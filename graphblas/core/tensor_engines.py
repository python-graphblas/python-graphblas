"""The TTM and MTTKRP engines of Tensors (UDT, Numba and the GraphBLAS products).

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import contextlib
import itertools
import math

import numpy as np

from ..dtypes import FP64
from ..exceptions import DimensionMismatch
from . import lib
from . import tensor_census as census
from . import tensor_layout as tl
from .matrix import Matrix
from .operator import get_typed_op
from .tensor_layout import Layout
from .utils import output_type


def _ttm_engines(updater, a, b, op, a_axes, b_axes):
    """A TTM by the first engine of ``_TTM_ENGINES`` before "products" that runs here, or by
    the products on compacted rows: ``(src, layout, shape)``; None for the plain products
    (``_contract``).
    """
    kernels = (
        len(a_axes) == 1
        and b.ndim == 2
        and a.ndim >= 2
        and op.name == "plus_times"
        and op.return_type == FP64
        and a.dtype == FP64
        and b.dtype == FP64
        and a._M._nvals >= max(_t._TTM_KERNEL_MIN_NNZ, 1)
        and b._M._nvals == math.prod(b._shape)
    )
    declined = []
    for engine in _t._TTM_ENGINES:
        if engine == "products":
            break
        if not kernels:
            continue
        run = _t._ttm_udt if engine == "udt" else _t._ttm_numba
        if (rv := run(updater, a, b, op, a_axes, b_axes)) is not None:
            if census.on:
                census.note("ttm", engine, declined=declined, kernels=kernels, nvals=a._M._nvals)
            return rv
        declined.append(engine)
    # The products, on the nonempty rows when they are few
    rv = _t._ttm_compacted(updater, a, b, op, a_axes, b_axes)
    if census.on and len(a_axes) == 1 and b.ndim == 2:
        census.note("ttm", "products" if rv is None else "products, compacted rows",
                    declined=declined, kernels=kernels, nvals=a._M._nvals)  # fmt: skip
    return rv


def _ttm_factor(b, kb):
    """The factor of a TTM as a dense C-contiguous (K x R) float64 array, K along its axis
    ``kb`` (the contracted one); a new array.
    """
    Bm = b._M if b._layout.rows == (0,) else tl.transpose(b._M)  # (b's axis 0 | axis 1)
    U = np.ascontiguousarray(Bm.to_dense(fill_value=0.0))
    if kb == 1:
        U = np.ascontiguousarray(U.T)
    return U


def _ttm_udt(updater, a, b, op, a_axes, b_axes):
    """TTM by one ``mxv`` with FP64[R] values (``tensor_udt.ttm``), when the JIT is on:
    ``(src, layout, shape)`` as from ``_contract``, or None. The contracted axis must be the
    column group by itself, so each nonempty row is an output fiber; the tensor moves to the
    cheapest layout that has it so. Where the Numba engine runs too, this one takes only
    the cases it is faster on (see ``_TTM_UDT_MAX_R``).
    """
    from . import tensor_udt

    before = tensor_udt.jit_control()
    if not tensor_udt.jit_ready():
        return None
    m, kb = a_axes[0], b_axes[0]
    free_a = tuple(x for x in range(a.ndim) if x != m)
    q = len(free_a)
    targets = frozenset(
        ((*perm, m), q)
        for perm in (itertools.permutations(free_a) if q <= 4 else [free_a])
        if tl._valid_split(a._shape, (*perm, m), q)
    )
    if not targets:
        return None
    nthreads = _nthreads(updater)
    _, path, state = tl.plan(a._shape, a._layout.state, targets, a._M._nvals, nthreads)
    if (
        _t._has_numba
        and "numba" in _t._TTM_ENGINES
        and (
            path
            or b._shape[1 - kb] > _t._TTM_UDT_MAX_R
            # A hypersparse output as SuiteSparse decides it (nonempty rows under its hyper
            # switch, 1/16): Uber's 14% of fibers, a sparse output, ran 1.4-1.5x faster with
            # Numba (phase 9a census), NELL-2's 0.3% 1.24x faster with UDT
            or tl.count_nonempty_rows(a._M) >= tl._global_hyper_switch() * a._M._nrows
        )
    ):
        return None  # (the Numba engine is faster here: see _TTM_UDT_MAX_R)
    layout = Layout.from_state(state)
    from .descriptor import lookup

    def compute():
        # (the relayout too: under the raised control it may compile)
        X = _in_layout(a, layout, updater.opts.get("nthreads"))
        return tensor_udt.ttm(X, _ttm_factor(b, kb), lookup(**updater.opts))

    if (src := tensor_udt.run(before, compute)) is None:
        return None
    out = {x: i for i, x in enumerate(free_a)}
    shape = (*(a._shape[x] for x in free_a), src._ncols)
    return src, Layout(tuple(out[x] for x in layout.rows), (len(shape) - 1,)), shape


def _ttm_numba(updater, a, b, op, a_axes, b_axes):
    """TTM (an FP64 tensor's axis contracted with a full FP64 factor, plus_times) by a Numba
    kernel over the stored arrays (``tensor_numba.ttm``): ``(src, layout, shape)`` as from
    ``_contract``, or None to use the product. The contracted axis must be the last column
    axis; the tensor stays as it is when it is (with leading column axes too: their runs are
    the output fibers), else it moves to the cheapest layout that has it there.

    The output is today's storage: hypersparse rows of R values, or full when every output
    fiber exists. (One mxm was 2.4-2.7x behind TACO in the stored order: its Gustavson saxpy
    on hypersparse rows, then the output's row hash; DESIGN_phase5 1.1.)
    """
    if not _t._has_numba or len(a_axes) != 1 or b.ndim != 2 or a.ndim < 2:
        return None
    if op.name != "plus_times" or op.return_type != FP64 or a.dtype != FP64 or b.dtype != FP64:
        return None
    if a._M._nvals < max(_t._TTM_KERNEL_MIN_NNZ, 1) or b._M._nvals != math.prod(b._shape):
        return None
    m, kb = a_axes[0], b_axes[0]
    free_a = tuple(x for x in range(a.ndim) if x != m)
    nthreads = _nthreads(updater)
    targets = frozenset(
        ((*perm, m), q)
        for perm in (itertools.permutations(free_a) if len(free_a) <= 4 else [free_a])
        for q in range(1, len(free_a) + 1)  # (with no row axis there is one row: no threads)
        if tl._valid_split(a._shape, (*perm, m), q)
    )
    if not targets:
        return None
    _, _, state = tl.plan(a._shape, a._layout.state, targets, a._M._nvals, nthreads)
    layout = Layout.from_state(state)
    X = _in_layout(a, layout, updater.opts.get("nthreads"))
    if X.ss.format not in {"csr", "hypercsr"}:
        return None
    from . import tensor_numba

    U = _ttm_factor(b, kb)
    R = U.shape[1]
    out = {x: i for i, x in enumerate(free_a)}
    rows = tuple(out[x] for x in (*layout.rows, *layout.cols[:-1]))
    shape = (*(a._shape[x] for x in free_a), R)
    nrows = tl.prod(shape, rows)
    with contextlib.ExitStack() as stack:
        cont = stack.enter_context(tl._Unloaded(X))
        p, h, ci = (stack.enter_context(tl._VectorArray(v)) for v in (cont.p, cont.h, cont.i))
        xv = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        kind, *parts = tensor_numba.ttm(
            (p.array, np.empty(0, np.int64) if h.array is None else h.array, ci.array, xv.array,
             bool(cont.iso)),
            tuple(a._shape[x] for x in layout.cols), U, nrows,
            updater.opts.get("nthreads") or nthreads,
        )  # fmt: skip
    # The parts are handed over, not copied or checked
    if kind == "full":
        src = tl.matrix_from_arrays(FP64, nrows, R, "fullr", x=parts[0], nvals=nrows * R)
    elif kind == "bitmap":
        V, B, nvals = parts
        src = tl.matrix_from_arrays(FP64, nrows, R, "bitmapr", x=V, b=B, nvals=nvals)
    else:
        hout, indptr, cols, V = parts
        src = tl.matrix_from_arrays(
            FP64, nrows, R, "hypercsr", x=V, h=hout, p=indptr, i=cols, nvals=V.size
        )
    return src, Layout(rows, (len(shape) - 1,)), shape


def _ttm_compacted(updater, a, b, op, a_axes, b_axes):
    """TTM in the stored order on a hypersparse tensor, by one product on its rows compacted:
    the nonempty rows (h dropped through the Container) times the factor, then the result's
    rows given h back. On hypersparse rows SuiteSparse runs a Gustavson saxpy; on the same
    rows stored compactly its bitmap / full kernel, with no index per output entry (NELL-2
    mode 2: 404-430 against 689-769 ms at 1 thread, DESIGN_phase5 1.1). Any type and
    semiring; ``(src, layout, shape)`` or None.
    """
    if len(a_axes) != 1 or b.ndim != 2 or a._layout.cols != a_axes or not a._layout.rows:
        return None
    M = a._M
    if M.ss.format != "hypercsr" or M._nvals == 0:
        return None
    nonempty = tl.count_nonempty_rows(M)
    if 2 * nonempty > M._nrows:
        return None  # (few rows to drop)
    kb = b_axes[0]
    Bm = b._M if b._layout.rows == (kb,) else tl.transpose(b._M)  # (b's axis kb | the other)
    A = M.dup()
    with tl._Unloaded(A) as cont:
        h, cont.h = cont.h, tl.new_vector_handle(cont.h)
        cont.nrows = nonempty
        cont.format = lib.GxB_SPARSE
    C = tl.new_matrix(op.return_type, nonempty, Bm._ncols)
    C(**updater.opts) << A.mxm(Bm, op)
    tl.give_rows(C, h, M._nrows)
    free_a = tuple(x for x in range(a.ndim) if x != a_axes[0])
    out = {x: i for i, x in enumerate(free_a)}
    shape = (*(a._shape[x] for x in free_a), b._shape[1 - kb])
    return C, Layout(tuple(out[x] for x in a._layout.rows), (len(shape) - 1,)), shape


def _as_factor(F, nrows, within):
    """A factor matrix (Matrix or ndarray) with ``nrows`` rows, as a Matrix."""
    if isinstance(F, np.ndarray):
        if F.ndim != 2:
            raise ValueError(f"{within}: factors must be 2-d; got shape {F.shape}")
        F = Matrix.from_dense(F)
    elif output_type(F) is Tensor:
        F = _compute(F, within).to_matrix()
    elif type(F) is not Matrix:
        raise TypeError(f"{within}: a factor must be a Matrix or ndarray, not {type(F)}")
    if F._nrows != nrows:
        raise DimensionMismatch(f"{within}: a factor has {F._nrows} rows; expected {nrows}")
    return F


def _mttkrp_expr(self, factors, mode, op):
    mode = _normalize_axis(mode, self.ndim)
    if self.ndim < 2:
        raise ValueError("mttkrp needs at least 2 dimensions")
    if len(factors) != self.ndim:
        raise ValueError(f"mttkrp needs {self.ndim} factors (one per axis); got {len(factors)}")
    others = [a for a in range(self.ndim) if a != mode]
    mats = {a: _as_factor(factors[a], self._shape[a], "mttkrp") for a in others}
    ranks = {M._ncols for M in mats.values()}
    if len(ranks) != 1:
        raise DimensionMismatch(f"mttkrp: factors have different ranks {sorted(ranks)}")
    R = ranks.pop()
    op = get_typed_op(op, self.dtype, mats[others[0]].dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="mttkrp", argname="op")
    return TensorExpression(
        "mttkrp",
        _t._mttkrp_recipe,
        [self],
        (self, mats, mode, R, op),
        dtype=op.return_type,
        shape=(self._shape[mode], R),
        op=op,
        expr_repr="{0.name}.mttkrp(mode=%d, op={op})" % mode,  # noqa: UP031
    )


def _mttkrp_numba(x, mats, mode, nthreads, *, layout=None):
    """MTTKRP of an FP64 tensor of 3 or more axes with full FP64 factors and plus_times by a
    Numba kernel over the stored arrays (``tensor_numba``) in ``layout`` (default: as stored),
    as a Matrix; or None to use the GraphBLAS products: no Numba, fewer axes, another type or
    semiring, sparse factors, an empty group, bitmap / full storage, or per-thread buffers too
    large for this layout.
    """
    if not _t._has_numba or x.ndim < 3 or x._M._nvals < max(_t._MTTKRP_KERNEL_MIN_NNZ, 1):
        return None
    layout = x._layout if layout is None else layout
    if not layout.rows or not layout.cols or x.dtype != FP64:
        return None
    for M in mats.values():
        if M.dtype != FP64 or M._nvals != M._nrows * M._ncols:
            return None
    X = _in_layout(x, layout, nthreads)
    if X.ss.format not in {"csr", "hypercsr"}:
        return None
    from . import tensor_numba

    order = (*layout.rows, *layout.cols)
    R = next(iter(mats.values()))._ncols
    nnz = X._nvals
    with contextlib.ExitStack() as stack:
        F = {}
        views = {}
        for ax, M in mats.items():
            if (arr := views.get(id(M))) is None:
                arr = views[id(M)] = _factor_array(M, stack)
            F[ax] = arr
        Fs = [F[ax] if ax != mode else np.empty((0, R)) for ax in order]
        cont = stack.enter_context(tl._Unloaded(X))
        p, h, ci = (stack.enter_context(tl._VectorArray(v)) for v in (cont.p, cont.h, cont.i))
        xv = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        rv = tensor_numba.mttkrp_nd(
            (p.array, np.empty(0, np.int64) if h.array is None else h.array, ci.array, xv.array,
             bool(cont.iso)),
            tuple(x._shape[a] for a in layout.rows),
            tuple(x._shape[a] for a in layout.cols),
            order.index(mode), Fs, x._shape[mode], nthreads,
            # Per-thread output buffers up to about the data's size
            max_buffer=max(R * nnz, 2**22),
        )  # fmt: skip
    if rv is None:
        return None
    out, touched = rv
    if touched.all():
        return Matrix.ss.import_fullr(values=out, take_ownership=True)
    # The reached rows only, each full: hypersparse arrays handed over (a build of 7e6
    # entries took 35 ms on hyper, against 12 for the kernel)
    rows = np.flatnonzero(touched)
    n = rows.size
    return Matrix.ss.import_hypercsr(
        nrows=out.shape[0], ncols=R, rows=rows, indptr=np.arange(0, (n + 1) * R, R),
        col_indices=np.tile(np.arange(R), n), values=out[rows].ravel(), sorted_cols=True,
        take_ownership=True,
    )  # fmt: skip


def _factor_array(M, stack):
    """A full FP64 factor as an (n x R) array: a view of its values through the Container
    while ``stack`` is open (Enron's 244268 x 64 factor is 125 MB to copy), else a copy.
    """
    nrows, ncols = M._nrows, M._ncols  # (M reads as empty while unloaded)
    if M.ss.format == "fullr" and not M.ss.is_iso and nrows * ncols > 0:
        cont = stack.enter_context(tl._Unloaded(M))
        x = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        if x.array is not None and x.array.size == nrows * ncols:
            return x.array.reshape(nrows, ncols)
        # (not expected: a full Matrix holds nrows * ncols values)
    return np.ascontiguousarray(M.to_dense(fill_value=0.0))


# The products' intermediates cost about this much per (row x R) entry, ns at 8 / 1 threads
# (Enron mode 1, R = 64: 0.92 s for a first intermediate of 567,936 rows; 26.9 s for 31.3e6)
C_MTTKRP_INTER = (2.0, 13.0)


def _mttkrp_order(x, mode, R, nthreads, *, products):
    """The order of the axes other than ``mode`` for the layout ``(mode, *order[:-1] |
    order[-1])``: by the cost to reach it, plus with ``products`` the size of the
    intermediates the products make (one row per distinct ``(mode, *order[:-k])``, k = 1, 2,
    ...), estimated from a sample of the entries. Relayout cost alone picked, for Enron mode 1
    at 1 thread, a first intermediate 55x larger than at 8 threads (26.9 s against 0.92).
    """
    others = [a for a in range(x.ndim) if a != mode]
    orders = list(itertools.permutations(others)) if len(others) <= 4 else [tuple(others)]
    orders = [order for order in orders if tl._valid_split(x._shape, (mode, *order), len(order))]
    nnz = x._M._nvals
    fibers = {}
    if products and len(orders) > 1 and nnz > 0:
        groups = {(mode, *order[:k]) for order in orders for k in range(1, len(order))}
        fibers = _sampled_fibers(x, groups)
    ti = 0 if nthreads > 1 else 1
    best = None
    for order in orders:
        dst = Layout((mode, *order[:-1]), (order[-1],))
        cost = tl.relayout_cost(x._shape, x._layout, dst, nnz, nthreads)
        for k in range(1, len(order)):
            cost += C_MTTKRP_INTER[ti] * R * fibers.get((mode, *order[:k]), 0.0)
        if best is None or cost < best[0]:
            best = (cost, order)
    return best[1]


def _sampled_fibers(x, groups):
    """Estimated distinct values of each group of axes among ``x``'s entries, from a sample of
    2^14 stored entries (``_sampled_nonempty_rows``); uniform occupancy for bitmap or full
    storage.
    """
    M = x._M
    nnz = M._nvals
    rows, cols = x._layout
    if M.ss.format not in {"csr", "hypercsr"}:
        return {g: tl.expected_rows(tl.prod(x._shape, g), nnz) for g in groups}
    q = np.sort(np.random.default_rng(0).integers(0, nnz, min(nnz, _SPLIT_SAMPLE)))
    with (
        tl._Unloaded(M) as cont,
        tl._VectorArray(cont.p) as p,
        tl._VectorArray(cont.h) as h,
        tl._VectorArray(cont.i) as ci,
    ):
        slot = np.searchsorted(p.array, q.astype(p.array.dtype), side="right") - 1
        r = slot if h.array is None else h.array[slot]
        c = ci.array[q]
    coords = {}
    tl.unravel(r.astype(np.int64), x._shape, rows, coords)
    tl.unravel(c.astype(np.int64), x._shape, cols, coords)
    coords = [coords[a] for a in range(x.ndim)]
    sample = np.arange(q.size)
    return {g: _sampled_nonempty_rows(coords, x._shape, g, nnz, sample) for g in groups}


def _krp_embedding(F, R):
    """W with ``W[j * R + r, r] = F[j, r]``: one product with it sums over j for each r."""
    j, r, v = F.to_coo()
    j = j.astype(np.int64)
    r = r.astype(np.int64)
    return tl.from_coo(j * R + r, r, v, F.dtype, F._nrows * R, R)


def _mttkrp_recipe(updater, x, mats, mode, R, op):
    nthreads = updater.opts.get("nthreads")
    kernel = op.name == "plus_times" and op.return_type == FP64
    numba_kernel = kernel and "numba" in _t._MTTKRP_ENGINES
    if (
        numba_kernel
        and (T := _t._mttkrp_numba(x, mats, mode, nthreads or _nthreads(updater))) is not None
    ):
        # In the stored layout: no relayout
        if census.on:
            census.note("mttkrp", "numba, stored layout", nvals=x._M._nvals, ndim=x.ndim)
        _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)
        return
    # Any order of the other axes works. For the kernel, take the one whose layout is
    # cheapest to reach
    order = _mttkrp_order(x, mode, R, _nthreads(updater), products=False)
    if numba_kernel:
        dst = tl.as_layout(((mode, *order[:-1]), (order[-1],)), x._shape)
        T = _t._mttkrp_numba(x, mats, mode, nthreads or _nthreads(updater), layout=dst)
        if T is not None:
            # Relayout, then the kernel that owns its output rows
            if census.on:
                census.note("mttkrp", "numba, after a relayout", nvals=x._M._nvals, ndim=x.ndim)
            _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)
            return
    # For the products, count their intermediates too
    others = list(_mttkrp_order(x, mode, R, _nthreads(updater), products=True))
    last = others[-1]
    rest = tuple(others[:-1])
    dst = tl.as_layout(((mode, *rest), (last,)), x._shape)
    X = _in_layout(x, dst, nthreads)
    # 1. TTM over the last axis: (mode, rest | last) @ F_last -> (mode, rest | r), by the
    # UDT engine where it runs (see _MTTKRP_ENGINES)
    T = None
    if (
        kernel
        and "udt" in _t._MTTKRP_ENGINES
        and x.dtype == FP64
        and mats[last]._nvals == mats[last]._nrows * R
    ):
        from . import tensor_udt
        from .descriptor import lookup

        before = tensor_udt.jit_control()
        if X._nvals >= _t._TTM_KERNEL_MIN_NNZ and tensor_udt.jit_ready():
            U = np.ascontiguousarray(mats[last].to_dense(fill_value=0.0), np.float64)
            T = tensor_udt.run(before, tensor_udt.ttm, X, U, lookup(**updater.opts))
    if census.on:
        census.note("mttkrp", "products" if T is None else "udt TTM, then products",
                    nvals=x._M._nvals, ndim=x.ndim, numba_eligible=numba_kernel)  # fmt: skip
    if T is None:
        T = tl.new_matrix(op.return_type, X._nrows, R)
        T(**updater.opts) << X.mxm(mats[last], op)
    # 2. Each remaining axis: move it next to r, then one product with its embedding
    rows = (mode, *rest)
    for a in reversed(rest):
        rows = rows[:-1]
        T = tl.reshape_owned(T, tl.prod(x._shape, rows), x._shape[a] * R)
        out = tl.new_matrix(op.return_type, T._nrows, R)
        out(**updater.opts) << T.mxm(_krp_embedding(mats[a], R), op)
        T = out
    _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)


# isort: off
from . import tensor as _t  # noqa: E402 (knobs are set on the hub)
from .tensor import (  # noqa: E402
    Tensor,
    TensorExpression,
    _SPLIT_SAMPLE,
    _compute,
    _in_layout,
    _normalize_axis,
    _nthreads,
    _sampled_nonempty_rows,
    _store,
)

# isort: on

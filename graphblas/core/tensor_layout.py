"""Physical layouts of a Tensor and the planner that moves data between them (experimental).

A Tensor keeps its entries in one Matrix held by row. Its ``Layout`` says which logical axes
are raveled (C order) into the Matrix rows and which into the columns. The planner works on
physical states ``(seq, split)``: the axis sequence ``rows + cols`` and ``len(rows)``.

A relayout is a sort by the new major key, so the planner builds it like an LSD radix sort
from cheap SuiteSparse primitives (DESIGN.md 3.2 and 7.1):

- RS: ``ss.reshape`` in storage order (moves the split, keeps the sequence)
- T: a materialized transpose (swaps the groups); cheap only into few output rows
- PERM_R: any permutation of the row axes as one ``P @ M``
- REBUILD: ``to_coo``, new indices, ``from_coo`` (any state to any state)

and runs Dijkstra over the states with the measured per-entry costs. Plans and the shape-only
helper matrices (permutations and selectors) are cached.
"""

import heapq
import itertools
import math
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from .. import semiring as _semiring
from ..dtypes import BOOL
from ..exceptions import _error_code_lookup
from . import NULL, ffi, lib
from .matrix import Matrix

# GraphBLAS dimensions go up to GrB_INDEX_MAX + 1 = 2^60.
MAX_GROUP = 2**60
# The planner searches states up to this many axes; above it a relayout is one REBUILD.
MAX_PLANNED_NDIM = 5


class Layout(NamedTuple):
    """Which logical axes are raveled into the rows and the columns of the stored Matrix.

    Each group is raveled in C order in the order given, so ``Layout((1, 0), (2,))`` stores
    ``T[i, j, k]`` at row ``j * shape[0] + i`` and column ``k``.
    """

    rows: tuple
    cols: tuple

    @property
    def seq(self):
        return self.rows + self.cols

    @property
    def state(self):
        return (self.rows + self.cols, len(self.rows))

    @classmethod
    def from_state(cls, state):
        seq, p = state
        return cls(seq[:p], seq[p:])

    def _short(self):
        rows = ",".join(map(str, self.rows))
        cols = ",".join(map(str, self.cols))
        return f"({rows}|{cols})"


def prod(shape, axes):
    return math.prod(shape[a] for a in axes)


def as_layout(layout, shape):
    """Validate ``layout`` (a Layout or a pair of axis sequences) for ``shape``."""
    try:
        rows, cols = layout
        rows = tuple(int(a) for a in rows)
        cols = tuple(int(a) for a in cols)
    except (TypeError, ValueError):
        raise TypeError(
            f"layout must be a Layout or a pair of axis sequences (rows, cols); got {layout!r}"
        ) from None
    ndim = len(shape)
    if sorted(rows + cols) != list(range(ndim)):
        raise ValueError(
            f"layout {layout!r} must partition the axes {tuple(range(ndim))} "
            "into rows and cols, each axis once"
        )
    for axes in (rows, cols):
        if prod(shape, axes) > MAX_GROUP:
            raise ValueError(
                f"axes {axes} of shape {shape} have {prod(shape, axes)} positions, more than "
                f"one Matrix dimension can hold (2**60); choose another layout"
            )
    return Layout(rows, cols)


def _valid_split(shape, seq, p):
    return prod(shape, seq[:p]) <= MAX_GROUP and prod(shape, seq[p:]) <= MAX_GROUP


def default_layout(shape, nnz=None):
    """The layout for new data: identity sequence, split by expected entries per row.

    ``p`` is the largest split in ``1..ndim-1`` with ``prod(shape[:p]) <= nnz / 4`` (about four
    entries per nonempty row), else 1; with no ``nnz`` (an empty Tensor) ``p = ndim - 1``.
    A 1-d Tensor is stored as one row (the bytes of a Vector); a 2-d one as the plain Matrix.
    """
    ndim = len(shape)
    seq = tuple(range(ndim))
    if ndim == 0:
        return Layout((), ())
    if ndim == 1:
        return Layout((), (0,))
    if ndim == 2:
        return Layout((0,), (1,))
    if nnz is None:
        p = ndim - 1
    else:
        p = 1
        for q in range(ndim - 1, 0, -1):
            if prod(shape, seq[:q]) * 4 <= nnz:
                p = q
                break
    if _valid_split(shape, seq, p):
        return Layout(seq[:p], seq[p:])
    for q in range(1, ndim):
        if _valid_split(shape, seq, q):
            return Layout(seq[:q], seq[q:])
    raise ValueError(f"shape {shape} has no split whose groups each fit in 2**60")


# ---------------------------------------------------------------------------------------------
# Raveling


def ravel(coords, shape, axes, n=None):
    """C-order linear index of ``axes`` (int64; every group is < 2^60)."""
    if not axes:
        if n is None:
            n = coords[0].size if coords else 0
        return np.zeros(n, np.int64)
    idx = coords[axes[0]].astype(np.int64, copy=True)
    for a in axes[1:]:
        idx *= shape[a]
        idx += coords[a]
    return idx


def unravel(idx, shape, axes, out):
    """Fill ``out[a]`` for each axis in ``axes`` from the linear index ``idx``."""
    if not axes:
        return
    if len(axes) == 1:
        out[axes[0]] = idx
        return
    for a in reversed(axes[1:]):
        idx, out[a] = np.divmod(idx, shape[a])
    out[axes[0]] = idx


def coords_of(M, shape, state, *, values=True, sort=False):
    """Logical coordinates (a list of int64 arrays) and values of the entries of ``M``."""
    seq, p = state
    r, c, v = M.to_coo(values=values, sort=sort)
    coords = [None] * len(shape)
    unravel(r.view(np.int64), shape, seq[:p], coords)
    unravel(c.view(np.int64), shape, seq[p:], coords)
    return coords, v


# ---------------------------------------------------------------------------------------------
# Matrices held by row, and the Container calls


def _check(rc, what):
    # Direct `lib` calls skip python-graphblas's `call`, so check their status here.
    if rc != 0:  # pragma: no cover (safety)
        exc = _error_code_lookup.get(rc, RuntimeError)
        raise exc(f"{what} failed with GrB_Info {rc}")


def _global_by_col():
    val = ffi.new("int32_t*")
    _check(lib.GxB_Global_Option_get_INT32(lib.GxB_FORMAT, val), "GxB_Global_Option_get")
    return val[0] == lib.GxB_BY_COL


def new_matrix(dtype, nrows, ncols, *, name=None):
    """An empty Matrix held by row whatever the global format is.

    SuiteSparse creates every n x 1 Matrix by column (as a Vector), whatever the global
    format; a per-Matrix setting on the empty Matrix sticks through later updates.
    """
    M = Matrix(dtype, nrows, ncols, name=name)
    if ncols == 1 or _global_by_col():
        M.ss.config["format"] = "by_row"
    return M


def from_coo(rows, cols, values, dtype, nrows, ncols, *, dup_op=None, name=None):
    """``Matrix.from_coo`` held by row. Under a by-column global format, build the transpose
    by column and flip it, which costs nothing extra.
    """
    if _global_by_col():
        M = Matrix.from_coo(
            cols, rows, values, dtype, nrows=ncols, ncols=nrows, dup_op=dup_op, name=name
        )
        flip_orientation(M)
    else:
        M = Matrix.from_coo(
            rows, cols, values, dtype, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    if not is_by_row(M):
        # An n x 1 result (see new_matrix); converting it costs O(nvals).
        M.ss.config["format"] = "by_row"
    return M


def is_by_row(M):
    return M.ss.orientation == "rowwise"


class _Unloaded:
    """Context manager: ``M`` unloaded into a GxB_Container, loaded back on exit."""

    def __init__(self, M):
        self.M = M

    def __enter__(self):
        # Pending work (zombies, pending tuples, jumbled rows) must not be unloaded.
        self.M.wait()
        self._Cp = ffi.new("GxB_Container*")
        _check(lib.GxB_Container_new(self._Cp), "GxB_Container_new")
        rc = lib.GxB_unload_Matrix_into_Container(self.M.gb_obj[0], self._Cp[0], NULL)
        if rc != 0:  # pragma: no cover (safety)
            lib.GxB_Container_free(self._Cp)
            _check(rc, "GxB_unload_Matrix_into_Container")
        return self._Cp[0]

    def __exit__(self, *exc):
        rc = lib.GxB_load_Matrix_from_Container(self.M.gb_obj[0], self._Cp[0], NULL)
        lib.GxB_Container_free(self._Cp)
        _check(rc, "GxB_load_Matrix_from_Container")


def flip_orientation(M):
    """In place, O(1): an (n x m) Matrix held by column becomes the (m x n) Matrix of its
    transpose held by row (or the reverse), sharing the same arrays (DESIGN.md 4.1, s8).
    """
    with _Unloaded(M) as cont:
        cont.nrows, cont.ncols = cont.ncols, cont.nrows
        cont.nrows_nonempty, cont.ncols_nonempty = cont.ncols_nonempty, cont.nrows_nonempty
        if cont.orientation == lib.GrB_COLMAJOR:
            cont.orientation = lib.GrB_ROWMAJOR
        else:
            cont.orientation = lib.GrB_COLMAJOR
    M._nrows, M._ncols = M._ncols, M._nrows
    return M


def nonempty_rows(M):
    """The number of nonempty rows of a hypersparse Matrix (O(1)); ``nrows`` otherwise."""
    if M.ss.format != "hypercsr":
        return M._nrows
    n = ffi.new("GrB_Index*")
    with _Unloaded(M) as cont:
        rc = lib.GrB_Vector_size(n, cont.h)
    _check(rc, "GrB_Vector_size")
    return n[0]


def steal_vector_as_row(v):
    """Turn a Vector (n x 1 by column) into a (1 x n) Matrix held by row, O(1). ``v`` is
    left empty and must not be used again.
    """
    M = Matrix._from_obj(
        ffi.new("GrB_Matrix*", ffi.cast("GrB_Matrix", v.gb_obj[0])), v.dtype, v._size, 1
    )
    v.gb_obj[0] = NULL
    return flip_orientation(M)


# ---------------------------------------------------------------------------------------------
# Small LRU caches for plans and shape-only matrices


class _LRU:
    """Least recently used entries are dropped past ``maxsize`` items or ``maxweight``."""

    def __init__(self, maxsize, maxweight=math.inf):
        self.maxsize = maxsize
        self.maxweight = maxweight
        self.weight = 0
        self.data = OrderedDict()

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            return None
        self.data.move_to_end(key)
        return item[0]

    def put(self, key, val, weight=0):
        if weight > self.maxweight:
            return
        if (old := self.data.pop(key, None)) is not None:
            self.weight -= old[1]
        self.data[key] = (val, weight)
        self.weight += weight
        while len(self.data) > self.maxsize or self.weight > self.maxweight:
            _, (_, w) = self.data.popitem(last=False)
            self.weight -= w

    def clear(self):
        self.data.clear()
        self.weight = 0


_plan_cache = _LRU(4096)
# Permutation and selector matrices depend only on the shape, so they are reused across calls.
# They hold one entry per position of a group (about 12 bytes each), so the cache is bounded
# by its total entries: 2**25 is about 400 MB.
_matrix_cache = _LRU(64, maxweight=2**25)


def clear_caches():
    _plan_cache.clear()
    _matrix_cache.clear()


# ---------------------------------------------------------------------------------------------
# Cost model: ns per stored entry, (8 threads, 1 thread), from DESIGN.md 3.10. The constants
# rank plans; they are not predictions (they run 1.3-2x low).

C_RS = (0.5, 2.1)
C_REBUILD = (9.0, 60.0)
C_PERM_ENTRY = (0.6, 2.0)
C_PERM_ROW = {"sparse": (1.0, 0.0), "hyper": (30.0, 185.0)}
T_TABLE = [
    (1e3, 0.52, 1.82),
    (1e4, 0.73, 2.65),
    (1e5, 2.2, 6.2),
    (3e5, 16.3, 10.0),
    (1e6, 12.3, 10.5),
    (1e7, 8.2, 14.5),
]
# Python and call overhead per step, so tiny tensors take the plan with the fewest steps.
C_STEP = 10_000.0
C_REBUILD_FIXED = 40_000.0


def t_cost(out_rows, ti):
    """Transpose cost per entry, log-linear in the number of output rows."""
    if out_rows <= T_TABLE[0][0]:
        return T_TABLE[0][1 + ti]
    for (r0, *a), (r1, *b) in itertools.pairwise(T_TABLE):
        if out_rows <= r1:
            f = (math.log(out_rows) - math.log(r0)) / (math.log(r1) - math.log(r0))
            return a[ti] + f * (b[ti] - a[ti])
    return T_TABLE[-1][1 + ti]


def rebuild_cost(nnz, ti):
    return C_REBUILD[ti] * nnz + C_REBUILD_FIXED


def _neighbors(state, shape, nnz, ti, hyper1, allow_perm, total_ok):
    seq, p = state
    n = len(seq)
    if total_ok:
        rs = C_RS[ti] * nnz + C_STEP
        for q in range(n + 1):
            if q != p and _valid_split(shape, seq, q):
                yield ("RS", q), (seq, q), rs
    # A transpose with an empty group is a reshape (RS covers it).
    if 0 < p < n:
        out_rows = prod(shape, seq[p:])
        cost = t_cost(out_rows, ti) * nnz
        single = False
        # A wide transpose can be faster on one thread (DESIGN.md 3.2); price that too.
        if ti == 0 and out_rows >= 1e5:
            cost1 = t_cost(out_rows, 1) * nnz
            if cost1 < cost:
                cost = cost1
                single = True
        yield ("T", single), (seq[p:] + seq[:p], n - p), cost + C_STEP
    if allow_perm and p >= 2:
        nrows = prod(shape, seq[:p])
        kind = "hyper" if nrows > 16 * nnz else "sparse"
        cost = C_PERM_ENTRY[ti] * nnz + C_PERM_ROW[kind][ti] * min(nrows, nnz)
        if hyper1:
            # About one entry per row: P @ M pays a hash lookup per row (DESIGN.md 3.1).
            cost = max(cost, C_REBUILD[ti] * nnz)
        cost += C_STEP
        for perm in itertools.permutations(seq[:p]):
            if perm != seq[:p]:
                yield ("PERM_R", perm), (perm + seq[p:], p), cost


def plan(shape, src, targets, nnz, nthreads, *, hyper1=False, allow_perm=True):
    """Cheapest edge sequence from state ``src`` to any state in ``targets`` (a frozenset).

    Returns ``(cost_ns, path, final_state)``. A path is a list of edges: ``("RS", q)``,
    ``("T", single_thread)``, ``("PERM_R", new_row_seq)`` or ``("REBUILD", state)``.
    """
    if src in targets:
        return 0.0, [], src
    ti = 0 if nthreads > 1 else 1
    key = (shape, src, targets, ti, hyper1, allow_perm, int(nnz).bit_length())
    if (rv := _plan_cache.get(key)) is not None:
        return rv
    fallback = min(targets)
    rebuild = rebuild_cost(nnz, ti)
    rv = (rebuild, [("REBUILD", fallback)], fallback)
    if len(shape) <= MAX_PLANNED_NDIM and nnz > 0:
        total_ok = math.prod(shape) < MAX_GROUP
        best = {src: 0.0}
        counter = itertools.count()
        heap = [(0.0, next(counter), src, ())]
        while heap:
            cost, _, state, path = heapq.heappop(heap)
            if state in targets:
                if cost <= rebuild:
                    rv = (cost, list(path), state)
                else:
                    rv = (rebuild, [("REBUILD", state)], state)
                break
            if cost > best.get(state, math.inf) or cost > rebuild:
                continue
            for edge, nxt, w in _neighbors(state, shape, nnz, ti, hyper1, allow_perm, total_ok):
                nc = cost + w
                if nc < best.get(nxt, math.inf):
                    best[nxt] = nc
                    heapq.heappush(heap, (nc, next(counter), nxt, (*path, edge)))
    _plan_cache.put(key, rv)
    return rv


# ---------------------------------------------------------------------------------------------
# Executing a plan


def _perm_semiring(M):
    # P has one entry per row, so the monoid never combines two values; pick a factory kernel
    # (DESIGN.md 7.2). An iso M stays iso only with any_second (plus_second drops it).
    if M.ss.is_iso:
        return _semiring.any_second
    if M.dtype == BOOL:
        return _semiring.lor_second
    return _semiring.plus_second


def _perm_matrix(shape, old_rows, new_rows, M):
    """P with ``P[new_row, old_row] = True``, rows sorted, so ``P @ M`` permutes the rows."""
    nrows = M._nrows
    nvals = M._nvals
    if nrows <= max(4 * nvals, 4096):
        key = ("P", tuple(shape[a] for a in old_rows), old_rows, new_rows)
        if (P := _matrix_cache.get(key)) is not None:
            return P
        # Iterate the new row ids in order so P's rows arrive sorted.
        newr = np.arange(nrows, dtype=np.int64)
        co = {}
        unravel(newr, shape, new_rows, co)
        oldr = ravel(co, shape, old_rows)
        P = from_coo(newr, oldr, True, BOOL, nrows, nrows)
        _matrix_cache.put(key, P, nrows)
        return P
    # Few nonempty rows: build P from those only (not cacheable).
    oldr = np.unique(M.to_coo(columns=False, values=False, sort=False)[0].view(np.int64))
    co = {}
    unravel(oldr, shape, old_rows, co)
    newr = ravel(co, shape, new_rows)
    return from_coo(newr, oldr, True, BOOL, nrows, nrows)


def perm_rows(M, shape, old_rows, new_rows):
    P = _perm_matrix(shape, old_rows, new_rows, M)
    out = new_matrix(M.dtype, M._nrows, M._ncols)
    out << P.mxm(M, _perm_semiring(M))
    return out


def rebuild(M, shape, src, dst):
    seq, p = dst
    iso = M.ss.is_iso
    coords, v = coords_of(M, shape, src, values=not iso)
    if iso:
        v = M.ss.iso_value.value if M._nvals > 0 else 0
    rows = ravel(coords, shape, seq[:p])
    cols = ravel(coords, shape, seq[p:], n=rows.size)
    return from_coo(rows, cols, v, M.dtype, prod(shape, seq[:p]), prod(shape, seq[p:]))


def transpose(M, *, single_thread=False):
    out = new_matrix(M.dtype, M._ncols, M._nrows)
    if single_thread:
        out(nthreads=1) << M.T
    else:
        out << M.T
    return out


def execute(M, shape, state, path):
    """Run ``path`` on ``M`` (held by row, in ``state``); return the new Matrix and state.

    ``M`` itself is never changed; intermediates are reshaped in place, which skips a copy
    of the values (7.1 ms vs 9.6 for 1e7 entries at 8 threads, and less peak memory).
    SuiteSparse 10.5.1's in-place reshape does not widen 32-bit indices, so a reshape whose
    new dimensions need 64-bit indices (past 2^32) gets wrong column indices; those use the
    copying reshape.
    """
    owned = False
    for edge in path:
        seq, p = state
        kind = edge[0]
        if kind == "RS":
            q = edge[1]
            nrows, ncols = prod(shape, seq[:q]), prod(shape, seq[q:])
            if M._nvals == 0:
                M = new_matrix(M.dtype, nrows, ncols)
            elif owned and nrows <= 2**31 and ncols <= 2**31:
                # Storage order (rowwise on a by-row Matrix), so the other 10.5.1 reshape bug
                # (order != orientation) cannot apply either.
                M.ss.reshape(nrows, ncols, inplace=True)
            else:
                M = M.ss.reshape(nrows, ncols)
            state = (seq, q)
            owned = True
        elif kind == "T":
            M = transpose(M, single_thread=edge[1])
            state = (seq[p:] + seq[:p], len(seq) - p)
        elif kind == "PERM_R":
            M = perm_rows(M, shape, seq[:p], edge[1])
            state = (edge[1] + seq[p:], p)
        else:
            M = rebuild(M, shape, state, edge[1])
            state = edge[1]
        owned = True
    return M, state


def _nthreads():
    val = ffi.new("int32_t*")
    _check(lib.GxB_Global_Option_get_INT32(lib.GxB_NTHREADS, val), "GxB_Global_Option_get")
    return val[0]


def density(M):
    """True when the rows hold fewer than about 2 entries each (the hypersparse branch)."""
    nvals = M._nvals
    if nvals < 65536:
        # Small: the choice does not matter, and the Container round trip costs time.
        return False
    return nvals < 2 * nonempty_rows(M)


def relayout(M, shape, src, targets, *, nthreads=None, allow_perm=None):
    """Move ``M`` from Layout ``src`` to a layout in ``targets`` (a Layout or a frozenset of
    states). Returns ``(M2, layout2)``; ``M2`` is ``M`` itself when no move is needed.
    """
    src_state = src.state
    if isinstance(targets, Layout):
        if targets == src:
            return M, src
        targets = frozenset([targets.state])
    elif src_state in targets:
        return M, src
    if nthreads is None:
        nthreads = _nthreads()
    nvals = M._nvals
    if allow_perm is None:
        allow_perm = not M.dtype._is_udt
    hyper1 = density(M)
    _, path, _ = plan(shape, src_state, targets, nvals, nthreads, hyper1=hyper1,
                      allow_perm=allow_perm)  # fmt: skip
    if nvals == 0:
        seq, p = _final_state(src_state, path)
        return new_matrix(M.dtype, prod(shape, seq[:p]), prod(shape, seq[p:])), Layout(
            seq[:p], seq[p:]
        )
    M, state = execute(M, shape, src_state, path)
    return M, Layout.from_state(state)


def _final_state(state, path):
    for edge in path:
        seq, p = state
        if edge[0] == "RS":
            state = (seq, edge[1])
        elif edge[0] == "T":
            state = (seq[p:] + seq[:p], len(seq) - p)
        elif edge[0] == "PERM_R":
            state = (edge[1] + seq[p:], p)
        else:
            state = edge[1]
    return state


def relayout_cost(shape, src, dst, nnz, nthreads):
    """Estimated ns to move ``nnz`` entries from Layout ``src`` to Layout ``dst``."""
    if src == dst:
        return 0.0
    return plan(shape, src.state, frozenset([dst.state]), nnz, nthreads)[0]


# ---------------------------------------------------------------------------------------------
# Selectors: kron-structured 0/1 matrices that reduce a subset of a group's axes in one mxm


def selector(group_shape, kept, *, transposed=False):
    """S with ``S[ravel(x[kept]), ravel(x)] = True`` over the positions ``x`` of a group, as a
    BOOL iso Matrix (``prod(kept) x prod(group)``); its transpose if ``transposed``.

    ``kept`` are positions within the group, in group order. ``S @ M`` reduces the other row
    axes and ``M @ S.T`` the other column axes, with semiring ``(monoid, second/first)``.
    """
    key = ("S", group_shape, kept, transposed)
    if (S := _matrix_cache.get(key)) is not None:
        return S
    n = math.prod(group_shape)
    src = np.arange(n, dtype=np.int64)
    co = {}
    axes = tuple(range(len(group_shape)))
    unravel(src, group_shape, axes, co)
    dst = ravel(co, group_shape, kept, n=n)
    nkept = prod(group_shape, kept)
    if transposed:
        S = from_coo(src, dst, True, BOOL, n, nkept)
    else:
        S = from_coo(dst, src, True, BOOL, nkept, n)
    _matrix_cache.put(key, S, n)
    return S

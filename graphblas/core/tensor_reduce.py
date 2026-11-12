"""Reductions of Tensors over axes: selectors, reduce_rowwise / columnwise, rebuilds.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import math

from .. import binary, monoid, semiring
from ..dtypes import BOOL, UINT32, UINT64
from . import tensor_census as census
from . import tensor_layout as tl
from .operator import get_semiring, get_typed_op
from .operator.monoid import TypedBuiltinMonoid
from .scalar import Scalar
from .tensor_layout import Layout
from .vector import Vector


def _reduce_aggregator(t, op, axis, keepdims):
    """``reduce`` with an aggregator: its monoid; or its semiring's multiply applied with the
    aggregator's initial value to every entry, then the semiring's monoid, then its finalize
    op, as Matrix aggregations do.
    """
    from .operator.agg import TypedAggregator

    agg = op.parent if isinstance(op, TypedAggregator) else op
    if agg._monoid is not None:
        return _reduce_expr(t, agg._monoid, axis, keepdims)
    if agg._semiring is None or agg._composite is not None or agg._custom is not None:
        raise NotImplementedError(
            f"Tensor.reduce supports aggregators made of a monoid or a semiring; not "
            f"agg.{agg.name}"
        )
    # The semiring typed as Matrix aggregations type it, by the values' type and the initial
    # value's (so integer powers of INT64 values stay INT64: logaddexp2 of -3 is log2(0)); its
    # result type is then the one Matrix gives (INT64 for count and exists)
    agg[t.dtype]  # (a KeyError for the types Matrix refuses, as sum_of_squares of BOOL)
    sr = get_typed_op(agg._semiring, t.dtype, agg._initdtype)
    if agg._switch:
        x = t.apply(sr.binaryop, left=agg._initval).new()
    else:
        x = t.apply(sr.binaryop, right=agg._initval).new()
    rv = _reduce_expr(x, sr.monoid, axis, keepdims)
    if agg._finalize is not None:
        rv = rv.new().apply(agg._finalize[sr.return_type])
    return rv


def _reduce_expr(self, op, axis, keepdims, *, name="reduce", cast=None):
    axes = _normalize_axes(axis, self.ndim)
    op = get_typed_op(op, self.dtype if cast is None else cast, kind="binary")
    if op.opclass == "BinaryOp" and op.monoid is not None:
        op = op.monoid
    self._expect_op(op, "Monoid", within=name, argname="op")
    if keepdims:
        shape = tuple(1 if a in axes else n for a, n in enumerate(self._shape))
    else:
        shape = tuple(n for a, n in enumerate(self._shape) if a not in axes)
    axis_repr = "None" if axis is None else repr(axis)
    return TensorExpression(
        name,
        _reduce_recipe,
        [self],
        (self, op, axes, keepdims, cast),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr=f"{{0.name}}.{name}({{op}}, axis={axis_repr})",
    )


def _reduce_recipe(updater, a, op, axes, keepdims, cast):
    shape, layout = a._shape, a._layout
    M = a._M
    if cast is not None and M.dtype != cast:
        M = M.dup(cast)
    elif not axes:
        _store(updater, M, shape, layout, share=a)
        return
    M, lay = _reduce_matrix(M, shape, layout, set(axes), op, updater.opts)
    if keepdims:
        out_shape = tuple(1 if x in axes else n for x, n in enumerate(shape))
        out_layout = _insert_unit_axes(lay, axes)
    else:
        kept = [x for x in range(len(shape)) if x not in axes]
        out_shape = tuple(shape[x] for x in kept)
        out_layout = _renumber(lay, {x: i for i, x in enumerate(kept)})
    if _fixed_layout(updater) is None:
        M, out_layout = _healthy(M, out_shape, out_layout)
    _store(updater, M, out_shape, out_layout, fresh=True)


def _selector_semiring(op, mult):
    """The semiring of monoid ``op`` and ``mult`` (first or second) that multiplies by a
    selector, whose entries are True, or for a row selector (second) 1 of the type
    ``_selector_dtype`` gives.
    """
    if op is monoid.lor[BOOL]:
        # land(True, x) is x, as first and second give, and SuiteSparse 10 has a factory
        # kernel for lor_land only: BOOL has no factory FIRST or SECOND semiring (Tensor.any
        # ran about 2x slower with the JIT at "off" or "run")
        return semiring.lor_land[BOOL]
    if mult is binary.second and _selector_dtype(op) is not BOOL:
        return semiring.plus_times[op.type]
    return get_semiring(op, mult)


def _selector_dtype(op):
    """The type of the row selector that monoid ``op`` reduces with (``S @ M``): BOOL, except
    for sums of UINT32 and UINT64, where 1 * x with a selector of that type is a factory
    kernel (plus_times) and plus_second is not: u7 summed over axis 0 or 1 at 1 thread with
    the JIT off, 53-64 ms against 20-28, as with the JIT on.

    Column selectors (``M @ S.T``) stay BOOL: with the JIT on, the plus_first kernel does
    not read the selector, and was 2x faster there than plus_times (0.87 against 1.73 ms for
    1e6 entries at 1 thread).
    """
    if op is monoid.plus[UINT32] or op is monoid.plus[UINT64]:
        return op.type
    return BOOL


def _row_reducer(M, shape, rows, kept, dtype=BOOL):
    """S with S @ M reducing the row axes not in ``kept`` (positions in ``rows``)."""
    if tl.stored_format(M) == "hypercsr":
        return tl.rows_selector(M, shape, rows, kept, dtype=dtype)
    return tl.selector(tuple(shape[x] for x in rows), kept, dtype=dtype)


def _vector_to_row(v):
    return tl.steal_vector_as_row(v)


def _exact_identity(op):
    """Whether monoid ``op``'s identity, combined with any value of its type, returns that
    value bit for bit: the built-in monoids on integers and BOOL, and any on every type.
    """
    return isinstance(op, TypedBuiltinMonoid) and (
        op.name == "any" or op.type.name not in {"FP32", "FP64", "FC32", "FC64"}
    )


def _selector_mxm(left, right, sr, dtype, opts, pad):
    """``left @ right`` under ``sr``, which multiplies by a selector, into a new Matrix.

    With ``pad``, ``right`` has one more column than the result, which no entry fills, so no
    row of the product has an entry in every column. SuiteSparse's saxpy computes a row that
    does (a dense column of the transpose it computes) by starting from the monoid's identity
    and adding each term (GB_COMPUTE_DENSE_C_j, 10.5.1 and 10.5.2). That identity is not exact
    for every value of a floating type, nor for a user monoid: under min, a fiber of NaNs
    reduces to +inf there. Every other row starts from its first term, which keeps a lone term
    as it is, a fiber of NaNs NaN and a fiber of -0.0 at -0.0. The padding column is then
    dropped.

    The dot method also starts from the first term, but folds the rest under plus from 0 (an
    OpenMP SIMD reduction), so a fiber of -0.0 sums to +0.0 there, as in
    Matrix.reduce_rowwise. SuiteSparse 10.5 picks it for this product only when ``right`` is
    hypersparse and ``left`` bitmap or full (it computes the transposes), and in a padded
    product ``left`` is then a sparse selector, so forcing saxpy changes no method today and
    cost nothing measured (2026-11-10); it keeps it so if that choice changes. The padding
    itself cost 0.92-1.14x against none, and 1.16-1.23x under plus over wide sparse rows
    (10.5.1 and 10.5.2, 2026-11-10).
    """
    out = tl.new_matrix(dtype, left._nrows, right._ncols)
    if not pad:
        out(**opts) << left.mxm(right, sr)
        return out
    # Sparse or hypersparse, so the Container drops the padding column in O(1)
    out.ss.config["sparsity_control"] = {"sparse", "hypersparse"}
    out(**dict(opts, axb_method="saxpy")) << left.mxm(right, sr)
    tl.drop_last_column(out)
    out.ss.config["sparsity_control"] = "auto"
    return out


def _reduce_matrix(M, shape, layout, A, op, opts):
    """Reduce the axes ``A`` of ``M`` (stored in ``layout``); return the result Matrix and
    its layout over the kept axes (with their original numbers).
    """
    ndim = len(shape)
    dtype = op.return_type
    if M._nvals == 0:
        # SuiteSparse (seen from 8.2 through 10.5.1): a reduction or product over a
        # dimension of size 0 returns every output position (value 0) when its operands are
        # iso and full, as an empty region of an iso Tensor and the iso selectors are. No
        # entries reduce to none.
        rows = tuple(x for x in layout.rows if x not in A)
        cols = tuple(x for x in layout.cols if x not in A)
        out = tl.new_matrix(dtype, tl.prod(shape, rows), tl.prod(shape, cols))
        return out, Layout(rows, cols)
    if len(A) == ndim:
        if census.on:
            census.note("reduce", "all axes: reduce_scalar")
        # As Matrix.reduce_scalar: on bitmap data SuiteSparse starts from the monoid's identity
        # (GB_reduce_to_scalar.c, 10.5.1 and 10.5.2), so entries all NaN reduce to +inf under
        # min (-inf under max) and all -0.0 to +0.0 under plus; other formats keep them
        s = Scalar(dtype)
        s(**opts) << M.reduce_scalar(op)
        out = tl.new_matrix(dtype, 1, 1)
        out[0, 0] = s  # (entries reduce to a value)
        return out, Layout((), ())
    nvals = M._nvals
    if M.dtype._is_udt:
        if census.on:
            census.note("reduce", "UDT: rebuild")
        return _reduce_rebuild(M, shape, layout, A, op)
    # Pass on only the descriptor options the user gave: building one costs microseconds.
    nthreads = opts.get("nthreads") or tl._nthreads()
    ti = _ti(nthreads)
    # A selector product on sparse data under a monoid whose identity is not exact is padded
    # (see _selector_mxm). On bitmap or full data, SuiteSparse runs it as a dot product, which
    # starts from the first term; under plus, a full one folds the rest from 0, as
    # Matrix.reduce_rowwise does.
    exact = _exact_identity(op)
    A = set(A)
    while A:
        rows, cols = layout
        Ac = A & set(cols)
        if Ac:
            if Ac == set(cols):
                if census.on:
                    census.note("reduce", "column group: reduce_rowwise", nvals=M._nvals)
                # As Matrix.reduce_rowwise: its dot kernels fold the terms after the first
                # under plus from 0 (GB_AxB_dot_cij.c), so a fiber of two or more -0.0 sums
                # to +0.0 on sparse and full data (bitmap keeps -0.0)
                v = Vector(dtype, M._nrows)
                v(**opts) << M.reduce_rowwise(op)
                M = _vector_to_row(v)
                layout = Layout((), rows)
            else:
                gshape = tuple(shape[x] for x in cols)
                # A column group has no index of its nonempty columns, so the selector costs
                # per position; past the data's own cost, rebuild with dup_op instead
                nv = M._nvals
                sel, reb = _shape_selector_cost(nv, math.prod(gshape), ti), tl.rebuild_cost(nv, ti)
                if census.on:
                    j = census.choose("reduce", [sel, reb], ["column selector", "rebuild"],
                                      int(sel > reb), nvals=nv)  # fmt: skip
                    sel, reb = (0, 1) if j == 0 else (1, 0)
                if sel > reb:
                    return _reduce_rebuild(M, shape, layout, A, op)
                kept = tuple(i for i, x in enumerate(cols) if x not in Ac)
                pad = not exact and tl.stored_format(M) in {"csr", "hypercsr"}
                St = tl.selector(gshape, kept, transposed=True, pad=pad)
                M = _selector_mxm(M, St, _selector_semiring(op, binary.first), dtype, opts, pad)
                layout = Layout(rows, tuple(cols[i] for i in kept))
            A -= Ac
            continue
        Ar = A & set(rows)
        if Ar == set(rows):
            # Rows with about one entry each: a columnwise scatter costs a cache miss per
            # entry; rebuilding with dup_op is 4-5x faster.
            if tl.density(M):
                if census.on:
                    census.note("reduce", "row group, ~1 entry per row: rebuild", nvals=nvals)
                return _reduce_rebuild(M, shape, layout, A, op)
            ncols = tl.prod(shape, cols)
            wide = ncols >= 1e5
            if (
                wide
                and nthreads > 1
                and len(cols) >= 2
                and math.prod(shape) < tl.MAX_GROUP
                and 4 * tl.prod(shape, rows) * shape[cols[0]] <= nvals
            ):
                # reduce_columnwise into many outputs scatters like a wide transpose and is
                # slow with threads; move one column axis into the rows and
                # reduce with a selector instead, when the selector is small next to the data
                # (1000^3 / 1e7: 6.6 ms vs 55.6). A larger selector costs more than a
                # single-threaded columnwise reduce (Uber: 32.8 ms vs 4.7).
                if census.on:
                    census.note("reduce", "row group, wide: reshape, then a selector",
                                nvals=nvals)  # fmt: skip
                q = len(rows) + 1
                seq = layout.seq
                M = M.ss.reshape(tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
                layout = Layout(seq[:q], seq[q:])
                continue
            # Into many outputs, a sum on 2 threads is as fast as on 1 or faster; on 4 and 8 it
            # can be 2-4x slower (FROSTT Uber, 4392 x 1.96e6: 4.0 / 4.0 / 17 / 15 ms at 1 / 2 /
            # 4 / 8 threads; 1e4 x 1e6 with 1e7 entries: 18 / 18 / 11 / 39 ms). MAX and MIN
            # scale instead (5e6 x 5e6 with 1e7 entries: 122 / 120 / 51 / 50 ms; 1e7 x 1e7
            # with 1e8: 0.66 / 0.66 / 0.32 / 0.28 s; no loss on the others), so they keep
            # every thread. (SuiteSparse 10.5.1, arm64.)
            capped = wide and getattr(op, "name", None) not in {"max", "min"}
            if census.on:
                census.note("reduce", "row group: reduce_columnwise" + (", 2 threads" if capped
                            and nthreads > 2 else ""), nvals=nvals)  # fmt: skip
            v = Vector(dtype, M._ncols)
            v(**(dict(opts, nthreads=min(2, nthreads)) if capped else opts)) << M.reduce_columnwise(
                op
            )
            M = _vector_to_row(v)
            layout = Layout((), cols)
        else:
            nv = M._nvals
            sel, reb = _row_selector_cost(M, tl.prod(shape, rows), ti), tl.rebuild_cost(nv, ti)
            if census.on:
                j = census.choose("reduce", [sel, reb], ["row selector", "rebuild"],
                                  int(sel > reb), nvals=nv)  # fmt: skip
                sel, reb = (0, 1) if j == 0 else (1, 0)
            if sel > reb:
                return _reduce_rebuild(M, shape, layout, A, op)
            kept = tuple(i for i, x in enumerate(rows) if x not in Ar)
            S = _row_reducer(M, shape, rows, kept, _selector_dtype(op))
            # The padding column (see _selector_mxm) is M's: a view of M with one more column
            V = None
            if not exact and tl.stored_format(M) in {"csr", "hypercsr"}:
                V = tl.padded_view(M)
                if V is None:  # pragma: no cover (2^31 columns)
                    V = M.dup()
                    V.resize(M._nrows, M._ncols + 1)
            sr = _selector_semiring(op, binary.second)
            out = _selector_mxm(S, M if V is None else V, sr, dtype, opts, V is not None)
            del V  # (before M, whose arrays it reads)
            M = out
            layout = Layout(tuple(rows[i] for i in kept), cols)
        A -= Ar
    return M, layout


def _reduce_rebuild(M, shape, layout, A, op):
    """Reduce by rebuilding with the monoid as ``dup_op``: best when rows hold about one
    entry each, and the path for UDTs.
    """
    coords, values = tl.coords_of(M, shape, layout.state)
    # The layout may already lack axes reduced by earlier steps
    kept = [x for x in sorted(layout.seq) if x not in A]
    kshape = tuple(shape[x] for x in kept)
    lay = tl.default_layout(kshape, values.size)
    rows = tuple(kept[i] for i in lay.rows)
    cols = tuple(kept[i] for i in lay.cols)
    r = tl.ravel(coords, shape, rows, n=values.size)
    c = tl.ravel(coords, shape, cols, n=values.size)
    out = tl.from_coo(
        r,
        c,
        values,
        op.return_type,
        tl.prod(shape, rows),
        tl.prod(shape, cols),
        dup_op=op.binaryop,
    )
    return out, Layout(rows, cols)


def _reduce_scalar_recipe(updater, a, op, allow_empty):
    updater << a._M.reduce_scalar(op, allow_empty=allow_empty)


# isort: off
from .tensor import (  # noqa: E402
    TensorExpression,
    _fixed_layout,
    _healthy,
    _insert_unit_axes,
    _normalize_axes,
    _renumber,
    _row_selector_cost,
    _shape_selector_cost,
    _store,
    _ti,
)

# isort: on

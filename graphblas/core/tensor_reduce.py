"""Reductions of Tensors over axes: selectors, reduce_rowwise / columnwise, rebuilds.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import math

from .. import binary, monoid, semiring
from ..dtypes import BOOL
from . import tensor_census as census
from . import tensor_layout as tl
from .operator import get_semiring, get_typed_op
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
    selector, whose entries are True.
    """
    if op is monoid.lor[BOOL]:
        # land(True, x) is x, as first and second give, and SuiteSparse 10 has a factory
        # kernel for lor_land only: BOOL has no factory FIRST or SECOND semiring (Tensor.any
        # ran about 2x slower with the JIT at "off" or "run")
        return semiring.lor_land[BOOL]
    return get_semiring(op, mult)


def _row_reducer(M, shape, rows, kept):
    """S with S @ M reducing the row axes not in ``kept`` (positions in ``rows``)."""
    if tl.stored_format(M) == "hypercsr":
        return tl.rows_selector(M, shape, rows, kept)
    return tl.selector(tuple(shape[x] for x in rows), kept)


def _vector_to_row(v):
    return tl.steal_vector_as_row(v)


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
    A = set(A)
    while A:
        rows, cols = layout
        Ac = A & set(cols)
        if Ac:
            if Ac == set(cols):
                if census.on:
                    census.note("reduce", "column group: reduce_rowwise", nvals=M._nvals)
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
                St = tl.selector(gshape, kept, transposed=True)
                out = tl.new_matrix(dtype, M._nrows, St._ncols)
                out(**opts) << M.mxm(St, _selector_semiring(op, binary.first))
                M = out
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
            if census.on:
                census.note("reduce", "row group: reduce_columnwise" + (", 2 threads" if wide
                            and nthreads > 2 else ""), nvals=nvals)  # fmt: skip
            v = Vector(dtype, M._ncols)
            # Into many outputs, 2 threads scatter as fast as 1 or faster; 4 and 8 can be
            # 3-4x slower (Uber 4.0 / 4.9 / 17 / 15 ms at 2 / 1 / 4 / 8 threads)
            v(**(dict(opts, nthreads=min(2, nthreads)) if wide else opts)) << M.reduce_columnwise(
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
            S = _row_reducer(M, shape, rows, kept)
            out = tl.new_matrix(dtype, S._nrows, M._ncols)
            out(**opts) << S.mxm(M, _selector_semiring(op, binary.second))
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

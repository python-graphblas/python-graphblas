"""Reductions of Tensors over axes: selectors, reduce_rowwise / columnwise, rebuilds.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import math
from functools import partial

import numpy as np

from .. import binary, monoid, semiring
from ..dtypes import BOOL, INT64, UINT32, UINT64
from . import tensor_census as census
from . import tensor_layout as tl
from .operator import get_semiring, get_typed_op
from .operator.monoid import TypedBuiltinMonoid
from .scalar import Scalar
from .tensor_layout import Layout
from .vector import Vector


def _reduce_aggregator(t, op, axis, keepdims):
    """``reduce`` with an aggregator, computed as Matrix aggregations compute it.

    An aggregator with a monoid reduces with it. One with a semiring applies the semiring's
    multiply, with the aggregator's initial value, to every entry, then reduces with the
    semiring's monoid, then applies its finalize op. An aggregator that picks another by the
    kind of number (``agg.L1norm``) reduces with the one for the Tensor's dtype, and an
    applybegin op, if any, is applied first. A composite one (``agg.mean``) computes each of
    its aggregations now and returns its finalize function's expression of them (for
    ``mean``, ``sum / count``), which Matrix's finalize functions build from infix operators
    that work on Tensors too. The ones of positions go to ``_reduce_positional``.
    """
    from .operator.agg import TypedAggregator, _by_kind

    agg = op.parent if isinstance(op, TypedAggregator) else op
    if isinstance(agg._custom, partial) and agg._custom.func is _by_kind:
        aggs = agg._custom.keywords["aggs"]
        agg = aggs.get(t.dtype.np_type.kind, aggs["other"])
    if isinstance(agg._custom, partial) and (kind := _positional_kind(agg._custom)) is not None:
        return _reduce_positional(t, agg, kind, axis, keepdims)
    if agg._monoid is not None:
        return _reduce_expr(t, agg._monoid, axis, keepdims)
    if agg._composite is not None:
        typed = agg[t.dtype]  # (a KeyError for the types Matrix refuses, as peak_to_peak of FC64)
        x = t
        if t.dtype.np_type.kind in "biu" and typed.return_type.np_type.kind in "fc":
            # Sum integer and BOOL values in the float result type, as numpy accumulates
            # means and variances in float64: in the values' type the sums, squares and
            # products wrap (decisions.md, 32)
            x = t.dup(typed.return_type)
        parts = [_reduce_aggregator(x, part, axis, keepdims).new() for part in agg._composite]
        # Matrix stores the finalize expression into an output of the aggregator's return type,
        # which can be narrower than the expression's own (FP32 for the mean of FP32 values,
        # where sum / count gives FP64), so this does too.
        return _aggregator_expr(t, typed, agg._finalize(*parts, {}), axis)
    if agg._semiring is None or agg._custom is not None:
        raise NotImplementedError(
            f"Tensor.reduce supports aggregators made of a monoid or a semiring; not "
            f"agg.{agg.name}"
        )
    # Type the semiring as Matrix aggregations do, by the values' type and the initial value's,
    # so the result type is the one Matrix returns (INT64 for count and exists). Integer powers
    # of INT64 values stay INT64 (logaddexp2 of -3 is log2(0)).
    agg[t.dtype]  # (a KeyError for the types Matrix refuses, as sum_of_squares of BOOL)
    sr = get_typed_op(agg._semiring, t.dtype, agg._initdtype)
    if agg._applybegin is not None:  # Such as abs, for L1norm without Numba
        t = t.apply(agg._applybegin).new()
    if agg._switch:
        x = t.apply(sr.binaryop, left=agg._initval).new()
    else:
        x = t.apply(sr.binaryop, right=agg._initval).new()
    rv = _reduce_expr(x, sr.monoid, axis, keepdims)
    if agg._finalize is not None:
        rv = rv.new().apply(agg._finalize[sr.return_type])
    return rv


def _aggregator_expr(t, typed, inner, axis):
    """``inner``, an expression that computes ``t.reduce(typed, axis)``, as one named for the
    typed aggregator and of its result type.
    """
    axis_repr = "None" if axis is None else repr(axis)
    return TensorExpression(
        "reduce", _delegate_recipe, [t], (inner,), dtype=typed.return_type, shape=inner.shape,
        op=typed, expr_repr=f"{{0.name}}.reduce({{op}}, axis={axis_repr})",
    )  # fmt: skip


def _delegate_recipe(updater, inner):
    inner.args[-2](updater, *inner.args[-1])


def _positional_kind(custom):
    """``(what, last)`` for an aggregator of positions, from its recipe in ``operator.agg``:
    ``what`` is "argmin", "argmax", "value" (``first``, ``last``) or "index"
    (``first_index``, ``last_index``). None for any other recipe.
    """
    from .operator.agg import _argminmax, _first_last, _first_last_index

    if custom.func is _argminmax:
        return ("argmin" if custom.keywords["monoid"] is monoid.min else "argmax"), False
    if custom.func is _first_last:
        return "value", custom.keywords["last"]
    if custom.func is _first_last_index:
        return "index", custom.keywords["last"]
    return None


def _reduce_positional(t, agg, kind, axis, keepdims):
    """``reduce`` with ``argmin``, ``argmax``, ``first``, ``last``, ``first_index`` or
    ``last_index``, as Matrix's ``reduce_rowwise`` computes them along a row.

    Each entry's position is its C-order index over the reduced axes (``_positions``). The
    index aggregators take one axis, as numpy's ``argmin`` and Matrix's (which refuses
    ``reduce_scalar``) do: ``first_index`` is the least position, and ``argmin`` the least
    position among the entries equal to their fiber's minimum (``eq``, so a NaN never
    matches). ``first`` and ``last`` take any axes: the value at the least or greatest
    position, so over every axis they return what ``reduce_scalar`` returns for a Matrix.
    """
    what, last = kind
    axes = _normalize_axes(axis, t.ndim)
    typed = agg[t.dtype]  # (a KeyError for the types Matrix refuses, as argmin of FC64)
    if what != "value" and len(axes) != 1:
        raise ValueError(
            f"Tensor.reduce with agg.{agg.name} takes one axis, as numpy's argmin does; got "
            f"axis={axis!r} for a {t.ndim}-d Tensor"
        )
    end = monoid.max if last else monoid.min
    pos = _positions(t, axes)
    if what == "index":
        return _aggregator_expr(t, typed, _reduce_expr(pos, end, axes, keepdims), axis)
    if what == "value":
        hit = pos.ewise_mult(_reduce_expr(pos, end, axes, True).new(), binary.eq).new()
        picked = t
    else:
        extreme = monoid.min if what == "argmin" else monoid.max
        hit = t.ewise_mult(_reduce_expr(t, extreme, axes, True).new(), binary.eq).new()
        picked = pos
    sub = Tensor(picked.dtype, picked._shape)
    sub(hit.V) << picked  # the entries at the hits: one per fiber for first and last
    inner = _reduce_expr(sub, monoid.any if what == "value" else monoid.min, axes, keepdims)
    return _aggregator_expr(t, typed, inner, axis)


def _positions(t, axes):
    """An INT64 Tensor with ``t``'s pattern and layout whose values are their entries' C-order
    index over ``axes``: one pass over the stored indices, and a build from keys already in
    storage order.
    """
    if math.prod(t._shape[a] for a in axes) >= 2**63:
        raise ValueError(
            f"the axes {axes} have {math.prod(t._shape[a] for a in axes)} positions, more "
            f"than an INT64 index can count"
        )
    M = t._M
    seq, p = t._layout.state
    rows, cols, _ = M.to_coo(values=False, sort=False)
    coords = [None] * t.ndim
    tl.unravel(rows.view(np.int64), t._shape, seq[:p], coords)
    tl.unravel(cols.view(np.int64), t._shape, seq[p:], coords)
    pos = np.zeros(rows.size, np.int64)
    for a in axes:
        pos = pos * t._shape[a] + coords[a]
    P = tl.from_coo(rows, cols, pos, INT64, M._nrows, M._ncols)
    return Tensor._from_M(P, t._shape, t._layout)


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
    """The semiring of monoid ``op`` and ``mult`` (first or second) for a product with a
    selector. The selector's entries are True, or for a row selector (``second``) 1 of the
    type ``_selector_dtype`` returns.
    """
    if op is monoid.lor[BOOL]:
        # land(True, x) is x, as with first and second. SuiteSparse 10 has a factory kernel
        # for lor_land but none for a BOOL FIRST or SECOND semiring (with those, Tensor.any
        # ran about 2x slower with the JIT "off" or at "run")
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
    """The selector S for which S @ M reduces the row axes not in ``kept`` (positions in
    ``rows``).
    """
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
    product ``left`` is then a sparse selector. So forcing saxpy changes no method and
    measured no cost (2026-11-10); it guards against a change in that choice. The padding
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
    """Reduce by rebuilding with the monoid as ``dup_op``: best when rows have about one
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


def _reduce_scalar_aggregator_recipe(updater, a, op):
    updater << a.reduce(op).new().to_scalar()


# isort: off
from .tensor import (  # noqa: E402
    Tensor,
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

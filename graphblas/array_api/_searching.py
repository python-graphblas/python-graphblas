"""Searching functions: argmax, argmin, count_nonzero, nonzero, searchsorted, where."""

import builtins
import math

import numpy as np

from ..core.tensor import Tensor
from ..core.vector import Vector
from . import _utils
from ._array import Array
from ._dtypes import check_supported
from ._utils import gb_dtype

__all__ = ["argmax", "argmin", "count_nonzero", "nonzero", "searchsorted", "where"]


def _operand(x, name):
    if isinstance(x, Array):
        return x
    if isinstance(x, (np.generic, np.ndarray)):
        from ._creation import asarray

        return asarray(x)
    if isinstance(x, (builtins.bool, int, float, complex)):
        return x
    raise TypeError(f"{name} needs arrays or Python scalars, not {type(x).__name__}")


def _probe(x):
    return np.empty(0, x.dtype) if isinstance(x, Array) else x


def _value(x):
    return np.asarray(x._fill) if isinstance(x, Array) else x


def where(condition, x1, x2, /):
    """``x1`` where ``condition`` is true, else ``x2`` (broadcast); numpy's ``where``.

    Where the condition stores nothing it reads its fill, so the result there is the operand
    that fill selects. When that operand needs no broadcast (or is a scalar, or stores
    nothing), the result is that operand with the condition's stored positions written over:
    O(entries of the condition and of that operand), the other operand read only at the
    condition's entries (``xp.where(graph, labels, big)``, with ``labels`` broadcast along an
    axis of ``graph``, stores ``graph``'s entries). Otherwise the entries are the union of
    the three arrays' entries, with the broadcast operands' entries replicated.
    """
    cond = _operand(condition, "where")
    if not isinstance(cond, Array):
        from ._creation import asarray

        cond = asarray(cond)
    a = _operand(x1, "where")
    b = _operand(x2, "where")
    rdt = check_supported(np.result_type(_probe(a), _probe(b)), "where")
    fill = np.where(np.asarray(cond._fill, dtype=bool), _value(a), _value(b)).astype(rdt)[()]
    shapes = [x.shape for x in (cond, a, b) if isinstance(x, Array)]
    shape = np.broadcast_shapes(*shapes)
    arrays = [x for x in (cond, a, b) if isinstance(x, Array) and x.ndim]
    if not shape or all(x._T().nvals == 0 for x in arrays):
        return Array._new(Tensor(gb_dtype(rdt), shape), fill)
    if (R := _where_at_condition(cond, a, b, shape, rdt)) is not None:
        return Array._new(R, fill)
    tensors = iter(_utils.aligned(arrays, shape))
    slots = [next(tensors) if isinstance(x, Array) and x.ndim else None for x in (cond, a, b)]
    constants = [_value(x) for x in (cond, a, b)]

    def func(*values):
        it = iter(values)
        c, v1, v2 = (
            next(it) if t is not None else k for t, k in zip(slots, constants, strict=True)
        )
        return np.where(np.asarray(c, dtype=bool), v1, v2).astype(rdt, copy=False)

    R = _utils.map_values(func, [t for t in slots if t is not None], rdt)
    return Array._new(R, fill)


def _where_at_condition(cond, a, b, shape, rdt):
    """``where(cond, a, b)``'s Tensor built from the operand ``cond``'s fill selects, with
    ``cond``'s stored positions written over (see ``where``), or None when that operand
    would have to be broadcast over stored entries, or ``cond`` itself broadcasts.
    """
    from .. import binary, select
    from ..dtypes import BOOL

    Tc = cond._T()
    if Tc.shape != shape:
        return None
    base = a if bool(cond._fill) else b
    if isinstance(base, Array) and base.ndim and base.shape != shape and base._T().nvals:
        return None
    gdt = gb_dtype(rdt)
    if not (isinstance(base, Array) and base.ndim and base._T().nvals):
        # The base holds only the result's fill: the condition's entries that select it
        # (those equal to its fill) would store the fill, so only the others are kept, and
        # the other operand is read there only (where(x > t, x, 0) stores x's entries > t)
        other = b if bool(cond._fill) else a
        if cond.dtype.kind != "c":
            # All in GraphBLAS, in parallel: the entries that select ``other`` (the truthy
            # ones, or the falsy ones under a truthy fill; NaN is truthy), holding its fill,
            # then its stored values written over. A broadcast ``other`` is read through a
            # diagonal scaling, without replicating it (xp.where(graph, labels, big)).
            pick = select.valueeq if bool(cond._fill) else select.valuene
            sel = Tc.select(pick, False if Tc.dtype == BOOL else 0).new()
            second = binary.second[gdt]
            R = sel.apply(second, right=_utils.gb_scalar(_value(other), rdt)).new(dtype=gdt)
            if isinstance(other, Array) and other.ndim and other._T().nvals:
                R(accum=second) << sel.ewise_mult(other._T(), second)
            return R
        coords, c = Tc.to_coo()
        c = np.asarray(c, dtype=bool)
        keep = c != bool(cond._fill)
        coords = tuple(i[keep] for i in coords)
        if isinstance(other, Array) and other.ndim:
            V = _values_at(coords, other, shape) if len(coords[0]) else c[keep]
        else:
            V = np.full(len(coords[0]), _value(other))
        V = np.asarray(V).astype(rdt, copy=False)
        return Tensor.from_coo(coords, V, gdt, shape=shape) if len(V) else Tensor(gdt, shape)
    coords, c = Tc.to_coo()
    c = np.asarray(c, dtype=bool)
    values = [
        (
            (_values_at(coords, x, shape) if len(c) else c)
            if isinstance(x, Array) and x.ndim
            else _value(x)
        )
        for x in (a, b)
    ]
    V = np.where(c, *values).astype(rdt, copy=False)
    R = Tensor.from_coo(coords, V, gdt, shape=shape) if len(c) else Tensor(gdt, shape)
    return base._T().ewise_add(R, binary.second[gdt]).new(dtype=gdt)


def _values_at(coords, x, shape):
    """The values of ``x`` (an Array that broadcasts to ``shape``) at ``coords`` of
    ``shape``: its stored value, or its fill. O(len(coords)), read from x's own entries.
    """
    lead = len(shape) - x.ndim
    n = len(coords[0])
    # (a broadcast axis of x has size 1: index 0 there)
    index = tuple(
        coords[lead + i] if size != 1 else np.zeros(n, np.int64) for i, size in enumerate(x.shape)
    )
    return x._T().vindex[index].to_dense(fill_value=x._fill)


def _check_array(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs an Array, not {type(x).__name__}")


def _arg(name, x, axis, keepdims):
    """The first position of the extreme value of each fiber (argmax, argmin), NaN counting
    as the extreme (numpy's rule), computed in GraphBLAS on the stored entries.

    With the fibers as the rows (or columns) of the stored Matrix: the extreme of each fiber
    (a reduction), the entries equal to it (the Matrix scaled by the extremes as a diagonal,
    under ``eq``), and the smallest position among those (a positional reduction); likewise
    the first stored NaN. A fiber with an unstored position holds the fill there: when the
    fill beats or ties the stored extreme, or is NaN, the fiber's first unstored position
    takes part. That is 0 unless the fiber stores position 0, and otherwise is found from
    that fiber's entries alone. Each step is a pass over the entries; nothing is sorted.
    """
    from ._manipulation import _normalize_axis, expand_dims, reshape

    _check_array(x, name)
    if x.dtype.kind == "c":
        raise TypeError(f"{name} needs a real-valued array, not {x.dtype}")
    if axis is None:
        out = _arg(name, reshape(x, (-1,)), 0, False)
        if keepdims:
            out = reshape(out, (1,) * x.ndim)
        return out
    axis = _normalize_axis(axis, x.ndim)
    L = x.shape[axis]
    if L == 0:
        raise ValueError(f"{name} of an empty axis")
    out_shape = tuple(n for a, n in enumerate(x.shape) if a != axis)
    T = x._T()
    if T.nvals == 0 or math.prod(out_shape) == 0:
        out = Array._new(Tensor(gb_dtype(np.int64), out_shape), 0)
        return expand_dims(out, axis) if keepdims else out
    R = _arg_fibers(name, x, T, axis)
    if out_shape:
        out = Array._new(Tensor.from_vector(R).reshape(out_shape), 0)
    else:
        out = Array._new(_utils.empty_0d(np.int64), np.int64(R.get(0, 0)))
    return expand_dims(out, axis) if keepdims else out


def _arg_fibers(name, x, T, axis):
    """A Vector over the fibers of ``T`` along ``axis`` (in C order) holding the first
    position of each one's extreme (see ``_arg``), storing only the positions that are not 0.
    """
    from .. import binary, monoid, select, semiring, unary
    from ..dtypes import BOOL, INT64

    L = T.shape[axis]
    others = tuple(a for a in range(T.ndim) if a != axis)
    # The fibers as the rows of M (byrow) or its columns: as stored when the layout allows
    if T._layout == (others, (axis,)):
        M, byrow = T._M, True
    elif T._layout == ((axis,), others):
        M, byrow = T._M, False
    else:
        M, byrow = T.to_matrix(rows=others), True
    nfibers = M.nrows if byrow else M.ncols
    if M.dtype == BOOL:
        top = monoid.lor if name == "argmax" else monoid.land
    else:
        top = monoid.max if name == "argmax" else monoid.min
    every = Vector(BOOL, L)
    every << True  # (iso: no values stored)

    def first(P, mask=None):
        # (the smallest position along each fiber of P's entries)
        v = Vector(INT64, nfibers)
        if byrow:
            v(mask=mask) << P.mxv(every, semiring.ss.min_firstj)
        else:
            v(mask=mask) << every.vxm(P, semiring.ss.min_secondi)
        return v

    ext = (M.reduce_rowwise(top) if byrow else M.reduce_columnwise(top)).new()  # (skips NaN)
    # The entries equal to their fiber's extreme. With a value at every diagonal position,
    # SuiteSparse runs the product as a row (column) scaling, one pass over M.
    if nfibers <= 4 * M.nvals:
        diag = Vector(ext.dtype, nfibers)
        diag << 0  # (the value at a fiber with no entries scales nothing)
        diag(accum=binary.second) << ext
    else:
        diag = ext
    D = diag.diag()
    eq = semiring.any_eq[M.dtype]
    hits = (D.mxm(M, eq) if byrow else M.mxm(D, eq)).new()
    R = first(hits.select(select.valueeq, True).new())
    del hits
    # The first stored NaN, which beats every value
    nan = None
    if x.dtype.kind == "f" and _utils.fact(x, "nan"):
        nan = first(M.apply(unary.isnan).new().select(select.valueeq, True).new())
        R(accum=binary.second) << nan
    # The fibers where the fill takes part, if they have an unstored position: those whose
    # stored extreme the fill beats or ties (and that store no NaN), or all, for a NaN fill
    fill = x._fill
    fill_nan = x.dtype.kind == "f" and np.isnan(fill)
    if fill_nan:
        need = ext
    else:
        beats = select.valuelt if name == "argmax" else select.valuegt
        need = ext.select(beats, fill).new()
        need(accum=binary.first) << ext.select(select.valueeq, fill)
        if nan is not None:
            keep = Vector(need.dtype, nfibers)
            keep(~nan.S, replace=True) << need
            need = keep
    if need.nvals:
        # A fiber whose first entry is past position 0 has its first unstored position at 0
        firsts = first(M, need.S)
        R(~firsts.select(select.valuegt, 0).new().S, replace=True) << R  # (there: 0)
        at_0 = firsts.select(select.valueeq, 0).new()
        if at_0.nvals:
            _fill_at_0(name, M, byrow, R, ext, nan, fill, fill_nan, at_0, L)
    return R.select(select.valuene, 0).new()


def _fill_at_0(name, M, byrow, R, ext, nan, fill, fill_nan, at_0, L):
    """Update R at the fibers ``at_0`` (those that take the fill and store position 0) with
    the fill's first position, read from those fibers' entries: the first position not equal
    to its rank among the fiber's entries, or the number of entries when they fill a prefix
    (none when that number is L: the fiber stores every position).
    """
    fibers = at_0.to_coo(values=False)[0]
    sub = M[fibers, :].new() if byrow else M[:, fibers].new().T.new()
    r, c, _ = sub.to_coo()  # (by fiber, positions sorted)
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    counts = np.diff(np.r_[starts, len(r)])
    rank = np.arange(len(r)) - np.repeat(starts, counts)
    gap = np.minimum.reduceat(np.where(c != rank, rank, np.repeat(counts, counts)), starts)
    best = R[fibers].new().to_dense(fill_value=L)
    if fill_nan:
        stored_nan = nan[fibers].new().to_dense(fill_value=L) if nan is not None else L
        new = np.where(gap < L, np.minimum(gap, stored_nan), best)
    else:
        e = ext[fibers].new().to_dense()
        wins = (fill > e) if name == "argmax" else (fill < e)
        new = np.where(gap < L, np.where(wins, gap, np.minimum(gap, best)), best)
    R[fibers] << Vector.from_coo(np.arange(len(fibers)), new.astype(np.int64), size=len(fibers))


def argmax(x, /, *, axis=None, keepdims=False):
    return _arg("argmax", x, axis, keepdims)


def argmin(x, /, *, axis=None, keepdims=False):
    return _arg("argmin", x, axis, keepdims)


def count_nonzero(x, /, *, axis=None, keepdims=False):
    from ._elementwise import not_equal
    from ._statistical import sum as sum_

    _check_array(x, "count_nonzero")
    return sum_(not_equal(x, 0), axis=axis, dtype=np.dtype(np.int64), keepdims=keepdims)


def nonzero(x, /):
    """The coordinates (C order) of the nonzero positions: O(entries) when the fill is zero,
    O(size) otherwise (every unstored position is nonzero).
    """
    from ._creation import asarray
    from ._elementwise import not_equal

    _check_array(x, "nonzero")
    if x.ndim == 0:
        raise ValueError("nonzero needs an array of at least one dimension")
    nz = not_equal(x, 0)
    if nz._fill:
        _utils.check_dense(x.size, "nonzero of an array whose fill is not zero")
        coords = np.nonzero(nz.todense())
    else:
        T = nz._T()
        if T.nvals == 0:
            coords = tuple(np.empty(0, np.int64) for _ in range(x.ndim))
        else:
            coords = T.select("==", True).new().to_coo(values=False)[0]
    return tuple(asarray(np.asarray(c, dtype=np.int64)) for c in coords)


def searchsorted(x1, x2, /, *, side="left", sorter=None):
    """Insertion positions in the sorted 1-d ``x1``: the stored values are searched, and the
    run of the fill counted where it sorts. O(entries of x1 + entries of x2).
    """
    from ._creation import asarray
    from ._indexing import take

    _check_array(x1, "searchsorted")
    if x1.ndim != 1:
        raise ValueError("searchsorted needs a 1-d x1")
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    if sorter is not None:
        x1 = take(x1, sorter)
    T = x1._T()
    stored = T.to_coo()[1] if T.nvals else np.empty(0, x1.dtype)
    k = x1.size - T.nvals
    fill = np.asarray([x1._fill])

    def count(v):
        n = np.searchsorted(stored, v, side=side)
        return (n + k * np.searchsorted(fill, v, side=side)).astype(np.int64)

    if not isinstance(x2, Array):
        x2 = asarray(x2)
    out_fill = count(np.asarray(x2._fill))
    if x2.ndim == 0 or x2._T().nvals == 0:
        return Array._new(Tensor(gb_dtype(np.int64), x2.shape), out_fill)
    return Array._new(_utils.map_values(count, [x2._T()], np.dtype(np.int64)), out_fill)

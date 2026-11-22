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
    """``x1`` where ``condition`` is true and ``x2`` elsewhere, broadcast as numpy's ``where``.

    Where the condition stores nothing, its fill decides, so the result there is the operand
    that the fill selects. When the condition needs no broadcast, and neither does that
    operand (or it is a scalar, or stores nothing), the result is that operand with the
    other's values written at the condition's entries that select the other. This costs
    O(entries of the condition and of that operand), and reads the other operand only at
    those entries. For example, ``xp.where(graph, labels, big)``, with ``labels`` broadcast
    along an axis of ``graph``, stores one entry per true entry of ``graph``.

    Otherwise the result stores the union of the three arrays' entries, with the entries of
    the broadcast operands replicated.
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
    """The Tensor of ``where(cond, a, b)``, built from the operand that ``cond``'s fill selects
    (see ``where``). None when ``cond`` broadcasts, or when that operand has entries and
    would have to be broadcast.
    """
    from .. import binary

    Tc = cond._T()
    if Tc.shape != shape:
        return None
    base = a if bool(cond._fill) else b
    if isinstance(base, Array) and base.ndim and base.shape != shape and base._T().nvals:
        return None
    gdt = gb_dtype(rdt)
    has_base = isinstance(base, Array) and base.ndim and base._T().nvals
    other = b if bool(cond._fill) else a
    if cond.dtype.kind != "c":
        # This path runs in GraphBLAS, in parallel. At the condition's entries that select
        # the base (those whose truth matches its fill's), the result has the base's values,
        # so the base supplies them. Only the other entries are computed, and only there is
        # the other operand read. So where(x > t, x, 0) stores x's entries above t, and
        # where(frontier, k, level) is a select and a union.
        R = _selecting(Tc, cond, other, shape, rdt)
        if has_base:
            R = base._T().ewise_add(R, binary.second[gdt]).new(dtype=gdt)
        return R
    if not has_base:
        # (a complex condition is converted to bool by numpy)
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


def _selecting(Tc, cond, other, shape, rdt):
    """The entries of ``Tc`` (cond's Tensor, not complex) that select ``other``, with
    ``other``'s values.

    Those are the truthy entries, or the falsy ones when the fill is truthy; NaN is truthy.
    Each gets ``other``'s fill, and then ``other``'s stored values are written over. A
    broadcast ``other`` is read through a diagonal scaling, without replicating it (as in
    ``xp.where(graph, labels, big)``).
    """
    from .. import binary, select
    from ..dtypes import BOOL

    gdt = gb_dtype(rdt)
    pick = select.valueeq if bool(cond._fill) else select.valuene
    sel = Tc.select(pick, False if Tc.dtype == BOOL else 0).new()
    second = binary.second[gdt]
    To = other._T() if isinstance(other, Array) and other.ndim else None
    if To is not None and other.shape != shape and other.size <= 4 * sel.nvals:
        # Stored at every position, the broadcast operand makes the product a diagonal
        # scaling, one pass. With gaps in the diagonal, SuiteSparse runs a general multiply,
        # 17x slower at 1e7 entries. Any fill but other's makes with_fill store every position.
        if To.nvals < other.size:
            To = _utils.with_fill(other, 0 if other._fill else 1)._T()
        return sel.ewise_mult(To, second).new(dtype=gdt)
    R = sel.apply(second, right=_utils.gb_scalar(_value(other), rdt)).new(dtype=gdt)
    if To is not None and To.nvals:
        R(accum=second) << sel.ewise_mult(To, second)
    return R


def _values_at(coords, x, shape):
    """The values of ``x`` (an Array that broadcasts to ``shape``) at ``coords`` of
    ``shape``: its stored value, or its fill. This costs O(number of coordinates) and reads
    only x's own entries.
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
    """The first position of the extreme value of each fiber (argmax, argmin), with NaN
    counting as the extreme, as in numpy. It is computed in GraphBLAS on the stored entries.

    The fibers are the rows (or columns) of the stored Matrix. A reduction finds the extreme
    of each fiber. Scaling the Matrix by the extremes as a diagonal, with the ``any_eq``
    semiring, finds the entries equal to their fiber's extreme, and a positional reduction
    finds the smallest position among those. The first stored NaN is found the same way.

    A fiber's unstored positions have the fill. When the fill beats or ties the stored
    extreme, or is NaN, the fiber's first unstored position takes part. That position is 0
    unless the fiber stores position 0; otherwise it is found from that fiber's entries
    alone. Each step is a pass over the entries, and nothing is sorted.
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
    """A Vector over the fibers of ``T`` along ``axis`` (in C order) with the first position
    of each fiber's extreme (see ``_arg``). It stores only the positions that are not 0.
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
    every << True  # (iso and full: O(1) memory)

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
    # The fibers where the fill takes part, if they have an unstored position: all of them
    # for a NaN fill, else those that store no NaN and whose stored extreme the fill beats or
    # ties
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
    """Update R at the fibers ``at_0`` (those where the fill takes part and position 0 is
    stored) with the fill's first position, read from those fibers' entries. That is the
    rank of the fiber's first entry whose position differs from its rank, or the number of
    entries when they fill a prefix. When that number is L, the fiber stores every position
    and the fill does not take part.
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
    """The coordinates (C order) of the nonzero positions. This costs O(entries) when the
    fill is zero, and O(size) otherwise, since then every unstored position is nonzero.
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
    """The positions where ``x2``'s values would be inserted into the sorted 1-d ``x1``.

    Only ``x1``'s stored values are searched; its unstored positions count as one run of the
    fill, where the fill sorts. This costs O(entries of x1 + entries of x2).
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

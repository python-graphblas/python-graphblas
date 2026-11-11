"""Searching functions: argmax, argmin, count_nonzero, nonzero, searchsorted, where."""

import builtins
import math

import numpy as np

from ..core.tensor import Tensor
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
    coords, c = Tc.to_coo()
    c = np.asarray(c, dtype=bool)
    gdt = gb_dtype(rdt)
    if not (isinstance(base, Array) and base.ndim and base._T().nvals):
        # The base holds only the result's fill: the condition's entries that select it
        # (those equal to its fill) would store the fill, so only the others are kept, and
        # the other operand is read there only (where(x > t, x, 0) stores x's entries > t)
        other = b if bool(cond._fill) else a
        if Tc.dtype == BOOL and isinstance(other, Array) and other.shape == shape:
            # (all in GraphBLAS, in parallel: the entries that select ``other``, holding
            # its value, or its fill where it stores nothing)
            sel = Tc.select(select.valueeq, not bool(cond._fill)).new()
            second = binary.second[gdt]
            R = sel.apply(second, right=_utils.gb_scalar(other._fill, rdt)).new(dtype=gdt)
            R(accum=second) << sel.ewise_mult(other._T(), second)
            return R
        keep = c != bool(cond._fill)
        coords = tuple(i[keep] for i in coords)
        if isinstance(other, Array) and other.ndim:
            V = _values_at(coords, other, shape) if len(coords[0]) else c[keep]
        else:
            V = np.full(len(coords[0]), _value(other))
        V = np.asarray(V).astype(rdt, copy=False)
        return Tensor.from_coo(coords, V, gdt, shape=shape) if len(V) else Tensor(gdt, shape)
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
    as the extreme (numpy's rule). O(entries log entries): the stored entries of each fiber
    are ranked, and the fill competes once if the fiber has an unstored position.
    """
    from ._manipulation import _normalize_axis, expand_dims, permute_dims, reshape

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
    others = tuple(a for a in range(x.ndim) if a != axis)
    T = permute_dims(x, (*others, axis))._T()
    nfibers = math.prod(out_shape)
    if T.nvals == 0 or nfibers == 0:
        out = Array._new(Tensor(gb_dtype(np.int64), out_shape), 0)
        return expand_dims(out, axis) if keepdims else out
    coords, values = T.to_coo()  # (C order: by fiber, then by position along the axis)
    fiber = (
        np.ravel_multi_index(coords[:-1], out_shape)
        if out_shape
        else np.zeros(len(values), np.int64)
    )
    pos = coords[-1]
    starts = np.flatnonzero(np.r_[True, fiber[1:] != fiber[:-1]])
    counts = np.diff(np.r_[starts, len(fiber)])
    # Rank the values, NaN first: then the lowest key, earliest position, wins
    isnan = np.isnan(values) if values.dtype.kind == "f" else np.zeros(len(values), bool)
    _, rank = np.unique(values, return_inverse=True)
    rank = rank.reshape(-1)
    if name == "argmax":
        rank = rank.max(initial=0) - rank
    key = np.where(isnan, -1, rank)
    order = np.lexsort((pos, key, fiber))
    best = order[starts]
    best_pos, best_val = pos[best], values[best]
    # The first position each fiber does not store (L if it stores them all)
    within = np.arange(len(pos)) - np.repeat(starts, counts)
    gap = pos != within
    first_gap = np.where(
        np.add.reduceat(gap, starts) > 0,
        _first_true(gap, starts, counts, within),
        counts,
    )
    fill = x._fill
    with np.errstate(invalid="ignore"):
        if fill != fill:  # (NaN)
            better = ~np.isnan(best_val) if best_val.dtype.kind == "f" else np.ones(len(best), bool)
            tie = ~better
        else:
            best_is_nan = (
                np.isnan(best_val) if best_val.dtype.kind == "f" else np.zeros(len(best), bool)
            )
            beats = (fill > best_val) if name == "argmax" else (fill < best_val)
            better = beats & ~best_is_nan
            tie = (fill == best_val) & ~best_is_nan
    unstored = first_gap < L
    result = np.where(
        unstored & better,
        first_gap,
        np.where(unstored & tie, np.minimum(first_gap, best_pos), best_pos),
    )
    fibers = fiber[starts]
    if out_shape:
        R = Tensor.from_coo(
            np.unravel_index(fibers, out_shape), result.astype(np.int64), gb_dtype(np.int64),
            shape=out_shape,
        )  # fmt: skip
        out = Array._new(R, 0)
    else:
        out = Array._new(_utils.empty_0d(np.int64), result[0])
    return expand_dims(out, axis) if keepdims else out


def _first_true(flags, starts, counts, within):
    """Per group (``starts``, ``counts``): ``within`` at the group's first True flag."""
    idx = np.where(flags, np.arange(len(flags)), len(flags))
    first = np.minimum.reduceat(idx, starts)
    first = np.minimum(first, len(flags) - 1)
    return within[first]


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

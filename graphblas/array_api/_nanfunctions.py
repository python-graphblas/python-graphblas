"""numpy's reductions that skip NaN, as extensions: ``nansum``, ``nanprod``, ``nanmax``,
``nanmin``, ``nanmean``, ``nanvar`` and ``nanstd``.

They take numpy's keywords, so a library that looks for them on an array's namespace finds
them here and never hands the Array to numpy (xarray does so for its float reductions, which
skip NaN by default). A NaN is skipped wherever it is: stored, or the fill, where the
unstored positions then count for nothing. Any other fill counts once per unstored position,
as in the reductions of the standard. A slice whose values are all NaN gives numpy's result:
0 for ``nansum``, 1 for ``nanprod`` and NaN for the others (numpy also warns; these do not).
"""

import math

import numpy as np

from .. import agg, binary, monoid, select, unary
from . import _utils
from ._array import Array
from ._dtypes import as_dtype, astype
from ._statistical import (
    _axes,
    _check_array,
    _keepdims,
    _mean_dtype,
    _reduce_tensor,
    _squared_magnitude,
    _times_count,
)
from ._statistical import max as _max
from ._statistical import mean as _mean
from ._statistical import min as _min
from ._statistical import prod as _prod
from ._statistical import sum as _sum
from ._statistical import var as _var
from ._utils import gb_dtype

__all__ = ["nanmax", "nanmean", "nanmin", "nanprod", "nanstd", "nansum", "nanvar"]


def _start(a, name, out):
    _check_array(a, name)
    if out is not None:
        raise TypeError(f"{name}: out= is not supported")


def _has_nan(a):
    """Whether ``a`` has a NaN: stored, or its fill at an unstored position."""
    if a.dtype.kind not in "fc":
        return False
    if a._T().nvals < a.size and np.isnan(a._fill):
        return True
    return _utils.fact(a, "nan")


def _replace_nan(a, value):
    from ._elementwise import isnan
    from ._searching import where

    return where(isnan(a), value, a)


def _count(a, axis, keepdims):
    """The number of values other than NaN in each slice (int64)."""
    from ._elementwise import isnan, logical_not

    return _sum(logical_not(isnan(a)), axis=axis, dtype=np.int64, keepdims=keepdims)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """The sum over ``axis``, NaN taken as 0 (numpy's ``nansum``). An extension."""
    _start(a, "nansum", out)
    return _sum(_replace_nan(a, 0) if _has_nan(a) else a, axis=axis, dtype=dtype, keepdims=keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """The product over ``axis``, NaN taken as 1 (numpy's ``nanprod``). An extension."""
    _start(a, "nanprod", out)
    return _prod(
        _replace_nan(a, 1) if _has_nan(a) else a, axis=axis, dtype=dtype, keepdims=keepdims
    )


def _nan_extreme(name, func, identity, a, axis, out, keepdims):
    from ._searching import where

    _start(a, name, out)
    if a.dtype.kind == "c":
        raise TypeError(f"{name} needs a real-valued array, not {a.dtype}")
    if not _has_nan(a):
        return func(a, axis=axis, keepdims=keepdims)
    r = func(_replace_nan(a, identity), axis=axis, keepdims=keepdims)
    return where(_count(a, axis, keepdims) == 0, np.nan, r)


def nanmax(a, axis=None, out=None, keepdims=False):
    """The maximum over ``axis``, skipping NaN (numpy's ``nanmax``); NaN where a slice has
    nothing else. An extension.
    """
    return _nan_extreme("nanmax", _max, -np.inf, a, axis, out, keepdims)


def nanmin(a, axis=None, out=None, keepdims=False):
    """The minimum over ``axis``, skipping NaN (numpy's ``nanmin``); NaN where a slice has
    nothing else. An extension.
    """
    return _nan_extreme("nanmin", _min, np.inf, a, axis, out, keepdims)


def _length(a, axis):
    return math.prod(a.shape[i] for i in _axes(a, axis))


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """The mean over ``axis`` of the values other than NaN (numpy's ``nanmean``), summed in
    ``dtype``; NaN where a slice has none. An extension.
    """
    from ._elementwise import divide

    _start(a, "nanmean", out)
    rdt = _mean_dtype(a, "nanmean") if dtype is None else as_dtype(dtype)
    if not _has_nan(a):
        if dtype is None:
            return _mean(a, axis=axis, keepdims=keepdims)
        return divide(_sum(a, axis=axis, dtype=rdt, keepdims=keepdims), _length(a, axis))
    total = _sum(_replace_nan(a, 0), axis=axis, dtype=rdt, keepdims=keepdims)
    # (numpy divides in float64, or complex128, and casts to the sum's dtype)
    out = divide(total, astype(_count(a, axis, keepdims), np.float64))
    return out if out.dtype == rdt else astype(out, rdt)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, correction=None):
    """The variance over ``axis`` of the values other than NaN (numpy's ``nanvar``), computed
    in ``dtype``, with ``ddof`` (or ``correction``) subtracted from each slice's count; NaN
    where that leaves no degree of freedom. An extension.

    It takes two passes, as numpy's does: each slice's mean, then the squared deviations from
    it. The stored values' deviations are summed by GraphBLAS; the fill's, when it is not
    NaN, is computed once per slice and multiplied by the slice's count of unstored
    positions.
    """
    _start(a, "nanvar", out)
    if correction is not None:
        if ddof != 0:
            raise ValueError("nanvar: ddof and correction cannot both be given")
        ddof = correction
    rdt = _mean_dtype(a, "nanvar") if dtype is None else as_dtype(dtype)
    if not _has_nan(a):
        x = a if dtype is None or a.dtype == rdt else astype(a, rdt)
        return _var(x, axis=axis, correction=ddof, keepdims=keepdims)
    # (an array with a NaN has no empty axis: the path above takes those)
    axes = _axes(a, axis)
    real_dt = np.dtype(rdt.char.lower()) if rdt.kind == "c" else rdt
    x = a if a.dtype == rdt else astype(a, rdt)
    out = _nanvar_each(x, ddof) if not axes else _nanvar_over(x, axes, ddof)
    out = out if out.dtype == real_dt else astype(out, real_dt)
    return _keepdims(out, a, axes, keepdims)


def _nanvar_each(x, ddof):
    """``nanvar`` over no axes: each element is its own slice."""
    from ._elementwise import divide, isnan, logical_not, subtract
    from ._searching import where

    dof = subtract(astype(logical_not(isnan(x)), np.float64), float(ddof))
    squares = _squared_magnitude(subtract(x, x))  # (0, or NaN for NaN and infinities)
    return where(dof <= 0, np.nan, divide(squares, dof))


def _reduced(T, fill, dtype):
    """An Array of the reduced Tensor ``T``, ``fill`` where it stores nothing (a 0-d Array
    holds its value as its fill).
    """
    if T.ndim:
        return Array._new(T, fill)
    return Array._new(_utils.empty_0d(dtype), T.to_coo()[1][0] if T.nvals else fill)


def _nanvar_over(x, axes, ddof):
    """``nanvar`` over ``axes`` (some), as an Array of the reduced shape."""
    from ._elementwise import add, binary_numpy, divide, subtract
    from ._searching import where

    rdt = x.dtype
    gdt = gb_dtype(rdt)
    real_dt = np.dtype(rdt.char.lower()) if rdt.kind == "c" else rdt
    L = math.prod(x.shape[i] for i in axes)
    T = x._T()
    fill = _utils.scalar(x._fill, rdt)
    fill_counts = not np.isnan(fill)
    # The stored values other than NaN, and per slice their count and sum
    N = T.apply(unary.isnan[gdt]).new().select(select.valueeq, True).new()
    V = T.apply(unary.identity[gdt]).new(mask=~N.S) if N.nvals else T
    C = _reduce_tensor(V, agg.count, axes)
    S = _reduce_tensor(V, monoid.plus[gdt], axes)
    K = _reduce_tensor(T, agg.count, axes)  # (stored entries per slice, NaN included)
    # Each slice's mean, at the slices with values in V: the fill counts L - (stored entries)
    # times unless it is NaN
    if fill_counts:
        KC = K.apply(unary.identity).new(mask=C.S)  # (on C's pattern)

        def mean_of(s, c, k):
            u = np.float64(L) - k
            return ((s + _times_count(u, np.asarray(fill, rdt))) / (c + u)).astype(rdt)

        M = _utils.map_values(mean_of, [S, C, KC], rdt)
    else:
        M = _utils.map_values(lambda s, c: (s / c).astype(rdt), [S, C], rdt)
    # The squared deviations of V from its slices' means, summed per slice
    minus = binary.minus[gdt]
    if M.ndim == 0:
        mean_value = M.to_coo()[1][0] if M.nvals else fill
        D = V.apply(minus, right=_utils.gb_scalar(mean_value, rdt)).new()
    else:
        D = V.ewise_mult(M.expand_dims(axes), minus).new()
    squares = _utils.map_values(lambda d: np.real(d * np.conj(d)).astype(real_dt), [D], real_dt)
    total = _reduced(_reduce_tensor(squares, monoid.plus[gb_dtype(real_dt)], axes), 0, real_dt)
    count = astype(_reduced(C, 0, np.dtype(np.int64)), np.float64)
    if fill_counts:
        # The unstored positions: each |fill - mean| ** 2 away (0 * inf must stay 0)
        unstored = subtract(float(L), _reduced(K, 0, np.dtype(np.int64)))
        count = add(count, unstored)
        means = _reduced(M, fill, rdt)  # (a slice with no value in V has only fills: their mean)
        deviation = subtract(Array._new(_utils.empty_0d(rdt), fill), means)
        missing = binary_numpy("nanvar", _times_count, unstored, _squared_magnitude(deviation))
        total = add(total, missing)
    dof = subtract(count, float(ddof))
    return where(dof <= 0, np.nan, divide(total, dof))


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, correction=None):
    """The standard deviation over ``axis`` of the values other than NaN (numpy's
    ``nanstd``): the square root of ``nanvar``. An extension.
    """
    from ._elementwise import sqrt

    _start(a, "nanstd", out)
    return sqrt(
        nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, correction=correction)
    )

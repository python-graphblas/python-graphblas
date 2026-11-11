"""Statistical functions, and the utility functions ``all``, ``any`` and ``diff``.

A reduction over a fiber of ``L`` positions, ``k`` of them unstored, is the monoid over the
stored entries combined with the fill ``k`` times: ``plus`` adds ``k * fill``, ``times``
multiplies by ``fill ** k``, and ``min``, ``max``, ``lor`` and ``land`` take the fill once
when ``k > 0``. The counts cost a second pass, skipped when the fill is the monoid's identity
or every position is stored. A fiber with no entries holds the result's fill. Floating-point
sums and products may round differently from numpy's (the order of the terms differs);
everything else is exact.
"""

import builtins
import math

import numpy as np

from .. import agg, binary, monoid, unary
from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._dtypes import as_dtype, check_supported
from ._utils import gb_dtype

__all__ = [
    "all",
    "any",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]


def _check_array(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs an Array, not {type(x).__name__}")


def _axes(x, axis):
    """The reduced axes, normalized and sorted."""
    if axis is None:
        return tuple(range(x.ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    norm = []
    for a in axes:
        a = int(a)
        if not -x.ndim <= a < x.ndim:
            raise IndexError(f"axis {a} is out of bounds for {x.ndim} dimensions")
        norm.append(a % x.ndim)
    if len(set(norm)) != len(norm):
        raise ValueError(f"repeated axis in {axis}")
    return tuple(sorted(norm))


def _out_shape(shape, axes, keepdims):
    if keepdims:
        return tuple(1 if i in axes else n for i, n in enumerate(shape))
    return tuple(n for i, n in enumerate(shape) if i not in axes)


def _keepdims(out, x, axes, keepdims):
    if keepdims and axes:
        from ._manipulation import expand_dims

        return expand_dims(out, axes)
    return out


def _reduce_tensor(T, op, axes):
    axis = None if len(axes) == T.ndim else axes
    return T.reduce(op, axis=axis).new()


def _combine(func, tensors, dtype, fill):
    """An Array whose entries are ``func`` of the aligned values of ``tensors`` (one
    pattern), and ``fill`` elsewhere.
    """
    return Array._new(_utils.map_values(func, tensors, dtype), fill)


def _int_power(base, exponent, dtype):
    """``base ** exponent`` wrapped to the integer ``dtype``, as repeated products wrap."""
    bits = dtype.itemsize * 8
    value = builtins.pow(int(base), int(exponent), 1 << bits)
    if dtype.kind == "i" and value >= 1 << (bits - 1):
        value -= 1 << bits
    return np.asarray(value).astype(dtype)[()]


def _unstored(L, c, ct):
    """``L - c`` (counts of unstored positions), in a dtype that does not overflow: uint64
    for integer sums (which wrap anyway), float64 otherwise (a fiber may hold 2**63 or more
    positions).
    """
    if ct.kind in "iub":
        return np.uint64(L % (1 << 64)) - np.asarray(c).astype(np.uint64)
    return np.float64(L) - c


def _scaled(k, fill, ct):
    """``fill`` added ``k`` times (k >= 0), as repeated additions give it: wrapping for
    integers, logical or for bool, and part by part for complex (0 * inf in one part must not
    make NaN in the other).
    """
    with np.errstate(all="ignore"):
        if ct.kind in "iu":
            f = np.asarray(fill).astype(np.int64 if ct.kind == "i" else np.uint64)
            return (np.asarray(k, dtype=np.uint64) * f.astype(np.uint64)).astype(ct)
        if ct.kind == "b":
            return np.logical_and(np.asarray(k) > 0, fill)
        if ct.kind == "c":
            out = np.empty(np.shape(k), ct)
            out.real = k * fill.real
            out.imag = k * fill.imag
            return out
        return np.asarray(np.multiply(k, fill)).astype(ct)


class _Monoid:
    def __init__(self, name, gb, npfunc, identity):
        self.name = name
        self.gb = gb
        self.npfunc = npfunc
        self.identity = identity  # (dtype -> the identity)

    def repeat(self, fill, L, ct):
        """The monoid over ``L`` copies of ``fill``."""
        if self.name == "sum":
            k = np.uint64(L % (1 << 64)) if ct.kind in "iub" else np.float64(L)
            return _scaled(k, fill, ct)[()]
        if self.name == "prod":
            with np.errstate(all="ignore"):
                if ct.kind in "iu":
                    return _int_power(fill, L, ct)
                if ct.kind == "b":
                    return fill
                return np.power(fill, np.float64(L), dtype=ct)
        return fill

    def combine(self, L, fill, ct):
        """``func(r, count)``: a fiber's reduction of its stored entries ``r`` combined with
        the fill at its ``L - count`` unstored positions.
        """
        name = self.name
        if name == "sum":
            # (a fiber with no unstored position must not see 0 * inf)
            return lambda r, c: np.where(c < L, r + _scaled(_unstored(L, c, ct), fill, ct), r)
        if name == "prod":
            if ct.kind in "iu":
                return lambda r, c: r * _int_powers(fill, _unstored(L, c, ct), ct)
            if ct.kind == "b":
                return lambda r, c: np.where(c < L, r & fill, r)
            return lambda r, c: np.where(c < L, r * np.power(fill, _unstored(L, c, ct)), r)
        func = {
            "min": np.minimum,
            "max": np.maximum,
            "any": np.logical_or,
            "all": np.logical_and,
        }[name]
        return lambda r, c: np.where(c < L, func(r, fill), r)


def _int_powers(base, exponents, dtype):
    """``base ** e`` for each count ``e`` (uint64), wrapped to the integer ``dtype``."""
    big = np.dtype("uint64" if dtype.kind == "u" else "int64")
    with np.errstate(all="ignore"):
        return np.power(big.type(base), exponents.astype(big)).astype(dtype)


def _min_identity(dt):
    if dt.kind == "f":
        return np.inf
    if dt.kind == "b":
        return True
    return np.iinfo(dt).max


def _max_identity(dt):
    if dt.kind == "f":
        return -np.inf
    if dt.kind == "b":
        return False
    return np.iinfo(dt).min


_MONOIDS = {
    "sum": _Monoid("sum", monoid.plus, np.sum, lambda dt: 0),
    "prod": _Monoid("prod", monoid.times, np.prod, lambda dt: 1),
    "min": _Monoid("min", monoid.min, np.min, _min_identity),
    "max": _Monoid("max", monoid.max, np.max, _max_identity),
    "any": _Monoid("any", monoid.lor, np.any, lambda dt: False),
    "all": _Monoid("all", monoid.land, np.all, lambda dt: True),
}


def _reduce(name, x, axis, keepdims, dtype=None):
    out = _reduce_main(name, x, axis, keepdims, dtype)
    if name == "sum" and out.dtype.kind in "fc":
        # numpy's sum starts from +0.0: a sum of -0.0 is 0.0 (GraphBLAS starts from the first)
        from ._elementwise import add

        out = add(out, 0.0)
    return out


def _reduce_main(name, x, axis, keepdims, dtype):
    _check_array(x, name)
    spec = _MONOIDS[name]
    if name in {"min", "max"} and x.dtype.kind == "c":
        raise TypeError(f"{name} needs a real-valued array, not {x.dtype}")
    if name in {"any", "all"}:
        ct = np.dtype(bool)
    elif name in {"sum", "prod"}:
        if dtype is not None:
            ct = as_dtype(dtype)
        else:
            ct = check_supported(spec.npfunc(np.zeros(1, x.dtype)).dtype, name)
    else:
        ct = x.dtype
    axes = _axes(x, axis)
    if x.ndim == 0:
        kwargs = {"dtype": ct} if name in {"sum", "prod"} else {}
        with np.errstate(all="ignore"):
            value = spec.npfunc(np.asarray(x._fill), **kwargs)
        return Array._new(Tensor(gb_dtype(ct), ()), value)
    if not axes:
        return _reduce_no_axes(name, x, ct)
    T = x._T()
    shape = _out_shape(x.shape, axes, keepdims=False)
    L = math.prod(x.shape[a] for a in axes)
    with np.errstate(all="ignore"):
        fill = _utils.scalar(x._fill, ct)
    if L == 0:
        if name in {"min", "max"}:
            raise ValueError(f"{name} of an empty axis, which has no identity")
        out = Array._new(Tensor(gb_dtype(ct), shape), spec.identity(ct))
        return _keepdims(out, x, axes, keepdims)
    fill_out = spec.repeat(fill, L, ct)
    if T.nvals == 0 or math.prod(shape) == 0:
        return _keepdims(Array._new(Tensor(gb_dtype(ct), shape), fill_out), x, axes, keepdims)
    if name in {"min", "max"} and ct.kind == "f" and _signed_zeros(x, fill):
        # GraphBLAS's min and max are C's fmin and fmax, which may take either of 0.0 and -0.0
        out = _reduce_numpy(x, axes, np.minimum if name == "min" else np.maximum, fill, ct)
        return _keepdims(out, x, axes, keepdims)
    nan_fix = name in {"min", "max"} and ct.kind == "f" and _has_nan(x)
    full = T.nvals == x.size
    if len(axes) == x.ndim:
        # Over every axis: one Scalar, and the count is the number of entries
        r = T.reduce_scalar(_gb_monoid(spec, ct)).new().value
        if nan_fix:
            value = np.nan
        elif full or _utils.same_value(fill, spec.identity(ct)):
            value = r
        else:
            with np.errstate(all="ignore"):
                count = np.array([T.nvals], dtype=np.int64)
                value = spec.combine(L, fill, ct)(np.array([r], dtype=ct), count)[0]
        out = Array._new(Tensor(gb_dtype(ct), ()), value)
        return _keepdims(out, x, axes, keepdims)
    R = _reduce_tensor(T, _gb_monoid(spec, ct), axes)
    if not nan_fix and (full or _utils.same_value(fill, spec.identity(ct))):
        out = Array._new(R, fill if full else fill_out)
    else:
        C = _reduce_tensor(T, agg.count, axes)
        combine = spec.combine(L, fill, ct)
        if nan_fix:
            # GraphBLAS's min and max skip NaN; numpy's propagate it
            N = _reduce_tensor(T.apply(unary.isnan).new(), monoid.lor[gb_dtype(bool)], axes)

            def func(r, c, n):
                return np.where(n, np.nan, combine(r, c))

            out = _combine(func, [R, C, N], ct, fill_out)
        else:
            out = _combine(combine, [R, C], ct, fill_out)
    return _keepdims(out, x, axes, keepdims)


def _reduce_no_axes(name, x, ct):
    """A reduction over no axes, as numpy computes it: the identity combined with each
    element (so ``sum`` turns ``-0.0`` into ``0.0``), or a copy for ``min`` and ``max``.
    """
    from ._elementwise import add, cast, multiply

    if name in {"sum", "prod"}:
        identity = Array._new(Tensor(gb_dtype(ct), ()), 0 if name == "sum" else 1)
        out = (add if name == "sum" else multiply)(identity, x if x.dtype == ct else cast(x, ct))
        return out if out.dtype == ct else cast(out, ct)
    return cast(x, ct) if x.dtype != ct else x._copy()


def _gb_monoid(spec, ct):
    # (a sum of booleans in bool is their logical or, a product their logical and)
    if ct.kind == "b" and spec.name in {"sum", "prod"}:
        return (monoid.lor if spec.name == "sum" else monoid.land)[gb_dtype(ct)]
    return spec.gb[gb_dtype(ct)]


def _signed_zeros(x, fill):
    """Whether a min or max might meet both zeros: a stored -0.0, or a fill of -0.0."""
    return _utils.summary(x)["negzero"] or bool(fill == 0 and np.signbit(fill))


def _reduce_numpy(x, axes, ufunc, fill, ct):
    """``ufunc.reduce`` over ``axes`` by numpy on the stored entries of each fiber, then with
    the fill once where a fiber has an unstored position: O(entries log entries).
    """
    from ._manipulation import permute_dims

    others = tuple(a for a in range(x.ndim) if a not in axes)
    shape = tuple(x.shape[a] for a in others)
    L = math.prod(x.shape[a] for a in axes)
    coords, values = permute_dims(x, others + tuple(axes))._T().to_coo()
    if shape:
        fiber = np.ravel_multi_index(coords[: len(others)], shape)
    else:
        fiber = np.zeros(len(values), np.int64)
    starts = np.flatnonzero(np.r_[True, fiber[1:] != fiber[:-1]])
    counts = np.diff(np.r_[starts, len(fiber)])
    with np.errstate(all="ignore"):
        r = ufunc.reduceat(values, starts)
        r = np.where(counts < L, ufunc(r, fill), r).astype(ct)
    if not shape:
        return Array._new(Tensor(gb_dtype(ct), ()), r[0])
    idx = np.unravel_index(fiber[starts], shape)
    return Array._new(Tensor.from_coo(idx, r, gb_dtype(ct), shape=shape), fill)


def _has_nan(x):
    return _utils.summary(x)["nan"]


def sum(x, /, *, axis=None, dtype=None, keepdims=False):  # noqa: A001 (the Array API's name)
    return _reduce("sum", x, axis, keepdims, dtype)


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
    return _reduce("prod", x, axis, keepdims, dtype)


def min(x, /, *, axis=None, keepdims=False):
    return _reduce("min", x, axis, keepdims)


def max(x, /, *, axis=None, keepdims=False):
    return _reduce("max", x, axis, keepdims)


def any(x, /, *, axis=None, keepdims=False):  # noqa: A001 (the Array API's name)
    return _reduce("any", x, axis, keepdims)


def all(x, /, *, axis=None, keepdims=False):  # noqa: A001 (the Array API's name)
    return _reduce("all", x, axis, keepdims)


def _mean_dtype(x, name):
    return check_supported(np.mean(np.zeros(1, x.dtype)).dtype, name)


def mean(x, /, *, axis=None, keepdims=False):
    from ._elementwise import divide

    _check_array(x, "mean")
    rdt = _mean_dtype(x, "mean")
    L = math.prod(x.shape[a] for a in _axes(x, axis))
    # (an empty axis divides 0 by 0, as numpy does: nan, or nan+nanj for complex)
    return divide(_reduce("sum", x, axis, keepdims, rdt), L)


def _squared_magnitude(x):
    from ._elementwise import conj, multiply, real

    if x.dtype.kind == "c":
        return real(multiply(x, conj(x)))
    return multiply(x, x)


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    """Two passes: the means, then the squared deviations from them (the stored entries'
    in GraphBLAS, and the fill's once per fiber, times its count of unstored positions).
    """
    from ._elementwise import add, binary_numpy, cast, divide, subtract

    _check_array(x, "var")
    rdt = _mean_dtype(x, "var")
    real_dt = np.dtype(rdt.char.lower()) if rdt.kind == "c" else rdt
    axes = _axes(x, axis)
    L = math.prod(x.shape[a] for a in axes)
    if L == 0:
        shape = _out_shape(x.shape, axes, keepdims)
        return Array._new(Tensor(gb_dtype(real_dt), shape), np.nan)
    denominator = builtins.max(L - correction, 0)
    if not axes:
        xr = cast(x, rdt)
        out = divide(_squared_magnitude(subtract(xr, xr)), denominator)
        return out if out.dtype == real_dt else cast(out, real_dt)
    T = x._T()
    m = mean(x, axis=axes)
    if T.nvals:
        C = _reduce_tensor(T, agg.count, axes)
        D = _deviations(T, m, C, axes, rdt)
        S2 = _utils.map_values(lambda d: np.real(d * np.conj(d)).astype(real_dt), [D], real_dt)
        S2 = _reduce_tensor(S2, monoid.plus[gb_dtype(real_dt)], axes)
    else:
        S2 = Tensor(gb_dtype(real_dt), m.shape)
        C = Tensor(gb_dtype(np.int64), m.shape)
    k = subtract(L, Array._new(C, 0))
    dev = subtract(Array._new(Tensor(gb_dtype(rdt), ()), x._fill), m)
    # k unstored positions, each (fill - mean) ** 2 away; none for k == 0 (the fill may be inf)
    missing = binary_numpy("var", _times_count, k, _squared_magnitude(dev))
    out = divide(add(Array._new(S2, 0), missing), denominator)
    if out.dtype != real_dt:
        out = cast(out, real_dt)
    return _keepdims(out, x, axes, keepdims)


def _deviations(T, m, C, axes, rdt):
    """Each entry of ``T`` minus its fiber's mean (``m``, ``C`` the entries per fiber)."""
    minus = binary.minus[gb_dtype(rdt)]
    if m.ndim == 0:
        return T.apply(minus, right=_utils.gb_scalar(m._fill, rdt)).new()
    Tm = m._T()
    if Tm.nvals != C.nvals:  # pragma: no cover (safety: store the mean of every fiber)
        keep_shape = _out_shape(T.shape, axes, keepdims=True)
        _utils.check_dense(math.prod(keep_shape), "var or std (the means)")
        M = _utils.full_tensor(rdt, keep_shape, m._fill)
        M(accum=binary.second) << Tm.expand_dims(axes)
        return T.ewise_mult(M, minus).new()
    # The means store exactly the fibers that hold entries (the pattern of the sums), and every
    # entry lies in one of them: broadcast them as they are, with no dense copy
    return T.ewise_mult(Tm.expand_dims(axes), minus).new()


def _times_count(k, d):
    return np.where(k == 0, d.dtype.type(0), k * d)


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    from ._elementwise import sqrt

    return sqrt(var(x, axis=axis, correction=correction, keepdims=keepdims))


def _cumulative(name, npfunc, x, axis, dtype, include_initial):
    _check_array(x, name)
    if x.ndim == 0:
        raise ValueError(f"{name} needs an array of at least one dimension")
    if axis is None:
        if x.ndim > 1:
            raise ValueError(f"{name} of an array of more than one dimension needs an axis")
        axis = 0
    axis = _axes(x, axis)[0]
    if dtype is None:
        reduce_name = "sum" if name == "cumulative_sum" else "prod"
        ct = check_supported(_MONOIDS[reduce_name].npfunc(np.zeros(1, x.dtype)).dtype, name)
    else:
        ct = as_dtype(dtype)
    _utils.check_dense(x.size + (x.size // builtins.max(x.shape[axis], 1)), name)
    with np.errstate(all="ignore"):
        out = npfunc(x.todense(), axis=axis, dtype=ct)
    if include_initial:
        shape = list(out.shape)
        shape[axis] = 1
        initial = np.zeros(shape, ct) if name == "cumulative_sum" else np.ones(shape, ct)
        out = np.concatenate([initial, out], axis=axis)
    from ._creation import _from_numpy

    return _from_numpy(out)


def cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False):
    return _cumulative("cumulative_sum", np.cumsum, x, axis, dtype, include_initial)


def cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False):
    return _cumulative("cumulative_prod", np.cumprod, x, axis, dtype, include_initial)


def diff(x, /, *, axis=-1, n=1, prepend=None, append=None):
    from ._elementwise import not_equal, subtract
    from ._manipulation import concat

    _check_array(x, "diff")
    if x.ndim == 0:
        raise ValueError("diff needs an array of at least one dimension")
    axis = _axes(x, axis)[0]
    n = int(n)
    if n < 0:
        raise ValueError(f"diff's n must be nonnegative, not {n}")
    if n == 0:
        return x._copy()  # (numpy's diff ignores prepend and append then)
    parts = [_diff_part(p, x, axis) for p in (prepend, x, append) if p is not None]
    if len(parts) > 1:
        x = concat(parts, axis=axis)
    # (numpy's diff of booleans is not_equal: subtracting booleans is an error)
    op = not_equal if x.dtype.kind == "b" else subtract
    before = (slice(None),) * axis
    for _ in range(n):
        if x.shape[axis] == 0:
            break
        x = op(x[(*before, slice(1, None))], x[(*before, slice(None, -1))])
    return x


def _diff_part(p, x, axis):
    """A ``prepend`` or ``append`` of ``diff``: an Array, or a scalar broadcast to ``x``'s
    shape with length 1 along ``axis`` (as numpy takes it).
    """
    if isinstance(p, Array) and p.ndim:
        return p
    from ._creation import asarray
    from ._manipulation import _broadcast_to

    shape = list(x.shape)
    shape[axis] = 1
    return _broadcast_to(asarray(p), tuple(shape))

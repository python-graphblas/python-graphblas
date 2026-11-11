"""Element-wise functions: numpy's answer, bit for bit, on every position.

A function of one array maps the stored entries and the fill alike. A function of two arrays
takes the union of their entries, an unstored position of one operand taking its fill there,
and the new fill is the function of the two fills. Python scalars are weak (NEP 50).

Where a GraphBLAS operator gives numpy's answer exactly for the dtypes at hand, the entries
are computed in GraphBLAS; otherwise numpy maps the stored values (the pattern stays as it is,
see ``_utils.map_values``). numpy also computes every fill, so an entry equal to the fill maps
to the new fill.
"""

import builtins

import numpy as np

from .. import binary, unary
from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._dtypes import check_supported
from ._utils import gb_dtype, gb_scalar

__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "pow",
    "real",
    "reciprocal",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

_FLOAT = "f"
_INEXACT = "fc"
_INTEGRAL = "iu"
_REAL = "biuf"


# Operands
def _as_operand(x, name):
    """An Array; a Python scalar stays as it is (weak); a numpy scalar or array becomes an
    Array (strong, as numpy treats it).
    """
    if isinstance(x, Array):
        return x
    if isinstance(x, (np.generic, np.ndarray)):
        from ._creation import asarray

        return asarray(x)
    if isinstance(x, (builtins.bool, int, float, complex)):
        return x
    raise TypeError(f"{name} needs arrays or Python scalars, not {type(x).__name__}")


def _probe(x):
    """What numpy's type resolution sees: an empty array of an Array's dtype, or the weak
    Python scalar itself.
    """
    if isinstance(x, Array):
        return np.empty(0, x.dtype)
    return x


def _fill_operand(x):
    """The value of ``x`` at an unstored position, typed as numpy would type ``x``."""
    if isinstance(x, Array):
        return np.asarray(x._fill)
    return x


def _result(T, fill):
    return Array._new(T, fill)


def _zero_d(dtype, fill):
    return Array._new(Tensor(gb_dtype(dtype), ()), fill)


# Unary functions
def _unary(name, ufunc, x, gb=None, *, const=None):
    """``ufunc`` of every position of ``x``.

    ``gb(dtype)`` gives a GraphBLAS UnaryOp equal to ``ufunc`` on that dtype, a callable
    taking the Tensor, or None; ``const(dtype)`` gives the value of every position when the
    result does not depend on the entries (``isnan`` of an integer array), else None.
    """
    x = _as_operand(x, name)
    if not isinstance(x, Array):
        from ._creation import asarray

        x = asarray(x)
    with np.errstate(all="ignore"):
        rdt = ufunc(np.empty(0, x.dtype)).dtype
        fill = ufunc(np.asarray(x._fill))
    check_supported(rdt, f"{name} of {x.dtype}")
    T = x._T()
    if T.ndim == 0:
        return _zero_d(rdt, fill)
    if const is not None and (value := const(x.dtype)) is not None:
        return _result(_utils.empty_tensor(rdt, T.shape), value)
    if T.nvals == 0:
        return _result(_utils.empty_tensor(rdt, T.shape), fill)
    op = gb(x.dtype) if gb is not None else None
    if op is None:
        R = _utils.map_values(ufunc, [T], rdt)
    elif op == "copy":
        R = T.dup(dtype=gb_dtype(rdt))
    elif callable(op) and not hasattr(op, "opclass"):
        R = op(T)
    else:
        R = T.apply(op).new(dtype=gb_dtype(rdt))
    return _result(R, fill)


def _typed(op, kinds):
    """``gb`` for _unary: ``op`` typed for dtypes of these kinds, else numpy."""

    def get(dtype):
        return op[gb_dtype(dtype)] if dtype.kind in kinds else None

    return get


def _copy_if(kinds):
    def get(dtype):
        return "copy" if dtype.kind in kinds else None

    return get


def _first_of(*getters):
    def get(dtype):
        for g in getters:
            if (op := g(dtype)) is not None:
                return op
        return None

    return get


def abs(x, /):  # noqa: A001 (the Array API's name)
    # (GraphBLAS's integer abs of the most negative value is undefined in C; numpy wraps)
    return _unary("abs", np.abs, x, _first_of(_typed(unary.abs, _FLOAT), _copy_if("ub")))


def negative(x, /):
    return _unary("negative", np.negative, x, _typed(unary.ainv, "iufc"))


def positive(x, /):
    return _unary("positive", np.positive, x, _copy_if("iufc"))


def sqrt(x, /):
    # IEEE square root is correctly rounded, in C as in numpy
    return _unary("sqrt", np.sqrt, x, _typed(unary.sqrt, _FLOAT))


def floor(x, /):
    return _unary("floor", np.floor, x, _first_of(_typed(unary.floor, _FLOAT), _copy_if("biu")))


def ceil(x, /):
    return _unary("ceil", np.ceil, x, _first_of(_typed(unary.ceil, _FLOAT), _copy_if("biu")))


def trunc(x, /):
    return _unary("trunc", np.trunc, x, _first_of(_typed(unary.trunc, _FLOAT), _copy_if("biu")))


def round(x, /):  # noqa: A001 (the Array API's name)
    # numpy rounds half to even; C's round (GraphBLAS's) rounds half away from zero
    return _unary("round", np.round, x, _copy_if("biu"))


def _const_unless(kinds, value):
    def get(dtype):
        return None if dtype.kind in kinds else value

    return get


def isnan(x, /):
    return _unary(
        "isnan", np.isnan, x, _typed(unary.isnan, _INEXACT), const=_const_unless("fc", False)
    )


def isinf(x, /):
    return _unary(
        "isinf", np.isinf, x, _typed(unary.isinf, _INEXACT), const=_const_unless("fc", False)
    )


def isfinite(x, /):
    return _unary(
        "isfinite",
        np.isfinite,
        x,
        _typed(unary.isfinite, _INEXACT),
        const=_const_unless("fc", True),
    )


def logical_not(x, /):
    return _unary("logical_not", np.logical_not, x, _typed(unary.lnot, "b"))


def bitwise_invert(x, /):
    def gb(dtype):
        if dtype.kind == "b":
            return unary.lnot[gb_dtype(dtype)]
        if dtype.kind in _INTEGRAL:
            return unary.bnot[gb_dtype(dtype)]
        return None

    return _unary("bitwise_invert", np.invert, x, gb)


def square(x, /):
    def gb(dtype):
        if dtype.kind not in "iuf":
            return None
        # x * x on the stored entries, as numpy's square computes it
        return lambda T: T.ewise_mult(T, binary.times[gb_dtype(dtype)]).new()

    return _unary("square", np.square, x, gb)


def reciprocal(x, /):
    return _unary("reciprocal", np.reciprocal, x, _typed(unary.minv, _FLOAT))


def conj(x, /):
    return _unary("conj", np.conj, x, _first_of(_typed(unary.conj, "c"), _copy_if(_REAL)))


def real(x, /):
    return _unary("real", np.real, x, _first_of(_typed(unary.creal, "c"), _copy_if(_REAL)))


def imag(x, /):
    def const(dtype):
        return None if dtype.kind == "c" else 0

    return _unary("imag", np.imag, x, _typed(unary.cimag, "c"), const=const)


def sign(x, /):
    return _unary("sign", np.sign, x)


def signbit(x, /):
    return _unary("signbit", np.signbit, x)


def _numpy_unary(name, ufunc):
    def func(x, /):
        return _unary(name, ufunc, x)

    func.__name__ = name
    func.__qualname__ = name
    return func


acos = _numpy_unary("acos", np.arccos)
acosh = _numpy_unary("acosh", np.arccosh)
asin = _numpy_unary("asin", np.arcsin)
asinh = _numpy_unary("asinh", np.arcsinh)
atan = _numpy_unary("atan", np.arctan)
atanh = _numpy_unary("atanh", np.arctanh)
cos = _numpy_unary("cos", np.cos)
cosh = _numpy_unary("cosh", np.cosh)
exp = _numpy_unary("exp", np.exp)
expm1 = _numpy_unary("expm1", np.expm1)
log = _numpy_unary("log", np.log)
log1p = _numpy_unary("log1p", np.log1p)
log2 = _numpy_unary("log2", np.log2)
log10 = _numpy_unary("log10", np.log10)
sin = _numpy_unary("sin", np.sin)
sinh = _numpy_unary("sinh", np.sinh)
tan = _numpy_unary("tan", np.tan)
tanh = _numpy_unary("tanh", np.tanh)


# Binary functions
_MANTISSA = {"f": {4: 24, 8: 53}, "c": {8: 24, 16: 53}}


def _exact_cast(src, dst):
    """Whether every value of dtype ``src`` converts to ``dst`` without change."""
    if src == dst or src.kind == "b":
        return True
    if src.kind in _INTEGRAL:
        if dst.kind in _INTEGRAL:
            return np.can_cast(src, dst, casting="safe")
        if dst.kind in _INEXACT and dst != np.dtype(bool):
            return src.itemsize * 8 - (src.kind == "i") <= _MANTISSA[dst.kind][dst.itemsize]
        return False
    if src.kind == "f" and dst.kind in _INEXACT:
        return _MANTISSA["f"][src.itemsize] <= _MANTISSA[dst.kind][dst.itemsize]
    if src.kind == "c" and dst.kind == "c":
        return src.itemsize <= dst.itemsize
    return False


def _python_scalar_fits(value, dtype):
    """Whether numpy converts the Python scalar ``value`` to ``dtype`` without changing it
    (out-of-range Python ints get special treatment in numpy's comparisons).
    """
    if dtype.kind in _INTEGRAL and isinstance(value, int):
        info = np.iinfo(dtype)
        return info.min <= value <= info.max
    return True


class _Binary:
    """How to compute one binary function.

    ``kind`` is "arith" (the op works in the result dtype), "cmp" (in the promoted dtype of
    the operands, with a bool result) or "logical" (in bool). ``gb(dtype)`` gives the
    GraphBLAS operator that equals ``ufunc`` on operands of the dtype it works in, or None.
    ``annihilator(dtype)`` gives the fill value that, held by both operands, makes the union
    equal to the intersection (``ewise_mult``), or None.
    """

    def __init__(self, name, ufunc, kind="arith", gb=None, annihilator=None):
        self.name = name
        self.ufunc = ufunc
        self.kind = kind
        self.gb = gb
        self.annihilator = annihilator


def _binary(spec, x1, x2):
    a = _as_operand(x1, spec.name)
    b = _as_operand(x2, spec.name)
    ufunc = spec.ufunc
    if not isinstance(a, Array) and not isinstance(b, Array):
        from ._creation import asarray

        a = asarray(a)
    with np.errstate(all="ignore"):
        rdt = ufunc(_probe(a), _probe(b)).dtype
        fill = ufunc(_fill_operand(a), _fill_operand(b))
    check_supported(rdt, f"{spec.name} of {_dtype_name(a)} and {_dtype_name(b)}")
    a_arr = isinstance(a, Array) and a.ndim > 0
    b_arr = isinstance(b, Array) and b.ndim > 0
    if not a_arr and not b_arr:
        return _zero_d(rdt, fill)
    if spec.kind == "arith":
        ct = rdt
    elif spec.kind == "cmp":
        try:
            ct = np.result_type(_probe(a), _probe(b))
        except (OverflowError, TypeError):
            ct = None  # (a Python int out of range: numpy compares it exactly)
    else:
        ct = np.dtype(bool)
    op = None if ct is None else _gb_op(spec, a, b, ct)
    if a_arr and b_arr:
        return _binary_arrays(spec, a, b, op, ct, rdt, fill)
    if a_arr:
        T, other, left = a._T(), b, False
    else:
        T, other, left = b._T(), a, True
    if T.nvals == 0:
        return _result(_utils.empty_tensor(rdt, T.shape), fill)
    if op is not None:
        s = gb_scalar(_fill_operand(other), ct)
        expr = T.apply(op, left=s) if left else T.apply(op, right=s)
        return _result(expr.new(dtype=gb_dtype(rdt)), fill)
    s = _fill_operand(other)
    if left:
        R = _utils.map_values(lambda v: ufunc(s, v), [T], rdt)
    else:
        R = _utils.map_values(lambda v: ufunc(v, s), [T], rdt)
    return _result(R, fill)


def _dtype_name(x):
    return str(x.dtype) if isinstance(x, Array) else type(x).__name__


def _gb_op(spec, a, b, ct):
    """The typed GraphBLAS op for ``spec`` working in ``ct``, when it gives numpy's answer;
    else None.
    """
    if spec.gb is None:
        return None
    for x in (a, b):
        if isinstance(x, Array):
            if spec.kind != "arith" and not _exact_cast(x.dtype, ct):
                return None
        elif not _python_scalar_fits(x, ct):
            return None
    return spec.gb(ct)


def _binary_arrays(spec, a, b, op, ct, rdt, fill):
    Ta, Tb = a._T(), b._T()
    try:
        shape = np.broadcast_shapes(Ta.shape, Tb.shape)
    except ValueError:
        raise ValueError(
            f"{spec.name}: shapes {Ta.shape} and {Tb.shape} do not broadcast"
        ) from None
    if Ta.nvals == 0 and Tb.nvals == 0:
        return _result(_utils.empty_tensor(rdt, shape), fill)
    if op is not None:
        fa = _utils.scalar(a._fill, ct)
        fb = _utils.scalar(b._fill, ct)
        zero = spec.annihilator(ct) if spec.annihilator is not None else None
        if zero is not None and _utils.same_value(fa, zero) and _utils.same_value(fb, zero):
            # Where either operand holds its fill the result is the fill: the intersection
            R = Ta.ewise_mult(Tb, op).new(dtype=gb_dtype(rdt))
            return _result(R, fill)
        _utils.expansion_check(Ta, Tb, shape)
        R = Ta.ewise_union(Tb, op, gb_scalar(fa, ct), gb_scalar(fb, ct), allow_expansion=True).new(
            dtype=gb_dtype(rdt)
        )
        return _result(R, fill)
    # numpy over the values: both operands' values on the union of entries, aligned
    _utils.expansion_check(Ta, Tb, shape)
    da, db = a.dtype, b.dtype
    A = Ta.ewise_union(
        Tb, binary.first[gb_dtype(da)], gb_scalar(a._fill, da), gb_scalar(0, da),
        allow_expansion=True,
    ).new()  # fmt: skip
    B = Ta.ewise_union(
        Tb, binary.second[gb_dtype(db)], gb_scalar(0, db), gb_scalar(b._fill, db),
        allow_expansion=True,
    ).new()  # fmt: skip
    return _result(_utils.map_values(spec.ufunc, [A, B], rdt), fill)


def _typed_for(op_by_kind):
    """``gb`` for _Binary: ``op_by_kind[dtype.kind]`` typed for the dtype, else None."""

    def get(dtype):
        op = op_by_kind.get(dtype.kind)
        return None if op is None else op[gb_dtype(dtype)]

    return get


def _zero(dtype):
    return _utils.scalar(0, dtype)


_SPECS = {
    "add": _Binary(
        "add", np.add, gb=_typed_for({"b": binary.lor, **dict.fromkeys("iufc", binary.plus)})
    ),
    "subtract": _Binary(
        "subtract", np.subtract, gb=_typed_for(dict.fromkeys("iufc", binary.minus))
    ),
    # Complex products and quotients are computed by numpy: C's formulas round otherwise.
    # A float zero fill is not an annihilator: 0.0 * -1.0 is -0.0, and 0.0 * inf is NaN.
    "multiply": _Binary(
        "multiply",
        np.multiply,
        gb=_typed_for({"b": binary.land, **dict.fromkeys("iuf", binary.times)}),
        annihilator=lambda dt: _zero(dt) if dt.kind in "biu" else None,
    ),
    "divide": _Binary("divide", np.divide, gb=_typed_for({"f": binary.truediv})),
    "equal": _Binary("equal", np.equal, "cmp", gb=_typed_for(dict.fromkeys("biufc", binary.eq))),
    "not_equal": _Binary(
        "not_equal", np.not_equal, "cmp", gb=_typed_for(dict.fromkeys("biufc", binary.ne))
    ),
    "less": _Binary("less", np.less, "cmp", gb=_typed_for(dict.fromkeys("biuf", binary.lt))),
    "less_equal": _Binary(
        "less_equal", np.less_equal, "cmp", gb=_typed_for(dict.fromkeys("biuf", binary.le))
    ),
    "greater": _Binary(
        "greater", np.greater, "cmp", gb=_typed_for(dict.fromkeys("biuf", binary.gt))
    ),
    "greater_equal": _Binary(
        "greater_equal", np.greater_equal, "cmp", gb=_typed_for(dict.fromkeys("biuf", binary.ge))
    ),
    "logical_and": _Binary(
        "logical_and",
        np.logical_and,
        "logical",
        gb=lambda dt: binary.land[gb_dtype(dt)],
        annihilator=_zero,
    ),
    "logical_or": _Binary(
        "logical_or", np.logical_or, "logical", gb=lambda dt: binary.lor[gb_dtype(dt)]
    ),
    "logical_xor": _Binary(
        "logical_xor", np.logical_xor, "logical", gb=lambda dt: binary.lxor[gb_dtype(dt)]
    ),
    "bitwise_and": _Binary(
        "bitwise_and",
        np.bitwise_and,
        gb=_typed_for({"b": binary.land, "i": binary.band, "u": binary.band}),
        annihilator=_zero,
    ),
    "bitwise_or": _Binary(
        "bitwise_or",
        np.bitwise_or,
        gb=_typed_for({"b": binary.lor, "i": binary.bor, "u": binary.bor}),
    ),
    "bitwise_xor": _Binary(
        "bitwise_xor",
        np.bitwise_xor,
        gb=_typed_for({"b": binary.lxor, "i": binary.bxor, "u": binary.bxor}),
    ),
    # GraphBLAS's max and min of floats drop NaN (C's fmax); numpy's propagate it
    "maximum": _Binary("maximum", np.maximum, gb=_typed_for(dict.fromkeys("biu", binary.max))),
    "minimum": _Binary("minimum", np.minimum, gb=_typed_for(dict.fromkeys("biu", binary.min))),
    "copysign": _Binary("copysign", np.copysign, gb=_typed_for({"f": binary.copysign})),
    # numpy's own definitions: floored division and remainder (with its results for division
    # by zero), integer powers, shifts past the width, and libm-free special values
    "floor_divide": _Binary("floor_divide", np.floor_divide),
    "remainder": _Binary("remainder", np.remainder),
    "pow": _Binary("pow", np.power),
    "atan2": _Binary("atan2", np.arctan2),
    "hypot": _Binary("hypot", np.hypot),
    "logaddexp": _Binary("logaddexp", np.logaddexp),
    "nextafter": _Binary("nextafter", np.nextafter),
    "bitwise_left_shift": _Binary("bitwise_left_shift", np.left_shift),
    "bitwise_right_shift": _Binary("bitwise_right_shift", np.right_shift),
}


def _make_binary(name):
    spec = _SPECS[name]

    def func(x1, x2, /):
        return _binary(spec, x1, x2)

    func.__name__ = name
    func.__qualname__ = name
    return func


add = _make_binary("add")
subtract = _make_binary("subtract")
multiply = _make_binary("multiply")
divide = _make_binary("divide")
equal = _make_binary("equal")
not_equal = _make_binary("not_equal")
less = _make_binary("less")
less_equal = _make_binary("less_equal")
greater = _make_binary("greater")
greater_equal = _make_binary("greater_equal")
logical_and = _make_binary("logical_and")
logical_or = _make_binary("logical_or")
logical_xor = _make_binary("logical_xor")
bitwise_and = _make_binary("bitwise_and")
bitwise_or = _make_binary("bitwise_or")
bitwise_xor = _make_binary("bitwise_xor")
maximum = _make_binary("maximum")
minimum = _make_binary("minimum")
copysign = _make_binary("copysign")
floor_divide = _make_binary("floor_divide")
remainder = _make_binary("remainder")
pow = _make_binary("pow")  # noqa: A001 (the Array API's name)
atan2 = _make_binary("atan2")
hypot = _make_binary("hypot")
logaddexp = _make_binary("logaddexp")
nextafter = _make_binary("nextafter")
bitwise_left_shift = _make_binary("bitwise_left_shift")
bitwise_right_shift = _make_binary("bitwise_right_shift")


def clip(x, /, min=None, max=None):
    """``x`` clipped to ``[min, max]`` in the dtype of ``x``; NaN propagates.

    With scalar bounds this is numpy's ``clip`` (which keeps ``-0.0`` against a bound of
    ``0.0``); with array bounds it is ``minimum(maximum(x, min), max)``, as numpy's ``clip``
    with array bounds is.
    """
    if not isinstance(x, Array):
        raise TypeError(f"clip needs an Array, not {type(x).__name__}")
    lo = _clip_bound(min, x)
    hi = _clip_bound(max, x)
    if lo is None and hi is None:
        out = x._copy()
    elif all(b is None or not isinstance(b, Array) or b.ndim == 0 for b in (lo, hi)):
        lo_v, hi_v = (b._fill if isinstance(b, Array) else b for b in (lo, hi))
        out = _unary("clip", lambda v: np.clip(v, lo_v, hi_v), x)
    else:
        out = x
        if lo is not None:
            out = maximum(out, lo)
        if hi is not None:
            out = minimum(out, hi)
    if out.dtype != x.dtype:
        out = cast(out, x.dtype)
    return out


def _clip_bound(bound, x):
    """A Python integer bound clamped to the range of an integer ``x`` (the Array API's clip
    keeps the dtype of ``x``, and numpy refuses an integer out of its range).
    """
    if isinstance(bound, (np.generic, np.ndarray)):
        return _as_operand(bound, "clip")
    if isinstance(bound, (builtins.bool, int, float)):
        if x.dtype.kind in _INTEGRAL and isinstance(bound, float):
            return bound
        if x.dtype.kind in _INTEGRAL and not _python_scalar_fits(bound, x.dtype):
            info = np.iinfo(x.dtype)
            return builtins.min(builtins.max(bound, int(info.min)), int(info.max))
    return bound


# Casting (astype)
def _gb_cast_ok(src, dst):
    """Whether GraphBLAS's typecast from ``src`` to ``dst`` gives numpy's ``astype``.

    C converts integers and floats as numpy does; float to integer differs where the value
    does not fit (NaN, inf, out of range: numpy's result is undefined there, GraphBLAS
    saturates), and complex to real is numpy's own rule.
    """
    if src.kind == "c":
        return dst.kind in "cb"
    if src.kind == "f":
        return dst.kind in "fcb"
    return True


def _astype(values, dt):
    # numpy's astype from complex to real keeps the real part (and warns that it does)
    if values.dtype.kind == "c" and dt.kind not in "cb":
        values = values.real
    return values.astype(dt)


def cast(x, dtype):
    """``x.astype(dtype)`` by numpy's rules, entries and fill alike."""
    dt = np.dtype(dtype)
    with np.errstate(all="ignore"):
        fill = _astype(np.asarray(x._fill), dt)
    T = x._T()
    if T.ndim == 0:
        return _zero_d(dt, fill)
    if T.nvals == 0:
        return _result(_utils.empty_tensor(dt, T.shape), fill)
    if _gb_cast_ok(x.dtype, dt):
        return _result(T.dup(dtype=gb_dtype(dt)), fill)
    return _result(_utils.map_values(lambda v: _astype(v, dt), [T], dt), fill)

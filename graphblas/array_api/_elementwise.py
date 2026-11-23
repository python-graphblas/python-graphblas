"""Element-wise functions.

Each function returns numpy's answer at every position, NaN and inf included. Transcendental
functions are within libm's rounding of numpy's results. Signed zeros are exact too, except
where the sign is unspecified: at a position the result does not store, and in maximum and
minimum of 0.0 and -0.0.

A function of one array maps the stored entries and the fill alike. A function of two arrays
computes on the union of their entries, using an operand's fill where it stores nothing, and
the result's fill is the function of the two fills. Python scalars are weak (NEP 50).

GraphBLAS computes the entries where its operator agrees exactly with numpy for the dtypes
at hand, and for the libm functions on real floating dtypes (see ``_libm_unary``); otherwise
numpy maps the stored values, keeping the pattern (see ``_utils.map_values``). The fill is
computed by the operator that maps the entries, or by numpy where the two agree exactly, so
an entry equal to the fill maps to the new fill.
"""

import builtins
import math

import numpy as np

from .. import binary, monoid, select, unary
from ..core.scalar import Scalar
from ..core.tensor import Tensor
from ..dtypes import BOOL, INT64
from . import _lazy, _utils
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
    """``x`` as an operand: an Array or a Python scalar (weak) as it is, and a numpy scalar or
    array as an Array (strong, as numpy treats it).
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
    """A stand-in for ``x`` in numpy's type resolution: an empty array of an Array's dtype,
    or a Python scalar itself (weak).
    """
    if isinstance(x, Array):
        return np.empty(0, x.dtype)
    return x


def _fill_operand(x):
    """The value of ``x`` at an unstored position, typed as numpy would type ``x``."""
    if isinstance(x, Array):
        return np.asarray(x._fill)
    return x


def _fill_of(ufunc, rdt, operands):
    """``ufunc`` of the operands' fills: the result's fill.

    When numpy raises on the fills (an integer to a negative power) and an operand stores
    every position, 0 stands in: that operand's fill is never seen, so neither is the
    result's.
    """
    # numpy's 0-d path can differ from its array loop (clip breaks a tie another way), so the
    # fill of an Array with a dimension is a 1-element array. A 0-d Array stays 0-d, and a
    # Python scalar stays weak.
    args = [
        (
            (np.reshape(x._fill, (1,)) if x.ndim else np.asarray(x._fill))
            if isinstance(x, Array)
            else x
        )
        for x in operands
    ]
    try:
        return np.asarray(ufunc(*args)).reshape(-1)[0]
    except (ValueError, ArithmeticError):
        if any(isinstance(x, Array) and x.ndim and x._T().nvals == x.size for x in operands):
            return _utils.scalar(0, rdt)
        raise


def _result(T, fill):
    if T.dtype == BOOL and T.ndim and T.nvals:
        if T.nvals == T.size:
            T, fill = _normalize_bool(T, fill)
        elif fill and 2 * T.nvals > T.size:
            T, fill = _majority_false(T)
    return Array._new(T, fill)


def _majority_false(T):
    """``(R, fill)`` with the values of the bool Tensor ``T``, whose fill is True.

    When more than half of the positions of ``T`` are stored False entries, ``R`` stores only
    the True positions (the unstored ones and the stored True entries) and ``fill`` is False.
    That costs O(size), which is under twice the entries. ``equal(deg, 0)`` of a ``deg`` that
    leaves a few zero degrees unstored is such a Tensor. Otherwise this returns ``(T, True)``.
    """
    ntrue = T.reduce_scalar(monoid.plus[INT64]).new().value
    if 2 * (T.nvals - ntrue) <= T.size:
        return T, True
    R = Tensor(BOOL, T.shape)
    R(~T.S)[...] << True
    if ntrue:
        R(T.select(select.valueeq, True).new().S)[...] << True
    return R, False


def _normalize_bool(T, fill):
    """``(T, False)`` for a bool Tensor ``T`` that stores every position, with its False
    entries dropped when at most a quarter of the entries are True.

    ``T`` never shows its fill, so it can have fill False, which lets its False entries be
    dropped. ``equal(deg, 0)`` of a fully stored ``deg`` then stores only its True entries.
    With more True entries the False ones stay: dropping them costs about as much as the
    comparison and saves a later consumer little.

    A sample of the leading entries first says whether dropping is likely, so that a
    mostly-True result pays only for the sample (a count over all 2e7 values cost as much as
    the comparison at 1 thread). Then one select of the True values decides, and is the
    result when they are few.
    """
    n = T.nvals
    rows = -(-_SAMPLE * T.shape[0] // n)  # (the leading rows with about _SAMPLE entries)
    if rows < T.shape[0]:
        sample = T[(slice(0, rows),) + (slice(None),) * (T.ndim - 1)].new()
        if 3 * sample.reduce_scalar(monoid.plus[INT64]).new().value > sample.nvals:
            return T, False  # (over a third True in the sample: dropping is unlikely)
    S = T.select(select.valueeq, True).new()
    return (S if 4 * S.nvals <= n else T), False


_SAMPLE = 2**16


def _zero_d(dtype, fill):
    return Array._new(_utils.empty_0d(dtype), fill)


# Unary functions
_ALL_FACTS = ("finite", "nan", "zero")


def _unary(name, ufunc, x, gb=None, *, const=None, keeps=()):
    """``ufunc`` of every position of ``x``.

    ``gb(dtype)`` says how to compute the entries for ``x``'s dtype: a GraphBLAS UnaryOp equal
    to ``ufunc`` there, ``"copy"`` when ``ufunc`` changes no value, a function of the Tensor,
    or None for numpy. ``const(dtype)`` returns the value of every position when the result
    does not depend on the entries (``isnan`` of an integer array), else None. ``keeps`` names
    the facts about the stored values (``_utils.fact``) that hold for the result when they
    hold for ``x``.
    """
    x = _as_operand(x, name)
    if not isinstance(x, Array):
        from ._creation import asarray

        x = asarray(x)
    op = gb(x.dtype) if gb is not None else None
    with np.errstate(all="ignore"):
        rdt = ufunc(np.empty(0, x.dtype)).dtype
        if getattr(op, "opclass", None) == "UnaryOp":
            # The fill by the op that maps the entries, so that an entry equal to the fill maps
            # to the new fill: libm's rounding may differ from numpy's by an ulp
            s = Scalar.from_value(x._fill, gb_dtype(x.dtype))
            fill = s.apply(op).new(dtype=gb_dtype(rdt)).value
        else:
            fill = _fill_of(ufunc, rdt, (x,))
    check_supported(rdt, f"{name} of {x.dtype}")
    T = x._T()
    if T.ndim == 0:
        return _zero_d(rdt, fill)
    if const is not None and (value := const(x.dtype)) is not None:
        return _result(_utils.empty_tensor(rdt, T.shape), value)
    if T.nvals == 0:
        return _result(_utils.empty_tensor(rdt, T.shape), fill)
    if op is None:
        R = _utils.map_values(ufunc, [T], rdt)
    elif op == "copy":
        R = T.dup(dtype=gb_dtype(rdt))
    elif callable(op) and not hasattr(op, "opclass"):
        R = op(T)
    else:
        R = T.apply(op).new(dtype=gb_dtype(rdt))
    return _utils.keep_facts(x, _result(R, fill), keeps)


def _typed(op, kinds):
    """A ``gb`` for ``_unary``: ``op`` typed for a dtype of one of ``kinds``, else None."""

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
    gb = _first_of(_typed(unary.abs, _FLOAT), _copy_if("ub"))
    # (|z| of a finite complex z may overflow, and |inf + nan j| is inf: only "zero" carries
    # over)
    keeps = ("zero",) if isinstance(x, Array) and x.dtype.kind == "c" else _ALL_FACTS
    return _unary("abs", np.abs, x, gb, keeps=keeps)


def negative(x, /):
    return _unary("negative", np.negative, x, _typed(unary.ainv, "iufc"), keeps=_ALL_FACTS)


def positive(x, /):
    return _unary("positive", np.positive, x, _copy_if("iufc"), keeps=_ALL_FACTS)


# (a rounding keeps a value finite, or NaN; it may make a zero)
_ROUNDING_KEEPS = ("finite", "nan")


def sqrt(x, /):
    # IEEE square root is correctly rounded, in C as in numpy
    return _unary("sqrt", np.sqrt, x, _typed(unary.sqrt, _FLOAT))


def floor(x, /):
    gb = _first_of(_typed(unary.floor, _FLOAT), _copy_if("biu"))
    return _unary("floor", np.floor, x, gb, keeps=_ROUNDING_KEEPS)


def ceil(x, /):
    gb = _first_of(_typed(unary.ceil, _FLOAT), _copy_if("biu"))
    return _unary("ceil", np.ceil, x, gb, keeps=_ROUNDING_KEEPS)


def trunc(x, /):
    gb = _first_of(_typed(unary.trunc, _FLOAT), _copy_if("biu"))
    return _unary("trunc", np.trunc, x, gb, keeps=_ROUNDING_KEEPS)


def round(x, /):  # noqa: A001 (the Array API's name)
    # numpy rounds half to even; C's round (GraphBLAS's) rounds half away from zero
    return _unary("round", np.round, x, _copy_if("biu"), keeps=_ROUNDING_KEEPS)


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
    gb = _first_of(_typed(unary.lnot, "b"), _typed(unary.bnot, _INTEGRAL))
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
    gb = _first_of(_typed(unary.conj, "c"), _copy_if(_REAL))
    return _unary("conj", np.conj, x, gb, keeps=_ALL_FACTS)


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


def _libm_unary(name, ufunc, op):
    """The namespace function ``name``: ``ufunc`` computed by the SuiteSparse unary op ``op``
    for real floating dtypes, and by numpy otherwise.

    SuiteSparse calls C's libm, in parallel: its results are within libm's rounding of
    numpy's, with special values (NaN, inf, signed zeros) exact. Complex values stay on
    numpy: C libraries treat the many special cases of complex arguments differently from
    platform to platform, and SuiteSparse's complex expm1 and log1p are ``cexp(z) - 1`` and
    ``clog(1 + z)``, inexact near 0. Integer inputs stay on numpy too, since numpy picks their
    result dtype (float16 for int8, for example).
    """

    def func(x, /):
        return _unary(name, ufunc, x, _typed(op, _FLOAT))

    func.__name__ = name
    func.__qualname__ = name
    return func


acos = _libm_unary("acos", np.arccos, unary.acos)
acosh = _libm_unary("acosh", np.arccosh, unary.acosh)
asin = _libm_unary("asin", np.arcsin, unary.asin)
asinh = _libm_unary("asinh", np.arcsinh, unary.asinh)
atan = _libm_unary("atan", np.arctan, unary.atan)
atanh = _libm_unary("atanh", np.arctanh, unary.atanh)
cos = _libm_unary("cos", np.cos, unary.cos)
cosh = _libm_unary("cosh", np.cosh, unary.cosh)
exp = _libm_unary("exp", np.exp, unary.exp)
log = _libm_unary("log", np.log, unary.log)
log2 = _libm_unary("log2", np.log2, unary.log2)
log10 = _libm_unary("log10", np.log10, unary.log10)
sin = _libm_unary("sin", np.sin, unary.sin)
sinh = _libm_unary("sinh", np.sinh, unary.sinh)
tan = _libm_unary("tan", np.tan, unary.tan)
tanh = _libm_unary("tanh", np.tanh, unary.tanh)
expm1 = _libm_unary("expm1", np.expm1, unary.expm1)
log1p = _libm_unary("log1p", np.log1p, unary.log1p)


# Binary functions
_MANTISSA = {"f": {4: 24, 8: 53}, "c": {8: 24, 16: 53}}


def _exact_cast(src, dst):
    """Whether every value of dtype ``src`` converts to ``dst`` without change."""
    if src == dst or src.kind == "b":
        return True
    if src.kind in _INTEGRAL:
        if dst.kind in _INTEGRAL:
            return np.can_cast(src, dst, casting="safe")
        if dst.kind in _INEXACT:
            return src.itemsize * 8 - (src.kind == "i") <= _MANTISSA[dst.kind][dst.itemsize]
        return False
    if src.kind == "f" and dst.kind in _INEXACT:
        return _MANTISSA["f"][src.itemsize] <= _MANTISSA[dst.kind][dst.itemsize]
    if src.kind == "c" and dst.kind == "c":
        return src.itemsize <= dst.itemsize
    return False


def _python_scalar_fits(value, dtype):
    """Whether numpy casts the Python scalar ``value`` to ``dtype``.

    It does not cast a Python int out of the range of an integer ``dtype``: numpy's
    comparisons compare such an int exactly.
    """
    if dtype.kind in _INTEGRAL and isinstance(value, int):
        info = np.iinfo(dtype)
        return info.min <= value <= info.max
    return True


class _Binary:
    """How to compute one binary function.

    ``kind`` says which dtype the op works in: "arith" in the result dtype, "cmp" in the
    promoted dtype of the operands (with a bool result), "logical" in bool. ``gb(dtype)``
    returns the GraphBLAS operator equal to ``ufunc`` on that dtype, or None.
    ``gb_values(a, b, dtype)``, used in place of ``gb``, may also look at the operands'
    values. ``annihilator(a, b, fill_a, fill_b, dtype)`` says whether the intersection
    (``ewise_mult``) has the union's values: whether, where either operand's value is its
    fill, the result is the result's fill whatever the other operand's value.
    """

    def __init__(self, name, ufunc, kind="arith", gb=None, annihilator=None, gb_values=None):
        self.name = name
        self.ufunc = ufunc
        self.kind = kind
        self.gb = gb
        self.annihilator = annihilator
        self.gb_values = gb_values


def _binary(spec, x1, x2):
    a = _as_operand(x1, spec.name)
    b = _as_operand(x2, spec.name)
    if spec.name == "multiply" and (out := _lazy.masked_multiply(a, b)) is not None:
        return out  # (an SDDMM)
    ufunc = spec.ufunc
    if not isinstance(a, Array) and not isinstance(b, Array):
        from ._creation import asarray

        a = asarray(a)
    if isinstance(a, Array) and isinstance(b, Array) and a.ndim and b.ndim:
        # An operand of size 1 that broadcasts against the other is one value, as a 0-d one
        # is: x - xp.max(x, keepdims=True) costs O(entries of x), not O(size of x)
        if b.size == 1 and b.ndim <= a.ndim:
            b = _to_0d(b)
        elif a.size == 1 and a.ndim <= b.ndim:
            a = _to_0d(a)
    with np.errstate(all="ignore"):
        rdt = ufunc(_probe(a), _probe(b)).dtype
        fill = _fill_of(ufunc, rdt, (a, b))
    check_supported(rdt, f"{spec.name} of {_dtype_name(a)} and {_dtype_name(b)}")
    a_arr = isinstance(a, Array) and a.ndim > 0
    b_arr = isinstance(b, Array) and b.ndim > 0
    if not a_arr and not b_arr:
        return _zero_d(rdt, fill)
    if spec.kind == "arith":
        ct = rdt
    elif spec.kind == "cmp":
        # (a Python int out of range of an integer dtype keeps that dtype here; _gb_op then
        # leaves the comparison to numpy, which compares such an int exactly)
        ct = np.result_type(_probe(a), _probe(b))
    else:
        ct = np.dtype(bool)
    op = _gb_op(spec, a, b, ct)
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


def _to_0d(x):
    """The one value of the size-1 Array ``x`` as a 0-d Array."""
    T = x._T()
    value = T.to_coo()[1][0] if T.nvals else x._fill
    return Array._new(_utils.empty_0d(x.dtype), value)


def _dtype_name(x):
    return str(x.dtype) if isinstance(x, Array) else type(x).__name__


def _gb_op(spec, a, b, ct):
    """The typed GraphBLAS op for ``spec`` working in ``ct``, when it gives numpy's answer;
    else None.
    """
    if spec.gb_values is not None:
        return spec.gb_values(a, b, ct)
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
        shape = Ta.shape if Ta.shape == Tb.shape else np.broadcast_shapes(Ta.shape, Tb.shape)
    except ValueError:
        raise ValueError(
            f"{spec.name}: shapes {Ta.shape} and {Tb.shape} do not broadcast"
        ) from None
    if Ta.nvals == 0 and Tb.nvals == 0:
        return _result(_utils.empty_tensor(rdt, shape), fill)
    if op is not None:
        fa = _utils.scalar(a._fill, ct)
        fb = _utils.scalar(b._fill, ct)
        if spec.annihilator is not None:
            # An operand that stores every position never shows its fill: take it as 0
            zero = _zero(ct)
            fa_seen = zero if _stores_all(a) else fa
            fb_seen = zero if _stores_all(b) else fb
            if spec.annihilator(a, b, fa_seen, fb_seen, ct):
                # Where either operand's value is its fill, the result is the result's fill:
                # the intersection has the union's values
                _utils.mult_expansion_check(Ta, Tb, shape)
                R = Ta.ewise_mult(Tb, op).new(dtype=gb_dtype(rdt))
                with np.errstate(all="ignore"):
                    return _result(R, spec.ufunc(fa_seen, fb_seen))
        if (e := _absorbing(spec, a, b, fa, fb, ct)) is not None:
            # Both fills absorb the op: where either operand's value is its fill, the result is
            # that fill. So the intersection has the union's values, and it replicates a
            # broadcast operand only when both broadcast.
            _utils.mult_expansion_check(Ta, Tb, shape)
            R = Ta.ewise_mult(Tb, op).new(dtype=gb_dtype(rdt))
            return _result(R, _utils.scalar(e, rdt))
        if spec.name in {"multiply", "divide"} and ct.kind == "f":
            out = _one_sided(spec, a, b, fa, fb, op, ct, rdt, shape)
            if out is not None:
                return out
        _utils.expansion_check(Ta, Tb, shape)
        expr = Ta.ewise_union(Tb, op, gb_scalar(fa, ct), gb_scalar(fb, ct), allow_expansion=True)
        return _result(expr.new(dtype=gb_dtype(rdt)), fill)
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


def _absorbing_values(name, ct):
    """The values ``e`` with ``op(e, y) == op(y, e) == e`` for the binary ``name`` working in
    ``ct``, for every y but those ``_absorbing`` excludes (NaN, the opposite infinity).
    """
    if name in {"add", "maximum", "logical_or"} and ct.kind == "b":
        return (True,)
    if name == "minimum" and ct.kind == "b":
        return (False,)
    if ct.kind == "f":
        return {"add": (np.inf, -np.inf), "maximum": (np.inf,), "minimum": (-np.inf,)}.get(name, ())
    if ct.kind in "iu":
        info = np.iinfo(ct)
        return {"maximum": (info.max,), "minimum": (info.min,)}.get(name, ())
    return ()


def _absorbing(spec, a, b, fa, fb, ct):
    """The fill ``e`` of both operands when ``e`` absorbs the op, else None.

    ``e`` absorbs the op when ``op(e, y)`` and ``op(y, e)`` are ``e`` (``_absorbing_values``):
    inf for add and maximum, -inf for add and minimum, True for logical_or, or an integer
    extreme. The result is then the intersection of the entries, with fill ``e``. An operand
    that stores every position never shows its fill, so it counts as having fill ``e``, but
    not both operands may. For floats, no operand may store NaN (numpy propagates it) nor,
    for add, the opposite infinity (inf + -inf is NaN), checked by the cached facts.
    """
    full_a, full_b = _stores_all(a), _stores_all(b)
    if full_a and full_b:
        return None  # (the union is the intersection already)
    for e in _absorbing_values(spec.name, ct):
        e = _utils.scalar(e, ct)
        if not (full_a or _utils.same_value(fa, e)) or not (full_b or _utils.same_value(fb, e)):
            continue
        if ct.kind == "f":
            opposite = "neginf" if e > 0 else "posinf"
            for x in (a, b):
                if _utils.fact(x, "nan") or (spec.name == "add" and _utils.fact(x, opposite)):
                    return None
        return e
    return None


def _power(x1, x2):
    """``np.power``, corrected where numpy computes a repeated exponent of 0.5 (a scalar, or
    an exponent of stride 0) as a square root. The Array API's pow(-inf, 0.5) is +inf and
    pow(-0.0, 0.5) is +0.0, where sqrt returns NaN and -0.0.
    """
    out = np.power(x1, x2)
    if out.dtype.kind == "f":
        x1 = np.asarray(x1)
        fix = (np.asarray(x2) == 0.5) & ((x1 == -np.inf) | (x1 == 0))
        if fix.any():
            out = np.where(fix, np.where(x1 == 0, 0.0, np.inf), out).astype(out.dtype)
    return out


def _typed_for(op_by_kind):
    """A ``gb`` for ``_Binary``: ``op_by_kind[dtype.kind]`` typed for the dtype, else None."""

    def get(dtype):
        op = op_by_kind.get(dtype.kind)
        return None if op is None else op[gb_dtype(dtype)]

    return get


def _zero(dtype):
    return _utils.scalar(0, dtype)


def _stores_all(x):
    """Whether ``x`` stores every position, so that its fill is never seen."""
    return x._T().nvals == x.size


def _both_fills_zero(a, b, fa, fb, ct):
    zero = _zero(ct)
    return _utils.same_value(fa, zero) and _utils.same_value(fb, zero)


def _multiply_intersects(a, b, fa, fb, ct):
    """Whether ``a * b`` stores only the intersection of entries: both fills are 0 (+0.0 for
    floats) and, for floats, no stored value is inf or NaN (whose product with 0.0 is NaN).

    numpy's product of an unstored zero and a negative value is -0.0, but that position reads
    as the fill, 0.0. The sign of a zero at an unstored position is unspecified, and keeping
    it would make a row scaling ``d[:, None] * A`` by a ``d`` with negative values store every
    position.
    """
    if not _both_fills_zero(a, b, fa, fb, ct):
        return False
    if ct.kind != "f":
        return True
    return _utils.fact(a, "finite") and _utils.fact(b, "finite")


def _one_sided(spec, a, b, fa, fb, op, ct, rdt, shape):
    """``a * b`` or ``a / b`` stored at the entries of one operand X only, or None when
    neither operand qualifies.

    X qualifies when its fill is 0, it does not store every position, it has the result's
    shape, and 0 times (or divided by) every value of the other operand Y is a zero. A row
    scaling ``A * d[:, None]`` or ``A / d`` then stores only A's entries. At X's entries, Y's
    value is its entry or its fill. A Y that broadcasts and is larger than X's entries is
    gathered at them; a smaller one is stored in full first. Either way Y is not replicated.
    """
    sides = [(a, b)] if spec.name == "divide" else [(a, b), (b, a)]
    for X, Y in sides:
        fx = fa if X is a else fb
        fy = fb if X is a else fa
        if _stores_all(X) or X._T().shape != shape or fx != 0:
            continue
        if _stores_all(Y):
            fy = _utils.scalar(1, ct)  # (never seen)
        if not _zero_times(spec, Y, fy):
            continue
        Tx, Ty = X._T(), Y._T()
        with np.errstate(all="ignore"):
            fill = spec.ufunc(fx, fy) if X is a else spec.ufunc(fy, fx)
        if Ty.shape != shape and not _stores_all(Y) and Tx.nvals < Y.size:
            # Y broadcasts, shows its fill, and is larger than X's entries: gather its value at
            # each of X's entries from Y itself, at the projected coordinates, in O(entries of
            # X). Storing Y in full would cost O(size of Y), and the union would replicate Y
            # along its broadcast axes: exp(v)[None, :] times an X of shape (1000, 1e6) with 4
            # entries would make v 1000 times. A Y no larger is stored in full below, so that
            # the broadcast ewise_mult needs no coordinates. (A of 1e6 x 1e6 with 1e7 entries
            # times y[:, None], y 10% stored, at 8 threads: 6 ms, against 147 ms by the gather.)
            Ty = _gather(Tx, Y, shape)
        elif not _stores_all(Y) and (Y.size <= Tx.nvals + 1024 or Ty.shape != shape):
            # (O(size of Y): every position of Y stored, so the product is an intersection)
            other = 0 if Y._fill != 0 else 1
            Ty = _utils.with_fill(Y, other)._T()
        Ta, Tb = (Tx, Ty) if X is a else (Ty, Tx)
        if Ty.nvals == Y.size:
            _utils.mult_expansion_check(Ta, Tb, shape)
            return _result(Ta.ewise_mult(Tb, op).new(dtype=gb_dtype(rdt)), fill)
        _utils.expansion_check(Ta, Tb, shape)
        R = Tensor(gb_dtype(rdt), shape)
        fa_s, fb_s = gb_scalar(fa, ct), gb_scalar(fb, ct)
        R(Tx.S) << Ta.ewise_union(Tb, op, fa_s, fb_s, allow_expansion=True)
        return _result(R, fill)
    return None


def _gather(Tx, Y, shape):
    """A Tensor with the pattern of ``Tx`` (of ``shape``) whose values are those of ``Y``
    there, its entry or its fill. ``Y`` is an Array that broadcasts to ``shape``.
    """
    coords = Tx.to_coo(values=False)[0]
    Ty = Y._T()
    lead = len(shape) - Y.ndim
    # (a broadcast axis of Y has size 1: index 0 there)
    index = tuple(
        coords[lead + i] if n != 1 else np.zeros(len(coords[0]), np.int64)
        for i, n in enumerate(Ty.shape)
    )
    values = Ty.vindex[index].to_dense(fill_value=Y._fill)
    return Tensor.from_coo(coords, values, Ty.dtype, shape=shape)


def _zero_times(spec, y, fy):
    """Whether 0 * v (or 0 / v) is a zero for every value v of ``y``, its fill ``fy`` too.

    For multiply every value must be finite; for divide no value may be 0 or NaN (0 / inf is
    0). The sign of the zero does not matter, since an unstored position reads the fill.
    """
    if spec.name == "divide":
        ok = fy != 0 and not np.isnan(fy)
        return ok and not _utils.fact(y, "zero") and not _utils.fact(y, "nan")
    return bool(np.isfinite(fy)) and _utils.fact(y, "finite")


def _max_min_gb(op):
    """A ``gb_values`` for maximum or minimum: the GraphBLAS ``op`` where it agrees with
    numpy, else None.

    GraphBLAS's max and min of floats are C's fmax and fmin, which skip NaN; numpy's
    propagate it. They agree when neither operand stores NaN (a cached fact) or has a NaN
    fill. Either may return either zero for 0.0 and -0.0: the Array API leaves the order of
    signed zeros unspecified.
    """

    def get(a, b, ct):
        if ct.kind in "biu":
            return op[gb_dtype(ct)]
        if ct.kind != "f":
            return None
        for x in (a, b):
            if isinstance(x, Array):
                if np.isnan(x._fill) or _utils.fact(x, "nan"):
                    return None
            elif isinstance(x, float) and np.isnan(x):
                return None
        return op[gb_dtype(ct)]

    return get


_SPECS = {
    "add": _Binary(
        "add", np.add, gb=_typed_for({"b": binary.lor, **dict.fromkeys("iufc", binary.plus)})
    ),
    "subtract": _Binary(
        "subtract", np.subtract, gb=_typed_for(dict.fromkeys("iufc", binary.minus))
    ),
    # numpy computes complex products and quotients, since C's formulas round differently
    "multiply": _Binary(
        "multiply",
        np.multiply,
        gb=_typed_for({"b": binary.land, **dict.fromkeys("iuf", binary.times)}),
        annihilator=_multiply_intersects,
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
        annihilator=_both_fills_zero,
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
        annihilator=_both_fills_zero,
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
    "maximum": _Binary("maximum", np.maximum, gb_values=_max_min_gb(binary.max)),
    "minimum": _Binary("minimum", np.minimum, gb_values=_max_min_gb(binary.min)),
    "copysign": _Binary("copysign", np.copysign, gb=_typed_for({"f": binary.copysign})),
    # numpy computes the rest. SuiteSparse has no built-in operator for numpy's floored
    # division and remainder (with its results for division by zero), logaddexp or nextafter.
    # Its integer powers differ from numpy's, and its shifts read the shift amount as INT8
    # and shift right when it is negative. atan2 and hypot stay on numpy too, which computes the
    # fill (``_fill_of``): SuiteSparse's libm versions may differ from it by an ulp.
    "floor_divide": _Binary("floor_divide", np.floor_divide),
    "remainder": _Binary("remainder", np.remainder),
    "pow": _Binary("pow", _power),
    "atan2": _Binary("atan2", np.arctan2),
    "hypot": _Binary("hypot", np.hypot),
    "logaddexp": _Binary("logaddexp", np.logaddexp),
    "nextafter": _Binary("nextafter", np.nextafter),
    "bitwise_left_shift": _Binary("bitwise_left_shift", np.left_shift),
    "bitwise_right_shift": _Binary("bitwise_right_shift", np.right_shift),
}


def binary_numpy(name, func, x1, x2):
    """``func`` (a numpy function of two arrays, which broadcasts) of two operands with the
    fill rules of a binary function, computed by numpy over the values (for internal use).
    """
    return _binary(_Binary(name, func), x1, x2)


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
    """Clip each value of ``x`` to the range [``min``, ``max``], as numpy's ``clip`` does.

    ``min`` and ``max`` are arrays that broadcast with ``x``, Python scalars, or None for no
    bound. The result has the dtype of ``x``, and NaN in ``x`` or in a bound propagates. It
    stores the union of the entries of ``x`` and of any array bounds.
    """
    if not isinstance(x, Array):
        raise TypeError(f"clip needs an Array, not {type(x).__name__}")
    lo = _clip_bound(min, x)
    hi = _clip_bound(max, x)
    if lo is None and hi is None:
        return x._copy()
    if all(not isinstance(b, Array) or b.ndim == 0 for b in (lo, hi)):
        lo_v, hi_v = (b._fill if isinstance(b, Array) else b for b in (lo, hi))
        out = _unary("clip", lambda v: np.clip(v, lo_v, hi_v), x)
    else:
        out = _clip_arrays(x, lo, hi)
    return out if out.dtype == x.dtype else cast(out, x.dtype)


def _clip_arrays(x, lo, hi):
    """Clip with array bounds as numpy does, over the union of the entries (aligned values);
    the result has the shape the three broadcast to.
    """
    operands = (x, lo, hi)
    shape = np.broadcast_shapes(*[b.shape for b in operands if isinstance(b, Array)])
    is_array = [isinstance(b, Array) and b.ndim > 0 for b in operands]
    with np.errstate(all="ignore"):
        rdt = np.clip(*map(_probe, operands)).dtype
        fill = _fill_of(np.clip, rdt, operands)
    if math.prod(shape) == 0:
        return _result(_utils.empty_tensor(rdt, shape), fill)

    def func(*arrays):
        it = iter(arrays)
        v, lo_v, hi_v = (
            next(it) if arr else _fill_operand(b) for b, arr in zip(operands, is_array, strict=True)
        )
        return np.clip(v, lo_v, hi_v)

    tensors = _utils.aligned([b for b, arr in zip(operands, is_array, strict=True) if arr], shape)
    return _result(_utils.map_values(func, tensors, rdt), fill)


def _clip_bound(bound, x):
    """``bound`` as an operand of clip, with a Python int clamped to the range of an integer
    ``x``: the Array API's clip keeps the dtype of ``x``, and NumPy 2.0 refuses an int out of
    its range.
    """
    if isinstance(bound, (np.generic, np.ndarray)):
        return _as_operand(bound, "clip")
    if isinstance(bound, (builtins.bool, int, float)):
        if x.dtype.kind in _INTEGRAL and isinstance(bound, float):
            return bound
        # MAINT 2026-11-22: NumPy 2.1 clamps such an int itself; drop this once that is the
        # minimum
        if x.dtype.kind in _INTEGRAL and not _python_scalar_fits(bound, x.dtype):
            info = np.iinfo(x.dtype)
            return builtins.min(builtins.max(bound, int(info.min)), int(info.max))
    return bound


# Casting (astype)
def _gb_cast_ok(src, dst):
    """Whether GraphBLAS's typecast from ``src`` to ``dst`` gives numpy's ``astype``.

    C converts integers and floats as numpy does, except float to integer where the value
    does not fit (NaN, inf, out of range): numpy's result is undefined there, and GraphBLAS
    saturates. Complex to complex or bool agrees too. Complex to a real type stays on numpy
    (to an integer, it differs as float to integer does).
    """
    if src.kind == "c":
        return dst.kind in "cb"
    if src.kind == "f":
        return dst.kind in "fcb"
    return True


def _astype(values, dt):
    # numpy's astype from complex to real keeps the real part, but warns; dropping the
    # imaginary part first avoids the warning
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

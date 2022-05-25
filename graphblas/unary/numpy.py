""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.readthedocs.io/en/stable/reference/numpysupported.html#math-operations

"""
import numpy as _np

from .. import _STANDARD_OPERATOR_NAMES
from .. import config as _config
from .. import unary as _unary
from ..dtypes import _supports_complex

_delayed = {}
_unary_names = {
    # Math operations
    "negative",
    "abs",
    "absolute",
    "cbrt",
    "fabs",
    "rint",
    "sign",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "positive",
    "sqrt",
    "square",
    "reciprocal",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "degrees",
    "radians",
    # Bit-twiddling functions
    "bitwise_not",
    "invert",
    # Comparison functions
    "logical_not",
    # Floating functions
    "isfinite",
    "isinf",
    "isnan",
    "signbit",
    "floor",
    "ceil",
    "trunc",
    "spacing",
    # Datetime functions
    # "nat",  # We need to see if our UDTs support datetime dtypes!
}
_numpy_to_graphblas = {
    "abs": "abs",
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctanh": "atanh",
    "bitwise_not": "bnot",
    "cbrt": "cbrt",
    "ceil": "ceil",
    "cos": "cos",
    "cosh": "cosh",
    "exp": "exp",
    "exp2": "exp2",
    "expm1": "expm1",
    # 'fabs': 'abs'  # should we rely on coercion?  fabs is only float
    "floor": "floor",
    "invert": "bnot",
    "isfinite": "isfinite",
    "isinf": "isinf",
    "isnan": "isnan",
    "log": "log",
    "log10": "log10",
    "log1p": "log1p",
    "log2": "log2",
    "logical_not": "lnot",  # should we?  result_type not the same
    "negative": "ainv",
    "positive": "identity",  # positive is supposed to check dtype, but doesn't in numba
    # "reciprocal": "minv",  # has differences.  We should investigate further.
    "rint": "round",
    # 'sign': 'signum'  # signum is float-only
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh",
    "trunc": "trunc",
}
# Not included: deg2rad degrees rad2deg radians signbit spacing square
if _supports_complex:
    _unary_names.update({"conj", "conjugate"})
    _numpy_to_graphblas["conj"] = "conj"
    _numpy_to_graphblas["conjugate"] = "conj"
# _graphblas_to_numpy = {val: key for key, val in _numpy_to_graphblas.items()}  # Soon...

_STANDARD_OPERATOR_NAMES.update(f"unary.numpy.{name}" for name in _unary_names)
__all__ = list(_unary_names)


def __dir__():
    return globals().keys() | _delayed.keys() | _unary_names


def __getattr__(name):
    if name in _delayed:
        func, kwargs = _delayed.pop(name)
        rv = func(**kwargs)
        globals()[name] = rv
        return rv
    if name not in _unary_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _config.get("mapnumpy") and name in _numpy_to_graphblas:
        globals()[name] = getattr(_unary, _numpy_to_graphblas[name])
    else:
        from .. import operator

        numpy_func = getattr(_np, name)

        def func(x):  # pragma: no cover
            return numpy_func(x)

        operator.UnaryOp.register_new(f"numpy.{name}", func)
        if name == "reciprocal":
            # numba doesn't match numpy here
            def reciprocal(x):
                return 1 if x else 0

            op = operator.UnaryOp.register_anonymous(reciprocal)
            globals()[name]._add(op["BOOL"])
    return globals()[name]

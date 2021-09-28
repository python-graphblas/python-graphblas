""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#math-operations

"""
import numpy as _np

from .. import binary as _binary
from .. import config as _config
from .. import operator as _operator

_binary_names = {
    # Math operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "logaddexp",
    "logaddexp2",
    "true_divide",
    "floor_divide",
    "power",
    "remainder",
    "mod",
    "fmod",
    "gcd",
    "lcm",
    # Trigonometric functions
    "arctan2",
    "hypot",
    # Bit-twiddling functions
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
    # Comparison functions
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    # Floating functions
    "copysign",
    "nextafter",
    "ldexp",
}
__all__ = list(_binary_names)
_numpy_to_graphblas = {
    # Monoids
    "add": "plus",
    "bitwise_and": "band",
    "bitwise_or": "bor",
    "bitwise_xor": "bxor",
    "equal": "eq",
    "fmax": "max",  # ignores nan
    "fmin": "min",  # ignores nan
    "logical_and": "land",
    "logical_or": "lor",
    "logical_xor": "lxor",
    "multiply": "times",
    # Other
    "arctan2": "atan2",
    "copysign": "copysign",
    "divide": "truediv",
    # "floor_divide": "floordiv",  # floor_divide does not cast to int, but floordiv does
    # "fmod": "fmod",  # not the same!
    "greater": "gt",
    "greater_equal": "ge",
    "ldexp": "ldexp",
    "less": "lt",
    "less_equal": "le",
    # "mod": "remainder",  # not the same!
    "not_equal": "ne",
    "power": "pow",
    # "remainder": "remainder",  # not the same!
    "subtract": "minus",
    "true_divide": "truediv",
}
# Not included: maximum, minimum, gcd, hypot, logaddexp, logaddexp2
# lcm, left_shift, nextafter, right_shift


def __dir__():
    return __all__


def __getattr__(name):
    if name not in _binary_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _config.get("mapnumpy") and name in _numpy_to_graphblas:
        globals()[name] = getattr(_binary, _numpy_to_graphblas[name])
    else:
        numpy_func = getattr(_np, name)
        _operator.BinaryOp.register_new(f"numpy.{name}", lambda x, y: numpy_func(x, y))
    return globals()[name]

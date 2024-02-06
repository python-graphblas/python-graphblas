"""Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.readthedocs.io/en/stable/reference/numpysupported.html#math-operations

"""

import numpy as _np

from .. import _STANDARD_OPERATOR_NAMES
from .. import binary as _binary
from .. import config as _config
from ..core import _supports_udfs

_delayed = {}
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
    "float_power",
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
    # Misc.
    # "around",  # <non-bool>_<int> -> <non-bool>
}
_STANDARD_OPERATOR_NAMES.update(f"binary.numpy.{name}" for name in _binary_names)
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
    "float_power": None,  # uses pow with everything coerced to float64 (constructed below)
    # "remainder": "remainder",  # not the same!
    "subtract": "minus",
    "true_divide": "truediv",
}
# _graphblas_to_numpy = {val: key for key, val in _numpy_to_graphblas.items()}  # Soon...
# Not included: maximum, minimum, gcd, hypot, logaddexp, logaddexp2
# lcm, left_shift, nextafter, right_shift

_commutative = {
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "equal",
    "fmax",
    "fmin",
    "gcd",
    "hypot",
    "lcm",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "not_equal",
}
_commutes_to = {
    "greater": "less",
    "greater_equal": "less_equal",
    "less": "greater",
    "less_equal": "greater_equal",
}
# Don't commute: arctan2, around, copysign, divide, floor_divide, fmod, ldexp,
# left_shift, mod, nextafter, power, remainder, right_shift, subtract, true_divide.
# If desired, we could create r-versions of these so they can commute to something.


def __dir__():
    if not _supports_udfs and not _config["mapnumpy"]:
        # float_power is special: it's constructed from builtin operators
        return globals().keys() | {"float_power"}  # FLAKY COVERAGE
    attrs = _delayed.keys() | _binary_names
    if not _supports_udfs:
        attrs &= _numpy_to_graphblas.keys()
    return attrs | globals().keys()


def __getattr__(name):
    if name in _delayed:
        delayed_func, kwargs = _delayed.pop(name)
        rv = delayed_func(**kwargs)
        globals()[name] = rv
        return rv
    if name not in _binary_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _config.get("mapnumpy") and name in _numpy_to_graphblas or name == "float_power":
        if name == "float_power":
            from ..core.operator import binary
            from ..dtypes import FP64

            new_op = binary.BinaryOp(f"numpy.{name}")
            builtin_op = _binary.pow
            for dtype in builtin_op.types:
                if dtype.name in {"FP32", "FC32", "FC64"}:
                    orig_dtype = dtype
                else:
                    orig_dtype = FP64
                orig_op = builtin_op[orig_dtype]
                cur_op = binary.TypedBuiltinBinaryOp(
                    new_op,
                    new_op.name,
                    dtype,
                    builtin_op.types[orig_dtype],
                    orig_op.gb_obj,
                    orig_op.gb_name,
                )
                new_op._add(cur_op)
            globals()[name] = new_op
        else:
            globals()[name] = getattr(_binary, _numpy_to_graphblas[name])
    elif not _supports_udfs:
        raise AttributeError(
            f"module {__name__!r} unable to compile UDF for {name!r}; "
            "install numba for UDF support"
        )
    else:
        numpy_func = getattr(_np, name)

        def func(x, y):  # pragma: no cover (numba)
            return numpy_func(x, y)

        _binary.register_new(f"numpy.{name}", func)
    rv = globals()[name]
    if name in _commutative:
        rv._commutes_to = rv
    elif name in _commutes_to and rv._commutes_to is None:
        rv._commutes_to = f"numpy.{_commutes_to[name]}"
    return rv

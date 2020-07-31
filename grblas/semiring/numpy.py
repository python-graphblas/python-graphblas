""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#math-operations

"""
import itertools
from .. import ops, binary, monoid
from ..binary.numpy import _binary_names
from ..monoid.numpy import _monoid_identities

_semiring_names = {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(_monoid_identities, _binary_names)
}

# Remove incompatible combinations
# <non-int>_<int>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(
        {"equal", "hypot", "logaddexp", "logaddexp2", "logical_and", "logical_or", "logical_xor",},
        {"gcd", "lcm", "left_shift", "right_shift"},
    )
}
# <non-float>_<float>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(
        {
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "equal",
            "gcd",
            "logical_and",
            "logical_or",
            "logical_xor",
        },
        {
            "arctan2",
            "copysign",
            "divide",
            "hypot",
            "ldexp",
            "logaddexp2",
            "logaddexp",
            "nextafter",
            "true_divide",
        },
    )
}
# <float>_<non-float>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(
        {"hypot", "logaddexp", "logaddexp2"}, {"bitwise_and", "bitwise_or", "bitwise_xor"},
    )
}
# <bool>_<non-bool>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(
        {"equal", "logical_and", "logical_or", "logical_xor"},
        {"floor_divide", "fmod", "mod", "power", "remainder", "subtract"},
    )
}
# <non-bool>_<bool>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in itertools.product(
        {"gcd", "hypot", "logaddexp", "logaddexp2"},
        {
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "not_equal",
        },
    )
}


def __dir__():
    return list(_semiring_names)


def __getattr__(name):
    if name not in _semiring_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    words = name.split("_")
    for i in range(1, len(words)):
        monoid_name = "_".join(words[:i])
        if not hasattr(monoid.numpy, monoid_name):
            continue
        binary_name = "_".join(words[i:])
        if hasattr(binary.numpy, binary_name):
            break
    ops.Semiring.register_new(
        f"numpy.{name}", getattr(monoid.numpy, monoid_name), getattr(binary.numpy, binary_name),
    )
    return globals()[name]

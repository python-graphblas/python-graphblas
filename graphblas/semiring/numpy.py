"""Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.readthedocs.io/en/stable/reference/numpysupported.html#math-operations

"""
import itertools as _itertools

from .. import _STANDARD_OPERATOR_NAMES
from .. import binary as _binary
from .. import config as _config
from .. import monoid as _monoid
from ..binary.numpy import _binary_names
from ..core import _supports_udfs
from ..monoid.numpy import _fmin_is_float, _monoid_identities

_delayed = {}
_semiring_names = {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in _itertools.product(_monoid_identities, _binary_names)
}

# Remove incompatible combinations
# <non-int>_<int>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in _itertools.product(
        {"equal", "hypot", "logaddexp", "logaddexp2"},
        {"gcd", "lcm", "left_shift", "right_shift"},
    )
}
# <non-float>_<float>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in _itertools.product(
        {
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "equal",
            "gcd",
        },
        {
            "arctan2",
            "copysign",
            "divide",
            "float_power",
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
    for monoid_name, binary_name in _itertools.product(
        {"hypot", "logaddexp", "logaddexp2"},
        {"bitwise_and", "bitwise_or", "bitwise_xor"},
    )
}
# <bool>_<non-bool>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in _itertools.product(
        {"equal"},
        {"floor_divide", "fmod", "mod", "power", "remainder", "subtract"},
    )
}
# <non-bool>_<bool>
_semiring_names -= {
    f"{monoid_name}_{binary_name}"
    for monoid_name, binary_name in _itertools.product(
        {"gcd", "hypot", "logaddexp", "logaddexp2"},
        {
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "not_equal",
        },
    )
}
if _fmin_is_float and not _config.get("mapnumpy"):
    # See: https://github.com/numba/numba/issues/8478
    # <non-float>_<float>
    _semiring_names -= {
        f"{monoid_name}_{binary_name}"
        for monoid_name, binary_name in _itertools.product(
            {
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                "equal",
                "gcd",
            },
            {"fmax", "fmin"},
        )
    }
    _semiring_names -= {
        f"{monoid_name}_{binary_name}"
        for monoid_name, binary_name in _itertools.product(
            {"fmax", "fmin"},
            {
                # <non-float>
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                # <bool>
                "equal",
                "greater",
                "greater_equal",
                "less",
                "less_equal",
                "not_equal",
                # <int>
                "gcd",
                "lcm",
                "left_shift",
                "right_shift",
            },
        )
    }


_STANDARD_OPERATOR_NAMES.update(f"semiring.numpy.{name}" for name in _semiring_names)
__all__ = list(_semiring_names)


def __dir__():
    if not _supports_udfs and not _config.get("mapnumpy"):
        return globals().keys()  # FLAKY COVERAGE
    attrs = _delayed.keys() | _semiring_names
    if not _supports_udfs:
        attrs &= {
            f"{monoid_name}_{binary_name}"
            for monoid_name, binary_name in _itertools.product(
                dir(_monoid.numpy), dir(_binary.numpy)
            )
        }
    return attrs | globals().keys()


def __getattr__(name):
    from ..core.operator import get_semiring

    if name in _delayed:
        func, kwargs = _delayed.pop(name)
        if type(kwargs["binaryop"]) is str:
            from ..binary import from_string

            kwargs["binaryop"] = from_string(kwargs["binaryop"])
        if type(kwargs["monoid"]) is str:
            from ..monoid import from_string

            kwargs["monoid"] = from_string(kwargs["monoid"])
        rv = func(**kwargs)
        globals()[name] = rv
        return rv
    if name not in _semiring_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    words = name.split("_")
    for i in range(1, len(words)):  # pragma: no branch
        monoid_name = "_".join(words[:i])
        if not hasattr(_monoid.numpy, monoid_name):
            continue
        binary_name = "_".join(words[i:])
        if hasattr(_binary.numpy, binary_name):  # pragma: no branch
            break
    get_semiring(
        getattr(_monoid.numpy, monoid_name),
        getattr(_binary.numpy, binary_name),
        name=f"numpy.{name}",
    )
    return globals()[name]

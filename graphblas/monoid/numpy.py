"""Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.readthedocs.io/en/stable/reference/numpysupported.html#math-operations

"""
import numpy as _np

from .. import _STANDARD_OPERATOR_NAMES
from .. import binary as _binary
from .. import config as _config
from .. import monoid as _monoid
from ..core import _has_numba, _supports_udfs
from ..dtypes import _supports_complex

if _has_numba:
    import numba as _numba

_delayed = {}
_complex_dtypes = {"FC32", "FC64"}
_float_dtypes = {"FP32", "FP64"}
_int_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
_bool_int_dtypes = _int_dtypes | {"BOOL"}

_monoid_identities = {
    # Math operations
    "add": 0,
    "multiply": 1,
    "logaddexp": dict.fromkeys(_float_dtypes, -_np.inf),
    "logaddexp2": dict.fromkeys(_float_dtypes, -_np.inf),
    "gcd": dict.fromkeys(_int_dtypes, 0),
    # Trigonometric functions
    "hypot": dict.fromkeys(_float_dtypes, 0.0),
    # Bit-twiddling functions
    "bitwise_and": {dtype: True if dtype == "BOOL" else -1 for dtype in _bool_int_dtypes},
    "bitwise_or": dict.fromkeys(_bool_int_dtypes, 0),
    "bitwise_xor": dict.fromkeys(_bool_int_dtypes, 0),
    # Comparison functions
    "equal": {"BOOL": True},
    "logical_and": {"BOOL": True},
    "logical_or": {"BOOL": True},
    "logical_xor": {"BOOL": False},
    "maximum": {
        "BOOL": False,
        "INT8": _np.iinfo(_np.int8).min,
        "UINT8": 0,
        "INT16": _np.iinfo(_np.int16).min,
        "UINT16": 0,
        "INT32": _np.iinfo(_np.int32).min,
        "UINT32": 0,
        "INT64": _np.iinfo(_np.int64).min,
        "UINT64": 0,
        "FP32": -_np.inf,
        "FP64": -_np.inf,
    },
    "minimum": {
        "BOOL": True,
        "INT8": _np.iinfo(_np.int8).max,
        "UINT8": _np.iinfo(_np.uint8).max,
        "INT16": _np.iinfo(_np.int16).max,
        "UINT16": _np.iinfo(_np.uint16).max,
        "INT32": _np.iinfo(_np.int32).max,
        "UINT32": _np.iinfo(_np.uint32).max,
        "INT64": _np.iinfo(_np.int64).max,
        "UINT64": _np.iinfo(_np.uint64).max,
        "FP32": _np.inf,
        "FP64": _np.inf,
    },
    "fmax": {
        # More conditionally added below
        "FP32": -_np.inf,  # or _np.nan?
        "FP64": -_np.inf,  # or _np.nan?
    },
    "fmin": {
        # More conditionally added below
        "FP32": _np.inf,  # or _np.nan?
        "FP64": _np.inf,  # or _np.nan?
    },
}
if _supports_complex:
    _monoid_identities["fmax"].update(dict.fromkeys(_complex_dtypes, complex(-_np.inf, -_np.inf)))
    _monoid_identities["fmin"].update(dict.fromkeys(_complex_dtypes, complex(_np.inf, _np.inf)))
    _monoid_identities["maximum"].update(
        dict.fromkeys(_complex_dtypes, complex(-_np.inf, -_np.inf))
    )
    _monoid_identities["minimum"].update(dict.fromkeys(_complex_dtypes, complex(_np.inf, _np.inf)))

# To increase import speed, only call njit when `_config.get("mapnumpy")` is False
if (
    _config.get("mapnumpy")
    or _has_numba
    and type(_numba.njit(lambda x, y: _np.fmax(x, y))(1, 2))  # pragma: no branch (numba)
    is not float
):
    # Incorrect behavior was introduced in numba 0.56.2 and numpy 1.23
    # See: https://github.com/numba/numba/issues/8478
    # MAINT: we may be able to remove the behavior-based check above in 2025
    _monoid_identities["fmax"].update(
        {
            "BOOL": False,
            "INT8": _np.iinfo(_np.int8).min,
            "UINT8": 0,
            "INT16": _np.iinfo(_np.int8).min,
            "UINT16": 0,
            "INT32": _np.iinfo(_np.int8).min,
            "UINT32": 0,
            "INT64": _np.iinfo(_np.int8).min,
            "UINT64": 0,
        }
    )
    _monoid_identities["fmin"].update(
        {
            "BOOL": True,
            "INT8": _np.iinfo(_np.int8).max,
            "UINT8": _np.iinfo(_np.uint8).max,
            "INT16": _np.iinfo(_np.int16).max,
            "UINT16": _np.iinfo(_np.uint16).max,
            "INT32": _np.iinfo(_np.int32).max,
            "UINT32": _np.iinfo(_np.uint32).max,
            "INT64": _np.iinfo(_np.int64).max,
            "UINT64": _np.iinfo(_np.uint64).max,
        }
    )
    _fmin_is_float = False
else:
    _fmin_is_float = True

_STANDARD_OPERATOR_NAMES.update(f"monoid.numpy.{name}" for name in _monoid_identities)
__all__ = list(_monoid_identities)
_numpy_to_graphblas = {
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
}
# _graphblas_to_numpy = {val: key for key, val in _numpy_to_graphblas.items()}  # Soon...
# Not included: maximum, minimum, gcd, hypot, logaddexp, logaddexp2

# True if ``monoid(x, x) == x`` for any x.
_idempotent = {
    "bitwise_and",
    "bitwise_or",
    "fmax",
    "fmin",
    "gcd",
    "logical_and",
    "logical_or",
    "maximum",
    "minimum",
}


def __dir__():
    if not _supports_udfs and not _config.get("mapnumpy"):
        return globals().keys()  # FLAKY COVERAGE
    attrs = _delayed.keys() | _monoid_identities.keys()
    if not _supports_udfs:
        attrs &= _numpy_to_graphblas.keys()
    return attrs | globals().keys()


def __getattr__(name):
    if name in _delayed:
        func, kwargs = _delayed.pop(name)
        if type(kwargs["binaryop"]) is str:
            from ..binary import from_string

            kwargs["binaryop"] = from_string(kwargs["binaryop"])
        rv = func(**kwargs)
        globals()[name] = rv
        return rv
    if name not in _monoid_identities:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _config.get("mapnumpy") and name in _numpy_to_graphblas:
        globals()[name] = getattr(_monoid, _numpy_to_graphblas[name])
    else:
        func = getattr(_binary.numpy, name)
        _monoid.register_new(
            f"numpy.{name}", func, _monoid_identities[name], is_idempotent=name in _idempotent
        )
    return globals()[name]

""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#math-operations

"""
import numpy as _np

from .. import binary as _binary
from .. import config as _config
from .. import monoid as _monoid
from .. import operator as _operator
from ..dtypes import _supports_complex

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
        "BOOL": False,
        "INT8": _np.iinfo(_np.int8).min,
        "UINT8": 0,
        "INT16": _np.iinfo(_np.int8).min,
        "UINT16": 0,
        "INT32": _np.iinfo(_np.int8).min,
        "UINT32": 0,
        "INT64": _np.iinfo(_np.int8).min,
        "UINT64": 0,
        "FP32": -_np.inf,  # or _np.nan?
        "FP64": -_np.inf,  # or _np.nan?
    },
    "fmin": {
        "BOOL": True,
        "INT8": _np.iinfo(_np.int8).max,
        "UINT8": _np.iinfo(_np.uint8).max,
        "INT16": _np.iinfo(_np.int16).max,
        "UINT16": _np.iinfo(_np.uint16).max,
        "INT32": _np.iinfo(_np.int32).max,
        "UINT32": _np.iinfo(_np.uint32).max,
        "INT64": _np.iinfo(_np.int64).max,
        "UINT64": _np.iinfo(_np.uint64).max,
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

_operator._STANDARD_OPERATOR_NAMES.update(f"monoid.numpy.{name}" for name in _monoid_identities)
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
# Not included: maximum, minimum, gcd, hypot, logaddexp, logaddexp2


def __dir__():
    return globals().keys() | _delayed.keys() | _monoid_identities.keys()


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
        _operator.Monoid.register_new(f"numpy.{name}", func, _monoid_identities[name])
    return globals()[name]

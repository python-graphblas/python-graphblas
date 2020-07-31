""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#math-operations

"""
import numpy as np
from .. import ops, binary

_float_dtypes = {"FP32", "FP64"}
_int_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
_bool_int_dtypes = _int_dtypes | {"BOOL"}

_monoid_identities = {
    # Math operations
    "add": 0,
    "multiply": 1,
    "logaddexp": dict.fromkeys(_float_dtypes, -np.inf),
    "logaddexp2": dict.fromkeys(_float_dtypes, -np.inf),
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
        "INT8": np.iinfo(np.int8).min,
        "UINT8": 0,
        "INT16": np.iinfo(np.int16).min,
        "UINT16": 0,
        "INT32": np.iinfo(np.int32).min,
        "UINT32": 0,
        "INT64": np.iinfo(np.int64).min,
        "UINT64": 0,
        "FP32": -np.inf,
        "FP64": -np.inf,
    },
    "minimum": {
        "BOOL": True,
        "INT8": np.iinfo(np.int8).max,
        "UINT8": np.iinfo(np.uint8).max,
        "INT16": np.iinfo(np.int16).max,
        "UINT16": np.iinfo(np.uint16).max,
        "INT32": np.iinfo(np.int32).max,
        "UINT32": np.iinfo(np.uint32).max,
        "INT64": np.iinfo(np.int64).max,
        "UINT64": np.iinfo(np.uint64).max,
        "FP32": np.inf,
        "FP64": np.inf,
    },
    "fmax": {
        "BOOL": False,
        "INT8": np.iinfo(np.int8).min,
        "UINT8": 0,
        "INT16": np.iinfo(np.int8).min,
        "UINT16": 0,
        "INT32": np.iinfo(np.int8).min,
        "UINT32": 0,
        "INT64": np.iinfo(np.int8).min,
        "UINT64": 0,
        "FP32": -np.inf,  # or np.nan?
        "FP64": -np.inf,  # or np.nan?
    },
    "fmin": {
        "BOOL": True,
        "INT8": np.iinfo(np.int8).max,
        "UINT8": np.iinfo(np.uint8).max,
        "INT16": np.iinfo(np.int16).max,
        "UINT16": np.iinfo(np.uint16).max,
        "INT32": np.iinfo(np.int32).max,
        "UINT32": np.iinfo(np.uint32).max,
        "INT64": np.iinfo(np.int64).max,
        "UINT64": np.iinfo(np.uint64).max,
        "FP32": np.inf,  # or np.nan?
        "FP64": np.inf,  # or np.nan?
    },
}


def __dir__():
    return list(_monoid_identities)


def __getattr__(name):
    if name not in _monoid_identities:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    func = getattr(binary.numpy, name)
    ops.Monoid.register_new(f"numpy.{name}", func, _monoid_identities[name])
    return globals()[name]

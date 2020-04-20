import numpy as np
from .. import dtypes, ops
from . import binary

_float_dtypes = {dtypes.FP32, dtypes.FP64}
_int_dtypes = {
    dtypes.INT8, dtypes.UINT8, dtypes.INT16, dtypes.UINT16,
    dtypes.INT32, dtypes.UINT32, dtypes.INT64, dtypes.UINT64,
}
_bool_int_dtypes = _int_dtypes | {dtypes.BOOL}

_monoid_identities = {
    # Math operations
    'add': 0,
    'multiply': 1,
    'logaddexp': dict.fromkeys(_float_dtypes, -np.inf),
    'logaddexp2': dict.fromkeys(_float_dtypes, -np.inf),
    'gcd': dict.fromkeys(_int_dtypes, 0),

    # Trigonometric functions
    'hypot': dict.fromkeys(_float_dtypes, 0.),

    # Bit-twiddling functions
    'bitwise_and': {dtype: True if dtype is dtypes.BOOL else -1 for dtype in _bool_int_dtypes},
    'bitwise_or': dict.fromkeys(_bool_int_dtypes, 0),
    'bitwise_xor': dict.fromkeys(_bool_int_dtypes, 0),

    # Comparison functions
    'equal': {dtypes.BOOL: True},
    'logical_and': {dtypes.BOOL: True},
    'logical_or': {dtypes.BOOL: True},
    'logical_xor': {dtypes.BOOL: False},
    'maximum': {
        dtypes.BOOL: False,
        dtypes.INT8: np.iinfo(np.int8).min,
        dtypes.UINT8: 0,
        dtypes.INT16: np.iinfo(np.int16).min,
        dtypes.UINT16: 0,
        dtypes.INT32: np.iinfo(np.int32).min,
        dtypes.UINT32: 0,
        dtypes.INT64: np.iinfo(np.int64).min,
        dtypes.UINT64: 0,
        dtypes.FP32: -np.inf,
        dtypes.FP64: -np.inf,
    },
    'minimum': {
        dtypes.BOOL: True,
        dtypes.INT8: np.iinfo(np.int8).max,
        dtypes.UINT8: np.iinfo(np.uint8).max,
        dtypes.INT16: np.iinfo(np.int16).max,
        dtypes.UINT16: np.iinfo(np.uint16).max,
        dtypes.INT32: np.iinfo(np.int32).max,
        dtypes.UINT32: np.iinfo(np.uint32).max,
        dtypes.INT64: np.iinfo(np.int64).max,
        dtypes.UINT64: np.iinfo(np.uint64).max,
        dtypes.FP32: np.inf,
        dtypes.FP64: np.inf,
    },
    'fmax': {
        dtypes.BOOL: False,
        dtypes.INT8: np.iinfo(np.int8).min,
        dtypes.UINT8: 0,
        dtypes.INT16: np.iinfo(np.int8).min,
        dtypes.UINT16: 0,
        dtypes.INT32: np.iinfo(np.int8).min,
        dtypes.UINT32: 0,
        dtypes.INT64: np.iinfo(np.int8).min,
        dtypes.UINT64: 0,
        dtypes.FP32: -np.inf,  # or np.nan?
        dtypes.FP64: -np.inf,  # or np.nan?
    },
    'fmin': {
        dtypes.BOOL: True,
        dtypes.INT8: np.iinfo(np.int8).max,
        dtypes.UINT8: np.iinfo(np.uint8).max,
        dtypes.INT16: np.iinfo(np.int16).max,
        dtypes.UINT16: np.iinfo(np.uint16).max,
        dtypes.INT32: np.iinfo(np.int32).max,
        dtypes.UINT32: np.iinfo(np.uint32).max,
        dtypes.INT64: np.iinfo(np.int64).max,
        dtypes.UINT64: np.iinfo(np.uint64).max,
        dtypes.FP32: np.inf,  # or np.nan?
        dtypes.FP64: np.inf,  # or np.nan?
    },
}


def __dir__():
    return list(_monoid_identities)


def __getattr__(name):
    if name not in _monoid_identities:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    func = getattr(binary, name)
    monoid = ops.Monoid.register_anonymous(func, _monoid_identities[name], name)
    globals()[name] = monoid
    return monoid

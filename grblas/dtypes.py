import re
import numpy as np
import numba
from . import lib


class DataType:
    __slots__ = ['name', 'gb_type', 'c_type', 'numba_type']

    def __init__(self, name, gb_type, c_type, numba_type):
        self.name = name
        self.gb_type = gb_type
        self.c_type = c_type
        self.numba_type = numba_type

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DataType):
            return self.gb_type == other.gb_type
        else:
            # Attempt to use `other` as a lookup key
            try:
                other = lookup(other)
                return self == other
            except KeyError:
                return False

    @classmethod
    def from_pytype(cls, pytype):
        if pytype is int:
            return INT64
        if pytype is float:
            return FP64
        if pytype is bool:
            return BOOL
        raise TypeError(f'Invalid pytype: {pytype}')


BOOL = DataType('BOOL', lib.GrB_BOOL, '_Bool', numba.types.bool_)
INT8 = DataType('INT8', lib.GrB_INT8, 'int8_t', numba.types.int8)
UINT8 = DataType('UINT8', lib.GrB_UINT8, 'uint8_t', numba.types.uint8)
INT16 = DataType('INT16', lib.GrB_INT16, 'int16_t', numba.types.int16)
UINT16 = DataType('UINT16', lib.GrB_UINT16, 'uint16_t', numba.types.uint16)
INT32 = DataType('INT32', lib.GrB_INT32, 'int32_t', numba.types.int32)
UINT32 = DataType('UINT32', lib.GrB_UINT32, 'uint32_t', numba.types.uint32)
INT64 = DataType('INT64', lib.GrB_INT64, 'int64_t', numba.types.int64)
UINT64 = DataType('UINT64', lib.GrB_UINT64, 'uint64_t', numba.types.uint64)
FP32 = DataType('FP32', lib.GrB_FP32, 'float', numba.types.float32)
FP64 = DataType('FP64', lib.GrB_FP64, 'double', numba.types.float64)

# Used for testing user-defined functions
_sample_values = {
    INT8.name: np.int8(1),
    UINT8.name: np.uint8(1),
    INT16.name: np.int16(1),
    UINT16.name: np.uint16(1),
    INT32.name: np.int32(1),
    UINT32.name: np.uint32(1),
    INT64.name: np.int64(1),
    UINT64.name: np.uint64(1),
    FP32.name: np.float32(0.5),
    FP64.name: np.float64(0.5),
    BOOL.name: np.bool_(True),
}

# Create register to easily lookup types by name, gb_type, or c_type
_registry = {}
for dtype in [BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64]:
    _registry[dtype.name] = dtype
    _registry[dtype.gb_type] = dtype
    _registry[dtype.c_type] = dtype
    _registry[dtype.numba_type] = dtype
    _registry[dtype.numba_type.name] = dtype
    val = _sample_values[dtype.name]
    _registry[val.dtype] = dtype
    _registry[val.dtype.name] = dtype
# Upcast numpy float16 to float32
_registry[np.dtype(np.float16)] = FP32
_registry['float16'] = FP32

# Add some common Python types as lookup keys
_registry[int] = DataType.from_pytype(int)
_registry[float] = DataType.from_pytype(float)
_registry[bool] = DataType.from_pytype(bool)


def lookup(key):
    # Check for silly lookup where key is already a DataType
    if isinstance(key, DataType):
        return key
    try:
        return _registry[key]
    except KeyError:
        if hasattr(key, 'name'):
            return _registry[key.name]
        else:
            try:
                return lookup(np.dtype(key))
            except Exception:
                pass
            raise


_bits_pattern = re.compile(r'\D+(\d+)$')


def unify(type1, type2):
    """
    Returns a type that can hold both type1 and type2

    For example:
    unify(INT32, INT64) -> INT64
    unify(INT8, UINT16) -> INT32
    unify(BOOL, UINT16) -> UINT16
    unify(FP32, INT32) -> FP32
    """
    if type1 == type2:
        return type1
    # Bools fit anywhere
    if type1 == BOOL:
        return type2
    if type2 == BOOL:
        return type1
    # Compute bit numbers for comparison
    num1 = int(_bits_pattern.match(type1.name).group(1))
    num2 = int(_bits_pattern.match(type2.name).group(1))
    # Floats anywhere requires floats
    if type1.name[0] == 'F' or type2.name[0] == 'F':
        return lookup(f'FP{max(num1, num2)}')
    # Both ints or both uints is easy
    if type1.name[0] == type2.name[0]:
        return type2 if num2 > num1 else type1
    # Mixed int/uint is a little harder
    # Need to double the uint bit number to safely store as int
    # Beyond 64, we have to move to float
    if type1.name[0] == 'U':
        num1 *= 2
    else:  # type2 is unsigned
        num2 *= 2
    maxnum = max(num1, num2)
    if maxnum > 64:
        return FP64
    return lookup(f'INT{maxnum}')

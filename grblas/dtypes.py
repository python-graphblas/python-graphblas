from . import lib
import numba


class DataType:
    __slots__ = ['name', 'gb_type', 'c_type', 'numba_type']

    def __init__(self, name, gb_type, c_type, numba_type):
        self.name = name
        self.gb_type = gb_type
        self.c_type = c_type
        self.numba_type = numba_type
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash((self.name, self.c_type))
    
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
    BOOL: True,
    INT8: -3,
    UINT8: 3,
    INT16: -3,
    UINT16: 3,
    INT32: -3,
    UINT32: 3,
    INT64: -3,
    UINT64: 3,
    FP32: 3.14,
    FP64: 3.14
}

# Create register to easily lookup types by name, gb_type, or c_type
_registry = {}
for x in _sample_values:
    _registry[x.name] = x
    _registry[x.gb_type] = x
    _registry[x.c_type] = x
del x
# Add some common Python types as lookup keys
_registry[int] = DataType.from_pytype(int)
_registry[float] = DataType.from_pytype(float)
_registry[bool] = DataType.from_pytype(bool)


def lookup(key):
    # Check for silly lookup where key is already a DataType
    if isinstance(key, DataType):
        return key
    return _registry[key]

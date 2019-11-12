import pytest
from _grblas import lib
from grblas import dtypes

all_dtypes = (
    dtypes.BOOL,
    dtypes.INT8,
    dtypes.INT16,
    dtypes.INT32,
    dtypes.INT64,
    dtypes.UINT8,
    dtypes.UINT16,
    dtypes.UINT32,
    dtypes.UINT64,
    dtypes.FP32,
    dtypes.FP64,
)

def test_names():
    assert dtypes.BOOL.name == 'BOOL'
    assert dtypes.INT8.name == 'INT8'
    assert dtypes.INT16.name == 'INT16'
    assert dtypes.INT32.name == 'INT32'
    assert dtypes.INT64.name == 'INT64'
    assert dtypes.UINT8.name == 'UINT8'
    assert dtypes.UINT16.name == 'UINT16'
    assert dtypes.UINT32.name == 'UINT32'
    assert dtypes.UINT64.name == 'UINT64'
    assert dtypes.FP32.name == 'FP32'
    assert dtypes.FP64.name == 'FP64'

def test_ctype():
    assert dtypes.BOOL.c_type == '_Bool'
    assert dtypes.INT8.c_type == 'int8_t'
    assert dtypes.INT16.c_type == 'int16_t'
    assert dtypes.INT32.c_type == 'int32_t'
    assert dtypes.INT64.c_type == 'int64_t'
    assert dtypes.UINT8.c_type == 'uint8_t'
    assert dtypes.UINT16.c_type == 'uint16_t'
    assert dtypes.UINT32.c_type == 'uint32_t'
    assert dtypes.UINT64.c_type == 'uint64_t'
    assert dtypes.FP32.c_type == 'float'
    assert dtypes.FP64.c_type == 'double'

def test_gbtype():
    assert dtypes.BOOL.gb_type == lib.GrB_BOOL
    assert dtypes.INT8.gb_type == lib.GrB_INT8
    assert dtypes.INT16.gb_type == lib.GrB_INT16
    assert dtypes.INT32.gb_type == lib.GrB_INT32
    assert dtypes.INT64.gb_type == lib.GrB_INT64
    assert dtypes.UINT8.gb_type == lib.GrB_UINT8
    assert dtypes.UINT16.gb_type == lib.GrB_UINT16
    assert dtypes.UINT32.gb_type == lib.GrB_UINT32
    assert dtypes.UINT64.gb_type == lib.GrB_UINT64
    assert dtypes.FP32.gb_type == lib.GrB_FP32
    assert dtypes.FP64.gb_type == lib.GrB_FP64

def test_lookup_by_name():
    for dt in all_dtypes:
        assert dtypes.lookup(dt.name) is dt

def test_lookup_by_ctype():
    for dt in all_dtypes:
        assert dtypes.lookup(dt.c_type) is dt

def test_lookup_by_gbtype():
    for dt in all_dtypes:
        assert dtypes.lookup(dt.gb_type) is dt

def test_lookup_by_dtype():
    assert dtypes.lookup(bool) == dtypes.BOOL
    assert dtypes.lookup(int) == dtypes.INT64
    assert dtypes.lookup(float) == dtypes.FP64

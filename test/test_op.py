import pytest
from grblas import lib
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, ops
from grblas import Vector, Matrix

def test_unaryop():
    assert UnaryOp.AINV['INT32'] == lib.GrB_AINV_INT32
    assert UnaryOp.AINV[dtypes.UINT16] == lib.GrB_AINV_UINT16

def test_binaryop():
    assert BinaryOp.PLUS['INT32'] == lib.GrB_PLUS_INT32
    assert BinaryOp.PLUS[dtypes.UINT16] == lib.GrB_PLUS_UINT16

def test_monoid():
    assert Monoid.MAX['INT32'] == lib.GxB_MAX_INT32_MONOID
    assert Monoid.MAX[dtypes.UINT16] == lib.GxB_MAX_UINT16_MONOID

def test_semiring():
    assert Semiring.MIN_PLUS['INT32'] == lib.GxB_MIN_PLUS_INT32
    assert Semiring.MIN_PLUS[dtypes.UINT16] == lib.GxB_MIN_PLUS_UINT16

def test_find_opclass_unaryop():
    assert ops.find_opclass(UnaryOp.MINV) == 'UnaryOp'
    assert ops.find_opclass(lib.GrB_MINV_INT64) == 'UnaryOp'

def test_find_opclass_binaryop():
    assert ops.find_opclass(BinaryOp.TIMES) == 'BinaryOp'
    assert ops.find_opclass(lib.GrB_TIMES_INT64) == 'BinaryOp'

def test_find_opclass_monoid():
    assert ops.find_opclass(Monoid.MAX) == 'Monoid'
    assert ops.find_opclass(lib.GxB_MAX_INT64_MONOID) == 'Monoid'

def test_find_opclass_semiring():
    assert ops.find_opclass(Semiring.PLUS_PLUS) == 'Semiring'
    assert ops.find_opclass(lib.GxB_PLUS_PLUS_INT64) == 'Semiring'

def test_find_opclass_invalid():
    assert ops.find_opclass('foobar') == ops.UNKNOWN_OPCLASS
    assert ops.find_opclass(lib.GrB_INP0) == ops.UNKNOWN_OPCLASS

def test_unaryop_udf():
    def plus_one(x):
        return x + 1
    UnaryOp.register_new('plus_one', plus_one)
    assert hasattr(UnaryOp, 'plus_one')
    assert UnaryOp.plus_one.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                      'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                      'FP32', 'FP64'}
    v = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    v[:] = v.apply(UnaryOp.plus_one)
    result = Vector.new_from_values([0,1,3], [2,3,-3], dtype=dtypes.INT32)
    assert v == result

def test_unaryop_udf_bool_result():
    pytest.xfail('not sure why numba has trouble compiling this')
    # def is_positive(x):
    #     return x > 0
    # UnaryOp.register_new('is_positive', is_positive)
    # assert hasattr(UnaryOp, 'is_positive')
    # assert UnaryOp.is_positive.types == {'INT8', 'INT16', 'INT32', 'INT64',
    #                                      'UINT8', 'UINT16', 'UINT32', 'UINT64',
    #                                      'FP32', 'FP64'}
    # v = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    # w = v.apply(UnaryOp.is_positive).new()
    # result = Vector.new_from_values([0,1,3], [True,True,False], dtype=dtypes.BOOL)
    # assert v == result

def test_binaryop_udf():
    def times_minus_sum(x, y):
        return x * y - (x + y)
    BinaryOp.register_new('bin_test_func', times_minus_sum)
    assert hasattr(BinaryOp, 'bin_test_func')
    assert BinaryOp.bin_test_func.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                            'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                            'FP32', 'FP64'}
    v1 = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    v2 = Vector.new_from_values([0,2,3], [2,3,7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, BinaryOp.bin_test_func).new()
    result = Vector.new_from_values([0,1,2,3], [-1,2,3,-31], dtype=dtypes.INT32)
    assert w == result

def test_monoid_udf():
    def plus_plus_one(x, y):
        return x + y + 1
    BinaryOp.register_new('plus_plus_one', plus_plus_one)
    Monoid.register_new('plus_plus_one', BinaryOp.plus_plus_one, -1)
    assert hasattr(Monoid, 'plus_plus_one')
    assert Monoid.plus_plus_one.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                          'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                          'FP32', 'FP64'}
    v1 = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    v2 = Vector.new_from_values([0,2,3], [2,3,7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, Monoid.plus_plus_one).new()
    result = Vector.new_from_values([0,1,2,3], [4,2,3,4], dtype=dtypes.INT32)
    assert w == result

def test_semiring_udf():
    def plus_plus_two(x, y):
        return x + y + 2
    BinaryOp.register_new('plus_plus_two', plus_plus_two)
    Semiring.register_new('extra_twos', Monoid.PLUS, BinaryOp.plus_plus_two)
    v = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    A = Matrix.new_from_values([0,0,0,0,3,3,3,3], [0,1,2,3,0,1,2,3], [2,3,4,5,6,7,8,9], dtype=dtypes.INT32)
    w = v.vxm(A, Semiring.extra_twos).new()
    result = Vector.new_from_values([0,1,2,3], [9,11,13,15], dtype=dtypes.INT32)
    assert w == result

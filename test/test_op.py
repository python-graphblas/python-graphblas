import pytest
from grblas import lib
from grblas import unary, binary, monoid, semiring
from grblas import dtypes, ops
from grblas import Vector, Matrix
from grblas.ops import UnaryOp, BinaryOp, Monoid, Semiring


def test_unaryop():
    assert unary.ainv['INT32'] == lib.GrB_AINV_INT32
    assert unary.ainv[dtypes.UINT16] == lib.GrB_AINV_UINT16


def test_binaryop():
    assert binary.plus['INT32'] == lib.GrB_PLUS_INT32
    assert binary.plus[dtypes.UINT16] == lib.GrB_PLUS_UINT16


def test_monoid():
    assert monoid.max['INT32'] == lib.GxB_MAX_INT32_MONOID
    assert monoid.max[dtypes.UINT16] == lib.GxB_MAX_UINT16_MONOID


def test_semiring():
    assert semiring.min_plus['INT32'] == lib.GxB_MIN_PLUS_INT32
    assert semiring.min_plus[dtypes.UINT16] == lib.GxB_MIN_PLUS_UINT16


def test_find_opclass_unaryop():
    assert ops.find_opclass(unary.minv) == 'UnaryOp'
    assert ops.find_opclass(lib.GrB_MINV_INT64) == 'UnaryOp'


def test_find_opclass_binaryop():
    assert ops.find_opclass(binary.times) == 'BinaryOp'
    assert ops.find_opclass(lib.GrB_TIMES_INT64) == 'BinaryOp'


def test_find_opclass_monoid():
    assert ops.find_opclass(monoid.max) == 'Monoid'
    assert ops.find_opclass(lib.GxB_MAX_INT64_MONOID) == 'Monoid'


def test_find_opclass_semiring():
    assert ops.find_opclass(semiring.plus_plus) == 'Semiring'
    assert ops.find_opclass(lib.GxB_PLUS_PLUS_INT64) == 'Semiring'


def test_find_opclass_invalid():
    assert ops.find_opclass('foobar') == ops.UNKNOWN_OPCLASS
    assert ops.find_opclass(lib.GrB_INP0) == ops.UNKNOWN_OPCLASS


def test_unaryop_udf():
    def plus_one(x):
        return x + 1
    UnaryOp.register_new('plus_one', plus_one)
    assert hasattr(unary, 'plus_one')
    assert unary.plus_one.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                    'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                    'FP32', 'FP64'}
    v = Vector.new_from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.plus_one)
    result = Vector.new_from_values([0, 1, 3], [2, 3, -3], dtype=dtypes.INT32)
    assert v.isequal(result)


def test_unaryop_udf_bool_result():
    pytest.xfail('not sure why numba has trouble compiling this')
    # def is_positive(x):
    #     return x > 0
    # unary.register_new('is_positive', is_positive)
    # assert hasattr(UnaryOp, 'is_positive')
    # assert unary.is_positive.types == {'INT8', 'INT16', 'INT32', 'INT64',
    #                                    'UINT8', 'UINT16', 'UINT32', 'UINT64',
    #                                    'FP32', 'FP64'}
    # v = Vector.new_from_values([0,1,3], [1,2,-4], dtype=dtypes.INT32)
    # w = v.apply(unary.is_positive).new()
    # result = Vector.new_from_values([0,1,3], [True,True,False], dtype=dtypes.BOOL)
    # assert v == result


def test_binaryop_udf():
    def times_minus_sum(x, y):
        return x * y - (x + y)
    BinaryOp.register_new('bin_test_func', times_minus_sum)
    assert hasattr(binary, 'bin_test_func')
    assert binary.bin_test_func.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                          'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                          'FP32', 'FP64'}
    v1 = Vector.new_from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.new_from_values([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, binary.bin_test_func).new()
    result = Vector.new_from_values([0, 1, 2, 3], [-1, 2, 3, -31], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_monoid_udf():
    def plus_plus_one(x, y):
        return x + y + 1
    BinaryOp.register_new('plus_plus_one', plus_plus_one)
    Monoid.register_new('plus_plus_one', binary.plus_plus_one, -1)
    assert hasattr(monoid, 'plus_plus_one')
    assert monoid.plus_plus_one.types == {'INT8', 'INT16', 'INT32', 'INT64',
                                          'UINT8', 'UINT16', 'UINT32', 'UINT64',
                                          'FP32', 'FP64'}
    v1 = Vector.new_from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.new_from_values([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, monoid.plus_plus_one).new()
    result = Vector.new_from_values([0, 1, 2, 3], [4, 2, 3, 4], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_semiring_udf():
    def plus_plus_two(x, y):
        return x + y + 2
    BinaryOp.register_new('plus_plus_two', plus_plus_two)
    Semiring.register_new('extra_twos', monoid.plus, binary.plus_plus_two)
    v = Vector.new_from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    A = Matrix.new_from_values([0, 0, 0, 0, 3, 3, 3, 3],
                               [0, 1, 2, 3, 0, 1, 2, 3],
                               [2, 3, 4, 5, 6, 7, 8, 9], dtype=dtypes.INT32)
    w = v.vxm(A, semiring.extra_twos).new()
    result = Vector.new_from_values([0, 1, 2, 3], [9, 11, 13, 15], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_binary_updates():
    assert not hasattr(binary, 'div')
    assert binary.cdiv['INT64'] == lib.GrB_DIV_INT64
    vec1 = Vector.new_from_values([0], [1], dtype=dtypes.INT64)
    vec2 = Vector.new_from_values([0], [2], dtype=dtypes.INT64)
    result = vec1.ewise_mult(vec2, binary.truediv).new()
    assert result.isclose(Vector.new_from_values([0], [0.5], dtype=dtypes.FP64), check_dtype=True)
    vec4 = Vector.new_from_values([0], [-3], dtype=dtypes.INT64)
    result2 = vec4.ewise_mult(vec2, binary.cdiv).new()
    assert result2.isequal(Vector.new_from_values([0], [-1], dtype=dtypes.INT64), check_dtype=True)
    result3 = vec4.ewise_mult(vec2, binary.floordiv).new()
    assert result3.isequal(Vector.new_from_values([0], [-2], dtype=dtypes.INT64), check_dtype=True)

import inspect
import itertools
import pickle
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import grblas
from grblas import Matrix, Scalar, Vector, agg, binary, dtypes, monoid, semiring, unary
from grblas.exceptions import (
    DimensionMismatch,
    IndexOutOfBound,
    InvalidValue,
    OutputNotEmpty,
)


@pytest.fixture
def A():
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
    ]
    return Matrix.from_values(*data)


@pytest.fixture
def v():
    data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    return Vector.from_values(*data)


def test_new():
    u = Vector.new(dtypes.INT8, 17)
    assert u.dtype == "INT8"
    assert u.nvals == 0
    assert u.size == 17


def test_large_vector():
    u = Vector.from_values([0, 2 ** 59], [0, 1])
    assert u.size == 2 ** 59 + 1
    assert u[2 ** 59].value == 1
    with pytest.raises(InvalidValue):
        Vector.from_values([0, 2 ** 64 - 2], [0, 1])
    with pytest.raises(OverflowError):
        Vector.from_values([0, 2 ** 64], [0, 1])


def test_dup(v):
    u = v.dup()
    assert u is not v
    assert u.dtype == v.dtype
    assert u.nvals == v.nvals
    assert u.size == v.size
    # Ensure they are not the same backend object
    v[0] = 1000
    assert u[0].value != 1000
    # extended functionality
    w = Vector.from_values([0, 1], [0, 2.5], dtype=dtypes.FP64)
    x = w.dup(dtype=dtypes.INT64)
    assert x.isequal(Vector.from_values([0, 1], [0, 2], dtype=dtypes.INT64), check_dtype=True)
    x = w.dup(mask=w.V)
    assert x.isequal(Vector.from_values([1], [2.5], dtype=dtypes.FP64), check_dtype=True)
    x = w.dup(dtype=dtypes.INT64, mask=w.V)
    assert x.isequal(Vector.from_values([1], [2], dtype=dtypes.INT64), check_dtype=True)


def test_from_values():
    u = Vector.from_values([0, 1, 3], [True, False, True])
    assert u.size == 4
    assert u.nvals == 3
    assert u.dtype == bool
    u2 = Vector.from_values([0, 1, 3], [12.3, 12.4, 12.5], size=17)
    assert u2.size == 17
    assert u2.nvals == 3
    assert u2.dtype == float
    u3 = Vector.from_values([0, 1, 1], [1, 2, 3], size=10, dup_op=binary.times)
    assert u3.size == 10
    assert u3.nvals == 2  # duplicates were combined
    assert u3.dtype == int
    assert u3[1].value == 6  # 2*3
    with pytest.raises(ValueError, match="Duplicate indices found"):
        # Duplicate indices requires a dup_op
        Vector.from_values([0, 1, 1], [True, True, True])
    with pytest.raises(ValueError, match="No indices provided. Unable to infer size."):
        Vector.from_values([], [])

    # Changed: Assume empty value is float64 (like numpy)
    # with pytest.raises(ValueError, match="No values provided. Unable to determine type"):
    w = Vector.from_values([], [], size=10)
    assert w.size == 10
    assert w.nvals == 0
    assert w.dtype == dtypes.FP64

    with pytest.raises(ValueError, match="No indices provided. Unable to infer size"):
        Vector.from_values([], [], dtype=dtypes.INT64)
    u4 = Vector.from_values([], [], size=10, dtype=dtypes.INT64)
    u5 = Vector.new(dtypes.INT64, size=10)
    assert u4.isequal(u5, check_dtype=True)

    # we check index dtype if given numpy array
    with pytest.raises(ValueError, match="indices must be integers, not float64"):
        Vector.from_values(np.array([1.2, 3.4]), [1, 2])
    # but coerce index if given Python lists (we defer to numpy casting)
    u6 = Vector.from_values([1.2, 3.4], [1, 2])
    assert u6.isequal(Vector.from_values([1, 3], [1, 2]))

    # mis-matched sizes
    with pytest.raises(ValueError, match="`indices` and `values` lengths must match"):
        Vector.from_values([0], [1, 2])


def test_from_values_scalar():
    u = Vector.from_values([0, 1, 3], 7)
    assert u.size == 4
    assert u.nvals == 3
    assert u.dtype == dtypes.INT64
    assert u.ss.is_iso
    assert u.reduce(monoid.any) == 7

    # ignore duplicate indices; iso trumps duplicates!
    u = Vector.from_values([0, 1, 1, 3], 7)
    assert u.size == 4
    assert u.nvals == 3
    assert u.ss.is_iso
    assert u.reduce(monoid.any) == 7
    with pytest.raises(ValueError, match="dup_op must be None"):
        Vector.from_values([0, 1, 1, 3], 7, dup_op=binary.plus)


def test_clear(v):
    v.clear()
    assert v.nvals == 0
    assert v.size == 7


def test_resize(v):
    assert v.size == 7
    assert v.nvals == 4
    v.resize(20)
    assert v.size == 20
    assert v.nvals == 4
    assert v[19].value is None
    v.resize(4)
    assert v.size == 4
    assert v.nvals == 2


def test_size(v):
    assert v.size == 7


def test_nvals(v):
    assert v.nvals == 4


def test_build(v):
    assert v.nvals == 4
    v.clear()
    v.build([0, 6], [1, 2])
    assert v.nvals == 2
    with pytest.raises(OutputNotEmpty):
        v.build([1, 5], [3, 4])
    assert v.nvals == 2  # should be unchanged
    # We can clear though
    v.build([1, 2, 5], [2, 3, 4], clear=True)
    assert v.nvals == 3
    v.clear()
    with pytest.raises(IndexOutOfBound):
        v.build([0, 11], [1, 1])
    w = Vector.new(int, size=3)
    w.build([0, 11], [1, 1], size=12)
    assert w.isequal(Vector.from_values([0, 11], [1, 1]))


def test_build_scalar(v):
    with pytest.raises(OutputNotEmpty):
        v.ss.build_scalar([1, 5], 3)
    v.clear()
    v.ss.build_scalar([1, 5], 3)
    assert v.nvals == 2
    assert v.ss.is_iso


def test_extract_values(v):
    idx, vals = v.to_values()
    np.testing.assert_array_equal(idx, (1, 3, 4, 6))
    np.testing.assert_array_equal(vals, (1, 1, 2, 0))
    assert idx.dtype == np.uint64
    assert vals.dtype == np.int64

    idx, vals = v.to_values(dtype=int)
    np.testing.assert_array_equal(idx, (1, 3, 4, 6))
    np.testing.assert_array_equal(vals, (1, 1, 2, 0))
    assert idx.dtype == np.uint64
    assert vals.dtype == np.int64

    idx, vals = v.to_values(dtype=float)
    np.testing.assert_array_equal(idx, (1, 3, 4, 6))
    np.testing.assert_array_equal(vals, (1, 1, 2, 0))
    assert idx.dtype == np.uint64
    assert vals.dtype == np.float64


def test_extract_input_mask():
    v = Vector.from_values([0, 1, 2], [0, 1, 2])
    m = Vector.from_values([0, 2], [0, 2])
    result = v[[0, 1]].new(input_mask=m.S)
    expected = Vector.from_values([0], [0], size=2)
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=m.S) << v[[0, 1]]
    assert result.isequal(expected)
    with pytest.raises(ValueError, match="Size of `input_mask` does not match size of input"):
        v[[0, 2]].new(input_mask=expected.S)
    with pytest.raises(TypeError, match="`input_mask` argument may only be used for extract"):
        v(input_mask=m.S) << 1


def test_extract_element(v):
    assert v[1].value == 1
    assert v[6].new() == 0
    with pytest.raises(TypeError, match="Invalid type for index"):
        v[object()]
    with pytest.raises(IndexError):
        v[100]


def test_set_element(v):
    assert v[0].value is None
    assert v[1].value == 1
    v[0] = 12
    v[1] << 9
    assert v[0].value == 12
    assert v[1].new() == 9


def test_remove_element(v):
    assert v[1].value == 1
    del v[1]
    assert v[1].value is None
    assert v[4].value == 2
    with pytest.raises(TypeError, match="Remove Element only supports"):
        del v[1:3]


def test_vxm(v, A):
    w = v.vxm(A, semiring.plus_times).new()
    result = Vector.from_values([0, 2, 3, 4, 5, 6], [3, 3, 0, 8, 14, 4])
    assert w.isequal(result)


def test_vxm_transpose(v, A):
    w = v.vxm(A.T, semiring.plus_times).new()
    result = Vector.from_values([0, 1, 6], [5, 16, 13])
    assert w.isequal(result)


def test_vxm_nonsquare(v):
    A = Matrix.from_values([0, 3], [0, 1], [10, 20], nrows=7, ncols=2)
    u = Vector.new(v.dtype, size=2)
    u().update(v.vxm(A, semiring.min_plus))
    result = Vector.from_values([1], [21])
    assert u.isequal(result)
    w1 = v.vxm(A, semiring.min_plus).new()
    assert w1.isequal(u)
    # Test the transpose case
    v2 = Vector.from_values([0, 1], [1, 2])
    w2 = v2.vxm(A.T, semiring.min_plus).new()
    assert w2.size == 7


def test_vxm_mask(v, A):
    val_mask = Vector.from_values([0, 1, 2, 3, 4], [True, False, False, True, True], size=7)
    struct_mask = Vector.from_values([0, 3, 4], [False, False, False], size=7)
    u = v.dup()
    u(struct_mask.S) << v.vxm(A, semiring.plus_times)
    result = Vector.from_values([0, 1, 3, 4, 6], [3, 1, 0, 8, 0], size=7)
    assert u.isequal(result)
    u = v.dup()
    u(~~struct_mask.S) << v.vxm(A, semiring.plus_times)
    assert u.isequal(result)
    u = v.dup()
    u(~struct_mask.S) << v.vxm(A, semiring.plus_times)
    result2 = Vector.from_values([2, 3, 4, 5, 6], [3, 1, 2, 14, 4], size=7)
    assert u.isequal(result2)
    u = v.dup()
    u(replace=True, mask=val_mask.V) << v.vxm(A, semiring.plus_times)
    result3 = Vector.from_values([0, 3, 4], [3, 0, 8], size=7)
    assert u.isequal(result3)
    u = v.dup()
    u(replace=True, mask=~~val_mask.V) << v.vxm(A, semiring.plus_times)
    assert u.isequal(result3)
    w = v.vxm(A, semiring.plus_times).new(mask=val_mask.V)
    assert w.isequal(result3)


def test_vxm_accum(v, A):
    w1 = v.dup()
    w1(binary.plus) << v.vxm(A, semiring.plus_times)
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 1, 3, 1, 10, 14, 4], size=7)
    assert w1.isequal(result)
    # Allow monoids
    w2 = v.dup()
    w2(monoid.plus) << v.vxm(A, semiring.plus_times)
    assert w2.isequal(result)
    w3 = v.dup()
    w3(accum=monoid.plus) << v.vxm(A, semiring.plus_times)
    assert w3.isequal(result)


def test_ewise_mult(v):
    # Binary, Monoid, and Semiring
    v2 = Vector.from_values([0, 3, 5, 6], [2, 3, 2, 1])
    result = Vector.from_values([3, 6], [3, 0])
    w = v.ewise_mult(v2, binary.times).new()
    assert w.isequal(result)
    w << v.ewise_mult(v2, monoid.times)
    assert w.isequal(result)
    with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
        v.ewise_mult(v2, semiring.plus_times)


def test_ewise_mult_change_dtype(v):
    # We want to divide by 2, converting ints to floats
    v2 = Vector.from_values([1, 3, 4, 6], [2, 2, 2, 2])
    assert v.dtype == dtypes.INT64
    assert v2.dtype == dtypes.INT64
    result = Vector.from_values([1, 3, 4, 6], [0.5, 0.5, 1.0, 0], dtype=dtypes.FP64)
    w = v.ewise_mult(v2, binary.cdiv[dtypes.FP64]).new()
    assert w.isequal(result), w
    # Here is the potentially surprising way to do things
    # Division is still done with ints, but results are then stored as floats
    result2 = Vector.from_values([1, 3, 4, 6], [0.0, 0.0, 1.0, 0.0], dtype=dtypes.FP64)
    w2 = v.ewise_mult(v2, binary.cdiv).new(dtype=dtypes.FP64)
    assert w2.isequal(result2), w2
    # Try with boolean dtype via auto-conversion
    result3 = Vector.from_values([1, 3, 4, 6], [True, True, False, True])
    w3 = v.ewise_mult(v2, binary.lt).new()
    assert w3.isequal(result3), w3


def test_ewise_add(v):
    # Binary, Monoid, and Semiring
    v2 = Vector.from_values([0, 3, 5, 6], [2, 3, 2, 1])
    result = Vector.from_values([0, 1, 3, 4, 5, 6], [2, 1, 3, 2, 2, 1])
    with pytest.raises(TypeError, match="require_monoid"):
        v.ewise_add(v2, binary.minus)
    w = v.ewise_add(v2, binary.max).new()  # ok if the binaryop is part of a monoid
    assert w.isequal(result)
    w = v.ewise_add(v2, binary.max, require_monoid=False).new()
    assert w.isequal(result)
    w.update(v.ewise_add(v2, monoid.max))
    assert w.isequal(result)
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        v.ewise_add(v2, semiring.max_times)
    # default is plus
    w = v.ewise_add(v2).new()
    result = v.ewise_add(v2, monoid.plus).new()
    assert w.isequal(result)
    # what about default for bool?
    b1 = Vector.from_values([0, 1, 2, 3], [True, False, True, False])
    b2 = Vector.from_values([0, 1, 2, 3], [True, True, False, False])
    with pytest.raises(KeyError, match="plus does not work"):
        b1.ewise_add(b2).new()


def test_extract(v):
    w = Vector.new(v.dtype, 3)
    result = Vector.from_values([0, 1], [1, 1], size=3)
    w << v[[1, 3, 5]]
    assert w.isequal(result)
    w() << v[1::2]
    assert w.isequal(result)
    w2 = v[1::2].new()
    assert w2.isequal(w)


def test_extract_array(v):
    w = Vector.new(v.dtype, 3)
    result = Vector.from_values(np.array([0, 1]), np.array([1, 1]), size=3)
    w << v[np.array([1, 3, 5])]
    assert w.isequal(result)


def test_extract_with_vector(v):
    with pytest.raises(TypeError, match="Invalid type for index"):
        v[v].new()
    with pytest.raises(TypeError, match="Invalid type for index"):
        v[v.S].new()


def test_extract_fancy_scalars(v):
    assert v.dtype == dtypes.INT64
    s = v[1].new()
    assert s == 1
    assert s.dtype == dtypes.INT64

    assert v.dtype == dtypes.INT64
    s = v[1].new(dtype=float)
    assert s == 1.0
    assert s.dtype == dtypes.FP64

    t = Scalar.new(float)
    with pytest.raises(TypeError, match="is not supported"):
        t(accum=binary.plus) << s
    with pytest.raises(TypeError, match="is not supported"):
        t(accum=binary.plus) << 1
    with pytest.raises(TypeError, match="Mask not allowed for Scalars"):
        t(mask=t) << s

    s << v[1]
    assert s.value == 1
    t = Scalar.new(float)
    t << v[1]
    assert t.value == 1.0
    t = Scalar.new(float)
    t() << v[1]
    assert t.value == 1.0
    with pytest.raises(TypeError, match="Scalar accumulation with extract element"):
        t(accum=binary.plus) << v[0]


def test_assign(v):
    u = Vector.from_values([0, 2], [9, 8])
    result = Vector.from_values([0, 1, 3, 4, 6], [9, 1, 1, 8, 0])
    w = v.dup()
    w[[0, 2, 4]] = u
    assert w.isequal(result)
    w = v.dup()
    w[:5:2] << u
    assert w.isequal(result)
    with pytest.raises(TypeError):
        w[:] << u()
    with pytest.raises(TypeError, match="Invalid type for index: Vector."):
        w[w] = 1


def test_assign_scalar(v):
    result = Vector.from_values([1, 3, 4, 5, 6], [9, 9, 2, 9, 0])
    w = v.dup()
    w[[1, 3, 5]] = 9
    assert w.isequal(result)
    w = v.dup()
    w[1::2] = 9
    assert w.isequal(result)
    w = Vector.from_values([0, 1, 2], [1, 1, 1])
    s = Scalar.from_value(9)
    w[0] = s
    assert w.isequal(Vector.from_values([0, 1, 2], [9, 1, 1]))
    w[:] = s
    assert w.isequal(Vector.from_values([0, 1, 2], [9, 9, 9]))
    with pytest.raises(TypeError, match="Bad type for arg"):
        w[:] = object()
    with pytest.raises(TypeError, match="Bad type for arg"):
        w[1] = object()
    w << 2
    assert w.isequal(Vector.from_values([0, 1, 2], [2, 2, 2]))


def test_assign_scalar_mask(v):
    mask = Vector.from_values([1, 2, 5, 6], [0, 0, 1, 0])
    result = Vector.from_values([1, 3, 4, 5, 6], [1, 1, 2, 5, 0])
    w = v.dup()
    w[:](mask.V) << 5
    assert w.isequal(result)
    w = v.dup()
    w(mask.V) << 5
    assert w.isequal(result)
    w = v.dup()
    w(mask.V)[:] << 5
    assert w.isequal(result)
    result2 = Vector.from_values([0, 1, 2, 3, 4, 6], [5, 5, 5, 5, 5, 5])
    w = v.dup()
    w[:](~mask.V) << 5
    assert w.isequal(result2)
    w = v.dup()
    w(~mask.V) << 5
    assert w.isequal(result2)
    w = v.dup()
    w(~mask.V)[:] << 5
    assert w.isequal(result2)
    result3 = Vector.from_values([1, 2, 3, 4, 5, 6], [5, 5, 1, 2, 5, 5])
    w = v.dup()
    w[:](mask.S) << 5
    assert w.isequal(result3)
    w = v.dup()
    w(mask.S) << 5
    assert w.isequal(result3)
    w = v.dup()
    w(mask.S)[:] << 5
    assert w.isequal(result3)
    result4 = Vector.from_values([0, 1, 3, 4, 6], [5, 1, 5, 5, 0])
    w = v.dup()
    w[:](~mask.S) << 5
    assert w.isequal(result4)
    w = v.dup()
    w(~mask.S) << Scalar.from_value(5)
    assert w.isequal(result4)
    w = v.dup()
    w(~mask.S)[:] << 5
    assert w.isequal(result4)


def test_subassign(A):
    v = Vector.from_values([0, 1, 2], [0, 1, 2])
    w = Vector.from_values([0, 1], [10, 20])
    m = Vector.from_values([1], [True])
    v[[0, 1]](m.S) << w
    result1 = Vector.from_values([0, 1, 2], [0, 20, 2])
    assert v.isequal(result1)
    with pytest.raises(DimensionMismatch):
        v[[0, 1]](v.S) << w
    with pytest.raises(DimensionMismatch):
        v[[0, 1]](m.S) << v

    v[[0, 1]](m.S) << 100
    result2 = Vector.from_values([0, 1, 2], [0, 100, 2])
    assert v.isequal(result2)
    with pytest.raises(DimensionMismatch):
        v[[0, 1]](v.S) << 99
    with pytest.raises(TypeError, match="Mask object must be type Vector"):
        v[[0, 1]](A.S) << 88
    with pytest.raises(TypeError, match="Mask object must be type Vector"):
        v[[0, 1]](A.S) << w

    # It may be nice for these to also raise
    v[[0, 1]](A.S)
    v[0](m.S)
    v[0](v.S, replace=True)


def test_assign_scalar_with_mask():
    v = Vector.from_values([0, 1, 2], [1, 2, 3])
    m = Vector.from_values([0, 2], [False, True])
    w1 = Vector.from_values([0], [50])
    w3 = Vector.from_values([0, 1, 2], [10, 20, 30])

    v(m.V)[:] << w3
    result = Vector.from_values([0, 1, 2], [1, 2, 30])
    assert v.isequal(result)

    v(m.V)[:] << 100
    result = Vector.from_values([0, 1, 2], [1, 2, 100])
    assert v.isequal(result)

    v(m.V, accum=binary.plus)[2] << 1000
    result = Vector.from_values([0, 1, 2], [1, 2, 1100])
    assert v.isequal(result)

    with pytest.raises(TypeError, match="Single element assign does not accept a submask"):
        v[2](w1.S) << w1

    with pytest.raises(TypeError, match="Single element assign does not accept a submask"):
        v[2](w1.S) << 7

    v[[2]](w1.S) << 7
    result = Vector.from_values([0, 1, 2], [1, 2, 7])
    assert v.isequal(result)


def test_apply(v):
    result = Vector.from_values([1, 3, 4, 6], [-1, -1, -2, 0])
    w = v.apply(unary.ainv).new()
    assert w.isequal(result)
    with pytest.raises(TypeError, match="apply only accepts UnaryOp with no scalars or BinaryOp"):
        v.apply(semiring.min_plus)


def test_apply_binary(v):
    result_right = Vector.from_values([1, 3, 4, 6], [False, False, True, False])
    w_right = v.apply(binary.gt, right=1).new()
    w_right2 = v.apply(binary.gt, right=Scalar.from_value(1)).new()
    assert w_right.isequal(result_right)
    assert w_right2.isequal(result_right)
    result_left = Vector.from_values([1, 3, 4, 6], [1, 1, 0, 2])
    w_left = v.apply(binary.minus, left=2).new()
    w_left2 = v.apply(binary.minus, left=Scalar.from_value(2)).new()
    assert w_left.isequal(result_left)
    assert w_left2.isequal(result_left)
    with pytest.raises(TypeError):
        v.apply(binary.plus, left=v)
    with pytest.raises(TypeError):
        v.apply(binary.plus, right=v)
    with pytest.raises(TypeError, match="Cannot provide both"):
        v.apply(binary.plus, left=1, right=1)
    # accept monoids
    w1 = v.apply(binary.plus, left=1).new()
    w2 = v.apply(monoid.plus, left=1).new()
    w3 = v.apply(monoid.plus, right=1).new()
    assert w1.isequal(w2)
    assert w1.isequal(w3)


def test_reduce(v):
    s = v.reduce(monoid.plus).new()
    assert s == 4
    assert s.dtype == dtypes.INT64
    assert v.reduce(binary.plus).value == 4
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        v.reduce(binary.minus)

    # Test accum
    s(accum=binary.times) << v.reduce(monoid.plus)
    assert s == 16
    # Test default for non-bool
    assert v.reduce().value == 4
    # Test default for bool
    b1 = Vector.from_values([0, 1], [True, False])
    with pytest.raises(KeyError, match="plus does not work"):
        # KeyError here is kind of weird
        b1.reduce()


def test_reduce_agg(v):
    s = v.reduce(agg.sum).new()
    assert s.dtype == "INT64"
    assert s == 4
    s = v.reduce(agg.sum[float]).new()
    assert s.dtype == "FP64"
    assert s == 4
    assert v.reduce(agg.prod) == 0
    assert v.reduce(agg.count) == 4
    assert v.reduce(agg.count_nonzero) == 3
    assert v.reduce(agg.count_zero) == 1
    assert v.reduce(agg.sum_of_squares) == 6
    assert v.reduce(agg.hypot).new().isclose(6 ** 0.5)
    assert v.reduce(agg.logaddexp).new().isclose(np.log(1 + 2 * np.e + np.e ** 2))
    assert v.reduce(agg.logaddexp2).new().isclose(np.log2(9))
    assert v.reduce(agg.mean) == 1
    assert v.reduce(agg.peak_to_peak) == 2
    assert v.reduce(agg.varp).new().isclose(0.5)
    assert v.reduce(agg.vars).new().isclose(2 / 3)
    assert v.reduce(agg.stdp).new().isclose(0.5 ** 0.5)
    assert v.reduce(agg.stds).new().isclose((2 / 3) ** 0.5)
    assert v.reduce(agg.L0norm) == 3
    assert v.reduce(agg.L1norm) == 4
    assert v.reduce(agg.L2norm).new().isclose(6 ** 0.5)
    assert v.reduce(agg.Linfnorm) == 2
    assert v.reduce(agg.exists) == 1
    w = binary.plus(v, 1).new()
    assert w.reduce(agg.geometric_mean).new().isclose(12 ** 0.25)
    assert w.reduce(agg.harmonic_mean).new().isclose(12 / 7)

    silly = agg.Aggregator(
        "silly",
        composite=[agg.varp, agg.stdp],
        finalize=lambda x, y: binary.times(x & y),
        types=[agg.varp],
    )
    s = v.reduce(silly).new()
    assert s.isclose(0.5 ** 1.5)

    s = Vector.new(int, size=5).reduce(silly).new()
    assert s.is_empty


def test_reduce_agg_argminmax(v):
    assert v.reduce(agg.argmin).value == 6
    assert v.reduce(agg.argmax).value == 4

    silly = agg.Aggregator(
        "silly",
        composite=[agg.argmin, agg.argmax],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.argmin],
    )
    s = v.reduce(silly).new()
    assert s == 10


def test_reduce_agg_firstlast(v):
    empty = Vector.new(int, size=4)
    assert empty.reduce(agg.first).value is None
    assert empty.reduce(agg.last).value is None

    assert v.reduce(agg.first).value == 1
    assert v.reduce(agg.last).value == 0

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first, agg.last],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first],
    )
    s = v.reduce(silly).new()
    assert s == 1


def test_reduce_agg_firstlast_index(v):
    assert v.reduce(agg.first_index).value == 1
    assert v.reduce(agg.last_index).value == 6

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first_index, agg.last_index],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first_index],
    )
    s = v.reduce(silly).new()
    assert s == 7


def test_reduce_agg_empty():
    v = Vector.new("UINT8", size=3)
    for attr, aggr in vars(agg).items():
        if not isinstance(aggr, agg.Aggregator):
            continue
        s = v.reduce(aggr).new()
        assert s.value is None


def test_reduce_coerce_dtype(v):
    assert v.dtype == dtypes.INT64
    s = v.reduce().new(dtype=float)
    assert s == 4.0
    assert s.dtype == dtypes.FP64
    t = Scalar.new(float)
    t << v.reduce(monoid.plus)
    assert t == 4.0
    t = Scalar.new(float)
    t() << v.reduce(monoid.plus)
    assert t == 4.0
    t(accum=binary.times) << v.reduce(monoid.plus)
    assert t == 16.0
    assert v.reduce(monoid.plus[dtypes.UINT64]).value == 4
    # Make sure we accumulate as a float, not int
    t.value = 1.23
    t(accum=binary.plus) << v.reduce()
    assert t == 5.23


def test_simple_assignment(v):
    # w[:] = v
    w = Vector.new(v.dtype, v.size)
    w << v
    assert w.isequal(v)


def test_isequal(v):
    assert v.isequal(v)
    u = Vector.from_values([1], [1])
    assert not u.isequal(v)
    u2 = Vector.from_values([1], [1], size=7)
    assert not u2.isequal(v)
    u3 = Vector.from_values([1, 3, 4, 6], [1.0, 1.0, 2.0, 0.0])
    assert not u3.isequal(v, check_dtype=True), "different datatypes are not equal"
    u4 = Vector.from_values([1, 3, 4, 6], [1.0, 1 + 1e-9, 1.999999999999, 0.0])
    assert not u4.isequal(v)
    u5 = Vector.from_values([1, 3, 4, 5], [1.0, 1.0, 2.0, 3], size=u4.size)
    assert not u4.isequal(u5)


@pytest.mark.slow
def test_isclose(v):
    assert v.isclose(v)
    u = Vector.from_values([1], [1])  # wrong size
    assert not u.isclose(v)
    u2 = Vector.from_values([1], [1], size=7)  # missing values
    assert not u2.isclose(v)
    u3 = Vector.from_values([1, 2, 3, 4, 6], [1, 1, 1, 2, 0], size=7)  # extra values
    assert not u3.isclose(v)
    u4 = Vector.from_values([1, 3, 4, 6], [1.0, 1.0, 2.0, 0.0])
    assert not u4.isclose(v, check_dtype=True), "different datatypes are not equal"
    u5 = Vector.from_values([1, 3, 4, 6], [1.0, 1 + 1e-9, 1.999999999999, 0.0])
    assert u5.isclose(v)
    u6 = Vector.from_values([1, 3, 4, 6], [1.0, 1 + 1e-4, 1.99999, 0.0])
    assert u6.isclose(v, rel_tol=1e-3)
    # isclose should consider `inf == inf`
    u7 = Vector.from_values([1, 3], [-np.inf, np.inf])
    assert u7.isclose(u7, rel_tol=1e-8)
    u4b = Vector.from_values([1, 3, 4, 5], [1.0, 1.0, 2.0, 0.0], size=u4.size)
    assert not u4.isclose(u4b)


def test_binary_op(v):
    v2 = v.dup()
    v2[1] = 0
    w = v.ewise_mult(v2, binary.gt).new()
    result = Vector.from_values([1, 3, 4, 6], [True, False, False, False])
    assert w.dtype == "BOOL"
    assert w.isequal(result)


def test_accum_must_be_binaryop(v):
    # THIS IS NOW OKAY
    w1 = v.dup()
    w1(accum=monoid.plus) << v.ewise_mult(v)
    w2 = v.dup()
    w2(accum=binary.plus) << v.ewise_mult(v)
    assert w1.isequal(w2)
    with pytest.raises(TypeError, match="Expected type: BinaryOp"):
        v(accum=semiring.min_plus)


def test_mask_must_be_value_or_structure(v):
    with pytest.raises(TypeError):
        v(mask=v) << v.ewise_mult(v)
    with pytest.raises(TypeError):
        v(mask=object()) << v.ewise_mult(v)


def test_incompatible_shapes(A, v):
    u = v[:-1].new()
    with pytest.raises(DimensionMismatch):
        A.mxv(u)
    with pytest.raises(DimensionMismatch):
        u.vxm(A)
    with pytest.raises(DimensionMismatch):
        u.ewise_add(v)
    with pytest.raises(DimensionMismatch):
        u.ewise_mult(v)


def test_del(capsys):
    # Exceptions in __del__ are printed to stderr
    import gc

    # shell_v does not have `gb_obj` attribute
    shell_v = Vector.__new__(Vector)
    del shell_v
    # v has `gb_obj` of NULL
    v = Vector.from_values([0, 1], [0, 1])
    gb_obj = v.gb_obj
    v.gb_obj = grblas.ffi.NULL
    del v
    # let's clean up so we don't have a memory leak
    v2 = Vector.__new__(Vector)
    v2.gb_obj = gb_obj
    del v2
    gc.collect()
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
def test_import_export(v, do_iso, methods):
    if do_iso:
        v(v.S) << 1
    v1 = v.dup()
    out_method, in_method = methods
    if out_method == "export":
        d = getattr(v1.ss, out_method)("sparse", give_ownership=True)
    else:
        d = getattr(v1.ss, out_method)("sparse")
    if do_iso:
        assert_array_equal(d["values"], [1])
    else:
        assert_array_equal(d["values"], [1, 1, 2, 0])
    assert d["size"] == 7
    assert_array_equal(d["indices"], [1, 3, 4, 6])
    if in_method == "import":
        w1 = Vector.ss.import_any(**d)
        assert w1.isequal(v)
        assert w1.ss.is_iso is do_iso
    else:
        v1.ss.pack_any(**d)
        assert v1.isequal(v)
        assert v1.ss.is_iso is do_iso

    v2 = v.dup()
    d = getattr(v2.ss, out_method)("bitmap")
    if do_iso:
        assert_array_equal(d["values"], [1])
    else:
        assert_array_equal(d["values"][d["bitmap"]], [1, 1, 2, 0])
    assert d["nvals"] == 4
    assert len(d["bitmap"]) == 7
    assert_array_equal(d["bitmap"], [0, 1, 0, 1, 1, 0, 1])
    if in_method == "import":
        w2 = Vector.ss.import_any(**d)
        assert w2.isequal(v)
        assert w2.ss.is_iso is do_iso
    else:
        v2.ss.pack_any(**d)
        assert v2.isequal(v)
        assert v2.ss.is_iso is do_iso
    del d["nvals"]
    if in_method == "import":
        w2b = Vector.ss.import_any(**d)
        assert w2b.isequal(v)
        assert w2b.ss.is_iso is do_iso
    else:
        v2.ss.pack_any(**d)
        assert v2.isequal(v)
        assert v2.ss.is_iso is do_iso
    d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]])
    if in_method == "import":
        w2c = Vector.ss.import_any(**d)
        if not do_iso:
            assert w2c.isequal(v)
        else:
            assert w2c.size == 2 * v.size
            assert w2c.nvals == 2 * v.nvals
        assert w2c.ss.is_iso is do_iso
    else:
        v2.ss.pack_any(**d)
        assert v2.isequal(v)
        assert v2.ss.is_iso is do_iso

    v3 = Vector.from_values([0, 1, 2], [1, 3, 5])
    if do_iso:
        v3(v3.S) << 1
    v3_copy = v3.dup()
    d = getattr(v3.ss, out_method)("full")
    if do_iso:
        assert_array_equal(d["values"], [1])
    else:
        assert_array_equal(d["values"], [1, 3, 5])
    if in_method == "import":
        w3 = Vector.ss.import_any(**d)
        assert w3.isequal(v3_copy)
        assert w3.ss.is_iso is do_iso
    else:
        v3.ss.pack_any(**d)
        assert v3.isequal(v3_copy)
        assert v3.ss.is_iso is do_iso

    v4 = v.dup()
    d = getattr(v4.ss, out_method)()
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["format"] in {"sparse", "bitmap", "full"}
    if in_method == "import":
        w4 = Vector.ss.import_any(**d)
        assert w4.isequal(v4)
        assert w4.ss.is_iso is do_iso
    else:
        v4.ss.pack_any(**d)
        assert v4.isequal(v)
        assert v4.ss.is_iso is do_iso

    # can't own if we can't write
    d = v.ss.export("sparse")
    d["indices"].flags.writeable = False
    # can't own a view
    size = len(d["values"])
    vals = np.zeros(2 * size, dtype=d["values"].dtype)
    vals[:size] = d["values"]
    view = vals[:size]
    w5 = Vector.ss.import_sparse(take_ownership=True, **dict(d, values=view))
    assert w5.isequal(v)
    assert w5.ss.is_iso is do_iso
    assert d["values"].flags.owndata
    assert d["values"].flags.writeable
    assert d["indices"].flags.owndata

    # now let's take ownership!
    d = v.ss.export("sparse", sort=True)
    w6 = Vector.ss.import_any(take_ownership=True, **d)
    assert w6.isequal(v)
    assert w6.ss.is_iso is do_iso
    assert not d["values"].flags.owndata
    assert not d["values"].flags.writeable
    assert not d["indices"].flags.owndata
    assert not d["indices"].flags.writeable

    with pytest.raises(ValueError, match="Invalid format: bad_name"):
        v.ss.export("bad_name")

    d = v.ss.export("sparse")
    del d["format"]
    with pytest.raises(TypeError, match="Cannot provide both"):
        Vector.ss.import_any(bitmap=d["values"], **d)

    # if we give the same value, make sure it's copied
    for format, key1, key2 in [
        ("sparse", "values", "indices"),
        ("bitmap", "values", "bitmap"),
    ]:
        # No assertions here, but code coverage should be "good enough"
        d = v.ss.export(format, raw=True)
        d[key1] = d[key2]
        Vector.ss.import_any(take_ownership=True, **d)


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
def test_import_export_auto(v, do_iso, methods):
    if do_iso:
        v(v.S) << 1
    v_orig = v.dup()
    out_method, in_method = methods
    for format in ["sparse", "bitmap"]:
        for (
            sort,
            raw,
            import_format,
            give_ownership,
            take_ownership,
            import_name,
        ) in itertools.product(
            [False, True],
            [False, True],
            [format, None],
            [False, True],
            [False, True],
            ["any", format],
        ):
            v2 = v.dup() if give_ownership or out_method == "unpack" else v
            if out_method == "export":
                d = v2.ss.export(format, sort=sort, raw=raw, give_ownership=give_ownership)
            else:
                d = v2.ss.unpack(format, sort=sort, raw=raw)
            if in_method == "import":
                import_func = getattr(Vector.ss, f"import_{import_name}")
            else:

                def import_func(**kwargs):
                    getattr(v2.ss, f"pack_{import_name}")(**kwargs)
                    return v2

            d["format"] = import_format
            other = import_func(take_ownership=take_ownership, **d)
            assert other.isequal(v_orig)
            assert other.ss.is_iso is do_iso
            d["format"] = "bad_format"
            with pytest.raises(ValueError, match="Invalid format"):
                import_func(**d)
    assert v.isequal(v_orig)
    assert v.ss.is_iso is do_iso
    assert v_orig.ss.is_iso is do_iso

    w = Vector.from_values([0, 1, 2], [10, 20, 30])
    if do_iso:
        w(w.S) << 1
    w_orig = w.dup()
    format = "full"
    for (raw, import_format, give_ownership, take_ownership, import_name,) in itertools.product(
        [False, True],
        [format, None],
        [False, True],
        [False, True],
        ["any", format],
    ):
        w2 = w.dup() if give_ownership or out_method == "unpack" else w
        if out_method == "export":
            d = w2.ss.export(format, raw=raw, give_ownership=give_ownership)
        else:
            d = w2.ss.unpack(format, raw=raw)
        if in_method == "import":
            import_func = getattr(Vector.ss, f"import_{import_name}")
        else:

            def import_func(**kwargs):
                getattr(w2.ss, f"pack_{import_name}")(**kwargs)
                return w2

        d["format"] = import_format
        other = import_func(take_ownership=take_ownership, **d)
        assert other.isequal(w_orig)
        assert other.ss.is_iso is do_iso
        d["format"] = "bad_format"
        with pytest.raises(ValueError, match="Invalid format"):
            import_func(**d)
    assert w.isequal(w_orig)
    assert w.ss.is_iso is do_iso
    assert w_orig.ss.is_iso is do_iso


def test_contains(v):
    assert 0 not in v
    assert 1 in v
    with pytest.raises(TypeError):
        [0] in v
    with pytest.raises(TypeError):
        (0,) in v


def test_iter(v):
    assert set(v) == {1, 3, 4, 6}


def test_wait(v):
    v2 = v.dup()
    v2.wait()
    assert v2.isequal(v)


def test_pickle(v):
    s = pickle.dumps(v)
    v2 = pickle.loads(s)
    assert v.isequal(v2, check_dtype=True)
    assert v.name == v2.name


def test_weakref(v):
    d = weakref.WeakValueDictionary()
    d["v"] = v
    assert d["v"] is v
    vS = v.S
    d["v.S"] = vS
    assert d["v.S"] is vS


def test_not_to_array(v):
    with pytest.raises(TypeError, match="Vector can't be directly converted to a numpy array"):
        np.array(v)


def test_vector_index_with_scalar():
    v = Vector.from_values([0, 1, 2], [10, 20, 30])
    expected = Vector.from_values([0, 1], [20, 10])
    for dtype in ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32"]:
        s1 = Scalar.from_value(1, dtype=dtype)
        assert v[s1] == 20
        s0 = Scalar.from_value(0, dtype=dtype)
        w = v[[s1, s0]].new()
        assert w.isequal(expected)
    for dtype in ["fp32", "fp64", "fc32", "fc64"]:
        s = Scalar.from_value(1, dtype=dtype)
        with pytest.raises(TypeError, match="An integer is required for indexing"):
            v[s]


def test_diag(v):
    indices, values = v.to_values()
    for k in range(-5, 5):
        A = grblas.ss.diag(v, k=k)
        size = v.size + abs(k)
        rows = indices + max(0, -k)
        cols = indices + max(0, k)
        expected = Matrix.from_values(rows, cols, values, nrows=size, ncols=size, dtype=v.dtype)
        assert expected.isequal(A)
        w = grblas.ss.diag(A, Scalar.from_value(k))
        assert v.isequal(w)
        assert w.dtype == "INT64"
        w = grblas.ss.diag(A.T, -k, dtype=float)
        assert v.isequal(w)
        assert w.dtype == "FP64"


def test_nbytes(v):
    assert v.ss.nbytes > 0


def test_inner(v):
    R = Matrix.new(v.dtype, nrows=1, ncols=v.size)  # row vector
    C = Matrix.new(v.dtype, nrows=v.size, ncols=1)  # column vector
    R[0, :] = v
    C[:, 0] = v
    expected = R.mxm(C).new()[0, 0].new()
    assert expected.isequal(v.inner(v).new())
    assert expected.isequal(v.inner(v).value)
    assert expected.isequal((v @ v).new())
    assert expected.isequal((v @ v).value)


def test_outer(v):
    R = Matrix.new(v.dtype, nrows=1, ncols=v.size)  # row vector
    C = Matrix.new(v.dtype, nrows=v.size, ncols=1)  # column vector
    R[0, :] = v
    C[:, 0] = v
    expected = C.mxm(R).new()
    result = v.outer(v).new()
    assert result.isequal(expected)
    result = v.outer(v, monoid.times).new()
    assert result.isequal(expected)


def test_auto(v):
    expected = binary.times(v & v).new()
    assert 0 not in expected
    assert 1 in expected
    for expr in [(v & v), binary.times(v & v)]:
        assert expr.size == expected.size
        assert expr.dtype == expected.dtype
        assert expr.shape == expected.shape
        assert expr.nvals == expected.nvals
        assert expr._nvals == expected._nvals
        assert expr.isclose(expected)
        val = expr.new(name="val")
        assert val.name == "val"
        assert expr._value is None
        assert expected.isclose(expr)
        assert expr.isequal(expected)
        assert expected.isequal(expr)
        assert 0 not in expr
        assert 1 in expr
        assert expr[0].value == expected[0].value
        assert expr[1].value == expected[1].value
        assert list(expr) == list(expected)
        k1, v1 = expected.to_values()
        k2, v2 = expr.to_values()
        assert_array_equal(k1, k2)
        assert_array_equal(v1, v2)
        unary.sqrt(expected).isequal(unary.sqrt(expr))
        assert expr.gb_obj is not expected.gb_obj
        assert expr.gb_obj is expr.gb_obj
        assert expr.name != expected.name
        expr.name = "new name"
        assert expr.name == "new name"
        # Probably no need for _name_html or _carg
        assert expr._name_html != expected._name_html
        assert expr._carg != expected._carg
        for method in [
            "ewise_add",
            "ewise_mult",
            "inner",
            "outer",
            "__matmul__",
            "__and__",
            "__or__",
            "__rmatmul__",
            "__rand__",
            "__ror__",
        ]:
            val1 = getattr(expected, method)(expected).new()
            val2 = getattr(expected, method)(expr)
            val3 = getattr(expr, method)(expected)
            val4 = getattr(expr, method)(expr)
            assert val1.isequal(val2)
            assert val1.isequal(val3)
            assert val1.isequal(val4)
            assert val1.isequal(val2.new())
            assert val1.isequal(val3.new())
            assert val1.isequal(val4.new())
        s1 = expected.reduce().new()
        s2 = expr.reduce()
        assert s1.isequal(s2.new())
        assert s1.isequal(s2)
        assert s1.is_empty == s2.is_empty
        assert -s1 == -s2
        assert complex(s1) == complex(s2)
        assert_array_equal(np.array([s1]), np.array([s2]))
    w = v.dup()
    expected = v.dup()
    expected(binary.plus) << (w & w).new()
    w(binary.plus) << (w & w)


def test_auto_assign(v):
    expected = v.dup()
    w = v[1:4].new()
    expr = w & w
    expected[:3] = expr.new()
    v[:3] = expr
    assert expected.isequal(v)
    with pytest.raises(TypeError):
        # Not yet supported, but we could!
        v[:3] = v[1:4]


def test_expr_is_like_vector(v):
    attrs = {attr for attr, val in inspect.getmembers(v)}
    expr_attrs = {attr for attr, val in inspect.getmembers(binary.times(v & v))}
    infix_attrs = {attr for attr, val in inspect.getmembers(v & v)}
    # Should we make any of these raise informative errors?
    expected = {
        "__call__",
        "__del__",
        "__delitem__",
        "__lshift__",
        "__setitem__",
        "_assign_element",
        "_delete_element",
        "_deserialize",
        "_extract_element",
        "_name_counter",
        "_prep_for_assign",
        "_prep_for_extract",
        "_update",
        "build",
        "clear",
        "from_pygraphblas",
        "from_values",
        "resize",
        "update",
    }
    assert attrs - expr_attrs == expected
    assert attrs - infix_attrs == expected | {
        "_expect_op",
        "_expect_type",
    }

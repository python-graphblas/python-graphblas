import inspect
import itertools
import pickle
import sys
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas
from graphblas import agg, binary, dtypes, indexunary, monoid, select, semiring, unary
from graphblas.exceptions import (
    DimensionMismatch,
    EmptyObject,
    IndexOutOfBound,
    InvalidObject,
    InvalidValue,
    OutputNotEmpty,
)

from .conftest import autocompute, compute

from graphblas import Matrix, Scalar, Vector  # isort:skip (for dask-graphblas)


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
    u = Vector(dtypes.INT8, 17)
    assert u.dtype == "INT8"
    assert u.nvals == 0
    assert u.size == 17


def test_large_vector():
    u = Vector.from_values([0, 2**59], [0, 1])
    assert u.size == 2**59 + 1
    assert u[2**59].new() == 1
    with pytest.raises(InvalidValue):
        Vector.from_values([0, 2**64 - 2], [0, 1])
    with pytest.raises(OverflowError):
        Vector.from_values([0, 2**64], [0, 1])


def test_dup(v):
    u = v.dup()
    assert u is not v
    assert u.dtype == v.dtype
    assert u.nvals == v.nvals
    assert u.size == v.size
    # Ensure they are not the same backend object
    v[0] = 1000
    assert u[0].new() != 1000
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
    assert u3[1].new() == 6  # 2*3
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
    u5 = Vector(dtypes.INT64, size=10)
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
    if hasattr(u, "ss"):  # pragma: no branch
        assert u.ss.is_iso
        assert u.ss.iso_value == 7
    assert u.reduce(monoid.any).new() == 7

    # ignore duplicate indices; iso trumps duplicates!
    u = Vector.from_values([0, 1, 1, 3], 7)
    assert u.size == 4
    assert u.nvals == 3
    if hasattr(u, "ss"):  # pragma: no branch
        assert u.ss.is_iso
        assert u.ss.iso_value == 7
    assert u.reduce(monoid.any).new() == 7
    with pytest.raises(ValueError, match="dup_op must be None"):
        Vector.from_values([0, 1, 1, 3], 7, dup_op=binary.plus)
    u[0] = 0
    with pytest.raises(ValueError, match="not iso"):
        u.ss.iso_value


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
    assert compute(v[19].new().value) is None
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
    w = Vector(int, size=3)
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
    assert v[1].new() == 1
    with pytest.raises(TypeError, match="enable automatic"):
        v[1].value
    assert v[6].new() == 0
    with pytest.raises(TypeError, match="Invalid type for index"):
        v[object()]
    with pytest.raises(IndexError):
        v[100]
    s = Scalar(int)
    s << v[1]
    assert s == 1


def test_set_element(v):
    assert compute(v[0].new().value) is None
    assert v[1].new() == 1
    v[0] = 12
    v[1] << 9
    assert v[0].new() == 12
    assert v[1].new() == 9


def test_remove_element(v):
    assert v[1].new() == 1
    del v[1]
    assert compute(v[1].new().value) is None
    assert v[4].new() == 2
    # with pytest.raises(TypeError, match="Remove Element only supports"):
    del v[1:3]  # Now okay


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
    u = Vector(v.dtype, size=2)
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
    # allow strings for accum
    w4 = v.dup()
    w4("+") << v.vxm(A, semiring.plus_times)
    assert w4.isequal(result)
    w5 = v.dup()
    w5(accum="plus") << v.vxm(A, semiring.plus_times)
    assert w5.isequal(result)


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
    # with pytest.raises(TypeError, match="require_monoid"):
    v.ewise_add(v2, binary.minus)  # okay now
    w = v.ewise_add(v2, binary.max).new()  # ok if the binaryop is part of a monoid
    assert w.isequal(result)
    w = v.ewise_add(v2, binary.max).new()
    assert w.isequal(result)
    w.update(v.ewise_add(v2, monoid.max))
    assert w.isequal(result)
    with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
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
    # with pytest.raises(TypeError, match="for BOOL datatype"):
    binary.plus(b1 | b2)  # now okay (btw, `binary.plus[bool].monoid is None`)


def test_extract(v):
    w = Vector(v.dtype, 3)
    result = Vector.from_values([0, 1], [1, 1], size=3)
    w << v[[1, 3, 5]]
    assert w.isequal(result)
    w() << v[1::2]
    assert w.isequal(result)
    w2 = v[1::2].new()
    assert w2.isequal(w)


def test_extract_array(v):
    w = Vector(v.dtype, 3)
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

    t = Scalar(float)
    with pytest.raises(TypeError, match="is not supported"):
        t(accum=binary.plus) << s
    with pytest.raises(TypeError, match="is not supported"):
        t(accum=binary.plus) << 1
    with pytest.raises(TypeError, match="Mask not allowed for Scalars"):
        t(mask=t) << s

    s << v[1]
    assert s.value == 1
    t = Scalar(float)
    t << v[1]
    assert t.value == 1.0
    t = Scalar(float)
    t() << v[1]
    assert t.value == 1.0
    with pytest.raises(TypeError, match="Scalar accumulation with extract element"):
        t(accum=binary.plus) << v[0]


def test_extract_negative_indices(v):
    assert v[-1].new() == 0
    assert compute(v[-v.size].new().value) is None
    assert v[[-v.size]].new().nvals == 0
    assert v[Scalar.from_value(-4)].new() == 1
    w = v[[-1, -3]].new()
    assert w.isequal(Vector.from_values([0, 1], [0, 2]))
    with pytest.raises(IndexError):
        v[-v.size - 1]
    with pytest.raises(IndexError):
        v[Scalar.from_value(-v.size - 1)]
    with pytest.raises(IndexError):
        v[[-v.size - 1]]


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
    w[0] = Scalar(int)
    assert w.isequal(Vector.from_values([1, 2], [2, 2]))


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


def test_assign_list():
    v = Vector(int, 4)
    v[[0, 1]] = [2, 3]
    expected = Vector.from_values([0, 1], [2, 3], size=4)
    assert v.isequal(expected)
    v[::2] = np.arange(2)
    expected = Vector.from_values([0, 1, 2], [0, 3, 1], size=4)
    assert v.isequal(expected)
    with pytest.raises(TypeError):
        v[0] = [1]
    with pytest.raises(TypeError):
        v()[0] = [1]
    with pytest.raises(ValueError, match="shape mismatch"):
        v[[0, 1]] = [1, 2, 3]
    with pytest.raises(ValueError, match="shape mismatch"):
        v[[0, 1]] = [[1, 2]]
    with pytest.raises(ValueError, match="shape mismatch"):
        v[[0, 1]] = [[1], [2]]
    with pytest.raises(TypeError):
        v[[0, 1]] = [2, object()]


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


def test_apply_empty(v):
    s = Scalar(int, is_cscalar=True)
    with pytest.raises(EmptyObject):
        v.apply(binary.plus, s).new()


def test_apply_indexunary(v):
    idx = [1, 3, 4, 6]
    vr = Vector.from_values(idx, idx)
    r1 = v.apply(indexunary.index).new()
    r2 = v.apply("index").new()
    r3 = indexunary.index(v).new()
    assert r1.isequal(vr)
    assert r2.isequal(vr)
    assert r3.isequal(vr)

    v2 = Vector.from_values(idx, [False, False, True, False])
    s2 = Scalar(dtypes.INT64)
    s2.value = 2
    w1 = v.apply("==", s2).new()
    w2 = v.apply(indexunary.valueeq, s2).new()
    w3 = v.apply(select.valueeq, s2).new()
    w4 = indexunary.valueeq(v, s2).new()
    assert w1.isequal(v2)
    assert w2.isequal(v2)
    assert w3.isequal(v2)
    assert w4.isequal(v2)
    with pytest.raises(TypeError, match="left"):
        v.apply(indexunary.valueeq, left=s2)


def test_select(v):
    result = Vector.from_values([1, 3], [1, 1], size=7)
    w1 = v.select(select.valueeq, 1).new()
    w2 = v.select("==", 1).new()
    w3 = select.rowle(v, 3).new()
    w4 = v.select("index<=", 3).new()
    w5 = select.value(v == 1).new()
    w6 = select.index(v < 4).new()
    assert w1.isequal(result)
    assert w2.isequal(result)
    assert w3.isequal(result)
    assert w4.isequal(result)
    assert w5.isequal(result)
    assert w6.isequal(result)
    with pytest.raises(TypeError, match="thunk"):
        v.select(select.valueeq, object())


@autocompute
def test_select_bools_and_masks(v):
    # Select with boolean and masks
    result = Vector.from_values([1, 3], [1, 1], size=7)
    w7 = v.select((v == 1).new()).new()
    assert w7.isequal(result)
    w7b = v.select(v == 1).new()  # we rewrite!
    assert w7b.isequal(result)
    w7c = v.select(~(v != 1)).new()
    assert w7c.isequal(result)
    w8 = v.select(w7.S).new()
    assert w8.isequal(result)
    w7[4] = 0
    w9 = v.select(w7.V).new()
    assert w9.isequal(result)
    with pytest.raises(TypeError, match="thunk"):
        v.select(w7.V, 777)
    with pytest.raises(TypeError, match="thunk"):
        v.select(v == 1, 777)
    with pytest.raises(TypeError):
        v.select(v.outer(v).new().S)
    # Make sure we use `replace`
    w9 << 1
    w9 << v.select(w7.V)
    assert w9.isequal(result)
    # and with masks
    w9 << 1
    w9[1] = 0
    w9(w9.V) << v.select(w7.V)
    result2 = Vector.from_values([1, 3], [0, 1], size=7)
    assert w9.isequal(result2)
    w9 << 1
    w9[1] = 0
    w9(w9.V, replace=True) << v.select(w7.V)
    del result2[1]
    assert w9.isequal(result2)
    w9 << 1
    w9[1] = 0
    w9(w9.V, binary.plus) << v.select(w7.V)
    w8 << 1
    w8[1] = 0
    w8(w8.V, binary.plus) << v.dup(mask=w7.V)
    assert w8.isequal(w9)
    w9 << 1
    w9(binary.plus) << v.select(w7.V)
    w8 << 1
    w8(binary.plus) << v.dup(mask=w7.V)
    assert w8.isequal(w9)


@pytest.mark.slow
def test_indexunary_udf(v):
    def twox_minusthunk(x, row, col, thunk):  # pragma: no cover
        return 2 * x - thunk

    indexunary.register_new("twox_minusthunk", twox_minusthunk)
    assert hasattr(indexunary, "twox_minusthunk")
    assert not hasattr(select, "twox_minusthunk")
    with pytest.raises(ValueError):
        select.register_anonymous(twox_minusthunk)
    with pytest.raises(TypeError, match="must be a function"):
        select.register_anonymous(object())
    expected = Vector.from_values([1, 3, 4, 6], [-2, -2, 0, -4], size=7)
    result = indexunary.twox_minusthunk(v, 4).new()
    assert result.isequal(expected)
    delattr(indexunary, "twox_minusthunk")

    def ii(x, idx, _, thunk):  # pragma: no cover
        return idx // 2 >= thunk

    select.register_new("ii", ii)
    assert hasattr(indexunary, "ii")
    assert hasattr(select, "ii")
    ii_apply = indexunary.register_anonymous(ii)
    expected = Vector.from_values([1, 3, 4, 6], [False, False, True, True], size=7)
    result = ii_apply(v, 2).new()
    assert result.isequal(expected)
    ii_select = select.register_anonymous(ii)
    expected = Vector.from_values([4, 6], [2, 0], size=7)
    result = ii_select(v, 2).new()
    assert result.isequal(expected)
    delattr(indexunary, "ii")
    delattr(select, "ii")


def test_reduce(v):
    s = v.reduce(monoid.plus).new()
    assert s == 4
    assert s.dtype == dtypes.INT64
    assert v.reduce(binary.plus).new() == 4
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        v.reduce(binary.minus)
    with pytest.raises(TypeError, match="to get the Monoid"):
        v.reduce(semiring.plus_times)
    with pytest.raises(TypeError, match="part of a Monoid for BOOL datatype"):
        v.reduce(binary.plus[bool])

    # Test accum
    s(accum=binary.times) << v.reduce(monoid.plus)
    assert s == 16
    # Test default for non-bool
    assert v.reduce().new() == 4
    # Test default for bool
    b1 = Vector.from_values([0, 1], [True, False])
    with pytest.raises(KeyError, match="plus does not work"):
        # KeyError here is kind of weird
        b1.reduce()
    # v is not empty, so it shouldn't matter how we reduce
    for allow_empty, is_cscalar in itertools.product([True, False], [True, False]):
        t = v.reduce(allow_empty=allow_empty).new(is_cscalar=is_cscalar)
        assert t == 4


def test_reduce_empty():
    w = Vector(int, 5)
    s = Scalar.from_value(16)
    s(accum=binary.times) << w.reduce(monoid.plus, allow_empty=True)
    assert s == 16
    s(accum=binary.times) << w.reduce(monoid.plus, allow_empty=False)
    assert s == 0
    assert w.reduce(monoid.plus, allow_empty=True).new().is_empty
    assert w.reduce(monoid.plus, allow_empty=False).new() == 0
    assert w.reduce(agg.sum, allow_empty=True).new().is_empty
    assert w.reduce(agg.sum, allow_empty=False).new() == 0
    assert w.reduce(agg.mean, allow_empty=True).new().is_empty
    with pytest.raises(ValueError):
        w.reduce(agg.mean, allow_empty=False)


def test_reduce_agg(v):
    s = v.reduce(agg.sum).new()
    assert s.dtype == "INT64"
    assert s == 4
    s = v.reduce(agg.sum[float]).new()
    assert s.dtype == "FP64"
    assert s == 4
    assert v.reduce(agg.prod).new() == 0
    assert v.reduce(agg.count).new() == 4
    assert v.reduce(agg.count_nonzero).new() == 3
    assert v.reduce(agg.count_zero).new() == 1
    assert v.reduce(agg.sum_of_squares).new() == 6
    assert v.reduce(agg.hypot).new().isclose(6**0.5)
    assert v.reduce(agg.logaddexp).new().isclose(np.log(1 + 2 * np.e + np.e**2))
    assert v.reduce(agg.logaddexp2).new().isclose(np.log2(9))
    assert v.reduce(agg.mean).new() == 1
    assert v.reduce(agg.peak_to_peak).new() == 2
    assert v.reduce(agg.varp).new().isclose(0.5)
    assert v.reduce(agg.vars).new().isclose(2 / 3)
    assert v.reduce(agg.stdp).new().isclose(0.5**0.5)
    assert v.reduce(agg.stds).new().isclose((2 / 3) ** 0.5)
    assert v.reduce(agg.L0norm).new() == 3
    assert v.reduce(agg.L1norm).new() == 4
    assert v.reduce(agg.L2norm).new().isclose(6**0.5)
    assert v.reduce(agg.Linfnorm).new() == 2
    assert v.reduce(agg.exists).new() == 1
    w = binary.plus(v, 1).new()
    assert w.reduce(agg.geometric_mean).new().isclose(12**0.25)
    assert w.reduce(agg.harmonic_mean).new().isclose(12 / 7)

    silly = agg.Aggregator(
        "silly",
        composite=[agg.varp, agg.stdp],
        finalize=lambda x, y: binary.times(x & y),
        types=[agg.varp],
    )
    s = v.reduce(silly).new()
    assert s.isclose(0.5**1.5)

    s = Vector(int, size=5).reduce(silly).new()
    assert s.is_empty


def test_reduce_agg_argminmax(v):
    assert v.reduce(agg.argmin).new() == 6
    assert v.reduce(agg.argmax).new() == 4

    silly = agg.Aggregator(
        "silly",
        composite=[agg.argmin, agg.argmax],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.argmin],
    )
    s = v.reduce(silly).new()
    assert s == 10


def test_reduce_agg_firstlast(v):
    empty = Vector(int, size=4)
    assert empty.reduce(agg.first).new().is_empty
    assert empty.reduce(agg.last).new().is_empty

    assert v.reduce(agg.first).new() == 1
    assert v.reduce(agg.last).new() == 0

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first, agg.last],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first],
    )
    s = v.reduce(silly).new()
    assert s == 1


def test_reduce_agg_firstlast_index(v):
    assert v.reduce(agg.first_index).new() == 1
    assert v.reduce(agg.last_index).new() == 6

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first_index, agg.last_index],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first_index],
    )
    s = v.reduce(silly).new()
    assert s == 7


def test_reduce_agg_empty():
    v = Vector("UINT8", size=3)
    for _attr, aggr in vars(agg).items():
        if not isinstance(aggr, agg.Aggregator):
            continue
        s = v.reduce(aggr).new()
        assert compute(s.value) is None


def test_reduce_coerce_dtype(v):
    assert v.dtype == dtypes.INT64
    s = v.reduce().new(dtype=float)
    assert s == 4.0
    assert s.dtype == dtypes.FP64
    t = Scalar(float)
    t << v.reduce(monoid.plus)
    assert t == 4.0
    t = Scalar(float)
    t() << v.reduce(monoid.plus)
    assert t == 4.0
    t(accum=binary.times) << v.reduce(monoid.plus)
    assert t == 16.0
    assert v.reduce(monoid.plus[dtypes.UINT64]).new() == 4
    # Make sure we accumulate as a float, not int
    t.value = 1.23
    t(accum=binary.plus) << v.reduce()
    assert t == 5.23


@autocompute
def test_reduce_call_agg(v):
    assert agg.sum(v) == 4
    assert agg.sum(v + v) == 8  # handles expression
    result = agg.max[float](v).new()  # typed agg is callable too
    assert result.dtype == "FP64"
    assert result == 2
    with pytest.raises(ValueError, match="rowwise"):
        agg.sum(v, rowwise=True)
    with pytest.raises(ValueError, match="columnwise"):
        agg.sum(v, columnwise=True)
    with pytest.raises(TypeError):
        agg.sum(7)


def test_simple_assignment(v):
    # w[:] = v
    w = Vector(v.dtype, v.size)
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
    shell_v = object.__new__(Vector)
    del shell_v
    # v has `gb_obj` of NULL
    v = Vector.from_values([0, 1], [0, 1])
    gb_obj = v.gb_obj
    v.gb_obj = graphblas.ffi.NULL
    del v
    # let's clean up so we don't have a memory leak
    v2 = object.__new__(Vector)
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
        assert [0] in v
    with pytest.raises(TypeError):
        assert (0,) in v


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
    for dtype in ["int8", "uint8", "int16", "uint16", "int32", "uint32"]:
        s1 = Scalar.from_value(1, dtype=dtype)
        assert v[s1].new() == 20
        s0 = Scalar.from_value(0, dtype=dtype)
        w = v[[s1, s0]].new()
        assert w.isequal(expected)
    for dtype in ["bool", "fp32", "fp64"] + ["fc32", "fc64"] if dtypes._supports_complex else []:
        s = Scalar.from_value(1, dtype=dtype)
        with pytest.raises(TypeError, match="An integer is required for indexing"):
            v[s]


def test_diag(v):
    indices, values = v.to_values()
    for k in range(-5, 5):
        # Construct diagonal matrix A
        A = graphblas.ss.diag(v, k=k)
        size = v.size + abs(k)
        rows = indices + max(0, -k)
        cols = indices + max(0, k)
        expected = Matrix.from_values(rows, cols, values, nrows=size, ncols=size, dtype=v.dtype)
        assert expected.isequal(A)
        A = v.diag(k)
        assert expected.isequal(A)

        # Extract diagonal from A
        w = graphblas.ss.diag(A, Scalar.from_value(k))
        assert v.isequal(w)
        assert w.dtype == "INT64"

        w = graphblas.ss.diag(A.T, -k, dtype=float)
        assert v.isequal(w)
        assert w.dtype == "FP64"


def test_nbytes(v):
    assert v.ss.nbytes > 0


def test_inner(v):
    R = Matrix(v.dtype, nrows=1, ncols=v.size)  # row vector
    C = Matrix(v.dtype, nrows=v.size, ncols=1)  # column vector
    R[0, :] = v
    C[:, 0] = v
    expected = R.mxm(C).new()[0, 0].new()
    assert expected.isequal(v.inner(v).new())
    assert expected.isequal((v @ v).new())
    s = Scalar(v.dtype)
    s << v.inner(v)
    assert s == 6
    s(binary.plus) << v.inner(v)
    assert s == 12
    with pytest.raises(TypeError, match="autocompute"):
        v << v.inner(v)
    with pytest.raises(TypeError, match="autocompute"):
        v(v.S) << v.inner(v)


@autocompute
def test_inner_infix(v):
    s = Scalar(v.dtype)
    s << v @ v
    assert s == 6
    s(binary.plus) << v @ v
    assert s == 12
    # These autocompute to a scalar on the right!
    v << v @ v
    expected = Vector(v.dtype, v.size)
    expected[:] = 6
    assert v.isequal(expected)
    v(v.S) << v @ v
    expected[:] = 6 * 6 * 7
    assert v.isequal(expected)
    v[:] = 6
    v << v.inner(v)
    assert v.isequal(expected)
    v[:] = 6
    v[:] = v @ v
    assert v.isequal(expected)


def test_outer(v):
    R = Matrix(v.dtype, nrows=1, ncols=v.size)  # row vector
    C = Matrix(v.dtype, nrows=v.size, ncols=1)  # column vector
    R[0, :] = v
    C[:, 0] = v
    expected = C.mxm(R).new()
    result = v.outer(v).new()
    assert result.isequal(expected)
    result = v.outer(v, monoid.times).new()
    assert result.isequal(expected)


@autocompute
def test_auto(v):
    v = v.dup(dtype=bool)
    expected = binary.land(v & v).new()
    assert 0 not in expected
    assert 1 in expected
    for expr in [(v & v), binary.land(v & v)]:
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
            # "ewise_add",
            # "ewise_mult",
            # "inner",
            # "outer",
            # "__matmul__",
            "__and__",
            "__or__",
            # "__rmatmul__",
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
        s1 = expected.reduce(monoid.lor).new()
        s2 = expr.reduce(monoid.lor)
        assert s1.isequal(s2.new())
        assert s1.isequal(s2)
        assert s1.is_empty == s2.is_empty
        assert ~s1 == ~s2
        assert complex(s1) == complex(s2)
        assert_array_equal(np.array([s1]), np.array([s2]))
        assert expected.isequal(expr.new())
    w = v.dup()
    expected = v.dup()
    expected(binary.plus) << (w & w).new()
    w(binary.plus) << (w & w)


@autocompute
def test_auto_assign(v):
    expected = v.dup()
    w = v[1:4].new(dtype=bool)
    expr = w & w
    expected[:3] = expr.new()
    v[:3] = expr
    assert expected.isequal(v)
    v[:3] = v[1:4]
    del expected[0]
    expected[1] = 1
    assert expected.isequal(v)


@autocompute
def test_expr_is_like_vector(v):
    w = v.dup(dtype=bool)
    attrs = {attr for attr, val in inspect.getmembers(w)}
    expr_attrs = {attr for attr, val in inspect.getmembers(binary.times(w & w))}
    infix_attrs = {attr for attr, val in inspect.getmembers(w & w)}
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
        "_from_obj",
        "_name_counter",
        "_parent",
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
    assert attrs - expr_attrs == expected, (
        "If you see this message, you probably added a method to Vector.  You may need to "
        "add an entry to `vector` or `matrix_vector` set in `graphblas._automethods.py` "
        "and then run `python -m graphblas._automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas._infixmethods`."
    )
    assert attrs - infix_attrs == expected


@autocompute
def test_index_expr_is_like_vector(v):
    w = v.dup(dtype=bool)
    attrs = {attr for attr, val in inspect.getmembers(w)}
    expr_attrs = {attr for attr, val in inspect.getmembers(w[[0, 1]])}
    expected = {
        "__del__",
        "__delitem__",
        "__setitem__",
        "_assign_element",
        "_delete_element",
        "_deserialize",
        "_extract_element",
        "_from_obj",
        "_name_counter",
        "_parent",
        "_prep_for_assign",
        "_prep_for_extract",
        "_update",
        "build",
        "clear",
        "from_pygraphblas",
        "from_values",
        "resize",
    }
    assert attrs - expr_attrs == expected, (
        "If you see this message, you probably added a method to Vector.  You may need to "
        "add an entry to `vector` or `matrix_vector` set in `graphblas._automethods.py` "
        "and then run `python -m graphblas._automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas._infixmethods`."
    )


def test_random(v):
    r1 = Vector.from_values([1], [1], size=v.size)
    r2 = Vector.from_values([3], [1], size=v.size)
    r3 = Vector.from_values([4], [2], size=v.size)
    r4 = Vector.from_values([6], [0], size=v.size)
    seen = set()
    for _i in range(1000000):  # pragma: no branch
        r = v.ss.selectk("random", 1)
        if r.isequal(r1):
            seen.add("r1")
        elif r.isequal(r2):
            seen.add("r2")
        elif r.isequal(r3):
            seen.add("r3")
        elif r.isequal(r4):
            seen.add("r4")
        else:  # pragma: no cover
            raise AssertionError()
        if len(seen) == 4:
            break
    for k in range(1, v.nvals + 1):
        r = v.ss.selectk("random", k)
        assert r.nvals == k
        assert monoid.any(v & r).new().nvals == k
    # test iso
    v(v.S) << 1
    for k in range(1, v.nvals + 1):
        r = v.ss.selectk("random", k)
        assert r.nvals == k
        assert monoid.any(v & r).new().nvals == k
    with pytest.raises(ValueError):
        v.ss.selectk("bad", 1)


def test_firstk(v):
    mixed_data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    iso_data = [[1, 3, 4, 6], [1, 1, 1, 1]]
    iso_v = v.dup()
    iso_v(iso_v.S) << 1
    for w, data in [(v, mixed_data), (iso_v, iso_data)]:
        for k in range(w.nvals + 1):
            x = w.ss.selectk("first", k)
            expected = Vector.from_values(data[0][:k], data[1][:k], size=w.size)
            assert x.isequal(expected)
    with pytest.raises(ValueError):
        v.ss.selectk("first", -1)


def test_lastk(v):
    mixed_data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    iso_data = [[1, 3, 4, 6], [1, 1, 1, 1]]
    iso_v = v.dup()
    iso_v(iso_v.S) << 1
    for w, data in [(v, mixed_data), (iso_v, iso_data)]:
        for k in range(w.nvals + 1):
            x = w.ss.selectk("last", k)
            expected = Vector.from_values(data[0][-k:], data[1][-k:], size=w.size)
            assert x.isequal(expected)


def test_largestk(v):
    w = v.ss.selectk("largest", 1)
    expected = Vector.from_values([4], [2], size=v.size)
    assert w.isequal(expected)

    w = v.ss.selectk("largest", 2)
    expected1 = Vector.from_values([1, 4], [1, 2], size=v.size)
    expected2 = Vector.from_values([3, 4], [1, 2], size=v.size)
    assert w.isequal(expected1) or w.isequal(expected2)

    w = v.ss.selectk("largest", 3)
    expected = Vector.from_values([1, 3, 4], [1, 1, 2], size=v.size)
    assert w.isequal(expected)


def test_smallestk(v):
    w = v.ss.selectk("smallest", 1)
    expected = Vector.from_values([6], [0], size=v.size)
    assert w.isequal(expected)

    w = v.ss.selectk("smallest", 2)
    expected1 = Vector.from_values([1, 6], [1, 0], size=v.size)
    expected2 = Vector.from_values([3, 6], [1, 0], size=v.size)
    assert w.isequal(expected1) or w.isequal(expected2)

    w = v.ss.selectk("smallest", 3)
    expected = Vector.from_values([1, 3, 6], [1, 1, 0], size=v.size)
    assert w.isequal(expected)


@pytest.mark.parametrize("do_iso", [False, True])
def test_compactify(do_iso):
    orig_indices = [1, 3, 4, 6]
    new_indices = [0, 1, 2, 3]
    if do_iso:
        v = Vector.from_values(orig_indices, 1)
    else:
        v = Vector.from_values(orig_indices, [1, 4, 2, 0])

    def check(v, expected, *args, stop=0, **kwargs):
        w = v.ss.compactify(*args, **kwargs)
        assert w.isequal(expected)
        for n in reversed(range(stop, 5)):
            expected = expected[:n].new()
            w = v.ss.compactify(*args, size=n, **kwargs)
            assert w.isequal(expected)

    def reverse(v):
        return v[::-1].new().ss.compactify("first", v.size)

    def check_reverse(v, expected, *args, stop=0, **kwargs):
        w = v.ss.compactify(*args, reverse=True, **kwargs)
        x = reverse(expected)
        assert w.isequal(x)
        for n in reversed(range(stop, 5)):
            x = reverse(expected[:n].new())
            w = v.ss.compactify(*args, size=n, reverse=True, **kwargs)
            assert w.isequal(x)

    expected = Vector.from_values(
        new_indices,
        1 if do_iso else [1, 4, 2, 0],
        size=4,
    )
    check(v, expected, "first")
    check_reverse(v, expected, "first")
    check(v, reverse(expected), "last")
    check_reverse(v, reverse(expected), "last")

    expected = Vector.from_values(
        new_indices,
        orig_indices,
        size=4,
    )
    check(v, expected, "first", asindex=True)
    check_reverse(v, expected, "first", asindex=True)
    check(v, reverse(expected), "last", asindex=True)
    check_reverse(v, reverse(expected), "last", asindex=True)

    expected = Vector.from_values(
        new_indices,
        1 if do_iso else [0, 1, 2, 4],
        size=4,
    )
    check(v, expected, "smallest")
    check_reverse(v, expected, "smallest")
    check(v, reverse(expected), "largest")
    check_reverse(v, reverse(expected), "largest")

    if not do_iso:
        expected = Vector.from_values(
            new_indices,
            [6, 1, 4, 3],
            size=4,
        )
        check(v, expected, "smallest", asindex=True, stop=3)
        check_reverse(v, expected, "smallest", asindex=True, stop=3)
        check(v, reverse(expected), "largest", asindex=True, stop=3)
        check_reverse(v, reverse(expected), "largest", asindex=True, stop=3)

    def compare(v, expected, isequal=True, sort=False, **kwargs):
        for _ in range(1000):
            w = v.ss.compactify("random", **kwargs)
            if sort:
                w = w.ss.compactify("smallest")
            if w.isequal(expected) == isequal:
                break
        else:
            raise AssertionError("random failed")

    with pytest.raises(AssertionError):
        compare(v, v[::-1].new())
    for asindex in [False, True]:
        compare(v, v.ss.compactify("first", asindex=asindex), asindex=asindex)
        compare(v, v.ss.compactify("first", 0, asindex=asindex), size=0, asindex=asindex)
        for i in range(1, 4):
            for how in ["first", "last", "smallest", "largest"]:
                w = v.ss.compactify(how, i, asindex=asindex, reverse=how == "last")
                sort = how in {"largest", "smallest"}
                if sort:
                    w = w.ss.compactify("smallest")
                compare(v, w, size=i, asindex=asindex, sort=sort)
                if not do_iso:
                    compare(v, w, size=i, asindex=asindex, isequal=True, sort=sort)
    with pytest.raises(ValueError):
        v.ss.compactify("bad_how")


def test_slice():
    v = Vector.from_values(np.arange(5), np.arange(5))
    w = v[0:0].new()
    assert w.size == 0
    w = v[2:0].new()
    assert w.size == 0
    w = v[::-1].new()
    expected = Vector.from_values(np.arange(5), np.arange(5)[::-1])
    assert w.isequal(expected)
    w = v[4:-3:-1].new()
    expected = Vector.from_values(np.arange(2), np.arange(5)[4:-3:-1])
    assert w.isequal(expected)


def test_concat(v):
    expected = Vector(v.dtype, size=2 * v.size)
    expected[: v.size] = v
    expected[v.size :] = v
    w1 = graphblas.ss.concat([v, v])
    assert w1.isequal(expected, check_dtype=True)
    w2 = Vector(v.dtype, size=2 * v.size)
    w2.ss.concat([v, v])
    assert w2.isequal(expected, check_dtype=True)
    with pytest.raises(TypeError):
        w2.ss.concat([[v, v]])
    w3 = graphblas.ss.concat([v, v], dtype=float)
    assert w3.isequal(expected)
    assert w3.dtype == float


def test_split(v):
    w1, w2 = v.ss.split(4)
    expected1 = Vector.from_values([1, 3], 1)
    expected2 = Vector.from_values([0, 2], [2, 0])
    assert w1.isequal(expected1)
    assert w2.isequal(expected2)
    x1, x2 = v.ss.split([4, 3], name="split")
    assert x1.isequal(expected1)
    assert x2.isequal(expected2)
    assert x1.name == "split_0"
    assert x2.name == "split_1"


def test_ndim(A, v):
    assert v.ndim == 1
    assert v.ewise_mult(v).ndim == 1
    assert (v & v).ndim == 1
    assert (A @ v).ndim == 1


def test_sizeof(v):
    assert sys.getsizeof(v) > v.nvals * 16


def test_ewise_union():
    v1 = Vector.from_values([0], [1], size=3)
    v2 = Vector.from_values([1], [2], size=3)
    result = v1.ewise_union(v2, binary.plus, 10, 20).new()
    expected = Vector.from_values([0, 1], [21, 12], size=3)
    assert result.isequal(expected)
    # Handle Scalars
    result = v1.ewise_union(v2, binary.plus, Scalar.from_value(10), Scalar.from_value(20)).new()
    assert result.isequal(expected)
    # Upcast if scalars are floats
    result = v1.ewise_union(v2, monoid.plus, 10.1, 20.2).new()
    expected = Vector.from_values([0, 1], [21.2, 12.1], size=3)
    assert result.isclose(expected)

    result = v1.ewise_union(v2, binary.minus, 0, 0).new()
    expected = Vector.from_values([0, 1], [1, -2], size=3)
    assert result.isequal(expected)
    result = (v1 - v2).new()
    assert result.isequal(expected)

    bad = Vector(int, size=1)
    with pytest.raises(DimensionMismatch):
        v1.ewise_union(bad, binary.plus, 0, 0)
    with pytest.raises(TypeError, match="Literal scalars"):
        v1.ewise_union(v2, binary.plus, v2, 20)
    with pytest.raises(TypeError, match="Literal scalars"):
        v1.ewise_union(v2, binary.plus, 10, v2)


def test_delete_via_scalar(v):
    del v[[1, 3]]
    assert v.isequal(Vector.from_values([4, 6], [2, 0]))
    del v[:]
    assert v.nvals == 0


def test_udt():
    record_dtype = np.dtype([("x", np.bool_), ("y", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "VectorUDT")
    assert udt._is_anonymous
    v = Vector(udt, size=3)
    a = np.zeros(1, dtype=record_dtype)
    v[0] = a[0]
    expected = Vector.from_values([0], a, size=3, dtype=udt)
    assert v.isequal(expected)
    v[1] = (1, 2)
    expected = Vector.from_values(
        [0, 1], np.array([(0, 0), (1, 2)], dtype=record_dtype), size=3, dtype=udt
    )
    assert v.isequal(expected)
    v[:] = 0
    zeros = Vector.from_values([0, 1, 2], np.zeros(3, dtype=record_dtype), dtype=udt)
    assert v.isequal(zeros)
    v(v.S)[:] = 1
    ones = Vector.from_values([0, 1, 2], np.ones(3, dtype=record_dtype), dtype=udt)
    assert v.isequal(ones)
    v[:](v.S) << 0
    assert v.isequal(zeros)
    s = Scalar(udt)
    s.value = (1, 1)
    v(v.S)[:] << s
    assert v.isequal(ones)
    s.value = 0
    v[:](v.S) << s
    assert v.isequal(zeros)
    t = v.reduce(monoid.any, allow_empty=True).new()
    assert s == t
    t = v.reduce(monoid.any, allow_empty=False).new()
    assert s == t
    v << binary.first(1, v)
    assert v.isequal(ones)
    v << binary.first(s, v)
    assert v.isequal(zeros)
    v << binary.second(v, (1, 1))
    assert v.isequal(ones)
    v << binary.second(v, s)
    assert v.isequal(zeros)
    assert v[0].new() == s
    assert v[:].new().isequal(v)
    indices, values = v.to_values()
    assert v.isequal(Vector.from_values(indices, values))
    result = unary.positioni(v).new()
    expected = Vector.from_values([0, 1, 2], [0, 1, 2])
    assert result.isequal(expected)
    result = binary.firsti(v & v).new()
    assert result.isequal(expected)
    result = semiring.min_firsti(v @ v).new()
    assert result == 0
    assert v.reduce(agg.any_value).new() == s
    assert v.reduce(agg.first).new() == s
    assert v.reduce(agg.last).new() == s
    assert v.reduce(agg.exists).new() == 1
    assert v.reduce(agg.first_index).new() == 0
    assert v.reduce(agg.last_index).new() == 2
    assert v.reduce(agg.count).new() == 3
    info = v.ss.export()
    result = Vector.ss.import_any(**info)
    assert result.isequal(v)
    for aggop in [agg.any_value, agg.first, agg.last, agg.count, agg.first_index, agg.last_index]:
        v.reduce(aggop).new()
    v.clear()
    v[[0, 1]] = [(2, 3), (4, 5)]
    expected = Vector.from_values([0, 1], [(2, 3), (4, 5)], dtype=udt, size=v.size)
    vv = Vector.ss.deserialize(v.ss.serialize(), unsafe=True)
    assert v.isequal(vv, check_dtype=True)
    vv = Vector.ss.deserialize(v.ss.serialize(), dtype=v.dtype)
    assert v.isequal(vv, check_dtype=True)
    with pytest.raises(ValueError):
        Vector.ss.deserialize(v.ss.serialize())

    # arrays as dtypes!
    np_dtype = np.dtype("(3,)uint16")
    udt2 = dtypes.register_anonymous(np_dtype, "has_subdtype")
    s = Scalar(udt2)
    s.value = [0, 0, 0]

    v = Vector(udt2, size=2)
    v[0] = 0
    w = Vector(udt2, size=2)
    w[0] = (0, 0, 0)
    assert v.isequal(w)
    assert v.ewise_mult(w, binary.eq).new().reduce(monoid.land).new()
    assert not v.ewise_mult(w, binary.ne).new().reduce(monoid.land).new()

    w[0] = np.array([0, 0, 1], dtype=np.uint8)
    assert not v.isequal(w)
    assert not v.ewise_mult(w, binary.eq).new().reduce(monoid.land).new()
    assert v.ewise_mult(w, binary.ne).new().reduce(monoid.land).new()
    v[:] = 1
    w[:] = np.array([1, 1, 1], dtype=np.uint8)
    assert v.isequal(w)

    indices, values = v.to_values()
    assert_array_equal(values, np.ones((2, 3), dtype=np.uint16))
    assert v.isequal(Vector.from_values(indices, values, dtype=v.dtype))
    assert v.isequal(Vector.from_values(indices, values))
    info = v.ss.export()
    result = Vector.ss.import_any(**info)
    assert result.isequal(v)

    s = v.reduce(monoid.any).new()
    assert (s.value == [1, 1, 1]).all()
    assert (s.value == 1).all()
    assert s == 1
    assert s != 0
    assert s == (1, 1, 1)
    assert s == [1, 1, 1]
    assert s == s
    t = Scalar.from_value(1)
    assert s == t
    assert t == s
    # Just make sure these work
    for aggop in [agg.any_value, agg.first, agg.last, agg.count, agg.first_index, agg.last_index]:
        v.reduce(aggop).new()
    v.clear()
    v[[0, 1]] = [[2, 3, 4], [5, 6, 7]]
    expected = Vector.from_values([0, 1], [[2, 3, 4], [5, 6, 7]], dtype=udt2, size=v.size)
    assert v.isequal(expected)
    vv = Vector.ss.deserialize(v.ss.serialize(), unsafe=True)
    assert v.isequal(vv, check_dtype=True)

    long_dtype = np.dtype([("x", np.bool_), ("y" * 1000, np.float64)], align=True)
    with pytest.warns(UserWarning, match="too large"):
        long_udt = dtypes.register_anonymous(long_dtype)
    v = Vector(long_udt, size=3)
    v[0] = 0
    v[1] = (1, 5.6)
    vv = Vector.ss.deserialize(v.ss.serialize(), dtype=long_udt)
    assert v.isequal(vv, check_dtype=True)
    with pytest.raises(Exception):
        # The size of the UDT name is limited
        Vector.ss.deserialize(v.ss.serialize(), unsafe=True)
    # May be able to look up non-anonymous dtypes by name if their names are too long
    named_long_dtype = np.dtype([("x", np.bool_), ("y" * 1000, np.float64)], align=False)
    with pytest.warns(UserWarning, match="too large"):
        named_long_udt = dtypes.register_new("LongUDT", named_long_dtype)
    v = Vector(named_long_udt, size=3)
    v[0] = 0
    v[1] = (1, 5.6)
    vv = Vector.ss.deserialize(v.ss.serialize())
    assert v.isequal(vv, check_dtype=True)


def test_infix_outer():
    v = Vector(int, 2)
    v += 1
    assert v.nvals == 0
    v[:] = 1
    v += 1
    assert v.reduce().new() == 4
    v += v
    assert v.reduce().new() == 8
    v += v + v
    assert v.reduce().new() == 24
    with pytest.raises(TypeError, match="autocompute"):
        v += v @ v
    with pytest.raises(TypeError, match="only supported for BOOL"):
        v ^= v
    with pytest.raises(TypeError, match="only supported for BOOL"):
        v |= True
    w = Vector(bool, 2)
    w |= True
    assert w.nvals == 0
    w[:] = False
    w |= True
    assert w.reduce(binary.plus[int]).new() == 2


def test_iteration(v):
    w = Vector(int, 2)
    assert list(w.ss.iterkeys()) == []
    assert list(w.ss.itervalues()) == []
    assert list(w.ss.iteritems()) == []

    indices, values = v.to_values()
    # This is what I would expect
    assert sorted(indices) == sorted(v.ss.iterkeys())
    assert sorted(values) == sorted(v.ss.itervalues())
    assert sorted(zip(indices, values)) == sorted(v.ss.iteritems())

    N = indices.size
    v = Vector.ss.import_bitmap(**v.ss.export("bitmap"))
    assert v.ss.format == "bitmap"
    assert len(list(v.ss.iterkeys(4))) == 2
    assert len(list(v.ss.itervalues(-3))) == 2
    assert len(list(v.ss.iteritems(-v.size))) == N
    assert len(list(v.ss.itervalues(-v.size - 1))) == N
    assert len(list(v.ss.iterkeys(v.size + 1))) == 0

    v = Vector.ss.import_sparse(**v.ss.export("sparse"))
    assert v.ss.format == "sparse"
    assert len(list(v.ss.iterkeys(2))) == 2
    assert len(list(v.ss.itervalues(N))) == 0
    assert len(list(v.ss.iteritems(N + 1))) == 0


def test_broadcasting(A, v):
    # Vector on left
    expected = A.dup()
    for i in range(A.ncols):
        expected(binary.plus)[:, i] = v
    result = (v + A).new()
    assert result.isequal(expected)
    result = v.ewise_add(A, monoid.plus).new()
    assert result.isequal(expected)
    result = v.ewise_union(A, monoid.plus, 0, 0).new()
    assert result.isequal(expected)
    result = binary.plus(v | A).new()
    assert result.isequal(expected)
    # use mask
    result(A.S, replace=True) << binary.plus(v | A)
    expected(A.S, replace=True) << expected
    assert result.isequal(expected)

    expected = semiring.any_plus(v.diag() @ A).new()
    result = v.ewise_mult(A, monoid.plus).new()
    assert result.isequal(expected)
    result = binary.plus(v & A).new()
    assert result.isequal(expected)

    # Vector on right
    expected = A.dup()
    for i in range(A.nrows):
        expected(binary.plus)[i, :] = v
    result = (A + v).new()
    assert result.isequal(expected)
    result = A.ewise_add(v, monoid.plus).new()
    assert result.isequal(expected)
    result = A.ewise_union(v, monoid.plus, 0, 0).new()
    assert result.isequal(expected)
    result = binary.plus(A | v).new()
    assert result.isequal(expected)
    result = A.dup()
    result += v
    assert result.isequal(expected)
    # use mask
    result(A.S, replace=True) << binary.plus(A | v)
    expected(A.S, replace=True) << expected
    assert result.isequal(expected)

    expected = semiring.any_plus(A @ v.diag()).new()
    result = A.ewise_mult(v, monoid.plus).new()
    assert result.isequal(expected)
    result = binary.plus(A & v).new()
    assert result.isequal(expected)
    with pytest.raises(TypeError, match="Bad dtype"):
        (A & v).new()
    (A.dup(bool) & v.dup(bool)).new()  # okay
    with pytest.raises(TypeError):
        v += A
    binary.minus(v | A)


def test_ewise_union_infix():
    v = Vector(int, 3)
    w = Vector(int, 3)
    v[0] = 1
    w[1] = 2
    result = binary.plus(v | w, left_default=10, right_default=20).new()
    expected = Vector.from_values([0, 1], [21, 12], size=3)
    assert expected.isequal(result)
    result = monoid.plus(v | w, left_default=10, right_default=20).new()
    assert expected.isequal(result)
    for op in [binary.plus, monoid.plus]:
        with pytest.raises(TypeError):
            op(v & w, left_default=10, right_default=20)
        with pytest.raises(TypeError):
            op(v @ w, left_default=10, right_default=20)
        with pytest.raises(TypeError):
            op(v | w, right_default=20)
        with pytest.raises(TypeError):
            op(v | w, left_default=20)
        with pytest.raises(TypeError):
            op(v | w, left_default=10, right_default=20, require_monoid=True)
        with pytest.raises(TypeError):
            op(v | w, left_default=10, right_default=20, require_monoid=False)
        with pytest.raises(TypeError):
            op(v | w, right_default=20, require_monoid=True)
        with pytest.raises(TypeError):
            op(v & w, 5, left_default=10, right_default=20)
        with pytest.raises(TypeError):
            op(v, w, left_default=10, right_default=20)


def test_reposition(v):
    indices, values = v.to_values()
    indices = indices.astype(int)

    def get_expected(offset, size):
        ind = indices + offset
        mask = (ind >= 0) & (ind < size)
        return Vector.from_values(ind[mask], values[mask], size=size)

    for offset in range(-v.size - 2, v.size + 3):
        result = v.reposition(offset).new()
        expected = get_expected(offset, v.size)
        assert result.isequal(expected)
        result = v.reposition(offset, size=3).new()
        expected = get_expected(offset, 3)
        assert result.isequal(expected)
        result = v.reposition(offset, size=10).new()
        expected = get_expected(offset, 10)
        assert result.isequal(expected)


def test_to_values_sort():
    # How can we get a vector to a jumbled state in SS so that export won't be sorted?
    N = 1000000
    a = np.unique(np.random.randint(N, size=100))
    expected = a.copy()
    np.random.shuffle(a)
    v = Vector.from_values(a, a, size=N)
    indices, values = v.to_values(sort=False)
    v = Vector.from_values(a, a, size=N)
    indices, values = v.to_values(sort=True)
    assert_array_equal(indices, expected)
    assert_array_equal(values, expected)


def test_lambda_udfs(v):
    result = v.apply(lambda x: x + 1).new()  # pragma: no branch
    expected = binary.plus(v, 1).new()
    assert result.isequal(expected)
    result = v.ewise_mult(v, lambda x, y: x + y).new()  # pragma: no branch
    expected = binary.plus(v & v).new()
    assert result.isequal(expected)
    result = v.ewise_add(v, lambda x, y: x + y).new()  # pragma: no branch
    assert result.isequal(expected)
    # Should we also allow lambdas for monoids with assumed identity of 0?
    # with pytest.raises(TypeError):
    v.ewise_add(v, lambda x, y: x + y)  # pragma: no branch
    with pytest.raises(TypeError):
        v.inner(v, lambda x, y: x + y)


def test_get(v):
    assert compute(v.get(0)) is None
    assert v.get(0, "mittens") == "mittens"
    assert v.get(1) == 1
    assert v.get(1, "mittens") == 1
    assert type(compute(v.get(1))) is int
    with pytest.raises(ValueError):
        # Not yet supported
        v.get([0, 1])


def test_ss_serialize(v):
    for compression, level, nthreads in itertools.product(
        [None, "none", "default", "lz4", "lz4hc"], [None, 1, 5, 9], [None, -1, 1, 10]
    ):
        if level is not None and compression != "lz4hc":
            with pytest.raises(TypeError, match="level argument"):
                v.ss.serialize(compression, level, nthreads=nthreads)
            continue
        a = v.ss.serialize(compression, level, nthreads=nthreads)
        w = Vector.ss.deserialize(a, nthreads=nthreads)
        assert v.isequal(w, check_dtype=True)
    b = a.tobytes()
    w = Vector.ss.deserialize(b)
    assert v.isequal(w, check_dtype=True)
    with pytest.raises(ValueError, match="compression argument"):
        v.ss.serialize("bad")
    with pytest.raises(ValueError, match="level argument"):
        v.ss.serialize("lz4hc", -1)
    with pytest.raises(ValueError, match="level argument"):
        v.ss.serialize("lz4hc", 0)
    with pytest.raises(InvalidObject):
        Vector.ss.deserialize(a[:-5])

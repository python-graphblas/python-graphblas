import pytest
from grblas import Matrix, Vector, Scalar
from grblas import unary, binary, monoid, semiring
from grblas import dtypes
from grblas.exceptions import IndexOutOfBound


@pytest.fixture
def A():
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
    ]
    return Matrix.new_from_values(*data)


@pytest.fixture
def v():
    data = [
        [1, 3, 4, 6],
        [1, 1, 2, 0]
    ]
    return Vector.new_from_values(*data)


def test_new_from_type():
    u = Vector.new_from_type(dtypes.INT8, 17)
    assert u.dtype == 'INT8'
    assert u.nvals == 0
    assert u.size == 17


def test_new_from_existing(v):
    u = Vector.new_from_existing(v)
    assert u is not v
    assert u.dtype == v.dtype
    assert u.nvals == v.nvals
    assert u.size == v.size
    # Ensure they are not the same backend object
    v[0] = 1000
    assert u[0].value != 1000


def test_new_from_values():
    u = Vector.new_from_values([0, 1, 3], [True, False, True])
    assert u.size == 4
    assert u.nvals == 3
    assert u.dtype == bool
    u2 = Vector.new_from_values([0, 1, 3], [12.3, 12.4, 12.5], size=17)
    assert u2.size == 17
    assert u2.nvals == 3
    assert u2.dtype == float
    u3 = Vector.new_from_values([0, 1, 1], [1, 2, 3], size=10, dup_op=binary.times)
    assert u3.size == 10
    assert u3.nvals == 2  # duplicates were combined
    assert u3.dtype == int
    assert u3[1].value == 6  # 2*3
    with pytest.raises(ValueError):
        # Duplicate indices requires a dup_op
        Vector.new_from_values([0, 1, 1], [True, True, True])


def test_clear(v):
    v.clear()
    assert v.nvals == 0
    assert v.size == 7


def test_resize(v):
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_size(v):
    assert v.size == 7


def test_nvals(v):
    assert v.nvals == 4


def test_rebuild(v):
    assert v.nvals == 4
    v.rebuild_from_values([0, 6], [1, 2])
    assert v.nvals == 2
    with pytest.raises(IndexOutOfBound):
        v.rebuild_from_values([0, 11], [1, 1])


def test_extract_values(v):
    idx, vals = v.to_values()
    assert idx == (1, 3, 4, 6)
    assert vals == (1, 1, 2, 0)


def test_extract_element(v):
    assert v[1].value == 1
    assert v[6].new() == 0


def test_set_element(v):
    assert v[0].value is None
    assert v[1].value == 1
    v[0] = 12
    v[1] << 9
    assert v[0].value == 12
    assert v[1].new() == 9


def test_remove_element(v):
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_vxm(v, A):
    w = v.vxm(A, semiring.plus_times).new()
    result = Vector.new_from_values([0, 2, 3, 4, 5, 6], [3, 3, 0, 8, 14, 4])
    assert w == result


def test_vxm_transpose(v, A):
    w = v.vxm(A.T, semiring.plus_times).new()
    result = Vector.new_from_values([0, 1, 6], [5, 16, 13])
    assert w == result


def test_vxm_nonsquare(v):
    A = Matrix.new_from_values([0, 3], [0, 1], [10, 20], nrows=7, ncols=2)
    u = Vector.new_from_type(v.dtype, size=2)
    u().update(v.vxm(A, semiring.min_plus))
    result = Vector.new_from_values([1], [21])
    assert u == result
    w1 = v.vxm(A, semiring.min_plus).new()
    assert w1 == u
    # Test the transpose case
    v2 = Vector.new_from_values([0, 1], [1, 2])
    w2 = v2.vxm(A.T, semiring.min_plus).new()
    assert w2.size == 7


def test_vxm_mask(v, A):
    mask = Vector.new_from_values([0, 3, 4], [True, True, True], size=7)
    u = Vector.new_from_existing(v)
    u(mask) << v.vxm(A, semiring.plus_times)
    result = Vector.new_from_values([0, 1, 3, 4, 6], [3, 1, 0, 8, 0], size=7)
    assert u == result
    u = Vector.new_from_existing(v)
    u(~mask) << v.vxm(A, semiring.plus_times)
    result2 = Vector.new_from_values([2, 3, 4, 5, 6], [3, 1, 2, 14, 4], size=7)
    assert u == result2
    u = Vector.new_from_existing(v)
    u(replace=True, mask=mask) << v.vxm(A, semiring.plus_times)
    result3 = Vector.new_from_values([0, 3, 4], [3, 0, 8], size=7)
    assert u == result3
    w = v.vxm(A, semiring.plus_times).new(mask=mask)
    assert w == result3


def test_vxm_accum(v, A):
    v(binary.plus) << v.vxm(A, semiring.plus_times)
    result = Vector.new_from_values([0, 1, 2, 3, 4, 5, 6], [3, 1, 3, 1, 10, 14, 4], size=7)
    assert v == result


def test_ewise_mult(v):
    # Binary, Monoid, and Semiring
    v2 = Vector.new_from_values([0, 3, 5, 6], [2, 3, 2, 1])
    result = Vector.new_from_values([3, 6], [3, 0])
    w = v.ewise_mult(v2, binary.times).new()
    assert w == result
    w << v.ewise_mult(v2, monoid.times)
    assert w == result
    w.update(v.ewise_mult(v2, semiring.plus_times))
    assert w == result


def test_ewise_mult_change_dtype(v):
    # We want to divide by 2, converting ints to floats
    v2 = Vector.new_from_values([1, 3, 4, 6], [2, 2, 2, 2])
    assert v.dtype == dtypes.INT64
    assert v2.dtype == dtypes.INT64
    result = Vector.new_from_values([1, 3, 4, 6], [0.5, 0.5, 1.0, 0], dtype=dtypes.FP64)
    w = v.ewise_mult(v2, binary.div[dtypes.FP64]).new()
    assert w == result
    # Here is the potentially surprising way to do things
    # Division is still done with ints, but results are then stored as floats
    result2 = Vector.new_from_values([1, 3, 4, 6], [0.0, 0.0, 1.0, 0.0], dtype=dtypes.FP64)
    w2 = v.ewise_mult(v2, binary.div).new(dtype=dtypes.FP64)
    assert w2 == result2


def test_ewise_add(v):
    # Binary, Monoid, and Semiring
    v2 = Vector.new_from_values([0, 3, 5, 6], [2, 3, 2, 1])
    result = Vector.new_from_values([0, 1, 3, 4, 5, 6], [2, 1, 3, 2, 2, 1])
    w = v.ewise_add(v2, binary.max).new()
    assert w == result
    w.update(v.ewise_add(v2, monoid.max))
    assert w == result
    w << v.ewise_add(v2, semiring.max_times)
    assert w == result


def test_extract(v):
    w = Vector.new_from_type(v.dtype, 3)
    result = Vector.new_from_values([0, 1], [1, 1], size=3)
    w << v[[1, 3, 5]]
    assert w == result
    w() << v[1::2]
    assert w == result
    w2 = v[1::2].new()
    assert w2 == w


def test_assign(v):
    u = Vector.new_from_values([0, 2], [9, 8])
    result = Vector.new_from_values([0, 1, 3, 4, 6], [9, 1, 1, 8, 0])
    w = Vector.new_from_existing(v)
    w[[0, 2, 4]] = u
    assert w == result
    w = Vector.new_from_existing(v)
    w[:5:2] << u
    assert w == result


def test_assign_scalar(v):
    result = Vector.new_from_values([1, 3, 4, 5, 6], [9, 9, 2, 9, 0])
    w = Vector.new_from_existing(v)
    w[[1, 3, 5]] = 9
    assert w == result
    w = Vector.new_from_existing(v)
    w[1::2] = 9
    assert w == result
    w = Vector.new_from_values([0, 1, 2], [1, 1, 1])
    s = Scalar.new_from_value(9)
    w[:] = s
    assert w == Vector.new_from_values([0, 1, 2], [9, 9, 9])


def test_assign_scalar_mask(v):
    mask = Vector.new_from_values([1, 2, 5, 6], [0, 0, 1, 0])
    result = Vector.new_from_values([1, 3, 4, 5, 6], [1, 1, 2, 5, 0])
    w = v.dup()
    w[:](mask) << 5
    assert w == result
    result2 = Vector.new_from_values([0, 1, 2, 3, 4, 6], [5, 5, 5, 5, 5, 5])
    w = v.dup()
    w[:](~mask) << 5
    assert w == result2
    result3 = Vector.new_from_values([1, 2, 3, 4, 5, 6], [5, 5, 1, 2, 5, 5])
    w = v.dup()
    w[:](mask.S) << 5
    assert w == result3
    result4 = Vector.new_from_values([0, 1, 3, 4, 6], [5, 1, 5, 5, 0])
    w = v.dup()
    w[:](~mask.S) << 5
    assert w == result4


def test_apply(v):
    result = Vector.new_from_values([1, 3, 4, 6], [-1, -1, -2, 0])
    w = v.apply(unary.ainv).new()
    assert w == result


def test_apply_binary(v):
    # Test bind-first and bind-second
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_reduce(v):
    s = v.reduce(monoid.plus).new()
    assert s == 4
    # Test accum
    s(accum=binary.times) << v.reduce(monoid.plus)
    assert s == 16


def test_simple_assignment(v):
    # w[:] = v
    w = Vector.new_from_type(v.dtype, v.size)
    w << v
    assert w == v


def test_equal(v):
    assert v == v
    u = Vector.new_from_values([1], [1])
    assert u != v
    u2 = Vector.new_from_values([1], [1], size=7)
    assert u2 != v
    u3 = Vector.new_from_values([1, 3, 4, 6], [1., 1., 2., 0.])
    assert not u3.isequal(v, strict_dtype=True), 'different datatypes are not equal'
    u4 = Vector.new_from_values([1, 3, 4, 6], [1., 1+1e-9, 1.999999999999, 0.])
    assert u4 == v
    u5 = Vector.new_from_values([1, 3, 4, 6], [1., 1+1e-4, 1.99999, 0.])
    assert u5.isequal(v, rel_tol=1e-3)


def test_binary_op(v):
    v2 = Vector.new_from_existing(v)
    v2[1] = 0
    w = v.ewise_mult(v2, binary.gt).new()
    result = Vector.new_from_values([1, 3, 4, 6], [True, False, False, False])
    assert w.dtype == 'BOOL'
    assert w == result

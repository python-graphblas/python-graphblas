import pytest
from _grblas import lib, ffi
from grblas import Matrix, Vector, Scalar
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, descriptor
from grblas.exceptions import IndexOutOfBound

@pytest.fixture
def A():
    data = [
        [3,0,3,5,6,0,6,1,6,2,4,1],
        [0,1,2,2,2,3,3,4,4,5,5,6],
        [3,2,3,1,5,3,7,8,3,1,7,4]
    ]
    return Matrix.new_from_values(*data)

@pytest.fixture
def v():
    data = [
        [1,3,4,6],
        [1,1,2,0]
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
    v.element[0] = 1000
    assert u.element[0] != 1000

def test_new_from_values():
    u = Vector.new_from_values([0, 1, 3], [True, False, True])
    assert u.size == 4
    assert u.nvals == 3
    assert u.dtype == bool
    u2 = Vector.new_from_values([0, 1, 3], [12.3, 12.4, 12.5], size=17)
    assert u2.size == 17
    assert u2.nvals == 3
    assert u2.dtype == float
    u3 = Vector.new_from_values([0, 1, 1], [1, 2, 3], size=10, dup_op=BinaryOp.TIMES)
    assert u3.size == 10
    assert u3.nvals == 2  # duplicates were combined
    assert u3.dtype == int
    assert u3.element[1] == 6  # 2*3
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
    assert tuple(idx) == (1, 3, 4, 6)
    assert tuple(vals) == (1, 1, 2, 0)

def test_extract_element(v):
    assert v.element[1] == 1
    assert v.element[6] == 0

def test_set_element(v):
    assert v.element[0] is None
    assert v.element[1] == 1
    v.element[0] = 12
    v.element[1] = 9
    assert v.element[0] == 12
    assert v.element[1] == 9

def test_remove_element(v):
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_vxm(v, A):
    pytest.skip()

def test_vxm_transpose(v, A):
    pytest.skip()

def test_vxm_nonsquare(v, A):
    pytest.skip()

def test_vxm_mask(v, A):
    pytest.skip()

def test_vxm_accum(v, A):
    pytest.skip()

def test_ewise_mult(v):
    # Binary, Monoid, and Semiring
    pytest.skip()

def test_ewise_add(v):
    # Binary, Monoid, and Semiring
    pytest.skip()

def test_extract(v):
    pytest.skip()

def test_assign(v):
    pytest.skip()

def test_assign_scalar(v):
    # Test block, row, column
    pytest.skip()

def test_apply(v):
    pytest.skip()

def test_apply_binary(v):
    # Test bind-first and bind-second
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_reduce(v):
    pytest.skip()

def test_simple_assignment(v):
    # w[:] = v
    pytest.skip()

def test_equal(v):
    pytest.skip()

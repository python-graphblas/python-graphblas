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
    C = Matrix.new_from_type(dtypes.INT8, 17, 12)
    assert C.dtype == 'INT8'
    assert C.nvals == 0
    assert C.nrows == 17
    assert C.ncols == 12

def test_new_from_existing(A):
    C = Matrix.new_from_existing(A)
    assert C is not A
    assert C.dtype == A.dtype
    assert C.nvals == A.nvals
    assert C.nrows == A.nrows
    assert C.ncols == A.ncols
    # Ensure they are not the same backend object
    A.element[0, 0] = 1000
    assert C.element[0, 0] != 1000

def test_new_from_values():
    C = Matrix.new_from_values([0, 1, 3], [1, 1, 2], [True, False, True])
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == bool
    C2 = Matrix.new_from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=3)
    assert C2.nrows == 17
    assert C2.ncols == 3
    assert C2.nvals == 3
    assert C2.dtype == float
    C3 = Matrix.new_from_values([0, 1, 1], [2, 1, 1], [1, 2, 3], nrows=10, dup_op=BinaryOp.TIMES)
    assert C3.nrows == 10
    assert C3.ncols == 3
    assert C3.nvals == 2  # duplicates were combined
    assert C3.dtype == int
    assert C3.element[1, 1] == 6  # 2*3
    with pytest.raises(ValueError):
        # Duplicate indices requires a dup_op
        Matrix.new_from_values([0, 1, 1], [2, 1, 1], [True, True, True])
    with pytest.raises(IndexOutOfBound):
        # Specified ncols can't hold provided indexes
        Matrix.new_from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=2)

def test_clear(A):
    A.clear()
    assert A.nvals == 0
    assert A.nrows == 7
    assert A.ncols == 7

def test_resize(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_nrows(A):
    assert A.nrows == 7

def test_ncols(A):
    assert A.ncols == 7

def test_nvals(A):
    assert A.nvals == 12

def test_rebuild(A):
    assert A.nvals == 12
    A.rebuild_from_values([0, 6], [0, 1], [1, 2])
    assert A.nvals == 2
    with pytest.raises(IndexOutOfBound):
        A.rebuild_from_values([0, 11], [0, 0], [1, 1])

def test_extract_values(A):
    rows, cols, vals = A.to_values()
    assert tuple(rows) == (0,0,1,1,2,3,3,4,5,6,6,6)
    assert tuple(cols) == (1,3,4,6,5,0,2,5,2,2,3,4)
    assert tuple(vals) == (2,3,8,4,1,3,3,7,1,5,7,3)

def test_extract_element(A):
    assert A.element[3, 0] == 3
    assert A.element[1, 6] == 4

def test_set_element(A):
    assert A.element[1, 1] is None
    assert A.element[3, 0] == 3
    A.element[1, 1] = 21
    A.element[3, 0] = -5
    assert A.element[1, 1] == 21
    assert A.element[3, 0] == -5

def test_remove_element(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_mxm(A):
    pytest.skip()

def test_mxm_transpose(A):
    pytest.skip()

def test_mxm_nonsquare(A):
    pytest.skip()

def test_mxm_mask(A):
    pytest.skip()

def test_mxm_accum(A):
    pytest.skip()

def test_mxv(A):
    pytest.skip()

def test_ewise_mult(A):
    # Binary, Monoid, and Semiring
    pytest.skip()

def test_ewise_add(A):
    # Binary, Monoid, and Semiring
    pytest.skip()

def test_extract(A):
    pytest.skip()

def test_extract_row(A):
    pytest.skip()

def test_extract_column(A):
    pytest.skip()

def test_assign(A):
    pytest.skip()

def test_assign_row(A, v):
    pytest.skip()

def test_assign_column(A, v):
    pytest.skip()

def test_assign_scalar(A):
    # Test block, row, column
    pytest.skip()

def test_apply(A):
    pytest.skip()

def test_apply_binary(A):
    # Test bind-first and bind-second
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_reduce_row(A):
    pytest.skip()

def test_reduce_column(A):
    pytest.skip()

def test_reduce_scalar(A):
    pytest.skip()

def test_reduce_scalar_accum(A):
    pytest.skip()

def test_transpose(A):
    # C[:] = A.T
    pytest.skip()

def test_kronecker(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_simple_assignment(A):
    # C[:] = A
    pytest.skip()

def test_equal(A):
    pytest.skip()

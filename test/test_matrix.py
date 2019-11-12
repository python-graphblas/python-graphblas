import pytest
from _grblas import lib, ffi
from grblas import Matrix, Vector, Scalar
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, descriptor
from grblas import REPLACE
from grblas.exceptions import IndexOutOfBound, DimensionMismatch

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
    A[:] = A.mxm(A, Semiring.PLUS_TIMES)
    result = Matrix.new_from_values(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6],
        [0, 2, 4, 6, 2, 3, 4, 5, 2, 1, 3, 5, 2, 5, 0, 2, 5],
        [9, 9, 16, 8, 20, 28, 12, 56, 1, 6, 9, 3, 7, 1, 21, 21, 26])
    assert A == result

def test_mxm_transpose(A):
    C = Matrix.new_from_existing(A)
    C[:] = A.mxm(A.T, Semiring.PLUS_TIMES)
    result = Matrix.new_from_values(
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
        [0, 6, 1, 6, 2, 4, 3, 5, 6, 2, 4, 3, 5, 6, 0, 1, 3, 5, 6],
        [13, 21, 80, 24, 1, 7, 18, 3, 15, 7, 49, 3, 1, 5, 21, 24, 15, 5, 83])
    assert C == result
    C[:] = A.T.mxm(A, Semiring.PLUS_TIMES)
    result2 = Matrix.new_from_values(
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6],
        [0, 2, 1, 3, 0, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 6, 5, 4, 6],
        [9, 9, 4, 6, 9, 35, 35, 15, 6, 35, 58, 21, 15, 21, 73, 32, 50, 32, 16])
    assert C == result2

def test_mxm_nonsquare():
    A = Matrix.new_from_values([0, 0, 0], [0, 2, 4], [1, 2, 3], nrows=1, ncols=5)
    B = Matrix.new_from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1)
    C = Matrix.new_from_type(A.dtype, nrows=1, ncols=1)
    C[:] = A.mxm(B, Semiring.MAX_PLUS)
    assert C.element[0, 0] == 33

def test_mxm_mask(A):
    mask = Matrix.new_from_values([0, 3, 4], [2, 3, 2], [True, True, True], nrows=7, ncols=7)
    C = Matrix.new_from_existing(A)
    C[mask] = A.mxm(A, Semiring.PLUS_TIMES)
    result = Matrix.new_from_values(
        [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 6, 6],
        [1, 2, 3, 4, 6, 5, 0, 2, 3, 2, 5, 2, 2, 3, 4],
        [2, 9, 3, 8, 4, 1, 3, 3, 9, 7, 7, 1, 5, 7, 3])
    assert C == result
    C = Matrix.new_from_existing(A)
    C[~mask] = A.mxm(A, Semiring.PLUS_TIMES)
    result2 = Matrix.new_from_values(
        [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 5, 6, 6, 6], 
        [0, 4, 6, 2, 3, 4, 5, 2, 1, 5, 5, 0, 2, 5],
        [9, 16, 8, 20, 28, 12, 56, 1, 6, 3, 1, 21, 21, 26])
    assert C == result2
    C = Matrix.new_from_existing(A)
    C[mask, REPLACE] = A.mxm(A, Semiring.PLUS_TIMES)
    result3 = Matrix.new_from_values(
        [0,3,4],
        [2,3,2],
        [9,9,7], nrows=7, ncols=7)
    assert C == result3

def test_mxm_accum(A):
    A[BinaryOp.PLUS] = A.mxm(A, Semiring.PLUS_TIMES)
    result = Matrix.new_from_values(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6],
        [0, 1, 2, 3, 4, 6, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 5, 2, 5, 2, 5, 0, 2, 3, 4, 5],
        [9, 2, 9, 3, 16, 8, 20, 28, 20, 56, 4, 1, 1, 3, 6, 3, 9, 3, 7, 7, 1, 1, 21, 26, 7, 3, 26])
    assert A == result

def test_mxv(A, v):
    v[:] = A.mxv(v, Semiring.PLUS_TIMES)
    result = Vector.new_from_values([0,1,6], [5,16,13])
    assert v == result

def test_ewise_mult(A):
    # Binary, Monoid, and Semiring
    B = Matrix.new_from_values([0,0,5], [1,2,2], [5,4,8], nrows=7, ncols=7)
    result = Matrix.new_from_values([0,5], [1,2], [10,8], nrows=7, ncols=7)
    C = Matrix.new_from_type(A.dtype, A.nrows, A.ncols)
    C[:] = A.ewise_mult(B, BinaryOp.TIMES)
    assert C == result
    C[:] = A.ewise_mult(B, Monoid.TIMES)
    assert C == result
    C[:] = A.ewise_mult(B, Semiring.PLUS_TIMES)
    assert C == result

def test_ewise_add(A):
    # Binary, Monoid, and Semiring
    B = Matrix.new_from_values([0,0,5], [1,2,2], [5,4,8], nrows=7, ncols=7)
    result = Matrix.new_from_values(
        [0,3,0,3,5,6,0,6,1,6,2,4,1],
        [2,0,1,2,2,2,3,3,4,4,5,5,6],
        [4,3,5,3,8,5,3,7,8,3,1,7,4]
    )
    C = Matrix.new_from_type(A.dtype, A.nrows, A.ncols)
    C[:] = A.ewise_add(B, BinaryOp.SECOND)  # possibly surprising, but SECOND(x, empty) == x
    assert C == result
    C[:] = A.ewise_add(B, Monoid.MAX)
    assert C == result
    C[:] = A.ewise_add(B, Semiring.MAX_MINUS)
    assert C == result

def test_extract(A):
    C = Matrix.new_from_type(A.dtype, 3, 4)
    result = Matrix.new_from_values([0,0,1,2,2,2], [0,2,1,1,2,3], [2,3,3,5,7,3], nrows=3, ncols=4)
    C[:] = A.extract[[0,3,6], [1,2,3,4]]
    assert C == result
    C[:] = A.extract[0::3, 1:5]
    assert C == result
    C[:] = A.extract[[0,3,6], 1:5:1]
    assert C == result

def test_extract_row(A):
    w = Vector.new_from_type(A.dtype, 3)
    result = Vector.new_from_values([1,2], [5,3], size=3)
    w[:] = A.extract[6, [0,2,4]]
    assert w == result
    w[:] = A.extract[6, :5:2]
    assert w == result
    w[:] = A.T.extract[[0,2,4], 6]
    assert w == result

def test_extract_column(A):
    w = Vector.new_from_type(A.dtype, 3)
    result = Vector.new_from_values([1,2], [3,1], size=3)
    w[:] = A.extract[[1,3,5], 2]
    assert w == result
    w[:] = A.extract[1:6:2, 2]
    assert w == result
    w[:] = A.T.extract[2, [1,3,5]]
    assert w == result

def test_assign(A):
    B = Matrix.new_from_values([0,0,1], [0,1,0], [9,8,7])
    result = Matrix.new_from_values(
        [0,0,2,3,0,3,5,6,0,6,1,6,4,1],
        [0,5,0,0,1,2,2,2,3,3,4,4,5,6],
        [9,8,7,3,2,3,1,5,3,7,8,3,7,4])
    C = Matrix.new_from_existing(A)
    C.assign[[0,2], [0,5]] = B
    assert C == result
    C = Matrix.new_from_existing(A)
    C.assign[:3:2, :6:5] = B
    assert C == result

def test_assign_wrong_dims(A):
    B = Matrix.new_from_values([0,0,1], [0,1,0], [9,8,7])
    with pytest.raises(DimensionMismatch):
        A.assign[[0,2,4], [0,5]] = B

def test_assign_row(A, v):
    result = Matrix.new_from_values(
        [3,3,5,6,6,1,6,2,4,1,0,0,0,0],
        [0,2,2,2,3,4,4,5,5,6,1,3,4,6],
        [3,3,1,5,7,8,3,1,7,4,1,1,2,0])
    C = Matrix.new_from_existing(A)
    C.assign[0, :] = v
    assert C == result

def test_assign_column(A, v):
    result = Matrix.new_from_values(
        [3,3,5,6,0,6,1,6,2,4,1,1,3,4,6],
        [0,2,2,2,3,3,4,4,5,5,6,1,1,1,1],
        [3,3,1,5,3,7,8,3,1,7,4,1,1,2,0])
    C = Matrix.new_from_existing(A)
    C.assign[:, 1] = v
    assert C == result

def test_assign_scalar(A):
    # Test block
    result_block = Matrix.new_from_values(
        [3,0,6,0,6,6,2,4,1,1,3,5,1,3,5],
        [0,1,2,3,3,4,5,5,6,2,2,2,4,4,4],
        [3,2,5,3,7,3,1,7,4,0,0,0,0,0,0])
    C = Matrix.new_from_existing(A)
    C.assign[[1,3,5], [2,4]] = 0
    assert C == result_block
    C = Matrix.new_from_existing(A)
    C.assign[1::2, 2:5:2] = 0
    assert C == result_block
    # Test row
    result_row = Matrix.new_from_values(
        [3,0,6,0,6,6,2,4,1,3,5,1,1],
        [0,1,2,3,3,4,5,5,6,2,2,2,4],
        [3,2,5,3,7,3,1,7,4,3,1,0,0])
    C = Matrix.new_from_existing(A)
    C.assign[1, [2,4]] = 0
    assert C == result_row
    C = Matrix.new_from_existing(A)
    C.assign[1, 2:5:2] = 0
    assert C == result_row
    # Test column
    result_column = Matrix.new_from_values(
        [3,0,6,0,6,6,2,4,1,1,1,3,5],
        [0,1,2,3,3,4,5,5,6,4,2,2,2],
        [3,2,5,3,7,3,1,7,4,8,0,0,0])
    C = Matrix.new_from_existing(A)
    C.assign[[1,3,5], 2] = 0
    assert C == result_column
    C = Matrix.new_from_existing(A)
    C.assign[1::2, 2] = 0
    assert C == result_column

def test_apply(A):
    result = Matrix.new_from_values(
        [3,0,3,5,6,0,6,1,6,2,4,1],
        [0,1,2,2,2,3,3,4,4,5,5,6],
        [-3,-2,-3,-1,-5,-3,-7,-8,-3,-1,-7,-4])
    A[:] = A.apply(UnaryOp.AINV)
    assert A == result

def test_apply_binary(A):
    # Test bind-first and bind-second
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_reduce_row(A):
    result = Vector.new_from_values([0,1,2,3,4,5,6], [5,12,1,6,7,1,15])
    w = Vector.new_from_type(A.dtype, 7)
    w[:] = A.reduce_rows(Monoid.PLUS)
    assert w == result

def test_reduce_column(A):
    result = Vector.new_from_values([0,1,2,3,4,5,6], [3,2,9,10,11,8,4])
    w = Vector.new_from_type(A.dtype, 7)
    w[:] = A.reduce_columns(Monoid.PLUS)
    assert w == result

def test_reduce_scalar(A):
    s = Scalar.new_from_type(A.dtype)
    s[:] = A.reduce_scalar(Monoid.PLUS)
    assert s == 47

def test_transpose(A):
    # C[:] = A.T
    rows, cols, vals = A.to_values()
    result = Matrix.new_from_values(cols, rows, vals)
    C = Matrix.new_from_type(A.dtype, A.ncols, A.nrows)
    C[:] = A.T
    assert C == result

def test_kronecker(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')

def test_simple_assignment(A):
    # C[:] = A
    C = Matrix.new_from_type(A.dtype, A.nrows, A.ncols)
    C[:] = A
    assert C == A

def test_equal(A, v):
    assert A == A
    assert A != v
    C = Matrix.new_from_values([1], [1], [1])
    assert C != A
    C2 = Matrix.new_from_values([1], [1], [1], nrows=7, ncols=7)
    assert C2 != A
    C3 = Matrix.new_from_values(
        [3,0,3,5,6,0,6,1,6,2,4,1],
        [0,1,2,2,2,3,3,4,4,5,5,6],
        [3.,2.,3.,1.,5.,3.,7.,8.,3.,1.,7.,4.])
    assert C3 != A, 'different datatypes are not equal'

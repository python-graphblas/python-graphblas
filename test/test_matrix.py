import pytest
from grblas import Matrix, Vector
from grblas import unary, binary, monoid, semiring
from grblas import dtypes
from grblas.exceptions import IndexOutOfBound, DimensionMismatch, OutputNotEmpty


@pytest.fixture
def A():
    #    0 1 2 3 4 5 6
    # 0 [- 2 - 3 - - -]
    # 1 [- - - - 8 - 4]
    # 2 [- - - - - 1 -]
    # 3 [3 - 3 - - - -]
    # 4 [- - - - - 7 -]
    # 5 [- - 1 - - - -]
    # 6 [- - 5 7 3 - -]
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
    ]
    return Matrix.from_values(*data)


@pytest.fixture
def v():
    data = [
        [1, 3, 4, 6],
        [1, 1, 2, 0]
    ]
    return Vector.from_values(*data)


def test_new():
    C = Matrix.new(dtypes.INT8, 17, 12)
    assert C.dtype == 'INT8'
    assert C.nvals == 0
    assert C.nrows == 17
    assert C.ncols == 12


def test_dup(A):
    C = A.dup()
    assert C is not A
    assert C.dtype == A.dtype
    assert C.nvals == A.nvals
    assert C.nrows == A.nrows
    assert C.ncols == A.ncols
    # Ensure they are not the same backend object
    A[0, 0] = 1000
    assert C[0, 0].value != 1000


def test_from_values():
    C = Matrix.from_values([0, 1, 3], [1, 1, 2], [True, False, True])
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == bool
    C2 = Matrix.from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=3)
    assert C2.nrows == 17
    assert C2.ncols == 3
    assert C2.nvals == 3
    assert C2.dtype == float
    C3 = Matrix.from_values([0, 1, 1], [2, 1, 1], [1, 2, 3], nrows=10, dup_op=binary.times)
    assert C3.nrows == 10
    assert C3.ncols == 3
    assert C3.nvals == 2  # duplicates were combined
    assert C3.dtype == int
    assert C3[1, 1].value == 6  # 2*3
    with pytest.raises(ValueError, match='Duplicate indices found'):
        # Duplicate indices requires a dup_op
        Matrix.from_values([0, 1, 1], [2, 1, 1], [True, True, True])
    with pytest.raises(IndexOutOfBound):
        # Specified ncols can't hold provided indexes
        Matrix.from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=2)
    with pytest.raises(ValueError, match='No values provided. Unable to determine type'):
        Matrix.from_values([], [], [])
    with pytest.raises(ValueError, match='No values provided. Unable to determine type'):
        Matrix.from_values([], [], [], nrows=3, ncols=4)
    with pytest.raises(ValueError, match='Unable to infer'):
        Matrix.from_values([], [], [], dtype=dtypes.INT64)
    C4 = Matrix.from_values([], [], [],  nrows=3, ncols=4, dtype=dtypes.INT64)
    C5 = Matrix.new(dtypes.INT64, nrows=3, ncols=4)
    assert C4.isequal(C5, check_dtype=True)


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


def test_build(A):
    assert A.nvals == 12
    A.clear()
    A.build([0, 6], [0, 1], [1, 2])
    assert A.nvals == 2
    with pytest.raises(OutputNotEmpty):
        A.build([1, 5], [2, 3], [3, 4])
    assert A.nvals == 2  # nothing should be modified
    # We can clear though
    A.build([1, 2, 5], [1, 2, 3], [2, 3, 4], clear=True)
    assert A.nvals == 3
    A.clear()
    with pytest.raises(IndexOutOfBound):
        A.build([0, 11], [0, 0], [1, 1])


def test_extract_values(A):
    rows, cols, vals = A.to_values()
    assert rows == (0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6)
    assert cols == (1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4)
    assert vals == (2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3)


def test_extract_element(A):
    assert A[3, 0].new() == 3
    assert A[1, 6].value == 4
    assert A.T[6, 1].value == 4


def test_set_element(A):
    assert A[1, 1].value is None
    assert A[3, 0].value == 3
    A[1, 1].update(21)
    A[3, 0] << -5
    assert A[1, 1].value == 21
    assert A[3, 0].new() == -5


def test_remove_element(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_mxm(A):
    C = A.mxm(A, semiring.plus_times).new()
    result = Matrix.from_values(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6],
        [0, 2, 4, 6, 2, 3, 4, 5, 2, 1, 3, 5, 2, 5, 0, 2, 5],
        [9, 9, 16, 8, 20, 28, 12, 56, 1, 6, 9, 3, 7, 1, 21, 21, 26])
    assert C.isequal(result)


def test_mxm_transpose(A):
    C = A.dup()
    C << A.mxm(A.T, semiring.plus_times)
    result = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
        [0, 6, 1, 6, 2, 4, 3, 5, 6, 2, 4, 3, 5, 6, 0, 1, 3, 5, 6],
        [13, 21, 80, 24, 1, 7, 18, 3, 15, 7, 49, 3, 1, 5, 21, 24, 15, 5, 83])
    assert C.isequal(result)
    C << A.T.mxm(A, semiring.plus_times)
    result2 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6],
        [0, 2, 1, 3, 0, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 6, 5, 4, 6],
        [9, 9, 4, 6, 9, 35, 35, 15, 6, 35, 58, 21, 15, 21, 73, 32, 50, 32, 16])
    assert C.isequal(result2)


def test_mxm_nonsquare():
    A = Matrix.from_values([0, 0, 0], [0, 2, 4], [1, 2, 3], nrows=1, ncols=5)
    B = Matrix.from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1)
    C = Matrix.new(A.dtype, nrows=1, ncols=1)
    C << A.mxm(B, semiring.max_plus)
    assert C[0, 0].value == 33
    C1 = A.mxm(B, semiring.max_plus).new()
    assert C1.isequal(C)
    C2 = A.T.mxm(B.T, semiring.max_plus).new()
    assert C2.nrows == 5
    assert C2.ncols == 5


def test_mxm_mask(A):
    val_mask = Matrix.from_values([0, 3, 4], [2, 3, 2], [True, True, True], nrows=7, ncols=7)
    struct_mask = Matrix.from_values([0, 3, 4], [2, 3, 2], [1, 0, 0], nrows=7, ncols=7)
    C = A.dup()
    C(val_mask.V) << A.mxm(A, semiring.plus_times)
    result = Matrix.from_values(
        [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 6, 6],
        [1, 2, 3, 4, 6, 5, 0, 2, 3, 2, 5, 2, 2, 3, 4],
        [2, 9, 3, 8, 4, 1, 3, 3, 9, 7, 7, 1, 5, 7, 3])
    assert C.isequal(result)
    C = A.dup()
    C(~val_mask.V) << A.mxm(A, semiring.plus_times)
    result2 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 5, 6, 6, 6],
        [0, 4, 6, 2, 3, 4, 5, 2, 1, 5, 5, 0, 2, 5],
        [9, 16, 8, 20, 28, 12, 56, 1, 6, 3, 1, 21, 21, 26])
    assert C.isequal(result2)
    C = A.dup()
    C(struct_mask.S, replace=True).update(A.mxm(A, semiring.plus_times))
    result3 = Matrix.from_values(
        [0, 3, 4],
        [2, 3, 2],
        [9, 9, 7], nrows=7, ncols=7)
    assert C.isequal(result3)
    C2 = A.mxm(A, semiring.plus_times).new(mask=struct_mask.S)
    assert C2.isequal(result3)
    with pytest.raises(TypeError, match="Mask must indicate"):
        A.mxm(A).new(mask=struct_mask)


def test_mxm_accum(A):
    A(binary.plus) << A.mxm(A, semiring.plus_times)
    result = Matrix.from_values(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6],
        [0, 1, 2, 3, 4, 6, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 5, 2, 5, 2, 5, 0, 2, 3, 4, 5],
        [9, 2, 9, 3, 16, 8, 20, 28, 20, 56, 4, 1, 1, 3, 6, 3, 9, 3, 7, 7, 1, 1, 21, 26, 7, 3, 26])
    assert A.isequal(result)


def test_mxv(A, v):
    w = A.mxv(v, semiring.plus_times).new()
    result = Vector.from_values([0, 1, 6], [5, 16, 13])
    assert w.isequal(result)


def test_ewise_mult(A):
    # Binary, Monoid, and Semiring
    B = Matrix.from_values([0, 0, 5], [1, 2, 2], [5, 4, 8], nrows=7, ncols=7)
    result = Matrix.from_values([0, 5], [1, 2], [10, 8], nrows=7, ncols=7)
    C = A.ewise_mult(B, binary.times).new()
    assert C.isequal(result)
    C() << A.ewise_mult(B, monoid.times)
    assert C.isequal(result)
    C << A.ewise_mult(B, semiring.plus_times)
    assert C.isequal(result)


def test_ewise_add(A):
    # Binary, Monoid, and Semiring
    B = Matrix.from_values([0, 0, 5], [1, 2, 2], [5, 4, 8], nrows=7, ncols=7)
    result = Matrix.from_values(
        [0, 3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [4, 3, 5, 3, 8, 5, 3, 7, 8, 3, 1, 7, 4]
    )
    with pytest.raises(TypeError, match="require_monoid"):
        A.ewise_add(B, binary.second)
    # surprising that SECOND(x, empty) == x, which is why user must opt-in to using binary ops in ewise_add
    C = A.ewise_add(B, binary.second, require_monoid=False).new()
    assert C.isequal(result)
    C << A.ewise_add(B, monoid.max)
    assert C.isequal(result)
    C << A.ewise_add(B, semiring.max_minus)
    assert C.isequal(result)


def test_extract(A):
    C = Matrix.new(A.dtype, 3, 4)
    result = Matrix.from_values([0, 0, 1, 2, 2, 2], [0, 2, 1, 1, 2, 3], [2, 3, 3, 5, 7, 3], nrows=3, ncols=4)
    C << A[[0, 3, 6], [1, 2, 3, 4]]
    assert C.isequal(result)
    C << A[0::3, 1:5]
    assert C.isequal(result)
    C << A[[0, 3, 6], 1:5:1]
    assert C.isequal(result)
    C2 = A[[0, 3, 6], [1, 2, 3, 4]].new()
    assert C2.isequal(result)


def test_extract_row(A):
    w = Vector.new(A.dtype, 3)
    result = Vector.from_values([1, 2], [5, 3], size=3)
    w << A[6, [0, 2, 4]]
    assert w.isequal(result)
    w << A[6, :5:2]
    assert w.isequal(result)
    w << A.T[[0, 2, 4], 6]
    assert w.isequal(result)
    w2 = A[6, [0, 2, 4]].new()
    assert w2.isequal(result)


def test_extract_column(A):
    w = Vector.new(A.dtype, 3)
    result = Vector.from_values([1, 2], [3, 1], size=3)
    w << A[[1, 3, 5], 2]
    assert w.isequal(result)
    w << A[1:6:2, 2]
    assert w.isequal(result)
    w << A.T[2, [1, 3, 5]]
    assert w.isequal(result)
    w2 = A[1:6:2, 2].new()
    assert w2.isequal(result)


def test_assign(A):
    B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
    result = Matrix.from_values(
        [0, 0, 2, 3, 0, 3, 5, 6, 0, 6, 1, 6, 4, 1],
        [0, 5, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6],
        [9, 8, 7, 3, 2, 3, 1, 5, 3, 7, 8, 3, 7, 4])
    C = A.dup()
    C()[[0, 2], [0, 5]] = B
    assert C.isequal(result)
    C = A.dup()
    C[:3:2, :6:5]() << B
    assert C.isequal(result)


def test_assign_wrong_dims(A):
    B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
    with pytest.raises(DimensionMismatch):
        A[[0, 2, 4], [0, 5]] = B


def test_assign_row(A, v):
    result = Matrix.from_values(
        [3, 3, 5, 6, 6, 1, 6, 2, 4, 1, 0, 0, 0, 0],
        [0, 2, 2, 2, 3, 4, 4, 5, 5, 6, 1, 3, 4, 6],
        [3, 3, 1, 5, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0])
    C = A.dup()
    C[0, :] = v
    assert C.isequal(result)


def test_assign_column(A, v):
    result = Matrix.from_values(
        [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4, 6],
        [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1, 1],
        [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0])
    C = A.dup()
    C[:, 1] = v
    assert C.isequal(result)


def test_assign_scalar(A):
    # Test block
    result_block = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 3, 5, 1, 3, 5],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4, 4, 4],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 0, 0, 0, 0, 0, 0])
    C = A.dup()
    C[[1, 3, 5], [2, 4]] = 0
    assert C.isequal(result_block)
    C = A.dup()
    C[1::2, 2:5:2] = 0
    assert C.isequal(result_block)
    # Test row
    result_row = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 3, 5, 1, 1],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 3, 1, 0, 0])
    C = A.dup()
    C[1, [2, 4]] = 0
    assert C.isequal(result_row)
    C = A.dup()
    C[1, 2:5:2] = 0
    assert C.isequal(result_row)
    # Test column
    result_column = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 1, 3, 5],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 4, 2, 2, 2],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 8, 0, 0, 0])
    C = A.dup()
    C[[1, 3, 5], 2] = 0
    assert C.isequal(result_column)
    C = A.dup()
    C[1::2, 2] = 0
    assert C.isequal(result_column)


def test_apply(A):
    result = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [-3, -2, -3, -1, -5, -3, -7, -8, -3, -1, -7, -4])
    C = A.apply(unary.ainv).new()
    assert C.isequal(result)


def test_apply_binary(A):
    # Test bind-first and bind-second
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_reduce_row(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    w = A.reduce_rows(monoid.plus).new()
    assert w.isequal(result)


def test_reduce_column(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6],  [3, 2, 9, 10, 11, 8, 4])
    w = A.reduce_columns(monoid.plus).new()
    assert w.isequal(result)


def test_reduce_scalar(A):
    s = A.reduce_scalar(monoid.plus).new()
    assert s == 47


def test_transpose(A):
    # C << A.T
    rows, cols, vals = A.to_values()
    result = Matrix.from_values(cols, rows, vals)
    C = Matrix.new(A.dtype, A.ncols, A.nrows)
    C << A.T
    assert C.isequal(result)
    C2 = A.T.new()
    assert C2.isequal(result)


def test_kronecker(A):
    pytest.xfail('Not implemented in GraphBLAS 1.2')


def test_simple_assignment(A):
    # C << A
    C = Matrix.new(A.dtype, A.nrows, A.ncols)
    C << A
    assert C.isequal(A)


def test_isequal(A, v):
    assert A.isequal(A)
    assert not A.isequal(v)
    C = Matrix.from_values([1], [1], [1])
    assert not C.isequal(A)
    C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)
    assert not C2.isequal(A)
    C3 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3., 2., 3., 1., 5., 3., 7., 8., 3., 1., 7., 4.])
    assert not C3.isequal(A, check_dtype=True), 'different datatypes are not equal'
    C4 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3., 2., 3., 1., 5., 3.000000000000000001, 7., 8., 3., 1-1e-11, 7., 4.])
    assert not C4.isequal(A)


def test_isclose(A, v):
    assert A.isclose(A)
    assert not A.isclose(v)
    C = Matrix.from_values([1], [1], [1])  # wrong size
    assert not C.isclose(A)
    C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)  # missing values
    assert not C2.isclose(A)
    C3 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 0],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 2],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 3])  # extra values
    assert not C3.isclose(A)
    C4 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3., 2., 3., 1., 5., 3., 7., 8., 3., 1., 7., 4.])
    assert not C4.isclose(A, check_dtype=True), 'different datatypes are not equal'
    C5 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3., 2., 3., 1., 5., 3.000000000000000001, 7., 8., 3., 1-1e-11, 7., 4.])
    assert C5.isclose(A)
    C6 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3., 2.000001, 3., 1., 5., 3., 7., 7.9999999, 3., 1., 7., 4.])
    assert C6.isclose(A, rel_tol=1e-3)


def test_transpose_equals(A):
    data = [
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
    ]
    B = Matrix.from_values(*data)
    assert A.isequal(B.T)
    assert B.isequal(A.T)
    assert A.T.isequal(B)
    assert A.T.isequal(A.T)
    assert A.isclose(A)
    assert A.isclose(B.T)
    assert B.isclose(A.T)
    assert A.T.isclose(B)
    assert A.T.isclose(A.T)

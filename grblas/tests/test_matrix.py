import pytest
import itertools
import grblas
import numpy as np
import pickle
import weakref
from grblas import Matrix, Vector, Scalar
from grblas import unary, binary, monoid, semiring
from grblas import dtypes
from grblas.exceptions import (
    IndexOutOfBound,
    DimensionMismatch,
    OutputNotEmpty,
    InvalidValue,
    DomainMismatch,
)


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
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
    ]
    return Matrix.from_values(*data)


@pytest.fixture
def v():
    data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    return Vector.from_values(*data)


def test_new():
    C = Matrix.new(dtypes.INT8, 17, 12)
    assert C.dtype == "INT8"
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
    # extended functionality
    D = Matrix.from_values([0, 1], [0, 1], [0, 2.5], dtype=dtypes.FP64)
    E = D.dup(dtype=dtypes.INT64)
    assert E.isequal(
        Matrix.from_values([0, 1], [0, 1], [0, 2], dtype=dtypes.INT64), check_dtype=True
    )
    E = D.dup(mask=D.V)
    assert E.isequal(Matrix.from_values([1], [1], [2.5], dtype=dtypes.FP64), check_dtype=True)
    E = D.dup(dtype=dtypes.INT64, mask=D.V)
    assert E.isequal(Matrix.from_values([1], [1], [2], dtype=dtypes.INT64), check_dtype=True)


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
    C3monoid = Matrix.from_values([0, 1, 1], [2, 1, 1], [1, 2, 3], nrows=10, dup_op=monoid.times)
    assert C3.isequal(C3monoid)

    with pytest.raises(ValueError, match="Duplicate indices found"):
        # Duplicate indices requires a dup_op
        Matrix.from_values([0, 1, 1], [2, 1, 1], [True, True, True])
    with pytest.raises(IndexOutOfBound):
        # Specified ncols can't hold provided indexes
        Matrix.from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=2)
    with pytest.raises(ValueError, match="No row indices provided. Unable to infer nrows."):
        Matrix.from_values([], [], [])

    # Changed: Assume empty value is float64 (like numpy)
    # with pytest.raises(ValueError, match="No values provided. Unable to determine type"):
    empty1 = Matrix.from_values([], [], [], nrows=3, ncols=4)
    assert empty1.dtype == dtypes.FP64
    assert empty1.nrows == 3
    assert empty1.ncols == 4
    assert empty1.nvals == 0

    with pytest.raises(ValueError, match="Unable to infer"):
        Matrix.from_values([], [], [], dtype=dtypes.INT64)
    with pytest.raises(ValueError, match="Unable to infer"):
        # could also raise b/c rows and columns are different sizes
        Matrix.from_values([0], [], [0], dtype=dtypes.INT64)
    C4 = Matrix.from_values([], [], [], nrows=3, ncols=4, dtype=dtypes.INT64)
    C5 = Matrix.new(dtypes.INT64, nrows=3, ncols=4)
    assert C4.isequal(C5, check_dtype=True)

    with pytest.raises(
        ValueError, match="`rows` and `columns` and `values` lengths must match: 1, 2, 1"
    ):
        Matrix.from_values([0], [1, 2], [0])


def test_clear(A):
    A.clear()
    assert A.nvals == 0
    assert A.nrows == 7
    assert A.ncols == 7


def test_resize(A):
    assert A.nrows == 7
    assert A.ncols == 7
    assert A.nvals == 12
    A.resize(10, 11)
    assert A.nrows == 10
    assert A.ncols == 11
    assert A.nvals == 12
    assert A[9, 9].value is None
    A.resize(4, 1)
    assert A.nrows == 4
    assert A.ncols == 1
    assert A.nvals == 1


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
    B = Matrix.new(int, nrows=2, ncols=2)
    B.build([0, 11], [0, 0], [1, 1], nrows=12)
    assert B.isequal(Matrix.from_values([0, 11], [0, 0], [1, 1], ncols=2))
    C = Matrix.new(int, nrows=2, ncols=2)
    C.build([0, 0], [0, 11], [1, 1], ncols=12)
    assert C.isequal(Matrix.from_values([0, 0], [0, 11], [1, 1], nrows=2))


def test_extract_values(A):
    rows, cols, vals = A.to_values(dtype=int)
    np.testing.assert_array_equal(rows, (0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6))
    np.testing.assert_array_equal(cols, (1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4))
    np.testing.assert_array_equal(vals, (2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3))
    assert rows.dtype == np.uint64
    assert cols.dtype == np.uint64
    assert vals.dtype == np.int64
    Trows, Tcols, Tvals = A.T.to_values(dtype=float)
    np.testing.assert_array_equal(rows, Tcols)
    np.testing.assert_array_equal(cols, Trows)
    np.testing.assert_array_equal(vals, Tvals)
    assert Trows.dtype == np.uint64
    assert Tcols.dtype == np.uint64
    assert Tvals.dtype == np.float64


def test_extract_element(A):
    assert A[3, 0].new() == 3
    assert A[1, 6].value == 4
    assert A.T[6, 1].value == 4
    s = A[0, 0].new()
    assert s.value is None
    assert s.dtype == "INT64"
    s = A[1, 6].new(dtype=float)
    assert s.value == 4.0
    assert s.dtype == "FP64"


def test_set_element(A):
    assert A[1, 1].value is None
    assert A[3, 0].value == 3
    A[1, 1].update(21)
    A[3, 0] << -5
    assert A[1, 1].value == 21
    assert A[3, 0].new() == -5


def test_remove_element(A):
    assert A[3, 0].value == 3
    del A[3, 0]
    assert A[3, 0].value is None
    assert A[6, 3].value == 7
    with pytest.raises(TypeError, match="Remove Element only supports"):
        del A[3:5, 3]


def test_mxm(A):
    C = A.mxm(A, semiring.plus_times).new()
    result = Matrix.from_values(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6],
        [0, 2, 4, 6, 2, 3, 4, 5, 2, 1, 3, 5, 2, 5, 0, 2, 5],
        [9, 9, 16, 8, 20, 28, 12, 56, 1, 6, 9, 3, 7, 1, 21, 21, 26],
    )
    assert C.isequal(result)


def test_mxm_transpose(A):
    C = A.dup()
    C << A.mxm(A.T, semiring.plus_times)
    result = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
        [0, 6, 1, 6, 2, 4, 3, 5, 6, 2, 4, 3, 5, 6, 0, 1, 3, 5, 6],
        [13, 21, 80, 24, 1, 7, 18, 3, 15, 7, 49, 3, 1, 5, 21, 24, 15, 5, 83],
    )
    assert C.isequal(result)
    C << A.T.mxm(A, semiring.plus_times)
    result2 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6],
        [0, 2, 1, 3, 0, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 6, 5, 4, 6],
        [9, 9, 4, 6, 9, 35, 35, 15, 6, 35, 58, 21, 15, 21, 73, 32, 50, 32, 16],
    )
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
        [2, 9, 3, 8, 4, 1, 3, 3, 9, 7, 7, 1, 5, 7, 3],
    )
    assert C.isequal(result)
    C = A.dup()
    C(~val_mask.V) << A.mxm(A, semiring.plus_times)
    result2 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 5, 6, 6, 6],
        [0, 4, 6, 2, 3, 4, 5, 2, 1, 5, 5, 0, 2, 5],
        [9, 16, 8, 20, 28, 12, 56, 1, 6, 3, 1, 21, 21, 26],
    )
    assert C.isequal(result2)
    C = A.dup()
    C(struct_mask.S, replace=True).update(A.mxm(A, semiring.plus_times))
    result3 = Matrix.from_values([0, 3, 4], [2, 3, 2], [9, 9, 7], nrows=7, ncols=7)
    assert C.isequal(result3)
    C2 = A.mxm(A, semiring.plus_times).new(mask=struct_mask.S)
    assert C2.isequal(result3)
    with pytest.raises(TypeError, match="Mask must indicate"):
        A.mxm(A).new(mask=struct_mask)


def test_mxm_accum(A):
    A(binary.plus) << A.mxm(A, semiring.plus_times)
    # fmt: off
    result = Matrix.from_values(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6],
        [0, 1, 2, 3, 4, 6, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 5, 2, 5, 2, 5, 0, 2, 3, 4, 5],
        [9, 2, 9, 3, 16, 8, 20, 28, 20, 56, 4, 1, 1, 3, 6, 3, 9, 3, 7, 7, 1, 1, 21, 26, 7, 3, 26],
    )
    # fmt: on
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
    with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
        A.ewise_mult(B, semiring.plus_times)


def test_ewise_add(A):
    # Binary, Monoid, and Semiring
    B = Matrix.from_values([0, 0, 5], [1, 2, 2], [5, 4, 8], nrows=7, ncols=7)
    result = Matrix.from_values(
        [0, 3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [4, 3, 5, 3, 8, 5, 3, 7, 8, 3, 1, 7, 4],
    )
    with pytest.raises(TypeError, match="require_monoid"):
        A.ewise_add(B, binary.second)
    # surprising that SECOND(x, empty) == x, which is why user
    # must opt-in to using binary ops in ewise_add
    C = A.ewise_add(B, binary.second, require_monoid=False).new()
    assert C.isequal(result)
    C << A.ewise_add(B, monoid.max)
    assert C.isequal(result)
    C << A.ewise_add(B, binary.max)
    assert C.isequal(result)
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        A.ewise_add(B, semiring.max_minus)


def test_extract(A):
    C = Matrix.new(A.dtype, 3, 4)
    result = Matrix.from_values(
        [0, 0, 1, 2, 2, 2], [0, 2, 1, 1, 2, 3], [2, 3, 3, 5, 7, 3], nrows=3, ncols=4
    )
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


def test_extract_input_mask():
    # A       M
    # 0 1 2   _ 0 1
    # 3 4 5   2 3 _
    A = Matrix.from_values(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5],
    )
    M = Matrix.from_values(
        [0, 0, 1, 1],
        [1, 2, 0, 1],
        [0, 1, 2, 3],
    )
    m = M[0, :].new()
    MT = M.T.new()
    # Matrix structure mask
    result = A[0, [0, 1]].new(input_mask=M.S)
    expected = Vector.from_values([1], [1])
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=M.S) << A[0, [0, 1]]
    assert result.isequal(expected)

    # Vector mask
    result = A[0, [0, 1]].new(input_mask=m.S)
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=m.S) << A[0, [0, 1]]
    assert result.isequal(expected)

    # Matrix value mask
    result = A[0, [1, 2]].new(input_mask=M.V)
    expected = Vector.from_values([1], [2], size=2)
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=M.V) << A[0, [1, 2]]
    assert result.isequal(expected)

    with pytest.raises(ValueError, match="Shape of `input_mask` does not match shape of input"):
        A[0, [0, 1]].new(input_mask=MT.S)
    with pytest.raises(ValueError, match="Shape of `input_mask` does not match shape of input"):
        m(input_mask=MT.S) << A[0, [0, 1]]
    with pytest.raises(
        ValueError, match="Size of `input_mask` Vector does not match ncols of Matrix"
    ):
        A[0, [0]].new(input_mask=expected.S)
    with pytest.raises(
        ValueError, match="Size of `input_mask` Vector does not match ncols of Matrix"
    ):
        m(input_mask=expected.S) << A[0, [0]]
    with pytest.raises(
        ValueError, match="Size of `input_mask` Vector does not match nrows of Matrix"
    ):
        A[[0], 0].new(input_mask=m.S)
    with pytest.raises(
        ValueError, match="Size of `input_mask` Vector does not match nrows of Matrix"
    ):
        m(input_mask=m.S) << A[[0], 0]
    with pytest.raises(
        TypeError, match="Got Vector `input_mask` when extracting a submatrix from a Matrix"
    ):
        A[[0], [0]].new(input_mask=expected.S)
    with pytest.raises(
        TypeError, match="Got Vector `input_mask` when extracting a submatrix from a Matrix"
    ):
        A(input_mask=expected.S) << A[[0], [0]]
    with pytest.raises(TypeError, match="mask is not allowed for single element extraction"):
        A[0, 0].new(input_mask=M.S)
    with pytest.raises(TypeError, match="mask and input_mask arguments cannot both be given"):
        A[0, [0, 1]].new(input_mask=M.S, mask=expected.S)
    with pytest.raises(TypeError, match="mask and input_mask arguments cannot both be given"):
        A(input_mask=M.S, mask=expected.S)
    with pytest.raises(TypeError, match=r"Mask must indicate values \(M.V\) or structure \(M.S\)"):
        A[0, [0, 1]].new(input_mask=M)
    with pytest.raises(TypeError, match=r"Mask must indicate values \(M.V\) or structure \(M.S\)"):
        A(input_mask=M)
    with pytest.raises(TypeError, match="Mask object must be type Vector"):
        expected[[0, 1]].new(input_mask=M.S)
    with pytest.raises(TypeError, match="Mask object must be type Vector"):
        expected(input_mask=M.S) << expected[[0, 1]]
    with pytest.raises(TypeError, match=r"new\(\) got an unexpected keyword argument 'input_mask'"):
        A.new(input_mask=M.S)
    with pytest.raises(TypeError, match="`input_mask` argument may only be used for extract"):
        A(input_mask=M.S) << A.apply(unary.ainv)
    with pytest.raises(TypeError, match="`input_mask` argument may only be used for extract"):
        A(input_mask=M.S)[[0], [0]] = 1
    with pytest.raises(TypeError, match="`input_mask` argument may only be used for extract"):
        A(input_mask=M.S)[[0], [0]]

    # With transpose input value
    # Matrix structure mask
    result = A.T[[0, 1], 0].new(input_mask=MT.S)
    expected = Vector.from_values([1], [1])
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=MT.S) << A.T[[0, 1], 0]
    assert result.isequal(expected)

    # Vector mask
    result = A.T[[0, 1], 0].new(input_mask=m.S)
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=m.S) << A.T[[0, 1], 0]
    assert result.isequal(expected)

    # Matrix value mask
    result = A.T[[1, 2], 0].new(input_mask=MT.V)
    expected = Vector.from_values([1], [2], size=2)
    assert result.isequal(expected)
    # again
    result.clear()
    result(input_mask=MT.V) << A.T[[1, 2], 0]
    assert result.isequal(expected)


def test_extract_with_matrix(A):
    with pytest.raises(TypeError, match="Invalid type for index"):
        A[A.T, 1].new()
    with pytest.raises(TypeError, match="Invalid type for index"):
        A[A, [1]].new()
    with pytest.raises(TypeError, match="Invalid type for index"):
        A[[0], A.V].new()


def test_assign(A):
    B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
    result = Matrix.from_values(
        [0, 0, 2, 3, 0, 3, 5, 6, 0, 6, 1, 6, 4, 1],
        [0, 5, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6],
        [9, 8, 7, 3, 2, 3, 1, 5, 3, 7, 8, 3, 7, 4],
    )
    C = A.dup()
    C()[[0, 2], [0, 5]] = B
    assert C.isequal(result)
    C = A.dup()
    C[:3:2, :6:5]() << B
    assert C.isequal(result)
    with pytest.raises(TypeError, match="will make the Matrix dense"):
        C << 1
    nvals = C.nvals
    C(C.S) << 1
    assert C.nvals == nvals
    assert C.reduce_scalar().value == nvals
    with pytest.raises(TypeError, match="Invalid type for index"):
        C[C, [1]] = C


def test_assign_wrong_dims(A):
    B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
    with pytest.raises(DimensionMismatch):
        A[[0, 2, 4], [0, 5]] = B


def test_assign_row(A, v):
    result = Matrix.from_values(
        [3, 3, 5, 6, 6, 1, 6, 2, 4, 1, 0, 0, 0, 0],
        [0, 2, 2, 2, 3, 4, 4, 5, 5, 6, 1, 3, 4, 6],
        [3, 3, 1, 5, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0],
    )
    C = A.dup()
    C[0, :] = v
    assert C.isequal(result)


def test_subassign_row_col():
    A = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    m = Vector.from_values([1], [True])
    v = Vector.from_values([0, 1], [10, 20])

    A[[0, 1], 0](m.S) << v
    result1 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 20, 4, 5, 6, 7, 8],
    )
    assert A.isequal(result1)

    A[1, [1, 2]](m.V, accum=binary.plus).update(v)
    result2 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 20, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result2)

    A[[0, 1], 0](m.S, binary.plus, replace=True) << v
    result3 = Matrix.from_values(
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 40, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result3)

    with pytest.raises(DimensionMismatch):
        A(m.S)[[0, 1], 0] << v

    A[[0, 1], 0](m.S) << 99
    result4 = Matrix.from_values(
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 99, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result4)

    A[[1, 2], 0](m.S, binary.plus, replace=True) << 100
    result5 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2],
        [1, 2, 1, 2, 0, 1, 2],
        [1, 2, 4, 25, 106, 7, 8],
    )
    assert A.isequal(result5)

    A[2, [0, 1]](m.S) << -1
    result6 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2],
        [1, 2, 1, 2, 0, 1, 2],
        [1, 2, 4, 25, 106, -1, 8],
    )
    assert A.isequal(result6)


def test_subassign_matrix():
    A = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    m = Matrix.from_values([1], [0], [True])
    v = Matrix.from_values([0, 1], [0, 0], [10, 20])
    mT = m.T.new()

    A[[0, 1], [0]](m.S) << v
    result1 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 20, 4, 5, 6, 7, 8],
    )
    assert A.isequal(result1)

    A[[1], [1, 2]](mT.V, accum=binary.plus) << v.T
    result2 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 20, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result2)

    A[[0, 1], [0]](m.S, binary.plus, replace=True) << v
    result3 = Matrix.from_values(
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 40, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result3)

    with pytest.raises(DimensionMismatch):
        A(m.S)[[0, 1], [0]] << v

    A[[0, 1], [0]](m.S) << 99
    result4 = Matrix.from_values(
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 99, 4, 25, 6, 7, 8],
    )
    assert A.isequal(result4)

    A[[1, 2], [0]](m.S, binary.plus, replace=True) << 100
    result5 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2],
        [1, 2, 1, 2, 0, 1, 2],
        [1, 2, 4, 25, 106, 7, 8],
    )
    assert A.isequal(result5)

    A[[2], [0, 1]](mT.S) << -1
    result6 = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2],
        [1, 2, 1, 2, 0, 1, 2],
        [1, 2, 4, 25, 106, -1, 8],
    )
    assert A.isequal(result6)


def test_assign_column(A, v):
    result = Matrix.from_values(
        [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4, 6],
        [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1, 1],
        [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0],
    )
    C = A.dup()
    C[:, 1] = v
    assert C.isequal(result)


def test_assign_row_scalar(A, v):
    C = A.dup()
    C[0, :](v.S) << v
    D = A.dup()
    D(v.S)[0, :] << v
    assert C.isequal(D)

    C[:, :](C.S) << 1

    with pytest.raises(
        TypeError, match="Unable to use Vector mask on Matrix assignment to a Matrix"
    ):
        C[:, :](v.S) << 1
    with pytest.raises(
        TypeError, match="Unable to use Vector mask on single element assignment to a Matrix"
    ):
        C[0, 0](v.S) << 1

    with pytest.raises(TypeError):
        C[0, 0](v.S) << v
    with pytest.raises(TypeError):
        C(v.S)[0, 0] << v
    with pytest.raises(TypeError):
        C[0, 0](C.S) << v
    with pytest.raises(TypeError):
        C(C.S)[0, 0] << v

    with pytest.raises(TypeError):
        C[0, 0](v.S) << C
    with pytest.raises(TypeError):
        C[0, 0](C.S) << C

    C = A.dup()
    C(v.S)[0, :] = 10
    result = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 0, 0],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 4, 6],
        [3, 10, 3, 1, 5, 10, 7, 8, 3, 1, 7, 4, 10, 10],
    )
    assert C.isequal(result)


def test_assign_row_col_matrix_mask():
    # A         B       v1      v2
    # 0 1       4 _     100     10
    # 2 _       0 5             20
    A = Matrix.from_values([0, 0, 1], [0, 1, 0], [0, 1, 2])
    B = Matrix.from_values([0, 1, 1], [0, 0, 1], [4, 0, 5])
    v1 = Vector.from_values([0], [100])
    v2 = Vector.from_values([0, 1], [10, 20])

    # row assign
    C = A.dup()
    C(B.S)[0, :] << v2
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 2])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, accum=binary.plus)[1, :] = v2
    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 12, 20])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, replace=True)[1, :] << v2
    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 10, 20])
    assert C.isequal(result)

    # col assign
    C = A.dup()
    C(B.S)[:, 0] = v2
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 20])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, accum=binary.plus)[:, 1] << v2
    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 20])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, replace=True)[:, 1] = v2
    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 2, 20])
    assert C.isequal(result)

    # row assign scalar (as a sanity check)
    C = A.dup()
    C(B.S)[0, :] = 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, accum=binary.plus)[1, :] << 100
    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 102, 100])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, replace=True)[1, :] = 100
    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 100, 100])
    assert C.isequal(result)

    # col assign scalar (as a sanity check)
    C = A.dup()
    C(B.S)[:, 0] << 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 100])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, accum=binary.plus)[:, 1] = 100
    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 100])
    assert C.isequal(result)

    C = A.dup()
    C(B.S, replace=True)[:, 1] << 100
    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 2, 100])
    assert C.isequal(result)

    # row subassign
    C = A.dup()
    C[0, :](v2.S) << v2
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 20, 2])
    assert C.isequal(result)

    C = A.dup()
    C[0, [0]](v1.S) << v1
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
    assert C.isequal(result)

    with pytest.raises(
        TypeError, match="Indices for subassign imply Vector submask, but got Matrix mask instead"
    ):
        C[0, :](B.S) << v2

    # col subassign
    C = A.dup()
    C[:, 0](v2.S) << v2
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 20])
    assert C.isequal(result)

    C = A.dup()
    C[[0], 0](v1.S) << v1
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
    assert C.isequal(result)

    with pytest.raises(
        TypeError, match="Indices for subassign imply Vector submask, but got Matrix mask instead"
    ):
        C[:, 0](B.S) << v2

    # row subassign scalar
    C = A.dup()
    C[0, :](v2.S) << 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 100, 2])
    assert C.isequal(result)

    C = A.dup()
    C[0, [0]](v1.S) << 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
    assert C.isequal(result)

    with pytest.raises(
        TypeError, match="Indices for subassign imply Vector submask, but got Matrix mask instead"
    ):
        C[:, 0](B.S) << 100

    # col subassign scalar
    C = A.dup()
    C[:, 0](v2.S) << 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 100])
    assert C.isequal(result)

    C = A.dup()
    C[[0], 0](v1.S) << 100
    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
    assert C.isequal(result)

    with pytest.raises(
        TypeError, match="Indices for subassign imply Vector submask, but got Matrix mask instead"
    ):
        C[:, 0](B.S) << 100

    # Bad subassign
    with pytest.raises(TypeError, match="Single element assign does not accept a submask"):
        C[0, 0](B.S) << 100


def test_assign_column_scalar(A, v):
    C = A.dup()
    C[:, 0](v.S) << v
    D = A.dup()
    D(v.S)[:, 0] << v
    assert C.isequal(D)

    C = A.dup()
    C[:, 1] = v
    C(v.S)[:, 1] = 10
    result = Matrix.from_values(
        [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4, 6],
        [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1, 1],
        [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 10, 10, 10, 10],
    )
    assert C.isequal(result)

    C(v.V, replace=True, accum=binary.plus)[:, 1] = 20
    result = Matrix.from_values(
        [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4],
        [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1],
        [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 30, 30, 30],
    )
    assert C.isequal(result)


def test_assign_scalar(A):
    # Test block
    result_block = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 3, 5, 1, 3, 5],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4, 4, 4],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 0, 0, 0, 0, 0, 0],
    )
    C = A.dup()
    C[[1, 3, 5], [2, 4]] = 0
    assert C.isequal(result_block)
    C = A.dup()
    C[[1, 3, 5], [2, 4]] = Scalar.from_value(0)
    assert C.isequal(result_block)
    C = A.dup()
    C[1::2, 2:5:2] = 0
    assert C.isequal(result_block)
    C = A.dup()
    C[1::2, 2:5:2] = Scalar.from_value(0)
    assert C.isequal(result_block)
    # Test row
    result_row = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 3, 5, 1, 1],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 3, 1, 0, 0],
    )
    C = A.dup()
    C[1, [2, 4]] = 0
    assert C.isequal(result_row)
    C = A.dup()
    C[1, 2] = Scalar.from_value(0)
    C[1, 4] = Scalar.from_value(0)
    assert C.isequal(result_row)
    C = A.dup()
    C[1, 2:5:2] = 0
    assert C.isequal(result_row)
    # Test column
    result_column = Matrix.from_values(
        [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 1, 3, 5],
        [0, 1, 2, 3, 3, 4, 5, 5, 6, 4, 2, 2, 2],
        [3, 2, 5, 3, 7, 3, 1, 7, 4, 8, 0, 0, 0],
    )
    C = A.dup()
    C[[1, 3, 5], 2] = 0
    assert C.isequal(result_column)
    C = A.dup()
    C[1::2, 2] = 0
    assert C.isequal(result_column)


def test_assign_bad(A):
    with pytest.raises(TypeError, match="Bad type"):
        A[0, 0] = object()
    with pytest.raises(TypeError, match="Bad type"):
        A[:, 0] = object()
    with pytest.raises(TypeError, match="Bad type"):
        A[0, :] = object()
    with pytest.raises(TypeError, match="Bad type"):
        A[:, :] = object()
    with pytest.raises(TypeError, match="Bad type"):
        A[0, 0] = A
    with pytest.raises(TypeError, match="Bad type"):
        A[:, 0] = A
    with pytest.raises(TypeError, match="Bad type"):
        A[0, :] = A
    v = A[0, :].new()
    with pytest.raises(TypeError, match="Bad type"):
        A[0, 0] = v
    with pytest.raises(TypeError, match="Bad type"):
        A[:, :] = v


def test_apply(A):
    result = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [-3, -2, -3, -1, -5, -3, -7, -8, -3, -1, -7, -4],
    )
    C = A.apply(unary.ainv).new()
    assert C.isequal(result)


def test_apply_binary(A):
    result_right = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        dtype=bool,
    )
    w_right = A.apply(binary.gt, right=1).new()
    w_right2 = A.apply(binary.gt, right=Scalar.from_value(1)).new()
    assert w_right.isequal(result_right)
    assert w_right2.isequal(result_right)
    result_left = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [5, 6, 5, 7, 3, 5, 1, 0, 5, 7, 1, 4],
    )
    w_left = A.apply(binary.minus, left=8).new()
    w_left2 = A.apply(binary.minus, left=Scalar.from_value(8)).new()
    assert w_left.isequal(result_left)
    assert w_left2.isequal(result_left)
    with pytest.raises(TypeError):
        A.apply(binary.plus, left=A)
    with pytest.raises(TypeError):
        A.apply(binary.plus, right=A)
    with pytest.raises(TypeError, match="Cannot provide both"):
        A.apply(binary.plus, left=1, right=1)

    # allow monoids
    w1 = A.apply(binary.plus, left=1).new()
    w2 = A.apply(monoid.plus, left=1).new()
    w3 = A.apply(monoid.plus, right=1).new()
    assert w1.isequal(w2)
    assert w1.isequal(w3)


def test_reduce_row(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    w = A.reduce_rows(monoid.plus).new()
    assert w.isequal(result)
    w2 = A.reduce_rows(binary.plus).new()
    assert w2.isequal(result)


def test_reduce_row_udf(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    binop = grblas.operator.BinaryOp.register_anonymous(lambda x, y: x + y)
    with pytest.raises(DomainMismatch):
        # Although allowed by the spec, SuiteSparse doesn't like user-defined binarops here
        A.reduce_rows(binop).new()
    # If the user creates a monoid from the binop, then we can use the monoid instead
    monoid = grblas.operator.Monoid.register_anonymous(binop, 0)
    w = A.reduce_rows(binop).new()
    assert w.isequal(result)
    w2 = A.reduce_rows(monoid).new()
    assert w2.isequal(result)


def test_reduce_column(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 9, 10, 11, 8, 4])
    w = A.reduce_columns(monoid.plus).new()
    assert w.isequal(result)
    w2 = A.reduce_columns(binary.plus).new()
    assert w2.isequal(result)


def test_reduce_scalar(A):
    s = A.reduce_scalar(monoid.plus).new()
    assert s == 47
    assert A.reduce_scalar(binary.plus).value == 47
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        A.reduce_scalar(binary.minus)

    # test dtype coercion
    assert A.dtype == dtypes.INT64
    s = A.reduce_scalar().new(dtype=float)
    assert s == 47.0
    assert s.dtype == dtypes.FP64
    t = Scalar.new(float)
    t << A.reduce_scalar(monoid.plus)
    assert t == 47.0
    t = Scalar.new(float)
    t() << A.reduce_scalar(monoid.plus)
    assert t == 47.0
    t(accum=binary.times) << A.reduce_scalar(monoid.plus)
    assert t == 47 * 47
    assert A.reduce_scalar(monoid.plus[dtypes.UINT64]).value == 47
    # Make sure we accumulate as a float, not int
    t.value = 1.23
    t(accum=binary.plus) << A.reduce_scalar()
    assert t == 48.23


def test_transpose(A):
    # C << A.T
    rows, cols, vals = A.to_values()
    result = Matrix.from_values(cols, rows, vals)
    C = Matrix.new(A.dtype, A.ncols, A.nrows)
    C << A.T
    assert C.isequal(result)
    C2 = A.T.new()
    assert C2.isequal(result)
    assert A.T.T is A
    C3 = A.T.new(dtype=float)
    assert C3.isequal(result)


def test_kronecker():
    # A  0 1     B  0 1 2
    # 0 [1 -]    0 [- 2 3]
    # 1 [2 3]    1 [8 - 4]
    #
    # C  0  1  2  3  4  5
    # 0 [-  2  3  -  -  - ]
    # 1 [8  -  4  -  -  - ]
    # 2 [-  4  6  -  6  9 ]
    # 3 [16 -  8  24 -  12]
    A = Matrix.from_values([0, 1, 1], [0, 0, 1], [1, 2, 3])
    B = Matrix.from_values([0, 0, 1, 1], [1, 2, 0, 2], [2, 3, 8, 4])
    result = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 2, 0, 2, 1, 2, 4, 5, 0, 2, 3, 5],
        [2, 3, 8, 4, 4, 6, 6, 9, 16, 8, 24, 12],
    )
    C = A.kronecker(B, binary.times).new()
    assert C.isequal(result)


def test_simple_assignment(A):
    # C << A
    C = Matrix.new(A.dtype, A.nrows, A.ncols)
    C << A
    assert C.isequal(A)


def test_assign_transpose(A):
    C = Matrix.new(A.dtype, A.ncols, A.nrows)
    C << A.T
    assert C.isequal(A.T.new())

    with pytest.raises(TypeError):
        C.T << A
    with pytest.raises(TypeError, match="does not support item assignment"):
        C.T[:, :] << A
    with pytest.raises(AttributeError):
        C[:, :].T << A

    C = Matrix.new(A.dtype, A.ncols + 1, A.nrows + 1)
    C[: A.ncols, : A.nrows] << A.T
    assert C[: A.ncols, : A.nrows].new().isequal(A.T.new())


def test_isequal(A, v):
    assert A.isequal(A)
    with pytest.raises(TypeError, match="Matrix"):
        A.isequal(v)  # equality is not type-checking
    C = Matrix.from_values([1], [1], [1])
    assert not C.isequal(A)
    D = Matrix.from_values([1], [2], [1])
    assert not C.isequal(D)
    D2 = Matrix.from_values([0], [2], [1], nrows=D.nrows, ncols=D.ncols)
    assert not D2.isequal(D)
    C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)
    assert not C2.isequal(A)
    C3 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3.0, 2.0, 3.0, 1.0, 5.0, 3.0, 7.0, 8.0, 3.0, 1.0, 7.0, 4.0],
    )
    assert not C3.isequal(A, check_dtype=True), "different datatypes are not equal"
    C4 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3.0, 2.0, 3.0, 1.0, 5.0, 3.000000000000000001, 7.0, 8.0, 3.0, 1 - 1e-11, 7.0, 4.0],
    )
    assert not C4.isequal(A)


@pytest.mark.slow
def test_isclose(A, v):
    assert A.isclose(A)
    with pytest.raises(TypeError, match="Matrix"):
        A.isclose(v)  # equality is not type-checking
    C = Matrix.from_values([1], [1], [1])  # wrong size
    assert not C.isclose(A)
    D = Matrix.from_values([1], [2], [1])
    assert not C.isclose(D)
    D2 = Matrix.from_values([0], [2], [1], nrows=D.nrows, ncols=D.ncols)
    assert not D2.isclose(D)
    C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)  # missing values
    assert not C2.isclose(A)
    C3 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 0],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 2],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 3],
    )  # extra values
    assert not C3.isclose(A)
    C4 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3.0, 2.0, 3.0, 1.0, 5.0, 3.0, 7.0, 8.0, 3.0, 1.0, 7.0, 4.0],
    )
    assert not C4.isclose(A, check_dtype=True), "different datatypes are not equal"
    # fmt: off
    C5 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3.0, 2.0, 3.0, 1.0, 5.0, 3.000000000000000001, 7.0, 8.0, 3.0, 1 - 1e-11, 7.0, 4.0],
    )
    # fmt: on
    assert C5.isclose(A)
    C6 = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3.0, 2.000001, 3.0, 1.0, 5.0, 3.0, 7.0, 7.9999999, 3.0, 1.0, 7.0, 4.0],
    )
    assert C6.isclose(A, rel_tol=1e-3)


@pytest.mark.slow
def test_transpose_equals(A):
    data = [
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
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


def test_transpose_exceptional():
    A = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [True, True, False, True])
    B = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])

    with pytest.raises(TypeError, match="not callable"):
        B.T(mask=A.V) << B.ewise_mult(B, op=binary.plus)
    with pytest.raises(AttributeError):
        B(mask=A.T.V) << B.ewise_mult(B, op=binary.plus)
    with pytest.raises(AttributeError):
        B.T(mask=A.T.V) << B.ewise_mult(B, op=binary.plus)
    with pytest.raises(TypeError, match="does not support item assignment"):
        B.T[1, 0] << 10
    with pytest.raises(TypeError, match="not callable"):
        B.T[1, 0]() << 10
    with pytest.raises(TypeError, match="not callable"):
        B.T()[1, 0] << 10
    with pytest.raises(AttributeError):
        B.T.dup()  # should use new instead
    # Not exceptional, but while we're here...
    C = B.T.new(mask=A.V)
    D = B.T.new()
    D = D.dup(mask=A.V)
    assert C.isequal(D)
    assert C.isequal(Matrix.from_values([0, 0, 1], [0, 1, 1], [1, 3, 4]))


def test_nested_matrix_operations():
    """ Make sure temporaries aren't garbage-collected too soon"""
    A = Matrix.new(int, 8, 8)
    A.ewise_mult(A.mxm(A.T).new()).new().reduce_scalar().new()
    A.ewise_mult(A.ewise_mult(A.ewise_mult(A.ewise_mult(A).new()).new()).new())


def test_bad_init():
    with pytest.raises(TypeError, match="CData"):
        Matrix(None, float, name="bad_matrix")


def test_no_equals(A):
    with pytest.raises(TypeError, match="not defined for objects of type"):
        A == A


def test_bad_update(A):
    with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
        A << None


def test_incompatible_shapes(A):
    B = A[:-1, :-1].new()
    with pytest.raises(DimensionMismatch):
        A.mxm(B)
    with pytest.raises(DimensionMismatch):
        A.ewise_add(B)
    with pytest.raises(DimensionMismatch):
        A.ewise_mult(B)


def test_del(capsys):
    # Exceptions in __del__ are printed to stderr
    import gc

    # shell_A does not have `gb_obj` attribute
    shell_A = Matrix.__new__(Matrix)
    del shell_A
    # A has `gb_obj` of NULL
    A = Matrix.from_values([0, 1], [0, 1], [0, 1])
    gb_obj = A.gb_obj
    A.gb_obj = grblas.ffi.NULL
    del A
    # let's clean up so we don't have a memory leak
    A2 = Matrix.__new__(Matrix)
    A2.gb_obj = gb_obj
    del A2
    gc.collect()
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


def test_import_export(A):
    A1 = A.dup()
    d = A1.ss.export("csr", give_ownership=True)
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert (d["indptr"] == [0, 2, 4, 5, 7, 8, 9, 12]).all()
    assert (d["col_indices"] == [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]).all()
    assert (d["values"] == [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3]).all()
    B1 = Matrix.ss.import_any(take_ownership=True, **d)
    assert B1.isequal(A)

    A2 = A.dup()
    d = A2.ss.export("csc")
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert (d["indptr"] == [0, 1, 2, 5, 7, 9, 11, 12]).all()
    assert (d["row_indices"] == [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1]).all()
    assert (d["values"] == [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]).all()
    B2 = Matrix.ss.import_any(**d)
    assert B2.isequal(A)

    A3 = A.dup()
    d = A3.ss.export("hypercsr")
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert (d["rows"] == [0, 1, 2, 3, 4, 5, 6]).all()
    assert (d["indptr"] == [0, 2, 4, 5, 7, 8, 9, 12]).all()
    assert (d["col_indices"] == [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]).all()
    assert (d["values"] == [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3]).all()
    B3 = Matrix.ss.import_any(**d)
    assert B3.isequal(A)

    A4 = A.dup()
    d = A4.ss.export("hypercsc")
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert (d["cols"] == [0, 1, 2, 3, 4, 5, 6]).all()
    assert (d["indptr"] == [0, 1, 2, 5, 7, 9, 11, 12]).all()
    assert (d["row_indices"] == [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1]).all()
    assert (d["values"] == [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]).all()
    B4 = Matrix.ss.import_any(**d)
    assert B4.isequal(A)

    A5 = A.dup()
    d = A5.ss.export("bitmapr")
    assert "nrows" not in d
    assert "ncols" not in d
    assert d["values"].shape == (7, 7)
    assert d["bitmap"].shape == (7, 7)
    assert d["nvals"] == 12
    bitmap = np.array(
        [
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]
    )
    assert (d["bitmap"] == bitmap).all()
    assert (
        d["values"].ravel("K")[d["bitmap"].ravel("K")] == [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3]
    ).all()
    del d["nvals"]
    B5 = Matrix.ss.import_any(**d)
    assert B5.isequal(A)
    d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=0)
    B5b = Matrix.ss.import_any(**d)
    assert B5b.isequal(A)

    A6 = A.dup()
    d = A6.ss.export("bitmapc")
    assert (d["bitmap"] == bitmap).all()
    assert (
        d["values"].ravel("K")[d["bitmap"].ravel("K")] == [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
    ).all()
    del d["nvals"]
    B6 = Matrix.ss.import_any(nrows=7, **d)
    assert B6.isequal(A)
    d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=1)
    B6b = Matrix.ss.import_any(ncols=7, **d)
    assert B6b.isequal(A)

    A7 = A.dup()
    d = A7.ss.export()
    B7 = Matrix.ss.import_any(**d)
    assert B7.isequal(A)

    A8 = A.dup()
    d = A8.ss.export("bitmapr", raw=True)
    del d["nrows"]
    del d["ncols"]
    with pytest.raises(ValueError, match="nrows and ncols must be provided"):
        Matrix.ss.import_any(**d)

    C = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    C1 = C.dup()
    d = C1.ss.export("fullr")
    assert "nrows" not in d
    assert "ncols" not in d
    assert d["values"].shape == (2, 2)
    assert (d["values"] == [[1, 2], [3, 4]]).all()
    assert d["values"].flags.c_contiguous
    D1 = Matrix.ss.import_any(ncols=2, **d)
    assert D1.isequal(C)

    C2 = C.dup()
    d = C2.ss.export("fullc")
    assert (d["values"] == [[1, 2], [3, 4]]).all()
    assert d["values"].flags.f_contiguous
    D2 = Matrix.ss.import_any(**d)
    assert D2.isequal(C)

    # all elements must have values
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullr")
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullc")

    a = np.array([0, 1, 2])
    for bad_combos in [
        ["indptr", "bitmap"],
        ["indptr"],
        ["indptr", "row_indices", "col_indices"],
        ["indptr", "rows", "cols"],
        ["indptr", "col_indices", "rows", "cols"],
        ["indptr", "rows"],
        ["indptr", "cols"],
        ["indptr", "row_indices", "rows"],
        ["indptr", "col_indices", "cols"],
        ["bitmap", "col_indices"],
        ["bitmap", "row_indices"],
        ["bitmap", "rows"],
        ["bitmap", "cols"],
    ]:
        with pytest.raises(TypeError):
            Matrix.ss.import_any(nrows=3, ncols=3, values=a, **dict.fromkeys(bad_combos, a))


def test_import_on_view():
    A = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    B = Matrix.ss.import_any(nrows=2, ncols=2, values=np.array([1, 2, 3, 4, 99, 99, 99])[:4])
    assert A.isequal(B)


def test_import_export_empty():
    A = Matrix.new(int, 2, 3)
    A1 = A.dup()
    d = A1.ss.export("csr")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert (d["indptr"] == [0, 0, 0]).all()
    assert len(d["col_indices"]) == 0
    assert len(d["values"]) == 0
    B1 = Matrix.ss.import_any(**d)
    assert B1.isequal(A)

    A2 = A.dup()
    d = A2.ss.export("csc")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert (d["indptr"] == [0, 0, 0, 0]).all()
    assert len(d["row_indices"]) == 0
    assert len(d["values"]) == 0
    B2 = Matrix.ss.import_any(**d)
    assert B2.isequal(A)

    A3 = A.dup()
    d = A3.ss.export("hypercsr")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["indptr"]) == 1
    assert d["indptr"][0] == 0
    assert len(d["col_indices"]) == 0
    assert len(d["values"]) == 0
    assert len(d["rows"]) == 0
    B3 = Matrix.ss.import_any(**d)
    assert B3.isequal(A)

    A4 = A.dup()
    d = A4.ss.export("hypercsc")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["indptr"]) == 1
    assert d["indptr"][0] == 0
    assert len(d["row_indices"]) == 0
    assert len(d["values"]) == 0
    assert len(d["cols"]) == 0
    B4 = Matrix.ss.import_any(**d)
    assert B4.isequal(A)

    A5 = A.dup()
    d = A5.ss.export("bitmapr")
    assert d["bitmap"].shape == (2, 3)
    assert d["bitmap"].flags.c_contiguous
    assert d["nvals"] == 0
    assert (d["bitmap"].ravel() == 6 * [0]).all()
    B5 = Matrix.ss.import_any(**d)
    assert B5.isequal(A)

    A6 = A.dup()
    d = A6.ss.export("bitmapc")
    assert d["bitmap"].shape == (2, 3)
    assert d["bitmap"].flags.f_contiguous
    assert d["nvals"] == 0
    assert (d["bitmap"].ravel() == 6 * [0]).all()
    B6 = Matrix.ss.import_any(**d)
    assert B6.isequal(A)

    # all elements must have values
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullr")
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullc")

    # if we give the same value, make sure it's copied
    for format, key1, key2 in [
        ("csr", "values", "col_indices"),
        ("hypercsr", "values", "col_indices"),
        ("csc", "values", "row_indices"),
        ("hypercsc", "values", "row_indices"),
        ("bitmapr", "values", "bitmap"),
        ("bitmapc", "values", "bitmap"),
    ]:
        # No assertions here, but code coverage should be "good enough"
        d = A.ss.export(format, raw=True)
        d[key1] = d[key2]
        Matrix.ss.import_any(take_ownership=True, **d)

    with pytest.raises(ValueError, match="Invalid format"):
        A.ss.export(format="bad_format")


def test_import_export_auto(A):
    A_orig = A.dup()
    for format in ["csr", "csc", "hypercsr", "hypercsc", "bitmapr", "bitmapc"]:
        for (
            sort,
            raw,
            import_format,
            give_ownership,
            take_ownership,
            import_func,
        ) in itertools.product(
            [False, True],
            [False, True],
            [format, None],
            [False, True],
            [False, True],
            [Matrix.ss.import_any, getattr(Matrix.ss, f"import_{format}")],
        ):
            A2 = A.dup() if give_ownership else A
            d = A2.ss.export(format, sort=sort, raw=raw, give_ownership=give_ownership)
            d["format"] = import_format
            other = import_func(take_ownership=take_ownership, **d)
            if (
                format == "bitmapc"
                and raw
                and import_format is None
                and import_func.__name__ == "import_any"
            ):
                # It's 1d, so we can't tell we're column-oriented w/o format keyword
                assert other.isequal(A_orig.T)
            else:
                assert other.isequal(A_orig)
            d["format"] = "bad_format"
            with pytest.raises(ValueError, match="Invalid format"):
                import_func(**d)
    assert A.isequal(A_orig)

    C = Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6])
    C_orig = C.dup()
    for format in ["fullr", "fullc"]:
        for raw, import_format, give_ownership, take_ownership, import_func in itertools.product(
            [False, True],
            [format, None],
            [False, True],
            [False, True],
            [Matrix.ss.import_any, getattr(Matrix.ss, f"import_{format}")],
        ):
            C2 = C.dup() if give_ownership else C
            d = C2.ss.export(format, raw=raw, give_ownership=give_ownership)
            d["format"] = import_format
            other = import_func(take_ownership=take_ownership, **d)
            if (
                format == "fullc"
                and raw
                and import_format is None
                and import_func.__name__ == "import_any"
            ):
                # It's 1d, so we can't tell we're column-oriented w/o format keyword
                assert other.isequal(
                    Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 3, 5, 2, 4, 6])
                )
            else:
                assert other.isequal(C_orig)
            d["format"] = "bad_format"
            with pytest.raises(ValueError, match="Invalid format"):
                import_func(**d)
    assert C.isequal(C_orig)


def test_no_bool_or_eq(A):
    with pytest.raises(TypeError, match="not defined"):
        bool(A)
    with pytest.raises(TypeError, match="not defined"):
        A == A
    with pytest.raises(TypeError, match="not defined"):
        bool(A.S)
    with pytest.raises(TypeError, match="not defined"):
        A.S == A.S
    expr = A.ewise_mult(A)
    with pytest.raises(TypeError, match="not defined"):
        bool(expr)
    with pytest.raises(TypeError, match="not defined"):
        expr == expr
    assigner = A[1, 2]()
    with pytest.raises(TypeError, match="not defined"):
        bool(assigner)
    with pytest.raises(TypeError, match="not defined"):
        assigner == assigner
    updater = A()
    with pytest.raises(TypeError, match="not defined"):
        bool(updater)
    with pytest.raises(TypeError, match="not defined"):
        updater == updater


def test_bool_eq_on_scalar_expressions(A):
    expr = A.reduce_scalar()
    assert expr == 47
    assert bool(expr)
    assert int(expr) == 47
    assert float(expr) == 47.0
    assert range(expr) == range(47)

    expr = A[0, 1]
    assert expr == 2
    assert bool(expr)
    assert int(expr) == 2
    assert float(expr) == 2.0
    assert range(expr) == range(2)

    expr = A[0, [1, 1]]
    with pytest.raises(TypeError, match="not defined"):
        expr == expr
    with pytest.raises(TypeError, match="not defined"):
        bool(expr)
    with pytest.raises(TypeError, match="not defined"):
        int(expr)
    with pytest.raises(TypeError, match="not defined"):
        float(expr)
    with pytest.raises(TypeError, match="not defined"):
        range(expr)


def test_contains(A):
    assert (0, 1) in A
    assert (1, 0) in A.T

    assert (0, 1) not in A.T
    assert (1, 0) not in A

    with pytest.raises(TypeError):
        1 in A
    with pytest.raises(TypeError):
        (1,) in A.T
    with pytest.raises(TypeError, match="Invalid index"):
        (1, [1, 2]) in A


def test_iter(A):
    assert set(A) == set(
        zip(
            [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
            [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        )
    )
    assert set(A.T) == set(
        zip(
            [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        )
    )


def test_wait(A):
    A2 = A.dup()
    A2.wait()
    assert A2.isequal(A)


def test_pickle(A):
    s = pickle.dumps(A)
    A2 = pickle.loads(s)
    assert A.isequal(A2, check_dtype=True)
    assert A.name == A2.name


def test_weakref(A):
    d = weakref.WeakValueDictionary()
    d["A"] = A
    assert d["A"] is A
    AT = A.T
    d["A.T"] = AT
    assert d["A.T"] is AT
    expr = A.mxm(A)
    d["expr"] = expr
    assert d["expr"] is expr


def test_not_to_array(A):
    with pytest.raises(TypeError, match="Matrix can't be directly converted to a numpy array"):
        np.array(A)
    with pytest.raises(
        TypeError, match="TransposedMatrix can't be directly converted to a numpy array"
    ):
        np.array(A.T)

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
    NotImplementedException,
    OutputNotEmpty,
)

from .conftest import autocompute, compute

from graphblas import Matrix, Scalar, Vector  # isort:skip (for dask-graphblas)


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
    C = Matrix(dtypes.INT8, 17, 12)
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
    assert C[0, 0].new() != 1000
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
    assert C3[1, 1].new() == 6  # 2*3
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
    C5 = Matrix(dtypes.INT64, nrows=3, ncols=4)
    assert C4.isequal(C5, check_dtype=True)

    with pytest.raises(
        ValueError, match="`rows` and `columns` and `values` lengths must match: 1, 2, 1"
    ):
        Matrix.from_values([0], [1, 2], [0])


def test_from_values_scalar():
    C = Matrix.from_values([0, 1, 3], [1, 1, 2], 7)
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == dtypes.INT64
    assert C.ss.is_iso
    assert C.ss.iso_value == 7
    assert C.reduce_scalar(monoid.any).new() == 7

    # iso drumps duplicates
    C = Matrix.from_values([0, 1, 3, 0], [1, 1, 2, 1], 7)
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == dtypes.INT64
    assert C.ss.is_iso
    assert C.ss.iso_value == 7
    assert C.reduce_scalar(monoid.any).new() == 7
    with pytest.raises(ValueError, match="dup_op must be None"):
        Matrix.from_values([0, 1, 3, 0], [1, 1, 2, 1], 7, dup_op=binary.plus)
    C[0, 0] = 0
    with pytest.raises(ValueError, match="not iso"):
        C.ss.iso_value


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
    assert compute(A[9, 9].new().value) is None
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
    B = Matrix(int, nrows=2, ncols=2)
    B.build([0, 11], [0, 0], [1, 1], nrows=12)
    assert B.isequal(Matrix.from_values([0, 11], [0, 0], [1, 1], ncols=2))
    C = Matrix(int, nrows=2, ncols=2)
    C.build([0, 0], [0, 11], [1, 1], ncols=12)
    assert C.isequal(Matrix.from_values([0, 0], [0, 11], [1, 1], nrows=2))


def test_build_scalar(A):
    assert A.nvals == 12
    with pytest.raises(OutputNotEmpty):
        A.ss.build_scalar([1, 5], [2, 3], 3)
    A.clear()
    A.ss.build_scalar([0, 6], [0, 1], 1)
    assert A.nvals == 2
    assert A.ss.is_iso
    A.clear()
    with pytest.raises(ValueError, match="lengths must match"):
        A.ss.build_scalar([0, 6], [0, 1, 2], 1)
    with pytest.raises(EmptyObject):
        A.ss.build_scalar([0, 5], [0, 1], None)


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
    assert A[1, 6].new() == 4
    with pytest.raises(TypeError, match="enable automatic"):
        A[1, 6].value
    assert A.T[6, 1].new() == 4
    s = A[0, 0].new()
    assert compute(s.value) is None
    assert s.dtype == "INT64"
    s = A[1, 6].new(dtype=float)
    assert s.value == 4.0
    assert s.dtype == "FP64"


def test_set_element(A):
    assert compute(A[1, 1].new().value) is None
    assert A[3, 0].new() == 3
    A[1, 1].update(21)
    A[3, 0] << -5
    assert A[1, 1].new() == 21
    assert A[3, 0].new() == -5


def test_remove_element(A):
    assert A[3, 0].new() == 3
    del A[3, 0]
    assert compute(A[3, 0].new().value) is None
    assert A[6, 3].new() == 7
    # with pytest.raises(TypeError, match="Remove Element only supports"):
    del A[3:5, 3]  # Now okay


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
    C = Matrix(A.dtype, nrows=1, ncols=1)
    C << A.mxm(B, semiring.max_plus)
    assert C[0, 0].new() == 33
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
    with pytest.raises(TypeError, match="Mask must be"):
        A.mxm(A).new(mask=struct_mask)  # would be okay if bool mask, but it's not


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
    # with pytest.raises(TypeError, match="require_monoid"):
    A.ewise_add(B, binary.second)  # okay now
    # surprising that SECOND(x, empty) == x
    C = A.ewise_add(B, binary.second).new()
    assert C.isequal(result)
    C << A.ewise_add(B, monoid.max)
    assert C.isequal(result)
    C << A.ewise_add(B, binary.max)
    assert C.isequal(result)
    with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
        A.ewise_add(B, semiring.max_minus)


def test_extract(A):
    C = Matrix(A.dtype, 3, 4)
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
    w = Vector(A.dtype, 3)
    result = Vector.from_values([1, 2], [5, 3], size=3)
    w << A[6, [0, 2, 4]]
    assert w.isequal(result)
    w << A[6, :5:2]
    assert w.isequal(result)
    w << A.T[[0, 2, 4], 6]
    assert w.isequal(result)
    w2 = A[6, [0, 2, 4]].new()
    assert w2.isequal(result)
    with pytest.raises(TypeError):
        # Should be list, not tuple (although tuple isn't so bad)
        A[6, (0, 2, 4)]
    w3 = A[6, np.array([0, 2, 4])].new()
    assert w3.isequal(result)
    with pytest.raises(TypeError, match="Invalid dtype"):
        A[6, np.array([0, 2, 4], dtype=float)]
    with pytest.raises(TypeError, match="Invalid number of dimensions"):
        A[6, np.array([[0, 2, 4]])]


def test_extract_column(A):
    w = Vector(A.dtype, 3)
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
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        A[0, 0].new(input_mask=M.S)
    with pytest.raises(TypeError, match="mask and input_mask arguments cannot both be given"):
        A[0, [0, 1]].new(input_mask=M.S, mask=expected.S)
    with pytest.raises(TypeError, match="mask and input_mask arguments cannot both be given"):
        A(input_mask=M.S, mask=expected.S)
    with pytest.raises(TypeError, match="Mask must be"):
        A[0, [0, 1]].new(input_mask=M)
    with pytest.raises(TypeError, match="Mask must be"):
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
    nvals = C.nvals
    C(C.S) << 1
    assert C.nvals == nvals
    assert C.reduce_scalar().new() == nvals
    with pytest.raises(TypeError, match="Invalid type for index"):
        C[C, [1]] = C
    C << 1  # Now okay
    assert C.nvals == C.nrows * C.ncols


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
    B = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], 1)
    B[1, 1] = Scalar(B.dtype)
    expected = Matrix.from_values([0, 0, 1], [0, 1, 0], 1)
    assert B.isequal(expected)


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


def test_apply_indexunary(A):
    ridx = [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1]
    cidx = [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    Ar = Matrix.from_values(ridx, cidx, ridx)
    r1 = A.apply("rowindex").new()
    r2 = A.apply(indexunary.rowindex).new()
    r3 = indexunary.rowindex(A).new()
    assert r1.isequal(Ar)
    assert r2.isequal(Ar)
    assert r3.isequal(Ar)

    Ac = Matrix.from_values(ridx, cidx, [c + 2 for c in cidx])
    c1 = A.apply("colindex", 2).new()
    c2 = A.apply(indexunary.colindex, 2).new()
    c3 = indexunary.colindex(A, thunk=2).new()
    assert c1.isequal(Ac)
    assert c2.isequal(Ac)
    assert c3.isequal(Ac)

    A3 = Matrix.from_values(ridx, cidx, [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    s3 = Scalar.from_value(3, dtypes.INT64)
    w1 = A.apply(indexunary.valueeq, s3).new()
    w2 = A.apply(select.valueeq, s3).new()
    w3 = A.apply("==", s3).new()
    w4 = indexunary.valueeq(A, s3).new()
    assert w1.isequal(A3)
    assert w2.isequal(A3)
    assert w3.isequal(A3)
    assert w4.isequal(A3)
    with pytest.raises(TypeError, match="left"):
        A.apply(select.valueeq, left=s3)


def test_select(A):
    A3 = Matrix.from_values([0, 3, 3, 6], [3, 0, 2, 4], [3, 3, 3, 3], nrows=7, ncols=7)
    w1 = A.select(select.valueeq, 3).new()
    w1b = A.select(indexunary.valueeq, 3).new()
    w2 = A.select("==", 3).new()
    w3 = select.value(A == 3).new()
    assert w1.isequal(A3)
    assert w1b.isequal(A3)
    assert w2.isequal(A3)
    assert w3.isequal(A3)

    A2cols = Matrix.from_values([3, 0, 3, 5, 6], [0, 1, 2, 2, 2], [3, 2, 3, 1, 5], nrows=7, ncols=7)
    w4 = select.colle(A, 2).new()
    w5 = A.select("col<=", 2).new()
    w6 = select.column(A < 3).new()
    assert w4.isequal(A2cols)
    assert w5.isequal(A2cols)
    assert w6.isequal(A2cols)

    Aupper = Matrix.from_values(
        [0, 0, 1, 2, 4, 1], [1, 3, 4, 5, 5, 6], [2, 3, 8, 1, 7, 4], nrows=7, ncols=7
    )
    w7 = A.select("TRIU").new()
    assert w7.isequal(Aupper)
    with pytest.raises(TypeError, match="thunk"):
        A.select(select.valueeq, object())


@autocompute
def test_select_bools_and_masks(A):
    A3 = Matrix.from_values([0, 3, 3, 6], [3, 0, 2, 4], [3, 3, 3, 3], nrows=7, ncols=7)
    # Select with boolean and masks
    w8 = A.select((A == 3).new()).new()
    assert w8.isequal(A3)
    w8b = A.select(A == 3).new()  # we rewrite!
    assert w8b.isequal(A3)
    w8c = A.select(~(A != 3)).new()
    assert w8c.isequal(A3)
    w9 = A.select(w8.S).new()
    assert w9.isequal(A3)
    w8[0, 1] = 0
    w10 = A.select(w8.V).new()
    assert w10.isequal(A3)
    with pytest.raises(TypeError, match="thunk"):
        A.select(w8.V, 777)
    with pytest.raises(TypeError, match="thunk"):
        A.select(A == 3, 777)
    with pytest.raises(TypeError):
        A.select(A[0, :].new().S)


@pytest.mark.slow
def test_indexunary_udf(A):
    def threex_minusthunk(x, row, col, thunk):  # pragma: no cover
        return 3 * x - thunk

    indexunary.register_new("threex_minusthunk", threex_minusthunk)
    assert hasattr(indexunary, "threex_minusthunk")
    assert not hasattr(select, "threex_minusthunk")
    with pytest.raises(ValueError):
        select.register_anonymous(threex_minusthunk)
    expected = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [5, 2, 5, -1, 11, 5, 17, 20, 5, -1, 17, 8],
    )
    result = indexunary.threex_minusthunk(A, 4).new()
    assert result.isequal(expected)
    delattr(indexunary, "threex_minusthunk")

    def iii(x, row, col, thunk):  # pragma: no cover
        return (row + col) // 2 >= thunk

    select.register_new("iii", iii)
    assert hasattr(indexunary, "iii")
    assert hasattr(select, "iii")
    iii_apply = indexunary.register_anonymous(iii)
    expected = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [False, False, True, True, True, False, True, True, True, True, True, True],
    )
    result = iii_apply(A, 2).new()
    assert result.isequal(expected)
    iii_select = select.register_anonymous(iii)
    expected = Matrix.from_values(
        [3, 5, 6, 6, 1, 6, 2, 4, 1], [2, 2, 2, 3, 4, 4, 5, 5, 6], [3, 1, 5, 7, 8, 3, 1, 7, 4]
    )
    result = iii_select(A, 2).new()
    assert result.isequal(expected)
    delattr(indexunary, "iii")
    delattr(select, "iii")


def test_reduce_row(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    w = A.reduce_rowwise(monoid.plus).new()
    assert w.isequal(result)
    w2 = A.reduce_rowwise(binary.plus).new()
    assert w2.isequal(result)


@pytest.mark.slow
def test_reduce_agg(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    w1 = A.reduce_rowwise(agg.sum).new()
    assert w1.isequal(result)
    w2 = A.T.reduce_columnwise(agg.sum).new()
    assert w2.isequal(result)

    counts = A.dup(dtype=bool).reduce_rowwise(monoid.plus[int]).new()
    w3 = A.reduce_rowwise(agg.count).new()
    assert w3.isequal(counts)
    w4 = A.T.reduce_columnwise(agg.count).new()
    assert w4.isequal(counts)

    Asquared = monoid.times(A & A).new()
    squared = Asquared.reduce_rowwise(monoid.plus).new()
    expected = unary.sqrt[float](squared).new()
    w5 = A.reduce_rowwise(agg.hypot).new()
    assert w5.isclose(expected)
    w6 = A.reduce_rowwise(monoid.numpy.hypot[float]).new()
    assert w6.isclose(expected)
    w7 = Vector(w5.dtype, size=w5.size)
    w7 << A.reduce_rowwise(agg.hypot)
    assert w7.isclose(expected)

    w8 = A.reduce_rowwise(agg.logaddexp).new()
    expected = A.reduce_rowwise(monoid.numpy.logaddexp[float]).new()
    assert w8.isclose(w8)

    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 9, 10, 11, 8, 4])
    w9 = A.reduce_columnwise(agg.sum).new()
    assert w9.isequal(result)
    w10 = A.T.reduce_rowwise(agg.sum).new()
    assert w10.isequal(result)

    counts = A.dup(dtype=bool).reduce_columnwise(monoid.plus[int]).new()
    w11 = A.reduce_columnwise(agg.count).new()
    assert w11.isequal(counts)
    w12 = A.T.reduce_rowwise(agg.count).new()
    assert w12.isequal(counts)

    w13 = A.reduce_rowwise(agg.mean).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [2.5, 6, 1, 3, 7, 1, 5])
    assert w13.isequal(expected)
    w14 = A.reduce_columnwise(agg.mean).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 3, 5, 5.5, 4, 4])
    assert w14.isequal(expected)

    w15 = A.reduce_rowwise(agg.exists).new()
    w16 = A.reduce_columnwise(agg.exists).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1, 1])
    assert w15.isequal(expected)
    assert w16.isequal(expected)

    assert A.reduce_scalar(agg.sum).new() == 47
    assert A.reduce_scalar(agg.prod).new() == 1270080
    assert A.reduce_scalar(agg.count).new() == 12
    assert A.reduce_scalar(agg.count_nonzero).new() == 12
    assert A.reduce_scalar(agg.count_zero).new() == 0
    assert A.reduce_scalar(agg.sum_of_squares).new() == 245
    assert A.reduce_scalar(agg.hypot).new().isclose(245**0.5)
    assert A.reduce_scalar(agg.logaddexp).new().isclose(8.6071076)
    assert A.reduce_scalar(agg.logaddexp2).new().isclose(9.2288187)
    assert A.reduce_scalar(agg.mean).new().isclose(47 / 12)
    assert A.reduce_scalar(agg.exists).new() == 1

    silly = agg.Aggregator(
        "silly",
        composite=[agg.varp, agg.stdp],
        finalize=lambda x, y: binary.times(x & y),
        types=[agg.varp],
    )
    v1 = A.reduce_rowwise(agg.varp).new()
    v2 = A.reduce_rowwise(agg.stdp).new()
    assert v1.isclose(binary.times(v2 & v2).new())
    v3 = A.reduce_rowwise(silly).new()
    assert v3.isclose(binary.times(v1 & v2).new())

    s1 = A.reduce_scalar(agg.varp).new()
    s2 = A.reduce_scalar(agg.stdp).new()
    assert s1.isclose(s2.value * s2.value)
    s3 = A.reduce_scalar(silly).new()
    assert s3.isclose(s1.value * s2.value)

    B = Matrix(int, nrows=4, ncols=5)
    assert B.reduce_scalar(agg.sum, allow_empty=True).new().is_empty
    assert B.reduce_scalar(agg.sum, allow_empty=False).new() == 0
    assert B.reduce_scalar(agg.vars, allow_empty=True).new().is_empty
    with pytest.raises(ValueError):
        B.reduce_scalar(agg.vars, allow_empty=False)


def test_reduce_agg_argminmax(A):
    # reduce_rowwise
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 6, 5, 0, 5, 2, 4])
    w1b = A.reduce_rowwise(agg.argmin).new()
    assert w1b.isequal(expected)
    w1c = A.T.reduce_columnwise(agg.argmin).new()
    assert w1c.isequal(expected)
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 4, 5, 0, 5, 2, 3])
    w2b = A.reduce_rowwise(agg.argmax).new()
    assert w2b.isequal(expected)
    w2c = A.T.reduce_columnwise(agg.argmax).new()
    assert w2c.isequal(expected)

    # reduce_cols
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 5, 0, 6, 2, 1])
    w7b = A.reduce_columnwise(agg.argmin).new()
    assert w7b.isequal(expected)
    w7c = A.T.reduce_rowwise(agg.argmin).new()
    assert w7c.isequal(expected)
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 6, 6, 1, 4, 1])
    w8b = A.reduce_columnwise(agg.argmax).new()
    assert w8b.isequal(expected)
    w8c = A.T.reduce_rowwise(agg.argmax).new()
    assert w8c.isequal(expected)

    # reduce_scalar
    with pytest.raises(
        ValueError, match="Aggregator argmin may not be used with Matrix.reduce_scalar"
    ):
        A.reduce_scalar(agg.argmin)

    silly = agg.Aggregator(
        "silly",
        composite=[agg.argmin, agg.argmax],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.argmin],
    )
    v1 = A.reduce_rowwise(agg.argmin).new()
    v2 = A.reduce_rowwise(agg.argmax).new()
    v3 = A.reduce_rowwise(silly).new()
    assert v3.isequal(binary.plus(v1 & v2).new())

    v1 = A.reduce_columnwise(agg.argmin).new()
    v2 = A.reduce_columnwise(agg.argmax).new()
    v3 = A.reduce_columnwise(silly).new()
    assert v3.isequal(binary.plus(v1 & v2).new())

    with pytest.raises(ValueError, match="Aggregator"):
        A.reduce_scalar(silly).new()


def test_reduce_agg_firstlast(A):
    # reduce_rowwise
    w1 = A.reduce_rowwise(agg.first).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [2, 8, 1, 3, 7, 1, 5])
    assert w1.isequal(expected)
    w1b = A.T.reduce_columnwise(agg.first).new()
    assert w1b.isequal(expected)
    w2 = A.reduce_rowwise(agg.last).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 4, 1, 3, 7, 1, 3])
    assert w2.isequal(expected)
    w2b = A.T.reduce_columnwise(agg.last).new()
    assert w2b.isequal(expected)

    # reduce_columnwise
    w3 = A.reduce_columnwise(agg.first).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 3, 3, 8, 1, 4])
    assert w3.isequal(expected)
    w3b = A.T.reduce_rowwise(agg.first).new()
    assert w3b.isequal(expected)
    w4 = A.reduce_columnwise(agg.last).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 5, 7, 3, 7, 4])
    assert w4.isequal(expected)
    w4b = A.T.reduce_rowwise(agg.last).new()
    assert w4b.isequal(expected)

    # reduce_scalar
    w5 = A.reduce_scalar(agg.first).new()
    assert w5 == 2
    w6 = A.reduce_scalar(agg.last).new()
    assert w6 == 3
    B = Matrix(float, nrows=2, ncols=3)
    assert B.reduce_scalar(agg.first).new().is_empty
    assert B.reduce_scalar(agg.last).new().is_empty
    w7 = B.reduce_rowwise(agg.first).new()
    assert w7.isequal(Vector(float, size=B.nrows))
    w8 = B.reduce_columnwise(agg.last).new()
    assert w8.isequal(Vector(float, size=B.ncols))

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first, agg.last],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first],
    )
    v1 = A.reduce_rowwise(agg.first).new()
    v2 = A.reduce_rowwise(agg.last).new()
    v3 = A.reduce_rowwise(silly).new()
    assert v3.isequal(binary.plus(v1 & v2).new())

    s1 = A.reduce_scalar(agg.first).new()
    s2 = A.reduce_scalar(agg.last).new()
    s3 = A.reduce_scalar(silly).new()
    assert s3.isequal(s1.value + s2.value)


def test_reduce_agg_firstlast_index(A):
    # reduce_rowwise
    w1 = A.reduce_rowwise(agg.first_index).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 4, 5, 0, 5, 2, 2])
    assert w1.isequal(expected)
    w1b = A.T.reduce_columnwise(agg.first_index).new()
    assert w1b.isequal(expected)
    w2 = A.reduce_rowwise(agg.last_index).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 6, 5, 2, 5, 2, 4])
    assert w2.isequal(expected)
    w2b = A.T.reduce_columnwise(agg.last_index).new()
    assert w2b.isequal(expected)

    # reduce_columnwise
    w3 = A.reduce_columnwise(agg.first_index).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 3, 0, 1, 2, 1])
    assert w3.isequal(expected)
    w3b = A.T.reduce_rowwise(agg.first_index).new()
    assert w3b.isequal(expected)
    w4 = A.reduce_columnwise(agg.last_index).new()
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 6, 6, 6, 4, 1])
    assert w4.isequal(expected)
    w4b = A.T.reduce_rowwise(agg.last_index).new()
    assert w4b.isequal(expected)

    # reduce_scalar
    with pytest.raises(ValueError, match="Aggregator first_index may not"):
        A.reduce_scalar(agg.first_index).new()
    with pytest.raises(ValueError, match="Aggregator last_index may not"):
        A.reduce_scalar(agg.last_index).new()

    silly = agg.Aggregator(
        "silly",
        composite=[agg.first_index, agg.last_index],
        finalize=lambda x, y: binary.plus(x & y),
        types=[agg.first_index],
    )
    v1 = A.reduce_rowwise(agg.first_index).new()
    v2 = A.reduce_rowwise(agg.last_index).new()
    v3 = A.reduce_rowwise(silly).new()
    assert v3.isequal(binary.plus(v1 & v2).new())

    with pytest.raises(ValueError, match="Aggregator"):
        A.reduce_scalar(silly).new()


def test_reduce_agg_empty():
    A = Matrix("UINT8", nrows=3, ncols=4)
    for B in [A, A.T]:
        ve = Vector(bool, size=B.nrows)
        we = Vector(bool, size=B.ncols)
        for attr, aggr in vars(agg).items():
            if not isinstance(aggr, agg.Aggregator):
                continue
            v = B.reduce_rowwise(aggr).new()
            assert ve.isequal(v)
            w = B.reduce_columnwise(aggr).new()
            assert we.isequal(w)
            if attr not in {"argmin", "argmax", "first_index", "last_index"}:
                s = B.reduce_scalar(aggr).new()
                assert compute(s.value) is None


def test_reduce_row_udf(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])

    def plus(x, y):  # pragma: no cover
        return x + y

    binop = graphblas.operator.BinaryOp.register_anonymous(plus)
    with pytest.raises(NotImplementedException):
        # Although allowed by the spec, SuiteSparse doesn't like user-defined binarops here
        A.reduce_rowwise(binop).new()
    # If the user creates a monoid from the binop, then we can use the monoid instead
    monoid = graphblas.operator.Monoid.register_anonymous(binop, 0)
    w = A.reduce_rowwise(binop).new()
    assert w.isequal(result)
    w2 = A.reduce_rowwise(monoid).new()
    assert w2.isequal(result)


def test_reduce_column(A):
    result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 9, 10, 11, 8, 4])
    w = A.reduce_columnwise(monoid.plus).new()
    assert w.isequal(result)
    w2 = A.reduce_columnwise(binary.plus).new()
    assert w2.isequal(result)


def test_reduce_scalar(A):
    s = A.reduce_scalar(monoid.plus).new()
    assert s == 47
    assert A.reduce_scalar(binary.plus).new() == 47
    with pytest.raises(TypeError, match="Expected type: Monoid"):
        A.reduce_scalar(binary.minus)

    # test dtype coercion
    assert A.dtype == dtypes.INT64
    s = A.reduce_scalar().new(dtype=float)
    assert s == 47.0
    assert s.dtype == dtypes.FP64
    t = Scalar(float)
    t << A.reduce_scalar(monoid.plus)
    assert t == 47.0
    t = Scalar(float)
    t() << A.reduce_scalar(monoid.plus)
    assert t == 47.0
    t(accum=binary.times) << A.reduce_scalar(monoid.plus)
    assert t == 47 * 47
    assert A.reduce_scalar(monoid.plus[dtypes.UINT64]).new() == 47
    # Make sure we accumulate as a float, not int
    t.value = 1.23
    t(accum=binary.plus) << A.reduce_scalar()
    assert t == 48.23


@autocompute
def test_reduce_call_agg(A):
    assert agg.sum(A) == 47
    assert agg.sum(A.T) == 47
    assert agg.sum(A + A) == 94  # handles expression
    result = agg.max[float](A).new()  # typed agg is callable too
    assert result.dtype == "FP64"
    assert result == 8
    expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
    result = agg.sum(A, rowwise=True)
    assert result.isequal(expected)
    result = agg.sum(A.T, columnwise=True)
    assert result.isequal(expected)
    with pytest.raises(ValueError, match="cannot both be True"):
        agg.sum(A, rowwise=True, columnwise=True)


def test_transpose(A):
    # C << A.T
    rows, cols, vals = A.to_values()
    result = Matrix.from_values(cols, rows, vals)
    C = Matrix(A.dtype, A.ncols, A.nrows)
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
    C = Matrix(A.dtype, A.nrows, A.ncols)
    C << A
    assert C.isequal(A)


def test_assign_transpose(A):
    C = Matrix(A.dtype, A.ncols, A.nrows)
    C << A.T
    assert C.isequal(A.T.new())

    with pytest.raises(TypeError):
        C.T << A
    with pytest.raises(TypeError, match="does not support item assignment"):
        C.T[:, :] << A
    with pytest.raises(TypeError, match="autocompute"):
        C[:, :].T << A

    C = Matrix(A.dtype, A.ncols + 1, A.nrows + 1)
    C[: A.ncols, : A.nrows] << A.T
    assert C[: A.ncols, : A.nrows].new().isequal(A.T.new())


def test_assign_list():
    A = Matrix(int, 3, 3)
    A[[0, 1], [1, 2]] = [[3, 4], [5, 6]]
    expected = Matrix.from_values([0, 0, 1, 1], [1, 2, 1, 2], [3, 4, 5, 6], nrows=3, ncols=3)
    assert A.isequal(expected)
    A[[0, 1], 1] = np.arange(2)
    expected = Matrix.from_values([0, 0, 1, 1], [1, 2, 1, 2], [0, 4, 1, 6], nrows=3, ncols=3)
    assert A.isequal(expected)
    A[0, 1:3] = [10, 20]
    expected = Matrix.from_values([0, 0, 1, 1], [1, 2, 1, 2], [10, 20, 1, 6], nrows=3, ncols=3)
    assert A.isequal(expected)
    with pytest.raises(TypeError):
        A[0, 1] = [0]
    with pytest.raises(TypeError):
        A()[0, 1] = [0]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], 1] = [0]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], [1, 2]] = [0]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], [1, 2]] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1, 2], [1]] = [1, 2, 3]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1, 2], [1]] = [[1, 2, 3]]
    with pytest.raises(TypeError):
        A[[0, 1], [1, 2]] = [[3, 4], [5, object()]]


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
    # with pytest.raises(AttributeError):
    # should use new instead--Now okay.
    assert B.T.dup().isequal(B.T.new())
    # Not exceptional, but while we're here...
    C = B.T.new(mask=A.V)
    D = B.T.new()
    D = D.dup(mask=A.V)
    assert C.isequal(D)
    assert C.isequal(Matrix.from_values([0, 0, 1], [0, 1, 1], [1, 3, 4]))


def test_nested_matrix_operations():
    """Make sure temporaries aren't garbage-collected too soon"""
    A = Matrix(int, 8, 8)
    A.ewise_mult(A.mxm(A.T).new()).new().reduce_scalar().new()
    A.ewise_mult(A.ewise_mult(A.ewise_mult(A.ewise_mult(A).new()).new()).new())


def test_bad_init():
    with pytest.raises(TypeError, match="Bad dtype"):
        Matrix(None, float, name="bad_matrix")


def test_equals(A):
    assert (A == A).new().reduce_scalar(monoid.land).new()


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
    shell_A = object.__new__(Matrix)
    del shell_A
    # A has `gb_obj` of NULL
    A = Matrix.from_values([0, 1], [0, 1], [0, 1])
    gb_obj = A.gb_obj
    A.gb_obj = graphblas.ffi.NULL
    del A
    # let's clean up so we don't have a memory leak
    A2 = object.__new__(Matrix)
    A2.gb_obj = gb_obj
    del A2
    gc.collect()
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
def test_import_export(A, do_iso, methods):
    if do_iso:
        A(A.S) << 1
    A1 = A.dup()
    out_method, in_method = methods
    if out_method == "export":
        d = getattr(A1.ss, out_method)("csr", give_ownership=True)
    else:
        d = getattr(A1.ss, out_method)("csr")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert_array_equal(d["indptr"], [0, 2, 4, 5, 7, 8, 9, 12])
    assert_array_equal(d["col_indices"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])
    if not do_iso:
        assert_array_equal(d["values"], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3])
    if in_method == "import":
        B1 = Matrix.ss.import_any(take_ownership=True, **d)
        assert B1.isequal(A)
        assert B1.ss.is_iso is do_iso
    else:
        A1.ss.pack_any(take_ownership=True, **d)
        assert A1.isequal(A)
        assert A1.ss.is_iso is do_iso

    A2 = A.dup()
    d = getattr(A2.ss, out_method)("csc")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert_array_equal(d["indptr"], [0, 1, 2, 5, 7, 9, 11, 12])
    assert_array_equal(d["row_indices"], [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1])
    if not do_iso:
        assert_array_equal(d["values"], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4])
    if in_method == "import":
        B2 = Matrix.ss.import_any(**d)
        assert B2.isequal(A)
        assert B2.ss.is_iso is do_iso
    else:
        A2.ss.pack_any(**d)
        assert A2.isequal(A)
        assert A2.ss.is_iso is do_iso

    A3 = A.dup()
    d = getattr(A3.ss, out_method)("hypercsr")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert_array_equal(d["rows"], [0, 1, 2, 3, 4, 5, 6])
    assert_array_equal(d["indptr"], [0, 2, 4, 5, 7, 8, 9, 12])
    assert_array_equal(d["col_indices"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])
    if not do_iso:
        assert_array_equal(d["values"], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3])
    if in_method == "import":
        B3 = Matrix.ss.import_any(**d)
        assert B3.isequal(A)
        assert B3.ss.is_iso is do_iso
    else:
        A3.ss.pack_any(**d)
        assert A3.isequal(A)
        assert A3.ss.is_iso is do_iso

    A4 = A.dup()
    d = getattr(A4.ss, out_method)("hypercsc")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert_array_equal(d["cols"], [0, 1, 2, 3, 4, 5, 6])
    assert_array_equal(d["indptr"], [0, 1, 2, 5, 7, 9, 11, 12])
    assert_array_equal(d["row_indices"], [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1])
    if not do_iso:
        assert_array_equal(d["values"], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4])
    if in_method == "import":
        B4 = Matrix.ss.import_any(**d)
        assert B4.isequal(A)
        assert B4.ss.is_iso is do_iso
    else:
        A4.ss.pack_any(**d)
        assert A4.isequal(A)
        assert A4.ss.is_iso is do_iso

    A5 = A.dup()
    d = getattr(A5.ss, out_method)("bitmapr")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert "nrows" not in d
    assert "ncols" not in d
    if not do_iso:
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
    assert_array_equal(d["bitmap"], bitmap)
    if not do_iso:
        assert_array_equal(
            d["values"].ravel("K")[d["bitmap"].ravel("K")], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3]
        )
    del d["nvals"]
    if in_method == "import":
        B5 = Matrix.ss.import_any(**d)
        assert B5.isequal(A)
        assert B5.ss.is_iso is do_iso
    else:
        A5.ss.pack_any(**d)
        assert A5.isequal(A)
        assert A5.ss.is_iso is do_iso
    d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=0)
    B5b = Matrix.ss.import_any(**d)
    if in_method == "import":
        # B5b == [A, A] (i.e, get 2d shape from bitmap first if possible)
        assert B5b.nvals == 2 * A.nvals
        assert B5b.nrows == 2 * A.nrows
        assert B5b.ncols == A.ncols
    else:
        A5.ss.pack_any(**d)
        assert A5.isequal(A)
        assert A5.ss.is_iso is do_iso

    A6 = A.dup()
    d = getattr(A6.ss, out_method)("bitmapc")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert_array_equal(d["bitmap"], bitmap)
    if not do_iso:
        assert_array_equal(
            d["values"].ravel("K")[d["bitmap"].ravel("K")], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
        )
    del d["nvals"]
    if in_method == "import":
        B6 = Matrix.ss.import_any(nrows=7, **d)
        assert B6.isequal(A)
        assert B6.ss.is_iso is do_iso
    else:
        A6.ss.pack_any(**d)
        assert A6.isequal(A)
        assert A6.ss.is_iso is do_iso
    d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=1)
    if in_method == "import":
        B6b = Matrix.ss.import_any(ncols=7, **d)
        assert B6b.isequal(A)
        assert B6b.ss.is_iso is do_iso
    else:
        A6.ss.pack_any(**d)
        assert A6.isequal(A)
        assert A6.ss.is_iso is do_iso

    A7 = A.dup()
    d = getattr(A7.ss, out_method)()
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    if in_method == "import":
        B7 = Matrix.ss.import_any(**d)
        assert B7.isequal(A)
        assert B7.ss.is_iso is do_iso
    else:
        A7.ss.pack_any(**d)
        assert A7.isequal(A)
        assert A7.ss.is_iso is do_iso

    A8 = A.dup()
    d = getattr(A8.ss, out_method)("bitmapr", raw=True)
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    del d["nrows"]
    del d["ncols"]
    if in_method == "import":
        with pytest.raises(ValueError, match="nrows and ncols must be provided"):
            Matrix.ss.import_any(**d)
    else:
        A8.ss.pack_any(**d)
        assert A8.isequal(A)
        assert A8.ss.is_iso is do_iso

    A9 = A.dup()
    d = getattr(A9.ss, out_method)("coo", sort=True)
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    assert d["nrows"] == 7
    assert d["ncols"] == 7
    assert d["rows"].shape == (12,)
    assert d["cols"].shape == (12,)
    assert d["sorted_cols"]
    assert_array_equal(d["rows"], [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6])
    assert_array_equal(d["cols"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])

    if do_iso:
        assert d["values"].shape == (1,)
    else:
        assert d["values"].shape == (12,)
    if in_method == "import":
        B8 = Matrix.ss.import_any(**d)
        assert B8.isequal(A)
        assert B8.ss.is_iso is do_iso
        del d["rows"]
        del d["format"]
        with pytest.raises(ValueError, match="coo requires both"):
            Matrix.ss.import_any(**d)
    else:
        A9.ss.pack_any(**d)
        assert A9.isequal(A)
        assert A9.ss.is_iso is do_iso

    C = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    if do_iso:
        C(C.S) << 1
    C1 = C.dup()
    d = getattr(C1.ss, out_method)("fullr")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
        assert "nrows" in d
        assert "ncols" in d
    else:
        assert "nrows" not in d
        assert "ncols" not in d
    assert d["values"].flags.c_contiguous
    if not do_iso:
        assert d["values"].shape == (2, 2)
        assert_array_equal(d["values"], [[1, 2], [3, 4]])
        if in_method == "import":
            D1 = Matrix.ss.import_any(ncols=2, **d)
            assert D1.isequal(C)
            assert D1.ss.is_iso is do_iso
        else:
            C1.ss.pack_any(**d)
            assert C1.isequal(C)
            assert C1.ss.is_iso is do_iso
    else:
        if in_method == "import":
            D1 = Matrix.ss.import_any(**d)
            assert D1.isequal(C)
            assert D1.ss.is_iso is do_iso
        else:
            C1.ss.pack_any(**d)
            assert C1.isequal(C)
            assert C1.ss.is_iso is do_iso

    C2 = C.dup()
    d = getattr(C2.ss, out_method)("fullc")
    assert d["is_iso"] is do_iso
    if do_iso:
        assert_array_equal(d["values"], [1])
    if not do_iso:
        assert_array_equal(d["values"], [[1, 2], [3, 4]])
    assert d["values"].flags.f_contiguous
    if in_method == "import":
        D2 = Matrix.ss.import_any(**d)
        assert D2.isequal(C)
        assert D2.ss.is_iso is do_iso
    else:
        C2.ss.pack_any(**d)
        assert C2.isequal(C)
        assert C2.ss.is_iso is do_iso

    # all elements must have values
    with pytest.raises(InvalidValue):
        getattr(A.dup().ss, out_method)("fullr")
    with pytest.raises(InvalidValue):
        getattr(A.dup().ss, out_method)("fullc")

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
    with pytest.raises(ValueError, match="Invalid format"):
        A.ss.export("coobad")
    D = Matrix.ss.import_csc(**A.ss.export("csc"))
    info = D.ss.export("coo", sort=True)
    assert info["sorted_rows"]
    E = Matrix.ss.import_any(**info)
    assert E.isequal(A)

    info = D.ss.export("coor")
    info["sorted_rows"] = False
    with pytest.raises(ValueError, match="sorted_rows must be True"):
        Matrix.ss.import_coor(**info)
    info["sorted_rows"] = True
    info["sorted_cols"] = False
    del info["format"]
    E = Matrix.ss.import_any(**info)
    assert E.isequal(D)

    info = D.ss.export("cooc")
    info["sorted_cols"] = False
    with pytest.raises(ValueError, match="sorted_cols must be True"):
        Matrix.ss.import_cooc(**info)
    info["sorted_cols"] = True
    info["sorted_rows"] = False
    del info["format"]
    E = Matrix.ss.import_any(**info)
    assert E.isequal(D)


def test_import_on_view():
    A = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    B = Matrix.ss.import_any(nrows=2, ncols=2, values=np.array([1, 2, 3, 4, 99, 99, 99])[:4])
    assert A.isequal(B)


def test_import_export_empty():
    A = Matrix(int, 2, 3)
    A1 = A.dup()
    d = A1.ss.export("csr")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert_array_equal(d["indptr"], [0, 0, 0])
    assert len(d["col_indices"]) == 0
    assert len(d["values"]) == 0
    B1 = Matrix.ss.import_any(**d)
    assert B1.isequal(A)

    A2 = A.dup()
    d = A2.ss.export("csc")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert_array_equal(d["indptr"], [0, 0, 0, 0])
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
    assert_array_equal(d["bitmap"].ravel(), 6 * [0])
    B5 = Matrix.ss.import_any(**d)
    assert B5.isequal(A)

    A6 = A.dup()
    d = A6.ss.export("bitmapc")
    assert d["bitmap"].shape == (2, 3)
    assert d["bitmap"].flags.f_contiguous
    assert d["nvals"] == 0
    assert_array_equal(d["bitmap"].ravel(), 6 * [0])
    B6 = Matrix.ss.import_any(**d)
    assert B6.isequal(A)

    # all elements must have values
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullr")
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullc")

    A7 = A.dup()
    d = A7.ss.export("coo")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["rows"]) == 0
    assert len(d["cols"]) == 0
    assert len(d["values"]) == 0

    # if we give the same value, make sure it's copied
    for format, key1, key2 in [
        ("csr", "values", "col_indices"),
        ("hypercsr", "values", "col_indices"),
        ("csc", "values", "row_indices"),
        ("hypercsc", "values", "row_indices"),
        ("bitmapr", "values", "bitmap"),
        ("bitmapc", "values", "bitmap"),
        ("coo", "values", "rows"),
    ]:
        # No assertions here, but code coverage should be "good enough"
        d = A.ss.export(format, raw=True)
        d[key1] = d[key2]
        Matrix.ss.import_any(take_ownership=True, **d)

    with pytest.raises(ValueError, match="Invalid format"):
        A.ss.export(format="bad_format")


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
def test_import_export_auto(A, do_iso, methods):
    if do_iso:
        A(A.S) << 1
    A_orig = A.dup()
    out_method, in_method = methods
    for format in [
        "csr",
        "csc",
        "hypercsr",
        "hypercsc",
        "bitmapr",
        "bitmapc",
        "coo",
        "coor",
        "cooc",
    ]:
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
            A2 = A.dup() if give_ownership or out_method == "unpack" else A
            if out_method == "export":
                d = A2.ss.export(format, sort=sort, raw=raw, give_ownership=give_ownership)
            else:
                d = A2.ss.unpack(format, sort=sort, raw=raw)
            if in_method == "import":
                import_func = getattr(Matrix.ss, f"import_{import_name}")
            else:

                def import_func(**kwargs):
                    getattr(A2.ss, f"pack_{import_name}")(**kwargs)
                    return A2

            d["format"] = import_format
            other = import_func(take_ownership=take_ownership, **d)
            if format == "bitmapc" and raw and import_format is None and import_name == "any":
                # It's 1d, so we can't tell we're column-oriented w/o format keyword
                assert other.isequal(A_orig.T)
            else:
                assert other.isequal(A_orig)
            assert other.ss.is_iso is do_iso
            d["format"] = "bad_format"
            with pytest.raises(ValueError, match="Invalid format"):
                import_func(**d)
    assert A.isequal(A_orig)
    assert A.ss.is_iso is do_iso
    assert A_orig.ss.is_iso is do_iso

    C = Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6])
    if do_iso:
        C(C.S) << 1
    C_orig = C.dup()
    for format in ["fullr", "fullc"]:
        for raw, import_format, give_ownership, take_ownership, import_name in itertools.product(
            [False, True],
            [format, None],
            [False, True],
            [False, True],
            ["any", format],
        ):
            assert C.shape == (C.nrows, C.ncols)
            C2 = C.dup() if give_ownership or out_method == "unpack" else C
            if out_method == "export":
                d = C2.ss.export(format, raw=raw, give_ownership=give_ownership)
            else:
                d = C2.ss.unpack(format, raw=raw)
            if in_method == "import":
                import_func = getattr(Matrix.ss, f"import_{import_name}")
            else:

                def import_func(**kwargs):
                    getattr(C2.ss, f"pack_{import_name}")(**kwargs)
                    return C2

            d["format"] = import_format
            other = import_func(take_ownership=take_ownership, **d)
            if format == "fullc" and raw and import_format is None and import_name == "any":
                # It's 1d, so we can't tell we're column-oriented w/o format keyword
                if do_iso:
                    values = [1, 1, 1, 1, 1, 1]
                else:
                    values = [1, 3, 5, 2, 4, 6]
                assert other.isequal(
                    Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], values)
                )
            else:
                assert other.isequal(C_orig)
            assert other.ss.is_iso is do_iso
            d["format"] = "bad_format"
            with pytest.raises(ValueError, match="Invalid format"):
                import_func(**d)
    assert C.isequal(C_orig)
    assert C.ss.is_iso is do_iso
    assert C_orig.ss.is_iso is do_iso


def test_no_bool_or_eq(A):
    with pytest.raises(TypeError, match="not defined"):
        bool(A)
    # with pytest.raises(TypeError, match="not defined"):
    assert (A == A) is not None
    with pytest.raises(TypeError, match="not defined"):
        bool(A.S)
    with pytest.raises(TypeError, match="not defined"):
        assert A.S == A.S
    expr = A.ewise_mult(A)
    with pytest.raises(TypeError, match="not defined"):
        bool(expr)
    with pytest.raises(TypeError, match="not enabled"):
        assert expr == expr
    assigner = A[1, 2]()
    with pytest.raises(TypeError, match="not defined"):
        bool(assigner)
    with pytest.raises(TypeError, match="not defined"):
        assert assigner == assigner
    updater = A()
    with pytest.raises(TypeError, match="not defined"):
        bool(updater)
    with pytest.raises(TypeError, match="not defined"):
        assert updater == updater


@autocompute
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
    # with pytest.raises(TypeError, match="not defined"):
    assert (expr == expr) is not None  # Now okay
    with pytest.raises(TypeError, match="not defined"):
        bool(expr)
    with pytest.raises(TypeError):
        int(expr)
    with pytest.raises(TypeError):
        float(expr)
    with pytest.raises(TypeError):
        range(expr)


def test_bool_eq_on_scalar_expressions_no_auto(A):
    expr = A.reduce_scalar()
    with pytest.raises(TypeError, match="autocompute"):
        assert expr == 47
    with pytest.raises(TypeError, match="autocompute"):
        bool(expr)
    with pytest.raises(TypeError, match="autocompute"):
        int(expr)


def test_contains(A):
    assert (0, 1) in A
    assert (1, 0) in A.T

    assert (0, 1) not in A.T
    assert (1, 0) not in A

    with pytest.raises(TypeError):
        assert 1 in A
    with pytest.raises(TypeError):
        assert (1,) in A.T
    with pytest.raises(TypeError, match="Invalid index"):
        assert (1, [1, 2]) in A


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


@pytest.mark.parametrize(
    "params",
    [
        (0, [], []),
        (1, [0, 4], [2, 7]),
        (3, [0, 1, 2], [3, 8, 1]),
        (10, [], []),
        (-1, [2], [3]),
        (-3, [0, 2, 3], [3, 1, 7]),
        (-10, [], []),
    ],
)
def test_diag(A, params):
    k, indices, values = params
    expected = Vector.from_values(indices, values, dtype=A.dtype, size=max(0, A.nrows - abs(k)))
    v = graphblas.ss.diag(A, k=k)
    assert expected.isequal(v)
    v[:] = 0
    v.ss.build_diag(A, k=k)
    assert expected.isequal(v)
    v = A.diag(k)
    assert expected.isequal(v)
    v = graphblas.ss.diag(A.T, k=-k)
    assert expected.isequal(v)
    v[:] = 0
    v.ss.build_diag(A.T, -k)
    assert expected.isequal(v)
    v = A.T.diag(-k)
    assert expected.isequal(v)


def test_normalize_chunks():
    from graphblas._ss.matrix import normalize_chunks

    shape = (20, 20)
    assert normalize_chunks(10, shape) == [[10, 10], [10, 10]]
    assert normalize_chunks(15.0, shape) == [[15, 5], [15, 5]]
    assert normalize_chunks(((10, 10), [10, 10]), shape) == [[10, 10], [10, 10]]
    assert normalize_chunks((15, [10, 10.0]), shape) == [[15, 5], [10, 10]]
    assert normalize_chunks((None, np.array([10, 10])), shape) == [[20], [10, 10]]
    assert normalize_chunks([[5, None], (None, 6)], shape) == [[5, 15], [14, 6]]
    assert normalize_chunks(np.array([10, 10]), shape) == [[10, 10], [10, 10]]

    with pytest.raises(TypeError, match="chunks argument must be a list"):
        normalize_chunks(None, shape)
    with pytest.raises(TypeError, match="None value in chunks"):
        normalize_chunks([[5, 5, None, None], 10], shape)
    with pytest.raises(TypeError, match="expected int or None, but got"):
        normalize_chunks([[15.5, 4.5], 10], shape)
    with pytest.raises(TypeError, match="Chunks for a dimension must be"):
        normalize_chunks([10, 10.5], shape)
    with pytest.raises(TypeError, match="must be integer dtype; got float64"):
        normalize_chunks([10, np.array([1.5, 2.5])], shape)
    with pytest.raises(TypeError, match="numpy array for chunks must be 1-dimension"):
        normalize_chunks([10, np.array([[1, 2], [3, 4]])], shape)

    with pytest.raises(ValueError, match="hunks argument must be of length 2"):
        normalize_chunks([10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks(-10, shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([-10, -10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([[-10, 30], [-10, 30]], shape)
    with pytest.raises(ValueError, match="chunks argument must be of length 2"):
        normalize_chunks([5, 5, 5], shape)
    with pytest.raises(ValueError, match="Chunks are too large"):
        normalize_chunks([[30, None], 10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([10, np.array([-1, 2])], shape)


def test_split(A):
    results = A.ss.split([4, 3])
    for results in [A.ss.split([4, 3]), A.ss.split([[4, None], 3], name="split")]:
        row_boundaries = [0, 4, 7]
        col_boundaries = [0, 3, 6, 7]
        for i, (i1, i2) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
            for j, (j1, j2) in enumerate(zip(col_boundaries[:-1], col_boundaries[1:])):
                expected = A[i1:i2, j1:j2].new()
                assert expected.isequal(results[i][j])
    with pytest.raises(DimensionMismatch):
        A.ss.split([[5, 5], 3])


def test_concat(A, v):
    B1 = graphblas.ss.concat([[A, A]], dtype=float)
    assert B1.dtype == "FP64"
    expected = Matrix(A.dtype, nrows=A.nrows, ncols=2 * A.ncols)
    expected[:, : A.ncols] = A
    expected[:, A.ncols :] = A
    assert B1.isequal(expected)

    B2 = Matrix(A.dtype, nrows=2 * A.nrows, ncols=A.ncols)
    B2.ss.concat([[A], [A]])
    expected = Matrix(A.dtype, nrows=2 * A.nrows, ncols=A.ncols)
    expected[: A.nrows, :] = A
    expected[A.nrows :, :] = A
    assert B2.isequal(expected)

    tiles = A.ss.split([4, 3])
    A2 = graphblas.ss.concat(tiles)
    assert A2.isequal(A)

    with pytest.raises(TypeError, match="tiles argument must be list or tuple"):
        graphblas.ss.concat(1)
    # with pytest.raises(TypeError, match="Each tile must be a Matrix"):
    assert graphblas.ss.concat([[A.T]]).isequal(A.T)
    with pytest.raises(TypeError, match="tiles must be lists or tuples"):
        graphblas.ss.concat([A])

    with pytest.raises(ValueError, match="tiles argument must not be empty"):
        graphblas.ss.concat([])
    with pytest.raises(ValueError, match="tiles must not be empty"):
        graphblas.ss.concat([[]])
    with pytest.raises(ValueError, match="tiles must all be the same length"):
        graphblas.ss.concat([[A], [A, A]])

    # Treat vectors like Nx1 matrices
    B3 = graphblas.ss.concat([[v, v]])
    expected = Matrix(v.dtype, nrows=v.size, ncols=2)
    expected[:, 0] = v
    expected[:, 1] = v
    assert B3.isequal(expected)

    B4 = graphblas.ss.concat([[v], [v]])
    expected = Matrix(v.dtype, nrows=2 * v.size, ncols=1)
    expected[: v.size, 0] = v
    expected[v.size :, 0] = v
    assert B4.isequal(expected)

    B5 = graphblas.ss.concat([[A, v]])
    expected = Matrix(v.dtype, nrows=v.size, ncols=A.ncols + 1)
    expected[:, : A.ncols] = A
    expected[:, A.ncols] = v
    assert B5.isequal(expected)

    with pytest.raises(TypeError, match=""):
        graphblas.ss.concat([v, [v]])
    with pytest.raises(TypeError):
        graphblas.ss.concat([[v], v])


def test_nbytes(A):
    assert A.ss.nbytes > 0


@autocompute
def test_auto(A, v):
    expected = binary.land[bool](A & A).new()
    B = A.dup(dtype=bool)
    for expr in [(B & B), binary.land[bool](A & A)]:
        assert expr.dtype == expected.dtype
        assert expr.nrows == expected.nrows
        assert expr.ncols == expected.ncols
        assert expr.shape == expected.shape
        assert expr.nvals == expected.nvals
        assert expr.isclose(expected)
        assert expected.isclose(expr)
        assert expr.isequal(expected)
        assert expected.isequal(expr)
        assert expr.mxv(v).isequal(expected.mxv(v))
        assert expected.T.mxv(v).isequal(expr.T.mxv(v))
        for method in [
            # "ewise_add",
            # "ewise_mult",
            # "mxm",
            # "__matmul__",
            "__and__",
            "__or__",
            # "kronecker",
        ]:
            val1 = getattr(expected, method)(expected).new()
            val2 = getattr(expected, method)(expr)
            val3 = getattr(expr, method)(expected)
            val4 = getattr(expr, method)(expr)
            assert val1.isequal(val2)
            assert val1.isequal(val3)
            assert val1.isequal(val4)
        for method in ["reduce_rowwise", "reduce_columnwise", "reduce_scalar"]:
            s1 = getattr(expected, method)(monoid.lor).new()
            s2 = getattr(expr, method)(monoid.lor)
            assert s1.isequal(s2.new())
            assert s1.isequal(s2)

    expected = binary.times(A & A).new()
    for expr in [binary.times(A & A)]:
        assert expr.dtype == expected.dtype
        assert expr.nrows == expected.nrows
        assert expr.ncols == expected.ncols
        assert expr.shape == expected.shape
        assert expr.nvals == expected.nvals
        assert expr.isclose(expected)
        assert expected.isclose(expr)
        assert expr.isequal(expected)
        assert expected.isequal(expr)
        assert expr.mxv(v).isequal(expected.mxv(v))
        assert expected.T.mxv(v).isequal(expr.T.mxv(v))
        for method in [
            "ewise_add",
            "ewise_mult",
            "mxm",
            # "__matmul__",
            # "__and__",
            # "__or__",
            "kronecker",
        ]:
            val1 = getattr(expected, method)(expected).new()
            val2 = getattr(expected, method)(expr)
            val3 = getattr(expr, method)(expected)
            val4 = getattr(expr, method)(expr)
            assert val1.isequal(val2)
            assert val1.isequal(val3)
            assert val1.isequal(val4)
        for method in ["reduce_rowwise", "reduce_columnwise", "reduce_scalar"]:
            s1 = getattr(expected, method)().new()
            s2 = getattr(expr, method)()
            assert s1.isequal(s2.new())
            assert s1.isequal(s2)

    expected = semiring.plus_times(A @ v).new()
    for expr in [(A @ v), (v @ A.T), semiring.plus_times(A @ v)]:
        assert expr.vxm(A).isequal(expected.vxm(A))
        assert expr.vxm(A).new(mask=expr.S).isequal(expected.vxm(A).new(mask=expected.S))
        assert expr.vxm(A).new(mask=expr.V).isequal(expected.vxm(A).new(mask=expected.V))


@autocompute
def test_auto_assign(A):
    expected = A.dup()
    B = A[1:4, 1:4].new(dtype=bool)
    expr = B & B
    expected[:3, :3] = expr.new()
    A[:3, :3] = expr
    assert expected.isequal(A)
    v = A[2:5, 5].new(dtype=bool)
    expr = v & v
    A[:3, 4] << expr
    expected[:3, 4] << expr.new()
    assert expected.isequal(A)
    C = A[1:4, 1:4].new()
    A[:3, :3] = A[1:4, 1:4]
    assert A[:3, :3].isequal(C)


@autocompute
def test_expr_is_like_matrix(A):
    B = A.dup(dtype=bool)
    attrs = {attr for attr, val in inspect.getmembers(B)}
    expr_attrs = {attr for attr, val in inspect.getmembers(binary.times(B & B))}
    infix_attrs = {attr for attr, val in inspect.getmembers(B & B)}
    transposed_attrs = {attr for attr, val in inspect.getmembers(B.T)}
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
        "If you see this message, you probably added a method to Matrix.  You may need to "
        "add an entry to `matrix` or `matrix_vector` set in `graphblas._automethods.py` "
        "and then run `python -m graphblas._automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas._infixmethods`."
    )
    assert attrs - infix_attrs == expected
    # TransposedMatrix is used differently than other expressions,
    # so maybe it shouldn't support everything.
    assert attrs - transposed_attrs == (expected | {"S", "V", "ss", "to_pygraphblas"}) - {
        "_prep_for_extract",
        "_extract_element",
    }


@autocompute
def test_index_expr_is_like_matrix(A):
    B = A.dup(dtype=bool)
    attrs = {attr for attr, val in inspect.getmembers(B)}
    expr_attrs = {attr for attr, val in inspect.getmembers(B[[0, 1], [0, 1]])}
    # Should we make any of these raise informative errors?
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
        "If you see this message, you probably added a method to Matrix.  You may need to "
        "add an entry to `matrix` or `matrix_vector` set in `graphblas._automethods.py` "
        "and then run `python -m graphblas._automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas._infixmethods`."
    )


def test_flatten(A):
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
    ]
    # row-wise
    indices = [row * A.ncols + col for row, col in zip(data[0], data[1])]
    expected = Vector.from_values(indices, data[2], size=A.nrows * A.ncols)
    for fmt in ["csr", "hypercsr", "bitmapr"]:
        B = Matrix.ss.import_any(**A.ss.export(format=fmt))
        v = B.ss.flatten()
        assert v.isequal(expected)
        C = v.ss.reshape(*B.shape)
        assert C.isequal(B)
    B(mask=~B.S)[:, :] = 10
    expected(mask=~expected.S)[:] = 10
    B = Matrix.ss.import_fullr(**B.ss.export(format="fullr"))
    v = B.ss.flatten()
    assert v.isequal(expected)
    C = v.ss.reshape(*B.shape)
    assert C.isequal(B)
    C = v.ss.reshape(B.shape)
    assert C.isequal(B)

    # column-wise
    indices = [col * A.nrows + row for row, col in zip(data[0], data[1])]
    expected = Vector.from_values(indices, data[2], size=A.nrows * A.ncols)
    for fmt in ["csc", "hypercsc", "bitmapc"]:
        B = Matrix.ss.import_any(**A.ss.export(format=fmt))
        v = B.ss.flatten(order="col")
        assert v.isequal(expected)
        C = v.ss.reshape(*B.shape, order="col")
        assert C.isequal(B)
    B(mask=~B.S)[:, :] = 10
    expected(mask=~expected.S)[:] = 10
    B = Matrix.ss.import_fullc(**B.ss.export(format="fullc"))
    v = B.ss.flatten(order="F")
    assert v.isequal(expected)
    C = v.ss.reshape(*B.shape, order="F")
    assert C.isequal(B)
    C = v.ss.reshape(B.shape, order="F")
    assert C.isequal(B)
    with pytest.raises(ValueError, match="Bad value for order"):
        A.ss.flatten(order="bad")
    with pytest.raises(ValueError, match="cannot reshape"):
        v.ss.reshape(100, 100)
    with pytest.raises(ValueError):
        v.ss.reshape(A.shape + (1,))


def test_autocompute_argument_messages(A, v):
    with pytest.raises(TypeError, match="autocompute"):
        A.ewise_mult(A & A)
    with pytest.raises(TypeError, match="autocompute"):
        A.mxv(A @ v)


@autocompute
def test_infix_sugar(A):
    assert type(A + 1) is not Matrix
    assert binary.plus(A, 1).isequal(A + 1)
    assert binary.plus(A.T, 1).isequal(A.T + 1)
    assert binary.plus(1, A).isequal(1 + A)
    assert binary.minus(A, 1).isequal(A - 1)
    assert binary.minus(1, A).isequal(1 - A)
    assert binary.times(A, 2).isequal(A * 2)
    assert binary.times(2, A).isequal(2 * A)
    assert binary.truediv(A, 2).isequal(A / 2)
    assert binary.truediv(5, A).isequal(5 / A)
    assert binary.floordiv(A, 2).isequal(A // 2)
    assert binary.floordiv(5, A).isequal(5 // A)
    assert binary.numpy.mod(A, 2).isequal(A % 2)
    assert binary.numpy.mod(5, A).isequal(5 % A)
    assert binary.pow(A, 2).isequal(A**2)
    assert binary.pow(2, A).isequal(2**A)
    assert binary.pow(A, 2).isequal(pow(A, 2))
    assert unary.ainv(A).isequal(-A)
    assert unary.ainv(A.T).isequal(-A.T)
    B = A.dup(dtype=bool)
    assert unary.lnot(B).isequal(~B)
    assert unary.lnot(B.T).isequal(~B.T)
    with pytest.raises(TypeError):
        assert unary.lnot(A).isequal(~A)
    with pytest.raises(TypeError):
        assert unary.lnot(A.T).isequal(~A.T)
    assert binary.lxor(True, B).isequal(True ^ B)
    assert binary.lxor(B, True).isequal(B ^ True)
    with pytest.raises(TypeError):
        A ^ True
    with pytest.raises(TypeError):
        A ^ B
    with pytest.raises(TypeError):
        6 ^ B
    assert binary.lt(A, 4).isequal(A < 4)
    assert binary.le(A, 4).isequal(A <= 4)
    assert binary.gt(A, 4).isequal(A > 4)
    assert binary.ge(A, 4).isequal(A >= 4)
    assert binary.eq(A, 4).isequal(A == 4)
    assert binary.ne(A, 4).isequal(A != 4)
    x, y = divmod(A, 3)
    assert binary.floordiv(A, 3).isequal(x)
    assert binary.numpy.mod(A, 3).isequal(y)
    assert binary.fmod(A, 3).isequal(y)
    assert A.isequal(binary.plus((3 * x) & y))
    x, y = divmod(-A, 3)
    assert binary.floordiv(-A, 3).isequal(x)
    assert binary.numpy.mod(-A, 3).isequal(y)
    # assert binary.fmod(-A, 3).isequal(y)  # The reason we use numpy.mod
    assert (-A).isequal(binary.plus((3 * x) & y))
    x, y = divmod(3, A)
    assert binary.floordiv(3, A).isequal(x)
    assert binary.numpy.mod(3, A).isequal(y)
    assert binary.fmod(3, A).isequal(y)
    assert binary.plus(binary.times(A & x) & y).isequal(3 * unary.one(A))
    x, y = divmod(-3, A)
    assert binary.floordiv(-3, A).isequal(x)
    assert binary.numpy.mod(-3, A).isequal(y)
    # assert binary.fmod(-3, A).isequal(y)  # The reason we use numpy.mod
    assert binary.plus(binary.times(A & x) & y).isequal(-3 * unary.one(A))

    assert binary.eq(A & A).isequal(A == A)
    assert binary.ne(A.T & A.T).isequal(A.T != A.T)
    assert binary.lt(A & A.T).isequal(A < A.T)
    assert binary.ge(A.T & A).isequal(A.T >= A)

    B = A.dup()
    B += 1
    assert type(B) is Matrix
    assert binary.plus(A, 1).isequal(B)
    B = A.dup()
    B -= 1
    assert type(B) is Matrix
    assert binary.minus(A, 1).isequal(B)
    B = A.dup()
    B *= 2
    assert type(B) is Matrix
    assert binary.times(A, 2).isequal(B)
    B = A.dup(dtype=float)
    B /= 2
    assert type(B) is Matrix
    assert binary.truediv(A, 2).isequal(B)
    B = A.dup()
    B //= 2
    assert type(B) is Matrix
    assert binary.floordiv(A, 2).isequal(B)
    B = A.dup()
    B %= 2
    assert type(B) is Matrix
    assert binary.numpy.mod(A, 2).isequal(B)
    B = A.dup()
    B **= 2
    assert type(B) is Matrix
    assert binary.pow(A, 2).isequal(B)
    B = A.dup(dtype=bool)
    B ^= True
    assert type(B) is Matrix
    assert B.isequal(~A.dup(dtype=bool))
    B = A.dup(dtype=bool)
    B ^= B
    assert type(B) is Matrix
    assert not B.reduce_scalar(agg.any).new()

    expr = binary.plus(A & A)
    assert unary.abs(expr).isequal(abs(expr))
    assert unary.ainv(expr).isequal(-expr)
    with pytest.raises(TypeError):
        assert unary.lnot(expr).isequal(~expr)
    with pytest.raises(TypeError):
        expr += 1
    with pytest.raises(TypeError):
        expr -= 1
    with pytest.raises(TypeError):
        expr *= 1
    with pytest.raises(TypeError):
        expr /= 1
    with pytest.raises(TypeError):
        expr //= 1
    with pytest.raises(TypeError):
        expr %= 1
    with pytest.raises(TypeError):
        expr **= 1
    with pytest.raises(TypeError):
        expr ^= 1


@pytest.mark.slow
def test_random(A):
    R = A.ss.selectk_rowwise("random", 1)
    counts = R.reduce_rowwise(agg.count).new()
    expected = Vector.from_values(range(A.ncols), 1)
    assert counts.isequal(expected)

    R = A.ss.selectk_columnwise("random", 1)
    counts = R.reduce_columnwise(agg.count).new()
    expected = Vector.from_values(range(A.nrows), 1)
    assert counts.isequal(expected)

    R = A.ss.selectk_rowwise("random", 2)
    counts = R.reduce_rowwise(agg.count).new()
    assert counts.reduce(monoid.min).new() == 1
    assert counts.reduce(monoid.max).new() == 2

    # test iso
    A(A.S) << 1
    R = A.ss.selectk_rowwise("random", 1)
    counts = R.reduce_rowwise(agg.count).new()
    expected = Vector.from_values(range(A.ncols), 1)
    assert counts.isequal(expected)

    with pytest.raises(ValueError):
        A.ss.selectk_rowwise("bad", 1)
    with pytest.raises(ValueError):
        A.ss.selectk_columnwise("bad", 1)
    with pytest.raises(ValueError):
        A.ss.selectk_columnwise("random", -1)


def test_firstk(A):
    B = A.ss.selectk_rowwise("first", 1)
    expected = Matrix.from_values(
        [0, 1, 2, 3, 4, 5, 6],
        [1, 4, 5, 0, 5, 2, 2],
        [2, 8, 1, 3, 7, 1, 5],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_rowwise("first", 2)
    expected = Matrix.from_values(
        [3, 0, 3, 5, 6, 0, 6, 1, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 1, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_rowwise("first", 3)
    assert B.isequal(A)

    B = A.ss.selectk_columnwise("first", 1)
    expected = Matrix.from_values(
        [3, 0, 3, 0, 1, 2, 1],
        [0, 1, 2, 3, 4, 5, 6],
        [3, 2, 3, 3, 8, 1, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_columnwise("first", 2)
    expected = Matrix.from_values(
        [3, 0, 3, 5, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 3, 7, 8, 3, 1, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_columnwise("first", 3)
    assert B.isequal(A)


def test_lastk(A):
    B = A.ss.selectk_rowwise("last", 1)
    expected = Matrix.from_values(
        [0, 3, 5, 6, 2, 4, 1],
        [3, 2, 2, 4, 5, 5, 6],
        [3, 3, 1, 3, 1, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_rowwise("last", 2)
    expected = Matrix.from_values(
        [3, 0, 3, 5, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 3, 7, 8, 3, 1, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_rowwise("last", 3)
    assert B.isequal(A)

    B = A.ss.selectk_columnwise("last", 1)
    expected = Matrix.from_values(
        [3, 0, 6, 6, 6, 4, 1],
        [0, 1, 2, 3, 4, 5, 6],
        [3, 2, 5, 7, 3, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_columnwise("last", 2)
    expected = Matrix.from_values(
        [3, 0, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 1, 5, 3, 7, 8, 3, 1, 7, 4],
        nrows=A.nrows,
        ncols=A.ncols,
    )
    assert B.isequal(expected)

    B = A.ss.selectk_columnwise("last", 3)
    assert B.isequal(A)


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.slow
def test_compactify(A, do_iso):
    if do_iso:
        r, c, v = A.to_values()
        A = Matrix.from_values(r, c, 1)
    rows = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
    new_cols = [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2]
    orig_cols = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]

    def check(A, expected, *args, stop=0, **kwargs):
        B = A.ss.compactify_rowwise(*args, **kwargs)
        assert B.isequal(expected)
        for n in reversed(range(stop, 4)):
            expected = expected[:, :n].new()
            B = A.ss.compactify_rowwise(*args, ncols=n, **kwargs)
            assert B.isequal(expected)

    def reverse(A):
        return A[:, ::-1].new().ss.compactify_rowwise("first", A.ncols)

    def check_reverse(A, expected, *args, stop=0, **kwargs):
        B = A.ss.compactify_rowwise(*args, reverse=True, **kwargs)
        C = reverse(expected)
        assert B.isequal(C)
        for n in reversed(range(stop, 4)):
            C = reverse(expected[:, :n].new())
            B = A.ss.compactify_rowwise(*args, ncols=n, reverse=True, **kwargs)
            assert B.isequal(C)

    expected = Matrix.from_values(
        rows,
        new_cols,
        1 if do_iso else [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3],
        nrows=A.nrows,
        ncols=3,
    )
    check(A, expected, "first")
    check_reverse(A, expected, "first")
    check(A, reverse(expected), "last")
    check_reverse(A, reverse(expected), "last")

    expected = Matrix.from_values(
        rows,
        new_cols,
        orig_cols,
        nrows=A.nrows,
        ncols=3,
    )
    check(A, expected, "first", asindex=True)
    check_reverse(A, expected, "first", asindex=True)
    check(A, reverse(expected), "last", asindex=True)
    check_reverse(A, reverse(expected), "last", asindex=True)

    expected = Matrix.from_values(
        rows,
        new_cols,
        1 if do_iso else [2, 3, 4, 8, 1, 3, 3, 7, 1, 3, 5, 7],
        nrows=A.nrows,
        ncols=3,
    )
    check(A, expected, "smallest")
    check_reverse(A, expected, "smallest")
    check(A, reverse(expected), "largest")
    check_reverse(A, reverse(expected), "largest")

    if not do_iso:
        expected = Matrix.from_values(
            rows,
            new_cols,
            [1, 3, 6, 4, 5, 0, 2, 5, 2, 4, 2, 3],
            nrows=A.nrows,
            ncols=3,
        )
        check(A, expected, "smallest", asindex=True, stop=2)
        check_reverse(A, expected, "smallest", asindex=True, stop=2)
        check(A, reverse(expected), "largest", asindex=True, stop=2)
        check_reverse(A, reverse(expected), "largest", asindex=True, stop=2)

    def compare(A, expected, isequal=True, **kwargs):
        for _ in range(1000):
            B = A.ss.compactify_rowwise("random", **kwargs)
            if B.isequal(expected) == isequal:
                break
        else:
            raise AssertionError("random failed")

    with pytest.raises(AssertionError):
        compare(A, A[:, ::-1].new())

    for asindex in [False, True]:
        compare(A, A.ss.compactify_rowwise("first", asindex=asindex), asindex=asindex)
        compare(A, A.ss.compactify_rowwise("first", 3, asindex=asindex), ncols=3, asindex=asindex)
        compare(A, A.ss.compactify_rowwise("first", 2, asindex=asindex), ncols=2, asindex=asindex)
        compare(
            A,
            A.ss.compactify_rowwise("first", 2, asindex=asindex),
            ncols=2,
            asindex=asindex,
            isequal=do_iso,
        )
        compare(A, A.ss.compactify_rowwise("first", 1, asindex=asindex), ncols=1, asindex=asindex)
        compare(
            A,
            A.ss.compactify_rowwise("first", 1, asindex=asindex),
            ncols=1,
            asindex=asindex,
            isequal=do_iso,
        )
        compare(A, A.ss.compactify_rowwise("last", 1, asindex=asindex), ncols=1, asindex=asindex)
        compare(
            A, A.ss.compactify_rowwise("smallest", 1, asindex=asindex), ncols=1, asindex=asindex
        )
        compare(A, A.ss.compactify_rowwise("largest", 1, asindex=asindex), ncols=1, asindex=asindex)
        compare(A, A.ss.compactify_rowwise("first", 0, asindex=asindex), ncols=0, asindex=asindex)

    B = A.ss.compactify_columnwise("first", nrows=1)
    expected = Matrix.from_values(
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6],
        1 if do_iso else [3, 2, 3, 3, 8, 1, 4],
    )
    assert B.isequal(expected)
    B = A.ss.compactify_columnwise("last", nrows=1, asindex=True)
    expected = Matrix.from_values(
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6],
        [3, 0, 6, 6, 6, 4, 1],
    )
    assert B.isequal(expected)
    with pytest.raises(ValueError):
        A.ss.compactify_rowwise("bad_how")


def test_deprecated(A):
    v = A.diag()
    with pytest.warns(DeprecationWarning):
        A.ss.diag(v)
    with pytest.warns(DeprecationWarning):
        v.ss.diag(A)
    with pytest.warns(DeprecationWarning):
        Matrix.new(int)
    with pytest.warns(DeprecationWarning):
        Vector.new(int)
    with pytest.warns(DeprecationWarning):
        Scalar.new(int)
    with pytest.warns(DeprecationWarning):
        A.S.mask
    with pytest.warns(DeprecationWarning):
        binary.plus(A | A, require_monoid=True)


def test_ndim(A):
    assert A.ndim == 2
    assert A.ewise_mult(A).ndim == 2
    assert (A & A).ndim == 2
    assert (A @ A).ndim == 2


def test_sizeof(A):
    assert sys.getsizeof(A) > A.nvals * 16


def test_ewise_union():
    A1 = Matrix.from_values([0], [0], [1], nrows=1, ncols=3)
    A2 = Matrix.from_values([0], [1], [2], nrows=1, ncols=3)
    result = A1.ewise_union(A2, binary.plus, 10, 20).new()
    expected = Matrix.from_values([0, 0], [0, 1], [21, 12], nrows=1, ncols=3)
    assert result.isequal(expected)

    # Test transposed
    A2transposed = A2.T.new()
    result = A1.ewise_union(A2transposed.T, binary.plus, 10, 20).new()
    assert result.isequal(expected)
    result = A1.T.ewise_union(A2transposed, binary.plus, 10, 20).new()
    assert result.isequal(expected.T.new())

    # Handle Scalars
    result = A1.ewise_union(A2, binary.plus, Scalar.from_value(10), Scalar.from_value(20)).new()
    assert result.isequal(expected)
    # Upcast if scalars are floats
    result = A1.ewise_union(A2, monoid.plus, 10.1, 20.2).new()
    expected = Matrix.from_values([0, 0], [0, 1], [21.2, 12.1], nrows=1, ncols=3)
    assert result.isclose(expected)

    result = A1.ewise_union(A2, binary.minus, 0, 0).new()
    expected = Matrix.from_values([0, 0], [0, 1], [1, -2], nrows=1, ncols=3)
    assert result.isequal(expected)
    result = (A1 - A2).new()
    assert result.isequal(expected)

    bad = Matrix(int, nrows=1, ncols=1)
    with pytest.raises(DimensionMismatch):
        A1.ewise_union(bad, binary.plus, 0, 0)
    with pytest.raises(TypeError, match="Literal scalars"):
        A1.ewise_union(A2, binary.plus, A2, 20)
    with pytest.raises(TypeError, match="Literal scalars"):
        A1.ewise_union(A2, binary.plus, 10, A2)


def test_delete_via_scalar(A):
    nvals = A.nvals
    del A[0, [1, 3]]
    assert A.nvals == nvals - 2
    assert A[0, :].new().nvals == 0
    del A[:, 0]
    assert A.nvals == nvals - 3
    assert A[:, 0].new().nvals == 0
    del A[:, :]
    assert A.nvals == 0


def test_iteration(A):
    B = Matrix(int, 2, 2)
    assert list(B.ss.iterkeys()) == []
    assert list(B.ss.itervalues()) == []
    assert list(B.ss.iteritems()) == []
    rows, columns, values = A.to_values()
    assert sorted(zip(rows, columns)) == sorted(A.ss.iterkeys())
    assert sorted(values) == sorted(A.ss.itervalues())
    assert sorted(zip(rows, columns, values)) == sorted(A.ss.iteritems())
    N = rows.size

    A = Matrix.ss.import_bitmapr(**A.ss.export("bitmapr"))
    assert A.ss.format == "bitmapr"
    assert len(list(A.ss.iterkeys(3))) == N - A[0, :3].new().nvals
    assert len(list(A.ss.iterkeys(-3))) == A[-1, -3:].new().nvals

    A = Matrix.ss.import_csr(**A.ss.export("csr"))
    assert A.ss.format == "csr"
    assert len(list(A.ss.iterkeys(3))) == N - 3
    assert len(list(A.ss.iterkeys(-3))) == 3
    assert len(list(A.ss.itervalues(N))) == 0
    assert len(list(A.ss.iteritems(N + 1))) == 0
    assert len(list(A.ss.iterkeys(N + 2))) == 0
    assert len(list(A.ss.iterkeys(-N))) == N
    assert len(list(A.ss.itervalues(-N - 1))) == N


def test_udt():
    record_dtype = np.dtype([("x", np.bool_), ("y", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "MatrixUDT")
    A = Matrix(udt, nrows=2, ncols=2)
    a = np.zeros(1, dtype=record_dtype)
    A[0, 0] = a[0]
    expected = Matrix.from_values([0], [0], a, nrows=2, ncols=2, dtype=udt)
    assert A.isequal(expected)
    A[0, 1] = (1, 2)
    expected = Matrix.from_values(
        [0, 0], [0, 1], np.array([(0, 0), (1, 2)], dtype=record_dtype), nrows=2, ncols=2, dtype=udt
    )
    assert A.isequal(expected)
    A[:, :] = 0
    zeros = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], 0, dtype=udt)
    assert A.isequal(zeros)
    A(A.S)[:, :] = 1
    ones = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], dtype=udt)
    assert A.isequal(ones)
    A[:, :](A.S) << 0
    assert A.isequal(zeros)
    s = Scalar(udt)
    s.value = (1, 1)
    A(A.S)[:, :] = s
    assert A.isequal(ones)
    s.value = 0
    A[:, :](A.S) << s
    assert A.isequal(zeros)
    A[0, :] = 1
    A[[1], :] = 1
    assert A.isequal(ones)
    A[:, [0]] = s
    A[:, 1] = s
    A[0, 0] = s
    assert A.isequal(zeros)
    t = A.reduce_scalar(monoid.any, allow_empty=True).new()
    assert s == t
    t = A.reduce_scalar(monoid.any, allow_empty=False).new()
    assert s == t
    A << binary.first(1, A)
    assert A.isequal(ones)
    A << binary.first(s, A)
    assert A.isequal(zeros)
    A << binary.second(A, 1)
    assert A.isequal(ones)
    A << binary.second(A, s)
    assert A.isequal(zeros)
    assert A[0, 0].new() == s
    assert A[:, :].new().isequal(A)
    expected = Vector.from_values([0, 1], s)
    assert A[0, :].new().isequal(expected)
    assert A.reduce_rowwise(monoid.any).new().isequal(expected)
    rows, cols, values = A.to_values()
    assert A.isequal(Matrix.from_values(rows, cols, values))
    assert A.isequal(Matrix.from_values(rows, cols, values, dtype=A.dtype))
    info = A.ss.export()
    result = A.ss.import_any(**info)
    assert result.isequal(A)
    info = A.ss.export("cooc")
    result = A.ss.import_any(**info)
    assert result.isequal(A)
    result = unary.positioni(A).new()
    expected = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    assert result.isequal(expected)
    AB = unary.one(select.tril(A).new()).new()
    BA = select.tril(unary.one(A).new()).new()
    assert AB.isequal(BA)

    # Just make sure these work
    for aggop in [agg.any_value, agg.first, agg.last, agg.count]:
        A.reduce_rowwise(aggop).new()
        A.reduce_columnwise(aggop).new()
        A.reduce_scalar(aggop).new()
    for aggop in [agg.first_index, agg.last_index]:
        A.reduce_rowwise(aggop).new()
        A.reduce_columnwise(aggop).new()
    A.clear()
    A[[0, 1], 1] = [(2, 3), (4, 5)]
    expected = Matrix.from_values([0, 1], [1, 1], [(2, 3), (4, 5)], dtype=udt)
    assert A.isequal(expected)
    A.clear()
    A[[0, 1], [1]] = [[(2, 3)], [(4, 5)]]
    assert A.isequal(expected)
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], [1]] = [[(2, 3), (4, 5)]]

    AA = Matrix.ss.deserialize(A.ss.serialize(), unsafe=True)
    assert A.isequal(AA, check_dtype=True)
    AA = Matrix.ss.deserialize(A.ss.serialize(), dtype=A.dtype)
    assert A.isequal(AA, check_dtype=True)
    with pytest.raises(ValueError):
        Matrix.ss.deserialize(A.ss.serialize())

    np_dtype = np.dtype("(3,)uint16")
    udt = dtypes.register_anonymous(np_dtype, "has_subdtype")
    A = Matrix(udt, nrows=2, ncols=2)
    A[:, :] = (1, 2, 3)
    rows, cols, values = A.to_values()
    assert_array_equal(values, np.array([[1, 2, 3]] * 4))
    result = Matrix.from_values(rows, cols, values)
    assert A.isequal(result)
    assert result.isequal(A)
    result = Matrix.from_values(rows, cols, values, dtype=udt)
    assert result.isequal(A)
    info = A.ss.export()
    result = A.ss.import_any(**info)
    assert result.isequal(A)
    info = A.ss.export("coor")
    result = A.ss.import_any(**info)
    assert result.isequal(A)
    A.clear()
    A[[0, 1], 1] = [(2, 3, 4), [5, 6, 7]]
    expected = Matrix.from_values([0, 1], [1, 1], [[2, 3, 4], [5, 6, 7]], dtype=udt)
    assert A.isequal(expected)
    A[[0, 1], [1]] = [[[2, 3, 4]], [[5, 6, 7]]]
    AA = Matrix.ss.deserialize(A.ss.serialize(), unsafe=True)
    assert A.isequal(AA, check_dtype=True)
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], [1]] = [[[2, 3, 4], [5, 6, 7]]]
    with pytest.raises(ValueError, match="shape mismatch"):
        A[[0, 1], [1]] = [[2, 3, 4], [5, 6, 7]]
    A = Matrix(udt, nrows=2, ncols=3)
    with pytest.raises(ValueError, match="include dtype shape"):
        A[[0, 1], :] = [[2, 3, 4], [5, 6, 7]]


def test_reposition(A):
    rows, cols, values = A.to_values()
    rows = rows.astype(int)
    cols = cols.astype(int)

    def get_expected(row_offset, col_offset, nrows, ncols, is_transposed):
        r = rows
        c = cols
        if is_transposed:
            r, c = c, r
        r = r + row_offset
        c = c + col_offset
        mask = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
        return Matrix.from_values(r[mask], c[mask], values[mask], nrows=nrows, ncols=ncols)

    for row_offset in range(-A.nrows - 2, A.nrows + 3, 3):
        for col_offset in range(-A.ncols - 2, A.ncols + 3, 3):
            for M in [A, A.T]:
                result = M.reposition(row_offset, col_offset).new()
                expected = get_expected(row_offset, col_offset, M.nrows, M.ncols, M._is_transposed)
                assert result.isequal(expected)
                result = M.reposition(row_offset, col_offset, nrows=3, ncols=10).new()
                expected = get_expected(row_offset, col_offset, 3, 10, M._is_transposed)
                assert result.isequal(expected)
                result = M.reposition(row_offset, col_offset, nrows=10, ncols=3).new()
                expected = get_expected(row_offset, col_offset, 10, 3, M._is_transposed)
                assert result.isequal(expected)

    result = A.reposition(3, 1).new(mask=A.S)
    expected = Matrix.from_values([3, 4, 6], [2, 5, 3], [2, 8, 3], nrows=A.nrows, ncols=A.ncols)
    assert result.isequal(expected)

    result(A.S, binary.plus) << A.reposition(3, 1)
    expected *= 2
    assert result.isequal(expected)

    result = A.T.reposition(-1, 1).new(mask=A.S)
    expected = Matrix.from_values(
        [0, 1, 1, 3, 4, 5], [1, 4, 6, 2, 5, 2], [2, 3, 1, 8, 7, 4], nrows=A.ncols, ncols=A.nrows
    )
    assert result.isequal(expected)

    result(A.S, binary.plus) << A.T.reposition(-1, 1)
    expected *= 2
    assert result.isequal(expected)


def test_to_values_sort():
    # How can we get a matrix to a jumbled state in SS so that export won't be sorted?
    N = 1000000
    r = np.unique(np.random.randint(N, size=100))
    c = np.unique(np.random.randint(N, size=100))
    r = r[: c.size].copy()  # make sure same length
    c = c[: r.size].copy()
    expected_rows = r.copy()
    np.random.shuffle(r)
    np.random.shuffle(c)
    A = Matrix.from_values(r, c, r, nrows=N, ncols=N)
    rows, cols, values = A.to_values(sort=False)
    A = Matrix.from_values(r, c, r, nrows=N, ncols=N)
    rows, cols, values = A.to_values(sort=True)
    assert_array_equal(rows, expected_rows)
    rows, cols, values = A.T.to_values(sort=True)
    assert_array_equal(cols, expected_rows)


def test_get(A):
    assert compute(A.get(0, 0)) is None
    assert A.get(0, 0, "mittens") == "mittens"
    assert A.get(0, 1) == 2
    assert compute(A.T.get(0, 1)) is None
    assert A.T.get(1, 0) == 2
    assert A.get(0, 1, "mittens") == 2
    assert type(compute(A.get(0, 1))) is int
    with pytest.raises(ValueError):
        # Not yet supported
        A.get(0, [0, 1])


@autocompute
def test_bool_as_mask(A):
    expected = select.value(A < 3).new()
    expected *= 3
    A(A < 3, binary.plus, replace=True) << A + A
    assert A.isequal(expected)


def test_ss_serialize(A):
    for compression, level, nthreads in itertools.product(
        [None, "none", "default", "lz4", "lz4hc"], [None, 1, 5, 9], [None, -1, 1, 10]
    ):
        if level is not None and compression != "lz4hc":
            with pytest.raises(TypeError, match="level argument"):
                A.ss.serialize(compression, level, nthreads=nthreads)
            continue
        a = A.ss.serialize(compression, level, nthreads=nthreads)
        C = Matrix.ss.deserialize(a, nthreads=nthreads)
        assert A.isequal(C, check_dtype=True)
    b = a.tobytes()
    C = Matrix.ss.deserialize(b)
    assert A.isequal(C, check_dtype=True)
    with pytest.raises(ValueError, match="compression argument"):
        A.ss.serialize("bad")
    with pytest.raises(ValueError, match="level argument"):
        A.ss.serialize("lz4hc", -1)
    with pytest.raises(ValueError, match="level argument"):
        A.ss.serialize("lz4hc", 0)
    with pytest.raises(InvalidObject):
        Matrix.ss.deserialize(a[:-5])

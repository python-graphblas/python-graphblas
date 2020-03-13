import pytest
from grblas import lib, ffi
from grblas import Matrix, Vector, Scalar
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, descriptor
from grblas.base import Updater
from grblas.exceptions import IndexOutOfBound, DimensionMismatch

def test_new_from_values_dtype_resolving():
    u = Vector.new_from_values([0,1,2], [1,2,3], dtype=dtypes.INT32)
    assert u.dtype == 'INT32'
    u = Vector.new_from_values([0,1,2], [1,2,3], dtype='INT32')
    assert u.dtype == dtypes.INT32
    M = Matrix.new_from_values([0,1,2], [2,0,1], [0,2,3], dtype=dtypes.UINT8)
    assert M.dtype == 'UINT8'
    M = Matrix.new_from_values([0,1,2], [2,0,1], [0,2,3], dtype=float)
    assert M.dtype == dtypes.FP64

def test_new_from_values_invalid_dtype():
    with pytest.raises(OverflowError):
        Matrix.new_from_values([0,1,2], [2,0,1], [0,2,3], dtype=dtypes.BOOL)

def test_resolve_ops_using_output_dtype():
    # C[:] = A.ewise_mult(B, BinaryOp.PLUS) <-- PLUS should use same dtype as C, not A or B
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3], dtype=dtypes.INT64)
    v = Vector.new_from_values([0, 1, 3], [0.1, 0.1, 0.1], dtype='FP64')
    w = Vector.new_from_type('FP32', u.size)
    w << u.ewise_mult(v, BinaryOp.PLUS)
    # Element[0] should be 1.1; check approximate equality
    assert abs(w[0].value - 1.1) < 1e-6

def test_order_of_updater_params_does_not_matter():
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3])
    mask = Vector.new_from_values([0, 3], [True, True])
    accum = BinaryOp.PLUS
    result = Vector.new_from_values([0, 3], [5, 10])
    # mask, accum, replace=
    v = Vector.new_from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask, accum, replace=True) << u.ewise_mult(u, BinaryOp.TIMES)
    assert v == result
    # accum, mask, replace=
    v = Vector.new_from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask, replace=True) << u.ewise_mult(u, BinaryOp.TIMES)
    assert v == result
    # accum, mask=, replace=
    v = Vector.new_from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask=mask, replace=True) << u.ewise_mult(u, BinaryOp.TIMES)
    assert v == result
    # mask, accum=, replace=
    v = Vector.new_from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask, accum=accum, replace=True) << u.ewise_mult(u, BinaryOp.TIMES)
    assert v == result
    # replace=, mask=, accum=
    v = Vector.new_from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(replace=True, mask=mask, accum=accum) << u.ewise_mult(u, BinaryOp.TIMES)
    assert v == result

def test_already_resolved_ops_allowed_in_updater():
    # C[BinaryOp.PLUS['FP64']] = ...
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3])
    u(BinaryOp.PLUS['INT64']) << u.ewise_mult(u, BinaryOp.TIMES['INT64'])
    result = Vector.new_from_values([0, 1, 3], [2, 6, 12])
    assert u == result

def test_updater_returns_updater():
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3])
    y = u(accum=BinaryOp.TIMES)
    assert isinstance(y, Updater)
    z = y << u.apply(UnaryOp.AINV)
    assert z is None
    assert isinstance(y, Updater)
    final_result = Vector.new_from_values([0, 1, 3], [-1, -4, -9])
    assert u == final_result

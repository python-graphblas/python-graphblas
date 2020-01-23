import pytest
from grblas import lib, ffi
from grblas import Matrix, Vector, Scalar
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, descriptor
from grblas import REPLACE
from grblas.exceptions import IndexOutOfBound, DimensionMismatch

def test_new_from_values_dtype_resolving():
    data = [[0,1,2], [1,2,3]]
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
    w[:] = u.ewise_mult(v, BinaryOp.PLUS)
    # Element[0] should be 1.1; check approximate equality
    assert abs(w.element[0] - 1.1) < 1e-6

def test_order_of_resolve_params_does_not_matter():
    # C[mask, accum, REPLACE] = ...
    # C[accum, REPLACE, mask] = ...
    # C[REPLACE, accum, mask] = ...
    # etc.
    from itertools import permutations
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3])
    mask = Vector.new_from_values([0, 3], [True, True])
    result = Vector.new_from_values([0, 3], [5, 10])
    for params in permutations([REPLACE, mask, BinaryOp.PLUS], 3):
        v = Vector.new_from_values([0,1,2,3], [4,3,2,1])
        v[params] = u.ewise_mult(u, BinaryOp.TIMES)
        assert v == result

def test_already_resolved_ops_allowed_in_resolve():
    # C[BinaryOp.PLUS['FP64']] = ...
    u = Vector.new_from_values([0, 1, 3], [1, 2, 3])
    u[BinaryOp.PLUS['INT64']] = u.ewise_mult(u, BinaryOp.TIMES['INT64'])
    result = Vector.new_from_values([0, 1, 3], [2, 6, 12])
    assert u == result

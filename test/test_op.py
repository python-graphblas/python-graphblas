import pytest
from _grblas import lib
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, ops

def test_unaryop():
    assert UnaryOp.AINV['INT32'] == lib.GrB_AINV_INT32
    assert UnaryOp.AINV[dtypes.UINT16] == lib.GrB_AINV_UINT16

def test_binaryop():
    assert BinaryOp.PLUS['INT32'] == lib.GrB_PLUS_INT32
    assert BinaryOp.PLUS[dtypes.UINT16] == lib.GrB_PLUS_UINT16

def test_monoid():
    assert Monoid.MAX['INT32'] == lib.GxB_MAX_INT32_MONOID
    assert Monoid.MAX[dtypes.UINT16] == lib.GxB_MAX_UINT16_MONOID

def test_semiring():
    assert Semiring.MIN_PLUS['INT32'] == lib.GxB_MIN_PLUS_INT32
    assert Semiring.MIN_PLUS[dtypes.UINT16] == lib.GxB_MIN_PLUS_UINT16

def test_find_opclass_unaryop():
    assert ops.find_opclass(UnaryOp.MINV) == 'UnaryOp'
    assert ops.find_opclass(lib.GrB_MINV_INT64) == 'UnaryOp'

def test_find_opclass_binaryop():
    assert ops.find_opclass(BinaryOp.TIMES) == 'BinaryOp'
    assert ops.find_opclass(lib.GrB_TIMES_INT64) == 'BinaryOp'

def test_find_opclass_monoid():
    assert ops.find_opclass(Monoid.MAX) == 'Monoid'
    assert ops.find_opclass(lib.GxB_MAX_INT64_MONOID) == 'Monoid'

def test_find_opclass_semiring():
    assert ops.find_opclass(Semiring.PLUS_PLUS) == 'Semiring'
    assert ops.find_opclass(lib.GxB_PLUS_PLUS_INT64) == 'Semiring'

def test_find_opclass_invalid():
    assert ops.find_opclass('foobar') == ops.UNKNOWN_OPCLASS
    assert ops.find_opclass(lib.GrB_INP0) == ops.UNKNOWN_OPCLASS

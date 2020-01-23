import pytest
from grblas import lib, ffi
from grblas import Scalar
from grblas import UnaryOp, BinaryOp, Monoid, Semiring
from grblas import dtypes, descriptor

@pytest.fixture
def s():
    return Scalar.new_from_value(5)


def test_new_from_type():
    s = Scalar.new_from_type(dtypes.INT8)
    assert s.dtype == 'INT8'
    assert s.value == 0  # must hold a value; initialized to 0
    s2 = Scalar.new_from_type(bool)
    assert s2.dtype == 'BOOL'
    assert s2.value == False  # must hold a value; initialized to False

def test_new_from_existing(s):
    s2 = Scalar.new_from_existing(s)
    assert s2.dtype == s.dtype
    assert s2.value == s.value

def test_new_from_value():
    s = Scalar.new_from_value(False)
    assert s.dtype == bool
    assert s.value == False
    s2 = Scalar.new_from_value(-1.1)
    assert s2.dtype == 'FP64'
    assert s2.value == -1.1

def test_clear(s):
    assert s.value == 5
    s.clear()
    assert s.value == 0  # must hold a value; reset to 0
    s2 = Scalar.new_from_value(True)
    assert s2.value == True
    s2.clear()
    assert s2.value == False  # must hold a value; reset to False

def test_equal(s):
    assert s.value == 5
    assert s == 5
    assert s != 27

def test_truthy(s):
    assert s, 's did not register as truthy'
    with pytest.raises(AssertionError):
        assert not s
    s2 = Scalar.new_from_value(True)
    assert s2
    with pytest.raises(AssertionError):
        assert not s2

def test_get_value(s):
    assert s.value == 5

def test_set_value(s):
    assert s.value == 5
    s.value = 12
    assert s.value == 12
    with pytest.raises(TypeError):
        s.value = 12.5

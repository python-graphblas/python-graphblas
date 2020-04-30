import pytest
from grblas import Scalar
from grblas import dtypes


@pytest.fixture
def s():
    return Scalar.from_value(5)


def test_new():
    s = Scalar.new(dtypes.INT8)
    assert s.dtype == 'INT8'
    assert s.value is None
    s.is_empty = False
    assert s.value == 0  # must hold a value; initialized to 0
    s2 = Scalar.new(bool)
    assert s2.dtype == 'BOOL'
    assert s2.value is None
    s2.is_empty = False
    assert s2.value is False  # must hold a value; initialized to False


def test_dup(s):
    s2 = s.dup()
    assert s2.dtype == s.dtype
    assert s2.value == s.value
    s3 = s.dup()
    assert s3.dtype == s.dtype
    assert s3.value == s.value


def test_from_value():
    s = Scalar.from_value(False)
    assert s.dtype == bool
    assert s.value is False
    s2 = Scalar.from_value(-1.1)
    assert s2.dtype == 'FP64'
    assert s2.value == -1.1


def test_clear(s):
    assert s.value == 5
    assert not s.is_empty
    s.clear()
    assert s.value is None
    assert s.is_empty
    s2 = Scalar.from_value(True)
    assert s2.value is True
    assert not s2.is_empty
    s2.clear()
    assert s2.value is None
    assert s2.is_empty


def test_equal(s):
    assert s.value == 5
    assert s == 5
    assert s != 27


def test_truthy(s):
    assert s, 's did not register as truthy'
    with pytest.raises(AssertionError):
        assert not s
    s2 = Scalar.from_value(True)
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


def test_isequal(s):
    assert s.isequal(5)
    assert s.isequal(5.0)
    assert s.isequal(5.0, check_dtype=True)  # No explicit dtype given; should we check?
    assert not s.isequal(None)
    with pytest.raises(TypeError):
        s.isequal(object())
    assert not s.isequal(Scalar.from_value(None, dtype=s.dtype))
    t = Scalar.from_value(5, dtype='INT8')
    assert s.isequal(t)
    assert not s.isequal(t, check_dtype=True)
    assert Scalar.from_value(None, dtype='INT8').isequal(Scalar.from_value(None, dtype='INT16'))


def test_isclose():
    s = Scalar.from_value(5.0)
    assert s.isclose(5)
    assert s.isclose(5, check_dtype=True)  # No explicit dtype given; should we check?
    assert not s.isclose(6)
    assert s.isclose(5.000000001)
    assert not s.isclose(5.000000001, rel_tol=1e-10)
    assert not s.isclose(None)
    with pytest.raises(TypeError):
        s.isclose(object())
    assert not s.isclose(Scalar.from_value(5), check_dtype=True)
    assert not s.isclose(Scalar.from_value(None, dtype=s.dtype))
    assert Scalar.from_value(None, dtype='FP64').isequal(Scalar.from_value(None, dtype='FP32'))

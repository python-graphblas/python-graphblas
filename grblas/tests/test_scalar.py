import pytest
from grblas import Scalar
from grblas import dtypes, binary
from grblas.scalar import _CScalar


@pytest.fixture
def s():
    return Scalar.from_value(5)


def test_new():
    s = Scalar.new(dtypes.INT8)
    assert s.dtype == "INT8"
    assert s.value is None
    s.value = 0
    assert s.is_empty is False
    s2 = Scalar.new(bool)
    assert s2.dtype == "BOOL"
    assert s2.value is None
    assert bool(s2) is False
    s2.value = False
    assert s2.is_empty is False


def test_dup(s):
    s2 = s.dup()
    assert s2.dtype == s.dtype
    assert s2.value == s.value
    s3 = s.dup()
    assert s3.dtype == s.dtype
    assert s3.value == s.value
    # extended functionality
    s4 = Scalar.from_value(-2.5, dtype=dtypes.FP64)
    s_empty = Scalar.new(dtypes.FP64)
    s_unempty = Scalar.from_value(0.0)
    for dtype, val in [
        ("INT8", -2),
        ("INT16", -2),
        ("INT32", -2),
        ("UINT8", 2 ** 8 - 2),
        ("UINT16", 2 ** 16 - 2),
        ("UINT32", 2 ** 32 - 2),
        ("UINT64", 2 ** 64 - 2),
        ("BOOL", True),
        ("FP32", -2.5),
    ]:
        s5 = s4.dup(dtype=dtype)
        assert s5.dtype == dtype and s5.value == val
        s6 = s_empty.dup(dtype=dtype)
        assert s6.is_empty
        assert s6.value is None
        s7 = s_unempty.dup(dtype=dtype)
        assert not s7.is_empty
        assert s7.value is not None


def test_from_value():
    s = Scalar.from_value(False)
    assert s.dtype == bool
    assert s.value is False
    s2 = Scalar.from_value(-1.1)
    assert s2.dtype == "FP64"
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
    assert s, "s did not register as truthy"
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
    t = Scalar.from_value(5, dtype="INT8")
    assert s.isequal(t)
    assert not s.isequal(t, check_dtype=True)
    assert Scalar.from_value(None, dtype="INT8").isequal(Scalar.from_value(None, dtype="INT16"))


@pytest.mark.slow
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
    assert Scalar.from_value(None, dtype="FP64").isequal(Scalar.from_value(None, dtype="FP32"))


def test_nvals(s):
    assert s.nvals == 1
    s.clear()
    assert s.nvals == 0


def test_unsupported_ops(s):
    with pytest.raises(AttributeError):
        s.S
    with pytest.raises(AttributeError):
        s.V
    with pytest.raises(AttributeError):
        s.T
    with pytest.raises(TypeError, match="is not subscriptable"):
        s[0]
    with pytest.raises(TypeError, match="does not support"):
        s[0] = 0
    with pytest.raises(TypeError, match="doesn't support"):
        del s[0]


def test_is_empty(s):
    with pytest.raises(AttributeError, match="can't set attribute"):
        s.is_empty = True


def test_update(s):
    s << 1
    assert s == 1
    s << Scalar.from_value(2)
    assert s == 2
    s << Scalar.from_value(3)
    assert s == 3
    with pytest.raises(TypeError, match="an integer is required"):
        s << Scalar.from_value(4.4)
    s() << 5
    assert s == 5
    with pytest.raises(TypeError, match="is not supported"):
        s(accum=binary.plus) << 6
    with pytest.raises(TypeError, match="Mask not allowed for Scalars"):
        s(s)


def test_not_hashable(s):
    with pytest.raises(TypeError, match="unhashable type"):
        {s}
    with pytest.raises(TypeError, match="unhashable type"):
        hash(s)


def test_cscalar():
    c1 = _CScalar(Scalar.from_value(5))
    assert c1 == _CScalar(Scalar.from_value(5))
    assert c1 == 5
    assert c1 != _CScalar(Scalar.from_value(6))
    assert c1 != 6
    assert repr(c1) == "5"
    assert c1._repr_html_() == c1.scalar._repr_html_()

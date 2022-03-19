import inspect
import pickle
import sys
import weakref

import numpy as np
import pytest

import grblas as gb
from grblas import binary, dtypes, replace

from .conftest import autocompute, compute

from grblas import Matrix, Scalar, Vector  # isort:skip


@pytest.fixture
def s():
    return Scalar.from_value(5)


def test_new():
    s = Scalar.new(dtypes.INT8)
    assert s.dtype == "INT8"
    assert compute(s.value) is None
    s.value = 0
    assert compute(s.is_empty) is False
    s2 = Scalar.new(bool)
    assert s2.dtype == "BOOL"
    assert compute(s2.value) is None
    assert bool(s2) is False
    s2.value = False
    assert compute(s2.is_empty) is False


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
        ("UINT8", 2**8 - 2),
        ("UINT16", 2**16 - 2),
        ("UINT32", 2**32 - 2),
        ("UINT64", 2**64 - 2),
        ("BOOL", True),
        ("FP32", -2.5),
    ]:
        s5 = s4.dup(dtype=dtype)
        assert s5.dtype == dtype and s5.value == val
        s6 = s_empty.dup(dtype=dtype)
        assert s6.is_empty
        assert compute(s6.value) is None
        s7 = s_unempty.dup(dtype=dtype)
        assert not s7.is_empty
        assert compute(s7.value) is not None


def test_from_value():
    s = Scalar.from_value(False)
    assert s.dtype == bool
    assert compute(s.value) is False
    s2 = Scalar.from_value(-1.1)
    assert s2.dtype == "FP64"
    assert s2.value == -1.1
    s3 = Scalar.from_value(s, dtype="INT64")
    assert s3.dtype == "INT64"
    assert s3.value == 0


def test_clear(s):
    assert s.value == 5
    assert not s.is_empty
    s.clear()
    assert compute(s.value) is None
    assert s.is_empty
    s2 = Scalar.from_value(True)
    assert compute(s2.value) is True
    assert not s2.is_empty
    s2.clear()
    assert compute(s2.value) is None
    assert s2.is_empty


def test_equal(s):
    assert s.value == 5
    assert s == 5
    assert s != 27


def test_casting(s):
    assert int(s) == 5
    assert type(int(s)) is int
    assert float(s) == 5.0
    assert type(float(s)) is float
    assert range(s) == range(5)
    assert complex(s) == complex(5)
    assert type(complex(s)) is complex


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
    if s._is_cscalar:
        with pytest.raises(TypeError):
            s.value = 12.5
    else:
        s.value = 12.5
        assert s == 12


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
    assert Scalar.from_value(None, dtype="FP64").isclose(Scalar.from_value(None, dtype="FP32"))


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
    if s._is_cscalar:
        with pytest.raises(TypeError, match="an integer is required"):
            s << Scalar.from_value(4.4)
    else:
        s << Scalar.from_value(4.4)
        assert s == 4
    s() << 5
    assert s == 5
    with pytest.raises(TypeError, match="is not supported"):
        s(accum=binary.plus) << 6
    with pytest.raises(TypeError, match="Mask not allowed for Scalars"):
        s(s)
    with pytest.raises(TypeError, match="input_mask not allowed for Scalars"):
        s(input_mask=s)
    with pytest.raises(TypeError, match="'replace' argument may not be True for Scalar"):
        s(replace=True)
    with pytest.raises(TypeError, match="'replace' argument may not be True for Scalar"):
        s(replace)


def test_not_hashable(s):
    with pytest.raises(TypeError, match="unhashable type"):
        {s}
    with pytest.raises(TypeError, match="unhashable type"):
        hash(s)


def test_pickle(s):
    blob = pickle.dumps(s)
    s2 = pickle.loads(blob)
    assert s.isequal(s2, check_dtype=True)
    assert s.name == s2.name


def test_weakref(s):
    d = weakref.WeakValueDictionary()
    d["s"] = s
    assert d["s"] is s


def test_scalar_to_numpy(s):
    for a, b in [
        (np.array(s), np.array(5, dtype=np.int64)),
        (np.array(s, dtype=float), np.array(5.0)),
        (np.array([s]), np.array([5], dtype=np.int64)),
        (np.array([s], dtype=float), np.array([5.0])),
        (np.array([s, s]), np.array([5, 5], dtype=np.int64)),
        (np.array([s, s], dtype=float), np.array([5.0, 5.0])),
    ]:
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype, (a, b)
        assert a.shape == b.shape


def test_neg():
    for dtype in sorted(
        (dtype for dtype in vars(dtypes).values() if isinstance(dtype, dtypes.DataType)),
        key=lambda x: x.name,
        reverse=True,  # XXX: segfault when False!!!
    ):
        s = Scalar.from_value(1, dtype=dtype)
        empty = Scalar.new(dtype)
        if dtype.name == "BOOL" or dtype.name.startswith("U"):
            with pytest.raises(TypeError, match="The negative operator, `-`, is not supported"):
                -s
            with pytest.raises(TypeError, match="The negative operator, `-`, is not supported"):
                -empty
        else:
            minus_s = Scalar.from_value(-1, dtype=dtype)
            assert s == -minus_s
            assert (-s).value == minus_s.value
            assert empty == -empty
            assert compute((-empty).value) is None


def test_invert():
    empty = Scalar.new(bool)
    assert empty == ~empty
    assert compute((~empty).value) is None
    not_s = Scalar.from_value(0, dtype=bool)
    s = Scalar.from_value(1, dtype=bool)
    assert ~s == not_s
    assert (~s).value == not_s.value
    assert not (s.value) == not_s.value
    bad = Scalar.new(int)
    with pytest.raises(TypeError, match="The invert operator"):
        ~bad


def test_wait(s):
    s.wait()


@autocompute
def test_expr_is_like_scalar(s):
    v = Vector.from_values([1], [2])
    attrs = {attr for attr, val in inspect.getmembers(s)}
    expr_attrs = {attr for attr, val in inspect.getmembers(v.inner(v))}
    infix_attrs = {attr for attr, val in inspect.getmembers(v @ v)}
    # Should we make any of these raise informative errors?
    expected = {
        "__call__",
        "__del__",
        "__imatmul__",
        "__lshift__",
        "_carg",
        "_deserialize",
        "_expr_name",
        "_expr_name_html",
        "_name_counter",
        "_update",
        "clear",
        "from_pygraphblas",
        "from_value",
        "update",
    }
    if s.is_cscalar:
        expected.add("_empty")
    assert attrs - expr_attrs == expected
    assert attrs - infix_attrs == expected


@autocompute
def test_index_expr_is_like_scalar(s):
    v = Vector.from_values([1], [2])
    attrs = {attr for attr, val in inspect.getmembers(s)}
    expr_attrs = {attr for attr, val in inspect.getmembers(v[0])}
    # Should we make any of these raise informative errors?
    expected = {
        "__del__",
        "__imatmul__",
        "_carg",
        "_deserialize",
        "_expr_name",
        "_expr_name_html",
        "_name_counter",
        "_update",
        "clear",
        "from_pygraphblas",
        "from_value",
    }
    if s.is_cscalar:
        expected.add("_empty")
    assert attrs - expr_attrs == expected


def test_ndim(s):
    assert s.ndim == 0
    v = Vector.from_values([1], [2])
    assert v.inner(v).ndim == 0
    assert (v @ v).ndim == 0


@pytest.mark.skipif("not dtypes._supports_complex")
# @pytest.mark.parametrize("dtype", ["FC32", "FC64"])  # This segfaults
@pytest.mark.parametrize("dtype", ["FC64", "FC32"])
def test_scalar_complex(dtype):
    s = Scalar.new(dtype)
    assert s.is_empty
    s.value = 1
    assert s == 1
    assert s.value == 1
    s.value = 2j  # segfault here!!!
    assert s == 2j
    assert s.value == 2j
    s << 3
    assert s == 3
    assert s.value == 3
    s << 4j
    assert s == 4j
    assert s.value == 4j
    s << 5 + 6j
    assert s == 5 + 6j
    assert s.value == 5 + 6j
    s.value = 7 + 8j
    assert s == 7 + 8j
    assert s.value == 7 + 8j
    s = Scalar.from_value(1j, dtype)
    assert s.dtype == dtype
    assert s == 1j
    assert s.value == 1j
    s = Scalar.from_value(2 + 3j, dtype)
    assert s.dtype == dtype
    assert s == 2 + 3j
    assert s.value == 2 + 3j


@autocompute
def test_scalar_expr(s):
    v = Vector.from_values([1], [2])
    expr = v.inner(v)
    t = expr._new_scalar(s.dtype)
    assert t.is_cscalar is s.is_cscalar
    assert (v @ v).is_cscalar is s.is_cscalar
    assert (v @ v).is_grbscalar is s.is_grbscalar
    assert (v @ v).new(is_cscalar=True).is_cscalar is True
    assert (v @ v).new(is_cscalar=False).is_cscalar is False  # pragma: is_grbscalar
    assert v[1].new(is_cscalar=True).is_cscalar is True
    assert v[1].new(is_cscalar=False).is_cscalar is False  # pragma: is_grbscalar
    expr = v.reduce()
    assert expr == 2  # Autocompute and cache value
    assert expr.new().is_cscalar is False  # b/c default reduce is to allow empty
    assert expr == 2  # Autocompute and cache value
    assert expr.new(is_cscalar=True).is_cscalar is True  # We should respect keyword


def test_sizeof(s):
    assert 1 < sys.getsizeof(s) < 1000


def test_concat(s):
    empty = Scalar.new(int)
    v = gb.ss.concat([s, s, empty])
    expected = Vector.from_values([0, 1], 5, size=3)
    assert v.isequal(expected)
    A = gb.ss.concat([[s, s, empty]])
    expected = Matrix.from_values([0, 0], [0, 1], 5, nrows=1, ncols=3)
    assert A.isequal(expected)
    A = gb.ss.concat([[s], [s], [empty]])
    expected = Matrix.from_values([0, 1], [0, 0], 5, nrows=3, ncols=1)
    assert A.isequal(expected)

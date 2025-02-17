import inspect
import pickle
import random
import sys
import types
import weakref

import numpy as np
import pytest

import graphblas as gb
from graphblas import backend, binary, dtypes, monoid, replace, select, unary
from graphblas.exceptions import EmptyObject

from .conftest import autocompute, compute, pypy

from graphblas import Matrix, Scalar, Vector  # isort:skip (for dask-graphblas)

suitesparse = backend == "suitesparse"


@pytest.fixture
def s():
    return Scalar.from_value(5)


def test_new():
    s = Scalar(dtypes.INT8)
    assert s.dtype == "INT8"
    assert compute(s.value) is None
    s.value = 0
    assert compute(s.is_empty) is False
    s2 = Scalar(bool)
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
    s_empty = Scalar(dtypes.FP64)
    s_unempty = Scalar.from_value(0.0)
    if s_empty.is_cscalar:
        # NumPy <2 wraps around; >=2 raises OverflowError
        uint_data = [
            ("UINT8", 2**8 - 2),
            ("UINT16", 2**16 - 2),
            ("UINT32", 2**32 - 2),
            ("UINT64", 2**64 - 2),
        ]
    else:
        # SuiteSparse clips
        uint_data = [
            ("UINT8", 0),
            ("UINT16", 0),
            ("UINT32", 0),
            ("UINT64", 0),
        ]
    for dtype, val in [
        ("INT8", -2),
        ("INT16", -2),
        ("INT32", -2),
        ("BOOL", True),
        ("FP32", -2.5),
        *uint_data,
    ]:
        if dtype.startswith("UINT") and s_empty.is_cscalar and not np.__version__.startswith("1."):
            with pytest.raises(OverflowError, match="out of bounds for uint"):
                s4.dup(dtype=dtype, name="s5")
            continue
        s5 = s4.dup(dtype=dtype, name="s5")
        assert s5.dtype == dtype
        assert s5.value == val
        s6 = s_empty.dup(dtype=dtype, name="s6")
        assert s6.is_empty
        assert compute(s6.value) is None
        s7 = s_unempty.dup(dtype=dtype, name="s7")
        assert not s7.is_empty
        assert compute(s7.value) is not None


def test_dup_clear(s):
    for is_cscalar in [True, False, None]:
        s2 = s.dup(clear=True, is_cscalar=is_cscalar)
        assert s2.dtype == s.dtype
        assert compute(s2.value) is None
        s3 = s.dup("FP64", clear=True, is_cscalar=is_cscalar)
        assert s3.dtype == "FP64"
        assert compute(s3.value) is None


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
    assert isinstance(int(s), int)
    assert float(s) == 5.0
    assert isinstance(float(s), float)
    assert range(s) == range(5)
    with pytest.raises(AttributeError, match="Scalar .* only .*__index__.*integral"):
        range(s.dup(float))
    assert complex(s) == complex(5)
    assert isinstance(complex(s), complex)


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
    with pytest.raises(TypeError, match="doesn't support|does not support"):
        del s[0]


def test_is_empty(s):
    with pytest.raises(AttributeError, match="can't set attribute|object has no setter"):
        s.is_empty = True


def test_update(s):
    s << 1
    assert s == 1
    s << Scalar.from_value(2)
    assert s == 2
    s << Scalar.from_value(3)
    assert s == 3
    if s._is_cscalar:
        with pytest.raises(TypeError, match="an integer is required|expected integer"):
            s << Scalar.from_value(4.4)
    else:
        s << Scalar.from_value(4.4)
        assert s == 4
    s() << 5
    assert s == 5
    # with pytest.raises(TypeError, match="is not supported"):
    s(accum=binary.plus) << 6  # Now okay
    assert s == 11
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
        _ = {s}
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


@autocompute
def test_neg():
    for dtype in sorted(
        (
            dtype
            for attr, dtype in vars(dtypes).items()
            if isinstance(dtype, dtypes.DataType) and attr not in {"_INDEX"}
        ),
        key=lambda x: x.name,
        reverse=random.choice([False, True]),  # used to segfault when False
    ):
        s = Scalar.from_value(1, dtype=dtype)
        empty = Scalar(dtype)
        if dtype._is_udt:
            with pytest.raises(KeyError, match="ainv does not work with"):
                -s
            with pytest.raises(KeyError, match="ainv does not work with"):
                -empty
        else:
            minus_s = Scalar.from_value(-1, dtype=dtype, is_cscalar=False)  # pragma: is_grbscalar
            assert s == -minus_s
            assert (-s).value == minus_s.value
            assert empty == -empty
            assert compute((-empty).value) is None


@autocompute
def test_invert():
    empty = Scalar(bool)
    assert empty == ~empty
    assert compute((~empty).value) is None
    not_s = Scalar.from_value(0, dtype=bool)
    s = Scalar.from_value(1, dtype=bool)
    assert ~s == not_s
    assert (~s).value == not_s.value
    compare = s.value == not_s.value
    assert not compare
    assert s.value != not_s.value
    bad = Scalar(int)
    with pytest.raises(TypeError, match="The invert operator"):
        ~bad


def test_wait(s):
    s.wait()
    s.wait("materialize")
    s.wait("complete")
    with pytest.raises(ValueError, match="`how` argument must be"):
        s.wait("badmode")


@autocompute
def test_expr_is_like_scalar(s):
    v = Vector.from_coo([1], [2])
    t = s.dup(bool)
    attrs = {attr for attr, val in inspect.getmembers(s)}
    expr_attrs = {attr for attr, val in inspect.getmembers(v.inner(v))}
    infix_attrs = {attr for attr, val in inspect.getmembers(v @ v)}
    scalar_infix_attrs = {attr for attr, val in inspect.getmembers(t & t)}
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
        "_from_obj",
        "_name_counter",
        "_update",
        "clear",
        "from_value",
        "update",
    }
    if s.is_cscalar:
        expected.add("_empty")
    ignore = {"__sizeof__", "_ewise_add", "_ewise_mult", "_ewise_union"}
    assert attrs - expr_attrs - ignore == expected, (
        "If you see this message, you probably added a method to Scalar.  You may need to "
        "add an entry to `scalar` set in `graphblas.core.automethods` "
        "and then run `python -m graphblas.core.automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas.core.infixmethods`."
    )
    assert attrs - infix_attrs - ignore == expected
    assert attrs - scalar_infix_attrs - ignore == expected
    # Make sure signatures actually match. `expr.dup` has `**opts`
    skip = {"__init__", "__repr__", "_repr_html_", "dup"}
    for expr in [v.inner(v), v @ v, t & t]:
        print(type(expr).__name__)
        for attr, val in inspect.getmembers(expr):
            if attr in skip or not isinstance(val, types.MethodType) or not hasattr(s, attr):
                continue
            val2 = getattr(s, attr)
            assert inspect.signature(val) == inspect.signature(val2), attr
            assert val.__doc__ == val2.__doc__


@autocompute
def test_index_expr_is_like_scalar(s):
    v = Vector.from_coo([1], [2])
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
        "_from_obj",
        "_name_counter",
        "_update",
        "clear",
        "from_value",
    }
    if s.is_cscalar:
        expected.add("_empty")
    ignore = {"__sizeof__", "_ewise_add", "_ewise_mult", "_ewise_union"}
    assert attrs - expr_attrs - ignore == expected, (
        "If you see this message, you probably added a method to Scalar.  You may need to "
        "add an entry to `scalar` set in `graphblas.core.automethods` "
        "and then run `python -m graphblas.core.automethods`.  If you're messing with infix "
        "methods, then you may need to run `python -m graphblas.core.infixmethods`."
    )
    # Make sure signatures actually match. `update` has different docstring.
    skip = {"__call__", "__init__", "__repr__", "_repr_html_", "update", "dup"}
    for attr, val in inspect.getmembers(v[0]):
        if attr in skip or not isinstance(val, types.MethodType) or not hasattr(s, attr):
            continue
        val2 = getattr(s, attr)
        assert inspect.signature(val) == inspect.signature(val2), attr
        assert val.__doc__ == val2.__doc__


@autocompute
def test_dup_expr(s):
    v = Vector.from_coo([1], [2])
    result = (s + s).dup()
    assert result.isequal(2 * s)
    result = (s + s).dup(is_cscalar=not s._is_cscalar)
    assert result.isequal(2 * s)
    assert result._is_cscalar != s._is_cscalar
    result = (s + s).dup(dtype=float)
    assert result.isequal(10.0, check_dtype=True)
    result = (s + s).dup(clear=True)
    assert result.isequal(s.dup(clear=True))
    b = s.dup(bool)
    result = (b | b).dup()
    assert result.isequal(b, check_dtype=True)
    result = (b | b).dup(clear=True)
    assert result.isequal(b.dup(clear=True), check_dtype=True)
    result = (b | b).dup(float)
    assert result.isequal(b.dup(float), check_dtype=True)
    result = (v @ v).dup()
    assert result.isequal(4)
    result = v[1].dup()
    assert result.isequal(2, check_dtype=True)
    result = v[1].dup(float)
    assert result.isequal(2.0, check_dtype=True)
    result = v[1].dup(clear=True)
    assert result.isequal(v[0])


def test_ndim(s):
    assert s.ndim == 0
    v = Vector.from_coo([1], [2])
    assert v.inner(v).ndim == 0
    assert (v @ v).ndim == 0


@pytest.mark.skipif("not dtypes._supports_complex")
# @pytest.mark.parametrize("dtype", ["FC32", "FC64"])  # This segfaults
@pytest.mark.parametrize("dtype", ["FC64", "FC32"])
def test_scalar_complex(dtype):
    s = Scalar(dtype)
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
    v = Vector.from_coo([1], [2])
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
    if (suitesparse or s._is_cscalar) and not pypy:
        assert 1 < sys.getsizeof(s) < 1000
    else:
        with pytest.raises(TypeError):  # flakey coverage (why?!)
            sys.getsizeof(s)


def test_ewise_union(s):
    t = Scalar(int)
    result = s.ewise_union(t, binary.plus, 10, 20).new()
    assert result == 25
    with pytest.raises(EmptyObject):
        s.ewise_union(t, binary.plus, 10, t).new()
    result = s.ewise_union(s, monoid.plus, 10, 20).new()
    assert result == 10
    result = t.ewise_union(t, binary.plus, 10, 20).new()
    assert result.is_empty
    with pytest.raises(EmptyObject):
        t.ewise_union(t, binary.plus, t, t).new()
    v = Vector(int, 2)
    with pytest.raises(TypeError, match="Literal scalars also"):
        s.ewise_union(v, binary.plus, 10, 20)
    with pytest.raises(TypeError, match="Literal scalars also"):
        s.ewise_union(t, binary.plus, v, 20)
    with pytest.raises(TypeError, match="Literal scalars also"):
        s.ewise_union(t, binary.plus, 10, v)


def test_ewise_mult_add(s):
    assert s.ewise_add(s).new() == 10
    assert s.ewise_mult(s).new() == 25
    v = Vector(int, 2)
    with pytest.raises(TypeError, match="Literal scalars also"):
        s.ewise_add(v)
    with pytest.raises(TypeError, match="Literal scalars also"):
        s.ewise_mult(v)


def test_select(s):
    assert select.value(s < 10).new() == s
    assert select.value(s > 10).new().is_empty
    assert select.valueeq(s, 5).new() == s
    assert select.valuene(5, s).new().is_empty
    with pytest.raises(TypeError):
        select.value(s | s)


@pytest.mark.skipif("not suitesparse")
def test_ss_concat(s):
    empty = Scalar(int)
    v = gb.ss.concat([s, s, empty])
    expected = Vector.from_coo([0, 1], 5, size=3)
    assert v.isequal(expected)
    A = gb.ss.concat([[s, s, empty]])
    expected = Matrix.from_coo([0, 0], [0, 1], 5, nrows=1, ncols=3)
    assert A.isequal(expected)
    A = gb.ss.concat([[s], [s], [empty]])
    expected = Matrix.from_coo([0, 1], [0, 0], 5, nrows=3, ncols=1)
    assert A.isequal(expected)


def test_record_from_dict():
    s = Scalar.from_value(
        {"x": 1, "y": {"a": 2, "b": 3}}, dtype={"x": int, "y": {"a": int, "b": int}}
    )
    assert s == (1, (2, 3))


def test_get(s):
    assert s.get() == 5
    assert s.get("mittens") == 5
    assert isinstance(compute(s.get()), int)
    s.clear()
    assert compute(s.get()) is None
    assert s.get("mittens") == "mittens"


def test_ss_descriptors(s):
    v = Vector.from_coo([0, 2], [10, 20])
    if suitesparse:
        with pytest.raises(ValueError, match="escriptor"):
            v[0].new(bad_opt=True)
        assert v[0].new(nthreads=4) == 10  # ignored, but okay
        with pytest.raises(ValueError, match="escriptor"):
            v.dup(bad_opt=True)
        v.dup(nthreads=4)
        with pytest.raises(ValueError, match="escriptor"):
            s(bad_opt=True) << 1
        s(nthreads=4) << 1  # ignored, but okay
        with pytest.raises(ValueError, match="escriptor"):
            s(bad_opt=True) << s
        s(nthreads=4) << s  # ignored, but okay
    else:
        with pytest.raises(ValueError, match="escriptor"):
            v[0].new(nthreads=4)
        with pytest.raises(ValueError, match="escriptor"):
            s(nthreads=4) << 1


@autocompute
def test_scalar_operators(s):
    assert -s == -5
    assert s + 1 == 6
    assert 1 + s == 6
    assert s - 1 == 4
    assert 1 - s == -4
    assert s * 2 == 10
    assert 2 * s == 10
    assert s * s == 25
    assert s**2 == 25
    assert unary.cos(0) == 1
    assert binary.plus(s | 2) == 7
    assert binary.plus(s, 2) == 7
    assert binary.plus(5, 2) == 7
    assert binary.plus(2, s) == 7
    assert (-s).apply(unary.abs) == 5
    with pytest.raises(TypeError):
        unary.sin(object())
    with pytest.raises(TypeError):
        binary.plus(object(), object())

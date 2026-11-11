"""graphblas.array_api: creation, data types, the Array object, basic indexing and the shape
functions, against numpy.
"""

import copy
import math
import pickle

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp
from graphblas import Matrix, Scalar, Tensor, Vector

from .test_array_api_elementwise import (
    DTYPES,
    assert_array_equal,
    check_invariants,
    inputs,
    make,
    np_call,
)


def test_namespace():
    assert xp.__array_api_version__ == "2025.12"
    for name in xp.__all__:
        assert hasattr(xp, name), name
    assert xp.float64 is np.dtype("float64")
    assert xp.bool == np.dtype(bool)
    x = xp.asarray([1, 2])
    assert x.__array_namespace__() is xp
    assert x.__array_namespace__(api_version="2024.12") is xp
    with pytest.raises(ValueError, match="version"):
        x.__array_namespace__(api_version="2019.12")


# asarray
def test_asarray_stores_what_differs_from_the_fill():
    x = xp.asarray([1.0, 0.0, -0.0, np.nan, 2.0])
    assert x.fill_value == 0.0
    assert x.nnz == 4  # -0.0 and NaN are stored
    assert_array_equal(x, np.array([1.0, 0.0, -0.0, np.nan, 2.0]))
    y = xp.asarray([1.0, np.nan, np.nan], fill_value=np.nan)
    assert y.nnz == 1
    assert_array_equal(y, np.array([1.0, np.nan, np.nan]))
    z = xp.asarray([[1.0, 2.0], [-0.0, 0.0]], fill_value=-0.0)
    assert z.nnz == 3
    assert_array_equal(z, np.array([[1.0, 2.0], [-0.0, 0.0]]))
    c = xp.asarray([complex(0, -0.0), 0j, complex(np.nan, 0)])
    assert c.nnz == 2
    assert_array_equal(c, np.array([complex(0, -0.0), 0j, complex(np.nan, 0)]))


@pytest.mark.parametrize(
    ("obj", "dtype"),
    [
        (True, "bool"),
        (3, "int64"),
        (2.5, "float64"),
        (1j, "complex128"),
        ([1, 2], "int64"),
        ([1.0, 2], "float64"),
        ([True, False], "bool"),
        ([[1, 0], [0, 3]], "int64"),
        ([], "float64"),
        (np.float32(1.5), "float32"),
        (np.arange(4, dtype=np.uint16), "uint16"),
    ],
)
def test_asarray_dtypes(obj, dtype):
    x = xp.asarray(obj)
    assert x.dtype == np.dtype(dtype)
    assert_array_equal(x, np.asarray(obj, dtype=dtype))


def test_asarray_dtype_and_copy():
    x = xp.asarray([1, 2, 3], dtype=xp.float32)
    assert x.dtype == xp.float32
    assert xp.asarray(x) is x
    assert xp.asarray(x, copy=False) is x
    y = xp.asarray(x, copy=True)
    assert y is not x
    y += 1
    assert_array_equal(x, np.array([1, 2, 3], np.float32))
    assert_array_equal(xp.asarray(x, dtype=xp.int8), np.array([1, 2, 3], np.int8))
    with pytest.raises(ValueError, match="copy"):
        xp.asarray(x, dtype=xp.int8, copy=False)
    with pytest.raises(ValueError, match="copy"):
        xp.asarray([1, 2], copy=False)
    with pytest.raises(TypeError):
        xp.asarray([1, 2], dtype=np.float16)
    with pytest.raises(TypeError):
        xp.asarray(np.array(["a"]))
    with pytest.raises(OverflowError):
        xp.asarray([300], dtype=xp.int8)
    with pytest.raises(ValueError, match="device"):
        xp.asarray([1], device="gpu")


def test_asarray_nested_arrays():
    x = xp.asarray([xp.asarray(1.0), xp.asarray(2.0)])
    assert_array_equal(x, np.array([1.0, 2.0]))
    x = xp.asarray([[1, 2], xp.asarray([3, 4])])
    assert_array_equal(x, np.array([[1, 2], [3, 4]]))


def test_asarray_graphblas_objects():
    v = Vector.from_coo([0, 3], [1.5, 2.5], size=5)
    x = xp.asarray(v)
    assert x.nnz == 2
    assert_array_equal(x, np.array([1.5, 0, 0, 2.5, 0]))
    x = xp.asarray(v, fill_value=np.inf)
    assert_array_equal(x, np.array([1.5, np.inf, np.inf, 2.5, np.inf]))
    A = Matrix.from_coo([0, 1], [1, 0], [5, 6], nrows=2, ncols=3)
    assert_array_equal(xp.asarray(A), np.array([[0, 5, 0], [6, 0, 0]]))
    assert_array_equal(xp.asarray(A.T), np.array([[0, 6], [5, 0], [0, 0]]))
    assert_array_equal(
        xp.asarray(A, dtype=xp.float32), np.array([[0, 5, 0], [6, 0, 0]], np.float32)
    )
    T = Tensor.from_coo([[0, 1], [1, 0], [2, 3]], [1.0, 2.0], shape=(2, 2, 4))
    x = xp.asarray(T, fill_value=-1.0)
    expected = np.full((2, 2, 4), -1.0)
    expected[0, 1, 2], expected[1, 0, 3] = 1.0, 2.0
    assert_array_equal(x, expected)
    T[0, 0, 0] = 9.0  # the Array has its own data
    assert_array_equal(x, expected)
    assert_array_equal(xp.asarray(Scalar.from_value(3)), np.array(3))
    assert_array_equal(xp.asarray(Scalar(float)), np.array(0.0))
    assert isinstance(x.tensor, Tensor)
    assert x.tensor.nvals == 2


def test_asarray_scipy_and_pydata_sparse():
    sp = pytest.importorskip("scipy.sparse")
    A = sp.csr_array(np.array([[0.0, 1.5], [2.0, 0.0]]))
    assert_array_equal(xp.asarray(A), np.array([[0.0, 1.5], [2.0, 0.0]]))
    sparse = pytest.importorskip("sparse")
    s = sparse.COO.from_numpy(np.array([[1.0, 2.0], [2.0, 3.0]]), fill_value=2.0)
    x = xp.asarray(s)
    assert x.fill_value == 2.0
    assert x.nnz == 2
    assert_array_equal(x, np.array([[1.0, 2.0], [2.0, 3.0]]))


def test_from_coo():
    x = xp.from_coo(([0, 2], [1, 0]), [1.0, 2.0], (3, 2), fill_value=5.0)
    assert_array_equal(x, np.array([[5.0, 1.0], [5.0, 5.0], [2.0, 5.0]]))
    # a repeated coordinate sums its values (or for bool), as pydata/sparse and scipy do
    x = xp.from_coo(([0, 0, 1], [1, 1, 0]), [1.0, 2.0, 5.0], (2, 2))
    assert_array_equal(x, np.array([[0.0, 3.0], [5.0, 0.0]]))
    x = xp.from_coo(([0, 0],), [True, False], (3,))
    assert_array_equal(x, np.array([True, False, False]))
    x = xp.from_coo(([1, 1],), np.array([100, 100], np.int8), (2,))
    assert_array_equal(x, np.array([0, -56], np.int8))  # (wraps, as numpy's int8 sum)
    # one value for every coordinate: stored once (iso), summed where coordinates repeat
    x = xp.from_coo(([0, 2, 0],), 7, (3,))
    assert_array_equal(x, np.array([14, 0, 7]))
    assert_array_equal(xp.from_coo(([0, 0],), True, (2,)), np.array([True, False]))
    assert_array_equal(xp.from_coo(([1, 0],), 2.5, (2,)), np.array([2.5, 2.5]))
    x = xp.from_coo(np.array([[0, 2], [1, 0]]), 7, (3, 2))
    assert x.dtype == np.dtype(int)
    assert_array_equal(x, np.array([[0, 7], [0, 0], [7, 0]]))
    x = xp.from_coo(([],), [], (4,), dtype=xp.int8)
    assert_array_equal(x, np.zeros(4, np.int8))
    coords, values = xp.from_coo(([3, 1],), [1.0, 2.0], (5,)).to_coo()
    np.testing.assert_array_equal(coords[0], [1, 3])
    np.testing.assert_array_equal(values, [2.0, 1.0])


# Creation functions
@pytest.mark.parametrize("shape", [(), 3, (0,), (2, 3), (2, 0, 4)])
@pytest.mark.parametrize("dtype", [None, "bool", "int16", "uint8", "float32", "complex64"])
def test_full_family(shape, dtype):
    dt = None if dtype is None else np.dtype(dtype)
    for name, value in [("zeros", 0), ("ones", 1), ("empty", 0)]:
        x = getattr(xp, name)(shape, dtype=dt)
        assert x.nnz == 0
        assert_array_equal(x, getattr(np, name if name != "empty" else "zeros")(shape, dtype=dt))
        like = getattr(xp, f"{name}_like")(x)
        assert like.nnz == 0
        assert_array_equal(like, np.full(x.shape, value, dtype=x.dtype))
    if dt is None:
        for value in [True, 3, 2.5, 1j, -0.0, np.nan]:
            assert_array_equal(xp.full(shape, value), np.full(shape, value))
    else:
        assert_array_equal(xp.full(shape, 1, dtype=dt), np.full(shape, 1, dtype=dt))
        assert_array_equal(xp.full_like(xp.zeros(shape), 2, dtype=dt), np.full(shape, 2, dtype=dt))
    with pytest.raises(OverflowError):
        xp.full(shape, 1000, dtype=xp.int8)


@pytest.mark.parametrize("dtype", [None, "int32", "bool", "complex128"])
def test_eye(dtype):
    for args in [(3,), (3, 5), (5, 3), (0,), (4, 4)]:
        for k in [-6, -2, -1, 0, 1, 3, 5]:
            assert_array_equal(xp.eye(*args, k=k, dtype=dtype), np.eye(*args, k=k, dtype=dtype))


@pytest.mark.parametrize(
    "args",
    [(5,), (2, 9), (2, 9, 3), (9, 2, -2), (0,), (1.5, 4.0, 0.5), (10, 0, -3), (0, 1, 0.1)],
)
@pytest.mark.parametrize("dtype", [None, "int16", "float32"])
def test_arange(args, dtype):
    if dtype == "int16" and any(isinstance(a, float) for a in args):
        return
    expected = np.arange(*args, dtype=dtype)
    if dtype is None and all(isinstance(a, int) for a in args):
        expected = expected.astype(np.int64)
    assert_array_equal(xp.arange(*args, dtype=dtype), expected)
    with pytest.raises(ValueError, match="step"):
        xp.arange(0, 3, 0)


@pytest.mark.parametrize(
    ("start", "stop", "num"), [(0, 1, 5), (2.5, -1, 4), (0, 1, 1), (0, 1, 0), (1j, 2, 3)]
)
@pytest.mark.parametrize("endpoint", [True, False])
def test_linspace(start, stop, num, endpoint):
    expected = np.linspace(start, stop, num, endpoint=endpoint)
    assert_array_equal(xp.linspace(start, stop, num, endpoint=endpoint), expected)
    if not isinstance(start, complex):
        assert_array_equal(
            xp.linspace(start, stop, num, endpoint=endpoint, dtype=xp.float32),
            np.linspace(start, stop, num, endpoint=endpoint, dtype=np.float32),
        )


def test_tril_triu():
    rng = np.random.default_rng(0)
    for shape in [(4, 4), (3, 5), (2, 3, 4), (0, 3)]:
        a = np.where(rng.random(shape) < 0.5, 0.0, rng.normal(size=shape))
        for fill in [0.0, 2.0]:
            dense = np.where(a == 0, fill, a)
            x = make(dense, fill)
            for k in [-3, -1, 0, 1, 2, 5]:
                assert_array_equal(xp.tril(x, k=k), np.tril(dense, k=k))
                assert_array_equal(xp.triu(x, k=k), np.triu(dense, k=k))
    with pytest.raises(ValueError, match="2 dimensions"):
        xp.tril(xp.zeros(3))


@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_meshgrid(indexing):
    a, b, c = np.array([1.0, 0.0, 2.0]), np.array([0.0, 5.0]), np.array([7.0, 0.0, 0.0, 1.0])
    for arrays in [(a,), (a, b), (a, b, c)]:
        out = xp.meshgrid(*[make(v, 0.0) for v in arrays], indexing=indexing)
        expected = np.meshgrid(*arrays, indexing=indexing)
        assert len(out) == len(expected)
        for o, e in zip(out, expected, strict=True):
            assert_array_equal(o, e)
    assert xp.meshgrid() == ()


def test_from_dlpack():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert_array_equal(xp.from_dlpack(a), a)
    x = make(a, 0.0)
    assert_array_equal(xp.from_dlpack(x), a)
    np.testing.assert_array_equal(np.from_dlpack(x), a)
    assert xp.from_dlpack(x, copy=False) is x
    assert xp.from_dlpack(x, copy=True) is not x
    with pytest.raises(ValueError, match="copy"):
        xp.from_dlpack(a, copy=False)


# Data type functions
def test_astype_all_pairs():
    for src in DTYPES:
        for dense, x in inputs(src, (3, 4), seed=1):
            for dst in DTYPES:
                # (numpy's astype from complex to real keeps the real part, with a warning)
                real = dense.real if src.kind == "c" and dst.kind not in "cb" else dense
                expected = np_call(real.astype, dst)
                if src.kind == "f" and dst.kind in "iu" and not np.isfinite(dense).all():
                    # numpy's cast of NaN and inf to an integer is undefined; any value will do
                    out = xp.astype(x, dst)
                    assert out.dtype == dst
                    finite = np.isfinite(dense)
                    np.testing.assert_array_equal(out.todense()[finite], expected[finite])
                    continue
                if src.kind == "f" and dst.kind in "iu":
                    big = np.abs(dense) >= 2.0 ** (8 * dst.itemsize - 1)
                    out = xp.astype(x, dst)
                    np.testing.assert_array_equal(out.todense()[~big], expected[~big])
                    continue
                assert_array_equal(xp.astype(x, dst), expected)
    x = xp.asarray([1, 2])
    assert xp.astype(x, xp.int64, copy=False) is x
    assert xp.astype(x, xp.int64) is not x


def test_can_cast():
    assert xp.can_cast(xp.int8, xp.int16)
    assert not xp.can_cast(xp.int16, xp.int8)
    assert xp.can_cast(xp.asarray([1], dtype=xp.float32), xp.float64)
    assert not xp.can_cast(xp.float64, xp.float32)
    assert xp.can_cast(xp.bool, xp.uint8)


def test_finfo_iinfo():
    for dt in [xp.float32, xp.float64, xp.complex64, xp.complex128]:
        info = xp.finfo(dt)
        ref = np.finfo(dt)
        assert info.bits == ref.bits
        assert info.eps == float(ref.eps)
        assert info.max == float(ref.max)
        assert info.min == float(ref.min)
        assert info.smallest_normal == float(ref.smallest_normal)
        assert isinstance(info.eps, float)
        assert xp.finfo(xp.asarray(1, dtype=dt)) == info
    assert xp.finfo(xp.complex64).dtype == xp.float32
    for dt in [xp.int8, xp.uint64, xp.int64]:
        info = xp.iinfo(dt)
        assert (info.bits, info.min, info.max) == (
            np.iinfo(dt).bits,
            int(np.iinfo(dt).min),
            int(np.iinfo(dt).max),
        )
        assert info.dtype == dt
    with pytest.raises(TypeError):
        xp.finfo(xp.int8)
    with pytest.raises(TypeError):
        xp.iinfo(xp.float32)


def test_isdtype():
    assert xp.isdtype(xp.int8, "signed integer")
    assert xp.isdtype(xp.uint8, "integral")
    assert not xp.isdtype(xp.bool, "numeric")
    assert xp.isdtype(xp.complex64, ("real floating", "complex floating"))
    assert xp.isdtype(xp.float32, xp.float32)
    assert not xp.isdtype(xp.float32, xp.float64)
    with pytest.raises(ValueError, match="kind"):
        xp.isdtype(xp.int8, "integer")
    with pytest.raises(TypeError):
        xp.isdtype("int8", "integral")


def test_result_type():
    assert xp.result_type(xp.int8, xp.uint8) == xp.int16
    assert xp.result_type(xp.asarray([1], dtype=xp.float32), 1.5) == xp.float32
    assert xp.result_type(xp.int8, 1) == xp.int8
    assert xp.result_type(xp.float32, xp.int8, xp.uint8) == np.result_type(
        np.result_type(np.float32, np.int8), np.uint8
    )
    with pytest.raises(ValueError, match="at least one"):
        xp.result_type(1, 2.0)


def test_info():
    info = xp.__array_namespace_info__()
    assert info.default_device() == "cpu"
    assert info.devices() == ("cpu",)
    assert info.default_dtypes()["real floating"] == xp.float64
    assert info.default_dtypes(device="cpu")["indexing"] == xp.int64
    assert set(info.dtypes()) == {str(dt) for dt in DTYPES}
    assert set(info.dtypes(kind="integral")) == {
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
    }  # fmt: skip
    assert set(info.dtypes(kind=("bool", "complex floating"))) == {
        "bool",
        "complex64",
        "complex128",
    }
    assert info.capabilities()["data-dependent shapes"]


# The Array object
def test_array_attributes():
    a = np.arange(24.0).reshape(2, 3, 4)
    x = make(a, 0.0)
    assert x.shape == (2, 3, 4)
    assert x.ndim == 3
    assert x.size == 24
    assert x.dtype == xp.float64
    assert x.device == "cpu"
    assert x.to_device("cpu") is x
    with pytest.raises(ValueError, match="device"):
        x.to_device("gpu")
    assert_array_equal(x.T, a.T)
    assert_array_equal(x.mT, np.swapaxes(a, -1, -2))
    with pytest.raises(ValueError, match="2 dimensions"):
        xp.zeros(3).mT
    assert x.nnz == 23
    assert x.fill_value == 0.0
    assert repr(x) == "Array(shape=(2, 3, 4), dtype=float64, nnz=23, fill_value=0.0)"
    assert repr(xp.asarray(1.5)) == "Array(1.5, dtype=float64)"
    with pytest.raises(TypeError, match="asarray"):
        xp.Array()
    assert len(x) == 2
    assert [y.shape for y in x] == [(3, 4), (3, 4)]
    with pytest.raises(TypeError, match="len"):
        _ = len(xp.asarray(1))
    with pytest.raises(TypeError):
        _ = iter(xp.asarray(1))
    with pytest.raises(TypeError):
        hash(x)


def test_metadata_is_stored():
    # shape, dtype, ndim and size never read the Tensor (one may not be computed yet)
    x = make(np.arange(24, dtype=np.int32).reshape(2, 3, 4), 0)
    x._tensor = None  # (any read of it would fail)
    assert (x.shape, x.dtype, x.ndim, x.size) == ((2, 3, 4), xp.int32, 3, 24)
    assert x.fill_value == 0
    y = xp.zeros((5, 0), dtype=xp.bool)
    y += xp.ones((5, 0), dtype=xp.bool)
    assert (y.shape, y.dtype) == ((5, 0), xp.bool)


def test_python_scalars():
    assert bool(xp.asarray(True)) is True
    assert int(xp.asarray(-3)) == -3
    assert float(xp.asarray(2.5)) == 2.5
    assert complex(xp.asarray(1 + 2j)) == 1 + 2j
    assert [10, 20, 30][xp.asarray(1)] == 20
    assert math.isnan(float(xp.asarray(np.nan)))
    with pytest.raises(TypeError):
        [1, 2][xp.asarray(1.0)]
    with pytest.raises(TypeError):
        bool(xp.asarray([1, 2]))
    with pytest.raises(TypeError):
        float(xp.asarray(1j))
    with pytest.raises(ValueError, match="NaN"):
        int(xp.asarray(np.nan))


def test_array_protocol_is_guarded():
    x = make(np.array([1.0, 0.0, 2.0]), 0.0)
    np.testing.assert_array_equal(np.asarray(x), [1.0, 0.0, 2.0])
    assert np.asarray(x, dtype=np.float32).dtype == np.float32
    with pytest.raises(ValueError, match="copies"):
        np.asarray(x, copy=False)
    big = xp.zeros((10**5, 10**5))
    with pytest.raises(xp.DensifyError, match="array_api_densify_limit"):
        np.asarray(big)
    with gb.config.set(array_api_densify_limit=2), pytest.raises(xp.DensifyError):
        x.todense()
    assert big.nnz == 0
    assert xp.sum(big) == 0.0


def test_copy_module():
    a = np.array([[1.0, 0.0], [2.0, 3.0]])
    x = make(a, 0.0)
    for func in [copy.copy, copy.deepcopy]:
        y = func(x)
        y[0, 0] = 5.0
        assert_array_equal(x, a)
        assert y[0, 0] == 5.0
        z = func(x @ x)
        assert_array_equal(z, a @ a)


def test_pickle_roundtrips():
    # (review finding L7) through coordinates, values, shape, dtype and fill; explicit fills
    # stay stored, and a pending product is computed first
    rng = np.random.default_rng(19)
    for dense, fill in [
        (np.array([1.0, 0.0, 2.0]), 0.0),
        (np.array([[1, 7], [7, 3]], np.int16), 7),
        (np.array([True, False, True]), True),
        (np.array([1j, 0, np.nan]), 0j),
        (np.array(2.5), 0.0),
        (np.zeros((0, 3)), 1.0),
    ]:
        x = make(dense, fill, rng)
        y = pickle.loads(pickle.dumps(x))
        assert_array_equal(y, dense)
        assert y.nnz == x.nnz
        assert y.fill_value == x.fill_value or np.isnan(fill)
    a = make(np.array([[1.0, 2.0], [0.0, 3.0]]), 0.0)
    P = a @ a
    assert_array_equal(pickle.loads(pickle.dumps(P)), a.todense() @ a.todense())


# Indexing and shape functions
@pytest.mark.parametrize(
    "key",
    [
        0,
        -1,
        (1, 2),
        (1, 2, 3),
        slice(None),
        slice(1, None),
        slice(None, None, -1),
        slice(None, None, 2),
        (slice(None), 1),
        (Ellipsis, 1),
        (1, Ellipsis),
        (Ellipsis,),
        (None, 0),
        (0, None, slice(None, None, -2)),
        (Ellipsis, None),
        (None, Ellipsis, None),
        (slice(5, 1, -1), slice(0, 0)),
        (),
        (1, slice(0, 2), -1),
    ],
)
def test_getitem_basic(key):
    rng = np.random.default_rng(2)
    a = np.where(rng.random((3, 4, 5)) < 0.5, 0.0, np.round(rng.normal(size=(3, 4, 5)), 1))
    for fill in [0.0, -1.5]:
        dense = np.where(a == 0, fill, a)
        x = make(dense, fill, rng)
        assert_array_equal(x[key], dense[key])


def test_getitem_errors_and_0d():
    x = make(np.arange(6.0).reshape(2, 3), 0.0)
    with pytest.raises(IndexError):
        x[2]
    with pytest.raises(IndexError):
        x[0, -4]
    with pytest.raises(IndexError):
        x[0, 0, 0]
    with pytest.raises(IndexError):
        x[..., 0, ...]
    z = xp.asarray(2.5)
    assert_array_equal(z[()], np.array(2.5))
    assert_array_equal(z[...], np.array(2.5))
    assert_array_equal(z[None], np.array([2.5]))
    assert_array_equal(x[xp.asarray(1)], np.arange(3.0, 6.0))


@pytest.mark.parametrize(
    ("shape", "new"),
    [
        ((2, 3, 4), (6, 4)),
        ((2, 3, 4), (4, -1)),
        ((2, 3, 4), (24,)),
        ((2, 3, 4), (2, 3, 4)),
        ((6,), (1, 6, 1)),
        ((1,), ()),
        ((), (1, 1)),
        ((), -1),
        ((0, 3), (3, 0)),
    ],
)
def test_reshape(shape, new):
    for dtype in [np.dtype("int16"), np.dtype("float64")]:
        for dense, x in inputs(dtype, shape, seed=3):
            assert_array_equal(xp.reshape(x, new), np.reshape(dense, new))
            assert_array_equal(xp.reshape(x, new, copy=True), np.reshape(dense, new))
    with pytest.raises(ValueError, match="reshape"):
        xp.reshape(xp.zeros((2, 3)), (4, 2))
    with pytest.raises(ValueError, match="-1"):
        xp.reshape(xp.zeros((2, 3)), (-1, -1))


def test_reshape_copy_false():
    x = make(np.arange(24.0).reshape(2, 3, 4), 0.0)
    out = xp.reshape(x, (2, 12), copy=False)
    assert_array_equal(out, np.arange(24.0).reshape(2, 12))


def test_permute_expand_squeeze():
    rng = np.random.default_rng(4)
    a = np.where(rng.random((2, 1, 3)) < 0.5, 0, rng.integers(1, 9, (2, 1, 3)))
    x = make(a, 0)
    for axes in [(0, 1, 2), (2, 0, 1), (-1, -2, -3)]:
        assert_array_equal(xp.permute_dims(x, axes), np.transpose(a, axes))
    with pytest.raises(ValueError, match="permutation"):
        xp.permute_dims(x, (0, 0, 1))
    for axis in [0, 1, 3, -1, -4, (0, 2), (0, -1)]:
        assert_array_equal(xp.expand_dims(x, axis=axis), np.expand_dims(a, axis))
    with pytest.raises(IndexError):
        xp.expand_dims(x, axis=5)
    assert_array_equal(xp.squeeze(x, axis=1), np.squeeze(a, axis=1))
    assert_array_equal(xp.squeeze(x, axis=(1,)), np.squeeze(a, axis=1))
    with pytest.raises(ValueError, match="squeeze"):
        xp.squeeze(x, axis=0)
    one = make(np.array([[7]]), 0)
    assert_array_equal(xp.squeeze(one, axis=(0, 1)), np.array(7))
    assert_array_equal(xp.expand_dims(xp.asarray(3), axis=0), np.array([3]))
    assert_array_equal(xp.permute_dims(xp.asarray(3), ()), np.array(3))
    check_invariants(xp.squeeze(one, axis=(0, 1)))


def test_asarray_of_arrays_stays_sparse():
    # a list of Arrays is stacked, not densified (review finding L7)
    import graphblas as gb

    x = xp.zeros((10**5, 10**5))
    x[1, 2] = 3.0
    s = xp.asarray([x, x])
    assert s.shape == (2, 10**5, 10**5)
    assert s.nnz == 2
    assert_array_equal(
        xp.asarray([[make(np.array([1, 0]), 0), make(np.array([0, 2]), 0)]]),
        np.array([[[1, 0], [0, 2]]]),
    )
    assert_array_equal(xp.asarray([xp.asarray([1, 2]), [3, 4]]), np.array([[1, 2], [3, 4]]))
    ragged = [make(np.array([1, 2]), 0), make(np.array([3]), 0)]
    with pytest.raises(ValueError, match="inhomogeneous"):  # (as numpy refuses it)
        xp.asarray(ragged)
    assert_array_equal(
        xp.asarray([make(np.array([1.0]), 0.0)], dtype=xp.float32), np.array([[1.0]], np.float32)
    )
    del gb


def test_asarray_byte_order():
    # (review finding L9)
    assert_array_equal(xp.asarray(np.array([1, 2], dtype=">i8")), np.array([1, 2]))
    assert_array_equal(
        xp.asarray(np.array([1.5, 0.0], dtype=">f4")), np.array([1.5, 0.0], np.float32)
    )

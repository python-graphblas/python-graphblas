"""graphblas.array_api: argument checking, rare storage cases and extension paths."""

import numpy as np
import pytest

import graphblas.array_api as xp
from graphblas.array_api import _utils
from graphblas.core.tensor import Tensor

from .test_array_api_elementwise import assert_array_equal, make


def test_array_arguments_are_checked():
    a = np.array([1.0, 2.0])
    for func in [xp.reshape, xp.squeeze, xp.permute_dims, xp.expand_dims]:
        with pytest.raises(TypeError, match="needs an Array"):
            func(a, 0)
    for func in [xp.sum, xp.diff, xp.cumulative_sum, xp.tril, xp.zeros_like, xp.clip]:
        with pytest.raises(TypeError, match="needs an Array"):
            func(a)
    with pytest.raises(TypeError, match="needs an Array"):
        xp.astype(a, xp.float32)
    with pytest.raises(TypeError, match="arrays or Python scalars"):
        xp.add(make(a, 0.0), [1, 2])


def test_scalars_alone():
    assert_array_equal(xp.sin(1.0), np.sin(np.float64(1.0)))
    assert_array_equal(xp.add(1, 2), np.array(3))
    assert_array_equal(xp.multiply(np.float32(2), 3), np.array(6, np.float32))


def test_shape_errors():
    x = make(np.arange(6.0).reshape(2, 3), 0.0)
    with pytest.raises(ValueError, match="negative"):
        xp.zeros((2, -1))
    with pytest.raises(ValueError, match="invalid shape"):
        xp.reshape(x, (-2, 3))
    with pytest.raises(ValueError, match="cannot reshape"):
        xp.reshape(x, (4, -1))
    with pytest.raises(ValueError, match="repeated"):
        xp.expand_dims(x, axis=(0, 0))
    with pytest.raises(ValueError, match="repeated"):
        xp.squeeze(xp.zeros((1, 1)), axis=(0, -2))
    assert_array_equal(xp.squeeze(x, axis=()), np.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError, match="do not broadcast"):
        x + make(np.ones(4), 0.0)
    with pytest.raises(ValueError, match="1-d"):
        xp.meshgrid(x)
    with pytest.raises(ValueError, match="one data type"):
        xp.meshgrid(xp.zeros(2), xp.zeros(2, dtype=xp.int8))
    with pytest.raises(ValueError, match="indexing"):
        xp.meshgrid(xp.zeros(2), indexing="yx")


def test_creation_errors():
    with pytest.raises(TypeError, match="Python scalar"):
        xp.full(3, "a")
    with pytest.raises(ValueError, match="copy"):
        xp.from_dlpack(np.ones(2), copy=False)
    with pytest.raises(ValueError, match="keeps its fill"):
        xp.asarray(xp.zeros(2), fill_value=1.0)
    assert_array_equal(xp.asarray(np.array([1, 2], np.float16)), np.array([1, 2], np.float32))
    assert_array_equal(xp.full(2, np.float32(1.5)), np.full(2, 1.5, np.float32))
    assert_array_equal(xp.full(2, xp.asarray(3, dtype=xp.int8)), np.full(2, 3, np.int8))
    assert_array_equal(xp.full_like(xp.zeros(2), xp.asarray(4.0)), np.full(2, 4.0))
    assert_array_equal(xp.from_coo(([1, 3],), [1, 2]), np.array([0, 1, 0, 2]))
    assert_array_equal(xp.from_coo((), 5, ()), np.array(5))
    assert_array_equal(xp.from_coo((), [], (), dtype=xp.int8), np.array(0, np.int8))


def test_sparse_input_errors():
    sp = pytest.importorskip("scipy.sparse")
    A = sp.coo_array((np.array([1.0, 2.0]), (np.array([0, 0]), np.array([1, 1]))), shape=(2, 2))
    assert_array_equal(xp.asarray(A), np.array([[0.0, 3.0], [0.0, 0.0]]))  # duplicates summed
    with pytest.raises(ValueError, match="fill value 0"):
        xp.asarray(A, fill_value=1.0)
    sparse = pytest.importorskip("sparse")
    s = sparse.GCXS.from_numpy(np.array([[1.0, 0.0], [0.0, 3.0]]))
    assert_array_equal(xp.asarray(s), np.array([[1.0, 0.0], [0.0, 3.0]]))
    with pytest.raises(ValueError, match="differs"):
        xp.asarray(s, fill_value=1.0)


def test_dtype_errors():
    with pytest.raises(TypeError, match="not a data type"):
        xp.asarray([1], dtype="not a dtype")
    assert xp.isdtype(np.float32, "real floating")
    with pytest.raises(ValueError, match="at least one"):
        xp.result_type()
    info = xp.__array_namespace_info__()
    with pytest.raises(ValueError, match="kind"):
        info.dtypes(kind="integer")
    with pytest.raises(ValueError, match="device"):
        info.default_dtypes(device="gpu")


def test_object_protocol_errors():
    x = make(np.array([1.0, 0.0]), 0.0)
    with pytest.raises(BufferError, match="copies"):
        x.__dlpack__(copy=False)
    assert x.__dlpack_device__() == np.empty(0).__dlpack_device__()
    np.testing.assert_array_equal(np.from_dlpack(x), [1.0, 0.0])
    with pytest.raises(TypeError, match="complex"):
        int(xp.asarray(1j))
    with pytest.raises(TypeError):
        x @ "a"
    with pytest.raises(TypeError):
        "a" @ x
    with pytest.raises(TypeError):
        x + "a"
    with pytest.raises(TypeError):
        "a" + x
    np.testing.assert_array_equal(np.from_dlpack(xp.from_dlpack(x)), [1.0, 0.0])
    capsule = x.__dlpack__(dl_device=x.__dlpack_device__(), max_version=(1, 0))
    assert capsule is not None
    with pytest.raises(TypeError, match="None"):
        xp.astype(x, None)


def test_clip_mixed_bounds():
    a = np.array([1, 5, 0, 9], np.int16)
    x = make(a, 0)
    lo = make(np.array([2, 2, 2, 2], np.int32), 2)
    out = xp.clip(x, min=lo, max=np.int16(6))
    assert out.dtype == xp.int16
    assert_array_equal(out, np.minimum(np.maximum(a, 2), 6).astype(np.int16))
    assert_array_equal(xp.clip(x, max=4.5), np.clip(a, None, 4.5).astype(np.int16))
    assert_array_equal(xp.clip(x, min=xp.asarray(np.int16(3))), np.clip(a, 3, None))
    hi = make(np.array([4, 4, 4, 4], np.int16), 4)
    assert_array_equal(xp.clip(x, max=hi), np.minimum(a, 4))


def test_complex_comparisons_mixed_precision():
    a = np.array([1 + 1j, 0, 2], np.complex64)
    b = np.array([1 + 1j, 3, 0], np.complex128)
    assert_array_equal(xp.equal(make(a, 0), make(b, 0)), a == b)
    assert_array_equal(xp.not_equal(make(a, 0), make(b, 0)), a != b)


def test_bool_products():
    a = np.array([[True, False, True], [True, True, True]])
    x = make(a, False)
    for axis in [None, 0, 1]:
        assert_array_equal(xp.prod(x, axis=axis, dtype=xp.bool), np.prod(a, axis=axis, dtype=bool))
    y = make(a, True)
    assert_array_equal(xp.prod(y, axis=1, dtype=xp.bool), np.prod(a, axis=1, dtype=bool))


def test_statistical_errors():
    with pytest.raises(ValueError, match="at least one dimension"):
        xp.cumulative_sum(xp.asarray(1.0))
    with pytest.raises(ValueError, match="at least one dimension"):
        xp.diff(xp.asarray(1.0))
    with pytest.raises(ValueError, match="nonnegative"):
        xp.diff(xp.zeros(3), n=-1)


def test_concat_errors():
    from graphblas.array_api._manipulation import _broadcast_to

    a = make(np.array([[1.0, 0.0]]), 0.0)
    b = make(np.array([[2.0, 2.0]]), 2.0)
    assert_array_equal(xp.concat([a, b]), np.array([[1.0, 0.0], [2.0, 2.0]]))
    assert_array_equal(xp.concat([xp.zeros((1, 2)), xp.zeros((2, 2))]), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least one"):
        xp.concat([])
    with pytest.raises(ValueError, match="at least 1"):
        xp.concat([xp.asarray(1.0)])
    with pytest.raises(ValueError, match="do not concatenate"):
        xp.concat([a, xp.zeros((1, 3))])
    with pytest.raises(ValueError, match="broadcast"):
        _broadcast_to(a, (3, 3))
    with pytest.raises(ValueError, match="cannot broadcast"):
        _broadcast_to(a, (2,))


def test_map_values_storage_cases():
    T = Tensor.from_coo([[0, 1], [1, 2]], [1.0, 2.0], shape=(2, 3))
    U = T.relayout(((1,), (0,)))
    out = _utils.map_values(np.add, [T, U], np.dtype(float))
    np.testing.assert_array_equal(out.to_dense(fill_value=0), [[0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    out = _utils.map_values(np.add, [T, T.dup()], np.dtype(float))  # (one Matrix, shared)
    np.testing.assert_array_equal(out.to_coo()[1], [2.0, 4.0])
    E = Tensor(T.dtype, (2, 3))
    assert _utils.map_values(np.sin, [E], np.dtype(float)).nvals == 0
    out = _utils._map_values_coo(np.add, [T, T], np.dtype(float))
    np.testing.assert_array_equal(out.to_coo()[1], [2.0, 4.0])
    out = _utils._map_values_coo(np.negative, [T], np.dtype(float))
    np.testing.assert_array_equal(out.to_coo()[1], [-1.0, -2.0])
    out = _utils._map_values_coo(lambda v: np.float64(7.0), [T], np.dtype(float))
    np.testing.assert_array_equal(out.to_coo()[1], [7.0, 7.0])
    assert _utils.full_tensor(np.dtype(float), (0, 3), 1.0).nvals == 0


def test_scalar_cache_is_bounded():
    for i in range(1100):
        _utils.gb_scalar(i, np.dtype(np.int64))
    assert len(_utils._SCALARS) <= 1025
    assert _utils.gb_scalar(-0.0, np.dtype(float)).value == 0.0


def test_summary_of_complex_values():
    x = make(np.array([1 + 0j, complex(-0.0, 1), complex(np.nan, 0)]), 0j)
    facts = _utils.summary(x)
    assert facts == {"finite": False, "nonnegative": False, "nan": True, "negzero": True}

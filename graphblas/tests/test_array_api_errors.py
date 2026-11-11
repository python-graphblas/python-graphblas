"""graphblas.array_api: argument checking, rare storage cases and extension paths."""

import numpy as np
import pytest

import graphblas as gb
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
    assert _utils.full_tensor(np.dtype(float), (0, 3), 1.0).nvals == 0


def test_scalar_cache_is_bounded():
    for i in range(1100):
        _utils.gb_scalar(i, np.dtype(np.int64))
    assert len(_utils._SCALARS) <= 1025
    assert _utils.gb_scalar(-0.0, np.dtype(float)).value == 0.0


def test_facts_of_stored_values():
    x = make(np.array([1 + 0j, complex(-0.0, 1), complex(np.nan, 0)]), 0j)
    assert [_utils.fact(x, k) for k in ["finite", "nan", "zero"]] == [False, True, False]
    assert x._facts == {"finite": False, "nan": True, "zero": False}
    y = make(np.array([-0.0, np.inf, 2.0]), 1.0)
    assert [_utils.fact(y, k) for k in ["finite", "nan", "zero"]] == [False, False, True]
    # NaN and finiteness exclude each other: one is known from the other
    z = make(np.array([np.nan, 2.0]), 0.0)
    assert _utils.fact(z, "nan")
    assert not _utils.fact(z, "finite")
    w = make(np.array([3.0, 2.0]), 0.0)
    assert _utils.fact(w, "finite")
    assert not _utils.fact(w, "nan")
    # integers are finite without a look; empty arrays hold nothing
    i = make(np.array([0, 5]), 1)
    assert _utils.fact(i, "finite")
    assert not _utils.fact(i, "nan")
    assert _utils.fact(i, "zero")
    e = xp.zeros(4)
    assert [_utils.fact(e, k) for k in ["finite", "nan", "zero"]] == [True, False, False]
    # infinities by sign; known absent from finiteness, and for integers
    u = make(np.array([np.inf, 2.0]), 0.0)
    assert [_utils.fact(u, k) for k in ["posinf", "neginf"]] == [True, False]
    assert _utils.fact(make(np.array([-np.inf]), 0.0), "neginf")
    assert not _utils.fact(w, "posinf")
    assert not _utils.fact(i, "neginf")
    # abs, negative, positive and conj keep all three; roundings keep two
    v = make(np.array([-1.5, np.nan, 0.0]), 0.0)
    for k in ["finite", "nan", "zero"]:
        _utils.fact(v, k)
    for func in [xp.abs, xp.negative, xp.positive, xp.conj]:
        assert func(v)._facts == v._facts
    for func in [xp.floor, xp.ceil, xp.trunc, xp.round]:
        assert func(v)._facts == {"finite": False, "nan": True}
    assert xp.exp(v)._facts is None
    assert xp.abs(xp.zeros(3))._facts is None
    # (review finding M4) |z| of a finite complex z may be inf: abs keeps only "zero" there
    z = make(np.array([1.5e308 + 1.5e308j, 1, 0, 0]), 0j)
    assert _utils.fact(z, "finite")
    _utils.fact(z, "zero")
    assert set(xp.abs(z)._facts) == {"zero"}
    zz = make(np.array([0.0, 0.0, 2.0, 3.0]), 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        assert_array_equal(xp.abs(z) * zz, np.abs(z.todense()) * zz.todense())


def test_thread3_argument_checks():
    x = make(np.arange(6.0).reshape(2, 3), 0.0)
    for func in [xp.take, xp.take_along_axis]:
        with pytest.raises(TypeError, match="Array"):
            func(np.zeros(3), np.array([0]))
    with pytest.raises(TypeError, match="integer"):
        xp.take(x, np.array([0.5]), axis=0)
    with pytest.raises(ValueError, match="dimension"):
        xp.take(xp.asarray(1.0), np.array([0]))
    with pytest.raises(TypeError, match="Arrays"):
        xp.matmul(x, [[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        xp.tensordot(x, x, axes=-1)
    with pytest.raises(ValueError, match="more axes"):
        xp.tensordot(x, x, axes=3)
    with pytest.raises(ValueError, match="as many"):
        xp.tensordot(x, x, axes=([0], [0, 1]))
    with pytest.raises(ValueError, match="dimension"):
        xp.vecdot(xp.asarray(1.0), x)
    with pytest.raises(ValueError, match="differs"):
        xp.vecdot(x, make(np.ones((2, 2)), 0.0))
    assert_array_equal(xp.vecdot(x, x, axis=1), np.vecdot(x.todense(), x.todense(), axis=1))
    with pytest.raises(ValueError, match="repeated"):
        xp.flip(x, axis=(0, 0))
    with pytest.raises(TypeError, match="Python scalars"):
        xp.where(x > 0, x, "a")
    with pytest.raises(TypeError, match="Array"):
        xp.argmax(np.zeros(3))
    with pytest.raises(ValueError, match="negative"):
        xp.linalg.cross(xp.zeros(3), xp.zeros(3), axis=0)
    with pytest.raises(TypeError, match="Array"):
        xp.linalg.vector_norm(np.zeros(3))
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    rtol = xp.asarray(1e-3)
    assert_array_equal(xp.linalg.pinv(make(a, 0.0), rtol=rtol), np.linalg.pinv(a, rtol=1e-3))
    assert int(xp.linalg.matrix_rank(make(a, 0.0), rtol=rtol)) == 2
    assert_array_equal(x @ x.mT, x.todense() @ x.todense().T)


def test_thread3_edge_cases():
    x = make(np.arange(6.0).reshape(2, 3), 0.0)
    assert_array_equal(x[np.array([], dtype=float)], np.zeros((0, 3)))
    assert_array_equal(x[xp.asarray(True)], x.todense()[None])
    assert_array_equal(x[np.array(False)], x.todense()[np.array(False)])
    y = make(np.arange(6.0).reshape(2, 3), 0.0)
    y[:, 0:0] = 5.0
    y[[0, 1], 0:0] = xp.zeros((2, 0))
    y[[], 1] = 3.0
    assert_array_equal(y, x.todense())
    e = xp.zeros((2, 3))
    e[0] = 0.0  # (deleting from an array that stores nothing)
    assert e.nnz == 0
    assert_array_equal(
        xp.take_along_axis(xp.zeros((2, 3)), np.zeros((2, 1), int), axis=1), np.zeros((2, 1))
    )
    assert_array_equal(xp.nonzero(xp.zeros(4))[0], np.zeros(0, np.int64))
    assert_array_equal(xp.sort(xp.zeros(3)), np.zeros(3))
    assert_array_equal(xp.sort(make(np.array([0.0, 2.0, 0.0]), 2.0)), np.array([0.0, 0.0, 2.0]))
    z = make(np.array([[2.0, 2.0], [2.0, 2.0]]), 2.0, np.random.default_rng(0), explicit=1.0)
    assert_array_equal(xp.sort(z, axis=0), np.full((2, 2), 2.0))
    full = make(np.array([[1.0, 2.0], [3.0, 1.0]]), 0.0)
    assert_array_equal(xp.unique_inverse(full).inverse_indices, np.array([[0, 1], [2, 0]]))
    assert_array_equal(xp.unique_values(xp.zeros((2, 0))), np.zeros(0))
    assert_array_equal(
        xp.unique_inverse(xp.zeros((2, 0))).inverse_indices, np.zeros((2, 0), np.int64)
    )
    assert_array_equal(xp.isin(full, full), np.ones((2, 2), bool))
    with np.errstate(all="ignore"):
        assert_array_equal(xp.var(xp.zeros((0, 3)), axis=0, keepdims=True), np.full((1, 3), np.nan))
    # tensordot with a nonzero fill on a 0-d result, and int32 * int8 promotion
    a = make(np.array([1.0, 0.0, 2.0]), 1.0)
    assert_array_equal(xp.tensordot(a, a, axes=1), np.tensordot(a.todense(), a.todense(), axes=1))
    i32 = make(np.array([[1, 2], [3, 4]], np.int32), 0)
    i8 = make(np.array([[1, 0], [0, 1]], np.int8), 0)
    assert_array_equal(
        xp.tensordot(i32, i8, axes=1), np.tensordot(i32.todense(), i8.todense(), axes=1)
    )
    inf = make(np.array([[np.inf, 1.0], [2.0, np.inf]]), np.inf)
    assert_array_equal(
        xp.tensordot(inf, i32, axes=1), np.tensordot(inf.todense(), i32.todense(), axes=1)
    )
    # both operands broadcast: an outer product of the entries, under the limit
    u, v = make(np.ones((3, 1, 4)), 0.0), make(np.ones((5, 4)), 0.0)
    assert_array_equal(xp.vecdot(u, v), np.vecdot(u.todense(), v.todense()))
    with gb.config.set(array_api_densify_limit=4), pytest.raises(xp.DensifyError):
        xp.vecdot(u, v)
    # numpy operands
    c = np.array([True, False, True])
    assert_array_equal(xp.where(c, np.array([1, 2, 3]), 0), np.where(c, [1, 2, 3], 0))
    assert_array_equal(xp.where(True, x, 0.0), x.todense())
    assert_array_equal(np.eye(3) @ x.mT, np.eye(3) @ x.todense().T)

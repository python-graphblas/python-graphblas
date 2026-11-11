"""graphblas.array_api linear algebra (main namespace and the linalg extension) against numpy."""

import math

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp

from .test_array_api_elementwise import assert_array_equal, check_invariants, make, np_call

rng = np.random.default_rng(21)


def _rand(shape, fill, dtype=np.float64, density=0.5):
    if np.dtype(dtype).kind == "b":
        a = rng.random(shape) < 0.5
        return np.where(rng.random(shape) < density, a, bool(fill))
    if np.dtype(dtype).kind in "iu":
        a = rng.integers(0, 5, shape)
    else:
        a = np.round(rng.normal(size=shape), 2)
    return np.where(rng.random(shape) < density, a, fill).astype(dtype)


def assert_close(x, expected):
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape
    got = x.todense()
    if got.dtype.kind in "fc":
        tol = 1e-5 if got.dtype.itemsize * (2 if got.dtype.kind == "f" else 1) == 8 else 1e-10
        np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(got, expected)


MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((4,), (4,)),
    ((2, 3, 4), (4, 5)),
    ((2, 1, 3, 4), (3, 4, 2)),
    ((0, 4), (4, 2)),
]


@pytest.mark.parametrize("shapes", MATMUL_SHAPES, ids=str)
def test_matmul(shapes):
    for dtype in [np.float64, np.int64, np.float32, np.complex128, np.uint8, np.bool_]:
        for fa, fb in [(0, 0), (1, 0), (0, 2), (1, 3)]:
            if np.dtype(dtype).kind == "b" and (fa or fb) and (fa, fb) != (1, 3):
                continue
            a, b = _rand(shapes[0], fa, dtype), _rand(shapes[1], fb, dtype)
            x1, x2 = make(a, fa), make(b, fb)
            expected = np.matmul(a, b)
            assert_close(xp.matmul(x1, x2), expected)
            assert_close(x1 @ x2, expected)
    with pytest.raises(ValueError, match="align"):
        xp.matmul(xp.zeros((2, 3)), xp.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least one"):
        xp.matmul(xp.asarray(1.0), xp.zeros(2))
    with pytest.raises(ValueError, match="broadcast"):
        xp.matmul(xp.zeros((2, 2, 3)), xp.zeros((3, 3, 2)))


def test_matmul_zero_fill_is_structural():
    # (DESIGN D6: as scipy.sparse; numpy would give NaN for 0 * inf)
    a = make(np.array([[np.inf, 0.0], [0.0, 1.0]]), 0.0)
    b = make(np.array([[0.0, 1.0], [1.0, 0.0]]), 0.0)
    assert_close(a @ b, np.array([[0.0, np.inf], [1.0, 0.0]]))
    # a fill that is not finite: numpy, densely
    c = make(np.array([[np.inf, 1.0], [np.inf, np.inf]]), np.inf)
    assert_array_equal(np_call(xp.matmul, c, b), np_call(np.matmul, c.todense(), b.todense()))
    with gb.config.set(array_api_densify_limit=2), pytest.raises(xp.DensifyError):
        c @ b


def test_matmul_inplace():
    a = make(np.eye(3), 0.0)
    b = make(np.arange(9.0).reshape(3, 3), 0.0)
    a @= b
    assert_close(a, np.arange(9.0).reshape(3, 3))


@pytest.mark.parametrize(
    "axes", [2, 1, 0, ([1, 2], [1, 0]), ([2], [0]), ([0, 1, 2], [2, 1, 0]), ((), ())]
)
def test_tensordot(axes):
    for dtype in [np.float64, np.int32, np.complex64]:
        for fa, fb in [(0, 0), (2, 0), (0, 1), (1, 1)]:
            a = _rand((3, 4, 3), fa, dtype)
            b = _rand((3, 4, 3), fb, dtype) if axes != 1 else _rand((3, 2), fb, dtype)
            if axes == 2:
                b = _rand((4, 3, 2), fb, dtype)
            if axes == 0:
                b = _rand((2,), fb, dtype)
            expected = np.tensordot(a, b, axes=axes)
            out = xp.tensordot(make(a, fa), make(b, fb), axes=axes)
            if np.dtype(dtype) == np.complex64:
                np.testing.assert_allclose(out.todense(), expected, rtol=1e-5, atol=1e-5)
                assert out.dtype == expected.dtype
            else:
                assert_close(out, expected)
    with pytest.raises(ValueError, match="differ"):
        xp.tensordot(xp.zeros((2, 3)), xp.zeros((2, 3)), axes=1)


@pytest.mark.parametrize("axis", [-1, -2])
def test_vecdot(axis):
    for dtype in [np.float64, np.int16, np.complex128, np.bool_]:
        for fa, fb in [(0, 0), (1, 0), (0, 2), (1, 2)]:
            if np.dtype(dtype).kind == "b" and (fa or fb) and (fa, fb) != (1, 2):
                continue
            for sa, sb in [((3, 4), (3, 4)), ((2, 3, 4), (3, 4)), ((3, 1), (1, 3, 1))]:
                a, b = _rand(sa, fa, dtype), _rand(sb, fb, dtype)
                if -axis > min(a.ndim, b.ndim) or a.shape[axis] != b.shape[axis]:
                    continue
                out = xp.vecdot(make(a, fa), make(b, fb), axis=axis)
                assert_close(out, np.vecdot(a, b, axis=axis))
    with pytest.raises(ValueError, match="axis"):
        xp.vecdot(xp.zeros(3), xp.zeros(3), axis=-2)
    a = np.array([1 + 2j, np.inf, 1j])
    b = np.array([2j, 1, 1 - 1j])
    assert_array_equal(np_call(xp.vecdot, make(a, np.inf), make(b, 0)), np_call(np.vecdot, a, b))


def test_matrix_transpose():
    a = _rand((2, 3, 4), 0.0)
    assert_array_equal(xp.matrix_transpose(make(a, 0.0)), np.swapaxes(a, -1, -2))
    assert_array_equal(xp.linalg.matrix_transpose(make(a, 1.0)), np.swapaxes(a, -1, -2))
    with pytest.raises(ValueError, match="2 dimensions"):
        xp.matrix_transpose(xp.zeros(3))


@pytest.mark.parametrize("offset", [-3, -1, 0, 1, 2, 5])
def test_diagonal_trace(offset):
    for fill in [0.0, 1.5]:
        a = _rand((2, 3, 4), fill)
        x = make(a, fill)
        assert_array_equal(
            xp.linalg.diagonal(x, offset=offset), np.linalg.diagonal(a, offset=offset)
        )
        assert_close(xp.linalg.trace(x, offset=offset), np.linalg.trace(a, offset=offset))
        b = _rand((4, 4), fill, np.int8)
        assert_close(
            xp.linalg.trace(make(b, int(fill)), offset=offset, dtype=xp.int8),
            np.linalg.trace(b, offset=offset, dtype=np.int8),
        )


def test_outer_cross():
    a, b = _rand((4,), 0.0), _rand((3,), 1.0)
    assert_array_equal(xp.linalg.outer(make(a, 0.0), make(b, 1.0)), np.outer(a, b))
    u, v = _rand((2, 3), 0.0, density=0.7), _rand((3,), 2.0, density=0.7)
    assert_array_equal(xp.linalg.cross(make(u, 0.0), make(v, 2.0)), np.cross(u, v))
    u, v = _rand((3, 2), 0.0, density=0.7), _rand((3, 2), 0.0, density=0.7)
    assert_array_equal(
        xp.linalg.cross(make(u, 0.0), make(v, 0.0), axis=-2), np.cross(u, v, axis=-2)
    )
    with pytest.raises(ValueError, match="size 3"):
        xp.linalg.cross(xp.zeros(2), xp.zeros(2))
    with pytest.raises(ValueError, match="1-d"):
        xp.linalg.outer(xp.zeros((2, 2)), xp.zeros(2))


@pytest.mark.parametrize("order", [2, 1, 0, math.inf, -math.inf, 3, -1, 0.5])
def test_vector_norm(order):
    for dtype, fill in [(np.float64, 0.0), (np.float64, -2.0), (np.complex128, 0), (np.int32, 0)]:
        a = _rand((3, 4), fill, dtype)
        x = make(a, fill)
        for axis in [None, 0, -1, (0, 1)]:
            for keepdims in [False, True]:
                expected = np_call(
                    np.linalg.vector_norm, a, axis=axis, keepdims=keepdims, ord=order
                )
                assert_close(
                    np_call(xp.linalg.vector_norm, x, axis=axis, keepdims=keepdims, ord=order),
                    expected,
                )


@pytest.mark.parametrize("order", ["fro", 1, -1, math.inf, -math.inf, 2, -2, "nuc"])
def test_matrix_norm(order):
    for fill in [0.0, 1.0]:
        a = _rand((2, 3, 4), fill)
        for keepdims in [False, True]:
            expected = np.linalg.matrix_norm(a, ord=order, keepdims=keepdims)
            assert_close(
                xp.linalg.matrix_norm(make(a, fill), ord=order, keepdims=keepdims), expected
            )
    with pytest.raises(ValueError, match="order"):
        xp.linalg.matrix_norm(xp.zeros((2, 2)), ord=3)


@pytest.mark.parametrize("n", [0, 1, 2, 5, -1, -2])
def test_matrix_power(n):
    a = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
    for fill in [0.0, 1.0]:
        dense = np.where(a == 0, fill, a)
        expected = np.linalg.matrix_power(dense, n)
        assert_close(xp.linalg.matrix_power(make(dense, fill), n), expected)
    b = np.stack([a, a.T])
    assert_close(xp.linalg.matrix_power(make(b, 0.0), 3), np.linalg.matrix_power(b, 3))
    with pytest.raises(ValueError, match="square"):
        xp.linalg.matrix_power(xp.zeros((2, 3)), 2)


def test_dense_algebra():
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    x = make(a, 0.0)
    la = np.linalg
    assert_close(xp.linalg.cholesky(x), la.cholesky(a))
    assert_close(xp.linalg.cholesky(x, upper=True), la.cholesky(a).T)
    assert_close(xp.linalg.det(x), la.det(a))
    w, _ = xp.linalg.eigh(x)
    assert_close(w, la.eigh(a).eigenvalues)
    assert_close(xp.linalg.eigvalsh(x), la.eigvalsh(a))
    assert_close(xp.linalg.eigvals(x), la.eigvals(a))
    assert xp.linalg.eig(x).eigenvectors.shape == (3, 3)
    assert_close(xp.linalg.inv(x), la.inv(a))
    assert_close(xp.linalg.pinv(x), la.pinv(a))
    assert_close(xp.linalg.matrix_rank(x), la.matrix_rank(a))
    q, r = xp.linalg.qr(x)
    assert_close(q @ r, a)
    sign, logdet = xp.linalg.slogdet(x)
    assert_close(sign, la.slogdet(a).sign)
    assert_close(logdet, la.slogdet(a).logabsdet)
    b = np.array([1.0, 0.0, 2.0])
    assert_close(xp.linalg.solve(x, make(b, 0.0)), la.solve(a, b))
    B = np.stack([a, a])
    assert_close(xp.linalg.solve(make(B, 0.0), make(B, 0.0)), la.solve(B, B))
    s = xp.linalg.svd(x, full_matrices=False).S
    assert_close(s, la.svd(a).S)
    assert_close(xp.linalg.svdvals(x), la.svdvals(a))
    assert_close(xp.linalg.inv(make(np.eye(2, dtype=np.int64), 0)), np.eye(2))
    with gb.config.set(array_api_densify_limit=4), pytest.raises(xp.DensifyError, match="dense"):
        xp.linalg.inv(x)
    with pytest.raises(ValueError, match="2 dimensions"):
        xp.linalg.det(xp.zeros(3))
    with pytest.raises(TypeError, match="Array"):
        xp.linalg.det(a)


def test_nonzero_fills_follow_d6():
    # (review finding M3, DESIGN D12c) a nonzero fill of an operand that stores every
    # position is taken as 0; any other nonzero fill goes dense (no rank-1 split, which turned
    # inf into NaN and lost every digit when the fill dwarfs the values)
    inf = make(np.array([np.inf]), 0.0)
    five = xp.from_coo(([0],), [1.0], (1,), fill_value=5.0)
    assert_close(xp.vecdot(inf, five), np.array(np.inf))
    m = make(np.array([[np.inf, 0.0]]), 0.0)
    k = xp.from_coo(([0, 1], [0, 0]), [1.0, 5.0], (2, 1), fill_value=5.0)
    assert_close(m @ k, [[np.inf]])
    big = xp.from_coo(([0, 1],), [1.0, 2.0], (2,), fill_value=1e20)
    assert_close(xp.vecdot(make(np.array([1.0, -1.0]), 0.0), big), np.array(-1.0))
    big32 = xp.from_coo(([0, 1],), np.array([1.0, 2.0], np.float32), (2,), fill_value=1e8)
    out = xp.vecdot(make(np.array([1.0, -1.0], np.float32), 0.0), big32)
    assert_close(out, np.array(-1.0, np.float32))
    # dense under the limit on the result, not only on the operands (review finding L8)
    with gb.config.set(array_api_densify_limit=1000):
        c = make(np.full((100, 1), np.inf), np.inf)
        d = make(np.full((1, 100), 2.0), np.inf)
        with pytest.raises(xp.DensifyError):
            c @ d
        with pytest.raises(xp.DensifyError):
            xp.tensordot(c, d, axes=0)


def test_tensordot_axes_out_of_bounds():
    # (review finding L2) as numpy: no wrapping modulo the number of axes
    x = make(np.ones((2, 3)), 0.0)
    with pytest.raises(IndexError, match="out of bounds"):
        xp.tensordot(x, x.T, axes=([5], [0]))
    with pytest.raises(IndexError, match="out of bounds"):
        xp.tensordot(x, x.T, axes=([1], [-3]))
    assert_close(xp.tensordot(x, x.T, axes=([-1], [0])), np.ones((2, 3)) @ np.ones((3, 2)))


def test_finite_fills_are_stored_in_full():
    # (review findings R1, R3) an operand with a finite nonzero fill that fits under the
    # densify limit is stored in full, whatever the other operands store: matrices of ones
    # by a large sparse matrix stay sparse, and explicit zeros in the other operand do not
    # change the answer (D10)
    rng = np.random.default_rng(20)
    n = 400
    a = np.zeros((n, n))
    a[rng.integers(0, n, 600), rng.integers(0, n, 600)] = rng.random(600) + 0.5
    A = xp.from_coo(np.nonzero(a), a[np.nonzero(a)], (n, n))
    with gb.config.set(array_api_densify_limit=10 * n):
        r1 = A @ xp.ones((n, 8))
        r2 = A @ xp.full((n, 8), 2.5)
    assert_close(r1, a @ np.ones((n, 8)))
    assert_close(r2, a @ np.full((n, 8), 2.5))
    b = np.ones((2, 1500))
    b[0, 0] = np.inf
    B = xp.asarray(b, fill_value=1.0)
    A1 = xp.from_coo(([0], [0]), np.array([1.0]), (3000, 2))
    r = np.arange(1, 2001)
    A2 = xp.from_coo((np.r_[0, r], np.r_[0, np.zeros(2000, int)]), np.r_[1.0, np.zeros(2000)],
                     (3000, 2))  # fmt: skip
    for x in [A1, A2, xp.prune(A2)]:
        assert (x @ B).todense()[1, 0] == 0.0  # (D6: an unstored zero times inf is no term)

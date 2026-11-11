"""graphblas.array_api: deferred contractions and their two push-down consumers (DESIGN D9)."""

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp
from graphblas.array_api import _lazy, _utils

from .test_array_api_elementwise import assert_array_equal, check_invariants, make

rng = np.random.default_rng(9)


def _rand(shape, fill=0.0, dtype=np.float64, density=0.5):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        a = rng.random(shape) < 0.5
    elif dtype.kind in "iu":
        a = rng.integers(1, 9, shape)
    else:
        a = np.round(rng.normal(size=shape) * 4, 1)
    return np.where(rng.random(shape) < density, a, fill).astype(dtype)


def assert_close(x, expected):
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape, (x.shape, expected.shape)
    got = x.todense()
    if got.dtype.kind in "fc":
        tol = 1e-5 if got.dtype in (np.float32, np.complex64) else 1e-9
        np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(got, expected)


def pending(x):
    return x._tensor is None


class _Seen(list):
    hints = None


@pytest.fixture
def masks(monkeypatch):
    """The masks the pending products were computed under (None: whole)."""
    seen = _Seen()
    compute = _lazy.Contraction.compute

    def recording(self, mask=None, **opts):
        seen.append(mask)
        hints.append(opts)
        return compute(self, mask, **opts)

    hints = []
    seen.hints = hints  # (the options each computation got)

    monkeypatch.setattr(_lazy.Contraction, "compute", recording)
    return seen


def test_contractions_are_pending(masks):
    a, b = _rand((4, 5)), _rand((5, 3))
    x1, x2 = make(a, 0.0), make(b, 0.0)
    for P, expected in [
        (x1 @ x2, a @ b),
        (xp.matmul(x1, x2), a @ b),
        (xp.tensordot(x1, x2, axes=1), a @ b),
        (xp.einsum("ij,jk->ik", x1, x2), a @ b),
        (xp.vecdot(x1, x1), np.vecdot(a, a)),
        (xp.matmul(x1, x2, semiring="lor_land"), (a != 0).astype(int) @ (b != 0) > 0),
    ]:
        assert pending(P)
        # Metadata and repr do not compute it
        assert P.shape == expected.shape
        assert P.ndim == expected.ndim
        assert P.size == expected.size
        assert P.dtype == expected.dtype
        assert P.fill_value == 0
        assert P.device == "cpu"
        assert repr(P) == (
            f"Array(shape={expected.shape}, dtype={expected.dtype}, "
            f"fill_value={P.fill_value.item()!r}, pending)"
        )
        assert pending(P)
        assert_close(P, expected)  # (computes it)
        assert not pending(P)
        assert P._pending is None  # (the snapshots are released)
        T = P._T()
        assert P._T() is T  # (computed once)
    assert masks == [None] * 6
    # Not deferred: 0-d results, empty operands, the kill switch, nonzero fills
    assert not pending(xp.vecdot(make(a[0], 0.0), make(a[1], 0.0)))
    assert not pending(xp.einsum("ij,ij->", x1, x1))
    assert not pending(x1 @ xp.zeros((5, 3)))
    assert not pending(x1 @ make(b, 1.0))
    with gb.config.set(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(_lazy, "enabled", False)
        P = x1 @ x2
        assert not pending(P)
        assert_close(P, a @ b)


def test_snapshots(masks):
    a, b = _rand((4, 5)), _rand((5, 3))
    x1, x2 = make(a, 0.0), make(b, 0.0)
    P = x1 @ x2
    # The snapshots share the operands' data (copy-on-write: O(1))
    T1, T2 = P._pending.tensors
    assert T1._M is x1._T()._M
    assert T2._M is x2._T()._M
    # Changing an operand afterwards changes neither the pending result nor the operand's
    # earlier readers
    x1[0, :] = 7.0
    x1[1, 1] = 0.0
    x2 += 1.0
    assert T1._M is not x1._T()._M
    assert_close(P, a @ b)
    a[0, :] = 7.0
    a[1, 1] = 0.0
    assert_close(x1 @ x2, a @ (b + 1.0))
    # The same Array on both sides
    P = x1 @ x1.T
    x1[...] = 0.0
    assert_close(P, a @ a.T)
    # __setitem__ on a pending Array computes it first
    P = make(a, 0.0) @ make(b, 0.0)
    P[0, 0] = -1.0
    expected = a @ b
    expected[0, 0] = -1.0
    assert_close(P, expected)
    P = make(a, 0.0) @ make(b, 0.0)
    P += 1.0
    assert_close(P, a @ b + 1.0)
    Q = make(a[:, :4], 0.0)
    Q @= make(2 * np.eye(4), 0.0)
    assert not pending(Q)
    assert_close(Q, 2 * a[:, :4])


def test_pending_operands_and_consumers():
    a, b, c = _rand((4, 5)), _rand((5, 3)), _rand((3, 2))
    x1, x2, x3 = make(a, 0.0), make(b, 0.0), make(c, 0.0)
    # A pending operand is computed (one level of deferral)
    P = x1 @ x2
    R = P @ x3
    assert not pending(P)
    assert pending(R)
    assert_close(R, a @ b @ c)
    # Everything else computes it
    for func, np_func in [
        (lambda p: p + 1, lambda p: p + 1),
        (xp.sum, np.sum),
        (lambda p: p.T, lambda p: p.T),
        (lambda p: p[[0, 2]], lambda p: p[[0, 2]]),
        (lambda p: xp.asarray(np.asarray(p)), lambda p: p),
        (lambda p: xp.stack(list(p)), lambda p: p),
    ]:
        P = x1 @ x2
        assert_close(func(P), np_func(a @ b))
        assert not pending(P)
    # (the stored entries: where some term meets)
    structure = (a != 0).astype(int) @ (b != 0).astype(int) > 0
    P = x1 @ x2
    assert P.nnz == structure.sum()
    assert not pending(P)
    P = x1 @ x2
    assert_close(xp.asarray(P.tensor), a @ b)
    assert not pending(P)


def _sddmm_inputs(n=12, k=3, dtype=np.float64):
    S = _rand((n, n), 0.0, dtype, density=0.2)
    A = _rand((n, k), 0.0, dtype, density=1.0)
    B = _rand((n, k), 0.0, dtype, density=1.0)
    return S, A, B


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.complex128])
def test_sddmm(dtype, masks):
    s, a, b = _sddmm_inputs(dtype=dtype)
    S, A, B = make(s, 0), make(a, 0), make(b, 0)
    expected = s * (a @ b.T)
    for func in [
        lambda P: xp.multiply(S, P),
        lambda P: xp.multiply(P, S),
        lambda P: S * P,
        lambda P: P * S,
    ]:
        P = xp.matmul(A, xp.matrix_transpose(B))
        masks.clear()
        R = func(P)
        assert len(masks) == 1
        assert masks[0] is not None  # (computed under S's structure)
        assert pending(P)  # (the masked product is not kept)
        assert_close(R, expected)
        assert R.nnz <= S.nnz
    # Other contractions, and S with explicit zeros
    S = make(s, 0, rng, explicit=0.3)
    for P in [
        xp.einsum("ik,jk->ij", A, B),
        xp.tensordot(A, B, axes=([1], [1])),
        xp.vecdot(A[:, None, :], B[None, :, :]),
    ]:
        masks.clear()
        assert_close(xp.multiply(S, P), expected)
        assert masks[0] is not None
    # A pending S is computed, then serves as the mask
    P = A @ xp.matrix_transpose(B)
    Q = make(np.eye(a.shape[0], dtype=dtype), 0) @ S
    assert_close(xp.multiply(Q, P), expected)


def test_sddmm_bool_and_semirings(masks):
    s, a, b = _sddmm_inputs(dtype=np.bool_)
    S, A, B = make(s, False), make(a, False), make(b, False)
    masks.clear()
    assert_close(xp.multiply(S, A @ xp.matrix_transpose(B)), s & (a @ b.T))
    assert masks[0] is not None
    # A fill other than 0 is computed whole: min_plus's inf
    s, a, b = _sddmm_inputs()
    S = make(s, 0.0)
    A, B = make(np.where(a == 0, np.inf, a), np.inf), make(np.where(b == 0, np.inf, b), np.inf)
    masks.clear()
    P = xp.matmul(A, xp.matrix_transpose(B), semiring="min_plus")
    expected = np.min(A.todense()[:, None, :] + B.todense()[None, :, :], axis=2)
    with np.errstate(invalid="ignore"):
        assert_close(xp.multiply(S, P), s * expected)
    assert masks == [None]


def test_sddmm_falls_back(masks):
    s, a, b = _sddmm_inputs()
    A, B = make(a, 0.0), make(b, 0.0)
    expected = a @ b.T
    # S's fill is not 0, S broadcasts, or S is a Python scalar: the product is computed whole
    for S, s_np in [(make(s, 1.0), s), (make(s[:1], 0.0), s[:1]), (2.0, 2.0)]:
        masks.clear()
        P = A @ xp.matrix_transpose(B)
        assert_close(xp.multiply(S, P), s_np * expected)
        assert [m is None for m in masks] == [True]
    # (an ndarray is an Array with fill 0)
    masks.clear()
    assert_close(xp.multiply(s, A @ xp.matrix_transpose(B)), s * expected)
    assert [m is None for m in masks] == [False]
    # An operand holding inf or NaN: exact, as numpy (0 * inf is NaN outside S)
    for bad in [np.inf, np.nan]:
        a2 = a.copy()
        a2[0, 0] = bad
        A2 = make(a2, 0.0)
        masks.clear()
        R = xp.multiply(make(s, 0.0), A2 @ xp.matrix_transpose(B))
        with np.errstate(invalid="ignore"):
            assert_close(R, s * (a2 @ b.T))
        assert [m is None for m in masks] == [True]


def test_sddmm_overflow_outside_s(masks):
    # A product that may overflow to inf outside S's entries is computed whole: 0 * inf there
    # is NaN, as numpy has it
    s = np.array([[1.0, 0.0], [0.0, 0.0]])
    a = np.full((2, 1), 1e200)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = s * (a @ a.T)
    assert np.isnan(expected[0, 1])
    for func in [
        lambda A, B: A @ xp.matrix_transpose(B),
        lambda A, B: xp.einsum("ik,jk->ij", A, B),
        lambda A, B: xp.vecdot(A[:, None, :], B[None, :, :]),
    ]:
        masks.clear()
        R = xp.multiply(make(s, 0.0), func(make(a, 0.0), make(a, 0.0)))
        assert_array_equal(R, expected)
        assert [m is None for m in masks] == [True]
    # The bound counts the terms of each sum (and keeps half the range for rounding): one
    # term of 6e153 * 6e153 is 3.6e307, three may reach 1.1e308
    a = np.full((3, 3), 6e153)
    for k, pushed in [(1, True), (3, False)]:
        masks.clear()
        A = make(a[:, :k], 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = s[:1, :1] * (a[:1, :k] @ a[:1, :k].T)
            R = xp.multiply(make(s[:1, :1], 0.0), A[:1] @ xp.matrix_transpose(A[:1]))
        assert_array_equal(R, expected)
        assert [m is not None for m in masks] == [pushed]
    # Large values well within the bound, float32 too, and complex
    for dtype, value in [(np.float64, 1e100), (np.float32, 1e15), (np.complex128, 1e100j)]:
        a = np.full((3, 2), value, dtype=dtype)
        s3 = np.eye(3, dtype=dtype)
        masks.clear()
        R = xp.multiply(make(s3, 0), make(a, 0) @ xp.matrix_transpose(make(a, 0)))
        assert_array_equal(R, s3 * (a @ a.T))
        assert [m is not None for m in masks] == [True]
    masks.clear()
    a = np.full((3, 2), 1e20, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.eye(3, dtype=np.float32) * (a @ a.T)
    R = xp.multiply(make(np.eye(3, dtype=np.float32), 0), make(a, 0) @ make(a.T, 0))
    assert_array_equal(R, expected)
    assert [m is None for m in masks] == [True]
    # Computing it whole is guarded: that whole may be dense
    big = make(np.full((4, 1), 1e200), 0.0)
    eye = make(np.eye(4), 0.0)
    with gb.config.set(array_api_densify_limit=10), pytest.raises(xp.DensifyError, match="inf"):
        xp.multiply(eye, big @ xp.matrix_transpose(big))


def test_max_abs():
    assert _lazy._max_abs(xp.zeros(3)._T()) == 0.0
    assert _lazy._max_abs(make(np.array([-5, 3], np.int64), 0)._T()) == 5.0
    assert _lazy._max_abs(make(np.array([True, False]), False)._T()) == 1.0
    assert _lazy._max_abs(make(np.array([3 + 4j, 1j]), 0)._T()) == 5.0
    assert _lazy._max_abs(make(np.array([-2.5, np.nan, 1.0]), 0.0)._T()) == 2.5


def test_finite_facts():
    x = make(np.array([1.0, np.inf]), 0.0)
    assert not _lazy._all_finite(x._T())
    assert not _lazy._all_finite(make(np.array([np.nan, 1.0]), 0.0)._T())
    assert not _lazy._all_finite(make(np.array([np.inf, -np.inf]), 0.0)._T())
    assert not _lazy._all_finite(make(np.array([1 + 0j, complex(0, np.inf)]), 0.0)._T())
    assert _lazy._all_finite(make(np.array([1.0, -2.0]), 0.0)._T())
    assert _lazy._all_finite(make(np.array([5, 7]), 0)._T())
    assert _lazy._all_finite(xp.zeros(3)._T())
    # Facts an Array already knows are taken as they are
    y = make(np.array([1.0, 2.0]), 0.0)
    assert _lazy._known_finite(y) is None
    _utils.fact(y, "finite")
    assert _lazy._known_finite(y) is True
    assert _lazy._known_finite(make(np.array([1, 2]), 0)) is True
    P = xp.matmul(y, make(np.array([[1.0], [2.0]]), 0.0))
    assert P._pending.finite == [True, None]


SLICES = [
    (1,),
    (slice(1, 3),),
    (slice(None), 2),
    (slice(None, None, -2), slice(1, 3)),
    (Ellipsis, 1),
    (None, slice(0, 2)),
    (2, None),
]


@pytest.mark.parametrize("key", SLICES, ids=str)
def test_getitem_slices_the_operands(key, masks):
    a, b = _rand((4, 5)), _rand((5, 3))
    x1, x2 = make(a, 0.0), make(b, 0.0)
    expected = (a @ b)[key]
    for P in [x1 @ x2, xp.tensordot(x1, x2, axes=1), xp.einsum("ij,jk->ik", x1, x2)]:
        masks.clear()
        R = P[key]
        assert pending(P)
        if R.ndim and not any(k is None for k in key):
            assert pending(R)
        assert_close(R, expected)
        assert masks == [None]  # (only the slice was computed)
        assert_close(P, a @ b)


def test_getitem_cases(masks):
    a, b = _rand((2, 4, 5)), _rand((2, 5, 3))
    x1, x2 = make(a, 0.0), make(b, 0.0)
    # Batched (equal batch axes), 1-d operands, three operands
    for P, expected in [
        (x1 @ x2, a @ b),
        (x1[0, 0] @ x2[0], a[0, 0] @ b[0]),
        (x1[0] @ x2[0, :, 0], a[0] @ b[0, :, 0]),
        (
            xp.einsum("bij,bjk,kl->bil", x1, x2, make(b[0].T, 0.0)),
            np.einsum("bij,bjk,kl->bil", a, b, b[0].T),
        ),
        (xp.tensordot(x1, x2, axes=([0, 2], [0, 1])), np.tensordot(a, b, ([0, 2], [0, 1]))),
    ]:
        key = (
            (1,) + (slice(1, None),) * (expected.ndim - 1)
            if expected.ndim > 1
            else (slice(1, None),)
        )
        masks.clear()
        R = P[key]
        assert pending(P)
        assert_close(R, expected[key])
    # Broadcast batch axes: no slicing of the operands, computed whole
    y1 = make(a[:1], 0.0)
    P = y1 @ x2
    assert_close(P[1, 1:], (a[:1] @ b)[1, 1:])
    assert not pending(P)
    # vecdot has no slicing; all-int keys, full slices and index arrays compute whole
    for P, expected, key in [
        (xp.vecdot(x1, x1), np.vecdot(a, a), (0, slice(1, 3))),
        (x1 @ x2, a @ b, (0, 1, 2)),
        (x1 @ x2, a @ b, (slice(None),)),
        (x1 @ x2, a @ b, (Ellipsis,)),
        (x1 @ x2, a @ b, (np.array([0, 1]),)),
        (x1[0, 0] @ x2[0], a[0, 0] @ b[0], (1,)),
    ]:
        assert pending(P)
        assert_close(P[key], expected[key])
        assert not pending(P)
    # Slices of empty extent, and slicing then the SDDMM
    P = make(a[0], 0.0) @ make(b[0], 0.0)
    assert_close(P[2:2], (a[0] @ b[0])[2:2])
    s = _rand((2, 3), 0.0, density=0.5)
    masks.clear()
    R = xp.multiply(make(s, 0.0), P[1:3])
    assert_close(R, s * (a[0] @ b[0])[1:3])
    assert [m is not None for m in masks] == [True]


def test_iteration_computes_once(masks):
    a, b = _rand((4, 5)), _rand((5, 3))
    P = make(a, 0.0) @ make(b, 0.0)
    rows = list(P)
    assert masks == [None]
    for row, expected in zip(rows, a @ b, strict=True):
        assert_close(row, expected)


def test_sddmm_dot_products(masks):
    # A @ B with B held by row (k x n) under a sparse mask: dot products at the mask's
    # entries, not SuiteSparse's masked saxpy over the whole product (found by the SAPS run)
    rng = np.random.default_rng(13)
    n, k = 60, 4
    s = np.where(rng.random((n, n)) < 0.02, rng.random((n, n)), 0.0)
    a, b = rng.random((n, k)) + 0.5, rng.random((k, n)) + 0.5
    S, A, B = make(s, 0.0), xp.asarray(a), xp.asarray(b)
    for P, expected in [
        (A @ B, s * (a @ b)),
        (xp.tensordot(A, B, axes=1), s * (a @ b)),
        (xp.einsum("ik,kj->ij", A, B), s * (a @ b)),
        (xp.einsum("ik,jk->ij", A, xp.asarray(b.T)), s * (a @ b)),
        ((A @ B)[: n // 2], s[: n // 2] * (a @ b)[: n // 2]),
    ]:
        masks.clear()
        masks.hints.clear()
        S2 = S if expected.shape == s.shape else S[: n // 2]
        assert_close(xp.multiply(S2, P), expected)
        assert masks.hints == [{"axb_method": "dot"}]
    # Not for a dense mask, a vecdot, or an einsum whose product is permuted after
    for S2, P, expected in [
        (make(np.ones((n, n)), 0.0), A @ B, a @ b),
        (S, xp.vecdot(A[:, None, :], xp.asarray(b.T)[None, :, :]), s * (a @ b)),
        (S, xp.einsum("ik,kj->ji", A, B), s * (a @ b).T),
    ]:
        masks.clear()
        masks.hints.clear()
        assert_close(xp.multiply(S2, P), expected)
        assert masks.hints == [{}]


@pytest.mark.parametrize(
    "op",
    ["iadd", "isub", "imul", "itruediv", "ifloordiv", "imod", "ipow", "imatmul", "iand", "ior",
     "ixor", "ilshift", "irshift"],
)  # fmt: skip
def test_inplace_operators_on_a_pending_array(op):
    # (review finding H1) an in-place operator replaces a pending product: later consumers
    # (the SDDMM, slicing) must read the new value, not the old product
    import operator

    func = getattr(operator, op)
    rng = np.random.default_rng(14)
    dtype = np.int64 if op in {"iand", "ior", "ixor", "ilshift", "irshift"} else np.float64
    a = rng.integers(1, 5, (4, 4)).astype(dtype)
    s = np.zeros((4, 4), dtype)
    s[0, 0], s[1, 2], s[3, 3] = 2, 3, 1
    s2 = np.eye(4, dtype=dtype)
    P = make(a, 0) @ make(a, 0)
    assert pending(P)
    with np.errstate(all="ignore"):
        expected = func((a @ a).copy(), s if op != "imatmul" else s2)
        P = func(P, make(s, 0) if op != "imatmul" else make(s2, 0))
        assert not pending(P)
        assert P._pending is None
        assert_close(P, expected)
        assert_close(xp.multiply(make(s2, 0), P), s2 * expected)
        assert_close(P[1:], expected[1:])

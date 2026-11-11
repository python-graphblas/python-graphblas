"""graphblas.array_api statistical and utility functions against numpy."""

import warnings

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp

from .test_array_api_elementwise import (
    DTYPES,
    assert_array_equal,
    check_invariants,
    inputs,
    make,
    np_call,
)

REDUCTIONS = [
    ("sum", np.sum),
    ("prod", np.prod),
    ("min", np.min),
    ("max", np.max),
    ("any", np.any),
    ("all", np.all),
    ("mean", np.mean),
    ("var", np.var),
    ("std", np.std),
]
SHAPES_AXES = [
    ((6,), [None, 0, -1, (0,), ()]),
    ((3, 4), [None, 0, 1, -2, (0, 1), (1, 0), ()]),
    ((2, 3, 4), [None, 0, 2, (0, 2), (1, 2), (0, 1, 2)]),
    ((), [None, ()]),
]


def assert_close(x, expected, scale=1.0):
    """``x`` equals ``expected`` up to rounding of floating-point sums (exactly otherwise);
    ``scale`` is the size of the largest terms summed, which bounds the rounding.
    """
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape
    got = x.todense()
    if got.dtype.kind in "fc":
        rtol = 1e-4 if got.dtype in (np.float32, np.complex64) else 1e-10
        atol = rtol * scale
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True)
        # an infinity's sign, and NaN where numpy has NaN, are not rounding
        assert (np.isnan(got) == np.isnan(expected)).all()
        assert (np.isinf(got) == np.isinf(expected)).all()
    else:
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(("name", "func"), REDUCTIONS, ids=[n for n, _ in REDUCTIONS])
@pytest.mark.parametrize(("shape", "axes"), SHAPES_AXES, ids=str)
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions(name, func, shape, axes, keepdims):
    for dtype in DTYPES:
        for dense, x in inputs(dtype, shape, seed=11):
            for axis in axes:
                kwargs = {"axis": axis, "keepdims": keepdims}
                try:
                    expected = np_call(func, dense, **kwargs)
                except (TypeError, ValueError) as exc:
                    with pytest.raises(type(exc)):
                        getattr(xp, name)(x, **kwargs)
                    continue
                if name in {"min", "max"} and dtype.kind == "c":
                    with pytest.raises(TypeError):
                        getattr(xp, name)(x, **kwargs)
                    continue
                if np.asarray(expected).dtype not in set(DTYPES):
                    continue
                out = getattr(xp, name)(x, **kwargs)
                exact = name in {"min", "max", "any", "all"} or dtype.kind in "biu"
                exact = exact and name not in {"mean", "var", "std"}
                try:
                    if exact:
                        assert_array_equal(out, np.asarray(expected))
                    elif name == "prod" and not np.isfinite(dense).all():
                        # (a product of infinities and zeros: NaN or not depends on the order)
                        pass
                    else:
                        finite = dense[np.isfinite(dense)] if dense.dtype.kind in "fc" else dense
                        scale = float(np.abs(finite.astype(np.complex128)).max(initial=1.0))
                        assert_close(out, expected, scale * dense.size)
                except AssertionError as exc:
                    raise AssertionError(
                        f"{name} of {dtype} fill {x.fill_value!r} axis={axis}: {exc}"
                    ) from None


@pytest.mark.parametrize("name", ["sum", "prod"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "float32", "bool"])
def test_sum_prod_dtype(name, dtype):
    func = getattr(np, name)
    for dense, x in inputs(np.dtype(dtype), (3, 5), seed=2):
        for out_dtype in [None, np.dtype("int64"), np.dtype("float64"), np.dtype("uint64")]:
            if out_dtype is not None and dtype == "float32" and out_dtype.kind in "iu":
                continue
            expected = np_call(func, dense, axis=1, dtype=out_dtype)
            out = getattr(xp, name)(x, axis=1, dtype=out_dtype)
            assert_close(out, expected)


def test_sum_wraps_like_numpy():
    a = np.array([[100, 100, 0, 0], [0, 0, 0, 50]], dtype=np.int8)
    x = make(a, 0)
    assert_array_equal(xp.sum(x, axis=1, dtype=xp.int8), np.sum(a, axis=1, dtype=np.int8))
    # a nonzero fill counted many times over
    x = make(np.full((2, 300), 7, dtype=np.int8), 0) + 0
    y = xp.full((2, 300), 7, dtype=xp.int8)
    assert_array_equal(
        xp.sum(y, axis=1, dtype=xp.int8), np.full(2, 7 * 300, np.int64).astype(np.int8)
    )
    assert_array_equal(
        xp.prod(y, axis=1, dtype=xp.int8),
        np.prod(np.full((2, 300), 7, np.int8), axis=1, dtype=np.int8),
    )
    assert_array_equal(xp.prod(y, axis=1), np.prod(np.full((2, 300), 7, np.int8), axis=1))
    del x


def test_min_max_nan_propagates():
    a = np.array([[1.0, np.nan, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0, 0, 0, 0]])
    x = make(a, 0.0)
    for name in ["min", "max"]:
        for axis in [None, 0, 1]:
            assert_array_equal(getattr(xp, name)(x, axis=axis), getattr(np, name)(a, axis=axis))
    x = make(np.array([[1.0, np.nan], [np.nan, np.nan]]), np.nan)
    assert_array_equal(xp.max(x, axis=1), np.array([np.nan, np.nan]))


def test_empty_axis():
    x = xp.zeros((0, 3))
    assert_array_equal(xp.sum(x, axis=0), np.zeros(3))
    assert_array_equal(xp.prod(x, axis=0), np.ones(3))
    assert_array_equal(xp.any(x, axis=0), np.zeros(3, bool))
    assert_array_equal(xp.all(x, axis=0), np.ones(3, bool))
    assert_array_equal(xp.sum(x, axis=1), np.zeros(0))
    with pytest.raises(ValueError, match="empty"):
        xp.max(x, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_array_equal(xp.mean(x, axis=0), np.mean(np.zeros((0, 3)), axis=0))
        assert_array_equal(xp.var(x, axis=0), np.var(np.zeros((0, 3)), axis=0))
    assert_array_equal(xp.max(x, axis=1), np.zeros(0))


@pytest.mark.parametrize("correction", [0, 1, 2.5, 4, 10])
def test_var_correction(correction):
    rng = np.random.default_rng(3)
    a = np.where(rng.random((3, 4)) < 0.5, 0.0, np.round(rng.normal(size=(3, 4)), 2))
    for fill in [0.0, 1.5]:
        dense = np.where(a == 0, fill, a)
        x = make(dense, fill)
        for axis in [None, 0, 1, ()]:
            for name in ["var", "std"]:
                expected = np_call(getattr(np, name), dense, axis=axis, ddof=correction)
                out = np_call(getattr(xp, name), x, axis=axis, correction=correction)
                assert_close(out, expected)


def test_reductions_on_full_arrays():
    a = np.arange(1.0, 13.0).reshape(3, 4)
    x = make(a, 0.0)
    assert x.nnz == 12
    for name, func in REDUCTIONS:
        for axis in [None, 0, 1]:
            assert_close(getattr(xp, name)(x, axis=axis), func(a, axis=axis))


def test_reductions_of_fill_only_arrays():
    x = xp.full((4, 5), 2.5)
    assert_close(xp.sum(x, axis=1), np.full(4, 12.5))
    assert xp.sum(x, axis=1).nnz == 0
    assert_close(xp.prod(x), np.prod(np.full((4, 5), 2.5)))
    assert_close(xp.var(x, axis=0), np.zeros(5))
    x = xp.full((3, 3), np.inf)
    assert_array_equal(np_call(xp.var, x, axis=0), np_call(np.var, np.full((3, 3), np.inf), axis=0))


@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
@pytest.mark.parametrize("include_initial", [False, True])
def test_cumulative(name, include_initial):
    func = np.cumsum if name == "cumulative_sum" else np.cumprod
    for dtype in [np.dtype(t) for t in ("bool", "int8", "uint16", "int64", "float32", "float64")]:
        for shape, axes in [((7,), [None, 0, -1]), ((3, 4), [0, 1, -1]), ((2, 3, 2), [0, 2])]:
            for dense, x in inputs(dtype, shape, seed=5):
                for axis in axes:
                    np_axis = 0 if axis is None else axis
                    expected = np_call(func, dense, axis=np_axis)
                    if include_initial:
                        initial = np.zeros if name == "cumulative_sum" else np.ones
                        shp = list(expected.shape)
                        shp[np_axis] = 1
                        expected = np.concatenate(
                            [initial(shp, expected.dtype), expected], axis=np_axis
                        )
                    out = getattr(xp, name)(x, axis=axis, include_initial=include_initial)
                    assert_close(out, expected)
    x = make(np.ones((2, 3)), 0.0)
    with pytest.raises(ValueError, match="axis"):
        getattr(xp, name)(x)
    out = getattr(xp, name)(make(np.array([1, 2, 3], np.int8), 0), dtype=xp.float32)
    assert out.dtype == xp.float32


def test_cumulative_is_guarded():
    with gb.config.set(array_api_densify_limit=10), pytest.raises(xp.DensifyError):
        xp.cumulative_sum(xp.zeros(100))


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_diff(n):
    for dtype in [np.dtype(t) for t in ("bool", "int16", "uint8", "float64", "complex64")]:
        for shape, axes in [((6,), [0, -1]), ((3, 4), [0, 1])]:
            for dense, x in inputs(dtype, shape, seed=6):
                for axis in axes:
                    expected = np_call(np.diff, dense, n=n, axis=axis)
                    assert_array_equal(xp.diff(x, n=n, axis=axis), expected)


def test_diff_prepend_append():
    a = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, -1.0]])
    x = make(a, 0.0)
    p = np.array([[5.0], [0.0]])
    q = np.array([[0.0, 2.0, 0.0]])
    assert_array_equal(xp.diff(x, axis=1, prepend=make(p, 0.0)), np.diff(a, axis=1, prepend=p))
    assert_array_equal(xp.diff(x, axis=0, append=make(q, 1.0)), np.diff(a, axis=0, append=q))
    assert_array_equal(xp.diff(x, axis=1, append=2.0), np.diff(a, axis=1, append=2.0))
    assert_array_equal(
        xp.diff(x, axis=1, n=2, prepend=make(p, 0.0), append=make(p, 7.0)),
        np.diff(a, axis=1, n=2, prepend=p, append=p),
    )


def test_axis_errors():
    x = make(np.ones((2, 3)), 0.0)
    with pytest.raises(IndexError):
        xp.sum(x, axis=2)
    with pytest.raises(ValueError, match="repeated"):
        xp.sum(x, axis=(0, -2))

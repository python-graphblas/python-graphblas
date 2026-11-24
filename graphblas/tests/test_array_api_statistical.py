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
    fills_for,
    inputs,
    make,
    np_call,
    random_values,
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
                        # (the Array API leaves the order of -0.0 and 0.0 in min and max to the
                        # library, and numpy's answer differs between x86 and arm64)
                        zero_signs = name not in {"min", "max"}
                        assert_array_equal(out, np.asarray(expected), zero_signs=zero_signs)
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


NAN_REDUCTIONS = ["nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanvar", "nanstd"]


def nan_inputs(dtype, shape, seed=0):
    """(dense, Array) pairs, one per fill of the dtype: floats get NaN stored at random, a
    slice along the last axis of NaN only, and (for fill NaN) NaN at unstored positions.
    """
    rng = np.random.default_rng(seed)
    for i, fill in enumerate(fills_for(dtype)):
        dense = random_values(rng, dtype, shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fillv = np.asarray(fill).astype(dtype)[()]
        if dense.ndim:
            dense = np.where(rng.random(shape) < 0.4, fillv, dense).astype(dtype)
            if dtype.kind in "fc":
                dense = np.where(rng.random(shape) < 0.2, np.nan, dense).astype(dtype)
                if dense.size and dense.ndim > 1:
                    dense[(0,) * (dense.ndim - 1)] = np.nan
        yield dense, make(dense, fillv, rng if i % 2 else None)


@pytest.mark.parametrize("name", NAN_REDUCTIONS)
@pytest.mark.parametrize(("shape", "axes"), SHAPES_AXES, ids=str)
@pytest.mark.parametrize("keepdims", [False, True])
def test_nan_reductions(name, shape, axes, keepdims):
    # numpy's nan-functions skip NaN, stored or as the fill; a slice of NaN only gives 0, 1
    # or NaN as numpy's do
    func = getattr(np, name)
    for dtype in DTYPES:
        for dense, x in nan_inputs(dtype, shape, seed=13):
            for axis in axes:
                for extra in (
                    [{}, {"ddof": 1}, {"ddof": 2}] if name in {"nanvar", "nanstd"} else [{}]
                ):
                    kwargs = {"axis": axis, "keepdims": keepdims, **extra}
                    if name in {"nanmax", "nanmin"} and dtype.kind == "c":
                        with pytest.raises(TypeError, match="real-valued"):
                            getattr(xp, name)(x, **kwargs)
                        continue
                    try:
                        expected = np_call(func, dense, **kwargs)
                    except (TypeError, ValueError) as exc:
                        with pytest.raises(type(exc)):
                            getattr(xp, name)(x, **kwargs)
                        continue
                    if np.asarray(expected).dtype not in set(DTYPES):
                        continue
                    out = getattr(xp, name)(x, **kwargs)
                    values = dense[~np.isnan(dense)] if dense.dtype.kind in "fc" else dense
                    try:
                        if name in {"nanmax", "nanmin"} or (
                            dtype.kind in "biu" and name in {"nansum", "nanprod"}
                        ):
                            assert_array_equal(out, np.asarray(expected), zero_signs=False)
                        elif name == "nanprod" and _order_matters(values):
                            pass  # (infinities, or overflow, and zeros: NaN or not by order)
                        else:
                            finite = (
                                values[np.isfinite(values)] if values.dtype.kind in "fc" else values
                            )
                            scale = float(np.abs(finite.astype(np.complex128)).max(initial=1.0))
                            if name in {"nanvar", "nanstd"}:
                                # (squares; capped so the tolerance fits float32)
                                single = dense.dtype in (np.float32, np.complex64)
                                scale = min(scale * scale, 1e30 if single else 1e300)
                            _assert_close_parts(out, expected, scale * max(dense.size, 1))
                    except AssertionError as exc:
                        raise AssertionError(
                            f"{name} of {dtype} fill {x.fill_value!r} {kwargs}: {exc}"
                        ) from None


def _order_matters(values):
    """Whether a product of ``values`` may be NaN or not by the order of its factors: it has a
    zero and an infinity, or factors whose product overflows.
    """
    if not values.size or not (values == 0).any():
        return False
    with np.errstate(all="ignore"):
        big = np.prod(np.abs(values[values != 0]).astype(np.float64))
    return not np.isfinite(values).all() or big > np.finfo(values.real.dtype).max


def _assert_close_parts(out, expected, scale):
    """``assert_close``, part by part for complex results (a complex infinity minus itself is
    NaN, which ``assert_allclose`` would report).
    """
    if out.dtype.kind != "c":
        assert_close(out, expected, scale)
        return
    expected = np.asarray(expected)
    assert out.dtype == expected.dtype
    assert_close(xp.real(out), expected.real, scale)
    assert_close(xp.imag(out), expected.imag, scale)


def test_nan_reductions_keywords():
    rng = np.random.default_rng(17)
    dense = np.where(rng.random((4, 5)) < 0.3, np.nan, rng.normal(size=(4, 5))).astype(np.float32)
    x = make(dense, np.nan, rng)
    # dtype: the sums' dtype (numpy rounds var's deviations to the input's dtype, float32
    # here, which these compute in dtype: so a float32 tolerance)
    for name in ["nansum", "nanprod", "nanmean", "nanvar", "nanstd"]:
        expected = np_call(getattr(np, name), dense, axis=1, dtype=np.float64)
        out = getattr(xp, name)(x, axis=1, dtype=xp.float64)
        assert out.dtype == expected.dtype
        np.testing.assert_allclose(out.todense(), expected, rtol=1e-6, equal_nan=True)
    ints = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert_close(
        xp.nanmean(make(ints, 0), axis=0, dtype=xp.float32),
        np.nanmean(ints, axis=0, dtype=np.float32),
    )
    # ddof and its Array API spelling, correction
    assert_close(
        xp.nanvar(x, axis=0, correction=1), np_call(np.nanvar, dense, axis=0, ddof=1), 10.0
    )
    with pytest.raises(ValueError, match="ddof and correction"):
        xp.nanvar(x, ddof=1, correction=1)
    with pytest.raises(TypeError, match="out="):
        xp.nansum(x, out=np.empty(5))
    with pytest.raises(TypeError, match="needs an Array"):
        xp.nanmean(dense)
    # empty axes, as numpy gives them (nanmax and nanmin of none raise)
    for shape in [(0, 3), (3, 0)]:
        e = make(np.zeros(shape), 0.0)
        for name in NAN_REDUCTIONS:
            for axis in [0, 1, None]:
                try:
                    expected = np_call(getattr(np, name), np.zeros(shape), axis=axis)
                except ValueError:
                    with pytest.raises(ValueError, match="empty axis"):
                        getattr(xp, name)(e, axis=axis)
                    continue
                assert_close(getattr(xp, name)(e, axis=axis), expected)


def test_nan_reductions_stay_sparse():
    # A 1e5 x 1e5 array with 1000 entries (50 of them NaN), filled with NaN or 0: every
    # nan-function, over each axis, under a densify limit below the result's size
    rng = np.random.default_rng(19)
    n, k = 10**5, 1000
    flat = rng.choice(n * 10, k, replace=False)
    rows, cols = flat // 10, flat % 10  # (ten columns, so that columns hold many entries)
    values = rng.normal(size=k)
    values[:50] = np.nan
    for fill in [np.nan, 0.0]:
        x = xp.from_coo((rows, cols), values, (n, n), fill_value=fill)
        with gb.config.set(array_api_densify_limit=10**4):
            for name in NAN_REDUCTIONS:
                for axis in [None, 0, 1]:
                    out = getattr(xp, name)(x, axis=axis)
                    assert out.nnz <= k
        # each column's values, against numpy on that column alone
        for name in NAN_REDUCTIONS:
            out = getattr(xp, name)(x, axis=0)
            for col in range(3):
                dense = np.full(n, fill)
                dense[rows[cols == col]] = values[cols == col]
                assert_close(out[col], np_call(getattr(np, name), dense), 10.0 * n)


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
    y = xp.full((2, 300), 7, dtype=xp.int8)
    assert_array_equal(
        xp.sum(y, axis=1, dtype=xp.int8), np.full(2, 7 * 300, np.int64).astype(np.int8)
    )
    assert_array_equal(
        xp.prod(y, axis=1, dtype=xp.int8),
        np.prod(np.full((2, 300), 7, np.int8), axis=1, dtype=np.int8),
    )
    assert_array_equal(xp.prod(y, axis=1), np.prod(np.full((2, 300), 7, np.int8), axis=1))


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
        xp.cumulative_sum(xp.full(100, 1.0))
    # with the identity as fill only the positions from each fiber's first entry are stored
    y = xp.zeros((20000, 20000))
    y[[1, 5, 9], [3, 19000, 7]] = 1.0
    assert xp.cumulative_sum(y, axis=1).nnz == 19997 + 1000 + 19993
    assert xp.cumulative_sum(xp.zeros(10**12)).nnz == 0
    with gb.config.set(array_api_densify_limit=100), pytest.raises(xp.DensifyError):
        xp.cumulative_sum(y, axis=1)


def test_cumulative_sparse_uneven_fibers():
    # one long fiber among many short ones: the fibers accumulate one by one
    a = np.zeros((200, 300))
    a[0] = np.arange(1.0, 301.0)
    a[np.arange(1, 200), np.arange(1, 200)] = -0.5
    x = make(a, 0.0)
    assert_array_equal(xp.cumulative_sum(x, axis=1), np.cumsum(a, axis=1))
    c = xp.zeros(1, dtype=xp.complex64)
    assert_array_equal(xp.cumulative_sum(c, include_initial=True), np.zeros(2, np.complex64))


@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
def test_cumulative_sparse_exact(name):
    # runs of unstored positions add (multiply by) the fill: 0.0 + -0.0 is 0.0, as in numpy
    func = np.cumsum if name == "cumulative_sum" else np.cumprod
    identity = 0 if name == "cumulative_sum" else 1
    rng = np.random.default_rng(17)
    specials = np.array([-0.0, 1.5, -2.0, 3.0, np.inf, np.nan, 0.5])
    for dtype in [np.float64, np.float32, np.int16, np.uint8, np.bool_]:
        for shape in [(9,), (3, 7), (2, 3, 5), (1, 40)]:
            if np.dtype(dtype).kind == "f":
                vals = rng.choice(specials, shape)
            else:
                vals = rng.integers(0, 4, shape)
            a = np.where(rng.random(shape) < 0.6, identity, vals).astype(dtype)
            x = make(a, identity, rng)
            for axis in range(len(shape)):
                for include_initial in [False, True]:
                    expected = np_call(func, a, axis=axis)
                    if include_initial:
                        s = list(expected.shape)
                        s[axis] = 1
                        initial = np.full(s, identity, expected.dtype)
                        expected = np.concatenate([initial, expected], axis=axis)
                    out = np_call(getattr(xp, name), x, axis=axis, include_initial=include_initial)
                    assert_array_equal(out, expected)


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


def test_var_of_a_huge_sparse_array():
    # The means are not stored densely: a 1e9 x 1e9 array with 4 entries
    n = 10**9
    coords = (np.array([0, 5, 5, 10**8]), np.array([3, 3, 7, 3]))
    x = xp.from_coo(coords, np.array([1.0, 2.0, 4.0, 8.0]), (n, n))
    v = xp.var(x, axis=0)
    assert v.nnz == 2
    assert v.fill_value == 0.0
    for col, values in [(3, [1.0, 2.0, 8.0]), (7, [4.0])]:
        values = np.array(values)
        mean = values.sum() / n
        expected = (((values - mean) ** 2).sum() + (n - values.size) * mean**2) / n
        assert float(v[col]) == pytest.approx(expected, rel=1e-12)


def test_sum_starts_from_positive_zero():
    # numpy's sum starts from +0.0: a sum of -0.0 is 0.0
    for a in [
        np.array([-0.0, -0.0]),
        np.array([[-0.0, 0.0], [-0.0, 1.0]]),
        np.array([-0.0], np.float32),
    ]:
        x = make(a, 0.0)
        for axis in [None, -1]:
            assert_array_equal(xp.sum(x, axis=axis), np.sum(a, axis=axis))
            assert_array_equal(xp.mean(x, axis=axis), np.mean(a, axis=axis))
    c = np.array([complex(-0.0, -0.0)])
    assert_array_equal(xp.sum(make(c, 0j)), np.sum(c))
    x = make(np.array([1.0, -0.0]), -0.0)
    assert_array_equal(xp.sum(x, axis=0), np.sum(np.array([1.0, -0.0])))
    assert_array_equal(xp.sum(xp.full((3,), -0.0)), np.sum(np.full(3, -0.0)))


def test_min_max_of_signed_zeros():
    # GraphBLAS's fmin and fmax may return either zero, and the Array API leaves the order of
    # -0.0 and 0.0 unspecified, so zero signs are not compared. NaN propagates.
    cases = [
        (np.array([-0.0, 0.0]), 1.0),
        (np.array([0.0, -0.0]), 1.0),
        (np.array([[0.0, 2.0], [-0.0, 3.0]]), 5.0),
        (np.array([[np.nan, -0.0], [np.nan, 0.0]]), np.nan),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), -0.0),
        (np.array([[-0.0, 1.0], [2.0, -0.0]]), 0.0),
    ]
    for a, fill in cases:
        for x in [make(a, fill), make(np.where(a == 0, fill, a), fill)]:
            dense = x.todense()
            for name in ["min", "max"]:
                for axis in [None, 0, -1]:
                    assert_array_equal(
                        getattr(xp, name)(x, axis=axis),
                        getattr(np, name)(dense, axis=axis),
                        zero_signs=False,
                    )


def test_complex_sum_of_infinite_fill():
    # k copies of inf+0j sum to inf+0j, not inf+nanj
    for fill in [complex(np.inf, 0), complex(np.nan, 0), complex(0, -np.inf)]:
        a = np.array([[1 + 1j, fill, fill], [fill, fill, fill]])
        x = make(a, fill)
        for axis in [None, 0, 1]:
            for name in ["sum", "mean"]:
                assert_array_equal(
                    np_call(getattr(xp, name), x, axis=axis),
                    np_call(getattr(np, name), a, axis=axis),
                )


def test_sum_in_bool():
    # a sum in bool is the logical or of the values, and a product their logical and
    a = np.array([[1, 0, 3], [0, 0, 2]])
    for fill in [0, 2]:
        x = make(np.where(a == 0, fill, a), fill)
        dense = x.todense()
        for axis in [None, 0, 1]:
            assert_array_equal(
                xp.sum(x, axis=axis, dtype=xp.bool), np.sum(dense, axis=axis, dtype=bool)
            )
            assert_array_equal(
                xp.prod(x, axis=axis, dtype=xp.bool), np.prod(dense, axis=axis, dtype=bool)
            )


def test_mean_of_empty_complex():
    # the mean of an empty complex array is nan+nanj (0j / 0), as in numpy
    for dt in [np.complex128, np.complex64, np.float32]:
        assert_array_equal(
            np_call(xp.mean, xp.zeros((0,), dtype=dt)), np_call(np.mean, np.zeros(0, dt))
        )
        assert_array_equal(
            np_call(xp.mean, xp.zeros((2, 0), dtype=dt), axis=1),
            np_call(np.mean, np.zeros((2, 0), dt), axis=1),
        )


def test_diff_n0_ignores_prepend_append():
    # as in numpy, diff with n=0 returns the values unchanged and ignores prepend and append
    a = np.array([1.0, 0.0, 2.0])
    x = make(a, 0.0)
    assert_array_equal(
        xp.diff(x, n=0, prepend=make(np.array([5.0]), 0.0)), np.diff(a, n=0, prepend=5.0)
    )
    assert_array_equal(xp.diff(x, n=0, append=7.0), np.diff(a, n=0, append=7.0))


def test_reductions_of_fibers_past_2_63_positions():
    # a fiber of 2**63 positions: counts do not overflow
    shape = (2**21,) * 3
    y = xp.full(shape, 1.0)
    y[0, 0, 0] = 3.0
    assert float(xp.sum(y)) == float(2**63 + 2)
    assert float(xp.sum(y, axis=(1, 2))[0]) == 2.0**42 + 2
    assert float(xp.max(y)) == 3.0
    z = xp.full(shape, 1, dtype=xp.int64)
    z[0, 0, 0] = 3
    assert int(xp.sum(z)) == 3 + (2**63 - 1) - 2**64  # (wraps, as integer sums do)
    assert int(xp.prod(z)) == 3


def _wrapped(value, dtype):
    """The Python int ``value`` wrapped to the integer ``dtype``, as numpy's products wrap."""
    bits = np.dtype(dtype).itemsize * 8
    value %= 1 << bits
    return value - (1 << bits) if np.dtype(dtype).kind == "i" and value >> (bits - 1) else value


@pytest.mark.parametrize("dtype", [xp.int64, xp.uint64, xp.int8])
# sizes 2**63 + 2**32, 2**64 + 2 and 2**70
@pytest.mark.parametrize("shape", [(2**31 + 1, 2**32), (239075442, 77158673929), (2**40, 2**30)])
def test_integer_prod_over_huge_fibers(dtype, shape):
    """An integer product over 2**63 positions or more equals the repeated product wrapped to
    the dtype, as numpy's does. The count of unstored positions does not wrap to a negative
    exponent, and an even fill gives 0 when the count wraps modulo 2**64 (to 0 for size
    2**64 + 2 and two entries).
    """
    L = shape[0] * shape[1]
    for fill in [2, 3, -3]:
        if fill < 0 and dtype == xp.uint64:
            continue
        x = xp.full(shape, fill, dtype=dtype)
        x[0, 0] = 5
        x[0, 1] = 7
        out = xp.prod(x, dtype=dtype)
        assert int(out) == _wrapped(35 * pow(fill, L - 2, 1 << 64), dtype)
        # by rows: row 0 has 2 entries, the others none
        rows = xp.prod(x[:3], axis=1, dtype=dtype)
        expected = [
            _wrapped(35 * pow(fill, shape[1] - 2, 1 << 64), dtype),
            _wrapped(pow(fill, shape[1], 1 << 64), dtype),
        ]
        assert [int(v) for v in rows.todense()] == [expected[0], expected[1], expected[1]]
        cols = xp.prod(x[:, :2], axis=0, dtype=dtype)
        assert [int(v) for v in cols.todense()] == [
            _wrapped(5 * pow(fill, shape[0] - 1, 1 << 64), dtype),
            _wrapped(7 * pow(fill, shape[0] - 1, 1 << 64), dtype),
        ]


@pytest.mark.parametrize("dtype", [xp.float64, xp.float32])
def test_float_prod_sign_past_2_53_positions(dtype):
    """A product of a negative fill over more than 2**53 positions has the sign of the exact
    count's parity, which float64 cannot hold.
    """
    L = 2**54 + 1
    for fill, odd, even in [(-1.0, -1.0, 1.0), (-0.0, -0.0, 0.0), (-np.inf, -np.inf, np.inf)]:
        x = xp.full((3, L), fill, dtype=dtype)
        x[0, 0] = 3.0  # L - 1 unstored: even
        x[1, 0] = 3.0  # L - 2 unstored: odd
        x[1, 1] = 2.0
        expected = np.array([3.0 * even, 6.0 * odd, odd], dtype=dtype)
        assert_array_equal(xp.prod(x, axis=1), expected)
        assert np.array_equal(np.signbit(xp.prod(x, axis=1).todense()), np.signbit(expected))
        out = xp.prod(xp.full((L,), fill, dtype=dtype))
        assert out.dtype == dtype
        assert np.signbit(float(out))
        assert float(out) == odd
        out = xp.prod(x)  # (3 * 3 * 2 times the fill to the power 3 * L - 3, which is even)
        assert float(out) == 18.0 * even
        assert not np.signbit(float(out))


def test_bool_sum_over_fibers_of_2_64_positions():
    """A sum in bool (a logical or) of True fills is True however many positions there are:
    the count of unstored positions does not wrap to 0 at 2**64.
    """
    x = xp.full((2**32, 2**32), True)
    assert bool(xp.sum(x, dtype=xp.bool))
    y = xp.full((3, 2**32, 2**32), True)
    y[0, 0, 0] = False
    assert_array_equal(xp.sum(y, axis=(1, 2), dtype=xp.bool), np.array([True, True, True]))
    z = xp.full((274177, 67280421310721), True)  # (2**64 + 1 positions)
    z[0, 0] = False  # (2**64 unstored)
    assert bool(xp.sum(z, dtype=xp.bool))
    assert not bool(xp.sum(xp.full((2**32, 2**32), False), dtype=xp.bool))


@pytest.mark.parametrize("shape", [(2**31 + 1, 2**32), (239075442, 77158673929), (2**40, 2**30)])
def test_var_and_std_over_huge_fibers(shape):
    """The namespace's var and std count the unstored positions in float64, so they work over
    a fiber of 2**63 positions or more.
    """
    L = shape[0] * shape[1]
    x = xp.full(shape, 2.0)
    x[0, 0] = 5.0
    expected = 9.0 * (L - 1) / L**2  # (L - 1 values of 2.0 and one of 5.0)
    assert float(xp.var(x)) == pytest.approx(expected, rel=1e-9)
    assert float(xp.std(x)) == pytest.approx(expected**0.5, rel=1e-9)
    assert float(xp.var(x, correction=1)) == pytest.approx(9.0 / L, rel=1e-9)
    assert float(xp.var(x, correction=0)) == pytest.approx(expected, rel=1e-9)  # (an int)
    y = xp.full((3, *shape), 2, dtype=xp.int64)
    y[0, 0, 0] = 5
    v = xp.var(y, axis=(1, 2)).todense()
    assert v[0] == pytest.approx(expected, rel=1e-9)
    assert v[1] == v[2] == 0.0
    assert float(xp.mean(x)) == pytest.approx(2.0 + 3.0 / L, rel=1e-12)

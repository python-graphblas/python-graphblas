"""graphblas.array_api set functions against numpy."""

from types import SimpleNamespace

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp

from .test_array_api_elementwise import _strict_signed_zeros  # noqa: F401 (autouse)
from .test_array_api_elementwise import assert_array_equal, check_invariants, inputs, make

DTYPES = [np.dtype(t) for t in ("bool", "int8", "uint32", "int64", "float32", "float64")]


def _check_values(out, expected):
    """Unique values: as numpy's, except that which of 0.0 and -0.0 stands for both is free."""
    check_invariants(out)
    got = out.todense()
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)  # (0.0 == -0.0, NaN matches NaN)


@pytest.mark.parametrize("shape", [(9,), (3, 4), (2, 3, 2), ()])
def test_unique(shape):
    for dtype in DTYPES:
        for dense, x in inputs(dtype, shape, seed=13):
            # (numpy's unique_* need not sort; np.unique does, as graphblas.array_api does)
            values, indices, inverse, counts = np.unique(
                dense, return_index=True, return_inverse=True, return_counts=True, equal_nan=False
            )
            e = SimpleNamespace(
                values=values,
                indices=indices,
                inverse_indices=inverse.reshape(dense.shape),
                counts=counts,
            )
            r = xp.unique_all(x)
            _check_values(r.values, e.values)
            assert_array_equal(r.counts, e.counts.astype(np.int64))
            # the first index of each value (of either zero, for 0.0 and -0.0)
            assert_array_equal(r.indices, e.indices.astype(np.int64))
            assert_array_equal(r.inverse_indices, e.inverse_indices.astype(np.int64))
            _check_values(xp.unique_values(x), e.values)
            c = xp.unique_counts(x)
            _check_values(c.values, e.values)
            assert_array_equal(c.counts, e.counts.astype(np.int64))
            i = xp.unique_inverse(x)
            _check_values(i.values, e.values)
            assert_array_equal(i.inverse_indices, e.inverse_indices.astype(np.int64))


def test_unique_is_sparse():
    x = xp.zeros((10**6, 10**6))
    x[3, [1, 5]] = np.array([2.0, -1.0])
    r = xp.unique_all(x)
    assert_array_equal(r.values, np.array([-1.0, 0.0, 2.0]))
    assert_array_equal(r.indices, np.array([3 * 10**6 + 5, 0, 3 * 10**6 + 1]))
    assert_array_equal(r.counts, np.array([1, 10**12 - 2, 1]))
    assert r.inverse_indices.nnz == 2
    assert r.inverse_indices.fill_value == 1


def test_unique_nan_fill():
    a = np.array([np.nan, 1.0, np.nan, 2.0])
    x = make(a, np.nan)
    _check_values(xp.unique_values(x), np.unique(a, equal_nan=False))
    assert_array_equal(xp.unique_counts(x).counts, np.ones(4, np.int64))
    with gb.config.set(array_api_densify_limit=2), pytest.raises(xp.DensifyError):
        xp.unique_values(x)


@pytest.mark.parametrize("invert", [False, True])
def test_isin(invert):
    for dtype in DTYPES:
        for (d1, x1), (d2, x2) in zip(
            inputs(dtype, (3, 4), seed=14), inputs(dtype, (5,), seed=15), strict=False
        ):
            expected = np.isin(d1, d2, invert=invert)
            assert_array_equal(xp.isin(x1, x2, invert=invert), expected)
    x = make(np.array([1, 0, 5, 2]), 0)
    assert_array_equal(xp.isin(x, 2), np.isin([1, 0, 5, 2], 2))
    assert_array_equal(xp.isin(3, x), np.array(False))
    assert_array_equal(xp.isin(xp.asarray(1.0), x), np.array(True))
    with pytest.raises(TypeError, match="Array"):
        xp.unique_values(np.zeros(3))

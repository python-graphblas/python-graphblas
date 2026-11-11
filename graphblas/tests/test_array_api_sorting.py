"""graphblas.array_api sort and argsort against numpy (stable sorts)."""

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp

from .test_array_api_elementwise import _strict_signed_zeros  # noqa: F401 (autouse)
from .test_array_api_elementwise import assert_array_equal, inputs, make

DTYPES = [np.dtype(t) for t in ("bool", "int8", "uint16", "int64", "float32", "float64")]


def _np_sort(a, axis, descending):
    if descending:
        return np.flip(np.sort(np.flip(a, axis=axis), axis=axis, kind="stable"), axis=axis)
    return np.sort(a, axis=axis, kind="stable")


def _np_argsort(a, axis, descending):
    if descending:
        n = a.shape[axis]
        flipped = np.argsort(np.flip(a, axis=axis), axis=axis, kind="stable")
        return n - 1 - np.flip(flipped, axis=axis)
    return np.argsort(a, axis=axis, kind="stable")


@pytest.mark.parametrize("descending", [False, True])
def test_sort(descending):
    for dtype in DTYPES:
        for shape, axes in [((9,), [0, -1]), ((3, 6), [0, 1]), ((2, 3, 4), [0, 1, 2])]:
            for dense, x in inputs(dtype, shape, seed=12):
                for axis in axes:
                    out = xp.sort(x, axis=axis, descending=descending)
                    assert_array_equal(out, _np_sort(dense, axis, descending))
                    assert out.fill_value == x.fill_value or np.isnan(out.fill_value)
                    out = xp.argsort(x, axis=axis, descending=descending)
                    assert_array_equal(out, _np_argsort(dense, axis, descending).astype(np.int64))


def test_sort_keeps_sparsity():
    x = xp.zeros((1000, 1000))
    x[5, [3, 7, 9]] = np.array([2.0, -1.0, 4.0])
    x[8, 0] = -0.0
    out = xp.sort(x, axis=1)
    assert out.nnz == 4
    row = out[5].todense()
    assert (row[[0, 998, 999]] == [-1.0, 2.0, 4.0]).all()
    assert np.signbit(out[8].todense()[0])  # -0.0 keeps its place among the zeros: first
    with gb.config.set(array_api_densify_limit=10**5), pytest.raises(xp.DensifyError):
        xp.argsort(x)


def test_sort_signed_zeros_and_nan_stable():
    a = np.array([0.0, -0.0, np.nan, 1.0, -0.0, 0.0, np.nan, -1.0])
    for fill in [0.0, -0.0, np.nan, 1.0]:
        for x in [make(a, fill), make(a, fill, np.random.default_rng(1), explicit=0.6)]:
            for descending in [False, True]:
                assert_array_equal(xp.sort(x, descending=descending), _np_sort(a, 0, descending))


def test_sort_errors():
    with pytest.raises(ValueError, match="dimension"):
        xp.sort(xp.asarray(1.0))
    with pytest.raises(TypeError, match="real"):
        xp.argsort(xp.zeros(2, dtype=xp.complex64))
    with pytest.raises(TypeError, match="Array"):
        xp.sort(np.zeros(3))

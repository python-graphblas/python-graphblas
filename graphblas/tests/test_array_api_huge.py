"""graphblas.array_api on arrays of more positions than numpy's largest array size on 32-bit
platforms such as WebAssembly (2**31 - 1), where index math over a whole shape cannot go
through numpy.
"""

import numpy as np
import pytest

import graphblas.array_api as xp
from graphblas.array_api import _utils

from .test_array_api_elementwise import assert_array_equal


def test_index_math_past_2_31_positions():
    """Index, assign, sort and accumulate along an axis of arrays of 3 * 10**10 and 2**42
    positions. Under Pyodide each raised ValueError or OverflowError from numpy.
    """
    y = xp.zeros((10**5, 10**5, 3))
    y[1, 2, 0] = 1.0
    y[7, 8, 2] = 3.0
    y[7, 8, 1] = 2.0
    pick = ([1, 7], [2, 8], slice(None))  # (two index arrays: one ravel over two axes)
    assert_array_equal(y[pick], np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]))
    assert_array_equal(xp.sort(y, axis=2)[pick], np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]]))
    s = xp.cumulative_sum(y, axis=2)
    assert s.nnz == 5
    assert_array_equal(s[pick], np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 5.0]]))
    y[[1, 7], [2, 8], [0, 2]] = np.array([4.0, 6.0])  # (scatter)
    assert_array_equal(y[pick], np.array([[4.0, 0.0, 0.0], [0.0, 2.0, 6.0]]))

    z = xp.zeros((2**40, 4))
    h = 2**39
    z[h : h + 3, 1:3] = 7.0  # (a region, by slices along the long axis)
    assert z.nnz == 6
    assert z[h:, 1:3].shape == (h, 2)
    assert_array_equal(z[h : h + 3, 1:3], np.full((3, 2), 7.0))
    assert_array_equal(xp.multiply(z, 2.0)[h + 1 : h + 2, :], np.array([[0.0, 14.0, 14.0, 0.0]]))


def test_index_helpers_in_int64_give_numpys_results(monkeypatch):
    """Where numpy's sizes cannot hold the shape (forced here), ``ravel`` and ``unravel`` give
    numpy's results; ``slice_length`` is ``len(range(...))`` of every slice.
    """
    rng = np.random.default_rng(0)
    cases = []
    for shape in [(3, 4, 5), (7,), (2, 1, 3), (6, 9)]:
        coords = tuple(rng.integers(0, n, 20) for n in shape)
        flat = np.ravel_multi_index(coords, shape)
        cases.append((shape, coords, flat, np.unravel_index(flat, shape)))
    monkeypatch.setattr(_utils, "_MAX_INTP", 0)
    for shape, coords, flat, unflat in cases:
        np.testing.assert_array_equal(_utils.ravel(coords, shape), flat)
        for got, want in zip(_utils.unravel(flat, shape), unflat, strict=True):
            np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="invalid dims|Maximum allowed dimension"):  # (2**80)
        _utils.ravel(([0], [0]), (2**40, 2**40))
    for n in range(12):
        values = [None, *range(-n - 2, n + 3)]
        for k in [slice(a, b, c) for a in values for b in values for c in [None, -3, -1, 1, 2]]:
            assert _utils.slice_length(k, n) == len(range(*k.indices(n))), (k, n)
    assert _utils.slice_length(slice(None), 2**60) == 2**60

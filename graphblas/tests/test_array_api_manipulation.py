"""graphblas.array_api manipulation functions against numpy."""

import numpy as np
import pytest

import graphblas.array_api as xp

from .test_array_api_elementwise import assert_array_equal, inputs, make


@pytest.mark.parametrize(
    ("shapes", "axis"),
    [
        ([(2, 3), (4, 3)], 0),
        ([(2, 3), (2, 1), (2, 4)], 1),
        ([(2, 3), (2, 3)], -1),
        ([(2, 3, 2), (2, 1, 2)], 1),
        ([(3,), (0,), (2,)], 0),
        ([(2, 3), (3,)], None),
    ],
)
def test_concat(shapes, axis):
    for dtype in [np.dtype("float64"), np.dtype("int16"), np.dtype("bool")]:
        parts = [list(inputs(dtype, shape, seed=i)) for i, shape in enumerate(shapes)]
        for i in range(len(parts[0])):
            dense = [p[i % len(p)][0] for p in parts]
            arrays = [p[i % len(p)][1] for p in parts]
            assert_array_equal(xp.concat(arrays, axis=axis), np.concatenate(dense, axis=axis))
    mixed = [make(np.array([1, 0, 2], np.int8), 0), make(np.array([1.5, 2.5]), 2.5)]
    expected = np.concatenate([np.array([1, 0, 2], np.int8), np.array([1.5, 2.5])])
    assert_array_equal(xp.concat(mixed), expected)


def test_concat_keeps_the_most_common_fill():
    big = xp.full((10,), 7.0)
    small = make(np.array([0.0, 1.0]), 0.0)
    out = xp.concat([small, big])
    assert out.fill_value == 7.0
    assert out.nnz == 2
    assert_array_equal(out, np.concatenate([[0.0, 1.0], np.full(10, 7.0)]))


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_stack(axis):
    for dtype in [np.dtype("float32"), np.dtype("uint8")]:
        a = list(inputs(dtype, (2, 3), seed=1))
        b = list(inputs(dtype, (2, 3), seed=2))
        for (da, xa), (db, xb) in zip(a, b, strict=False):
            assert_array_equal(xp.stack([xa, xb, xa], axis=axis), np.stack([da, db, da], axis=axis))
    assert_array_equal(xp.stack([xp.asarray(1), xp.asarray(2)]), np.array([1, 2]))
    with pytest.raises(ValueError, match="one shape"):
        xp.stack([xp.zeros(2), xp.zeros(3)])
    with pytest.raises(ValueError, match="at least one"):
        xp.stack([])

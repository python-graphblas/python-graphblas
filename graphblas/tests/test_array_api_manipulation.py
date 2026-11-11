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


DTYPES3 = [np.dtype("float64"), np.dtype("int8"), np.dtype("bool")]


def _cases(shape, seed=0):
    for dtype in DTYPES3:
        yield from inputs(dtype, shape, seed=seed)


def test_broadcast():
    assert xp.broadcast_shapes((2, 1), (3,), ()) == (2, 3)
    assert xp.broadcast_shapes() == ()
    with pytest.raises(ValueError, match="broadcast"):
        xp.broadcast_shapes((2,), (3,))
    for dense, x in _cases((3, 1)):
        for shape in [(3, 1), (3, 4), (2, 3, 4), (0, 3, 1)]:
            assert_array_equal(xp.broadcast_to(x, shape), np.broadcast_to(dense, shape))
        with pytest.raises(ValueError, match="broadcast"):
            xp.broadcast_to(x, (3,))
        y = make(np.array([1, 0, 2, 0], dtype=dense.dtype), np.zeros((), dense.dtype)[()])
        out = xp.broadcast_arrays(x, y, xp.asarray(1))
        expected = np.broadcast_arrays(dense, y.todense(), np.array(1))
        assert isinstance(out, tuple)
        for o, e in zip(out, expected, strict=True):
            assert_array_equal(o, e)


@pytest.mark.parametrize("axis", [0, 1, -1, 2])
def test_unstack(axis):
    for dense, x in _cases((2, 3, 4)):
        out = xp.unstack(x, axis=axis)
        assert isinstance(out, tuple)
        expected = [np.take(dense, i, axis=axis) for i in range(dense.shape[axis])]
        assert len(out) == len(expected)
        for o, e in zip(out, expected, strict=True):
            assert_array_equal(o, e)
    with pytest.raises(ValueError, match="dimension"):
        xp.unstack(xp.asarray(1.0))


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2), ()])
def test_flip(axis):
    for dense, x in _cases((2, 3, 4)):
        assert_array_equal(xp.flip(x, axis=axis), np.flip(dense, axis=axis))
    assert_array_equal(xp.flip(xp.asarray(3)), np.array(3))


@pytest.mark.parametrize(
    ("source", "destination"), [(0, 2), (-1, 0), ((0, 1), (2, 0)), ((), ()), ((0, 1, 2), (2, 1, 0))]
)
def test_moveaxis(source, destination):
    for dense, x in _cases((2, 3, 4)):
        assert_array_equal(
            xp.moveaxis(x, source, destination), np.moveaxis(dense, source, destination)
        )
    with pytest.raises(ValueError, match="as many"):
        xp.moveaxis(xp.zeros((2, 3)), (0, 1), 0)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_repeat(axis):
    for dense, x in _cases((2, 3)):
        n = dense.size if axis is None else dense.shape[axis]
        for repeats in [0, 1, 3, np.arange(n) % 3, np.array([2])]:
            if isinstance(repeats, np.ndarray):
                r = xp.asarray(repeats)
                expected = np.repeat(dense, np.broadcast_to(repeats, (n,)), axis=axis)
            else:
                r = repeats
                expected = np.repeat(dense, repeats, axis=axis)
            assert_array_equal(xp.repeat(x, r, axis=axis), expected)
    with pytest.raises(ValueError, match="nonnegative"):
        xp.repeat(xp.zeros(3), -1)
    with pytest.raises(TypeError, match="integer"):
        xp.repeat(xp.zeros(3), xp.asarray([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    ("shift", "axis"),
    [(1, None), (-7, None), (2, 0), (-1, 1), ((1, 2), (0, 1)), (3, (0, 1)), (1, (1, 1))],
)
def test_roll(shift, axis):
    for dense, x in _cases((3, 4)):
        assert_array_equal(xp.roll(x, shift, axis=axis), np.roll(dense, shift, axis=axis))
    with pytest.raises(ValueError, match="one shift"):
        xp.roll(xp.zeros((2, 2)), (1, 2, 3), axis=(0, 1))


@pytest.mark.parametrize("reps", [(2,), (2, 3), (1, 1), (0, 2), (2, 1, 3), (3,)])
def test_tile(reps):
    for dense, x in _cases((2, 3)):
        assert_array_equal(xp.tile(x, reps), np.tile(dense, reps))
    assert_array_equal(xp.tile(xp.asarray(2.0), (3,)), np.tile(np.array(2.0), 3))
    with pytest.raises(ValueError, match="nonnegative"):
        xp.tile(xp.zeros(2), (-1,))

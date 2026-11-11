"""graphblas.array_api indexing (get and set), take and take_along_axis against numpy."""

import itertools

import numpy as np
import pytest

import graphblas.array_api as xp

from .test_array_api_elementwise import assert_array_equal, inputs, make

rng = np.random.default_rng(0)
A = np.arange(60).reshape(3, 4, 5)
MASK = (A % 7) < 3


def _keys():
    return [
        [0, 2],
        np.array([2, -1, 0, 0]),
        ([1, 0], [3, 3]),
        ([[0], [2]], [1, 0, 1], [4, 4, 0]),
        (2, [1, 3]),
        ([1], 2, [0, 4]),
        (0, Ellipsis, [4, 1]),
        ([0, 1], slice(None), [1, 3]),
        (slice(None), [2, 0], [1, 1]),
        (slice(1, None), [3, 0, 3]),
        (slice(None, None, -1), slice(1, 3), np.array([[0, 1], [2, 3]])),
        (None, [1], slice(None), 2),
        ([2, 1], None, 0),
        (Ellipsis, [3, 0]),
        MASK,
        MASK[:, :, 0],
        (slice(None), MASK[0]),
        (MASK[:, 0, :].any(axis=1),),
        (MASK[:, 0, 0], 1),
        True,
        False,
        (np.array([], dtype=np.int64),),
        (np.zeros((3, 4, 5), bool),),
        (Ellipsis, None),
    ]


@pytest.mark.parametrize("key", range(len(_keys())))
def test_getitem_advanced(key):
    key = _keys()[key]
    for dtype in [np.dtype("float64"), np.dtype("int16"), np.dtype("bool"), np.dtype("complex64")]:
        for dense, x in inputs(dtype, (3, 4, 5), seed=3):
            assert_array_equal(x[key], dense[key])
            # index arrays given as Arrays
            if (
                isinstance(key, np.ndarray)
                or isinstance(key, tuple)
                and any(isinstance(k, np.ndarray) for k in key)
            ):
                k2 = key if not isinstance(key, np.ndarray) else (key,)
                k2 = tuple(xp.asarray(k) if isinstance(k, np.ndarray) else k for k in k2)
                assert_array_equal(x[k2], dense[key])


def test_boolean_mask_from_comparison():
    for fill in [0.0, 1.5]:
        a = np.where(rng.random((4, 5)) < 0.5, fill, np.round(rng.normal(size=(4, 5)), 1))
        x = make(a, fill)
        for mask in [x > 0, x < 0.5, x == fill, xp.isnan(x)]:
            assert_array_equal(x[mask], a[mask.todense()])
        assert_array_equal(x[x > 100], a[a > 100])
    big = xp.zeros((10**5, 10**5))
    mask = big != 0
    assert big[mask].shape == (0,)


def test_getitem_errors():
    x = make(A.astype(float), 0.0)
    with pytest.raises(IndexError):
        x[[0, 3]]
    with pytest.raises(IndexError):
        x[[0], [-5]]
    with pytest.raises(IndexError):
        x[MASK[:2]]
    with pytest.raises(IndexError):
        xp.zeros((0, 0))[np.zeros((0, 1), bool)]
    assert xp.zeros((2, 3))[np.zeros((0, 3), bool)].shape == (0,)
    with pytest.raises(IndexError):
        x[np.array([0.5])]
    with pytest.raises(IndexError):
        x[xp.asarray([0.5])]
    with pytest.raises(IndexError):
        x["a"]
    with pytest.raises(IndexError):
        x[..., 0, ...]
    with pytest.raises(IndexError):
        x[MASK, 0]
    with pytest.raises(IndexError):
        x[xp.asarray(1.5)]


SET_CASES = [
    ((0, 1), 9.0),
    ((0, [1, 2]), 9.0),
    (([0, 1], [0, 2]), 7.0),
    (([0, 1, 0], [0, 2, 0], [1, 1, 1]), np.array([1.0, 2.0, 3.0])),
    ((slice(None), [0, 0], 1), np.array([5.0, 6.0])),
    (([1], slice(1, 3)), 0.0),
    ((Ellipsis, 1), -1.0),
    ((1,), np.arange(20.0).reshape(4, 5)),
    ((slice(None), slice(1, 3)), np.array([1.0, 0.0, 2.0, 0.0, 3.0])),
    ((slice(None, None, -1), 2), 0.0),
    (([2, 0, 2],), np.arange(20.0).reshape(4, 5) * 0),
    (MASK, 0.0),
    (MASK, 4.0),
    (MASK[:, :, 0], np.array([1.0, 0.0, 2.0, 0.0, 3.0])),
    (([0, 2], [1, 3]), np.array([[1.0], [2.0]])),
    ((), 3.5),
    ((Ellipsis,), np.zeros(5)),
    ((None, 0), 1.0),
]


@pytest.mark.parametrize("case", range(len(SET_CASES)))
def test_setitem(case):
    key, value = SET_CASES[case]
    for fill in [0.0, 2.5, -0.0]:
        dense = np.where(A % 3 == 0, fill, A.astype(float))
        x = make(dense, fill)
        expected = dense.copy()
        expected[key] = value
        for v in [value, make(np.asarray(value), 0.0), make(np.asarray(value), fill)]:
            y = make(dense, fill)
            y[key] = v
            assert_array_equal(y, expected)
            assert y.fill_value == x.fill_value
    y = make(dense, fill)
    y[MASK.tolist()] = 1.0  # (a nested list is an index array too)
    expected = dense.copy()
    expected[MASK] = 1.0
    assert_array_equal(y, expected)


def test_setitem_dtypes_and_errors():
    x = make(np.array([1, 0, 3], np.int8), 0)
    x[0] = 1.9
    assert_array_equal(x, np.array([1, 0, 3], np.int8))
    with pytest.raises(OverflowError):
        x[1] = 300
    with pytest.raises(TypeError, match="complex"):
        x[1] = 1j
    with pytest.raises(ValueError, match="cannot assign"):
        x[:2] = xp.asarray([1, 2, 3])
    with pytest.raises(ValueError, match="cannot assign"):
        x[[0, 1]] = xp.asarray([1, 2, 3])
    y = make(np.zeros((2, 2)), 0.0)
    with pytest.raises(ValueError, match="cannot assign"):
        y[[0, 1], [0, 1]] = np.ones(3)
    z = xp.asarray(1.5)
    z[False] = 3.0
    assert_array_equal(z, np.array(1.5))
    z[True] = 2.5
    assert_array_equal(z, np.array(2.5))
    q = make(np.array([1.0, 2.0]), 0.0)
    q[False] = 3.0
    assert_array_equal(q, np.array([1.0, 2.0]))
    z[()] = 2.0
    assert_array_equal(z, np.array(2.0))
    with pytest.raises(ValueError, match="0-d"):
        z[...] = xp.asarray([1.0])
    # the cached summary of the values does not survive an assignment
    w = make(np.array([1.0, 0.0, 2.0]), 0.0)
    assert_array_equal(w * make(np.array([0.0, 3.0, 0.0]), 0.0), np.zeros(3))
    w[0] = -1.0
    assert_array_equal(w * make(np.array([0.0, 3.0, 0.0]), 0.0), np.array([-0.0, 0.0, 0.0]))
    # an assignment does not change another array that shares its data
    a = make(np.array([1.0, 0.0, 2.0]), 0.0)
    b = xp.asarray(a, copy=True)
    c = a[:]
    a[[0, 2]] = 5.0
    assert_array_equal(b, np.array([1.0, 0.0, 2.0]))
    assert_array_equal(c, np.array([1.0, 0.0, 2.0]))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_take(axis):
    for dtype in [np.dtype("float64"), np.dtype("uint8"), np.dtype("bool")]:
        shape = (7,) if axis is None else (3, 4)
        for dense, x in inputs(dtype, shape, seed=2):
            n = dense.shape[0 if axis is None else axis]
            for idx in [np.array([0, n - 1, 1, 1]), np.array([-1, 0]), np.array([], np.int64)]:
                expected = np.take(dense, idx, axis=axis)
                assert_array_equal(xp.take(x, xp.asarray(idx), axis=axis), expected)
                assert_array_equal(xp.take(x, idx, axis=axis), expected)
            with pytest.raises(IndexError):
                xp.take(x, xp.asarray([n]), axis=axis)
    with pytest.raises(ValueError, match="axis"):
        xp.take(xp.zeros((2, 2)), xp.asarray([0]))
    with pytest.raises(TypeError, match="integer"):
        xp.take(xp.zeros(2), xp.asarray([0.0]))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_take_along_axis(axis):
    for dense, x in inputs(np.dtype("float32"), (3, 4), seed=5):
        n = dense.shape[axis]
        idx = np.argsort(dense, axis=axis)
        assert_array_equal(
            xp.take_along_axis(x, xp.asarray(idx), axis=axis),
            np.take_along_axis(dense, idx, axis=axis),
        )
        shape = [3, 4]
        shape[axis] = 2
        idx = rng.integers(-n, n, shape)
        assert_array_equal(
            xp.take_along_axis(x, xp.asarray(idx), axis=axis),
            np.take_along_axis(dense, idx, axis=axis),
        )
        bshape = [1, 1]
        bshape[axis] = 5
        idx = rng.integers(0, n, bshape)
        assert_array_equal(
            xp.take_along_axis(x, idx, axis=axis), np.take_along_axis(dense, idx, axis=axis)
        )
    with pytest.raises(ValueError, match="dimensions"):
        xp.take_along_axis(xp.zeros((2, 2)), xp.asarray([0]))


def _index_patterns():
    """Keys mixing ints, slices, integer arrays, ``...`` and ``None`` with at least one
    integer array, as numpy accepts them for a 4-d array.
    """
    parts = [slice(None), 1, np.array([0, 2]), np.array([[1], [0]]), Ellipsis, None, slice(0, 2)]
    for n in range(1, 5):
        for key in itertools.product(parts, repeat=n):
            if not any(isinstance(k, np.ndarray) for k in key):
                continue
            if sum(k is Ellipsis for k in key) > 1:
                continue
            yield key


def test_mixed_keys_order_axes_as_numpy():
    # (shim finding B1) the advanced axes go first when the advanced indices are apart, a
    # "..." that covers no axis included; in place when they are next to each other
    count = 0
    for a, key in itertools.chain(
        ((np.arange(36.0).reshape(3, 4, 3) + 1, k) for k in _index_patterns()),
        ((np.arange(108.0).reshape(3, 4, 3, 3) + 1, k) for k in _index_patterns()),
    ):
        x = make(a, 0.0)
        try:
            expected = a[key]
        except IndexError:
            continue
        count += 1
        assert_array_equal(x[key], expected)
        if any(k is None for k in key):
            continue  # (a value for a key with new axes is not read with them yet)
        b = a.copy()
        value = np.arange(expected.size, dtype=np.float64).reshape(expected.shape) - 50
        b[key] = value
        y = make(a, 0.0)
        y[key] = make(value, 0.0) if value.ndim else xp.asarray(value)
        assert_array_equal(y, b)
    assert count > 2000
    x = make(np.arange(24.0).reshape(2, 3, 4), 0.0)
    assert_array_equal(x[:, 1, ..., [0, 2]], np.array([[4.0, 16.0], [6.0, 18.0]]))

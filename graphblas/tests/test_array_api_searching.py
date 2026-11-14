"""graphblas.array_api searching functions against numpy."""

import numpy as np
import pytest

import graphblas.array_api as xp

from .test_array_api_elementwise import DTYPES, assert_array_equal, inputs, make, np_call

WHERE_SHAPES = [
    ((3, 4), (3, 4), (3, 4)),
    ((3, 4), (4,), (3, 1)),
    ((3, 1), (1, 4), ()),
    ((), (2, 3), (2, 3)),
    ((2, 1, 4), (3, 1), (4,)),
    ((), (), ()),
]


@pytest.mark.parametrize("shapes", WHERE_SHAPES, ids=str)
def test_where(shapes):
    rng = np.random.default_rng(1)
    for dt1, dt2 in [
        ("float64", "float64"),
        ("int8", "uint8"),
        ("float32", "int16"),
        ("bool", "complex64"),
    ]:
        dt1, dt2 = np.dtype(dt1), np.dtype(dt2)
        for (c, xc), (a, xa), (b, xb) in zip(
            inputs(np.dtype(bool), shapes[0], seed=2),
            inputs(dt1, shapes[1], seed=3),
            inputs(dt2, shapes[2], seed=4),
            strict=False,
        ):
            assert_array_equal(xp.where(xc, xa, xb), np.where(c, a, b))
            for s in [0, 1.5, True]:
                assert_array_equal(xp.where(xc, xa, s), np.where(c, a, s))
                assert_array_equal(xp.where(xc, s, xb), np.where(c, s, b))
    # a condition that is not bool counts by truth value
    a = np.where(rng.random((3, 4)) < 0.5, 0.0, rng.normal(size=(3, 4)))
    x = make(a, 0.0)
    assert_array_equal(xp.where(x, x, -1.0), np.where(a, a, -1.0))
    assert_array_equal(xp.where(x > 0, x, 0.0), np.where(a > 0, a, 0.0))
    with pytest.raises(ValueError, match="broadcast"):
        xp.where(make(np.ones(3, bool), False), x, x)


def test_where_stores_the_condition_and_the_selected_operand():
    # Where the condition stores nothing, its fill selects one operand: the result is that
    # operand with the condition's stored positions written over, and the other operand is
    # read only there (no replication of a broadcast operand)
    rng = np.random.default_rng(7)
    n = 60
    c = rng.random((n, n)) < 0.05
    labels = np.arange(n)
    big = np.iinfo(np.int64).max
    G = make(c, False, rng)  # (with explicit False entries)
    W = xp.where(G, xp.arange(n), big)
    assert_array_equal(W, np.where(c, labels, big))
    assert W.nnz == c.sum()  # (G's explicit False entries select big, the fill: not stored)
    assert W.fill_value == big
    # where(x > t, x, 0): x's entries above t only, read there only
    a = np.where(rng.random((n, n)) < 0.3, rng.normal(size=(n, n)), 0.0)
    X = make(a, 0.0, rng)
    R = xp.where(xp.greater(X, 0.5), X, 0.0)
    assert_array_equal(R, np.where(a > 0.5, a, 0.0))
    assert R.nnz == (a > 0.5).sum()
    R = xp.where(xp.less_equal(X, 0.5), 0.0, X)  # (a condition with fill True)
    assert_array_equal(R, np.where(a <= 0.5, 0.0, a))
    assert R.nnz == (a > 0.5).sum()
    # other layouts of the condition, and a fill of True (the base is then x1)
    for cond, cn in [(xp.matrix_transpose(G), c.T), (xp.logical_not(G), ~c)]:
        a = rng.normal(size=(n, n))
        a[rng.random((n, n)) < 0.5] = 0
        a[0, 0] = np.nan
        for x1, x2, e in [
            (make(a, 0.0), xp.arange(n) * 1.0, np.where(cn, a, labels * 1.0)),
            (xp.arange(n) * 1.0, make(a, 0.0, rng), np.where(cn, labels * 1.0, a)),
            (xp.arange(n)[:, None] * 1.0, -1.0, np.where(cn, labels[:, None] * 1.0, -1.0)),
        ]:
            assert_array_equal(xp.where(cond, x1, x2), e)
    # A condition that is not bool selects by truth value (NaN is true), with a broadcast
    # operand read at its entries, or over a base with entries (where(frontier, k, level));
    # a complex condition goes through numpy
    f = np.where(c, np.where(rng.random((n, n)) < 0.3, np.nan, 2.5), 0.0)
    f[0, 0] = -0.0
    level = make(a, 0.0)
    for cond, fill in [(f, 0.0), (np.where(c, 0, 3).astype(np.int8), 3), (f + 1j * f, 0j)]:
        for x1, x2 in [
            (xp.arange(n), big),
            (big, xp.arange(n)[:, None]),
            (labels[0], -1),
            (7, level),
            (xp.arange(n), level),
        ]:
            e = np.where(cond, np.asarray(x1), np.asarray(x2))
            assert_array_equal(xp.where(make(cond, fill), x1, x2), e)
    # The selected operand broadcasts over stored entries, or the condition broadcasts: the
    # union (the result really holds the operand's entries along the broadcast axis)
    v = np.where(rng.random(n) < 0.5, rng.normal(size=n), 0.0)
    assert_array_equal(xp.where(G, -1.0, make(v, 0.0)), np.where(c, -1.0, v))
    assert_array_equal(xp.where(make(c[0], False), make(a, 0.0), 2.0), np.where(c[0], a, 2.0))
    # An empty condition: the selected operand
    E = xp.zeros((n, n), dtype=xp.bool)
    assert_array_equal(xp.where(E, 1.0, make(a, 0.0)), a)
    assert xp.where(E, 1.0, make(a, 0.0)).nnz == make(a, 0.0).nnz


def test_where_dtypes():
    for dt in DTYPES:
        for dense, x in inputs(dt, (5,), seed=6):
            c = dense.astype(bool) if dt.kind != "c" else dense.real.astype(bool)
            assert_array_equal(xp.where(xp.asarray(c), x, x), np.where(c, dense, dense))
            assert_array_equal(
                np_call(xp.where, xp.asarray(c), x, 2), np_call(np.where, c, dense, 2)
            )


SEARCH_DTYPES = [np.dtype(t) for t in ("bool", "int8", "uint16", "int64", "float32", "float64")]


@pytest.mark.parametrize("name", ["argmax", "argmin"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmax_argmin(name, keepdims):
    func = getattr(np, name)
    for dtype in SEARCH_DTYPES:
        for shape, axes in [((7,), [None, 0, -1]), ((3, 5), [None, 0, 1]), ((2, 3, 4), [0, 1, 2])]:
            for dense, x in inputs(dtype, shape, seed=8):
                for axis in axes:
                    expected = func(dense, axis=axis, keepdims=keepdims)
                    out = getattr(xp, name)(x, axis=axis, keepdims=keepdims)
                    assert_array_equal(out, np.asarray(expected, dtype=np.int64))
    with pytest.raises(ValueError, match="empty"):
        getattr(xp, name)(xp.zeros((0, 3)), axis=0)
    with pytest.raises(TypeError, match="real"):
        getattr(xp, name)(xp.zeros(3, dtype=xp.complex64))
    assert_array_equal(getattr(xp, name)(xp.zeros((2, 0)), axis=0), np.zeros(0, np.int64))


def test_argmax_ties_with_the_fill():
    # the fill ties with a stored value: the first of either position
    for a, fill in [
        ([0.0, 3.0, 0.0, 3.0], 3.0),
        ([3.0, 3.0, 1.0, 0.0], 3.0),
        ([-0.0, 0.0, -1.0], 0.0),
        ([np.nan, 1.0, 1.0, np.nan], np.nan),
        ([1.0, 1.0, np.nan, 1.0], 1.0),
    ]:
        a = np.array(a)
        for x in [make(a, fill), make(a, fill, np.random.default_rng(0), explicit=0.7)]:
            assert int(xp.argmax(x)) == np.argmax(a)
            assert int(xp.argmin(x)) == np.argmin(a)


def test_argmax_argmin_random():
    # numpy's rules (first occurrence, NaN the extreme, the fill at unstored positions)
    # over fills, densities, shapes, axes and stored layouts
    rng = np.random.default_rng(12)
    pools = {
        "float64": [0.0, -0.0, 1.0, -1.0, 2.5, np.nan, np.inf, -np.inf],
        "float32": [0.0, -0.0, 1.0, -1.0, np.nan, np.inf],
        "int8": [-3, -1, 0, 1, 2],
        "uint16": [0, 1, 2, 3],
        "bool": [False, True],
    }
    shapes = [(1,), (6,), (1, 1), (3, 4), (6, 1), (1, 7), (2, 3, 4), (3, 1, 2, 2)]
    for _ in range(250):
        dt = np.dtype(list(pools)[rng.integers(len(pools))])
        pool = np.array(pools[dt.name], dt)
        shape = shapes[rng.integers(len(shapes))]
        fill = pool[rng.integers(len(pool))]
        values = rng.choice(pool[: rng.integers(1, len(pool) + 1)], shape)
        dense = np.where(rng.random(shape) < rng.choice([0.0, 0.2, 0.6, 1.0]), values, fill)
        x = make(dense, fill, rng, explicit=0.2)
        if dense.ndim > 1 and rng.random() < 0.5:
            # (a stored layout other than the default)
            perm = tuple(int(a) for a in rng.permutation(dense.ndim))
            inverse = tuple(int(a) for a in np.argsort(perm))
            x = xp.permute_dims(xp.asarray(xp.permute_dims(x, perm), copy=True), inverse)
        axis = int(rng.integers(-dense.ndim, dense.ndim)) if rng.random() < 0.8 else None
        keepdims = rng.random() < 0.3
        for name in ["argmax", "argmin"]:
            expected = getattr(np, name)(dense, axis=axis, keepdims=keepdims)
            out = getattr(xp, name)(x, axis=axis, keepdims=keepdims)
            assert_array_equal(out, np.asarray(expected, dtype=np.int64))


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), ()])
def test_count_nonzero(axis):
    for dtype in [*SEARCH_DTYPES, np.dtype("complex64")]:
        for dense, x in inputs(dtype, (3, 4), seed=9):
            for keepdims in [False, True]:
                expected = np.count_nonzero(dense, axis=axis, keepdims=keepdims)
                out = xp.count_nonzero(x, axis=axis, keepdims=keepdims)
                assert_array_equal(out, np.asarray(expected, dtype=np.int64))


def test_nonzero():
    for dtype in [*SEARCH_DTYPES, np.dtype("complex128")]:
        for shape in [(7,), (3, 4), (2, 3, 2)]:
            for dense, x in inputs(dtype, shape, seed=10):
                out = xp.nonzero(x)
                expected = np.nonzero(dense)
                assert isinstance(out, tuple)
                assert len(out) == len(expected)
                for o, e in zip(out, expected, strict=True):
                    assert_array_equal(o, e.astype(np.int64))
    with pytest.raises(ValueError, match="dimension"):
        xp.nonzero(xp.asarray(1))
    big = xp.zeros((10**6, 10**6))
    big[3, 4] = 1.0
    out = xp.nonzero(big)
    assert [int(c[0]) for c in out] == [3, 4]


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted(side):
    rng = np.random.default_rng(11)
    for fill in [0.0, 2.0, -np.inf, np.nan]:
        for _ in range(20):
            a = np.sort(np.where(rng.random(9) < 0.5, fill, np.round(rng.normal(size=9) * 2)))
            x1 = make(a, fill)
            v = np.array([-10.0, -1.0, 0.0, fill, 1.0, 2.0, 2.5, np.inf, np.nan])
            for x2, vv in [(make(v, 0.0), v), (make(v, fill), v), (xp.asarray(1.0), np.array(1.0))]:
                out = xp.searchsorted(x1, x2, side=side)
                assert_array_equal(out, np.searchsorted(a, vv, side=side).astype(np.int64))
            assert int(xp.searchsorted(x1, 2.0, side=side)) == np.searchsorted(a, 2.0, side=side)
    a = np.array([3, 0, 1, 0, 5])
    sorter = np.argsort(a, kind="stable")
    out = xp.searchsorted(make(a, 0), xp.asarray([0, 4]), side=side, sorter=xp.asarray(sorter))
    assert_array_equal(out, np.searchsorted(a, [0, 4], side=side, sorter=sorter).astype(np.int64))
    with pytest.raises(ValueError, match="1-d"):
        xp.searchsorted(xp.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="side"):
        xp.searchsorted(xp.zeros(2), 1.0, side="middle")

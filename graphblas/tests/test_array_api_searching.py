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


def test_where_dtypes():
    for dt in DTYPES:
        for dense, x in inputs(dt, (5,), seed=6):
            c = dense.astype(bool) if dt.kind != "c" else dense.real.astype(bool)
            assert_array_equal(xp.where(xp.asarray(c), x, x), np.where(c, dense, dense))
            assert_array_equal(
                np_call(xp.where, xp.asarray(c), x, 2), np_call(np.where, c, dense, 2)
            )

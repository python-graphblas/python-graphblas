"""graphblas.array_api: the semiring= extension of matmul, tensordot and vecdot, and einsum."""

import numpy as np
import pytest

import graphblas as gb
import graphblas.array_api as xp

from .test_array_api_elementwise import check_invariants, make

rng = np.random.default_rng(58)


def _rand(shape, fill, dtype=np.float64, density=0.5):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        a = rng.random(shape) < 0.5
    elif dtype.kind in "iu":
        a = rng.integers(0, 9, shape)
    else:
        a = np.round(rng.normal(size=shape) * 4, 1)
    return np.where(rng.random(shape) < density, a, fill).astype(dtype)


def assert_close(x, expected, fill=None):
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape, (x.shape, expected.shape)
    if fill is not None and x.ndim:
        assert x.fill_value == fill or (np.isnan(fill) and np.isnan(x.fill_value))
    got = x.todense()
    if got.dtype.kind in "fc":
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(got, expected)


# (name, numpy add-monoid reduction, numpy multiply, identity) for the oracles
SEMIRINGS = {
    "min_plus": (np.min, np.add, np.inf),
    "max_plus": (np.max, np.add, -np.inf),
    "min_max": (np.min, np.maximum, np.inf),
    "max_min": (np.max, np.minimum, -np.inf),
}


def semiring_einsum(subscripts, *arrays, reduce, mult):
    """The numpy oracle: each term's operands combined by ``mult``, then ``reduce`` over the
    summed indices (every index spelled out, so each operand broadcasts over all of them).
    """
    inputs, output = subscripts.replace(" ", "").split("->")
    terms = inputs.split(",")
    letters = sorted(set("".join(terms)))
    total = None
    for term, a in zip(terms, arrays, strict=True):
        a = np.asarray(a)
        perm = [term.index(c) for c in letters if c in term]
        a = np.transpose(a, perm)
        shape = [a.shape[sorted(term, key=letters.index).index(c)] if c in term else 1
                 for c in letters]  # fmt: skip
        a = a.reshape(shape)
        total = a if total is None else mult(total, a)
    with np.errstate(invalid="ignore"):
        summed = tuple(i for i, c in enumerate(letters) if c not in output)
        out = reduce(total, axis=summed) if summed else total
    kept = [c for c in letters if c in output]
    return np.transpose(out, [kept.index(c) for c in output])


@pytest.mark.parametrize("name", list(SEMIRINGS))
@pytest.mark.parametrize(
    "shapes",
    [((3, 4), (4, 5)), ((4,), (4, 5)), ((3, 4), (4,)), ((4,), (4,)), ((2, 1, 3, 4), (3, 4, 2))],
    ids=str,
)
def test_matmul_semirings(name, shapes):
    reduce, mult, e = SEMIRINGS[name]
    for dtype in [np.float64, np.float32]:
        a, b = _rand(shapes[0], e, dtype), _rand(shapes[1], e, dtype)
        x1, x2 = make(a, e, rng), make(b, e)
        a2 = a[None] if a.ndim == 1 else a
        b2 = b[:, None] if b.ndim == 1 else b
        expected = reduce(mult(a2[..., :, :, None], b2[..., None, :, :]), axis=-2)
        if a.ndim == 1:
            expected = expected[..., 0, :]
        if b.ndim == 1:
            expected = expected[..., 0]
        expected = np.asarray(expected, dtype)
        for sr in [name, gb.semiring.from_string(name)]:
            assert_close(xp.matmul(x1, x2, semiring=sr), expected, e)
            assert_close(xp.linalg.matmul(x1, x2, semiring=sr), expected, e)
        # Stored values equal to the fill do not change the answer
        assert_close(xp.matmul(make(a, e, rng), make(b, e, rng), semiring=name), expected, e)


def test_matmul_semiring_dtypes():
    a, b = _rand((3, 4), np.inf, np.float32), _rand((4, 2), np.inf)
    expected = np.min(a[:, :, None] + b[None], axis=1)
    R = xp.matmul(make(a, np.inf), make(b, np.inf), semiring="min_plus")
    assert_close(R, expected.astype(np.float64), np.inf)
    # A typed semiring keeps its type
    R = xp.matmul(make(a, np.inf), make(b, np.inf), semiring=gb.semiring.min_plus["FP32"])
    assert_close(R, expected.astype(np.float32), np.inf)
    # Bools: lor_land (the default), and plus_times is the same thing
    p, q = _rand((3, 4), False, bool), _rand((4, 5), False, bool)
    for sr in [None, "lor_land", "plus_times", gb.semiring.lor_land]:
        assert_close(xp.matmul(make(p, False), make(q, False), semiring=sr), p @ q, False)
    # land_lor's identity is True
    expected = np.all(p[:, :, None] | q[None], axis=1)
    R = xp.matmul(make(p, True), make(q, True), semiring="land_lor")
    assert_close(R, expected, True)
    # Numbers through a bool semiring: nonzero values are True
    f, g = _rand((3, 4), 0.0), _rand((4, 2), 0.0)
    R = xp.matmul(make(f, 0.0), make(g, 0.0), semiring="lor_land")
    assert_close(R, (f != 0).astype(int) @ (g != 0).astype(int) > 0, False)
    # Unsigned max_times and plus_min: 0 annihilates times and min there
    u, v = _rand((3, 4), 0, np.uint8), _rand((4, 5), 0, np.uint8)
    expected = np.max(u[:, :, None] * v[None], axis=1)
    assert_close(xp.matmul(make(u, 0), make(v, 0), semiring="max_times"), expected, 0)
    expected = np.sum(np.minimum(u[:, :, None], v[None]), axis=1, dtype=np.uint8)
    assert_close(xp.matmul(make(u, 0), make(v, 0), semiring="plus_min"), expected, 0)
    # An explicit plus_times is the default, nonzero fills included
    for fa, fb in [(0, 0), (1, 0), (2, 3)]:
        a, b = _rand((3, 4), fa), _rand((4, 2), fb)
        assert_close(xp.matmul(make(a, fa), make(b, fb), semiring="plus_times"), a @ b)
    # Integer plus_times (any fill 0)
    i, j = _rand((3, 4), 0, np.int16), _rand((4, 2), 0, np.int16)
    assert_close(xp.matmul(make(i, 0), make(j, 0), semiring="plus_times"), i @ j, 0)


def test_contraction_skips_structurally():
    # As 0 * inf in plus_times (D6): an infinity meeting an unstored inf is no NaN
    a = make(np.array([[-np.inf, np.inf]]), np.inf)
    b = make(np.array([[np.inf], [-np.inf]]), np.inf)
    assert a.nnz == b.nnz == 1
    assert_close(xp.matmul(a, b, semiring="min_plus"), [[np.inf]], np.inf)
    # numpy: inf + -inf is NaN
    with np.errstate(invalid="ignore"):
        assert np.isnan(np.min(a.todense()[:, :, None] + b.todense()[None], axis=1)).all()


@pytest.mark.parametrize(
    ("xshape", "yshape", "axes"),
    [
        ((3, 4, 5), (5, 6), 1),
        ((3, 4, 5), (4, 5, 2), 2),
        ((3, 4, 5), (5, 6, 3), ([0, 2], [2, 0])),
        ((3, 4), (2, 5), 0),
        ((4,), (4,), 1),
    ],
    ids=str,
)
def test_tensordot_semirings(xshape, yshape, axes):
    for name, (reduce, mult, e) in SEMIRINGS.items():
        a, b = _rand(xshape, e), _rand(yshape, e)
        if isinstance(axes, int):
            a_axes, b_axes = list(range(len(xshape) - axes, len(xshape))), list(range(axes))
        else:
            a_axes, b_axes = axes
        la = "abcde"[: len(xshape)]
        lb = list("vwxyz"[: len(yshape)])
        for i, j in zip(a_axes, b_axes, strict=True):
            lb[j] = la[i]
        lb = "".join(lb)
        out = "".join(c for c in la if c not in lb) + "".join(c for c in lb if c not in la)
        subs = f"{la},{lb}->{out}"
        expected = semiring_einsum(subs, a, b, reduce=reduce, mult=mult)
        R = xp.tensordot(make(a, e), make(b, e, rng), axes=axes, semiring=name)
        assert_close(R, expected, e)
        assert_close(xp.linalg.tensordot(make(a, e), make(b, e), axes=axes, semiring=name),
                     expected, e)  # fmt: skip


@pytest.mark.parametrize(
    ("xshape", "yshape", "axis"),
    [((3, 4), (3, 4), -1), ((3, 4), (4,), -1), ((2, 3, 4), (3, 4), -2), ((3, 1, 4), (5, 4), -1)],
    ids=str,
)
def test_vecdot_semirings(xshape, yshape, axis):
    for name, (reduce, mult, e) in SEMIRINGS.items():
        a, b = _rand(xshape, e), _rand(yshape, e)
        a_b, b_b = np.broadcast_arrays(a, b)
        expected = reduce(mult(a_b, b_b), axis=axis)
        for func in [xp.vecdot, xp.linalg.vecdot]:
            R = func(make(a, e, rng), make(b, e), axis=axis, semiring=name)
            assert_close(R, expected, e)
    # lor_land of bools is the default's any(x1 & x2)
    p, q = _rand((3, 4), False, bool), _rand((3, 4), False, bool)
    assert_close(xp.vecdot(make(p, False), make(q, False), semiring="lor_land"),
                 np.any(p & q, axis=-1), False)  # fmt: skip


EINSUMS = [
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("ik,jk->ij", [(3, 4), (5, 4)]),
    ("ij,jk,kl->il", [(3, 4), (4, 5), (5, 2)]),
    ("ijk,jr,kr->ir", [(3, 4, 5), (4, 2), (5, 2)]),
    ("ij,ij->i", [(3, 4), (3, 4)]),
    ("i,j->ij", [(3,), (4,)]),
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),
    ("ij->ji", [(3, 4)]),
    ("ij->i", [(3, 4)]),
    ("ijk->", [(2, 3, 4)]),
    ("ij,jk", [(3, 4), (4, 5)]),
    ("ij,j->i", [(3, 4), (4,)]),
]


@pytest.mark.parametrize(("subscripts", "shapes"), EINSUMS, ids=[s for s, _ in EINSUMS])
def test_einsum(subscripts, shapes):
    # The default: numpy's einsum, zero fills structural, other fills densely through numpy
    for dtype in [np.float64, np.int64, np.bool_, np.complex128]:
        for fills in [(0,) * len(shapes), (1,) + (0,) * (len(shapes) - 1)]:
            if np.dtype(dtype).kind == "b" and any(fills):
                fills = (True,) * len(shapes)
            arrays = [_rand(s, f, dtype) for s, f in zip(shapes, fills, strict=True)]
            expected = np.einsum(subscripts, *arrays)
            R = xp.einsum(subscripts, *[make(a, f) for a, f in zip(arrays, fills, strict=True)])
            assert_close(R, expected)
    # Semirings
    full = subscripts if "->" in subscripts else subscripts + "->ik"
    for name, (reduce, mult, e) in SEMIRINGS.items():
        arrays = [_rand(s, e) for s in shapes]
        expected = semiring_einsum(full, *arrays, reduce=reduce, mult=mult)
        R = xp.einsum(subscripts, *[make(a, e, rng) for a in arrays], semiring=name)
        assert_close(R, np.asarray(expected), e)


def test_einsum_operands_and_errors():
    a = _rand((3, 4), 0.0)
    x = make(a, 0.0)
    # numpy operands, empty operands and sizes
    assert_close(xp.einsum("ij,kj->ik", a, x), a @ a.T)
    assert_close(xp.einsum("ij,jk->ik", x, xp.zeros((4, 2))), np.zeros((3, 2)))
    assert_close(xp.einsum("ij,jk->ik", xp.zeros((3, 0)), xp.zeros((0, 2))), np.zeros((3, 2)))
    R = xp.einsum("ij,jk->ik", xp.full((3, 0), np.inf), xp.full((0, 2), np.inf),
                  semiring="min_plus")  # fmt: skip
    assert_close(R, np.full((3, 2), np.inf), np.inf)
    # A 0-d operand
    assert_close(xp.einsum(",ij->ij", xp.asarray(2.0), x), 2 * a)
    assert_close(xp.einsum(",ij->ij", xp.asarray(0.0), x), 0 * a)
    with pytest.raises(TypeError, match="subscripts"):
        xp.einsum(x, x)
    with pytest.raises(ValueError, match="at least one"):
        xp.einsum("ij")
    with pytest.raises(TypeError, match="Arrays"):
        xp.einsum("ij->ij", [[1.0]])
    with pytest.raises(NotImplementedError, match="not supported"):
        xp.einsum("...ij->...ji", x)
    with pytest.raises(NotImplementedError, match="diagonal"):
        xp.einsum("ii->i", make(np.eye(3), 0.0))
    with pytest.raises(ValueError, match="fill is inf"):
        xp.einsum("ij,jk->ik", x, x.T, semiring="min_plus")
    with pytest.raises(ValueError, match="operand 1"):
        xp.einsum("ij,jk->ik", xp.full((3, 4), np.inf), x.T, semiring="min_plus")
    # One operand only sums: the identity is all its fill needs
    b = _rand((3, 4), -np.inf)
    R = xp.einsum("ij->i", make(b, -np.inf), semiring="max_times")
    assert_close(R, np.max(b, axis=1), -np.inf)
    with gb.config.set(array_api_densify_limit=5), pytest.raises(xp.DensifyError):
        xp.einsum("ij,jk->ik", make(a, 1.0), x.T)


def test_semiring_errors():
    a = make(_rand((3, 4), np.inf), np.inf)
    b = make(_rand((4, 2), np.inf), np.inf)
    z = make(_rand((4, 2), 0.0), 0.0)
    # Fills that are not the identity
    with pytest.raises(ValueError, match="x2 has the fill 0.0"):
        xp.matmul(a, z, semiring="min_plus")
    with pytest.raises(ValueError, match="x1 has the fill inf"):
        xp.tensordot(a, z, axes=1, semiring="max_plus")
    with pytest.raises(ValueError, match="fill is -inf"):
        xp.vecdot(a, a, semiring="max_plus")
    with pytest.raises(ValueError, match="fill is inf"):
        xp.matmul(make(_rand((3, 4), np.nan), np.nan), b, semiring="min_plus")
    with pytest.raises(ValueError, match="fill_value=xp.inf"):
        xp.matmul(z.T, z, semiring="min_plus")
    with pytest.raises(ValueError, match="fill_value=-xp.inf"):
        xp.matmul(z.T, z, semiring="max_plus")
    t = xp.zeros((4, 2), dtype=xp.bool)
    with pytest.raises(ValueError, match="fill_value=True"):
        xp.matmul(t.T, t, semiring="land_lor")
    # Semirings whose identity does not annihilate the multiply: no fill works
    for sr, dtype in [
        ("max_times", np.float64),
        ("plus_pair", np.float64),
        ("any_pair", np.float64),
        ("min_first", np.float64),
        ("plus_first", np.float64),
        ("min_plus", np.int64),
        ("max_plus", np.uint8),
        ("min_times", np.uint16),
        ("lor_eq", np.float64),
    ]:
        e = np.asarray(0).astype(dtype)
        x = make(_rand((3, 4), e, dtype), e)
        y = make(_rand((4, 2), e, dtype), e)
        with pytest.raises(ValueError, match="cannot skip"):
            xp.matmul(x, y, semiring=sr)
        with pytest.raises(ValueError, match="cannot skip"):
            xp.einsum("ij,jk->ik", x, y, semiring=sr)
    with pytest.raises(ValueError, match="cannot skip"):
        xp.matmul(a, b, semiring=gb.semiring.ss.min_firsti)
    with pytest.raises(ValueError, match="unknown semiring"):
        xp.matmul(a, b, semiring="min.nope")
    with pytest.raises(TypeError, match="Semiring or its name"):
        xp.matmul(a, b, semiring=gb.binary.plus)
    with pytest.raises(TypeError, match="does not work with complex128"):
        xp.matmul(xp.astype(a, xp.complex128), b, semiring="min_plus")
    with pytest.raises(TypeError, match="does not work with bool"):
        xp.vecdot(xp.zeros(3, dtype=xp.bool), xp.zeros(3, dtype=xp.bool), semiring="min_plus")


def test_probes():
    from graphblas.array_api._linalg import _probes

    for dt in ["bool", "int8", "uint64", "float32", "complex64"]:
        p = _probes(np.dtype(dt))
        assert p.dtype == np.dtype(dt)
        if p.dtype.kind in "fc":
            assert np.isfinite(p).all()
    assert {-128, 127, 0} <= set(_probes(np.dtype("int8")).tolist())


def test_explicit_fills_do_not_change_values():
    # (review finding M2, DESIGN D12a) a stored entry equal to the fill meets an inf as an
    # unstored position does: no term
    a = xp.from_coo(([0], [0]), [0.0], (1, 1))
    b = make(np.array([[np.inf]]), 0.0)
    assert_close(a @ b, [[0.0]])
    assert_close(xp.prune(a) @ b, [[0.0]])
    a = xp.from_coo(([0], [0]), [np.inf], (1, 1), fill_value=np.inf)
    b = make(np.array([[-np.inf]]), np.inf)
    for x in [a, xp.prune(a)]:
        assert_close(xp.matmul(x, b, semiring="min_plus"), [[np.inf]], np.inf)
    # with no inf or NaN anywhere, explicit fills stay (nothing to prune)
    a = xp.from_coo(([0, 0], [0, 1]), [0.0, 2.0], (1, 2))
    b = make(np.array([[1.0], [3.0]]), 0.0)
    assert_close(a @ b, [[6.0]])


def test_nan_propagates_through_min_and_max():
    # (review finding M1, DESIGN D12b) a stored NaN term makes the min or max NaN, as numpy's
    # do, although SuiteSparse's fmin and fmax skip it; positions it does not reach keep theirs
    inf = np.inf
    a = np.array([[np.nan, 5.0], [inf, 2.0]])
    b = np.array([[1.0, inf], [3.0, 4.0]])
    A, B = make(a, inf), make(b, inf)
    for name, (reduce, mult, e) in [("min_plus", SEMIRINGS["min_plus"]),
                                    ("min_max", SEMIRINGS["min_max"])]:  # fmt: skip
        with np.errstate(invalid="ignore"):
            expected = reduce(mult(a[:, :, None], b[None]), axis=1)
        # (D6: a NaN meeting an unstored inf is no term, so [0, 1] keeps its value)
        expected[0, 1] = reduce(mult(a[0, 1], b[1, 1]))
        assert_close(xp.matmul(A, B, semiring=name), expected, e)
        assert_close(xp.tensordot(A, B, axes=1, semiring=name), expected, e)
        assert_close(xp.einsum("ij,jk->ik", A, B, semiring=name), expected, e)
    m = make(np.array([[np.nan, 5.0]]), -inf)
    n = make(np.array([[1.0], [3.0]]), -inf)
    assert_close(xp.matmul(m, n, semiring="max_plus"), [[np.nan]], -inf)
    assert_close(xp.vecdot(m, xp.matrix_transpose(n), semiring="max_plus"), [np.nan], -inf)
    x = make(np.array([2.0, np.nan, 1.0]), inf)
    assert_close(xp.einsum("b->", x, semiring="min_max"), np.array(np.nan))
    assert_close(xp.einsum("b->", make(np.array([2.0, 1.0]), inf), semiring="min_max"),
                 np.array(1.0))  # fmt: skip


def test_soundness_is_cached_per_semiring_object():
    # (review finding L1) two semirings of one name: each is checked
    from graphblas.core.operator import Semiring

    s1 = Semiring.register_anonymous(gb.monoid.min, gb.binary.plus, name="same_name")
    s2 = Semiring.register_anonymous(gb.monoid.min, gb.binary.minus, name="same_name")
    a = make(np.array([[1.0, 2.0]]), np.inf)
    b = make(np.array([[1.0], [3.0]]), np.inf)
    assert_close(xp.matmul(a, b, semiring=s1), [[2.0]], np.inf)
    with pytest.raises(ValueError, match="cannot skip"):
        xp.matmul(a, b, semiring=s2)


def test_vecdot_plus_times_conjugates():
    # (review finding L4) the default semiring, given or not, conjugates x1
    x = make(np.array([1j, 2.0]), 0j)
    y = make(np.array([1j, 1.0]), 0j)
    for semiring in [None, "plus_times"]:
        assert_close(xp.vecdot(x, y, semiring=semiring), np.array(3 + 0j))


def test_einsum_with_scalars_and_ones_stays_sparse():
    # (review finding L5) a 0-d operand is a scalar factor, and a small operand with a finite
    # nonzero fill (a vector of ones: row sums) is stored in full, not densified with the rest
    rng = np.random.default_rng(18)
    n = 400
    a = np.zeros((n, n))
    a[rng.integers(0, n, 600), rng.integers(0, n, 600)] = rng.random(600) + 0.5
    A = xp.from_coo(np.nonzero(a), a[np.nonzero(a)], (n, n))
    two, ones = xp.asarray(2.0), xp.ones(n)
    with gb.config.set(array_api_densify_limit=10 * n):
        outs = [
            xp.einsum("ij,j->i", A, ones),
            A @ ones,
            xp.tensordot(A, ones, axes=1),
            xp.einsum("ij,->ij", A, two),
            xp.einsum(",ij,->ji", two, A, two),
            xp.einsum("ij,", A, two),
        ]
    for out, expected in zip(
        outs, [a.sum(1), a.sum(1), a.sum(1), 2 * a, 4 * a.T, 2 * a], strict=True
    ):
        assert_close(out, expected)
    assert_close(xp.einsum(",->", two, xp.asarray(3.0)), np.array(6.0))
    # a non-finite fill that shows goes dense, as numpy computes it (inf * 0 is NaN there)
    c = np.array([[1.0, np.inf], [np.inf, 2.0]])
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    with np.errstate(invalid="ignore"):
        assert_close(xp.einsum("ij,->", make(d, 0.0), xp.asarray(np.inf)),
                     np.einsum("ij,->", d, np.inf))  # fmt: skip
    with np.errstate(invalid="ignore"):
        expected = np.einsum("ij,jk->ik", c, d)
    assert_close(xp.einsum("ij,jk->ik", make(c, np.inf), make(d, 0.0)), expected)

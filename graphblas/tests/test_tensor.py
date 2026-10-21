import gc
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, backend, binary, dtypes, monoid
from graphblas.core import tensor_layout as tl
from graphblas.core.tensor import Layout, Tensor, TensorExpression
from graphblas.exceptions import DimensionMismatch, IndexOutOfBound

if backend != "suitesparse":
    pytest.skip("Tensor needs the suitesparse backend", allow_module_level=True)


def all_layouts(ndim):
    return [
        Layout(seq[:p], seq[p:])
        for seq in itertools.permutations(range(ndim))
        for p in range(ndim + 1)
    ]


def random_tensor(shape, rng, *, density=0.3, dtype=dtypes.FP64, layout=None, iso=False):
    """A Tensor and its numpy model: dense values plus a boolean "present" array."""
    present = rng.random(shape) < density
    if dtype == dtypes.BOOL:
        values = rng.random(shape) < 0.5
    elif dtype == dtypes.INT64:
        values = rng.integers(-5, 6, shape)
    else:
        values = np.round(rng.random(shape) * 10 - 5, 1)
    if iso:
        values = np.full(shape, values.flat[0] if values.size else 1, dtype=values.dtype)
    values = np.where(present, values, 0).astype(dtype.np_type)
    coords = np.nonzero(present)
    vals = values[present]
    if iso:
        vals = vals[0] if vals.size else values.dtype.type(1)
    T = Tensor.from_coo(coords, vals, dtype, shape=shape, layout=layout)
    return T, values, present


def check(T, values, present, *, approx=False):
    """Compare a Tensor with its numpy model, entry by entry."""
    assert type(T) is Tensor
    T._check_invariants()
    assert T.shape == values.shape
    assert T.nvals == present.sum()
    coords, vals = T.to_coo()
    got = np.zeros(values.shape, bool)
    # An empty coords tuple indexes the one element of a 0-d array, entry or not
    got[coords] = vals.size > 0
    np.testing.assert_array_equal(got, present)
    if approx:
        np.testing.assert_allclose(vals, values[present], rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_array_equal(vals, values[present])


def model_update(cv, cp, zv, zp, *, mask=None, accum=None, replace=False):
    """GraphBLAS C<mask, replace> accum= Z on numpy models."""
    if accum is not None:
        both = cp & zp
        tv = np.where(zp, zv, cv).astype(np.result_type(cv, zv))
        tv[both] = accum(cv[both], zv[both])
        tp = cp | zp
    else:
        tv, tp = zv, zp
    if mask is None:
        return tv, tp
    rp = np.where(mask, tp, False if replace else cp)
    rv = np.where(mask, tv, cv)
    return np.where(rp, rv, 0), rp


# --------------------------------------------------------------------------------------------
# M1: data model


def test_layout_validation():
    assert tl.as_layout(((0, 1), (2,)), (2, 3, 4)) == Layout((0, 1), (2,))
    assert tl.as_layout([[2], [0, 1]], (2, 3, 4)) == Layout((2,), (0, 1))
    with pytest.raises(ValueError, match="partition"):
        tl.as_layout(((0, 1), (1,)), (2, 3, 4))
    with pytest.raises(ValueError, match="partition"):
        tl.as_layout(((0,), (2,)), (2, 3, 4))
    with pytest.raises(TypeError, match="pair"):
        tl.as_layout("01|2", (2, 3, 4))
    with pytest.raises(ValueError, match="2\\*\\*60"):
        tl.as_layout(((0, 1, 2), (3,)), (2**21, 2**21, 2**21, 2))
    assert tl.as_layout(((0, 1), (2,)), (2**30, 2**30, 2**40))
    assert Layout((0, 1), (2,))._short() == "(0,1|2)"
    assert Layout((), ())._short() == "(|)"


def test_default_layout():
    assert tl.default_layout(()) == Layout((), ())
    assert tl.default_layout((5,)) == Layout((), (0,))
    assert tl.default_layout((5, 6)) == Layout((0,), (1,))
    assert tl.default_layout((10, 20, 30)) == Layout((0, 1), (2,))
    # DESIGN.md 7.6: cases D, H and Uber
    assert tl.default_layout((1000, 1000, 1000), 10**7) == Layout((0, 1), (2,))
    assert tl.default_layout((2**19,) * 3, 10**6) == Layout((0,), (1, 2))
    assert tl.default_layout((183, 24, 1140, 1717), 3_309_490) == Layout((0, 1), (2, 3))
    # Every split of (2**29,)*4 has a group past 2**60 except the middle one
    assert tl.default_layout((2**29,) * 4, 10) == Layout((0, 1), (2, 3))
    with pytest.raises(ValueError, match="no split"):
        tl.default_layout((2**61, 2, 2), 10)


def test_new():
    T = Tensor(dtypes.FP64, 3, 4, 5)
    U = Tensor(float, (3, 4, 5), name="U")
    for X in [T, U]:
        X._check_invariants()
        assert X.shape == (3, 4, 5)
        assert X.ndim == 3
        assert X.size == 60
        assert X.nvals == 0
        assert X.dtype == dtypes.FP64
        assert X.layout == Layout((0, 1), (2,))
        assert not X.layout_pinned
    assert U.name == "U"
    V = Tensor(int, 2, 3, layout=((1,), (0,)))
    assert V.layout == Layout((1,), (0,))
    assert V.layout_pinned
    assert V.dtype == dtypes.INT64
    Z = Tensor(bool)
    assert Z.shape == ()
    assert Z.ndim == 0
    assert Z.size == 1
    Z._check_invariants()
    E = Tensor(float, 3, 0, 2)
    assert E.size == 0
    E._check_invariants()
    with pytest.raises(ValueError, match="negative"):
        Tensor(float, 3, -1)
    with pytest.raises(TypeError, match="shape"):
        Tensor(float, 3, "x")


def test_from_coo_forms():
    i = np.array([0, 1, 2, 2])
    j = np.array([1, 0, 3, 3])
    k = np.array([4, 2, 0, 1])
    vals = np.array([1.5, 2.5, 3.5, 4.5])
    A = Tensor.from_coo((i, j, k), vals, shape=(3, 4, 5))
    B = Tensor.from_coo(np.stack([i, j, k]), vals, shape=(3, 4, 5))
    C = Tensor.from_coo([list(i), list(j), list(k)], list(vals))
    for X in [A, B, C]:
        X._check_invariants()
        assert X.isequal(A, check_dtype=True)
    assert C.shape == (3, 4, 5)
    coords, v = A.to_coo()
    np.testing.assert_array_equal(coords[0], i)
    np.testing.assert_array_equal(coords[2], k)
    np.testing.assert_array_equal(v, vals)
    assert all(c.dtype == np.int64 for c in coords)
    # Scalar values are stored once (iso)
    D = Tensor.from_coo((i, j, k), 7, shape=(3, 4, 5))
    assert D.ss.is_iso
    assert D.dtype == dtypes.INT64
    assert D.to_coo()[1].tolist() == [7] * 4
    # dtype and layout
    E = Tensor.from_coo((i, j, k), vals, dtypes.FP32, shape=(3, 4, 5), layout=((2,), (0, 1)))
    assert E.dtype == dtypes.FP32
    assert E.layout == Layout((2,), (0, 1))
    assert E.layout_pinned
    assert E.isequal(A)
    # duplicates
    F = Tensor.from_coo(([0, 0], [1, 1], [2, 2]), [1.0, 2.0], dup_op=binary.plus)
    assert F.nvals == 1
    assert F.to_coo()[1].tolist() == [3.0]
    with pytest.raises(ValueError, match="dup_op"):
        Tensor.from_coo(([0, 0], [1, 1], [2, 2]), [1.0, 2.0])


def test_from_coo_errors_and_edges():
    with pytest.raises(IndexOutOfBound, match="axis 1"):
        # Raveled, (0, 4, 0) would land on (1, 0, 0) silently
        Tensor.from_coo(([0], [4], [0]), [1.0], shape=(2, 4, 3))
    with pytest.raises(IndexOutOfBound, match="axis 0"):
        Tensor.from_coo(([-1], [0], [0]), [1.0], shape=(2, 4, 3))
    with pytest.raises(ValueError, match="same length"):
        Tensor.from_coo(([0, 1], [0], [0]), [1.0], shape=(2, 4, 3))
    with pytest.raises(ValueError, match="infer shape"):
        Tensor.from_coo(([], [], []), [])
    with pytest.raises(ValueError, match="axes"):
        Tensor.from_coo(([0], [0]), [1.0], shape=(2, 4, 3))
    with pytest.raises(TypeError, match="integers"):
        Tensor.from_coo(([0.5], [0], [0]), [1.0], shape=(2, 4, 3))
    with pytest.raises(ValueError, match="1-d"):
        Tensor.from_coo(np.zeros((2, 2, 2), int), [1.0], shape=(2, 4, 3))
    E = Tensor.from_coo(([], [], []), [], dtypes.FP64, shape=(2, 3, 4))
    assert E.nvals == 0
    E._check_invariants()
    # 0-d and 1-d
    Z = Tensor.from_coo((), 5.0)
    assert Z.shape == ()
    assert Z.to_scalar().value == 5.0
    coords, v = Z.to_coo()
    assert coords == ()
    assert v.tolist() == [5.0]
    v = Tensor.from_coo(([3, 1],), [1, 2], shape=(6,))
    assert v.layout == Layout((), (0,))
    w = v.to_vector()
    assert w.isequal(Vector.from_coo([1, 3], [2, 1], size=6))
    # Size-1 axes and an n x 1 stored Matrix (held by row, unlike SuiteSparse's default)
    U = Tensor.from_coo(([0, 1], [0, 2], [0, 0]), [1.0, 2.0], shape=(2, 3, 1))
    U._check_invariants()
    U = Tensor.from_coo(
        ([0, 1], [0, 2], [0, 0]), [1.0, 2.0], shape=(2, 3, 1), layout=((0, 1), (2,))
    )
    U._check_invariants()
    assert U.ss.matrix.shape == (6, 1)


@pytest.mark.parametrize("layout", all_layouts(3))
def test_to_coo_order(layout):
    rng = np.random.default_rng(1)
    T, values, present = random_tensor((3, 4, 5), rng, layout=layout)
    check(T, values, present)
    # sort=False returns stored order without moving data
    coords, vals = T.to_coo(sort=False)
    got = np.zeros(values.shape, values.dtype)
    got[coords] = vals
    np.testing.assert_array_equal(got, values)
    assert T.to_coo(values=False)[1] is None
    assert T.to_coo(indices=False)[0] is None


def test_from_matrix_and_back():
    A = Matrix.from_coo([0, 1, 2], [1, 2, 0], [1.0, 2.0, 3.0], nrows=3, ncols=4)
    T = Tensor.from_matrix(A)
    T._check_invariants()
    assert T.shape == (3, 4)
    assert T.layout == Layout((0,), (1,))
    assert T.to_matrix().isequal(A)
    # A held by column is stored as its transpose held by row, with no transpose
    B = A.dup()
    B.ss.config["format"] = "by_col"
    TB = Tensor.from_matrix(B)
    TB._check_invariants()
    assert TB.layout == Layout((1,), (0,))
    assert TB.isequal(T)
    assert TB.to_matrix().isequal(A)
    assert B.ss.orientation == "columnwise"  # the user's Matrix is untouched
    # A.T
    TT = Tensor.from_matrix(A.T, name="TT")
    TT._check_invariants()
    assert TT.name == "TT"
    assert TT.shape == (4, 3)
    assert TT.to_matrix().isequal(A.T.new())
    # copy=False shares, and the Tensor copies before changing it
    C = A.dup()
    TC = Tensor.from_matrix(C, copy=False)
    assert TC.ss.matrix is C
    TC << TC * 10
    assert C.isequal(A)
    TC2 = Tensor.from_matrix(C, copy=False)
    TC2(TC2.S, accum=binary.plus) << TC2
    assert C.isequal(A)
    assert TC2.to_matrix().isequal((A * 2).new())
    with pytest.raises(TypeError, match="Matrix"):
        Tensor.from_matrix(Vector(float, 3))
    with pytest.raises(TypeError, match="Vector"):
        Tensor.from_vector(A)
    v = Vector.from_coo([0, 4], [1.0, 2.0], size=6)
    Tv = Tensor.from_vector(v)
    Tv._check_invariants()
    assert Tv.to_vector().isequal(v)
    assert v.nvals == 2


def test_to_matrix_rows():
    rng = np.random.default_rng(2)
    T, values, present = random_tensor((3, 4, 5), rng, layout=((2,), (0, 1)))
    M = T.to_matrix()  # stored grouping
    assert M.shape == (5, 12)
    assert M is not T.ss.matrix
    M2 = T.to_matrix(copy=False)
    assert M2 is T.ss.matrix
    for rows in [(0,), (1,), (2,), (0, 1), (1, 0), (2, 0), (0, 1, 2), ()]:
        cols = tuple(a for a in range(3) if a not in rows)
        A = T.to_matrix(rows)
        rshape = tuple(values.shape[a] for a in rows)
        cshape = tuple(values.shape[a] for a in cols)
        dense = values.transpose(rows + cols).reshape(math.prod(rshape), math.prod(cshape))
        pattern = present.transpose(rows + cols).reshape(dense.shape)
        r, c, v = A.to_coo()
        np.testing.assert_array_equal(v, dense[r, c])
        assert A.nvals == pattern.sum()
    T2 = Tensor.from_coo(([0, 1], [1, 0]), [1.0, 2.0], shape=(2, 2), layout=((1,), (0,)))
    assert T2.to_matrix().isequal(Matrix.from_coo([0, 1], [1, 0], [1.0, 2.0]))
    with pytest.raises(ValueError, match="1-d"):
        T.to_vector()
    with pytest.raises(ValueError, match="0-d"):
        T.to_scalar()


def test_repr():
    T = Tensor.from_coo(([0, 1, 2], [1, 0, 3], [4, 2, 0]), [1.0, 2.0, 3.0], name="X")
    r = repr(T)
    assert '"X"' in r
    assert "layout (0|1,2)" in r
    assert "(0, 1, 4): 1.0" in r
    assert "storage order" in repr(T.permute_dims((2, 0, 1)))
    assert "StructuralMask" in repr(T.S)
    big = Tensor.from_coo((np.arange(20), np.arange(20)), 1.0)
    assert "..." in repr(big)
    assert "TensorExpression" in repr(T + T)
    assert "X.ewise_add(X" in repr(T + T)
    assert repr(Tensor(float, 2, 3))


def test_dup_and_clear():
    rng = np.random.default_rng(3)
    T, values, present = random_tensor((3, 4, 5), rng)
    D = T.dup()
    check(D, values, present)
    assert D.ss.matrix is T.ss.matrix  # copy-on-write
    D2 = T.dup(dtypes.INT64, layout=((1,), (2, 0)), name="D2")
    assert D2.dtype == dtypes.INT64
    assert D2.layout == Layout((1,), (2, 0))
    assert D2.layout_pinned
    check(D2, values.astype(np.int64), present)
    E = T.dup(clear=True)
    assert E.nvals == 0
    assert E.shape == T.shape
    D.clear()
    assert D.nvals == 0
    check(T, values, present)


def test_vanilla_check(monkeypatch):
    from graphblas.core import tensor

    monkeypatch.setattr(tensor, "backend", "suitesparse-vanilla")
    with pytest.raises(NotImplementedError, match="suitesparse"):
        Tensor(float, 2, 3)


def test_by_col_global_format():
    """Every stored Matrix stays held by row when the global format is by column (F6)."""
    rng = np.random.default_rng(4)
    prev = gb.ss.config["format"]
    gb.ss.config["format"] = "by_col"
    try:
        T, values, present = random_tensor((3, 4, 5), rng)
        assert T.ss.format.endswith("r")
        check(T, values, present)
        for layout in all_layouts(3)[::5]:
            U = T.relayout(layout)
            check(U, values, present)
        U, uvals, upres = random_tensor((3, 4, 5), rng, layout=((2,), (1, 0)))
        S = (T + U).new()
        check(S, np.where(present, values, 0) + np.where(upres, uvals, 0), present | upres)
        R = T.reduce(axis=(0, 2)).new()
        check(R, values.sum(axis=(0, 2)), present.any(axis=(0, 2)), approx=True)
        R = T.reduce(axis=1).new()
        check(R, values.sum(axis=1), present.any(axis=1), approx=True)
        Z = T.sum().new()
        Z._check_invariants()
        v = Vector.from_coo([0, 2], [1.0, 2.0], size=3)
        Tv = Tensor.from_vector(v)
        Tv._check_invariants()
        assert Tv.to_vector().isequal(v)
    finally:
        gb.ss.config["format"] = prev


# --------------------------------------------------------------------------------------------
# M2: relayout


def _rebuild_reference(T, layout):
    coords, vals = T.to_coo()
    return Tensor.from_coo(coords, vals, T.dtype, shape=T.shape, layout=layout)


@pytest.mark.parametrize("shape", [(3, 4, 5), (1, 7, 2), (50, 1, 40)])
def test_relayout_all_pairs(shape):
    """Every planned relayout between all 24 x 24 3-d states equals a REBUILD."""
    rng = np.random.default_rng(5)
    present = rng.random(shape) < 0.3
    coords = np.nonzero(present)
    vals = rng.random(present.sum())
    states = all_layouts(3)
    for src in states:
        T = Tensor.from_coo(coords, vals, shape=shape, layout=src)
        for dst in states:
            # Price as if large so the plans use the real primitives (not REBUILD), at both
            # thread counts, as the s12 spike did.
            for nthreads in [8, 1]:
                _, path, final = tl.plan(shape, src.state, frozenset([dst.state]), 10**7, nthreads)
                assert final == dst.state
                M, state = tl.execute(T.ss.matrix, shape, src.state, path)
                assert state == dst.state
                ref = Tensor.from_coo(coords, vals, shape=shape, layout=dst)
                assert M.isequal(ref.ss.matrix), (src, dst, path)
                assert tl.is_by_row(M)
            # And through the public API, with the real nnz
            U = T.relayout(dst)
            U._check_invariants()
            assert U.layout == dst
            assert U.layout_pinned
            assert U.ss.matrix.isequal(ref.ss.matrix)


def test_relayout_plans_match_spike():
    """The planner finds the measured best chains of s1c/s12 for 1000^3, 1e7 nnz."""
    shape = (1000, 1000, 1000)
    src = ((0, 1, 2), 2)
    expected = {
        (1, 0, 2): ["PERM_R"],
        (2, 0, 1): ["T", "RS"],
        (1, 2, 0): ["T", "RS", "T", "RS"],
        (2, 1, 0): ["PERM_R", "T", "RS"],
        (0, 2, 1): ["T", "RS", "PERM_R"],
    }
    for tgt, kinds in expected.items():
        _, path, _ = tl.plan(shape, src, frozenset([(tgt, 2)]), 10**7, 8)
        assert [e[0] for e in path] == kinds, (tgt, path)
    # Hypersparse case H (about 1 entry per row): no P @ M, whose rows pay a hash lookup each;
    # REBUILD (27 / 141 ms at 8 / 1 threads) or a PERM_C chain (25 / 104 ms)
    _, path, _ = tl.plan(
        (2**19,) * 3, ((0, 1, 2), 1), frozenset([((1, 0, 2), 2)]), 10**6, 8, hyper1=True
    )
    assert "PERM_R" not in [e[0] for e in path]


def test_perm_cols_paths():
    """PERM_C reorders the column group's axes in the Container: through a cached map, by
    computing the indices (a column group far wider than the data), by REBUILD for full
    storage; any dtype, iso values, an empty Matrix; the input is never changed.
    """
    rng = np.random.default_rng(66)
    udt = dtypes.register_anonymous(np.dtype([("tensor_a", "<i4"), ("tensor_b", "<f8")]))
    cases = [((6, 7, 8), 0.3), ((5, 2**20, 2**20), None), ((3, 4, 5), 1.0), ((3, 4, 5), 0.0)]
    for shape, density in cases:
        if density is None:
            coords = [rng.integers(0, n, 400) for n in shape]
            lin = np.unique(np.ravel_multi_index(coords, shape))
            coords = np.unravel_index(lin, shape)
        else:
            coords = np.nonzero(rng.random(shape) < density)
        n = coords[0].size
        udt_vals = np.zeros(n, dtype=udt.np_type)
        udt_vals["tensor_a"] = np.arange(n)
        for vals, dtype in [(rng.random(n), None), (1, "INT8"), (udt_vals, udt)]:
            T = Tensor.from_coo(coords, vals, dtype, shape=shape, layout=((0,), (1, 2)))
            M = T.ss.matrix
            before = M.to_coo()
            for path in [[("PERM_C", (2, 1))], [("RS", 1), ("PERM_C", (2, 1))]]:
                R, state = tl.execute(M, shape, ((0, 1, 2), 1), path)
                assert state == ((0, 2, 1), 1)
                U = Tensor.from_coo(coords, vals, dtype, shape=shape, layout=((0,), (2, 1)))
                # (compared as arrays: isequal on a UDT needs Numba)
                r, c, v = R.to_coo()
                r2, c2, v2 = U.ss.matrix.to_coo()
                np.testing.assert_array_equal(r, r2)
                np.testing.assert_array_equal(c, c2)
                np.testing.assert_array_equal(v, v2)
                for got, expected in zip(M.to_coo(), before, strict=True):
                    np.testing.assert_array_equal(got, expected)
            assert tl._col_map_ok(M.ncols, M.nvals) == (density is not None)
    # The planner takes PERM_C where it replaces T PERM_R T: rows of about 10 entries
    shape = (10**6, 10, 100)
    _, path, _ = tl.plan(shape, ((0, 1, 2), 1), frozenset([((0, 2, 1), 1)]), 10**7, 8)
    assert path == [("PERM_C", (2, 1))]


def test_big_total_split_move():
    """Past 2^60 positions in total reshape is unavailable; a split move is a REBUILD whose keys
    arrive sorted, priced as such, and chains use it.
    """
    shape = (2**21, 2**21, 2**20)
    src, dst = ((0, 1, 2), 1), ((0, 1, 2), 2)
    cost, path, _ = tl.plan(shape, src, frozenset([dst]), 10**7, 8)
    assert path == [("REBUILD", dst)]
    assert cost == tl.rebuild_cost(10**7, 0, sorted_keys=True)
    assert cost < tl.rebuild_cost(10**7, 0)
    rng = np.random.default_rng(67)
    coords = [rng.integers(0, n, 300) for n in shape]
    T = Tensor.from_coo(coords, rng.random(300), shape=shape, layout=((0,), (1, 2)))
    coords, vals = T.to_coo()
    for layout in [((0, 1), (2,)), ((2,), (0, 1)), ((1, 0), (2,))]:
        _, path, _ = tl.plan(shape, src, frozenset([Layout(*layout).state]), 10**7, 8)
        M, state = tl.execute(T.ss.matrix, shape, src, path)
        ref = Tensor.from_coo(coords, vals, shape=shape, layout=layout)
        assert state == ref.layout.state
        assert M.isequal(ref.ss.matrix), (layout, path)


def test_relayout_hypersparse():
    rng = np.random.default_rng(6)
    shape = (2**19,) * 3
    n = 2000
    coords = [rng.integers(0, 2**19, n) for _ in range(3)]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = rng.random(lin.size)
    T = Tensor.from_coo(coords, vals, shape=shape)
    assert T.layout == Layout((0,), (1, 2))
    for dst in all_layouts(3):
        if not all(math.prod(shape[a] for a in g) <= 2**60 for g in dst):
            continue
        U = T.relayout(dst)
        U._check_invariants()
        ref = Tensor.from_coo(coords, vals, shape=shape, layout=dst)
        assert U.ss.matrix.isequal(ref.ss.matrix), dst
    # The planner with the hypersparse density class and real primitives
    src = T.layout.state
    for dst in all_layouts(3)[::3]:
        if not all(math.prod(shape[a] for a in g) <= 2**60 for g in dst):
            continue
        _, path, _ = tl.plan(shape, src, frozenset([dst.state]), 10**6, 8, hyper1=True)
        M, state = tl.execute(T.ss.matrix, shape, src, path)
        assert state == dst.state
        assert M.isequal(Tensor.from_coo(coords, vals, shape=shape, layout=dst).ss.matrix)


def test_relayout_4d_and_big():
    rng = np.random.default_rng(7)
    shape = (3, 1, 4, 5)
    T, values, present = random_tensor(shape, rng)
    states = all_layouts(4)
    for i in rng.choice(len(states), 40, replace=False):
        src = states[i]
        S = T.relayout(src)
        for j in rng.choice(len(states), 5, replace=False):
            dst = states[j]
            _, path, _ = tl.plan(shape, src.state, frozenset([dst.state]), 10**7, 8)
            M, state = tl.execute(S.ss.matrix, shape, src.state, path)
            assert state == dst.state
            assert M.isequal(_rebuild_reference(T, dst).ss.matrix), (src, dst, path)
        check(S, values, present)
    # Total size past 2**60: no reshape edges; groups past 2**60 are not states (F13)
    shape = (2**19,) * 4
    n = 300
    coords = [rng.integers(0, 2**19, n) for _ in range(4)]
    T = Tensor.from_coo(coords, 1.0, shape=shape)
    assert T.layout == Layout((0,), (1, 2, 3))
    for dst in [Layout((2, 3), (0, 1)), Layout((1, 0), (3, 2)), Layout((0, 1, 2), (3,))]:
        U = T.relayout(dst)
        U._check_invariants()
        assert U.isequal(T)
        assert U.ss.is_iso
    with pytest.raises(ValueError, match="2\\*\\*60"):
        T.relayout(((0, 1, 2, 3), ()))


def test_relayout_6d_rebuilds():
    rng = np.random.default_rng(8)
    T, values, present = random_tensor((2, 3, 1, 2, 2, 3), rng)
    _, path, _ = tl.plan(T.shape, T.layout.state, frozenset([((5, 4, 3, 2, 1, 0), 3)]), 10**7, 8)
    assert [e[0] for e in path] == ["REBUILD"]
    U = T.relayout(((5, 4, 3), (2, 1, 0)))
    check(U, values, present)


@pytest.mark.parametrize("dtype", [dtypes.BOOL, dtypes.INT64, dtypes.FP64, dtypes.FC64])
@pytest.mark.parametrize("iso", [False, True])
def test_relayout_dtypes_iso(dtype, iso):
    rng = np.random.default_rng(9)
    shape = (4, 5, 6)
    present = rng.random(shape) < 0.4
    coords = np.nonzero(present)
    if iso:
        vals = dtype.np_type.type(1)
    else:
        vals = (rng.random(present.sum()) * 10).astype(dtype.np_type)
    T = Tensor.from_coo(coords, vals, dtype, shape=shape)
    assert T.ss.is_iso == iso
    for dst in all_layouts(3):
        # Large pricing exercises PERM_R (P @ M) for every dtype
        _, path, _ = tl.plan(shape, T.layout.state, frozenset([dst.state]), 10**7, 8)
        M, _ = tl.execute(T.ss.matrix, shape, T.layout.state, path)
        assert M.isequal(_rebuild_reference(T, dst).ss.matrix, check_dtype=True)
        assert M.ss.is_iso == iso, (dst, path)
        U = T.relayout(dst)
        assert U.ss.is_iso == iso
        assert U.isequal(T, check_dtype=True)


def test_relayout_fast_path_and_cache():
    rng = np.random.default_rng(10)
    T, values, present = random_tensor((3, 4, 5), rng)
    U = T.relayout(T.layout)
    assert U.ss.matrix is T.ss.matrix
    tl.clear_caches()
    dst = Layout((2,), (1, 0))
    T.relayout(dst)
    n = len(tl._plan_cache.data)
    assert n >= 1
    T.relayout(dst)
    assert len(tl._plan_cache.data) == n
    # inplace
    V = T.dup()
    V.relayout(dst, inplace=True)
    assert V.layout == dst
    assert V.layout_pinned
    check(V, values, present)
    check(T, values, present)
    # Shape-only matrices are cached; empty tensors move without data
    E = Tensor(float, 3, 4, 5)
    F = E.relayout(dst)
    F._check_invariants()
    assert F.nvals == 0


def test_flip_orientation_and_helpers():
    A = Matrix.from_coo([0, 1, 2], [1, 2, 0], [1.0, 2.0, 3.0], nrows=3, ncols=5)
    for fmt in ["sparse", "hypersparse", "bitmap"]:
        B = A.dup()
        B.ss.config["format"] = "by_col"
        B.ss.config["sparsity_control"] = fmt
        tl.flip_orientation(B)
        assert tl.is_by_row(B)
        assert B.shape == (5, 3)
        assert B.isequal(A.T.new())
    H = Matrix.from_coo([0, 2**40], [1, 2], [1.0, 2.0], nrows=2**41, ncols=4)
    assert H.ss.format == "hypercsr"
    assert tl.nonempty_rows(H) == 2
    assert tl.nonempty_rows(A) == 3
    v = Vector.from_coo([1, 3], [1.0, 2.0], size=5)
    M = tl.steal_vector_as_row(v.dup())
    assert M.shape == (1, 5)
    assert tl.is_by_row(M)
    S = tl.selector((2, 3), (1,))
    assert S.shape == (3, 6)
    assert S.isequal(Matrix.from_coo([0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5], True))
    assert tl.selector((2, 3), (1,)) is S
    St = tl.selector((2, 3), (0,), transposed=True)
    assert St.shape == (6, 2)
    # The shape-only cache is bounded by total entries
    lru = tl._LRU(10, maxweight=100)
    lru.put("a", 1, 60)
    lru.put("b", 2, 30)
    assert lru.get("a") == 1
    lru.put("c", 3, 30)  # over the budget: drops "b", the least recently used
    assert lru.get("b") is None
    assert lru.weight == 90
    lru.put("d", 4, 101)  # heavier than the whole budget: not kept
    assert lru.get("d") is None
    lru.put("a", 5, 10)
    assert lru.get("a") == 5
    assert lru.weight == 40


# --------------------------------------------------------------------------------------------
# M3: shape operations and copy-on-write


@pytest.mark.parametrize("layout", all_layouts(3)[::3])
def test_shape_ops(layout):
    rng = np.random.default_rng(11)
    T, values, present = random_tensor((2, 3, 4), rng, layout=layout)
    for axes in itertools.permutations(range(3)):
        P = T.permute_dims(axes)
        assert P.ss.matrix is T.ss.matrix
        check(P, values.transpose(axes), present.transpose(axes))
    check(T.permute_dims((-1, 0, 1)), values.transpose(2, 0, 1), present.transpose(2, 0, 1))
    check(T.T, values.T, present.T)
    check(T.mT, np.swapaxes(values, -1, -2), np.swapaxes(present, -1, -2))
    for axis in [0, 1, 2, 3, -1, -4, (0, 2), (1, -1)]:
        E = T.expand_dims(axis)
        assert E.ss.matrix is T.ss.matrix
        check(E, np.expand_dims(values, axis), np.expand_dims(present, axis))
    E = T.expand_dims((0, 3))
    check(E.squeeze(), values, present)
    check(E.squeeze(0), values[None, ...].reshape(2, 3, 1, 4), present.reshape(2, 3, 1, 4))
    check(E.squeeze((0, 3)), values, present)
    with pytest.raises(ValueError, match="squeeze"):
        T.squeeze(0)
    with pytest.raises(ValueError, match="permutation"):
        T.permute_dims((0, 0, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        T.permute_dims((0, 1, 3))
    with pytest.raises(ValueError, match="repeated"):
        T.expand_dims((0, 0))
    with pytest.raises(TypeError, match="int"):
        T.expand_dims("x")


def test_shape_ops_small_ndim():
    Z = Tensor.from_coo((), 3.0)
    assert Z.T.shape == ()
    with pytest.raises(ValueError, match="mT"):
        Z.mT
    E = Z.expand_dims(0).expand_dims(0)
    assert E.shape == (1, 1)
    assert E.squeeze().to_scalar().value == 3.0
    v = Tensor.from_coo(([1, 3],), [1.0, 2.0], shape=(4,))
    check(v.T, np.array([0, 1, 0, 2.0]), np.array([False, True, False, True]))
    M = v.expand_dims(1)
    assert M.shape == (4, 1)
    M._check_invariants()
    assert M.to_matrix().isequal(Matrix.from_coo([1, 3], [0, 0], [1.0, 2.0], nrows=4, ncols=1))


def test_copy_on_write():
    rng = np.random.default_rng(12)
    T, values, present = random_tensor((2, 3, 4), rng)
    V = T.permute_dims((2, 0, 1))
    vv, vp = values.transpose(2, 0, 1), present.transpose(2, 0, 1)
    assert len(T._store.owners) == 2
    # Mutating in place (mask without replace keeps old values) copies first
    T(T.S, accum=binary.plus) << T
    check(T, values * 2, present)
    check(V, vv, vp)
    assert T.ss.matrix is not V.ss.matrix
    # The view can be written too
    W = V.T
    wv, wp = vv.T, vp.T
    V << V * 3
    check(V, vv * 3, vp)
    check(W, wv, wp)
    # Replacing (<< without mask) detaches without copying
    X = W.mT
    xv, xp = np.swapaxes(wv, -1, -2), np.swapaxes(wp, -1, -2)
    W << W + W
    check(W, wv * 2, wp)
    check(X, xv, xp)
    # clear and relayout(inplace=True)
    Y = X.dup()
    Z = Y.T
    Y.clear()
    assert Y.nvals == 0
    check(Z, xv.T, xp.T)
    Z2 = Z.permute_dims((0, 1, 2))
    Z.relayout(((2,), (0, 1)), inplace=True)
    assert Z2.ss.matrix is not Z.ss.matrix
    check(Z, xv.T, xp.T)
    check(Z2, xv.T, xp.T)
    # A Matrix handed out without copying keeps its values when the Tensor changes
    M = Z2.to_matrix(copy=False)
    M0 = M.dup()
    Z2(Z2.S) << 1.0
    assert M.isequal(M0)
    check(Z2, np.where(xp.T, 1.0, 0), xp.T)
    M2 = Z2.ss.matrix
    M2c = M2.dup()
    Z2(Z2.S, accum=binary.plus) << Z2
    assert M2.isequal(M2c)
    check(Z2, np.where(xp.T, 2.0, 0), xp.T)


def test_copy_on_write_weak_owners():
    rng = np.random.default_rng(13)
    T, values, present = random_tensor((2, 3, 4), rng)
    V = T.T
    del V
    gc.collect()
    assert len(T._store.owners) == 1
    M = T._M
    # No other owner left, so an in-place update does not copy
    T(T.S, accum=binary.plus) << T
    assert T._M is M
    check(T, values * 2, present)
    # Assigning a Tensor shares it
    U = Tensor(float, 2, 3, 4)
    U << T
    assert U.ss.matrix is T.ss.matrix
    U(U.S, accum=binary.times) << U
    check(U, values * values * 4, present)
    check(T, values * 2, present)
    T << T
    check(T, values * 2, present)


# --------------------------------------------------------------------------------------------
# M4: expressions, <<, masks, accumulators


def test_expression_basics():
    rng = np.random.default_rng(14)
    T, values, present = random_tensor((2, 3, 4), rng)
    expr = T + T
    assert type(expr) is TensorExpression
    assert expr.shape == (2, 3, 4)
    assert expr.ndim == 3
    assert expr.size == 24
    assert expr.dtype == dtypes.FP64
    if not gb.config.get("autocompute"):
        with pytest.raises(TypeError, match="new"):
            expr.nvals
    with gb.config.set(autocompute=True):
        e2 = T + T
        assert e2.nvals == present.sum()
        assert e2.layout == T.layout
        check(e2.new(), values * 2, present)
        assert e2.isequal(T * 2)
        assert e2.to_coo()[1].tolist() == (values * 2)[present].tolist()
        check((e2 * 2).new(), values * 4, present)
    R = expr.new(name="R")
    assert R.name == "R"
    check(R, values * 2, present)
    R = expr.new(dtypes.INT64)
    assert R.dtype == dtypes.INT64
    check(R, (values * 2).astype(np.int64), present)
    R = expr.new(layout=((2,), (0, 1)))
    assert R.layout == Layout((2,), (0, 1))
    assert R.layout_pinned
    check(R, values * 2, present)
    R = expr.dup()
    check(R, values * 2, present)


@pytest.mark.parametrize("target_layout", [None, ((2,), (0, 1)), ((0, 1), (2,))])
@pytest.mark.parametrize("mask_layout", [((1,), (2, 0)), ((0, 1), (2,))])
def test_masks_and_accum(target_layout, mask_layout):
    rng = np.random.default_rng(15)
    shape = (3, 4, 5)
    A, av, ap = random_tensor(shape, rng, layout=((2, 0), (1,)))
    B, bv, bp = random_tensor(shape, rng)
    Mk, mv, mp = random_tensor(shape, rng, dtype=dtypes.BOOL, density=0.5, layout=mask_layout)
    zv = np.where(ap, av, 0) + np.where(bp, bv, 0)
    zp = ap | bp
    masks = {
        "S": (Mk.S, mp),
        "V": (Mk.V, mp & mv),
        "~S": (~Mk.S, ~mp),
        "~V": (~Mk.V, ~(mp & mv)),
        "bare": (Mk, mp & mv),
    }
    for (mname, (mask, mpos)), accum, replace in itertools.product(
        masks.items(), [None, binary.plus, binary.minus], [False, True]
    ):
        label = (mname, accum, replace)
        C, cv, cp = random_tensor(shape, rng, layout=target_layout)
        C(mask, accum=accum, replace=replace) << A + B
        np_accum = None if accum is None else {"plus": np.add, "minus": np.subtract}[accum.name]
        ev, ep = model_update(cv, cp, zv, zp, mask=mpos, accum=np_accum, replace=replace)
        check(C, ev, ep, approx=True)
        if target_layout is not None:
            assert C.layout == target_layout, label
    # .new(mask=)
    for mask, mpos in masks.values():
        R = (A + B).new(mask=mask)
        check(R, np.where(mpos & zp, zv, 0), mpos & zp, approx=True)
    # accum without mask
    C, cv, cp = random_tensor(shape, rng, layout=target_layout)
    C(binary.plus) << A + B
    ev, ep = model_update(cv, cp, zv, zp, accum=np.add)
    check(C, ev, ep, approx=True)


def test_update_layout_rules():
    rng = np.random.default_rng(16)
    shape = (3, 4, 5)
    A, av, ap = random_tensor(shape, rng, layout=((2,), (0, 1)))
    # Unpinned target adopts the result's layout; pinned keeps its own (F3)
    C = Tensor(float, shape)
    C << A * 2
    assert C.layout == A.layout
    assert not C.layout_pinned
    P = Tensor(float, shape, layout=((1, 0), (2,)))
    P << A * 2
    assert P.layout == Layout((1, 0), (2,))
    check(P, av * 2, ap)
    # A mask without replace keeps the target's layout (old values survive)
    D = random_tensor(shape, rng)[0]
    D0 = D.layout
    D(A.S) << A * 2
    assert D.layout == D0
    # With replace on an unpinned target, nothing old survives: the smaller side moves
    E = random_tensor(shape, rng)[0]
    E(A.S, replace=True) << A * 2
    check(E, av * 2, ap)
    # Mask shape and type errors
    with pytest.raises(DimensionMismatch):
        C(Tensor(bool, 3, 4, 6).S) << A * 2
    with pytest.raises(TypeError, match="Tensor mask"):
        C(Matrix(bool, 3, 4).S) << A * 2
    with pytest.raises(TypeError, match="boolean"):
        C(A) << A * 2
    with pytest.raises(DimensionMismatch):
        Tensor(float, 3, 4) << A * 2
    with pytest.raises(TypeError, match="from_matrix"):
        C << Matrix(float, 3, 4)
    with pytest.raises(TypeError, match="input_mask"):
        C(input_mask=A.S) << A * 2


def test_assign_scalar_and_tensor():
    rng = np.random.default_rng(17)
    shape = (2, 3, 4)
    A, av, ap = random_tensor(shape, rng)
    C = Tensor(float, shape, layout=((2,), (0, 1)))
    C(A.S) << 7
    check(C, np.where(ap, 7.0, 0), ap)
    C(~A.S) << Scalar.from_value(1.5)
    check(C, np.where(ap, 7.0, 1.5), np.ones(shape, bool))
    C << 2
    check(C, np.full(shape, 2.0), np.ones(shape, bool))
    assert C.layout == Layout((2,), (0, 1))
    C(binary.plus) << 1
    check(C, np.full(shape, 3.0), np.ones(shape, bool))
    # Tensor of another dtype casts
    Ti = Tensor(int, shape)
    Ti << A
    assert Ti.dtype == dtypes.INT64
    check(Ti, av.astype(np.int64), ap)
    # A pinned target relays the assigned Tensor
    P = Tensor(float, shape, layout=((1,), (2, 0)))
    P << A
    assert P.layout == Layout((1,), (2, 0))
    check(P, av, ap)
    P(A.S, binary.plus) << A
    check(P, av * 2, ap)


def test_in_place_infix():
    rng = np.random.default_rng(18)
    A, av, ap = random_tensor((2, 3, 4), rng)
    B, bv, bp = random_tensor((2, 3, 4), rng, layout=((1, 2), (0,)))
    C = A.dup()
    C += B
    check(C, np.where(ap, av, 0) + np.where(bp, bv, 0), ap | bp, approx=True)
    C = A.dup()
    C *= B
    check(C, np.where(ap & bp, av * bv, 0), ap & bp, approx=True)
    C = A.dup()
    C -= 1
    check(C, np.where(ap, av - 1, 0), ap, approx=True)
    with pytest.raises(DimensionMismatch):
        C += Matrix(float, 2, 3)
    C += Matrix(float, 3, 4)  # a Matrix operand broadcasts over the leading axis
    check(C, np.where(ap, av - 1, 0), ap, approx=True)
    expr = A + B
    with pytest.raises(TypeError, match="new"):
        expr += B
    with pytest.raises(TypeError):
        A + "x"


# --------------------------------------------------------------------------------------------
# M5: element-wise, same shape

EWISE_OPS = {
    "plus": (binary.plus, np.add),
    "times": (binary.times, np.multiply),
    "minus": (binary.minus, np.subtract),
    "first": (binary.first, lambda x, y: x),
    "second": (binary.second, lambda x, y: y),
    "truediv": (binary.truediv, np.true_divide),
    "lor": (binary.lor, np.logical_or),
}


def _np_op(name, x, y, dtype):
    if name == "truediv":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.true_divide(x, y)
    if name == "lor":
        return np.logical_or(x != 0, y != 0)
    if dtype == dtypes.BOOL and name in {"plus", "times", "minus"}:
        return {"plus": np.logical_or, "times": np.logical_and, "minus": np.not_equal}[name](x, y)
    return EWISE_OPS[name][1](x, y)


@pytest.mark.parametrize("dtype", [dtypes.FP64, dtypes.INT64, dtypes.BOOL])
@pytest.mark.parametrize("opname", list(EWISE_OPS))
def test_ewise(dtype, opname):
    rng = np.random.default_rng(19)
    shape = (3, 4, 5)
    op = EWISE_OPS[opname][0]
    if dtype == dtypes.BOOL and opname == "truediv":
        return
    layouts = [((0, 1), (2,)), ((2,), (1, 0)), ((1,), (0, 2))]
    for la, lb in [(layouts[0], layouts[0]), (layouts[0], layouts[1]), (layouts[2], layouts[0])]:
        A, av, ap = random_tensor(shape, rng, dtype=dtype, layout=la)
        B, bv, bp = random_tensor(shape, rng, dtype=dtype, layout=lb)
        for (X, xv, xp), (Y, yv, yp) in [((A, av, ap), (B, bv, bp)), ((B, bv, bp), (A, av, ap))]:
            both = xp & yp
            with np.errstate(divide="ignore", invalid="ignore"):
                zv = np.where(both, _np_op(opname, xv, yv, dtype), 0)
            # ewise_mult: intersection
            R = X.ewise_mult(Y, op).new()
            ev = zv.astype(R.dtype.np_type)
            check(R, np.where(both, ev, 0), both, approx=R.dtype != dtypes.BOOL)
            # ewise_add: union, the present side passes through
            R = X.ewise_add(Y, op).new()
            uv = np.where(both, zv, np.where(xp, xv, yv)).astype(R.dtype.np_type)
            check(R, np.where(xp | yp, uv, 0), xp | yp, approx=R.dtype != dtypes.BOOL)
            if dtype != dtypes.BOOL:
                # ewise_union with defaults 0 (-1 on the right)
                R = X.ewise_union(Y, op, 0, -1).new()
                xx = np.where(xp, xv, 0)
                yy = np.where(yp, yv, -1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    wv = _np_op(opname, xx, yy, dtype)
                check(R, np.where(xp | yp, wv, 0).astype(R.dtype.np_type), xp | yp, approx=True)


def test_ewise_infix_and_scalars():
    rng = np.random.default_rng(20)
    shape = (2, 3, 4)
    A, av, ap = random_tensor(shape, rng, layout=((2,), (0, 1)))
    B, bv, bp = random_tensor(shape, rng)
    a0, b0 = np.where(ap, av, 0), np.where(bp, bv, 0)
    check((A + B).new(), a0 + b0, ap | bp, approx=True)
    check((A - B).new(), a0 - b0, ap | bp, approx=True)
    check((A * B).new(), np.where(ap & bp, av * bv, 0), ap & bp, approx=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.where(ap & bp, av / bv, 0)
    R = (A / B).new()
    np.testing.assert_allclose(R.to_coo()[1], dv[ap & bp])
    check((A * 2).new(), np.where(ap, av * 2, 0), ap)
    check((2 * A).new(), np.where(ap, av * 2, 0), ap)
    check((A - 1).new(), np.where(ap, av - 1, 0), ap, approx=True)
    check((1 - A).new(), np.where(ap, 1 - av, 0), ap, approx=True)
    check((A + Scalar.from_value(1.0)).new(), np.where(ap, av + 1, 0), ap, approx=True)
    check(A.apply(gb.unary.abs).new(), np.abs(a0), ap)
    check(A.apply(binary.minus, left=10).new(), np.where(ap, 10 - av, 0), ap, approx=True)
    with pytest.raises(DimensionMismatch, match="broadcast"):
        A + Tensor(float, 2, 3)
    with pytest.raises(DimensionMismatch, match="broadcast"):
        A.ewise_add(Matrix(float, 2, 3))
    with pytest.raises(TypeError):
        A.ewise_add("x")
    with pytest.raises(TypeError):
        A.ewise_add(B, gb.semiring.plus_times)
    if not gb.config.get("autocompute"):
        with pytest.raises(TypeError, match="autocompute"):
            (A + B) + A
    # An operand that is the target itself
    C = A.dup()
    C << C + B
    check(C, a0 + b0, ap | bp, approx=True)
    C = A.dup()
    C(C.S) << C * C
    check(C, np.where(ap, av * av, 0), ap, approx=True)


@pytest.mark.parametrize("opname", ["times", "first", "plus", "second"])
def test_ewise_iso(opname):
    """Iso inputs give iso results exactly when the same Matrix op does (F7)."""
    rng = np.random.default_rng(21)
    shape = (3, 4, 5)
    op = EWISE_OPS[opname][0]
    A = random_tensor(shape, rng, iso=True)[0]
    B = random_tensor(shape, rng, iso=True, layout=((2,), (0, 1)))[0]
    assert A.ss.is_iso
    assert B.ss.is_iso
    for method in ["ewise_mult", "ewise_add"]:
        R = getattr(A, method)(B, op).new(layout=A.layout)
        ref = getattr(A.ss.matrix, method)(B.relayout(A.layout).ss.matrix, op).new()
        assert R.ss.is_iso == ref.ss.is_iso, method
        assert R.ss.matrix.isequal(ref)
    S = (A * 3).new()
    assert S.ss.is_iso


# --------------------------------------------------------------------------------------------
# M6: reductions

NP_REDUCE = {
    "plus": (np.add, 0),
    "times": (np.multiply, 1),
    "max": (np.maximum, -np.inf),
    "min": (np.minimum, np.inf),
}


def np_reduce(name, values, present, axes, keepdims=False):
    ufunc, ident = NP_REDUCE[name]
    vals = np.where(present, values.astype(float), ident)
    out = ufunc.reduce(vals, axis=axes, keepdims=keepdims)
    pattern = np.logical_or.reduce(present, axis=axes, keepdims=keepdims)
    return np.where(pattern, out, 0), pattern


AXES_3D = [0, 1, 2, (0, 1), (0, 2), (1, 2), (2, 0), None, -1, ()]


@pytest.mark.parametrize("layout", all_layouts(3))
def test_reduce_all_layouts(layout):
    rng = np.random.default_rng(22)
    shape = (3, 4, 5)
    T, values, present = random_tensor(shape, rng, layout=layout, density=0.2)
    for axis in AXES_3D:
        axes = tuple(range(3)) if axis is None else axis
        for keepdims in [False, True]:
            R = T.reduce(monoid.plus, axis, keepdims=keepdims).new()
            ev, ep = np_reduce("plus", values, present, axes, keepdims)
            check(R, ev, ep, approx=True)
    for name in ["max", "min", "times"]:
        R = T.reduce(getattr(monoid, name), axis=(0, 2)).new()
        ev, ep = np_reduce(name, values, present, (0, 2))
        check(R, ev, ep, approx=True)


@pytest.mark.parametrize("dtype", [dtypes.INT64, dtypes.BOOL])
def test_reduce_dtypes(dtype):
    rng = np.random.default_rng(23)
    shape = (2, 3, 4)
    T, values, present = random_tensor(shape, rng, dtype=dtype, layout=((1,), (2, 0)))
    for axis in [0, 1, 2, (0, 2), None]:
        axes = tuple(range(3)) if axis is None else axis
        if dtype == dtypes.BOOL:
            R = T.any(axis).new()
            ep = np.logical_or.reduce(present, axis=axes)
            ev = np.logical_or.reduce(present & values, axis=axes)
            check(R, ev & ep, ep)
            R = T.all(axis).new()
            ev = np.logical_and.reduce(~present | values, axis=axes)
            check(R, ev & ep, ep)
            R = T.reduce(monoid.lor, axis).new()
            assert R.dtype == dtypes.BOOL
        else:
            for name, meth in [("plus", "sum"), ("times", "prod"), ("max", "max"), ("min", "min")]:
                R = getattr(T, meth)(axis).new()
                assert R.dtype == dtypes.INT64
                ev, ep = np_reduce(name, values, present, axes)
                check(R, ev.astype(np.int64), ep)
            R = T.any(axis).new()
            assert R.dtype == dtypes.BOOL
            ep = np.logical_or.reduce(present, axis=axes)
            ev = np.logical_or.reduce(present & (values != 0), axis=axes)
            check(R, ev & ep, ep)


def test_reduce_scalar():
    rng = np.random.default_rng(24)
    T, values, present = random_tensor((3, 4, 5), rng, layout=((2,), (0, 1)))
    s = T.reduce_scalar().new()
    assert s.value == pytest.approx(values[present].sum())
    s = T.reduce_scalar(monoid.max).new()
    assert s.value == values[present].max()
    s = Scalar(float)
    s << T.reduce_scalar(monoid.min)
    assert s.value == values[present].min()
    E = Tensor(float, 2, 3)
    assert E.reduce_scalar().new().is_empty
    assert E.reduce_scalar(allow_empty=False).new().value == 0
    Z = T.sum().new()
    assert Z.shape == ()
    assert Z.to_scalar().value == pytest.approx(values[present].sum())
    Z = E.sum().new()
    assert Z.nvals == 0
    with pytest.raises(TypeError):
        T.reduce(binary.minus)
    with pytest.raises(ValueError, match="repeated"):
        T.reduce(axis=(0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        T.reduce(axis=3)
    assert "reduce" in repr(T.reduce(axis=1))


def test_reduce_4d():
    rng = np.random.default_rng(25)
    shape = (2, 3, 1, 4)
    for layout in [((0, 1), (2, 3)), ((3, 1), (0, 2)), ((2,), (3, 0, 1)), ((0, 1, 2, 3), ())]:
        T, values, present = random_tensor(shape, rng, layout=layout)
        for axis in [0, 1, 3, (0, 1), (1, 3), (0, 2, 3), (0, 1, 3)]:
            R = T.sum(axis).new()
            ev, ep = np_reduce("plus", values, present, axis)
            check(R, ev, ep, approx=True)


def test_reduce_strategies(monkeypatch):
    rng = np.random.default_rng(26)
    # Wide columnwise reduction: with enough data one column axis moves into the rows for a
    # selector; with little, a single-threaded columnwise reduce
    shape = (2, 400, 300)
    for density in [0.05, 0.01]:
        T, values, present = random_tensor(shape, rng, layout=((0,), (1, 2)), density=density)
        R = T.sum(0).new(nthreads=4)
        check(R, *np_reduce("plus", values, present, 0), approx=True)
        R = T.sum(0).new(nthreads=1)
        check(R, *np_reduce("plus", values, present, 0), approx=True)
    # Hypersparse rows with about one entry each: REBUILD with the monoid as dup_op
    monkeypatch.setattr(tl, "density", lambda M: True)
    T, values, present = random_tensor((3, 4, 5), rng, layout=((0, 1), (2,)))
    for axis in [0, 1, 2, (0, 1), (1, 2), (0, 2)]:
        R = T.reduce(monoid.max, axis).new()
        check(R, *np_reduce("max", values, present, axis))
    monkeypatch.undo()
    # Groups much larger than the data use REBUILD too
    shape = (2**20, 2**20, 3)
    coords = [rng.integers(0, 2**20, 50), rng.integers(0, 2**20, 50), rng.integers(0, 3, 50)]
    T = Tensor.from_coo(coords, np.ones(50), shape=shape, layout=((2,), (0, 1)), dup_op=binary.plus)
    R = T.sum(0).new()
    lin = np.ravel_multi_index((coords[1], coords[2]), (2**20, 3))
    assert R.nvals == np.unique(lin).size
    assert R.to_coo()[1].sum() == pytest.approx(50.0)
    R._check_invariants()
    R = T.sum((0, 1)).new()
    assert R.to_coo()[1].sum() == pytest.approx(50.0)


def test_reduce_target_layout():
    rng = np.random.default_rng(27)
    T, values, present = random_tensor((3, 4, 5), rng)
    C = Tensor(float, 3, 5, layout=((1,), (0,)))
    C << T.sum(1)
    assert C.layout == Layout((1,), (0,))
    check(C, *np_reduce("plus", values, present, 1), approx=True)
    D, dv, dp = random_tensor((3, 5), rng)
    D(binary.plus) << T.sum(1)
    ev, ep = np_reduce("plus", values, present, 1)
    check(D, np.where(dp, dv, 0) + ev, dp | ep, approx=True)
    # A result is not left as an n x 1 Matrix (review F4); one row is fine
    R = T.sum(2, keepdims=True).new()
    assert R.layout.cols
    check(R, *np_reduce("plus", values, present, 2, keepdims=True), approx=True)
    U = T.relayout(((0, 1, 2), ()))
    R = U.sum(1, keepdims=True).new()
    assert R.layout.cols
    check(R, *np_reduce("plus", values, present, 1, keepdims=True), approx=True)
    R = U.sum((0, 2), keepdims=True).new()
    assert R.layout.cols
    check(R, *np_reduce("plus", values, present, (0, 2), keepdims=True), approx=True)


def test_udt_relayout():
    """UDT data moves by reshape, transpose and REBUILD only (no P @ M semiring)."""
    udt = dtypes.register_anonymous(np.dtype([("tensor_a", "<i4"), ("tensor_b", "<f8")]))
    rng = np.random.default_rng(28)
    shape = (3, 4, 5)
    present = rng.random(shape) < 0.3
    coords = np.nonzero(present)
    vals = np.zeros(present.sum(), dtype=udt.np_type)
    vals["tensor_a"] = np.arange(vals.size)
    vals["tensor_b"] = rng.random(vals.size)
    T = Tensor.from_coo(coords, vals, udt, shape=shape)
    assert T.dtype == udt
    for dst in all_layouts(3):
        U = T.relayout(dst)
        U._check_invariants()
        got_coords, got = U.to_coo()
        for a, b in zip(got_coords, coords, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(got, vals)
    V = T.permute_dims((2, 1, 0))
    got_coords, got = V.to_coo(sort=False)
    assert got.size == vals.size


@pytest.mark.parametrize("layout", [((0, 1), (2,)), ((2,), (0, 1)), ((1,), (2, 0))])
def test_size_zero_axes(layout):
    rng = np.random.default_rng(29)
    shape = (3, 0, 4)
    T, values, present = random_tensor(shape, rng, layout=layout)
    assert T.nvals == 0
    for dst in all_layouts(3)[::4]:
        check(T.relayout(dst), values, present)
    check((T + T).new(), values, present)
    for axis in [0, 1, 2, (0, 2), None]:
        R = T.sum(axis).new()
        ev, ep = np_reduce("plus", values, present, axis if axis is not None else (0, 1, 2))
        check(R, ev, ep)


# --------------------------------------------------------------------------------------------
# M7: tensordot


def np_tensordot(xv, xp, yv, yp, axes):
    zv = np.tensordot(np.where(xp, xv, 0), np.where(yp, yv, 0), axes)
    zp = np.tensordot(xp.astype(np.int64), yp.astype(np.int64), axes) > 0
    return np.where(zp, zv, 0), zp


@pytest.mark.parametrize(
    ("xshape", "yshape", "axes"),
    [
        ((3, 4, 5), (5, 6), 1),
        ((3, 4, 5), (4, 5, 2), 2),
        ((3, 4, 5), (5, 6, 3), ([0, 2], [2, 0])),
        ((3, 4, 5), (6, 4), ([-2], [-1])),
        ((2, 3), (4,), 0),
        ((3, 4, 5), (3, 4, 5), 3),
        ((2, 3, 4, 5), (5, 4, 3), ([1, 2, 3], [2, 1, 0])),
        ((4, 5), (5,), 1),
        ((5,), (5, 4, 3), 1),
    ],
)
def test_tensordot(xshape, yshape, axes):
    rng = np.random.default_rng(30)
    nx, ny = len(xshape), len(yshape)
    for lx, ly in [(None, None), (all_layouts(nx)[-1], all_layouts(ny)[1])]:
        X, xv, xp = random_tensor(xshape, rng, layout=lx, density=0.4)
        Y, yv, yp = random_tensor(yshape, rng, layout=ly, density=0.4)
        R = X.tensordot(Y, axes).new()
        ev, ep = np_tensordot(xv, xp, yv, yp, axes)
        check(R, ev, ep, approx=True)


def test_tensordot_semirings_masks_and_errors():
    rng = np.random.default_rng(31)
    X, xv, xp = random_tensor((3, 4, 5), rng, layout=((2,), (0, 1)), density=0.5)
    Y, yv, yp = random_tensor((5, 6), rng, density=0.5)
    # min_plus by brute force
    R = X.tensordot(Y, 1, op=gb.semiring.min_plus).new()
    big = np.where(xp, xv, np.inf)[..., None] + np.where(yp, yv, np.inf)[None, None]
    ev = big.min(axis=2)
    ep = np.isfinite(ev)
    check(R, np.where(ep, ev, 0), ep, approx=True)
    # mask, accum and a pinned target
    ev, ep = np_tensordot(xv, xp, yv, yp, 1)
    C, cv, cp = random_tensor((3, 4, 6), rng, layout=((2, 1), (0,)))
    Mk, _, mp = random_tensor((3, 4, 6), rng, dtype=dtypes.BOOL, density=0.5)
    C(Mk.S, binary.plus) << X.tensordot(Y, 1)
    rv, rp = model_update(cv, cp, ev, ep, mask=mp, accum=np.add)
    check(C, rv, rp, approx=True)
    assert C.layout == Layout((2, 1), (0,))
    assert "tensordot" in repr(X.tensordot(Y, 1))
    with pytest.raises(DimensionMismatch, match="contract"):
        X.tensordot(Y, ([0], [0]))
    with pytest.raises(ValueError, match="axes"):
        X.tensordot(Y, 4)
    with pytest.raises(ValueError, match="same number"):
        X.tensordot(Y, ([0, 1], [0]))
    with pytest.raises(ValueError, match="repeated"):
        X.tensordot(Y, ([2, 2], [0, 0]))
    with pytest.raises(ValueError, match="pair"):
        X.tensordot(Y, ([2], [0], [1]))
    with pytest.raises(TypeError):
        X.tensordot(Y, 1, op=binary.plus)


# --------------------------------------------------------------------------------------------
# Phase 2a.1: core hooks (Matrix, Vector, Scalar and operators meet a Tensor)


def test_core_hooks():
    rng = np.random.default_rng(40)
    X, xv, xp = random_tensor((3, 4, 5), rng)
    A = Matrix.from_coo([0, 1, 3], [1, 2, 4], [1.0, 2.0, 3.0], nrows=4, ncols=5)
    av, ap = A.to_dense(0.0), A.to_dense(0.0) != 0
    # Infix with a Matrix on the left defers to Tensor rules (broadcast over the first axis)
    R = (A + X).new()
    assert type(R) is Tensor
    check(R, *np_bcast("add", av, ap, xv, xp, np.add), approx=True)
    check((A * X).new(), *np_bcast("mult", av, ap, xv, xp, np.multiply), approx=True)
    check((A - X).new(), *np_bcast("union", av, ap, xv, xp, np.subtract), approx=True)
    check((A < X).new(), *np_bcast("mult", av, ap, xv, xp, np.less))
    check((X >= A).new(), *np_bcast("mult", xv, xp, av, ap, np.greater_equal))
    check(binary.plus(A | X).new(), *np_bcast("add", av, ap, xv, xp, np.add), approx=True)
    check(binary.minus(X & A).new(), *np_bcast("mult", xv, xp, av, ap, np.subtract))
    v = Vector.from_coo([0, 2], [2.0, 3.0], size=5)
    vv, vp = v.to_dense(0.0), v.to_dense(0.0) != 0
    check((v * X).new(), *np_bcast("mult", vv, vp, xv, xp, np.multiply), approx=True)
    check((X / v).new(), *np_bcast("mult", xv, xp, vv, vp, np.true_divide), approx=True)
    # In-place Matrix ops and Matrix methods keep Matrix rules and raise clearly
    with pytest.raises(TypeError, match="(?i)in-place"):
        A += X
    with pytest.raises(TypeError, match="(?i)in-place"):
        A |= X
    with pytest.raises(TypeError, match="Tensor mask"):
        A(X.S) << A
    with pytest.raises(TypeError, match="Matrix methods do not take Tensor"):
        A.ewise_add(X)
    with pytest.raises(TypeError, match="cannot be assigned"):
        A << X
    with pytest.raises(TypeError):
        Scalar.from_value(1.0) & X
    # Named operators on a Tensor
    check(gb.unary.abs(X).new(), np.abs(xv), xp)
    check(binary.plus(X, 1).new(), np.where(xp, xv + 1, 0), xp, approx=True)
    check(binary.minus(1, X).new(), np.where(xp, 1 - xv, 0), xp, approx=True)
    assert gb.agg.sum(X).new().value == pytest.approx(xv[xp].sum())
    with pytest.raises(ValueError, match="reduce"):
        gb.agg.sum(X, rowwise=True)
    check(gb.select.valuegt(X, 0).new(), np.where(xp & (xv > 0), xv, 0), xp & (xv > 0))
    with pytest.raises(NotImplementedError, match="value"):
        gb.select.tril(X)
    assert "infix" in repr(X | X)
    assert (X | X).shape == X.shape


# --------------------------------------------------------------------------------------------
# Phase 2a.2 / 2a.3: Matrix, Vector and ndarray operands; broadcasting


def np_bcast(kind, xv, xp, yv, yp, ufunc, left_default=0, right_default=0):
    """Numpy broadcasting with presence: mult = intersection, add = union passing values
    through, union = union with defaults.
    """
    xv, yv = np.broadcast_arrays(xv, yv)
    xp, yp = np.broadcast_arrays(xp, yp)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "mult":
            present = xp & yp
            vals = ufunc(xv, yv)
        elif kind == "add":
            present = xp | yp
            vals = np.where(xp & yp, ufunc(xv, yv), np.where(xp, xv, yv))
        else:
            present = xp | yp
            vals = ufunc(np.where(xp, xv, left_default), np.where(yp, yv, right_default))
    return np.where(present, vals, 0), present


BCAST_SHAPES = [
    ((3, 4, 5), (5,)),
    ((3, 4, 5), (4, 5)),
    ((3, 4, 5), (4, 1)),
    ((3, 4, 5), (3, 1, 1)),
    ((3, 4, 5), (3, 1, 5)),
    ((3, 4, 5), (1, 1, 1)),
    ((3, 1, 5), (1, 4, 1)),
    ((3, 1, 5), (4, 1)),
    ((2, 1, 3, 1), (4, 1, 5)),
    ((2, 3, 4, 5), (3, 1, 5)),
]


@pytest.mark.parametrize(("xshape", "yshape"), BCAST_SHAPES)
def test_broadcast(xshape, yshape):
    rng = np.random.default_rng(41)
    ops = [
        (binary.times, np.multiply),
        (binary.minus, np.subtract),
        (binary.lt, np.less),
        (binary.first, lambda x, y: x),
        (binary.truediv, np.true_divide),
    ]
    xlayouts = all_layouts(len(xshape))[:: max(1, len(all_layouts(len(xshape))) // 5)]
    for i, lx in enumerate(xlayouts):
        X, xv, xp = random_tensor(xshape, rng, layout=lx, density=0.5)
        Y, yv, yp = random_tensor(yshape, rng, density=0.6)
        for op, ufunc in ops:
            for (A, av, ap), (B, bv, bp) in [
                ((X, xv, xp), (Y, yv, yp)),
                ((Y, yv, yp), (X, xv, xp)),
            ]:
                R = A.ewise_mult(B, op).new()
                check(R, *np_bcast("mult", av, ap, bv, bp, ufunc), approx=R.dtype != dtypes.BOOL)
                if i == 0 and op in (binary.times, binary.minus):
                    R = A.ewise_add(B, op).new()
                    check(R, *np_bcast("add", av, ap, bv, bp, ufunc), approx=True)
                    R = A.ewise_union(B, op, 1, -1).new()
                    check(R, *np_bcast("union", av, ap, bv, bp, ufunc, 1, -1), approx=True)
        # infix and comparisons
        check((X * Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.multiply), approx=True)
        check((Y + X).new(), *np_bcast("add", yv, yp, xv, xp, np.add), approx=True)
        check((X == Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.equal))
        check((Y != X).new(), *np_bcast("mult", yv, yp, xv, xp, np.not_equal))
        check((X > Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.greater))


def test_broadcast_operands_and_dtypes():
    rng = np.random.default_rng(42)
    X, xv, xp = random_tensor((3, 4, 5), rng, dtype=dtypes.INT64)
    B, bv, bp = random_tensor((4, 5), rng, dtype=dtypes.BOOL, density=0.7)
    check((X * B).new(), *np_bcast("mult", xv, xp, bv.astype(np.int64), bp, np.multiply))
    check(X.ewise_mult(B, binary.first).new(), *np_bcast("mult", xv, xp, bv, bp, lambda x, y: x))
    # ndarray operands are full Tensors
    d = rng.random((4, 5))
    check(
        (X * d).new(), *np_bcast("mult", xv, xp, d, np.ones_like(d, bool), np.multiply), approx=True
    )
    check(
        (d - X).new(),
        *np_bcast("union", d, np.ones_like(d, bool), xv, xp, np.subtract),
        approx=True,
    )
    check(
        X.ewise_mult(d[0]).new(),
        *np_bcast("mult", xv, xp, d[0], np.ones(5, bool), np.multiply),
        approx=True,
    )
    # an empty broadcast operand, a one-value operand, iso
    E = Tensor(float, 4, 1)
    assert (X * E).new().nvals == 0
    one = Tensor.from_coo(([0], [0], [0]), [2.0], shape=(1, 1, 1))
    check((X * one).new(), np.where(xp, xv * 2.0, 0), xp)
    Xi, xiv, xip = random_tensor((3, 4, 5), rng, iso=True)
    Vi = Tensor.from_coo(([0, 3],), 1.0, shape=(5,))
    R = (Xi * Vi).new()
    check(
        R,
        *np_bcast(
            "mult",
            xiv,
            xip,
            np.array([1.0, 0, 0, 1, 0]),
            np.array([1, 0, 0, 1, 0], bool),
            np.multiply,
        ),
    )
    with pytest.raises(DimensionMismatch, match="broadcast"):
        X + Tensor(float, 3, 4)


def test_broadcast_guard():
    rng = np.random.default_rng(43)
    X, xv, xp = random_tensor((3, 4, 5), rng)
    v = Tensor.from_coo(([0, 2],), [1.0, 2.0], shape=(5,))
    with gb.config.set(tensor_max_expansion=10):
        with pytest.raises(ValueError, match="replicate 24 entries"):
            X + v
        with pytest.raises(ValueError, match="allow_expansion"):
            X.ewise_union(v, binary.minus, 0, 0)
        R = X.ewise_add(v, allow_expansion=True).new()
        check(
            R,
            *np_bcast(
                "add", xv, xp, np.array([1.0, 0, 2, 0, 0]), np.array([1, 0, 1, 0, 0], bool), np.add
            ),
        )
        # the intersection with a full operand never replicates
        check(
            (X * v).new(),
            *np_bcast(
                "mult",
                xv,
                xp,
                np.array([1.0, 0, 2, 0, 0]),
                np.array([1, 0, 1, 0, 0], bool),
                np.multiply,
            ),
        )
    # both broadcasting: the cheaper side (b, 2 entries x 3) is replicated
    a = Tensor.from_coo(([0, 1, 2], [0, 0, 0]), 1.0, shape=(3, 1))
    b = Tensor.from_coo(([0, 0], [1, 3]), 1.0, shape=(1, 5))
    with gb.config.set(tensor_max_expansion=5), pytest.raises(ValueError, match="replicate 6"):
        a * b
    R = (a * b).new()
    assert R.shape == (3, 5)
    assert R.nvals == 6


@pytest.mark.parametrize("layout", all_layouts(3)[::2])
def test_broadcast_mult_strategies(layout, monkeypatch):
    """Exact group, part of a group (expanded through a selector) and relayout paths."""
    rng = np.random.default_rng(44)
    X, xv, xp = random_tensor((3, 4, 5), rng, layout=layout, density=0.5)
    for yshape in [(5,), (4, 1), (3, 1, 1), (4, 5), (3, 1, 5), (3, 4, 1)]:
        Y, yv, yp = random_tensor(yshape, rng, density=0.7)
        for op, ufunc in [(binary.minus, np.subtract), (gb.binary.atan2, np.arctan2)]:
            check(X.ewise_mult(Y, op).new(), *np_bcast("mult", xv, xp, yv, yp, ufunc), approx=True)
            check(Y.ewise_mult(X, op).new(), *np_bcast("mult", yv, yp, xv, xp, ufunc), approx=True)
    # hypersparse rows: no selector expansion
    monkeypatch.setattr(tl, "density", lambda M: True)
    Y, yv, yp = random_tensor((4, 1), rng, density=0.7)
    check((X * Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.multiply), approx=True)


# --------------------------------------------------------------------------------------------
# Phase 2a.4: matmul


def np_matmul(xv, xp, yv, yp):
    v = np.matmul(xv, yv)
    p = np.matmul(xp.astype(np.int64), yp.astype(np.int64)) > 0
    return np.where(p, v, 0), p


MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((4,), (4,)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (4, 5)),
    ((3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (5, 4, 6)),
    ((2, 3, 4), (4,)),
    ((5,), (2, 5, 3)),
    ((7, 2, 3), (7, 3, 2)),
    ((2, 3, 2, 4), (3, 4, 2)),
]


@pytest.mark.parametrize(("xshape", "yshape"), MATMUL_SHAPES)
@pytest.mark.parametrize("nthreads", [1, 8])
def test_matmul(xshape, yshape, nthreads):
    rng = np.random.default_rng(45)
    for lx in [None, all_layouts(len(xshape))[-1]]:
        X, xv, xp = random_tensor(xshape, rng, layout=lx, density=0.5)
        Y, yv, yp = random_tensor(yshape, rng, density=0.5)
        R = X.matmul(Y).new(nthreads=nthreads)
        check(R, *np_matmul(xv, xp, yv, yp), approx=True)
        check((X @ Y).new(nthreads=nthreads), *np_matmul(xv, xp, yv, yp), approx=True)


def test_matmul_semirings_operands_and_errors(monkeypatch):
    rng = np.random.default_rng(46)
    X, xv, xp = random_tensor((3, 2, 4), rng, density=0.6)
    Y, yv, yp = random_tensor((3, 4, 5), rng, density=0.6)
    # min_plus by brute force
    R = gb.semiring.min_plus(X @ Y).new()
    big = np.where(xp, xv, np.inf)[..., :, None] + np.where(yp, yv, np.inf)[:, None, :, :]
    ev = big.min(axis=2)
    check(R, np.where(np.isfinite(ev), ev, 0), np.isfinite(ev), approx=True)
    # Matrix and ndarray operands on either side
    A = Matrix.from_coo([0, 1, 3], [1, 0, 4], [1.0, 2.0, 3.0], nrows=4, ncols=5)
    av, ap = A.to_dense(0.0), A.to_dense(0.0) != 0
    check((X @ A).new(), *np_matmul(xv, xp, av, ap), approx=True)
    B = Matrix.from_coo([0, 1], [1, 0], [1.0, 2.0], nrows=2, ncols=2)
    bv, bp = B.to_dense(0.0), B.to_dense(0.0) != 0
    check((B @ X).new(), *np_matmul(bv, bp, xv, xp), approx=True)
    d = rng.random((4, 3))
    check((X @ d).new(), *np_matmul(xv, xp, d, np.ones_like(d, bool)), approx=True)
    # The embedding falls back to a rebuild when it cannot change the indices in place
    monkeypatch.setattr(tl, "embed_block_diagonal", lambda *args: False)
    check(X.matmul(Y).new(nthreads=8), *np_matmul(xv, xp, yv, yp), approx=True)
    monkeypatch.undo()
    with pytest.raises(DimensionMismatch, match="last axis"):
        X @ Tensor(float, 5, 2)
    with pytest.raises(DimensionMismatch, match="batch"):
        X @ Tensor(float, 2, 4, 2)
    with pytest.raises(ValueError, match="0-d"):
        X.matmul(Tensor(float))
    with pytest.raises(TypeError):
        X.matmul(Y, binary.plus)
    assert "matmul" in repr(X.matmul(Y))


def test_embed_block_diagonal():
    rng = np.random.default_rng(47)
    for rows_per_block, nblocks, K, fmt in [
        (3, 4, 5, "sparse"),
        (100, 10, 50, "hypersparse"),
        (3, 4, 2**20, "hypersparse"),
    ]:
        n = 40
        r = rng.integers(0, rows_per_block * nblocks, n)
        c = rng.integers(0, K, n)
        lin = np.unique(r.astype(np.int64) * K + c)
        r, c = lin // K, lin % K
        M = Matrix.from_coo(r, c, np.arange(r.size, dtype=float), nrows=r.max() + 1, ncols=K)
        M.resize(rows_per_block * nblocks, K)
        M.ss.config["sparsity_control"] = fmt
        assert tl.embed_block_diagonal(M, rows_per_block, nblocks)
        ref = Matrix.from_coo(
            r,
            c + (r // rows_per_block) * K,
            np.arange(r.size, dtype=float),
            nrows=rows_per_block * nblocks,
            ncols=nblocks * K,
        )
        assert M.isequal(ref)
    # bitmap storage has no column index array
    M = Matrix.from_dense(np.ones((4, 3)))
    assert not tl.embed_block_diagonal(M, 2, 2)
    E = Matrix(float, 4, 3)
    E.ss.config["sparsity_control"] = "sparse"
    assert tl.embed_block_diagonal(E, 2, 2)
    assert E.shape == (4, 6)


# --------------------------------------------------------------------------------------------
# Phase 2a.5: einsum, mttkrp, unfold, TTV


def np_einsum(sub, *models):
    vals = [m[1] for m in models]
    patterns = [m[2].astype(np.int64) for m in models]
    p = np.einsum(sub, *patterns) > 0
    return np.where(p, np.einsum(sub, *vals), 0), p


EINSUM_CASES = [
    ("ijk,jr,kr->ir", [(3, 4, 5), (4, 2), (5, 2)]),
    ("ijk,ir,kr->jr", [(3, 4, 5), (3, 2), (5, 2)]),
    ("ijk,ir,jr->kr", [(3, 4, 5), (3, 2), (4, 2)]),
    ("ijk,jr,kr->ri", [(3, 4, 5), (4, 2), (5, 2)]),
    ("ijk,rj,kr->ir", [(3, 4, 5), (2, 4), (5, 2)]),
    ("ijkl,jr,kr,lr->ir", [(2, 3, 4, 5), (3, 2), (4, 2), (5, 2)]),
    ("ijkl,ir,jr,kr->lr", [(2, 3, 4, 5), (2, 2), (3, 2), (4, 2)]),
    ("ijk,kl->ijl", [(3, 4, 5), (5, 6)]),
    ("ijk,jl->ilk", [(3, 4, 5), (4, 6)]),
    ("ijk,il->ljk", [(3, 4, 5), (3, 2)]),
    ("ijk,k->ij", [(3, 4, 5), (5,)]),
    ("ijk,j->ik", [(3, 4, 5), (4,)]),
    ("ijk->kji", [(3, 4, 5)]),
    ("ijk->i", [(3, 4, 5)]),
    ("ijk->", [(3, 4, 5)]),
    ("ijk", [(3, 4, 5)]),
    ("ijk,jk->i", [(3, 4, 5), (4, 5)]),
    ("ijk,jkr->ir", [(3, 4, 5), (4, 5, 2)]),
    ("bij,bjk->bik", [(3, 4, 2), (3, 2, 5)]),
    ("bij,bjk->kib", [(3, 4, 2), (3, 2, 5)]),
    ("ij,kl->ijkl", [(2, 3), (2, 2)]),
    ("ijk,ijk->ijk", [(3, 4, 5), (3, 4, 5)]),
    ("ijk,jk->ijk", [(3, 4, 5), (4, 5)]),
    ("ijk,ijk->", [(3, 4, 5), (3, 4, 5)]),
    ("ij,jk,kl->il", [(3, 4), (4, 5), (5, 2)]),
    ("ijk,jl,km->ilm", [(3, 4, 5), (4, 2), (5, 3)]),
]


@pytest.mark.parametrize(("sub", "shapes"), EINSUM_CASES)
def test_einsum(sub, shapes):
    rng = np.random.default_rng(48)
    for trial in range(2):
        models = []
        for k, shape in enumerate(shapes):
            layout = all_layouts(len(shape))[-1] if trial and k == 0 else None
            models.append(random_tensor(shape, rng, layout=layout, density=0.6))
        R = gb.core.tensor.einsum(sub, *[m[0] for m in models]).new()
        check(R, *np_einsum(sub, *models), approx=True)


def test_einsum_operands_semirings_and_errors(monkeypatch):
    from graphblas.core import tensor as tensor_mod

    einsum = tensor_mod.einsum
    rng = np.random.default_rng(49)
    X, xv, xp = random_tensor((3, 4, 5), rng, density=0.6)
    B = rng.random((4, 2))
    C = Matrix.from_dense(rng.random((5, 2)))
    ones2 = np.ones((4, 2), bool), np.ones((5, 2), bool)
    R = einsum("ijk,jr,kr->ir", X, B, C).new()
    check(
        R,
        *np_einsum("ijk,jr,kr->ir", (X, xv, xp), (B, B, ones2[0]), (C, C.to_dense(), ones2[1])),
        approx=True,
    )
    # min_plus by brute force for a plain contraction
    Y, yv, yp = random_tensor((5, 2), rng, density=0.6)
    R = einsum("ijk,kl->ijl", X, Y, op=gb.semiring.min_plus).new()
    big = np.where(xp, xv, np.inf)[..., None] + np.where(yp, yv, np.inf)[None, None]
    ev = big.min(axis=2)
    check(R, np.where(np.isfinite(ev), ev, 0), np.isfinite(ev), approx=True)
    # mask on the result
    Mk, _, mp = random_tensor((3, 4, 2), rng, dtype=dtypes.BOOL, density=0.5)
    ev, ep = np_einsum("ijk,kl->ijl", (X, xv, xp), (Y, yv, yp))
    check(
        einsum("ijk,kl->ijl", X, Y).new(mask=Mk.S), np.where(mp & ep, ev, 0), mp & ep, approx=True
    )
    # MTTKRP goes to Tensor.mttkrp's recipe, never the Khatri-Rao product, in both orders
    called = []
    orig = tensor_mod._mttkrp_recipe
    monkeypatch.setattr(
        tensor_mod, "_mttkrp_recipe", lambda *a, **k: called.append(1) or orig(*a, **k)
    )
    einsum("ijk,jr,kr->ir", X, B, C).new()
    assert called
    R = einsum("ijk,rj,kr->ri", X, B.T, C).new()
    assert len(called) == 2
    check(
        R.T,
        *np_einsum("ijk,jr,kr->ir", (X, xv, xp), (B, B, ones2[0]), (C, C.to_dense(), ones2[1])),
        approx=True,
    )
    monkeypatch.undo()
    # without opt_einsum the greedy path is used (the plan is cached by shapes: start afresh)
    tensor_mod._einsum_plan.cache_clear()
    monkeypatch.setitem(__import__("sys").modules, "opt_einsum", None)
    A1, a1v, a1p = random_tensor((3, 4), rng)
    A2, a2v, a2p = random_tensor((4, 5), rng)
    A3, a3v, a3p = random_tensor((5, 2), rng)
    R = einsum("ij,jk,kl->il", A1, A2, A3).new()
    check(
        R, *np_einsum("ij,jk,kl->il", (A1, a1v, a1p), (A2, a2v, a2p), (A3, a3v, a3p)), approx=True
    )
    monkeypatch.undo()
    tensor_mod._einsum_plan.cache_clear()
    with pytest.raises(NotImplementedError, match="diagonal"):
        einsum("iik->k", Tensor(float, 3, 3, 4))
    with pytest.raises(NotImplementedError, match=r"\.\.\."):
        einsum("...k->k", X)
    with pytest.raises(ValueError, match="terms"):
        einsum("ijk,jr->ir", X)
    with pytest.raises(DimensionMismatch, match="sizes"):
        einsum("ijk,jr->ir", X, Tensor(float, 5, 2))
    with pytest.raises(ValueError, match="does not match"):
        einsum("ij->i", X)
    with pytest.raises(ValueError, match="output"):
        einsum("ijk->iq", X)
    with pytest.raises(TypeError):
        einsum("ijk->i", X, op=binary.plus)
    with pytest.raises(TypeError, match="Tensor-like"):
        einsum("ijk->i", "x")
    assert "einsum" in repr(einsum("ijk->i", X))


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_mttkrp(mode):
    rng = np.random.default_rng(50)
    X, xv, xp = random_tensor((3, 4, 5), rng, density=0.5)
    factors = [rng.random((n, 3)) for n in X.shape]
    factors[1] = Matrix.from_dense(factors[1])
    dense = [f if isinstance(f, np.ndarray) else f.to_dense() for f in factors]
    letters = "ijk"
    sub = letters + "," + ",".join(c + "r" for k, c in enumerate(letters) if k != mode)
    sub += "->" + letters[mode] + "r"
    models = [(X, xv, xp)] + [
        (None, dense[k], np.ones_like(dense[k], bool)) for k in range(3) if k != mode
    ]
    check(X.mttkrp(factors, mode).new(), *np_einsum(sub, *models), approx=True)
    with pytest.raises(ValueError, match="factors"):
        X.mttkrp(factors[:2], mode)
    bad = list(factors)
    bad[(mode + 1) % 3] = rng.random((7, 3))
    with pytest.raises(DimensionMismatch, match="rows"):
        X.mttkrp(bad, mode)
    bad[(mode + 1) % 3] = rng.random((X.shape[(mode + 1) % 3], 2))
    with pytest.raises(DimensionMismatch, match="ranks"):
        X.mttkrp(bad, mode)
    with pytest.raises(TypeError, match="factor"):
        X.mttkrp(["x", "y", "z"], mode)


@pytest.fixture(scope="module")
def numba_boundscheck():
    """Compile the Numba kernels with bounds checks while the kernel tests run: an
    out-of-bounds write corrupts the heap silently (it aborted LLVM later, in another test).
    """
    if not gb.core._has_numba:
        yield
        return
    import numba

    from graphblas.core import tensor_numba

    prev = numba.config.BOUNDSCHECK
    numba.config.BOUNDSCHECK = 1
    tensor_numba._compiled.clear()
    yield
    numba.config.BOUNDSCHECK = prev
    tensor_numba._compiled.clear()


def _mttkrp_model(xv, xp, dense, mode):
    letters = "ijklm"[: xv.ndim]
    sub = letters + "," + ",".join(c + "r" for k, c in enumerate(letters) if k != mode)
    sub += "->" + letters[mode] + "r"
    models = [(None, xv, xp)] + [
        (None, dense[k], np.ones_like(dense[k], bool)) for k in range(xv.ndim) if k != mode
    ]
    return np_einsum(sub, *models)


def _kernel_spy(monkeypatch):
    """Record whether each call of the Numba MTTKRP kernel ran (True) or declined (False)."""
    from graphblas.core import tensor, tensor_numba

    calls = []
    orig = tensor_numba.mttkrp_nd

    def spy(*args, **kwargs):
        rv = orig(*args, **kwargs)
        calls.append(rv is not None)
        return rv

    monkeypatch.setattr(tensor_numba, "mttkrp_nd", spy)
    monkeypatch.setattr(tensor, "_MTTKRP_KERNEL_MIN_NNZ", 0)
    return calls, spy


@pytest.mark.usefixtures("numba_boundscheck")
@pytest.mark.parametrize("nthreads", [1, 8])
@pytest.mark.parametrize("shape", [(20, 25, 30), (8, 9, 10, 11), (4, 5, 6, 5, 4)])
def test_mttkrp_kernel_layouts(shape, nthreads, monkeypatch):
    """The Numba MTTKRP kernel reads every layout in place, for every mode (owned output rows,
    per-thread buffers, or blocks of the first column axis; runs of column entries read both
    ways), iso values too: 3-, 4- and 5-d against numpy's einsum.
    """
    pytest.importorskip("numba")
    from graphblas.core import tensor_numba

    calls, _ = _kernel_spy(monkeypatch)
    rng = np.random.default_rng(68 + len(shape))
    ndim = len(shape)
    dense = [rng.random((n, 3)) for n in shape]
    factors = [Matrix.from_dense(d) for d in dense]
    layouts = [layout for layout in all_layouts(ndim) if layout.rows and layout.cols]
    if ndim == 5:
        layouts = layouts[::23]  # 21 of the 480, every split
    for i, layout in enumerate(layouts):
        # Empty slices along every axis, so some output rows are absent
        X, xv, xp = random_tensor(shape, rng, density=0.02, layout=layout, iso=i % 4 == 0)
        assert X.ss.format in {"csr", "hypercsr"}
        for mode in range(ndim):
            for runs in [True, False] if len(layout.cols) > 1 else [None]:
                for colblock in [False, True] if mode == layout.cols[0] else [None]:
                    monkeypatch.setattr(tensor_numba, "_FORCE_RUNS", runs)
                    monkeypatch.setattr(tensor_numba, "_FORCE_COLBLOCK", colblock)
                    calls.clear()
                    R = X.mttkrp(factors, mode).new(nthreads=nthreads)
                    check(R, *_mttkrp_model(xv, xp, dense, mode), approx=True)
                    assert calls == [True], (layout, mode, runs, colblock)


@pytest.mark.usefixtures("numba_boundscheck")
def test_mttkrp_kernel_paths(monkeypatch):
    """Views, R = 1, the relayout handover and the GraphBLAS fallbacks of the MTTKRP kernel."""
    pytest.importorskip("numba")
    from graphblas.core import tensor, tensor_numba

    calls, spy = _kernel_spy(monkeypatch)
    rng = np.random.default_rng(69)
    shape = (8, 9, 10, 11)
    X, xv, xp = random_tensor(shape, rng, density=0.02, layout=((0, 1), (2, 3)))
    dense = [rng.random((n, 3)) for n in shape]
    factors = [Matrix.from_dense(d) for d in dense]
    # A permuted view: the kernel reads the same arrays in the view's layout
    perm = (2, 0, 3, 1)
    V = X.permute_dims(perm)
    vv, vp = xv.transpose(perm), xp.transpose(perm)
    vdense = [dense[a] for a in perm]
    vfactors = [factors[a] for a in perm]
    for mode in range(4):
        calls.clear()
        check(V.mttkrp(vfactors, mode).new(), *_mttkrp_model(vv, vp, vdense, mode), approx=True)
        assert calls == [True]
    # R = 1; one Matrix for two axes (unloaded once) and an ndarray factor
    Y, yv, yp = random_tensor((8, 9, 9, 7), rng, density=0.03, layout=((0, 3), (1, 2)))
    same = Matrix.from_dense(rng.random((9, 1)))
    d1 = [rng.random((8, 1)), same.to_dense(), same.to_dense(), rng.random((7, 1))]
    for mode in range(4):
        calls.clear()
        R = Y.mttkrp([Matrix.from_dense(d1[0]), same, same, d1[3]], mode).new()
        check(R, *_mttkrp_model(yv, yp, d1, mode), approx=True)
        assert calls == [True]
    # Many more output rows than chunks: each chunk owns several, some of them absent
    Z, zv, zp = random_tensor((600, 6, 7, 3), rng, density=0.01, layout=((0, 1), (2, 3)))
    zd = [rng.random((n, 2)) for n in Z.shape]
    for mode in [0, 2]:
        calls.clear()
        R = Z.mttkrp([Matrix.from_dense(d) for d in zd], mode).new(nthreads=8)
        check(R, *_mttkrp_model(zv, zp, zd, mode), approx=True)
        assert calls == [True]
    # Buffers too large in the stored layout: blocks of the first column axis (mode 2), else
    # a relayout and then the kernel that owns its rows
    monkeypatch.setattr(tensor_numba, "mttkrp_nd", lambda *a, **k: spy(*a, **dict(k, max_buffer=0)))
    for mode in [1, 2, 3]:
        calls.clear()
        R = X.mttkrp(factors, mode).new(nthreads=8)
        check(R, *_mttkrp_model(xv, xp, dense, mode), approx=True)
        assert calls == ([True] if mode == 2 else [False, True])
    monkeypatch.setattr(tensor_numba, "mttkrp_nd", spy)
    # The GraphBLAS products: no Numba, an empty tensor, a sparse factor, another dtype or
    # semiring; the kernel never runs
    calls.clear()
    with monkeypatch.context() as m:
        m.setattr(tensor, "_has_numba", False)
        check(X.mttkrp(factors, 0).new(), *_mttkrp_model(xv, xp, dense, 0), approx=True)
    E = Tensor.from_coo([np.empty(0, np.int64)] * 4, np.empty(0), shape=shape)
    check(E.mttkrp(factors, 2).new(), np.zeros((10, 3)), np.zeros((10, 3), bool))
    sparse = list(factors)
    sparse[1] = factors[1].dup()
    del sparse[1][0, 0]
    dense_sparse = [d.copy() for d in dense]
    dense_sparse[1][0, 0] = 0
    sparse_result = X.mttkrp(sparse, 0).new()
    for dtype in [dtypes.INT64, dtypes.FP32]:
        Xi, xiv, xip = random_tensor(shape, rng, density=0.02, dtype=dtype)
        R = Xi.mttkrp(factors, 3).new()
        check(R, *_mttkrp_model(xiv, xip, dense, 3), approx=True)
    R = X.mttkrp(factors, 0, op=gb.semiring.max_times).new()
    assert R.dtype == dtypes.FP64
    assert calls == []
    # The sparse factor's result matches the kernel's on the same values with zeros stored
    assert sparse_result.isclose(X.mttkrp(dense_sparse, 0).new())
    assert calls == [True]


@pytest.mark.usefixtures("numba_boundscheck")
@pytest.mark.parametrize("nthreads", [1, 8])
def test_mttkrp_empty_groups(nthreads, monkeypatch):
    """Layouts with an empty group (all axes in the rows or all in the columns), as stored
    and as squeeze / expand_dims views make them: the kernel relayouts first (it needs both
    groups), the products work as usual; every mode, against numpy's einsum.
    """
    from graphblas.core import tensor

    has_kernel = tensor._has_numba
    if has_kernel:
        calls, _ = _kernel_spy(monkeypatch)
    rng = np.random.default_rng(71)
    shape = (9, 10, 11, 12)
    cases = []
    for i, layout in enumerate(
        [((), (0, 1, 2, 3)), ((0, 1, 2, 3), ()), ((), (2, 0, 3, 1)), ((3, 1, 0, 2), ())]
    ):
        X, xv, xp = random_tensor(shape, rng, density=0.03, layout=layout, iso=i == 2)
        assert not X.layout.rows or not X.layout.cols
        cases.append((X, xv, xp))
    # squeeze of a leading size-1 axis in the rows leaves the rows empty
    Y, yv, yp = random_tensor((1, 20, 30, 40), rng, density=0.01, layout=((0,), (1, 2, 3)))
    S = Y.squeeze(0)
    assert S.layout.rows == ()
    cases.append((S, yv[0], yp[0]))
    # expand_dims of tensors with an empty group: a size-1 axis, the group stays empty
    for X, xv, xp in cases[:2]:
        E = X.expand_dims(2)
        assert not E.layout.rows or not E.layout.cols
        cases.append((E, np.expand_dims(xv, 2), np.expand_dims(xp, 2)))
    # and an empty tensor in an empty-group layout
    E = Tensor.from_coo(
        [np.empty(0, np.int64)] * 4, np.empty(0), shape=shape, layout=((), (0, 1, 2, 3))
    )
    cases.append((E, np.zeros(shape), np.zeros(shape, bool)))
    for X, xv, xp in cases:
        dense = [rng.random((n, 3)) for n in X.shape]
        factors = [Matrix.from_dense(d) for d in dense]
        for mode in range(X.ndim):
            expected = _mttkrp_model(xv, xp, dense, mode)
            if has_kernel:
                calls.clear()
            check(X.mttkrp(factors, mode).new(nthreads=nthreads), *expected, approx=True)
            if has_kernel:
                # Declined in the stored layout (no call), run after the relayout
                assert calls == ([] if X.nvals == 0 else [True]), (X.layout, mode)
            with monkeypatch.context() as m:
                m.setattr(tensor, "_has_numba", False)
                check(X.mttkrp(factors, mode).new(nthreads=nthreads), *expected, approx=True)


@pytest.mark.parametrize("shape", [(6, 7, 3000, 4), (6, 3000, 7, 4)])
def test_mttkrp_products_order(shape, monkeypatch):
    """Without the kernel, MTTKRP contracts the widest axis first: its intermediates have one
    row per distinct (mode, rest) fiber, which the order choice now counts from a sample
    (relayout cost alone picks the other order here).
    """
    from graphblas.core import tensor

    rng = np.random.default_rng(70)
    coords = [rng.integers(0, n, 30000) for n in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = 1.0 + rng.random(lin.size)
    X = Tensor.from_coo(coords, vals, shape=shape, layout=((0, 1), (2, 3)))
    wide = int(np.argmax(shape))
    for nthreads in [1, 8]:
        assert tensor._mttkrp_order(X, 0, 64, nthreads, products=True)[-1] == wide
    # The estimates are low, never high, and within 3x here
    groups = {(0, 1), (0, 1, 2), (0, 1, 3), (0, 2, 3)}
    est = tensor._sampled_fibers(X, groups)
    for g in groups:
        exact = np.unique(np.ravel_multi_index([coords[a] for a in g], [shape[a] for a in g]))
        assert exact.size / 3 <= est[g] <= exact.size * 1.01, (g, est[g], exact.size)
    xv = np.zeros(shape)
    xv[coords] = vals
    dense = [rng.random((n, 2)) for n in shape]
    monkeypatch.setattr(tensor, "_has_numba", False)
    R = X.mttkrp([Matrix.from_dense(d) for d in dense], 0).new(nthreads=1)
    check(R, *_mttkrp_model(xv, xv != 0, dense, 0), approx=True)


def _selected(mask, shape):
    """The positions a mask selects, as a dense bool array (it masks an assignment of True)."""
    if type(mask.parent) is Tensor or len(shape) > 2:
        R = Tensor(dtypes.BOOL, shape)
        R(mask) << Tensor.from_dense(np.ones(shape, bool))
        out = np.zeros(shape, bool)
        coords, _ = R.to_coo()
    else:
        R = Matrix(dtypes.BOOL, *shape) if len(shape) == 2 else Vector(dtypes.BOOL, *shape)
        ones = np.ones(shape, bool)
        R(mask) << (Matrix if len(shape) == 2 else Vector).from_dense(ones)
        out = np.zeros(shape, bool)
        coords = R.to_coo()[:-1]
    out[tuple(coords)] = True
    return out


def _mask_kinds(X, values):
    """The four masks of X with the numpy positions each selects."""
    present = values >= 0  # -1 marks a missing entry; 0 is a stored zero
    return {
        "S": (X.S, present),
        "V": (X.V, present & (values != 0)),
        "~S": (~X.S, ~present),
        "~V": (~X.V, ~(present & (values != 0))),
    }


def _masked_data(rng, shape):
    values = np.where(rng.random(shape) < 0.4, rng.integers(0, 3, shape), -1)
    return values, values[values >= 0].astype(np.float64)


def test_mask_combinations(monkeypatch):
    """`|` and `&` of S, V, ~S and ~V masks of Tensors (the union used to raise "plus does not
    work with BOOL"), and mixed with Matrix and Vector masks: against numpy, and against Matrix
    for the same data in 2-d.
    """
    rng = np.random.default_rng(72)
    ops = {"|": lambda a, b: a | b, "&": lambda a, b: a & b}
    # 3-d, Tensor with Tensor, against numpy
    shape = (4, 5, 6)
    xv, xvals = _masked_data(rng, shape)
    yv, yvals = _masked_data(rng, shape)
    X = Tensor.from_coo(np.nonzero(xv >= 0), xvals, shape=shape)
    Y = Tensor.from_coo(np.nonzero(yv >= 0), yvals, shape=shape, layout=((2,), (0, 1)))
    for (k1, (m1, p1)), (k2, (m2, p2)) in itertools.product(
        _mask_kinds(X, xv).items(), _mask_kinds(Y, yv).items()
    ):
        for name, op in ops.items():
            np.testing.assert_array_equal(
                _selected(op(m1, m2), shape), op(p1, p2), err_msg=f"{k1} {name} {k2}"
            )
    # 2-d: Tensor-Tensor, Tensor-Matrix and Matrix-Tensor give what Matrix-Matrix gives
    shape = (5, 6)
    av, avals = _masked_data(rng, shape)
    bv, bvals = _masked_data(rng, shape)
    A = Matrix.from_coo(*np.nonzero(av >= 0), avals, nrows=5, ncols=6)
    B = Matrix.from_coo(*np.nonzero(bv >= 0), bvals, nrows=5, ncols=6)
    TA, TB = Tensor.from_matrix(A), Tensor.from_matrix(B)
    for (k1, (ma, p1)), (k2, (mb, p2)) in itertools.product(
        _mask_kinds(A, av).items(), _mask_kinds(B, bv).items()
    ):
        mta, mtb = _mask_kinds(TA, av)[k1][0], _mask_kinds(TB, bv)[k2][0]
        for name, op in ops.items():
            expected = _selected(op(ma, mb), shape)
            np.testing.assert_array_equal(expected, op(p1, p2))
            for left, right in [(mta, mtb), (mta, mb), (ma, mtb)]:
                got = _selected(op(left, right), shape)
                np.testing.assert_array_equal(got, expected, err_msg=f"{k1} {name} {k2}")
    # With a Vector: where Matrix combines (a union broadcasting the Vector along the rows),
    # the Tensor gives the same; where Matrix refuses a mask of another shape, so does the
    # Tensor (masks do not broadcast)
    cv, cvals = _masked_data(rng, (6,))
    v = Vector.from_coo(np.flatnonzero(cv >= 0), cvals, size=6)
    for (k1, (ma, _p1)), (k2, (mv, _)) in itertools.product(
        _mask_kinds(A, av).items(), _mask_kinds(v, cv).items()
    ):
        mta = _mask_kinds(TA, av)[k1][0]
        for name, op in ops.items():
            try:
                expected = _selected(op(ma, mv), shape)
            except (TypeError, DimensionMismatch):
                with pytest.raises((TypeError, DimensionMismatch)):
                    _selected(op(mta, mv), shape)
                continue
            np.testing.assert_array_equal(
                _selected(op(mta, mv), shape), expected, err_msg=f"{k1} {name} v.{k2}"
            )
    # A mask of a Matrix applies to a 2-d Tensor result directly; of another shape, it raises
    R = TA.apply(gb.unary.one).new(mask=B.S)
    np.testing.assert_array_equal(_selected(R.S, shape), (av >= 0) & (bv >= 0))
    with pytest.raises(TypeError, match="Tensor mask"):
        TA.apply(gb.unary.one).new(mask=v.S)


def test_relayout_ends_sparse_when_rows_fill():
    """A relayout that ends with a reshape gives its result SuiteSparse's usual hyper switch
    (a reshape leaves 1.0, keeping full rows hypersparse, which sends mxm to its Gustavson
    saxpy); rows that stay mostly empty stay hypersparse.
    """
    rng = np.random.default_rng(75)
    shape = (40, 50, 300)
    lin = np.unique(rng.integers(0, math.prod(shape), 20000))
    X = Tensor.from_coo(np.unravel_index(lin, shape), 1.0, shape=shape, layout=((0,), (1, 2)))
    M, state = tl.execute(X.ss.matrix, shape, ((0, 1, 2), 1), [("RS", 2)])
    assert state == ((0, 1, 2), 2)
    assert M.ss.format == "csr"  # 2000 rows, about 10 entries each
    assert M.ss.config["hyper_switch"] == gb.ss.config["hyper_switch"]
    T = Tensor._from_M(M, shape, Layout((0, 1), (2,)))
    T._check_invariants()
    sparse_shape = (40, 5000, 30)
    lin = np.unique(rng.integers(0, math.prod(sparse_shape), 2000))
    Y = Tensor.from_coo(np.unravel_index(lin, sparse_shape), 1.0, shape=sparse_shape)
    M, _ = tl.execute(Y.ss.matrix, sparse_shape, Y.layout.state, [("RS", 2)])
    assert M.ss.format == "hypercsr"  # 2000 entries in 200000 rows


def test_scalar_index():
    """A GraphBLAS Scalar indexes as the int it holds, on any axis and with any integer dtype;
    an empty or non-integer Scalar raises TypeError, as for Matrix.
    """
    rng = np.random.default_rng(74)
    X, xv, xp = random_tensor((4, 5, 6), rng, layout=((1,), (2, 0)))
    for axis, i in [(0, 1), (1, -1), (2, 5), (1, 0)]:
        for dtype in ["INT8", "INT64", "UINT32"]:
            if i < 0 and dtype.startswith("U"):
                continue
            key = [slice(None)] * 3
            key[axis] = Scalar.from_value(i, dtype=dtype)
            ikey = list(key)
            ikey[axis] = i
            got = X[tuple(key)].new()
            check(got, np_outer_index(xv, tuple(ikey)), np_outer_index(xp, tuple(ikey)))
    # With slices and lists on other axes, and all three (one element)
    key = (Scalar.from_value(2), [0, 3], slice(1, 5))
    ikey = (2, [0, 3], slice(1, 5))
    check(X[key].new(), np_outer_index(xv, ikey), np_outer_index(xp, ikey))
    s = [Scalar.from_value(i) for i in (3, 2, 1)]
    assert X[s[0], s[1], s[2]].new().isequal(X[3, 2, 1].new())
    A = Matrix.from_dense(np.arange(12.0).reshape(3, 4))
    for bad in [Scalar("INT64"), Scalar.from_value(1.0), Scalar.from_value(True)]:
        with pytest.raises(TypeError):
            A[bad, :].new()
        with pytest.raises(TypeError):
            X[bad, :, :]
    with pytest.raises(IndexOutOfBound):
        X[:, Scalar.from_value(5), :]


def test_infix_bool_and_operator_waits():
    """`X | Y` and `X & Y` of BOOL Tensors default to lor and land, as for Matrix; an infix
    expression leaves the operator to its caller, so `op(X | Y)` works for dtypes the default
    operator does not cover (UDTs, and BOOL under monoid.any, as mask unions use).
    """
    rng = np.random.default_rng(73)
    av, _ = _masked_data(rng, (5, 6))
    bv, _ = _masked_data(rng, (5, 6))
    A = Matrix.from_coo(*np.nonzero(av >= 0), av[av >= 0] > 0, nrows=5, ncols=6)
    B = Matrix.from_coo(*np.nonzero(bv >= 0), bv[bv >= 0] > 1, nrows=5, ncols=6)
    TA, TB = Tensor.from_matrix(A), Tensor.from_matrix(B)
    for expr, ref in [(TA | TB, A | B), (TA & TB, A & B)]:
        got = expr.new()
        assert got.dtype == dtypes.BOOL
        assert got.to_matrix().isequal(ref.new())
    assert (monoid.any(TA | TB)).new().to_matrix().isequal(monoid.any(A | B).new())
    with pytest.raises(DimensionMismatch):
        TA | Tensor(bool, 5, 7)  # mismatched shapes still raise at once
    udt = dtypes.register_anonymous(np.dtype([("tensor_a", "<i4"), ("tensor_b", "<f8")]))
    vals = np.zeros(3, dtype=udt.np_type)
    vals["tensor_a"] = [1, 2, 3]
    U = Tensor.from_coo(([0, 1, 2], [0, 1, 2]), vals, udt, shape=(3, 3))
    W = Tensor.from_coo(([0, 2], [1, 2]), vals[:2], udt, shape=(3, 3))
    expr = U | W  # building `U | W` no longer needs plus on the UDT
    if gb.core._has_numba:  # (operators on UDTs need Numba)
        assert binary.first(expr).new().nvals == 4


def _ttm_model(xv, xp, U, mode, kb=0):
    """Numpy's TTM: values and presence (a full factor reaches every r of a nonempty fiber)."""
    Uk = U if kb == 0 else U.T
    values = np.tensordot(xv, Uk, ([mode], [0]))
    present = np.tensordot(xp.astype(np.int64), np.ones(Uk.shape, np.int64), ([mode], [0])) > 0
    return np.where(present, values, 0), present


def _ttm_spy(monkeypatch):
    from graphblas.core import tensor, tensor_numba

    calls = []
    orig = tensor_numba.ttm

    def spy(*args, **kwargs):
        rv = orig(*args, **kwargs)
        calls.append(rv[0])
        return rv

    monkeypatch.setattr(tensor_numba, "ttm", spy)
    monkeypatch.setattr(tensor, "_TTM_KERNEL_MIN_NNZ", 0)
    monkeypatch.setattr(tensor, "_TTM_ENGINES", ("numba",))
    return calls


@pytest.mark.usefixtures("numba_boundscheck")
@pytest.mark.parametrize("nthreads", [1, 8])
@pytest.mark.parametrize("shape", [(12, 13, 14), (6, 7, 8, 9), (4, 5, 6, 5, 4)])
def test_ttm_kernel_layouts(shape, nthreads, monkeypatch):
    """The Numba TTM kernel for every mode and layout (the axis read in place as the last
    column axis, its leading column axes making the output fibers; else after a relayout),
    either side of the factor, iso values: 3-, 4- and 5-d against numpy.
    """
    pytest.importorskip("numba")
    calls = _ttm_spy(monkeypatch)
    rng = np.random.default_rng(76 + len(shape))
    ndim = len(shape)
    layouts = all_layouts(ndim)
    if ndim == 5:
        layouts = layouts[::29]
    for i, layout in enumerate(layouts):
        X, xv, xp = random_tensor(shape, rng, density=0.02, layout=layout, iso=i % 5 == 0)
        assert X.ss.format in {"csr", "hypercsr"}
        for mode in range(ndim):
            U = rng.random((shape[mode], 3))
            kb = (i + mode) % 2  # contract the factor's rows or its columns
            F = Matrix.from_dense(U if kb == 0 else U.T)
            calls.clear()
            R = X.tensordot(F, ([mode], [kb])).new(nthreads=nthreads)
            check(R, *_ttm_model(xv, xp, U, mode), approx=True)
            assert calls, (layout, mode)


@pytest.mark.usefixtures("numba_boundscheck")
def test_ttm_kernel_paths(monkeypatch):
    """TTM outputs in every storage (hypersparse rows, bitmap, full), views, empty groups,
    R = 1, and the GraphBLAS product where the kernel cannot run.
    """
    pytest.importorskip("numba")
    from graphblas.core import tensor

    calls = _ttm_spy(monkeypatch)
    rng = np.random.default_rng(80)
    # Every (i, j) fiber present: full output; about 70% of them: bitmap
    shape = (20, 30, 400)
    for share, kind in [(1.0, "full"), (0.7, "bitmap")]:
        ij = np.flatnonzero(rng.random(600) < share)
        k = rng.integers(0, 400, (ij.size, 3))
        lin = np.unique(
            np.ravel_multi_index((np.repeat(ij // 30, 3), np.repeat(ij % 30, 3), k.ravel()), shape)
        )
        coords = np.unravel_index(lin, shape)
        X = Tensor.from_coo(coords, rng.random(lin.size), shape=shape, layout=((0, 1), (2,)))
        xv = np.zeros(shape)
        xv[coords] = X.to_coo()[1]
        U = rng.random((400, 4))
        calls.clear()
        R = X.tensordot(Matrix.from_dense(U), ([2], [0])).new()
        check(R, *_ttm_model(xv, xv != 0, U, 2), approx=True)
        assert calls == [kind]
    # Hypersparse output; a permuted view; layouts with an empty group (squeeze too)
    X, xv, xp = random_tensor((10, 11, 12, 13), rng, density=0.02, layout=((2, 0), (3, 1)))
    U = rng.random((11, 2))
    calls.clear()
    check(
        X.tensordot(Matrix.from_dense(U), ([1], [0])).new(), *_ttm_model(xv, xp, U, 1), approx=True
    )
    assert calls == ["hyper"]
    perm = (3, 1, 0, 2)
    V = X.permute_dims(perm)
    for mode in range(4):
        U = rng.random((V.shape[mode], 2))
        expected = _ttm_model(xv.transpose(perm), xp.transpose(perm), U, mode)
        check(V.tensordot(Matrix.from_dense(U), ([mode], [0])).new(), *expected, approx=True)
    for layout in [((), (0, 1, 2, 3)), ((0, 1, 2, 3), ())]:
        E, ev, ep = random_tensor((10, 11, 12, 13), rng, density=0.02, layout=layout)
        for mode in range(4):
            U = rng.random((E.shape[mode], 2))
            check(
                E.tensordot(Matrix.from_dense(U), ([mode], [0])).new(),
                *_ttm_model(ev, ep, U, mode),
                approx=True,
            )
    Y, yv, yp = random_tensor((1, 20, 30, 40), rng, density=0.01, layout=((0,), (1, 2, 3)))
    S = Y.squeeze(0)
    for mode in range(3):
        U = rng.random((S.shape[mode], 2))
        check(
            S.tensordot(Matrix.from_dense(U), ([mode], [0])).new(),
            *_ttm_model(yv[0], yp[0], U, mode),
            approx=True,
        )
    # R = 1, and an ndarray factor
    U = rng.random((12, 1))
    calls.clear()
    check(X.tensordot(U, ([2], [0])).new(), *_ttm_model(xv, xp, U, 2), approx=True)
    assert calls
    # The product: no Numba, a sparse factor, another dtype or semiring, an empty tensor
    calls.clear()
    U = rng.random((13, 3))
    with monkeypatch.context() as m:
        m.setattr(tensor, "_has_numba", False)
        check(
            X.tensordot(Matrix.from_dense(U), ([3], [0])).new(),
            *_ttm_model(xv, xp, U, 3),
            approx=True,
        )
    Us = Matrix.from_dense(U)
    del Us[0, 0]
    got = X.tensordot(Us, ([3], [0])).new()
    U0 = U.copy()
    U0[0, 0] = 0
    vals, _ = _ttm_model(xv, xp, U0, 3)
    present = (
        np.tensordot(
            xp.astype(np.int64),
            (Us.to_dense(fill_value=0) != 0).astype(np.int64) | (np.arange(13) != 0)[:, None],
            ([3], [0]),
        )
        > 0
    )
    check(got, np.where(present, vals, 0), present, approx=True)
    Xi, xiv, xip = random_tensor((10, 11, 12, 13), rng, density=0.02, dtype=dtypes.INT64)
    check(
        Xi.tensordot(Matrix.from_dense(U), ([3], [0])).new(),
        *_ttm_model(xiv, xip, U, 3),
        approx=True,
    )
    X.tensordot(Matrix.from_dense(U), ([3], [0]), op=gb.semiring.max_times).new()
    E = Tensor.from_coo([np.empty(0, np.int64)] * 4, np.empty(0), shape=(10, 11, 12, 13))
    check(
        E.tensordot(Matrix.from_dense(U), ([3], [0])).new(),
        np.zeros((10, 11, 12, 3)),
        np.zeros((10, 11, 12, 3), bool),
    )
    assert calls == []


def _ttm_udt_spy(monkeypatch, engines=("udt",)):
    from graphblas.core import tensor, tensor_udt

    calls = []
    orig = tensor_udt.ttm

    def spy(*args, **kwargs):
        rv = orig(*args, **kwargs)
        calls.append(None if rv is None else rv.ss.format)
        return rv

    monkeypatch.setattr(tensor_udt, "ttm", spy)
    monkeypatch.setattr(tensor, "_TTM_KERNEL_MIN_NNZ", 0)
    monkeypatch.setattr(tensor, "_TTM_ENGINES", engines)
    return calls


def _require_jit_compile():
    from graphblas.core import tensor_udt

    if not tensor_udt.jit_ready() or tensor_udt.semiring(2) is None:
        pytest.skip("the UDT TTM engine needs the SuiteSparse JIT with a compiler")


@pytest.mark.parametrize("nthreads", [1, 8])
@pytest.mark.parametrize("shape", [(12, 13, 14), (6, 7, 8, 9), (4, 5, 6, 5, 4)])
def test_ttm_udt_layouts(shape, nthreads, monkeypatch):
    """The array-UDT TTM engine (one mxv with values of R doubles) for every mode and
    layout (after a relayout where the axis is not the column group by itself), either side
    of the factor, iso values, R = 1: 3-, 4- and 5-d against numpy.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    _require_jit_compile()
    calls = _ttm_udt_spy(monkeypatch)
    rng = np.random.default_rng(90 + len(shape))
    ndim = len(shape)
    layouts = all_layouts(ndim)
    if ndim == 5:
        layouts = layouts[::29]
    for i, layout in enumerate(layouts):
        X, xv, xp = random_tensor(shape, rng, density=0.03, layout=layout, iso=i % 5 == 0)
        for mode in range(ndim):
            U = rng.random((shape[mode], 1 if i % 7 == 3 else 3))
            kb = (i + mode) % 2
            F = Matrix.from_dense(U if kb == 0 else U.T)
            calls.clear()
            R = X.tensordot(F, ([mode], [kb])).new(nthreads=nthreads)
            check(R, *_ttm_model(xv, xp, U, mode), approx=True)
            assert calls, (layout, mode)
            assert calls[0] in {"hypercsr", "bitmapr", "fullr"}


def test_ttm_udt_paths(monkeypatch, capfd):
    """UDT engine outputs in every storage, views and empty groups, an empty tensor, the
    JIT kernel (burble), its type and ops made once per R, and JIT off.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    _require_jit_compile()
    from graphblas.core import tensor_udt

    calls = _ttm_udt_spy(monkeypatch)
    rng = np.random.default_rng(95)
    shape = (20, 30, 400)
    for share, kind in [(1.0, "fullr"), (0.7, "bitmapr"), (0.2, "hypercsr")]:
        ij = np.flatnonzero(rng.random(600) < share)
        k = rng.integers(0, 400, (ij.size, 3))
        lin = np.unique(
            np.ravel_multi_index((np.repeat(ij // 30, 3), np.repeat(ij % 30, 3), k.ravel()), shape)
        )
        coords = np.unravel_index(lin, shape)
        X = Tensor.from_coo(coords, rng.random(lin.size), shape=shape, layout=((0, 1), (2,)))
        xv = np.zeros(shape)
        xv[coords] = X.to_coo()[1]
        U = rng.random((400, 5))
        calls.clear()
        R = X.tensordot(Matrix.from_dense(U), ([2], [0])).new()
        check(R, *_ttm_model(xv, xv != 0, U, 2), approx=True)
        assert calls == [kind]
    # The product's kernel is the JIT one (a UDT product has no other but the generic one,
    # which burble names)
    import graphblas.core.tensor_udt as tu

    M = X._M
    for control, generic in [("on", False), ("off", True)]:
        prev = gb.ss.config["jit_c_control"]
        gb.ss.config["jit_c_control"] = control
        capfd.readouterr()
        gb.ss.config["burble"] = True
        try:
            tu.ttm(M, rng.random((400, 5)))
        finally:
            gb.ss.config["burble"] = False
            gb.ss.config["jit_c_control"] = prev
        out = capfd.readouterr().out
        assert "GrB_mxv" in out
        assert ("generic" in out) == generic, out
    # The type and semiring of an R are made once
    first = tensor_udt._SEMIRINGS[5]
    X.tensordot(Matrix.from_dense(U), ([2], [0])).new()
    assert tensor_udt._SEMIRINGS[5] is first
    assert tensor_udt.semiring(5) is first
    # Views, empty groups, an empty tensor
    X, xv, xp = random_tensor((10, 11, 12, 13), rng, density=0.03, layout=((2, 0), (3, 1)))
    V = X.permute_dims((3, 1, 0, 2))
    for mode in range(4):
        U = rng.random((V.shape[mode], 2))
        expected = _ttm_model(xv.transpose(3, 1, 0, 2), xp.transpose(3, 1, 0, 2), U, mode)
        check(V.tensordot(Matrix.from_dense(U), ([mode], [0])).new(), *expected, approx=True)
    for layout in [((), (0, 1, 2, 3)), ((0, 1, 2, 3), ())]:
        E, ev, ep = random_tensor((10, 11, 12, 13), rng, density=0.03, layout=layout)
        for mode in range(4):
            U = rng.random((E.shape[mode], 2))
            check(
                E.tensordot(Matrix.from_dense(U), ([mode], [0])).new(),
                *_ttm_model(ev, ep, U, mode),
                approx=True,
            )
    calls.clear()
    Z = Tensor(dtypes.FP64, (5, 6, 7))
    check(
        Z.tensordot(rng.random((7, 2)), ([2], [0])).new(),
        np.zeros((5, 6, 2)),
        np.zeros((5, 6, 2), bool),
    )
    # JIT off: not this engine; the result is the same
    X, xv, xp = random_tensor((10, 11, 12), rng, density=0.05, layout=((0, 1), (2,)))
    U = rng.random((12, 3))
    prev = gb.ss.config["jit_c_control"]
    try:
        gb.ss.config["jit_c_control"] = "off"
        calls.clear()
        check(X.tensordot(U, ([2], [0])).new(), *_ttm_model(xv, xp, U, 2), approx=True)
        assert calls == []
    finally:
        gb.ss.config["jit_c_control"] = prev


_JIT_FAILURE_SCRIPT = """
import sys
import warnings
if sys.argv[2] == "nonumba":
    sys.modules["numba"] = None
import numpy as np
import graphblas as gb
from graphblas.core import _has_numba, tensor, tensor_udt
from graphblas.core.tensor import Tensor
print("package:", gb.__file__)
if not tensor_udt.jit_ready():  # (the probe compiles a test type into the empty cache)
    print("skip: no JIT compiler")
    sys.exit(0)
gb.ss.config["jit_c_compiler_name"] = "/nonexistent/bin/cc"
gb.ss.config["jit_c_control"] = sys.argv[1]
tensor._TTM_KERNEL_MIN_NNZ = 0
rng = np.random.default_rng(0)
shape = (30, 40, 50)
xv = np.where(rng.random(shape) < 0.002, rng.random(shape), 0)
X = Tensor.from_coo(np.nonzero(xv), xv[xv != 0], shape=shape, layout=((0, 1), (2,)))
U = rng.random((50, 11))
expected = np.tensordot(xv, U, ([2], [0]))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    err = [
        float(np.abs(X.tensordot(U, ([2], [0])).new().to_dense(fill_value=0) - expected).max())
        for _ in range(2)
    ]
print("max_err:", max(err))
print("control:", gb.ss.config["jit_c_control"])
print("compiler:", gb.ss.config["jit_c_compiler_name"])
print("failure:", tensor_udt._FAILURE and tensor_udt._FAILURE[:2])
print("warnings:", sum("array-UDT engine" in str(w.message) for w in caught))
print("has_numba:", _has_numba)
"""


@pytest.mark.parametrize(("control", "numba"), [("run", "numba"), ("on", "nonumba")])
def test_ttm_udt_jit_failure(control, numba, tmp_path):
    """A compile that fails with the JIT on (a compiler gone missing after the probe, an
    empty JIT cache, a fresh R): the UDT engine gives up once, warns, restores the JIT
    control the user had, and the next engine (Numba, or the products without it) gives
    the right result; later calls do not try the engine again.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root), GRAPHBLAS_CACHE_PATH=str(tmp_path / "jit"))
    result = subprocess.run(
        [sys.executable, "-c", _JIT_FAILURE_SCRIPT, control, numba],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=600,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines["package"] == str(repo_root / "graphblas" / "__init__.py"), report
    if "skip" in lines:
        pytest.skip(lines["skip"])
    assert float(lines["max_err"]) < 1e-12, report
    assert lines["control"] == control, report  # (restored)
    assert lines["compiler"] == "/nonexistent/bin/cc", report  # (not touched)
    assert lines["failure"] == f"({control!r}, 'load')", report  # (SuiteSparse had set "load")
    assert lines["warnings"] == "1", report
    assert lines["has_numba"] == str(numba == "numba"), report


def _engine_spies(monkeypatch):
    """Spies on every TTM and MTTKRP engine: the names of those that produced a result."""
    from graphblas.core import tensor, tensor_numba, tensor_udt

    ran = []

    def spy(name, module, attr, ok=lambda rv: rv is not None):
        orig = getattr(module, attr)

        def wrapper(*args, **kwargs):
            rv = orig(*args, **kwargs)
            if ok(rv):
                ran.append(name)
            return rv

        monkeypatch.setattr(module, attr, wrapper)

    spy("udt", tensor_udt, "ttm")
    spy("numba", tensor_numba, "ttm")
    spy("numba", tensor_numba, "mttkrp_nd")
    spy("compacted", tensor, "_ttm_compacted")
    spy("products", tensor, "_contract", ok=lambda rv: True)
    monkeypatch.setattr(tensor, "_TTM_KERNEL_MIN_NNZ", 0)
    monkeypatch.setattr(tensor, "_MTTKRP_KERNEL_MIN_NNZ", 0)
    return ran


def test_engines_selectable(monkeypatch):
    """Every TTM engine (udt, numba, compacted, products) and MTTKRP engine (numba, udt,
    products) can be forced, and each gives the numpy result; a forced engine that does not
    run here falls back to the products.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    pytest.importorskip("numba")
    _require_jit_compile()
    from graphblas.core import tensor

    ran = _engine_spies(monkeypatch)
    rng = np.random.default_rng(99)
    X, xv, xp = random_tensor((30, 40, 50), rng, density=0.003, layout=((0, 1), (2,)))
    assert X.ss.format == "hypercsr"
    U = rng.random((50, 4))
    for engine in ["udt", "numba", "compacted", "products"]:
        monkeypatch.setattr(tensor, "_TTM_ENGINES", (engine,))
        ran.clear()
        check(X.tensordot(U, ([2], [0])).new(), *_ttm_model(xv, xp, U, 2), approx=True)
        assert ran == [engine], engine
    # Not in the stored order: compacted does not run, so the products do
    monkeypatch.setattr(tensor, "_TTM_ENGINES", ("compacted",))
    ran.clear()
    U0 = rng.random((30, 4))
    check(X.tensordot(U0, ([0], [0])).new(), *_ttm_model(xv, xp, U0, 0), approx=True)
    assert ran == ["products"]
    dense = [rng.random((n, 3)) for n in X.shape]
    factors = [Matrix.from_dense(d) for d in dense]
    for engine, expected in [("numba", ["numba"]), ("udt", ["udt"]), ("products", [])]:
        monkeypatch.setattr(tensor, "_MTTKRP_ENGINES", (engine,))
        for mode in range(3):
            ran.clear()
            check(X.mttkrp(factors, mode).new(), *_mttkrp_model(xv, xp, dense, mode), approx=True)
            assert ran == expected, (engine, mode)


_JIT_NEW_KERNEL_SCRIPT = """
import sys
import warnings
sys.modules["numba"] = None  # (so the UDT engine takes the dense case too)
import numpy as np
import graphblas as gb
from graphblas.core import tensor, tensor_udt
from graphblas.core.tensor import Tensor
print("package:", gb.__file__)
if not tensor_udt.jit_ready():
    print("skip: no JIT compiler")
    sys.exit(0)
tensor._TTM_KERNEL_MIN_NNZ = 0
attempts = []
orig = tensor_udt.ttm
def spy(*args, **kwargs):
    try:
        rv = orig(*args, **kwargs)
    except Exception as exc:
        attempts.append(type(exc).__name__)
        raise
    attempts.append("ok" if rv is not None else "none")
    return rv
tensor_udt.ttm = spy
rng = np.random.default_rng(0)
shape = (300, 300, 40)
flat = np.unique(rng.choice(np.prod(shape), 20000, replace=False))
xv = np.zeros(shape)
xv.flat[flat] = rng.standard_normal(flat.size)
X = Tensor.from_coo(np.nonzero(xv), xv[xv != 0], shape=shape, layout=((0, 1), (2,)))
dv = rng.standard_normal((30, 30, 40))
Xd = Tensor.from_dense(dv, layout=((0, 1), (2,)))
F = [rng.standard_normal((n, 7)) for n in (30, 30, 40)]
def run(T, tv):
    if sys.argv[1] == "ttm":
        got = T.tensordot(F[2], ([2], [0])).new().to_dense(fill_value=0)
        return float(np.abs(got - np.tensordot(tv, F[2], ([2], [0]))).max())
    Fs = [np.resize(f, (n, 7)) for f, n in zip(F, T.shape)]
    got = T.mttkrp(Fs, 0).new().to_dense(fill_value=0)
    return float(np.abs(got - np.einsum("ijk,jr,kr->ir", tv, Fs[1], Fs[2])).max())
err = [run(X, xv)]  # the engine's type, ops and kernels for R = 7 compile here
first = list(attempts)
gb.ss.config["jit_c_compiler_name"] = "/nonexistent/bin/cc"
gb.ss.config["jit_c_control"] = "on"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    err += [run(Xd, dv), run(Xd, dv)]  # a dense tensor: a new kernel, which cannot compile
print("max_err:", max(err))
print("attempts:", first, attempts[len(first):])
print("control:", gb.ss.config["jit_c_control"])
print("failure:", tensor_udt._FAILURE and tensor_udt._FAILURE[:2])
print("warnings:", sum("array-UDT engine" in str(w.message) for w in caught))
"""


@pytest.mark.parametrize("case", ["ttm", "mttkrp"])
def test_udt_engine_new_kernel_jit_failure(case, tmp_path):
    """The engine's ops for an R are compiled, then the compiler breaks, and a case needs a
    new kernel (a dense tensor): the JIT error is caught there too, for TTM and for the
    MTTKRP products' first step, and the products give the result (Numba blocked, so the
    UDT engine is the one tried).
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root), GRAPHBLAS_CACHE_PATH=str(tmp_path / "jit"))
    result = subprocess.run(
        [sys.executable, "-c", _JIT_NEW_KERNEL_SCRIPT, case],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=600,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines["package"] == str(repo_root / "graphblas" / "__init__.py"), report
    if "skip" in lines:
        pytest.skip(lines["skip"])
    assert float(lines["max_err"]) < 1e-10, report
    # Compiled and ran; then one attempt that failed, and none after
    assert lines["attempts"] == "['ok'] ['JitError']", report
    assert lines["control"] == "on", report
    assert lines["failure"] == "('on', 'load')", report
    assert lines["warnings"] == "1", report


def test_ttm_engine_choice(monkeypatch):
    """With both engines, the UDT one takes the stored order with a hypersparse output and
    R <= _TTM_UDT_MAX_R; the Numba one the rest. Without Numba, the UDT one takes all.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    pytest.importorskip("numba")
    _require_jit_compile()
    from graphblas.core import tensor, tensor_numba

    udt = _ttm_udt_spy(monkeypatch, engines=("udt", "numba"))
    nb = []
    orig = tensor_numba.ttm
    monkeypatch.setattr(tensor_numba, "ttm", lambda *a, **k: nb.append(1) or orig(*a, **k))
    rng = np.random.default_rng(97)
    # Few fibers: hypersparse output
    H, hv, hp = random_tensor((30, 40, 50), rng, density=0.002, layout=((0, 1), (2,)))
    # Most fibers: bitmap output
    B, bv, bp = random_tensor((10, 12, 50), rng, density=0.2, layout=((0, 1), (2,)))
    cases = [
        (H, hv, hp, 2, 4, "udt"),
        (H, hv, hp, 2, tensor._TTM_UDT_MAX_R + 1, "numba"),
        (H, hv, hp, 0, 4, "numba"),  # (a relayout)
        (B, bv, bp, 2, 4, "numba"),
    ]
    for X, xv, xp, mode, R, engine in cases:
        U = rng.random((X.shape[mode], R))
        udt.clear()
        nb.clear()
        check(X.tensordot(U, ([mode], [0])).new(), *_ttm_model(xv, xp, U, mode), approx=True)
        assert (bool(udt), bool(nb)) == (engine == "udt", engine == "numba"), (mode, R, engine)
    monkeypatch.setattr(tensor, "_has_numba", False)
    for X, xv, xp, mode, R, _ in cases:
        U = rng.random((X.shape[mode], R))
        udt.clear()
        check(X.tensordot(U, ([mode], [0])).new(), *_ttm_model(xv, xp, U, mode), approx=True)
        assert udt


def test_ttm_compacted(monkeypatch):
    """Without the kernel, TTM in the stored order on hypersparse rows runs one product on the
    rows compacted (h dropped, then given back): any dtype and semiring, as the product.
    """
    from graphblas.core import tensor

    monkeypatch.setattr(tensor, "_has_numba", False)
    calls = []
    orig = tensor._ttm_compacted

    def spy(*args, **kwargs):
        rv = orig(*args, **kwargs)
        calls.append(rv is not None)
        return rv

    monkeypatch.setattr(tensor, "_ttm_compacted", spy)
    rng = np.random.default_rng(81)
    shape = (60, 70, 30)
    for iso, dtype in [(False, dtypes.FP64), (True, dtypes.FP64), (False, dtypes.INT64)]:
        X, xv, xp = random_tensor(
            shape, rng, density=0.004, layout=((0, 1), (2,)), iso=iso, dtype=dtype
        )
        assert X.ss.format == "hypercsr"
        U = rng.random((30, 4))
        for kb in [0, 1]:
            calls.clear()
            R = X.tensordot(Matrix.from_dense(U if kb == 0 else U.T), ([2], [kb])).new()
            check(R, *_ttm_model(xv, xp, U, 2), approx=True)
            assert calls == [True]
    # A sparse factor and another semiring: the same as the product
    Us = Matrix.from_dense(U)
    del Us[3, 1]
    for factor, op in [(Us, gb.semiring.plus_times), (Matrix.from_dense(U), gb.semiring.max_times)]:
        got = X.tensordot(factor, ([2], [0]), op=op).new()
        with monkeypatch.context() as m:
            m.setattr(tensor, "_ttm_compacted", lambda *a, **k: None)
            # (close: the two kernels sum in different orders)
            assert got.isclose(X.tensordot(factor, ([2], [0]), op=op).new())
    # Not for rows that are mostly nonempty, nor outside the stored order
    Y, yv, yp = random_tensor((10, 11, 30), rng, density=0.1, layout=((0, 1), (2,)))
    calls.clear()
    check(
        Y.tensordot(Matrix.from_dense(U), ([2], [0])).new(), *_ttm_model(yv, yp, U, 2), approx=True
    )
    calls.clear()
    U1 = rng.random((70, 2))
    check(
        X.tensordot(Matrix.from_dense(U1), ([1], [0])).new(),
        *_ttm_model(xv, xp, U1, 1),
        approx=True,
    )
    assert calls == [False]


_KERNEL_CACHE_SCRIPT = """
import sys
import numpy as np
import graphblas as gb
from graphblas import Matrix
from graphblas.core import tensor, tensor_numba
from graphblas.core.tensor import Tensor
print("package:", gb.__file__)
tensor._TTM_KERNEL_MIN_NNZ = 0
rng = np.random.default_rng(0)
xv = np.where(rng.random((20, 30, 40)) < 0.01, rng.random((20, 30, 40)), 0)
X = Tensor.from_coo(np.nonzero(xv), xv[xv != 0], shape=xv.shape)
U = rng.random((40, 3))
got = X.tensordot(Matrix.from_dense(U), ([2], [0])).new().to_dense(fill_value=0)
print("max_err:", float(np.abs(got - np.tensordot(xv, U, ([2], [0]))).max()))
k = next(iter(tensor_numba._compiled.values()))
print("cache_hits:", sum(k.stats.cache_hits.values()))
"""


def _run_kernel_cache_script(cache):
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root), GRAPHBLAS_KERNEL_CACHE=cache)
    return subprocess.Popen(
        [sys.executable, "-c", _KERNEL_CACHE_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _kernel_cache_lines(proc):
    out, err = proc.communicate(timeout=600)
    assert proc.returncode == 0, f"stdout:\n{out}\nstderr:\n{err}"
    lines = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    assert float(lines["max_err"]) < 1e-12
    return lines


def test_kernel_disk_cache(tmp_path):
    """Generated kernels go to files in the cache directory, so a second process loads the
    compiled code from Numba's cache; two processes may fill one cache at once; a read-only or
    disabled cache compiles in memory.
    """
    pytest.importorskip("numba")
    cache = tmp_path / "kernels"
    # Two processes at once on an empty cache: both work
    procs = [_run_kernel_cache_script(str(cache)) for _ in range(2)]
    for proc in procs:
        _kernel_cache_lines(proc)
    files = sorted(p.name for p in cache.glob("k_*.py"))
    assert files
    assert not list(cache.glob("*.tmp"))
    # A later process loads the compiled code
    assert int(_kernel_cache_lines(_run_kernel_cache_script(str(cache)))["cache_hits"]) > 0
    # Read-only: compiles in memory, writes nothing
    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        assert int(_kernel_cache_lines(_run_kernel_cache_script(str(ro)))["cache_hits"]) == 0
        assert not list(ro.iterdir())
    finally:
        ro.chmod(0o755)
    # Disabled
    assert int(_kernel_cache_lines(_run_kernel_cache_script(""))["cache_hits"]) == 0


def test_kernel_disk_cache_guards(tmp_path, monkeypatch):
    """A cache file that is not the kernel's source (corrupt, tampered with, or a digest
    prefix collision) is neither run nor overwritten: the kernel compiles in memory. A failed
    rename leaves no temporary file behind.
    """
    pytest.importorskip("numba")
    from graphblas.core import tensor_numba

    source = (
        "def kernel(x):\n"
        "    s = 0.0\n"
        "    for i in prange(x.size):\n"
        "        s += x[i]\n"
        "    return s\n"
    )
    x = np.arange(10.0)
    monkeypatch.setenv(tensor_numba._CACHE_ENV, str(tmp_path / "kernels"))
    k = tensor_numba._compile(source)
    assert k(x) == 45
    assert type(k._cache).__name__ != "NullCache"
    [path] = (tmp_path / "kernels").glob("k_*.py")
    tampered = path.read_bytes().replace(b"s += x[i]", b"s += 2 * x[i]")
    path.write_bytes(tampered)
    k = tensor_numba._compile(source)
    assert k(x) == 45
    assert type(k._cache).__name__ == "NullCache"
    assert path.read_bytes() == tampered

    def fail(self, target):
        raise OSError("rename failed")

    monkeypatch.setenv(tensor_numba._CACHE_ENV, str(tmp_path / "other"))
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", fail)
        k = tensor_numba._compile(source)
    assert k(x) == 45
    assert type(k._cache).__name__ == "NullCache"
    assert list((tmp_path / "other").iterdir()) == []


@pytest.mark.parametrize("shape", [(9, 10, 11), (5, 6, 7, 8), (3, 4, 5, 4, 3)])
def test_mttkrp_products_udt_first(shape, monkeypatch):
    """Without the Numba kernel, the products' first step (a TTM over the last axis) runs
    on the array-UDT engine when the JIT is on: every mode and a sample of layouts, iso
    values, against numpy and the plain products; a sparse factor keeps the product.
    """
    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    _require_jit_compile()
    from graphblas.core import tensor

    calls = _ttm_udt_spy(monkeypatch)
    monkeypatch.setattr(tensor, "_has_numba", False)
    rng = np.random.default_rng(70 + len(shape))
    ndim = len(shape)
    dense = [rng.random((n, 3)) for n in shape]
    factors = [Matrix.from_dense(d) for d in dense]
    layouts = all_layouts(ndim)[:: {3: 1, 4: 17, 5: 173}[ndim]]
    for i, layout in enumerate(layouts):
        X, xv, xp = random_tensor(shape, rng, density=0.05, layout=layout, iso=i % 4 == 0)
        for mode in range(ndim):
            calls.clear()
            R = X.mttkrp(factors, mode).new()
            check(R, *_mttkrp_model(xv, xp, dense, mode), approx=True)
            assert calls, (layout, mode)
            with monkeypatch.context() as m:
                m.setattr(tensor, "_MTTKRP_ENGINES", ("products",))
                calls.clear()
                assert R.isclose(X.mttkrp(factors, mode).new(), rel_tol=1e-12)
                assert calls == []
    sparse = [f.dup() for f in factors]
    for f in sparse:
        del f[0, 0]
    calls.clear()
    sdense = [f.to_dense(fill_value=0.0) for f in sparse]
    check(X.mttkrp(sparse, 0).new(), *_mttkrp_model(xv, xp, sdense, 0), approx=True)
    assert calls == []


def test_mttkrp_without_numba():
    """With Numba blocked, the Tensor imports and MTTKRP runs the GraphBLAS products."""
    script = """
import sys
sys.modules["numba"] = None
import numpy as np
import graphblas as gb
from graphblas.core import _has_numba
from graphblas.core.tensor import Tensor
print("package:", gb.__file__)
rng = np.random.default_rng(0)
err = 0.0
for shape, sub in [((6, 7, 8), "ijk,ir,kr->jr"), ((5, 6, 7, 8), "ijkl,ir,kr,lr->jr")]:
    xv = np.where(rng.random(shape) < 0.3, rng.random(shape), 0)
    X = Tensor.from_coo(np.nonzero(xv), xv[xv != 0], shape=xv.shape)
    F = [rng.random((n, 3)) for n in xv.shape]
    got = X.mttkrp(F, 1).new().to_dense(fill_value=0)
    expected = np.einsum(sub, xv, *[F[a] for a in range(len(shape)) if a != 1])
    err = max(err, float(np.abs(got - expected).max()))
print("has_numba:", _has_numba, "numba_module:", "graphblas.core.tensor_numba" in sys.modules)
print("max_err:", err)
"""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(repo_root)),
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines["package"] == str(repo_root / "graphblas" / "__init__.py"), report
    assert lines["has_numba"] == "False numba_module: False", report
    assert float(lines["max_err"]) < 1e-12, report


def test_unfold():
    rng = np.random.default_rng(51)
    X, xv, _ = random_tensor((3, 4, 5), rng, layout=((2,), (1, 0)))
    for mode in range(3):
        M = X.unfold(mode)
        np.testing.assert_allclose(
            M.to_dense(0.0), np.moveaxis(xv, mode, 0).reshape(xv.shape[mode], -1)
        )
        M = X.unfold(mode, order="F")
        np.testing.assert_allclose(
            M.to_dense(0.0), np.moveaxis(xv, mode, 0).reshape(xv.shape[mode], -1, order="F")
        )
    assert X.unfold(2) is not X.ss.matrix
    with pytest.raises(ValueError, match="order"):
        X.unfold(0, order="X")


@pytest.mark.parametrize("layout", all_layouts(3)[::2])
def test_ttv(layout):
    rng = np.random.default_rng(52)
    X, xv, xp = random_tensor((3, 4, 5), rng, layout=layout, density=0.5)
    for axis in range(3):
        w, wv, wp = random_tensor((X.shape[axis],), rng, density=0.8)
        rest = "ijk".replace("ijk"[axis], "")
        sub = f"ijk,{'ijk'[axis]}->{rest}"
        check(
            X.tensordot(w, ([axis], [0])).new(),
            *np_einsum(sub, (X, xv, xp), (w, wv, wp)),
            approx=True,
        )
        sub2 = f"{'ijk'[axis]},ijk->{rest}"
        check(
            w.tensordot(X, ([0], [axis])).new(),
            *np_einsum(sub2, (w, wv, wp), (X, xv, xp)),
            approx=True,
        )
        # a non-commutative multiply: sum of (x - w) over the present pairs
        R = X.tensordot(w, ([axis], [0]), op=gb.semiring.plus_minus).new()
        diff = np.moveaxis(xv, axis, -1) - wv
        both = np.moveaxis(xp, axis, -1) & wp
        check(R, np.where(both, diff, 0).sum(-1), both.any(-1), approx=True)
        R = w.tensordot(X, ([0], [axis]), op=gb.semiring.plus_minus).new()
        check(R, np.where(both, -diff, 0).sum(-1), both.any(-1), approx=True)
        # Vector and ndarray operands
        v = Vector.from_dense(wv)
        check(
            X.tensordot(v, ([axis], [0])).new(),
            *np_einsum(sub, (X, xv, xp), (v, wv, np.ones_like(wp))),
            approx=True,
        )
        check(
            X.tensordot(wv, ([axis], [0])).new(),
            *np_einsum(sub, (X, xv, xp), (v, wv, np.ones_like(wp))),
            approx=True,
        )


# --------------------------------------------------------------------------------------------
# Phase 2a.6: indexing, from_dense/to_dense, from_matrices/to_matrices


def np_outer_index(arr, key):
    key = key if isinstance(key, tuple) else (key,)
    if any(k is Ellipsis for k in key):
        at = next(i for i, k in enumerate(key) if k is Ellipsis)
        key = key[:at] + (slice(None),) * (arr.ndim - len(key) + 1) + key[at + 1 :]
    key = key + (slice(None),) * (arr.ndim - len(key))
    lists, drop = [], []
    for a, k in enumerate(key):
        if isinstance(k, (int, np.integer)):
            lists.append([k % arr.shape[a]])
            drop.append(a)
        elif isinstance(k, slice):
            lists.append(list(range(*k.indices(arr.shape[a]))))
        else:
            lists.append([i % arr.shape[a] for i in k])
    out = arr[np.ix_(*lists)]
    return out.reshape([n for a, n in enumerate(out.shape) if a not in drop])


INDEX_KEYS = [
    (1, slice(None), slice(None)),
    (slice(None), 2, slice(None)),
    (slice(None), slice(None), 3),
    (slice(1, 3),),
    (slice(None), slice(1, 3)),
    (Ellipsis, slice(0, 5, 2)),
    (1, 2, 3),
    ([0, 2], slice(None), [4, 0]),
    (-1, slice(None, None, -1)),
    (slice(1, 2), 3, Ellipsis),
    (np.array([2, 0]), np.array([1, 1, 3])),
    2,
]


@pytest.mark.parametrize("layout", all_layouts(3)[::3])
def test_indexing(layout, monkeypatch):
    rng = np.random.default_rng(53)
    X, xv, xp = random_tensor((3, 4, 5), rng, layout=layout)
    for key in INDEX_KEYS:
        R = X[key].new()
        check(R, np_outer_index(xv, key), np_outer_index(xp, key))
    # the coordinate filter used when the positions are too many to list
    from graphblas.core import tensor as tensor_mod

    monkeypatch.setattr(tensor_mod, "_MAX_LISTED", 0)
    for key in INDEX_KEYS[7:11]:
        check(X[key].new(), np_outer_index(xv, key), np_outer_index(xp, key))
    monkeypatch.undo()
    with pytest.raises(IndexOutOfBound):
        X[3]
    with pytest.raises(IndexOutOfBound):
        X[:, [0, 4]]
    with pytest.raises(IndexError, match="too many"):
        X[0, 0, 0, 0]
    with pytest.raises(IndexError, match="ellipsis"):
        X[..., 0, ...]
    with pytest.raises(NotImplementedError, match="expand_dims"):
        X[None]
    with pytest.raises(TypeError, match="boolean"):
        X[np.array([True, False, True])]
    with pytest.raises(TypeError, match="boolean"):
        X[np.array([True, False, True])] = 1  # (assignment takes the same keys)


def test_indexing_hypersparse_slab():
    rng = np.random.default_rng(54)
    shape = (2**19,) * 3
    coords = [rng.integers(0, 2**19, 2000) for _ in range(3)]
    coords[0][:50] = 7
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    T = Tensor.from_coo(coords, 1.0, shape=shape, layout=((0, 1), (2,)))
    R = T[7].new()
    assert R.shape == (2**19, 2**19)
    assert R.nvals == int((coords[0] == 7).sum())
    R = T[5:9, :, 100:2000].new()
    keep = (coords[0] >= 5) & (coords[0] < 9) & (coords[2] >= 100) & (coords[2] < 2000)
    assert R.nvals == int(keep.sum())


def test_dense_and_matrices():
    rng = np.random.default_rng(55)
    d = rng.random((3, 4, 5))
    for layout in [None, ((2,), (1, 0)), ((0, 1, 2), ())]:
        T = Tensor.from_dense(d, layout=layout)
        T._check_invariants()
        assert T.nvals == d.size
        np.testing.assert_array_equal(T.to_dense(), d)
    d0 = np.where(d < 0.5, 0, d)
    T = Tensor.from_dense(d0, missing_value=0)
    check(T, d0, d0 != 0)
    with pytest.raises(TypeError, match="fill_value"):
        T.to_dense()
    np.testing.assert_array_equal(T.to_dense(0), d0)
    assert Tensor.from_dense(np.float64(3.0)).to_scalar().value == 3.0
    mats = [Matrix.from_dense(rng.random((3, 4))) for _ in range(5)]
    mats[1] = mats[1].T.new().T  # a TransposedMatrix
    for axis in range(3):
        T = Tensor.from_matrices(mats, axis=axis)
        T._check_invariants()
        ref = np.stack(
            [m.new().to_dense() if hasattr(m, "_matrix") else m.to_dense() for m in mats], axis=axis
        )
        np.testing.assert_allclose(T.to_dense(), ref)
        back = T.to_matrices(axis)
        assert len(back) == 5
        for b, m in zip(back, mats, strict=True):
            assert b.isequal(m.new() if hasattr(m, "_matrix") else m)
    with pytest.raises(TypeError, match="Matrices"):
        Tensor.from_matrices([])
    with pytest.raises(DimensionMismatch):
        Tensor.from_matrices([Matrix(float, 2, 3), Matrix(float, 3, 2)])
    with pytest.raises(ValueError, match="3-d"):
        Tensor(float, 2, 3).to_matrices()
    assert Tensor(float, 0, 3, 4).to_matrices() == []
    E = Tensor(float, 2, 0, 4).to_matrices()
    assert len(E) == 2
    assert E[0].shape == (0, 4)


def test_scalar_comparisons_and_pow():
    rng = np.random.default_rng(56)
    X, xv, xp = random_tensor((3, 4, 5), rng)
    check((X > 0).new(), np.where(xp, xv > 0, False), xp)
    check((0 < X).new(), np.where(xp, xv > 0, False), xp)
    check((X == 1).new(), np.where(xp, xv == 1, False), xp)
    check((X**2).new(), np.where(xp, xv**2, 0), xp, approx=True)
    check((2**X).new(), np.where(xp, 2.0**xv, 0), xp, approx=True)
    assert hash(X) == hash(X)  # comparisons do not make Tensors unhashable


def test_mttkrp_large_dims():
    """An intermediate past 2**31 columns takes the copying reshape (SS 10.5.1 bug)."""
    rng = np.random.default_rng(57)
    shape = (3, 2**29, 5)
    coords = [rng.integers(0, n, 40) for n in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = rng.random(lin.size)
    X = Tensor.from_coo(coords, vals, shape=shape)
    # The same data with the used j relabeled 0..m-1, for numpy
    uj, jinv = np.unique(coords[1], return_inverse=True)
    xs = np.zeros((3, uj.size, 5))
    xs[coords[0], jinv, coords[2]] = vals
    bs = rng.random((uj.size, 8))
    B = Matrix.from_coo(
        np.repeat(uj, 8), np.tile(np.arange(8), uj.size), bs.ravel(), nrows=2**29, ncols=8
    )
    C = rng.random((5, 8))
    R = X.mttkrp([None, B, C], 0).new()
    expected = np.einsum("ijk,jr,kr->ir", xs, bs, C)
    present = np.zeros(3, bool)
    present[coords[0]] = True
    check(
        R, np.where(present[:, None], expected, 0), np.repeat(present[:, None], 8, 1), approx=True
    )


def _small_tensor(shape, rng, layout=None):
    """random_tensor, also for 0-d shapes."""
    if shape:
        return random_tensor(shape, rng, layout=layout, density=0.6)
    if rng.random() < 0.8:
        value = np.round(rng.random() * 10 - 5, 1)
        return Tensor.from_coo((), value), np.array(value), np.array(True)
    return Tensor(float), np.array(0.0), np.array(False)


def _shapes_12(ndim):
    return list(itertools.product((1, 2), repeat=ndim))


@pytest.mark.parametrize("xndim", [1, 2, 3])
def test_broadcast_sweep(xndim):
    """Every shape pair over sizes {1, 2} (lower rank and 0-d too), both operand orders,
    several layouts: a lower-rank operand whose missing axes are size 1 still covers the
    result (broadcast factor 1) and once took the wrong code path.
    """
    rng = np.random.default_rng(58)
    ops = [
        ("mult", binary.times, np.multiply),
        ("mult", binary.minus, np.subtract),
        ("mult", binary.lt, np.less),
        ("add", binary.plus, np.add),
        ("union", binary.minus, np.subtract),
    ]
    for xshape in _shapes_12(xndim):
        xlayouts = all_layouts(xndim)
        for li in {0, len(xlayouts) // 2, len(xlayouts) - 1}:
            X, xv, xp = _small_tensor(xshape, rng, layout=xlayouts[li])
            for yndim in range(xndim + 1):
                for yshape in _shapes_12(yndim):
                    ylayouts = all_layouts(yndim)
                    ylay = ylayouts[rng.integers(len(ylayouts))] if yndim else None
                    Y, yv, yp = _small_tensor(yshape, rng, layout=ylay)
                    for kind, op, ufunc in ops:
                        for (A, av, ap), (B, bv, bp) in [
                            ((X, xv, xp), (Y, yv, yp)),
                            ((Y, yv, yp), (X, xv, xp)),
                        ]:
                            if kind == "mult":
                                R = A.ewise_mult(B, op).new()
                                ev, ep = np_bcast("mult", av, ap, bv, bp, ufunc)
                            elif kind == "add":
                                R = A.ewise_add(B, op).new()
                                ev, ep = np_bcast("add", av, ap, bv, bp, ufunc)
                            else:
                                R = A.ewise_union(B, op, 1, -1).new()
                                ev, ep = np_bcast("union", av, ap, bv, bp, ufunc, 1, -1)
                            check(R, ev, ep, approx=R.dtype != dtypes.BOOL)


def test_relayout_rejects_layout_of_another_rank():
    T = Tensor.from_coo(([0], [2]), [5.0], shape=(1, 4))
    with pytest.raises(ValueError, match="does not match"):
        tl.relayout(T.ss.matrix, (1, 1, 4), T.layout, Layout((0,), (1, 2)))


# --------------------------------------------------------------------------------------------
# Phase 2b: planner pricing and strategy choices


def test_nonempty_row_ids_and_perm_pricing():
    rng = np.random.default_rng(60)
    r = np.array([1, 1, 5, 9, 9, 9])
    c = np.array([0, 2, 1, 0, 1, 3])
    for fmt in ["sparse", "hypersparse", "bitmap"]:
        M = Matrix.from_coo(r, c, 1.0, nrows=12, ncols=4)
        M.ss.config["sparsity_control"] = fmt
        np.testing.assert_array_equal(tl.nonempty_row_ids(M), [1, 5, 9])
        M2 = M.dup()
        assert M2.isequal(M)
    assert tl.nonempty_row_ids(Matrix(float, 5, 3)).size == 0
    assert tl.expected_rows(100, 0) == 0
    assert tl.expected_rows(10, 1000) == pytest.approx(10)
    assert tl.expected_rows(10**8, 10**7) == pytest.approx(9.5e6, rel=0.01)
    # skew (1e5, 1e3, 10) at 1e7, permuted (1, 0, 2): PERM_R over 1e8 rows of about one entry
    # each once won on price (26 ms predicted) and took 3.5 s; it must cost more than a REBUILD
    shape = (1000, 100_000, 10)
    cost, path, _ = tl.plan(shape, ((1, 0, 2), 1), frozenset([((0, 1, 2), 1)]), 10**7, 8)
    assert cost < tl.rebuild_cost(10**7, 0)
    for state, edge in zip(_states_along(((1, 0, 2), 1), path), path, strict=True):
        if edge[0] == "PERM_R":
            assert tl.prod(shape, state[0][: state[1]]) <= 4 * 10**7, path
    # A small version runs through the same plan and gives the REBUILD result
    X, xv, xp = random_tensor((40, 300, 3), rng, density=0.05)
    Xv = X.permute_dims((1, 0, 2))
    check(
        Xv.relayout(tl.default_layout(Xv.shape, X.nvals)),
        xv.transpose(1, 0, 2),
        xp.transpose(1, 0, 2),
    )


def _states_along(state, path):
    out = []
    for edge in path:
        out.append(state)
        state = tl._final_state(state, [edge])
    return out


def test_default_split_follows_nonempty_rows(monkeypatch):
    from graphblas.core import tensor

    rng = np.random.default_rng(61)
    # Every (i, j) row holds 5 entries: the rows are well filled, so (0,1|2)
    shape = (10, 12, 50)
    i, j = np.meshgrid(np.arange(10), np.arange(12), indexing="ij")
    i, j = np.repeat(i.ravel(), 5), np.repeat(j.ravel(), 5)
    k = np.concatenate([rng.choice(50, 5, replace=False) for _ in range(120)])
    T = Tensor.from_coo((i, j, k), 1.0, shape=shape)
    assert T.layout == Layout((0, 1), (2,))
    T._check_invariants()
    # About one entry per (i, j) row: (0|1,2)
    coords = [rng.integers(0, n, 30) for n in (40, 50, 60)]
    lin = np.unique(np.ravel_multi_index(coords, (40, 50, 60)))
    T = Tensor.from_coo(np.unravel_index(lin, (40, 50, 60)), 1.0, shape=(40, 50, 60))
    assert T.layout == Layout((0,), (1, 2))
    T._check_invariants()
    # Shuffled input is judged by a sample of its row keys (then settled): the same layouts
    order = rng.permutation(lin.size)
    shuffled = [c[order] for c in np.unravel_index(lin, (40, 50, 60))]
    T = Tensor.from_coo(shuffled, 1.0, shape=(40, 50, 60))
    assert T.layout == Layout((0,), (1, 2))
    T._check_invariants()
    order = rng.permutation(i.size)
    T = Tensor.from_coo((i[order], j[order], k[order]), 1.0, shape=shape)
    assert T.layout == Layout((0, 1), (2,))
    # Hypersparse rows need more entries each: 4 per row over 1/20 of the rows is too few
    shape = (40, 50, 100)
    ij = rng.choice(2000, 100, replace=False)
    i2, j2 = np.repeat(ij // 50, 4), np.repeat(ij % 50, 4)
    k2 = np.concatenate([rng.choice(100, 4, replace=False) for _ in range(100)])
    T = Tensor.from_coo((i2, j2, k2), 1.0, shape=shape)
    assert T.layout == Layout((0,), (1, 2))
    T._check_invariants()
    # About 2 entries per (i, j) row: thin rows are kept only when (0|1,2)'s column group
    # would be wider than the data (400 * 2000 positions for 2e5 entries), not when it is
    # narrower (400 * 50); exact, and with the sampled check first
    for past in [2**23, 2**16]:
        monkeypatch.setattr(tensor, "_SPLIT_SAMPLE_PAST", past)
        for big, expected in [
            ((300, 400, 2000), Layout((0, 1), (2,))),
            ((300, 400, 50), Layout((0,), (1, 2))),
        ]:
            lin = np.unique(rng.integers(0, math.prod(big), 200_000))
            T = Tensor.from_coo(np.unravel_index(lin, big), 1.0, shape=big)
            assert T.layout == expected
    # 4-d moves the split one axis at a time
    coords = [rng.integers(0, n, 40) for n in (5, 6, 7, 8)]
    lin = np.unique(np.ravel_multi_index(coords, (5, 6, 7, 8)))
    T = Tensor.from_coo(np.unravel_index(lin, (5, 6, 7, 8)), 1.0, shape=(5, 6, 7, 8))
    T._check_invariants()
    assert T.layout.seq == (0, 1, 2, 3)
    np.testing.assert_array_equal(T.to_coo()[0][0], np.unravel_index(lin, (5, 6, 7, 8))[0])


def _hyper_rows_tensor(rng, shape, n):
    coords = [rng.integers(0, s, n) for s in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = np.round(rng.random(lin.size) * 10 - 5, 1)
    T = Tensor.from_coo(coords, vals, shape=shape, layout=((0, 1), (2,)))
    T.ss.matrix.ss.config["sparsity_control"] = "hypersparse"
    T._store.exported = False
    values = np.zeros(shape)
    present = np.zeros(shape, bool)
    values[coords] = vals
    present[coords] = True
    return T, values, present


def test_hypersparse_row_group_paths():
    """Reductions, TTV and broadcasts over a hypersparse row group use selectors built from
    its nonempty rows (sized by the data, not the group's positions).
    """
    rng = np.random.default_rng(62)
    X, xv, xp = _hyper_rows_tensor(rng, (60, 70, 5), 150)
    assert X.ss.format == "hypercsr"
    for axis in [0, 1, (0, 2), (1, 2), 2, (0, 1)]:
        check(X.sum(axis).new(), *np_reduce("plus", xv, xp, axis), approx=True)
        check(X.reduce(monoid.max, axis).new(), *np_reduce("max", xv, xp, axis))
    for axis in range(3):
        w, wv, wp = random_tensor((X.shape[axis],), rng, density=0.7)
        rest = "ijk".replace("ijk"[axis], "")
        sub = f"ijk,{'ijk'[axis]}->{rest}"
        check(
            X.tensordot(w, ([axis], [0])).new(),
            *np_einsum(sub, (X, xv, xp), (w, wv, wp)),
            approx=True,
        )
        R = w.tensordot(X, ([0], [axis]), op=gb.semiring.plus_minus).new()
        diff = np.moveaxis(xv, axis, -1) - wv
        both = np.moveaxis(xp, axis, -1) & wp
        check(R, np.where(both, -diff, 0).sum(-1), both.any(-1), approx=True)
    for yshape in [(60, 1, 1), (70, 1), (60, 70, 1), (5,)]:
        Y, yv, yp = random_tensor(yshape, rng, density=0.7)
        check((X * Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.multiply), approx=True)
        check(
            Y.ewise_mult(X, binary.minus).new(),
            *np_bcast("mult", yv, yp, xv, xp, np.subtract),
            approx=True,
        )


@pytest.mark.parametrize("nthreads", [1, 8])
def test_wide_column_group_paths(nthreads):
    """A column group far wider than the data: selector, split move or REBUILD, by cost."""
    rng = np.random.default_rng(63)
    shape = (3, 400, 500)
    coords = [rng.integers(0, s, 300) for s in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = rng.random(lin.size)
    X = Tensor.from_coo(coords, vals, shape=shape, layout=((0,), (1, 2)))
    xv = np.zeros(shape)
    xp = np.zeros(shape, bool)
    xv[coords] = vals
    xp[coords] = True
    for axis in [0, 1, 2, (1, 2), (0, 2)]:
        check(X.sum(axis).new(nthreads=nthreads), *np_reduce("plus", xv, xp, axis), approx=True)
    for axis in range(3):
        w, wv, wp = random_tensor((shape[axis],), rng, density=0.7)
        rest = "ijk".replace("ijk"[axis], "")
        sub = f"ijk,{'ijk'[axis]}->{rest}"
        R = X.tensordot(w, ([axis], [0])).new(nthreads=nthreads)
        check(R, *np_einsum(sub, (X, xv, xp), (w, wv, wp)), approx=True)
    for yshape in [(400, 1), (500,), (3, 1, 1)]:
        Y, yv, yp = random_tensor(yshape, rng, density=0.7)
        R = (X * Y).new(nthreads=nthreads)
        check(R, *np_bcast("mult", xv, xp, yv, yp, np.multiply), approx=True)


def test_transpose_thread_choices():
    """Into many rows a direct transpose wins at 1 thread (radix chains pay a reshape per
    output row) and runs on 2 threads when that beats the 8-thread price.
    """
    shape = (1000, 1000, 1000)
    src = ((2, 0, 1), 1)  # (k | i,j)
    dst = frozenset([((0, 1, 2), 2)])  # (i,j | k): a transpose into 1e6 rows
    _, path1, _ = tl.plan(shape, src, dst, 10**7, 1)
    assert path1 == [("T", None)]
    _, path8, _ = tl.plan(shape, src, dst, 10**7, 8)
    assert [e[0] for e in path8] == ["RS", "T", "RS", "T", "RS"]
    # A wide transpose with no cheap chain (2-d): 2 threads
    _, path, _ = tl.plan((1000, 10**6), ((0, 1), 1), frozenset([((1, 0), 1)]), 10**7, 8)
    assert path == [("T", 2)]
    M = Matrix.from_coo([0, 1], [5, 3], [1.0, 2.0], nrows=2, ncols=6)
    assert tl.transpose(M, nthreads=2).isequal(M.T.new())


def test_broadcast_gather_path():
    """A full small operand along axes inside a wide group: its values are gathered onto the
    stored entries, then one same-pattern ewise_mult.
    """
    rng = np.random.default_rng(64)
    shape = (8, 300, 400)
    coords = [rng.integers(0, s, 200) for s in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = rng.random(lin.size)
    xv = np.zeros(shape)
    xp = np.zeros(shape, bool)
    xv[coords] = vals
    xp[coords] = True
    for layout in [((0,), (1, 2)), ((0, 1), (2,)), ((2, 0), (1,))]:
        X = Tensor.from_coo(coords, vals, shape=shape, layout=layout)
        for yshape in [(300, 1), (400,), (8, 1, 1), (300, 400), (8, 300, 1)]:
            yv = rng.integers(1, 5, yshape)
            Y = Tensor.from_dense(yv)
            yp = np.ones(yshape, bool)
            check((X * Y).new(), *np_bcast("mult", xv, xp, yv, yp, np.multiply), approx=True)
            R = Y.ewise_mult(X, binary.minus).new()
            check(R, *np_bcast("mult", yv, yp, xv, xp, np.subtract), approx=True)
    W = tl.gather_operand(X.ss.matrix, shape, X.layout, (1,), np.arange(300.0), dtypes.FP64)
    assert W.nvals == X.nvals
    assert (
        tl.gather_operand(
            Matrix(float, 3, 3), (3, 3), Layout((0,), (1,)), (0,), np.ones(3), dtypes.FP64
        )
        is None
    )


def test_extract_filter_path():
    """A slab along an axis inside a wide column group filters the stored coordinates
    instead of listing every column index (or both ways agree).
    """
    rng = np.random.default_rng(65)
    shape = (6, 2**12, 2**10)
    coords = [rng.integers(0, s, 300) for s in shape]
    lin = np.unique(np.ravel_multi_index(coords, shape))
    coords = np.unravel_index(lin, shape)
    vals = rng.random(lin.size)
    X = Tensor.from_coo(coords, vals, shape=shape, layout=((0,), (1, 2)))
    dense = np.zeros(shape)
    present = np.zeros(shape, bool)
    dense[coords] = vals
    present[coords] = True
    for key in [
        (slice(None), slice(None), 7),
        (slice(None), 5, slice(None)),
        (2, slice(None), [3, 9]),
        (slice(None), [1, 4000], slice(10, 20)),
    ]:
        check(X[key].new(), np_outer_index(dense, key), np_outer_index(present, key))
    # Keys that index only axes of a hypersparse row group pick rows from h (when its rows
    # hold enough entries to beat the filter): ranges, steps, integers, an empty result
    shape = (300, 400, 50)
    ij = rng.choice(300 * 400, 600, replace=False)
    k = np.concatenate([rng.choice(50, 20, replace=False) for _ in range(600)])
    coords = (np.repeat(ij // 400, 20), np.repeat(ij % 400, 20), k)
    vals = np.round(rng.random(k.size) * 10 - 5, 1)
    X = Tensor.from_coo(coords, vals, shape=shape, layout=((0, 1), (2,)))
    assert X.ss.format == "hypercsr"
    xv = np.zeros(shape)
    xp = np.zeros(shape, bool)
    xv[coords] = vals
    xp[coords] = True
    for key in [
        (slice(None), slice(40, 80), slice(None)),
        (slice(10, 200, 3), slice(None, None, -7), slice(None)),
        (slice(None), 37, slice(None)),
        (12, slice(100, 300), slice(None)),
        (5, 6, slice(None)),
        (slice(0, 0), slice(None), slice(None)),
    ]:
        for nthreads in (8, 1):
            R = X[key].new(nthreads=nthreads)
            check(R, np_outer_index(xv, key), np_outer_index(xp, key))


def _random_reshape(shape, rng):
    """A random target shape of the same size: the sizes' prime factors regrouped, with
    some size-1 axes, up to 6 axes.
    """
    size = math.prod(shape)
    if size == 0:
        return tuple(int(n) for n in rng.permutation([0, *([1] * int(rng.integers(0, 3)))]))
    factors = []
    for n in shape:
        p = 2
        while n > 1:
            while n % p == 0:
                factors.append(p)
                n //= p
            p += 1
    out = [1] * int(rng.integers(1, 5))
    for f in factors:
        out[int(rng.integers(0, len(out)))] *= f
    for _ in range(int(rng.integers(0, 2))):
        out.insert(int(rng.integers(0, len(out) + 1)), 1)
    return tuple(out[:6]) if math.prod(out[:6]) == size else tuple(out)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_reshape_random(ndim):
    """``reshape`` against numpy: random shapes (size-1 and size-0 axes too), every layout
    (a sample at 4-5-d), views, and target shapes that split and fuse runs across the
    groups; a view exactly when the layout holds the result.
    """
    from graphblas.core.tensor import _reshape_layout

    rng = np.random.default_rng(100 + ndim)
    layouts = all_layouts(ndim)[:: {1: 1, 2: 1, 3: 1, 4: 7, 5: 61}[ndim]]
    for i, layout in enumerate(layouts):
        shape = tuple(int(n) for n in rng.choice([1, 2, 3, 4, 6], ndim))
        if i % 11 == 5:
            shape = (*shape[:-1], 0)
        X, xv, xp = random_tensor(shape, rng, density=0.4, layout=layout, iso=i % 6 == 0)
        views = [tuple(range(ndim))]
        if ndim > 1:
            views.append(tuple(int(a) for a in rng.permutation(ndim)))
        for perm in views:
            V = X.permute_dims(perm)
            vv, vp = xv.transpose(perm), xp.transpose(perm)
            for _ in range(4):
                new = _random_reshape(V.shape, rng)
                R = V.reshape(new)
                check(R, vv.reshape(new), vp.reshape(new))
                if math.prod(new) > 0:
                    holds = _reshape_layout(V.shape, V.layout, new) is not None
                    assert (R._M is V._M) == holds, (V.layout, V.shape, new)
                    if not holds:
                        with pytest.raises(ValueError, match="copy=False"):
                            V.reshape(new, copy=False)
                C = V.reshape(new, copy=True)
                check(C, vv.reshape(new), vp.reshape(new))
                assert C._M is not V._M


def test_reshape_views_and_moves():
    """Splits and fuses within one group are views (O(1), sharing the data copy-on-write);
    across the groups, or out of order, the data moves once to a healthy layout.
    """
    rng = np.random.default_rng(7)
    X, xv, xp = random_tensor((4, 6, 10), rng, density=0.3, layout=((0, 1), (2,)))
    for new, view in [
        ((24, 10), True),
        ((2, 2, 6, 10), True),
        ((4, 6, 2, 5), True),
        ((1, 4, 6, 1, 10), True),
        ((4, 60), False),
        ((10, 24), False),
        ((240,), False),
    ]:
        R = X.reshape(new)
        check(R, xv.reshape(new), xp.reshape(new))
        assert (R._M is X._M) == view, new
        if not view:
            assert R.layout.rows and R.layout.cols or R.ndim == 1, R.layout
    # Out of order in its group: moves
    Y, yv, yp = random_tensor((4, 6, 10), rng, density=0.3, layout=((1, 0), (2,)))
    R = Y.reshape(24, 10)
    check(R, yv.reshape(24, 10), yp.reshape(24, 10))
    assert R._M is not Y._M
    assert Y.reshape(2, 2, 6, 10)._M is Y._M  # (axis 0 split in place)
    # Copy-on-write: changing the view leaves the source alone, and the reverse
    V = X.reshape(24, 10)
    V << V.apply(gb.unary.ainv)
    check(X, xv, xp)
    check(V, -xv.reshape(24, 10), xp.reshape(24, 10))
    W = X.reshape(2, 2, 6, 10)
    X << X.apply(gb.unary.ainv)
    check(W, xv.reshape(2, 2, 6, 10), xp.reshape(2, 2, 6, 10))
    # ravel, flatten, -1, the argument forms
    check(X.ravel(), -xv.ravel(), xp.ravel())
    assert X.ravel()._M is not X._M  # (24 x 10 is not one row)
    F = X.flatten()
    check(F, -xv.ravel(), xp.ravel())
    assert X.reshape((2, -1, 10)).shape == X.reshape(2, -1, 10).shape == (2, 12, 10)
    # 0-d and size-1
    S = Tensor.from_coo((), [5.0], shape=())
    check(S.reshape(1, 1, 1), np.full((1, 1, 1), 5.0), np.ones((1, 1, 1), bool))
    check(S.reshape(1, 1).reshape(()), np.array(5.0), np.array(True))


def test_reshape_errors():
    X = Tensor.from_coo(np.array([[0, 1], [1, 2], [2, 3]]), [1.0, 2.0], shape=(2, 3, 4))
    with pytest.raises(ValueError, match="cannot reshape a Tensor of size 24 into shape"):
        X.reshape(5, 5)
    with pytest.raises(ValueError, match="one unknown"):
        X.reshape(-1, -1)
    with pytest.raises(ValueError, match="negative"):
        X.reshape(-2, -12)
    with pytest.raises(ValueError, match="size 24"):
        X.reshape(-1, 5)
    with pytest.raises(TypeError, match="ints"):
        X.reshape(2.0, 12)
    E = Tensor(dtypes.FP64, (0, 3))
    with pytest.raises(ValueError, match="size 0"):
        E.reshape(-1, 0)
    assert E.reshape(3, 0, 5).shape == (3, 0, 5)
    assert E.reshape(3, 0, copy=False).nvals == 0


def test_user_guide_examples():
    """Every example on the docs' Tensors page (``docs/user_guide/tensor.rst``) runs and
    prints what the page shows.
    """
    import doctest

    if backend != "suitesparse":
        pytest.skip("SuiteSparse only")
    pytest.importorskip("scipy")
    path = Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "tensor.rst"
    if not path.exists():  # pragma: no cover (an installed package has no docs)
        pytest.skip("no docs directory")
    failed, attempted = doctest.testfile(
        str(path),
        module_relative=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert attempted >= 40
    assert failed == 0


def test_expression_methods():
    """A TensorExpression offers the Tensor's methods on its computed value, as Matrix
    expressions do: ``(X * X).sum()`` with autocompute on; without it, the TypeError that
    says to call ``.new()``, as for Matrix.
    """
    from graphblas.core.tensor import TensorExpression

    rng = np.random.default_rng(13)
    X, _, _ = random_tensor((3, 4, 5), rng, density=0.4)
    F = [rng.random((n, 2)) for n in X.shape]
    calls = {
        "to_dense": ((), {"fill_value": 0}),
        "to_matrix": ((), {}),
        "to_scipy_sparse": ((), {}),
        "isclose": ((X * X,), {}),
        "permute_dims": (((2, 0, 1),), {}),
        "expand_dims": ((1,), {}),
        "squeeze": ((), {}),
        "reshape": ((12, 5), {}),
        "ravel": ((), {}),
        "flatten": ((), {}),
        "ewise_add": ((X,), {}),
        "ewise_mult": ((X,), {}),
        "ewise_union": ((X, binary.plus, 0, 0), {}),
        "apply": ((gb.unary.ainv,), {}),
        "select": ((gb.select.valuegt, 1.0), {}),
        "reduce": ((monoid.plus,), {"axis": 1}),
        "reduce_scalar": ((monoid.max,), {}),
        "sum": ((), {"axis": (0, 2)}),
        "prod": ((), {}),
        "min": ((), {"axis": 0}),
        "max": ((), {}),
        "any": ((), {}),
        "all": ((), {}),
        "tensordot": ((F[2], ([2], [0])), {}),
        "matmul": ((Tensor.from_dense(rng.random((3, 5, 2))),), {}),
        "mttkrp": ((F, 0), {}),
        "unfold": ((1,), {}),
    }

    def value(rv):
        if isinstance(rv, TensorExpression) or type(rv).__name__.endswith(("Expr", "Expression")):
            rv = rv.new()
        if isinstance(rv, Tensor):
            return rv.to_dense(fill_value=0.0)
        if isinstance(rv, Matrix):
            return rv.to_dense(fill_value=0.0)
        if hasattr(rv, "toarray"):
            return rv.toarray()
        return rv

    E = X * X
    D = E.new()
    with gb.config.set(autocompute=True):
        for name, (args, kwargs) in calls.items():
            got = value(getattr(E, name)(*args, **kwargs))
            want = value(getattr(D, name)(*args, **kwargs))
            np.testing.assert_array_equal(got, want, err_msg=name)
        np.testing.assert_array_equal(value(E.T), value(D.T))
        np.testing.assert_array_equal(value(E.mT), value(D.mT))
        np.testing.assert_array_equal(value(E[1, :, 2:]), value(D[1, :, 2:]))
    assert E.to_scalar.__doc__ == Tensor.to_scalar.__doc__
    with gb.config.set(autocompute=False):
        for name, call in [
            ("sum", E.sum),
            ("to_dense", E.to_dense),
            ("T", lambda: E.T),
            ("__getitem__", lambda: E[0]),
        ]:
            with pytest.raises(TypeError, match=f"{name} not enabled .* Use `.new\\(\\)`"):
                call()
    A = Matrix.from_dense(np.eye(3))
    with gb.config.set(autocompute=False), pytest.raises(TypeError, match="Use `.new"):
        (A @ A).reduce_rowwise()  # (the same rule for Matrix)


def _model_assign(cv, cp, sel, rv, rp, *, mask=None, accum=None, replace=False, submask=False):
    """GraphBLAS assign on numpy models (values and presence): the region's positions in C
    order, the last occurrence of a repeated position winning, as for Matrix (SuiteSparse).
    ``rv``/``rp`` are the value's values and presence over the region; ``mask`` is a boolean
    array of the Tensor's shape (assign) or the region's (``submask``).
    """
    zv, zp = cv.copy(), cp.copy()
    axes = [[None] if isinstance(s, int) else range(len(s)) for s in sel]
    last = {}
    for r in itertools.product(*axes):
        t = tuple(s if isinstance(s, int) else int(s[i]) for s, i in zip(sel, r, strict=True))
        last[t] = tuple(i for i in r if i is not None)
    for t, r in last.items():
        v, p = rv[r], rp[r]
        if accum is None:
            zv[t], zp[t] = (v if p else 0), p
        elif p and cp[t]:
            zv[t] = accum(cv[t], v)
        elif p:
            zv[t], zp[t] = v, True
    if mask is None:
        return zv, zp
    nv, np_ = cv.copy(), cp.copy()
    if submask:
        for t, r in last.items():
            if mask[r]:
                nv[t], np_[t] = zv[t], zp[t]
            elif replace:
                np_[t] = False
    else:
        nv = np.where(mask, zv, cv)
        np_ = np.where(mask, zp, cp & (not replace))
    return np.where(np_, nv, 0), np_


def _random_key(n, rng, *, dups):
    kind = rng.choice(["int", "slice", "list", "array", "full"])
    if kind == "int":
        return int(rng.integers(-n, n))
    if kind == "slice":
        a, b = sorted(rng.integers(0, n + 1, 2))
        return slice(int(a), int(b) if b > a else None, int(rng.choice([1, 1, 2])))
    if kind == "full":
        return slice(None)
    k = int(rng.integers(1, n + 1))
    idx = rng.choice(n, k, replace=False) if not dups else rng.integers(-n, n, k)
    return [int(i) for i in idx] if kind == "list" else np.asarray(idx)


def _region_model(v_shape_value, shape, rng, kind):
    """A value for a region of ``shape``: (value, values, presence) broadcast to ``shape``."""
    if kind == "scalar":
        x = float(rng.integers(-9, 10))
        return x, np.full(shape, x), np.ones(shape, bool)
    if kind == "empty":
        return gb.Scalar(dtypes.FP64), np.zeros(shape), np.zeros(shape, bool)
    # A Tensor of a broadcastable shape: drop some leading axes, set some sizes to 1
    vshape = list(shape[int(rng.integers(0, len(shape))) :] if kind == "broadcast" else shape)
    if kind == "broadcast":
        vshape = [1 if rng.random() < 0.4 else n for n in vshape]
    V, vv, vp = random_tensor(tuple(vshape), rng, density=0.6)
    bv, bp = np.broadcast_to(vv, shape), np.broadcast_to(vp, shape)
    if kind == "matrix" and len(vshape) == 2:
        return V.to_matrix((0,)), bv, bp
    if kind == "vector" and len(vshape) == 1:
        return V.to_vector(), bv, bp
    return V, bv, bp


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_assign_random(ndim):
    """``T[idx] << value`` and its masked forms against a numpy model of GraphBLAS assign:
    1-5-d, every layout (sampled at 4-5-d), outer keys of every kind (ints, slices, lists,
    arrays, Scalars, negatives), values broadcast, accum, assign and subassign masks of each
    kind, replace.
    """
    rng = np.random.default_rng(300 + ndim)
    layouts = all_layouts(ndim)[:: {1: 1, 2: 1, 3: 1, 4: 5, 5: 41}[ndim]]
    plus = binary.plus
    n_checks = 0
    for layout in layouts:
        shape = tuple(int(n) for n in rng.integers(1, 5, ndim))
        for trial in range(3):
            T, cv, cp = random_tensor(shape, rng, density=0.5, layout=layout)
            keys = tuple(_random_key(n, rng, dups=False) for n in shape)
            if trial == 1 and isinstance(keys[0], int):
                keys = (gb.Scalar.from_value(keys[0]), *keys[1:])
            keyed = tuple(k.value if isinstance(k, gb.Scalar) else k for k in keys)
            sel = [
                k if isinstance(k, int) else np.arange(n)[k]
                for k, n in zip(keyed, shape, strict=True)
            ]
            sel = [
                (s + n if s < 0 else s) if isinstance(s, int) else s
                for s, n in zip(sel, shape, strict=True)
            ]
            rshape = tuple(len(s) for s in sel if not isinstance(s, int))
            kinds = ["tensor", "broadcast", "scalar", "empty", "matrix", "vector"]
            kind = rng.choice(kinds if rshape else ["scalar", "empty"])
            value, rv, rp = _region_model(None, rshape, rng, kind)
            accum = plus if rng.random() < 0.3 else None
            how = rng.choice(["plain", "setitem", "assign_mask", "submask"][: 4 if rshape else 3])
            mask = mbool = None
            replace = False
            if how in {"assign_mask", "submask"}:
                mshape = rshape if how == "submask" else shape
                Mt, mv, mp = random_tensor(mshape, rng, density=0.5, dtype=dtypes.BOOL)
                mkind = rng.choice(["S", "V", "~S", "~V"])
                mask = {"S": Mt.S, "V": Mt.V, "~S": ~Mt.S, "~V": ~Mt.V}[mkind]
                mbool = {"S": mp, "V": mp & mv, "~S": ~mp, "~V": ~(mp & mv)}[mkind]
                replace = rng.random() < 0.4
            ev, ep = _model_assign(
                cv,
                cp,
                sel,
                rv,
                rp,
                mask=mbool,
                accum=(np.add if accum else None),
                replace=replace,
                submask=how == "submask",
            )
            if how == "plain":
                if accum is None:
                    T[keys] << value
                else:
                    T(accum=accum)[keys] << value
            elif how == "setitem":
                if accum is None:
                    T[keys] = value
                else:
                    T(accum=accum)[keys] = value
            elif how == "assign_mask":
                T(mask, accum=accum, replace=replace)[keys] << value
            else:
                T[keys](mask, accum=accum, replace=replace) << value
            check(T, ev, ep)
            assert T.layout == Layout(*layout)
            n_checks += 1
    assert n_checks == 3 * len(layouts)


def test_assign_views_duplicates_and_matrix_parity():
    """Assigning into a view leaves its source and siblings alone (copy-on-write); repeated
    indices: the last occurrence wins, as for Matrix; 2-d Tensors assign as Matrices do
    (masks both ways, accum, replace); ``del``; errors.
    """
    rng = np.random.default_rng(320)
    X, xv, xp = random_tensor((4, 5, 6), rng, density=0.5, layout=((2, 0), (1,)))
    V = X.permute_dims((2, 0, 1))
    W = X.reshape(20, 6)  # (a sibling sharing the data)
    V[1, :, 2:4] = 9.0
    check(X, xv, xp)
    check(W, xv.reshape(20, 6), xp.reshape(20, 6))
    vv, vp = xv.transpose(2, 0, 1).copy(), xp.transpose(2, 0, 1).copy()
    vv[1, :, 2:4], vp[1, :, 2:4] = 9.0, True
    check(V, vv, vp)
    X[0] << -1.0
    check(V, vv, vp)
    # Repeated indices: the last occurrence wins, its absence too
    T, tv, tp = random_tensor((3, 4, 5), rng, density=0.5)
    val, rv, rp = random_tensor((3, 2), rng, density=0.6)
    sel = [np.array([2, 0, 2]), 1, np.array([4, 4])]
    T[[2, 0, 2], 1, [4, 4]] << val
    check(T, *_model_assign(tv, tp, sel, rv, rp))
    # 2-d: the same as Matrix, for each form
    for form in range(6):
        A = Matrix.from_dense(rng.integers(0, 3, (6, 7)).astype(float), missing_value=0.0)
        Am = Matrix.from_dense(rng.integers(0, 2, (6, 7)).astype(bool), missing_value=False)
        B = Matrix.from_dense(rng.integers(0, 3, (3, 2)).astype(float), missing_value=0.0)
        Bm = Matrix.from_dense(rng.integers(0, 2, (3, 2)).astype(bool), missing_value=False)
        T = Tensor.from_matrix(A)
        rows, cols = [5, 1, 3], [0, 6]
        if form == 0:
            A[rows, cols] << B
            T[rows, cols] << B
        elif form == 1:
            A(Am.V, replace=True)[rows, cols] << B
            T(Am.V, replace=True)[rows, cols] << B
        elif form == 2:
            A[rows, cols](Bm.S, binary.plus) << B
            T[rows, cols](Bm.S, binary.plus) << B
        elif form == 3:
            A[rows, cols](~Bm.V, replace=True) << 4.0
            T[rows, cols](~Bm.V, replace=True) << 4.0
        elif form == 4:
            del A[rows, 2:5]
            del T[rows, 2:5]
        else:
            A(~Am.S, binary.times)[rows, cols] << B
            T(~Am.S, binary.times)[rows, cols] << B
        assert T.to_matrix().isequal(A), form
    T = Tensor.from_dense(np.zeros((2, 3, 4)))
    with pytest.raises(DimensionMismatch, match="does not broadcast"):
        T[0] << Tensor.from_dense(np.ones((4, 3)))
    with pytest.raises(DimensionMismatch, match="mask"):
        T[0](Tensor.from_dense(np.ones((2, 3, 4), bool)).S) << 1.0
    with pytest.raises(IndexError, match="too many"):
        T[0, 0, 0, 0] = 1.0
    with pytest.raises(IndexOutOfBound):
        T[2] = 1.0


def _gather_check(R, wv, wp):
    """``R`` against numpy's result on the values (``wv``) and on the presence (``wp``)."""
    check(R, np.where(wp, wv, 0), wp)


@pytest.mark.parametrize("engines", [("extract", "numba", "graphblas"), ("numba",), ("graphblas",)])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_gather(ndim, engines, monkeypatch):
    """``take``, ``take_along_axis`` and ``vindex`` against numpy, values and presence, with
    each engine forced: every layout (sampled at 4-d), sparse, hypersparse, bitmap and full
    storage, negatives, duplicates, empty and multi-d index arrays, broadcasting.
    """
    from graphblas.core import tensor_index

    if "numba" in engines:
        pytest.importorskip("numba")
    monkeypatch.setattr(tensor_index, "_GATHER_ENGINES", engines)
    rng = np.random.default_rng(400 + ndim)
    layouts = all_layouts(ndim)[:: 1 if ndim < 4 else 9]
    for i, layout in enumerate(layouts):
        shape = tuple(int(n) for n in rng.integers(2, 6, ndim))
        density = [0.05, 0.3, 0.8, 1.0][i % 4]
        X, xv, xp = random_tensor(shape, rng, density=density, layout=layout, iso=i % 5 == 2)
        for axis in [None, *range(ndim)]:
            n = math.prod(shape) if axis is None else shape[axis]
            for ishape in [(), (3,), (2, 3), (0,)]:
                idx = rng.integers(-n, n, ishape)
                _gather_check(X.take(idx, axis), np.take(xv, idx, axis), np.take(xp, idx, axis))
        axis = int(rng.integers(0, ndim))
        ishape = [1 if rng.random() < 0.3 else n for n in shape]
        ishape[axis] = int(rng.integers(0, 4))
        idx = rng.integers(-shape[axis], shape[axis], ishape)
        wrapped = np.where(idx < 0, idx + shape[axis], idx)
        _gather_check(
            X.take_along_axis(idx, axis),
            np.take_along_axis(xv, wrapped, axis),
            np.take_along_axis(xp, wrapped, axis),
        )
        keys = []
        for a, n in enumerate(shape):
            r = rng.random()
            if r < 0.2:
                keys.append(int(rng.integers(-n, n)))
            elif r < 0.3:
                keys.append(gb.Scalar.from_value(int(rng.integers(0, n))))
            else:
                keys.append(rng.integers(-n, n, (2, 1, 3)[-(1 + a % 3) :]))
        nkeys = tuple(k.value if isinstance(k, gb.Scalar) else k for k in keys)
        _gather_check(X.vindex[tuple(keys)], xv[nkeys], xp[nkeys])


def test_gather_errors_and_options():
    X = Tensor.from_dense(np.arange(24.0).reshape(2, 3, 4))
    with pytest.raises(IndexError, match="out of bounds"):
        X.take([3], axis=1)
    with pytest.raises(IndexError, match="out of bounds"):
        X.vindex[0, 0, [4]]
    assert X.take([-1], axis=1).shape == (2, 1, 4)
    with pytest.raises(IndexError, match="negative"):
        X.take([-1], axis=1, wraparound=False)
    with pytest.raises(IndexError, match="negative"):
        X.vindex(wraparound=False)[0, -1, 0]
    with pytest.raises(IndexError, match="negative"):
        X.take_along_axis(np.full((2, 3, 1), -1), 2, wraparound=False)
    with pytest.raises(TypeError, match="boolean"):
        X.take(np.array([True, False]), axis=0)
    with pytest.raises(IndexError, match="one index per axis"):
        X.vindex[[0, 1]]
    with pytest.raises(IndexError, match="ints, Scalars"):
        X.vindex[0, :, 1]
    with pytest.raises(ValueError, match="dimensions"):
        X.take_along_axis(np.zeros((2, 3), int), 1)
    E = Tensor(dtypes.FP64, (2, 3))
    assert E.vindex[[0, 1], [2, 0]].nvals == 0
    assert X.vindex[1, 2, 3].to_dense() == 23.0
    # numpy's take modes
    xv = np.arange(24.0).reshape(2, 3, 4)
    idx = np.array([[-7, -1, 0], [3, 5, 11]])
    for mode in ["wrap", "clip"]:
        for axis in [None, 1, 2]:
            want = np.take(xv, idx, axis, mode=mode)
            np.testing.assert_array_equal(X.take(idx, axis, mode=mode).to_dense(), want)
    with pytest.raises(IndexError, match="negative"):
        X.take([-1], axis=1, mode="wrap", wraparound=False)
    with pytest.raises(IndexError, match="out of bounds"):
        X.take([5], axis=1, mode="raise")
    with pytest.raises(ValueError, match="mode"):
        X.take([0], axis=1, mode="clamp")
    with pytest.raises(IndexError, match="out of bounds"):
        Tensor(dtypes.FP64, (2, 0)).take([0], axis=1, mode="wrap")


@pytest.mark.parametrize("max_listed", [None, 0])
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_extract_repeated_indices(ndim, max_listed, monkeypatch):
    """Outer indexing with repeated indices repeats the entries (numpy's ``np.ix_``, and
    Matrix), on every layout and storage, by SuiteSparse's extract or (``_MAX_LISTED`` 0) the
    coordinate filter, which used to keep one of them.
    """
    from graphblas.core import tensor

    if max_listed is not None:
        monkeypatch.setattr(tensor, "_MAX_LISTED", max_listed)
    rng = np.random.default_rng(500 + ndim)
    for i, layout in enumerate(all_layouts(ndim)[:: 1 if ndim < 4 else 7]):
        shape = tuple(int(n) for n in rng.integers(2, 6, ndim))
        X, xv, xp = random_tensor(shape, rng, density=[0.05, 0.4, 1.0][i % 3], layout=layout)
        keys = [
            rng.integers(0, n, int(rng.integers(1, 5))) if rng.random() < 0.7 else slice(None)
            for n in shape
        ]
        sel = np.ix_(*[np.arange(n)[k] for k, n in zip(keys, shape, strict=True)])
        check(X[tuple(keys)].new(), xv[sel], xp[sel])
    A = Matrix.from_dense(np.arange(1.0, 13.0).reshape(3, 4))
    T = Tensor.from_matrix(A)
    assert T[[2, 2, 0], [1, 1]].new().to_matrix().isequal(A[[2, 2, 0], [1, 1]].new())


def test_reduce_aggregators():
    """``reduce`` with aggregators made of a monoid or a semiring gives what Matrix gives
    (values and dtypes); composite ones raise.
    """
    from graphblas import agg

    rng = np.random.default_rng(600)
    for dtype in [dtypes.FP64, dtypes.INT64, dtypes.INT8, dtypes.BOOL]:
        X = random_tensor((3, 4, 5), rng, density=0.5, dtype=dtype, layout=((2, 0), (1,)))[0]
        A = X.permute_dims((0, 2, 1)).reshape(15, 4)  # (axis 1 last)
        M = A.to_matrix((0,))
        names = "count exists sum max count_nonzero count_zero sum_of_squares sum_of_inverses"
        for name in [*names.split(), "hypot", "logaddexp", "logaddexp2"]:
            a = getattr(agg, name)
            try:
                want = M.reduce_rowwise(a).new()
            except KeyError:
                with pytest.raises(KeyError):
                    X.reduce(a, axis=1)
                continue
            got = X.reduce(a, axis=1).new()
            assert got.dtype == want.dtype, (dtype, name)
            assert got.reshape(15).isclose(want, rel_tol=1e-12), (dtype, name)
            assert X.reduce(a[dtype], axis=1).new().isclose(got)
    with pytest.raises(NotImplementedError, match="agg.mean"):
        X.reduce(agg.mean)


def test_isequal_matrix_vector_scalar():
    """``isequal`` / ``isclose`` compare a Tensor with a Matrix, Vector or Scalar of its shape
    by values; another shape is not equal.
    """
    rng = np.random.default_rng(601)
    X = random_tensor((3, 4), rng, density=0.5)[0]
    A = X.to_matrix((0,))
    assert X.isequal(A)
    assert X.isclose(A)
    assert X.T.isequal(A.T)
    assert not X.isequal(A.T.new())
    B = A.dup()
    B[0, 0] = 99.0
    assert not X.isequal(B)
    v = X[1].new()
    assert v.isequal(v.to_vector())
    w = v.to_vector()
    w[0] = 1e9
    assert not v.isequal(w)
    assert not v.isequal(gb.Vector(v.dtype, 5))
    assert not v.isequal(A)
    s = X[1, 2].new()
    assert s.isequal(s.to_scalar())
    assert not s.isequal(gb.Scalar.from_value(1e9))
    assert Tensor(dtypes.FP64, ()).isequal(gb.Scalar(dtypes.FP64))
    assert not X.isequal(gb.Scalar.from_value(1.0))
    assert X.isequal(A, check_dtype=True)
    assert not X.isequal(A.dup(dtypes.INT64), check_dtype=True)
    with pytest.raises(TypeError, match="isequal"):
        X.isequal(np.ones((3, 4)))


def test_einsum_plan_cache():
    """Einsum's plan (parse, MTTKRP pattern, path) is cached by subscripts and shapes; another
    shape plans again, and a result never depends on the cache.
    """
    from graphblas.core import tensor as tensor_mod

    einsum = tensor_mod.einsum
    rng = np.random.default_rng(700)
    X, xv, xp = random_tensor((3, 4, 5), rng, density=0.6)
    Y, yv, yp = random_tensor((5, 2), rng, density=0.6)
    tensor_mod._einsum_plan.cache_clear()
    for sub in ["ijk,kl->ijl", "ijk,kl->lij", "ijk,kl->il"]:
        want = np_einsum(sub, (X, xv, xp), (Y, yv, yp))
        check(einsum(sub, X, Y).new(), *want, approx=True)
        check(einsum(sub, X, Y).new(), *want, approx=True)
    info = tensor_mod._einsum_plan.cache_info()
    assert (info.hits, info.misses) == (3, 3)
    Z, zv, zp = random_tensor((5, 3), rng, density=0.6)
    check(einsum("ijk,kl->ijl", X, Z).new(), *np_einsum("ijk,kl->ijl", (X, xv, xp), (Z, zv, zp)),
          approx=True)  # fmt: skip
    assert tensor_mod._einsum_plan.cache_info().misses == 4
    # An explicit path (a list: unhashable) plans without the cache
    A1 = random_tensor((3, 4), rng)[0]
    R = einsum("ij,jk,kl->il", A1, A1.T, A1, optimize=[(0, 1), (0, 1)]).new()
    assert R.shape == (3, 4)


def test_batched_contraction_embedded_format(monkeypatch):
    """The block-diagonal operand of a batched contraction may take any format again after the
    embedding (bitmap when few batches make it dense: SuiteSparse's bitmap saxpy instead of
    the sparse one's fine tasks); the result is unchanged.
    """
    rng = np.random.default_rng(701)
    X, xv, xp = random_tensor((3, 6, 5), rng, density=0.5, layout=((0, 1), (2,)))
    x = Tensor.from_dense(rng.random((3, 6)))
    seen = []
    orig = Matrix.mxm

    def mxm(self, other, op=gb.semiring.plus_times):
        seen.append(set(self.ss.config["sparsity_control"]))
        return orig(self, other, op)

    monkeypatch.setattr(Matrix, "mxm", mxm)
    R = gb.core.tensor.einsum("lu,luv->lv", x, X).new()
    monkeypatch.undo()
    assert seen
    assert seen[-1] & {"auto", "bitmap"}
    ev = np.einsum("lu,luv->lv", x.to_dense(), np.where(xp, xv, 0))
    ep = np.einsum("lu,luv->lv", np.ones((3, 6)), xp.astype(float)) > 0
    check(R, np.where(ep, ev, 0), ep, approx=True)


def test_einsum_last_contraction_on_target(monkeypatch):
    """The last contraction of a multi-operand einsum runs on the target, so a mask or an
    accumulator applies in its product (a regular path query's step); results as numpy's.
    """
    from graphblas.core import tensor as tensor_mod

    rng = np.random.default_rng(702)
    F, fv, fp = random_tensor((3, 6), rng, density=0.4)
    R, rv, rp = random_tensor((2, 3, 3), rng, density=0.5)
    G, gv, gp = random_tensor((2, 6, 6), rng, density=0.4)
    want = np_einsum("qu,lqp,luv->pv", (F, fv, fp), (R, rv, rp), (G, gv, gp))
    V, vv, vp = random_tensor((3, 6), rng, density=0.5)
    stored = []
    orig = tensor_mod._store

    def store(updater, src, *args, **kwargs):
        stored.append(type(src).__name__)
        return orig(updater, src, *args, **kwargs)

    monkeypatch.setattr(tensor_mod, "_store", store)
    R1 = tensor_mod.einsum("qu,lqp,luv->pv", F, R, G).new(mask=~V.S)
    monkeypatch.undo()
    assert stored[-1] == "MatrixExpression"  # the product itself, masked
    keep = want[1] & ~vp
    check(R1, np.where(keep, want[0], 0), keep, approx=True)
    C = V.dup()
    C(accum=binary.plus) << tensor_mod.einsum("qu,lqp,luv->pv", F, R, G)
    both = want[1] | vp
    cv = np.where(want[1], want[0], 0) + np.where(vp, vv, 0)
    check(C, np.where(both, cv, 0), both, approx=True)


def test_census():
    """The private census records decisions only while on, and ``force`` takes an offered
    option (here the ewise layout of the smaller operand) without changing the result.
    """
    from graphblas.core import tensor_census as census

    rng = np.random.default_rng(703)
    X, xv, xp = random_tensor((4, 5, 6), rng, density=0.6, layout=((0, 1), (2,)))
    Y, yv, yp = random_tensor((4, 5, 6), rng, density=0.2, layout=((2,), (0, 1)))
    want = (np.where(xp & yp, xv * yv, 0), xp & yp)
    assert not census.on
    n0 = len(census.records)
    (X * Y).new()
    assert len(census.records) == n0  # off: nothing recorded
    census.start()
    try:
        check((X * Y).new(), *want, approx=True)
        recs = census.stop()
        lay = [r for r in recs if r["point"] == "ewise.layout"]
        assert lay
        assert set(lay[0]["costs"]) == {"operand 0", "operand 1"}
        other = next(k for k in lay[0]["costs"] if k != lay[0]["choice"])
        census.start()
        census.force["ewise.layout"] = other
        R = (X * Y).new()
        forced = [r for r in census.stop() if r["point"] == "ewise.layout"]
        assert forced[0]["choice"] == other
        assert forced[0]["forced"]
        check(R, *want, approx=True)
    finally:
        census.stop()
        census.force.clear()
        census.records.clear()
    assert not census.on

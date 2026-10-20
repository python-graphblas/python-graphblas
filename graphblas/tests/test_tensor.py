import gc
import itertools
import math

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
    # Hypersparse case H (about 1 entry per row): REBUILD
    _, path, _ = tl.plan(
        (2**19,) * 3, ((0, 1, 2), 1), frozenset([((1, 0, 2), 2)]), 10**6, 8, hyper1=True
    )
    assert path[0][0] == "REBUILD"


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
    # MTTKRP goes to Tensor.mttkrp, never the Khatri-Rao product
    called = []
    orig = Tensor.mttkrp
    monkeypatch.setattr(Tensor, "mttkrp", lambda *a, **k: called.append(1) or orig(*a, **k))
    einsum("ijk,jr,kr->ir", X, B, C).new()
    assert called
    monkeypatch.undo()
    # without opt_einsum the greedy path is used
    monkeypatch.setitem(__import__("sys").modules, "opt_einsum", None)
    A1, a1v, a1p = random_tensor((3, 4), rng)
    A2, a2v, a2p = random_tensor((4, 5), rng)
    A3, a3v, a3p = random_tensor((5, 2), rng)
    R = einsum("ij,jk,kl->il", A1, A2, A3).new()
    check(
        R, *np_einsum("ij,jk,kl->il", (A1, a1v, a1p), (A2, a2v, a2p), (A3, a3v, a3p)), approx=True
    )
    monkeypatch.undo()
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
    with pytest.raises(NotImplementedError):
        X[0] = 1


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


def test_default_split_follows_nonempty_rows():
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
    # Shuffled input is built at the widest split and settled: the same layouts
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
    # Large sample-checked cases (more entries than the sample size). About 2 entries per
    # (i, j) row: thin rows are kept only when (0|1,2)'s column group would be wider than the
    # data (400 * 2000 positions for 2e5 entries), not when it is narrower (400 * 50).
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

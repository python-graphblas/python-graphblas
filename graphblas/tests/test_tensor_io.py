"""Tensor conversions with pydata/sparse and scipy.sparse, against the libraries' own todense."""

import itertools

import numpy as np
import pytest

from graphblas import backend, binary, dtypes
from graphblas.core.tensor import Layout, Tensor
from graphblas.core.tensor import _has_container as tensor_has_container
from graphblas.core.tensor_io import _scipy_coo_needs

from .test_tensor import all_layouts, check

if backend != "suitesparse" or not tensor_has_container():
    pytest.skip("Tensor needs SuiteSparse:GraphBLAS 10 or later", allow_module_level=True)


ss = pytest.importorskip("scipy.sparse")
try:
    import sparse
except ImportError:  # (pydata sparse needs Numba)
    sparse = None
needs_sparse = pytest.mark.skipif(sparse is None, reason="needs pydata sparse")

pytestmark = pytest.mark.skipif(backend != "suitesparse", reason="Tensor needs SuiteSparse")

SHAPES = [(), (7,), (4, 5), (3, 4, 5), (2, 3, 4, 3), (2, 2, 3, 2, 3)]
DTYPES = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]
if dtypes._supports_complex:
    DTYPES += [np.complex64, np.complex128]


def _model(shape, rng, dtype=np.float64, *, density=0.4, iso=False):
    """Coordinates (C order), values with explicit zeros among them, and the dense model."""
    present = np.asarray(rng.random(shape) < density)
    if np.dtype(dtype) == np.bool_:
        values = rng.random(shape) < 0.5
    elif np.issubdtype(dtype, np.unsignedinteger):
        values = rng.integers(0, 6, shape)
    elif np.issubdtype(dtype, np.complexfloating):
        values = rng.integers(-3, 4, shape) + 1j * rng.integers(-3, 4, shape)
    else:
        values = rng.integers(-5, 6, shape)
    values = np.asarray(values).astype(dtype)
    if iso:
        values = np.full(shape, 1, dtype)
    else:
        values[present & (rng.random(shape) < 0.2)] = 0  # explicit zeros
    coords = np.ascontiguousarray(np.argwhere(present).T, np.int64)  # (also 0-d)
    return coords, values[present], present, np.where(present, values, 0).astype(dtype)


def _c_order(coords, shape):
    """Whether the coordinates are strictly increasing in C order."""
    if coords.shape[1] < 2 or not shape:
        return True
    lin = np.ravel_multi_index(tuple(coords), shape)
    return bool((np.diff(lin) > 0).all())


def _scipy_coords(A):
    """A scipy COO array's coordinates (scipy before 1.13 has ``row`` and ``col``, not
    ``coords``).
    """
    return (A.row, A.col) if A.ndim == 2 else A.coords


def _todense(s):
    """``s.todense()``, but a 0-d COO with an entry gives its value: pydata sparse before
    0.16 fails to make that one dense (``x[()] = data`` with ``data`` of length 1).
    """
    if s.shape == () and s.nnz and isinstance(s, sparse.COO):
        return np.asarray(s.data[0])
    return s.todense()


def _same_pydata(got, want):
    """Same shape, dtype, entries (explicit zeros too) and values; COO in C order."""
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    assert got.fill_value == 0
    assert got.nnz == want.nnz
    np.testing.assert_array_equal(_todense(got), _todense(want))
    g, w = got.asformat("coo"), want.asformat("coo")
    np.testing.assert_array_equal(g.coords, w.coords)
    np.testing.assert_array_equal(g.data, w.data)
    if isinstance(got, sparse.COO):
        assert _c_order(got.coords, got.shape)


@needs_sparse
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_pydata_round_trip(shape, dtype):
    rng = np.random.default_rng(len(shape) * 100 + np.dtype(dtype).num)
    coords, vals, present, dense = _model(shape, rng, dtype)
    s = sparse.COO(coords, vals, shape=shape)
    T = Tensor.from_pydata_sparse(s)
    assert T.dtype == dtypes.lookup_dtype(dtype)
    check(T, dense, present)
    np.testing.assert_array_equal(T.to_dense(fill_value=0), _todense(s))
    # (pydata cannot turn a 0-d GCXS or DOK back into COO)
    for fmt in ["coo", "gcxs", "dok"] if shape else ["coo"]:
        _same_pydata(T.to_pydata_sparse(fmt), s)
        T2 = Tensor.from_pydata_sparse(s.asformat(fmt))
        check(T2, dense, present)


@needs_sparse
@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
def test_pydata_gcxs_compressed_axes(ndim):
    """GCXS with any compressed axes imports into that layout; export uses the Tensor's
    layout when GCXS can hold it, else the axes asked for, or the cheapest it can hold.
    """
    rng = np.random.default_rng(ndim)
    shape = SHAPES[ndim]
    coords, vals, present, dense = _model(shape, rng)
    s = sparse.COO(coords, vals, shape=shape)
    subsets = [c for k in range(1, ndim) for c in itertools.combinations(range(ndim), k)]
    for ca in subsets:
        g = sparse.GCXS(s, compressed_axes=ca)
        T = Tensor.from_pydata_sparse(g)
        assert T.layout == Layout(ca, tuple(a for a in range(ndim) if a not in ca))
        assert not T.layout_pinned
        check(T, dense, present)
        out = T.to_pydata_sparse("gcxs")
        assert out.compressed_axes == ca  # (no relayout)
        _same_pydata(out, s)
        for other in subsets[:: max(1, len(subsets) // 4)]:
            out = T.to_pydata_sparse("gcxs", compressed_axes=other)
            assert out.compressed_axes == other
            _same_pydata(out, s)
    for lay in all_layouts(ndim)[:: 1 if ndim < 4 else 13]:
        T = Tensor.from_coo(coords, vals, shape=shape, layout=lay)
        out = T.to_pydata_sparse("gcxs")
        held = list(lay.rows) == sorted(lay.rows) and list(lay.cols) == sorted(lay.cols)
        if 0 < len(lay.rows) < ndim and held:
            assert out.compressed_axes == lay.rows  # (no relayout)
        _same_pydata(out, s)
        _same_pydata(T.to_pydata_sparse("coo"), s)


@needs_sparse
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_pydata_views_and_layouts(ndim):
    """Permuted, expanded and squeezed views, every stored layout, iso and empty."""
    rng = np.random.default_rng(10 + ndim)
    shape = SHAPES[ndim]
    coords, vals, present, dense = _model(shape, rng)
    for lay in all_layouts(ndim)[:: 1 if ndim < 4 else 7]:
        T = Tensor.from_coo(coords, vals, shape=shape, layout=lay)
        for perm in list(itertools.permutations(range(ndim)))[:: 1 if ndim < 4 else 5]:
            V = T.permute_dims(perm)
            want = sparse.COO.from_numpy(np.transpose(dense, perm))
            want = sparse.COO(  # (explicit zeros of the model kept)
                np.array(np.nonzero(np.transpose(present, perm)), np.int64).reshape(ndim, -1),
                np.transpose(dense, perm)[np.transpose(present, perm)],
                shape=want.shape,
            )
            for fmt in ["coo", "gcxs"]:
                _same_pydata(V.to_pydata_sparse(fmt), want)
    E = Tensor.from_coo(coords, vals, shape=shape).expand_dims(1)
    _same_pydata(
        E.to_pydata_sparse(), sparse.COO(np.insert(coords, 1, 0, axis=0), vals, shape=E.shape)
    )
    _same_pydata(E.squeeze(1).to_pydata_sparse("gcxs"), sparse.COO(coords, vals, shape=shape))
    # iso
    icoords, ivals, ipresent, idense = _model(shape, rng, iso=True)
    T = Tensor.from_coo(icoords, 1.0, shape=shape)
    for fmt in ["coo", "gcxs"]:
        _same_pydata(T.to_pydata_sparse(fmt), sparse.COO(icoords, ivals, shape=shape))
    check(Tensor.from_pydata_sparse(sparse.COO(icoords, ivals, shape=shape)), idense, ipresent)
    # empty, with and without a zero-size axis
    for eshape in [shape, (*shape[:-1], 0)]:
        e = sparse.COO(np.zeros((ndim, 0), np.int64), np.array([], np.float64), shape=eshape)
        T = Tensor.from_pydata_sparse(e)
        assert T.shape == eshape
        assert T.nvals == 0
        for fmt in ["coo", "gcxs", "dok"]:
            _same_pydata(T.to_pydata_sparse(fmt), e)
            assert Tensor.from_pydata_sparse(e.asformat(fmt)).nvals == 0


@needs_sparse
def test_pydata_gcxs_repeated_or_unsorted_columns():
    """A GCXS whose rows repeat a column, or list them out of order (pydata checks neither),
    imports as pydata's todense reads it: repeats summed (or combined with ``dup_op``). Its
    arrays were imported as they were, which made a Matrix SuiteSparse finds invalid
    (``GxB_Matrix_fprint``: GrB_INVALID_OBJECT) holding both entries.
    """
    from graphblas.core import ffi, lib

    def valid(T):
        M = T.ss.matrix
        return lib.GxB_Matrix_fprint(M._carg, b"M", lib.GxB_SILENT, ffi.NULL) == lib.GrB_SUCCESS

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for indices, indptr in [
        ([1, 1, 0, 2, 3, 3], [0, 2, 3, 6]),  # repeats, sorted
        ([1, 0, 1, 2, 3, 2], [0, 3, 3, 6]),  # repeats, unsorted; an empty row
        ([2, 0, 1, 3, 1, 0], [0, 3, 4, 6]),  # unsorted, no repeat
        ([0, 1, 2, 0, 1, 3], [0, 3, 3, 6]),  # sorted and distinct: imported as it is
    ]:
        arrays = (data, np.array(indices), np.array(indptr))
        g = sparse.GCXS(arrays, shape=(3, 4), compressed_axes=(0,))
        T = Tensor.from_pydata_sparse(g)
        assert valid(T)
        assert T.layout == Layout((0,), (1,))
        want = g.todense()
        check(T, want, want != 0)
        # dup_op combines the repeats; BOOL ORs them
        T = Tensor.from_pydata_sparse(g, dup_op=binary.max)
        assert valid(T)
        r = np.repeat(np.arange(3), np.diff(indptr))
        want = np.zeros((3, 4))
        np.maximum.at(want, (r, np.array(indices)), data)
        check(T, want, want != 0)
        T = Tensor.from_pydata_sparse(g.astype(bool))
        assert valid(T)
        check(T, want != 0, want != 0)
    # 3-d, compressed along axis 1: rows of axis 1, columns of (0, 2) raveled
    g = sparse.GCXS(
        (data, np.array([5, 5, 0, 1, 1, 2]), np.array([0, 3, 6])),
        shape=(2, 2, 3),
        compressed_axes=(1,),
    )
    T = Tensor.from_pydata_sparse(g)
    assert valid(T)
    assert T.layout == Layout((1,), (0, 2))
    want = g.todense()
    check(T, want, want != 0)


@needs_sparse
def test_pydata_options_and_errors():
    rng = np.random.default_rng(3)
    shape = (3, 4, 5)
    coords, vals, present, dense = _model(shape, rng)
    s = sparse.COO(coords, vals, shape=shape)
    for src in [s, sparse.GCXS(s, compressed_axes=(1,))]:
        T = Tensor.from_pydata_sparse(src, layout=((2,), (0, 1)), name="X")
        assert T.layout == Layout((2,), (0, 1))
        assert T.layout_pinned
        assert T.name == "X"
        check(T, dense, present)
    # A fill value other than 0 has no meaning for a Tensor
    for fill in [1.0, np.nan]:
        with pytest.raises(ValueError, match="fill value of 0"):
            Tensor.from_pydata_sparse(sparse.COO(coords, vals, shape=shape, fill_value=fill))
    with pytest.raises(ValueError, match="fill value of 0"):
        Tensor.from_pydata_sparse(sparse.GCXS(s, compressed_axes=(0,)).astype(int) + 1)
    with pytest.raises(TypeError, match="sparse"):
        Tensor.from_pydata_sparse(dense)
    T = Tensor.from_pydata_sparse(s)
    with pytest.raises(ValueError, match="Invalid format"):
        T.to_pydata_sparse("csr")
    with pytest.raises(ValueError, match="gcxs"):
        T.to_pydata_sparse("coo", compressed_axes=(0,))
    with pytest.raises(ValueError, match="some but not all"):
        T.to_pydata_sparse("gcxs", compressed_axes=(0, 1, 2))
    out = T.to_pydata_sparse("gcxs", compressed_axes=(-1, 0))
    assert out.compressed_axes == (0, 2)
    # 1-d and 0-d GCXS go through COO
    _same_pydata(Tensor.from_pydata_sparse(s[1, 2]).to_pydata_sparse("gcxs"), s[1, 2])


@pytest.mark.parametrize("shape", SHAPES[1:], ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_scipy_round_trip(shape, dtype):
    rng = np.random.default_rng(len(shape) * 100 + np.dtype(dtype).num + 7)
    coords, vals, present, dense = _model(shape, rng, dtype)
    if need := _scipy_coo_needs(len(shape)):  # (this scipy has no such array)
        with pytest.raises(NotImplementedError, match=f"requires scipy {need} or later"):
            Tensor.from_coo(coords, vals, shape=shape).to_scipy_sparse()
        return
    A = ss.coo_array((vals, tuple(coords)), shape=shape)
    T = Tensor.from_scipy_sparse(A)
    assert T.dtype == dtypes.lookup_dtype(dtype)
    check(T, dense, present)
    np.testing.assert_array_equal(T.to_dense(fill_value=0), A.todense())
    B = T.to_scipy_sparse()
    assert B.format == "coo"
    assert B.shape == shape
    assert B.dtype == np.dtype(dtype)
    assert B.nnz == A.nnz  # (explicit zeros kept)
    np.testing.assert_array_equal(B.todense(), A.todense())
    if B.has_canonical_format:
        assert _c_order(np.array(_scipy_coords(B)).reshape(len(shape), -1), shape)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_scipy_layouts_and_duplicates(ndim):
    rng = np.random.default_rng(20 + ndim)
    shape = SHAPES[ndim]
    coords, vals, present, dense = _model(shape, rng)
    if need := _scipy_coo_needs(ndim):  # (this scipy has no such array)
        with pytest.raises(NotImplementedError, match=f"requires scipy {need} or later"):
            Tensor.from_coo(coords, vals, shape=shape).to_scipy_sparse()
        return
    for lay in all_layouts(ndim)[:: 1 if ndim < 4 else 7]:
        T = Tensor.from_coo(coords, vals, shape=shape, layout=lay)
        B = T.to_scipy_sparse()
        # The stored order (no relayout), canonical only in the identity axis order
        assert B.has_canonical_format == (lay.seq == tuple(range(ndim)))
        if B.has_canonical_format:
            assert _c_order(np.array(_scipy_coords(B)), shape)
        assert B.nnz == len(vals)
        np.testing.assert_array_equal(B.todense(), dense)
        for perm in list(itertools.permutations(range(ndim)))[:: 1 if ndim < 4 else 5]:
            np.testing.assert_array_equal(
                T.permute_dims(perm).to_scipy_sparse().todense(), np.transpose(dense, perm)
            )
    # Repeated coordinates add up as in scipy, or by dup_op
    rep = rng.integers(0, len(vals), len(vals))
    dcoords = np.concatenate([coords, coords[:, rep]], axis=1)
    dvals = np.concatenate([vals, vals[rep] + 1])
    A = ss.coo_array((dvals, tuple(dcoords)), shape=shape)
    assert not A.has_canonical_format
    T = Tensor.from_scipy_sparse(A)
    check(T, A.todense(), present)
    T = Tensor.from_scipy_sparse(A, dup_op=binary.max, layout=Layout((), tuple(range(ndim))))
    want = np.full(shape, -np.inf)
    np.maximum.at(want, tuple(dcoords), dvals)
    check(T, np.where(present, want, 0), present)
    assert T.layout_pinned
    Ab = ss.coo_array((dvals != 0, tuple(dcoords)), shape=shape)
    check(Tensor.from_scipy_sparse(Ab), Ab.todense(), present)
    # Empty, and a zero-size axis
    for eshape in [shape, (*shape[:-1], 0)]:
        E = ss.coo_array((np.array([]), tuple(np.zeros((ndim, 0), np.int64))), shape=eshape)
        T = Tensor.from_scipy_sparse(E)
        assert T.shape == eshape
        assert T.nvals == 0
        assert T.to_scipy_sparse().shape == eshape
        assert T.to_scipy_sparse().nnz == 0


def test_scipy_2d_formats():
    rng = np.random.default_rng(5)
    coords, vals, present, dense = _model((6, 7), rng)
    A = ss.coo_array((vals, tuple(coords)), shape=(6, 7))
    for fmt in ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"]:
        S = A.asformat(fmt)
        T = Tensor.from_scipy_sparse(S)
        if fmt in {"csr", "csc"}:
            assert T.layout == (Layout((0,), (1,)) if fmt == "csr" else Layout((1,), (0,)))
            check(T, dense, present)  # (explicit zeros kept)
        np.testing.assert_array_equal(T.to_dense(fill_value=0), S.toarray())
        for lay in all_layouts(2):
            T2 = Tensor.from_coo(coords, vals, shape=(6, 7), layout=lay)
            B = T2.to_scipy_sparse(fmt)
            assert B.format == fmt
            np.testing.assert_array_equal(B.toarray(), dense)
            if fmt in {"csr", "csc", "coo"}:
                assert B.nnz == len(vals)
    # A CSR array with repeated columns in a row: added up
    R = ss.csr_array((np.array([1.0, 2.0, 4.0]), np.array([1, 1, 0]), np.array([0, 2, 3])))
    assert not R.has_canonical_format
    check(
        Tensor.from_scipy_sparse(R),
        np.array([[0, 3.0], [4.0, 0]]),
        np.array([[False, True], [True, False]]),
    )
    # The stored CSR of layout (1|0) is the CSC of the Tensor
    T = Tensor.from_scipy_sparse(A.asformat("csc"))
    np.testing.assert_array_equal(T.to_scipy_sparse("csc").toarray(), dense)


def test_scipy_errors(monkeypatch):
    import scipy

    T = Tensor.from_coo(np.array([[0, 1], [1, 2], [2, 3]]), [1.0, 2.0], shape=(2, 3, 4))
    with pytest.raises(ValueError, match="'coo' only"):
        T.to_scipy_sparse("csr")
    # scipy.sparse is 2-d only before scipy 1.13, has 1-d COO from 1.13 and n-d from 1.15
    V = Tensor.from_coo(np.array([[0, 2]]), [1.0, 2.0], shape=(3,))
    M = Tensor.from_coo(np.array([[0, 1], [1, 0]]), [1.0, 2.0], shape=(2, 2))
    real_1d = _scipy_coo_needs(1) is None  # (the real scipy makes the array below)
    monkeypatch.setattr(scipy, "__version__", "1.14.1")
    with pytest.raises(NotImplementedError, match="3-d .* requires scipy 1.15 or later"):
        T.to_scipy_sparse()
    if real_1d:
        assert V.to_scipy_sparse().shape == (3,)
    monkeypatch.setattr(scipy, "__version__", "1.12.0")
    with pytest.raises(NotImplementedError, match="1-d .* requires scipy 1.13 .* scipy 1.12.0"):
        V.to_scipy_sparse()
    assert M.to_scipy_sparse().shape == (2, 2)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="no 0-d"):
        Tensor.from_coo((), [1.0], shape=()).to_scipy_sparse()
    with pytest.raises(TypeError, match="scipy.sparse"):
        Tensor.from_scipy_sparse(np.eye(2))
    with pytest.raises(ValueError, match="Invalid format"):
        Tensor.from_coo(np.array([[0], [1]]), [1.0], shape=(2, 2)).to_scipy_sparse("xyz")

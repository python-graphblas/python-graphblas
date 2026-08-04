import functools
from io import BytesIO, StringIO

import numpy as np
import pytest

import graphblas as gb
from graphblas import Matrix, Vector, dtypes
from graphblas.exceptions import GraphblasException

try:
    import networkx as nx
except ImportError:  # pragma: no cover (import)
    nx = None

try:
    import scipy.sparse as ss
except ImportError:  # pragma: no cover (import)
    ss = None

try:
    import sparse
except ImportError:  # pragma: no cover (import)
    sparse = None

try:
    import awkward._v2 as ak
except ImportError:
    try:
        import awkward as ak
    except ImportError:  # pragma: no cover (import)
        ak = None

try:
    import fast_matrix_market as fmm
except ImportError:  # pragma: no cover (import)
    fmm = None

suitesparse = gb.backend == "suitesparse"


@pytest.mark.skipif("not ss")
def test_vector_to_from_numpy():
    a = np.array([0.0, 2.0, 4.1])
    v = Vector.from_dense(a, 0)
    assert v.isequal(gb.Vector.from_coo([1, 2], [2.0, 4.1]), check_dtype=True)
    a2 = v.to_dense(0)
    np.testing.assert_array_equal(a, a2)

    csr = gb.io.to_scipy_sparse(v, "csr")
    assert csr.nnz == 2
    # 2023-06-25: scipy 1.11.0 added `sparray` and changed e.g. `ss.isspmatrix_csr`
    assert isinstance(csr, getattr(ss, "sparray", ss.spmatrix))
    assert csr.format == "csr"
    np.testing.assert_array_equal(csr.toarray(), np.array([[0.0, 2.0, 4.1]]))

    csc = gb.io.to_scipy_sparse(v, "csc")
    assert csc.nnz == 2
    # 2023-06-25: scipy 1.11.0 added `sparray` and changed e.g. `ss.isspmatrix_csc`
    assert isinstance(csc, getattr(ss, "sparray", ss.spmatrix))
    assert csc.format == "csc"
    np.testing.assert_array_equal(csc.toarray(), np.array([[0.0, 2.0, 4.1]]).T)

    # default to csr-like
    coo = gb.io.to_scipy_sparse(v, "coo")
    assert coo.shape == csr.shape
    # 2023-06-25: scipy 1.11.0 added `sparray` and changed e.g. `ss.isspmatrix_coo`
    assert isinstance(coo, getattr(ss, "sparray", ss.spmatrix))
    assert coo.format == "coo"
    assert coo.nnz == 2
    np.testing.assert_array_equal(coo.toarray(), np.array([[0.0, 2.0, 4.1]]))


@pytest.mark.skipif("not ss")
@pytest.mark.parametrize("a", [np.array([7, 0]), np.array([0, 0]), np.array([])])
def test_vector_to_from_numpy_correct_size(a):
    # Make sure we use the right size
    v = Vector.from_dense(a, 0)
    assert v.shape == a.shape
    b = v.to_dense(0)
    np.testing.assert_array_equal(a, b)
    csr = gb.io.to_scipy_sparse(v, "csr")
    np.testing.assert_array_equal(a[None, :], csr.toarray())
    csc = gb.io.to_scipy_sparse(v, "csc")
    np.testing.assert_array_equal(a[:, None], csc.toarray())


@pytest.mark.skipif("not ss")
def test_matrix_to_from_numpy():
    a = np.array([[1.0, 0.0], [2.0, 3.7]])
    M = Matrix.from_dense(a, 0)
    assert M.isequal(gb.Matrix.from_coo([0, 1, 1], [0, 0, 1], [1.0, 2.0, 3.7]), check_dtype=True)
    a2 = M.to_dense(0)
    np.testing.assert_array_equal(a, a2)

    for format in ["csr", "csc", "coo"]:
        sparse = gb.io.to_scipy_sparse(M, format)
        # 2023-06-25: scipy 1.11.0 added `sparray` and changed e.g. `ss.isspmatrix_csr`
        assert isinstance(sparse, getattr(ss, "sparray", ss.spmatrix))
        assert sparse.format == format
        assert sparse.nnz == 3
        np.testing.assert_array_equal(sparse.toarray(), a)
        M2 = gb.io.from_scipy_sparse(sparse)
        assert M.isequal(M2, check_dtype=True)

    with pytest.raises(ValueError, match="Invalid format"):
        gb.io.to_scipy_sparse(M, "bad format")


@pytest.mark.skipif("not nx or not ss")
def test_matrix_to_from_networkx():
    M = gb.Matrix.from_coo([0, 1, 1], [0, 0, 1], [1, 2, 3])
    G = gb.io.to_networkx(M)
    a = np.array([[1, 0], [2, 3]])
    G2 = nx.from_numpy_array(a, create_using=nx.DiGraph)
    assert G.number_of_edges() == G2.number_of_edges() == 3
    assert G.number_of_nodes() == G2.number_of_nodes() == 2
    np.testing.assert_array_equal(nx.to_numpy_array(G), a)

    # No weights
    G3 = gb.io.to_networkx(M, edge_attribute=None)
    a2 = np.array([[1, 0], [1, 1]])
    G4 = nx.from_numpy_array(a2, create_using=nx.DiGraph)
    assert G3.number_of_edges() == G4.number_of_edges() == 3
    assert G3.number_of_nodes() == G4.number_of_nodes() == 2
    np.testing.assert_array_equal(nx.to_numpy_array(G3), a2)

    M2 = gb.io.from_networkx(G, dtype=int)
    assert M.isequal(M2, check_dtype=True)
    # Check iso-value
    G = nx.DiGraph()
    edges = [
        (1, 2),
        (1, 3),
        # 2 is a dangling node
        (3, 1),
        (3, 2),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 4),
        (5, 6),
        (6, 4),
    ]
    G.add_edges_from(edges)
    G.add_node(0)
    M = gb.io.from_networkx(G, nodelist=range(7))
    if suitesparse:
        assert M.ss.is_iso
    rows, cols = zip(*edges, strict=True)
    expected = gb.Matrix.from_coo(rows, cols, 1)
    assert expected.isequal(M)
    # Test empty
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXError):
        gb.io.from_networkx(G)
    G.add_node(0)
    M = gb.io.from_networkx(G)
    assert M.nvals == 0
    assert M.shape == (1, 1)


@pytest.mark.skipif("not nx")
def test_from_networkx_undirected():
    # Undirected graphs go through from_networkx's direct COO path (no scipy),
    # which symmetrizes off-diagonal entries and keeps self-loops single-counted.
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 2.0), (0, 0, 7.0), (1, 2, 3.0)])
    M = gb.io.from_networkx(G)
    expected = gb.Matrix.from_coo([0, 0, 1, 1, 2], [0, 1, 0, 2, 1], [7.0, 2.0, 2.0, 3.0, 3.0])
    assert M.isequal(expected, check_dtype=True)

    # weight=None ignores edge weights (all entries 1) and yields an int Matrix
    M_none = gb.io.from_networkx(G, weight=None)
    expected_none = gb.Matrix.from_coo([0, 0, 1, 1, 2], [0, 1, 0, 2, 1], 1)
    assert M_none.isequal(expected_none, check_dtype=True)


@pytest.mark.skipif("not nx or not ss")
@pytest.mark.parametrize(
    "graph_cls", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph] if nx else []
)
def test_from_networkx_rejects_array_weights(graph_cls):
    # Sequence weights of uniform length infer a 2-D numeric array, which passes
    # a dtype-kind test but which from_coo would happily read as a UDT. scipy has
    # always rejected these, so the direct path must defer rather than quietly
    # widen what from_networkx accepts.
    G = graph_cls()
    G.add_edge(0, 1, weight=[1, 2])
    G.add_edge(1, 0, weight=[3, 4])
    with pytest.raises(ValueError, match="must be 1-D"):
        gb.io.from_networkx(G)


@pytest.mark.skipif("not nx or not ss")
@pytest.mark.parametrize(
    "graph_cls", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph] if nx else []
)
def test_from_networkx_rejects_string_weights(graph_cls):
    # String weights infer a 1-D <U array, so a shape test alone would let them
    # through; only the dtype-kind half of the guard sends these to scipy, which
    # rejects them the way it always has.
    G = graph_cls()
    G.add_edge(0, 1, weight="a")
    G.add_edge(1, 2, weight="b")
    with pytest.raises(ValueError, match="does not support dtype"):
        gb.io.from_networkx(G)


@pytest.mark.skipif("not nx")
def test_from_networkx_unsupported_dtype_is_valueerror(monkeypatch):
    # scipy < 1.15 builds a string-dtype coo array happily and only fails converting
    # it to csr, and networkx re-raises that failure as a NetworkXError blaming the
    # sparse format. from_networkx restates it as the ValueError newer scipy raises
    # directly, which is what the caller can act on; monkeypatching the fallback
    # exercises the old behavior on any scipy.
    import graphblas.io._networkx as _gnx

    def _raise_networkx_error(*args, **kwargs):
        raise nx.NetworkXError("Unknown sparse matrix format: csr")

    monkeypatch.setattr(_gnx, "_from_networkx_via_scipy", _raise_networkx_error)
    G = nx.Graph()
    G.add_edge(0, 1, weight="a")
    G.add_edge(1, 2, weight="b")
    with pytest.raises(ValueError, match="does not support dtype"):
        gb.io.from_networkx(G)


@pytest.mark.skipif("not nx or not ss")
@pytest.mark.parametrize(
    "graph_cls", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph] if nx else []
)
def test_from_networkx_matches_scipy(graph_cls):
    # Both simple graphs and multigraphs now build the coo directly (no scipy
    # round-trip); the result must reproduce nx.to_scipy_sparse_array exactly,
    # including parallel-edge and parallel-self-loop weight sums.
    G = graph_cls()
    G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 0, 4.0), (0, 0, 5.0)])
    if G.is_multigraph():
        G.add_edge(0, 1, weight=1.5)  # parallel edge -> weights summed
        G.add_edge(0, 0, weight=2.5)  # parallel self-loop -> diagonal summed
    G.add_edge(3, 4)  # missing weight attr -> default 1
    G.add_node(9)  # isolated node

    A = nx.to_scipy_sparse_array(G, weight="weight")
    reference = gb.io.from_scipy_sparse(A)
    M = gb.io.from_networkx(G, weight="weight")
    assert M.isequal(reference, check_dtype=True)
    assert M.shape == reference.shape


@pytest.mark.skipif("not nx")
def test_from_networkx_multigraph_is_scipy_free(monkeypatch):
    # A numeric-weight multigraph must ingest via the direct coo path, not the
    # scipy fallback, so networkx ingest no longer requires scipy for this case.
    import graphblas.io._networkx as _gnx

    def _boom(*args, **kwargs):  # pragma: no cover (only runs if the direct path regresses)
        raise AssertionError("scipy fallback should not be used for a numeric multigraph")

    monkeypatch.setattr(_gnx, "_from_networkx_via_scipy", _boom)
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from([(0, 1, 2.0), (0, 1, 3.0), (2, 0, 1.0), (1, 1, 4.0), (1, 1, 0.5)])
    M = gb.io.from_networkx(G)
    assert M[0, 1].new().value == 5.0  # parallel edges summed
    assert M[1, 1].new().value == 4.5  # parallel self-loops summed


@pytest.mark.skipif("not ss")
@pytest.mark.parametrize("engine", ["auto", "scipy", "fmm"])
def test_mmread_mmwrite(engine):
    if engine == "fmm" and fmm is None:  # pragma: no cover (import)
        pytest.skip("needs fast_matrix_market")
    try:
        # Importing this module evaluates ``pytest.mark.thread_unsafe``, which
        # fails unless the marker is registered; see the ``markers`` entry in
        # pyproject.toml.
        from scipy.io.tests import test_mmio
    except ImportError:
        # Test files are mysteriously missing from some conda-forge builds
        pytest.skip("scipy.io.tests.test_mmio unavailable :(")

    p31 = 2**31
    p63 = 2**63
    m31 = -p31
    m63 = -p63
    p311 = p31 - 1
    p312 = p31 - 2
    p631 = p63 - 1
    m631 = m63 + 1
    m632 = m63 + 2

    # Use example Matrix Market files from scipy
    examples = {
        "_32bit_integer_dense_example": (
            False,
            Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [p311, p311, p312, p312]),
        ),
        "_32bit_integer_sparse_example": (
            False,
            Matrix.from_coo([0, 1], [0, 1], [p311, p312]),
        ),
        "_64bit_integer_dense_example": (
            False,
            Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [p31, m31, m632, p631]),
        ),
        "_64bit_integer_sparse_general_example": (
            False,
            Matrix.from_coo([0, 0, 1], [0, 1, 1], [p31, p631, p631]),
        ),
        "_64bit_integer_sparse_symmetric_example": (
            False,
            Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [p31, m631, m631, p631]),
        ),
        "_64bit_integer_sparse_skew_example": (
            False,
            Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [p31, m631, p631, p631]),
        ),
        "_over64bit_integer_dense_example": (True, None),
        "_over64bit_integer_sparse_example": (True, None),
        "_general_example": (
            False,
            Matrix.from_coo(
                [0, 0, 1, 2, 3, 3, 3, 4],
                [0, 3, 1, 2, 1, 3, 4, 4],
                [1, 6, 10.5, 0.015, 250.5, -280, 33.32, 12],
            ),
        ),
        "_skew_example": (
            False,
            Matrix.from_coo(
                [0, 1, 1, 2, 3, 3, 3, 4, 4],
                [0, 1, 3, 2, 1, 3, 4, 3, 4],
                [1, 10.5, -250.5, 0.015, 250.5, -280, 0, 0, 12],
            ),
        ),
        "_symmetric_example": (
            False,
            Matrix.from_coo(
                [0, 1, 1, 2, 3, 3, 3, 4, 4],
                [0, 1, 3, 2, 1, 3, 4, 3, 4],
                [1, 10.5, 250.5, 0.015, 250.5, -280, 8, 8, 12],
            ),
        ),
        "_symmetric_pattern_example": (
            False,
            Matrix.from_coo(
                [0, 1, 1, 2, 3, 3, 3, 4, 4],
                [0, 1, 3, 2, 1, 3, 4, 3, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
        ),
        "_empty_lines_example": (
            False,
            Matrix.from_coo(
                [0, 0, 1, 2, 3, 3, 3, 4],
                [0, 3, 1, 2, 1, 3, 4, 4],
                [1, 6, 10.5, 0.015, 250.5, -280, 33.32, 12],
            ),
        ),
    }
    if dtypes._supports_complex:
        examples["_hermitian_example"] = (
            False,
            Matrix.from_coo(
                [0, 1, 1, 2, 3, 3, 3, 4, 4],
                [0, 1, 3, 2, 1, 3, 4, 3, 4],
                [1, 10.5, 250.5 - 22.22j, 0.015, 250.5 + 22.22j, -280, -33.32j, 33.32j, 12],
            ),
        )
    success = 0
    for example, (over64, expected) in examples.items():
        if not hasattr(test_mmio, example):  # pragma: no cover
            continue
        mm_in = StringIO(getattr(test_mmio, example))
        if over64:
            with pytest.raises((OverflowError, ValueError)):
                # fast_matrix_market v1.4.5 raises ValueError instead of OverflowError
                M = gb.io.mmread(mm_in, engine)
        else:
            if (
                example == "_empty_lines_example"
                and engine in {"fmm", "auto"}
                and fmm is not None
                and fmm.__version__ == "1.4.5"
            ):
                # `fast_matrix_market` __version__ v1.4.5 does not handle this, but v1.5.0 does
                continue
            M = gb.io.mmread(mm_in, engine)
            if not M.isequal(expected):  # pragma: no cover (debug)
                print(example)
                print("Expected:")
                print(expected)
                print("Got:")
                print(M)
                raise AssertionError("Matrix M not as expected.  See print output above")
            mm_out = BytesIO()
            gb.io.mmwrite(mm_out, M, engine)
            mm_out.flush()
            mm_out.seek(0)
            mm_out_str = b"".join(mm_out.readlines()).decode()
            mm_out.seek(0)
            M2 = gb.io.mmread(mm_out, engine)
            if not M2.isequal(expected):  # pragma: no cover (debug)
                print(example)
                print("Expected:")
                print(expected)
                print("Got:")
                print(M2)
                print("Matrix Market file:")
                print(mm_out_str)
                raise AssertionError("Matrix M2 not as expected.  See print output above")
        success += 1
    assert success > 0


@pytest.mark.skipif("not ss")
def test_mmread_asks_scipy_for_a_sparse_array(monkeypatch):
    """We pass ``spmatrix=False`` so scipy's changing default never reaches us.

    Deliberately does not use scipy's own example files: they are absent from
    some conda-forge builds, which skips ``test_mmread_mmwrite`` entirely.
    """
    import inspect

    import scipy.io as sio

    if "spmatrix" not in inspect.signature(sio.mmread).parameters:  # pragma: no cover (old scipy)
        pytest.skip("scipy too old to accept spmatrix=")

    seen = []
    real = sio.mmread

    @functools.wraps(real)  # keep the signature the caller inspects
    def spy(source, **kwargs):
        seen.append(kwargs)
        return real(source, **kwargs)

    monkeypatch.setattr(sio, "mmread", spy)
    mm = StringIO("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 3.0\n2 2 4.0\n")
    M = gb.io.mmread(mm)

    assert seen == [{"spmatrix": False}]
    assert M.isequal(Matrix.from_coo([0, 1], [0, 1], [3.0, 4.0]))

    # An explicit value wins: the guard only fills in a default.
    seen.clear()
    mm.seek(0)
    gb.io.mmread(mm, spmatrix=True)
    assert seen == [{"spmatrix": True}]


@pytest.mark.skipif("not ss")
def test_from_scipy_sparse_duplicates():
    a = ss.coo_array(([1, 2, 3, 4], ([0, 1, 2, 2], [2, 1, 0, 0])))
    np.testing.assert_array_equal(a.toarray(), np.array([[0, 0, 1], [0, 2, 0], [7, 0, 0]]))
    with pytest.raises(ValueError, match="Duplicate indices found"):
        gb.io.from_scipy_sparse(a)
    a2 = gb.io.from_scipy_sparse(a, dup_op=gb.binary.plus)
    expected = gb.Matrix.from_coo([0, 1, 2], [2, 1, 0], [1, 2, 7])
    assert a2.isequal(expected)


@pytest.mark.skipif("not ss")
@pytest.mark.parametrize("engine", ["auto", "scipy", "fast_matrix_market"])
def test_matrix_market_sparse_duplicates(engine):
    if engine == "fast_matrix_market" and fmm is None:  # pragma: no cover (import)
        pytest.skip("needs fast_matrix_market")
    string = """%%MatrixMarket matrix coordinate real general
        3 3 4
        1 3 1
        2 2 2
        3 1 3
        3 1 4"""
    mm = StringIO(string)
    with pytest.raises(ValueError, match="Duplicate indices found"):
        gb.io.mmread(mm, engine)
    # mm.seek(0)  # Doesn't work with `fast_matrix_market` 1.4.5
    mm = StringIO(string)
    a = gb.io.mmread(mm, engine, dup_op=gb.binary.plus)
    expected = gb.Matrix.from_coo([0, 1, 2], [2, 1, 0], [1, 2, 7])
    assert a.isequal(expected)


@pytest.mark.skipif("not ss")
def test_matrix_market_bad_engine():
    A = gb.Matrix.from_coo([0, 0, 3, 5], [1, 4, 0, 2], [1, 0, 2, -1], nrows=7, ncols=6)
    with pytest.raises(ValueError, match="Bad engine value"):
        gb.io.mmwrite(BytesIO(), A, engine="bad_engine")
    mm_out = BytesIO()
    gb.io.mmwrite(mm_out, A)
    mm_out.seek(0)
    with pytest.raises(ValueError, match="Bad engine value"):
        gb.io.mmread(mm_out, engine="bad_engine")


@pytest.mark.skipif("not ss")
@pytest.mark.skipif("fmm is not None")
def test_matrix_market_fmm_engine_unavailable():
    # Naming the deprecated engine warns before anything else, so the caller hears
    # about the deprecation even when the install that would satisfy it is missing.
    A = gb.Matrix.from_coo([0, 0, 3, 5], [1, 4, 0, 2], [1, 0, 2, -1], nrows=7, ncols=6)
    with (
        pytest.warns(DeprecationWarning, match="fast_matrix_market is no longer maintained"),
        pytest.raises(ImportError, match="required to write Matrix Market files"),
    ):
        gb.io.mmwrite(BytesIO(), A, engine="fmm")
    with (
        pytest.warns(DeprecationWarning, match="fast_matrix_market is no longer maintained"),
        pytest.raises(ImportError, match="required to read Matrix Market files"),
    ):
        gb.io.mmread(BytesIO(), engine="fast_matrix_market")


@pytest.mark.skipif("not ss")
def test_scipy_sparse():
    a = np.arange(12).reshape(3, 4)
    for a in [np.arange(12).reshape(3, 4), np.ones((3, 4)), np.zeros((3, 4))]:
        for fmt in ["bsr", "csr", "csc", "coo", "lil", "dia", "dok"]:
            sa = getattr(ss, f"{fmt}_array")(a)
            A = gb.io.from_scipy_sparse(sa)
            for M in [A, A.T.new().T]:
                if fmt == "dia" and M.nvals > 0:  # "dia" format is weird
                    assert sa.nnz == M.nrows * M.ncols
                else:
                    assert sa.nnz == M.nvals
                assert sa.shape == M.shape
                sa2 = gb.io.to_scipy_sparse(M, fmt)
                assert (sa != sa2).nnz == 0


@pytest.mark.skipif("not ak")
@pytest.mark.xfail(np.__version__[:5] in {"1.25.", "1.26."}, reason="awkward bug with numpy >=1.25")
def test_awkward_roundtrip():
    # Vector
    v = gb.Vector.from_coo([1, 3, 5], [20, 21, -5], size=22)
    for dtype in ["int16", "float32", "bool"]:
        v1 = v.dup(dtype=dtype)
        kv = gb.io.to_awkward(v1)
        assert isinstance(kv, ak.Array)
        v2 = gb.io.from_awkward(kv)
        assert v2.isequal(v1)
    # Matrix
    m = gb.Matrix.from_coo([0, 0, 3, 5], [1, 4, 0, 2], [1, 0, 2, -1], nrows=7, ncols=6)
    for dtype in ["int16", "float32", "bool"]:
        for format in ["csr", "csc", "hypercsr", "hypercsc"]:
            m1 = m.dup(dtype=dtype)
            km = gb.io.to_awkward(m1, format=format)
            assert isinstance(km, ak.Array)
            m2 = gb.io.from_awkward(km)
            assert m2.isequal(m1)


@pytest.mark.skipif("not ak")
@pytest.mark.xfail(np.__version__[:5] in {"1.25.", "1.26."}, reason="awkward bug with numpy >=1.25")
def test_awkward_iso_roundtrip():
    # Vector
    v = gb.Vector.from_coo([1, 3, 5], [20, 20, 20], size=22)
    if suitesparse:
        assert v.ss.is_iso
    kv = gb.io.to_awkward(v)
    assert isinstance(kv, ak.Array)
    v2 = gb.io.from_awkward(kv)
    assert v2.isequal(v)
    # Matrix
    m = gb.Matrix.from_coo([0, 0, 3, 5], [1, 4, 0, 2], [1, 1, 1, 1], nrows=7, ncols=6)
    if suitesparse:
        assert m.ss.is_iso
    for format in ["csr", "csc", "hypercsr", "hypercsc"]:
        km = gb.io.to_awkward(m, format=format)
        assert isinstance(km, ak.Array)
        m2 = gb.io.from_awkward(km)
        assert m2.isequal(m)


@pytest.mark.skipif("not ak")
def test_awkward_errors():
    v = gb.Vector.from_coo([1, 3, 5], [20, 20, 20], size=22)
    m = gb.Matrix.from_coo([0, 0, 3, 5], [1, 4, 0, 2], [1, 1, 1, 1], nrows=7, ncols=6)
    with pytest.raises(ValueError, match="Missing parameters"):
        gb.io.from_awkward(ak.Array([1, 2, 3]))
    kv = gb.io.to_awkward(v)
    kv = ak.with_parameter(kv, "format", "csr")
    with pytest.raises(ValueError, match="Invalid format for Vector"):
        gb.io.from_awkward(kv)
    km = gb.io.to_awkward(m)
    km = ak.with_parameter(km, "format", "dcsr")
    with pytest.raises(ValueError, match="Invalid format for Matrix"):
        gb.io.from_awkward(km)
    with pytest.raises(ValueError, match="Invalid format for Vector"):
        gb.io.to_awkward(v, format="csr")
    with pytest.raises(ValueError, match="Invalid format for Matrix"):
        gb.io.to_awkward(m, format="dcsr")
    with pytest.raises(TypeError):
        gb.io.to_awkward(gb.Scalar.from_value(5))


@pytest.mark.skipif("not sparse")
@pytest.mark.slow
def test_vector_to_from_pydata_sparse():
    coords = np.array([0, 1, 2, 3, 4], dtype="int64")
    data = np.array([10, 20, 30, 40, 50], dtype="int64")
    s = sparse.COO(coords, data, shape=(5,))
    v = gb.io.from_pydata_sparse(s)
    assert v.isequal(gb.Vector.from_coo(coords, data, dtype=dtypes.INT64), check_dtype=True)

    t = gb.io.to_pydata_sparse(v)
    assert t.shape == s.shape
    assert (t == s).all()


@pytest.mark.skipif("not sparse")
@pytest.mark.slow
def test_matrix_to_from_pydata_sparse():
    coords = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype="int64")
    data = np.array([10, 20, 30, 40, 50], dtype="int64")
    s = sparse.COO(coords, data, shape=(5, 5))
    v = gb.io.from_pydata_sparse(s)
    assert v.isequal(gb.Matrix.from_coo(*coords, data, dtype=dtypes.INT64), check_dtype=False)

    t = gb.io.to_pydata_sparse(v)
    assert t.shape == s.shape
    assert (t == s).all()

    # test ndim
    e = sparse.random(shape=(5, 5, 5), density=0)
    with pytest.raises(GraphblasException):
        gb.io.from_pydata_sparse(e)

    # test GCXS array conversion
    indptr = np.array([0, 2, 3, 6], dtype="int64")
    indices = np.array([0, 2, 2, 0, 1, 2], dtype="int64")
    data = np.array([1, 2, 3, 4, 5, 6], dtype="int64")

    g = sparse.GCXS((data, indices, indptr), shape=(3, 3), compressed_axes=[0])
    w = gb.io.from_pydata_sparse(g)
    coords = g.asformat("coo").coords
    data = g.asformat("coo").data
    assert w.isequal(gb.Matrix.from_coo(*coords, data, dtype=dtypes.INT64), check_dtype=False)

    r = gb.io.to_pydata_sparse(w, format="gcxs")
    assert r.shape == g.shape
    assert (r == g).all()
    with pytest.raises(ValueError, match="format"):
        gb.io.to_pydata_sparse(w, format="badformat")
    with pytest.raises(TypeError, match="sparse.pydata"):
        gb.io.from_pydata_sparse(w)

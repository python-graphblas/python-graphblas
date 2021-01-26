import pytest
import grblas as gb
import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None
try:
    import scipy.sparse as ss
except ImportError:  # pragma: no cover
    ss = None


@pytest.mark.skipif("not ss")
def test_vector_to_from_numpy():
    a = np.array([0.0, 2.0, 4.1])
    v = gb.io.from_numpy(a)
    assert v.isequal(gb.Vector.from_values([1, 2], [2.0, 4.1]), check_dtype=True)
    a2 = gb.io.to_numpy(v)
    np.testing.assert_array_equal(a, a2)

    csr = gb.io.to_scipy_sparse_matrix(v, "csr")
    assert csr.nnz == 2
    assert ss.isspmatrix_csr(csr)
    np.testing.assert_array_equal(csr.toarray(), np.array([[0.0, 2.0, 4.1]]))

    csc = gb.io.to_scipy_sparse_matrix(v, "csc")
    assert csc.nnz == 2
    assert ss.isspmatrix_csc(csc)
    np.testing.assert_array_equal(csc.toarray(), np.array([[0.0, 2.0, 4.1]]).T)

    # default to csr-like
    coo = gb.io.to_scipy_sparse_matrix(v, "coo")
    assert coo.shape == csr.shape
    assert ss.isspmatrix_coo(coo)
    assert coo.nnz == 2
    np.testing.assert_array_equal(coo.toarray(), np.array([[0.0, 2.0, 4.1]]))


@pytest.mark.skipif("not ss")
def test_matrix_to_from_numpy():
    a = np.array([[1.0, 0.0], [2.0, 3.7]])
    M = gb.io.from_numpy(a)
    assert M.isequal(gb.Matrix.from_values([0, 1, 1], [0, 0, 1], [1.0, 2.0, 3.7]), check_dtype=True)
    a2 = gb.io.to_numpy(M)
    np.testing.assert_array_equal(a, a2)

    for format in ["csr", "csc", "coo"]:
        sparse = gb.io.to_scipy_sparse_matrix(M, format)
        assert getattr(ss, f"isspmatrix_{format}")(sparse)
        assert sparse.nnz == 3
        np.testing.assert_array_equal(sparse.toarray(), a)
        M2 = gb.io.from_scipy_sparse_matrix(sparse)
        assert M.isequal(M2, check_dtype=True)

    with pytest.raises(gb.exceptions.GrblasException, match="Invalid format"):
        gb.io.to_scipy_sparse_matrix(M, "bad format")

    with pytest.raises(gb.exceptions.GrblasException, match="ndim must be"):
        gb.io.from_numpy(np.array([[[1.0, 0.0], [2.0, 3.7]]]))


@pytest.mark.skipif("not nx or not ss")
def test_matrix_to_from_networkx():
    M = gb.Matrix.from_values([0, 1, 1], [0, 0, 1], [1, 2, 3])
    G = gb.io.to_networkx(M)
    a = np.array([[1, 0], [2, 3]])
    G2 = nx.from_numpy_array(a, create_using=nx.DiGraph)
    assert G.number_of_edges() == G2.number_of_edges() == 3
    assert G.number_of_nodes() == G2.number_of_nodes() == 2
    np.testing.assert_array_equal(nx.to_numpy_array(G), a)

    M2 = gb.io.from_networkx(G, dtype=int)
    assert M.isequal(M2, check_dtype=True)

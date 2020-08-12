from . import Matrix, Vector
from .dtypes import lookup_dtype, FP64
from .exceptions import GrblasException
from .matrix import TransposedMatrix


def draw(m):
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover
        print("`draw` requires networkx to be installed")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("`draw` requires matplotlib to be installed")
        return

    if not isinstance(m, (Matrix, TransposedMatrix)):
        print(f"Can only draw a Matrix, not {type(m)}")
        return

    g = to_networkx(m)
    pos = nx.spring_layout(g)
    edge_labels = {(i, j): d["weight"] for i, j, d in g.edges(data=True)}
    nx.draw_networkx(g, pos, node_color="red", node_size=500)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()


def from_networkx(g, dtype=FP64):
    """
    Returns a Matrix
    """
    import networkx as nx

    ss = nx.convert_matrix.to_scipy_sparse_matrix(g)
    nrows, ncols = ss.shape
    rows, cols = ss.nonzero()
    m = Matrix.from_values(rows, cols, ss.data, nrows=nrows, ncols=ncols, dtype=dtype)
    return m


# TODO: add parameter to indicate empty value (default is 0 and NaN)
def from_numpy(m):
    """
    If m.ndim == 1, returns a Vector
    if m.ndim == 2, returns a Matrix
    if m.ndim > 2, raises an error
    dtype is inferred from m.dtype
    """
    if m.ndim > 2:
        raise GrblasException("m.ndim must be <= 2")

    try:
        from scipy.sparse import csr_matrix
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to import from numpy")

    if m.ndim == 1:
        ss = csr_matrix([m])
        _, size = ss.shape
        dtype = lookup_dtype(m.dtype)
        g = Vector.from_values(ss.indices, ss.data, size=size, dtype=dtype)
        return g
    else:
        ss = csr_matrix(m)
        return from_scipy_sparse_matrix(ss)


def from_scipy_sparse_matrix(m):
    """
    dtype is inferred from m.dtype
    """
    ss = m.tocsr()
    nrows, ncols = ss.shape
    rows, cols = ss.nonzero()
    dtype = lookup_dtype(m.dtype)
    g = Matrix.from_values(rows, cols, ss.data, nrows=nrows, ncols=ncols, dtype=dtype)
    return g


# TODO: add parameters to allow different networkx classes and attribute names
def to_networkx(m):
    import networkx as nx

    g = nx.DiGraph()
    for row, col, val in zip(*m.to_values()):
        g.add_edge(row, col, weight=val)
    return g


def to_numpy(m):
    try:
        import scipy  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to export to numpy")
    if type(m) is Vector:
        return to_scipy_sparse_matrix(m).toarray()[0]
    else:
        sparse = to_scipy_sparse_matrix(m, "coo")
        return sparse.toarray()


def to_scipy_sparse_matrix(m, format="csr"):
    """
    format: str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
    """
    import scipy.sparse as ss

    format = format.lower()
    if type(m) is Vector:
        indices, data = m.to_values()
        if format == "csc":
            return ss.csc_matrix((data, indices, [0, len(data)]))
        else:
            rv = ss.csr_matrix((data, indices, [0, len(data)]))
            if format == "csr":
                return rv
    else:
        rows, cols, data = m.to_values()
        rv = ss.coo_matrix((data, (rows, cols)), shape=m.shape)
        if format == "coo":
            return rv
    if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
        raise GrblasException(f"Invalid format: {format}")
    return rv.asformat(format)

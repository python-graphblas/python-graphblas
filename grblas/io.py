from . import Matrix, Vector
from .dtypes import lookup_dtype, FP64
from .exceptions import GrblasException
from .matrix import TransposedMatrix


def draw(m):
    try:
        import networkx as nx
    except ImportError:
        print("`draw` requires networkx to be installed")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
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
    except ImportError:
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


def to_numpy(m, format="array"):
    try:
        import scipy  # noqa
    except ImportError:
        raise ImportError("scipy is required to export to numpy")

    ss = to_scipy_sparse_matrix(m, "coo")
    format = format.lower()
    if format == "matrix":
        return ss.todense()
    elif format == "array":
        return ss.toarray()
    else:
        raise GrblasException(f"Invalid format: {format}")


def to_scipy_sparse_matrix(m, format="csr"):
    """
    format: str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
    """
    from scipy.sparse import coo_matrix

    nrows, ncols = m.nrows, m.ncols
    rows, cols, data = m.to_values()
    ss = coo_matrix((list(data), (list(rows), list(cols))), shape=(nrows, ncols))
    format = format.lower()
    if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
        raise GrblasException(f"Invalid format: {format}")
    return ss.asformat(format)

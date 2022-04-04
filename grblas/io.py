from . import Matrix, Vector, backend
from .dtypes import lookup_dtype
from .exceptions import GrblasException
from .matrix import TransposedMatrix
from .utils import output_type


def draw(m):  # pragma: no cover
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


def from_networkx(G, nodelist=None, dtype=None, weight="weight", name=None):
    """
    Returns a Matrix
    """
    import networkx as nx

    dtype = dtype if dtype is None else lookup_dtype(dtype).np_type
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
    nrows, ncols = A.shape
    data = A.data
    if data.size == 0:
        return Matrix.new(A.dtype, nrows=nrows, ncols=ncols, name=name)
    if backend == "suitesparse":
        is_iso = (data[[0]] == data).all()
        if is_iso:
            data = data[[0]]
        M = Matrix.ss.import_csr(
            nrows=nrows,
            ncols=ncols,
            indptr=A.indptr,
            col_indices=A.indices,
            values=data,
            is_iso=is_iso,
            sorted_cols=getattr(A, "_has_sorted_indices", False),
            take_ownership=True,
            name=name,
        )
    else:  # pragma: no cover
        rows, cols = A.nonzero()
        M = Matrix.from_values(rows, cols, data, nrows=nrows, ncols=ncols, name=name)
    return M


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
        from scipy.sparse import coo_matrix, csr_matrix
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to import from numpy") from None

    if m.ndim == 1:
        ss = csr_matrix(m)
        _, size = ss.shape
        dtype = lookup_dtype(m.dtype)
        g = Vector.from_values(ss.indices, ss.data, size=size, dtype=dtype)
        return g
    else:
        ss = coo_matrix(m)
        return from_scipy_sparse_matrix(ss)


def from_scipy_sparse_matrix(m, *, dup_op=None, name=None):
    """
    dtype is inferred from m.dtype
    """
    ss = m.tocoo()
    nrows, ncols = ss.shape
    dtype = lookup_dtype(m.dtype)
    g = Matrix.from_values(
        ss.row, ss.col, ss.data, nrows=nrows, ncols=ncols, dtype=dtype, dup_op=dup_op, name=name
    )
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
        raise ImportError("scipy is required to export to numpy") from None
    if output_type(m) is Vector:
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
    if output_type(m) is Vector:
        indices, data = m.to_values()
        if format == "csc":
            return ss.csc_matrix((data, indices, [0, len(data)]), shape=(m._size, 1))
        else:
            rv = ss.csr_matrix((data, indices, [0, len(data)]), shape=(1, m._size))
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


def mmread(source, *, dup_op=None, name=None):
    """Read the contents of a Matrix Market filename or file into a new Matrix.

    This uses `scipy.io.mmread`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html

    For more information on the Matrix Market format, see:
    https://math.nist.gov/MatrixMarket/formats.html
    """
    try:
        from scipy.io import mmread  # noqa
        from scipy.sparse import coo_matrix  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to read Matrix Market files") from None
    array = mmread(source)
    if isinstance(array, coo_matrix):
        nrows, ncols = array.shape
        return Matrix.from_values(
            array.row, array.col, array.data, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    # SS, SuiteSparse-specific: import_full
    return Matrix.ss.import_fullr(values=array, take_ownership=True, name=name)


def mmwrite(target, matrix, *, comment="", field=None, precision=None, symmetry=None):
    """Write matrix to Matrix Market file `target`.

    This uses `scipy.io.mmwrite`.  See documentation for the parameters here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html
    """
    try:
        from scipy.io import mmwrite  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to write Matrix Market files") from None

    # Note: it would be preferable to write the data in full format if it is indeed full.
    #       However, there may be a bug: https://github.com/scipy/scipy/issues/13634
    #
    # SS, SuiteSparse-specific: format
    # fmt = matrix.ss.format
    # if fmt == 'fullr' or fmt == 'fullc':
    #    array = matrix.ss.export(format=fmt)['values']
    # else:

    array = to_scipy_sparse_matrix(matrix, format="coo")
    mmwrite(target, array, comment=comment, field=field, precision=precision, symmetry=symmetry)

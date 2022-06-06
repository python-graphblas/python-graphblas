import warnings

import numpy as np

from . import Matrix, Vector, backend
from .dtypes import lookup_dtype
from .exceptions import GraphblasException
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
        return Matrix(A.dtype, nrows=nrows, ncols=ncols, name=name)
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
        raise GraphblasException("m.ndim must be <= 2")

    try:
        from scipy.sparse import coo_array, csr_array
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to import from numpy") from None

    if m.ndim == 1:
        A = csr_array(m)
        _, size = A.shape
        dtype = lookup_dtype(m.dtype)
        g = Vector.from_values(A.indices, A.data, size=size, dtype=dtype)
        return g
    else:
        A = coo_array(m)
        return from_scipy_sparse(A)


def from_scipy_sparse_matrix(m, *, dup_op=None, name=None):
    """
    dtype is inferred from m.dtype
    """
    warnings.warn(
        "`from_scipy_sparse_matrix` is deprecated; please use `from_scipy_sparse` instead.",
        DeprecationWarning,
    )
    A = m.tocoo()
    nrows, ncols = A.shape
    dtype = lookup_dtype(m.dtype)
    g = Matrix.from_values(
        A.row, A.col, A.data, nrows=nrows, ncols=ncols, dtype=dtype, dup_op=dup_op, name=name
    )
    return g


def from_scipy_sparse(A, *, dup_op=None, name=None):
    """Create a Matrix from a scipy.sparse array or matrix.

    Input data in "csr" or "csc" format will be efficient when importing with SuiteSparse:GraphBLAS
    """
    nrows, ncols = A.shape
    dtype = lookup_dtype(A.dtype)
    if A.nnz == 0:
        return Matrix(dtype, nrows=nrows, ncols=ncols, name=name)
    if backend == "suitesparse" and A.format in {"csr", "csc"}:
        data = A.data
        is_iso = (data[[0]] == data).all()
        if is_iso:
            data = data[[0]]
        if A.format == "csr":
            return Matrix.ss.import_csr(
                nrows=nrows,
                ncols=ncols,
                indptr=A.indptr,
                col_indices=A.indices,
                values=data,
                is_iso=is_iso,
                sorted_cols=getattr(A, "_has_sorted_indices", False),
                name=name,
            )
        else:
            return Matrix.ss.import_csc(
                nrows=nrows,
                ncols=ncols,
                indptr=A.indptr,
                row_indices=A.indices,
                values=data,
                is_iso=is_iso,
                sorted_rows=getattr(A, "_has_sorted_indices", False),
                name=name,
            )
    if A.format != "coo":
        A = A.tocoo()
    return Matrix.from_values(
        A.row, A.col, A.data, nrows=nrows, ncols=ncols, dtype=dtype, dup_op=dup_op, name=name
    )


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
        return to_scipy_sparse(m).toarray()[0]
    else:
        sparse = to_scipy_sparse(m, "coo")
        return sparse.toarray()


def to_scipy_sparse_matrix(m, format="csr"):  # pragma: no cover
    """
    format: str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
    """
    import scipy.sparse as ss

    warnings.warn(
        "`to_scipy_sparse_matrix` is deprecated; please use `to_scipy_sparse` instead.",
        DeprecationWarning,
    )
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
        raise GraphblasException(f"Invalid format: {format}")
    return rv.asformat(format)


def to_scipy_sparse(A, format="csr"):
    """Create a scipy.sparse array (not matrix) from a GraphBLAS Matrix or Vector

    Zero values will not be included in the scipy.sparse array.

    Parameters
    ----------
    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
    """
    import scipy.sparse as ss

    format = format.lower()
    if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
        raise ValueError(f"Invalid format: {format}")
    if output_type(A) is Vector:
        indices, data = A.to_values()
        if format == "csc":
            return ss.csc_array((data, indices, [0, len(data)]), shape=(A._size, 1))
        else:
            rv = ss.csr_array((data, indices, [0, len(data)]), shape=(1, A._size))
            if format == "csr":
                return rv
    elif backend == "suitesparse" and format in {"csr", "csc"}:
        if A._is_transposed:
            info = A.T.ss.export("csc" if format == "csr" else "csr", sort=True)
            if "col_indices" in info:
                info["row_indices"] = info["col_indices"]
            else:
                info["col_indices"] = info["row_indices"]
        else:
            info = A.ss.export(format, sort=True)
        if info["is_iso"]:
            info["values"] = np.broadcast_to(info["values"], (A._nvals,))
        if format == "csr":
            return ss.csr_array(
                (info["values"], info["col_indices"], info["indptr"]), shape=A.shape
            )
        else:
            return ss.csc_array(
                (info["values"], info["row_indices"], info["indptr"]), shape=A.shape
            )
    else:
        rows, cols, data = A.to_values()
        rv = ss.coo_array((data, (rows, cols)), shape=A.shape)
        if format == "coo":
            return rv
    return rv.asformat(format)


def mmread(source, *, dup_op=None, name=None):
    """Read the contents of a Matrix Market filename or file into a new Matrix.

    This uses `scipy.io.mmread`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html

    For more information on the Matrix Market format, see:
    https://math.nist.gov/MatrixMarket/formats.html
    """
    try:
        from scipy.io import mmread
        from scipy.sparse import isspmatrix_coo
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to read Matrix Market files") from None
    array = mmread(source)
    if isspmatrix_coo(array):
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
        from scipy.io import mmwrite
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to write Matrix Market files") from None
    if backend == "suitesparse" and matrix.ss.format in {"fullr", "fullc"}:
        array = matrix.ss.export()["values"]
    else:
        array = to_scipy_sparse(matrix, format="coo")
    mmwrite(target, array, comment=comment, field=field, precision=precision, symmetry=symmetry)

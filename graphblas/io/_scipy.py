from .. import backend
from ..core.matrix import Matrix
from ..core.utils import normalize_values, output_type
from ..core.vector import Vector
from ..dtypes import lookup_dtype


def from_scipy_sparse(A, *, dup_op=None, name=None):
    """Create a Matrix from a scipy.sparse array or matrix.

    Input data in "csr" or "csc" format will be efficient when importing with SuiteSparse:GraphBLAS.

    Parameters
    ----------
    A : scipy.sparse
        Scipy sparse array or matrix
    dup_op : BinaryOp, optional
        Aggregation function for formats that allow duplicate entries (e.g. coo)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`

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
    if A.format == "csr":
        return Matrix.from_csr(A.indptr, A.indices, A.data, ncols=ncols, name=name)
    if A.format == "csc":
        return Matrix.from_csc(A.indptr, A.indices, A.data, nrows=nrows, name=name)
    if A.format != "coo":
        A = A.tocoo()
    return Matrix.from_coo(
        A.row, A.col, A.data, nrows=nrows, ncols=ncols, dtype=dtype, dup_op=dup_op, name=name
    )


def to_scipy_sparse(A, format="csr"):
    """Create a scipy.sparse array from a GraphBLAS Matrix or Vector.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str
        {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}

    Returns
    -------
    scipy.sparse array

    """
    import scipy.sparse as ss

    format = format.lower()
    if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
        raise ValueError(f"Invalid format: {format}")
    if output_type(A) is Vector:
        indices, data = A.to_coo()
        if format == "csc":
            return ss.csc_array((data, indices, [0, len(data)]), shape=(A._size, 1))
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
        values = normalize_values(A, info["values"], None, (A._nvals,), info["is_iso"])
        if format == "csr":
            return ss.csr_array((values, info["col_indices"], info["indptr"]), shape=A.shape)
        return ss.csc_array((values, info["row_indices"], info["indptr"]), shape=A.shape)
    elif format == "csr":
        indptr, cols, vals = A.to_csr()
        return ss.csr_array((vals, cols, indptr), shape=A.shape)
    elif format == "csc":
        indptr, rows, vals = A.to_csc()
        return ss.csc_array((vals, rows, indptr), shape=A.shape)
    else:
        rows, cols, data = A.to_coo()
        rv = ss.coo_array((data, (rows, cols)), shape=A.shape)
        if format == "coo":
            return rv
    return rv.asformat(format)

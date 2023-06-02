from ..core.matrix import Matrix
from ..core.utils import output_type
from ..core.vector import Vector
from ..exceptions import GraphblasException
from ._scipy import from_scipy_sparse, to_scipy_sparse


def from_pydata_sparse(s, *, dup_op=None, name=None):
    """Create a Vector or a Matrix from a pydata.sparse array or matrix.

    Input data in "gcxs" format will be efficient when importing with SuiteSparse:GraphBLAS.

    Parameters
    ----------
    s : sparse
        PyData sparse array or matrix (see https://sparse.pydata.org)
    dup_op : BinaryOp, optional
        Aggregation function for formats that allow duplicate entries (e.g. coo)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Vector`
    :class:`~graphblas.Matrix`
    """
    try:
        import sparse
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to import from pydata sparse") from None
    if not isinstance(s, sparse.SparseArray):
        raise TypeError(
            "from_pydata_sparse only accepts objects from the `sparse` library; "
            "see https://sparse.pydata.org"
        )
    if s.ndim > 2:
        raise GraphblasException("m.ndim must be <= 2")

    if s.ndim == 1:
        # the .asformat('coo') makes it easier to convert dok/gcxs using a single approach
        _s = s.asformat("coo")
        return Vector.from_coo(
            _s.coords, _s.data, dtype=_s.dtype, size=_s.shape[0], dup_op=dup_op, name=name
        )
    # handle two-dimensional arrays
    if isinstance(s, sparse.GCXS):
        return from_scipy_sparse(s.to_scipy_sparse(), dup_op=dup_op, name=name)
    if isinstance(s, (sparse.DOK, sparse.COO)):
        _s = s.asformat("coo")
        return Matrix.from_coo(
            *_s.coords,
            _s.data,
            nrows=_s.shape[0],
            ncols=_s.shape[1],
            dtype=_s.dtype,
            dup_op=dup_op,
            name=name,
        )
    raise ValueError(f"Unknown sparse array type: {type(s).__name__}")  # pragma: no cover (safety)


def to_pydata_sparse(A, format="coo"):
    """Create a pydata.sparse array from a GraphBLAS Matrix or Vector.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str
        {'coo', 'dok', 'gcxs'}

    Returns
    -------
    sparse array (see https://sparse.pydata.org)

    """
    try:
        from sparse import COO
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to export to pydata sparse") from None

    format = format.lower()
    if format not in {"coo", "dok", "gcxs"}:
        raise ValueError(f"Invalid format: {format}")

    if output_type(A) is Vector:
        indices, values = A.to_coo(sort=False)
        s = COO(indices, values, shape=A.shape)
    else:
        if format == "gcxs":
            B = to_scipy_sparse(A, format="csr")
        else:
            # obtain an intermediate conversion via hardcoded 'coo' intermediate object
            B = to_scipy_sparse(A, format="coo")
        # convert to pydata.sparse
        s = COO.from_scipy_sparse(B)

    # express in the desired format
    return s.asformat(format)

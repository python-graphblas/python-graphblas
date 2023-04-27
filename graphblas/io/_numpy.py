from warnings import warn

from ..core.utils import output_type
from ..core.vector import Vector
from ..dtypes import lookup_dtype
from ..exceptions import GraphblasException
from ._scipy import from_scipy_sparse, to_scipy_sparse


def from_numpy(m):  # pragma: no cover (deprecated)
    """Create a sparse Vector or Matrix from a dense numpy array.

    .. deprecated:: 2023.2.0
        `from_numpy` will be removed in a future release.
        Use `Vector.from_dense` or `Matrix.from_dense` instead.
        Will be removed in version 2023.10.0 or later

    A value of 0 is considered as "missing".

    - m.ndim == 1 returns a `Vector`
    - m.ndim == 2 returns a `Matrix`
    - m.ndim > 2 raises an error

    dtype is inferred from m.dtype

    Parameters
    ----------
    m : np.ndarray
        Input array

    See Also
    --------
    Matrix.from_dense
    Vector.from_dense
    from_scipy_sparse

    Returns
    -------
    Vector or Matrix
    """
    warn(
        "`graphblas.io.from_numpy` is deprecated; "
        "use `Matrix.from_dense` and `Vector.from_dense` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if m.ndim > 2:
        raise GraphblasException("m.ndim must be <= 2")

    try:
        from scipy.sparse import coo_array, csr_array
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to import from numpy") from None

    if m.ndim == 1:
        A = csr_array(m)
        _, size = A.shape
        dtype = lookup_dtype(m.dtype)
        return Vector.from_coo(A.indices, A.data, size=size, dtype=dtype)
    A = coo_array(m)
    return from_scipy_sparse(A)


def to_numpy(m):  # pragma: no cover (deprecated)
    """Create a dense numpy array from a sparse Vector or Matrix.

    .. deprecated:: 2023.2.0
        `to_numpy` will be removed in a future release.
        Use `Vector.to_dense` or `Matrix.to_dense` instead.
        Will be removed in version 2023.10.0 or later

    Missing values will become 0 in the output.

    numpy dtype will match the GraphBLAS dtype

    Parameters
    ----------
    m : Vector or Matrix
        GraphBLAS Vector or Matrix

    See Also
    --------
    to_scipy_sparse
    Matrix.to_dense
    Vector.to_dense

    Returns
    -------
    np.ndarray
    """
    warn(
        "`graphblas.io.to_numpy` is deprecated; "
        "use `Matrix.to_dense` and `Vector.to_dense` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        import scipy  # noqa: F401
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to export to numpy") from None
    if output_type(m) is Vector:
        return to_scipy_sparse(m).toarray()[0]
    sparse = to_scipy_sparse(m, "coo")
    return sparse.toarray()

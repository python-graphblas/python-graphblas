from .._ss.matrix import _concat_mn
from ..base import _expect_type
from ..dtypes import INT64
from ..matrix import Matrix, TransposedMatrix
from ..scalar import Scalar
from ..vector import Vector


class _graphblas_ss:
    """Used in `_expect_type`"""


_graphblas_ss.__name__ = "graphblas.ss"
_graphblas_ss = _graphblas_ss()


def diag(x, k=0, dtype=None, *, name=None):
    """
    GxB_Matrix_diag, GxB_Vector_diag

    Extract a diagonal Vector from a Matrix, or construct a diagonal Matrix
    from a Vector.  Unlike ``Matrix.diag`` and ``Vector.diag``, this function
    returns a new object.

    Parameters
    ----------
    x : Vector or Matrix
        The Vector to assign to the diagonal, or the Matrix from which to
        extract the diagonal.
    k : int, default 0
        Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
        and `k<0` for diagonals below the main diagonal.

    See Also
    --------
    Vector.diag
    Matrix.diag
    Vector.ss.build_diag
    Matrix.ss.build_diag

    """
    x = _expect_type(
        _graphblas_ss, x, (Matrix, TransposedMatrix, Vector), within="diag", argname="x"
    )
    if type(k) is not Scalar:
        k = Scalar.from_value(k, INT64, is_cscalar=True, name="")
    if dtype is None:
        dtype = x.dtype
    typ = type(x)
    if typ is Vector:
        size = x._size + abs(k.value)
        rv = Matrix(dtype, nrows=size, ncols=size, name=name)
        rv.ss.build_diag(x, k)
    else:
        if k.value < 0:
            size = min(x._nrows + k.value, x._ncols)
        else:
            size = min(x._ncols - k.value, x._nrows)
        if size < 0:
            size = 0
        rv = Vector(dtype, size=size, name=name)
        rv.ss.build_diag(x, k)
    return rv


def concat(tiles, dtype=None, *, name=None):
    """
    GxB_Matrix_concat

    Concatenate a 2D list of Matrix objects into a new Matrix, or a 1D list of
    Vector objects into a new Vector.  To concatenate into existing objects,
    use ``Matrix.ss.concat`` or `Vector.ss.concat`.

    Vectors may be used as `Nx1` Matrix objects when creating a new Matrix.

    This performs the opposite operation as ``split``.

    See Also
    --------
    Matrix.ss.split
    Matrix.ss.concat
    Vector.ss.split
    Vector.ss.concat

    """
    tiles, m, n, is_matrix = _concat_mn(tiles)
    if is_matrix:
        if dtype is None:
            dtype = tiles[0][0].dtype
        nrows = sum(row_tiles[0]._nrows for row_tiles in tiles)
        ncols = sum(tile._ncols for tile in tiles[0])
        rv = Matrix(dtype, nrows=nrows, ncols=ncols, name=name)
        rv.ss._concat(tiles, m, n)
    else:
        if dtype is None:
            dtype = tiles[0].dtype
        size = sum(tile._nrows for tile in tiles)
        rv = Vector(dtype, size=size, name=name)
        rv.ss._concat(tiles, m)
    return rv

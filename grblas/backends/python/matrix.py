from scipy.sparse import csr_matrix
from .base import BasePointer, GraphBlasContainer
from .context import handle_panic, return_error
from .constants import GrB_Info


class MatrixPtr(BasePointer):
    def set_matrix(self, matrix):
        self.instance = matrix


class Matrix(GraphBlasContainer):
    def __init__(self, matrix):
        assert isinstance(matrix, csr_matrix)
        self.matrix = matrix

    @classmethod
    def new_from_dtype(cls, dtype, nrows, ncols):
        matrix = csr_matrix((nrows, ncols), dtype=dtype)
        return cls(matrix)

    @classmethod
    def new_from_existing(cls, other):
        matrix = csr_matrix(other)
        return cls(matrix)

    @classmethod
    def get_pointer(cls):
        return MatrixPtr()


@handle_panic
def Matrix_new(A: MatrixPtr, dtype: type, nrows: int, ncols: int):
    if nrows <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, 'nrows must be > 0')
    if ncols <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, 'ncols must be > 0')
    matrix = Matrix.new_from_dtype(dtype, nrows, ncols)
    A.set_matrix(matrix)
    return GrB_Info.GrB_SUCCESS


@handle_panic
def Matrix_dup(C: MatrixPtr, A: Matrix):
    matrix = Matrix.new_from_existing(A)
    C.set_matrix(matrix)
    return GrB_Info.GrB_SUCCESS


@handle_panic
def Matrix_resize(C: Matrix, nrows: int, ncols: int):
    if nrows <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, 'nrows must be > 0')
    if ncols <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, 'ncols must be > 0')
    C.matrix.resize((nrows, ncols))
    return GrB_Info.GrB_SUCCESS


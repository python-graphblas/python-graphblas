import numba
import numpy as np
from scipy.sparse import csr_matrix

from .base import BasePointer, GraphBlasContainer
from .context import handle_panic, return_error
from .exceptions import GrB_Info


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
        return_error(GrB_Info.GrB_INVALID_VALUE, "nrows must be > 0")
    if ncols <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, "ncols must be > 0")
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
        return_error(GrB_Info.GrB_INVALID_VALUE, "nrows must be > 0")
    if ncols <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, "ncols must be > 0")
    C.matrix.resize((nrows, ncols))
    return GrB_Info.GrB_SUCCESS


# TODO: this is just the essential code; it needs to handle descriptors, masks, accumulators, etc
@handle_panic
def mxm(C, A, B, semiring):
    cr, cc = C.shape
    ar, ac = A.shape
    br, bc = B.shape
    if cr != ar:
        return_error(GrB_Info.GrB_DIMENSION_MISMATCH, "C.nrows != A.nrows")
    if cc != bc:
        return_error(GrB_Info.GrB_DIMENSION_MISMATCH, "C.ncols != B.ncols")
    if ac != br:
        return_error(GrB_Info.GrB_DIMENSION_MISMATCH, "A.nrows != B.ncols")
    b = B.tocsc()
    d, i, ip = _sparse_matmul(
        A.data,
        A.indices,
        A.indptr,
        b.data,
        b.indices,
        b.indptr,
        semiring.plus.op,
        semiring.times,
        semiring.plus.identity,
        C.dtype,
    )
    C.data = d
    C.indices = i
    C.indptr = ip
    return GrB_Info.GrB_SUCCESS


@numba.njit
def _sparse_matmul(
    a_data,
    a_indices,
    a_indptr,
    b_data,
    b_indices,
    b_indptr,
    plus,
    times,
    identity,
    dtype,
):
    # Final array size is unknown, so we give ourselves room and then adjust on the fly
    tmp_output_size = a_data.size * 2
    data = np.empty((tmp_output_size,), dtype=dtype)
    indices = np.empty((tmp_output_size,), dtype=a_indices.dtype)
    indptr = np.empty((a_indptr.size,), dtype=a_indptr.dtype)
    output_counter = 0
    for iptr in range(a_indptr.size - 1):
        indptr[iptr] = output_counter
        for jptr in range(b_indptr.size - 1):
            a_counter = a_indptr[iptr]
            a_stop = a_indptr[iptr + 1]
            b_counter = b_indptr[jptr]
            b_stop = b_indptr[jptr + 1]
            val = identity
            nonempty = False
            while a_counter < a_stop and b_counter < b_stop:
                a_k = a_indices[a_counter]
                b_k = b_indices[b_counter]
                if a_k == b_k:
                    val = plus(val, times(a_data[a_counter], b_data[b_counter]))
                    nonempty = True
                    a_counter += 1
                    b_counter += 1
                elif a_k < b_k:
                    a_counter += 1
                else:
                    b_counter += 1
            if nonempty:
                if output_counter >= tmp_output_size:
                    # We filled up the allocated space; copy existing data to a larger array
                    tmp_output_size *= 2
                    new_data = np.empty((tmp_output_size,), dtype=data.dtype)
                    new_indices = np.empty((tmp_output_size,), dtype=indices.dtype)
                    new_data[:output_counter] = data[:output_counter]
                    new_indices[:output_counter] = indices[:output_counter]
                    data = new_data
                    indices = new_indices
                data[output_counter] = val
                indices[output_counter] = jptr
                output_counter += 1
    # Add final entry to indptr (should indicate nnz in the output)
    nnz = output_counter
    indptr[iptr + 1] = nnz
    # Trim output arrays
    data = data[:nnz]
    indices = indices[:nnz]

    return (data, indices, indptr)

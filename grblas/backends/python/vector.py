from scipy.sparse import csr_matrix

from .base import BasePointer, GraphBlasContainer
from .context import handle_panic, return_error
from .exceptions import GrB_Info


class VectorPtr(BasePointer):
    def set_vector(self, vector):
        self.instance = vector


class Vector(GraphBlasContainer):
    def __init__(self, vector):
        assert isinstance(vector, csr_matrix)
        self.vector = vector

    @classmethod
    def new_from_dtype(cls, dtype, nsize):
        vector = csr_matrix((1, nsize), dtype=dtype)
        return cls(vector)

    @classmethod
    def new_from_existing(cls, other):
        vector = csr_matrix(other)
        return cls(vector)

    @classmethod
    def get_pointer(cls):
        return VectorPtr()


@handle_panic
def Vector_new(A: VectorPtr, dtype: type, nsize: int):
    if nsize <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, "nsize must be > 0")
    vector = Vector.new_from_dtype(dtype, nsize)
    A.set_vector(vector)
    return GrB_Info.GrB_SUCCESS


@handle_panic
def Vector_dup(C: VectorPtr, A: Vector):
    vector = Vector.new_from_existing(A)
    C.set_vector(vector)
    return GrB_Info.GrB_SUCCESS


@handle_panic
def Vector_resize(C: Vector, nsize: int):
    if nsize <= 0:
        return_error(GrB_Info.GrB_INVALID_VALUE, "nsize must be > 0")
    C.vector.resize((1, nsize))
    return GrB_Info.GrB_SUCCESS

import numpy as np
from . import vector, matrix


GrB_ALL = object()
GrB_NULL = object()


class GrB_Type:
    GrB_BOOL = np.bool
    GrB_INT8 = np.int8
    GrB_UINT8 = np.uint8
    GrB_INT16 = np.int16
    GrB_UINT16 = np.uint16
    GrB_INT32 = np.int32
    GrB_UINT32 = np.uint32
    GrB_INT64 = np.int64
    GrB_UINT64 = np.uint64
    GrB_FP32 = np.float32
    GrB_FP64 = np.float64
    GrB_Index = np.uint64


class GrB_Info:
    GrB_SUCCESS = object()
    # API Errors
    GrB_UNINITIALIZED_OBJECT = object()
    GrB_NULL_POINTER = object()
    GrB_INVALID_VALUE = object()
    GrB_INVALID_INDEX = object()
    GrB_DOMAIN_MISMATCH = object()
    GrB_DIMENSION_MISMATCH = object()
    GrB_OUTPUT_NOT_EMPTY = object()
    GrB_NO_VALUE = object()
    # Execution Errors
    GrB_OUT_OF_MEMORY = object()
    GrB_INSUFFICIENT_SPACE = object()
    GrB_INVALID_OBJECT = object()
    GrB_INDEX_OUT_OF_BOUNDS = object()
    GrB_PANIC = object()


class GrB_Mode:
    GrB_BLOCKING = object()
    GrB_NONBLOCKING = object()


# Aliases
GrB_Vector = vector.Vector
GrB_Matrix = matrix.Matrix
# Algebra Methods
# Vector Methods
GrB_Vector_new = vector.Vector_new
GrB_Vector_dup = vector.Vector_dup
GrB_Vector_resize = vector.Vector_resize
# Matrix Methods
GrB_Matrix_new = matrix.Matrix_new
GrB_Matrix_dup = matrix.Matrix_dup
GrB_Matrix_resize = matrix.Matrix_resize
# Descriptor Methods
# Operations
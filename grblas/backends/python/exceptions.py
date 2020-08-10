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


class GraphBlasException(Exception):
    pass


last_error_message = None


def GrB_error():
    return last_error_message


def return_error(error, msg=""):
    global last_error_message
    last_error_message = msg
    return error

from . import ffi, lib


class GrblasException(Exception):
    pass


class NoValue(GrblasException):
    pass


class UninitializedObject(GrblasException):
    pass


class InvalidObject(GrblasException):
    pass


class NullPointer(GrblasException):
    pass


class InvalidValue(GrblasException):
    pass


class InvalidIndex(GrblasException):
    pass


class DomainMismatch(GrblasException):
    pass


class DimensionMismatch(GrblasException):
    pass


class OutputNotEmpty(GrblasException):
    pass


class OutOfMemory(GrblasException):
    pass


class InsufficientSpace(GrblasException):
    pass


class IndexOutOfBound(GrblasException):
    pass


class Panic(GrblasException):
    pass


# Our errors
class UdfParseError(GrblasException):
    pass


_error_code_lookup = {
    # Warning
    lib.GrB_NO_VALUE: NoValue,
    # API Errors
    lib.GrB_UNINITIALIZED_OBJECT: UninitializedObject,
    lib.GrB_INVALID_OBJECT: InvalidObject,
    lib.GrB_NULL_POINTER: NullPointer,
    lib.GrB_INVALID_VALUE: InvalidValue,
    lib.GrB_INVALID_INDEX: InvalidIndex,
    lib.GrB_DOMAIN_MISMATCH: DomainMismatch,
    lib.GrB_DIMENSION_MISMATCH: DimensionMismatch,
    lib.GrB_OUTPUT_NOT_EMPTY: OutputNotEmpty,
    # Execution Errors
    lib.GrB_OUT_OF_MEMORY: OutOfMemory,
    lib.GrB_INSUFFICIENT_SPACE: InsufficientSpace,
    lib.GrB_INDEX_OUT_OF_BOUNDS: IndexOutOfBound,
    lib.GrB_PANIC: Panic,
}


def is_error(response_code, error_class):
    if response_code in _error_code_lookup:
        if _error_code_lookup[response_code] == error_class:  # pragma: no branch
            return True
    return False


GrB_SUCCESS = lib.GrB_SUCCESS


def check_status(response_code):
    if response_code != GrB_SUCCESS:
        text = ffi.string(lib.GrB_error()).decode()
        raise _error_code_lookup[response_code](text)

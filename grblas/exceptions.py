from . import ffi, lib
from .dtypes import libget


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
GrB_SUCCESS = lib.GrB_SUCCESS
GrB_NO_VALUE = lib.GrB_NO_VALUE


def check_status(response_code, args_or_name, carg_or_None=None):
    if response_code != GrB_SUCCESS:
        if response_code == GrB_NO_VALUE:
            return NoValue
        if carg_or_None is not None:
            type_name = args_or_name
            carg = carg_or_None
        else:
            if type(args_or_name) in {tuple, list}:
                arg = args_or_name[0]
            else:
                arg = args_or_name
            type_name = type(arg).__name__
            carg = arg._carg

        error_func = libget(f"GrB_{type_name}_error")
        string = ffi.new("char**")
        error_func(string, carg)
        text = ffi.string(string[0]).decode()
        raise _error_code_lookup[response_code](text)

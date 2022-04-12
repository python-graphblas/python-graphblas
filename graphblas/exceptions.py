from . import ffi, lib
from .utils import _Pointer, libget


class GraphblasException(Exception):
    pass


class NoValue(GraphblasException):
    pass


class UninitializedObject(GraphblasException):
    pass


class InvalidObject(GraphblasException):
    pass


class NullPointer(GraphblasException):
    pass


class InvalidValue(GraphblasException):
    pass


class InvalidIndex(GraphblasException):
    pass


class DomainMismatch(GraphblasException):
    pass


class DimensionMismatch(GraphblasException):
    pass


class OutputNotEmpty(GraphblasException):
    pass


class OutOfMemory(GraphblasException):
    pass


class InsufficientSpace(GraphblasException):
    pass


class IndexOutOfBound(GraphblasException):
    pass


class Panic(GraphblasException):
    pass


class EmptyObject(GraphblasException):
    pass


class NotImplementedException(GraphblasException):
    pass


# Our errors
class UdfParseError(GraphblasException):
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
    lib.GrB_EMPTY_OBJECT: EmptyObject,
    # Execution Errors
    lib.GrB_OUT_OF_MEMORY: OutOfMemory,
    lib.GrB_INSUFFICIENT_SPACE: InsufficientSpace,
    lib.GrB_INDEX_OUT_OF_BOUNDS: IndexOutOfBound,
    lib.GrB_PANIC: Panic,
    lib.GrB_NOT_IMPLEMENTED: NotImplementedException,
    # GxB Errors
    lib.GxB_EXHAUSTED: StopIteration,
}
GrB_SUCCESS = lib.GrB_SUCCESS
GrB_NO_VALUE = lib.GrB_NO_VALUE


def check_status(response_code, args):
    if response_code == GrB_SUCCESS:
        return
    if response_code == GrB_NO_VALUE:
        return NoValue
    if type(args) is list:
        arg = args[0]
    else:
        arg = args
    if hasattr(arg, "_exc_arg"):
        arg = arg._exc_arg
    if type(arg) is _Pointer:
        arg = arg.val
    type_name = type(arg).__name__
    carg = arg._carg
    return check_status_carg(response_code, type_name, carg)


def check_status_carg(response_code, type_name, carg):
    if response_code == GrB_SUCCESS:
        return
    if response_code == GrB_NO_VALUE:  # pragma: no cover
        return NoValue
    try:
        error_func = libget(f"GrB_{type_name}_error")
    except AttributeError:  # pragma: no cover
        text = (
            f"Unable to get the error string for type {type_name}.  "
            "This is most likely a bug in graphblas.  Please report this as an issue at:\n"
            "    https://github.com/metagraph-dev/graphblas/issues\n"
            "Thanks (and sorry)!"
        )
    else:
        string = ffi.new("char**")
        error_func(string, carg)
        text = ffi.string(string[0]).decode()
    raise _error_code_lookup[response_code](text)

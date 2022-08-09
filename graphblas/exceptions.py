from . import ffi, lib
from .utils import _Pointer, libget


class GraphblasException(Exception):
    pass


class NoValue(GraphblasException):
    pass


class UninitializedObject(GraphblasException):
    pass


class InvalidObject(GraphblasException):
    """One of the collection objects (input or output)
    is in an invalid state due to a previous error.
    """


class NullPointer(GraphblasException):
    pass


class InvalidValue(GraphblasException):
    pass


class InvalidIndex(GraphblasException):
    """Provided index specifies a location outside the dimensions.

    This error is always raised immediately, even in non-blocking mode.
    """


class DomainMismatch(GraphblasException):
    """The domains (i.e. data types) of the inputs or outputs
    are incompatible for the operation.
    """


class DimensionMismatch(GraphblasException):
    """The input or output dimensions (i.e. shape) are not compatible."""


class OutputNotEmpty(GraphblasException):
    """Attempt to call :meth:`~graphblas.Matrix.build` on a non-empty object."""


class OutOfMemory(GraphblasException):
    """GraphBLAS ran out of memory when allocating space for the operation."""


class InsufficientSpace(GraphblasException):
    pass


class IndexOutOfBound(GraphblasException):
    """A provided index falls outside the dimensions.

    In non-blocking mode, this error can be deferred.
    """


class Panic(GraphblasException):
    """Unknown internal GraphBLAS error."""


class EmptyObject(GraphblasException):
    """A provided Scalar object is empty, but requires a value.

    This could happen, for example, if an empty Scalar is provided as the
    ``right`` argument to :meth:`~graphblas.Matrix.apply`.
    """


class NotImplementedException(GraphblasException):
    """The backend GraphBLAS implementation does not support
    the operation for the provided inputs.
    """


# Our errors
class UdfParseError(GraphblasException):
    """Unable to parse the user-defined function."""


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
            "    https://github.com/python-graphblas/python-graphblas/issues\n"
            "Thanks (and sorry)!"
        )
    else:
        string = ffi.new("char**")
        error_func(string, carg)
        text = ffi.string(string[0]).decode()
    raise _error_code_lookup[response_code](text)

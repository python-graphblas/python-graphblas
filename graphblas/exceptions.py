from .core import ffi as _ffi
from .core import lib as _lib
from .core.utils import _Pointer
from .core.utils import libget as _libget


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


# SuiteSparse errors
class JitError(GraphblasException):
    """SuiteSparse:GraphBLAS error using JIT."""


# Our errors
class UdfParseError(GraphblasException):
    """SuiteSparse:GraphBLAS unable to parse the user-defined function."""


_error_code_lookup = {
    # Warning
    _lib.GrB_NO_VALUE: NoValue,
    # API Errors
    _lib.GrB_UNINITIALIZED_OBJECT: UninitializedObject,
    _lib.GrB_INVALID_OBJECT: InvalidObject,
    _lib.GrB_NULL_POINTER: NullPointer,
    _lib.GrB_INVALID_VALUE: InvalidValue,
    _lib.GrB_INVALID_INDEX: InvalidIndex,
    _lib.GrB_DOMAIN_MISMATCH: DomainMismatch,
    _lib.GrB_DIMENSION_MISMATCH: DimensionMismatch,
    _lib.GrB_OUTPUT_NOT_EMPTY: OutputNotEmpty,
    _lib.GrB_EMPTY_OBJECT: EmptyObject,
    # Execution Errors
    _lib.GrB_OUT_OF_MEMORY: OutOfMemory,
    _lib.GrB_INSUFFICIENT_SPACE: InsufficientSpace,
    _lib.GrB_INDEX_OUT_OF_BOUNDS: IndexOutOfBound,
    _lib.GrB_PANIC: Panic,
    _lib.GrB_NOT_IMPLEMENTED: NotImplementedException,
}
GrB_SUCCESS = _lib.GrB_SUCCESS
GrB_NO_VALUE = _lib.GrB_NO_VALUE

# SuiteSparse-specific errors
if hasattr(_lib, "GxB_EXHAUSTED"):
    _error_code_lookup[_lib.GxB_EXHAUSTED] = StopIteration
if hasattr(_lib, "GxB_JIT_ERROR"):  # Added in 9.x
    _error_code_lookup[_lib.GxB_JIT_ERROR] = JitError


def check_status(response_code, args):
    if response_code == GrB_SUCCESS:
        return
    if response_code == GrB_NO_VALUE:
        return NoValue
    if isinstance(args, list):
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
    if response_code == GrB_NO_VALUE:  # pragma: no cover (safety)
        return NoValue
    try:
        error_func = _libget(f"GrB_{type_name}_error")
    except AttributeError:  # pragma: no cover (sanity)
        text = (
            f"Unable to get the error string for type {type_name}.  "
            "This is most likely a bug in graphblas.  Please report this as an issue at:\n"
            "    https://github.com/python-graphblas/python-graphblas/issues\n"
            "Thanks (and sorry)!"
        )
    else:
        string = _ffi.new("char**")
        error_func(string, carg)
        text = _ffi.string(string[0]).decode()
    raise _error_code_lookup[response_code](text)

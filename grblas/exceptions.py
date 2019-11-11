from _grblas import lib, ffi


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


_error_code_lookup = {
    1:  NoValue,
    2:  UninitializedObject,
    3:  InvalidObject,
    4:  NullPointer,
    5:  InvalidValue,
    6:  InvalidIndex,
    7:  DomainMismatch,
    8:  DimensionMismatch,
    9:  OutputNotEmpty,
    10: OutOfMemory,
    11: InsufficientSpace,
    12: IndexOutOfBound,
    13: Panic,
}


def is_error(response_code, error_class):
    if response_code in _error_code_lookup:
        if _error_code_lookup[response_code] == error_class:
            return True
    return False


def check_status(response_code):
    if response_code != lib.GrB_SUCCESS:
        raise _error_codes[res](ffi.string(lib.GrB_error()))

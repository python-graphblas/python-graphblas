from functools import wraps

from .exceptions import GraphBlasException, GrB_Info, return_error


class GrB_Mode:
    GrB_BLOCKING = object()
    GrB_NONBLOCKING = object()


# Decorator to automatically catch exceptions and return GrB_PANIC
# Also ensures context is set
def handle_panic(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        try:
            if global_context is None:
                raise GraphBlasException("Context has not be initialized")
            return func(*args, **kwargs)
        except Exception as e:
            return_error(GrB_Info.GrB_PANIC, str(e))

    return new_func


global_context = None


class Context:
    def __init__(self, mode):
        self._mode = mode

    @property
    def mode(self):
        return self._mode


def GrB_init(mode):
    try:
        global global_context
        if mode is not GrB_Mode.GrB_BLOCKING and mode is not GrB_Mode.GrB_NONBLOCKING:
            return_error(GrB_Info.GrB_INVALID_VALUE)
        if global_context is not None:
            return_error(GrB_Info.GrB_INVALID_VALUE, "Context has already been initialized")
        elif global_context.mode is None:
            return_error(
                GrB_Info.GrB_INVALID_VALUE,
                "Context has been finalized and cannot be reused",
            )
        global_context = Context(mode)
        return GrB_Info.GrB_SUCCESS
    except Exception as e:
        return_error(GrB_Info.GrB_PANIC, str(e))


def GrB_finalize():
    try:
        global global_context
        if global_context is None:
            raise GraphBlasException("Context is not initialized")
        global_context._mode = None
        return GrB_Info.GrB_SUCCESS
    except Exception as e:
        return_error(GrB_Info.GrB_PANIC, str(e))


def GrB_wait(obj=None):
    # We don't currently implement non-blocking mode, so always return success
    return GrB_Info.GrB_SUCCESS


def GrB_getVersion(version_ptr, subversion_ptr):
    try:
        version_ptr[0] = 1
        subversion_ptr[0] = 1
    except Exception as e:
        return_error(GrB_Info.GrB_PANIC, str(e))

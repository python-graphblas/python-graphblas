import importlib as _importlib

from . import backends, mask  # noqa
from ._version import get_versions  # noqa


class replace:
    """Singleton to indicate ``replace=True`` when updating objects.

    >>> C(mask, replace) << A.mxm(B)

    """

    def __repr__(self):
        return "replace"


replace = replace()


backend = None
_init_params = None
_SPECIAL_ATTRS = {
    "Matrix",
    "Recorder",
    "Scalar",
    "Vector",
    "_agg",
    "_ss",
    "agg",
    "base",
    "binary",
    "descriptor",
    "dtypes",
    "exceptions",
    "expr",
    "ffi",
    "formatting",
    "infix",
    "io",
    "lib",
    "matrix",
    "monoid",
    "op",
    "operator",
    "recorder",
    "scalar",
    "semiring",
    "ss",
    "tests",
    "unary",
    "utils",
    "vector",
}


def __getattr__(name):
    """Auto-initialize if special attrs used without explicit init call by user"""
    if name in _SPECIAL_ATTRS:
        if _init_params is None:
            _init("suitesparse", None, automatic=True)
        if name not in globals():
            _load(name)
        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys() | _SPECIAL_ATTRS)


def init(backend="suitesparse", blocking=False):
    """Initialize the chosen backend.

    Parameters
    ----------
    backend : str, one of {"suitesparse"}
    blocking : bool
        Whether to call GrB_init with GrB_BLOCKING or GrB_NONBLOCKING

    """
    _init(backend, blocking)


def _init(backend_arg, blocking, automatic=False):
    global _init_params, backend, lib, ffi

    passed_params = dict(backend=backend_arg, blocking=blocking, automatic=automatic)
    if _init_params is not None:
        if blocking is None:
            passed_params["blocking"] = _init_params["blocking"]
        if _init_params != passed_params:
            from .exceptions import GrblasException

            if _init_params.get("automatic"):
                raise GrblasException("grblas objects accessed prior to manual initialization")
            else:
                raise GrblasException(
                    "grblas initialized multiple times with different init parameters"
                )
        # Already initialized with these parameters; nothing more to do
        return

    backend = backend_arg
    if backend == "suitesparse":
        from suitesparse_graphblas import ffi, initialize, is_initialized, lib

        if is_initialized():
            mode = ffi.new("GrB_Mode*")
            assert lib.GxB_Global_Option_get(lib.GxB_MODE, mode) == 0
            is_blocking = mode[0] == lib.GrB_BLOCKING
            if blocking is None:
                passed_params["blocking"] = is_blocking
            elif is_blocking != blocking:
                raise RuntimeError(
                    f"GraphBLAS has already been initialized with `blocking={is_blocking}`"
                )
        else:
            if blocking is None:
                blocking = False
                passed_params["blocking"] = blocking
            initialize(blocking=blocking, memory_manager="c")
    else:
        raise ValueError(f'Bad backend name.  Must be "suitesparse".  Got: {backend}')
    _init_params = passed_params


_NEEDS_OPERATOR = {"_agg", "agg", "base", "io", "matrix", "scalar", "vector", "recorder", "ss"}


def _load(name):
    if name in {"Matrix", "Vector", "Scalar", "Recorder"}:
        module_name = name.lower()
        if module_name not in globals():
            _load(module_name)
        module = globals()[module_name]
        val = getattr(module, name)
        globals()[name] = val
    else:
        # Everything else is a module
        if name in _NEEDS_OPERATOR and "operator" not in globals():
            # Avoid circular imports
            _load("operator")
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module


__version__ = get_versions()["version"]
del get_versions

__all__ = [key for key in __dir__() if not key.startswith("_")]

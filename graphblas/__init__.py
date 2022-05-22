import sys as _sys
from importlib import import_module as _import_module

from ._version import get_versions


class replace:
    """Singleton to indicate ``replace=True`` when updating objects.

    >>> C(mask, replace) << A.mxm(B)

    """

    def __repr__(self):
        return "replace"

    def __reduce__(self):
        return "replace"


replace = replace()


def get_config():
    import os

    import donfig
    import yaml

    filename = os.path.join(os.path.dirname(__file__), "graphblas.yaml")
    config = donfig.Config("graphblas")
    with open(filename) as f:
        defaults = yaml.safe_load(f)
    config.update_defaults(defaults)
    return config


config = get_config()
del get_config

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
    "indexunary",
    "infix",
    "io",
    "lib",
    "mask",
    "matrix",
    "monoid",
    "op",
    "operator",
    "recorder",
    "scalar",
    "select",
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
            if f"graphblas.{name}" in _sys.modules:
                globals()[name] = _sys.modules[f"graphblas.{name}"]
            else:
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

    passed_params = {"backend": backend_arg, "blocking": blocking, "automatic": automatic}
    if _init_params is not None:
        if blocking is None:
            passed_params["blocking"] = _init_params["blocking"]
        if _init_params != passed_params:
            from .exceptions import GraphblasException

            if _init_params.get("automatic"):
                raise GraphblasException(
                    "graphblas objects accessed prior to manual initialization"
                )
            else:
                raise GraphblasException(
                    "graphblas initialized multiple times with different init parameters"
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
            initialize(blocking=blocking, memory_manager="numpy")
    else:
        raise ValueError(f'Bad backend name.  Must be "suitesparse".  Got: {backend}')
    _init_params = passed_params


# Ideally this is in operator.py, but lives here to avoid circular references
_STANDARD_OPERATOR_NAMES = set()


def _load(name):
    if name in {"Matrix", "Vector", "Scalar", "Recorder"}:
        module_name = name.lower()
        if module_name not in globals():
            _load(module_name)
        module = globals()[module_name]
        globals()[name] = getattr(module, name)
    else:
        # Everything else is a module
        globals()[name] = _import_module(f".{name}", __name__)


__version__ = get_versions()["version"]
del get_versions

__all__ = [key for key in __dir__() if not key.startswith("_")]

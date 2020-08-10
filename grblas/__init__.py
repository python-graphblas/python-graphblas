import importlib as _importlib
from . import backends, mask  # noqa

backend = None
_init_params = None
_SPECIAL_ATTRS = {
    "ffi",
    "lib",
    "Matrix",
    "Vector",
    "Scalar",
    "base",
    "descriptor",
    "dtypes",
    "exceptions",
    "expr",
    "formatting",
    "io",
    "ops",
    "unary",
    "binary",
    "monoid",
    "semiring",
    "matrix",
    "vector",
    "scalar",
    "tests",
}


def __getattr__(name):
    """Auto-initialize if special attrs used without explicit init call by user"""
    if name in _SPECIAL_ATTRS:
        if _init_params is None:
            _init("suitesparse", True, automatic=True)
        if name not in globals():
            _load(name)
        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has not attribute {name!r}")


def __dir__():
    return list(globals().keys() | _SPECIAL_ATTRS)


def init(backend="suitesparse", blocking=True):
    """Initialize the chosen backend.

    Parameters
    ----------
    backend : str, {"suitesparse", "pygraphblas"}
    blocking : bool
        Whether to call GrB_init with GrB_BLOCKING or GrB_NONBLOCKING

    The "pygraphblas" backend uses the pygraphblas Python package.
    If this backend is chosen, then GrB_init is not called (we defer to pygraphblas)
    and the `blocking` parameter is ignored.

    Choosing "pygraphblas" allows objects to be converted between the two libraries.
    Specifically, `Scalar`, `Vector`, and `Matrix` objects from `grblas` will now
    have `to_pygraphblas` and `from_pygraphblas` methods.
    """
    _init(backend, blocking)


def _init(backend_arg, blocking, automatic=False):
    global _init_params, backend, lib, ffi

    passed_params = dict(backend=backend_arg, blocking=blocking, automatic=automatic)
    if _init_params is None:
        _init_params = passed_params
    else:
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
    if backend == "pygraphblas":  # pragma: no cover
        import _pygraphblas, pygraphblas  # noqa

        lib = _pygraphblas.lib
        ffi = _pygraphblas.ffi
        # I think pygraphblas is only non-blocking.
        # Don't call GrB_init; defer to pygraphblas.
    else:
        ffi_backend = _importlib.import_module(f".backends.{backend}", __name__)
        lib = ffi_backend.lib
        ffi = ffi_backend.ffi
        # This must be called before anything else happens
        if blocking:
            ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_BLOCKING)
        else:
            ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_NONBLOCKING)


def _load(name):
    if name in {"Matrix", "Vector", "Scalar"}:
        module_name = name.lower()
        if module_name not in globals():
            _load(module_name)
        module = globals()[module_name]
        val = getattr(module, name)
        globals()[name] = val
    else:
        # Everything else is a module
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module


from ._version import get_versions  # noqa

__version__ = get_versions()["version"]
del get_versions

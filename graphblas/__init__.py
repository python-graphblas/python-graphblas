import sys as _sys
from importlib import import_module as _import_module


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
    from pathlib import Path

    import donfig
    import yaml

    config = donfig.Config("graphblas")
    path = Path(__file__).parent / "graphblas.yaml"
    with path.open() as f:
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
    "agg",
    "binary",
    "core",
    "dtypes",
    "exceptions",
    "indexunary",
    "io",
    "monoid",
    "op",
    "select",
    "semiring",
    "ss",
    "unary",
    "viz",
}


def __getattr__(name):
    """Auto-initialize if special attrs used without explicit init call by user."""
    if name in _SPECIAL_ATTRS:
        if _init_params is None:
            _init("suitesparse", None, automatic=True)
            # _init("suitesparse-vanilla", None, automatic=True)
        if name == "ss" and backend != "suitesparse":
            raise AttributeError(
                f'module {__name__!r} only has attribute "ss" when backend is "suitesparse"'
            )
        if name not in globals():
            if f"graphblas.{name}" in _sys.modules:
                globals()[name] = _sys.modules[f"graphblas.{name}"]
            else:
                _load(name)
        return globals()[name]
    if name == "_autoinit":
        if _init_params is None:
            _init("suitesparse", None, automatic=True)
        return
    if name == "__version__":
        from importlib.metadata import version

        try:
            return globals().setdefault("__version__", version("python-graphblas"))
        except Exception as exc:  # pragma: no cover (safety)
            raise AttributeError(
                "`graphblas.__version__` not available. This may mean python-graphblas was "
                "incorrectly installed or not installed at all. For local development, you may "
                "want to do an editable install via `python -m pip install -e path/to/graphblas`."
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = globals().keys() | _SPECIAL_ATTRS
    if backend is not None and backend != "suitesparse":
        names.remove("ss")
    names.add("__version__")
    return list(names)


def init(backend="suitesparse", blocking=False):
    """Initialize the chosen backend.

    Parameters
    ----------
    backend : str, one of {"suitesparse", "suitesparse-vanilla"}
    blocking : bool
        Whether to call GrB_init with GrB_BLOCKING or GrB_NONBLOCKING

    """
    _init(backend, blocking)


def _init(backend_arg, blocking, automatic=False):
    global _init_params, backend

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
            raise GraphblasException(
                "graphblas initialized multiple times with different init parameters"
            )
        # Already initialized with these parameters; nothing more to do
        return

    backend = backend_arg
    if backend in {"suitesparse", "suitesparse-vanilla"}:
        try:
            from suitesparse_graphblas import ffi, initialize, is_initialized, lib
        except ImportError:  # pragma: no cover (import)
            raise ImportError(
                f"suitesparse_graphblas is required for {backend!r} backend. "
                "It may be installed with pip or conda:\n\n"
                "    $ pip install suitesparse-graphblas\n"
                "    $ conda install -c conda-forge python-suitesparse-graphblas\n\n"
                "SuiteSparse:GraphBLAS is the primary C implementation and backend of "
                "python-graphblas and is what we recommend to most users. If you are "
                "installing python-graphblas with pip, we recommend installing with one "
                "of the following to automatically include suitespare-graphblas:\n\n"
                "    $ pip install python-graphblas[suitesparse]\n"
                "    $ pip install python-graphblas[default]"
            ) from None

        if is_initialized():
            mode = ffi.new("int32_t*")
            if lib.GxB_Global_Option_get_INT32(lib.GxB_MODE, mode) != 0:
                raise RuntimeError("Could not get GraphBLAS mode")  # pragma: no cover (safety)
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
        if backend == "suitesparse-vanilla":
            # Exclude functions that start with GxB

            class Lib:
                pass

            orig_lib = lib
            lib = Lib()
            for key, val in vars(orig_lib).items():
                # TODO: handle GxB objects
                if callable(val) and key.startswith("GxB") or "FC32" in key or "FC64" in key:
                    continue
                setattr(lib, key, getattr(orig_lib, key))
            for key in ["GxB_BACKWARDS", "GxB_STRIDE"]:
                delattr(lib, key)
    else:
        raise ValueError(
            f'Bad backend name.  Must be "suitesparse" or "suitesparse-vanilla".  Got: {backend}'
        )
    _init_params = passed_params

    from . import core

    core.ffi = ffi
    core.lib = lib
    core.NULL = ffi.NULL


# Ideally this is in operator.py, but lives here to avoid circular references
_STANDARD_OPERATOR_NAMES = set()


def _load(name):
    if name in {"Matrix", "Vector", "Scalar", "Recorder"}:
        module = _import_module(f".core.{name.lower()}", __name__)
        globals()[name] = getattr(module, name)
    else:
        # Everything else is a module
        globals()[name] = _import_module(f".{name}", __name__)


__all__ = [key for key in __dir__() if not key.startswith("_")]

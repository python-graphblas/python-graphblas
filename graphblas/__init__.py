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

    class Config(donfig.Config):
        """donfig Config that rejects silent attribute writes.

        Options are set with ``config.set(name=value)`` (optionally as a
        ``with config.set(name=value):`` block) and read with ``config[name]``.
        Plain attribute assignment (``config.name = value``) would otherwise
        create a dead instance attribute and leave the real option unchanged,
        so we raise instead.
        """

        def __init__(self, *args, **kwargs):
            # Writes are unrestricted until donfig's own __init__ finishes, and
            # the attributes it set there become the allowlist. Deriving that
            # list beats hardcoding donfig's internals: a donfig release that
            # adds or renames a field can't then break attribute access here.
            object.__setattr__(self, "_initializing", True)
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_donfig_attrs", frozenset(self.__dict__) - {"_initializing"})
            object.__setattr__(self, "_initializing", False)

        def __setattr__(self, key, value):
            # An instance built without running __init__ has no allowlist yet,
            # so stay permissive rather than raising a confusing AttributeError
            # from the guard.
            if self.__dict__.get("_initializing", True) or key in self._donfig_attrs:
                object.__setattr__(self, key, value)
                return
            if key in self.config:
                raise AttributeError(
                    f"Cannot set config option {key!r} by attribute assignment. "
                    f"Use `graphblas.config.set({key}=...)` to change it (optionally "
                    f"in a `with` block for a scoped change) and "
                    f"`graphblas.config[{key!r}]` to read it."
                )
            raise AttributeError(
                f"Unknown config option {key!r}; known options are {sorted(self.config)}."
            )

    config = Config("graphblas")
    path = Path(__file__).parent / "graphblas.yaml"
    with path.open() as f:
        defaults = yaml.safe_load(f)
    config.update_defaults(defaults)
    return config


config = get_config()
del get_config

# None until a backend is initialized. Touching a special attribute such as
# gb.Matrix auto-initializes, as does an explicit init(), and sets this to
# "suitesparse" or "suitesparse-vanilla". Reading gb.backend is not itself a
# special-attribute access, so it never triggers initialization.
backend = None
_init_params = None
_SPECIAL_ATTRS = {
    "MAX_SIZE",  # The maximum size of Vector and Matrix dimensions (GrB_INDEX_MAX + 1)
    "Matrix",
    "Recorder",
    "Scalar",
    "Tensor",  # Experimental: n-dimensional sparse arrays (graphblas.core.tensor)
    "Vector",
    "agg",
    "binary",
    "core",
    "dtypes",
    "einsum",  # Experimental, with Tensor
    "exceptions",
    "indexbinary",
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

    if backend_arg == "suitesparse":
        # Checked before anything is initialized, so a bad value changes nothing
        jit_c_control = _jit_c_control_config()
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
    if backend == "suitesparse":
        from .core.ss.jit_config import _set_jit_c_control_at_init

        _set_jit_c_control_at_init(jit_c_control)


_JIT_C_CONTROLS = ("on", "load", "run", "pause", "off")


def _jit_c_control_config():
    """``config["jit_c_control"]``, the SuiteSparse JIT control set at init (see graphblas.yaml).

    None when it is not set, which is the default: ``'run'``, raised by UDT work.
    """
    val = config.get("jit_c_control")
    if val is None:
        return None
    # YAML 1.1 reads an unquoted on/off as a bool, so a config file saying
    # ``jit_c_control: off`` arrives here as False.
    if val is True or val is False:
        return "on" if val else "off"
    if isinstance(val, str) and val.lower() in _JIT_C_CONTROLS:
        return val.lower()
    raise ValueError(
        "Bad value for graphblas.config['jit_c_control'] (or the GRAPHBLAS_JIT_C_CONTROL "
        f"environment variable): {val!r}. Must be one of: "
        + ", ".join(repr(x) for x in _JIT_C_CONTROLS)
        + ", or unset"
    )


# Ideally this is in operator.py, but lives here to avoid circular references
_STANDARD_OPERATOR_NAMES = set()


def _load(name):
    if name in {"Matrix", "Vector", "Scalar", "Recorder"}:
        module = _import_module(f".core.{name.lower()}", __name__)
        globals()[name] = getattr(module, name)
    elif name in {"Tensor", "einsum"}:
        module = _import_module(".core.tensor", __name__)
        globals()[name] = getattr(module, name)
    elif name == "MAX_SIZE":
        from .core import lib

        globals()[name] = lib.GrB_INDEX_MAX + 1
    else:
        # Everything else is a module
        globals()[name] = _import_module(f".{name}", __name__)


__all__ = [key for key in __dir__() if not key.startswith("_")]

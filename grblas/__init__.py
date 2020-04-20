import importlib
from . import backends  # noqa

_init_params = None
_SPECIAL_ATTRS = ["lib", "ffi", "Matrix", "Vector", "Scalar",
                  "base", "exceptions", "matrix", "ops", "scalar", "vector"
                  "unary", "binary", "monoid", "semiring", "numpyops"]


def __getattr__(name):
    """Auto-initialize if special attrs used without explicit init call by user"""
    if name in _SPECIAL_ATTRS:
        _init("suitesparse", True, automatic=True)

        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has not attribute {name!r}")


def __dir__():
    attrs = list(globals())
    if "lib" not in attrs:
        attrs += _SPECIAL_ATTRS
    return attrs


def init(backend="suitesparse", blocking=True):
    _init(backend, blocking)


def _init(backend, blocking, automatic=False):
    global _init_params, lib, ffi, Matrix, Vector, Scalar
    global base, exceptions, matrix, ops, scalar, vector
    global unary, binary, monoid, semiring, numpyops

    passed_params = dict(backend=backend, blocking=blocking, automatic=automatic)
    if _init_params is None:
        _init_params = passed_params
    else:
        if _init_params != passed_params:
            from .exceptions import GrblasException
            if _init_params.get("automatic"):
                raise GrblasException("grblas objects accessed prior to manual initialization")
            else:
                raise GrblasException("grblas initialized multiple times with different init parameters")

        # Already initialized with these parameters; nothing more to do
        return

    ffi_backend = importlib.import_module(f'.backends.{backend}', __name__)

    lib = ffi_backend.lib
    ffi = ffi_backend.ffi

    # This must be called before anything else happens
    if blocking:
        ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_BLOCKING)
    else:
        ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_NONBLOCKING)

    exceptions = importlib.import_module(f".exceptions", __name__)
    unary = importlib.import_module(f".unary", __name__)
    binary = importlib.import_module(f".binary", __name__)
    monoid = importlib.import_module(f".monoid", __name__)
    semiring = importlib.import_module(f".semiring", __name__)
    ops = importlib.import_module(f".ops", __name__)
    base = importlib.import_module(f".base", __name__)
    matrix = importlib.import_module(f".matrix", __name__)
    vector = importlib.import_module(f".vector", __name__)
    scalar = importlib.import_module(f".scalar", __name__)
    numpyops = importlib.import_module(f".numpyops", __name__)
    from .matrix import Matrix
    from .vector import Vector
    from .scalar import Scalar

    ops.UnaryOp._initialize()
    ops.BinaryOp._initialize()
    ops.Monoid._initialize()
    ops.Semiring._initialize()

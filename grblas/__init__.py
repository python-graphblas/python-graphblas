import importlib
from . import backends

_init_params = None
_SPECIAL_ATTRS = ["lib", "ffi", "Matrix", "Vector", "Scalar", "UnaryOp", "BinaryOp", "Monoid", "Semiring"]


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
    global lib, ffi, REPLACE, Matrix, Vector, Scalar, UnaryOp, BinaryOp, Monoid, Semiring, _init_params

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

    globals()["lib"] = ffi_backend.lib
    globals()["ffi"] = ffi_backend.ffi

    # This must be called before anything else happens
    if blocking:
        ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_BLOCKING)
    else:
        ffi_backend.lib.GrB_init(ffi_backend.lib.GrB_NONBLOCKING)

    from .base import REPLACE
    from .matrix import Matrix
    from .vector import Vector
    from .scalar import Scalar
    from .ops import UnaryOp, BinaryOp, Monoid, Semiring

    UnaryOp._initialize()
    BinaryOp._initialize()
    Monoid._initialize()
    Semiring._initialize()

    globals()["Matrix"] = Matrix
    globals()["Vector"] = Vector
    globals()["Scalar"] = Scalar
    globals()["UnaryOp"] = UnaryOp
    globals()["BinaryOp"] = BinaryOp
    globals()["Monoid"] = Monoid
    globals()["Semiring"] = Semiring
    globals()["REPLACE"] = REPLACE

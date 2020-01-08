import importlib
from . import backends

lib = None
ffi = None
REPLACE = None
Matrix = None
Vector = None
Scalar = None
UnaryOp, BinaryOp, Monoid, Semiring = None, None, None, None

def init(backend, blocking=True):
    global lib, ffi, REPLACE, Matrix, Vector, Scalar, UnaryOp, BinaryOp, Monoid, Semiring

    ffi_backend = importlib.import_module(f'.backends.{backend}', __name__)

    lib = ffi_backend.lib
    ffi = ffi_backend.ffi

    # This must be called before anything else happens
    if blocking:
        lib.GrB_init(lib.GrB_BLOCKING)
    else:
        lib.GrB_init(lib.GrB_NONBLOCKING)

    from .base import REPLACE
    from .matrix import Matrix
    from .vector import Vector
    from .scalar import Scalar
    from .ops import UnaryOp, BinaryOp, Monoid, Semiring

    UnaryOp._initialize()
    BinaryOp._initialize()
    Monoid._initialize()
    Semiring._initialize()

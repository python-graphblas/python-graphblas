from _grblas import lib, ffi
# This must be called before anything else happens
lib.GrB_init(lib.GrB_BLOCKING)

from .base import REPLACE
from .matrix import Matrix
from .vector import Vector
from .scalar import Scalar
from .ops import UnaryOp, BinaryOp, Monoid, Semiring

UnaryOp._initialize()
BinaryOp._initialize()
Monoid._initialize()
Semiring._initialize()
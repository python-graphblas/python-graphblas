from numba import njit
from numba import types as nt

from .types import GrB_Type


class OpContainer:
    def _compile(self, signatures, nosuffix=False):
        def ncompiler(func):
            funcname = f"GrB_{func.__name__.upper()}"
            for sig in signatures:
                if nosuffix:
                    typed_name = funcname
                else:
                    primary_dtype = sig.args[0]
                    suffix = GrB_Type.lookup_name(primary_dtype)
                    typed_name = f"{funcname}_{suffix}"
                jitted_func = njit(sig)(func)
                setattr(self, typed_name, jitted_func)

        return ncompiler


GrB_UnaryOp = OpContainer()
GrB_BinaryOp = OpContainer()


##################################
# Useful collections of signatures
##################################
_unary_bool = [nt.boolean(nt.boolean)]
_unary_int = [
    nt.uint8(nt.uint8),
    nt.int8(nt.int8),
    nt.uint16(nt.uint16),
    nt.int16(nt.int16),
    nt.uint32(nt.uint32),
    nt.int32(nt.int32),
    nt.uint64(nt.uint64),
    nt.int64(nt.int64),
]
_unary_float = [nt.float32(nt.float32), nt.float64(nt.float64)]
_unary_all = _unary_bool + _unary_int + _unary_float

_binary_bool = [nt.boolean(nt.boolean, nt.boolean)]
_binary_int = [
    nt.uint8(nt.uint8, nt.uint8),
    nt.int8(nt.int8, nt.int8),
    nt.uint16(nt.uint16, nt.uint16),
    nt.int16(nt.int16, nt.int16),
    nt.uint32(nt.uint32, nt.uint32),
    nt.int32(nt.int32, nt.int32),
    nt.uint64(nt.uint64, nt.uint64),
    nt.int64(nt.int64, nt.int64),
]
_binary_float = [nt.float32(nt.float32, nt.float32), nt.float64(nt.float64, nt.float64)]
_binary_all = _binary_bool + _binary_int + _binary_float

_binary_int_to_bool = [
    nt.boolean(nt.uint8, nt.uint8),
    nt.boolean(nt.int8, nt.int8),
    nt.boolean(nt.uint16, nt.uint16),
    nt.boolean(nt.int16, nt.int16),
    nt.boolean(nt.uint32, nt.uint32),
    nt.boolean(nt.int32, nt.int32),
    nt.boolean(nt.uint64, nt.uint64),
    nt.boolean(nt.int64, nt.int64),
]
_binary_float_to_bool = [
    nt.boolean(nt.float32, nt.float32),
    nt.boolean(nt.float64, nt.float64),
]
_binary_all_to_bool = _binary_bool + _binary_int_to_bool + _binary_float_to_bool

# NOTE:
# Most of these feel very simple and redundant, but there is a reason
# Even if an equivalent function exists in numpy or scipy,
# numba can't accept a passed-in function unless it is a jit'd function

#################
# Unary Operators
#################


@GrB_UnaryOp._compile(_unary_all)
def identity(x):
    """Identity"""
    return x


@GrB_UnaryOp._compile(_unary_int + _unary_float)
def abs(x):
    """Absolute value"""
    return abs(x)


@GrB_UnaryOp._compile(_unary_int + _unary_float)
def ainv(x):
    """Additive inverse"""
    return -x


@GrB_UnaryOp._compile(_unary_float)
def minv(x):
    """Multiplicative inverse"""
    return 1 / x


@GrB_UnaryOp._compile(_unary_bool, nosuffix=True)
def lnot(x):
    """Logical inverse"""
    return not x


@GrB_UnaryOp._compile(_unary_int)
def bnot(x):
    """Bitwise complement"""
    return ~x


##################
# Binary Operators
##################


@GrB_BinaryOp._compile(_binary_bool, nosuffix=True)
def lor(x, y):
    """Logical OR"""
    return x | y


@GrB_BinaryOp._compile(_binary_bool, nosuffix=True)
def land(x, y):
    """Logical AND"""
    return x & y


@GrB_BinaryOp._compile(_binary_bool, nosuffix=True)
def lxor(x, y):
    """Logical XOR"""
    return x ^ y


@GrB_BinaryOp._compile(_binary_bool, nosuffix=True)
def lxnor(x, y):
    """Logical XNOR"""
    return not (x ^ y)


@GrB_BinaryOp._compile(_binary_int)
def bor(x, y):
    """Bitwise OR"""
    return x | y


@GrB_BinaryOp._compile(_binary_int)
def band(x, y):
    """Bitwise AND"""
    return x & y


@GrB_BinaryOp._compile(_binary_int)
def bxor(x, y):
    """Bitwise XOR"""
    return x ^ y


@GrB_BinaryOp._compile(_binary_int)
def bxnor(x, y):
    """Bitwise XNOR"""
    return ~(x ^ y)


@GrB_BinaryOp._compile(_binary_all_to_bool)
def eq(x, y):
    """Equal"""
    return x == y


@GrB_BinaryOp._compile(_binary_all_to_bool)
def ne(x, y):
    """Not equal"""
    return x != y


@GrB_BinaryOp._compile(_binary_all_to_bool)
def gt(x, y):
    """Greater than"""
    return x > y


@GrB_BinaryOp._compile(_binary_all_to_bool)
def lt(x, y):
    """Less than"""
    return x < y


@GrB_BinaryOp._compile(_binary_all_to_bool)
def ge(x, y):
    """Greater than or equal"""
    return x >= y


@GrB_BinaryOp._compile(_binary_all_to_bool)
def le(x, y):
    """Less than or equal"""
    return x <= y


@GrB_BinaryOp._compile(_binary_all)
def first(x, y):
    """First argument"""
    return x


@GrB_BinaryOp._compile(_binary_all)
def second(x, y):
    """Second argument"""
    return y


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def min(x, y):
    """Minimum"""
    return min(x, y)


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def max(x, y):
    """Maximum"""
    return max(x, y)


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def plus(x, y):
    """Addition"""
    return x + y


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def minus(x, y):
    """Subtraction"""
    return x - y


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def times(x, y):
    """Multiplication"""
    return x * y


@GrB_BinaryOp._compile(_binary_int + _binary_float)
def div(x, y):
    """Division"""
    return x / y


class Monoid:
    def __init__(self, binaryop, identity):
        self.op = binaryop
        self.identity = identity


class Semiring:
    def __init__(self, plus_monoid, times_operator):
        self.plus = plus_monoid
        self.times = times_operator

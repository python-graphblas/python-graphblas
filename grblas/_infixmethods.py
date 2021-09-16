"""
comparisons = {
    "lt": "lt",
    "le": "le",
    "gt": "gt",
    "ge": "ge",
    "eq": "eq",
    "ne": "ne",
}
operations = {
    "add": "plus",
    "sub": "minus",
    "mul": "times",
    "truediv": "truediv",
    "floordiv": "floordiv",
    "mod": "numpy.mod",
    "pow": "pow",
}
custom = {
    "abs",
    "divmod",
    "invert",
    "neg",
    "rdivmod",
}
# Skipped: rshift, xor, pos
# Already used for syntax: lshift, and, or

for method, op in sorted(comparisons.items()):
    print(
        f'def __{method}__(self, other):\n'
        f'    return comparison(self, other, "__{method}__", binary.{op})\n\n'
    )
for method, op in sorted(operations.items()):
    print(
        f'def __{method}__(self, other):\n'
        f'    return binary.{op}(self, other)\n\n'
    )
    print(
        f'def __r{method}__(self, other):\n'
        f'    return binary.{op}(other, self)\n\n'
    )
    print(
        f'def __i{method}__(self, other):\n'
        f'    self << binary.{op}(self, other)\n'
        '    return self\n\n'
    )
methods = sorted(
    {f"__{x}__" for x in custom}
    | {f"__{x}__" for x in comparisons}
    | {f"__{x}__" for x in operations}
    | {f"__r{x}__" for x in operations}
    | {f"__i{x}__" for x in operations}
)
print(
    "d = globals()\n"
    f"for name in {methods}:\n"
    "    val = d[name]\n"
    "    setattr(Vector, name, val)\n"
    "    setattr(Matrix, name, val)\n"
    "    if not name.startswith('__i') or name == '__invert__':\n"
    "        setattr(TransposedMatrix, name, val)\n"
    "        setattr(VectorExpression, name, val)\n"
    "        setattr(MatrixExpression, name, val)\n"
    "        setattr(VectorInfixExpr, name, val)\n"
    "        setattr(MatrixInfixExpr, name, val)\n"
)
"""
from . import binary, unary
from .infix import MatrixInfixExpr, VectorInfixExpr
from .matrix import Matrix, MatrixExpression, TransposedMatrix
from .utils import output_type
from .vector import Vector, VectorExpression


def comparison(self, other, method, op):
    type1 = output_type(self)
    type2 = output_type(other)
    if (
        type1 is type2
        or type1 is Matrix
        and type2 is TransposedMatrix
        or type1 is TransposedMatrix
        and type2 is Matrix
    ):
        return op(self & other)
    return op(self, other)


def __divmod__(self, other):
    return (binary.floordiv(self, other), binary.numpy.mod(self, other))


def __rdivmod__(self, other):
    return (binary.floordiv(other, self), binary.numpy.mod(other, self))


def __abs__(self):
    return unary.abs(self)


def __invert__(self):
    return unary.lnot(self)


def __neg__(self):
    return unary.ainv(self)


# Paste here
def __eq__(self, other):
    return comparison(self, other, "__eq__", binary.eq)


def __ge__(self, other):
    return comparison(self, other, "__ge__", binary.ge)


def __gt__(self, other):
    return comparison(self, other, "__gt__", binary.gt)


def __le__(self, other):
    return comparison(self, other, "__le__", binary.le)


def __lt__(self, other):
    return comparison(self, other, "__lt__", binary.lt)


def __ne__(self, other):
    return comparison(self, other, "__ne__", binary.ne)


def __add__(self, other):
    return binary.plus(self, other)


def __radd__(self, other):
    return binary.plus(other, self)


def __iadd__(self, other):
    self << binary.plus(self, other)
    return self


def __floordiv__(self, other):
    return binary.floordiv(self, other)


def __rfloordiv__(self, other):
    return binary.floordiv(other, self)


def __ifloordiv__(self, other):
    self << binary.floordiv(self, other)
    return self


def __mod__(self, other):
    return binary.numpy.mod(self, other)


def __rmod__(self, other):
    return binary.numpy.mod(other, self)


def __imod__(self, other):
    self << binary.numpy.mod(self, other)
    return self


def __mul__(self, other):
    return binary.times(self, other)


def __rmul__(self, other):
    return binary.times(other, self)


def __imul__(self, other):
    self << binary.times(self, other)
    return self


def __pow__(self, other):
    return binary.pow(self, other)


def __rpow__(self, other):
    return binary.pow(other, self)


def __ipow__(self, other):
    self << binary.pow(self, other)
    return self


def __sub__(self, other):
    return binary.minus(self, other)


def __rsub__(self, other):
    return binary.minus(other, self)


def __isub__(self, other):
    self << binary.minus(self, other)
    return self


def __truediv__(self, other):
    return binary.truediv(self, other)


def __rtruediv__(self, other):
    return binary.truediv(other, self)


def __itruediv__(self, other):
    self << binary.truediv(self, other)
    return self


d = globals()
for name in [
    "__abs__",
    "__add__",
    "__divmod__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__iadd__",
    "__ifloordiv__",
    "__imod__",
    "__imul__",
    "__invert__",
    "__ipow__",
    "__isub__",
    "__itruediv__",
    "__le__",
    "__lt__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__pow__",
    "__radd__",
    "__rdivmod__",
    "__rfloordiv__",
    "__rmod__",
    "__rmul__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
]:
    val = d[name]
    setattr(Vector, name, val)
    setattr(Matrix, name, val)
    if not name.startswith("__i") or name == "__invert__":
        setattr(TransposedMatrix, name, val)
        setattr(VectorExpression, name, val)
        setattr(MatrixExpression, name, val)
        setattr(VectorInfixExpr, name, val)
        setattr(MatrixInfixExpr, name, val)

import numpy as np
from .. import ops

_unary_names = {
    # Math operations
    'negative',
    'abs',
    'absolute',
    'fabs',
    'rint',
    'sign',
    'conj',
    'exp',
    'exp2',
    'log',
    'log2',
    'log10',
    'expm1',
    'log1p',
    'sqrt',
    'square',
    'reciprocal',
    'conjugate',

    # Trigonometric functions
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'deg2rad',
    'rad2deg',
    'degrees',
    'radians',

    # Bit-twiddling functions
    'bitwise_not',
    'invert',

    # Comparison functions
    'logical_not',

    # Floating functions
    'isfinite',
    'isinf',
    'isnan',
    'signbit',
    'floor',
    'ceil',
    'trunc',
    'spacing',
}


def __dir__():
    return list(_unary_names)


def __getattr__(name):
    if name not in _unary_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    numpy_func = getattr(np, name)
    func = ops.UnaryOp.register_anonymous(lambda x: numpy_func(x))
    globals()[name] = func
    return func

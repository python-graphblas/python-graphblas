""" Create UDFs of numpy functions supported by numba.

See list of numpy ufuncs supported by numpy here:

https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#math-operations

"""
import numpy as np
from .. import ops, unary

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
    ops.UnaryOp.register_new(f'numpy.{name}', lambda x: numpy_func(x))
    rv = globals()[name]
    if name in {'invert', 'bitwise_not'}:
        # numba has difficulty compiling with bool dtypes, so fix our hack
        rv._specific_types['BOOL'] = unary.numpy.logical_not._specific_types['BOOL']
    return rv

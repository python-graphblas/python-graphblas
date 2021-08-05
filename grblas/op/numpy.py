from ..binary import numpy as _np_binary
from ..semiring import numpy as _np_semiring
from ..unary import numpy as _np_unary

_op_to_mod = dict.fromkeys(_np_unary.__all__, _np_unary)
_op_to_mod.update(dict.fromkeys(_np_binary.__all__, _np_binary))
_op_to_mod.update(dict.fromkeys(_np_semiring.__all__, _np_semiring))
__all__ = list(_op_to_mod)


def __dir__():
    return __all__


def __getattr__(name):
    try:
        rv = getattr(_op_to_mod[name], name)
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = rv
    return rv

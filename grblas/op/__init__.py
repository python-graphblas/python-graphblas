# All items are dynamically added by classes in operator.py
# This module acts as a container of all UnaryOp, BinaryOp, and Semiring instances
_delayed = {}
from grblas import operator  # noqa isort:skip
from . import numpy  # noqa isort:skip

del operator


def __dir__():
    from grblas.operator import OpBase, ParameterizedUdf

    keys = set(_delayed)
    keys.add("numpy")
    keys.update(
        key for key, val in globals().items() if isinstance(val, (OpBase, ParameterizedUdf))
    )
    return keys


def __getattr__(key):
    if key in _delayed:
        module = _delayed.pop(key)
        rv = getattr(module, key)
        globals()[key] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")

# All items are dynamically added by classes in operator.py
# This module acts as a container of UnaryOp instances
_delayed = {}
from grblas import operator  # noqa isort:skip
from . import numpy  # noqa isort:skip

del operator


def __dir__():
    from grblas.operator import ParameterizedUnaryOp, UnaryOp

    keys = set(_delayed)
    keys.add("numpy")
    keys.update(
        key for key, val in globals().items() if isinstance(val, (UnaryOp, ParameterizedUnaryOp))
    )
    return keys


def __getattr__(key):
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
        rv = func(**kwargs)
        globals()[key] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")

# All items are dynamically added by classes in operator.py
# This module acts as a container of all UnaryOp, BinaryOp, and Semiring instances
_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys()


def __getattr__(key):
    if key in _delayed:
        module = _delayed.pop(key)
        rv = getattr(module, key)
        globals()[key] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator  # noqa: E402 isort:skip
from . import numpy  # noqa: E402, F401 isort:skip

del operator

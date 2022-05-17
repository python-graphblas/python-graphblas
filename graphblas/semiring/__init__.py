# All items are dynamically added by classes in operator.py
# This module acts as a container of Semiring instances
_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys()


def __getattr__(key):
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
        if type(kwargs["binaryop"]) is str:
            from ..binary import from_string

            kwargs["binaryop"] = from_string(kwargs["binaryop"])
        if type(kwargs["monoid"]) is str:
            from ..monoid import from_string

            kwargs["monoid"] = from_string(kwargs["monoid"])
        rv = func(**kwargs)
        globals()[key] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from .. import operator  # noqa isort:skip
from . import numpy  # noqa isort:skip

del operator

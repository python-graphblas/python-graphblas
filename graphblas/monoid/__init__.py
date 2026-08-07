# All items are dynamically added by classes in operator.py
# This module acts as a container of Monoid instances
_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys() | {"ss"}


def __getattr__(key):
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
        if isinstance(kwargs["binaryop"], str):
            from ..binary import from_string

            kwargs["binaryop"] = from_string(kwargs["binaryop"])
        rv = func(**kwargs)
        globals()[key] = rv
        return rv
    if key == "ss":
        from .. import backend

        if backend != "suitesparse":
            raise AttributeError(
                f'module {__name__!r} only has attribute "ss" when backend is "suitesparse"'
            )
        from importlib import import_module

        ss = import_module(".ss", __name__)
        globals()["ss"] = ss
        return ss
    from ..core.utils import _module_attr_error

    raise _module_attr_error(__name__, key, __dir__())


from ..core import operator  # noqa: E402 isort:skip
from . import numpy  # noqa: E402 isort:skip

del operator

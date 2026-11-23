# All items are dynamically added by classes in operator.py.
# This module acts as a container of IndexBinaryOp instances.
from ..core import _build_delayed

_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys() | {"ss"}


def __getattr__(key):
    if (rv := _build_delayed(globals(), _delayed, key)) is not None:
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
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator  # noqa: E402 isort:skip

del operator

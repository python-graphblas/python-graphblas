# All items are dynamically added by classes in operator.py
# This module acts as a container of all UnaryOp, BinaryOp, and Semiring instances
from ..core import _build_delayed

_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys() | {"ss"}


def _build(key, module):
    return getattr(module, key)


def __getattr__(key):
    if (rv := _build_delayed(globals(), _delayed, key, _build)) is not None:
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
    if not _supports_udfs:
        from .. import binary, semiring

        if key in binary._udfs or key in semiring._udfs:
            raise AttributeError(
                f"module {__name__!r} unable to compile UDF for {key!r}; "
                "install numba for UDF support"
            )
    from ..core.utils import _module_attr_error

    raise _module_attr_error(__name__, key, __dir__())


from ..core import operator, _supports_udfs  # noqa: E402 isort:skip
from . import numpy  # noqa: E402 isort:skip

del operator

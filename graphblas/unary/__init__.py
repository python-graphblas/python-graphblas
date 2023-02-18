# All items are dynamically added by classes in operator.py
# This module acts as a container of UnaryOp instances
_delayed = {}
_deprecated = {}


def __dir__():
    return globals().keys() | _delayed.keys() | _deprecated.keys() | {"ss"}


def __getattr__(key):
    if key in _deprecated:
        import warnings

        alt = {"positioni": "indexunary.rowindex", "positionj": "indexunary.colindex"}.get(key, "")
        if alt:
            alt = f"`gb.{alt}` or "
        warnings.warn(
            f"`gb.unary.{key}` is deprecated; please use {alt}`gb.unary.ss.{key}` instead. "
            f"`{key}` is specific to SuiteSparse:GraphBLAS. "
            f"`gb.unary.{key}` will be removed in version 2023.9.0 or later.",
            DeprecationWarning,
            stacklevel=2,
        )
        rv = _deprecated[key]
        globals()[key] = rv
        return rv
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
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
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator  # noqa: E402 isort:skip
from . import numpy  # noqa: E402 isort:skip

del operator

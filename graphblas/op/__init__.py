# All items are dynamically added by classes in operator.py
# This module acts as a container of all UnaryOp, BinaryOp, and Semiring instances
_delayed = {}
_deprecated = {}


def __dir__():
    return globals().keys() | _delayed.keys() | _deprecated.keys() | {"ss"}


def __getattr__(key):
    if key in _deprecated:
        import warnings

        warnings.warn(
            f"`gb.op.{key}` is deprecated; please use `gb.op.ss.{key}` instead. "
            f"`{key}` is specific to SuiteSparse:GraphBLAS. "
            f"`gb.op.{key}` will be removed in version 2023.9.0 or later.",
            DeprecationWarning,
            stacklevel=2,
        )
        rv = _deprecated[key]
        globals()[key] = rv
        return rv
    if key in _delayed:
        module = _delayed.pop(key)
        rv = getattr(module, key)
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
    if not _supports_udfs:
        from .. import binary, semiring

        if key in binary._udfs or key in semiring._udfs:
            raise AttributeError(
                f"module {__name__!r} unable to compile UDF for {key!r}; "
                "install numba for UDF support"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator, _supports_udfs  # noqa: E402 isort:skip
from . import numpy  # noqa: E402 isort:skip

del operator

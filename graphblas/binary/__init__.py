# All items are dynamically added by classes in operator.py
# This module acts as a container of BinaryOp instances
from ..core import _supports_udfs

_delayed = {}
_delayed_commutes_to = {
    "absfirst": "abssecond",
    "abssecond": "absfirst",
    "floordiv": "rfloordiv",
    "rfloordiv": "floordiv",
    "rpow": "pow",
}
_deprecated = {}
_udfs = {
    "absfirst",
    "abssecond",
    "binom",
    "floordiv",
    "isclose",
    "rfloordiv",
    "rpow",
}


def __dir__():
    return globals().keys() | _delayed.keys() | _deprecated.keys() | {"ss"}


def __getattr__(key):
    if key in _deprecated:
        import warnings

        warnings.warn(
            f"`gb.binary.{key}` is deprecated; please use `gb.binary.ss.{key}` instead. "
            f"`{key}` is specific to SuiteSparse:GraphBLAS. "
            f"`gb.binary.{key}` will be removed in version 2023.9.0 or later.",
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
        if key in _delayed_commutes_to:
            other_key = _delayed_commutes_to[key]
            other = globals().get(other_key, other_key)
            rv._commutes_to = other
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
    if not _supports_udfs and key in _udfs:
        raise AttributeError(
            f"module {__name__!r} unable to compile UDF for {key!r}; "
            "install numba for UDF support"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator  # noqa: E402 isort:skip
from . import numpy  # noqa: E402 isort:skip

del operator

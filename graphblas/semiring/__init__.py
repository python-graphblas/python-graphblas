# All items are dynamically added by classes in operator.py
# This module acts as a container of Semiring instances
from ..core import _supports_udfs

_delayed = {}
_deprecated = {}
_udfs = {
    # Used by aggregators
    "max_absfirst",
    "max_abssecond",
    "plus_absfirst",
    "plus_abssecond",
    "plus_rpow",
    # floordiv
    "any_floordiv",
    "max_floordiv",
    "min_floordiv",
    "plus_floordiv",
    "times_floordiv",
    # rfloordiv
    "any_rfloordiv",
    "max_rfloordiv",
    "min_rfloordiv",
    "plus_rfloordiv",
    "times_rfloordiv",
}


def __dir__():
    return globals().keys() | _delayed.keys() | _deprecated.keys() | {"ss"}


def __getattr__(key):
    if key in _deprecated:
        import warnings

        warnings.warn(
            f"`gb.semiring.{key}` is deprecated; please use `gb.semiring.ss.{key}` instead. "
            f"`{key}` is specific to SuiteSparse:GraphBLAS. "
            f"`gb.semiring.{key}` will be removed in version 2023.9.0 or later.",
            DeprecationWarning,
            stacklevel=2,
        )
        rv = _deprecated[key]
        globals()[key] = rv
        return rv
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

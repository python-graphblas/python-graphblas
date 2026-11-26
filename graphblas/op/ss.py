from ..core import _build_delayed, operator

_delayed = {}

del operator


def __getattr__(name):
    from . import _build

    if (rv := _build_delayed(globals(), _delayed, name, _build)) is not None:
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_delayed})

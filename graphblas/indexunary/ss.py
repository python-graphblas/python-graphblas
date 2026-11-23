from ..core import _build_delayed, operator
from ..core.ss.indexunary import register_new  # noqa: F401

_delayed = {}

del operator


def __getattr__(name):
    if (rv := _build_delayed(globals(), _delayed, name)) is not None:
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

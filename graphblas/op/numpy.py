from ..binary import numpy as _np_binary
from ..core import _build_delayed, _supports_udfs
from ..semiring import numpy as _np_semiring
from ..unary import numpy as _np_unary

_delayed = {}
_op_to_mod = dict.fromkeys(_np_unary.__all__, _np_unary)
_op_to_mod.update(dict.fromkeys(_np_binary.__all__, _np_binary))
_op_to_mod.update(dict.fromkeys(_np_semiring.__all__, _np_semiring))
__all__ = list(_op_to_mod)


def __dir__():
    attrs = _delayed.keys() | _op_to_mod.keys()
    if not _supports_udfs:
        attrs &= _np_unary.__dir__() | _np_binary.__dir__() | _np_semiring.__dir__()
    return attrs | globals().keys()


def __getattr__(name):
    from . import _build

    if (rv := _build_delayed(globals(), _delayed, name, _build)) is not None:
        return rv
    try:
        rv = getattr(_op_to_mod[name], name)
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = rv
    return rv

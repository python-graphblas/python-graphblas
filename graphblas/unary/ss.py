from ..core import operator
from ..core.ss import _IS_SSGB7

if not _IS_SSGB7:
    from ..core.ss.unary import register_new  # noqa: F401

_delayed = {}

del operator

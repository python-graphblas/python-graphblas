"""An Array API (2025.12) namespace over ``graphblas.Tensor``: ``import graphblas.array_api as xp``.

**THIS API IS EXPERIMENTAL AND MAY CHANGE.**

An Array is logically dense: its value at a position is its stored entry there, or its
``fill_value`` where it stores none (pydata/sparse's model). Results equal numpy's on the same
dense data, and keep the stored entries sparse where the fill values allow.
"""

from . import (
    _constants,
    _creation,
    _dtypes,
    _elementwise,
    _indexing,
    _info,
    _linalg,
    _manipulation,
    _searching,
    _set,
    _sorting,
    _statistical,
)
from ._array import Array
from ._constants import *  # noqa: F403
from ._creation import *  # noqa: F403
from ._dtypes import *  # noqa: F403
from ._dtypes import DTYPES as _DTYPES
from ._elementwise import *  # noqa: F403
from ._indexing import *  # noqa: F403
from ._info import *  # noqa: F403
from ._linalg import *  # noqa: F403
from ._manipulation import *  # noqa: F403
from ._searching import *  # noqa: F403
from ._set import *  # noqa: F403
from ._sorting import *  # noqa: F403
from ._statistical import *  # noqa: F403
from ._utils import DensifyError

__array_api_version__ = "2025.12"

bool = _DTYPES["bool"]  # noqa: A001 (the Array API's name)
int8 = _DTYPES["int8"]
int16 = _DTYPES["int16"]
int32 = _DTYPES["int32"]
int64 = _DTYPES["int64"]
uint8 = _DTYPES["uint8"]
uint16 = _DTYPES["uint16"]
uint32 = _DTYPES["uint32"]
uint64 = _DTYPES["uint64"]
float32 = _DTYPES["float32"]
float64 = _DTYPES["float64"]
complex64 = _DTYPES["complex64"]
complex128 = _DTYPES["complex128"]

__all__ = [
    "Array",
    "DensifyError",
    "__array_api_version__",
    "bool",
    "complex64",
    "complex128",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
__all__ += _constants.__all__
__all__ += _creation.__all__
__all__ += _dtypes.__all__
__all__ += _elementwise.__all__
__all__ += _indexing.__all__
__all__ += _info.__all__
__all__ += _linalg.__all__
__all__ += _manipulation.__all__
__all__ += _searching.__all__
__all__ += _set.__all__
__all__ += _sorting.__all__
__all__ += _statistical.__all__

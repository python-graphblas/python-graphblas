"""An Array API (2025.12) namespace backed by ``graphblas.Tensor``.

**THIS API IS EXPERIMENTAL AND MAY CHANGE.**

Import it as ``import graphblas.array_api as xp``. An Array is logically dense, as in
pydata/sparse: its value at a position is its stored entry there, or its ``fill_value`` where
it stores none. Results match numpy's on the same dense data, and keep the stored entries
sparse where the fill values allow. Where they differ:

- transcendental functions are within libm's rounding of numpy's results, and sums may round
  differently;
- the sign of a zero is unspecified at a position the result does not store, and where 0.0
  and -0.0 compare equal (maximum, minimum, max, min and the unique functions);
- contractions such as ``matmul`` skip the unstored positions of an operand whose fill is 0,
  so ``0 * inf`` there is not NaN (see the Technical Notes page
  :ref:`Contractions in the Array API Namespace <tn_array_api_contractions>`).

An operation that would allocate a dense array larger than the ``array_api_densify_limit``
config option raises ``DensifyError``.
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
    linalg,
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

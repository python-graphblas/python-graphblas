import numpy as np

from .. import backend
from ..dtypes import _INDEX
from . import lib
from .expr import _ALL_INDICES, AxisIndex
from .scalar import Scalar, _as_scalar
from .utils import _CArray

if backend == "suitesparse":
    _gxb_range = Scalar.from_value(lib.GxB_RANGE, dtype=_INDEX, name="GxB_RANGE", is_cscalar=True)
    _gxb_stride = Scalar.from_value(
        lib.GxB_STRIDE, dtype=_INDEX, name="GxB_STRIDE", is_cscalar=True
    )
    _gxb_backwards = Scalar.from_value(
        lib.GxB_BACKWARDS, dtype=_INDEX, name="GxB_BACKWARDS", is_cscalar=True
    )


def slice_to_index(index, size):
    start, stop, step = index.indices(size)
    length = len(range(start, stop, step))
    if length == size and step == 1:
        # [:] means all indices; use special GrB_ALL indicator
        return AxisIndex(size, _ALL_INDICES, _as_scalar(size, _INDEX, is_cscalar=True), size)
    if backend != "suitesparse":
        # Danger zone: compute all the indices
        return AxisIndex(
            length,
            _CArray(np.arange(start, stop, step, dtype=np.uint64)),
            _as_scalar(length, _INDEX, is_cscalar=True),
            size,
        )

    # SS, SuiteSparse-specific: slicing.
    # For non-SuiteSparse, do: index = list(range(size)[index])
    # SuiteSparse indexing is inclusive for both start and stop, and unsigned.
    if step < 0:
        if start < 0:
            start = stop = 0  # Must be empty
        return AxisIndex(length, _CArray([start, stop + 1, -step]), _gxb_backwards, size)
    if stop > 0:
        stop -= 1
    elif start == 0:
        # [0:0] slice should be empty, so change to [1:0]
        start += 1
    if step == 1:
        return AxisIndex(length, _CArray([start, stop]), _gxb_range, size)
    return AxisIndex(length, _CArray([start, stop, step]), _gxb_stride, size)

from . import lib
from .dtypes import _INDEX
from .expr import _ALL_INDICES, AxisIndex
from .scalar import Scalar, _as_scalar
from .utils import _CArray

gxb_range = Scalar.from_value(lib.GxB_RANGE, dtype=_INDEX, name="GxB_RANGE", is_cscalar=True)
gxb_stride = Scalar.from_value(lib.GxB_STRIDE, dtype=_INDEX, name="GxB_STRIDE", is_cscalar=True)
gxb_backwards = Scalar.from_value(
    lib.GxB_BACKWARDS, dtype=_INDEX, name="GxB_BACKWARDS", is_cscalar=True
)


def slice_to_index(index, size):
    start, stop, step = index.indices(size)
    length = len(range(start, stop, step))
    if length == size and step == 1:
        # [:] means all indices; use special GrB_ALL indicator
        return AxisIndex(size, _ALL_INDICES, _as_scalar(size, _INDEX, is_cscalar=True), size)
    # SS, SuiteSparse-specific: slicing.
    # For non-SuiteSparse, do: index = list(range(size)[index])
    # SuiteSparse indexing is inclusive for both start and stop, and unsigned.
    if step < 0:
        if start < 0:
            start = stop = 0  # Must be empty
        return AxisIndex(length, _CArray([start, stop + 1, -step]), gxb_backwards, size)
    if stop > 0:
        stop -= 1
    elif start == 0:
        # [0:0] slice should be empty, so change to [1:0]
        start += 1
    if step == 1:
        return AxisIndex(length, _CArray([start, stop]), gxb_range, size)
    else:
        return AxisIndex(length, _CArray([start, stop, step]), gxb_stride, size)

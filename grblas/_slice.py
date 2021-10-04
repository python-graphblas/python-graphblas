from . import lib
from .dtypes import _INDEX
from .expr import _ALL_INDICES, AxisIndex
from .scalar import Scalar, _CScalar
from .utils import _CArray

gxb_range = _CScalar(Scalar.from_value(lib.GxB_RANGE, dtype=_INDEX, name="GxB_RANGE"))
gxb_stride = _CScalar(Scalar.from_value(lib.GxB_STRIDE, dtype=_INDEX, name="GxB_STRIDE"))
gxb_backwards = _CScalar(Scalar.from_value(lib.GxB_BACKWARDS, dtype=_INDEX, name="GxB_BACKWARDS"))


def slice_to_index(index, size):
    start, stop, step = index.indices(size)
    length = len(range(start, stop, step))
    if length == size and step == 1:
        # [:] means all indices; use special GrB_ALL indicator
        return AxisIndex(size, _ALL_INDICES, _CScalar(size))
    # SS, SuiteSparse-specific: slicing.
    # For non-SuiteSparse, do: index = list(range(size)[index])
    # SuiteSparse indexing is inclusive for both start and stop, and unsigned.
    if step < 0:
        return AxisIndex(length, _CArray([start, stop + 1, -step]), gxb_backwards)
    if stop > 0:
        stop -= 1
    elif start == 0:
        # [0:0] slice should be empty, so change to [1:0]
        start += 1
    if step == 1:
        return AxisIndex(length, _CArray([start, stop]), gxb_range)
    else:
        return AxisIndex(length, _CArray([start, stop, step]), gxb_stride)

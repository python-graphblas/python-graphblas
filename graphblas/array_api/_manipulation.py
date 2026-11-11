"""Manipulation functions: shape changes share the stored data (copy-on-write)."""

import math

import numpy as np

from .. import binary
from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._utils import gb_dtype

__all__ = ["expand_dims", "permute_dims", "reshape", "squeeze"]


def _check_array(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs an Array, not {type(x).__name__}")


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if not -ndim <= axis < ndim:
        raise IndexError(f"axis {axis} is out of bounds for {ndim} dimensions")
    return axis % ndim


def _new_shape(shape, size):
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    shape = tuple(int(n) for n in shape)
    unknown = [i for i, n in enumerate(shape) if n == -1]
    if len(unknown) > 1:
        raise ValueError("reshape takes at most one -1")
    if any(n < -1 for n in shape):
        raise ValueError(f"invalid shape {shape}")
    if unknown:
        known = math.prod(n for n in shape if n != -1)
        if known == 0 or size % known:
            raise ValueError(f"cannot reshape an array of size {size} into shape {shape}")
        i = unknown[0]
        shape = (*shape[:i], size // known, *shape[i + 1 :])
    if math.prod(shape) != size:
        raise ValueError(f"cannot reshape an array of size {size} into shape {shape}")
    return shape


def reshape(x, /, shape, *, copy=None):
    _check_array(x, "reshape")
    shape = _new_shape(shape, x.size)
    T = x._T()
    if x.ndim == 0 or T.nvals == 0:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    if shape == ():
        # (one position: its entry, as a 0-d Array's value)
        return Array._new(Tensor(T.dtype, ()), T.to_coo()[1][0])
    if shape == T.shape:
        return x._copy() if copy else Array._new(T.dup(), x._fill)
    return Array._new(T.reshape(shape, copy=copy), x._fill)


def permute_dims(x, /, axes):
    _check_array(x, "permute_dims")
    axes = tuple(_normalize_axis(a, x.ndim) for a in axes)
    if sorted(axes) != list(range(x.ndim)):
        raise ValueError(f"axes {axes} must be a permutation of range({x.ndim})")
    if x.ndim == 0:
        return x._copy()
    return Array._new(x._T().permute_dims(axes), x._fill)


def expand_dims(x, /, axis=0):
    _check_array(x, "expand_dims")
    axes = axis if isinstance(axis, tuple) else (axis,)
    ndim = x.ndim + len(axes)
    norm = sorted(_normalize_axis(a, ndim) for a in axes)
    if len(set(norm)) != len(norm):
        raise ValueError(f"repeated axis in {axis}")
    T = x._T()
    if x.ndim == 0:
        return Array._new(Tensor(T.dtype, (1,) * ndim), x._fill)
    return Array._new(T.expand_dims(tuple(norm)), x._fill)


def squeeze(x, /, axis):
    _check_array(x, "squeeze")
    axes = axis if isinstance(axis, tuple) else (axis,)
    norm = [_normalize_axis(a, x.ndim) for a in axes]
    if len(set(norm)) != len(norm):
        raise ValueError(f"repeated axis in {axis}")
    for a in norm:
        if x.shape[a] != 1:
            raise ValueError(f"cannot squeeze axis {a} of size {x.shape[a]}")
    if not norm:
        return x._copy()
    T = x._T()
    shape = tuple(n for i, n in enumerate(x.shape) if i not in norm)
    if not shape:
        return reshape(x, ())
    return Array._new(T.squeeze(tuple(norm)), x._fill)


def _broadcast_to(x, shape):
    """``x`` broadcast to ``shape``: its entries replicated over the broadcast axes."""
    shape = tuple(shape)
    if np.broadcast_shapes(x.shape, shape) != shape:
        raise ValueError(f"cannot broadcast shape {x.shape} to {shape}")
    if x.shape == shape:
        return x._copy()
    T = x._T()
    if x.ndim == 0 or T.nvals == 0:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    E = Tensor(T.dtype, shape)
    _utils.expansion_check(T, E, shape)
    first = binary.first[gb_dtype(x.dtype)]
    return Array._new(T.ewise_add(E, first, allow_expansion=True).new(), x._fill)


def _concat(arrays, axis):
    """The arrays joined along ``axis``. Parts whose fill differs from the first part's store
    their fill explicitly (densely, under the limit).
    """
    from ._creation import _from_numpy
    from ._dtypes import result_type
    from ._elementwise import cast

    arrays = list(arrays)
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    ndim = arrays[0].ndim
    if ndim == 0 or any(a.ndim != ndim for a in arrays):
        raise ValueError("concatenation needs arrays of one positive number of dimensions")
    axis = _normalize_axis(axis, ndim)
    shape = list(arrays[0].shape)
    for a in arrays[1:]:
        if any(a.shape[i] != n for i, n in enumerate(shape) if i != axis):
            raise ValueError(f"shapes {arrays[0].shape} and {a.shape} do not concatenate")
    shape[axis] = sum(a.shape[axis] for a in arrays)
    dtype = result_type(*arrays)
    arrays = [a if a.dtype == dtype else cast(a, dtype) for a in arrays]
    fill = arrays[0]._fill
    coords, values = [], []
    offset = 0
    for a in arrays:
        if not _utils.same_value(a._fill, fill):
            _utils.check_dense(a.size, "concatenating arrays with different fill values")
            a = _from_numpy(a.todense(), fill)
        T = a._T()
        if T.nvals:
            idx, vals = T.to_coo()
            idx = list(idx)
            idx[axis] = idx[axis] + offset
            coords.append(idx)
            values.append(vals)
        offset += a.shape[axis]
    shape = tuple(shape)
    if not values:
        return Array._new(Tensor(gb_dtype(dtype), shape), fill)
    coords = tuple(np.concatenate([c[i] for c in coords]) for i in range(ndim))
    T = Tensor.from_coo(coords, np.concatenate(values), gb_dtype(dtype), shape=shape)
    return Array._new(T, fill)

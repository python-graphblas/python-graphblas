"""Indexing: ``x[key]`` (and later ``x[key] = value``, ``take``, ``take_along_axis``).

Basic keys so far: integers, slices, ``...`` and ``None``, as numpy and the Array API index.
"""

import operator

import numpy as np

from ..core.tensor import Tensor
from ._array import Array

__all__ = []


def _basic_key(x, key):
    """``key`` as a tuple of ints and slices for every axis, and the positions of the new
    axes (None) in the result; None when the key is not basic.
    """
    if not isinstance(key, tuple):
        key = (key,)
    items = []
    for k in key:
        if k is None or k is Ellipsis or isinstance(k, slice):
            items.append(k)
        elif isinstance(k, (bool, np.bool_)):
            return None  # (a boolean index is a mask)
        elif isinstance(k, (int, np.integer)):
            items.append(int(k))
        elif isinstance(k, Array) and k.ndim == 0 and k.dtype.kind in "iu":
            items.append(operator.index(k))
        else:
            return None
    if sum(k is Ellipsis for k in items) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    nindexed = sum(k is not None and k is not Ellipsis for k in items)
    if nindexed > x.ndim:
        raise IndexError(f"too many indices: {nindexed} for an array of {x.ndim} dimensions")
    if Ellipsis in items:
        i = items.index(Ellipsis)
        items[i : i + 1] = [slice(None)] * (x.ndim - nindexed)
    else:
        items.extend([slice(None)] * (x.ndim - nindexed))
    axes_key = []
    new_axes = []
    out_axis = 0
    axis = 0
    for k in items:
        if k is None:
            new_axes.append(out_axis)
            out_axis += 1
            continue
        n = x.shape[axis]
        if isinstance(k, int):
            if not -n <= k < n:
                raise IndexError(f"index {k} is out of bounds for axis {axis} with size {n}")
            axes_key.append(k % n)
        else:
            axes_key.append(k)
            out_axis += 1
        axis += 1
    return tuple(axes_key), new_axes


def getitem(x, key):
    basic = _basic_key(x, key)
    if basic is None:
        return _advanced_getitem(x, key)
    axes_key, new_axes = basic
    T = x._T()
    if x.ndim == 0:
        out = x._copy()
    elif all(k == slice(None) for k in axes_key):
        out = Array._new(T.dup(), x._fill)
    else:
        shape = tuple(
            len(range(*k.indices(n)))
            for k, n in zip(axes_key, T.shape, strict=True)
            if isinstance(k, slice)
        )
        if T.nvals == 0 or 0 in shape:
            out = Array._new(Tensor(T.dtype, shape), x._fill)
        else:
            out = Array._new(T[axes_key].new(), x._fill)
    if new_axes:
        from ._manipulation import expand_dims

        out = expand_dims(out, tuple(new_axes))
    return out


def _advanced_getitem(x, key):
    raise NotImplementedError(
        f"indexing with {key!r}: only integers, slices, ... and None are supported so far"
    )


def setitem(x, key, value):
    raise NotImplementedError("assignment by index is not supported yet")

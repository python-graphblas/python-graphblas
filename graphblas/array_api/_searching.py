"""Searching functions: argmax, argmin, count_nonzero, nonzero, searchsorted, where."""

import builtins

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._dtypes import check_supported
from ._utils import gb_dtype

__all__ = ["where"]


def _operand(x, name):
    if isinstance(x, Array):
        return x
    if isinstance(x, (np.generic, np.ndarray)):
        from ._creation import asarray

        return asarray(x)
    if isinstance(x, (builtins.bool, int, float, complex)):
        return x
    raise TypeError(f"{name} needs arrays or Python scalars, not {type(x).__name__}")


def _probe(x):
    return np.empty(0, x.dtype) if isinstance(x, Array) else x


def _value(x):
    return np.asarray(x._fill) if isinstance(x, Array) else x


def where(condition, x1, x2, /):
    """``x1`` where ``condition`` is true, else ``x2`` (broadcast); numpy's ``where``.

    The entries are the union of the three arrays' entries, O(entries) (with the broadcast
    operands' entries replicated, as for any union).
    """
    cond = _operand(condition, "where")
    if not isinstance(cond, Array):
        from ._creation import asarray

        cond = asarray(cond)
    a = _operand(x1, "where")
    b = _operand(x2, "where")
    rdt = check_supported(np.result_type(_probe(a), _probe(b)), "where")
    fill = np.where(np.asarray(cond._fill, dtype=bool), _value(a), _value(b)).astype(rdt)[()]
    shapes = [x.shape for x in (cond, a, b) if isinstance(x, Array)]
    shape = np.broadcast_shapes(*shapes)
    arrays = [x for x in (cond, a, b) if isinstance(x, Array) and x.ndim]
    if not shape or all(x._T().nvals == 0 for x in arrays):
        return Array._new(Tensor(gb_dtype(rdt), shape), fill)
    tensors = iter(_utils.aligned(arrays, shape))
    slots = [next(tensors) if isinstance(x, Array) and x.ndim else None for x in (cond, a, b)]
    constants = [_value(x) for x in (cond, a, b)]

    def func(*values):
        it = iter(values)
        c, v1, v2 = (
            next(it) if t is not None else k for t, k in zip(slots, constants, strict=True)
        )
        return np.where(np.asarray(c, dtype=bool), v1, v2).astype(rdt, copy=False)

    R = _utils.map_values(func, [t for t in slots if t is not None], rdt)
    return Array._new(R, fill)

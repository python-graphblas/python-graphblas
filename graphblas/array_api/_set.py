"""Set functions: isin, unique_all, unique_counts, unique_inverse, unique_values.

The unique values of an array are those of its stored entries plus the fill, counted once per
unstored position: O(entries log entries). NaNs are distinct (each is its own value), so a
fill of NaN with unstored positions makes as many distinct values: that case goes through
numpy densely, under the densify limit. Values come sorted, as numpy's.
"""

from typing import NamedTuple

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._utils import gb_dtype

__all__ = ["isin", "unique_all", "unique_counts", "unique_inverse", "unique_values"]


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def _vector(a):
    from ._creation import _from_numpy

    return _from_numpy(np.asarray(a))


def _unique(x, *, index=False, inverse=False, counts=False):
    """The sorted unique values of ``x`` (as numpy's unique with equal_nan=False), and the
    first flat index, the inverse (an Array of x's shape) and the counts when asked.
    """
    if not isinstance(x, Array):
        raise TypeError(f"needs an Array, not {type(x).__name__}")
    T = x._T()
    fill = x._fill
    k = x.size - T.nvals
    if k and x.dtype.kind in "fc" and np.isnan(fill):
        _utils.check_dense(x.size, "unique of an array whose fill is NaN (each NaN is distinct)")
        return _unique_dense(x, index, inverse, counts)
    if T.nvals:
        coords, values = T.to_coo()
        flat = np.ravel_multi_index(coords, x.shape) if x.ndim else np.zeros(1, np.int64)
    else:
        coords = tuple(np.empty(0, np.int64) for _ in range(x.ndim))
        values, flat = np.empty(0, x.dtype), np.empty(0, np.int64)
    if k:
        # The fill once, at its first unstored position, with the other k - 1 added later
        first = _first_unstored(flat, x.size)
        at = np.searchsorted(flat, first)
        values = np.insert(values, at, fill)
        flat = np.insert(flat, at, first)
    out = np.unique(
        values, return_index=True, return_inverse=True, return_counts=True, equal_nan=False
    )
    uvalues, uindex, uinverse, ucounts = out
    uinverse = uinverse.reshape(-1)
    result = {"values": _vector(uvalues)}
    if k:
        fill_slot = uinverse[at]
        ucounts = ucounts.copy()
        ucounts[fill_slot] += k - 1
    if index:
        result["indices"] = _vector(flat[uindex].astype(np.int64))
    if counts:
        result["counts"] = _vector(ucounts.astype(np.int64))
    if inverse:
        inv_fill = int(uinverse[at]) if k else 0
        if k:
            uinverse = np.delete(uinverse, at)
        if x.ndim == 0:
            inv = Array._new(
                Tensor(gb_dtype(np.int64), ()), uinverse[0] if len(uinverse) else inv_fill
            )
        elif T.nvals:
            R = Tensor.from_coo(
                coords, uinverse.astype(np.int64), gb_dtype(np.int64), shape=x.shape
            )
            inv = Array._new(R, inv_fill)
        else:
            inv = Array._new(Tensor(gb_dtype(np.int64), x.shape), inv_fill)
        result["inverse_indices"] = inv
    return result


def _first_unstored(flat, size):
    """The smallest flat position in ``range(size)`` not in the sorted ``flat``."""
    gaps = np.flatnonzero(flat != np.arange(len(flat)))
    return int(gaps[0]) if len(gaps) else len(flat)


def _unique_dense(x, index, inverse, counts):
    from ._creation import _from_numpy

    dense = x.todense()
    uvalues, uindex, uinverse, ucounts = np.unique(
        dense.reshape(-1), return_index=True, return_inverse=True, return_counts=True,
        equal_nan=False,
    )  # fmt: skip
    result = {"values": _vector(uvalues)}
    if index:
        result["indices"] = _vector(uindex.astype(np.int64))
    if counts:
        result["counts"] = _vector(ucounts.astype(np.int64))
    if inverse:
        result["inverse_indices"] = _from_numpy(uinverse.reshape(x.shape).astype(np.int64))
    return result


def unique_values(x, /):
    return _unique(x)["values"]


def unique_counts(x, /):
    r = _unique(x, counts=True)
    return UniqueCountsResult(r["values"], r["counts"])


def unique_inverse(x, /):
    r = _unique(x, inverse=True)
    return UniqueInverseResult(r["values"], r["inverse_indices"])


def unique_all(x, /):
    r = _unique(x, index=True, inverse=True, counts=True)
    return UniqueAllResult(r["values"], r["indices"], r["inverse_indices"], r["counts"])


def isin(x1, x2, /, *, invert=False):
    """Whether each element of ``x1`` is among ``x2``'s values (its stored values and its
    fill): O(entries of both).
    """
    from ._creation import asarray

    x1 = x1 if isinstance(x1, Array) else asarray(x1)
    x2 = x2 if isinstance(x2, Array) else asarray(x2)
    T2 = x2._T()
    values = T2.to_coo(indices=False)[1] if T2.nvals else np.empty(0, x2.dtype)
    if x2.size > T2.nvals:
        values = np.append(values, x2._fill)

    def member(v):
        return np.isin(v, values, invert=invert)

    fill = member(np.asarray(x1._fill))
    T1 = x1._T()
    if x1.ndim == 0 or T1.nvals == 0:
        return Array._new(Tensor(gb_dtype(bool), x1.shape), fill)
    return Array._new(_utils.map_values(member, [T1], np.dtype(bool)), fill)

"""Sorting functions: sort and argsort.

``sort`` keeps the result as sparse as its input: along each fiber, the values below the fill
come first, then the run of values equal to the fill (the unstored positions among them),
then the values above it, NaN last; equal values keep their order (stable). Descending order
reverses the values but keeps equal values in their order, NaN first (as
``flip(sort(flip(x)))``). O(entries log entries).

``argsort``'s result is a permutation of every position, dense: it goes through numpy, under
the densify limit.
"""

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._array import Array

__all__ = ["argsort", "sort"]


def _check(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs an Array, not {type(x).__name__}")
    if x.ndim == 0:
        raise ValueError(f"{name} needs an array of at least one dimension")
    if x.dtype.kind == "c":
        raise TypeError(f"{name} needs a real-valued array, not {x.dtype}")


def sort(x, /, *, axis=-1, descending=False, stable=True):
    from ._manipulation import _normalize_axis, permute_dims

    _check(x, "sort")
    axis = _normalize_axis(axis, x.ndim)
    L = x.shape[axis]
    T = x._T()
    if T.nvals == 0 or L <= 1:
        return x._copy()
    others = tuple(a for a in range(x.ndim) if a != axis)
    perm = (*others, axis)
    coords, values = permute_dims(x, perm)._T().to_coo()  # (by fiber, then position)
    fshape = tuple(x.shape[a] for a in others)
    fiber = np.ravel_multi_index(coords[:-1], fshape) if fshape else np.zeros(len(values), np.int64)
    pos = coords[-1]
    fill = x._fill
    with np.errstate(invalid="ignore"):
        if values.dtype.kind == "f":
            nan_v, nan_f = np.isnan(values), bool(np.isnan(fill))
            equal = (values == fill) | (nan_v & nan_f)
            below = (values < fill) | (~nan_v & nan_f)
        else:
            equal = values == fill
            below = values < fill
    above = ~(equal | below)
    if descending:
        below, above = above, below
    # Rank the values (np.unique sorts NaN last and makes them one value)
    _, rank = np.unique(values, return_inverse=True)
    rank = rank.reshape(-1)
    if descending:
        rank = rank.max(initial=0) - rank
    # Groups per fiber: 0 below the fill, 1 equal to it, 2 above it
    group = np.where(below, 0, np.where(equal, 1, 2))
    order = np.lexsort((pos, rank, group, fiber))
    f_o, g_o, p_o = fiber[order], group[order], pos[order]
    # Rank of each entry within its (fiber, group), and the sizes of the groups per fiber
    key = f_o * 3 + g_o
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, len(key)])
    within = np.arange(len(key)) - np.repeat(starts, sizes)
    fibers, fiber_index = np.unique(f_o, return_inverse=True)
    counts = np.zeros((len(fibers), 3), np.int64)
    np.add.at(counts, (fiber_index, g_o), 1)
    n_below = counts[fiber_index, 0]
    n_equal = counts[fiber_index, 1]
    unstored = L - counts.sum(axis=1)[fiber_index]
    # An entry equal to the fill keeps its place among the run of the fill (stable): the
    # positions before it in the run are every position before it but the other groups'
    others_before = _count_before(f_o, p_o, g_o != 1)
    new_pos = np.where(
        g_o == 0,
        within,
        np.where(g_o == 1, n_below + p_o - others_before, n_below + n_equal + unstored + within),
    )
    vals = values[order]
    keep = (g_o != 1) | _utils.differs(vals, fill)
    if not keep.any():
        return Array._new(Tensor(T.dtype, x.shape), fill)
    out_coords = list(np.unravel_index(f_o[keep], fshape)) if fshape else []
    out_coords.append(new_pos[keep])
    R = Tensor.from_coo(out_coords, vals[keep], T.dtype, shape=tuple(x.shape[a] for a in perm))
    inverse = tuple(int(i) for i in np.argsort(perm))
    return permute_dims(Array._new(R, fill), inverse)


def _count_before(fiber, pos, flags):
    """For each entry: how many flagged entries of its fiber have a smaller position."""
    order = np.lexsort((pos, fiber))
    f, fl = fiber[order], flags[order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    csum = np.cumsum(fl) - fl  # (flagged before, over all fibers)
    base = np.repeat(csum[starts], np.diff(np.r_[starts, len(f)]))
    out = np.empty(len(fiber), np.int64)
    out[order] = csum - base
    return out


def argsort(x, /, *, axis=-1, descending=False, stable=True):
    from ._creation import _from_numpy
    from ._manipulation import _normalize_axis

    _check(x, "argsort")
    axis = _normalize_axis(axis, x.ndim)
    _utils.check_dense(x.size, "argsort (every position's index)")
    dense = x.todense()
    if descending:
        # Descending and stable: equal values keep their order, as flip(sort(flip(x)))
        flipped = np.flip(dense, axis=axis)
        out = x.shape[axis] - 1 - np.flip(np.argsort(flipped, axis=axis, kind="stable"), axis=axis)
    else:
        out = np.argsort(dense, axis=axis, kind="stable")
    return _from_numpy(out.astype(np.int64, copy=False))

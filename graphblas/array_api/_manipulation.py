"""Manipulation functions: shape changes share the stored data (copy-on-write)."""

import math

import numpy as np

from .. import binary, select
from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._utils import gb_dtype

__all__ = [
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "prune",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]


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


def broadcast_to(x, /, shape):
    """``x`` broadcast to ``shape``: its entries replicated over the broadcast axes (the
    result's entries, under the densify limit for the replication).
    """
    _check_array(x, "broadcast_to")
    return _broadcast_to(x, shape)


def broadcast_shapes(*shapes):
    return tuple(np.broadcast_shapes(*shapes))


def broadcast_arrays(*arrays):
    for x in arrays:
        _check_array(x, "broadcast_arrays")
    shape = np.broadcast_shapes(*[x.shape for x in arrays])
    return tuple(_broadcast_to(x, shape) for x in arrays)


def _broadcast_to(x, shape):
    shape = tuple(int(n) for n in shape)
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


def concat(arrays, /, *, axis=0):
    """Join arrays along ``axis`` (None: flattened first), O(entries).

    The result's fill is the one held by the most positions; a part with another fill stores
    its fill explicitly (O(that part's size), under the densify limit).
    """
    from ._dtypes import result_type
    from ._elementwise import cast

    arrays = list(arrays)
    if not arrays:
        raise ValueError("concat needs at least one array")
    for a in arrays:
        _check_array(a, "concat")
    if axis is None:
        arrays = [reshape(a, (-1,)) for a in arrays]
        axis = 0
    ndim = arrays[0].ndim
    if ndim == 0 or any(a.ndim != ndim for a in arrays):
        raise ValueError("concat needs arrays of one number of dimensions, at least 1")
    axis = _normalize_axis(axis, ndim)
    shape = list(arrays[0].shape)
    for a in arrays[1:]:
        if any(a.shape[i] != n for i, n in enumerate(shape) if i != axis):
            raise ValueError(f"shapes {arrays[0].shape} and {a.shape} do not concatenate")
    shape[axis] = sum(a.shape[axis] for a in arrays)
    shape = tuple(shape)
    dtype = result_type(*arrays)
    arrays = [a if a.dtype == dtype else cast(a, dtype) for a in arrays]
    fill = _common_fill(arrays)
    coords, values = [], []
    offset = 0
    for a in arrays:
        T = _utils.with_fill(a, fill)._T()
        if T.nvals:
            # (stored order: no relayout to C order; the build sorts)
            idx, vals = T.to_coo(sort=False)
            idx = list(idx)
            idx[axis] = idx[axis] + offset
            coords.append(idx)
            values.append(vals)
        offset += a.shape[axis]
    if not values:
        return Array._new(Tensor(gb_dtype(dtype), shape), fill)
    coords = tuple(np.concatenate([c[i] for c in coords]) for i in range(ndim))
    T = Tensor.from_coo(coords, np.concatenate(values), gb_dtype(dtype), shape=shape)
    return Array._new(T, fill)


def _common_fill(arrays):
    """The fill held by the most positions of ``arrays`` (the first such, on ties)."""
    totals = []
    for a in arrays:
        for i, (fill, _) in enumerate(totals):
            if _utils.same_value(fill, a._fill):
                totals[i] = (fill, totals[i][1] + a.size)
                break
        else:
            totals.append((a._fill, a.size))
    best = max(n for _, n in totals)
    return next(fill for fill, n in totals if n == best)


def stack(arrays, /, *, axis=0):
    arrays = list(arrays)
    if not arrays:
        raise ValueError("stack needs at least one array")
    for a in arrays:
        _check_array(a, "stack")
    if len({a.shape for a in arrays}) > 1:
        raise ValueError(f"stack needs arrays of one shape, not {[a.shape for a in arrays]}")
    axis = _normalize_axis(axis, arrays[0].ndim + 1)
    return concat([expand_dims(a, axis=axis) for a in arrays], axis=axis)


def unstack(x, /, *, axis=0):
    _check_array(x, "unstack")
    if x.ndim == 0:
        raise ValueError("unstack needs an array of at least one dimension")
    axis = _normalize_axis(axis, x.ndim)
    before = (slice(None),) * axis
    return tuple(x[(*before, i)] for i in range(x.shape[axis]))


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    norm = tuple(_normalize_axis(a, ndim) for a in axes)
    if len(set(norm)) != len(norm):
        raise ValueError(f"repeated axis in {axis}")
    return norm


def flip(x, /, *, axis=None):
    _check_array(x, "flip")
    axes = _axes(axis, x.ndim)
    if x.ndim == 0:
        return x._copy()
    return x[tuple(slice(None, None, -1) if a in axes else slice(None) for a in range(x.ndim))]


def moveaxis(x, source, destination, /):
    _check_array(x, "moveaxis")
    src = source if isinstance(source, tuple) else (source,)
    dst = destination if isinstance(destination, tuple) else (destination,)
    if len(src) != len(dst):
        raise ValueError("moveaxis needs as many destinations as sources")
    src = _axes(src, x.ndim)
    dst = _axes(dst, x.ndim)
    order = [a for a in range(x.ndim) if a not in src]
    for d, s in sorted(zip(dst, src, strict=True)):
        order.insert(d, s)
    return permute_dims(x, tuple(order))


def _entries(x):
    """The stored entries of ``x`` as (coordinates, values), in stored order."""
    T = x._T()
    if T.nvals == 0:
        return [np.empty(0, np.int64) for _ in range(x.ndim)], np.empty(0, x.dtype)
    idx, values = T.to_coo(sort=False)
    return list(idx), values


def _from_entries(coords, values, shape, x):
    """An Array of ``x``'s dtype and fill storing ``values`` at ``coords`` in ``shape``."""
    shape = tuple(shape)
    if len(values) == 0:
        return Array._new(Tensor(x._T().dtype, shape), x._fill)
    T = Tensor.from_coo(tuple(coords), values, x._T().dtype, shape=shape)
    return Array._new(T, x._fill)


def repeat(x, repeats, /, *, axis=None):
    """Each element repeated along ``axis``, O(entries of the result)."""
    _check_array(x, "repeat")
    if axis is None:
        x = reshape(x, (-1,))
        axis = 0
    axis = _normalize_axis(axis, x.ndim)
    n = x.shape[axis]
    if isinstance(repeats, Array):
        if repeats.dtype.kind not in "iu" or repeats.ndim != 1:
            raise TypeError("repeats must be an int or a 1-d integer array")
        counts = repeats.todense().astype(np.int64)
    else:
        counts = np.asarray(int(repeats), dtype=np.int64)
    counts = np.broadcast_to(counts, (n,))
    if (counts < 0).any():
        raise ValueError("repeats must be nonnegative")
    starts = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=starts[1:])
    shape = list(x.shape)
    shape[axis] = int(starts[-1])
    coords, values = _entries(x)
    k = counts[coords[axis]]
    which = np.repeat(np.arange(len(values)), k)
    within = np.arange(which.size) - np.repeat(np.cumsum(k) - k, k)
    coords = [c[which] for c in coords]
    coords[axis] = starts[:-1][coords[axis]] + within
    return _from_entries(coords, values[which], shape, x)


def roll(x, /, shift, *, axis=None):
    _check_array(x, "roll")
    if axis is None:
        return reshape(roll(reshape(x, (-1,)), shift, axis=0), x.shape)
    shifts = shift if isinstance(shift, tuple) else (shift,)
    axes = axis if isinstance(axis, tuple) else (axis,)
    if len(shifts) == 1 and len(axes) > 1:
        shifts = shifts * len(axes)
    if len(shifts) != len(axes):
        raise ValueError("roll needs one shift per axis, or one shift")
    total = [0] * x.ndim
    for s, a in zip(shifts, axes, strict=True):
        total[_normalize_axis(a, x.ndim)] += int(s)
    coords, values = _entries(x)
    for a, s in enumerate(total):
        if s and x.shape[a]:
            coords[a] = (coords[a] + s) % x.shape[a]
    return _from_entries(coords, values, x.shape, x)


def tile(x, repetitions, /):
    """``x`` repeated ``repetitions`` times along each axis, O(entries of the result)."""
    _check_array(x, "tile")
    reps = tuple(int(r) for r in repetitions)
    if any(r < 0 for r in reps):
        raise ValueError("repetitions must be nonnegative")
    ndim = max(len(reps), x.ndim)
    reps = (1,) * (ndim - len(reps)) + reps
    if x.ndim < ndim:
        x = reshape(x, (1,) * (ndim - x.ndim) + x.shape)
    shape = tuple(n * r for n, r in zip(x.shape, reps, strict=True))
    coords, values = _entries(x)
    ntiles = math.prod(reps)
    if ntiles == 0 or len(values) == 0:
        return Array._new(Tensor(x._T().dtype, shape), x._fill)
    tiles = np.unravel_index(np.arange(ntiles), reps)
    out = []
    for c, t, n in zip(coords, tiles, x.shape, strict=True):
        out.append((c[None, :] + (t * n)[:, None]).reshape(-1))
    return _from_entries(out, np.tile(values, ntiles), shape, x)


def prune(x, /):
    """``x`` without the stored entries that equal its fill value (an extension): the same
    values, fewer stored entries. "Equal" is "identical": with fill 0.0 a stored -0.0 stays,
    and with fill NaN the stored NaNs go. ``nnz`` counts stored entries, these included,
    until they are pruned.
    """
    if not isinstance(x, Array):
        raise TypeError(f"prune needs an Array, not {type(x).__name__}")
    T = x._T()
    fill = x._fill
    if T.nvals == 0:
        return x._copy()
    kind = x.dtype.kind
    if kind in "biu" or (kind == "f" and fill != 0 and not np.isnan(fill)):
        # (a value equal to a nonzero number is that number)
        return Array._new(T.select(select.valuene, fill).new(), fill)
    if fill == 0 and not _utils.fact(x, "zero"):
        return x._copy()  # (no stored zero of either sign)
    keep = _utils.map_values(lambda v: _utils.differs(v, fill), [T], np.dtype(bool))
    R = Tensor(T.dtype, T.shape)
    R(keep.V) << T
    return Array._new(R, fill)

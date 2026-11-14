"""Indexing: ``x[key]``, ``x[key] = value``, ``take`` and ``take_along_axis``.

Keys follow numpy: integers, slices, ``...`` and ``None`` (basic indexing, extracted in
GraphBLAS), and integer and boolean arrays (advanced indexing, gathered from the stored
entries, so a position the array does not store is the fill in the result too). The Array API
standard specifies an index of integers and integer arrays only (vectorized indexing), or one
boolean array alone; other combinations follow numpy.

Costs: basic indexing is O(entries in the region). An integer index array is read densely
(its size is the result's); a boolean index array costs O(its entries) when its fill is
False, O(its size) when its fill is True. Assigning a scalar other than the fill, or an array
whose fill is not ``x``'s, to a region stores every position of the region; assigning by
several index arrays writes every selected position (O(selection)).
"""

import builtins
import math
import operator

import numpy as np

from ..core.tensor import Tensor
from . import _lazy, _utils
from ._array import Array
from ._utils import gb_dtype

__all__ = ["take", "take_along_axis"]


# Keys
class _Adv:
    """An integer index array (advanced indexing) for one axis, nonnegative."""

    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


class _NewAxis:
    """``None``, or a 0-d boolean index: a new axis of length 1 (0 for False)."""

    __slots__ = ("length",)

    def __init__(self, length=1):
        self.length = length


class _Gap:
    """A ``...`` that covers no axis. It takes no axis, but separates the advanced indices on
    either side of it, as in numpy: ``x[:, 1, ..., [0, 2]]`` of a 3-d ``x`` puts the
    broadcast advanced axis first.
    """

    __slots__ = ()


_GAP = _Gap()


def _is_int(k):
    return isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))


def _index_array(k):
    """An index of array type: a boolean Array as it is, else a numpy array; None when ``k``
    is not an array.
    """
    if isinstance(k, Array):
        if k.dtype.kind == "b":
            return k
        if k.dtype.kind not in "iu":
            raise IndexError(
                f"arrays used as indices must be of integer or boolean type: {k.dtype}"
            )
        return k.todense()
    if isinstance(k, (list, np.ndarray)):
        arr = np.asarray(k)
        if arr.dtype.kind not in "biu":
            if arr.size == 0:
                return arr.astype(np.int64)
            raise IndexError(
                f"arrays used as indices must be of integer or boolean type: {arr.dtype}"
            )
        return arr
    return None


def _mask_coords(mask, shape):
    """The coordinates, in C order, where the boolean ``mask`` (an Array or ndarray) is True;
    ``shape`` is that of the axes it covers.
    """
    if mask.shape != tuple(shape):
        # (the standard lets a dimension of the mask be 0 instead of the array's)
        if any(m not in (n, 0) for m, n in zip(mask.shape, shape, strict=True)):
            raise IndexError(f"boolean index of shape {mask.shape} does not match {shape}")
        return tuple(np.empty(0, np.int64) for _ in mask.shape)
    if isinstance(mask, np.ndarray):
        return np.nonzero(mask)
    if not mask._fill:
        T = mask._T()
        if T.nvals == 0:
            return tuple(np.empty(0, np.int64) for _ in mask.shape)
        # (stored entries may be False, equal to the fill)
        return T.select("==", True).new().to_coo(values=False)[0]
    _utils.check_dense(mask.size, "a boolean index whose fill is True")
    return np.nonzero(mask.todense())


def _normalize(x, key):
    """``key`` as a list of items: an int, slice or _Adv for each axis of ``x``, and _NewAxis
    items for new axes. A boolean array of M dimensions becomes M _Adv items.
    """
    if not isinstance(key, tuple):
        key = (key,)
    items = []
    for k in key:
        if k is None:
            items.append(_NewAxis())
        elif k is Ellipsis or isinstance(k, slice):
            items.append(k)
        elif isinstance(k, (bool, np.bool_)):
            items.append(_NewAxis(1 if k else 0))
        elif _is_int(k):
            items.append(int(k))
        elif isinstance(k, Array) and k.ndim == 0:
            if k.dtype.kind == "b":
                items.append(_NewAxis(1 if k._fill else 0))
            elif k.dtype.kind in "iu":
                items.append(operator.index(k))
            else:
                raise IndexError(f"a 0-d index must be an integer or boolean, not {k.dtype}")
        else:
            arr = _index_array(k)
            if arr is None:
                raise IndexError(f"invalid index {k!r}")
            items.append(arr)
    if sum(k is Ellipsis for k in items if not isinstance(k, (np.ndarray, Array))) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    def width(k):
        if isinstance(k, (np.ndarray, Array)) and k.dtype.kind == "b":
            return k.ndim
        return 0 if k is Ellipsis or isinstance(k, _NewAxis) else 1

    consumed = sum(width(k) for k in items)
    if consumed > x.ndim:
        raise IndexError(f"too many indices: {consumed} for an array of {x.ndim} dimensions")
    i = next((i for i, k in enumerate(items) if k is Ellipsis), None)
    if i is None:
        items.extend([slice(None)] * (x.ndim - consumed))
    else:
        items[i : i + 1] = [slice(None)] * (x.ndim - consumed) or [_GAP]
    out = []
    axis = 0
    for k in items:
        if isinstance(k, (_NewAxis, _Gap)):
            out.append(k)
            continue
        if isinstance(k, (np.ndarray, Array)) and k.dtype.kind == "b":
            if k.ndim == 0:  # (a 0-d boolean numpy array)
                out.append(_NewAxis(1 if k[()] else 0))
                continue
            coords = _mask_coords(k, x.shape[axis : axis + k.ndim])
            out.extend(_Adv(np.asarray(c, dtype=np.int64)) for c in coords)
            axis += k.ndim
            continue
        n = x.shape[axis]
        if isinstance(k, np.ndarray):
            idx = k.astype(np.int64, copy=False)
            if idx.size and (idx.min() < -n or idx.max() >= n):
                raise IndexError(f"index out of bounds for axis {axis} with size {n}")
            out.append(_Adv(np.where(idx < 0, idx + n, idx)))
        elif isinstance(k, int):
            if not -n <= k < n:
                raise IndexError(f"index {k} is out of bounds for axis {axis} with size {n}")
            out.append(k % n)
        else:
            out.append(k)
        axis += 1
    return out


def _has_adv(items):
    return any(isinstance(k, _Adv) for k in items)


def _adjacent(items):
    """Whether the advanced items (index arrays, with the ints among them) are next to each
    other; numpy then keeps their axes in place, else puts them first.
    """
    positions = [i for i, k in enumerate(items) if isinstance(k, (_Adv, int))]
    return positions == list(range(positions[0], positions[-1] + 1))


def _finish(out, items, nadv):
    """``out`` holds the advanced axes (``nadv`` of them) first, then the sliced axes in
    order: move the advanced ones where numpy puts them and insert the new axes.
    """
    from ._manipulation import expand_dims, permute_dims

    comps = []
    placed = True
    if nadv or _has_adv(items):
        placed = not _adjacent(items)
        if placed:
            comps.append("adv")
    for k in items:
        if isinstance(k, (_NewAxis, slice)):
            comps.append(k)
        elif isinstance(k, _Gap):
            continue
        elif not placed:
            comps.append("adv")
            placed = True
    order = []
    new = []
    rest = nadv
    for c in comps:
        if isinstance(c, str):
            order.extend(range(nadv))
        elif isinstance(c, slice):
            order.append(rest)
            rest += 1
        else:
            new.append((len(order) + len(new), c.length))
    if order != list(range(len(order))):
        out = permute_dims(out, tuple(order))
    if new:
        out = expand_dims(out, tuple(p for p, _ in new))
        if any(length == 0 for _, length in new):
            empty = {p for p, length in new if length == 0}
            out = out[tuple(slice(0, 0) if a in empty else slice(None) for a in range(out.ndim))]
    return out


def _basic_shape(shape, axes_key):
    return tuple(
        len(range(*k.indices(n)))
        for k, n in zip(axes_key, shape, strict=True)
        if isinstance(k, slice)
    )


# getitem
def getitem(x, key):
    items = _normalize(x, key)
    axes_key = [k for k in items if not isinstance(k, (_NewAxis, _Gap))]
    if _has_adv(items):
        out, nadv = _advanced_extract(x, axes_key)
    else:
        # (a pending product slices its operands instead)
        if (out := _lazy.sliced(x, axes_key)) is None:
            out = _basic_extract(x, tuple(axes_key))
        nadv = 0
    return _finish(out, items, nadv)


def _basic_extract(x, axes_key):
    T = x._T()
    if x.ndim == 0 or all(k == slice(None) for k in axes_key):
        return x._copy()
    shape = _basic_shape(T.shape, axes_key)
    if T.nvals == 0 or 0 in shape:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    return Array._new(T[axes_key].new(), x._fill)


def _advanced_extract(x, axes_key):
    """``x`` indexed by ``axes_key`` (one item per axis) with the advanced axes first, then the
    sliced ones in order, and the number of advanced axes.
    """
    sliced = tuple(k if isinstance(k, slice) else slice(None) for k in axes_key)
    y = _basic_extract(x, sliced) if any(k != slice(None) for k in sliced) else x
    adv_axes = [a for a, k in enumerate(axes_key) if not isinstance(k, slice)]
    rest = [a for a, k in enumerate(axes_key) if isinstance(k, slice)]
    indices = np.broadcast_arrays(
        *[
            k.index if isinstance(k, _Adv) else np.asarray(k, np.int64)
            for k in axes_key
            if not isinstance(k, slice)
        ]
    )
    S2 = indices[0].shape
    adv_shape = tuple(y.shape[a] for a in adv_axes)
    rest_shape = tuple(y.shape[a] for a in rest)
    T = y._T()
    if T.nvals == 0 or math.prod(S2) == 0 or math.prod(rest_shape) == 0:
        return Array._new(Tensor(T.dtype, S2 + rest_shape), x._fill), len(S2)
    if not rest:
        # Vectorized (pointwise) indexing of every axis
        return Array._new(T.vindex[tuple(indices)], x._fill), len(S2)
    Tp = T.permute_dims(tuple(adv_axes) + tuple(rest))
    if len(adv_axes) > 1:
        flat = np.ravel_multi_index(tuple(indices), adv_shape)
        Tp = Tp.reshape((math.prod(adv_shape), *rest_shape))
    else:
        flat = indices[0]
    return Array._new(Tp.take(flat.astype(np.int64, copy=False), axis=0), x._fill), len(S2)


# setitem
def setitem(x, key, value):
    if not isinstance(x, Array):  # pragma: no cover (safety)
        raise TypeError(type(x))
    value = _setitem_value(x, value)
    items = _normalize(x, key)
    axes_key = [k for k in items if not isinstance(k, (_NewAxis, _Gap))]
    if any(isinstance(k, _NewAxis) and k.length == 0 for k in items):
        return  # (a False index selects nothing)
    if isinstance(value, Array) and not _has_adv(items):
        value = _without_new_axes(value, items)
        if value.ndim == 0:
            value = _scalar_value(value._fill, x.dtype)
    if x.ndim == 0:
        if isinstance(value, Array):
            raise ValueError(f"cannot assign an array of shape {value.shape} to a 0-d array")
        x._fill = value
    else:
        adv = [k for k in axes_key if isinstance(k, _Adv)]
        if (
            len(adv) == 1
            and adv[0].index.ndim == 1
            and not any(isinstance(k, int) for k in axes_key)
        ):
            # One 1-d index array among slices: outer (GraphBLAS) and vectorized indexing agree
            _region_setitem(
                x, tuple(k.index if isinstance(k, _Adv) else k for k in axes_key), value
            )
        elif adv:
            _scatter_setitem(x, [k for k in items if not isinstance(k, _NewAxis)], value)
        else:
            _region_setitem(x, tuple(axes_key), value)
    x._facts = None


def _without_new_axes(value, items):
    """``value`` for a basic key with new axes (``None``): the selection numpy broadcasts it
    against has those axes, of length 1, which the region assigned to does not; drop the
    value's axes that line up with them.
    """
    kinds = [k for k in items if isinstance(k, (_NewAxis, slice))]
    offset = len(kinds) - value.ndim
    drop = [p - offset for p, k in enumerate(kinds) if isinstance(k, _NewAxis) and p - offset >= 0]
    if not drop:
        return value
    if any(value.shape[q] != 1 for q in drop):
        raise ValueError(
            f"cannot assign an array of shape {value.shape} to a selection of "
            f"{len(kinds)} dimensions whose new axes have length 1"
        )
    from ._manipulation import squeeze

    return squeeze(value, axis=tuple(drop))


def _setitem_value(x, value):
    """The value cast to ``x``'s dtype: a numpy scalar, or an Array of at least 1 dimension."""
    from ._creation import asarray
    from ._elementwise import cast

    dt = x.dtype
    if isinstance(value, (builtins.bool, int, float, complex, np.generic)):
        return _scalar_value(value, dt)
    if not isinstance(value, Array):
        value = asarray(value)
    if value.ndim == 0:
        return _scalar_value(value._fill, dt)
    return value if value.dtype == dt else cast(value, dt)


def _scalar_value(value, dt):
    if isinstance(value, (complex, np.complexfloating)) and dt.kind != "c":
        raise TypeError(f"cannot assign a complex value to an array of {dt}")
    with np.errstate(all="ignore"):
        return np.array(value, dtype=dt)[()]


def _region_setitem(x, key, value):
    """Assign to the region an outer index selects (GraphBLAS assign)."""
    T = x._T()
    region = tuple(
        len(range(*k.indices(n))) if isinstance(k, slice) else len(k)
        for k, n in zip(key, x.shape, strict=True)
        if not isinstance(k, int)
    )
    if not isinstance(value, Array):
        if math.prod(region) == 0:
            return
        if _utils.same_value(value, x._fill):
            if T.nvals:
                del T[key]
        else:
            T[key] = _utils.gb_scalar(value, x.dtype)
        return
    try:
        ok = np.broadcast_shapes(value.shape, region) == region
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(
            f"cannot assign an array of shape {value.shape} to a region of shape {region}"
        )
    if math.prod(region) == 0:
        return
    # The region takes the value's structure: an unstored position of the value deletes
    T[key] = _utils.with_fill(value, x._fill)._T()


def _scatter_setitem(x, axes_key, value):
    """Assign by several index arrays: write every selected position."""
    # numpy lays out the selection; index broadcast views of the coordinates with the key (a
    # ``...`` that covers no axis stays: it decides where the advanced axes go)
    key = tuple(
        k.index if isinstance(k, _Adv) else Ellipsis if isinstance(k, _Gap) else k for k in axes_key
    )
    coords = []
    for a, n in enumerate(x.shape):
        grid = np.arange(n, dtype=np.int64).reshape([n if b == a else 1 for b in range(x.ndim)])
        coords.append(np.asarray(np.broadcast_to(grid, x.shape)[key]))
    shape = coords[0].shape
    size = math.prod(shape)
    if size == 0:
        return
    _utils.check_dense(size, "assigning to the selected positions")
    if isinstance(value, Array):
        try:
            values = np.broadcast_to(value.todense(), shape)
        except ValueError:
            raise ValueError(
                f"cannot assign an array of shape {value.shape} to a selection of shape {shape}"
            ) from None
    else:
        values = np.broadcast_to(np.asarray(value, dtype=x.dtype), shape)
    coords = [c.reshape(-1) for c in coords]
    values = values.reshape(-1)
    # The last of repeated positions wins, as in numpy
    flat = np.ravel_multi_index(coords, x.shape)
    _, last = np.unique(flat[::-1], return_index=True)
    keep = np.sort(flat.size - 1 - last)
    coords = [c[keep] for c in coords]
    values = values[keep]
    stored = _utils.differs(values, x._fill)
    T = x._T()
    gbdt = gb_dtype(x.dtype)
    if stored.any():
        U = Tensor.from_coo([c[stored] for c in coords], values[stored], gbdt, shape=x.shape)
        T(U.S) << U
    if not stored.all() and T.nvals:
        D = Tensor.from_coo([c[~stored] for c in coords], True, shape=x.shape)
        T(D.S) << Tensor(gbdt, x.shape)


# take, take_along_axis
def _index_operand(indices, name):
    if isinstance(indices, Array):
        if indices.dtype.kind not in "iu":
            raise TypeError(f"{name} needs integer indices, not {indices.dtype}")
        return indices.todense().astype(np.int64, copy=False)
    arr = np.asarray(indices)
    if arr.dtype.kind not in "iu" and arr.size:
        raise TypeError(f"{name} needs integer indices, not {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _check_bounds(idx, n, axis):
    if idx.size and (idx.min() < -n or idx.max() >= n):
        raise IndexError(f"index out of bounds for axis {axis} with size {n}")


def take(x, indices, /, *, axis=None):
    """``x``'s elements at ``indices`` along ``axis`` (the flattened array if None)."""
    from ._manipulation import _normalize_axis

    if not isinstance(x, Array):
        raise TypeError(f"take needs an Array, not {type(x).__name__}")
    idx = _index_operand(indices, "take")
    if x.ndim == 0:
        raise ValueError("take needs an array of at least one dimension")
    if axis is None:
        if x.ndim > 1:
            raise ValueError("take needs an axis for an array of more than one dimension")
        axis = 0
    axis = _normalize_axis(axis, x.ndim)
    _check_bounds(idx, x.shape[axis], axis)
    T = x._T()
    shape = x.shape[:axis] + idx.shape + x.shape[axis + 1 :]
    if T.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    return Array._new(T.take(idx, axis=axis), x._fill)


def take_along_axis(x, indices, /, *, axis=-1):
    from ._manipulation import _normalize_axis

    if not isinstance(x, Array):
        raise TypeError(f"take_along_axis needs an Array, not {type(x).__name__}")
    idx = _index_operand(indices, "take_along_axis")
    if idx.ndim != x.ndim:
        raise ValueError("take_along_axis needs indices with as many dimensions as x")
    axis = _normalize_axis(axis, x.ndim)
    _check_bounds(idx, x.shape[axis], axis)
    others = [s for a, s in enumerate(x.shape) if a != axis]
    iothers = [s for a, s in enumerate(idx.shape) if a != axis]
    bshape = list(np.broadcast_shapes(tuple(others), tuple(iothers)))
    bshape.insert(axis, idx.shape[axis])
    shape = tuple(bshape)
    T = x._T()
    if T.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    return Array._new(T.take_along_axis(idx, axis=axis), x._fill)

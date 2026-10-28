"""Tensor index assignment and vectorized gather (``take``, ``take_along_axis``, ``vindex``).

**Assignment** follows python-graphblas Matrix assignment, extended to n dimensions:
``T[idx] << value`` and ``T[idx] = value``, ``T(mask, accum, replace)[idx] << value``
(GrB_assign: the mask has T's shape) and ``T[idx](mask, accum, replace) << value``
(GxB_subassign: the mask has the region's shape). ``idx`` is outer, as for extraction. The
value is broadcast (numpy rules) to the region's shape, and the region takes its structure: a
region entry the value lacks is deleted, a scalar fills the region. With repeated indices the
last occurrence wins, as for Matrix (SuiteSparse).

An outer region of any layout is an I x J submatrix of the stored Matrix (each group's
positions are the raveled outer product of its axes' indices), so every assignment is one
SuiteSparse assign or subassign on the stored Matrix, with the value and a region mask moved
to the region's matching layout.
"""

import math

import numpy as np

from ..dtypes import BOOL
from ..exceptions import DimensionMismatch, IndexOutOfBound
from . import tensor_census as census
from . import tensor_layout as tl
from .expr import Updater
from .scalar import Scalar
from .tensor_layout import Layout


def _resolve(t, keys):
    """``(keys, sel)`` as extraction resolves them: one entry per axis, an int (axis dropped),
    a range (a slice) or an int64 array of coordinates (negatives wrapped once).
    """
    keys = keys if isinstance(keys, tuple) else (keys,)
    if any(k is None for k in keys):
        raise NotImplementedError("None (a new axis) in a Tensor index; use expand_dims")
    if (n_ellipsis := sum(k is Ellipsis for k in keys)) > 1:
        raise IndexError("an index can only have a single ellipsis")
    if n_ellipsis:
        at = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:at] + (slice(None),) * (t.ndim - len(keys) + 1) + keys[at + 1 :]
    if len(keys) > t.ndim:
        raise IndexError(f"too many indices for a Tensor of {t.ndim} dimensions")
    keys = keys + (slice(None),) * (t.ndim - len(keys))
    keys = tuple(_scalar_index(k, a) for a, k in enumerate(keys))
    sel = [_axis_index(k, n, a) for a, (k, n) in enumerate(zip(keys, t._shape, strict=True))]
    return keys, sel


def _region_value(value, shape, dtype):
    """The value as a Tensor of the region's ``shape`` (numpy broadcasting), or a Scalar or
    Python scalar to fill it with.
    """
    from .base import BaseExpression
    from .scalar import Scalar
    from .tensor import Tensor, _as_tensor, _broadcast_shapes, _expand

    if isinstance(value, Scalar) or np.isscalar(value) and not isinstance(value, (str, bytes)):
        return value
    if isinstance(value, BaseExpression) and value.output_type is Scalar:
        return value.new()
    if hasattr(value, "_to_expr") and not isinstance(value, Tensor):
        value = value._to_expr()  # (an infix expression)
    if isinstance(value, BaseExpression):
        value = value.new()
    t = _as_tensor(value, "assignment")
    if t is None:
        raise TypeError(f"Bad type for the assigned value: {type(value)}")
    if t.ndim == 0 and shape != ():
        # (a GrB_Scalar, which may be empty when assigned; a C scalar may not)
        return t._M[0, 0].new(is_cscalar=False)  # pragma: is_grbscalar
    try:
        broadcast = _broadcast_shapes(t._shape, shape)
    except ValueError:
        broadcast = None
    if broadcast != shape:
        raise DimensionMismatch(
            f"the value's shape {t._shape} does not broadcast to the region's shape {shape}"
        )
    if t._shape != shape:
        if t.ndim < len(shape):
            t = t.expand_dims(tuple(range(len(shape) - t.ndim)))
        t = _expand(t, shape)
    return t


def assign(t, keys, value, *, mask=None, accum=None, replace=False, is_submask=False, opts=None):
    """``t[keys](...) << value`` (``is_submask``) or ``t(...)[keys] << value``, in place."""
    from .tensor import Tensor, _as_tensor

    opts = opts or {}
    keys, sel = _resolve(t, keys)
    kept = [a for a in range(t.ndim) if not isinstance(sel[a], int)]
    renum = {a: i for i, a in enumerate(kept)}
    shape = tuple(len(sel[a]) for a in kept)
    rows, cols = t._layout
    ridx = _group_index(t._shape, rows, sel, keys)
    cidx = _group_index(t._shape, cols, sel, keys)
    if ridx is None or cidx is None:
        raise NotImplementedError(
            f"the region lists more than {2**26} rows or columns of the stored Matrix; "
            "relayout the Tensor so that whole axes share a group"
        )
    # The region in the stored Matrix's row and column order
    region = Layout(
        tuple(renum[a] for a in rows if a in renum), tuple(renum[a] for a in cols if a in renum)
    )
    nthreads = opts.get("nthreads")
    rhs = _region_value(value, shape, t.dtype)
    if isinstance(rhs, Tensor):
        V = rhs._M
        if rhs._layout != region:
            V, _ = tl.relayout(V, shape, rhs._layout, region, nthreads=nthreads)
    else:
        V = rhs
    M_mask = None
    if mask is not None:
        parent = mask.parent if type(mask.parent) is Tensor else _as_tensor(mask.parent, "mask")
        want = shape if is_submask else t._shape
        if parent is None or parent._shape != want:
            raise DimensionMismatch(
                f"the mask's shape {getattr(mask.parent, 'shape', None)} does not match "
                f"{'the region' if is_submask else 'the Tensor'}'s shape {want}"
            )
        M = parent._M
        target = region if is_submask else t._layout
        if parent._layout != target:
            M, _ = tl.relayout(M, want, parent._layout, target, nthreads=nthreads)
        M_mask = type(mask)(M)
    t._own()
    M = t._M
    kwargs = dict(accum=accum, **opts)
    if M_mask is not None:
        kwargs.update(mask=M_mask, replace=replace)
    if is_submask:
        M[ridx, cidx](**kwargs) << V
    else:
        M(**kwargs)[ridx, cidx] << V


class TensorUpdater(Updater):
    """``T(mask, accum, replace)``: updates with ``<<``, and assignment by index."""

    __slots__ = ()

    def __getitem__(self, keys):
        return TensorAssigner(self, keys, is_submask=False)

    def __setitem__(self, keys, value):
        self._assign(keys, value, is_submask=False)

    def __delitem__(self, keys):
        from .scalar import Scalar

        # Delete by assigning an empty scalar, as Matrix does
        scalar = Scalar(self.parent.dtype, is_cscalar=False, name="s_empty")  # pragma: is_grbscalar
        self._assign(keys, scalar, is_submask=False)

    def _assign(self, keys, value, *, is_submask):
        kw = self.kwargs
        assign(self.parent, keys, value, mask=kw.get("mask"), accum=kw.get("accum"),
               replace=kw.get("replace", False), is_submask=is_submask, opts=self.opts)  # fmt: skip


class TensorAssigner:
    """``T[idx](mask, accum, replace)`` or ``T(mask, accum, replace)[idx]``: receives ``<<``."""

    __slots__ = "updater", "keys", "is_submask"

    def __init__(self, updater, keys, *, is_submask):
        self.updater = updater
        self.keys = keys
        self.is_submask = is_submask

    def __lshift__(self, value):
        self.updater._assign(self.keys, value, is_submask=self.is_submask)

    update = __lshift__

    def __eq__(self, other):
        raise TypeError(f"__eq__ not defined for objects of type {type(self)}.")

    __hash__ = None

    def __bool__(self):
        raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")


# ---------------------------------------------------------------------------------------------
# Gather: take, take_along_axis, vindex

# The gather engines, tried in this order where each runs (all kept, to measure; tests force
# some): "extract" (one SuiteSparse extract, for ``take``), "numba" (binary searches over the
# stored arrays, per point), "graphblas" (the points as an iso pattern Matrix, intersected with
# the stored Matrix by ``ewise_mult``), which runs every case.
_GATHER_ENGINES = ("extract", "numba", "graphblas")


def _index_array(index, size, axis, wraparound, mode="raise"):
    """An int64 array of coordinates along ``axis`` (an ndarray of any shape; a Scalar or int
    is 0-d). Negatives count from the end once, unless ``wraparound`` is False (they raise
    IndexError); out of range raises IndexError, or with numpy's ``take`` modes ``"wrap"`` and
    ``"clip"`` wraps or clips (a negative clips to 0).
    """
    from .scalar import Scalar

    if isinstance(index, Scalar):
        index = _scalar_index(index, axis)
    arr = np.asarray(index)
    if arr.dtype == bool:
        raise TypeError("boolean indices are not supported for Tensor")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"indices must be integers; got {arr.dtype} (axis {axis})")
    arr = arr.astype(np.int64)
    if not arr.size:
        return arr
    if not wraparound and arr.min() < 0:
        raise IndexError(f"an index is negative (wraparound=False) for axis {axis} of size {size}")
    if size and mode == "wrap":
        return np.mod(arr, size)
    if size and mode == "clip":
        return np.clip(arr, 0, size - 1)
    if arr.min() < -size or arr.max() >= size:
        raise IndexError(f"an index is out of bounds for axis {axis} of size {size}")
    return np.where(arr < 0, arr + size, arr)


def gather(t, coords, shape):
    """The Tensor of ``shape`` whose position k (C order) holds ``t`` at the point
    ``coords[a][k]`` (one flat int64 array per axis of ``t``), missing where ``t`` is.
    """
    from .tensor import Tensor

    n = math.prod(shape)
    if n == 0:
        return Tensor(t.dtype, shape)
    # The lookups trust the raveled positions (the Numba one reads the row pointers with them),
    # so a coordinate out of range must not get that far
    for a, (c, size) in enumerate(zip(coords, t._shape, strict=True)):
        if c.min() < 0 or c.max() >= size:
            raise IndexError(f"a coordinate is out of bounds for axis {a} of size {size}")
    if t._M._nvals == 0:
        return Tensor(t.dtype, shape)
    found, vals = lookup(t, coords, n)
    pos = np.flatnonzero(found)
    out = np.unravel_index(pos, shape) if shape else ()
    return Tensor.from_coo(out, vals[pos], t.dtype, shape=shape)


def lookup(t, coords, n):
    """``(found, values)`` of ``t`` at the ``n`` points ``coords`` (an int64 array per axis,
    in range), by the first lookup engine of ``_GATHER_ENGINES`` that runs.
    """
    rows = tl.ravel(coords, t._shape, t._layout.rows, n=n)
    cols = tl.ravel(coords, t._shape, t._layout.cols, n=n)
    for engine in _GATHER_ENGINES:
        if engine == "numba":
            rv = _lookup_numba(t, rows, cols)
        elif engine == "graphblas":
            rv = _lookup_graphblas(t, rows, cols)
        else:
            continue
        if rv is not None:
            found, vals = rv
            break
    else:
        engine = "graphblas"
        found, vals = _lookup_graphblas(t, rows, cols)
    if census.on:
        census.note("gather", engine, points=int(rows.size), format=t._M.ss.format)
    return found, vals


def _lookup_numba(t, rows, cols):
    from . import _has_numba

    M = t._M
    if not _has_numba or M.dtype._is_udt:
        return None
    fmt = tl.stored_format(M)
    np_type = M.dtype.np_type
    with tl._Unloaded(M) as cont:
        if fmt in {"bitmapr", "fullr"}:
            k = rows * M._ncols + cols
            with tl._VectorArray(cont.x, np_type) as x:
                vals = x.array[0 if cont.iso else k].copy() if x.array is not None else None
                vals = np.broadcast_to(vals, k.shape).copy() if cont.iso else vals
            if fmt == "fullr":
                return np.ones(k.size, bool), vals
            with tl._VectorArray(cont.b, np.int8) as b:
                return b.array[k].astype(bool), vals
        if fmt not in {"csr", "hypercsr"}:
            # Held by column: only a Matrix the Tensor shares (from_matrix(copy=False),
            # T.ss.matrix) that the user flipped; the GraphBLAS lookup takes it
            return None
        from .tensor_numba import lookup

        with (
            tl._VectorArray(cont.p) as p,
            tl._VectorArray(cont.h) as h,
            tl._VectorArray(cont.i) as ci,
            tl._VectorArray(cont.x, np_type) as x,
        ):
            vals = np.empty(rows.size, np_type)
            found = np.zeros(rows.size, bool)
            hh = h.array if h.array is not None else np.empty(0, np.int64)
            lookup(p.array, hh, ci.array, x.array, bool(cont.iso), rows, cols, vals, found)
    return found, vals


def _lookup_graphblas(t, rows, cols):
    """The points' entries by SuiteSparse: the distinct points as an iso pattern Matrix,
    ``ewise_mult(first)`` with the stored Matrix, then each point's hit by binary search.
    """
    from .. import binary
    from .matrix import Matrix

    M = t._M
    if M._nrows * M._ncols < 2**63:
        # One int64 key per point, row-major: sorted keys are sorted (row, col) pairs
        def encode(r, c):
            return r * M._ncols + c

        ukey, inverse = np.unique(encode(rows, cols), return_inverse=True)
        urows, ucols = np.divmod(ukey, M._ncols)
    else:
        # 2^63 positions or more: the key is the row's rank among the points' distinct rows,
        # then the column's, which fits (below points^2) and sorts as (row, col) does
        rv, cv = np.unique(rows), np.unique(cols)

        def encode(r, c):
            return np.searchsorted(rv, r) * cv.size + np.searchsorted(cv, c)

        ukey, inverse = np.unique(encode(rows, cols), return_inverse=True)
        urows, ucols = rv[ukey // cv.size], cv[ukey % cv.size]
    P = Matrix.from_coo(urows, ucols, True, nrows=M._nrows, ncols=M._ncols)
    hit = M.ewise_mult(P, binary.first).new()
    hr, hc, hv = hit.to_coo()
    # Each hit is one of the points: its key's position among theirs (a binary search)
    at = np.searchsorted(ukey, encode(hr.view(np.int64), hc.view(np.int64)))
    found_u = np.zeros(urows.size, bool)
    vals_u = np.empty(urows.size, hv.dtype)
    found_u[at] = True
    vals_u[at] = hv
    inverse = inverse.ravel()
    return found_u[inverse], vals_u[inverse]


def take(t, indices, axis=None, *, mode="raise", wraparound=True):
    """Numpy's ``take``: along ``axis`` (the flattened Tensor in C order when None) at
    ``indices`` (any shape); the result's shape puts ``indices.shape`` in place of the axis.
    """
    from .tensor import _normalize_axis

    if mode not in {"raise", "wrap", "clip"}:
        raise ValueError(f"mode must be 'raise', 'wrap' or 'clip'; got {mode!r}")
    if axis is None:
        return take(t.reshape(-1), indices, 0, mode=mode, wraparound=wraparound)
    axis = _normalize_axis(axis, t.ndim)
    idx = _index_array(indices, t._shape[axis], axis, wraparound, mode)
    shape = t._shape[:axis] + idx.shape + t._shape[axis + 1 :]
    if "extract" in _GATHER_ENGINES:
        if census.on:
            census.note("gather", "extract", points=int(idx.size))
        keys = (slice(None),) * axis + (idx.ravel(),)
        out = t[keys].new()
        return out.reshape(shape)
    grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij") if shape else []
    coords = []
    for a in range(t.ndim):
        if a < axis:
            coords.append(grids[a].ravel())
        elif a == axis:
            coords.append(
                idx[tuple(grids[axis : axis + idx.ndim])].ravel()
                if idx.ndim
                else np.full(math.prod(shape), int(idx))
            )
        else:
            coords.append(grids[a - 1 + idx.ndim].ravel())
    return gather(t, coords, shape)


def take_along_axis(t, indices, axis, *, wraparound=True):
    """Numpy's ``take_along_axis``: ``indices`` has the Tensor's ndim and broadcasts against
    it except along ``axis``; ``out[..., i, ...] = t[..., indices[..., i, ...], ...]``.
    """
    from .tensor import _normalize_axis

    axis = _normalize_axis(axis, t.ndim)
    arr = np.asarray(indices)
    if arr.ndim != t.ndim:
        raise ValueError(f"indices must have the Tensor's {t.ndim} dimensions; got {arr.ndim}")
    idx = _index_array(arr, t._shape[axis], axis, wraparound)
    tshape = list(t._shape)
    tshape[axis] = 1
    try:
        shape = np.broadcast_shapes(tuple(tshape), idx.shape)
    except ValueError:
        raise ValueError(
            f"indices of shape {idx.shape} do not broadcast against {t._shape} along the "
            f"other axes"
        ) from None
    # As numpy's _make_along_axis_idx: each other axis counts over the Tensor's own size, so
    # along a size-1 axis that the indices broadcast over the coordinate stays 0
    coords = []
    for a, size in enumerate(t._shape):
        if a == axis:
            coords.append(np.broadcast_to(idx, shape).ravel())
        else:
            along = [1] * t.ndim
            along[a] = size
            grid = np.arange(size, dtype=np.int64).reshape(along)
            coords.append(np.broadcast_to(grid, shape).ravel())
    return gather(t, coords, shape)


def vindex(t, keys, *, wraparound=True):
    """Vectorized (pointwise) integer indexing, numpy's advanced indexing: one int, Scalar or
    integer array per axis, broadcast together; the result has their broadcast shape.
    """
    keys = keys if isinstance(keys, tuple) else (keys,)
    if len(keys) != t.ndim:
        raise IndexError(
            f"vindex needs one index per axis ({t.ndim}); got {len(keys)} (slices are not "
            "supported: use T[...] for outer indexing)"
        )
    if any(isinstance(k, slice) or k is Ellipsis or k is None for k in keys):
        raise IndexError("vindex takes ints, Scalars and integer arrays only")
    idx = [
        _index_array(k, n, a, wraparound)
        for a, (k, n) in enumerate(zip(keys, t._shape, strict=True))
    ]
    shape = np.broadcast_shapes(*(x.shape for x in idx))
    coords = [np.broadcast_to(x, shape).ravel() for x in idx]
    return gather(t, coords, shape)


class VIndexer:
    """``T.vindex[i0, i1, ...]``; ``T.vindex(wraparound=False)[...]`` refuses negatives."""

    __slots__ = "t", "wraparound"

    def __init__(self, t, wraparound=True):
        self.t = t
        self.wraparound = wraparound

    def __call__(self, *, wraparound=True):
        return VIndexer(self.t, wraparound)

    def __getitem__(self, keys):
        return vindex(self.t, keys, wraparound=self.wraparound)


def _scalar_index(index, axis):
    """A GraphBLAS Scalar indexes as the int it holds, as for Matrix (which refuses empty and
    non-integer Scalars with TypeError too).
    """
    if type(index) is not Scalar:
        return index
    if not np.issubdtype(index.dtype.np_type, np.integer):
        raise TypeError(
            f"An integer is required for indexing.  Got: {index.dtype} (index for axis {axis})"
        )
    if index._is_empty:
        raise TypeError(f"The index for axis {axis} is an empty Scalar")
    return int(index.value)


def _axis_index(index, size, axis):
    """An int (axis dropped), a range (a slice: kept as one, so that indexing costs follow the
    entries rather than the axis lengths) or a sorted-or-not int64 array of coordinates.
    """
    if isinstance(index, (int, np.integer)):
        i = int(index)
        if not -size <= i < size:
            raise IndexOutOfBound(f"index {i} is out of bounds for axis {axis} of size {size}")
        return i + size if i < 0 else i
    if isinstance(index, slice):
        return range(*index.indices(size))
    arr = np.asarray(index)
    if arr.dtype == bool:
        raise TypeError("boolean indices are not supported for Tensor")
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f"bad index for axis {axis}: {index!r}")
    arr = arr.astype(np.int64)
    if arr.size and (arr.min() < -size or arr.max() >= size):
        raise IndexOutOfBound(f"an index is out of bounds for axis {axis} of size {size}")
    return np.where(arr < 0, arr + size, arr)


def _extract_expr(self, keys):
    keys, sel = _resolve(self, keys)
    shape = tuple(len(x) for x in sel if not isinstance(x, int))
    expr = TensorIndexExpr(
        "extract",
        _extract_recipe,
        [self],
        (self, sel, keys),
        dtype=self.dtype,
        shape=shape,
        op=None,
        expr_repr="{0.name}[...]",
    )
    expr._parent = self
    expr._keys = keys
    return expr


def _group_index(shape, axes, sel, keys, *, build=True):
    """The rows (or columns) of one group to extract: ``slice(None)`` for all of them, a slice
    when they are a contiguous range of raveled positions (a leading int or range with the
    rest of the group whole), else an array of raveled positions (None past ``_MAX_LISTED``).
    With ``build=False``, the number of those positions instead of the array: it is sized by
    the index, not the data (a reversed full slice lists every row), so it is built only once
    the extract is chosen.
    """
    full = [isinstance(keys[a], slice) and keys[a] == slice(None) for a in axes]
    if all(full):
        return slice(None)
    # Contiguous: single positions, then one contiguous range, then whole axes
    k = 0
    while k < len(axes) and isinstance(sel[axes[k]], int):
        k += 1
    if k < len(axes):
        x = sel[axes[k]]
        if isinstance(x, range):
            contiguous = len(x) == 1 or len(x) > 0 and x.step == 1
        else:
            contiguous = x.size > 0 and (x.size == 1 or bool(np.all(np.diff(x) == 1)))
        tail_full = all(full[k + 1 :])
    else:
        contiguous, tail_full = True, True
    if contiguous and tail_full:
        start = 0
        for a in axes[:k]:
            start = start * shape[a] + sel[a]
        tail = math.prod(shape[a] for a in axes[k + 1 :])
        if k < len(axes):
            x = sel[axes[k]]
            start = (start * shape[axes[k]] + int(x[0])) * tail
            return slice(start, start + len(x) * tail)
        return slice(start, start + 1)
    listed = math.prod(1 if isinstance(s, int) else len(s) for s in (sel[a] for a in axes))
    if listed > _t._MAX_LISTED:
        return None
    if not build:
        return listed
    idx = np.zeros(1, np.int64)
    for a in axes:
        x = _as_array(sel[a])
        idx = (idx[:, None] * shape[a] + x[None, :]).ravel()
    return idx


def _as_array(s):
    """An axis selection (int, range or array) as an int64 array."""
    if isinstance(s, int):
        return np.array([s], np.int64)
    if isinstance(s, range):
        return np.arange(s.start, s.stop, s.step, dtype=np.int64)
    return s


def _positions(s, coord):
    """For coordinates ``coord`` along an axis selected by ``s`` (a range, or an array without
    repeats), whether each is selected and its position in ``s``. Arithmetic for a range, a
    binary search for an array: sized by ``s`` and the coordinates, not the axis length.
    """
    if isinstance(s, range):
        d = coord - s.start
        q = d // s.step
        return (d % s.step == 0) & (q >= 0) & (q < len(s)), q
    order = np.argsort(s, kind="stable")
    ss = s[order]
    i = np.minimum(np.searchsorted(ss, coord), max(ss.size - 1, 0))
    if not ss.size:
        return np.zeros(coord.size, bool), i
    return ss[i] == coord, order[i]


# Extraction costs (ns, 8 / 1 threads, measured 2026-10-20): a GraphBLAS extract pays per
# nonempty row and per listed index of each group (u7 X[:, :, k]: 1e6 rows, one index, 3.9 ms;
# hyper (2^19)^3 X[:, :, k]: 4.5e5 rows, 2^19 indices, 8.6 ms); filtering the coordinates pays
# per stored entry and per indexed axis (hyper 2.4 ms for 1e6 entries; u6 X[:, :, k0:k1] at
# 1 thread 8.4 ms for 1e6, where the extract took 3.6).
C_X_ROW = (3.9, 19.0)

C_X_LIST = (13.0, 74.0)

C_FILTER = (2.4, 7.0)

C_FILTER_AXIS = (1.0, 1.5)


def _extract_recipe(updater, t, sel, keys):
    rows, cols = t._layout
    kept = [a for a in range(t.ndim) if not isinstance(sel[a], int)]
    renum = {a: i for i, a in enumerate(kept)}
    layout = Layout(
        tuple(renum[a] for a in rows if a in renum), tuple(renum[a] for a in cols if a in renum)
    )
    shape = tuple(len(sel[a]) for a in kept)
    # (positions to list are counted here, and listed only if the extract is chosen)
    ridx = _group_index(t._shape, rows, sel, keys, build=False)
    cidx = _group_index(t._shape, cols, sel, keys, build=False)
    listed = None
    if ridx is not None and cidx is not None:
        listed = sum(x for x in (ridx, cidx) if isinstance(x, int))
        if not any(isinstance(x, int) for x in (ridx, cidx)):
            if census.on:
                census.note("extract", "slices: extract")
            _store(updater, t._M[ridx, cidx], shape, layout)
            return
    # Repeated indices repeat the entries, as for Matrix (SuiteSparse's extract does; the
    # filter below expands them; the row pick takes slices and ints only)
    repeated = {
        a for a, s in enumerate(sel) if isinstance(s, np.ndarray) and np.unique(s).size != s.size
    }
    ti = _ti(_nthreads(updater))
    axes = [a for a in range(t.ndim) if not (isinstance(keys[a], slice) and keys[a] == slice(None))]
    nv = t._M._nvals
    options = [((C_FILTER[ti] + C_FILTER_AXIS[ti] * len(axes)) * nv, "filter")]
    if listed is not None:
        options.append((C_X_ROW[ti] * tl.nonempty_rows(t._M) + C_X_LIST[ti] * listed, "extract"))
    if (
        nv
        and tl.stored_format(t._M) == "hypercsr"
        and not t.dtype._is_udt
        and all(a in rows and (isinstance(keys[a], slice) or isinstance(sel[a], int)) for a in axes)
    ):
        # Pick rows among the nonempty ones (h) of a hypersparse row group: sized by the data
        # (NELL-2's (0,1|2), j in a range: the extract listed 1.1e7 rows, the filter read 7.7e7
        # entries). Numpy work per nonempty row, then P @ M over the kept ones.
        frac = math.prod(
            1 / t._shape[a] if isinstance(sel[a], int) else len(sel[a]) / t._shape[a] for a in axes
        )
        hrows = tl.nonempty_rows(t._M)
        cost = (
            tl.C_PBUILD_ROW[ti] * hrows
            + tl.C_PERM_ROW["hyper"][ti] * frac * hrows
            + tl.C_PERM_ENTRY[ti] * frac * nv
        )
        options.append((cost, "rows"))
    i = min(range(len(options)), key=lambda k: options[k][0])
    if census.on:
        i = census.choose("extract", [o[0] for o in options], [o[1] for o in options], i,
                          nvals=nv, repeated=bool(repeated))  # fmt: skip
    _, how = options[i]
    if how == "extract":
        if isinstance(ridx, int):
            ridx = _group_index(t._shape, rows, sel, keys)
        if isinstance(cidx, int):
            cidx = _group_index(t._shape, cols, sel, keys)
        _store(updater, t._M[ridx, cidx], shape, layout)
        return
    if how == "rows":
        S = _hyper_rows_pick(t, sel, keys, renum, shape, layout)
        _store(updater, S.mxm(t._M, tl._perm_semiring(t._M)).new(), shape, layout, fresh=True)
        return
    # Filter the stored coordinates: cheaper than listing many indices (a slab of a wide
    # column group), and the only way past _MAX_LISTED positions
    r, c, vals = t._M.to_coo(sort=False)
    r, c = r.view(np.int64), c.view(np.int64)
    keep = np.ones(vals.size, bool)
    for a in range(t.ndim):
        s = sel[a]
        if isinstance(keys[a], slice) and keys[a] == slice(None):
            continue
        coord = _axis_coords(r, c, t._shape, rows, cols, a)
        if isinstance(s, int):
            keep &= coord == s
        else:
            keep &= _positions(s, coord)[0]
    r, c, vals = r[keep], c[keep], vals[keep]
    out = {}
    for a in kept:
        coord = _axis_coords(r, c, t._shape, rows, cols, a)
        s = sel[a]
        if isinstance(keys[a], slice) and keys[a] == slice(None):
            out[renum[a]] = coord
        elif a in repeated:
            # Each entry once per position its coordinate is listed at
            order = np.argsort(s, kind="stable")
            ss = s[order]
            left = np.searchsorted(ss, coord, "left")
            count = np.searchsorted(ss, coord, "right") - left
            rep = np.repeat(np.arange(coord.size), count)
            offs = np.arange(rep.size) - np.repeat(np.cumsum(count) - count, count)
            r, c, vals = r[rep], c[rep], vals[rep]
            out = {k: v[rep] for k, v in out.items()}
            out[renum[a]] = order[left[rep] + offs]
        else:
            out[renum[a]] = _positions(s, coord)[1]
    n = vals.size
    rr = tl.ravel(out, shape, layout.rows, n=n)
    cc = tl.ravel(out, shape, layout.cols, n=n)
    M = tl.from_coo(rr, cc, vals, t.dtype, tl.prod(shape, layout.rows), tl.prod(shape, layout.cols))
    _store(updater, M, shape, layout, fresh=True)


def _hyper_rows_pick(t, sel, keys, renum, shape, layout):
    """``S`` with ``S @ t._M`` extracting ``t[keys]`` when the keys index only axes of the row
    group: built over the nonempty rows, mapping each kept one to its row in the result.
    """
    rows = t._layout.rows
    h = tl.nonempty_row_ids(t._M)
    co = {}
    tl.unravel(h, tuple(t._shape[a] for a in rows), tuple(range(len(rows))), co)
    keep = np.ones(h.size, bool)
    positions = {}
    for i, a in enumerate(rows):
        s = sel[a]
        if isinstance(s, int):
            keep &= co[i] == s
        elif not (isinstance(keys[a], slice) and keys[a] == slice(None)):
            positions[a] = _positions(s, co[i])
            keep &= positions[a][0]
    n = int(np.count_nonzero(keep))
    out = {}
    for i, a in enumerate(rows):
        if a in renum:
            c = co[i][keep]
            out[renum[a]] = positions[a][1][keep] if a in positions else c
    dst = tl.ravel(out, shape, layout.rows, n=n)
    nrows = tl.prod(shape, layout.rows)
    return tl.from_coo(dst, h[keep], True, BOOL, nrows, t._M._nrows)


def _axis_coords(r, c, shape, rows, cols, axis):
    """Coordinates along ``axis`` of entries at linear rows ``r`` and columns ``c``."""
    group, lin = (rows, r) if axis in rows else (cols, c)
    stride = tl.prod(shape, group[group.index(axis) + 1 :])
    return (lin // stride) % shape[axis]


# isort: off
from . import tensor as _t  # noqa: E402 (knobs are set on the hub)
from .tensor import (  # noqa: E402
    TensorIndexExpr,
    _nthreads,
    _store,
    _ti,
)

# isort: on

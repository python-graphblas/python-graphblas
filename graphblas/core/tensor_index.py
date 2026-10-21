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

from ..exceptions import DimensionMismatch
from . import tensor_census as census
from . import tensor_layout as tl
from .expr import Updater
from .tensor_layout import Layout


def _resolve(t, keys):
    """``(keys, sel)`` as extraction resolves them: one entry per axis, an int (axis dropped)
    or an int64 array of coordinates (negatives wrapped once).
    """
    from .tensor import _axis_index, _scalar_index

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
    from .tensor import Tensor, _as_tensor, _expand

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
        return t.to_scalar()
    try:
        broadcast = np.broadcast_shapes(t._shape, shape)
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
    from .tensor import Tensor, _as_tensor, _group_index

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

        self._assign(keys, Scalar(self.parent.dtype), is_submask=False)

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
    from .tensor import _scalar_index

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
    if n == 0 or t._M._nvals == 0:
        return Tensor(t.dtype, shape)
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
    pos = np.flatnonzero(found)
    out = np.unravel_index(pos, shape) if shape else ()
    return Tensor.from_coo(out, vals[pos], t.dtype, shape=shape)


def _lookup_numba(t, rows, cols):
    from . import _has_numba

    M = t._M
    if not _has_numba or M.dtype._is_udt:
        return None
    fmt = M.ss.format
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
        key = rows * M._ncols + cols
        ukey, inverse = np.unique(key, return_inverse=True)
        urows, ucols = np.divmod(ukey, M._ncols)
    else:  # pragma: no cover (more than 2^63 positions)
        uniq, inverse = np.unique(np.stack([rows, cols]), axis=1, return_inverse=True)
        urows, ucols, ukey = uniq[0], uniq[1], None
    P = Matrix.from_coo(urows, ucols, True, nrows=M._nrows, ncols=M._ncols)
    hit = M.ewise_mult(P, binary.first).new()
    hr, hc, hv = hit.to_coo()
    hr, hc = hr.view(np.int64), hc.view(np.int64)
    if ukey is not None:
        at = np.searchsorted(ukey, hr * M._ncols + hc)
    else:  # pragma: no cover (more than 2^63 positions: match each hit by row and column)
        at = np.array(
            [np.flatnonzero((urows == r) & (ucols == c))[0] for r, c in zip(hr, hc, strict=True)]
        )
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
    grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    coords = [
        np.broadcast_to(idx, shape).ravel() if a == axis else grids[a].ravel()
        for a in range(t.ndim)
    ]
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

"""Tensor: an n-dimensional sparse array on one SuiteSparse Matrix (experimental).

Semantics follow GraphBLAS for missing values (no fill value) and numpy / the Array API for
shapes and axes. Operations are lazy, as for Matrix: they return a ``TensorExpression``
that computes on ``.new()`` or ``T(mask, accum, replace) << expr``.

A Tensor stores its entries in one Matrix held by row (``T.ss.matrix``) plus a ``Layout``:
which axes are raveled (C order) into the rows and which into the columns. ``permute_dims``,
``.T``, ``.mT``, ``expand_dims``, ``squeeze``, and ``reshape`` where the layout holds the
result, only change the layout, in O(1); the data moves inside an operation when that
operation needs it (``tensor_layout.relayout``).
Tensors that share a stored Matrix copy it before one of them is changed in place
(copy-on-write), so a Tensor never sees another's writes.

**THIS API IS EXPERIMENTAL AND MAY CHANGE.** SuiteSparse:GraphBLAS 10 or later, with the
suitesparse backend. ``gb.Tensor`` and ``gb.einsum``; the user guide's "Tensors" page
covers the model, the operations and the engines.
"""

import contextlib
import itertools
import math
import operator
import weakref

import numpy as np

from .. import backend, binary, config, monoid, semiring
from ..dtypes import BOOL, FP64, lookup_dtype, unify
from ..exceptions import DimensionMismatch, IndexOutOfBound
from . import _has_numba, automethods, lib
from . import tensor_layout as tl
from . import utils
from .base import BaseExpression, BaseType
from .expr import InfixExprBase, Updater
from .mask import StructuralMask, ValueMask
from .matrix import Matrix, TransposedMatrix
from .operator import get_semiring, get_typed_op
from .scalar import Scalar, ScalarExpression, _union_default
from .tensor_layout import Layout
from .utils import output_type
from .vector import Vector

__all__ = ["Layout", "Tensor", "TensorExpression", "einsum"]


def _check_backend():
    if backend != "suitesparse":
        raise NotImplementedError("Tensor requires the suitesparse backend")
    if not hasattr(lib, "GxB_Container_new"):  # pragma: no cover (old SuiteSparse)
        raise NotImplementedError("Tensor requires SuiteSparse:GraphBLAS 10 or later")


def _as_shape(shape):
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError:
        raise TypeError(f"shape must be a sequence of ints; got {shape!r}") from None
    if any(n < 0 for n in shape):
        raise ValueError(f"shape may not have negative sizes; got {shape}")
    return shape


def _normalize_axis(axis, n):
    try:
        axis = operator.index(axis)
    except TypeError:
        raise TypeError(f"axis must be an int; got {axis!r}") from None
    if not -n <= axis < n:
        raise ValueError(f"axis {axis} is out of bounds for {n} axes")
    return axis + n if axis < 0 else axis


def _normalize_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (tuple, list)):
        axes = tuple(_normalize_axis(a, ndim) for a in axis)
        if len(set(axes)) != len(axes):
            raise ValueError(f"repeated axis in {axis}")
        return tuple(sorted(axes))
    return (_normalize_axis(axis, ndim),)


def _renumber(layout, mapping):
    return Layout(tuple(mapping[a] for a in layout.rows), tuple(mapping[a] for a in layout.cols))


def _reshape_shape(shape, old_shape):
    """The new shape of a reshape, with -1 resolved, checked against the old size."""
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError:
        raise TypeError(f"shape must be ints or a sequence of ints; got {shape!r}") from None
    size = math.prod(old_shape)
    unknown = [i for i, n in enumerate(shape) if n == -1]
    if len(unknown) > 1:
        raise ValueError("can only specify one unknown dimension (-1)")
    if any(n < -1 for n in shape):
        raise ValueError(f"negative dimensions are not allowed; got {shape}")
    if unknown:
        known = math.prod(n for n in shape if n != -1)
        if known == 0 or size % known:
            raise ValueError(f"cannot reshape a Tensor of size {size} into shape {shape}")
        shape = tuple(size // known if n == -1 else n for n in shape)
    if math.prod(shape) != size:
        raise ValueError(f"cannot reshape a Tensor of size {size} into shape {shape}")
    return shape


def _reshape_chunks(old_shape, new_shape):
    """The runs of a C-order reshape: ``[(old axes, new axes), ...]`` with the same size in
    each, over the axes of size other than 1, in order. Sizes are positive.
    """
    old = [a for a, n in enumerate(old_shape) if n != 1]
    new = [b for b, n in enumerate(new_shape) if n != 1]
    chunks = []
    i = j = 0
    while i < len(old):
        ia, jb = [old[i]], [new[j]]
        pa, pb = old_shape[old[i]], new_shape[new[j]]
        while pa != pb:
            if pa < pb:
                i += 1
                ia.append(old[i])
                pa *= old_shape[old[i]]
            else:
                j += 1
                jb.append(new[j])
                pb *= new_shape[new[j]]
        chunks.append((ia, jb))
        i += 1
        j += 1
    return chunks


def _reshape_layout(old_shape, layout, new_shape):
    """The layout of the reshape to ``new_shape`` that the stored Matrix of ``layout`` holds
    as it is: each run of old axes (``_reshape_chunks``) lies within one group, in order,
    and becomes the run's new axes there. None when some run does not. Sizes positive.
    """
    groups = []
    for group in (layout.rows, layout.cols):
        groups.append([a for a in group if old_shape[a] != 1])
    out = ([], [])
    for ia, jb in _reshape_chunks(old_shape, new_shape):
        for g, group in enumerate(groups):
            if ia[0] in group:
                k = group.index(ia[0])
                if group[k : k + len(ia)] != ia:
                    return None
                out[g].append((k, jb))
                break
    rows, cols = (tuple(b for _, jb in sorted(runs) for b in jb) for runs in out)
    units = [b for b, n in enumerate(new_shape) if n == 1]
    return _insert_unit_axes(Layout(rows, cols), units)


def _reshape_targets(old_shape, new_shape):
    """The layout states of ``old_shape`` whose stored Matrix holds the reshape: the runs as
    blocks, in any order, split between them with both groups nonempty (a group of all the
    axes stores the result as one row or column).
    """
    chunks = [ia for ia, _ in _reshape_chunks(old_shape, new_shape)]
    units = [a for a, n in enumerate(old_shape) if n == 1]
    orders = itertools.permutations(chunks) if len(chunks) <= 5 else [chunks]
    targets = set()
    for order in orders:
        for k in range(1, len(order)):
            rows = tuple(a for ia in order[:k] for a in ia)
            cols = tuple(a for ia in order[k:] for a in ia)
            state = _insert_unit_axes(Layout(rows, cols), units).state
            if tl._valid_split(old_shape, *state):
                targets.add(state)
    return frozenset(targets)


def _reshape(t, shape):
    """``t.reshape(shape)`` (``shape`` resolved): a view when the layout holds it; else the
    data moved once to a layout that does (the runs as blocks, split between two of them);
    else, when the reshape is one run, the data in C order (a relayout to the identity
    sequence) with the stored Matrix reshaped to the new shape's default split (DESIGN.md
    5.3).
    """
    if math.prod(shape) == 0:
        return Tensor(t.dtype, shape)
    if (layout := _reshape_layout(t._shape, t._layout, shape)) is not None:
        return t._view(shape, layout)
    if targets := _reshape_targets(t._shape, shape):
        M, src = tl.relayout(t._M, t._shape, t._layout, targets)
        return Tensor._from_M(M, shape, _reshape_layout(t._shape, src, shape))
    identity = tuple(range(t.ndim))
    targets = frozenset(
        (identity, p) for p in range(t.ndim + 1) if tl._valid_split(t._shape, identity, p)
    )
    M, _ = tl.relayout(t._M, t._shape, t._layout, targets)
    layout = tl.default_layout(shape, M._nvals)
    nrows, ncols = tl.prod(shape, layout.rows), tl.prod(shape, layout.cols)
    if M is not t._M and nrows <= 2**31 and ncols <= 2**31:
        M.ss.reshape(nrows, ncols, inplace=True)
    else:
        # (not the shared Matrix in place; SuiteSparse 10.5.1's in-place reshape cannot widen
        # 32-bit indices)
        M = M.ss.reshape(nrows, ncols)
    return Tensor._from_M(M, shape, layout)


def _reduce_aggregator(t, op, axis, keepdims):
    """``reduce`` with an aggregator: its monoid; or its semiring's multiply applied with the
    aggregator's initial value to every entry, then the semiring's monoid, then its finalize
    op, as Matrix aggregations do.
    """
    from .operator.agg import TypedAggregator

    agg = op.parent if isinstance(op, TypedAggregator) else op
    if agg._monoid is not None:
        return _reduce_expr(t, agg._monoid, axis, keepdims)
    if agg._semiring is None or agg._composite is not None or agg._custom is not None:
        raise NotImplementedError(
            f"Tensor.reduce supports aggregators made of a monoid or a semiring; not "
            f"agg.{agg.name}"
        )
    # The semiring typed as Matrix aggregations type it, by the values' type and the initial
    # value's (so integer powers of INT64 values stay INT64: logaddexp2 of -3 is log2(0)); its
    # result type is then the one Matrix gives (INT64 for count and exists)
    agg[t.dtype]  # (a KeyError for the types Matrix refuses, as sum_of_squares of BOOL)
    sr = get_typed_op(agg._semiring, t.dtype, agg._initdtype)
    if agg._switch:
        x = t.apply(sr.binaryop, left=agg._initval).new()
    else:
        x = t.apply(sr.binaryop, right=agg._initval).new()
    rv = _reduce_expr(x, sr.monoid, axis, keepdims)
    if agg._finalize is not None:
        rv = rv.new().apply(agg._finalize[sr.return_type])
    return rv


def _insert_unit_axes(layout, unit_axes):
    """Add size-1 axes to a layout. They do not change the stored Matrix, so they may go
    anywhere; put each before the next larger axis so an identity sequence stays one.
    """
    rows, cols = list(layout.rows), list(layout.cols)
    for u in sorted(unit_axes):
        for group in (rows, cols):
            pos = next((i for i, a in enumerate(group) if a > u), None)
            if pos is not None:
                group.insert(pos, u)
                break
        else:
            cols.append(u)
    return Layout(tuple(rows), tuple(cols))


def _nthreads(updater):
    nthreads = updater.opts.get("nthreads")
    return tl._nthreads() if nthreads is None else nthreads


def _is_scalar_like(x):
    return isinstance(x, (bool, int, float, complex, np.generic)) or output_type(x) is Scalar


def _is_tensor_like(x):
    return output_type(x) is Tensor


def _compute(x, within):
    """The Tensor of an expression operand (with autocompute), else a TypeError with a hint."""
    if type(x) is TensorExpression:
        if not config.get("autocompute"):
            raise TypeError(
                f"{within} needs a Tensor, not a TensorExpression; use `.new()` first.\n\n"
                "Hint: use `graphblas.config.set(autocompute=True)` to enable "
                "automatic computation of expressions."
            )
        return x._get_value()
    return x


def _as_tensor(x, within):
    """``x`` as a Tensor when it is Tensor-like: a Tensor, a Matrix (2-d) or Vector (1-d), an
    expression of one (computed with autocompute), or a numpy array (full). None otherwise.

    A Matrix held by row is used without a copy; a Vector, a Matrix held by column and an
    ndarray are copied once.
    """
    typ = output_type(x)
    if typ is Tensor:
        return _compute(x, within)
    if typ in {Matrix, Vector, TransposedMatrix}:
        if type(x) not in {Matrix, Vector, TransposedMatrix}:
            if not config.get("autocompute"):
                raise TypeError(
                    f"{within} needs a {typ.__name__}, not an expression; use `.new()` "
                    "first.\n\nHint: use `graphblas.config.set(autocompute=True)` to enable "
                    "automatic computation of expressions."
                )
            x = x.new()
        if type(x) is Vector:
            return Tensor.from_vector(x)
        return Tensor.from_matrix(x, copy=False)
    if typ is np.ndarray:
        return Tensor.from_dense(x)
    return None


def _mask_arg(x):
    """A bare BOOL Tensor used as a mask means its values (as for Matrix)."""
    x = _compute(x, "mask")
    if type(x) is Tensor:
        if x.dtype != BOOL:
            raise TypeError(
                f"Mask must be boolean objects (got {x.dtype}) "
                "or indicate values (T.V) or structure (T.S)"
            )
        return x.V
    return x


class _Owners:
    """The Tensors that share one stored Matrix. Held weakly, so no ``__del__`` bookkeeping
    is needed (review F2). Keyed by ``id`` because a WeakSet compares its members with
    ``==``, which will build an expression once Tensor has comparison operators.
    """

    __slots__ = "_refs", "__weakref__"

    def __init__(self, owner):
        self._refs = weakref.WeakValueDictionary({id(owner): owner})

    def add(self, obj):
        self._refs[id(obj)] = obj

    def discard(self, obj):
        if self._refs.get(id(obj)) is obj:
            del self._refs[id(obj)]

    def __contains__(self, obj):
        return self._refs.get(id(obj)) is obj

    def __len__(self):
        return len(self._refs)


class _Store:
    # `exported`: the Matrix was handed out (`to_matrix(copy=False)`, `ss.matrix`,
    # `from_matrix(copy=False)`), so the next in-place change must copy first.
    __slots__ = "owners", "exported", "__weakref__"

    def __init__(self, owner):
        self.owners = _Owners(owner)
        self.exported = False


class Tensor(BaseType):
    """An n-dimensional sparse array.

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.** See the user guide's "Tensors" page.

    Parameters
    ----------
    dtype :
        Data type of the elements.
    *shape : ints, or one tuple of ints
        ``Tensor(FP64, 3, 4, 5)`` or ``Tensor(FP64, (3, 4, 5))``.
    layout : Layout or (rows, cols), optional
        Which axes are raveled into the rows and the columns of the stored Matrix. A layout
        given here is kept by later ``<<`` updates (it is "pinned"); otherwise an update
        adopts whatever layout its result has.
    name : str, optional
    """

    __slots__ = "_M", "_shape", "_layout", "_store", "_pinned"
    _is_tensor = True
    _is_transposed = False
    _name_counter = itertools.count()

    def __new__(cls, dtype=FP64, *shape, layout=None, name=None):
        _check_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = _as_shape(shape)
        dtype = lookup_dtype(dtype)
        if layout is None:
            lay = tl.default_layout(shape)
        else:
            lay = tl.as_layout(layout, shape)
        M = tl.new_matrix(dtype, tl.prod(shape, lay.rows), tl.prod(shape, lay.cols))
        return cls._from_M(M, shape, lay, pinned=layout is not None, name=name)

    @classmethod
    def _from_M(cls, M, shape, layout, *, pinned=False, name=None, store=None):
        self = object.__new__(cls)
        self.dtype = M.dtype
        self.name = f"T_{next(Tensor._name_counter)}" if name is None else name
        self._M = M
        self._shape = shape
        self._layout = layout
        self._pinned = pinned
        if store is None:
            self._store = _Store(self)
        else:
            store.owners.add(self)
            self._store = store
        return self

    def _view(self, shape, layout):
        """A Tensor sharing this one's Matrix with another shape or layout, in O(1)."""
        return Tensor._from_M(self._M, shape, layout, pinned=self._pinned, store=self._store)

    # Copy-on-write. Every path that changes ``_M`` calls one of these first.
    def _shared(self):
        store = self._store
        return store.exported or len(store.owners) > 1

    def _own(self):
        """Before changing ``_M`` in place: copy it if anyone else holds it."""
        if self._shared():
            self._store.owners.discard(self)
            self._store = _Store(self)
            self._M = self._M.dup()

    def _set_M(self, M, layout):
        """Replace ``_M``; others that held the old one keep it unchanged."""
        if self._shared():
            self._store.owners.discard(self)
            self._store = _Store(self)
        self._M = M
        self._layout = layout

    def _share(self, other):
        if other._store is self._store:
            return
        self._store.owners.discard(self)
        self._store = other._store
        self._store.owners.add(self)
        self._M = other._M
        self._layout = other._layout

    def _check_invariants(self):
        """Raise AssertionError if the stored Matrix does not match the shape and layout."""
        M = self._M
        shape, (rows, cols) = self._shape, self._layout
        checks = [
            (sorted(rows + cols) == list(range(len(shape))), "layout is not a partition"),
            (M._nrows == tl.prod(shape, rows) == M.nrows, "nrows does not match the rows"),
            (M._ncols == tl.prod(shape, cols) == M.ncols, "ncols does not match the cols"),
            (tl.is_by_row(M), f"stored Matrix is not held by row ({M.ss.format})"),
            (M.dtype == self.dtype, f"dtype {M.dtype} != {self.dtype}"),
            (self in self._store.owners, "not an owner of its store"),
        ]
        for ok, msg in checks:
            if not ok:
                raise AssertionError(f"{msg}: shape {shape}, layout {self._layout}, {M.shape}")

    # Properties
    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        """Number of positions, ``prod(shape)`` (numpy's meaning); ``nvals`` counts entries."""
        return math.prod(self._shape)

    @property
    def nvals(self):
        return self._M.nvals

    @property
    def _nvals(self):
        # As Matrix and Vector have it: mask combination (mask.py) compares sizes with it
        return self._M._nvals

    @property
    def layout(self):
        return self._layout

    @property
    def layout_pinned(self):
        """Whether ``<<`` keeps this layout (it was set explicitly) or adopts the result's."""
        return self._pinned

    @property
    def ss(self):
        return _TensorSS(self)

    @property
    def S(self):
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    def __call__(self, *optional_mask_accum_replace, mask=None, accum=None, replace=False, **opts):
        from .tensor_index import TensorUpdater

        args = [_mask_arg(arg) for arg in optional_mask_accum_replace]
        if mask is not None:
            mask = _mask_arg(mask)
        u = super().__call__(*args, mask=mask, accum=accum, replace=replace, **opts)
        return TensorUpdater(self, opts=u.opts, **u.kwargs)

    def __getitem__(self, keys):
        """Extract with outer (orthogonal) indexing, as GraphBLAS extract does: ints drop
        their axis; slices, lists and int arrays keep it (``T[[0, 1], [0, 1]]`` is 2 x 2,
        not numpy's two elements). ``...`` fills in full slices. Returns a TensorExpression
        (an all-int key gives a 0-d Tensor), which also receives an assignment:
        ``T[idx] << value``, ``T[idx](mask, accum, replace) << value`` (the mask has the
        region's shape: GxB_subassign).
        """
        return _extract_expr(self, keys)

    def __setitem__(self, keys, value):
        """``T[idx] = value``: assign as Matrix does (GraphBLAS assign), the value broadcast
        to the region's shape (numpy rules); the region takes the value's structure, and a
        scalar fills it. ``T(mask, accum, replace)[idx] = value`` masks over the whole Tensor
        (GrB_assign).
        """
        self()._assign(keys, value, is_submask=False)

    def __delitem__(self, keys):
        """``del T[idx]``: delete the region's entries."""
        del self()[keys]

    @classmethod
    def from_matrices(cls, mats, axis=0, *, name=None):
        """Stack equally shaped Matrices along a new ``axis`` (0, 1 or 2) into a 3-d Tensor.

        The data is one vertical concatenation (``ss.concat``); ``axis`` only chooses which
        logical axis the stacking is.
        """
        _check_backend()
        mats = [
            m.new() if type(m) is TransposedMatrix else _compute(m, "from_matrices") for m in mats
        ]
        mats = [m.to_matrix() if type(m) is Tensor else m for m in mats]
        if not mats or any(type(m) is not Matrix for m in mats):
            raise TypeError("from_matrices needs a non-empty sequence of Matrices")
        nrows, ncols = mats[0].shape
        if any(m.shape != (nrows, ncols) for m in mats):
            raise DimensionMismatch("from_matrices: the Matrices must have the same shape")
        axis = _normalize_axis(axis, 3)
        dtype = mats[0].dtype
        for m in mats[1:]:
            dtype = unify(dtype, m.dtype)
        M = tl.new_matrix(dtype, len(mats) * nrows, ncols)
        M.ss.concat([[m] for m in mats])
        r_ax, c_ax = (a for a in range(3) if a != axis)
        shape = [0, 0, 0]
        shape[axis], shape[r_ax], shape[c_ax] = len(mats), nrows, ncols
        return cls._from_M(M, tuple(shape), Layout((axis, r_ax), (c_ax,)), name=name)

    def to_matrices(self, axis=0):
        """The Matrices of a 3-d Tensor along ``axis`` (the other two axes in order)."""
        if self.ndim != 3:
            raise ValueError(f"to_matrices needs a 3-d Tensor; this one has shape {self._shape}")
        axis = _normalize_axis(axis, 3)
        r_ax, c_ax = (a for a in range(3) if a != axis)
        target = tl.as_layout(((axis, r_ax), (c_ax,)), self._shape)
        M, _ = tl.relayout(self._M, self._shape, self._layout, target)
        n, nrows = self._shape[axis], self._shape[r_ax]
        if n == 0:
            return []
        if nrows == 0 or M._ncols == 0:
            return [Matrix(self.dtype, nrows, M._ncols) for _ in range(n)]
        return [row[0] for row in M.ss.split((nrows, M._ncols))]

    def __repr__(self, mask=None):
        return _format_tensor(self, mask)

    # Construction and conversion
    @classmethod
    def from_coo(
        cls, indices, values=1.0, dtype=None, *, shape=None, layout=None, dup_op=None, name=None
    ):
        """Create a Tensor from coordinates and values.

        Parameters
        ----------
        indices : sequence of 1-d integer arrays, or an (ndim, nvals) integer array
            ``indices[a]`` holds the coordinates along axis ``a`` (the ``np.nonzero`` form).
        values : array or scalar, default 1.0
            A scalar stores one value for every entry (iso).
        dtype : optional
            Inferred from ``values`` if not given.
        shape : tuple of ints, optional
            Inferred from the largest index along each axis if not given.
        layout : Layout or (rows, cols), optional
            The stored layout (pinned). The default is the identity axis order with the
            split chosen for about four entries per nonempty row (DESIGN.md 7.6).
        dup_op : BinaryOp, optional
            Combines values at repeated coordinates; repeated coordinates raise without it.
        name : str, optional
        """
        _check_backend()
        if isinstance(indices, np.ndarray):
            if indices.ndim != 2:
                raise ValueError(
                    f"indices must be a sequence of 1-d arrays or an (ndim, nvals) array; "
                    f"got an array with {indices.ndim} dimensions"
                )
            coords = list(indices)
        else:
            coords = [np.asarray(idx) for idx in indices]
        ndim = len(coords)
        for idx in coords:
            if idx.ndim != 1:
                raise ValueError("each array in indices must be 1-d")
            if idx.size and not np.issubdtype(idx.dtype, np.integer):
                raise TypeError(f"indices must be integers; got dtype {idx.dtype}")
        coords = [idx.astype(np.int64, copy=False) for idx in coords]
        if ndim == 0:
            n = 1 if np.ndim(values) == 0 else len(values)
        else:
            n = coords[0].size
            if any(idx.size != n for idx in coords):
                raise ValueError("all arrays in indices must have the same length")
        if shape is None:
            if n == 0 and ndim > 0:
                raise ValueError("No indices provided. Unable to infer shape.")
            shape = tuple(int(idx.max()) + 1 for idx in coords)
        shape = _as_shape(shape)
        if len(shape) != ndim:
            raise ValueError(f"shape {shape} has {len(shape)} axes; indices have {ndim}")
        for axis, idx in enumerate(coords):
            # A per-axis check: an index past its axis would silently land in the next row
            # once raveled.
            if n and (idx.min() < 0 or idx.max() >= shape[axis]):
                raise IndexOutOfBound(
                    f"index out of bounds along axis {axis} of size {shape[axis]}: "
                    f"min {idx.min()}, max {idx.max()}"
                )
        # Reshape works only below a total size of 2**60; past it keep the shape-based split
        rows = settle = None
        if layout is None and ndim >= 3 and n > 0:
            lay, rows, settle = _data_split_layout(coords, shape, n)
        elif layout is None:
            lay = tl.default_layout(shape, n)
        else:
            lay = tl.as_layout(layout, shape)
        if rows is None:
            rows = tl.ravel(coords, shape, lay.rows, n=n)
        cols = tl.ravel(coords, shape, lay.cols, n=n)
        M = tl.from_coo(
            rows,
            cols,
            values,
            dtype,
            tl.prod(shape, lay.rows),
            tl.prod(shape, lay.cols),
            dup_op=dup_op,
        )
        if settle:
            M, lay = _settle_split(M, shape, lay)
        return cls._from_M(M, shape, lay, pinned=layout is not None, name=name)

    @classmethod
    def from_matrix(cls, A, *, copy=True, name=None):
        """A 2-d Tensor from a Matrix (or ``A.T``).

        With ``copy=False`` the Tensor uses A's data without copying when A is held by row
        (use it read-only: changing A changes the Tensor). A Matrix held by column is copied
        and stored as its transpose held by row, which costs no transpose.
        """
        _check_backend()
        if type(A) is TransposedMatrix:
            T = cls.from_matrix(A._matrix, copy=copy)
            return T._view((A._nrows, A._ncols), _renumber(T._layout, {0: 1, 1: 0}))._named(name)
        if type(A) is not Matrix:
            raise TypeError(f"from_matrix needs a Matrix; got {type(A)}")
        shape = (A._nrows, A._ncols)
        if tl.is_by_row(A):
            M = A.dup() if copy else A
            T = cls._from_M(M, shape, Layout((0,), (1,)), name=name)
            T._store.exported = M is A
            return T
        M = tl.flip_orientation(A.dup())
        return cls._from_M(M, shape, Layout((1,), (0,)), name=name)

    @classmethod
    def from_vector(cls, v, *, name=None):
        """A 1-d Tensor from a Vector (a copy, stored as one row like the Vector itself)."""
        _check_backend()
        if type(v) is not Vector:
            raise TypeError(f"from_vector needs a Vector; got {type(v)}")
        M = tl.steal_vector_as_row(v.dup())
        return cls._from_M(M, (v._size,), Layout((), (0,)), name=name)

    @classmethod
    def from_dense(cls, values, missing_value=None, *, dtype=None, layout=None, name=None):
        """A Tensor from a numpy array (or array-like): every position is present, except
        those equal to ``missing_value`` when it is given (as ``Matrix.from_dense``).
        """
        _check_backend()
        values = np.asarray(values)
        shape = values.shape
        if layout is None:
            lay = tl.default_layout(shape, values.size)
        else:
            lay = tl.as_layout(layout, shape)
        nrows, ncols = tl.prod(shape, lay.rows), tl.prod(shape, lay.cols)
        arr = np.transpose(values, lay.rows + lay.cols).reshape(nrows, ncols)
        M = Matrix.from_dense(arr, missing_value, dtype=dtype)
        if not tl.is_by_row(M):
            # A by-column global format, or an n x 1 array (SuiteSparse keeps those by column)
            M.ss.config["format"] = "by_row"
        return cls._from_M(M, shape, lay, pinned=layout is not None, name=name)

    def to_dense(self, fill_value=None, dtype=None):
        """A numpy array; raises if an entry is missing and ``fill_value`` is not given."""
        M, lay = self._M, self._layout
        identity = tuple(range(self.ndim))
        if lay.seq != identity:
            targets = frozenset(
                (identity, p)
                for p in range(self.ndim + 1)
                if tl._valid_split(self._shape, identity, p)
            )
            M, lay = tl.relayout(M, self._shape, lay, targets)
        return M.to_dense(fill_value, dtype).reshape(self._shape)

    def _named(self, name):
        if name is not None:
            self.name = name
        return self

    def to_coo(self, dtype=None, *, indices=True, values=True, sort=True):
        """Return ``(indices, values)``: a tuple of int64 index arrays (one per axis) and the
        values.

        With ``sort=True`` the entries come in logical C order. A Tensor whose layout is not
        the identity axis order is relaid out for that first, which costs a relayout of the
        whole Tensor; ``sort=False`` returns the stored order without moving data.
        """
        M, lay = self._M, self._layout
        identity = tuple(range(self.ndim))
        if sort and lay.seq != identity:
            targets = frozenset(
                (identity, p)
                for p in range(self.ndim + 1)
                if tl._valid_split(self._shape, identity, p)
            )
            M, lay = tl.relayout(M, self._shape, lay, targets)
        r, c, v = M.to_coo(dtype, rows=indices, columns=indices, values=values, sort=sort)
        if not indices:
            return None, v
        coords = [None] * self.ndim
        tl.unravel(r.view(np.int64), self._shape, lay.rows, coords)
        tl.unravel(c.view(np.int64), self._shape, lay.cols, coords)
        return tuple(coords), v

    @classmethod
    def from_pydata_sparse(cls, s, *, dup_op=None, layout=None, name=None):
        """Create a Tensor from a pydata/sparse array (COO, GCXS or DOK) of any rank.

        Explicit zeros stay entries. The array's fill value must be 0: a Tensor has no fill
        value. Values are copied once; a COO's coordinates are raveled into the Matrix's row
        and column indices. A GCXS array (ndim >= 2) imports its arrays as they are, one copy
        each, and the Tensor takes its layout: the compressed axes as rows, the others as
        columns.

        Parameters
        ----------
        s : sparse.SparseArray
            See https://sparse.pydata.org
        dup_op : BinaryOp, optional
            Combines values at repeated coordinates (pydata's COO has none).
        layout : Layout or (rows, cols), optional
            The stored layout (pinned), as in ``from_coo``.
        name : str, optional
        """
        _check_backend()
        from .tensor_io import from_pydata_sparse

        return from_pydata_sparse(cls, s, dup_op=dup_op, layout=layout, name=name)

    def to_pydata_sparse(self, format="coo", *, compressed_axes=None):
        """Create a pydata/sparse array with fill value 0.

        Parameters
        ----------
        format : {"coo", "gcxs", "dok"}, default "coo"
            "coo" is in C order, so a Tensor whose layout is not the identity axis order is
            relaid out first. "gcxs" uses the Tensor's layout as it is when GCXS can hold it
            (both groups nonempty and in increasing order), else the cheapest such layout.
        compressed_axes : sequence of ints, optional
            The axes GCXS compresses (the rows of a layout), for "gcxs".
        """
        from .tensor_io import to_pydata_sparse

        return to_pydata_sparse(self, format, compressed_axes=compressed_axes)

    @classmethod
    def from_scipy_sparse(cls, A, *, dup_op=None, layout=None, name=None):
        """Create a Tensor from a scipy.sparse array: an n-d ``coo_array``, or any 2-d format.

        Explicit zeros stay entries; repeated coordinates add up as in scipy unless
        ``dup_op`` says otherwise. A COO array's coordinates are raveled into the Matrix's row
        and column indices and its values copied once. A canonical 2-d CSR or CSC array
        imports its arrays (one copy each) into layout ``(0|1)`` or ``(1|0)``.

        Parameters
        ----------
        A : scipy.sparse array or matrix
        dup_op : BinaryOp, optional
            Combines values at repeated coordinates; default ``plus`` (``lor`` for BOOL).
        layout : Layout or (rows, cols), optional
            The stored layout (pinned), as in ``from_coo``.
        name : str, optional
        """
        _check_backend()
        from .tensor_io import from_scipy_sparse

        return from_scipy_sparse(cls, A, dup_op=dup_op, layout=layout, name=name)

    def to_scipy_sparse(self, format="coo"):
        """Create a scipy.sparse array (``coo_array`` for any rank but 0).

        A COO array comes in the stored order with no relayout; it is in C order (marked
        canonical) when the layout is the identity axis order. A 2-d Tensor may also give
        the other scipy formats, as ``graphblas.io.to_scipy_sparse``: "csr" from layout
        ``(0|1)`` and "csc" from ``(1|0)`` take the stored arrays (one copy).
        """
        from .tensor_io import to_scipy_sparse

        return to_scipy_sparse(self, format)

    def to_matrix(self, rows=None, *, copy=True):
        """A Matrix with the axes ``rows`` (in that order) raveled into the rows and the other
        axes (in increasing order) into the columns, both in C order.

        ``rows=None`` gives the stored grouping for ndim != 2 (no data moves) and the plain
        Matrix (``rows=(0,)``) for a 2-d Tensor. With ``copy=False`` the result may be the
        stored Matrix itself; use it read-only.
        """
        if rows is None and self.ndim != 2:
            M = self._M
        else:
            if rows is None:
                rows = (0,)
            rows = tuple(_normalize_axis(a, self.ndim) for a in rows)
            cols = tuple(a for a in range(self.ndim) if a not in rows)
            target = tl.as_layout((rows, cols), self._shape)
            M, _ = tl.relayout(self._M, self._shape, self._layout, target)
        if M is self._M:
            if copy:
                return M.dup()
            self._store.exported = True
        return M

    def to_vector(self):
        """The Vector of a 1-d Tensor."""
        if self.ndim != 1:
            raise ValueError(f"to_vector needs a 1-d Tensor; this one has shape {self._shape}")
        if self._layout.rows:
            return self._M[:, 0].new()
        return self._M[0, :].new()

    def to_scalar(self):
        """The Scalar of a 0-d Tensor."""
        if self.ndim != 0:
            raise ValueError(f"to_scalar needs a 0-d Tensor; this one has shape {self._shape}")
        return self._M[0, 0].new()

    def dup(self, dtype=None, *, layout=None, clear=False, name=None):
        """A copy (sharing data copy-on-write when the layout and dtype stay the same)."""
        dtype = self.dtype if dtype is None else lookup_dtype(dtype)
        if clear:
            return Tensor(dtype, self._shape, layout=layout, name=name)
        M, lay = self._M, self._layout
        if layout is not None:
            M, lay = tl.relayout(M, self._shape, lay, tl.as_layout(layout, self._shape))
        if dtype != self.dtype:
            M = M.dup(dtype)
        if M is self._M:
            T = Tensor._from_M(M, self._shape, lay, name=name, store=self._store)
        else:
            T = Tensor._from_M(M, self._shape, lay, name=name)
        T._pinned = self._pinned if layout is None else True
        return T

    def relayout(self, layout, *, inplace=False):
        """Store the data in ``layout``; the layout is then pinned (kept by ``<<``)."""
        target = tl.as_layout(layout, self._shape)
        M, lay = tl.relayout(self._M, self._shape, self._layout, target)
        if inplace:
            if M is not self._M:
                self._set_M(M, lay)
            self._pinned = True
            return None
        if M is self._M:
            T = self._view(self._shape, lay)
        else:
            T = Tensor._from_M(M, self._shape, lay)
        T._pinned = True
        return T

    def clear(self):
        """Remove all entries; the shape and layout stay."""
        rows, cols = self._layout
        self._set_M(
            tl.new_matrix(self.dtype, tl.prod(self._shape, rows), tl.prod(self._shape, cols)),
            self._layout,
        )

    def wait(self, how="materialize"):
        self._M.wait(how)

    def _comparand(self, other, within):
        """``other`` for ``isequal`` / ``isclose``: a Tensor, or a Matrix, Vector or Scalar as
        the Tensor of its shape (2-d, 1-d, 0-d).
        """
        if type(other) is Scalar:
            if other._is_empty:
                return Tensor(other.dtype, ())
            return Tensor.from_coo((), [other.value], other.dtype, shape=())
        if output_type(other) in {Matrix, Vector, TransposedMatrix}:
            return _as_tensor(other, within)
        return self._expect_type(other, Tensor, within=within, argname="other")

    def isequal(self, other, *, check_dtype=False):
        """Whether ``other`` has the same shape, the same entries and equal values. ``other``
        is a Tensor, or a Matrix, Vector or Scalar (compared as a 2-, 1- or 0-d Tensor).
        """
        other = self._comparand(other, "isequal")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._shape != other._shape or self._M.nvals != other._M.nvals:
            return False
        A, B = _same_layout(self, other)
        return A.isequal(B, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """Whether ``other`` has the same shape, the same entries and close values (a Tensor,
        Matrix, Vector or Scalar, as for ``isequal``).
        """
        other = self._comparand(other, "isclose")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._shape != other._shape or self._M.nvals != other._M.nvals:
            return False
        A, B = _same_layout(self, other)
        return A.isclose(B, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)

    # Shape operations: metadata only, sharing the stored Matrix (copy-on-write)
    def permute_dims(self, axes):
        """Reorder the axes: axis ``i`` of the result is axis ``axes[i]`` of this Tensor."""
        axes = tuple(_normalize_axis(a, self.ndim) for a in axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(f"axes {axes} must be a permutation of range({self.ndim})")
        new = {old: i for i, old in enumerate(axes)}
        shape = tuple(self._shape[a] for a in axes)
        return self._view(shape, _renumber(self._layout, new))

    @property
    def T(self):
        """All axes reversed (numpy's ``.T``); a Tensor sharing this one's data."""
        return self.permute_dims(tuple(range(self.ndim - 1, -1, -1)))

    @property
    def mT(self):
        """The last two axes swapped (the Array API's matrix transpose)."""
        if self.ndim < 2:
            raise ValueError("mT needs at least 2 dimensions")
        axes = list(range(self.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return self.permute_dims(axes)

    def expand_dims(self, axis=0):
        """Insert a size-1 axis at position ``axis`` (an int or a tuple of ints)."""
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        ndim = self.ndim + len(axes)
        axes = sorted(_normalize_axis(a, ndim) for a in axes)
        if len(set(axes)) != len(axes):
            raise ValueError(f"repeated axis in {axis}")
        old = [a for a in range(ndim) if a not in axes]
        new = dict(zip(range(self.ndim), old, strict=True))
        shape = [1] * ndim
        for i, a in enumerate(old):
            shape[a] = self._shape[i]
        layout = _insert_unit_axes(_renumber(self._layout, new), axes)
        return self._view(tuple(shape), layout)

    def squeeze(self, axis=None):
        """Remove size-1 axes (all of them, or the ones in ``axis``)."""
        if axis is None:
            axes = [a for a, n in enumerate(self._shape) if n == 1]
        else:
            axes = _normalize_axes(axis, self.ndim)
            for a in axes:
                if self._shape[a] != 1:
                    raise ValueError(f"cannot squeeze axis {a} of size {self._shape[a]}")
        keep = [a for a in range(self.ndim) if a not in axes]
        new = {a: i for i, a in enumerate(keep)}
        rows = tuple(new[a] for a in self._layout.rows if a in new)
        cols = tuple(new[a] for a in self._layout.cols if a in new)
        return self._view(tuple(self._shape[a] for a in keep), Layout(rows, cols))

    def reshape(self, *shape, copy=None):
        """The same entries in a new shape, in C order (numpy's and the Array API's
        ``reshape``): ``T.reshape(2, -1)`` or ``T.reshape((2, -1))``; one size may be -1.

        A reshape splits and fuses runs of axes. When each such run lies within one group of
        the layout, in order, the stored Matrix holds the result as it is: the result is a
        view in O(1), sharing the data copy-on-write as ``permute_dims`` does (size-1 axes
        never matter). Otherwise the data moves once, by a planned relayout to a layout
        where that holds, and the result has data of its own.

        Parameters
        ----------
        shape : ints, or a tuple of ints
        copy : bool, optional
            True: the result has data of its own, copied now. False: raise ValueError where
            the data would have to move. None (default): a view where possible. (A view
            behaves as a copy either way: changing it, or this Tensor, copies the data
            first.)
        """
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
            shape = shape[0]
        shape = _reshape_shape(shape, self._shape)
        if (
            copy is False
            and math.prod(shape) > 0
            and not _reshape_layout(self._shape, self._layout, shape)
        ):
            raise ValueError(
                f"reshape from {self._shape} to {shape} with copy=False: the layout "
                f"{self._layout._short()} does not hold the result, so the data would move"
            )
        T = _reshape(self, shape)
        if copy and T._M is self._M:
            # (a copy of the stored Matrix: a view already behaves as one, copy-on-write)
            T = Tensor._from_M(T._M.dup(), T._shape, T._layout, pinned=T._pinned)
        return T

    def take(self, indices, axis=None, *, mode="raise", wraparound=True):
        """Numpy's ``take``: the entries at ``indices`` (any shape) along ``axis``, or of the
        flattened Tensor (C order) when ``axis`` is None; the result's shape puts
        ``indices.shape`` in place of the axis. A position that lands on a missing entry is
        missing. Negative indices count from the end once (``wraparound=False``: they raise
        IndexError). ``mode`` is numpy's: out of range raises IndexError (``"raise"``), wraps
        (``"wrap"``) or clips (``"clip"``, where a negative index clips to 0).
        """
        from .tensor_index import take

        return take(self, indices, axis, mode=mode, wraparound=wraparound)

    def take_along_axis(self, indices, axis=-1, *, wraparound=True):
        """Numpy's ``take_along_axis``: ``indices`` has the Tensor's ndim and broadcasts
        against it except along ``axis``; ``out[..., i, ...] = T[..., indices[..., i, ...],
        ...]``, missing where the Tensor is. Negatives as for ``take``.
        """
        from .tensor_index import take_along_axis

        return take_along_axis(self, indices, axis, wraparound=wraparound)

    @property
    def vindex(self):
        """Vectorized (pointwise) integer indexing, as numpy's advanced indexing (dask's and
        zarr's ``vindex``): ``T.vindex[i0, i1, ...]`` with one int, Scalar or integer array per
        axis, broadcast together; the result has their broadcast shape and is missing where
        the Tensor is. ``T[...]`` stays outer indexing. ``T.vindex(wraparound=False)[...]``
        refuses negative indices.
        """
        from .tensor_index import VIndexer

        return VIndexer(self)

    def ravel(self):
        """The 1-d Tensor of all positions in C order: ``reshape(-1)``, a view where
        possible (numpy's ``ravel``).
        """
        return self.reshape(-1)

    def flatten(self):
        """The 1-d Tensor of all positions in C order, with data of its own:
        ``reshape(-1, copy=True)`` (numpy's ``flatten``).
        """
        return self.reshape(-1, copy=True)

    # Element-wise operations
    def ewise_add(self, other, op=monoid.plus, *, allow_expansion=False):
        """Element-wise ``op`` on the union of entries.

        Shapes broadcast as in numpy. A position present in only one operand takes its value,
        so a broadcast operand is replicated over the broadcast axes; that raises when it would
        replicate more than ``graphblas.config["tensor_max_expansion"]`` entries unless
        ``allow_expansion=True``.
        """
        return _ewise_expr(self, "ewise_add", other, op, allow_expansion=allow_expansion)

    def ewise_mult(self, other, op=binary.times):
        """Element-wise ``op`` on the intersection of entries (numpy broadcasting).

        When one operand covers the whole result shape the other is applied as a diagonal
        scaling, without replicating it.
        """
        return _ewise_expr(self, "ewise_mult", other, op)

    def ewise_union(self, other, op, left_default, right_default, *, allow_expansion=False):
        """Element-wise ``op`` on the union, with defaults for a missing side (numpy
        broadcasting, replicated as for ``ewise_add``).
        """
        return _ewise_expr(
            self,
            "ewise_union",
            other,
            op,
            left_default,
            right_default,
            allow_expansion=allow_expansion,
        )

    # Hooks for `op(X | Y)`, `op(X & Y)` and `semiring(X @ Y)` (operator._call_op)
    def _ewise_add(self, other, op=monoid.plus, is_infix=False):
        return self.ewise_add(other, op)

    def _ewise_mult(self, other, op=binary.times, is_infix=False):
        return self.ewise_mult(other, op)

    def _ewise_union(self, other, op, left_default, right_default, is_infix=False):
        return self.ewise_union(other, op, left_default, right_default)

    def select(self, op, thunk=None):
        """Keep the entries whose values pass ``op`` (a value selector such as
        ``select.valuegt``); index-based selectors are not supported yet.
        """
        if not getattr(op, "name", "").startswith("value"):
            raise NotImplementedError(
                "Tensor.select supports value selectors (select.value*) only; index-based "
                "selectors would need logical coordinates"
            )
        probe = self._M.select(op, thunk)
        return TensorExpression(
            "select",
            _select_recipe,
            [self],
            (self, probe.op, thunk),
            dtype=probe.dtype,
            shape=self._shape,
            op=probe.op,
            expr_repr="{0.name}.select({op})",
        )

    def apply(self, op, right=None, *, left=None):
        """Apply ``op`` to every stored value (with a scalar ``left`` or ``right`` for a
        BinaryOp).
        """
        probe = self._M.apply(op, right=right, left=left)
        return TensorExpression(
            "apply",
            _apply_recipe,
            [self],
            (self, probe.op, right, left),
            dtype=probe.dtype,
            shape=self._shape,
            op=probe.op,
            expr_repr="{0.name}.apply({op})",
        )

    # Reductions
    def reduce(self, op=monoid.plus, axis=None, *, keepdims=False):
        """Reduce over ``axis`` (None: all axes, giving a 0-d Tensor) with the monoid ``op``,
        or an aggregator made of one (``agg.sum``, ``agg.max``, ...) or of a semiring
        (``agg.count``, ``agg.count_nonzero``, ``agg.sum_of_squares``, ``agg.hypot``, ...).
        Composite aggregators (``agg.mean``, ``agg.varp``, ...) and ones that need positions
        (``agg.ss.argmin``, ``agg.ss.first``, ...) are not supported.

        A position whose fiber has no entries stays missing (it is not 0).
        """
        from .operator.agg import Aggregator, TypedAggregator

        if isinstance(op, (Aggregator, TypedAggregator)):
            return _reduce_aggregator(self, op, axis, keepdims)
        return _reduce_expr(self, op, axis, keepdims)

    # Contractions
    def matmul(self, other, op=semiring.plus_times):
        """Matrix product with numpy's ``matmul`` rules (also ``X @ Y``).

        The last two axes multiply as matrices; leading (batch) axes broadcast. A 1-d operand
        is promoted to a matrix (a row on the left, a column on the right) and the added axis
        is removed from the result. Batch axes that both operands have are done in one
        product over a block-diagonal embedding.
        """
        return _matmul_expr(self, other, op)

    def _matmul(self, other, op=semiring.plus_times, is_infix=False):
        return _matmul_expr(self, other, op)

    def tensordot(self, other, axes=2, op=semiring.plus_times):
        """Contract ``axes`` of this Tensor with ``axes`` of ``other`` (numpy.tensordot).

        ``axes`` is an int N (the last N axes of this Tensor with the first N of ``other``)
        or a pair of axis sequences. The result has this Tensor's free axes, then those of
        ``other``. ``op`` is the semiring; a position is present when at least one pair of
        entries meets there. ``other`` may be a Tensor, Matrix, Vector or ndarray.

        A vector contracted over an axis inside one group of the stored layout (TTV) scales
        by a diagonal and reduces with a selector, without moving the data (DESIGN.md 3.3).
        """
        return _tensordot_expr(self, other, axes, op)

    def mttkrp(self, factors, mode, op=semiring.plus_times):
        """Matricized tensor times Khatri-Rao product for ``mode``: an (I_mode x R) Tensor.

        ``factors`` holds one (I_n x R) factor per axis (Matrix or ndarray; the entry for
        ``mode`` is ignored). For 3-d mode 0 this is ``einsum('ijk,jr,kr->ir', X, B, C)``,
        computed as one TTM and then one product per remaining axis against an (I_n R x R)
        embedding of its factor, never materializing the Khatri-Rao product (DESIGN.md 3.7).
        With Numba installed, a 3-d FP64 tensor with full FP64 factors and ``plus_times``
        takes a kernel over the stored arrays instead, in the layout it is stored in.
        """
        return _mttkrp_expr(self, factors, mode, op)

    def unfold(self, mode, order="C"):
        """The mode-``mode`` unfolding as a Matrix of shape (I_mode, prod(other sizes)).

        ``order="C"`` (numpy and tensorly): the other axes in increasing order, the last
        varying fastest, as ``np.moveaxis(X, mode, 0).reshape(I_mode, -1)``. ``order="F"``
        (Kolda and Bader): the first of the other axes varies fastest.
        """
        mode = _normalize_axis(mode, self.ndim)
        others = tuple(a for a in range(self.ndim) if a != mode)
        if order in {"F", "f", "fortran"}:
            others = others[::-1]
        elif order not in {"C", "c"}:
            raise ValueError(f"order must be 'C' or 'F', not {order!r}")
        target = tl.as_layout(((mode,), others), self._shape)
        M, _ = tl.relayout(self._M, self._shape, self._layout, target)
        return M.dup() if M is self._M else M

    def reduce_scalar(self, op=monoid.plus, *, allow_empty=True):
        """Reduce all entries to a Scalar."""
        probe = self._M.reduce_scalar(op, allow_empty=allow_empty)
        return ScalarExpression(
            "reduce_scalar",
            None,
            [self, _reduce_scalar_recipe, (self, probe.op, allow_empty)],
            op=probe.op,
            dtype=probe.dtype,
            expr_repr="{0.name}.reduce_scalar({op})",
            is_cscalar=not allow_empty,
        )

    def sum(self, axis=None, *, keepdims=False):
        return _reduce_expr(self, monoid.plus, axis, keepdims, name="sum")

    def prod(self, axis=None, *, keepdims=False):
        return _reduce_expr(self, monoid.times, axis, keepdims, name="prod")

    def min(self, axis=None, *, keepdims=False):
        return _reduce_expr(self, monoid.min, axis, keepdims, name="min")

    def max(self, axis=None, *, keepdims=False):
        return _reduce_expr(self, monoid.max, axis, keepdims, name="max")

    def any(self, axis=None, *, keepdims=False):
        """Whether any stored value along ``axis`` is true (nonzero); missing stays missing."""
        return _reduce_expr(self, monoid.lor[BOOL], axis, keepdims, name="any", cast=BOOL)

    def all(self, axis=None, *, keepdims=False):
        """Whether every stored value along ``axis`` is true (nonzero)."""
        return _reduce_expr(self, monoid.land[BOOL], axis, keepdims, name="all", cast=BOOL)

    # Updates
    def _update(self, expr, mask=None, accum=None, replace=False, input_mask=None, *, opts):
        if input_mask is not None:
            raise TypeError("input_mask is not supported for Tensor")
        updater = Updater(self, mask=mask, accum=accum, replace=replace, opts=opts)
        if type(expr) is TensorExpression:
            expr.args[-2](updater, *expr.args[-1])
        elif type(expr) is Tensor:
            _store(updater, expr._M, expr._shape, expr._layout, share=expr)
        elif _is_scalar_like(expr):
            _fill(updater, expr)
        else:
            hint = ""
            if output_type(expr) in {Matrix, Vector, TransposedMatrix}:
                hint = " Use Tensor.from_matrix or Tensor.from_vector first."
            raise TypeError(
                f"Assignment value must be a Tensor, TensorExpression or scalar, not "
                f"{type(expr)}.{hint}"
            )


def _ewise_infix(left, right, op, kind):
    ltensor = _as_tensor(left, "infix operator")
    rtensor = _as_tensor(right, "infix operator")
    if ltensor is not None and rtensor is not None:
        if kind == "add":
            return ltensor.ewise_add(rtensor, op)
        if kind == "union":
            return ltensor.ewise_union(rtensor, op, False, False)
        return ltensor.ewise_mult(rtensor, op)
    if ltensor is not None and _is_scalar_like(right):
        return ltensor.apply(op, right=right)
    if _is_scalar_like(left) and rtensor is not None:
        return rtensor.apply(op, left=left)
    return NotImplemented


def _binary_op(name):
    # Looked up when used: without Numba some (floordiv) are UDFs and cannot be compiled
    if name.startswith("numpy."):
        return getattr(binary.numpy, name[6:])
    return getattr(binary, name)


def _make_infix(name, kind):
    def forward(self, other):
        return _ewise_infix(self, other, _binary_op(name), kind)

    def reflected(self, other):
        return _ewise_infix(other, self, _binary_op(name), kind)

    def inplace(self, other):
        if type(self) is not Tensor:
            raise TypeError(f"in-place operators are not supported for {type(self).__name__}")
        expr = _ewise_infix(self, other, _binary_op(name), kind)
        if expr is NotImplemented:
            raise TypeError(f"unsupported operand type for in-place op: {type(other)}")
        self << expr
        return self

    return forward, reflected, inplace


# As for Matrix: `+` is the union, `-` the union with 0 for a missing side, and the others
# (comparisons included) the intersection.
_INFIX = {
    "add": ("plus", "add"),
    "sub": ("minus", "union"),
    "mul": ("times", "mult"),
    "truediv": ("truediv", "mult"),
    "floordiv": ("floordiv", "mult"),
    "mod": ("numpy.mod", "mult"),
    "pow": ("pow", "mult"),
}
_COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")


class TensorExpression(BaseExpression):
    """A lazy Tensor result. Compute it with ``.new()`` or ``T(...) << expr``."""

    __slots__ = ("_shape",)
    output_type = Tensor
    _is_tensor = True

    def __init__(self, method_name, func, expr_args, func_args, *, dtype, shape, op, expr_repr):
        super().__init__(
            method_name,
            None,
            [*expr_args, func, func_args],
            op=op,
            dtype=dtype,
            expr_repr=expr_repr,
        )
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return math.prod(self._shape)

    def construct_output(self, dtype=None, *, layout=None, name=None):
        return Tensor(self.dtype if dtype is None else dtype, self._shape, layout=layout, name=name)

    def new(self, dtype=None, *, mask=None, layout=None, name=None, **opts):
        """Compute into a new Tensor (in ``layout`` if given, which is then pinned)."""
        if (
            mask is None
            and layout is None
            and self._value is not None
            and (dtype is None or self._value.dtype == dtype)
        ):
            rv = self._value
            if name is not None:
                rv.name = name
            self._value = None
            return rv
        output = self.construct_output(dtype, layout=layout, name=name)
        if mask is None:
            updater = output(**opts)
        else:
            # The output starts empty, so masking with replace gives the same result and
            # lets the output take the result's layout.
            updater = output(mask=mask, replace=True, **opts)
        self.args[-2](updater, *self.args[-1])
        return output

    dup = new

    def __repr__(self):
        return (
            f"gb.TensorExpression  shape {self._shape}  dtype {self.dtype}\n"
            f"{self._format_expr()}"
        )

    _get_value = automethods._get_value

    @property
    def nvals(self):
        return self._get_value("nvals")

    @property
    def layout(self):
        return self._get_value("layout")

    def to_coo(self, dtype=None, *, indices=True, values=True, sort=True):
        return self._get_value().to_coo(dtype, indices=indices, values=values, sort=sort)

    def isequal(self, other, *, check_dtype=False):
        return self._get_value().isequal(other, check_dtype=check_dtype)

    @property
    def T(self):
        return self._get_value("T")

    @property
    def mT(self):
        return self._get_value("mT")

    def __getitem__(self, keys):
        return self._get_value("__getitem__")(keys)


class TensorIndexExpr(TensorExpression):
    """``T[idx]``: an extraction (``.new()``, or an operand), or the target of an assignment:
    ``T[idx] << value`` and ``T[idx](mask, accum, replace) << value`` (the mask has the
    region's shape: GxB_subassign).
    """

    __slots__ = ("_parent", "_keys")

    def __lshift__(self, value):
        self._parent()._assign(self._keys, value, is_submask=False)

    def update(self, value):
        self._parent()._assign(self._keys, value, is_submask=False)

    def __call__(self, *args, **kwargs):
        from .tensor_index import TensorAssigner

        updater = self._parent(*args, **kwargs)
        return TensorAssigner(
            updater, self._keys, is_submask=updater.kwargs.get("mask") is not None
        )


def _computed(name):
    """``TensorExpression.name``: the Tensor method on the computed value, as Matrix
    expressions have their Matrix's methods: with ``gb.config.set(autocompute=True)``,
    ``(X * X).sum()``; otherwise a TypeError that says to call ``.new()`` first.
    """

    def method(self, *args, **kwargs):
        return self._get_value(name)(*args, **kwargs)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = getattr(Tensor, name).__doc__
    return method


# (not ``dup``: an expression's ``dup`` is ``new``)
for _name in (
    # conversions
    "to_dense", "to_matrix", "to_vector", "to_scalar", "to_matrices", "to_pydata_sparse",
    "to_scipy_sparse", "isclose",
    # shape operations
    "permute_dims", "expand_dims", "squeeze", "reshape", "ravel", "flatten",
    # operations
    "ewise_add", "ewise_mult", "ewise_union", "apply", "select", "reduce", "reduce_scalar",
    "sum", "prod", "min", "max", "any", "all", "tensordot", "matmul", "mttkrp", "unfold",
):  # fmt: skip
    setattr(TensorExpression, _name, _computed(_name))
del _name


class _TensorInfixExpr(InfixExprBase):
    """``X | Y``, ``X & Y`` or ``X @ Y`` waiting for an operator, as in ``binary.plus(X | Y)``
    or ``semiring.min_plus(X @ Y)``; ``.new()`` uses the default operator.
    """

    __slots__ = ()
    output_type = TensorExpression

    def __init__(self, left, right):
        # Matrix, Vector and ndarray operands follow Tensor rules once they meet a Tensor
        left = _as_tensor(left, self._infix)
        right = _as_tensor(right, self._infix)
        if left is None or right is None:
            raise TypeError(f"Bad types for `{self._infix}`: {type(left)}, {type(right)}")
        super().__init__(left, right)
        # Mismatched shapes raise here, as for Matrix
        self._validate()

    def _validate(self):
        self._to_expr()

    @property
    def shape(self):
        return self._to_expr().shape

    @property
    def ndim(self):
        return len(self.shape)

    def new(self, dtype=None, *, mask=None, layout=None, name=None, **opts):
        return self._to_expr().new(dtype, mask=mask, layout=layout, name=name, **opts)

    def __repr__(self):
        return f"gb.Tensor infix expression\n{self._format_expr()}"


class _TensorEwiseInfixExpr(_TensorInfixExpr):
    __slots__ = ()

    def _validate(self):
        # With an operator every dtype has: the default one may not exist for these dtypes,
        # and `monoid.any(X | Y)` (how a mask union is built) brings its own
        try:
            getattr(self.left, self.method_name)(self.right, binary.first)
        except KeyError:
            pass  # (no `first` for this pair of dtypes; the operator call checks)

    def _to_expr(self):
        if self._expr is None:
            if self.left.dtype == BOOL and self.right.dtype == BOOL:
                # As for Matrix: `x | y` is lor, `x & y` land
                op = binary.lor if self.method_name == "ewise_add" else binary.land
                self._expr = getattr(self.left, self.method_name)(self.right, op)
            else:
                self._expr = getattr(self.left, self.method_name)(self.right)
        return self._expr


class TensorEwiseAddExpr(_TensorEwiseInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _infix = "|"


class TensorEwiseMultExpr(_TensorEwiseInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _infix = "&"


class TensorMatMulExpr(_TensorInfixExpr):
    __slots__ = ()
    method_name = "matmul"
    _infix = "@"


def _infix_or(self, other):
    return TensorEwiseAddExpr(self, other)


def _infix_ror(self, other):
    return TensorEwiseAddExpr(other, self)


def _infix_and(self, other):
    return TensorEwiseMultExpr(self, other)


def _infix_rand(self, other):
    return TensorEwiseMultExpr(other, self)


def _infix_matmul(self, other):
    return TensorMatMulExpr(self, other)


def _infix_rmatmul(self, other):
    return TensorMatMulExpr(other, self)


for _cls in (Tensor, TensorExpression):
    for _name, (_op, _kind) in _INFIX.items():
        _fwd, _ref, _inp = _make_infix(_op, _kind)
        setattr(_cls, f"__{_name}__", _fwd)
        setattr(_cls, f"__r{_name}__", _ref)
        if _cls is Tensor:
            setattr(_cls, f"__i{_name}__", _inp)
    for _name in _COMPARISONS:
        # Set after the class body, as for Matrix, so Tensors stay hashable
        setattr(_cls, f"__{_name}__", _make_infix(_name, "mult")[0])
    _cls.__or__ = _infix_or
    _cls.__ror__ = _infix_ror
    _cls.__and__ = _infix_and
    _cls.__rand__ = _infix_rand
    _cls.__matmul__ = _infix_matmul
    _cls.__rmatmul__ = _infix_rmatmul
    # numpy defers `ndarray op T` to the Tensor's reflected operator
    _cls.__array_ufunc__ = None
del _name, _op, _kind, _fwd, _ref, _inp, _cls

utils._output_types[Tensor] = Tensor
utils._output_types[TensorExpression] = Tensor
for _cls in (TensorEwiseAddExpr, TensorEwiseMultExpr, TensorMatMulExpr):
    utils._output_types[_cls] = Tensor
del _cls
utils._tensor_types.add(Tensor)


class _TensorSS:
    __slots__ = ("_parent",)

    def __init__(self, parent):
        self._parent = parent

    @property
    def format(self):
        """The SuiteSparse format of the stored Matrix (always held by row)."""
        return self._parent._M.ss.format

    @property
    def is_iso(self):
        return self._parent._M.ss.is_iso

    @property
    def matrix(self):
        """The stored Matrix itself. Use it read-only: the Tensor copies its data before its
        next in-place change, but changes made through this Matrix reach the Tensor.
        """
        parent = self._parent
        parent._store.exported = True
        return parent._M


# ---------------------------------------------------------------------------------------------
# Storing a result: masks, accumulators and the output layout (DESIGN.md 6, 7.3, 7.4)


def _materialize(src):
    out = tl.new_matrix(src.dtype, src._nrows, src._ncols)
    out << src
    return out


def _output_mask(updater):
    """The update's mask as a Tensor mask: a mask of a Matrix or Vector of the output's shape
    becomes the same mask of it as a Tensor (as Tensor operations treat such operands; mask
    combinations make them, as in ``T.S & A.V``); one of another shape is refused, since
    masks do not broadcast.
    """
    mask = updater.kwargs.get("mask")
    if mask is None or type(mask.parent) is Tensor:
        return mask
    parent = _as_tensor(mask.parent, "mask")
    if parent is None or parent._shape != updater.parent._shape:
        raise TypeError(
            f"A Tensor needs a Tensor mask (or one of a Matrix or Vector of its shape, "
            f"{updater.parent._shape}); got a mask of {type(mask.parent).__name__}"
        )
    mask = type(mask)(parent)
    updater.kwargs["mask"] = mask
    return mask


def _mask_in_layout(mask, shape, layout, nthreads):
    parent = mask.parent
    if type(parent) is not Tensor:
        raise TypeError(f"A Tensor needs a Tensor mask; got a mask of {type(parent).__name__}")
    if parent._shape != shape:
        raise DimensionMismatch(f"mask shape {parent._shape} does not match {shape}")
    M = parent._M
    if parent._layout != layout:
        M, _ = tl.relayout(M, shape, parent._layout, layout, nthreads=nthreads)
    return type(mask)(M)


def _fixed_layout(updater):
    """The layout an update must end in, or None if the target may adopt another one."""
    target = updater.parent
    kw = updater.kwargs
    if (
        target._pinned
        or kw.get("accum") is not None
        or (kw.get("mask") is not None and not kw.get("replace", False))
    ):
        # Old values can survive (accum, or a mask without replace), or the user chose it.
        return target._layout
    return None


def _store(updater, src, shape, layout, *, fresh=False, share=None):
    """Store ``src`` (a Matrix or MatrixExpression in ``layout``) into the updater's target.

    The mask and accumulator apply only here, on the last step. ``fresh`` means ``src`` is a
    temporary Matrix the target may adopt; ``share`` is a Tensor whose stored Matrix ``src``
    is, which the target may share (copy-on-write) instead of copying.
    """
    target = updater.parent
    if target._shape != shape:
        raise DimensionMismatch(f"Tensor shapes do not match: {target._shape} and {shape}")
    kw = updater.kwargs
    mask = _output_mask(updater)
    accum = kw.get("accum")
    replace = kw.get("replace", False)
    opts = updater.opts
    nthreads = opts.get("nthreads")
    L = _fixed_layout(updater)
    if L is None:
        L = layout
        if mask is not None and mask.parent._layout != layout:
            mparent = mask.parent
            # Move whichever is smaller, the result or the mask (DESIGN.md 7.3 rule 4).
            if (
                type(mparent) is Tensor
                and isinstance(src, Matrix)
                and src._nvals < mparent._M._nvals
            ):
                L = mparent._layout
    if layout != L:
        if not isinstance(src, Matrix):
            src = _materialize(src)
        src, _ = tl.relayout(src, shape, layout, L, nthreads=nthreads)
        fresh = True
        share = None
    if mask is not None:
        mask = _mask_in_layout(mask, shape, L, nthreads)
    nrows = tl.prod(shape, L.rows)
    ncols = tl.prod(shape, L.cols)
    if mask is None and accum is None:
        if src.dtype == target.dtype:
            if share is not None:
                target._share(share)
                return
            if fresh:
                target._set_M(src, L)
                return
        if target._layout == L and not target._shared():
            target._M(**opts) << src
            return
        out = tl.new_matrix(target.dtype, nrows, ncols)
        out(**opts) << src
        target._set_M(out, L)
        return
    if target._layout != L:
        # Only with a mask and replace on an unpinned target: no old value survives.
        out = tl.new_matrix(target.dtype, nrows, ncols)
        out(mask=mask, replace=replace, **opts) << src
        target._set_M(out, L)
        return
    target._own()
    target._M(mask=mask, accum=accum, replace=replace, **opts) << src


def _fill(updater, value):
    target = updater.parent
    kw = updater.kwargs
    mask = _output_mask(updater)
    accum = kw.get("accum")
    replace = kw.get("replace", False)
    opts = updater.opts
    L = target._layout
    if mask is not None:
        mask = _mask_in_layout(mask, target._shape, L, opts.get("nthreads"))
    if mask is None and accum is None:
        out = tl.new_matrix(target.dtype, target._M._nrows, target._M._ncols)
        out(**opts) << value
        target._set_M(out, L)
        return
    target._own()
    target._M(mask=mask, accum=accum, replace=replace, **opts) << value


def _same_layout(a, b):
    """The stored Matrices of ``a`` and ``b`` in one layout (moving the smaller one)."""
    if a._layout == b._layout:
        return a._M, b._M
    if b._M._nvals <= a._M._nvals:
        return a._M, tl.relayout(b._M, b._shape, b._layout, a._layout)[0]
    return tl.relayout(a._M, a._shape, a._layout, b._layout)[0], b._M


def _choose_layout(updater, operands, result_nvals):
    """The layout to compute a same-shape op in: the candidate (an operand's layout, the
    fixed output layout, or the mask's) with the least estimated relayout cost.
    """
    fixed = _fixed_layout(updater)
    mask = _output_mask(updater)
    mlayout = None
    if mask is not None and mask.parent._shape == updater.parent._shape:
        # (a mask of another shape raises when stored)
        mlayout = mask.parent._layout
    layouts = {t._layout for t in operands}
    if fixed is not None:
        layouts.add(fixed)
    if mlayout is not None:
        layouts.add(mlayout)
    if len(layouts) == 1:
        return layouts.pop()
    shape = operands[0]._shape
    nthreads = _nthreads(updater)
    # Ties go to the larger operand's layout (DESIGN.md 7.3 rule 5).
    ops = sorted(operands, key=lambda t: -t._M._nvals)
    candidates = [t._layout for t in ops]
    if fixed is not None:
        candidates.append(fixed)
    if mlayout is not None:
        candidates.append(mlayout)
    best = None
    for L in candidates:
        cost = sum(tl.relayout_cost(shape, t._layout, L, t._M._nvals, nthreads) for t in ops)
        final = L if fixed is None else fixed
        if fixed is not None:
            cost += tl.relayout_cost(shape, L, fixed, result_nvals, nthreads)
        if mlayout is not None:
            cost += tl.relayout_cost(shape, mlayout, final, mask.parent._M._nvals, nthreads)
        if best is None or cost < best[0]:
            best = (cost, L)
    return best[1]


def _in_layout(t, layout, nthreads):
    if t._layout == layout:
        return t._M
    return tl.relayout(t._M, t._shape, t._layout, layout, nthreads=nthreads)[0]


# ---------------------------------------------------------------------------------------------
# Element-wise


def _ewise_expr(
    self, method, other, op, left_default=None, right_default=None, *, allow_expansion=False
):
    if (tensor := _as_tensor(other, method)) is None:
        self._expect_type(other, Tensor, within=method, argname="other", op=op)
    other = tensor
    shape = self._shape
    if other._shape != shape:
        try:
            shape = tuple(int(n) for n in np.broadcast_shapes(self._shape, other._shape))
        except ValueError:
            raise DimensionMismatch(
                f"Tensor shapes {self._shape} and {other._shape} do not broadcast in {method}"
            ) from None
    untyped_op = op
    extra = ()
    if method == "ewise_union":
        temp_op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        left = _union_default(self, left_default, temp_op.type, op, "left_default")
        right = _union_default(self, right_default, temp_op.type2, op, "right_default")
        op1 = get_typed_op(op, self.dtype, right.dtype, kind="binary")
        op2 = get_typed_op(op, left.dtype, other.dtype, kind="binary")
        if op1 is not op2:
            left_dtype = unify(op1.type, op2.type, is_right_scalar=True)
            right_dtype = unify(op1.type2, op2.type2, is_left_scalar=True)
            op = get_typed_op(op, left_dtype, right_dtype, kind="binary")
        else:
            op = op1
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        extra = (left, right)
    else:
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method, argname="op")
    expr_repr = "{0.name}.%s({1.name}, op={op})" % method  # noqa: UP031
    if shape == self._shape == other._shape:
        return TensorExpression(
            method,
            _ewise_recipe,
            [self, other],
            (method, self, other, op, extra),
            dtype=op.return_type,
            shape=shape,
            op=op,
            expr_repr=expr_repr,
        )
    _check_expansion(method, (self, other), shape, allow_expansion)
    return TensorExpression(
        method,
        _bcast_recipe,
        [self, other],
        (method, self, other, op, untyped_op, extra, shape),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr=expr_repr,
    )


# ---------------------------------------------------------------------------------------------
# Broadcasting (DESIGN.md 5.4, 7.2): numpy shape rules with missing-value semantics. The
# intersection (ewise_mult, comparisons) scales by a diagonal and never replicates when one
# operand covers the whole result shape; the union replicates the broadcast operand, as numpy
# would, behind an O(1) size guard (review F1).


def _broadcast_factor(tshape, shape):
    """How many times an operand of ``tshape`` repeats to fill ``shape``."""
    off = len(shape) - len(tshape)
    factor = math.prod(shape[:off])
    for a, n in enumerate(tshape):
        if n == 1:
            factor *= shape[a + off]
    return factor


def _check_expansion(method, operands, shape, allow):
    sizes = [(_broadcast_factor(t._shape, shape), t._M._nvals) for t in operands]
    if method == "ewise_mult":
        if any(f == 1 for f, _ in sizes):
            return  # one operand covers the result: diagonal scaling, no replication
        replicated = min(f * n for f, n in sizes)
    else:
        replicated = sum(f * n for f, n in sizes if f > 1)
    limit = config.get("tensor_max_expansion", 10**9)
    if replicated > limit and not allow:
        raise ValueError(
            f"{method} of shapes {operands[0]._shape} and {operands[1]._shape} would replicate "
            f"{replicated} entries to broadcast (the limit is {limit}). numpy semantics fill "
            "every broadcast position; use ewise_mult for the intersection, pass "
            "allow_expansion=True, or raise `graphblas.config` option tensor_max_expansion."
        )


def _squeezed(t, shape):
    """``t`` without its broadcast size-1 axes, and the output axis of each remaining axis."""
    off = len(shape) - t.ndim
    drop = [a for a, n in enumerate(t._shape) if n == 1 and shape[a + off] != 1]
    keep = [a + off for a in range(t.ndim) if a not in drop]
    return (t.squeeze(drop) if drop else t), keep


def _expand(t, shape):
    """``t`` replicated over its broadcast axes to ``shape``, as a Tensor (kron with ones)."""
    if t._shape == shape:
        return t
    ts, out_of = _squeezed(t, shape)
    bcast = tuple(o for o in range(len(shape)) if o not in out_of)
    rows = bcast + tuple(out_of[a] for a in ts._layout.rows)
    cols = tuple(out_of[a] for a in ts._layout.cols)
    layout = tl.as_layout((rows, cols), shape)
    reps = tl.prod(shape, bcast)
    nrows, ncols = tl.prod(shape, rows), tl.prod(shape, cols)
    if reps == 1:
        return Tensor._from_M(ts._M, shape, layout, store=ts._store)
    ones = Matrix.from_scalar(True, reps, 1)
    M = tl.new_matrix(t.dtype, nrows, ncols)
    # kron(ones, M)[(b, r), c] = M[r, c]: the broadcast axes lead the rows
    M << ones.kronecker(ts._M, binary.second)
    return Tensor._from_M(M, shape, layout)


def _lift_rank(t, ndim):
    """``t`` with leading size-1 axes added up to ``ndim`` axes (metadata only)."""
    if t.ndim == ndim:
        return t
    return t.expand_dims(tuple(range(ndim - t.ndim)))


def _one_term_monoid(dtype):
    # Diagonal scaling sums one term per output, so any monoid gives the same value; pick one
    # with factory kernels for the output type.
    if dtype == BOOL:
        return monoid.lor
    if dtype._is_udt:
        return monoid.any
    return monoid.plus


def _bcast_recipe(updater, method, a, b, op, untyped_op, extra, shape):
    # Give both operands the result's rank first. A lower-rank operand whose missing leading
    # axes are all size 1 repeats once (broadcast factor 1) yet has fewer axes than the
    # result, so "covers the result" must also mean "has its axes".
    a = _lift_rank(a, len(shape))
    b = _lift_rank(b, len(shape))
    if method == "ewise_mult":
        _bcast_mult(updater, a, b, op, untyped_op, shape)
    else:
        _ewise_recipe(updater, method, _expand(a, shape), _expand(b, shape), op, extra)


def _bcast_mult(updater, a, b, op, untyped_op, shape):
    fa = _broadcast_factor(a._shape, shape)
    fb = _broadcast_factor(b._shape, shape)
    if fa > 1 and fb > 1:
        # Both broadcast (an outer product, for example): replicate the cheaper one
        if fa * a._M._nvals <= fb * b._M._nvals:
            a = _expand(a, shape)
        else:
            b = _expand(b, shape)
        fa = _broadcast_factor(a._shape, shape)
    if fa == 1:
        big, small, big_left = a, b, True
    else:
        big, small, big_left = b, a, False
    nthreads = updater.opts.get("nthreads")
    ts, F = _squeezed(small, shape)
    if ts._M._nvals == 0:
        L = big._layout
        empty = tl.new_matrix(op.return_type, tl.prod(shape, L.rows), tl.prod(shape, L.cols))
        _store(updater, empty, shape, L, fresh=True)
        return
    if not F:
        # The small side is one value
        value = ts._M[0, 0].new()
        if big_left:
            src = big._M.apply(op, right=value)
        else:
            src = big._M.apply(op, left=value)
        _store(updater, src, shape, big._layout)
        return
    flip = getattr(untyped_op, "commutes_to", None)
    # M @ diag(v) multiplies (big, small); diag(v) @ M multiplies (small, big)
    col_op = untyped_op if big_left else flip
    row_op = flip if big_left else untyped_op
    if hasattr(untyped_op, "binaryop"):  # a Monoid
        col_op = row_op = untyped_op.binaryop
    Fset = set(F)
    M, L = big._M, big._layout
    nv = M._nvals
    ti = _ti(nthreads or tl._nthreads())
    seq, p = L.state
    # Options: (cost, split, group side, expand from the nonempty rows); split "gather" puts
    # the small side's values on the stored entries, None relayouts
    options = []
    if (
        nv > 0
        and not M.dtype._is_udt
        and not ts.dtype._is_udt
        and ts._M._nvals == math.prod(ts._shape)
        and M.ss.format in {"csr", "hypercsr"}
        and not M.ss.is_iso
    ):
        need_rows = any(o in L.rows for o in F)
        cost = (C_GATHER["rows" if need_rows else "cols"] + C_EWISE_SAME[ti]) * nv
        if M.ss.format == "hypercsr":
            cost += C_EWISE_HROW[ti] * tl.nonempty_rows(M)
        options.append((cost, "gather", None, False))
    for q in range(len(seq) + 1):
        if not tl._valid_split(shape, seq, q):
            continue
        move = 0.0 if q == p else _rs_cost(shape, seq, q, nv, ti)
        rows, cols = seq[:q], seq[q:]
        if col_op is not None and Fset == set(cols):
            options.append((move + C_DIAG[ti] * nv, q, "cols", False))
        if row_op is not None and Fset == set(rows):
            options.append((move + C_DIAG[ti] * nv, q, "rows", False))
        if q != p:
            continue
        if row_op is not None and Fset < set(rows):
            hyper = M.ss.format == "hypercsr"
            if hyper:
                cost = C_DIAG[ti] * nv + C_HROW[ti] * tl.nonempty_rows(M)
            else:
                cost = C_DIAG[ti] * nv + C_SEL_POS[ti] * tl.prod(shape, rows)
            options.append((cost, q, "rows", hyper))
        if col_op is not None and Fset < set(cols):
            cost = C_DIAG[ti] * nv + C_SEL_POS[ti] * tl.prod(shape, cols)
            options.append((cost, q, "cols", False))
    # Or relayout so the small side's axes are exactly one group
    rest = tuple(o for o in range(len(shape)) if o not in Fset)
    targets = set()
    orders = itertools.permutations(F) if len(shape) <= tl.MAX_PLANNED_NDIM else [tuple(F)]
    for f in orders:
        for r in itertools.permutations(rest) if len(shape) <= 5 else [rest]:
            if col_op is not None and tl._valid_split(shape, r + f, len(r)):
                targets.add((r + f, len(r)))
            if row_op is not None and tl._valid_split(shape, f + r, len(f)):
                targets.add((f + r, len(f)))
    if targets:
        cost = tl.plan(shape, L.state, frozenset(targets), nv, nthreads or tl._nthreads())[0]
        options.append((cost + C_DIAG[ti] * nv, None, None, False))
    if not options:
        # No diagonal side works (an operator with no flipped form): replicate instead
        _ewise_recipe(
            updater,
            "ewise_mult",
            _expand(a, shape) if big is b else a,
            _expand(b, shape) if big is a else b,
            op,
            (),
        )
        return
    _, q, side, from_rows = min(options, key=lambda o: o[0])
    if q == "gather":
        values = ts.to_dense().ravel()
        W = tl.gather_operand(M, shape, L, tuple(F), values, ts.dtype)
        if W is not None:
            src = M.ewise_mult(W, op) if big_left else W.ewise_mult(M, op)
            _store(updater, src, shape, L)
            return
        q = None
    if q is None:
        M, L = tl.relayout(M, shape, L, frozenset(targets), nthreads=nthreads)
        side = "cols" if set(L.cols) == Fset and col_op is not None else "rows"
    elif q != p:
        M = M.ss.reshape(tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
        L = Layout(seq[:q], seq[q:])
    group = L.cols if side == "cols" else L.rows
    # The small side as a Vector over `group` (expanded over the group's other axes)
    inv = {o: i for i, o in enumerate(F)}
    sub = tuple(o for o in group if o in Fset)
    vM, _ = tl.relayout(
        ts._M, ts._shape, ts._layout, Layout((), tuple(inv[o] for o in sub)), nthreads=nthreads
    )
    v = vM[0, :].new()
    if from_rows:
        v = _expand_on_rows(M, shape, group, tuple(group.index(o) for o in sub), v)
    elif len(sub) < len(group):
        St = tl.selector(
            tuple(shape[o] for o in group),
            tuple(i for i, o in enumerate(group) if o in Fset),
            transposed=True,
        )
        e = Vector(v.dtype, St._nrows)
        e << St.mxv(v, semiring.lor_second if v.dtype == BOOL else semiring.plus_second)
        v = e
    D = v.diag()
    if side == "cols":
        sr = get_semiring(_one_term_monoid(op.return_type), col_op)
        src = M.mxm(D, sr)
    else:
        sr = get_semiring(_one_term_monoid(op.return_type), row_op)
        src = D.mxm(M, sr)
    _store(updater, src, shape, L)


def _expand_on_rows(M, shape, rows, sub_pos, v):
    """``v`` (over the row axes at ``sub_pos``) expanded onto the nonempty rows of ``M``: a
    Vector over the row group with ``e[r] = v[coordinates of r]``, sized by the data.
    """
    h = tl.nonempty_row_ids(M)
    gshape = tuple(shape[a] for a in rows)
    co = {}
    tl.unravel(h, gshape, tuple(range(len(gshape))), co)
    vidx = tl.ravel(co, gshape, sub_pos, n=h.size)
    idx, vals = v.to_coo()
    dense = np.zeros(v._size, vals.dtype)
    present = np.zeros(v._size, bool)
    dense[idx] = vals
    present[idx] = True
    keep = present[vidx]
    return Vector.from_coo(h[keep], dense[vidx[keep]], v.dtype, size=tl.prod(shape, rows))


def _ewise_recipe(updater, method, a, b, op, extra):
    if method == "ewise_mult":
        est = min(a._M._nvals, b._M._nvals)
    else:
        est = a._M._nvals + b._M._nvals
    L = _choose_layout(updater, (a, b), est)
    nthreads = updater.opts.get("nthreads")
    A = _in_layout(a, L, nthreads)
    B = A if b is a else _in_layout(b, L, nthreads)
    if method == "ewise_add":
        src = A.ewise_add(B, op)
    elif method == "ewise_mult":
        src = A.ewise_mult(B, op)
    else:
        src = A.ewise_union(B, op, *extra)
    _store(updater, src, a._shape, L)


def _select_recipe(updater, a, op, thunk):
    L = _choose_layout(updater, (a,), a._M._nvals)
    A = _in_layout(a, L, updater.opts.get("nthreads"))
    _store(updater, A.select(op, thunk), a._shape, L)


def _apply_recipe(updater, a, op, right, left):
    L = _choose_layout(updater, (a,), a._M._nvals)
    A = _in_layout(a, L, updater.opts.get("nthreads"))
    _store(updater, A.apply(op, right=right, left=left), a._shape, L)


# ---------------------------------------------------------------------------------------------
# Reductions (DESIGN.md 7.2)


def _reduce_expr(self, op, axis, keepdims, *, name="reduce", cast=None):
    axes = _normalize_axes(axis, self.ndim)
    op = get_typed_op(op, self.dtype if cast is None else cast, kind="binary")
    if op.opclass == "BinaryOp" and op.monoid is not None:
        op = op.monoid
    self._expect_op(op, "Monoid", within=name, argname="op")
    if keepdims:
        shape = tuple(1 if a in axes else n for a, n in enumerate(self._shape))
    else:
        shape = tuple(n for a, n in enumerate(self._shape) if a not in axes)
    axis_repr = "None" if axis is None else repr(axis)
    return TensorExpression(
        name,
        _reduce_recipe,
        [self],
        (self, op, axes, keepdims, cast),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr=f"{{0.name}}.{name}({{op}}, axis={axis_repr})",
    )


def _reduce_recipe(updater, a, op, axes, keepdims, cast):
    shape, layout = a._shape, a._layout
    M = a._M
    if cast is not None and M.dtype != cast:
        M = M.dup(cast)
    elif not axes:
        _store(updater, M, shape, layout, share=a)
        return
    M, lay = _reduce_matrix(M, shape, layout, set(axes), op, updater.opts)
    if keepdims:
        out_shape = tuple(1 if x in axes else n for x, n in enumerate(shape))
        out_layout = _insert_unit_axes(lay, axes)
    else:
        kept = [x for x in range(len(shape)) if x not in axes]
        out_shape = tuple(shape[x] for x in kept)
        out_layout = _renumber(lay, {x: i for i, x in enumerate(kept)})
    if _fixed_layout(updater) is None:
        M, out_layout = _healthy(M, out_shape, out_layout)
    _store(updater, M, out_shape, out_layout, fresh=True)


# The default split puts as many axes in the rows as the data fills. A row group indexes its
# nonempty rows (h), so selectors and expansions over it follow the data, while a column group
# costs per position (round 1: NELL-2's (0|1,2) has a 2.6e8-position column group). But rows
# cost too: a hypersparse row costs a hash lookup in ewise and mxm (about 19 / 130 ns at 8 / 1
# threads: Zipf's (0,1|2), 4.3 entries per row, took 53 ms for X + Y against 9.6 in (0|1,2)), and
# a sparse row about 3 / 16 ns. So rows must hold MIN_ROW_ENTRIES entries each, or
# MIN_HYPER_ROW_ENTRIES when they would be hypersparse (NELL-2's hold 228 each).
MIN_ROW_ENTRIES = 1.5
MIN_HYPER_ROW_ENTRIES = 32
FULL_ROW_ENTRIES = 4
# The sampled check costs about 0.3 ms; the exact one about 1.1 ns per entry (ravel, order
# check and count), so sorted input is sampled only past 2^23 entries
_SPLIT_SAMPLE = 2**14
_SPLIT_SAMPLE_PAST = 2**23


def _rows_ok(nvals, nonempty, positions, narrower_cols):
    """Whether rows of ``nvals`` entries over ``nonempty`` of ``positions`` rows are worth it.

    Thin sparse rows (under FULL_ROW_ENTRIES each) cost about 2x in ewise (u6, 1.6 per row:
    X + Y 3.9 ms vs 1.5), so they pay off only when the alternative, one axis fewer in the
    rows, has a column group of more positions than entries (repro case 1: 2.6x).
    """
    hyper = nonempty * 16 < positions  # SuiteSparse's default hyper switch is 1/16
    if hyper:
        return nvals >= MIN_HYPER_ROW_ENTRIES * nonempty
    if nvals >= FULL_ROW_ENTRIES * nonempty:
        return True
    return nvals >= MIN_ROW_ENTRIES * nonempty and narrower_cols > nvals


def _data_split_layout(coords, shape, n):
    """``(layout, row index or None, settle)``: identity order split at the largest p whose
    rows hold enough entries each. A sample of the row keys rejects thin splits cheaply (its
    distinct count can only err low, i.e. toward rows that look full): always for large
    input, and for unsorted input, whose exact count would need a sort. Sorted input (C
    order, the common case) is judged exactly from its row keys, which the build then
    reuses; unsorted input is built at the split the sample allows and settled afterwards
    (``settle=True``).
    """
    ndim = len(shape)
    seq = tuple(range(ndim))
    if math.prod(shape) >= tl.MAX_GROUP:
        # Reshape-based relayouts are unavailable; keep the shape-based rule
        return tl.default_layout(shape, n), None, False
    sample = np.random.default_rng(0).integers(0, n, min(n, _SPLIT_SAMPLE))

    def thin_by_sample(p, positions, narrower_cols):
        low = _sampled_nonempty_rows(coords, shape, seq[:p], n, sample)
        return not _rows_ok(n, low, positions, narrower_cols)

    for p in range(ndim - 1, 0, -1):
        if not tl._valid_split(shape, seq, p):
            continue
        positions = tl.prod(shape, seq[:p])
        if positions * FULL_ROW_ENTRIES <= n:
            return Layout(seq[:p], seq[p:]), None, False  # full enough even if all nonempty
        narrower_cols = tl.prod(shape, seq[p - 1 :])
        large = n > _SPLIT_SAMPLE_PAST
        if large and thin_by_sample(p, positions, narrower_cols):
            continue
        keys = tl.ravel(coords, shape, seq[:p], n=n)
        if not np.all(keys[1:] >= keys[:-1]):
            if not large and thin_by_sample(p, positions, narrower_cols):
                continue
            return Layout(seq[:p], seq[p:]), keys, True
        nonempty = int(np.count_nonzero(keys[1:] != keys[:-1])) + 1
        if _rows_ok(n, nonempty, positions, narrower_cols):
            return Layout(seq[:p], seq[p:]), keys, False
    for p in range(1, ndim):
        if tl._valid_split(shape, seq, p):
            return Layout(seq[:p], seq[p:]), None, False
    return tl.default_layout(shape, n), None, False


def _sampled_nonempty_rows(coords, shape, row_axes, n, sample):
    """A low estimate of the distinct row keys: the distinct count d in a sample of s
    entries, solving D * (1 - exp(-s / D)) = d (rows of equal size; skewed rows give less).
    """
    keys = np.sort(tl.ravel([c[sample] for c in coords], shape, row_axes))
    s = sample.size
    d = int(np.count_nonzero(keys[1:] != keys[:-1])) + 1  # np.unique is 3x slower
    if d >= s:
        return n
    lo, hi = float(d), float(n)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid * -math.expm1(-s / mid) < d:
            lo = mid
        else:
            hi = mid
    return lo


def _settle_split(M, shape, layout):
    """Move the split left (one reshape each) while the rows are too thin (``_rows_ok``),
    counted through the Container.
    """
    seq, p = layout.state
    nvals = M._nvals
    while p > 1 and not _rows_ok(
        nvals, tl.count_nonempty_rows(M), tl.prod(shape, seq[:p]), tl.prod(shape, seq[p - 1 :])
    ):
        q = p - 1
        if not tl._valid_split(shape, seq, q):
            break
        nrows, ncols = tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:])
        if nrows <= 2**31 and ncols <= 2**31:
            M.ss.reshape(nrows, ncols, inplace=True)
        else:
            # SuiteSparse 10.5.1's in-place reshape cannot widen 32-bit indices
            M = M.ss.reshape(nrows, ncols)
        p = q
    return M, Layout(seq[:p], seq[p:])


def _healthy(M, shape, layout):
    """Move an empty column group (an n x 1 Matrix, held as n rows of one entry at most) to
    the default split, by one reshape (review F4).

    An empty row group is left alone: one row holding every entry costs 12 bytes per entry
    like a Vector, and SuiteSparse runs ewise, apply and selector products on it as fast as
    on a square Matrix (measured at 1e6 entries), so a reshape there would only add time.
    """
    ndim = len(shape)
    if ndim == 0 or layout.cols or math.prod(shape) >= tl.MAX_GROUP:
        return M, layout
    seq = layout.seq
    nvals = M._nvals
    if ndim == 1:
        p = 0
    else:
        p = 1
        for q in range(ndim - 1, 0, -1):
            if tl.prod(shape, seq[:q]) * 4 <= nvals:
                p = q
                break
    if not tl._valid_split(shape, seq, p):
        return M, layout
    if nvals == 0:
        M = tl.new_matrix(M.dtype, tl.prod(shape, seq[:p]), tl.prod(shape, seq[p:]))
    else:
        M = M.ss.reshape(tl.prod(shape, seq[:p]), tl.prod(shape, seq[p:]))
    return M, Layout(seq[:p], seq[p:])


def _selector_ok(n, nvals):
    # A shape-only selector has one entry per position of the group; past this it costs
    # more than the data, and REBUILD is used instead.
    return n <= max(4 * nvals, 2**16)


# Strategy costs in ns, (8 threads, 1 thread), measured 2026-10-20 on (3000, 2000, 10000) at
# 7.7e6 entries. A shape-only selector costs per position of its group, whatever the data;
# one built from the nonempty rows of a hypersparse row group costs per nonempty row.
C_SEL_ENTRY = (0.7, 2.7)
C_SEL_POS = (2.2, 12.0)
C_SEL_BUILD = 15.0  # per position of a selector too big to cache (built with numpy per call)
C_HROW = (75.0, 255.0)  # per nonempty row: build from h, then a hash lookup per row
C_MXV = (0.66, 3.3)
C_VXM = (4.6, 11.4)
C_DIAG = (0.6, 2.1)
# Broadcast by gathering the small side's values onto the stored entries (numpy, one thread):
# per entry when only column coordinates are needed, more when row ids must be expanded; then
# a same-pattern ewise_mult, which pays per row of a hypersparse row group too.
C_GATHER = {"cols": 2.5, "rows": 4.5}
C_EWISE_SAME = (0.9, 5.0)
C_EWISE_HROW = (7.0, 48.0)


def _ti(nthreads):
    return 0 if nthreads > 1 else 1


def _shape_selector_cost(nvals, positions, ti):
    cost = C_SEL_ENTRY[ti] * nvals + C_SEL_POS[ti] * positions
    if positions > tl._matrix_cache.maxweight:
        cost += C_SEL_BUILD * positions
    return cost


def _row_selector_cost(M, positions, ti):
    """Cost of a selector over M's row group: from the nonempty rows when M is hypersparse."""
    nvals = M._nvals
    if M.ss.format == "hypercsr":
        return C_SEL_ENTRY[ti] * nvals + C_HROW[ti] * tl.nonempty_rows(M)
    return _shape_selector_cost(nvals, positions, ti)


def _rs_cost(shape, seq, q, nvals, ti):
    return tl.rs_cost(shape, seq, q, nvals, ti)


def _row_reducer(M, shape, rows, kept):
    """S with S @ M reducing the row axes not in ``kept`` (positions in ``rows``)."""
    if M.ss.format == "hypercsr":
        return tl.rows_selector(M, shape, rows, kept)
    return tl.selector(tuple(shape[x] for x in rows), kept)


def _vector_to_row(v):
    return tl.steal_vector_as_row(v)


def _reduce_matrix(M, shape, layout, A, op, opts):
    """Reduce the axes ``A`` of ``M`` (stored in ``layout``); return the result Matrix and
    its layout over the kept axes (with their original numbers).
    """
    ndim = len(shape)
    dtype = op.return_type
    if len(A) == ndim:
        s = Scalar(dtype)
        s(**opts) << M.reduce_scalar(op)
        out = tl.new_matrix(dtype, 1, 1)
        if not s._is_empty:
            out[0, 0] = s
        return out, Layout((), ())
    nvals = M._nvals
    if M.dtype._is_udt:
        return _reduce_rebuild(M, shape, layout, A, op)
    # Pass on only the descriptor options the user gave: building one costs microseconds.
    nthreads = opts.get("nthreads") or tl._nthreads()
    ti = _ti(nthreads)
    A = set(A)
    while A:
        rows, cols = layout
        Ac = A & set(cols)
        if Ac:
            if Ac == set(cols):
                v = Vector(dtype, M._nrows)
                v(**opts) << M.reduce_rowwise(op)
                M = _vector_to_row(v)
                layout = Layout((), rows)
            else:
                gshape = tuple(shape[x] for x in cols)
                # A column group has no index of its nonempty columns, so the selector costs
                # per position; past the data's own cost, rebuild with dup_op instead
                nv = M._nvals
                if _shape_selector_cost(nv, math.prod(gshape), ti) > tl.rebuild_cost(nv, ti):
                    return _reduce_rebuild(M, shape, layout, A, op)
                kept = tuple(i for i, x in enumerate(cols) if x not in Ac)
                St = tl.selector(gshape, kept, transposed=True)
                out = tl.new_matrix(dtype, M._nrows, St._ncols)
                out(**opts) << M.mxm(St, get_semiring(op, binary.first))
                M = out
                layout = Layout(rows, tuple(cols[i] for i in kept))
            A -= Ac
            continue
        Ar = A & set(rows)
        if Ar == set(rows):
            # Rows with about one entry each: a columnwise scatter costs a cache miss per
            # entry; rebuilding with dup_op is 4-5x faster (s13).
            if tl.density(M):
                return _reduce_rebuild(M, shape, layout, A, op)
            ncols = tl.prod(shape, cols)
            wide = ncols >= 1e5
            if (
                wide
                and nthreads > 1
                and len(cols) >= 2
                and math.prod(shape) < tl.MAX_GROUP
                and 4 * tl.prod(shape, rows) * shape[cols[0]] <= nvals
            ):
                # reduce_columnwise into many outputs scatters like a wide transpose and is
                # slow with threads (DESIGN.md 3.2); move one column axis into the rows and
                # reduce with a selector instead, when the selector is small next to the data
                # (1000^3 / 1e7: 6.6 ms vs 55.6). A larger selector costs more than a
                # single-threaded columnwise reduce (Uber: 32.8 ms vs 4.7).
                q = len(rows) + 1
                seq = layout.seq
                M = M.ss.reshape(tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
                layout = Layout(seq[:q], seq[q:])
                continue
            v = Vector(dtype, M._ncols)
            # Into many outputs, 2 threads scatter as fast as 1 or faster; 4 and 8 can be
            # 3-4x slower (Uber 4.0 / 4.9 / 17 / 15 ms at 2 / 1 / 4 / 8 threads)
            v(**(dict(opts, nthreads=min(2, nthreads)) if wide else opts)) << M.reduce_columnwise(
                op
            )
            M = _vector_to_row(v)
            layout = Layout((), cols)
        else:
            nv = M._nvals
            if _row_selector_cost(M, tl.prod(shape, rows), ti) > tl.rebuild_cost(nv, ti):
                return _reduce_rebuild(M, shape, layout, A, op)
            kept = tuple(i for i, x in enumerate(rows) if x not in Ar)
            S = _row_reducer(M, shape, rows, kept)
            out = tl.new_matrix(dtype, S._nrows, M._ncols)
            out(**opts) << S.mxm(M, get_semiring(op, binary.second))
            M = out
            layout = Layout(tuple(rows[i] for i in kept), cols)
        A -= Ar
    return M, layout


def _reduce_rebuild(M, shape, layout, A, op):
    """Reduce by rebuilding with the monoid as ``dup_op``: best when rows hold about one
    entry each (DESIGN.md 3.12), and the path for UDTs.
    """
    coords, values = tl.coords_of(M, shape, layout.state)
    # The layout may already lack axes reduced by earlier steps
    kept = [x for x in sorted(layout.seq) if x not in A]
    kshape = tuple(shape[x] for x in kept)
    lay = tl.default_layout(kshape, values.size)
    rows = tuple(kept[i] for i in lay.rows)
    cols = tuple(kept[i] for i in lay.cols)
    r = tl.ravel(coords, shape, rows, n=values.size)
    c = tl.ravel(coords, shape, cols, n=values.size)
    out = tl.from_coo(
        r,
        c,
        values,
        op.return_type,
        tl.prod(shape, rows),
        tl.prod(shape, cols),
        dup_op=op.binaryop,
    )
    return out, Layout(rows, cols)


# ---------------------------------------------------------------------------------------------
# Contractions (DESIGN.md 7.2: relayout so the contracted axes are exactly one group, then mxm)


def _tensordot_axes(axes, ndim_a, ndim_b):
    if isinstance(axes, (tuple, list)):
        if len(axes) != 2:
            raise ValueError("axes must be an int or a pair of axis sequences")
        a_axes, b_axes = axes
        a_axes = (a_axes,) if not isinstance(a_axes, (tuple, list)) else tuple(a_axes)
        b_axes = (b_axes,) if not isinstance(b_axes, (tuple, list)) else tuple(b_axes)
        if len(a_axes) != len(b_axes):
            raise ValueError(f"axes {axes} must pair the same number of axes")
        a_axes = tuple(_normalize_axis(x, ndim_a) for x in a_axes)
        b_axes = tuple(_normalize_axis(x, ndim_b) for x in b_axes)
        if len(set(a_axes)) != len(a_axes) or len(set(b_axes)) != len(b_axes):
            raise ValueError(f"repeated axis in {axes}")
        return a_axes, b_axes
    n = operator.index(axes)
    if not 0 <= n <= min(ndim_a, ndim_b):
        raise ValueError(f"axes={n} must be in 0..{min(ndim_a, ndim_b)}")
    return tuple(range(ndim_a - n, ndim_a)), tuple(range(n))


def _tensordot_expr(self, other, axes, op):
    if (tensor := _as_tensor(other, "tensordot")) is None:
        self._expect_type(other, Tensor, within="tensordot", argname="other", op=op)
    other = tensor
    a_axes, b_axes = _tensordot_axes(axes, self.ndim, other.ndim)
    for x, y in zip(a_axes, b_axes, strict=True):
        if self._shape[x] != other._shape[y]:
            raise DimensionMismatch(
                f"axis {x} of size {self._shape[x]} cannot contract with axis {y} of size "
                f"{other._shape[y]}"
            )
    op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="tensordot", argname="op")
    shape = tuple(n for x, n in enumerate(self._shape) if x not in a_axes) + tuple(
        n for y, n in enumerate(other._shape) if y not in b_axes
    )
    return TensorExpression(
        "tensordot",
        _tensordot_recipe,
        [self, other],
        (self, other, op, a_axes, b_axes),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.tensordot({1.name}, axes=%s, op={op})"  # noqa: UP031
        % repr((a_axes, b_axes)).replace("{", "{{").replace("}", "}}"),
    )


def _orders(axes):
    # Any order of the free axes gives the same product (only the output layout differs),
    # so offer them all to the planner when there are few.
    if len(axes) <= 3:
        return list(itertools.permutations(axes))
    return [axes]


def _contract(updater, a, b, op, a_axes, b_axes, a_batch=(), b_batch=()):
    """``a`` and ``b`` contracted over ``a_axes`` with ``b_axes`` (paired), with the paired
    batch axes ``a_batch`` / ``b_batch`` kept once. Returns ``(src, layout, shape)``, where
    ``src`` is a Matrix or MatrixExpression and the output axes are the batch axes (in
    ``a_batch`` order), then a's free axes, then b's, each in axis order.

    Without batch axes: relayout so the contracted axes are exactly a's column group and b's
    row group, in the same order, then one mxm. With batch axes they lead both row groups and
    a's columns are embedded block-diagonally, so one mxm does every batch (DESIGN.md 3.6).
    Only a is embedded: the product keeps b's columns, so it is narrow and needs no
    unembedding, and it beats a loop over the batches at 1 thread too (unlike s5, whose wide
    product had to be sorted).
    """
    nthreads = _nthreads(updater)
    opt_nthreads = updater.opts.get("nthreads")
    free_a = tuple(x for x in range(a.ndim) if x not in a_axes and x not in a_batch)
    free_b = tuple(y for y in range(b.ndim) if y not in b_axes and y not in b_batch)
    k, nb = len(a_axes), len(a_batch)
    na, nbv = a._M._nvals, b._M._nvals
    best = None
    # The contracted (and batch) axes must come in the same order on both sides; try each
    # order and the free-axis orders, and take the cheapest pair of relayouts.
    sigmas = list(itertools.permutations(range(k))) if k <= 3 else [tuple(range(k))]
    batch_orders = list(itertools.permutations(range(nb))) if nb <= 3 else [tuple(range(nb))]
    for sigma in sigmas:
        ca = tuple(a_axes[i] for i in sigma)
        cb = tuple(b_axes[i] for i in sigma)
        for beta in batch_orders:
            ba = tuple(a_batch[i] for i in beta)
            bb = tuple(b_batch[i] for i in beta)
            ta = frozenset(
                (ba + fa + ca, nb + len(fa))
                for fa in _orders(free_a)
                if tl._valid_split(a._shape, ba + fa + ca, nb + len(fa))
            )
            tb = frozenset(
                (bb + cb + fb, nb + k)
                for fb in _orders(free_b)
                if tl._valid_split(b._shape, bb + cb + fb, nb + k)
            )
            if not ta or not tb:
                continue
            cost_a, _, sa = tl.plan(a._shape, a._layout.state, ta, na, nthreads)
            cost_b, _, sb = tl.plan(b._shape, b._layout.state, tb, nbv, nthreads)
            if best is None or cost_a + cost_b < best[0]:
                best = (cost_a + cost_b, sa, sb, beta)
    if best is None:
        raise ValueError("the contracted or free axes do not fit in a Matrix dimension (2**60)")
    _, sa, sb, beta = best
    A = _in_layout(a, Layout.from_state(sa), opt_nthreads)
    B = _in_layout(b, Layout.from_state(sb), opt_nthreads)
    out_batch = {x: a_batch.index(x) for x in a_batch}
    out_a = {x: nb + i for i, x in enumerate(free_a)}
    out_b = {y: nb + len(free_a) + i for i, y in enumerate(free_b)}
    rows = tuple(out_batch[x] for x in sa[0][:nb]) + tuple(
        out_a[x] for x in sa[0][nb : nb + len(free_a)]
    )
    layout = Layout(rows, tuple(out_b[y] for y in sb[0][nb + k :]))
    shape = (
        tuple(a._shape[x] for x in a_batch)
        + tuple(a._shape[x] for x in free_a)
        + tuple(b._shape[y] for y in free_b)
    )
    if not nb:
        return A.mxm(B, op), layout, shape
    nblocks = tl.prod(a._shape, a_batch)
    rows_per_block = A._nrows // max(nblocks, 1)
    K = A._ncols
    if A is a._M or not tl.is_by_row(A):
        A = A.dup()
    if A.ss.format not in {"csr", "hypercsr"}:
        A.ss.config["sparsity_control"] = ["sparse", "hypersparse"]
    if not tl.embed_block_diagonal(A, rows_per_block, nblocks):
        # The new column indices need a wider integer type: rebuild instead
        r, c, v = A.to_coo()
        c = c + (r // rows_per_block) * K
        A = tl.from_coo(r, c, v, A.dtype, A._nrows, nblocks * K)
    return A.mxm(B, op), layout, shape


def _matmul_expr(self, other, op):
    if (tensor := _as_tensor(other, "matmul")) is None:
        self._expect_type(other, Tensor, within="matmul", argname="other", op=op)
    other = tensor
    if self.ndim == 0 or other.ndim == 0:
        raise ValueError("matmul does not take 0-d operands; use ewise_mult or `*`")
    op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="matmul", argname="op")
    x = self.expand_dims(0) if self.ndim == 1 else self
    y = other.expand_dims(1) if other.ndim == 1 else other
    if x._shape[-1] != y._shape[-2]:
        raise DimensionMismatch(
            f"matmul: the last axis of {self._shape} does not match {other._shape}"
        )
    try:
        bshape = tuple(int(n) for n in np.broadcast_shapes(x._shape[:-2], y._shape[:-2]))
    except ValueError:
        raise DimensionMismatch(
            f"matmul: batch axes of {self._shape} and {other._shape} do not broadcast"
        ) from None
    full = (*bshape, x._shape[-2], y._shape[-1])
    drop = ((len(bshape),) if self.ndim == 1 else ()) + (
        (len(bshape) + 1,) if other.ndim == 1 else ()
    )
    shape = tuple(n for i, n in enumerate(full) if i not in drop)
    return TensorExpression(
        "matmul",
        _matmul_recipe,
        [self, other],
        (x, y, op, bshape, drop),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.matmul({1.name}, op={op})",
    )


def _matmul_recipe(updater, x, y, op, bshape, drop):
    nbatch = len(bshape)
    # Size-1 batch axes broadcast; drop them and remember each kept axis's output axis
    xs, x_out = _squeezed_batch(x, bshape)
    ys, y_out = _squeezed_batch(y, bshape)
    xb = {o: a for a, o in enumerate(x_out) if o < nbatch}
    yb = {o: a for a, o in enumerate(y_out) if o < nbatch}
    both = [o for o in range(nbatch) if o in xb and o in yb]
    a_batch = tuple(xb[o] for o in both)
    b_batch = tuple(yb[o] for o in both)
    kx, ky = xs.ndim - 1, ys.ndim - 2
    src, layout, _ = _contract(updater, xs, ys, op, (kx,), (ky,), a_batch, b_batch)
    # _contract numbers its output: batch (in `both` order), xs free axes, ys free axes
    free_x = [a for a in range(xs.ndim) if a != kx and a not in a_batch]
    free_y = [a for a in range(ys.ndim) if a != ky and a not in b_batch]
    to_out = list(both) + [x_out[a] for a in free_x] + [y_out[a] for a in free_y]
    layout = _renumber(layout, dict(enumerate(to_out)))
    full = (*bshape, x._shape[-2], y._shape[-1])
    present = set(to_out)
    layout = _insert_unit_axes(layout, [o for o in range(len(full)) if o not in present])
    if drop:
        keep = {o: i for i, o in enumerate(o for o in range(len(full)) if o not in drop)}
        layout = Layout(
            tuple(keep[o] for o in layout.rows if o in keep),
            tuple(keep[o] for o in layout.cols if o in keep),
        )
        full = tuple(n for i, n in enumerate(full) if i not in drop)
    _store(updater, src, full, layout)


def _squeezed_batch(t, bshape):
    """``t`` without its size-1 batch axes; the output axis of each remaining axis (the last
    two map to ``len(bshape)`` and ``len(bshape) + 1``).
    """
    nbatch = len(bshape)
    off = nbatch - (t.ndim - 2)
    drop = [a for a in range(t.ndim - 2) if t._shape[a] == 1]
    out = [a + off for a in range(t.ndim) if a not in drop]
    return (t.squeeze(drop) if drop else t), out


def _ttv(updater, x, w, axis, op, x_left):
    """TTV: ``x`` contracted with the 1-d ``w`` over ``axis``, by the cheapest of: one mxv
    (``axis`` is the column group), one vxm (the row group), a weighted selector (``axis`` is
    inside a group), each possibly after moving the split; or the general relayout and
    mxm (returns False for that).
    """
    mult = op.binaryop
    flip = getattr(mult, "commutes_to", None)
    # (x, w) is the order the multiply must see: mxv and a column selector give (x, w), vxm
    # and a row selector give (w, x)
    xw_op = mult if x_left else flip
    wx_op = flip if x_left else mult
    M, layout = x._M, x._layout
    seq, p = layout.state
    n = len(seq)
    nv = M._nvals
    nthreads = _nthreads(updater)
    ti = _ti(nthreads)
    options = []  # (cost, kind, split)
    if layout.cols == (axis,) and xw_op is not None:
        # One pass with no data movement; the planner's relayout estimates run low, so do
        # not let them compete (round 1: relayout + mxm 56 ms vs vxm 36 ms)
        options.append((0.0, "mxv", p))
    elif layout.rows == (axis,) and wx_op is not None and M._ncols <= 4 * max(nv, 1):
        # vxm writes into the whole column group: fine up to a few times the data (round 1
        # repro: 2.6x, 36 ms vs 56), slow into a hypersparse one (hyper 2^38: 87 ms vs 23)
        options.append((0.0, "vxm", p))
    for q in range(n + 1) if not options else ():
        if not tl._valid_split(x._shape, seq, q):
            continue
        move = 0.0 if q == p else _rs_cost(x._shape, seq, q, nv, ti)
        rows, cols = seq[:q], seq[q:]
        if cols == (axis,) and xw_op is not None:
            options.append((move + C_MXV[ti] * nv, "mxv", q))
        elif rows == (axis,) and wx_op is not None and tl.prod(x._shape, cols) <= 4 * nv:
            options.append((move + C_VXM[ti] * nv, "vxm", q))
        elif q == p and axis in rows and len(rows) > 1 and wx_op is not None:
            options.append((_row_selector_cost(M, tl.prod(x._shape, rows), ti), "rowsel", q))
        elif q == p and axis in cols and len(cols) > 1 and xw_op is not None:
            cost = _shape_selector_cost(nv, tl.prod(x._shape, cols), ti) + C_DIAG[ti] * nv
            options.append((cost, "colsel", q))
    others = tuple(a for a in range(n) if a != axis)
    targets = frozenset(
        ((*o, axis), n - 1)
        for o in (itertools.permutations(others) if n <= tl.MAX_PLANNED_NDIM else [others])
        if tl._valid_split(x._shape, (*o, axis), n - 1)
    )
    if targets and (not options or options[0][0] > 0):
        general = tl.plan(x._shape, layout.state, targets, nv, nthreads)[0] + C_MXV[ti] * nv
        options.append((general, "general", None))
    if not options:
        return False
    _, kind, q = min(options, key=lambda o: o[0])
    if kind == "general":
        return False
    if q != p:
        M = M.ss.reshape(tl.prod(x._shape, seq[:q]), tl.prod(x._shape, seq[q:]))
    rows, cols = seq[:q], seq[q:]
    shape = tuple(m for a, m in enumerate(x._shape) if a != axis)
    renum = {a: i for i, a in enumerate(a for a in range(n) if a != axis)}
    opt_nthreads = updater.opts.get("nthreads")
    wv = _in_layout(w, Layout((), (0,)), opt_nthreads)[0, :].new()
    if kind in {"mxv", "vxm"}:
        v = Vector(op.return_type, M._nrows if kind == "mxv" else M._ncols)
        if kind == "mxv":
            sr = op if xw_op is mult else get_semiring(op.monoid, xw_op)
            v(**updater.opts) << M.mxv(wv, sr)
            out = Layout((), tuple(renum[a] for a in rows))
        else:
            sr = op if wx_op is mult else get_semiring(op.monoid, wx_op)
            v(**updater.opts) << wv.vxm(M, sr)
            out = Layout((), tuple(renum[a] for a in cols))
        _store(updater, tl.steal_vector_as_row(v), shape, out, fresh=True)
        return True
    group = rows if kind == "rowsel" else cols
    gshape = tuple(x._shape[g] for g in group)
    pos = group.index(axis)
    kept_pos = tuple(i for i in range(len(group)) if i != pos)
    if kind == "rowsel" and M.ss.format == "hypercsr":
        # A weighted selector over the nonempty rows only: S[kept, r] = w[axis coordinate]
        idx, vals = wv.to_coo()
        dense = np.zeros(x._shape[axis], vals.dtype)
        present = np.zeros(x._shape[axis], bool)
        dense[idx] = vals
        present[idx] = True
        Sw = tl.rows_selector(M, x._shape, rows, kept_pos, weights=(dense, present, pos))
    else:
        # S[kept, g] = w[axis coordinate of g] over the positions g of the group
        e = Vector(wv.dtype, tl.prod(x._shape, group))
        St = tl.selector(gshape, (pos,), transposed=True)
        e << St.mxv(wv, semiring.lor_second if wv.dtype == BOOL else semiring.plus_second)
        if kind == "rowsel":
            S = tl.selector(gshape, kept_pos)
            Sw = tl.new_matrix(e.dtype, S._nrows, S._ncols)
            Sw << S.mxm(e.diag(), semiring.any_second)
        else:
            S = tl.selector(gshape, kept_pos, transposed=True)
            Sw = tl.new_matrix(e.dtype, S._nrows, S._ncols)
            Sw << e.diag().mxm(S, semiring.any_first)
    if kind == "rowsel":
        src = Sw.mxm(M, get_semiring(op.monoid, wx_op))
        out = Layout(tuple(renum[a] for a in rows if a != axis), tuple(renum[a] for a in cols))
    else:
        src = M.mxm(Sw, get_semiring(op.monoid, xw_op))
        out = Layout(tuple(renum[a] for a in rows), tuple(renum[a] for a in cols if a != axis))
    _store(updater, src, shape, out)
    return True


def _tensordot_recipe(updater, a, b, op, a_axes, b_axes):
    if len(a_axes) == 1 and (b.ndim == 1 or a.ndim == 1) and a.ndim + b.ndim > 2:
        x_left = b.ndim == 1
        x, w, axis = (a, b, a_axes[0]) if x_left else (b, a, b_axes[0])
        if _ttv(updater, x, w, axis, op, x_left):
            return
    if (rv := _ttm_engines(updater, a, b, op, a_axes, b_axes)) is not None:
        src, layout, shape = rv
        _store(updater, src, shape, layout, fresh=True)
        return
    src, layout, shape = _contract(updater, a, b, op, a_axes, b_axes)
    if not layout.cols and layout.rows and _fixed_layout(updater) is None:
        M, layout = _healthy(_materialize(src), shape, layout)
        _store(updater, M, shape, layout, fresh=True)
    else:
        _store(updater, src, shape, layout)


# The TTM engines compile per R on first use, so small tensors keep the product
_TTM_KERNEL_MIN_NNZ = 2**16
# The TTM engines, tried in this order where each runs (all are kept, to measure; tests and
# benchmarks set this to force some):
#   "udt"        one mxv with FP64[R] values (tensor_udt.ttm; JIT on, FP64 plus_times)
#   "numba"      the Numba kernel (tensor_numba.ttm; FP64 plus_times)
#   "compacted"  one product on the nonempty rows (_ttm_compacted; stored order, any op)
#   "products"   one product after a relayout (_contract), which runs every TTM, so it is
#                also what any list falls back to
_TTM_ENGINES = ("udt", "numba", "compacted", "products")
# Where both engines run, the UDT one is faster only in the stored order with a hypersparse
# output and R up to 32 (NELL-2 mode 2: 77 / 139 / 329 ms against 92 / 174 / 354 at R = 8 /
# 16 / 32, 1 thread; 14 / 24 / 55 against 18 / 31 / 62 at 8); slower with a bitmap output
# (u7: 25-86 against 27-77 ms at R = 16, 1 thread, and 1.4-1.8x slower at 8), at R = 64,
# and after a relayout (NELL-2 modes 0 and 1: 1.6-1.8x slower). Phase 5c measurements.
_TTM_UDT_MAX_R = 32


def _ttm_engines(updater, a, b, op, a_axes, b_axes):
    """A TTM by the first engine of ``_TTM_ENGINES`` before "products" that runs here:
    ``(src, layout, shape)``, or None for the products (``_contract``).
    """
    kernels = (
        len(a_axes) == 1
        and b.ndim == 2
        and a.ndim >= 2
        and op.name == "plus_times"
        and op.return_type == FP64
        and a.dtype == FP64
        and b.dtype == FP64
        and a._M._nvals >= max(_TTM_KERNEL_MIN_NNZ, 1)
        and b._M._nvals == math.prod(b._shape)
    )
    for engine in _TTM_ENGINES:
        if engine == "products":
            break
        if engine == "compacted":
            run = _ttm_compacted
        elif not kernels:
            continue
        else:
            run = _ttm_udt if engine == "udt" else _ttm_numba
        if (rv := run(updater, a, b, op, a_axes, b_axes)) is not None:
            return rv
    return None


def _ttm_factor(b, kb):
    """The factor of a TTM as a dense C-contiguous (K x R) float64 array, K along its axis
    ``kb`` (the contracted one); a new array.
    """
    Bm = b._M if b._layout.rows == (0,) else tl.transpose(b._M)  # (b's axis 0 | axis 1)
    U = np.ascontiguousarray(Bm.to_dense(fill_value=0.0))
    if kb == 1:
        U = np.ascontiguousarray(U.T)
    return U


def _ttm_udt(updater, a, b, op, a_axes, b_axes):
    """TTM by one ``mxv`` with FP64[R] values (``tensor_udt.ttm``), when the JIT is on:
    ``(src, layout, shape)`` as from ``_contract``, or None. The contracted axis must be the
    column group by itself, so each nonempty row is an output fiber; the tensor moves to the
    cheapest layout that has it so. Where the Numba engine runs too, this one takes only
    the cases it is faster on (see ``_TTM_UDT_MAX_R``).
    """
    from . import tensor_udt

    before = tensor_udt.jit_control()
    if not tensor_udt.jit_ready():
        return None
    m, kb = a_axes[0], b_axes[0]
    free_a = tuple(x for x in range(a.ndim) if x != m)
    q = len(free_a)
    targets = frozenset(
        ((*perm, m), q)
        for perm in (itertools.permutations(free_a) if q <= 4 else [free_a])
        if tl._valid_split(a._shape, (*perm, m), q)
    )
    if not targets:
        return None
    nthreads = _nthreads(updater)
    _, path, state = tl.plan(a._shape, a._layout.state, targets, a._M._nvals, nthreads)
    if (
        _has_numba
        and "numba" in _TTM_ENGINES
        and (
            path
            or b._shape[1 - kb] > _TTM_UDT_MAX_R
            or 2 * tl.count_nonempty_rows(a._M) >= a._M._nrows
        )
    ):
        return None  # (the Numba engine is faster here: see _TTM_UDT_MAX_R)
    layout = Layout.from_state(state)
    from .descriptor import lookup

    def compute():
        # (the relayout too: under the raised control it may compile)
        X = _in_layout(a, layout, updater.opts.get("nthreads"))
        return tensor_udt.ttm(X, _ttm_factor(b, kb), lookup(**updater.opts))

    if (src := tensor_udt.run(before, compute)) is None:
        return None
    out = {x: i for i, x in enumerate(free_a)}
    shape = (*(a._shape[x] for x in free_a), src._ncols)
    return src, Layout(tuple(out[x] for x in layout.rows), (len(shape) - 1,)), shape


def _ttm_numba(updater, a, b, op, a_axes, b_axes):
    """TTM (an FP64 tensor's axis contracted with a full FP64 factor, plus_times) by a Numba
    kernel over the stored arrays (``tensor_numba.ttm``): ``(src, layout, shape)`` as from
    ``_contract``, or None to use the product. The contracted axis must be the last column
    axis; the tensor stays as it is when it is (with leading column axes too: their runs are
    the output fibers), else it moves to the cheapest layout that has it there.

    The output is today's storage: hypersparse rows of R values, or full when every output
    fiber exists. (One mxm was 2.4-2.7x behind TACO in the stored order: its Gustavson saxpy
    on hypersparse rows, then the output's row hash; DESIGN_phase5 1.1.)
    """
    if not _has_numba or len(a_axes) != 1 or b.ndim != 2 or a.ndim < 2:
        return None
    if op.name != "plus_times" or op.return_type != FP64 or a.dtype != FP64 or b.dtype != FP64:
        return None
    if a._M._nvals < max(_TTM_KERNEL_MIN_NNZ, 1) or b._M._nvals != math.prod(b._shape):
        return None
    m, kb = a_axes[0], b_axes[0]
    free_a = tuple(x for x in range(a.ndim) if x != m)
    nthreads = _nthreads(updater)
    targets = frozenset(
        ((*perm, m), q)
        for perm in (itertools.permutations(free_a) if len(free_a) <= 4 else [free_a])
        for q in range(1, len(free_a) + 1)  # (with no row axis there is one row: no threads)
        if tl._valid_split(a._shape, (*perm, m), q)
    )
    if not targets:
        return None
    _, _, state = tl.plan(a._shape, a._layout.state, targets, a._M._nvals, nthreads)
    layout = Layout.from_state(state)
    X = _in_layout(a, layout, updater.opts.get("nthreads"))
    if X.ss.format not in {"csr", "hypercsr"}:
        return None
    from . import tensor_numba

    U = _ttm_factor(b, kb)
    R = U.shape[1]
    out = {x: i for i, x in enumerate(free_a)}
    rows = tuple(out[x] for x in (*layout.rows, *layout.cols[:-1]))
    shape = (*(a._shape[x] for x in free_a), R)
    nrows = tl.prod(shape, rows)
    with contextlib.ExitStack() as stack:
        cont = stack.enter_context(tl._Unloaded(X))
        p, h, ci = (stack.enter_context(tl._VectorArray(v)) for v in (cont.p, cont.h, cont.i))
        xv = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        kind, *parts = tensor_numba.ttm(
            (p.array, np.empty(0, np.int64) if h.array is None else h.array, ci.array, xv.array,
             bool(cont.iso)),
            tuple(a._shape[x] for x in layout.cols), U, nrows,
            updater.opts.get("nthreads") or nthreads,
        )  # fmt: skip
    # The parts are handed over, not copied or checked
    if kind == "full":
        src = tl.matrix_from_arrays(FP64, nrows, R, "fullr", x=parts[0], nvals=nrows * R)
    elif kind == "bitmap":
        V, B, nvals = parts
        src = tl.matrix_from_arrays(FP64, nrows, R, "bitmapr", x=V, b=B, nvals=nvals)
    else:
        hout, indptr, cols, V = parts
        src = tl.matrix_from_arrays(
            FP64, nrows, R, "hypercsr", x=V, h=hout, p=indptr, i=cols, nvals=V.size
        )
    return src, Layout(rows, (len(shape) - 1,)), shape


def _ttm_compacted(updater, a, b, op, a_axes, b_axes):
    """TTM in the stored order on a hypersparse tensor, by one product on its rows compacted:
    the nonempty rows (h dropped through the Container) times the factor, then the result's
    rows given h back. On hypersparse rows SuiteSparse runs a Gustavson saxpy; on the same
    rows stored compactly its bitmap / full kernel, with no index per output entry (NELL-2
    mode 2: 404-430 against 689-769 ms at 1 thread, DESIGN_phase5 1.1). Any type and
    semiring; ``(src, layout, shape)`` or None.
    """
    if len(a_axes) != 1 or b.ndim != 2 or a._layout.cols != a_axes or not a._layout.rows:
        return None
    M = a._M
    if M.ss.format != "hypercsr" or M._nvals == 0:
        return None
    nonempty = tl.count_nonempty_rows(M)
    if 2 * nonempty > M._nrows:
        return None  # (few rows to drop)
    kb = b_axes[0]
    Bm = b._M if b._layout.rows == (kb,) else tl.transpose(b._M)  # (b's axis kb | the other)
    A = M.dup()
    with tl._Unloaded(A) as cont:
        h, cont.h = cont.h, tl.new_vector_handle(cont.h)
        cont.nrows = nonempty
        cont.format = lib.GxB_SPARSE
    C = tl.new_matrix(op.return_type, nonempty, Bm._ncols)
    C(**updater.opts) << A.mxm(Bm, op)
    tl.give_rows(C, h, M._nrows)
    free_a = tuple(x for x in range(a.ndim) if x != a_axes[0])
    out = {x: i for i, x in enumerate(free_a)}
    shape = (*(a._shape[x] for x in free_a), b._shape[1 - kb])
    return C, Layout(tuple(out[x] for x in a._layout.rows), (len(shape) - 1,)), shape


def _as_factor(F, nrows, within):
    """A factor matrix (Matrix or ndarray) with ``nrows`` rows, as a Matrix."""
    if isinstance(F, np.ndarray):
        if F.ndim != 2:
            raise ValueError(f"{within}: factors must be 2-d; got shape {F.shape}")
        F = Matrix.from_dense(F)
    elif output_type(F) is Tensor:
        F = _compute(F, within).to_matrix()
    elif type(F) is not Matrix:
        raise TypeError(f"{within}: a factor must be a Matrix or ndarray, not {type(F)}")
    if F._nrows != nrows:
        raise DimensionMismatch(f"{within}: a factor has {F._nrows} rows; expected {nrows}")
    return F


def _mttkrp_expr(self, factors, mode, op):
    mode = _normalize_axis(mode, self.ndim)
    if self.ndim < 2:
        raise ValueError("mttkrp needs at least 2 dimensions")
    if len(factors) != self.ndim:
        raise ValueError(f"mttkrp needs {self.ndim} factors (one per axis); got {len(factors)}")
    others = [a for a in range(self.ndim) if a != mode]
    mats = {a: _as_factor(factors[a], self._shape[a], "mttkrp") for a in others}
    ranks = {M._ncols for M in mats.values()}
    if len(ranks) != 1:
        raise DimensionMismatch(f"mttkrp: factors have different ranks {sorted(ranks)}")
    R = ranks.pop()
    op = get_typed_op(op, self.dtype, mats[others[0]].dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="mttkrp", argname="op")
    return TensorExpression(
        "mttkrp",
        _mttkrp_recipe,
        [self],
        (self, mats, mode, R, op),
        dtype=op.return_type,
        shape=(self._shape[mode], R),
        op=op,
        expr_repr="{0.name}.mttkrp(mode=%d, op={op})" % mode,  # noqa: UP031
    )


# The kernels compile per R on first use (about 0.3 s each), so small tensors keep the products
_MTTKRP_KERNEL_MIN_NNZ = 2**16


def _mttkrp_numba(x, mats, mode, nthreads, *, layout=None):
    """MTTKRP of an FP64 tensor of 3 or more axes with full FP64 factors and plus_times by a
    Numba kernel over the stored arrays (``tensor_numba``) in ``layout`` (default: as stored),
    as a Matrix; or None to use the GraphBLAS products: no Numba, fewer axes, another type or
    semiring, sparse factors, an empty group, bitmap / full storage, or per-thread buffers too
    large for this layout.
    """
    if not _has_numba or x.ndim < 3 or x._M._nvals < max(_MTTKRP_KERNEL_MIN_NNZ, 1):
        return None
    layout = x._layout if layout is None else layout
    if not layout.rows or not layout.cols or x.dtype != FP64:
        return None
    for M in mats.values():
        if M.dtype != FP64 or M._nvals != M._nrows * M._ncols:
            return None
    X = _in_layout(x, layout, nthreads)
    if X.ss.format not in {"csr", "hypercsr"}:
        return None
    from . import tensor_numba

    order = (*layout.rows, *layout.cols)
    R = next(iter(mats.values()))._ncols
    nnz = X._nvals
    with contextlib.ExitStack() as stack:
        F = {}
        views = {}
        for ax, M in mats.items():
            if (arr := views.get(id(M))) is None:
                arr = views[id(M)] = _factor_array(M, stack)
            F[ax] = arr
        Fs = [F[ax] if ax != mode else np.empty((0, R)) for ax in order]
        cont = stack.enter_context(tl._Unloaded(X))
        p, h, ci = (stack.enter_context(tl._VectorArray(v)) for v in (cont.p, cont.h, cont.i))
        xv = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        rv = tensor_numba.mttkrp_nd(
            (p.array, np.empty(0, np.int64) if h.array is None else h.array, ci.array, xv.array,
             bool(cont.iso)),
            tuple(x._shape[a] for a in layout.rows),
            tuple(x._shape[a] for a in layout.cols),
            order.index(mode), Fs, x._shape[mode], nthreads,
            # Per-thread output buffers up to about the data's size
            max_buffer=max(R * nnz, 2**22),
        )  # fmt: skip
    if rv is None:
        return None
    out, touched = rv
    if touched.all():
        return Matrix.ss.import_fullr(values=out, take_ownership=True)
    # The reached rows only, each full: hypersparse arrays handed over (a build of 7e6
    # entries took 35 ms on hyper, against 12 for the kernel)
    rows = np.flatnonzero(touched)
    n = rows.size
    return Matrix.ss.import_hypercsr(
        nrows=out.shape[0], ncols=R, rows=rows, indptr=np.arange(0, (n + 1) * R, R),
        col_indices=np.tile(np.arange(R), n), values=out[rows].ravel(), sorted_cols=True,
        take_ownership=True,
    )  # fmt: skip


def _factor_array(M, stack):
    """A full FP64 factor as an (n x R) array: a view of its values through the Container
    while ``stack`` is open (Enron's 244268 x 64 factor is 125 MB to copy), else a copy.
    """
    nrows, ncols = M._nrows, M._ncols  # (M reads as empty while unloaded)
    if M.ss.format == "fullr" and not M.ss.is_iso and nrows * ncols > 0:
        cont = stack.enter_context(tl._Unloaded(M))
        x = stack.enter_context(tl._VectorArray(cont.x, np.float64))
        if x.array is not None and x.array.size == nrows * ncols:
            return x.array.reshape(nrows, ncols)
        # (not expected: a full Matrix holds nrows * ncols values)
    return np.ascontiguousarray(M.to_dense(fill_value=0.0))


# The products' intermediates cost about this much per (row x R) entry, ns at 8 / 1 threads
# (Enron mode 1, R = 64: 0.92 s for a first intermediate of 567,936 rows; 26.9 s for 31.3e6)
C_MTTKRP_INTER = (2.0, 13.0)


def _mttkrp_order(x, mode, R, nthreads, *, products):
    """The order of the axes other than ``mode`` for the layout ``(mode, *order[:-1] |
    order[-1])``: by the cost to reach it, plus with ``products`` the size of the
    intermediates the products make (one row per distinct ``(mode, *order[:-k])``, k = 1, 2,
    ...), estimated from a sample of the entries. Relayout cost alone picked, for Enron mode 1
    at 1 thread, a first intermediate 55x larger than at 8 threads (26.9 s against 0.92).
    """
    others = [a for a in range(x.ndim) if a != mode]
    orders = list(itertools.permutations(others)) if len(others) <= 4 else [tuple(others)]
    orders = [order for order in orders if tl._valid_split(x._shape, (mode, *order), len(order))]
    nnz = x._M._nvals
    fibers = {}
    if products and len(orders) > 1 and nnz > 0:
        groups = {(mode, *order[:k]) for order in orders for k in range(1, len(order))}
        fibers = _sampled_fibers(x, groups)
    ti = 0 if nthreads > 1 else 1
    best = None
    for order in orders:
        dst = Layout((mode, *order[:-1]), (order[-1],))
        cost = tl.relayout_cost(x._shape, x._layout, dst, nnz, nthreads)
        for k in range(1, len(order)):
            cost += C_MTTKRP_INTER[ti] * R * fibers.get((mode, *order[:k]), 0.0)
        if best is None or cost < best[0]:
            best = (cost, order)
    return best[1]


def _sampled_fibers(x, groups):
    """Estimated distinct values of each group of axes among ``x``'s entries, from a sample of
    2^14 stored entries (``_sampled_nonempty_rows``); uniform occupancy for bitmap or full
    storage.
    """
    M = x._M
    nnz = M._nvals
    rows, cols = x._layout
    if M.ss.format not in {"csr", "hypercsr"}:
        return {g: tl.expected_rows(tl.prod(x._shape, g), nnz) for g in groups}
    q = np.sort(np.random.default_rng(0).integers(0, nnz, min(nnz, _SPLIT_SAMPLE)))
    with (
        tl._Unloaded(M) as cont,
        tl._VectorArray(cont.p) as p,
        tl._VectorArray(cont.h) as h,
        tl._VectorArray(cont.i) as ci,
    ):
        slot = np.searchsorted(p.array, q.astype(p.array.dtype), side="right") - 1
        r = slot if h.array is None else h.array[slot]
        c = ci.array[q]
    coords = {}
    tl.unravel(r.astype(np.int64), x._shape, rows, coords)
    tl.unravel(c.astype(np.int64), x._shape, cols, coords)
    coords = [coords[a] for a in range(x.ndim)]
    sample = np.arange(q.size)
    return {g: _sampled_nonempty_rows(coords, x._shape, g, nnz, sample) for g in groups}


def _krp_embedding(F, R):
    """W with ``W[j * R + r, r] = F[j, r]``: one product with it sums over j for each r."""
    j, r, v = F.to_coo()
    j = j.astype(np.int64)
    r = r.astype(np.int64)
    return tl.from_coo(j * R + r, r, v, F.dtype, F._nrows * R, R)


# The MTTKRP engines, tried in this order where each runs (all are kept, to measure; tests
# and benchmarks set this to force some):
#   "numba"     the Numba kernel (tensor_numba.mttkrp_nd; FP64 plus_times)
#   "udt"       the products, their first step (a TTM over the last axis) by the array-UDT
#               engine (tensor_udt.ttm; JIT on, FP64 plus_times, a full factor)
#   "products"  the GraphBLAS products (two-mxm for 3-d), which run every MTTKRP, so they
#               are also what any list falls back to
# Phase 5c, "udt" against "products" (Numba off), R = 16: NELL-2 mode 0 702 -> 142 ms at 1
# thread and 142 -> 29 at 8; u7 82-122 -> 38-77 at 1, the same at 8; NELL-2 modes 1 and 2
# within 3% at 1 thread, 7-13% slower at 8 (load rising to 3.5).
_MTTKRP_ENGINES = ("numba", "udt", "products")


def _mttkrp_recipe(updater, x, mats, mode, R, op):
    nthreads = updater.opts.get("nthreads")
    kernel = op.name == "plus_times" and op.return_type == FP64
    numba_kernel = kernel and "numba" in _MTTKRP_ENGINES
    if (
        numba_kernel
        and (T := _mttkrp_numba(x, mats, mode, nthreads or _nthreads(updater))) is not None
    ):
        # In the stored layout: no relayout
        _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)
        return
    # Any order of the other axes works. For the kernel, take the one whose layout is
    # cheapest to reach
    order = _mttkrp_order(x, mode, R, _nthreads(updater), products=False)
    if numba_kernel:
        dst = tl.as_layout(((mode, *order[:-1]), (order[-1],)), x._shape)
        T = _mttkrp_numba(x, mats, mode, nthreads or _nthreads(updater), layout=dst)
        if T is not None:
            # Relayout, then the kernel that owns its output rows
            _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)
            return
    # For the products, count their intermediates too
    others = list(_mttkrp_order(x, mode, R, _nthreads(updater), products=True))
    last = others[-1]
    rest = tuple(others[:-1])
    dst = tl.as_layout(((mode, *rest), (last,)), x._shape)
    X = _in_layout(x, dst, nthreads)
    # 1. TTM over the last axis: (mode, rest | last) @ F_last -> (mode, rest | r), by the
    # UDT engine where it runs (see _MTTKRP_ENGINES)
    T = None
    if (
        kernel
        and "udt" in _MTTKRP_ENGINES
        and x.dtype == FP64
        and mats[last]._nvals == mats[last]._nrows * R
    ):
        from . import tensor_udt
        from .descriptor import lookup

        before = tensor_udt.jit_control()
        if X._nvals >= _TTM_KERNEL_MIN_NNZ and tensor_udt.jit_ready():
            U = np.ascontiguousarray(mats[last].to_dense(fill_value=0.0), np.float64)
            T = tensor_udt.run(before, tensor_udt.ttm, X, U, lookup(**updater.opts))
    if T is None:
        T = tl.new_matrix(op.return_type, X._nrows, R)
        T(**updater.opts) << X.mxm(mats[last], op)
    # 2. Each remaining axis: move it next to r, then one product with its embedding
    rows = (mode, *rest)
    for a in reversed(rest):
        rows = rows[:-1]
        nrows, ncols = tl.prod(x._shape, rows), x._shape[a] * R
        if nrows <= 2**31 and ncols <= 2**31:
            T.ss.reshape(nrows, ncols, inplace=True)
        else:
            # SuiteSparse 10.5.1's in-place reshape cannot widen 32-bit indices
            T = T.ss.reshape(nrows, ncols)
        out = tl.new_matrix(op.return_type, T._nrows, R)
        out(**updater.opts) << T.mxm(_krp_embedding(mats[a], R), op)
        T = out
    _store(updater, T, (x._shape[mode], R), Layout((0,), (1,)), fresh=True)


# ---------------------------------------------------------------------------------------------
# einsum


def einsum(subscripts, *operands, op=semiring.plus_times, optimize="greedy"):
    """Einstein summation over sparse Tensors with a semiring (numpy.einsum subscripts).

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.**

    ``op``'s multiply combines the operands and its monoid sums over the contracted indices;
    a result position is present when at least one term exists. Operands may be Tensors,
    Matrices, Vectors or ndarrays. Each index appears at most once per operand (no
    diagonals) and ``...`` is not supported. With three or more operands the pairwise order
    comes from ``opt_einsum.contract_path`` when it is installed (else a greedy choice by
    result size); the MTTKRP pattern (``'ijk,jr,kr->ir'``) goes to ``Tensor.mttkrp`` instead,
    which never forms the Khatri-Rao product.
    """
    if "." in subscripts:
        raise NotImplementedError("einsum: `...` is not supported")
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        inputs, output = subscripts.split("->")
    else:
        inputs, output = subscripts, None
    terms = inputs.split(",")
    if len(terms) != len(operands):
        raise ValueError(f"einsum: {len(terms)} terms for {len(operands)} operands")
    tensors = []
    for term, x in zip(terms, operands, strict=True):
        if (t := _as_tensor(x, "einsum")) is None:
            raise TypeError(f"einsum operands must be Tensor-like, not {type(x)}")
        if len(term) != t.ndim:
            raise ValueError(f"einsum: term {term!r} does not match an operand of shape {t._shape}")
        if len(set(term)) != len(term):
            raise NotImplementedError(f"einsum: repeated index in {term!r} (a diagonal)")
        tensors.append(t)
    sizes = {}
    for term, t in zip(terms, tensors, strict=True):
        for c, n in zip(term, t._shape, strict=True):
            if sizes.setdefault(c, n) != n:
                raise DimensionMismatch(f"einsum: index {c!r} has sizes {sizes[c]} and {n}")
    if output is None:
        counts = {}
        for term in terms:
            for c in term:
                counts[c] = counts.get(c, 0) + 1
        output = "".join(sorted(c for c, k in counts.items() if k == 1))
    if len(set(output)) != len(output) or any(c not in sizes for c in output):
        raise ValueError(f"einsum: bad output {output!r}")
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = unify(dtype, t.dtype)
    typed = get_typed_op(op, dtype, dtype, kind="semiring")
    if typed.opclass != "Semiring":
        raise TypeError(f"einsum needs a Semiring, not {typed.opclass}")
    return TensorExpression(
        "einsum",
        _einsum_recipe,
        tensors,
        (terms, output, tensors, typed, optimize),
        dtype=typed.return_type,
        shape=tuple(sizes[c] for c in output),
        op=typed,
        expr_repr=f"einsum({subscripts!r}, ...)".replace("{", "{{").replace("}", "}}"),
    )


def _mttkrp_pattern(terms, output, tensors):
    """``(index of X, factors, mode, rank first)`` when the einsum is an MTTKRP: one operand X
    of 3+ axes, and 2-d factors that each pair one index of X with a common rank index r,
    covering every index of X but the mode's, with output (mode, r) or (r, mode).
    """
    big = [i for i, t in enumerate(terms) if len(t) >= 3]
    if len(terms) < 3 or len(output) != 2 or len(big) != 1:
        return None
    xi = big[0]
    xterm = terms[xi]
    mats = [None] * len(xterm)
    ranks = set()
    for i, term in enumerate(terms):
        if i == xi:
            continue
        inside = [c for c in term if c in xterm]
        if len(term) != 2 or len(inside) != 1:
            return None
        r = term[1] if term[0] == inside[0] else term[0]
        ranks.add(r)
        ax = xterm.index(inside[0])
        if mats[ax] is not None:
            return None
        F = tensors[i].to_matrix()
        mats[ax] = F if term[0] == inside[0] else F.T.new()
    missing = [ax for ax, F in enumerate(mats) if F is None]
    if len(ranks) != 1 or len(missing) != 1:
        return None
    r = ranks.pop()
    mode = missing[0]
    if set(output) != {xterm[mode], r}:
        return None
    return xi, mats, mode, output[0] == r


def _einsum_recipe(updater, terms, output, tensors, op, optimize):
    if (mt := _mttkrp_pattern(terms, output, tensors)) is not None:
        xi, mats, mode, transposed = mt
        R = tensors[xi].mttkrp(mats, mode, op).new(**updater.opts)
        result, letters = R, (output[1] + output[0]) if transposed else output
    else:
        ops = list(zip(terms, tensors, strict=True))
        path = _einsum_path(terms, output, tensors, optimize)
        for i, j in path:
            (ta, A), (tb, B) = ops[i], ops[j]
            for k in sorted((i, j), reverse=True):
                del ops[k]
            keep = set(output).union(*(t for t, _ in ops))
            ops.append(_einsum_pair(ta, A, tb, B, keep, op, updater.opts))
        ((letters, result),) = ops
        extra = [k for k, c in enumerate(letters) if c not in output]
        if extra:
            result = result.reduce(op.monoid, axis=tuple(extra)).new(**updater.opts)
            letters = "".join(c for c in letters if c in output)
    final = result.permute_dims(tuple(letters.index(c) for c in output))
    _store(updater, final._M, final._shape, final._layout, share=final)


def _einsum_path(terms, output, tensors, optimize):
    n = len(terms)
    if n == 1:
        return []
    if n > 2 and optimize:
        try:
            import opt_einsum
        except ImportError:
            pass
        else:
            sub = ",".join(terms) + "->" + output
            shapes = [t._shape for t in tensors]
            return [tuple(p) for p in opt_einsum.contract_path(sub, *shapes, shapes=True,
                                                               optimize=optimize)[0]]  # fmt: skip
    # Greedy: contract the pair with the smallest result (product of its kept index sizes)
    path = []
    cur = list(terms)
    sizes = {}
    for term, t in zip(terms, tensors, strict=True):
        sizes.update(zip(term, t._shape, strict=True))
    while len(cur) > 1:
        best = None
        for i, j in itertools.combinations(range(len(cur)), 2):
            rest = set(output).union(*(cur[k] for k in range(len(cur)) if k not in (i, j)))
            letters = (set(cur[i]) | set(cur[j])) & rest
            cost = math.prod(sizes[c] for c in letters)
            if best is None or cost < best[0]:
                best = (cost, i, j, "".join(sorted(letters)))
        _, i, j, letters = best
        path.append((i, j))
        cur = [cur[k] for k in range(len(cur)) if k not in (i, j)] + [letters]
    return path


def _einsum_pair(ta, A, tb, B, keep, op, opts):
    """Contract one pair; returns (letters, Tensor)."""
    # Indices that appear in only one operand and nowhere later are summed out first
    only = tuple(k for k, c in enumerate(ta) if c not in tb and c not in keep)
    if only:
        A = A.reduce(op.monoid, axis=only).new(**opts)
        ta = "".join(c for k, c in enumerate(ta) if k not in only)
    only = tuple(k for k, c in enumerate(tb) if c not in ta and c not in keep)
    if only:
        B = B.reduce(op.monoid, axis=only).new(**opts)
        tb = "".join(c for k, c in enumerate(tb) if k not in only)
    shared = [c for c in ta if c in tb]
    batch = [c for c in shared if c in keep]
    contracted = [c for c in shared if c not in keep]
    free_a = [c for c in ta if c not in tb]
    free_b = [c for c in tb if c not in ta]
    if contracted:
        a_axes = tuple(ta.index(c) for c in contracted)
        b_axes = tuple(tb.index(c) for c in contracted)
        a_batch = tuple(ta.index(c) for c in batch)
        b_batch = tuple(tb.index(c) for c in batch)
        if batch:
            R = _batched_contract_expr(A, B, op, a_axes, b_axes, a_batch, b_batch).new(**opts)
        else:
            R = A.tensordot(B, (a_axes, b_axes), op).new(**opts)
        return "".join(batch + free_a + free_b), R
    # No contraction: an element-wise product, broadcasting over the free indices
    letters = batch + free_a + free_b
    pa = A.permute_dims(tuple(ta.index(c) for c in batch + free_a))
    pb = B.permute_dims(tuple(tb.index(c) for c in batch + free_b))
    if free_b:
        pa = pa.expand_dims(tuple(range(len(batch) + len(free_a), len(letters))))
    if free_a:
        pb = pb.expand_dims(tuple(range(len(batch), len(batch) + len(free_a))))
    R = pa.ewise_mult(pb, op.binaryop).new(**opts)
    return "".join(letters), R


def _batched_contract_expr(a, b, op, a_axes, b_axes, a_batch, b_batch):
    free_a = [x for x in range(a.ndim) if x not in a_axes and x not in a_batch]
    free_b = [y for y in range(b.ndim) if y not in b_axes and y not in b_batch]
    shape = (
        tuple(a._shape[x] for x in a_batch)
        + tuple(a._shape[x] for x in free_a)
        + tuple(b._shape[y] for y in free_b)
    )
    return TensorExpression(
        "contract",
        _batched_contract_recipe,
        [a, b],
        (a, b, op, a_axes, b_axes, a_batch, b_batch),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.contract({1.name}, op={op})",
    )


def _batched_contract_recipe(updater, a, b, op, a_axes, b_axes, a_batch, b_batch):
    src, layout, shape = _contract(updater, a, b, op, a_axes, b_axes, a_batch, b_batch)
    _store(updater, src, shape, layout)


# ---------------------------------------------------------------------------------------------
# Indexing (outer semantics)


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
    """An int (axis dropped) or a sorted-or-not int64 array of coordinates (axis kept)."""
    if isinstance(index, (int, np.integer)):
        i = int(index)
        if not -size <= i < size:
            raise IndexOutOfBound(f"index {i} is out of bounds for axis {axis} of size {size}")
        return i + size if i < 0 else i
    if isinstance(index, slice):
        return np.arange(*index.indices(size), dtype=np.int64)
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
    keys = keys if isinstance(keys, tuple) else (keys,)
    if any(k is None for k in keys):
        raise NotImplementedError("None (a new axis) in a Tensor index; use expand_dims")
    if (n_ellipsis := sum(k is Ellipsis for k in keys)) > 1:
        raise IndexError("an index can only have a single ellipsis")
    if n_ellipsis:
        at = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:at] + (slice(None),) * (self.ndim - len(keys) + 1) + keys[at + 1 :]
    if len(keys) > self.ndim:
        raise IndexError(f"too many indices for a Tensor of {self.ndim} dimensions")
    keys = keys + (slice(None),) * (self.ndim - len(keys))
    keys = tuple(_scalar_index(k, a) for a, k in enumerate(keys))
    sel = [_axis_index(k, n, a) for a, (k, n) in enumerate(zip(keys, self._shape, strict=True))]
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


# Past this many listed positions per group, extraction filters the coordinates instead
_MAX_LISTED = 2**26


def _group_index(shape, axes, sel, keys):
    """The rows (or columns) of one group to extract: ``slice(None)`` for all of them, a slice
    when they are a contiguous range of raveled positions (a leading int or range with the
    rest of the group whole), else an array of raveled positions.
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
            return slice(start, start + x.size * tail)
        return slice(start, start + 1)
    lists = [np.array([s]) if isinstance(s, int) else s for s in (sel[a] for a in axes)]
    if math.prod(len(x) for x in lists) > _MAX_LISTED:
        return None
    idx = np.zeros(1, np.int64)
    for a, x in zip(axes, lists, strict=True):
        idx = (idx[:, None] * shape[a] + x[None, :]).ravel()
    return idx


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
    ridx = _group_index(t._shape, rows, sel, keys)
    cidx = _group_index(t._shape, cols, sel, keys)
    listed = None
    if ridx is not None and cidx is not None:
        listed = sum(x.size for x in (ridx, cidx) if isinstance(x, np.ndarray))
        if not listed:
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
        and t._M.ss.format == "hypercsr"
        and not t.dtype._is_udt
        and all(a in rows and (isinstance(keys[a], slice) or isinstance(sel[a], int)) for a in axes)
    ):
        # Pick rows among the nonempty ones (h) of a hypersparse row group: sized by the data
        # (NELL-2's (0,1|2), j in a range: the extract listed 1.1e7 rows, the filter read 7.7e7
        # entries). Numpy work per nonempty row, then P @ M over the kept ones.
        frac = math.prod(
            1 / t._shape[a] if isinstance(sel[a], int) else sel[a].size / t._shape[a] for a in axes
        )
        hrows = tl.nonempty_rows(t._M)
        cost = (
            tl.C_PBUILD_ROW[ti] * hrows
            + tl.C_PERM_ROW["hyper"][ti] * frac * hrows
            + tl.C_PERM_ENTRY[ti] * frac * nv
        )
        options.append((cost, "rows"))
    _, how = min(options)
    if how == "extract":
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
            pos = np.full(t._shape[a], -1, np.int64)
            pos[s] = np.arange(s.size)
            keep &= pos[coord] >= 0
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
            pos = np.full(t._shape[a], -1, np.int64)
            pos[s] = np.arange(s.size)
            out[renum[a]] = pos[coord]
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
            pos = np.full(t._shape[a], -1, np.int64)
            pos[s] = np.arange(s.size)
            positions[a] = pos
            keep &= pos[co[i]] >= 0
    n = int(np.count_nonzero(keep))
    out = {}
    for i, a in enumerate(rows):
        if a in renum:
            c = co[i][keep]
            out[renum[a]] = positions[a][c] if a in positions else c
    dst = tl.ravel(out, shape, layout.rows, n=n)
    nrows = tl.prod(shape, layout.rows)
    return tl.from_coo(dst, h[keep], True, BOOL, nrows, t._M._nrows)


def _axis_coords(r, c, shape, rows, cols, axis):
    """Coordinates along ``axis`` of entries at linear rows ``r`` and columns ``c``."""
    group, lin = (rows, r) if axis in rows else (cols, c)
    stride = tl.prod(shape, group[group.index(axis) + 1 :])
    return (lin // stride) % shape[axis]


def _reduce_scalar_recipe(updater, a, op, allow_empty):
    updater << a._M.reduce_scalar(op, allow_empty=allow_empty)


# ---------------------------------------------------------------------------------------------
# Repr


def _format_tensor(T, mask=None, max_entries=10):
    name = T.name if mask is None else mask.name
    kind = "gb.Tensor" if mask is None else type(mask).__name__
    header = (
        f'"{name}"  nvals {T._M.nvals}  shape {T._shape}  dtype {T.dtype}  '
        f"layout {T._layout._short()}  format {T._M.ss.format}"
    )
    lines = [f"{kind}  {header}"]
    if nvals := T._M.nvals:
        rows, cols, vals = T._M.ss.head(max_entries)
        coords = [None] * T.ndim
        tl.unravel(rows.astype(np.int64), T._shape, T._layout.rows, coords)
        tl.unravel(cols.astype(np.int64), T._shape, T._layout.cols, coords)
        order = "" if T._layout.seq == tuple(range(T.ndim)) else " (storage order)"
        lines.append(f"entries{order}:")
        for i in range(len(vals)):
            idx = tuple(int(c[i]) for c in coords)
            lines.append(f"  {idx}: {vals[i]}")
        if nvals > max_entries:
            lines.append("  ...")
    return "\n".join(lines)

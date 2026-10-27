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

The operation families live in their own modules (tensor_shape, tensor_ewise, tensor_reduce,
tensor_contract, tensor_engines, tensor_einsum, tensor_index), which this module imports at
its end.

Timings in the comments of these modules were measured on an Apple M5 Pro (6 performance and
12 efficiency cores) with SuiteSparse:GraphBLAS 10.5.1, at 8 and 1 threads, written
"8 / 1". The data they name: u6 and u7 are uniform random 1000^3 tensors with 1e6 and 1e7
entries; skew is uniform (100000, 1000, 10) with 1e7; hyper is uniform (2^19)^3 with 1e6;
Zipf is 100000^3 with 1e7 entries whose coordinates are Zipf-distributed (a = 1.5); NELL-2
(77M entries), Uber (3.3M), NIPS (3.1M) and Enron (54M) are FROSTT tensors (frostt.io).
"""

import itertools
import math
import operator
import weakref

import numpy as np

from .. import backend, binary, config, monoid, semiring
from ..dtypes import BOOL, FP64, lookup_dtype, unify
from ..exceptions import DimensionMismatch, IndexOutOfBound
from . import _has_numba, automethods, lib  # noqa: F401 (_has_numba: a knob, see tensor_engines)
from . import tensor_census as census
from . import tensor_layout as tl
from . import utils
from .base import BaseExpression, BaseType
from .expr import InfixExprBase, Updater
from .mask import StructuralMask, ValueMask
from .matrix import Matrix, TransposedMatrix
from .scalar import Scalar, ScalarExpression
from .tensor_layout import Layout
from .utils import output_type
from .vector import Vector

__all__ = ["Layout", "Tensor", "TensorExpression", "einsum"]


def _has_container():
    """Whether this SuiteSparse has the Container API (SuiteSparse:GraphBLAS 10), which the
    Tensor is built on.
    """
    return hasattr(lib, "GxB_Container_new")


def _check_backend():
    """Every public way to make a Tensor calls this first (einsum makes its operands Tensors),
    so another backend, or a SuiteSparse without the Container API, fails at once with one
    message.
    """
    if backend != "suitesparse":
        raise NotImplementedError("Tensor requires the suitesparse backend")
    if not _has_container():
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
    is needed (see "Copy-on-write views" in ai-notes/tensor/design.md). Keyed by ``id``
    because a WeakSet compares its members with ``==``, which will build an expression once
    Tensor has comparison operators.
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
        if other._store is not self._store:
            self._store.owners.discard(self)
            self._store = other._store
            self._store.owners.add(self)
        # Also when the store is already ours: a permuted view of it (``T << T.T``) holds the
        # same Matrix in another layout
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
            split chosen for about four entries per nonempty row.
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
            if census.on:
                census.note("layout.default", "shape rule", ndim=ndim, nvals=n)
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
            before = lay
            M, lay = _settle_split(M, shape, lay)
            if census.on:
                census.note("layout.settle", "moved" if lay != before else "kept", nvals=n)
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
            Combines values at repeated coordinates, which only a GCXS array can hold
            (pydata's COO sums them); by default they are summed, as pydata's ``todense``
            does.
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

        scipy.sparse has n-d COO arrays from scipy 1.15 on and 1-d ones from 1.13; with an
        older scipy, those ranks raise ``NotImplementedError``.

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
        by a diagonal and reduces with a selector, without moving the data.
        """
        return _tensordot_expr(self, other, axes, op)

    def mttkrp(self, factors, mode, op=semiring.plus_times):
        """Matricized tensor times Khatri-Rao product for ``mode``: an (I_mode x R) Tensor.

        ``factors`` holds one (I_n x R) factor per axis (Matrix or ndarray; the entry for
        ``mode`` is ignored). For 3-d mode 0 this is ``einsum('ijk,jr,kr->ir', X, B, C)``,
        computed as one TTM and then one product per remaining axis against an (I_n R x R)
        embedding of its factor, never materializing the Khatri-Rao product.
        With Numba installed, a 3-d FP64 tensor with full FP64 factors and ``plus_times``
        takes a kernel over the stored arrays instead, in the layout it is stored in. Both
        sum over one axis before multiplying by the next factor, which needs a semiring
        whose multiply distributes over its monoid and commutes (see ``einsum``); others
        take einsum's way, ``mult(mult(X, F_0), F_1)``... in axis order for each term.
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
# Storing a result: masks, accumulators and the output layout


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
            # Move whichever is smaller, the result or the mask.
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


def _broadcast_shapes(*shapes):
    """The shape numpy's ``broadcast_shapes`` gives, in Python ints: numpy refuses a result of
    2^63 positions or more (its size is an intp), which a Tensor can have (FROSTT's NELL-1 has
    1.6e20). Raises ValueError when the shapes do not broadcast.
    """
    ndim = max((len(s) for s in shapes), default=0)
    out = [1] * ndim
    for s in shapes:
        for a, n in enumerate(s, ndim - len(s)):
            if n != 1:
                if out[a] not in {1, n}:
                    raise ValueError(f"shapes {shapes} do not broadcast")
                out[a] = int(n)
    return tuple(out)


def _flipped(op):
    """The typed op that gives ``op(x, y)`` as ``flipped(y, x)``, for a product that puts the
    operands the other way round; or None. Only a flip of the same kind and result type gives
    the same values: pow's is rpow, a Numba UDF that wraps on integer overflow where
    GraphBLAS's pow saturates (and gives INT8 on BOOL).
    """
    flip = getattr(op, "commutes_to", None)
    if flip is None or type(flip) is not type(op) or flip.return_type != op.return_type:
        return None
    return flip


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
        if census.on:
            census.note("ewise.layout", "one layout")
        return layouts.pop()
    shape = operands[0]._shape
    nthreads = _nthreads(updater)
    # Ties go to the larger operand's layout.
    ops = sorted(operands, key=lambda t: -t._M._nvals)
    candidates = [t._layout for t in ops]
    if fixed is not None:
        candidates.append(fixed)
    if mlayout is not None:
        candidates.append(mlayout)
    best = None
    costs = []
    for L in candidates:
        cost = sum(tl.relayout_cost(shape, t._layout, L, t._M._nvals, nthreads) for t in ops)
        final = L if fixed is None else fixed
        if fixed is not None:
            cost += tl.relayout_cost(shape, L, fixed, result_nvals, nthreads)
        if mlayout is not None:
            cost += tl.relayout_cost(shape, mlayout, final, mask.parent._M._nvals, nthreads)
        costs.append(cost)
        if best is None or cost < best[0]:
            best = (cost, L)
    if census.on:
        labels = [f"operand {k}" for k in range(len(ops))]
        labels += ["output"] * (fixed is not None) + ["mask"] * (mlayout is not None)
        i = census.choose("ewise.layout", costs, labels, candidates.index(best[1]))
        return candidates[i]
    return best[1]


def _in_layout(t, layout, nthreads):
    if t._layout == layout:
        return t._M
    return tl.relayout(t._M, t._shape, t._layout, layout, nthreads=nthreads)[0]


# The default split puts as many axes in the rows as the data fills. A row group indexes its
# nonempty rows (h), so selectors and expansions over it follow the data, while a column group
# costs per position (NELL-2's (0|1,2) has a 2.6e8-position column group). But rows
# cost too: a hypersparse row costs a hash lookup in ewise and mxm (about 19 / 130 ns at 8 / 1
# threads: Zipf's (0,1|2), 4.3 entries per row, took 53 ms for X + Y against 9.6 in (0|1,2)), and
# a sparse row about 3 / 16 ns. So rows must hold MIN_ROW_ENTRIES entries each, or
# MIN_HYPER_ROW_ENTRIES when they would be hypersparse (NELL-2's hold 228 each).
MIN_ROW_ENTRIES = 1.5
MIN_HYPER_ROW_ENTRIES = 32
FULL_ROW_ENTRIES = tl.FULL_ROW_ENTRIES
# The sampled check costs about 0.3 ms; the exact one about 1.1 ns per entry (ravel, order
# check and count), so sorted input is sampled only past 2^23 entries
_SPLIT_SAMPLE = 2**14
_SPLIT_SAMPLE_PAST = 2**23


def _rows_ok(nvals, nonempty, positions, narrower_cols):
    """Whether rows of ``nvals`` entries over ``nonempty`` of ``positions`` rows are worth it.

    Thin sparse rows (under FULL_ROW_ENTRIES each) cost about 2x in ewise (u6, 1.6 per row:
    X + Y 3.9 ms vs 1.5), so they pay off only when the alternative, one axis fewer in the
    rows, has a column group of more positions than entries (2.6x then).
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
        if census.on:
            census.note("layout.default", "past 2^60: shape rule", ndim=ndim, nvals=n)
        return tl.default_layout(shape, n), None, False
    sample = np.random.default_rng(0).integers(0, n, min(n, _SPLIT_SAMPLE))
    sampled_out = 0  # splits the sample rejected

    def thin_by_sample(p, positions, narrower_cols):
        nonlocal sampled_out
        low = _sampled_nonempty_rows(coords, shape, seq[:p], n, sample)
        thin = not _rows_ok(n, low, positions, narrower_cols)
        sampled_out += thin
        return thin

    def chosen(choice, p):
        if census.on:
            census.note("layout.default", choice, ndim=ndim, nvals=n, split=p,
                        sampled_out=sampled_out)  # fmt: skip

    for p in range(ndim - 1, 0, -1):
        if not tl._valid_split(shape, seq, p):
            continue
        positions = tl.prod(shape, seq[:p])
        if positions * FULL_ROW_ENTRIES <= n:
            chosen("full enough by shape", p)
            return Layout(seq[:p], seq[p:]), None, False  # full enough even if all nonempty
        narrower_cols = tl.prod(shape, seq[p - 1 :])
        large = n > _SPLIT_SAMPLE_PAST
        if large and thin_by_sample(p, positions, narrower_cols):
            continue
        keys = tl.ravel(coords, shape, seq[:p], n=n)
        if not np.all(keys[1:] >= keys[:-1]):
            if not large and thin_by_sample(p, positions, narrower_cols):
                continue
            chosen("unsorted: sampled, settled after", p)
            return Layout(seq[:p], seq[p:]), keys, True
        nonempty = int(np.count_nonzero(keys[1:] != keys[:-1])) + 1
        if _rows_ok(n, nonempty, positions, narrower_cols):
            chosen("sorted: exact count", p)
            return Layout(seq[:p], seq[p:]), keys, False
    for p in range(1, ndim):
        if tl._valid_split(shape, seq, p):
            chosen("no split full enough: p = 1", p)
            return Layout(seq[:p], seq[p:]), None, False
    chosen("shape rule", None)
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
        M = tl.reshape_owned(M, tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
        p = q
    return M, Layout(seq[:p], seq[p:])


def _healthy(M, shape, layout):
    """Move an empty column group (an n x 1 Matrix, held as n rows of one entry at most) to
    the default split, by one reshape (see "Choosing a default layout" in
    ai-notes/tensor/design.md).

    An empty row group is left alone: one row holding every entry costs 12 bytes per entry
    like a Vector, and SuiteSparse runs ewise, apply and selector products on it as fast as
    on a square Matrix (measured at 1e6 entries), so a reshape there would only add time.
    """
    ndim = len(shape)
    if ndim == 0 or layout.cols or math.prod(shape) >= tl.MAX_GROUP:
        return M, layout
    seq = layout.seq
    nvals = M._nvals
    p = 0 if ndim == 1 else tl.full_split(shape, seq, nvals)
    if not tl._valid_split(shape, seq, p):
        return M, layout
    if census.on:
        census.note("layout.healthy", "reshaped an empty column group", ndim=ndim)
    if nvals == 0:
        M = tl.new_matrix(M.dtype, tl.prod(shape, seq[:p]), tl.prod(shape, seq[p:]))
    else:
        M = M.ss.reshape(tl.prod(shape, seq[:p]), tl.prod(shape, seq[p:]))
    return M, Layout(seq[:p], seq[p:])


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
    if tl.stored_format(M) == "hypercsr":
        return C_SEL_ENTRY[ti] * nvals + C_HROW[ti] * tl.nonempty_rows(M)
    return _shape_selector_cost(nvals, positions, ti)


def _rs_cost(shape, seq, q, nvals, ti):
    return tl.rs_cost(shape, seq, q, nvals, ti)


# The TTM engines compile per R on first use, so small tensors keep the product
_TTM_KERNEL_MIN_NNZ = 2**16
# The TTM engines, tried in this order where each runs (all are kept, to measure; tests and
# benchmarks set this to force some):
#   "udt"        one mxv with FP64[R] values (tensor_udt.ttm; JIT on, FP64 plus_times)
#   "numba"      the Numba kernel (tensor_numba.ttm; FP64 plus_times)
#   "products"   one product: in the stored order on hypersparse rows, on the nonempty rows
#                compacted (_ttm_compacted, any type and semiring; NELL-2 mode 2: 1.6-1.7x
#                over the plain product); else after a relayout (_contract). It runs every
#                TTM, so it is also what any list falls back to
_TTM_ENGINES = ("udt", "numba", "products")
# Where both engines run, the UDT one is faster only in the stored order with a hypersparse
# output (fewer nonempty fibers than SuiteSparse's hyper switch, 1/16) and R up to 32
# (NELL-2 mode 2: 77 / 139 / 329 ms against 92 / 174 / 354 at R = 8 / 16 / 32, 1 thread;
# 14 / 24 / 55 against 18 / 31 / 62 at 8); slower with a bitmap output
# (u7: 25-86 against 27-77 ms at R = 16, 1 thread, and 1.4-1.8x slower at 8), at R = 64,
# and after a relayout (NELL-2 modes 0 and 1: 1.6-1.8x slower).
_TTM_UDT_MAX_R = 32


# The kernels compile per R on first use (about 0.3 s each), so small tensors keep the products
_MTTKRP_KERNEL_MIN_NNZ = 2**16


# The MTTKRP engines, tried in this order where each runs (all are kept, to measure; tests
# and benchmarks set this to force some):
#   "numba"     the Numba kernel (tensor_numba.mttkrp_nd; FP64 plus_times)
#   "udt"       the products, their first step (a TTM over the last axis) by the array-UDT
#               engine (tensor_udt.ttm; JIT on, FP64 plus_times, a full factor)
#   "products"  the GraphBLAS products (two-mxm for 3-d), which run every MTTKRP, so they
#               are also what any list falls back to
# "udt" against "products" (Numba off), R = 16: NELL-2 mode 0 702 -> 142 ms at 1
# thread and 142 -> 29 at 8; u7 82-122 -> 38-77 at 1, the same at 8; NELL-2 modes 1 and 2
# within 3% at 1 thread, 7-13% slower at 8 (load rising to 3.5).
_MTTKRP_ENGINES = ("numba", "udt", "products")


# Past this many listed positions per group, extraction filters the coordinates instead
_MAX_LISTED = 2**26


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


# isort: off
# The operation families, in their own modules. Imported last: they import this module's
# names, and the functions above call theirs at run time. The private names tests and
# benchmarks use stay reachable here (graphblas.core.tensor._ttm_compacted, ...).
from .tensor_shape import (  # noqa: E402 (also re-exports)
    _reshape,
    _reshape_layout,
    _reshape_shape,
)
from .tensor_ewise import (  # noqa: E402, F401 (also re-exports)
    _apply_recipe,
    _bcast_mult,
    _ewise_expr,
    _expand,
    _select_recipe,
)
from .tensor_reduce import (  # noqa: E402, F401 (also re-exports)
    _reduce_aggregator,
    _reduce_expr,
    _reduce_matrix,
    _reduce_scalar_recipe,
)
from .tensor_contract import (  # noqa: E402, F401 (also re-exports)
    _contract,
    _orders,
    _matmul_expr,
    _tensordot_expr,
    _ttv,
)
from .tensor_engines import (  # noqa: E402, F401 (also re-exports)
    _as_factor,
    _mttkrp_expr,
    _mttkrp_numba,
    _mttkrp_order,
    _mttkrp_recipe,
    _sampled_fibers,
    _ttm_compacted,
    _ttm_engines,
    _ttm_numba,
    _ttm_udt,
)
from .tensor_einsum import (  # noqa: E402, F401 (also re-exports)
    _einsum_plan,
    einsum,
)
from .tensor_index import (  # noqa: E402, F401 (also re-exports)
    _axis_index,
    _extract_expr,
    _extract_recipe,
    _group_index,
    _scalar_index,
)

# isort: on

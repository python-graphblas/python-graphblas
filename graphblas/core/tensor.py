"""Tensor: an n-dimensional sparse array on one SuiteSparse Matrix (experimental).

Semantics follow GraphBLAS for missing values (no fill value) and numpy / the Array API for
shapes and axes. Operations are lazy, as for Matrix: they return a ``TensorExpression``
that computes on ``.new()`` or ``T(mask, accum, replace) << expr``.

A Tensor stores its entries in one Matrix held by row (``T.ss.matrix``) plus a ``Layout``:
which axes are raveled (C order) into the rows and which into the columns. ``permute_dims``,
``.T``, ``.mT``, ``expand_dims`` and ``squeeze`` only change the layout, in O(1); the data
moves inside an operation when that operation needs it (``tensor_layout.relayout``).
Tensors that share a stored Matrix copy it before one of them is changed in place
(copy-on-write), so a Tensor never sees another's writes.

SuiteSparse:GraphBLAS 10 or later only. Import with ``from graphblas.core.tensor import Tensor``.

Phase 1 covers the data model, relayout, the shape operations, same-shape element-wise
operations and reductions. Broadcasting, indexing, contractions and Matrix/Vector operands
are not implemented yet.
"""

import itertools
import math
import operator
import weakref

import numpy as np

from .. import backend, binary, config, monoid
from ..dtypes import BOOL, FP64, lookup_dtype, unify
from ..exceptions import DimensionMismatch, IndexOutOfBound
from . import automethods, lib
from . import tensor_layout as tl
from . import utils
from .base import BaseExpression, BaseType
from .expr import Updater
from .mask import StructuralMask, ValueMask
from .matrix import Matrix, TransposedMatrix
from .operator import get_semiring, get_typed_op
from .scalar import Scalar, ScalarExpression, _union_default
from .tensor_layout import Layout
from .utils import output_type
from .vector import Vector

__all__ = ["Layout", "Tensor", "TensorExpression"]


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


def _normalize_axis(axis, ndim, *, extra=0):
    n = ndim + extra
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
    """The owners of one stored Matrix: Tensors that view it, and Matrices handed out with
    ``copy=False``. Held weakly, so no ``__del__`` bookkeeping is needed. Keyed by ``id``
    because a WeakSet compares its members with ``==``, which builds an expression for
    Matrix (and will for Tensor).
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
    """An n-dimensional sparse array (experimental).

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
        args = [_mask_arg(arg) for arg in optional_mask_accum_replace]
        if mask is not None:
            mask = _mask_arg(mask)
        return super().__call__(*args, mask=mask, accum=accum, replace=replace, **opts)

    def __getitem__(self, keys):
        raise NotImplementedError("Tensor indexing is not implemented yet")

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
        if layout is None:
            lay = tl.default_layout(shape, n)
        else:
            lay = tl.as_layout(layout, shape)
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

    def isequal(self, other, *, check_dtype=False):
        """Whether ``other`` has the same shape, the same entries and equal values."""
        other = self._expect_type(other, Tensor, within="isequal", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._shape != other._shape or self._M.nvals != other._M.nvals:
            return False
        A, B = _same_layout(self, other)
        return A.isequal(B, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """Whether ``other`` has the same shape, the same entries and close values."""
        other = self._expect_type(other, Tensor, within="isclose", argname="other")
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

    # Element-wise operations
    def ewise_add(self, other, op=monoid.plus):
        """Element-wise ``op`` on the union of entries (same shape; no broadcasting yet)."""
        return _ewise_expr(self, "ewise_add", other, op)

    def ewise_mult(self, other, op=binary.times):
        """Element-wise ``op`` on the intersection of entries (same shape)."""
        return _ewise_expr(self, "ewise_mult", other, op)

    def ewise_union(self, other, op, left_default, right_default):
        """Element-wise ``op`` on the union, with defaults for a missing side (same shape)."""
        return _ewise_expr(self, "ewise_union", other, op, left_default, right_default)

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
        """Reduce over ``axis`` (None: all axes, giving a 0-d Tensor) with the monoid ``op``.

        A position whose fiber has no entries stays missing (it is not 0).
        """
        return _reduce_expr(self, op, axis, keepdims)

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
    if _is_tensor_like(left) and _is_tensor_like(right):
        left = _compute(left, "infix operator")
        if kind == "add":
            return left.ewise_add(right, op)
        if kind == "union":
            return left.ewise_union(right, op, False, False)
        return left.ewise_mult(right, op)
    if _is_tensor_like(left) and _is_scalar_like(right):
        return _compute(left, "infix operator").apply(op, right=right)
    if _is_scalar_like(left) and _is_tensor_like(right):
        return _compute(right, "infix operator").apply(op, left=left)
    return NotImplemented


def _make_infix(op, kind):
    def forward(self, other):
        return _ewise_infix(self, other, op, kind)

    def reflected(self, other):
        return _ewise_infix(other, self, op, kind)

    def inplace(self, other):
        if type(self) is not Tensor:
            raise TypeError(f"in-place operators are not supported for {type(self).__name__}")
        expr = _ewise_infix(self, other, op, kind)
        if expr is NotImplemented:
            raise TypeError(f"unsupported operand type for in-place op: {type(other)}")
        self << expr
        return self

    return forward, reflected, inplace


_INFIX = {
    "add": (binary.plus, "add"),
    "sub": (binary.minus, "union"),
    "mul": (binary.times, "mult"),
    "truediv": (binary.truediv, "mult"),
}


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


for _name, (_op, _kind) in _INFIX.items():
    _fwd, _ref, _inp = _make_infix(_op, _kind)
    for _cls in (Tensor, TensorExpression):
        setattr(_cls, f"__{_name}__", _fwd)
        setattr(_cls, f"__r{_name}__", _ref)
    setattr(Tensor, f"__i{_name}__", _inp)
del _name, _op, _kind, _fwd, _ref, _inp, _cls

utils._output_types[Tensor] = Tensor
utils._output_types[TensorExpression] = Tensor


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
    mask = kw.get("mask")
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
    mask = kw.get("mask")
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
    mask = updater.kwargs.get("mask")
    mlayout = mask.parent._layout if mask is not None and type(mask.parent) is Tensor else None
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


def _ewise_expr(self, method, other, op, left_default=None, right_default=None):
    other = self._expect_type(other, Tensor, within=method, argname="other", op=op)
    if self._shape != other._shape:
        raise DimensionMismatch(
            f"Tensor shapes must match in {method} (broadcasting is not implemented yet): "
            f"{self._shape} and {other._shape}"
        )
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
    return TensorExpression(
        method,
        _ewise_recipe,
        [self, other],
        (method, self, other, op, extra),
        dtype=op.return_type,
        shape=self._shape,
        op=op,
        expr_repr="{0.name}.%s({1.name}, op={op})" % method,  # noqa: UP031
    )


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
    if M.dtype._is_udt or tl.density(M):
        return _reduce_rebuild(M, shape, layout, A, op)
    # Pass on only the descriptor options the user gave: building one costs microseconds.
    nthreads = opts.get("nthreads") or tl._nthreads()
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
                if not _selector_ok(math.prod(gshape), nvals):
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
            ncols = tl.prod(shape, cols)
            wide = ncols >= 1e5
            if (
                wide
                and nthreads > 1
                and len(cols) >= 2
                and math.prod(shape) < tl.MAX_GROUP
                and _selector_ok(tl.prod(shape, rows) * shape[cols[0]], nvals)
            ):
                # reduce_columnwise into many outputs scatters like a wide transpose and is
                # slow with threads (DESIGN.md 3.2); move one column axis into the rows and
                # reduce with a selector instead.
                q = len(rows) + 1
                seq = layout.seq
                M = M.ss.reshape(tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
                layout = Layout(seq[:q], seq[q:])
                continue
            v = Vector(dtype, M._ncols)
            v(**(dict(opts, nthreads=1) if wide else opts)) << M.reduce_columnwise(op)
            M = _vector_to_row(v)
            layout = Layout((), cols)
        else:
            gshape = tuple(shape[x] for x in rows)
            if not _selector_ok(math.prod(gshape), nvals):
                return _reduce_rebuild(M, shape, layout, A, op)
            kept = tuple(i for i, x in enumerate(rows) if x not in Ar)
            S = tl.selector(gshape, kept)
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
    kept = [x for x in range(len(shape)) if x not in A]
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

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

import numpy as np

from ..exceptions import DimensionMismatch
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

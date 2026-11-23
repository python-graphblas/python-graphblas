"""Reshape for Tensors: a view when the layout represents the result, else a relayout.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import itertools
import math
import operator

import numpy as np

from . import tensor_layout as tl
from .tensor_layout import Layout


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
    """The layout of ``new_shape`` under which the stored Matrix of ``layout``, unchanged,
    represents the reshape: each run of old axes (``_reshape_chunks``) lies within one group,
    in order, and becomes the run's new axes there. None when some run does not. Sizes are
    positive.
    """
    groups = []
    for group in (layout.rows, layout.cols):
        groups.append([a for a in group if old_shape[a] != 1])
    out = ([], [])
    for ia, jb in _reshape_chunks(old_shape, new_shape):
        g = 0 if ia[0] in groups[0] else 1  # (a run starts at an axis of size other than 1)
        k = groups[g].index(ia[0])
        if groups[g][k : k + len(ia)] != ia:
            return None
        out[g].append((k, jb))
    rows, cols = (tuple(b for _, jb in sorted(runs) for b in jb) for runs in out)
    units = [b for b, n in enumerate(new_shape) if n == 1]
    return _insert_unit_axes(Layout(rows, cols), units)


def _reshape_targets(old_shape, new_shape):
    """The layout states of ``old_shape`` whose stored Matrix represents the reshape: the
    runs as blocks, in any order, split between them with both groups nonempty (a group of
    all the axes stores the result as one row or column).
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
    """``t.reshape(shape)`` (``shape`` resolved): a view when the layout represents it; else the
    data moved once to a layout that does (the runs as blocks, split between two of them);
    else, when the reshape is one run, the data in C order (a relayout to the identity
    sequence) with the stored Matrix reshaped to the new shape's default split.
    """
    if math.prod(shape) == 0:
        return Tensor(t.dtype, shape)
    if (layout := _reshape_layout(t._shape, t._layout, shape)) is not None:
        return t._view(shape, layout)
    if targets := _reshape_targets(t._shape, shape):
        M, src = tl.relayout(t._M, t._shape, t._layout, targets)
        return Tensor._from_M(M, shape, _reshape_layout(t._shape, src, shape))
    M, _ = tl.c_order(t._M, t._shape, t._layout)
    layout = tl.default_layout(shape, M._nvals)
    nrows, ncols = tl.prod(shape, layout.rows), tl.prod(shape, layout.cols)
    # (never the shared Matrix in place)
    M = M.ss.reshape(nrows, ncols) if M is t._M else tl.reshape_owned(M, nrows, ncols)
    return Tensor._from_M(M, shape, layout)


# numpy.pad's modes; all but "constant" raise NotImplementedError, other names ValueError
_NUMPY_PAD_MODES = frozenset(
    {"constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect"}
    | {"symmetric", "wrap", "empty"}
)


def _pad(t, pad_width, mode):
    """``t.pad(pad_width, mode)``: the entries rebuilt in ``t``'s layout of the padded shape,
    each coordinate moved by its axis's leading pad.

    The move keeps the order of each group's raveled index, so the new keys arrive in storage
    order and the build skips its sort.
    """
    if mode != "constant":
        if not callable(mode) and mode not in _NUMPY_PAD_MODES:
            raise ValueError(f"pad: mode {mode!r} is not supported")
        raise NotImplementedError(
            f"pad: only mode='constant', whose new positions are missing; got mode={mode!r}"
        )
    widths = np.asarray(pad_width)
    if widths.dtype.kind != "i":  # (unsigned too, as numpy)
        raise TypeError(f"pad_width must be of integral type; got {pad_width!r}")
    if (widths < 0).any():  # (before broadcasting: a 0-d Tensor has no pairs to check)
        raise ValueError(f"pad_width can't contain negative values; got {pad_width!r}")
    try:
        widths = np.broadcast_to(widths, (t.ndim, 2)).tolist()
    except ValueError:
        raise ValueError(
            f"pad_width {pad_width!r} does not broadcast to {t.ndim} (before, after) pairs"
        ) from None
    # (Python ints: a sum of int64 widths could wrap before the 2**60 check)
    shape = tuple(n + b + a for n, (b, a) in zip(t._shape, widths, strict=True))
    layout = tl.as_layout((t._layout.rows, t._layout.cols), shape)
    if shape == t._shape:
        return t.dup()
    M = t._M
    coords, values = tl.coords_of(M, t._shape, t._layout.state)
    for a, (b, _) in enumerate(widths):
        if b:
            coords[a] = coords[a] + b
    n = values.shape[0]
    rows = tl.ravel(coords, shape, layout.rows, n)
    cols = tl.ravel(coords, shape, layout.cols, n)
    P = tl.from_coo(
        rows, cols, values, t.dtype, tl.prod(shape, layout.rows), tl.prod(shape, layout.cols)
    )
    return Tensor._from_M(P, shape, layout)


# isort: off
from .tensor import (  # noqa: E402
    Tensor,
    _insert_unit_axes,
)

# isort: on

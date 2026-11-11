"""Private helpers shared by the modules of ``graphblas.array_api``."""

import contextlib
import math

import numpy as np

from .. import config
from ..core import tensor_layout as tl
from ..core.scalar import Scalar
from ..core.tensor import Tensor
from ..core.tensor_ewise import _broadcast_factor
from ..dtypes import lookup_dtype


class DensifyError(ValueError):
    """An operation would allocate a dense array larger than the namespace allows."""


def densify_limit():
    return config.get("array_api_densify_limit", 10**8)


def check_dense(n, what):
    """Raise DensifyError when ``what`` would hold ``n`` elements densely past the limit."""
    if n > (limit := densify_limit()):
        raise DensifyError(
            f"{what} would hold {n} elements densely, more than the limit of {limit}. Raise "
            "`graphblas.config` option array_api_densify_limit "
            "(`with gb.config.set(array_api_densify_limit=...)`) to allow it."
        )


def gb_dtype(dtype):
    return lookup_dtype(np.dtype(dtype))


def np_dtype(gbdtype):
    return np.dtype(gbdtype.np_type)


def scalar(value, dtype):
    """``value`` as a numpy scalar of ``dtype`` (numpy's cast, warnings silenced)."""
    with np.errstate(all="ignore"):
        return np.asarray(value).astype(dtype, copy=False)[()]


def gb_scalar(value, dtype):
    """A GraphBLAS Scalar of the numpy ``dtype`` holding ``value`` cast as numpy casts it."""
    return Scalar.from_value(scalar(value, dtype), gb_dtype(dtype))


def same_value(a, b):
    """Whether two scalars are identical as values: NaN matches NaN, -0.0 differs from 0.0."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind == "c" or b.dtype.kind == "c":
        return same_value(a.real, b.real) and same_value(a.imag, b.imag)
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        if np.isnan(a) or np.isnan(b):
            return bool(np.isnan(a) and np.isnan(b))
        return bool(a == b and np.signbit(a) == np.signbit(b))
    return bool(np.equal(a, b))


def differs(values, fill):
    """Boolean array: where ``values`` is not identical to the scalar ``fill`` (invariant 4)."""
    kind = values.dtype.kind
    if kind == "c":
        return differs(values.real, np.real(fill)) | differs(values.imag, np.imag(fill))
    if kind == "f":
        if np.isnan(fill):
            return ~np.isnan(values)
        return (values != fill) | (np.signbit(values) != np.signbit(fill))
    return values != fill


def expansion_check(Ta, Tb, shape):
    """Before a union of ``Ta`` and ``Tb`` broadcast to ``shape``: raise past the limit.

    The union replicates each broadcast operand's entries over the axes it broadcasts along.
    """
    replicated = 0
    for T in (Ta, Tb):
        factor = _broadcast_factor(T.shape, shape)
        if factor > 1:
            replicated += factor * T.nvals
    check_dense(replicated, f"broadcasting shapes {Ta.shape} and {Tb.shape} to {shape}")


def empty_tensor(dtype, shape):
    return Tensor(gb_dtype(dtype), tuple(shape))


def full_tensor(dtype, shape, value):
    """A Tensor holding ``value`` at every position (iso and full: O(1) memory)."""
    T = Tensor(gb_dtype(dtype), tuple(shape))
    if math.prod(shape) > 0:
        T << gb_scalar(value, dtype)
    return T


def tensor_values(T):
    """The stored values of ``T`` in storage order (a copy)."""
    return T._M.to_coo(rows=False, columns=False, sort=False)[2]


def _x_array(cont, dtype, stack):
    """The value array of an unloaded Container as a numpy view (None when it is empty)."""
    return stack.enter_context(tl._VectorArray(cont.x, np.dtype(dtype))).array


def map_values(func, tensors, dtype):
    """A Tensor with the entries of ``tensors`` (all of one shape, layout and pattern) whose
    values are ``func(*value_arrays)``, computed by numpy over the stored values.

    The pattern is copied in GraphBLAS and the values written into the copy's own array
    through a GxB_Container, so the entries are never rebuilt from coordinates.
    """
    T0 = tensors[0]
    shape, layout = T0._shape, T0._layout
    gbdt = gb_dtype(dtype)
    Ms = [T._M for T in tensors]
    if T0._M._nvals == 0:
        return Tensor._from_M(tl.new_matrix(gbdt, Ms[0]._nrows, Ms[0]._ncols), shape, layout)
    if any(T._layout != layout for T in tensors[1:]):
        Ms = [Ms[0]] + [tl.relayout(T._M, shape, T._layout, layout)[0] for T in tensors[1:]]
    # An iso Matrix has one value; the template for the result is one that holds them all,
    # if any does, so that its value array has the length of func's output.
    isos = [M.ss.is_iso for M in Ms]
    template = Ms[isos.index(False)] if False in isos else Ms[0]
    W = tl.new_matrix(gbdt, template._nrows, template._ncols)
    W << template
    with contextlib.ExitStack() as stack:
        unloaded = {}
        for M in Ms:
            if id(M) not in unloaded:
                unloaded[id(M)] = stack.enter_context(tl._Unloaded(M))
        conts = [unloaded[id(M)] for M in Ms]
        wcont = stack.enter_context(tl._Unloaded(W))
        formats = {c.format for c in conts} | {wcont.format}
        nvals = {c.nvals for c in conts} | {wcont.nvals}
        if len(formats) == 1 and len(nvals) == 1:
            arrays = [_x_array(c, np_dtype(M.dtype), stack) for c, M in zip(conts, Ms, strict=True)]
            out = _x_array(wcont, dtype, stack)
            with np.errstate(all="ignore"):
                result = func(*arrays)
            result = np.asarray(result)
            if out is not None and result.shape == out.shape:
                out[...] = result
                return Tensor._from_M(W, shape, layout)
    # (formats differ: a fallback through coordinates, which aligns the values by position)
    return _map_values_coo(func, tensors, dtype)  # pragma: no cover (safety)


def _map_values_coo(func, tensors, dtype):
    coords = None
    arrays = []
    for T in tensors:
        idx, vals = T.to_coo()
        if coords is None:
            coords = idx
        arrays.append(vals)
    with np.errstate(all="ignore"):
        result = np.asarray(func(*arrays), dtype=dtype)
    if result.ndim == 0:
        result = np.broadcast_to(result, coords[0].shape if coords else (1,))
    T0 = tensors[0]
    return Tensor.from_coo(coords, result, gb_dtype(dtype), shape=T0._shape)

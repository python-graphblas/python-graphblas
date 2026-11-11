"""Linear algebra in the main namespace: matmul, matrix_transpose, tensordot and vecdot.

Contractions treat a fill of zero as a structural zero (DESIGN D6), as scipy.sparse and
pydata/sparse do: a product with an unstored position is skipped, so 0 * inf there is not NaN,
unlike numpy. Sums may round differently from numpy's (the order of the terms differs). A
nonzero fill is the fill times a matrix of ones plus an array with fill zero, which turns into
sums of the operands broadcast over the result: the result is then dense in general (under the
densify limit). A fill that is not finite, or True, makes the result go through numpy densely.
"""

import math

import numpy as np

from .. import semiring
from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._dtypes import check_supported
from ._utils import gb_dtype

__all__ = ["matmul", "matrix_transpose", "tensordot", "vecdot"]


def _check(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs Arrays, not {type(x).__name__}")


def _operand(x):
    """A numpy array as an Array (``ndarray @ Array`` comes here through ``__rmatmul__``)."""
    if isinstance(x, np.ndarray):
        from ._creation import asarray

        return asarray(x)
    return x


def _semiring(dt):
    return (semiring.lor_land if dt.kind == "b" else semiring.plus_times)[gb_dtype(dt)]


def _is_zero(fill):
    return not bool(fill)


def _dense_ok(*arrays):
    """Whether the decomposition A = A0 + fill works: finite fills, not bool."""
    for x in arrays:
        if _is_zero(x._fill):
            continue
        if x.dtype.kind == "b" or (x.dtype.kind in "fc" and not np.isfinite(x._fill)):
            return False
    return True


def _dense(func, *arrays):
    from ._creation import _from_numpy

    for x in arrays:
        _utils.check_dense(x.size, "a contraction with a fill that is not finite or is True")
    out = np.asarray(func(*[x.todense() for x in arrays]))
    return _from_numpy(out)


def _zero_fill(x):
    """``x`` minus its fill (exact where the fill is 0), and the fill."""
    from ._elementwise import subtract

    if _is_zero(x._fill):
        return x, None
    return subtract(x, x._fill), x._fill


def _rdt(name, func, x1, x2):
    probe = func(
        np.ones((1,) * max(x1.ndim, 1), x1.dtype), np.ones((1,) * max(x2.ndim, 1), x2.dtype)
    )
    return check_supported(np.asarray(probe).dtype, name)


def matmul(x1, x2, /):
    from ._elementwise import add, multiply
    from ._manipulation import expand_dims
    from ._statistical import sum as sum_

    x1, x2 = _operand(x1), _operand(x2)
    _check(x1, "matmul")
    _check(x2, "matmul")
    if x1.ndim == 0 or x2.ndim == 0:
        raise ValueError("matmul needs arrays of at least one dimension")
    K = x1.shape[-1]
    if x2.shape[-2 if x2.ndim > 1 else 0] != K:
        raise ValueError(f"matmul: shapes {x1.shape} and {x2.shape} do not align")
    b1 = x1.shape[:-2]
    b2 = x2.shape[:-2]
    try:
        np.broadcast_shapes(b1, b2)
    except ValueError:
        raise ValueError(f"matmul: shapes {x1.shape} and {x2.shape} do not broadcast") from None
    rdt = _rdt("matmul", np.matmul, x1, x2)
    if not _dense_ok(x1, x2):
        return _dense(np.matmul, x1, x2)
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    out = _product(a0, b0, rdt)
    if fa is None and fb is None:
        return out
    terms = []
    if fa is not None:
        # fa * (ones @ b0): b0 summed over its rows, one row of the result
        s = sum_(b0, axis=-2 if x2.ndim > 1 else 0, dtype=rdt)
        if x1.ndim > 1 and x2.ndim > 1:
            s = expand_dims(s, axis=-2)
        terms.append(multiply(fa, s))
    if fb is not None:
        r = sum_(a0, axis=-1, dtype=rdt)
        if x2.ndim > 1 and x1.ndim > 1:
            r = expand_dims(r, axis=-1)
        terms.append(multiply(fb, r))
    if fa is not None and fb is not None:
        terms.append(multiply(multiply(fa, fb), K))
    for t in terms:
        out = add(out, t)
    from ._elementwise import cast

    return out if out.dtype == rdt else cast(out, rdt)


def _product(a, b, rdt):
    """``a @ b`` of arrays with fill 0, in GraphBLAS."""
    Ta, Tb = a._T(), b._T()
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    shape = shape + (a.shape[-2:-1] if a.ndim > 1 else ()) + (b.shape[-1:] if b.ndim > 1 else ())
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(rdt), shape), 0)
    return Array._new(Ta.matmul(Tb, _semiring(rdt)).new(dtype=gb_dtype(rdt)), 0)


def matrix_transpose(x, /):
    _check(x, "matrix_transpose")
    return x.mT


def _tensordot_axes(x1, x2, axes):
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError("tensordot's axes must be nonnegative")
        if axes > min(x1.ndim, x2.ndim):
            raise ValueError("tensordot: more axes than the arrays have")
        return tuple(range(x1.ndim - axes, x1.ndim)), tuple(range(axes))
    a, b = axes
    a = tuple(int(i) % x1.ndim for i in (a if isinstance(a, (list, tuple)) else (a,)))
    b = tuple(int(i) % x2.ndim for i in (b if isinstance(b, (list, tuple)) else (b,)))
    if len(a) != len(b):
        raise ValueError("tensordot needs as many axes of each array")
    return a, b


def tensordot(x1, x2, /, *, axes=2):
    from ._elementwise import add, cast, multiply
    from ._manipulation import reshape
    from ._statistical import sum as sum_

    _check(x1, "tensordot")
    _check(x2, "tensordot")
    a_axes, b_axes = _tensordot_axes(x1, x2, axes)
    for a, b in zip(a_axes, b_axes, strict=True):
        if x1.shape[a] != x2.shape[b]:
            raise ValueError(f"tensordot: axis {a} of x1 and axis {b} of x2 differ in size")
    rdt = check_supported(np.result_type(x1.dtype, x2.dtype), "tensordot")
    if not _dense_ok(x1, x2):
        return _dense(lambda a, b: np.tensordot(a, b, axes=(a_axes, b_axes)), x1, x2)
    free1 = tuple(n for i, n in enumerate(x1.shape) if i not in a_axes)
    free2 = tuple(n for i, n in enumerate(x2.shape) if i not in b_axes)
    shape = free1 + free2
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    Ta, Tb = a0._T(), b0._T()
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        out = Array._new(Tensor(gb_dtype(rdt), shape), 0)
    else:
        R = Ta.tensordot(Tb, (list(a_axes), list(b_axes)), _semiring(rdt)).new(dtype=gb_dtype(rdt))
        out = Array._new(R, 0)
    if fa is not None:
        s = sum_(b0, axis=b_axes, dtype=rdt) if b_axes else cast(b0, rdt)
        out = add(out, multiply(fa, reshape(s, (1,) * len(free1) + free2)))
    if fb is not None:
        r = sum_(a0, axis=a_axes, dtype=rdt) if a_axes else cast(a0, rdt)
        out = add(out, multiply(fb, reshape(r, free1 + (1,) * len(free2))))
    if fa is not None and fb is not None:
        K = math.prod(x1.shape[a] for a in a_axes)
        out = add(out, multiply(multiply(fa, fb), K))
    return out if out.dtype == rdt else cast(out, rdt)


def vecdot(x1, x2, /, *, axis=-1):
    """Sum over ``axis`` of ``conj(x1) * x2`` (broadcast), with zero fills structural."""
    from ._elementwise import add, cast, conj, multiply
    from ._statistical import sum as sum_

    _check(x1, "vecdot")
    _check(x2, "vecdot")
    ndim = min(x1.ndim, x2.ndim)
    if ndim == 0:
        raise ValueError("vecdot needs arrays of at least one dimension")
    if not -ndim <= axis < 0:
        if 0 <= axis < ndim and x1.ndim == x2.ndim:
            axis -= x1.ndim
        else:
            raise ValueError(f"vecdot's axis must be in [-{ndim}, -1], not {axis}")
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("vecdot: the axis differs in size")
    shape = np.broadcast_shapes(x1.shape, x2.shape)
    rdt = check_supported(np.result_type(x1.dtype, x2.dtype), "vecdot")
    if not _dense_ok(x1, x2):
        return _dense(lambda a, b: np.vecdot(a, b, axis=axis), x1, x2)
    if x1.dtype.kind == "c":
        x1 = conj(x1)
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    out_shape = tuple(n for i, n in enumerate(shape) if i != len(shape) + axis)
    Ta, Tb = a0._T(), b0._T()
    gbdt = gb_dtype(rdt)
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        out = Array._new(Tensor(gbdt, out_shape), 0)
    else:
        mult = _semiring(rdt).binaryop
        if _both_broadcast(Ta.shape, Tb.shape, shape):
            # (an outer product: ewise_mult replicates one operand)
            _utils.expansion_check(Ta, Tb, shape)
        P = Array._new(Ta.ewise_mult(Tb, mult).new(dtype=gbdt), 0)
        out = sum_(P, axis=axis, dtype=rdt) if rdt.kind != "b" else _any(P, axis)
    if fa is not None:
        out = add(out, multiply(fa, sum_(b0, axis=axis, dtype=rdt)))
    if fb is not None:
        out = add(out, multiply(fb, sum_(a0, axis=axis, dtype=rdt)))
    if fa is not None and fb is not None:
        out = add(out, multiply(multiply(fa, fb), shape[axis]))
    return out if out.dtype == rdt else cast(out, rdt)


def _any(x, axis):
    from ._statistical import any as any_

    return any_(x, axis=axis)


def _both_broadcast(sa, sb, shape):
    from ..core.tensor_ewise import _broadcast_factor

    return _broadcast_factor(sa, shape) > 1 and _broadcast_factor(sb, shape) > 1

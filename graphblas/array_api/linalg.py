"""The Array API's ``linalg`` extension (``xp.linalg``).

matmul, matrix_transpose, tensordot, vecdot, diagonal, trace, vector_norm, outer, cross, the
Frobenius, 1 and inf matrix norms, and matrix_power with n >= 0 work on the stored entries.
The decompositions and solvers (cholesky, det, eig, eigh, eigvals, eigvalsh, inv, pinv, qr,
slogdet, solve, svd, svdvals, matrix_rank, and matrix_norm of order 2, -2 or "nuc") are dense
algebra: they go through numpy densely, under the densify limit.
"""

import math
from typing import NamedTuple

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._array import Array
from ._linalg import matmul, matrix_transpose, tensordot, vecdot

__all__ = [
    "cholesky",
    "cross",
    "det",
    "diagonal",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
]


class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array


class EigResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array


class QRResult(NamedTuple):
    Q: Array
    R: Array


class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array


class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array


def _check(x, name, ndim=2):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs an Array, not {type(x).__name__}")
    if x.ndim < ndim:
        raise ValueError(f"{name} needs an array of at least {ndim} dimensions")


def _float(x):
    """``x`` in a floating-point dtype (numpy's linalg computes integers in float64)."""
    from ._elementwise import cast

    return x if x.dtype.kind in "fc" else cast(x, np.dtype(np.float64))


def _dense_call(name, func, *arrays):
    from ._creation import _from_numpy

    for x in arrays:
        _utils.check_dense(x.size, f"linalg.{name} (dense algebra)")
    out = func(*[x.todense() for x in arrays])
    if isinstance(out, tuple):
        return tuple(_from_numpy(np.asarray(o)) for o in out)
    return _from_numpy(np.asarray(out))


# Sparse
def diagonal(x, /, *, offset=0):
    """The diagonals of the last two axes: O(entries)."""
    _check(x, "diagonal")
    n, m = x.shape[-2:]
    offset = int(offset)
    length = max(0, min(n, m - offset) if offset >= 0 else min(n + offset, m))
    shape = (*x.shape[:-2], length)
    T = x._T()
    if T.nvals == 0 or length == 0:
        return Array._new(Tensor(T.dtype, shape), x._fill)
    coords, values = T.to_coo(sort=False)
    i, j = coords[-2], coords[-1]
    keep = j - i == offset
    k = i if offset >= 0 else j
    out = [c[keep] for c in coords[:-2]] + [k[keep]]
    if not keep.any():
        return Array._new(Tensor(T.dtype, shape), x._fill)
    return Array._new(Tensor.from_coo(out, values[keep], T.dtype, shape=shape), x._fill)


def trace(x, /, *, offset=0, dtype=None):
    from ._statistical import sum as sum_

    return sum_(diagonal(x, offset=offset), axis=-1, dtype=dtype)


def outer(x1, x2, /):
    from ._elementwise import multiply
    from ._manipulation import expand_dims

    for x in (x1, x2):
        _check(x, "outer", ndim=1)
        if x.ndim != 1:
            raise ValueError("outer needs 1-d arrays")
    return multiply(expand_dims(x1, axis=1), expand_dims(x2, axis=0))


def cross(x1, x2, /, *, axis=-1):
    """The cross product of 3-vectors along ``axis`` (numpy's products and differences)."""
    from ._elementwise import multiply, subtract
    from ._manipulation import stack

    for x in (x1, x2):
        _check(x, "cross", ndim=1)
    if not -min(x1.ndim, x2.ndim) <= axis < 0:
        raise ValueError(f"cross's axis must be negative and within both arrays, not {axis}")
    if x1.shape[axis] != 3 or x2.shape[axis] != 3:
        raise ValueError("cross needs vectors of size 3 along the axis")
    np.broadcast_shapes(x1.shape, x2.shape)

    def part(x, i):
        return x[(Ellipsis, i) + (slice(None),) * (-1 - axis)]

    a = [part(x1, i) for i in range(3)]
    b = [part(x2, i) for i in range(3)]
    c = [
        subtract(multiply(a[1], b[2]), multiply(a[2], b[1])),
        subtract(multiply(a[2], b[0]), multiply(a[0], b[2])),
        subtract(multiply(a[0], b[1]), multiply(a[1], b[0])),
    ]
    return stack(c, axis=axis)


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):  # noqa: A002 (the Array API's name)
    """O(entries): the norms are reductions of ``abs(x)``."""
    from ._elementwise import abs as abs_
    from ._elementwise import pow as pow_
    from ._elementwise import sqrt, square
    from ._searching import count_nonzero
    from ._statistical import max as max_
    from ._statistical import min as min_
    from ._statistical import sum as sum_

    if not isinstance(x, Array):
        raise TypeError(f"vector_norm needs an Array, not {type(x).__name__}")
    a = abs_(_float(x))
    if axis is None and x.ndim:
        axis = tuple(range(x.ndim))
    if ord == 2:
        return sqrt(sum_(square(a), axis=axis, keepdims=keepdims))
    if ord == 1:
        return sum_(a, axis=axis, keepdims=keepdims)
    if ord == math.inf:
        return max_(a, axis=axis, keepdims=keepdims)
    if ord == -math.inf:
        return min_(a, axis=axis, keepdims=keepdims)
    if ord == 0:
        from ._elementwise import cast

        return cast(count_nonzero(a, axis=axis, keepdims=keepdims), a.dtype)
    p = float(ord)
    with np.errstate(divide="ignore"):
        inv = 1.0 / p
    return pow_(sum_(pow_(a, p), axis=axis, keepdims=keepdims), inv)


def matrix_norm(x, /, *, keepdims=False, ord="fro"):  # noqa: A002 (the Array API's name)
    from ._elementwise import abs as abs_
    from ._elementwise import sqrt, square
    from ._manipulation import expand_dims
    from ._statistical import max as max_
    from ._statistical import min as min_
    from ._statistical import sum as sum_

    _check(x, "matrix_norm")
    a = abs_(_float(x))
    if ord == "fro":
        out = sqrt(sum_(square(a), axis=(-2, -1)))
    elif ord in (1, -1):
        cols = sum_(a, axis=-2)
        out = max_(cols, axis=-1) if ord == 1 else min_(cols, axis=-1)
    elif ord in (math.inf, -math.inf):
        rows = sum_(a, axis=-1)
        out = max_(rows, axis=-1) if ord == math.inf else min_(rows, axis=-1)
    elif ord in (2, -2, "nuc"):
        s = svdvals(x)
        if ord == 2:
            out = max_(s, axis=-1)
        elif ord == -2:
            out = min_(s, axis=-1)
        else:
            out = sum_(s, axis=-1)
    else:
        raise ValueError(f"invalid matrix_norm order {ord!r}")
    if keepdims:
        out = expand_dims(expand_dims(out, axis=-1), axis=-1)
    return out


def matrix_power(x, n, /):
    """``x`` to the integer power ``n``: repeated sparse products for n >= 0; inv for n < 0."""
    _check(x, "matrix_power")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("matrix_power needs square matrices")
    n = int(n)
    if n < 0:
        return matrix_power(inv(x), -n)
    if n == 0:
        from ._creation import eye
        from ._manipulation import broadcast_to

        return broadcast_to(eye(x.shape[-1], dtype=x.dtype), x.shape)
    result = None
    base = x
    while n:
        if n & 1:
            result = base if result is None else matmul(result, base)
        n >>= 1
        if n:
            base = matmul(base, base)
    return result


# Dense algebra (numpy under the densify limit)
def cholesky(x, /, *, upper=False):
    _check(x, "cholesky")

    def func(a):
        L = np.linalg.cholesky(a)
        return np.conj(np.swapaxes(L, -1, -2)) if upper else L

    return _dense_call("cholesky", func, _float(x))


def det(x, /):
    _check(x, "det")
    return _dense_call("det", np.linalg.det, _float(x))


def eigh(x, /):
    _check(x, "eigh")
    return EighResult(*_dense_call("eigh", lambda a: tuple(np.linalg.eigh(a)), _float(x)))


def eigvalsh(x, /):
    _check(x, "eigvalsh")
    return _dense_call("eigvalsh", np.linalg.eigvalsh, _float(x))


def eig(x, /):
    _check(x, "eig")
    return EigResult(*_dense_call("eig", lambda a: tuple(np.linalg.eig(a)), _float(x)))


def eigvals(x, /):
    _check(x, "eigvals")
    return _dense_call("eigvals", np.linalg.eigvals, _float(x))


def inv(x, /):
    _check(x, "inv")
    return _dense_call("inv", np.linalg.inv, _float(x))


def pinv(x, /, *, rtol=None):
    _check(x, "pinv")
    if isinstance(rtol, Array):
        rtol = rtol.todense()
    return _dense_call("pinv", lambda a: np.linalg.pinv(a, rtol=rtol), _float(x))


def matrix_rank(x, /, *, rtol=None):
    _check(x, "matrix_rank")
    if isinstance(rtol, Array):
        rtol = rtol.todense()
    return _dense_call("matrix_rank", lambda a: np.linalg.matrix_rank(a, rtol=rtol), _float(x))


def qr(x, /, *, mode="reduced"):
    _check(x, "qr")
    return QRResult(*_dense_call("qr", lambda a: tuple(np.linalg.qr(a, mode=mode)), _float(x)))


def slogdet(x, /):
    _check(x, "slogdet")
    return SlogdetResult(*_dense_call("slogdet", lambda a: tuple(np.linalg.slogdet(a)), _float(x)))


def solve(x1, x2, /):
    _check(x1, "solve")
    _check(x2, "solve", ndim=1)
    a, b = _float(x1), _float(x2)

    def func(a, b):
        if b.ndim == 1:  # (the standard's rule: a vector only when x2 is 1-d)
            return np.linalg.solve(a, b[..., None])[..., 0]
        return np.linalg.solve(a, b)

    return _dense_call("solve", func, a, b)


def svd(x, /, *, full_matrices=True):
    _check(x, "svd")
    out = _dense_call(
        "svd", lambda a: tuple(np.linalg.svd(a, full_matrices=full_matrices)), _float(x)
    )
    return SVDResult(*out)


def svdvals(x, /):
    _check(x, "svdvals")
    return _dense_call("svdvals", lambda a: np.linalg.svd(a, compute_uv=False), _float(x))

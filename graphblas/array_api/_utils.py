"""Private helpers shared by the modules of ``graphblas.array_api``."""

import math

import numpy as np

from .. import binary, config, monoid, unary
from ..core.scalar import Scalar
from ..core.tensor import Tensor
from ..core.tensor_ewise import _broadcast_factor
from ..dtypes import DataType, lookup_dtype


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


_SCALARS = {}


def gb_scalar(value, dtype):
    """A GraphBLAS Scalar of the numpy ``dtype`` holding ``value`` cast as numpy casts it.

    The Scalars are only read (as operands), so they are cached by their bytes: a new one
    costs about as much as a small element-wise operation.
    """
    value = scalar(value, dtype)
    key = (value.dtype.str, value.tobytes())
    if (s := _SCALARS.get(key)) is None:
        if len(_SCALARS) > 1024:
            _SCALARS.clear()
        s = _SCALARS[key] = Scalar.from_value(value, gb_dtype(dtype))
    return s


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
    if Ta.shape == Tb.shape:
        return
    replicated = 0
    for T in (Ta, Tb):
        factor = _broadcast_factor(T.shape, shape)
        if factor > 1:
            replicated += factor * T.nvals
    check_dense(replicated, f"broadcasting shapes {Ta.shape} and {Tb.shape} to {shape}")


def mult_expansion_check(Ta, Tb, shape):
    """Before an intersection (ewise_mult) of ``Ta`` and ``Tb`` broadcast to ``shape``: when
    both broadcast, one is replicated (an outer product); raise past the limit.
    """
    fa = _broadcast_factor(Ta.shape, shape)
    fb = _broadcast_factor(Tb.shape, shape)
    if fa > 1 and fb > 1:
        replicated = min(fa * Ta.nvals, fb * Tb.nvals)
        check_dense(replicated, f"an outer product of shapes {Ta.shape} and {Tb.shape}")


def empty_tensor(dtype, shape):
    shape = tuple(shape)
    return empty_0d(dtype) if not shape else Tensor(gb_dtype(dtype), shape)


_EMPTY_0D = {}


def empty_0d(dtype):
    """An empty 0-d Tensor of ``dtype`` (a numpy dtype or a graphblas one): a 0-d Array
    stores nothing, its value is its fill. An O(1) copy-on-write copy of one made once per
    dtype: making one costs a Matrix and a format setting, about 6 us, half of a 0-d
    operation (SAPS's time-stepping loops make one per operation).
    """
    gdt = dtype if isinstance(dtype, DataType) else gb_dtype(dtype)
    if (T := _EMPTY_0D.get(gdt)) is None:
        T = _EMPTY_0D[gdt] = Tensor(gdt, ())
    return T.dup()


def full_tensor(dtype, shape, value):
    """A Tensor holding ``value`` at every position (iso and full: O(1) memory)."""
    T = Tensor(gb_dtype(dtype), tuple(shape))
    if math.prod(shape) > 0:
        T << gb_scalar(value, dtype)
    return T


def union_pattern(tensors, shape):
    """A BOOL Tensor of ``shape`` with an entry wherever one of ``tensors``, broadcast to
    ``shape``, has one.
    """
    pair = binary.pair[lookup_dtype(bool)]
    P = None
    for T in tensors:
        if P is None:
            if T.shape == shape:
                P = T.apply(pair, right=True).new()
                continue
            P = Tensor(lookup_dtype(bool), shape)
        expansion_check(P, T, shape)
        P = P.ewise_add(T, pair, allow_expansion=True).new()
    return P


def aligned(arrays, shape):
    """Tensors of ``shape`` with one pattern, the union of the arrays' entries (broadcast),
    holding each array's values there: its entry, or its fill where it has none.
    """
    tensors = [x._T() for x in arrays]
    if len(tensors) == 1 and tensors[0].shape == shape:
        return tensors
    P = union_pattern(tensors, shape)
    out = []
    for x, T in zip(arrays, tensors, strict=True):
        dt = x.dtype
        second = binary.second[gb_dtype(dt)]
        V = P.ewise_union(T, second, gb_scalar(0, dt), gb_scalar(x._fill, dt), allow_expansion=True)
        out.append(V.new())
    return out


def with_fill(x, fill):
    """``x`` with the fill ``fill``: where it stores nothing, its old fill is stored (every
    position it does not store is written: O(size), under the densify limit).
    """
    from ._array import Array

    fill = scalar(fill, x.dtype)
    if same_value(fill, x._fill):
        return x
    check_dense(x.size, "giving an array another fill value")
    T = full_tensor(x.dtype, x.shape, x._fill)
    Tx = x._T()
    if Tx.nvals:
        T(accum=binary.second[gb_dtype(x.dtype)]) << Tx
    return Array._new(T, fill)


# Facts about an Array's stored values (not its fill), each found by one parallel GraphBLAS
# pass and cached on the Array; ``fact`` documents them. For each: the unary op that says
# whether a value has the property, the monoid that combines those answers, and the answer when
# the Array stores nothing.
# (A number in place of the unary op: whether a value equals it.)
_FACT_SCANS = {
    "finite": (unary.isfinite, monoid.land, True),
    "nan": (unary.isnan, monoid.lor, False),
    "zero": (0, monoid.lor, False),
    "posinf": (np.inf, monoid.lor, False),
    "neginf": (-np.inf, monoid.lor, False),
}


def fact(x, name):
    """A fact about the stored values of the Array ``x``, computed once (cached on ``x``):
    "finite" (every value finite), "nan" (some value is NaN), "zero" (some value is 0 or
    -0.0), "posinf" or "neginf" (some value is inf, or -inf). Integers and bools are finite,
    with no NaN or infinity, without a look; each other fact is one
    parallel GraphBLAS pass with no copy of the values (an apply to bool, then a reduction
    that stops at the first deciding value).
    """
    if name not in _FACT_SCANS:  # pragma: no cover (safety)
        raise KeyError(name)
    return x._fact(name, lambda x: _compute_fact(x, name))


def _compute_fact(x, name):
    facts = x._facts
    if name != "zero" and x.dtype.kind not in "fc":
        return name == "finite"
    if name in {"nan", "posinf", "neginf"} and facts.get("finite"):
        return False
    if name == "finite" and facts.get("nan"):
        return False
    return tensor_fact(x._T(), name)


def tensor_fact(T, name):
    """``fact`` of the stored values of the Tensor ``T``, by one pass, not cached."""
    op, combine, empty = _FACT_SCANS[name]
    if T.nvals == 0:
        return empty
    dt = T.dtype
    if not callable(op):
        B = T.apply(binary.eq[dt], right=gb_scalar(op, np_dtype(dt))).new()
    else:
        B = T.apply(op[dt]).new()
    return bool(B.reduce_scalar(combine[lookup_dtype(bool)]).new().value)


def keep_facts(src, dst, names=("finite", "nan", "zero")):
    """Give ``dst`` the facts ``src`` already knows among ``names`` (for results whose stored
    values keep them: ``abs`` and ``negative`` keep all three, a rounding keeps finiteness).
    """
    if facts := src._facts:
        kept = {k: v for k, v in facts.items() if k in names}
        if kept:
            if dst._facts is None:
                dst._facts = {}
            dst._facts.update(kept)
    return dst


def map_values(func, tensors, dtype):
    """A Tensor with the entries of ``tensors`` (all of one shape and pattern) whose values
    are ``func(*value_arrays)`` in ``dtype``, computed by numpy over the stored values
    (``Tensor._map_values``: the values in place, the pattern copied in GraphBLAS).
    """
    return tensors[0]._map_values(func, tensors[1:], gb_dtype(dtype))

"""Deferred contractions (DESIGN D9, scoped to one level of deferral).

matmul, tensordot, vecdot and einsum return an Array whose Tensor is not computed yet when
they skip the operands' unstored positions (their fills are the semiring's identity) and the
result has a dimension. Such a pending Array holds copy-on-write snapshots (``Tensor.dup``,
O(1)) of its operands' Tensors, so changing an operand afterwards does not change it (the
operand copies its data then, and the snapshots keep the old data alive until the result is
computed), and computes its Tensor on the first read (``Array._T``), once.

Two consumers use a pending product P without computing it whole:

- ``multiply(S, P)`` or ``multiply(P, S)``, with S of P's shape and both fills 0, computes P
  only at S's stored positions (a masked product, then the multiply: an SDDMM). Elsewhere S
  is 0, and 0 times P's value there is 0 when that value is finite. That holds when the
  operands are finite and no sum of products can overflow: the number of terms of each sum
  times the largest magnitude of each operand stays below the dtype's largest value (one
  GraphBLAS pass per operand for each). Otherwise P is computed whole, as numpy would, under
  the densify limit: the whole of a product meant for an SDDMM may be dense.
- ``P[key]`` with a basic index (ints and slices) slices the operands along the indexed axes
  of the result, then contracts (pending again).

Everything else reads the Tensor, which computes it. Values do not depend on whether a result
was deferred; ``enabled = False`` turns deferral off, for debugging.
"""

import numpy as np

from .. import monoid, unary
from ..core.tensor import Tensor
from ..dtypes import FC64, FP64
from . import _utils
from ._array import Array
from ._utils import gb_dtype, np_dtype

# Private kill switch: False computes every contraction when it is called
enabled = True


class Contraction:
    """A contraction not computed yet.

    ``build(*tensors)`` gives its TensorExpression; ``tensors`` are snapshots of the operands'
    Tensors. ``spec``, ``(terms, output, semiring)`` of an einsum that computes it, lets a
    slice of the result slice the operands instead (None: no such einsum). ``finite`` holds
    each operand's finiteness: True, False, or None when not checked yet. ``nterms``: the
    number of terms each result position sums (the product of the contracted sizes).
    """

    __slots__ = "tensors", "build", "gbdt", "spec", "finite", "nterms"

    def __init__(self, tensors, build, gbdt, spec, finite, nterms):
        self.tensors = tensors
        self.build = build
        self.gbdt = gbdt
        self.spec = spec
        self.finite = finite
        self.nterms = nterms

    def compute(self, mask=None):
        return self.build(*self.tensors).new(dtype=self.gbdt, mask=mask)

    def all_finite(self):
        """Whether every value of the product is finite: every operand value is finite, and
        ``nterms`` times the largest magnitude of each operand is below half the largest
        value of the result's dtype (the half covers the rounding of the sums).
        """
        for i, ok in enumerate(self.finite):
            if ok is None:
                ok = self.finite[i] = _all_finite(self.tensors[i])
            if not ok:
                return False
        dt = np_dtype(self.gbdt)
        if dt.kind not in "fc":
            return True  # (integers wrap, as numpy's do: 0 times a wrapped sum is 0)
        bound = float(self.nterms)
        for T in self.tensors:
            bound *= _max_abs(T)  # (an overflow here is inf, and fails the test)
        return bound < float(np.finfo(dt).max) / 2


def _all_finite(T):
    return np_dtype(T.dtype).kind not in "fc" or _utils.tensor_fact(T, "finite")


def _max_abs(T):
    """The largest magnitude of the stored values of ``T``, as a float."""
    if T.nvals == 0:
        return 0.0
    kind = np_dtype(T.dtype).kind
    A = T.apply(unary.abs[FC64 if kind == "c" else FP64]).new()
    return float(A.reduce_scalar(monoid.max[FP64]).new().value)


def _known_finite(x):
    """Whether ``x``'s values are finite, if ``x`` already knows (from its cached facts);
    else None.
    """
    if x.dtype.kind not in "fc":
        return True
    facts = x._facts
    return None if facts is None else facts.get("finite")


def contraction(arrays, build, shape, dtype, fill, spec=None, nterms=1):
    """The Array of ``build(*tensors)`` over the arrays' Tensors, with fill ``fill``: pending
    when deferral is on and the result has a dimension, else computed now. ``nterms``: the
    number of terms each result position sums.
    """
    tensors = [x._T() for x in arrays]
    gbdt = gb_dtype(dtype)
    if not enabled or not shape:
        return Array._new(build(*tensors).new(dtype=gbdt), fill)
    snapshots = [T.dup() for T in tensors]
    finite = [_known_finite(x) for x in arrays]
    node = Contraction(snapshots, build, gbdt, spec, finite, nterms)
    return Array._deferred(node, shape, dtype, fill)


def masked_multiply(a, b):
    """``multiply(a, b)`` when one of them is a pending product P and the other, S, lets P be
    computed at S's stored positions only (see the module's docstring); else None.
    """
    if not isinstance(a, Array) or not isinstance(b, Array):
        return None
    if b._pending is not None:
        S, P = a, b
    elif a._pending is not None:
        S, P = b, a
    else:
        return None
    if S.shape != P.shape or bool(S._fill) or bool(P._fill):
        return None
    node = P._pending
    if not node.all_finite():
        _utils.check_dense(
            P.size,
            "multiply(S, P) where P may hold inf or NaN outside S's entries (P computed whole)",
        )
        return None
    PS = Array._new(node.compute(mask=S._T().S), P._fill)
    from ._elementwise import multiply

    return multiply(PS, S) if P is a else multiply(S, PS)


def sliced(x, axes_key):
    """``x[axes_key]`` (an int or a slice per axis) of a pending ``x`` that an einsum
    computes: its operands sliced along the indexed axes, contracted (pending again). None
    when that does not apply.
    """
    node = x._pending
    if node is None or node.spec is None or all(k == slice(None) for k in axes_key):
        return None
    terms, output, typed = node.spec
    dropped = {c for c, k in zip(output, axes_key, strict=True) if not isinstance(k, slice)}
    new_terms = ["".join(c for c in term if c not in dropped) for term in terms]
    out = "".join(c for c in output if c not in dropped)
    if not out or not all(new_terms):
        return None  # (a 0-d result or operand: computed whole, then indexed)
    shape = tuple(
        len(range(*k.indices(n)))
        for k, n in zip(axes_key, x.shape, strict=True)
        if isinstance(k, slice)
    )
    keys = dict(zip(output, axes_key, strict=True))
    tensors = []
    for term, T in zip(terms, node.tensors, strict=True):
        key = tuple(keys.get(c, slice(None)) for c in term)
        if any(k != slice(None) for k in key):
            T = T[key].new()
        tensors.append(T)
    if 0 in shape or any(T.nvals == 0 for T in tensors):
        return Array._new(Tensor(node.gbdt, shape), x._fill)
    from ..core.tensor_einsum import einsum

    subscripts = ",".join(new_terms) + "->" + out

    def build(*tensors):
        return einsum(subscripts, *tensors, op=typed)

    # (a slice of a finite operand is finite)
    spec = (tuple(new_terms), out, typed)
    sub = Contraction(tensors, build, node.gbdt, spec, list(node.finite), node.nterms)
    return Array._deferred(sub, shape, x.dtype, x._fill)

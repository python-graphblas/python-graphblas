"""Deferred contractions, one level deep.

matmul, tensordot, vecdot and einsum return a pending Array, whose Tensor is not computed yet,
when they skip the operands' unstored positions (see ``_linalg``) and the result has a
dimension. A pending Array stores O(1) copy-on-write snapshots (``Tensor.dup``) of its
operands' Tensors. Changing an operand afterwards does not change the result: the operand
copies its data then, and the snapshots keep the old data until the result is computed. The
Tensor is computed on the first read (``Array._T``), once.

Two consumers use a pending product P without computing it whole:

- ``multiply(S, P)`` or ``multiply(P, S)``, with S of P's shape and both fills 0, computes P
  only at S's stored positions (a masked product, then the multiply: an SDDMM). Elsewhere S
  is 0, and 0 times P's value there is a zero when that value is finite. For floats, that
  holds when the operands are finite and no sum can overflow (``Contraction.all_finite``).
  Otherwise P is computed whole, then multiplied.
- ``P[key]`` with a basic index (ints and slices) slices the operands along the indexed axes
  of the result, then contracts them (pending again).

Everything else reads the Tensor, which computes it. Values do not depend on whether a result
was deferred. Setting ``enabled = False`` turns deferral off, for debugging.
"""

import math

import numpy as np

from .. import binary, monoid, select, semiring, unary
from ..core.tensor import Tensor
from ..dtypes import FP64, lookup_dtype
from . import _utils
from ._array import Array
from ._utils import gb_dtype, np_dtype

enabled = True


class Contraction:
    """A contraction not computed yet.

    ``tensors`` are snapshots of the operands' Tensors, and ``build(*tensors)`` returns its
    TensorExpression. ``spec`` is ``(terms, output, semiring)`` of an einsum that computes it,
    which lets a slice of the result slice the operands instead, or None when there is no such
    einsum. ``finite`` says, for each operand, whether its values are finite: True, False, or
    None when not checked yet. ``nterms`` is the number of terms each result position sums
    (the product of the contracted sizes). ``product`` says whether it is one product of its
    two operands, so that a mask goes into that product.
    """

    __slots__ = "tensors", "build", "gbdt", "spec", "finite", "nterms", "product"

    def __init__(self, tensors, build, gbdt, spec, finite, nterms, product=False):
        self.tensors = tensors
        self.build = build
        self.gbdt = gbdt
        self.spec = spec
        self.finite = finite
        self.nterms = nterms
        self.product = product

    def compute(self, mask=None, **opts):
        return self.build(*self.tensors).new(dtype=self.gbdt, mask=mask, **opts)

    def all_finite(self):
        """Whether every value of the result is finite: every operand value is finite, and
        ``nterms`` times the product of the operands' largest magnitudes is below half the
        largest value of the result's dtype (the half covers the rounding of the sums).
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
    """The largest magnitude of the stored values of ``T``, as a float (NaN skipped)."""
    if T.nvals == 0:
        return 0.0
    if np_dtype(T.dtype).kind == "c":
        # (FC64 exists only on the suitesparse backend; looking it up here lets this module
        # import on any backend, as core.tensor does)
        A = T.apply(unary.abs[lookup_dtype("FC64")]).new()
        return float(A.reduce_scalar(monoid.max[FP64]).new().value)
    # (two reductions in place, not one over a copy of |values|: an SDDMM with full operands
    # of 1e6 x 64 values and 1e7 entries in S took 0.136 s this way at 8 threads, and 0.154 s
    # with the copy)
    hi = T.reduce_scalar(monoid.max[FP64]).new().value
    lo = T.reduce_scalar(monoid.min[FP64]).new().value
    return max(abs(hi), abs(lo))


def _known_finite(x):
    """Whether the stored values of ``x`` are finite, if its cached facts say; else None."""
    if x.dtype.kind not in "fc":
        return True
    facts = x._facts
    return None if facts is None else facts.get("finite")


def contraction(arrays, build, shape, dtype, fill, spec=None, nterms=1, product=False, typed=None):
    """The Array of ``build(*tensors)`` over the arrays' Tensors, with fill ``fill``: pending
    when deferral is on and the result has a dimension, else computed at once.

    ``typed`` is the semiring, and ``build(*tensors, op=)`` builds the product with another
    one. ``nterms`` is the number of terms each result position sums; for ``product``, see
    ``Contraction``.

    When an operand stores inf or NaN, the operands' stored entries equal to their fill are
    dropped first (``prune``). An inf or NaN then contributes no term with an explicit fill,
    as it contributes none with an unstored position. When a float min or max monoid would
    skip a NaN term, the result is computed at once, with NaN at each position that has one
    (see ``_with_nan_terms``).
    """
    if any(x.dtype.kind in "fc" and not _utils.fact(x, "finite") for x in arrays):
        from ._manipulation import prune

        arrays = [prune(x) for x in arrays]
    tensors = [x._T() for x in arrays]
    gbdt = gb_dtype(dtype)
    nan = (
        typed is not None
        and typed.monoid.name in {"min", "max"}
        and np_dtype(typed.return_type).kind == "f"
        and any(x.dtype.kind == "f" and _utils.fact(x, "nan") for x in arrays)
    )
    if nan:
        return Array._new(_with_nan_terms(build, tensors, gbdt), fill)
    if not enabled or not shape:
        return Array._new(build(*tensors).new(dtype=gbdt), fill)
    snapshots = [T.dup() for T in tensors]
    finite = [_known_finite(x) for x in arrays]
    node = Contraction(snapshots, build, gbdt, spec, finite, nterms, product)
    return Array._deferred(node, shape, dtype, fill)


def _with_nan_terms(build, tensors, gbdt):
    """``build(*tensors)``, with NaN at each position where a term has a stored NaN operand.

    SuiteSparse's float min and max are C's fmin and fmax, which skip NaN; numpy's propagate
    it. Those positions are the any_pair product of one operand's NaN entries with the other
    operands' patterns.
    """
    R = build(*tensors).new(dtype=gbdt)
    bool_ = lookup_dtype(bool)
    patterns = [T.apply(unary.one[bool_]).new() for T in tensors]
    N = None
    for i, T in enumerate(tensors):
        # (every operand is float here: its fill is the identity of min or max, an infinity)
        nan = T.apply(unary.isnan).new().select(select.valueeq, True).new()
        if nan.nvals == 0:
            continue
        ops = [nan if j == i else P for j, P in enumerate(patterns)]
        Ni = build(*ops, op=semiring.any_pair[bool_]).new()
        N = Ni if N is None else N.ewise_add(Ni, binary.lor).new()
    if N is not None and N.nvals:
        R(N.S) << _utils.gb_scalar(np.nan, np_dtype(gbdt))
    return R


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
    if S is P or S.shape != P.shape or bool(S._fill) or bool(P._fill):
        return None  # (P * P: computed once, as a whole)
    node = P._pending
    if not node.all_finite():
        return None  # (P is computed whole, a sparse product as without deferral)
    TS = S._T()
    hint = _dot_hint(node, TS.nvals, P.shape)
    PS = Array._new(node.compute(mask=TS.S, **hint), P._fill)
    from ._elementwise import multiply

    return multiply(PS, S) if P is a else multiply(S, PS)


def _dot_hint(node, nmask, shape=()):
    """``{"axb_method": "dot"}`` for a product under a mask of ``nmask`` entries when dot
    products at the mask's entries cost well below a saxpy over the whole product, else ``{}``.

    For A @ B with B stored by row, SuiteSparse chooses a masked saxpy, which does all of
    A @ B's work. An SDDMM with full A (n x 32) and B (32 x n), n = 4000, and 4e4 mask entries
    took 0.6 s that way at 1 thread, and 0.6 ms as dot products (SuiteSparse transposes B for
    them).
    """
    if not node.product:
        return {}
    Ta, Tb = node.tensors
    k = max(node.nterms, 1)
    # (a batched product pairs entries within each batch only)
    if not shape:
        nbatch = 1
    elif node.spec is not None:
        (ta, tb), output = node.spec[:2]
        nbatch = math.prod(shape[output.index(c)] for c in output if c in ta and c in tb)
    else:
        nbatch = math.prod(shape[:-2])  # (a matmul whose batch axes broadcast)
    saxpy = Ta.nvals * Tb.nvals / (k * max(nbatch, 1))
    dot = nmask * k * (Ta.nvals / max(Ta.size, 1) + Tb.nvals / max(Tb.size, 1))
    return {"axb_method": "dot"} if 4 * dot < saxpy else {}


def sliced(x, axes_key):
    """``x[axes_key]`` (an int or a slice per axis) of a pending ``x`` that an einsum
    computes: its operands sliced along the indexed axes, then contracted (pending again).
    None when that does not apply.
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

    def build(*tensors, op=typed):
        return einsum(subscripts, *tensors, op=op)

    spec = (tuple(new_terms), out, typed)
    product = _one_product(new_terms, out)
    # (a slice of a finite operand is finite)
    sub = Contraction(tensors, build, node.gbdt, spec, list(node.finite), node.nterms, product)
    return Array._deferred(sub, shape, x.dtype, x._fill)


def _one_product(terms, output):
    """Whether an einsum of ``terms`` into ``output`` is one product of two operands, whose
    result needs no permutation or sum after it (so a mask goes into the product).
    """
    from ..core.tensor_einsum import _direct_axes

    return len(terms) == 2 and _direct_axes(*terms, output) is not None

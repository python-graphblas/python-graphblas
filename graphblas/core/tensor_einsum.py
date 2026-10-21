"""einsum over Tensors with any semiring.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import functools
import itertools
import math

import numpy as np

from .. import semiring
from ..dtypes import lookup_dtype, unify
from ..exceptions import DimensionMismatch
from . import tensor_census as census
from .operator import get_typed_op


def einsum(subscripts, *operands, op=semiring.plus_times, optimize="greedy"):
    """Einstein summation over sparse Tensors with a semiring (numpy.einsum subscripts).

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.**

    ``op``'s multiply combines the operands and its monoid sums over the contracted indices;
    a result position is present when at least one term exists. Operands may be Tensors,
    Matrices, Vectors or ndarrays. Each index appears at most once per operand (no
    diagonals) and ``...`` is not supported. With three or more operands the pairwise order
    comes from ``opt_einsum.contract_path`` when it is installed (else a greedy choice by
    result size); the MTTKRP pattern (``'ijk,jr,kr->ir'``) goes to ``Tensor.mttkrp`` instead,
    which never forms the Khatri-Rao product.
    """
    pre = []
    for x in operands:
        if type(x) is np.ndarray:
            pre.append(x)  # converted below, as a factor Matrix in an MTTKRP
        elif (t := _as_tensor(x, "einsum")) is None:
            raise TypeError(f"einsum operands must be Tensor-like, not {type(x)}")
        else:
            pre.append(t)
    shapes = tuple(x.shape for x in pre)
    try:
        plan = _einsum_plan(subscripts, shapes, optimize)
    except TypeError:  # an unhashable ``optimize`` (an explicit path): plan without the cache
        plan = _einsum_plan.__wrapped__(subscripts, shapes, optimize)
    terms, output, shape, mt, path, direct = plan
    if census.on:
        kind = "mttkrp" if mt is not None else "one contraction" if direct is not None else None
        census.note("einsum", kind or f"{len(terms)} operands, pairwise", subscripts=subscripts)
    dtype = pre[0].dtype if type(pre[0]) is not np.ndarray else lookup_dtype(pre[0].dtype)
    for x in pre[1:]:
        dtype = unify(dtype, x.dtype if type(x) is not np.ndarray else lookup_dtype(x.dtype))
    typed = get_typed_op(op, dtype, dtype, kind="semiring")
    if typed.opclass != "Semiring":
        raise TypeError(f"einsum needs a Semiring, not {typed.opclass}")
    expr_repr = f"einsum({subscripts!r}, ...)".replace("{", "{{").replace("}", "}}")
    if mt is not None:
        # An MTTKRP: Tensor.mttkrp's expression, given the factors as they came (an ndarray
        # becomes a factor Matrix with one copy)
        xi, factor_of, mode, rank_first = mt
        x = pre[xi] if type(pre[xi]) is not np.ndarray else Tensor.from_dense(pre[xi])
        factors = [None] * x.ndim
        for i, ax, transposed in factor_of:
            factors[ax] = pre[i].T if transposed else pre[i]
        inner = _mttkrp_expr(x, factors, mode, typed)
        if rank_first:
            func, func_args = _einsum_mttkrp_transposed_recipe, (inner,)
        else:
            func, func_args = _einsum_delegate_recipe, (inner,)
        return TensorExpression(
            "einsum", func, [x], func_args, dtype=inner.dtype, shape=shape, op=typed,
            expr_repr=expr_repr,
        )  # fmt: skip
    tensors = [x if type(x) is not np.ndarray else Tensor.from_dense(x) for x in pre]
    if direct is not None:
        # One contraction whose result's axes are already in the output's order: run its
        # expression's recipe on einsum's target, with nothing in between
        inner = _contraction_expr(*tensors, typed, direct)
        return TensorExpression(
            "einsum", _einsum_delegate_recipe, tensors, (inner,), dtype=inner.dtype,
            shape=shape, op=typed, expr_repr=expr_repr,
        )  # fmt: skip
    return TensorExpression(
        "einsum",
        _einsum_recipe,
        tensors,
        (terms, output, tensors, typed, path),
        dtype=typed.return_type,
        shape=shape,
        op=typed,
        expr_repr=expr_repr,
    )


@functools.lru_cache(maxsize=1024)
def _einsum_plan(subscripts, shapes, optimize):
    """What ``einsum`` needs from the subscripts and the operands' shapes alone, cached: the
    terms, the output letters and shape, the MTTKRP pattern (or None), the pairwise path, and
    for one contraction already in the output's order its axes (or None).
    """
    if "." in subscripts:
        raise NotImplementedError("einsum: `...` is not supported")
    subscripts = subscripts.replace(" ", "")
    if "->" in subscripts:
        inputs, output = subscripts.split("->")
    else:
        inputs, output = subscripts, None
    terms = tuple(inputs.split(","))
    if len(terms) != len(shapes):
        raise ValueError(f"einsum: {len(terms)} terms for {len(shapes)} operands")
    for term, shape in zip(terms, shapes, strict=True):
        if len(term) != len(shape):
            raise ValueError(f"einsum: term {term!r} does not match an operand of shape {shape}")
        if len(set(term)) != len(term):
            raise NotImplementedError(f"einsum: repeated index in {term!r} (a diagonal)")
    sizes = {}
    for term, shape in zip(terms, shapes, strict=True):
        for c, n in zip(term, shape, strict=True):
            if sizes.setdefault(c, n) != n:
                raise DimensionMismatch(f"einsum: index {c!r} has sizes {sizes[c]} and {n}")
    if output is None:
        counts = {}
        for term in terms:
            for c in term:
                counts[c] = counts.get(c, 0) + 1
        output = "".join(sorted(c for c, k in counts.items() if k == 1))
    if len(set(output)) != len(output) or any(c not in sizes for c in output):
        raise ValueError(f"einsum: bad output {output!r}")
    shape = tuple(sizes[c] for c in output)
    if (mt := _mttkrp_pattern(terms, output)) is not None:
        return terms, output, shape, mt, None, None
    path = tuple(_einsum_path(terms, output, shapes, optimize))
    direct = _direct_axes(*terms, output) if len(terms) == 2 else None
    return terms, output, shape, None, path, direct


def _direct_axes(ta, tb, output):
    """``(a_axes, b_axes, a_batch, b_batch)`` when contracting terms ``ta`` and ``tb`` into
    ``output`` is one contraction (tensordot, or batched) whose result's axes are already in
    the output's order, with nothing to sum out first; else None.
    """
    keep = set(output)
    shared = [c for c in ta if c in tb]
    batch = [c for c in shared if c in keep]
    contracted = [c for c in shared if c not in keep]
    free_a = [c for c in ta if c not in tb]
    free_b = [c for c in tb if c not in ta]
    if (
        not contracted
        or not all(c in keep for c in free_a + free_b)
        or "".join(batch + free_a + free_b) != output
    ):
        return None
    return (
        tuple(ta.index(c) for c in contracted),
        tuple(tb.index(c) for c in contracted),
        tuple(ta.index(c) for c in batch),
        tuple(tb.index(c) for c in batch),
    )


def _contraction_expr(A, B, op, direct):
    a_axes, b_axes, a_batch, b_batch = direct
    if a_batch:
        return _batched_contract_expr(A, B, op, a_axes, b_axes, a_batch, b_batch)
    return A.tensordot(B, (a_axes, b_axes), op)


def _mttkrp_pattern(terms, output):
    """``(index of X, ((operand, axis of X, transposed), ...), mode, rank first)`` when the
    einsum is an MTTKRP: one operand X of 3+ axes, and 2-d factors that each pair one index
    of X with a common rank index r, covering every index of X but the mode's, with output
    (mode, r) or (r, mode). A factor written (r, index) is ``transposed``.
    """
    big = [i for i, t in enumerate(terms) if len(t) >= 3]
    if len(terms) < 3 or len(output) != 2 or len(big) != 1:
        return None
    xi = big[0]
    xterm = terms[xi]
    seen = [False] * len(xterm)
    factor_of = []
    ranks = set()
    for i, term in enumerate(terms):
        if i == xi:
            continue
        inside = [c for c in term if c in xterm]
        if len(term) != 2 or len(inside) != 1:
            return None
        r = term[1] if term[0] == inside[0] else term[0]
        ranks.add(r)
        ax = xterm.index(inside[0])
        if seen[ax]:
            return None
        seen[ax] = True
        factor_of.append((i, ax, term[0] != inside[0]))
    missing = [ax for ax, s in enumerate(seen) if not s]
    if len(ranks) != 1 or len(missing) != 1:
        return None
    r = ranks.pop()
    mode = missing[0]
    if set(output) != {xterm[mode], r}:
        return None
    return xi, tuple(factor_of), mode, output[0] == r


def _einsum_delegate_recipe(updater, inner):
    inner.args[-2](updater, *inner.args[-1])


def _einsum_mttkrp_transposed_recipe(updater, inner):
    R = inner.new(**updater.opts).permute_dims((1, 0))
    _store(updater, R._M, R._shape, R._layout, share=R)


def _einsum_recipe(updater, terms, output, tensors, op, path):
    ops = list(zip(terms, tensors, strict=True))
    for step, (i, j) in enumerate(path):
        (ta, A), (tb, B) = ops[i], ops[j]
        for k in sorted((i, j), reverse=True):
            del ops[k]
        if step == len(path) - 1 and (direct := _direct_axes(ta, tb, output)) is not None:
            # The last contraction runs on einsum's target: a mask (an RPQ's ~visited) or an
            # accumulator goes into its product, not onto a computed temporary
            inner = _contraction_expr(A, B, op, direct)
            inner.args[-2](updater, *inner.args[-1])
            return
        keep = set(output).union(*(t for t, _ in ops))
        ops.append(_einsum_pair(ta, A, tb, B, keep, op, updater.opts))
    ((letters, result),) = ops
    extra = [k for k, c in enumerate(letters) if c not in output]
    if extra:
        result = result.reduce(op.monoid, axis=tuple(extra)).new(**updater.opts)
        letters = "".join(c for c in letters if c in output)
    final = result.permute_dims(tuple(letters.index(c) for c in output))
    _store(updater, final._M, final._shape, final._layout, share=final)


def _einsum_path(terms, output, shapes, optimize):
    n = len(terms)
    if n == 1:
        return []
    if n > 2 and optimize:
        try:
            import opt_einsum
        except ImportError:
            pass
        else:
            sub = ",".join(terms) + "->" + output
            return [tuple(p) for p in opt_einsum.contract_path(sub, *shapes, shapes=True,
                                                               optimize=optimize)[0]]  # fmt: skip
    # Greedy: contract the pair with the smallest result (product of its kept index sizes)
    path = []
    cur = list(terms)
    sizes = {}
    for term, shape in zip(terms, shapes, strict=True):
        sizes.update(zip(term, shape, strict=True))
    while len(cur) > 1:
        best = None
        for i, j in itertools.combinations(range(len(cur)), 2):
            rest = set(output).union(*(cur[k] for k in range(len(cur)) if k not in (i, j)))
            letters = (set(cur[i]) | set(cur[j])) & rest
            cost = math.prod(sizes[c] for c in letters)
            if best is None or cost < best[0]:
                best = (cost, i, j, "".join(sorted(letters)))
        _, i, j, letters = best
        path.append((i, j))
        cur = [cur[k] for k in range(len(cur)) if k not in (i, j)] + [letters]
    return path


def _einsum_pair(ta, A, tb, B, keep, op, opts):
    """Contract one pair; returns (letters, Tensor)."""
    # Indices that appear in only one operand and nowhere later are summed out first
    only = tuple(k for k, c in enumerate(ta) if c not in tb and c not in keep)
    if only:
        A = A.reduce(op.monoid, axis=only).new(**opts)
        ta = "".join(c for k, c in enumerate(ta) if k not in only)
    only = tuple(k for k, c in enumerate(tb) if c not in ta and c not in keep)
    if only:
        B = B.reduce(op.monoid, axis=only).new(**opts)
        tb = "".join(c for k, c in enumerate(tb) if k not in only)
    shared = [c for c in ta if c in tb]
    batch = [c for c in shared if c in keep]
    contracted = [c for c in shared if c not in keep]
    free_a = [c for c in ta if c not in tb]
    free_b = [c for c in tb if c not in ta]
    if contracted:
        a_axes = tuple(ta.index(c) for c in contracted)
        b_axes = tuple(tb.index(c) for c in contracted)
        a_batch = tuple(ta.index(c) for c in batch)
        b_batch = tuple(tb.index(c) for c in batch)
        if batch:
            R = _batched_contract_expr(A, B, op, a_axes, b_axes, a_batch, b_batch).new(**opts)
        else:
            R = A.tensordot(B, (a_axes, b_axes), op).new(**opts)
        return "".join(batch + free_a + free_b), R
    # No contraction: an element-wise product, broadcasting over the free indices
    letters = batch + free_a + free_b
    pa = A.permute_dims(tuple(ta.index(c) for c in batch + free_a))
    pb = B.permute_dims(tuple(tb.index(c) for c in batch + free_b))
    if free_b:
        pa = pa.expand_dims(tuple(range(len(batch) + len(free_a), len(letters))))
    if free_a:
        pb = pb.expand_dims(tuple(range(len(batch), len(batch) + len(free_a))))
    R = pa.ewise_mult(pb, op.binaryop).new(**opts)
    return "".join(letters), R


# isort: off
from .tensor import (  # noqa: E402
    Tensor,
    TensorExpression,
    _as_tensor,
    _store,
)
from .tensor_contract import _batched_contract_expr  # noqa: E402
from .tensor_engines import _mttkrp_expr  # noqa: E402

# isort: on

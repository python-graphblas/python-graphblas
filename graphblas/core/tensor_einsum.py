"""einsum over Tensors with any semiring.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import functools
import itertools
import math
import operator

import numpy as np

from .. import semiring
from ..dtypes import lookup_dtype, unify
from ..exceptions import DimensionMismatch
from . import tensor_census as census
from .operator import get_typed_op


def einsum(subscripts, *operands, op=semiring.plus_times, optimize="greedy"):
    """Einstein summation over sparse Tensors with a semiring, using numpy.einsum's subscripts.

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.**

    The multiply of the semiring ``op`` combines the operands of each term left to right
    (``mult(mult(a, b), c)``), and its monoid sums the terms over the contracted indices. A
    result position has an entry when at least one term exists there. Operands may be
    Tensors, Matrices, Vectors or ndarrays. As in numpy, ``...`` stands for the axes the
    letters leave, which broadcast against each other (a size-1 axis against any size), and
    an index repeated within one operand takes that operand's diagonal: ``"ii->i"`` is the
    diagonal and ``"ii"`` the trace. A diagonal costs one pass over the operand's entries.

    When the multiply distributes over the monoid (``plus_times``, ``min_plus``,
    ``max_plus``, ``lor_land``, ``min_max``, any ``min``, ``max`` or ``any`` with ``first``,
    ``second`` or ``pair``, ...), an index that only one operand has is summed out first,
    and three or more operands are contracted in pairs. The pairs come in the order
    ``optimize`` chooses when the multiply also commutes, else left to right. The MTTKRP
    pattern (``"ijk,jr,kr->ir"``) then goes to ``Tensor.mttkrp``, which never forms the
    Khatri-Rao product. Other semirings (``plus_pair``, ``plus_first``, ``max_times``, ...)
    form every term, left to right, and sum only at the end, which takes more memory.
    Regrouping is exact up to floating-point rounding and integer overflow: an integer
    ``min_plus`` or ``max_plus`` sum that overflows may wrap at another step than the
    term-by-term sum does.

    ``optimize`` chooses the pairwise order. It may be an ``opt_einsum`` strategy (default
    ``"greedy"``, or True), used through ``opt_einsum.contract_path`` when that is installed;
    without ``opt_einsum``, or with ``optimize=False``, the order is a greedy choice by result
    size. It may also be an explicit path, as ``opt_einsum.contract_path`` and
    ``numpy.einsum_path`` return it: a sequence of pairs of positions in the list of operands
    left at each step, each pair's result going last. numpy's leading ``"einsum_path"`` may
    stay. An explicit path is used as given, with or without ``opt_einsum``.
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
    if (rewritten := _einsum_rewrite(subscripts, shapes)) is not None:
        subscripts, preps = rewritten
        pre = [x if p is None else _prepare_operand(x, *p) for x, p in zip(pre, preps, strict=True)]
        shapes = tuple(x.shape for x in pre)
    try:
        plan = _einsum_plan(subscripts, shapes, optimize)
    except TypeError:  # an unhashable ``optimize`` (an explicit path): plan without the cache
        plan = _einsum_plan.__wrapped__(subscripts, shapes, optimize)
    terms, output, shape, mt, path, direct = plan
    dtype = pre[0].dtype if type(pre[0]) is not np.ndarray else lookup_dtype(pre[0].dtype)
    for x in pre[1:]:
        dtype = unify(dtype, x.dtype if type(x) is not np.ndarray else lookup_dtype(x.dtype))
    typed = get_typed_op(op, dtype, dtype, kind="semiring")
    if typed.opclass != "Semiring":
        raise TypeError(f"einsum needs a Semiring, not {typed.opclass}")
    grouping = _grouping(typed)
    if census.on:
        if mt is not None and grouping == "any order":
            kind = "mttkrp"
        elif direct is not None:
            kind = "one contraction"
        else:
            kind = f"{len(terms)} operands, " + (
                "pairwise" if grouping == "any order" else grouping
            )
        census.note("einsum", kind, subscripts=subscripts)
    expr_repr = f"einsum({subscripts!r}, ...)".replace("{", "{{").replace("}", "}}")
    if mt is not None and grouping == "any order":
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
        # expression's recipe on einsum's target, with nothing in between (one product, so
        # right for any semiring)
        inner = _contraction_expr(*tensors, typed, direct)
        return TensorExpression(
            "einsum", _einsum_delegate_recipe, tensors, (inner,), dtype=inner.dtype,
            shape=shape, op=typed, expr_repr=expr_repr,
        )  # fmt: skip
    return TensorExpression(
        "einsum",
        _einsum_recipe,
        tensors,
        (terms, output, tensors, typed, path, grouping),
        dtype=typed.return_type,
        shape=shape,
        op=typed,
        expr_repr=expr_repr,
    )


# (monoid, multiply) pairs whose multiply distributes over the monoid on both sides,
# a (x) (b (+) c) = (a (x) b) (+) (a (x) c) and likewise on the right, so summing an index out
# early, or contracting in steps, gives the sum of the terms. Each multiply here is associative.
# This is exact up to rounding and integer overflow. plus_times stays exact when it wraps
# (wrapping arithmetic is a ring), but a min_plus or max_plus sum that overflows can wrap at
# another step once regrouped. They stay in the set: they are common, and an overflowing sum is
# no meaningful answer either way. max_times and min_times do not distribute over signed values
# even without overflow, so they are left out.
_DISTRIBUTIVE = frozenset(
    {("plus", "times"), ("min", "plus"), ("max", "plus"), ("min", "max"), ("max", "min"),
     ("lor", "land"), ("land", "lor"), ("lxor", "land"), ("bor", "band"), ("band", "bor"),
     ("bxor", "band")}
)  # fmt: skip
# Monoids with x (+) x = x: first, second, pair and the monoid's own op distribute over them
_IDEMPOTENT = frozenset({"min", "max", "lor", "land", "bor", "band"})
_ASSOCIATIVE = frozenset(
    {"times", "plus", "min", "max", "first", "second", "pair", "oneb", "any", "land", "lor",
     "lxor", "band", "bor", "bxor"}
)  # fmt: skip


def _grouping(op):
    """How the terms of the typed semiring ``op`` may be grouped. "any order": sum an index
    out early, contract in pairs in any order (``opt_einsum``'s), either operand on either
    side. "in order": the same, but contract left to right, each intermediate on the left
    (a multiply that is not commutative). "terms": form every term left to right, and sum
    only at the end (any semiring).
    """
    monoid, mult = op.monoid.name, op.binaryop.name
    if (monoid, mult) in _DISTRIBUTIVE or (
        monoid in _IDEMPOTENT and mult in {monoid, "first", "second", "pair", "oneb"}
    ):
        return "any order" if op.binaryop.is_commutative else "in order"
    if monoid == "any":
        # Any one term's value: summing early or in steps still gives one, as long as each
        # term multiplies its operands left to right
        commutes = op.binaryop.is_commutative and mult in _ASSOCIATIVE
        return "any order" if commutes else "in order"
    return "terms"


@functools.lru_cache(maxsize=1024)
def _einsum_plan(subscripts, shapes, optimize):
    """What ``einsum`` needs from the subscripts and the operands' shapes alone, cached: the
    terms, the output letters and shape, the MTTKRP pattern (or None), the pairwise path, and
    for one contraction already in the output's order its axes (or None). ``einsum`` rewrites
    ``...`` and repeated indices first (``_einsum_rewrite``); a caller that plans without
    that rewrite gets NotImplementedError for them.
    """
    if "." in subscripts:
        raise NotImplementedError("einsum: `...` is not supported here")
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
            raise NotImplementedError(f"einsum: repeated index in {term!r} (a diagonal) here")
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


@functools.lru_cache(maxsize=1024)
def _einsum_rewrite(subscripts, shapes):
    """Rewrite numpy's ``...`` and repeated indices for operands of ``shapes``, cached.

    Returns ``(subscripts, preps)``: the subscripts with a letter for each axis, no letter
    twice in a term and an explicit output, and for each operand None or ``(squeeze,
    groups)`` for ``_prepare_operand``. None when ``subscripts`` has neither.

    ``...`` gets letters the subscripts do not use, right-aligned as numpy broadcasts. A
    size-1 axis under it that meets a longer one is squeezed and its letter dropped from that
    operand, which is how einsum broadcasts: the other operands' entries pair with every
    position of a letter it lacks.
    """
    subscripts = subscripts.replace(" ", "")
    inputs, arrow, output = subscripts.partition("->")
    terms = inputs.split(",")
    if "." not in subscripts and all(len(set(t)) == len(t) for t in terms):
        return None
    if len(terms) != len(shapes):
        raise ValueError(f"einsum: {len(terms)} terms for {len(shapes)} operands")
    free = [c for c in _LETTERS if c not in subscripts]
    parts = []
    for term, shape in zip(terms, shapes, strict=True):
        head, dots, tail = term.partition("...")
        if "." in head or "." in tail:
            raise ValueError(f"einsum: a '.' in {term!r} is not part of an ellipsis ('...')")
        if (n := len(shape) - len(head) - len(tail)) < 0 or not dots and n:
            raise ValueError(
                f"einsum: term {term!r} has {'more' if n < 0 else 'fewer'} indices than an "
                f"operand of shape {shape}"
            )
        parts.append((head, tail, n))
    nell = max(n for _, _, n in parts)
    if nell > len(free):
        raise ValueError(f"einsum: {nell} axes under '...' and {len(free)} unused letters")
    ell = free[:nell]
    sizes = dict.fromkeys(ell, 1)
    for (head, _, n), shape in zip(parts, shapes, strict=True):
        for c, m in zip(ell[nell - n :], shape[len(head) : len(head) + n], strict=True):
            if m != 1 and sizes[c] not in {1, m}:
                raise DimensionMismatch(
                    f"einsum: axes of sizes {sizes[c]} and {m} under '...' do not broadcast"
                )
            sizes[c] = max(sizes[c], m)
    if arrow:
        out_head, out_dots, out_tail = output.partition("...")
        if "." in out_head or "." in out_tail:
            raise ValueError(f"einsum: a '.' in output {output!r} is not part of an ellipsis")
        if nell and not out_dots:
            raise ValueError(
                f"einsum: the output {output!r} has no '...' for the {nell} axes under '...'"
            )
        output = out_head + "".join(ell) + out_tail
    else:
        # numpy's implicit output, counted on the given subscripts: the axes under ``...``,
        # then the letters that appear once, sorted (so "ii" is a trace)
        counts = {}
        for c in inputs.replace("...", "").replace(",", ""):
            counts[c] = counts.get(c, 0) + 1
        output = "".join(ell) + "".join(sorted(c for c, k in counts.items() if k == 1))
    new_terms, preps = [], []
    for (head, tail, n), shape in zip(parts, shapes, strict=True):
        letters, squeeze, kept = [*head], [], [*shape[: len(head)]]
        for i, c in enumerate(ell[nell - n :]):
            if shape[len(head) + i] == 1 and sizes[c] > 1:
                squeeze.append(len(head) + i)
            else:
                letters.append(c)
                kept.append(shape[len(head) + i])
        letters += tail
        kept += shape[len(shape) - len(tail) :]
        unique = list(dict.fromkeys(letters))
        groups = None
        if len(unique) < len(letters):
            groups = tuple(tuple(i for i, c in enumerate(letters) if c == u) for u in unique)
            for u, g in zip(unique, groups, strict=True):
                if len({kept[i] for i in g}) > 1:
                    raise DimensionMismatch(
                        f"einsum: index {u!r} has sizes {[kept[i] for i in g]} in term {term!r}"
                    )
        new_terms.append("".join(unique))
        preps.append(None if not squeeze and groups is None else (tuple(squeeze), groups))
    return ",".join(new_terms) + "->" + output, tuple(preps)


_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _prepare_operand(x, squeeze, groups):
    """``x`` (a Tensor or an ndarray) with the axes ``squeeze`` removed, then, if ``groups``
    is given, its diagonal: an axis for each group of axes, with the entries whose
    coordinates agree across each group.
    """
    if type(x) is np.ndarray:
        if squeeze:
            x = x.squeeze(axis=squeeze)
        if groups is not None:
            labels = [0] * x.ndim
            for k, g in enumerate(groups):
                for i in g:
                    labels[i] = k
            x = np.einsum(x, labels, list(range(len(groups))))
        return x
    if squeeze:
        x = x.squeeze(squeeze)
    if groups is None:
        return x
    # One pass over the entries in their stored order, then a build
    coords, values = x.to_coo(sort=False)
    keep = np.ones(values.shape[0], bool)
    for g in groups:
        for i in g[1:]:
            keep &= coords[i] == coords[g[0]]
    shape = tuple(x.shape[g[0]] for g in groups)
    return Tensor.from_coo([coords[g[0]][keep] for g in groups], values[keep], x.dtype, shape=shape)


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


def _einsum_recipe(updater, terms, output, tensors, op, path, grouping):
    ops = list(zip(terms, tensors, strict=True))
    if grouping != "any order":
        # Left to right, each intermediate in place of its left operand
        path = [(0, 1)] * (len(ops) - 1)
    for step, (i, j) in enumerate(path):
        (ta, A), (tb, B) = ops[i], ops[j]
        for k in sorted((i, j), reverse=True):
            del ops[k]
        if step == len(path) - 1 and (direct := _direct_axes(ta, tb, output)) is not None:
            # The last contraction runs on einsum's target: a mask (a regular path query's
            # ~visited) or an accumulator goes into its product, not onto a computed temporary
            inner = _contraction_expr(A, B, op, direct)
            inner.args[-2](updater, *inner.args[-1])
            return
        keep = set(output).union(*(t for t, _ in ops))
        if grouping == "terms" and ops:
            keep |= set(ta) | set(tb)  # no sum before the last step
        elif grouping == "terms":
            # The last step contracts only the indices both have; the others are summed out
            # after (below)
            keep |= set(ta) ^ set(tb)
        pair = _einsum_pair(ta, A, tb, B, keep, op, updater.opts)
        if grouping == "any order":
            ops.append(pair)  # (where opt_einsum's paths expect it)
        else:
            ops.insert(0, pair)
    ((letters, result),) = ops
    extra = [k for k, c in enumerate(letters) if c not in output]
    if extra:
        result = result.reduce(op.monoid, axis=tuple(extra)).new(**updater.opts)
        letters = "".join(c for c in letters if c in output)
    final = result.permute_dims(tuple(letters.index(c) for c in output))
    _store(updater, final._M, final._shape, final._layout, share=final)


def _einsum_path(terms, output, shapes, optimize):
    n = len(terms)
    if not isinstance(optimize, (str, bool)) and optimize is not None:
        return _explicit_path(optimize, n)
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


def _explicit_path(path, n):
    """An explicit pairwise path for ``n`` operands, checked: pairs of distinct positions among
    the operands left at each step, down to one.
    """
    steps = list(path)
    if steps and isinstance(steps[0], str) and steps[0] == "einsum_path":
        steps = steps[1:]  # (numpy.einsum_path's form)
    out = []
    left = n
    for step in steps:
        try:
            pair = tuple(operator.index(k) for k in step)
        except TypeError:
            raise TypeError(
                f"einsum: each step of an explicit path must be a pair of operand positions; "
                f"got {step!r}"
            ) from None
        if len(pair) != 2 or pair[0] == pair[1] or not all(0 <= k < left for k in pair):
            raise ValueError(
                f"einsum: path step {step!r} does not name two of the {left} operands left at "
                "that step"
            )
        out.append(pair)
        left -= 1
    if left != 1:
        raise ValueError(f"einsum: the path leaves {left} operands; it must contract them to one")
    return out


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

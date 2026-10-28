"""Element-wise operations and broadcasting for Tensors.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import itertools
import math

import numpy as np

from .. import binary, config, monoid
from ..dtypes import BOOL, unify
from ..exceptions import DimensionMismatch
from . import tensor_census as census
from . import tensor_layout as tl
from .matrix import Matrix
from .operator import get_semiring, get_typed_op
from .scalar import _union_default
from .tensor_layout import Layout
from .vector import Vector


def _ewise_expr(
    self, method, other, op, left_default=None, right_default=None, *, allow_expansion=False
):
    if (tensor := _as_tensor(other, method)) is None:
        self._expect_type(other, Tensor, within=method, argname="other", op=op)
    other = tensor
    shape = self._shape
    if other._shape != shape:
        try:
            shape = _broadcast_shapes(self._shape, other._shape)
        except ValueError:
            raise DimensionMismatch(
                f"Tensor shapes {self._shape} and {other._shape} do not broadcast in {method}"
            ) from None
    untyped_op = op
    extra = ()
    if method == "ewise_union":
        temp_op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        left = _union_default(self, left_default, temp_op.type, op, "left_default")
        right = _union_default(self, right_default, temp_op.type2, op, "right_default")
        op1 = get_typed_op(op, self.dtype, right.dtype, kind="binary")
        op2 = get_typed_op(op, left.dtype, other.dtype, kind="binary")
        if op1 is not op2:
            left_dtype = unify(op1.type, op2.type, is_right_scalar=True)
            right_dtype = unify(op1.type2, op2.type2, is_left_scalar=True)
            op = get_typed_op(op, left_dtype, right_dtype, kind="binary")
        else:
            op = op1
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        extra = (left, right)
    else:
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method, argname="op")
    expr_repr = "{0.name}.%s({1.name}, op={op})" % method  # noqa: UP031
    if shape == self._shape == other._shape:
        return TensorExpression(
            method,
            _ewise_recipe,
            [self, other],
            (method, self, other, op, extra),
            dtype=op.return_type,
            shape=shape,
            op=op,
            expr_repr=expr_repr,
        )
    _check_expansion(method, (self, other), shape, allow_expansion)
    return TensorExpression(
        method,
        _bcast_recipe,
        [self, other],
        (method, self, other, op, untyped_op, extra, shape),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr=expr_repr,
    )


def _broadcast_factor(tshape, shape):
    """How many times an operand of ``tshape`` repeats to fill ``shape``."""
    off = len(shape) - len(tshape)
    factor = math.prod(shape[:off])
    for a, n in enumerate(tshape):
        if n == 1:
            factor *= shape[a + off]
    return factor


def _check_expansion(method, operands, shape, allow):
    sizes = [(_broadcast_factor(t._shape, shape), t._M._nvals) for t in operands]
    if method == "ewise_mult":
        if any(f == 1 for f, _ in sizes):
            return  # one operand covers the result: diagonal scaling, no replication
        replicated = min(f * n for f, n in sizes)
    else:
        replicated = sum(f * n for f, n in sizes if f > 1)
    limit = config.get("tensor_max_expansion", 10**9)
    if replicated > limit and not allow:
        raise ValueError(
            f"{method} of shapes {operands[0]._shape} and {operands[1]._shape} would replicate "
            f"{replicated} entries to broadcast (the limit is {limit}). numpy semantics fill "
            "every broadcast position; use ewise_mult for the intersection, pass "
            "allow_expansion=True, or raise `graphblas.config` option tensor_max_expansion."
        )


def _squeezed(t, shape):
    """``t`` without its broadcast size-1 axes, and the output axis of each remaining axis."""
    off = len(shape) - t.ndim
    drop = [a for a, n in enumerate(t._shape) if n == 1 and shape[a + off] != 1]
    keep = [a + off for a in range(t.ndim) if a not in drop]
    return (t.squeeze(drop) if drop else t), keep


def _expand(t, shape):
    """``t`` replicated over its broadcast axes to ``shape``, as a Tensor (kron with ones)."""
    if t._shape == shape:
        return t
    ts, out_of = _squeezed(t, shape)
    bcast = tuple(o for o in range(len(shape)) if o not in out_of)
    rows = tuple(out_of[a] for a in ts._layout.rows)
    cols = tuple(out_of[a] for a in ts._layout.cols)
    # (more than one copy, or none: the callers give ``t`` the result's rank first, so a
    # broadcast axis of ``t`` has size 1 and the result's another size)
    reps = tl.prod(shape, bcast)
    if reps * ts._M._nvals > tl.MAX_GROUP:
        raise ValueError(
            f"broadcasting shape {t._shape} to {shape} replicates its {ts._M._nvals} entries "
            f"{reps} times, more than a Matrix can hold (2**60 entries)"
        )
    # The broadcast axes lead the rows, or, past 2^60 positions there, trail the columns
    if tl._valid_split(shape, bcast + rows + cols, len(bcast) + len(rows)):
        layout = Layout(bcast + rows, cols)
        # kron(ones, M)[(b, r), c] = M[r, c]
        expr = Matrix.from_scalar(True, reps, 1).kronecker(ts._M, binary.second)
    elif tl._valid_split(shape, rows + cols + bcast, len(rows)):
        layout = Layout(rows, cols + bcast)
        # kron(M, ones)[r, (c, b)] = M[r, c]
        expr = ts._M.kronecker(Matrix.from_scalar(True, 1, reps), binary.first)
    else:
        raise ValueError(
            f"broadcasting shape {t._shape} to {shape} replicates it {reps} times, more than "
            "one Matrix dimension can hold (2**60) next to its own axes"
        )
    M = tl.new_matrix(t.dtype, tl.prod(shape, layout.rows), tl.prod(shape, layout.cols))
    M << expr
    return Tensor._from_M(M, shape, layout)


def _lift_rank(t, ndim):
    """``t`` with leading size-1 axes added up to ``ndim`` axes (metadata only)."""
    if t.ndim == ndim:
        return t
    return t.expand_dims(tuple(range(ndim - t.ndim)))


def _one_term_monoid(dtype):
    # Diagonal scaling sums one term per output, so any monoid gives the same value; pick one
    # with factory kernels for the output type.
    if dtype == BOOL:
        return monoid.lor
    if dtype._is_udt:
        return monoid.any
    return monoid.plus


def _bcast_recipe(updater, method, a, b, op, untyped_op, extra, shape):
    # Give both operands the result's rank first. A lower-rank operand whose missing leading
    # axes are all size 1 repeats once (broadcast factor 1) yet has fewer axes than the
    # result, so "covers the result" must also mean "has its axes".
    a = _lift_rank(a, len(shape))
    b = _lift_rank(b, len(shape))
    if method == "ewise_mult":
        _bcast_mult(updater, a, b, op, untyped_op, shape)
    else:
        if census.on:
            census.note("bcast", "union: replicate")
        _ewise_recipe(updater, method, _expand(a, shape), _expand(b, shape), op, extra)


def _bcast_mult(updater, a, b, op, untyped_op, shape):
    fa = _broadcast_factor(a._shape, shape)
    fb = _broadcast_factor(b._shape, shape)
    if fa > 1 and fb > 1:
        # Both broadcast (an outer product, for example): replicate the cheaper one
        if fa * a._M._nvals <= fb * b._M._nvals:
            a = _expand(a, shape)
        else:
            b = _expand(b, shape)
        fa = _broadcast_factor(a._shape, shape)
    if fa == 1:
        big, small, big_left = a, b, True
    else:
        big, small, big_left = b, a, False
    nthreads = updater.opts.get("nthreads")
    ts, F = _squeezed(small, shape)
    if census.on and (ts._M._nvals == 0 or not F):
        census.note("bcast", "empty operand" if ts._M._nvals == 0 else "one value: apply")
    if ts._M._nvals == 0:
        L = big._layout
        empty = tl.new_matrix(op.return_type, tl.prod(shape, L.rows), tl.prod(shape, L.cols))
        _store(updater, empty, shape, L, fresh=True)
        return
    if not F:
        # The small side is one value
        value = ts._M[0, 0].new()
        if big_left:
            src = big._M.apply(op, right=value)
        else:
            src = big._M.apply(op, left=value)
        _store(updater, src, shape, big._layout)
        return
    flip = getattr(untyped_op, "commutes_to", None) if _flipped(op) is not None else None
    # M @ diag(v) multiplies (big, small); diag(v) @ M multiplies (small, big)
    col_op = untyped_op if big_left else flip
    row_op = flip if big_left else untyped_op
    if hasattr(untyped_op, "binaryop"):  # a Monoid
        col_op = row_op = untyped_op.binaryop
    Fset = set(F)
    M, L = big._M, big._layout
    nv = M._nvals
    ti = _ti(nthreads or tl._nthreads())
    seq, p = L.state
    # Options: (cost, split, group side, expand from the nonempty rows); split "gather" puts
    # the small side's values on the stored entries, None relayouts
    options = []
    if (
        nv > 0
        and not M.dtype._is_udt
        and not ts.dtype._is_udt
        and ts._M._nvals == math.prod(ts._shape)
        and tl.stored_format(M) in {"csr", "hypercsr"}
        and not M.ss.is_iso
    ):
        need_rows = any(o in L.rows for o in F)
        cost = (C_GATHER["rows" if need_rows else "cols"] + C_EWISE_SAME[ti]) * nv
        if tl.stored_format(M) == "hypercsr":
            cost += C_EWISE_HROW[ti] * tl.nonempty_rows(M)
        options.append((cost, "gather", None, False))

    def hyper_rows(state):
        # A reshape into hypersparse rows (more positions than SuiteSparse's 1/16 of nonempty
        # ones) makes the diagonal product pay a hash lookup per nonempty row, as the gather's
        # does: skew, hyper, Zipf, NIPS and Enron along a single last axis took 2-15x their
        # estimate without this term
        nrows = tl.prod(shape, state[0][: state[1]])
        R = tl.expected_rows(nrows, nv)
        return C_EWISE_HROW[ti] * R if nrows > 16 * R else 0.0

    # SuiteSparse's reshape refuses 2^60 positions or more in total: no split moves then (the
    # relayout option below replaces them by REBUILDs, see tensor_layout._neighbors)
    total_ok = math.prod(shape) < tl.MAX_GROUP
    for q in range(len(seq) + 1):
        if not tl._valid_split(shape, seq, q) or q != p and not total_ok:
            continue
        move = 0.0 if q == p else _rs_cost(shape, seq, q, nv, ti) + hyper_rows((seq, q))
        rows, cols = seq[:q], seq[q:]
        if col_op is not None and Fset == set(cols):
            options.append((move + C_DIAG[ti] * nv, q, "cols", False))
        if row_op is not None and Fset == set(rows):
            options.append((move + C_DIAG[ti] * nv, q, "rows", False))
        if q != p:
            continue
        if row_op is not None and Fset < set(rows):
            hyper = tl.stored_format(M) == "hypercsr"
            if hyper:
                cost = C_DIAG[ti] * nv + C_HROW[ti] * tl.nonempty_rows(M)
            else:
                cost = C_DIAG[ti] * nv + C_SEL_POS[ti] * tl.prod(shape, rows)
            options.append((cost, q, "rows", hyper))
        if col_op is not None and Fset < set(cols):
            cost = C_DIAG[ti] * nv + C_SEL_POS[ti] * tl.prod(shape, cols)
            options.append((cost, q, "cols", False))
    # Or relayout so the small side's axes are exactly one group
    rest = tuple(o for o in range(len(shape)) if o not in Fset)
    targets = set()
    orders = itertools.permutations(F) if len(shape) <= tl.MAX_PLANNED_NDIM else [tuple(F)]
    for f in orders:
        for r in itertools.permutations(rest) if len(shape) <= 5 else [rest]:
            if col_op is not None and tl._valid_split(shape, r + f, len(r)):
                targets.add((r + f, len(r)))
            if row_op is not None and tl._valid_split(shape, f + r, len(f)):
                targets.add((f + r, len(f)))
    # Those in this axis order are the split moves above (q), offered once: only the others
    targets = frozenset(t for t in targets if t[0] != seq)
    if targets:
        cost = tl.plan(shape, L.state, targets, nv, nthreads or tl._nthreads())[0]
        options.append((cost + C_DIAG[ti] * nv, None, None, False))
    if not options:
        # No diagonal side works (an operator with no flipped form, and past 2^60 positions in
        # total, where neither split moves nor relayouts that put the small side's axes in one
        # group fit): look the small side's values up at the big side's entries instead, which
        # costs by the entries (replicating it would build the broadcast shape's groups)
        if census.on:
            census.note("bcast", "no diagonal side: lookup at the entries")
        _store(updater, _mult_at_entries(M, shape, L, ts, F, op, big_left), shape, L)
        return
    i = min(range(len(options)), key=lambda k: options[k][0])
    if census.on:

        def label(o):
            if o[1] == "gather":
                return "gather"
            if o[1] is None:
                return "relayout, diagonal"
            inside = Fset != set(seq[: o[1]] if o[2] == "rows" else seq[o[1] :])
            name = f"diagonal {o[2]}"
            if o[3]:
                name += ", nonempty rows"
            elif inside:
                name += ", selector-expanded"
            return name + (", reshape" if o[1] != p else "")

        i = census.choose("bcast", [o[0] for o in options], [label(o) for o in options], i,
                          nvals=nv)  # fmt: skip
    _, q, side, from_rows = options[i]
    if q == "gather":
        values = ts.to_dense().ravel()
        W = tl.gather_operand(M, shape, L, tuple(F), values, ts.dtype)
        if W is not None:
            src = M.ewise_mult(W, op) if big_left else W.ewise_mult(M, op)
            _store(updater, src, shape, L)
            return
        q = None
    if q is None:
        M, L = tl.relayout(M, shape, L, targets, nthreads=nthreads)
        side = "cols" if set(L.cols) == Fset and col_op is not None else "rows"
    elif q != p:
        M = M.ss.reshape(tl.prod(shape, seq[:q]), tl.prod(shape, seq[q:]))
        L = Layout(seq[:q], seq[q:])
    group = L.cols if side == "cols" else L.rows
    # The small side as a Vector over `group` (expanded over the group's other axes)
    inv = {o: i for i, o in enumerate(F)}
    sub = tuple(o for o in group if o in Fset)
    vM, _ = tl.relayout(
        ts._M, ts._shape, ts._layout, Layout((), tuple(inv[o] for o in sub)), nthreads=nthreads
    )
    v = vM[0, :].new()
    if from_rows:
        v = _expand_on_rows(M, shape, group, tuple(group.index(o) for o in sub), v)
    elif len(sub) < len(group):
        St = tl.selector(
            tuple(shape[o] for o in group),
            tuple(i for i, o in enumerate(group) if o in Fset),
            transposed=True,
        )
        e = Vector(v.dtype, St._nrows)
        e << St.mxv(v, tl._perm_semiring(v))
        v = e
    D = v.diag()
    if side == "cols":
        sr = get_semiring(_one_term_monoid(op.return_type), col_op)
        src = M.mxm(D, sr)
    else:
        sr = get_semiring(_one_term_monoid(op.return_type), row_op)
        src = D.mxm(M, sr)
    _store(updater, src, shape, L)


def _mult_at_entries(M, shape, layout, ts, F, op, big_left):
    """``M`` (of ``shape``, stored in ``layout``) times ``ts`` (the small side, whose axes are
    the output axes ``F``) by ``op``, through the entries: the small side's values at each
    entry's coordinates along ``F`` (the gather's lookup), as a Matrix of ``M``'s pattern
    where there is one, then one ewise_mult in the operands' order.
    """
    from .tensor_index import lookup

    r, c, _ = M.to_coo(values=False, sort=False)
    r, c = r.view(np.int64), c.view(np.int64)
    coords = {}
    tl.unravel(r, shape, layout.rows, coords)
    tl.unravel(c, shape, layout.cols, coords)
    found, vals = lookup(ts, [coords[o] for o in F], r.size)
    W = tl.from_coo(r[found], c[found], vals[found], ts.dtype, M._nrows, M._ncols)
    return M.ewise_mult(W, op) if big_left else W.ewise_mult(M, op)


def _expand_on_rows(M, shape, rows, sub_pos, v):
    """``v`` (over the row axes at ``sub_pos``) expanded onto the nonempty rows of ``M``: a
    Vector over the row group with ``e[r] = v[coordinates of r]``, sized by the data.
    """
    h = tl.nonempty_row_ids(M)
    gshape = tuple(shape[a] for a in rows)
    co = {}
    tl.unravel(h, gshape, tuple(range(len(gshape))), co)
    vidx = tl.ravel(co, gshape, sub_pos, n=h.size)
    idx, vals = v.to_coo()
    dense = np.zeros(v._size, vals.dtype)
    present = np.zeros(v._size, bool)
    dense[idx] = vals
    present[idx] = True
    keep = present[vidx]
    return Vector.from_coo(h[keep], dense[vidx[keep]], v.dtype, size=tl.prod(shape, rows))


def _ewise_recipe(updater, method, a, b, op, extra):
    if method == "ewise_mult":
        est = min(a._M._nvals, b._M._nvals)
    else:
        est = a._M._nvals + b._M._nvals
    L = _choose_layout(updater, (a, b), est)
    nthreads = updater.opts.get("nthreads")
    A = _in_layout(a, L, nthreads)
    B = A if b is a else _in_layout(b, L, nthreads)
    if method == "ewise_add":
        src = A.ewise_add(B, op)
    elif method == "ewise_mult":
        src = A.ewise_mult(B, op)
    else:
        src = A.ewise_union(B, op, *extra)
    _store(updater, src, a._shape, L)


def _select_recipe(updater, a, op, thunk):
    L = _choose_layout(updater, (a,), a._M._nvals)
    A = _in_layout(a, L, updater.opts.get("nthreads"))
    _store(updater, A.select(op, thunk), a._shape, L)


def _apply_recipe(updater, a, op, right, left):
    L = _choose_layout(updater, (a,), a._M._nvals)
    A = _in_layout(a, L, updater.opts.get("nthreads"))
    _store(updater, A.apply(op, right=right, left=left), a._shape, L)


# isort: off
from .tensor import (  # noqa: E402
    C_DIAG,
    C_EWISE_HROW,
    C_EWISE_SAME,
    C_GATHER,
    C_HROW,
    C_SEL_POS,
    Tensor,
    TensorExpression,
    _as_tensor,
    _broadcast_shapes,
    _choose_layout,
    _flipped,
    _in_layout,
    _rs_cost,
    _store,
    _ti,
)

# isort: on

"""Contractions of Tensors: tensordot, matmul, batched products and TTV.

Private: part of ``graphblas.core.tensor``, which imports it at its end.
"""

import itertools
import operator

import numpy as np

from .. import semiring
from ..exceptions import DimensionMismatch
from . import tensor_census as census
from . import tensor_layout as tl
from .operator import get_semiring, get_typed_op
from .tensor_layout import Layout
from .vector import Vector


def _tensordot_axes(axes, ndim_a, ndim_b):
    if isinstance(axes, (tuple, list)):
        if len(axes) != 2:
            raise ValueError("axes must be an int or a pair of axis sequences")
        a_axes, b_axes = axes
        a_axes = (a_axes,) if not isinstance(a_axes, (tuple, list)) else tuple(a_axes)
        b_axes = (b_axes,) if not isinstance(b_axes, (tuple, list)) else tuple(b_axes)
        if len(a_axes) != len(b_axes):
            raise ValueError(f"axes {axes} must pair the same number of axes")
        a_axes = tuple(_normalize_axis(x, ndim_a) for x in a_axes)
        b_axes = tuple(_normalize_axis(x, ndim_b) for x in b_axes)
        if len(set(a_axes)) != len(a_axes) or len(set(b_axes)) != len(b_axes):
            raise ValueError(f"repeated axis in {axes}")
        return a_axes, b_axes
    n = operator.index(axes)
    if not 0 <= n <= min(ndim_a, ndim_b):
        raise ValueError(f"axes={n} must be in 0..{min(ndim_a, ndim_b)}")
    return tuple(range(ndim_a - n, ndim_a)), tuple(range(n))


def _tensordot_expr(self, other, axes, op):
    if (tensor := _as_tensor(other, "tensordot")) is None:
        self._expect_type(other, Tensor, within="tensordot", argname="other", op=op)
    other = tensor
    a_axes, b_axes = _tensordot_axes(axes, self.ndim, other.ndim)
    for x, y in zip(a_axes, b_axes, strict=True):
        if self._shape[x] != other._shape[y]:
            raise DimensionMismatch(
                f"axis {x} of size {self._shape[x]} cannot contract with axis {y} of size "
                f"{other._shape[y]}"
            )
    op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="tensordot", argname="op")
    shape = tuple(n for x, n in enumerate(self._shape) if x not in a_axes) + tuple(
        n for y, n in enumerate(other._shape) if y not in b_axes
    )
    return TensorExpression(
        "tensordot",
        _tensordot_recipe,
        [self, other],
        (self, other, op, a_axes, b_axes),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.tensordot({1.name}, axes=%s, op={op})"  # noqa: UP031
        % repr((a_axes, b_axes)).replace("{", "{{").replace("}", "}}"),
    )


def _orders(axes):
    # Any order of the free axes gives the same product (only the output layout differs),
    # so offer them all to the planner when there are few.
    if len(axes) <= 3:
        return list(itertools.permutations(axes))
    return [axes]


def _contract(updater, a, b, op, a_axes, b_axes, a_batch=(), b_batch=()):
    """``a`` and ``b`` contracted over ``a_axes`` with ``b_axes`` (paired), with the paired
    batch axes ``a_batch`` / ``b_batch`` kept once. Returns ``(src, layout, shape)``, where
    ``src`` is a Matrix or MatrixExpression and the output axes are the batch axes (in
    ``a_batch`` order), then a's free axes, then b's, each in axis order.

    Without batch axes: relayout so the contracted axes are exactly a's column group and b's
    row group, in the same order, then one mxm. With batch axes they lead both row groups and
    a's columns are embedded block-diagonally, so one mxm does every batch.
    Only a is embedded: the product keeps b's columns, so it is narrow and needs no
    unembedding, and it beats a loop over the batches at 1 thread too.
    """
    nthreads = _nthreads(updater)
    opt_nthreads = updater.opts.get("nthreads")
    free_a = tuple(x for x in range(a.ndim) if x not in a_axes and x not in a_batch)
    free_b = tuple(y for y in range(b.ndim) if y not in b_axes and y not in b_batch)
    k, nb = len(a_axes), len(a_batch)
    na, nbv = a._M._nvals, b._M._nvals
    shape = (
        tuple(a._shape[x] for x in a_batch)
        + tuple(a._shape[x] for x in free_a)
        + tuple(b._shape[y] for y in free_b)
    )
    if na == 0 or nbv == 0:
        # SuiteSparse (seen from 8.2 through 10.5.1): a product over an inner dimension of
        # size 0 returns every output position when both operands are iso and full (see
        # tensor_reduce._reduce_matrix). An operand with no entries gives no entries.
        layout = tl.default_layout(shape)
        out = tl.new_matrix(
            op.return_type, tl.prod(shape, layout.rows), tl.prod(shape, layout.cols)
        )
        return out, layout, shape
    best = None
    # The contracted (and batch) axes must come in the same order on both sides; try each
    # order and the free-axis orders, and take the cheapest pair of relayouts.
    sigmas = list(itertools.permutations(range(k))) if k <= 3 else [tuple(range(k))]
    batch_orders = list(itertools.permutations(range(nb))) if nb <= 3 else [tuple(range(nb))]
    for sigma in sigmas:
        ca = tuple(a_axes[i] for i in sigma)
        cb = tuple(b_axes[i] for i in sigma)
        for beta in batch_orders:
            ba = tuple(a_batch[i] for i in beta)
            bb = tuple(b_batch[i] for i in beta)
            ta = frozenset(
                (ba + fa + ca, nb + len(fa))
                for fa in _orders(free_a)
                if tl._valid_split(a._shape, ba + fa + ca, nb + len(fa))
            )
            tb = frozenset(
                (bb + cb + fb, nb + k)
                for fb in _orders(free_b)
                if tl._valid_split(b._shape, bb + cb + fb, nb + k)
            )
            if not ta or not tb:
                continue
            cost_a, _, sa = tl.plan(a._shape, a._layout.state, ta, na, nthreads)
            cost_b, _, sb = tl.plan(b._shape, b._layout.state, tb, nbv, nthreads)
            if best is None or cost_a + cost_b < best[0]:
                best = (cost_a + cost_b, sa, sb, beta)
    if best is None:
        raise ValueError("the contracted or free axes do not fit in a Matrix dimension (2**60)")
    _, sa, sb, beta = best
    A = _in_layout(a, Layout.from_state(sa), opt_nthreads)
    B = _in_layout(b, Layout.from_state(sb), opt_nthreads)
    out_batch = {x: a_batch.index(x) for x in a_batch}
    out_a = {x: nb + i for i, x in enumerate(free_a)}
    out_b = {y: nb + len(free_a) + i for i, y in enumerate(free_b)}
    rows = tuple(out_batch[x] for x in sa[0][:nb]) + tuple(
        out_a[x] for x in sa[0][nb : nb + len(free_a)]
    )
    layout = Layout(rows, tuple(out_b[y] for y in sb[0][nb + k :]))
    if census.on:
        census.note("contract", "batched, block-diagonal" if nb else "one mxm",
                    moved=[A is not a._M, B is not b._M], cost=best[0])  # fmt: skip
    if not nb:
        return A.mxm(B, op), layout, shape
    nblocks = tl.prod(a._shape, a_batch)
    rows_per_block = A._nrows // max(nblocks, 1)
    K = A._ncols
    if A is a._M or not tl.is_by_row(A):
        A = A.dup()
    if tl.stored_format(A) not in {"csr", "hypercsr"}:
        A.ss.config["sparsity_control"] = ["sparse", "hypersparse"]
    if not tl.embed_block_diagonal(A, rows_per_block, nblocks):
        # The new column indices need a wider integer type: rebuild instead
        if census.on:
            census.note("contract", "embedding by rebuild (wider indices)")
        r, c, v = A.to_coo()
        c = c + (r // rows_per_block) * K
        A = tl.from_coo(r, c, v, A.dtype, A._nrows, nblocks * K)
    # Then let SuiteSparse choose its format again: with few batches the embedded operand is
    # dense enough to go bitmap, and its bitmap saxpy is 5x faster at 8 threads than the
    # sparse saxpy's fine tasks (Higgs, 4 layers: x . A over the nodes, 88 -> 23 ms)
    A.ss.config["sparsity_control"] = ["hypersparse", "sparse", "bitmap", "full"]
    return A.mxm(B, op), layout, shape


def _matmul_expr(self, other, op):
    if (tensor := _as_tensor(other, "matmul")) is None:
        self._expect_type(other, Tensor, within="matmul", argname="other", op=op)
    other = tensor
    if self.ndim == 0 or other.ndim == 0:
        raise ValueError("matmul does not take 0-d operands; use ewise_mult or `*`")
    op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
    self._expect_op(op, "Semiring", within="matmul", argname="op")
    x = self.expand_dims(0) if self.ndim == 1 else self
    y = other.expand_dims(1) if other.ndim == 1 else other
    if x._shape[-1] != y._shape[-2]:
        raise DimensionMismatch(
            f"matmul: the last axis of {self._shape} does not match {other._shape}"
        )
    try:
        bshape = tuple(int(n) for n in np.broadcast_shapes(x._shape[:-2], y._shape[:-2]))
    except ValueError:
        raise DimensionMismatch(
            f"matmul: batch axes of {self._shape} and {other._shape} do not broadcast"
        ) from None
    full = (*bshape, x._shape[-2], y._shape[-1])
    drop = ((len(bshape),) if self.ndim == 1 else ()) + (
        (len(bshape) + 1,) if other.ndim == 1 else ()
    )
    shape = tuple(n for i, n in enumerate(full) if i not in drop)
    return TensorExpression(
        "matmul",
        _matmul_recipe,
        [self, other],
        (x, y, op, bshape, drop),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.matmul({1.name}, op={op})",
    )


def _matmul_recipe(updater, x, y, op, bshape, drop):
    nbatch = len(bshape)
    # Size-1 batch axes broadcast; drop them and remember each kept axis's output axis
    xs, x_out = _squeezed_batch(x, bshape)
    ys, y_out = _squeezed_batch(y, bshape)
    xb = {o: a for a, o in enumerate(x_out) if o < nbatch}
    yb = {o: a for a, o in enumerate(y_out) if o < nbatch}
    both = [o for o in range(nbatch) if o in xb and o in yb]
    a_batch = tuple(xb[o] for o in both)
    b_batch = tuple(yb[o] for o in both)
    kx, ky = xs.ndim - 1, ys.ndim - 2
    src, layout, _ = _contract(updater, xs, ys, op, (kx,), (ky,), a_batch, b_batch)
    # _contract numbers its output: batch (in `both` order), xs free axes, ys free axes
    free_x = [a for a in range(xs.ndim) if a != kx and a not in a_batch]
    free_y = [a for a in range(ys.ndim) if a != ky and a not in b_batch]
    to_out = list(both) + [x_out[a] for a in free_x] + [y_out[a] for a in free_y]
    layout = _renumber(layout, dict(enumerate(to_out)))
    full = (*bshape, x._shape[-2], y._shape[-1])
    present = set(to_out)
    layout = _insert_unit_axes(layout, [o for o in range(len(full)) if o not in present])
    if drop:
        keep = {o: i for i, o in enumerate(o for o in range(len(full)) if o not in drop)}
        layout = Layout(
            tuple(keep[o] for o in layout.rows if o in keep),
            tuple(keep[o] for o in layout.cols if o in keep),
        )
        full = tuple(n for i, n in enumerate(full) if i not in drop)
    _store(updater, src, full, layout)


def _squeezed_batch(t, bshape):
    """``t`` without its size-1 batch axes; the output axis of each remaining axis (the last
    two map to ``len(bshape)`` and ``len(bshape) + 1``).
    """
    nbatch = len(bshape)
    off = nbatch - (t.ndim - 2)
    drop = [a for a in range(t.ndim - 2) if t._shape[a] == 1]
    out = [a + off for a in range(t.ndim) if a not in drop]
    return (t.squeeze(drop) if drop else t), out


def _ttv(updater, x, w, axis, op, x_left):
    """TTV: ``x`` contracted with the 1-d ``w`` over ``axis``, by the cheapest of: one mxv
    (``axis`` is the column group), one vxm (the row group), a weighted selector (``axis`` is
    inside a group), each possibly after moving the split; or the general relayout and
    mxm (returns False for that).
    """
    mult = op.binaryop
    flip = getattr(mult, "commutes_to", None)
    # (x, w) is the order the multiply must see: mxv and a column selector give (x, w), vxm
    # and a row selector give (w, x)
    xw_op = mult if x_left else flip
    wx_op = flip if x_left else mult
    M, layout = x._M, x._layout
    seq, p = layout.state
    n = len(seq)
    nv = M._nvals
    if nv == 0 or w._M._nvals == 0:
        # SuiteSparse (seen from 8.2 through 10.5.1): mxv and vxm over a dimension of size 0
        # return every output position when the operands are iso and full (see
        # tensor_reduce._reduce_matrix). An operand with no entries gives no entries.
        shape = tuple(m for a, m in enumerate(x._shape) if a != axis)
        renum = {a: i for i, a in enumerate(a for a in range(n) if a != axis)}
        out = Layout(*(tuple(renum[a] for a in g if a != axis) for g in layout))
        src = tl.new_matrix(op.return_type, tl.prod(shape, out.rows), tl.prod(shape, out.cols))
        _store(updater, src, shape, out, fresh=True)
        return True
    nthreads = _nthreads(updater)
    ti = _ti(nthreads)
    options = []  # (cost, kind, split)
    if layout.cols == (axis,) and xw_op is not None:
        # One pass with no data movement; the planner's relayout estimates run low, so do
        # not let them compete (relayout + mxm 56 ms against vxm 36 ms, measured)
        options.append((0.0, "mxv", p))
    elif layout.rows == (axis,) and wx_op is not None and M._ncols <= 4 * max(nv, 1):
        # vxm writes into the whole column group: fine up to a few times the data (at 2.6x,
        # 36 ms vs 56), slow into a hypersparse one (hyper 2^38: 87 ms vs 23)
        options.append((0.0, "vxm", p))
    for q in range(n + 1) if not options else ():
        if not tl._valid_split(x._shape, seq, q):
            continue
        move = 0.0 if q == p else _rs_cost(x._shape, seq, q, nv, ti)
        rows, cols = seq[:q], seq[q:]
        if cols == (axis,) and xw_op is not None:
            options.append((move + C_MXV[ti] * nv, "mxv", q))
        elif rows == (axis,) and wx_op is not None and tl.prod(x._shape, cols) <= 4 * nv:
            options.append((move + C_VXM[ti] * nv, "vxm", q))
        elif q == p and axis in rows and len(rows) > 1 and wx_op is not None:
            options.append((_row_selector_cost(M, tl.prod(x._shape, rows), ti), "rowsel", q))
        elif q == p and axis in cols and len(cols) > 1 and xw_op is not None:
            cost = _shape_selector_cost(nv, tl.prod(x._shape, cols), ti) + C_DIAG[ti] * nv
            options.append((cost, "colsel", q))
    others = tuple(a for a in range(n) if a != axis)
    targets = frozenset(
        ((*o, axis), n - 1)
        for o in (itertools.permutations(others) if n <= tl.MAX_PLANNED_NDIM else [others])
        if tl._valid_split(x._shape, (*o, axis), n - 1)
    )
    if targets and (not options or options[0][0] > 0):
        general = tl.plan(x._shape, layout.state, targets, nv, nthreads)[0] + C_MXV[ti] * nv
        options.append((general, "general", None))
    if not options:
        if census.on:
            census.note("ttv", "no option: general")
        return False
    i = min(range(len(options)), key=lambda k: options[k][0])
    if census.on:
        labels = [o[1] + (", reshape" if o[2] not in {None, p} else "") for o in options]
        i = census.choose("ttv", [o[0] for o in options], labels, i, nvals=nv)
    _, kind, q = options[i]
    if kind == "general":
        return False
    if q != p:
        M = M.ss.reshape(tl.prod(x._shape, seq[:q]), tl.prod(x._shape, seq[q:]))
    rows, cols = seq[:q], seq[q:]
    shape = tuple(m for a, m in enumerate(x._shape) if a != axis)
    renum = {a: i for i, a in enumerate(a for a in range(n) if a != axis)}
    opt_nthreads = updater.opts.get("nthreads")
    wv = _in_layout(w, Layout((), (0,)), opt_nthreads)[0, :].new()
    if kind in {"mxv", "vxm"}:
        v = Vector(op.return_type, M._nrows if kind == "mxv" else M._ncols)
        if kind == "mxv":
            sr = op if xw_op is mult else get_semiring(op.monoid, xw_op)
            v(**updater.opts) << M.mxv(wv, sr)
            out = Layout((), tuple(renum[a] for a in rows))
        else:
            sr = op if wx_op is mult else get_semiring(op.monoid, wx_op)
            v(**updater.opts) << wv.vxm(M, sr)
            out = Layout((), tuple(renum[a] for a in cols))
        _store(updater, tl.steal_vector_as_row(v), shape, out, fresh=True)
        return True
    group = rows if kind == "rowsel" else cols
    gshape = tuple(x._shape[g] for g in group)
    pos = group.index(axis)
    kept_pos = tuple(i for i in range(len(group)) if i != pos)
    if kind == "rowsel" and tl.stored_format(M) == "hypercsr":
        # A weighted selector over the nonempty rows only: S[kept, r] = w[axis coordinate]
        idx, vals = wv.to_coo()
        dense = np.zeros(x._shape[axis], vals.dtype)
        present = np.zeros(x._shape[axis], bool)
        dense[idx] = vals
        present[idx] = True
        Sw = tl.rows_selector(M, x._shape, rows, kept_pos, weights=(dense, present, pos))
    else:
        # S[kept, g] = w[axis coordinate of g] over the positions g of the group
        e = Vector(wv.dtype, tl.prod(x._shape, group))
        St = tl.selector(gshape, (pos,), transposed=True)
        e << St.mxv(wv, tl._perm_semiring(wv))
        if kind == "rowsel":
            S = tl.selector(gshape, kept_pos)
            Sw = tl.new_matrix(e.dtype, S._nrows, S._ncols)
            Sw << S.mxm(e.diag(), semiring.any_second)
        else:
            S = tl.selector(gshape, kept_pos, transposed=True)
            Sw = tl.new_matrix(e.dtype, S._nrows, S._ncols)
            Sw << e.diag().mxm(S, semiring.any_first)
    if kind == "rowsel":
        src = Sw.mxm(M, get_semiring(op.monoid, wx_op))
        out = Layout(tuple(renum[a] for a in rows if a != axis), tuple(renum[a] for a in cols))
    else:
        src = M.mxm(Sw, get_semiring(op.monoid, xw_op))
        out = Layout(tuple(renum[a] for a in rows), tuple(renum[a] for a in cols if a != axis))
    _store(updater, src, shape, out)
    return True


def _tensordot_recipe(updater, a, b, op, a_axes, b_axes):
    if len(a_axes) == 1 and (b.ndim == 1 or a.ndim == 1) and a.ndim + b.ndim > 2:
        x_left = b.ndim == 1
        x, w, axis = (a, b, a_axes[0]) if x_left else (b, a, b_axes[0])
        if _ttv(updater, x, w, axis, op, x_left):
            return
    if (rv := _ttm_engines(updater, a, b, op, a_axes, b_axes)) is not None:
        src, layout, shape = rv
        _store(updater, src, shape, layout, fresh=True)
        return
    src, layout, shape = _contract(updater, a, b, op, a_axes, b_axes)
    if not layout.cols and layout.rows and _fixed_layout(updater) is None:
        M, layout = _healthy(_materialize(src), shape, layout)
        _store(updater, M, shape, layout, fresh=True)
    else:
        _store(updater, src, shape, layout)


def _batched_contract_expr(a, b, op, a_axes, b_axes, a_batch, b_batch):
    free_a = [x for x in range(a.ndim) if x not in a_axes and x not in a_batch]
    free_b = [y for y in range(b.ndim) if y not in b_axes and y not in b_batch]
    shape = (
        tuple(a._shape[x] for x in a_batch)
        + tuple(a._shape[x] for x in free_a)
        + tuple(b._shape[y] for y in free_b)
    )
    return TensorExpression(
        "contract",
        _batched_contract_recipe,
        [a, b],
        (a, b, op, a_axes, b_axes, a_batch, b_batch),
        dtype=op.return_type,
        shape=shape,
        op=op,
        expr_repr="{0.name}.contract({1.name}, op={op})",
    )


def _batched_contract_recipe(updater, a, b, op, a_axes, b_axes, a_batch, b_batch):
    src, layout, shape = _contract(updater, a, b, op, a_axes, b_axes, a_batch, b_batch)
    _store(updater, src, shape, layout)


# isort: off
from .tensor import (  # noqa: E402
    C_DIAG,
    C_MXV,
    C_VXM,
    Tensor,
    TensorExpression,
    _as_tensor,
    _fixed_layout,
    _healthy,
    _in_layout,
    _insert_unit_axes,
    _materialize,
    _normalize_axis,
    _nthreads,
    _renumber,
    _row_selector_cost,
    _rs_cost,
    _shape_selector_cost,
    _store,
    _ti,
)
from .tensor_engines import _ttm_engines  # noqa: E402

# isort: on

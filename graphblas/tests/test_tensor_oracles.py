"""Randomized oracles for the experimental Tensor (slow tests: run them with ``--runslow``).

Each oracle draws random cases (shapes of 1-5 dims with size-1 axes, random layouts and
permuted views, iso values and explicit zeros, masks, accumulators) and compares every result
with an independent numpy model of its values and of which positions hold a value. The models
share no code with ``test_tensor.py`` or with the implementation. The Tensor is used through
its public API, except for the switches that force an engine (``_TTM_ENGINES``,
``_MTTKRP_ENGINES``, ``tensor_index._GATHER_ENGINES``, ``_has_numba``, the kernels' size
thresholds and the Numba kernels' ``_FORCE_RUNS`` / ``_FORCE_COLBLOCK``) and the census
(``tensor_census``), which says which engine ran, so a pass that never reached an engine
cannot hide.

Every oracle has teeth: ``test_oracle_teeth`` plants a known error in its model and requires
failures, to show that the comparison would catch a real one.

Environment variables:

- ``GRAPHBLAS_TENSOR_ORACLE_SCALE``: multiplies every case count (default 1, sized for CI;
  use 20 or more for a thorough run).
- ``GRAPHBLAS_TENSOR_ORACLE_SEED``: the seed. By default it is the session's
  ``GRAPHBLAS_TEST_SEED``, which ``conftest.py`` prints, so each run draws new cases and a
  failure names the seed that reruns it.
"""

import contextlib
import itertools
import os
import zlib
from collections import Counter

import numpy as np
import pytest

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, backend, binary, monoid, semiring, unary
from graphblas.core import ffi, lib, tensor, tensor_census, tensor_index

if backend != "suitesparse":
    pytest.skip("Tensor needs the suitesparse backend", allow_module_level=True)
try:
    gb.Tensor(float, (1,))
except NotImplementedError:
    pytest.skip("Tensor needs SuiteSparse:GraphBLAS 10 or later", allow_module_level=True)


Tensor = gb.Tensor
pytestmark = pytest.mark.slow
SCALE = float(os.environ.get("GRAPHBLAS_TENSOR_ORACLE_SCALE", "1"))


def cases(n):
    return max(1, round(n * SCALE))


def base_seed():
    for var in ("GRAPHBLAS_TENSOR_ORACLE_SEED", "GRAPHBLAS_TEST_SEED"):
        if seed := os.environ.get(var):
            return int(seed)
    return 0


@pytest.fixture
def rng(request):
    """A stream of its own for each test, from the seed and the test's name."""
    return np.random.default_rng([base_seed(), zlib.crc32(request.node.name.encode())])


@pytest.fixture(autouse=True)
def _restore_nthreads():
    prev = gb.ss.config["nthreads"]
    yield
    gb.ss.config["nthreads"] = prev


@contextlib.contextmanager
def census():
    """The census's records while the block runs; what it held before is put back."""
    was_on, saved = tensor_census.on, list(tensor_census.records)
    tensor_census.start()
    try:
        yield tensor_census.records
    finally:
        tensor_census.on = was_on
        tensor_census.records[:] = saved


def choices(records, start, point):
    return [r["choice"] for r in records[start:] if r["point"] == point]


# ---------------------------------------------------------------------------------------------
# Models: dense values plus a boolean array of the positions that hold a value


def rand_model(rng, shape, density=0.4, *, kind=None):
    present = rng.random(shape) < density
    kind = kind or rng.choice(["int", "int", "iso", "normal"])
    if kind == "iso":
        values = np.full(shape, 2.5)
    elif kind == "normal":
        values = rng.standard_normal(shape)
    else:  # small integers: explicit zeros and ties
        values = rng.integers(-5, 6, shape).astype(float)
    return np.where(present, values, 0.0), present


def random_layout(rng, ndim):
    perm = [int(a) for a in rng.permutation(ndim)]
    k = int(rng.integers(0, ndim + 1))  # 0 or ndim: an empty group
    return tuple(perm[:k]), tuple(perm[k:])


def build(rng, values, present, *, view=True, dtype=None):
    """A Tensor holding the model, in a random layout, or as a permuted view of a Tensor
    built in another axis order.
    """
    ndim = values.ndim
    if view and ndim > 1 and rng.random() < 0.4:
        perm = rng.permutation(ndim)
        inv = np.argsort(perm)
        T = build(rng, values.transpose(inv), present.transpose(inv), view=False, dtype=dtype)
        return T.permute_dims([int(a) for a in perm])
    coords = np.nonzero(present)
    return Tensor.from_coo(
        coords, values[coords], dtype or values.dtype, shape=values.shape,
        layout=random_layout(rng, ndim),
    )  # fmt: skip


def model_of(obj, shape=None):
    """The model of a Tensor, Matrix, Vector or Scalar."""
    if isinstance(obj, Scalar):
        values, present = np.zeros(()), np.zeros((), bool)
        if obj.nvals:
            values[()], present[()] = obj.value, True
        return values, present
    if isinstance(obj, Matrix):
        r, c, v = obj.to_coo()
        values, present = np.zeros(obj.shape, v.dtype), np.zeros(obj.shape, bool)
        values[r, c], present[r, c] = v, True
        return values, present
    if isinstance(obj, Vector):
        i, v = obj.to_coo()
        values, present = np.zeros(obj.shape, v.dtype), np.zeros(obj.shape, bool)
        values[i], present[i] = v, True
        return values, present
    coords, v = obj.to_coo()
    values, present = np.zeros(obj.shape, v.dtype), np.zeros(obj.shape, bool)
    if obj.ndim:
        values[coords], present[coords] = v, True
    elif v.size:
        values[()], present[()] = v[0], True
    return values, present


def matches(obj, values, present, *, rtol=1e-10):
    present = np.asarray(present)
    if tuple(getattr(obj, "shape", ())) != present.shape:
        return False
    got_values, got_present = model_of(obj)
    if not (got_present == present).all():
        return False
    want = np.broadcast_to(values, present.shape)[present].astype(float)
    return bool(np.allclose(got_values[present].astype(float), want, rtol=rtol, atol=1e-10))


def valid(T):
    """Whether SuiteSparse finds the stored Matrix well formed (kernels hand it arrays
    unchecked).
    """
    M = T.ss.matrix
    return lib.GxB_Matrix_fprint(M._carg, b"M", lib.GxB_SILENT, ffi.NULL) == lib.GrB_SUCCESS


def update_model(cv, cp, zv, zp, *, mask=None, accum=None, replace=False):
    """``C(mask, accum, replace) << Z`` in GraphBLAS, on models."""
    if accum is None:
        tv, tp = zv, zp
    else:
        tv = np.where(cp & zp, accum(cv, zv), np.where(cp, cv, zv))
        tp = cp | zp
    if mask is None:
        return tv, tp
    return np.where(mask, tv, cv), np.where(mask, tp, cp & (not replace))


def random_mask(rng, shape, make=build):
    mv, mp = rand_model(rng, shape, 0.5)
    mv = np.where(mp, rng.choice([0.0, 1.0], shape), 0.0)
    M = make(rng, mv, mp)
    kind = str(rng.choice(["S", "V", "~S", "~V"]))
    base = mp if "S" in kind else mp & (mv != 0)
    arg = {"S": M.S, "V": M.V, "~S": ~M.S, "~V": ~M.V}[kind]
    return arg, ~base if kind.startswith("~") else base, kind


class Oracle:
    """Failures and counts of one oracle run."""

    def __init__(self, name):
        self.name = name
        self.fails = []
        self.stats = Counter()

    def fail(self, what, why):
        self.fails.append(f"{what}: {why}")

    def run(self, what, thunk):
        try:
            rv = thunk()
            if type(rv) not in {Tensor, Matrix, Vector, Scalar} and hasattr(rv, "new"):
                rv = rv.new()  # an expression
        except Exception as exc:
            self.fail(what, f"raised {type(exc).__name__}: {str(exc)[:200]}")
            return None
        return rv

    def expect(self, what, thunk, values, present, *, rtol=1e-10, count=None):
        self.stats[count or what.split(" ", 1)[0]] += 1
        rv = self.run(what, thunk)
        if rv is not None and not matches(rv, values, present, rtol=rtol):
            self.fail(what, "values or presence differ from the model")
        return rv

    def verdict(self, *, teeth=False):
        checks = dict(self.stats)
        if teeth:
            assert self.fails, f"{self.name}: the planted error was not caught ({checks})"
            return
        assert sum(self.stats.values()), f"{self.name}: nothing was checked"
        assert not self.fails, (
            f"{self.name}: {len(self.fails)} failures ({checks}); rerun with "
            f"GRAPHBLAS_TENSOR_ORACLE_SEED={base_seed()}\n" + "\n".join(self.fails[:12])
        )


def outer_index(values, key):
    """Numpy outer indexing (each index applies to its own axis; an int drops it)."""
    if any(k is Ellipsis for k in key):
        i = key.index(Ellipsis)
        key = key[:i] + (slice(None),) * (values.ndim - len(key) + 1) + key[i + 1 :]
    key = key + (slice(None),) * (values.ndim - len(key))
    axis = 0
    for k in key:
        if isinstance(k, (int, np.integer)):
            values = np.take(values, k, axis=axis)
        elif isinstance(k, slice):
            values = np.moveaxis(np.moveaxis(values, axis, 0)[k], 0, axis)
            axis += 1
        else:
            values = np.take(values, np.asarray(k), axis=axis)
            axis += 1
    return values


def random_key(rng, n, *, ints=True):
    """An index for an axis of size n: an int (negatives too), a slice or a list (no
    duplicates, since GraphBLAS leaves assignment to repeated indices undefined).
    """
    r = rng.random()
    if ints and r < 0.2:
        return int(rng.integers(-n, n))
    if r < 0.45:
        a, b = sorted(int(x) for x in rng.integers(0, n + 1, 2))
        return slice(a, b, int(rng.choice([1, 1, 2])))
    if r < 0.55:
        return slice(None, None, -int(rng.choice([1, 2]))) if rng.random() < 0.5 else slice(None)
    picks = rng.choice(n, int(rng.integers(1, n + 1)), replace=False)
    picks = [int(x) - (n if rng.random() < 0.2 else 0) for x in picks]
    return picks if rng.random() < 0.5 else np.array(picks)


# ---------------------------------------------------------------------------------------------
# Element-wise operations, reductions, tensordot, and updates with masks and accumulators


def oracle_ewise_reduce(rng, n, teeth=False):
    o = Oracle("ewise_reduce")
    for _ in range(n):
        ndim = int(rng.integers(1, 5))
        shape = tuple(int(rng.integers(1, 5)) for _ in range(ndim))
        av, ap = rand_model(rng, shape)
        bv, bp = rand_model(rng, shape)
        A, B = build(rng, av, ap), build(rng, bv, bp)
        desc = f"{shape} {A.layout} {B.layout}"
        cv, cp = np.where(ap & bp, av + bv, np.where(ap, av, bv)), ap | bp
        C = o.expect(f"ewise_add {desc}", lambda: A.ewise_add(B, binary.plus).new(), cv, cp)
        # Not commutative: the operand order must be kept (the teeth swap it)
        dv = av - bv if teeth else bv - av
        o.expect(f"ewise_mult {desc}", lambda: B.ewise_mult(A, binary.minus).new(), dv, ap & bp)
        uv = np.where(bp, bv, 10.0) - np.where(ap, av, -10.0)
        o.expect(
            f"ewise_union {desc}", lambda: B.ewise_union(A, binary.minus, 10.0, -10.0).new(),
            uv, ap | bp,
        )  # fmt: skip
        if C is not None:
            for k in range(1, ndim + 1):
                for axes in itertools.combinations(range(ndim), k):
                    op, npop, identity = [
                        (monoid.max, np.max, -np.inf), (monoid.plus, np.sum, 0.0),
                        (monoid.min, np.min, np.inf),
                    ][int(rng.integers(3))]  # fmt: skip
                    want = npop(np.where(cp, cv, identity), axis=axes)
                    o.expect(
                        f"reduce {op.name} {axes} {desc}",
                        lambda: C.reduce(op, axis=axes).new(), want, cp.any(axis=axes),
                    )  # fmt: skip
        # tensordot over one axis with a 2-d operand
        axis = int(rng.integers(ndim))
        yv, yp = rand_model(rng, (shape[axis], 3), 0.5)
        Y = build(rng, yv, yp)
        want = np.tensordot(av, yv, axes=([axis], [0]))
        wantp = np.tensordot(ap.astype(int), yp.astype(int), axes=([axis], [0])) > 0
        o.expect(
            f"tensordot axis {axis} {desc}", lambda: A.tensordot(Y, axes=([axis], [0])).new(),
            want, wantp,
        )  # fmt: skip
        # C(mask, accum, replace) << A * 10, into a target in its own layout
        tv, tp = rand_model(rng, shape)
        target = build(rng, tv, tp)
        kw, mask = {}, None
        if rng.random() < 0.7:
            mask_arg, mask, mkind = random_mask(rng, shape)
            if rng.random() < 0.4:
                kw["replace"] = True
        accum = None
        if rng.random() < 0.5:
            kw["accum"], accum = binary.max, np.maximum
        wv, wp = update_model(
            tv, tp, av * 10, ap, mask=mask, accum=accum, replace=kw.get("replace", False)
        )
        what = f"update mask={mask is not None and mkind} {sorted(kw)} {desc} {target.layout}"
        if mask is None:
            o.run(what, lambda: target(**kw) << A * 10)
        else:
            o.run(what, lambda: target(mask_arg, **kw) << A * 10)
        o.stats["update"] += 1
        if not matches(target, wv, wp):
            o.fail(what, "values or presence differ from the model")
    return o


# ---------------------------------------------------------------------------------------------
# Broadcasting, Matrix and Vector operands, matmul, einsum, outer indexing


MATMUL_SHAPES = [
    ((2, 3, 4), (4, 5)), ((3, 4), (2, 4, 5)), ((2, 1, 3, 4), (5, 4, 2)), ((4,), (2, 4, 3)),
    ((2, 3, 4), (4,)), ((1, 3, 4), (6, 4, 2)), ((3, 4), (4, 5)), ((4,), (4,)),
]  # fmt: skip
EINSUMS = [
    "ijk,jr,kr->ir", "ijk,jr,kr->jr", "ijk,kl->ijl", "ijk,jl->ilk", "bij,bjk->bik",
    "ijk,ijk->ij", "ijk->kji", "ijk->j", "ij,jk,kl->il", "ijk,lk->ijl", "i,j->ij", "ijk,k->ij",
    "ijkl,jr,kr,lr->ir", "ijk,ijk->",
]  # fmt: skip


def oracle_broadcast_contract(rng, n, teeth=False):
    o = Oracle("broadcast_contract")
    dims = {"i": 3, "j": 4, "k": 5, "l": 2, "r": 2, "b": 2}
    for case in range(n):
        # Broadcasting against an operand of lower rank, with size-1 axes
        ndim = int(rng.integers(1, 5))
        shape = tuple(int(rng.integers(1, 5)) for _ in range(ndim))
        av, ap = rand_model(rng, shape)
        A = build(rng, av, ap)
        k = int(rng.integers(1, ndim + 1))
        bshape = tuple(1 if rng.random() < 0.3 else d for d in shape[ndim - k :])
        bv, bp = rand_model(rng, bshape)
        B = build(rng, bv, bp)
        bvb, bpb = np.broadcast_to(bv, shape), np.broadcast_to(bp, shape)
        desc = f"{shape} {bshape}"
        o.expect(f"bcast mult {desc}", lambda: A * B, av * bvb, ap & bpb)
        o.expect(f"bcast mult reflected {desc}", lambda: B * A, bvb * av, ap & bpb)
        minus = np.where(bpb, bvb, 0) - av if teeth else av - np.where(bpb, bvb, 0)
        o.expect(f"bcast minus {desc}", lambda: A - B, minus, ap | bpb)
        o.expect(
            f"bcast first {desc}", lambda: A.ewise_add(B, binary.first).new(),
            np.where(ap, av, bvb), ap | bpb,
        )  # fmt: skip
        # Matrix and Vector operands follow Tensor rules (the Tensor's reflected operator)
        wv, wp = rand_model(rng, shape[-1:])
        w = Vector.from_coo(np.nonzero(wp)[0], wv[wp], size=shape[-1])
        o.expect(f"Vector * Tensor {desc}", lambda: w * A, wv * av, ap & wp)
        if ndim >= 2:
            mv, mp = rand_model(rng, shape[-2:])
            r, c = np.nonzero(mp)
            M = Matrix.from_coo(r, c, mv[mp], nrows=shape[-2], ncols=shape[-1])
            o.expect(
                f"Matrix + Tensor {desc}", lambda: M + A,
                np.where(mp, mv, 0) + np.where(ap, av, 0), ap | mp,
            )  # fmt: skip
        # matmul with numpy's rules: 1-d operands, broadcast batch axes
        ash, bsh = MATMUL_SHAPES[case % len(MATMUL_SHAPES)]
        xv, xp = rand_model(rng, ash)
        yv, yp = rand_model(rng, bsh)
        X, Y = build(rng, xv, xp), build(rng, yv, yp)
        want, wantp = np.matmul(xv, yv), np.matmul(xp.astype(int), yp.astype(int)) > 0
        o.expect(f"matmul {ash} @ {bsh}", lambda: X @ Y, want, wantp)
        if len(ash) >= 2 and len(bsh) >= 2:
            a2, b2 = np.where(xp, xv, np.inf), np.where(yp, yv, np.inf)
            mp_ = np.min(a2[..., :, :, None] + b2[..., None, :, :], axis=-2)
            o.expect(
                f"min_plus {ash} @ {bsh}", lambda: semiring.min_plus(X @ Y).new(), mp_,
                np.isfinite(mp_),
            )  # fmt: skip
        # einsum with plus_times: present where some product of present entries is
        spec = EINSUMS[case % len(EINSUMS)]
        ins = spec.split("->")[0].split(",")
        models = [rand_model(rng, tuple(dims[c] for c in s)) for s in ins]
        operands = [build(rng, v, p) for v, p in models]
        want = np.einsum(spec, *[v for v, _ in models])
        wantp = np.einsum(spec, *[p.astype(int) for _, p in models]) > 0
        o.expect(f"einsum {spec}", lambda: gb.einsum(spec, *operands).new(), want, wantp)
        # mttkrp with dense ndarray factors: a fiber with an entry gives a full row
        xv, xp = rand_model(rng, (4, 5, 6))
        X = build(rng, xv, xp)
        F = [rng.random((d, 3)) for d in (4, 5, 6)]
        mode = int(rng.integers(3))
        others = [m for m in range(3) if m != mode]
        sub = f"ijk,{'ijk'[others[0]]}r,{'ijk'[others[1]]}r->{'ijk'[mode]}r"
        want = np.einsum(sub, xv, F[others[0]], F[others[1]])
        wantp = np.broadcast_to(xp.any(axis=tuple(others))[:, None], want.shape)
        o.expect(f"mttkrp mode {mode}", lambda: X.mttkrp(F, mode).new(), want, wantp)
        # Outer indexing
        xv, xp = rand_model(rng, (5, 6, 7))
        X = build(rng, xv, xp)
        key = tuple(random_key(rng, d) for d in (5, 6, 7)[: int(rng.integers(1, 4))])
        if rng.random() < 0.2 and len(key) < 3:
            key = (Ellipsis, *key)
        o.expect(f"extract {key}", lambda: X[key], outer_index(xv, key), outer_index(xp, key))
    # A union broadcast that would replicate past its limit refuses (the guard)
    big = Tensor.from_coo(([0], [0], [1]), [1.0], shape=(2**20, 2**20, 4))
    v = Vector.from_coo([0, 1, 2, 3], 1.0, size=4)
    o.stats["guard"] += 1
    try:
        (big + v).new()
        o.fail("union broadcast guard", "did not refuse")
    except ValueError as exc:
        if "allow_expansion" not in str(exc):
            o.fail("union broadcast guard", f"refused with {exc}")
    return o


# ---------------------------------------------------------------------------------------------
# MTTKRP: every mode, layout and view, R in {1, 2, 5, 16}, the kernel's variants forced


def random_tensor_coords(rng, ndim_range, *, sparse):
    ndim = int(rng.integers(*ndim_range))
    if sparse:  # larger and sparser: (hyper)csr storage, where the kernels run
        shape = tuple(int(rng.integers(1, 14 if ndim < 5 else 9)) for _ in range(ndim))
        density = float(rng.choice([0.01, 0.03, 0.1]))
    else:
        shape = tuple(int(rng.integers(1, 8)) for _ in range(ndim))
        density = float(rng.choice([0.05, 0.3, 0.8, 1.0]))
    size = int(np.prod(shape))
    nnz = int(rng.binomial(size, density)) if rng.random() > 0.05 else 0
    flat = np.sort(rng.choice(size, nnz, replace=False))
    coords = list(np.unravel_index(flat, shape))
    kind = rng.choice(["normal", "iso", "zeros"])
    if kind == "iso":
        vals = np.full(nnz, 1.5)
    elif kind == "zeros":
        vals = rng.choice([0.0, 1.0, -2.0], nnz)
    else:
        vals = rng.standard_normal(nnz)
    return shape, coords, vals, kind


def random_view(rng, X, shape, coords, *, squeeze=True, maxdim=5):
    """X, or a permuted, expanded or squeezed view of it, with its shape and coordinates."""
    ndim = len(shape)
    view = rng.choice(["none", "none", "permute", "expand", "squeeze"])
    if view == "permute":
        p = [int(a) for a in rng.permutation(ndim)]
        return X.permute_dims(p), tuple(shape[a] for a in p), [coords[a] for a in p], view
    if view == "expand" and ndim < maxdim:
        ax = int(rng.integers(0, ndim + 1))
        nnz = len(coords[0]) if coords else 0
        coords = [*coords[:ax], np.zeros(nnz, np.int64), *coords[ax:]]
        return X.expand_dims(ax), (*shape[:ax], 1, *shape[ax:]), coords, view
    if view == "squeeze" and squeeze and 1 in shape and ndim > 3:
        ax = shape.index(1)
        return X.squeeze(ax), shape[:ax] + shape[ax + 1 :], coords[:ax] + coords[ax + 1 :], view
    return X, shape, coords, "none"


def oracle_mttkrp(rng, n, teeth=False):
    o = Oracle("mttkrp")
    numba_variants = gb.core._has_numba
    if numba_variants:
        from graphblas.core import tensor_numba
    letters = "abcdefgh"
    with census() as records:
        for _ in range(n):
            shape, coords, vals, kind = random_tensor_coords(rng, (3, 6), sparse=rng.random() < 0.5)
            layout = random_layout(rng, len(shape))
            X = Tensor.from_coo(tuple(coords), vals, shape=shape, layout=layout)
            X, shape, coords, view = random_view(rng, X, shape, coords)
            ndim = len(shape)
            R = int(rng.choice([1, 2, 5, 16]))
            F = [rng.standard_normal((d, R)) for d in shape]
            matrices = rng.random() < 0.5
            factors = [Matrix.from_dense(f) for f in F] if matrices else F
            if numba_variants:
                tensor_numba._FORCE_RUNS = rng.choice([None, True, False])
                tensor_numba._FORCE_COLBLOCK = rng.choice([None, True, False])
            nthreads = int(rng.choice([1, 4]))
            gb.ss.config["nthreads"] = nthreads
            dense = np.zeros(shape)
            np.add.at(dense, tuple(coords), vals)
            for mode in range(ndim):
                what = f"{shape} {layout} view={view} {kind} R={R} mode={mode} nt={nthreads}"
                start = len(records)
                arg = mode - ndim if rng.random() < 0.2 else mode
                res = o.run(what, lambda: X.mttkrp(factors, arg).new())
                if res is None:
                    continue
                for choice in choices(records, start, "mttkrp"):
                    o.stats[choice] += 1
                m = (mode + 1) % ndim if teeth else mode
                subs = [letters[:ndim]] + [letters[a] + "r" for a in range(ndim) if a != m]
                ops = [dense] + [F[a] for a in range(ndim) if a != m]
                want = np.einsum(",".join(subs) + "->" + letters[m] + "r", *ops)
                fibers = np.zeros(shape[m], bool)
                fibers[coords[m]] = True
                o.stats["checks"] += 1
                if not valid(res):
                    o.fail(what, "the stored Matrix is malformed")
                elif not matches(res, want, np.broadcast_to(fibers[:, None], want.shape)):
                    o.fail(what, "values or presence differ from the model")
    if numba_variants:
        tensor_numba._FORCE_RUNS = tensor_numba._FORCE_COLBLOCK = None
    return o


# ---------------------------------------------------------------------------------------------
# TTM (tensordot of one axis with a 2-d factor): every engine


def oracle_ttm(rng, n, teeth=False):
    o = Oracle("ttm")
    numba_variants = gb.core._has_numba
    if numba_variants:
        from graphblas.core import tensor_numba
    with census() as records:
        for _ in range(n):
            ndim = int(rng.integers(2, 6))
            style = rng.choice(["small", "sparse", "hyper"])
            if style == "small":
                shape = tuple(int(rng.integers(1, 7)) for _ in range(ndim))
                nnz = int(rng.binomial(int(np.prod(shape)), float(rng.choice([0.1, 0.5, 1.0]))))
            elif style == "sparse":
                shape = tuple(int(rng.integers(2, 13 if ndim < 5 else 8)) for _ in range(ndim))
                nnz = int(rng.binomial(int(np.prod(shape)), float(rng.choice([0.02, 0.1]))))
            else:  # many rows, few of them nonempty: hypersparse storage
                big = 60 if ndim < 4 else 25
                shape = (*(int(rng.integers(20, big)) for _ in range(ndim - 1)), 8)
                nnz = int(rng.integers(0, 60))
            size = int(np.prod(shape))
            nnz = min(nnz, size)
            coords = list(np.unravel_index(np.sort(rng.choice(size, nnz, replace=False)), shape))
            kind = rng.choice(["normal", "iso", "zeros"])
            vals = {
                "iso": lambda: np.full(nnz, 1.5),
                "zeros": lambda: rng.choice([0.0, 1.0, -2.0], nnz),
                "normal": lambda: rng.standard_normal(nnz),
            }[kind]()
            if rng.random() < 0.15:
                vals = np.round(vals * 3).astype(np.int64)
            if style == "hyper" and rng.random() < 0.7:
                layout = (tuple(range(ndim - 1)), (ndim - 1,))
            else:
                layout = random_layout(rng, ndim)
            X = Tensor.from_coo(tuple(coords), vals, shape=shape, layout=layout)
            X, shape, coords, view = random_view(rng, X, shape, coords, squeeze=False)
            ndim = len(shape)
            if numba_variants:
                tensor_numba._FORCE_RUNS = rng.choice([None, True, False])
            nthreads = int(rng.choice([1, 4]))
            gb.ss.config["nthreads"] = nthreads
            dense, present = np.zeros(shape), np.zeros(shape, bool)
            dense[tuple(coords)], present[tuple(coords)] = vals, True
            for m in range(ndim):
                R = int(rng.choice([1, 3, 16]))
                kb = int(rng.integers(0, 2))  # the factor's contracted axis
                Ud = rng.standard_normal((shape[m], R) if kb == 0 else (R, shape[m]))
                Up = np.ones(Ud.shape, bool)
                fkind = rng.choice(["numpy", "matrix", "sparse"])
                if fkind == "numpy":
                    U = Ud
                elif fkind == "matrix":
                    U = Matrix.from_dense(Ud)
                else:
                    Up = rng.random(Ud.shape) < 0.6
                    r, c = np.nonzero(Up)
                    U = Matrix.from_coo(r, c, Ud[r, c], nrows=Ud.shape[0], ncols=Ud.shape[1])
                semi = str(rng.choice(["plus_times", "plus_times", "min_plus"]))
                what = (
                    f"{shape} {layout} {style} view={view} {kind} {vals.dtype} axis={m} "
                    f"kb={kb} R={R} {fkind} {semi} nt={nthreads}"
                )
                start = len(records)
                op = getattr(semiring, semi)
                res = o.run(what, lambda: X.tensordot(U, axes=([m], [kb]), op=op).new())
                if res is None:
                    continue
                for choice in choices(records, start, "ttm"):
                    o.stats[choice] += 1
                Uo, Po = (Ud, Up) if kb == 0 else (Ud.T, Up.T)
                if teeth:  # the wrong factor axis, or the factor's rows reversed
                    Uo, Po = (Uo.T, Po.T) if Uo.shape[0] == Uo.shape[1] else (Uo[::-1], Po[::-1])
                if semi == "plus_times":
                    want = np.tensordot(dense, np.where(Po, Uo, 0), axes=([m], [0]))
                else:
                    moved = np.moveaxis(np.where(present, dense, np.inf), m, -1)[..., :, None]
                    want = np.min(moved + np.where(Po, Uo, np.inf), axis=-2)
                wantp = np.tensordot(present.astype(int), Po.astype(int), axes=([m], [0])) > 0
                o.stats["checks"] += 1
                if not valid(res):
                    o.fail(what, "the stored Matrix is malformed")
                elif not matches(res, want, wantp):
                    o.fail(what, "values or presence differ from the model")
    if numba_variants:
        tensor_numba._FORCE_RUNS = None
    return o


# ---------------------------------------------------------------------------------------------
# Reshape, ravel and flatten; copy-on-write both ways


def random_target_shape(rng, shape):
    size = int(np.prod(shape))
    if size == 0 or rng.random() < 0.3:  # random splits and fuses of adjacent axes
        s = list(shape)
        for _ in range(int(rng.integers(1, 4))):
            if len(s) >= 2 and rng.random() < 0.5:
                i = int(rng.integers(0, len(s) - 1))
                s[i : i + 2] = [s[i] * s[i + 1]]
            else:
                i = int(rng.integers(0, len(s)))
                divs = [d for d in range(1, s[i] + 1) if s[i] % d == 0] or [1]
                d = int(rng.choice(divs))
                s[i : i + 1] = [d, s[i] // d if d else 0]
        return tuple(s) if len(s) <= 6 else (*s[:5], int(np.prod(s[5:])))
    factors, rest = [], size  # a random factorization into 1-5 factors
    for _ in range(int(rng.integers(0, 5))):
        d = int(rng.choice([d for d in range(1, rest + 1) if rest % d == 0]))
        factors.append(d)
        rest //= d
    factors.append(rest)
    rng.shuffle(factors)
    return tuple(int(f) for f in factors)


def oracle_reshape(rng, n, teeth=False):
    o = Oracle("reshape")
    order = "F" if teeth else "C"
    for _ in range(n):
        ndim = int(rng.integers(1, 6))
        shape = tuple(int(rng.integers(0 if rng.random() < 0.05 else 1, 6)) for _ in range(ndim))
        xv, xp = rand_model(rng, shape, float(rng.choice([0.1, 0.5, 1.0])))
        X = build(rng, xv, xp)
        target = random_target_shape(rng, shape)
        arg = list(target)
        if arg and rng.random() < 0.3 and int(np.prod(target)) > 0:
            arg[int(rng.integers(0, len(arg)))] = -1
        copy = rng.choice([None, True, False])
        what = f"{shape} {X.layout} -> {tuple(arg)} copy={copy}"
        try:
            R = X.reshape(tuple(arg), copy=copy)
        except ValueError as exc:
            if copy is False:  # allowed to refuse, never to return a wrong result
                o.stats["copy=False refused"] += 1
            else:
                o.fail(what, f"raised ValueError: {exc}")
            continue
        except Exception as exc:
            o.fail(what, f"raised {type(exc).__name__}: {exc}")
            continue
        o.stats["reshape"] += 1
        wantv, wantp = xv.reshape(target, order=order), xp.reshape(target, order=order)
        if not matches(R, wantv, wantp):
            o.fail(what, "values or presence differ from the model")
            continue
        # Copy-on-write: writing either side leaves the other as it was
        if R.nvals:
            R << R.apply(unary.ainv)
            o.stats["copy-on-write"] += 1
            if not matches(X, xv, xp):
                o.fail(what, "writing the result changed the source")
        if X.nvals:
            before = model_of(R)
            X << X.apply(unary.ainv)
            o.stats["copy-on-write"] += 1
            if not matches(R, *before):
                o.fail(what, "writing the source changed the result")
    for _ in range(max(1, n // 8)):
        shape = tuple(int(rng.integers(1, 5)) for _ in range(int(rng.integers(1, 5))))
        xv, xp = rand_model(rng, shape, 0.5)
        X = build(rng, xv, xp)
        for method in ("ravel", "flatten"):
            o.expect(
                f"{method} {shape} {X.layout}", getattr(X, method),
                xv.reshape(-1, order=order), xp.reshape(-1, order=order), count=method,
            )  # fmt: skip
    return o


# ---------------------------------------------------------------------------------------------
# Assignment: a model of GraphBLAS assign and subassign, validated against Matrix first


def region_of(shape, key):
    """The flat positions an outer index selects, shaped like the region."""
    return outer_index(np.arange(int(np.prod(shape))).reshape(shape), key)


def assign_model(cv, cp, key, vv, vp, mask, how, accum, replace, region_shape, teeth):
    """GraphBLAS assign (how="assign": the mask is over all of C) or subassign (the mask is
    over the region).
    """
    replace = replace and not teeth
    flat = region_of(cv.shape, key)
    vv, vp = np.broadcast_to(vv, region_shape), np.broadcast_to(vp, region_shape)
    zv, zp = cv.copy().reshape(-1), cp.copy().reshape(-1)
    old_v, old_p = zv[flat], zp[flat]
    if accum is None:
        new_v, new_p = vv, vp
    else:
        new_p = old_p | vp
        new_v = np.where(old_p & vp, accum(old_v, vv), np.where(old_p, old_v, vv))
    if how == "subassign":
        m = np.ones(region_shape, bool) if mask is None else np.broadcast_to(mask, region_shape)
        zv[flat] = np.where(m, new_v, old_v)
        zp[flat] = np.where(m, new_p, old_p & (not replace))
        return zv.reshape(cv.shape), zp.reshape(cv.shape)
    zv[flat], zp[flat] = new_v, new_p  # assign: Z = C with Z(region) = new; then C<M> = Z
    zv, zp = zv.reshape(cv.shape), zp.reshape(cv.shape)
    if mask is None:
        return zv, zp
    if replace:
        return np.where(mask, zv, 0.0), mask & zp
    return np.where(mask, zv, cv), np.where(mask, zp, cp)


def make_object(rng, kind, values, present):
    """A Matrix, Vector, Scalar or Tensor (in a random layout) holding the model."""
    if kind == "scalar" or values.ndim == 0:
        if present[()]:
            return Scalar.from_value(float(values[()]))
        return Scalar(float, is_cscalar=False)  # pragma: is_grbscalar
    coords = np.nonzero(present)
    if kind == "matrix":
        nrows, ncols = values.shape
        return Matrix.from_coo(*coords, values[coords], nrows=nrows, ncols=ncols, dtype=float)
    if kind == "vector":
        return Vector.from_coo(coords[0], values[coords], size=values.shape[0], dtype=float)
    return build(rng, values, present, view=False, dtype=float)


ACCUMS = {None: None, "plus": (binary.plus, np.add), "max": (binary.max, np.maximum)}


def oracle_assign(rng, n, teeth=False, *, target="tensor"):
    o = Oracle(f"assign ({target})")
    on_matrix = target == "matrix"
    for _ in range(n):
        if on_matrix:
            shape = tuple(int(rng.integers(1, 6)) for _ in range(2))
        else:
            shape = tuple(int(rng.integers(1, 5)) for _ in range(int(rng.integers(1, 6))))
        cv, cp = rand_model(rng, shape, kind="int")
        C = make_object(rng, "matrix", cv, cp) if on_matrix else build(rng, cv, cp, dtype=float)
        key = tuple(random_key(rng, d) for d in shape)
        region_shape = region_of(shape, key).shape
        vkind = rng.choice(["object", "scalar", "python", "broadcast"])
        if vkind == "python":
            vv, vp = np.array(float(rng.integers(-3, 4))), np.array(True)
            value = float(vv)
        elif vkind == "scalar":
            vv, vp = rand_model(rng, (), 0.8, kind="int")
            value = make_object(rng, "scalar", vv, vp)
        else:
            vshape = region_shape
            if vkind == "broadcast" and not on_matrix and region_shape:
                vshape = tuple(1 if rng.random() < 0.5 else s for s in region_shape)
                if rng.random() < 0.3:
                    vshape = vshape[int(rng.integers(0, len(vshape) + 1)) :]
            vv, vp = rand_model(rng, vshape, 0.6, kind="int")
            if on_matrix:
                vk = {2: "matrix", 1: "vector"}.get(len(vshape), "scalar")
            else:
                vk = "tensor" if vshape else "scalar"
            value = make_object(rng, vk, vv, vp)
        how = str(rng.choice(["assign", "subassign"]))
        mshape = shape if how == "assign" else region_shape
        mask = mask_arg = None
        if rng.random() < 0.6 and (how == "assign" or region_shape):
            mk = "tensor"
            if on_matrix:
                mk = {2: "matrix", 1: "vector"}[len(mshape)]
            mask_arg, mask, _ = random_mask(
                rng, mshape, make=lambda rng, v, p, mk=mk: make_object(rng, mk, v, p)
            )
        acc = rng.choice(list(ACCUMS))
        kw = {}
        if acc is not None:
            kw["accum"] = ACCUMS[acc][0]
        if mask is not None and (teeth or rng.random() < 0.3):  # (the teeth ignore replace)
            kw["replace"] = True
        what = f"{shape} {key} value={vkind} {how} mask={mask is not None} {sorted(kw)}"
        if how == "assign":
            if mask is None:
                o.run(what, lambda: C(**kw)[key] << value)
            else:
                o.run(what, lambda: C(mask_arg, **kw)[key] << value)
        elif mask is None:
            o.run(what, lambda: C[key](**kw) << value)
        else:
            o.run(what, lambda: C[key](mask_arg, **kw) << value)
        want = assign_model(
            cv, cp, key, vv, vp, mask, how, None if acc is None else ACCUMS[acc][1],
            kw.get("replace", False), region_shape, teeth,
        )  # fmt: skip
        o.stats[how] += 1
        if not matches(C, *want):
            o.fail(what, "values or presence differ from the model")
    return o


# ---------------------------------------------------------------------------------------------
# Gather: take, take_along_axis and vindex against numpy


def oracle_gather(rng, n, teeth=False):
    o = Oracle("gather")
    with census() as records:
        for _ in range(n):
            shape = tuple(int(rng.integers(1, 6)) for _ in range(int(rng.integers(1, 5))))
            xv, xp = rand_model(rng, shape, 0.5)
            X = build(rng, xv, xp)
            if teeth:  # the model reads the next position along the last axis
                xv, xp = np.roll(xv, -1, axis=-1), np.roll(xp, -1, axis=-1)
            ndim = len(shape)
            axis = int(rng.integers(0, ndim))
            d = shape[axis]
            start = len(records)
            ishape = tuple(int(x) for x in rng.integers(0, 4, int(rng.integers(0, 3))))
            idx = rng.integers(-d, d, size=ishape)
            o.expect(
                f"take {shape} {X.layout} axis={axis} {idx.tolist()}",
                lambda: X.take(idx, axis=axis), np.take(xv, idx, axis=axis),
                np.take(xp, idx, axis=axis),
            )  # fmt: skip
            along = list(shape)
            along[axis] = int(rng.integers(0, 4))
            idx = rng.integers(-d, d, size=tuple(along))
            o.expect(
                f"take_along_axis {shape} {X.layout} axis={axis}",
                lambda: X.take_along_axis(idx, axis), np.take_along_axis(xv, idx, axis),
                np.take_along_axis(xp, idx, axis),
            )  # fmt: skip
            bshape = tuple(int(x) for x in rng.integers(1, 4, int(rng.integers(1, 3))))
            arrays = tuple(rng.integers(-s, s, size=bshape) for s in shape)
            o.expect(
                f"vindex {shape} {X.layout} {bshape}", lambda: X.vindex[arrays],
                xv[arrays], xp[arrays],
            )  # fmt: skip
            for choice in choices(records, start, "gather"):
                o.stats[f"engine {choice}"] += 1
    return o


# ---------------------------------------------------------------------------------------------
# einsum: the plan cache against new layouts, masked products, batched matmul


def oracle_einsum(rng, n, teeth=False):
    o = Oracle("einsum")
    letters = "abcdefg"
    specs = []
    for _ in range(max(1, n // 5)):
        used = list(letters[: int(rng.integers(2, 6))])
        terms = [
            "".join(rng.choice(used, int(rng.integers(1, min(4, len(used)) + 1)), replace=False))
            for _ in range(int(rng.integers(2, 4)))
        ]
        present = sorted(set("".join(terms)))
        out = "".join(c for c in present if rng.random() < 0.5)
        specs.append((",".join(terms) + "->" + out, {c: int(rng.integers(1, 5)) for c in present}))
    for case in range(n):
        spec, sizes = specs[case % len(specs)]  # reused, so a cached plan meets new layouts
        ins, out = spec.split("->")
        models = [rand_model(rng, tuple(sizes[c] for c in t), kind="int") for t in ins.split(",")]
        operands = [build(rng, v, p) for v, p in models]
        oshape = tuple(sizes[c] for c in out)
        want = np.einsum(spec, *[v for v, _ in models])
        wantp = np.einsum(spec, *[p.astype(int) for _, p in models]) > 0
        if not oshape or rng.random() < 0.5:
            o.expect(f"einsum {spec} {sizes}", lambda: gb.einsum(spec, *operands).new(), want,
                     wantp)  # fmt: skip
            continue
        mask_arg, mask, mkind = random_mask(rng, oshape)
        cv, cp = rand_model(rng, oshape, 0.3)
        C = build(rng, cv, cp)
        what = f"einsum {spec} {sizes} mask={mkind}"
        o.run(what, lambda: C(mask_arg) << gb.einsum(spec, *operands))
        if teeth:  # the model drops the mask
            mask = np.ones(oshape, bool)
        o.stats["einsum masked"] += 1
        if not matches(C, np.where(mask, want, cv), np.where(mask, wantp, cp)):
            o.fail(what, "values or presence differ from the model")
    for _ in range(max(4, n // 3)):
        b, i, k, j = (int(x) for x in rng.integers(1, 5, 4))
        av, ap = rand_model(rng, (b, i, k))
        bv, bp = rand_model(rng, (b, k, j))
        A, B = build(rng, av, ap), build(rng, bv, bp)
        want, wantp = av @ bv, (ap.astype(int) @ bp.astype(int)) > 0
        o.expect(f"batched matmul {(b, i, k, j)}", lambda: A @ B, want, wantp)
    return o


# ---------------------------------------------------------------------------------------------
# io: pydata/sparse COO and GCXS, scipy n-d COO with duplicates


def oracle_io(rng, n, teeth=False):
    try:
        import sparse
    except ImportError:  # (pydata sparse needs Numba)
        pytest.skip("needs pydata sparse")
    ssp = pytest.importorskip("scipy.sparse")
    o = Oracle("io")

    def compare(what, shape, coords, vals, got_shape, got_coords, got_vals):
        want_v, want_p = np.zeros(shape), np.zeros(shape, bool)
        if len(shape):
            want_v[tuple(coords)], want_p[tuple(coords)] = vals, True
        if teeth and len(vals):  # the model drops one entry
            want_p[tuple(c[0] for c in coords)] = False
        got = np.zeros(tuple(got_shape), bool), np.zeros(tuple(got_shape))
        o.stats[what] += 1
        if tuple(got_shape) != tuple(shape):
            o.fail(what, f"shape {tuple(got_shape)}, not {shape}")
            return
        if len(shape):
            got[0][tuple(got_coords)], got[1][tuple(got_coords)] = True, got_vals
        if not (got[0] == want_p).all() or not np.allclose(
            got[1][want_p].astype(float), want_v[want_p].astype(float)
        ):
            o.fail(f"{what} {shape} nnz={len(vals)}", "entries differ")

    for _ in range(n):
        ndim = int(rng.integers(1, 6))
        shape = tuple(int(rng.integers(0 if rng.random() < 0.05 else 1, 7)) for _ in range(ndim))
        size = int(np.prod(shape))
        nnz = int(rng.binomial(size, float(rng.choice([0.0, 0.1, 0.5, 1.0])))) if size else 0
        coords = list(np.unravel_index(np.sort(rng.choice(size, nnz, replace=False)), shape))
        kind = rng.choice(["normal", "iso", "zeros", "int", "bool"])
        vals = {
            "iso": lambda: np.full(nnz, 2.5),
            "zeros": lambda: rng.choice([0.0, 1.0], nnz),
            "int": lambda: rng.integers(-3, 4, nnz).astype(np.int32),
            "bool": lambda: rng.random(nnz) < 0.5,
            "normal": lambda: rng.standard_normal(nnz),
        }[kind]()
        X = Tensor.from_coo(tuple(coords), vals, shape=shape, layout=random_layout(rng, ndim))
        if rng.random() < 0.4:
            p = [int(a) for a in rng.permutation(ndim)]
            X = X.permute_dims(p)
            shape, coords = tuple(shape[a] for a in p), [coords[a] for a in p]
        s = X.to_pydata_sparse()
        compare("to pydata COO", shape, coords, vals, s.shape, s.coords, s.data)
        Y = Tensor.from_pydata_sparse(sparse.COO(np.array(coords).reshape(ndim, -1), vals, shape))
        compare("from pydata COO", shape, coords, vals, Y.shape, *Y.to_coo())
        if ndim >= 2:
            axes = rng.choice(ndim, int(rng.integers(1, ndim)), replace=False)
            axes = tuple(sorted(int(a) for a in axes))
            g = X.to_pydata_sparse(format="gcxs", compressed_axes=axes)
            if tuple(g.compressed_axes) != axes:
                o.fail("to GCXS", f"compressed axes {g.compressed_axes}, not {axes}")
            gc = g.tocoo()
            compare("to GCXS", shape, coords, vals, gc.shape, gc.coords, gc.data)
            Y = Tensor.from_pydata_sparse(g)
            compare("from GCXS", shape, coords, vals, Y.shape, *Y.to_coo())
        a = X.to_scipy_sparse()
        compare("to scipy COO", shape, coords, vals, a.shape, a.coords, a.data)
        if nnz and vals.dtype != bool:  # duplicates add up
            dup = rng.integers(0, nnz, max(1, nnz // 3))
            cc = [np.concatenate([c, c[dup]]) for c in coords]
            vv = np.concatenate([vals, vals[dup]])
            order = rng.permutation(len(vv))
            A = ssp.coo_array((vv[order], tuple(c[order] for c in cc)), shape=shape)
            summed = vals.copy()
            np.add.at(summed, dup, vals[dup])
            Y = Tensor.from_scipy_sparse(A)
            compare("from scipy COO, duplicates", shape, coords, summed, Y.shape, *Y.to_coo())
        if nnz and vals.dtype.kind == "f":  # a fill value other than 0 is refused
            s = sparse.COO(np.array(coords).reshape(ndim, -1), vals, shape, fill_value=1.0)
            o.stats["fill_value refused"] += 1
            try:
                Tensor.from_pydata_sparse(s)
                o.fail(f"fill_value {shape}", "accepted")
            except ValueError:
                pass
    return o


# ---------------------------------------------------------------------------------------------
# The tests


@pytest.fixture
def engines(monkeypatch):
    """Let the kernels run on small inputs (they wait for 2^16 entries otherwise)."""
    monkeypatch.setattr(tensor, "_TTM_KERNEL_MIN_NNZ", 0)
    monkeypatch.setattr(tensor, "_MTTKRP_KERNEL_MIN_NNZ", 0)
    return monkeypatch


def test_ewise_reduce(rng):
    oracle_ewise_reduce(rng, cases(40)).verdict()


def test_broadcast_contract(rng):
    oracle_broadcast_contract(rng, cases(40)).verdict()


@pytest.mark.parametrize("numba", [True, False])
def test_mttkrp(rng, engines, numba):
    if numba and not gb.core._has_numba:
        pytest.skip("needs numba")
    engines.setattr(tensor, "_has_numba", numba and gb.core._has_numba)
    o = oracle_mttkrp(rng, cases(20))
    o.verdict()
    ran_kernel = sum(v for k, v in o.stats.items() if k.startswith("numba"))
    if numba:
        assert ran_kernel, f"the Numba kernels never ran: {dict(o.stats)}"
    else:
        assert not ran_kernel, dict(o.stats)


@pytest.mark.parametrize(
    "engine_list", [None, ("numba",), ("udt",), ("products",)], ids=str
)  # fmt: skip
def test_ttm(rng, engines, engine_list):
    if engine_list == ("numba",) and not gb.core._has_numba:
        pytest.skip("needs numba")
    if engine_list is not None:
        engines.setattr(tensor, "_TTM_ENGINES", engine_list)
    o = oracle_ttm(rng, cases(30))
    o.verdict()
    if engine_list is not None and engine_list != ("products",):
        (engine,) = engine_list
        if not o.stats[engine]:
            # The results were right, but the engine never ran (the UDT engine needs the JIT)
            pytest.skip(f"the {engine} engine did not run here: {dict(o.stats)}")
    if engine_list == ("products",):
        assert not o.stats["numba"], dict(o.stats)
        assert not o.stats["udt"], dict(o.stats)


def test_ttm_without_numba(rng, engines):
    engines.setattr(tensor, "_has_numba", False)
    o = oracle_ttm(rng, cases(15))
    o.verdict()
    assert not o.stats["numba"], dict(o.stats)


def test_reshape(rng):
    oracle_reshape(rng, cases(150)).verdict()


def test_assign_model_matches_matrix(rng):
    """The assignment model is right where GraphBLAS itself is the ground truth: on Matrix."""
    oracle_assign(rng, cases(40), target="matrix").verdict()


def test_assign(rng):
    oracle_assign(rng, cases(50)).verdict()


@pytest.mark.parametrize(
    "engine_list", [None, ("numba", "graphblas"), ("graphblas",)], ids=str
)  # fmt: skip
def test_gather(rng, monkeypatch, engine_list):
    if engine_list is not None:
        monkeypatch.setattr(tensor_index, "_GATHER_ENGINES", engine_list)
    o = oracle_gather(rng, cases(60))
    o.verdict()
    if engine_list == ("graphblas",):
        assert not o.stats["engine numba"], dict(o.stats)
        assert not o.stats["engine extract"], dict(o.stats)


def test_einsum(rng):
    oracle_einsum(rng, cases(80)).verdict()


def test_io(rng):
    oracle_io(rng, cases(60)).verdict()


# The planted errors, and enough cases that each is all but certain to show: the operands of
# a non-commutative operator swapped, the wrong mode or factor axis, F order for C order,
# "replace" ignored, the mask dropped, a gather one position off, an entry dropped.
TEETH = {
    "ewise_reduce": (oracle_ewise_reduce, 20, {}),
    "broadcast_contract": (oracle_broadcast_contract, 10, {}),
    "mttkrp": (oracle_mttkrp, 10, {}),
    "ttm": (oracle_ttm, 10, {}),
    "reshape": (oracle_reshape, 40, {}),
    "assign_matrix": (oracle_assign, 40, {"target": "matrix"}),
    "assign": (oracle_assign, 40, {}),
    "gather": (oracle_gather, 10, {}),
    "einsum": (oracle_einsum, 60, {}),
    "io": (oracle_io, 10, {}),
}


@pytest.mark.parametrize("name", list(TEETH))
def test_oracle_teeth(rng, engines, name):
    """Each oracle catches the error planted in its model."""
    oracle, n, kwargs = TEETH[name]
    oracle(rng, n, teeth=True, **kwargs).verdict(teeth=True)

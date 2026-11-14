"""torch.sparse COO (pytorch 2.14.1 CPU, conda-forge), float64.

Results that torch leaves uncoalesced are coalesced inside the timed call (and the record
says so in its note).
"""

import datasets as D
import numpy as np
import torch
from common import NA, dense_guard, parse_axes, parse_perm, ttm_guard

torch.sparse.check_sparse_tensor_invariants.disable()


def _bshape(shape, a):
    return (shape[a],) + (1,) * (len(shape) - 1 - a)


def _sp(coords, vals, shape, coalesced=False):
    idx = torch.from_numpy(np.stack(coords))
    T = torch.sparse_coo_tensor(
        idx,
        torch.from_numpy(np.ascontiguousarray(vals)),
        tuple(shape),
        is_coalesced=coalesced,
        check_invariants=False,
    )
    return T if coalesced else T.coalesce()


def _try(fn):
    """Run ``fn`` once; a library error becomes NA with the message."""
    try:
        fn()
    except RuntimeError as e:  # NotImplementedError is a RuntimeError
        raise NA(f"{type(e).__name__}: {str(e).splitlines()[0][:160]}") from None
    return fn


NOTES = {
    "add_perm": "coalesce() included (X + Z.permute(...) is uncoalesced)",
    "mul_perm": "coalesce() included",
    "perm": "permute(p).coalesce(); for p keeping dim 0, rebuilt first "
    "(torch flags it coalesced unsorted)",
    "slice": "select(a, i).coalesce() (select leaves it uncoalesced); ranges by narrow_copy",
    "ttv": "torch.sparse.sum(X * v.view(...), dim=m) (sparse result)",
    "ttm": "bmm(sparse, dense U.expand) after permute+coalesce: dense output",
    "mttkrp": "gather + index_add_ over nnz x R (idiomatic torch)",
    "construct": "sparse_coo_tensor(...).coalesce()",
}


class Adapter:
    name = "torch"
    threads = (8, 1)

    def __init__(self, ds, coords, vals, shape):
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.nd = len(shape)
        self.stacked = torch.from_numpy(np.stack(coords))
        self.tvals = torch.from_numpy(vals)
        self.X = _sp(coords, vals, shape, coalesced=True)

    def info(self):
        return {
            "version": f"torch {torch.__version__}",
            "parallel": torch.__config__.parallel_info().splitlines()[-1],
        }

    def set_threads(self, nt):
        torch.set_num_threads(nt)

    def extra_cases(self, group):
        return []

    def note(self, group, case):
        if group == "ewise":
            return NOTES.get(case, "")
        if group == "permute":
            return NOTES["perm"]
        return NOTES.get(group, "")

    def setup(self, group):
        shape, nd = self.shape, self.nd
        if group == "construct":
            order = torch.from_numpy(np.random.default_rng(99).permutation(self.vals.size))
            self.shuf = self.stacked[:, order].contiguous(), self.tvals[order].contiguous()
        elif group == "ewise":
            yc, yv = D.second_operand(self.ds, self.coords, self.vals, shape)
            self.Y = _sp(yc, yv, shape)
            roll = tuple(range(1, nd)) + (0,)
            self.inv = (nd - 1,) + tuple(range(nd - 1))
            self.Z = _sp([yc[a] for a in roll], yv, tuple(shape[a] for a in roll))
        elif group == "bcast":
            vecs = D.vectors(self.ds, shape)
            self.bvecs = [
                torch.from_numpy(v.reshape(_bshape(shape, a))) for a, v in enumerate(vecs)
            ]
            M12 = D.dense_matrix12(self.ds, shape) if nd == 3 else None
            self.M12 = None if M12 is None else torch.from_numpy(M12)
        elif group == "ttv":
            self.vecs = [torch.from_numpy(v) for v in D.vectors(self.ds, shape)]
        elif group in {"ttm", "mttkrp"}:
            self.facs = {
                R: [torch.from_numpy(F) for F in D.factors(self.ds, shape, R)] for R in (16, 64)
            }
            if group == "mttkrp":
                self.idx = self.X.indices()
                self.xv = self.X.values()
        elif group == "contract":
            (r, c, v, L), (c2, v2, shape2) = D.tdot_operands(self.ds, shape)
            self.Ms = _sp([r, c], v, (shape[2], L))
            self.Y2 = _sp(c2, v2, shape2)
        elif group == "bmm":
            c2, v2, shape2 = D.load(self.ds, which=1)
            self.B = _sp(c2, v2, shape2)

    def case(self, group, case):
        X, shape, nd = self.X, self.shape, self.nd
        if group == "construct":
            if case == "from_coo_sorted":
                return lambda: torch.sparse_coo_tensor(self.stacked, self.tvals, shape).coalesce()
            if case == "from_coo_shuffled":
                i, v = self.shuf
                return lambda: torch.sparse_coo_tensor(i, v, shape).coalesce()
            if case == "to_coo":
                return lambda: (X.indices(), X.values())
        if group == "ewise":
            if case == "add_same":
                return lambda: X + self.Y
            if case == "mul_same":
                return lambda: X * self.Y
            if case == "add_perm":
                return lambda: (X + self.Z.permute(self.inv)).coalesce()
            if case == "mul_perm":
                return lambda: (X * self.Z.permute(self.inv)).coalesce()
        if group == "bcast":
            if case == "mul_mat12":
                if self.M12 is None:
                    raise NA("dense (s1, s2) matrix too large")
                return lambda: X * self.M12
            v = self.bvecs[int(case[-1])]
            return lambda: X * v
        if group == "reduce":
            axes = parse_axes(case)
            if axes is None:
                return lambda: torch.sparse.sum(X)
            return lambda: torch.sparse.sum(X, dim=axes)
        if group == "permute":
            p = parse_perm(case)

            def run():
                P = X.permute(p)
                if P.is_coalesced():
                    # torch flags permute() that keeps dim 0 as coalesced without re-sorting,
                    # so coalesce() would be a no-op; rebuild to force the sort
                    P = torch.sparse_coo_tensor(P._indices(), P._values(), P.shape)
                return P.coalesce()

            return run
        if group == "ttv":
            m = int(case[3:])
            v = self.vecs[m].view(_bshape(shape, m))
            return lambda: torch.sparse.sum(X * v, dim=m)
        if group == "ttm":
            if nd != 3:
                raise NA("bmm route is 3-d only")
            m, R = int(case[3]), int(case.split("_r")[1])
            ttm_guard(self.ds, self.coords, self.shape, m, R)
            others = [a for a in range(3) if a != m]
            dense_guard([shape[a] for a in others] + [R])
            U = self.facs[R][m]
            order = (others[0], others[1], m)
            Ue = U.expand(shape[others[0]], shape[m], R)
            if m == 2:
                return lambda: torch.bmm(X, Ue)
            return lambda: torch.bmm(X.permute(order).coalesce(), Ue)
        if group == "mttkrp":
            m = int(case[6])
            need = X._nnz() * 16 * 8 * 3
            if need > 12 * 2**30:
                raise NA(f"gather + index_add needs nnz x R temporaries (~{need / 2**30:.0f} GiB)")
            F = self.facs[16]
            others = [a for a in range(nd) if a != m]
            idx, xv = self.idx, self.xv

            def run():
                P = xv[:, None] * F[others[0]][idx[others[0]]]
                for a in others[1:]:
                    P.mul_(F[a][idx[a]])
                out = torch.zeros(shape[m], 16, dtype=torch.float64)
                out.index_add_(0, idx[m], P)
                return out

            return run
        if group == "contract":
            if case == "tdot1":
                return _try(lambda: torch.tensordot(X, self.Ms, dims=([2], [0])))
            return _try(lambda: torch.tensordot(X, self.Y2, dims=([1, 2], [0, 1])))
        if group == "bmm":
            return _try(lambda: torch.bmm(X, self.B))
        if group == "slice":
            key = D.slices(shape)[case]
            a = next(i for i, k in enumerate(key) if k != slice(None))
            k = key[a]
            if isinstance(k, int):
                return lambda: X.select(a, k).coalesce()
            return lambda: X.narrow_copy(a, k.start, k.stop - k.start)
        raise NA(f"unknown case {group}/{case}")

    def canon(self, r):
        if isinstance(r, tuple):
            return ("coo", list(r[0].numpy()), r[1].numpy(), self.shape)
        if r.layout == torch.sparse_coo:
            r = r.coalesce()
            if r.ndim == 0:
                return ("dense", r.to_dense().numpy())
            return ("coo", list(r.indices().numpy()), r.values().numpy(), tuple(r.shape))
        return ("dense", r.numpy())

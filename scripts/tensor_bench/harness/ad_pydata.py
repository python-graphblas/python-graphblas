"""pydata/sparse COO (Numba backend), single-threaded."""

import datasets as D
import numpy as np
import sparse
from common import NA, dense_guard, parse_axes, parse_perm, ttm_guard

LETTERS = "ijklmn"


def _bshape(shape, a):
    return (shape[a],) + (1,) * (len(shape) - 1 - a)


def mttkrp_subscripts(nd, m):
    idx = LETTERS[:nd]
    ins = [idx] + [f"{idx[a]}r" for a in range(nd) if a != m]
    return ",".join(ins) + f"->{idx[m]}r"


class Adapter:
    name = "pydata"
    threads = (1,)

    def __init__(self, ds, coords, vals, shape):
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.nd = len(shape)
        self.stacked = np.stack(coords)
        self.X = sparse.COO(self.stacked, vals, shape=shape, has_duplicates=False, sorted=True)

    def info(self):
        return {"version": f"sparse {sparse.__version__}"}

    def set_threads(self, nt):
        pass

    def extra_cases(self, group):
        if group == "mttkrp":
            return [f"mttkrp{m}_r16_ttm" for m in range(self.nd)]
        return []

    def _coo(self, coords, vals, shape):
        return sparse.COO(np.stack(coords), vals, shape=shape, has_duplicates=False, sorted=False)

    def setup(self, group):
        shape, nd = self.shape, self.nd
        if group == "construct":
            order = np.random.default_rng(99).permutation(self.vals.size)
            self.shuf = self.stacked[:, order], self.vals[order]
        elif group == "ewise":
            yc, yv = D.second_operand(self.ds, self.coords, self.vals, shape)
            self.Y = self._coo(yc, yv, shape)
            roll = tuple(range(1, nd)) + (0,)
            self.inv = (nd - 1,) + tuple(range(nd - 1))
            self.Z = self._coo([yc[a] for a in roll], yv, tuple(shape[a] for a in roll))
        elif group == "bcast":
            vecs = D.vectors(self.ds, shape)
            self.bvecs = [v.reshape(_bshape(shape, a)) for a, v in enumerate(vecs)]
            self.M12 = D.dense_matrix12(self.ds, shape) if nd == 3 else None
        elif group == "ttv":
            self.vecs = D.vectors(self.ds, shape)
        elif group in {"ttm", "mttkrp"}:
            self.facs = {R: D.factors(self.ds, shape, R) for R in (16, 64)}
        elif group == "contract":
            (r, c, v, L), (c2, v2, shape2) = D.tdot_operands(self.ds, shape)
            self.Ms = sparse.COO(np.stack([r, c]), v, shape=(shape[2], L))
            self.Y2 = self._coo(c2, v2, shape2)
        elif group == "bmm":
            c2, v2, shape2 = D.load(self.ds, which=1)
            self.B = self._coo(c2, v2, shape2)

    def case(self, group, case):
        X, shape = self.X, self.shape
        if group == "construct":
            if case == "from_coo_sorted":
                return lambda: sparse.COO(self.stacked, self.vals, shape=shape)
            if case == "from_coo_shuffled":
                c, v = self.shuf
                return lambda: sparse.COO(c, v, shape=shape)
            if case == "to_coo":
                return lambda: (X.coords, X.data)
        if group == "ewise":
            if case == "add_same":
                return lambda: X + self.Y
            if case == "mul_same":
                return lambda: X * self.Y
            if case == "add_perm":
                return lambda: X + self.Z.transpose(self.inv)
            if case == "mul_perm":
                return lambda: X * self.Z.transpose(self.inv)
        if group == "bcast":
            if case == "mul_mat12":
                if self.M12 is None:
                    raise NA("dense (s1, s2) matrix too large")
                return lambda: X * self.M12
            v = self.bvecs[int(case[-1])]
            return lambda: X * v
        if group == "reduce":
            axes = parse_axes(case)
            return lambda: X.sum(axis=axes)
        if group == "permute":
            p = parse_perm(case)
            return lambda: X.transpose(p)
        if group == "ttv":
            m = int(case[3:])
            dense_guard([s for a, s in enumerate(shape) if a != m])
            v = self.vecs[m]
            return lambda: sparse.tensordot(X, v, axes=([m], [0]))
        if group == "ttm":
            m, R = int(case[3]), int(case.split("_r")[1])
            ttm_guard(self.ds, self.coords, self.shape, m, R)
            dense_guard([s for a, s in enumerate(shape) if a != m] + [R])
            U = self.facs[R][m]
            return lambda: sparse.tensordot(X, U, axes=([m], [0]))
        if group == "mttkrp":
            m = int(case[6])
            F = self.facs[16]
            others = [a for a in range(self.nd) if a != m]
            if case.endswith("_ttm"):
                # TTM over the last other mode (dense result), then a dense einsum
                last = others[-1]
                dense_guard(
                    [shape[a] for a in others[:-1]] + [shape[m], 16], "dense TTM intermediate"
                )
                sub = LETTERS[: self.nd]
                kept = "".join(sub[a] for a in range(self.nd) if a != last)
                rest = ",".join(f"{sub[a]}r" for a in others[:-1])
                esub = f"{kept}r,{rest}->{sub[m]}r"
                return lambda: np.einsum(
                    esub,
                    sparse.tensordot(X, F[last], axes=([last], [0])),
                    *[F[a] for a in others[:-1]],
                )
            # sparse.einsum materializes the nnz x R broadcast product as COO
            need = X.nnz * 16 * (8 * (self.nd + 2))
            if need > 12 * 2**30:
                raise NA(
                    f"einsum materializes nnz*R = {X.nnz * 16:.3g} COO entries "
                    f"(~{need / 2**30:.0f} GiB)"
                )
            sub = mttkrp_subscripts(self.nd, m)
            return lambda: sparse.einsum(sub, X, *[F[a] for a in others])
        if group == "contract":
            if case == "tdot1":
                return lambda: sparse.tensordot(X, self.Ms, axes=([2], [0]))
            return lambda: sparse.tensordot(X, self.Y2, axes=([1, 2], [0, 1]))
        if group == "bmm":
            return lambda: X @ self.B
        if group == "slice":
            key = D.slices(shape)[case]
            return lambda: X[key]
        raise NA(f"unknown case {group}/{case}")

    def canon(self, r):
        if isinstance(r, tuple):
            return ("coo", list(r[0]), r[1], self.shape)
        if isinstance(r, np.ndarray) or np.isscalar(r):
            return ("dense", np.asarray(r))
        if not isinstance(r, sparse.COO):
            r = r.tocoo()
        if r.ndim == 0:
            return ("dense", np.asarray(r.todense()))
        return ("coo", list(r.coords), r.data, r.shape)

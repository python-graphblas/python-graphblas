"""Dense numpy reference (small datasets only). Values are positive, so presence == nonzero."""

import datasets as D
import numpy as np
from ad_pydata import mttkrp_subscripts
from common import NA, parse_axes, parse_perm


def _dense(coords, vals, shape):
    A = np.zeros(shape)
    A[tuple(coords)] = vals
    return A


def _bshape(shape, a):
    return (shape[a],) + (1,) * (len(shape) - 1 - a)


class Adapter:
    name = "numpy"
    threads = (1,)

    def __init__(self, ds, coords, vals, shape):
        if ds not in D.SMALL:
            raise NA("dense reference only for small datasets")
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.nd = len(shape)
        self.X = _dense(coords, vals, shape)

    def info(self):
        return {"version": f"numpy {np.__version__}"}

    def set_threads(self, nt):
        pass

    def extra_cases(self, group):
        return []

    def setup(self, group):
        shape, nd = self.shape, self.nd
        if group == "ewise":
            yc, yv = D.second_operand(self.ds, self.coords, self.vals, shape)
            self.Y = _dense(yc, yv, shape)
        elif group == "bcast":
            self.bvecs = [
                v.reshape(_bshape(shape, a)) for a, v in enumerate(D.vectors(self.ds, shape))
            ]
            self.M12 = D.dense_matrix12(self.ds, shape) if nd == 3 else None
        elif group == "ttv":
            self.vecs = D.vectors(self.ds, shape)
        elif group in {"ttm", "mttkrp"}:
            self.facs = {R: D.factors(self.ds, shape, R) for R in (16, 64)}
        elif group == "contract":
            (r, c, v, L), (c2, v2, shape2) = D.tdot_operands(self.ds, shape)
            self.Ms = _dense([r, c], v, (shape[2], L))
            self.Y2 = _dense(c2, v2, shape2)
        elif group == "bmm":
            c2, v2, shape2 = D.load(self.ds, which=1)
            self.B = _dense(c2, v2, shape2)

    def case(self, group, case):
        X, shape = self.X, self.shape
        if group == "construct":
            return lambda: X
        if group == "ewise":
            if case.startswith("add"):
                return lambda: X + self.Y
            return lambda: X * self.Y
        if group == "bcast":
            if case == "mul_mat12":
                return lambda: X * self.M12
            return lambda: X * self.bvecs[int(case[-1])]
        if group == "reduce":
            axes = parse_axes(case)
            return lambda: X.sum(axis=axes)
        if group == "permute":
            return lambda: np.ascontiguousarray(X.transpose(parse_perm(case)))
        if group == "ttv":
            m = int(case[3:])
            return lambda: np.tensordot(X, self.vecs[m], axes=([m], [0]))
        if group == "ttm":
            m, R = int(case[3]), int(case.split("_r")[1])
            return lambda: np.tensordot(X, self.facs[R][m], axes=([m], [0]))
        if group == "mttkrp":
            m = int(case[6])
            F = self.facs[16]
            return lambda: np.einsum(
                mttkrp_subscripts(self.nd, m), X, *[F[a] for a in range(self.nd) if a != m]
            )
        if group == "contract":
            if case == "tdot1":
                return lambda: np.tensordot(X, self.Ms, axes=([2], [0]))
            return lambda: np.tensordot(X, self.Y2, axes=([1, 2], [0, 1]))
        if group == "bmm":
            return lambda: X @ self.B
        if group == "slice":
            key = D.slices(shape)[case]
            return lambda: X[key]
        raise NA(f"unknown case {group}/{case}")

    def canon(self, r):
        return ("dense", np.asarray(r))

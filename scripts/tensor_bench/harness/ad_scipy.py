"""scipy.sparse n-d coo_array (scipy 1.18), single-threaded.

scipy keeps COO non-canonical after +, transpose and construction (duplicates and order are
resolved lazily). The main cases canonicalize with ``sum_duplicates()`` where the result can
hold duplicates or is unsorted (construction, +, permutation); the ``*_raw`` extra cases time
the same call without it.
"""

import datasets as D
import numpy as np
import scipy
import scipy.sparse as ss
from common import NA, dense_guard, parse_axes, parse_perm, ttm_guard

LETTERS = "ijklmn"


def _bshape(shape, a):
    return (shape[a],) + (1,) * (len(shape) - 1 - a)


def _canon(A):
    A.sum_duplicates()
    return A


class Adapter:
    name = "scipy"
    threads = (1,)

    def __init__(self, ds, coords, vals, shape):
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.nd = len(shape)
        self.X = self._coo(coords, vals, shape, canonical=True)

    @staticmethod
    def _coo(coords, vals, shape, canonical=False):
        A = ss.coo_array((vals, tuple(coords)), shape=tuple(shape))
        if canonical:
            A.has_canonical_format = True
        else:
            A.sum_duplicates()
        return A

    def info(self):
        return {"version": f"scipy {scipy.__version__}"}

    def set_threads(self, nt):
        pass

    def extra_cases(self, group):
        if group == "construct":
            return ["from_coo_sorted_raw", "from_coo_shuffled_raw"]
        if group == "ewise":
            return ["add_same_raw", "add_perm_raw"]
        if group == "permute":
            return (
                ["perm_" + "".join(map(str, p)) + "_raw" for p in [(2, 0, 1)]]
                if self.nd == 3
                else []
            )
        return []

    def setup(self, group):
        shape, nd = self.shape, self.nd
        if group == "construct":
            order = np.random.default_rng(99).permutation(self.vals.size)
            self.shuf = [c[order] for c in self.coords], self.vals[order]
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
            self.Ms = ss.coo_array((v, (r, c)), shape=(shape[2], L))
            self.Y2 = self._coo(c2, v2, shape2)
        elif group == "bmm":
            c2, v2, shape2 = D.load(self.ds, which=1)
            self.B = self._coo(c2, v2, shape2)

    def case(self, group, case):
        X, shape = self.X, self.shape
        if group == "construct":
            if case.startswith("from_coo"):
                c, v = (self.coords, self.vals) if "sorted" in case else self.shuf
                if case.endswith("_raw"):
                    return lambda: ss.coo_array((v, tuple(c)), shape=shape)
                return lambda: _canon(ss.coo_array((v, tuple(c)), shape=shape))
            if case == "to_coo":
                return lambda: (X.coords, X.data)
        if group == "ewise":
            if case.startswith("add"):
                if case.endswith("_raw"):
                    return lambda: X + (self.Y if "same" in case else self.Z.transpose(self.inv))
                if "same" in case:
                    return lambda: _canon(X + self.Y)
                return lambda: _canon(X + self.Z.transpose(self.inv))
            # multiply goes through 2-d CSR, so its result has no duplicates and is in order
            if "same" in case:
                return lambda: X.multiply(self.Y)
            return lambda: X.multiply(self.Z.transpose(self.inv))
        if group == "bcast":
            # n-d multiply does not broadcast ("inconsistent shapes: >2D multiply() does not
            # yet support broadcasting"); the one-line workaround is np.broadcast_to
            if case == "mul_mat12":
                if self.M12 is None:
                    raise NA("dense (s1, s2) matrix too large")
                return lambda: X.multiply(np.broadcast_to(self.M12, shape))
            v = self.bvecs[int(case[-1])]
            return lambda: X.multiply(np.broadcast_to(v, shape))
        if group == "reduce":
            axes = parse_axes(case)
            if axes is not None:
                dense_guard([s for a, s in enumerate(shape) if a not in axes], "dense sum output")
                # _sum_nd reshapes to 2-d and multiplies by a dense ones vector over the
                # reduced axes
                dense_guard(
                    [s for a, s in enumerate(shape) if a in axes],
                    "dense ones vector over the reduced axes",
                )
            return lambda: X.sum(axis=axes)
        if group == "permute":
            p = parse_perm(case)
            if case.endswith("_raw"):
                return lambda: X.transpose(p)
            return lambda: _canon(X.transpose(p))
        if group == "ttv":
            m = int(case[3:])
            dense_guard([s for a, s in enumerate(shape) if a != m])
            v = self.vecs[m]
            return lambda: X.tensordot(v, axes=([m], [0]))
        if group == "ttm":
            m, R = int(case[3]), int(case.split("_r")[1])
            ttm_guard(self.ds, self.coords, self.shape, m, R)
            dense_guard([s for a, s in enumerate(shape) if a != m] + [R])
            U = self.facs[R][m]
            return lambda: X.tensordot(U, axes=([m], [0]))
        if group == "mttkrp":
            # no einsum in scipy: TTM over the last other mode (dense), then a dense einsum
            m = int(case[6])
            F = self.facs[16]
            others = [a for a in range(self.nd) if a != m]
            last = others[-1]
            dense_guard([shape[a] for a in others[:-1]] + [shape[m], 16], "dense TTM intermediate")
            sub = LETTERS[: self.nd]
            kept = "".join(sub[a] for a in range(self.nd) if a != last)
            rest = ",".join(f"{sub[a]}r" for a in others[:-1])
            esub = f"{kept}r,{rest}->{sub[m]}r"
            return lambda: np.einsum(
                esub, X.tensordot(F[last], axes=([last], [0])), *[F[a] for a in others[:-1]]
            )
        if group == "contract":
            if case == "tdot1":
                return lambda: X.tensordot(self.Ms, axes=([2], [0]))
            return lambda: X.tensordot(self.Y2, axes=([1, 2], [0, 1]))
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
        r = r.copy()
        r.sum_duplicates()
        return ("coo", list(r.coords), r.data, r.shape)

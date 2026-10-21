"""python-graphblas Tensor (SuiteSparse:GraphBLAS 10 or later).

The SuiteSparse JIT is configured with ``fix_jit_config()`` (the compiler of the active conda
environment), as the maintainers benchmark it; the machine record says whether it worked.
``TENSOR_BENCH_REQUIRE_JIT=1`` refuses to run unless the JIT is on.
``TENSOR_BENCH_GB_ENGINE`` (``numba``, ``udt`` or ``products``) forces one engine for TTM and
MTTKRP, as run.py's ``graphblas-numba``, ``graphblas-udt`` and ``graphblas-products`` do.
"""

import os

import datasets as D
import numpy as np
from common import NA, parse_axes, parse_perm, ttm_guard

import graphblas as gb
import graphblas.core.tensor as _tensor
from graphblas import Matrix, Vector
from graphblas.core import tensor_layout as tl
from graphblas.core.ss import jit_config
from graphblas.core.tensor import Tensor

jit_config.fix_jit_config()
if os.environ.get("TENSOR_BENCH_REQUIRE_JIT") and gb.ss.config["jit_c_control"] != "on":
    raise SystemExit(f"the SuiteSparse JIT is {gb.ss.config['jit_c_control']!r}, not 'on'")
ENGINE = os.environ.get("TENSOR_BENCH_GB_ENGINE")
if ENGINE:
    _tensor._TTM_ENGINES = (ENGINE,)
    _tensor._MTTKRP_ENGINES = (ENGINE,) if ENGINE != "udt" else ("udt", "products")


def _done(T):
    T.wait()
    return T


def _bshape(shape, a):
    """Shape of a vector along axis ``a`` for numpy broadcasting against ``shape``."""
    return (shape[a],) + (1,) * (len(shape) - 1 - a)


class Adapter:
    name = "graphblas"
    threads = (8, 1)
    LAYOUT = None  # the Tensor's default layout

    def __init__(self, ds, coords, vals, shape):
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.nd = len(shape)
        self.X = _done(Tensor.from_coo(coords, vals, shape=shape, layout=self.LAYOUT))

    def info(self):
        return {
            "version": f"python-graphblas {getattr(gb, '__version__', '?')}, "
            f"SuiteSparse:GraphBLAS {'.'.join(map(str, gb.ss.about['library_version']))}",
            "graphblas_path": os.path.dirname(gb.__file__),
            "layout": self.X.layout._short(),
            "format": self.X.ss.format,
            "jit": gb.ss.config["jit_c_control"],
            "jit_cc": gb.ss.config["jit_c_compiler_name"],
            "engine": ENGINE or "default",
        }

    def set_threads(self, nt):
        gb.ss.config["nthreads"] = nt

    def extra_cases(self, group):
        if group == "mttkrp":
            return [f"mttkrp{m}_r64" for m in range(self.nd)]
        return []

    def setup(self, group):
        shape, nd = self.shape, self.nd
        if group == "construct":
            order = np.random.default_rng(99).permutation(self.vals.size)
            self.shuf = [c[order] for c in self.coords], self.vals[order]
        elif group == "ewise":
            yc, yv = D.second_operand(self.ds, self.coords, self.vals, shape)
            self.Y = _done(Tensor.from_coo(yc, yv, shape=shape, layout=self.LAYOUT))
            # Z holds Y with the axes rolled (Z[j,k,i] = Y[i,j,k]); Y == Z.permute_dims(inv)
            roll = tuple(range(1, nd)) + (0,)
            inv = (nd - 1,) + tuple(range(nd - 1))
            Z = _done(Tensor.from_coo([yc[a] for a in roll], yv, shape=[shape[a] for a in roll]))
            self.Yp = Z.permute_dims(inv)
        elif group == "bcast":
            vecs = D.vectors(self.ds, shape)
            self.bvecs = [
                Tensor.from_dense(v.reshape(_bshape(shape, a))) for a, v in enumerate(vecs)
            ]
            M12 = D.dense_matrix12(self.ds, shape) if nd == 3 else None
            self.M12 = None if M12 is None else Tensor.from_dense(M12)
        elif group == "ttv":
            self.vecs = [Vector.from_dense(v) for v in D.vectors(self.ds, shape)]
        elif group in {"ttm", "mttkrp"}:
            self.facs = {
                R: [Matrix.from_dense(F) for F in D.factors(self.ds, shape, R)] for R in (16, 64)
            }
        elif group == "contract":
            (r, c, v, L), (c2, v2, shape2) = D.tdot_operands(self.ds, shape)
            self.Ms = Matrix.from_coo(r, c, v, nrows=shape[2], ncols=L)
            self.Y2 = _done(Tensor.from_coo(c2, v2, shape=shape2))
        elif group == "bmm":
            c2, v2, shape2 = D.load(self.ds, which=1)
            self.B = _done(Tensor.from_coo(c2, v2, shape=shape2))

    def case(self, group, case):
        X = self.X
        if group == "construct":
            if case == "from_coo_sorted":
                return lambda: _done(
                    Tensor.from_coo(self.coords, self.vals, shape=self.shape, layout=self.LAYOUT)
                )
            if case == "from_coo_shuffled":
                c, v = self.shuf
                return lambda: _done(Tensor.from_coo(c, v, shape=self.shape, layout=self.LAYOUT))
            if case == "to_coo":
                return lambda: X.to_coo()
        if group == "ewise":
            other = self.Y if case.endswith("same") else self.Yp
            if case.startswith("add"):
                return lambda: _done((X + other).new())
            return lambda: _done((X * other).new())
        if group == "bcast":
            if case == "mul_mat12":
                if self.M12 is None:
                    raise NA("dense (s1, s2) matrix too large")
                return lambda: _done((X * self.M12).new())
            v = self.bvecs[int(case[-1])]
            return lambda: _done((X * v).new())
        if group == "reduce":
            axes = parse_axes(case)
            return lambda: _done(X.sum(axis=axes).new())
        if group == "permute":
            p = parse_perm(case)
            Xv = X.permute_dims(p)
            target = tl.default_layout(Xv.shape, X.nvals)
            return lambda: _done(Xv.relayout(target))
        if group == "ttv":
            m = int(case[3:])
            v = self.vecs[m]
            return lambda: _done(X.tensordot(v, ([m], [0])).new())
        if group == "ttm":
            m, R = int(case[3]), int(case.split("_r")[1])
            ttm_guard(self.ds, self.coords, self.shape, m, R)
            U = self.facs[R][m]
            return lambda: _done(X.tensordot(U, ([m], [0])).new())
        if group == "mttkrp":
            m, R = int(case[6]), int(case.split("_r")[1])
            F = self.facs[R]
            return lambda: _done(X.mttkrp(F, m).new())
        if group == "contract":
            if case == "tdot1":
                return lambda: _done(X.tensordot(self.Ms, ([2], [0])).new())
            return lambda: _done(X.tensordot(self.Y2, ([1, 2], [0, 1])).new())
        if group == "bmm":
            return lambda: _done((X @ self.B).new())
        if group == "slice":
            key = D.slices(self.shape)[case]
            return lambda: _done(X[key].new())
        raise NA(f"unknown case {group}/{case}")

    def note(self, group, case):
        if group == "permute":
            return f"result layout {self._last_layout} (identity order = materialized C order)"
        return ""

    def canon(self, r):
        if isinstance(r, tuple):
            coords, vals = r
            return ("coo", coords, vals, self.shape)
        self._last_layout = r.layout._short()
        coords, vals = r.to_coo()
        return ("coo", coords, vals, r.shape)

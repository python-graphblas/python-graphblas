"""Multilayer PageRank (versatility) on a layer x src x dst Tensor.

The model is the standard multilayer PageRank of De Domenico et al. (2015, Nat. Commun. 6:6868,
"Ranking in interconnected multilayer networks reveals versatile nodes"), as muxViz computes it:
ordinary PageRank on the supra-graph whose vertices are the (layer, node) pairs, with the intra-
layer edges A(l, u, v) and an inter-layer edge of weight `omega` between each node's copies in
every two layers (a multiplex: the layer coupling C is all ones off the diagonal). A walker at
(l, u) follows an edge with probability proportional to its weight; with probability 1 - d it
teleports to a uniformly random (layer, node); a copy with no out-weight teleports. A node's
versatility is the sum of its copies' PageRank over the layers.

On the Tensor, one step is (x = r / s, s the copies' out-weights, a (layer, node) Tensor):

    intra = einsum("lu,luv->lv", x, A)       # contract over nodes inside each layer
    inter = einsum("lu,lm->mu", x, omega C)  # contract over layers at each node
                                             # (for a multiplex: omega * (x.sum(0) - x))
    r = d * (intra + inter) + (1 - d + d * dangling) / (L n)

Baselines: the per-layer loop over a list of gb.Matrix (one vxm per layer, the layer coupling
as a loop over layer pairs), a dense numpy power iteration (EU airlines only), a scipy
power iteration on the sparse supra-adjacency matrix, and, for exactness, scipy's sparse direct
solve of the PageRank linear system on the supra-graph.

    python fetch_data.py eu-airlines higgs
    python demos/multiplex_pagerank.py eu [--reps 5]
    python demos/multiplex_pagerank.py higgs [--reps 3] [--threads 1,8]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import graphblas as gb
from graphblas import Matrix, Tensor, Vector, binary, einsum, monoid
from graphblas.core.ss import jit_config

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

# isort: on


jit_config.fix_jit_config()

DAMP, TOL, ITERMAX = 0.85, 1e-12, 1000
OMEGA = 1.0  # the inter-layer edge weight (--omega)
GENERAL = False  # --general-coupling: no multiplex shortcut, the coupling as a contraction


def load_eu():
    """(layer, src, dst) arrays of the 37-layer EU airline multiplex, both directions."""
    path = paths.data_dir() / "eu_airlines" / "edges.csv"
    e = np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64)
    s, t, layer = e[:, 0], e[:, 1], e[:, 3] - 1  # layers are 1-based in the file
    n = int(max(s.max(), t.max())) + 1
    L = int(layer.max()) + 1
    return np.concatenate([layer, layer]), np.concatenate([s, t]), np.concatenate([t, s]), L, n


def load_higgs():
    """(layer, src, dst) of the 4-layer Higgs Twitter multiplex as given (directed)."""
    import pandas as pd

    names = ["social", "retweet", "reply", "mention"]
    parts = []
    for i, name in enumerate(names):
        path = paths.data_dir() / "higgs_twitter" / f"higgs-{name}_network.edgelist.gz"
        df = pd.read_csv(path, sep=" ", header=None, usecols=[0, 1], dtype=np.int64)
        parts.append((np.full(len(df), i), df[0].to_numpy() - 1, df[1].to_numpy() - 1))
    layer, s, t = (np.concatenate(x) for x in zip(*parts, strict=True))
    return layer, s, t, len(names), 456626


def coupling(L):
    """Omega * C for a multiplex: every two layers coupled, no self-coupling."""
    return OMEGA * (np.ones((L, L)) - np.eye(L))


# ---------------------------------------------------------------- the Tensor
def tensor_pagerank(A, K, *, tol=TOL, itermax=ITERMAX):
    """A: (L, n, n) Tensor of edge weights; K: (L, L) ndarray of coupling weights."""
    L, n, _ = A.shape
    N = L * n
    Kt = Tensor.from_dense(K, missing_value=0.0)
    s = (
        A.sum(axis=2)
        .new()
        .ewise_add(  # each copy's out-weight: its edges, then its couplings
            Tensor.from_dense(np.broadcast_to(K.sum(axis=1)[:, None], (L, n)), missing_value=0.0),
            binary.plus,
        )
        .new()
    )
    inv_s = s.apply(gb.unary.minv).new()
    dangling = N - s.nvals  # copies with no out-weight (none when omega > 0 and L > 1)
    # A multiplex (K = omega off the diagonal) couples a copy to the sum of the others: omega *
    # (sum over layers - itself), as the loop does; any other K takes the contraction
    multiplex = not GENERAL and L > 1 and np.allclose(K, K[0, 1] * (1 - np.eye(L)))
    r = Tensor.from_dense(np.full((L, n), 1.0 / N))
    for it in range(1, itermax + 1):
        x = (r * inv_s).new()
        intra = einsum("lu,luv->lv", x, A).new()  # over nodes, inside each layer
        if multiplex:  # over layers, at each node
            total = x.sum(axis=0, keepdims=True).new()
            inter = total.ewise_mult(x, binary.minus).new().apply(binary.times, K[0, 1]).new()
        else:
            inter = einsum("lu,lm->mu", x, Kt).new()
        dmass = 0.0
        if dangling:
            dmass = (
                r.reduce_scalar().new().value
                - r.ewise_mult(inv_s, binary.first).new().sum().new().to_scalar().value
            )
        c = (1 - DAMP + DAMP * dmass) / N
        new = intra.ewise_add(inter, binary.plus).new().apply(binary.times, DAMP).new()
        new = new.apply(binary.plus, c).new()
        if new.nvals != N:  # a copy nothing reaches gets only the teleport share
            full = Tensor.from_dense(np.full((L, n), c))
            new = full.ewise_add(new, binary.second).new()
        err = new.ewise_add(r, binary.minus).new().apply(gb.unary.abs).new().sum().new()
        err = err.to_scalar().value
        r = new
        if err < tol:
            return r, it
    return r, itermax


# ------------------------------------------------------------- per-layer loop
def loop_pagerank(As, K, *, tol=TOL, itermax=ITERMAX):
    """As: list of L (n x n) Matrices (the per-label list LAGraph-style code keeps)."""
    L = len(As)
    n = As[0].nrows
    N = L * n
    inv_s = []
    dangling = 0
    for la, A in enumerate(As):
        s = A.reduce_rowwise(monoid.plus).new()
        s = (
            Vector.from_dense(np.full(n, K[la].sum())).ewise_add(s, binary.plus).new()
            if K[la].sum()
            else s
        )
        dangling += n - s.nvals
        inv_s.append(s.apply(gb.unary.minv).new())
    r = [Vector.from_dense(np.full(n, 1.0 / N)) for _ in range(L)]
    for it in range(1, itermax + 1):
        x = [(r[la] * inv_s[la]).new() for la in range(L)]
        dmass = 0.0
        if dangling:
            dmass = sum(
                r[la].reduce().value
                - r[la].ewise_mult(inv_s[la], binary.first).new().reduce().value
                for la in range(L)
            )
        c = (1 - DAMP + DAMP * dmass) / N
        # The coupling: inter[m] = sum over l of K[l, m] x[l]. For a multiplex (K = omega off
        # the diagonal) that is omega * (sum of all x - x[m]): L - 1 additions, not L^2
        multiplex = not GENERAL and L > 1 and np.allclose(K, K[0, 1] * (1 - np.eye(L)))
        if multiplex:
            total = x[0].dup()
            for la in range(1, L):
                total(binary.plus) << x[la]
        new = []
        for m in range(L):
            acc = x[m].vxm(As[m]).new()  # intra-layer
            if multiplex:
                acc(binary.plus) << total.ewise_add(x[m], binary.minus).new().apply(
                    binary.times, K[0, 1]
                )
            else:
                for la in range(L):
                    if K[la, m]:
                        acc(binary.plus) << x[la].apply(binary.times, K[la, m])
            acc = acc.apply(binary.times, DAMP).new()
            full = Vector.from_dense(np.full(n, c))
            full(binary.plus) << acc
            new.append(full)
        err = sum(
            new[la].ewise_add(r[la], binary.minus).new().apply(gb.unary.abs).new().reduce().value
            for la in range(L)
        )
        r = new
        if err < tol:
            return r, it
    return r, itermax


# ------------------------------------------------------------------- numpy
def supra_matrix(layer, s, t, w, K, L, n):
    """The supra-graph's transition matrix P (scipy CSR), P[(l,u), (m,v)] = weight / out-weight."""
    rows = [layer * n + s]
    cols = [layer * n + t]
    vals = [w]
    for la in range(L):
        for m in range(L):
            if K[la, m]:
                rows.append(la * n + np.arange(n))
                cols.append(m * n + np.arange(n))
                vals.append(np.full(n, K[la, m]))
    S = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(L * n, L * n)
    )
    out = np.asarray(S.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    return sp.diags(inv) @ S, out == 0


def scipy_pagerank(P, dangling, *, tol=TOL, itermax=ITERMAX):
    N = P.shape[0]
    PT = P.T.tocsr()
    r = np.full(N, 1.0 / N)
    for it in range(1, itermax + 1):
        new = DAMP * (PT @ r) + (1 - DAMP + DAMP * r[dangling].sum()) / N
        err = np.abs(new - r).sum()
        r = new
        if err < tol:
            return r, it
    return r, itermax


def dense_pagerank(Ad, K, *, tol=TOL, itermax=ITERMAX):
    """Power iteration in numpy on the dense (L, n, n) weights: the same two contractions."""
    L, n, _ = Ad.shape
    N = L * n
    s = Ad.sum(axis=2) + K.sum(axis=1)[:, None]
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
    r = np.full((L, n), 1.0 / N)
    for it in range(1, itermax + 1):
        x = r * inv_s
        new = DAMP * (np.einsum("lu,luv->lv", x, Ad) + np.einsum("lu,lm->mu", x, K))
        new += (1 - DAMP + DAMP * r[s == 0].sum()) / N
        err = np.abs(new - r).sum()
        r = new
        if err < tol:
            return r, it
    return r, itermax


def exact(P, dangling):
    """Solve r = d P^T r + (1 - d + d * dangling . r) / N with sum(r) = 1, directly."""
    N = P.shape[0]
    if dangling.any():
        # (I - d P^T - d/N 1 dangling^T) r = (1 - d)/N 1; dense rank-one term: small N only
        Mx = (
            sp.identity(N) - DAMP * P.T - (DAMP / N) * sp.csr_matrix(np.outer(np.ones(N), dangling))
        )
    else:
        Mx = sp.identity(N, format="csc") - DAMP * P.T.tocsc()
    r = spla.spsolve(Mx.tocsc(), np.full(N, (1 - DAMP) / N))
    return r / r.sum()


def best(f, reps):
    times, rv = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        rv = f()
        times.append(time.perf_counter() - t0)
    return min(times), rv


def dense_of(r, L, n):
    if isinstance(r, Tensor):
        return r.to_dense(fill_value=0.0)
    return np.stack([v.to_dense(fill_value=0.0) for v in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", choices=["eu", "higgs"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--json")
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--general-coupling", action="store_true")
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ../data)"
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    global OMEGA, GENERAL
    OMEGA, GENERAL = args.omega, args.general_coupling
    t0 = time.perf_counter()
    layer, s, t, L, n = load_eu() if args.data == "eu" else load_higgs()
    w = np.ones(layer.size)
    # Repeated (layer, src, dst) edges count once: structural edges
    key = np.unique((layer * n + s) * n + t)
    layer, rest = np.divmod(key, n * n)
    s, t = np.divmod(rest, n)
    w = np.ones(key.size)
    print(
        f"loaded {args.data} in {time.perf_counter() - t0:.1f} s: {L} layers, {n} nodes, "
        f"{key.size} directed edges",
        flush=True,
    )
    K = coupling(L)
    A = Tensor.from_coo((layer, s, t), w, shape=(L, n, n))
    As = [Matrix.from_coo(s[layer == la], t[layer == la], 1.0, nrows=n, ncols=n) for la in range(L)]
    P, dangling = supra_matrix(layer, s, t, w, K, L, n)
    results = {
        "data": args.data,
        "L": L,
        "n": n,
        "edges": int(key.size),
        "omega": OMEGA,
        "general_coupling": GENERAL,
        "load1_start": os.getloadavg()[0],
    }

    rs, its = scipy_pagerank(P, dangling)
    ref = rs.reshape(L, n)
    if args.data == "eu":
        ex = exact(P, dangling).reshape(L, n)
        print(f"exact solve vs scipy power iteration: max |diff| {np.abs(ex - ref).max():.2e}")
        results["exact_vs_scipy_maxdiff"] = float(np.abs(ex - ref).max())
        ref = ex
        rd, itd = dense_pagerank(A.to_dense(fill_value=0.0), K)
        print(f"dense numpy: {itd} iterations, max |diff| vs exact {np.abs(rd - ref).max():.2e}")
        results["dense_vs_exact_maxdiff"] = float(np.abs(rd - ref).max())

    for nt in [int(x) for x in args.threads.split(",")]:
        gb.ss.config["nthreads"] = nt
        tt, (rt, itt) = best(lambda: tensor_pagerank(A, K), args.reps)
        tl_, (rl, itl) = best(lambda: loop_pagerank(As, K), args.reps)
        dt, dl = dense_of(rt, L, n), dense_of(rl, L, n)
        print(
            f"{nt} threads: Tensor {tt * 1e3:.1f} ms ({itt} iterations), "
            f"per-layer Matrix loop {tl_ * 1e3:.1f} ms ({itl}); "
            f"max |Tensor - loop| {np.abs(dt - dl).max():.2e}, "
            f"max |Tensor - reference| {np.abs(dt - ref).max():.2e}",
            flush=True,
        )
        results[f"tensor_ms_{nt}t"] = round(tt * 1e3, 2)
        results[f"loop_ms_{nt}t"] = round(tl_ * 1e3, 2)
        results[f"iterations_{nt}t"] = (itt, itl)
        results[f"tensor_vs_loop_maxdiff_{nt}t"] = float(np.abs(dt - dl).max())
        results[f"tensor_vs_ref_maxdiff_{nt}t"] = float(np.abs(dt - ref).max())
        # The baselines: the per-layer loop (same algorithm, so equal to rounding) and the
        # reference (exact solve for EU airlines, scipy's power iteration for Higgs)
        assert np.abs(dt - dl).max() <= 1e-14, results
        assert np.abs(dt - ref).max() <= 1e-10, results
    tsp, (rs, its) = best(lambda: scipy_pagerank(P, dangling), args.reps)
    print(f"scipy supra-matrix power iteration (1 thread): {tsp * 1e3:.1f} ms ({its} iterations)")
    results["scipy_ms"] = round(tsp * 1e3, 2)
    if args.data == "eu":
        td, _ = best(lambda: dense_pagerank(A.to_dense(fill_value=0.0), K), args.reps)
        print(f"dense numpy power iteration: {td * 1e3:.1f} ms")
        results["dense_ms"] = round(td * 1e3, 2)
    vers = dense_of(rt, L, n).sum(axis=0)
    top = np.argsort(-vers)[:10]
    names = [str(i) for i in range(n)]
    if args.data == "eu":  # ICAO airport codes
        with open(paths.data_dir() / "eu_airlines" / "nodes.csv") as f:
            names = [line.split(",")[1] for line in f if not line.startswith("#")]
    results["top10_versatility"] = [(names[i], float(vers[i])) for i in top]
    print(
        "top 10 versatile nodes (node, versatility):",
        ", ".join(f"{names[i]} {vers[i]:.5f}" for i in top),
    )
    results["load1_end"] = os.getloadavg()[0]
    print(json.dumps(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

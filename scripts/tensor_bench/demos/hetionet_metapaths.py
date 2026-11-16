"""All two-hop Compound-Gene-Disease metapaths of Hetionet v1.0 in one einsum.

The Tensor side: three relation stacks (predicate x src x dst) and two einsums give the path
counts of all 9 metapaths CxGxD (x in {binds, upregulates, downregulates} for the
compound-gene edge, {associates, upregulates, downregulates} for the gene-disease edge) and,
for each metapath and each known compound-disease relation (treats, palliates), how many known
pairs the metapath reaches: the metapath-counts overlap statistic. A degree-weighted variant
(DWPC, Hetionet's feature, w = 0.4) scales the stacks by deg ** -w along each axis first.

The baseline: the per-predicate loop over Matrices that cbizon/metapath-counts runs (a dict of
(src_type, predicate, dst_type) -> gb.Matrix, one mxm per metapath, one intersection per target
relation), and the same with scipy (hetmatpy's stack). Results must be identical (counts) or
equal to 1e-12 (DWPC).

    python fetch_data.py hetionet
    python demos/hetionet_metapaths.py [--reps 5] [--threads 1,8]
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import graphblas as gb
from graphblas import Matrix, Tensor, binary, einsum, monoid, semiring
from graphblas.core.ss import jit_config

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

# isort: on


jit_config.fix_jit_config()

# Metaedge abbreviations in the SIF file; G<-D edges are stored Disease -> Gene there
CG_PREDS = ["CbG", "CuG", "CdG"]  # compound binds / upregulates / downregulates gene
DG_PREDS = ["DaG", "DuG", "DdG"]  # disease associates / upregulates / downregulates gene
CD_PREDS = ["CtD", "CpD"]  # compound treats / palliates disease
GG_PREDS = ["GiG", "GcG", "Gr>G"]  # gene interacts / covaries / regulates gene (larger case)
W = 0.4  # Hetionet's DWPC damping exponent


def load():
    index = {}
    with open(paths.data_dir() / "hetionet" / "hetionet-v1.0-nodes.tsv") as f:
        next(f)
        for line in f:
            node_id, _name, kind = line.rstrip("\n").split("\t")
            kinds = index.setdefault(kind, {})
            kinds[node_id] = len(kinds)
    wanted = set(CG_PREDS + DG_PREDS + CD_PREDS + GG_PREDS)
    edges = {p: ([], []) for p in wanted}
    with gzip.open(paths.data_dir() / "hetionet" / "hetionet-v1.0-edges.sif.gz", "rt") as f:
        next(f)
        for line in f:
            src, metaedge, dst = line.rstrip("\n").split("\t")
            if metaedge in wanted:
                edges[metaedge][0].append(src)
                edges[metaedge][1].append(dst)
    C, G, D = index["Compound"], index["Gene"], index["Disease"]
    out = {}
    for p in CG_PREDS:
        out[p] = (np.array([C[s] for s in edges[p][0]]), np.array([G[t] for t in edges[p][1]]))
    for p in DG_PREDS:  # stored D -> G; we want G -> D
        out[p] = (np.array([G[t] for t in edges[p][1]]), np.array([D[s] for s in edges[p][0]]))
    for p in CD_PREDS:
        out[p] = (np.array([C[s] for s in edges[p][0]]), np.array([D[t] for t in edges[p][1]]))
    for p in GG_PREDS:
        out[p] = (np.array([G[s] for s in edges[p][0]]), np.array([G[t] for t in edges[p][1]]))
    return out, (len(C), len(G), len(D))


def stack(edges, preds, shape):
    p = np.concatenate([np.full(edges[q][0].size, i) for i, q in enumerate(preds)])
    s = np.concatenate([edges[q][0] for q in preds])
    t = np.concatenate([edges[q][1] for q in preds])
    return Tensor.from_coo((p, s, t), 1.0, shape=(len(preds), *shape))


def scaled(T):
    """T[p, i, j] * (deg_p(i) * deg_p(j)) ** -W, degrees within each predicate p."""
    d_src = T.sum(axis=2).new().apply(binary.pow, -W).new()  # (p, i)
    d_dst = T.sum(axis=1).new().apply(binary.pow, -W).new()  # (p, j)
    return (T * d_src.expand_dims(2) * d_dst.expand_dims(1)).new()


def tensor_side(CG, GD, CD):
    P = einsum("pcg,qgd->pqcd", CG, GD, op=semiring.plus_pair).new()
    overlap = einsum("pqcd,rcd->pqr", P, CD, op=semiring.plus_pair).new()
    DWPC = einsum("pcg,qgd->pqcd", scaled(CG), scaled(GD)).new()
    return P, overlap, DWPC


def matrix_side(mats):
    """The per-predicate loop: one mxm per metapath, one intersection per target relation."""
    P, overlap, DWPC = {}, {}, {}
    w = {}
    for name, A in mats.items():
        if name in CD_PREDS:
            continue
        r = A.reduce_rowwise(monoid.plus).new().apply(binary.pow, -W).new()
        c = A.reduce_columnwise(monoid.plus).new().apply(binary.pow, -W).new()
        w[name] = (
            r.diag().mxm(A, semiring.plus_times).new().mxm(c.diag(), semiring.plus_times).new()
        )
    for p in CG_PREDS:
        for q in DG_PREDS:
            P[p, q] = mats[p].mxm(mats[q], semiring.plus_pair).new()
            DWPC[p, q] = w[p].mxm(w[q], semiring.plus_times).new()
            for r in CD_PREDS:
                overlap[p, q, r] = P[p, q].ewise_mult(mats[r], binary.pair).new().nvals
    return P, overlap, DWPC


def cgg_tensor(CG, GG):
    return einsum("pcg,qgh->pqch", CG, GG, op=semiring.plus_pair).new()


def cgg_loop(mats):
    return {
        (p, q): mats[p].mxm(mats[q], semiring.plus_pair).new() for p in CG_PREDS for q in GG_PREDS
    }


def scipy_side(smats):
    P = {}
    overlap = {}
    for p in CG_PREDS:
        for q in DG_PREDS:
            P[p, q] = (smats[p] @ smats[q]).tocsr()
            for r in CD_PREDS:
                overlap[p, q, r] = int(P[p, q].multiply(smats[r]).count_nonzero())
    return P, overlap


def best(f, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rv = f()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ../data)"
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    t0 = time.perf_counter()
    edges, (nC, nG, nD) = load()
    print(
        f"loaded Hetionet in {time.perf_counter() - t0:.1f} s: {nC} compounds, {nG} genes, "
        f"{nD} diseases; edges " + ", ".join(f"{p} {edges[p][0].size}" for p in edges),
        flush=True,
    )
    CG = stack(edges, CG_PREDS, (nC, nG))
    GD = stack(edges, DG_PREDS, (nG, nD))
    CD = stack(edges, CD_PREDS, (nC, nD))
    GG = stack(edges, GG_PREDS, (nG, nG))
    shapes = {
        **dict.fromkeys(CG_PREDS, (nC, nG)),
        **dict.fromkeys(DG_PREDS, (nG, nD)),
        **dict.fromkeys(CD_PREDS, (nC, nD)),
        **dict.fromkeys(GG_PREDS, (nG, nG)),
    }
    mats = {
        p: Matrix.from_coo(*edges[p], 1.0, nrows=shapes[p][0], ncols=shapes[p][1]) for p in edges
    }
    smats = {p: sp.csr_array((np.ones(edges[p][0].size), edges[p]), shape=shapes[p]) for p in edges}

    # Correctness: every metapath's counts, DWPC and overlaps against the loop and scipy
    P, overlap, DWPC = tensor_side(CG, GD, CD)
    lP, lover, lDWPC = matrix_side(mats)
    sP, sover = scipy_side(smats)
    over = overlap.to_dense(fill_value=0)
    n_paths = 0
    for i, p in enumerate(CG_PREDS):
        for j, q in enumerate(DG_PREDS):
            got = P[i, j, :, :].new().to_matrix()  # (an INT64 count Matrix, C x D)
            assert got.isequal(lP[p, q]), (p, q)
            assert got.isequal(gb.io.from_scipy_sparse(sP[p, q]).dup(got.dtype)), (p, q)
            gw = DWPC[i, j, :, :].new().to_matrix()
            assert gw.isclose(lDWPC[p, q], rel_tol=1e-12), (p, q)
            n_paths += got.nvals
            for k, r in enumerate(CD_PREDS):
                assert over[i, j, k] == lover[p, q, r] == sover[p, q, r], (p, q, r)
    print(
        f"checked: 9 metapaths, {n_paths} (compound, disease) pairs with a path, "
        f"{overlap.nvals} overlap counts, DWPC to 1e-12: tensor == Matrix loop == scipy",
        flush=True,
    )
    print("overlap (metapath x target relation):", flush=True)
    for i, p in enumerate(CG_PREDS):
        for j, q in enumerate(DG_PREDS):
            print(
                f"  C-{p[1]}-G-{q[1]}-D: "
                + ", ".join(f"{r} {int(over[i, j, k])}" for k, r in enumerate(CD_PREDS)),
                flush=True,
            )

    # The larger family: Compound-x-Gene-y-Gene, 9 metapaths over 20945 genes
    Q = cgg_tensor(CG, GG)
    lQ = cgg_loop(mats)
    for i, p in enumerate(CG_PREDS):
        for j, q in enumerate(GG_PREDS):
            assert Q[i, j, :, :].new().to_matrix().isequal(lQ[p, q]), (p, q)
    print(
        f"checked: CxGyG, 9 metapaths, {Q.nvals} (compound, gene) path counts: tensor == "
        f"Matrix loop",
        flush=True,
    )

    rows = []
    for nt in [int(x) for x in args.threads.split(",")]:
        gb.ss.config["nthreads"] = nt
        tensor_side(CG, GD, CD)  # (JIT kernels loaded)
        t_tensor, _ = best(lambda: tensor_side(CG, GD, CD), args.reps)
        t_loop, _ = best(lambda: matrix_side(mats), args.reps)

        def counts():
            P = einsum("pcg,qgd->pqcd", CG, GD, op=semiring.plus_pair).new()
            return einsum("pqcd,rcd->pqr", P, CD, op=semiring.plus_pair).new()

        t_counts, _ = best(counts, args.reps)
        t_loop_counts, _ = best(
            lambda: {
                (p, q, r): mats[p]
                .mxm(mats[q], semiring.plus_pair)
                .new()
                .ewise_mult(mats[r], binary.pair)
                .new()
                .nvals
                for p in CG_PREDS
                for q in DG_PREDS
                for r in CD_PREDS
            },
            args.reps,
        )
        t_cgg, _ = best(lambda: cgg_tensor(CG, GG), args.reps)
        t_cgg_loop, _ = best(lambda: cgg_loop(mats), args.reps)
        rows.append(
            dict(
                nthreads=nt,
                cgd_all_tensor_ms=round(t_tensor, 1),
                cgd_all_loop_ms=round(t_loop, 1),
                cgd_counts_overlap_tensor_ms=round(t_counts, 1),
                cgd_counts_overlap_loop_ms=round(t_loop_counts, 1),
                cgg_tensor_ms=round(t_cgg, 1),
                cgg_loop_ms=round(t_cgg_loop, 1),
                load1=round(os.getloadavg()[0], 2),
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    t_scipy, _ = best(lambda: scipy_side(smats), args.reps)
    t_scipy_cgg, _ = best(
        lambda: {(p, q): smats[p] @ smats[q] for p in CG_PREDS for q in GG_PREDS}, args.reps
    )
    print(
        json.dumps(
            dict(
                scipy_cgd_counts_overlap_ms=round(t_scipy, 1),
                scipy_cgg_ms=round(t_scipy_cgg, 1),
                load1=round(os.getloadavg()[0], 2),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

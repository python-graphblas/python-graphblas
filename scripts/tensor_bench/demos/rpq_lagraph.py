"""Regular path queries on a label x src x dst Tensor, against LAGraph's RegularPathQuery.

LAGraph's experimental LAGraph_RegularPathQuery (Belyanin and Grigoriev, SPbU; arXiv 2412.10287)
keeps one adjacency Matrix per edge label for the graph (G) and for the query's NFA (R), and
runs a BFS over (state, node) pairs: each step, for every label i,

    symbol_frontier = R_i^T F            # NFA states reachable by label i
    next<!visited> |= symbol_frontier G_i  # graph nodes reachable by label i

until no new pair appears; a node is reachable when it is visited with a final state.

On the Tensor the labels are an axis: G is (label, node, node), R is (label, state, state), and
one step is one einsum over (label, state, node) with lor_land, under the ~visited mask:

    nxt = einsum("qu,lqp,luv->pv", F, R, G, op=lor_land).new(mask=~V.S)

With a source axis the same expression answers every source separately in one run:

    nxt = einsum("squ,lqp,luv->spv", F, R, G, op=lor_land).new(mask=~V.S)

Cases:
  lagraph  the four cases of LAGraph's test_RegularPathQuery.c on data/rpq_data, against the
           expected outputs in that test and a per-label Matrix loop written as LAGraph's C is
  hetionet Compound -binds-> Gene (-interacts- Gene)* -associated with-> Disease on Hetionet
           v1.0 (47,031 nodes, the three labels), all compounds at once and per compound

    python fetch_data.py lagraph-rpq hetionet
    python demos/rpq_lagraph.py lagraph
    python demos/rpq_lagraph.py hetionet [--sources 100] [--reps 3] [--threads 1,8]
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import graphblas as gb
from graphblas import Matrix, Tensor, Vector, binary, dtypes, einsum, semiring
from graphblas.core.ss import jit_config

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

# isort: on


jit_config.fix_jit_config()

LOR_LAND = semiring.lor_land


def read_mtx(path):
    """(nrows, rows, cols) of a pattern MatrixMarket file, 0-based; symmetric adds (j, i)."""
    with open(path) as f:
        header = f.readline()
        lines = [line for line in f if not line.startswith("%")]
    nrows, _ncols, _ = (int(x) for x in lines[0].split())
    ij = (
        np.array([[int(x) for x in line.split()[:2]] for line in lines[1:]], np.int64).reshape(
            -1, 2
        )
        - 1
    )
    rows, cols = ij[:, 0], ij[:, 1]
    if "symmetric" in header:
        off = rows != cols
        rows, cols = np.concatenate([rows, cols[off]]), np.concatenate([cols, rows[off]])
    return nrows, rows, cols


# ---------------------------------------------------------------- the Tensor
def rpq_tensor(G, R, QS, QF, S):
    """Nodes reachable from any of S by a path whose labels the NFA accepts (LAGraph's output)."""
    nr, ng = R.shape[1], G.shape[1]
    q0, s0 = np.meshgrid(QS, S, indexing="ij")
    F = Tensor.from_coo((q0.ravel(), s0.ravel()), True, shape=(nr, ng))
    V = F.dup()
    while F.nvals:
        F = einsum("qu,lqp,luv->pv", F, R, G, op=LOR_LAND).new(mask=~V.S)
        V(binary.lor) << F
    final = Tensor.from_coo((np.asarray(QF),), True, shape=(nr,))
    return einsum("q,qv->v", final, V, op=LOR_LAND).new()


def product_graph(G, R):
    """The product of the graph and the NFA: E(q, u, p, v) when some label takes q to p in the
    NFA and u to v in the graph. One contraction over the label axis, done once per query.
    """
    E = einsum("lqp,luv->qupv", R, G, op=LOR_LAND).new()
    # Stored with (state, node) as rows, the way each step contracts it, once (else every
    # step would move it: the planner does not keep a moved operand)
    return E.relayout(((0, 1), (2, 3)))


def rpq_tensor_product(E, QS, QF, S):
    """``rpq_tensor`` on the product graph: one masked contraction over (state, node) per step."""
    nr, ng = E.shape[0], E.shape[1]
    q0, s0 = np.meshgrid(QS, S, indexing="ij")
    F = Tensor.from_coo((q0.ravel(), s0.ravel()), True, shape=(nr, ng))
    V = F.dup()
    while F.nvals:
        F = einsum("qu,qupv->pv", F, E, op=LOR_LAND).new(mask=~V.S)
        V(binary.lor) << F
    final = Tensor.from_coo((np.asarray(QF),), True, shape=(nr,))
    return einsum("q,qv->v", final, V, op=LOR_LAND).new()


def rpq_tensor_product_per_source(E, QS, QF, S):
    """``rpq_tensor_per_source`` on the product graph."""
    nr, ng = E.shape[0], E.shape[1]
    ns = len(S)
    si, q0 = np.meshgrid(np.arange(ns), QS, indexing="ij")
    F = Tensor.from_coo(
        (si.ravel(), q0.ravel(), np.asarray(S)[si.ravel()]), True, shape=(ns, nr, ng)
    )
    V = F.dup()
    while F.nvals:
        F = einsum("squ,qupv->spv", F, E, op=LOR_LAND).new(mask=~V.S)
        V(binary.lor) << F
    final = Tensor.from_coo((np.asarray(QF),), True, shape=(nr,))
    return einsum("q,sqv->sv", final, V, op=LOR_LAND).new()


def rpq_tensor_per_source(G, R, QS, QF, S):
    """(source, node): which nodes each source reaches, all sources at once (a source axis)."""
    nr, ng = R.shape[1], G.shape[1]
    ns = len(S)
    si, q0 = np.meshgrid(np.arange(ns), QS, indexing="ij")
    F = Tensor.from_coo(
        (si.ravel(), q0.ravel(), np.asarray(S)[si.ravel()]), True, shape=(ns, nr, ng)
    )
    V = F.dup()
    while F.nvals:
        F = einsum("squ,lqp,luv->spv", F, R, G, op=LOR_LAND).new(mask=~V.S)
        V(binary.lor) << F
    final = Tensor.from_coo((np.asarray(QF),), True, shape=(nr,))
    return einsum("q,sqv->sv", final, V, op=LOR_LAND).new()


# ------------------------------------------------------- the per-label loop
def rpq_loop(Gs, Rs, QS, QF, S, ng, nr):
    """LAGraph_RegularPathQuery.c's loop in python-graphblas: one Matrix per label."""
    frontier = Matrix(dtypes.BOOL, nr, ng)
    frontier[list(QS), list(S)] = True
    visited = frontier.dup()
    while True:
        nxt = Matrix(dtypes.BOOL, nr, ng)
        for A, B in zip(Gs, Rs, strict=True):
            if A is None or B is None:
                continue
            symbol_frontier = B.T.mxm(frontier, LOR_LAND).new()
            nxt(~visited.S, binary.lor) << symbol_frontier.mxm(A, LOR_LAND)
        visited(binary.lor) << nxt
        if not nxt.nvals:
            break
        frontier = nxt
    final = Vector(dtypes.BOOL, nr)
    final[list(QF)] = True
    return final.vxm(visited, LOR_LAND).new()


# ------------------------------------------------------------------ cases
LAGRAPH_CASES = [  # from LAGraph's experimental/test/test_RegularPathQuery.c (1-based ids)
    ("simple 1 or more", ["1_a.mtx"], "1_meta.txt", "1_sources.txt", [2, 4, 6, 7]),
    ("simple kleene star", ["2_a.mtx", "2_b.mtx"], "2_meta.txt", "2_sources.txt", [2, 6, 8]),
    (
        "kleene star of the conjunction",
        ["3_a.mtx", "3_b.mtx"],
        "3_meta.txt",
        "3_sources.txt",
        [3, 6],
    ),
    ("simple repeat from n to m times", ["", "4_b.mtx"], "4_meta.txt", "4_sources.txt", [3, 4, 6]),
]


def run_lagraph():
    d = paths.data_dir() / "lagraph_rpq"
    graph = [read_mtx(os.path.join(d, f)) for f in ["a.mtx", "b.mtx"]]
    ng = graph[0][0]
    nl = len(graph)
    Gs = [Matrix.from_coo(r, c, True, nrows=ng, ncols=ng) for _, r, c in graph]
    lab = np.concatenate([np.full(r.size, i) for i, (_, r, _) in enumerate(graph)])
    G = Tensor.from_coo(
        (lab, np.concatenate([r for _, r, _ in graph]), np.concatenate([c for _, _, c in graph])),
        True,
        shape=(nl, ng, ng),
    )
    out = []
    for name, nfa_files, meta, sources, expected in LAGRAPH_CASES:
        with open(os.path.join(d, meta)) as f:
            nums = [int(x) for x in f.read().split()]
        nqs = nums[0]
        QS = [q - 1 for q in nums[1 : 1 + nqs]]
        QF = [q - 1 for q in nums[2 + nqs : 2 + nqs + nums[1 + nqs]]]
        with open(os.path.join(d, sources)) as f:
            S = [int(x) - 1 for x in f.read().split()]
        nfa = [read_mtx(os.path.join(d, f)) if f else None for f in nfa_files]
        nr = max(x[0] for x in nfa if x is not None)
        nfa += [None] * (nl - len(nfa))
        Rs = [
            None if x is None else Matrix.from_coo(x[1], x[2], True, nrows=nr, ncols=nr)
            for x in nfa
        ]
        rl, rr, rc = [], [], []
        for i, x in enumerate(nfa):
            if x is not None:
                rl.append(np.full(x[1].size, i))
                rr.append(x[1])
                rc.append(x[2])
        R = Tensor.from_coo(
            (np.concatenate(rl), np.concatenate(rr), np.concatenate(rc)), True, shape=(nl, nr, nr)
        )
        got_t = sorted(int(i) + 1 for i in rpq_tensor(G, R, QS, QF, S).to_coo()[0][0])
        got_p = sorted(
            int(i) + 1 for i in rpq_tensor_product(product_graph(G, R), QS, QF, S).to_coo()[0][0]
        )
        got_l = sorted(int(i) + 1 for i in rpq_loop(Gs, Rs, QS, QF, S, ng, nr).to_coo()[0])
        per = rpq_tensor_per_source(G, R, QS, QF, S)
        union = sorted({int(v) + 1 for v in per.to_coo()[0][1]})
        ok = got_t == got_l == union == got_p == expected
        print(
            f"{name:32s} expected {expected}  tensor {got_t}  product graph {got_p}  loop {got_l}  "
            f"per-source union {union}  {'OK' if ok else 'MISMATCH'}"
        )
        out.append(
            {
                "case": name,
                "expected": expected,
                "tensor": got_t,
                "product_graph": got_p,
                "loop": got_l,
                "per_source_union": union,
                "ok": ok,
            }
        )
        assert ok, out[-1]
    return out


def load_hetionet():
    """Hetionet as one node space with the labels CbG, GiG (both ways) and GaD (DaG reversed)."""
    index, kinds = {}, {}
    with open(paths.data_dir() / "hetionet" / "hetionet-v1.0-nodes.tsv") as f:
        next(f)
        for line in f:
            node_id, _name, kind = line.rstrip("\n").split("\t")
            index[node_id] = len(index)
            kinds.setdefault(kind, []).append(index[node_id])
    want = {"CbG": [], "GiG": [], "DaG": []}
    with gzip.open(paths.data_dir() / "hetionet" / "hetionet-v1.0-edges.sif.gz", "rt") as f:
        next(f)
        for line in f:
            s, e, t = line.rstrip("\n").split("\t")
            if e in want:
                want[e].append((index[s], index[t]))
    e = {k: np.array(v, np.int64) for k, v in want.items()}
    labels = [  # (name, src, dst)
        ("binds", e["CbG"][:, 0], e["CbG"][:, 1]),
        (
            "interacts",
            np.concatenate([e["GiG"][:, 0], e["GiG"][:, 1]]),
            np.concatenate([e["GiG"][:, 1], e["GiG"][:, 0]]),
        ),
        ("associates", e["DaG"][:, 1], e["DaG"][:, 0]),  # gene -> disease
    ]
    return labels, len(index), kinds


def run_hetionet(nsources, reps, threads):
    labels, ng, kinds = load_hetionet()
    nl = len(labels)
    print(f"Hetionet: {ng} nodes; " + ", ".join(f"{n} {s.size}" for n, s, _ in labels))
    Gs = [Matrix.from_coo(s, t, True, nrows=ng, ncols=ng) for _, s, t in labels]
    G = Tensor.from_coo(
        (
            np.concatenate([np.full(s.size, i) for i, (_, s, _) in enumerate(labels)]),
            np.concatenate([s for _, s, _ in labels]),
            np.concatenate([t for _, _, t in labels]),
        ),
        True,
        shape=(nl, ng, ng),
    )
    # binds . interacts* . associates: states 0 -b-> 1, 1 -i-> 1, 1 -a-> 2 (final)
    nr = 3
    trans = [(0, 0, 1), (1, 1, 1), (2, 1, 2)]  # (label, state, state')
    Rs = [Matrix.from_coo([q], [p], True, nrows=nr, ncols=nr) for _, q, p in trans]
    R = Tensor.from_coo(
        tuple(np.array(x) for x in zip(*trans, strict=True)), True, shape=(nl, nr, nr)
    )
    QS, QF = [0], [2]
    compounds = np.array(kinds["Compound"])
    rng = np.random.default_rng(0)
    S = sorted(rng.choice(compounds, min(nsources, compounds.size), replace=False).tolist())
    res = {"nodes": ng, "labels": {n: int(s.size) for n, s, _ in labels}, "sources": len(S)}

    def timed(f):
        ts, rv = [], None
        for _ in range(reps):
            t0 = time.perf_counter()
            rv = f()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3, rv

    for nt in threads:
        gb.ss.config["nthreads"] = nt
        t_all_t, r_t = timed(lambda: rpq_tensor(G, R, QS, QF, S))
        t_all_l, r_l = timed(lambda: rpq_loop(Gs, Rs, QS, QF, S, ng, nr))
        same = r_t.to_vector().isequal(r_l)
        t_per_t, per = timed(lambda: rpq_tensor_per_source(G, R, QS, QF, S))
        t_per_l, rows = timed(lambda: [rpq_loop(Gs, Rs, QS, QF, [s], ng, nr) for s in S])
        per_m = per.to_matrix()
        same_per = all(per_m[i, :].new().isequal(rows[i]) for i in range(len(S)))
        print(
            f"{nt} threads: union over {len(S)} sources: Tensor {t_all_t:.1f} ms, "
            f"loop {t_all_l:.1f} ms ({r_t.nvals} diseases, equal: {same}); "
            f"per source: Tensor (one run with a source axis) "
            f"{t_per_t:.1f} ms, loop (one run per source) {t_per_l:.1f} ms ({per.nvals} (compound, "
            f"disease) pairs, equal: {same_per})",
            flush=True,
        )
        t_prod, E = timed(lambda: product_graph(G, R))
        t_all_p, r_p = timed(lambda E=E: rpq_tensor_product(E, QS, QF, S))
        t_per_p, per_p = timed(lambda E=E: rpq_tensor_product_per_source(E, QS, QF, S))
        same_p = r_p.isequal(r_t) and per_p.isequal(per)
        E0 = einsum("lqp,luv->qupv", R, G, op=LOR_LAND).new()  # as einsum stores it
        t_all_p0, _ = timed(lambda E0=E0: rpq_tensor_product(E0, QS, QF, S))
        print(
            f"{nt} threads, the product graph ({E.nvals} entries, built once in {t_prod:.1f} ms): "
            f"union {t_all_p:.1f} ms, per source {t_per_p:.1f} ms (equal: {same_p}); union without "
            f"the one relayout (layout {E0.layout}) {t_all_p0:.1f} ms",
            flush=True,
        )
        res[f"{nt}t"] = {
            "union_tensor_ms": round(t_all_t, 1),
            "union_loop_ms": round(t_all_l, 1),
            "per_source_tensor_ms": round(t_per_t, 1),
            "per_source_loop_ms": round(t_per_l, 1),
            "union_nvals": int(r_t.nvals),
            "pairs": int(per.nvals),
            "product_build_ms": round(t_prod, 1),
            "product_union_ms": round(t_all_p, 1),
            "product_per_source_ms": round(t_per_p, 1),
            "product_union_no_relayout_ms": round(t_all_p0, 1),
            "equal": bool(same and same_per and same_p),
        }
        assert same, res[f"{nt}t"]
        assert same_per, res[f"{nt}t"]
        assert same_p, res[f"{nt}t"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["lagraph", "hetionet"])
    ap.add_argument("--sources", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--json")
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ../data)"
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    if args.case == "lagraph":
        res = run_lagraph()
    else:
        res = run_hetionet(args.sources, args.reps, [int(x) for x in args.threads.split(",")])
    res = {"case": args.case, "result": res, "load1": os.getloadavg()[0]}
    print(json.dumps(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

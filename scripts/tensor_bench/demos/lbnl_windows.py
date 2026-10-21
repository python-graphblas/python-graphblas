"""Graph Challenge statistics for every time window of FROSTT's LBNL network tensor, and the
multi-timescale (binary) sums, as a few Tensor operations.

Data: FROSTT lbnl-network (sender IP x sender port x destination IP x destination port x time,
1.7M entries; `python fetch_data.py frostt-lbnl`). The time axis is padded to a power-of-two
number of windows of L seconds and split by ``reshape`` into (window, offset), a view; one
reduction sums out both ports and the offset, giving the (src, dst, window) traffic tensor.
The nine statistics of the Anonymized Network Sensing Graph Challenge (CAIDA/ILANDS-sensor
python-v1 analytics.py: n_links, n_packets, n_sources, n_destinations, max_packets,
max_source_packets, max_destination_packets, max_fan_out, max_fan_in) are then reductions that
return every window's value at once. "Packets" here are the tensor's values (FROSTT's traffic
values, floats), summed as the challenge sums packets. The binary multi-timescale sums are one
more reshape and reduction per level.

The baseline is the ILANDS-style loop: one gb.Matrix per window (built from that window's
entries, ports and seconds summed by the build), the statistics per Matrix with ILANDS's
functions, and the binary sums by adding consecutive window matrices level by level. Counts
must be identical and sums equal to 1e-12.

    python demos/lbnl_windows.py [--window 3600] [--reps 3] [--threads 1,8]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import graphblas as gb
from graphblas import Matrix, Tensor, agg, monoid, unary
from graphblas.core.ss import jit_config

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

# isort: on


jit_config.fix_jit_config()

STATS = [
    "n_links",
    "n_packets",
    "n_sources",
    "n_destinations",
    "max_packets",
    "max_source_packets",
    "max_destination_packets",
    "max_fan_out",
    "max_fan_in",
]
COUNTS = {"n_links", "n_sources", "n_destinations", "max_fan_out", "max_fan_in"}


def load():
    df = pd.read_csv(paths.data_dir() / "frostt" / "lbnl-network.tns.gz", sep=" ", header=None)
    coords = [df[i].to_numpy(np.int64) - 1 for i in range(5)]
    return coords, df[5].to_numpy(np.float64), tuple(int(c.max()) + 1 for c in coords)


# ---- Tensor side ----------------------------------------------------------------------------


def tensor_stats(A):
    """The nine statistics of every window of the (src, dst, window) traffic tensor A."""
    W = A.shape[2]
    one = A.apply(unary.one).new()
    rows, rows_n = A.sum(axis=1).new(), one.sum(axis=1).new()  # (src, window)
    cols, cols_n = A.sum(axis=0).new(), one.sum(axis=0).new()  # (dst, window)
    out = {
        "n_links": one.sum(axis=(0, 1)),
        "n_packets": A.sum(axis=(0, 1)),
        "max_packets": A.max(axis=(0, 1)),
        "n_sources": rows_n.apply(unary.one).new().sum(axis=0),
        "max_source_packets": rows.max(axis=0),
        "max_fan_out": rows_n.max(axis=0),
        "n_destinations": cols_n.apply(unary.one).new().sum(axis=0),
        "max_destination_packets": cols.max(axis=0),
        "max_fan_in": cols_n.max(axis=0),
    }
    return {k: v.new().to_dense(fill_value=0.0) if W else np.zeros(0) for k, v in out.items()}


def tensor_side(coords, vals, dims, L, levels):
    nS, _, nD, _, _ = dims
    W = 2**levels
    X = Tensor.from_coo(coords, vals, shape=(*dims[:4], W * L))
    # (window, offset) from the time axis: a view; then ports and offsets summed out at once
    A = X.reshape(*dims[:4], W, L).sum(axis=(1, 3, 5)).new()  # (src, dst, window)
    out = [tensor_stats(A)]
    for _ in range(levels):  # binary sums: pairs of consecutive windows
        A = A.reshape(nS, nD, A.shape[2] // 2, 2).sum(axis=3).new()
        out.append(tensor_stats(A))
    return out


# ---- the ILANDS-style per-window Matrix loop (statistics as in ILANDS analytics.py) ---------


def matrix_stats(M):
    rows = M.reduce_rowwise(monoid.plus).new()
    cols = M.reduce_columnwise(monoid.plus).new()
    rows_n = M.reduce_rowwise(agg.count).new()
    cols_n = M.reduce_columnwise(agg.count).new()
    v = lambda s: 0.0 if s.value is None else float(s.value)  # noqa: E731
    return {
        "n_links": float(M.reduce_scalar(agg.count).new().value or 0),
        "n_packets": v(M.reduce_scalar(monoid.plus).new()),
        "max_packets": v(M.reduce_scalar(monoid.max).new()),
        "n_sources": float(rows_n.reduce(agg.count).new().value or 0),
        "max_source_packets": v(rows.reduce(monoid.max).new()),
        "max_fan_out": v(rows_n.reduce(monoid.max).new()),
        "n_destinations": float(cols_n.reduce(agg.count).new().value or 0),
        "max_destination_packets": v(cols.reduce(monoid.max).new()),
        "max_fan_in": v(cols_n.reduce(monoid.max).new()),
    }


def matrix_side(coords, vals, dims, L, levels):
    nS, _, nD, _, _ = dims
    W = 2**levels
    src, _, dst, _, t = coords
    w = t // L
    order = np.argsort(w, kind="stable")
    bounds = np.searchsorted(w[order], np.arange(W + 1))
    mats = []
    for k in range(W):
        sel = order[bounds[k] : bounds[k + 1]]
        mats.append(
            Matrix.from_coo(
                src[sel], dst[sel], vals[sel], nrows=nS, ncols=nD, dup_op=gb.binary.plus
            )
        )
    out = []
    while True:
        stats = [matrix_stats(M) for M in mats]
        out.append({s: np.array([st[s] for st in stats]) for s in STATS})
        if len(mats) == 1:
            break
        mats = [mats[i].ewise_add(mats[i + 1], monoid.plus).new() for i in range(0, len(mats), 2)]
    return out


def best(f, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rv = f()
        times.append(time.perf_counter() - t0)
    return min(times), rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3600, help="window length (time-axis units)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ../data)"
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    L = args.window
    t0 = time.perf_counter()
    coords, vals, dims = load()
    levels = math.ceil(math.log2(math.ceil(dims[4] / L)))
    print(
        f"loaded lbnl-network in {time.perf_counter() - t0:.1f} s: shape {dims}, {vals.size} "
        f"entries; windows of {L}: {math.ceil(dims[4] / L)} used, padded to {2**levels} "
        f"({levels} binary levels)",
        flush=True,
    )
    got = tensor_side(coords, vals, dims, L, levels)
    want = matrix_side(coords, vals, dims, L, levels)
    for level, (g, m) in enumerate(zip(got, want, strict=True)):
        for s in STATS:
            if s in COUNTS:
                assert np.array_equal(g[s], m[s]), (level, s)
            else:
                assert np.allclose(g[s], m[s], rtol=1e-12, atol=0), (level, s)
    print(
        f"checked: 9 statistics x {sum(len(g['n_links']) for g in got)} windows over "
        f"{levels + 1} timescales: tensor == Matrix loop (counts exact, sums to 1e-12)",
        flush=True,
    )
    busiest = int(np.argmax(got[0]["n_packets"]))
    print("busiest window", busiest, {s: got[0][s][busiest] for s in STATS}, flush=True)
    for nt in [int(x) for x in args.threads.split(",")]:
        gb.ss.config["nthreads"] = nt
        t_tensor, _ = best(lambda: tensor_side(coords, vals, dims, L, levels), args.reps)
        t_loop, _ = best(lambda: matrix_side(coords, vals, dims, L, levels), args.reps)
        print(
            json.dumps(
                dict(
                    window=L,
                    windows=2**levels,
                    levels=levels + 1,
                    nthreads=nt,
                    tensor_s=round(t_tensor, 3),
                    loop_s=round(t_loop, 3),
                    load1=round(os.getloadavg()[0], 2),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

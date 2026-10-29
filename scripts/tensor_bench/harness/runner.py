"""Run kernel groups for one (library, dataset) in this process and append JSON lines.

    python runner.py --lib graphblas --dataset u7 --groups ewise,reduce --out results.jsonl

Each case: one warm-up call, then best of N (see common.timeit); every result is
fingerprinted for the cross-library check (common.fingerprint). ``run.py`` runs each
(library, dataset, group) in a fresh process through this script.
"""

import argparse
import importlib
import os
import platform
import time
import traceback

import datasets as D
import numpy as np
from common import NA, Recorder, cases, fingerprint, peak_rss_mb, rss_mb, timeit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default=None, help="comma list overriding the adapter's")
    ap.add_argument("--cases", default=None, help="comma list: only these cases")
    args = ap.parse_args()

    t0 = time.perf_counter()
    coords, vals, shape = D.load(args.dataset)
    load_s = time.perf_counter() - t0
    base = {
        # run.py labels forced-engine variants (graphblas-numba, ...) of the same adapter
        "lib": os.environ.get("TENSOR_BENCH_LIB_LABEL", args.lib),
        "dataset": args.dataset,
        "shape": list(shape),
        "nnz": int(vals.size),
        "host": platform.node(),
    }
    rec = Recorder(args.out, base)
    groups = args.groups.split(",")
    try:
        mod = importlib.import_module(f"ad_{args.lib}")
        if hasattr(mod, "JIT"):
            # the SuiteSparse JIT control (run.py --jit) goes on every result
            base["jit"] = mod.JIT
        t0 = time.perf_counter()
        ad = mod.Adapter(args.dataset, coords, vals, shape)
        build_s = time.perf_counter() - t0
    except NA as e:
        for g in groups:
            rec.write(group=g, case="*", status="n/a", note=str(e))
        return
    rss0 = rss_mb()
    rec.write(
        group="_setup",
        case="_setup",
        status="info",
        info=ad.info(),
        load_s=round(load_s, 3),
        build_s=round(build_s, 3),
        rss_after_build_mb=round(rss0, 1),
    )
    threads = ad.threads
    if args.threads and ad.threads != (1,):
        # multithreaded libraries take the requested counts; single-threaded ones run once
        threads = tuple(map(int, args.threads.split(",")))
    only = set(args.cases.split(",")) if args.cases else None
    for group in groups:
        try:
            ad.setup(group)
        except NA as e:
            rec.write(group=group, case="*", status="n/a", note=str(e))
            continue
        for case in cases(group, shape, args.dataset) + ad.extra_cases(group):
            if only and case not in only:
                continue
            for nt in threads:
                ad.set_threads(nt)
                common = dict(group=group, case=case, nthreads=nt)
                try:
                    fn = ad.case(group, case)
                except NA as e:
                    rec.write(**common, status="n/a", note=str(e))
                    break
                rec.write(**common, status="start")
                rss_before = rss_mb()
                try:
                    best, med, n, first, res = timeit(fn)
                except NA as e:
                    rec.write(**common, status="n/a", note=str(e))
                    break
                except Exception as e:
                    rec.write(
                        **common,
                        status="error",
                        note=f"{type(e).__name__}: {str(e)[:300]}",
                        tb=traceback.format_exc()[-1500:],
                    )
                    break
                rss_after = rss_mb()
                try:
                    fp = fingerprint(ad.canon(res))
                except Exception as e:
                    fp = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                del res
                extra = {}
                if hasattr(ad, "note"):
                    extra["note"] = ad.note(group, case)
                if hasattr(ad, "prep_seconds"):
                    ps = ad.prep_seconds(group, case)
                    if ps is not None:
                        extra["prep_ms"] = round(1e3 * ps, 3)
                rec.write(
                    **common,
                    status="ok",
                    best_ms=round(1e3 * best, 6),
                    median_ms=round(1e3 * med, 6),
                    n=n,
                    first_ms=round(1e3 * first, 4),
                    fp=fp,
                    rss_before_mb=round(rss_before, 1),
                    rss_after_mb=round(rss_after, 1),
                    **extra,
                )
    rec.write(
        group="_end",
        case="_end",
        status="info",
        info=ad.info(),
        final_peak_rss_mb=round(peak_rss_mb(), 1),
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

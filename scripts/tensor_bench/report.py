"""Tables and a cross-library fingerprint check from run.py's results.

    python report.py results/*.jsonl                 # markdown tables + check, to stdout
    python report.py results/*.jsonl --csv all.csv   # also every timed row, flat
    python report.py results/*.jsonl --teeth         # show that the check catches errors

Each table has one row per (dataset, case) and one column per (library, threads), with the
best time in ms; the fastest is in bold. With results from several machines, each machine
gets its own tables (``--host`` picks one).

**The check.** Every result carries a fingerprint of its output: the shape, the entry count, an
order-independent hash of the entries' linear indices, the sum of the values and a weighted
sum (``harness/common.py``). For each (dataset, case), every result is compared with a
reference: the Tensor (``graphblas``) at the highest thread count when there is one, else the
first result. Counts and hashes must be equal and the sums equal to 1e-8 relative. Results
from different machines are compared too: the data and seeds are the same everywhere.
``--teeth`` plants three wrong fingerprints (a value, an index, a count) in a copy of the
results and fails unless the check flags all three. The exit status is 1 on any mismatch.
"""

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
from common import fp_match

# isort: on


LIB_ORDER = [
    "graphblas",
    "graphblas-numba",
    "graphblas-udt",
    "graphblas-products",
    "splatt-allmode",
    "splatt-twomode",
    "splatt-onemode",
    "taco",
    "finch",
    "numba",
    "torch",
    "scipy",
    "pydata",
    "gcxs",
    "numpy",
]
GROUP_ORDER = [
    "construct",
    "ewise",
    "bcast",
    "reduce",
    "permute",
    "ttv",
    "ttm",
    "mttkrp",
    "contract",
    "bmm",
    "slice",
]


def load(paths):
    rows = []
    for p in paths:
        host = None
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "machine":
                    host = r.get("host")
                r.setdefault("host", host)
                rows.append(r)
    return rows


def timed(rows):
    return [
        r
        for r in rows
        if r.get("status") == "ok" and "best_ms" in r and r.get("group") not in ("_setup", "_end")
    ]


def col(r):
    return (r["lib"], r.get("nthreads"))


def colname(c):
    lib, nt = c
    return f"{lib} {nt}t" if nt is not None else lib


def fmt(ms):
    if ms is None:
        return ""
    if ms < 1:
        return f"{ms:.3f}"
    if ms < 10:
        return f"{ms:.2f}"
    if ms < 100:
        return f"{ms:.1f}"
    return f"{ms:.0f}"


def check(rows, verbose=True):
    """(matches, mismatches, messages) over every fingerprinted result."""
    by = defaultdict(list)
    for r in rows:
        fp = r.get("fp")
        if isinstance(fp, dict) and "count" in fp:
            case = r["case"].removesuffix("_raw").removesuffix("_ttm")
            if r.get("group") == "construct":
                case = "from_coo_sorted"  # every construction and export holds the same tensor
            by[(r["dataset"], case)].append(r)
    ok = bad = 0
    msgs = []
    for key, rs in sorted(by.items()):
        gb = [r for r in rs if r["lib"] == "graphblas"]
        ref = max(gb, key=lambda r: r.get("nthreads") or 0) if gb else rs[0]
        for r in rs:
            if r is ref:
                continue
            if fp_match(ref["fp"], r["fp"]):
                ok += 1
            else:
                bad += 1
                msgs.append(
                    f"MISMATCH {key[0]} {key[1]}: {r.get('host')} {colname(col(r))} vs "
                    f"{ref.get('host')} {colname(col(ref))}\n    got {r['fp']}\n    ref {ref['fp']}"
                )
    if verbose:
        for m in msgs:
            print(m)
    return ok, bad, msgs


def teeth(rows):
    """Plant three wrong fingerprints in a copy; True if the check flags each of them."""
    t = copy.deepcopy(rows)
    _, base_bad, _ = check(t, verbose=False)
    candidates = [
        r
        for r in t
        if isinstance(r.get("fp"), dict) and "count" in r["fp"] and r["lib"] != "graphblas"
    ]
    # one victim per library where there are enough libraries
    victims, seen = [], set()
    for r in candidates:
        if r["lib"] not in seen:
            victims.append(r)
            seen.add(r["lib"])
    victims += [r for r in candidates if r not in victims]
    if len(victims) < 3:
        print("teeth: fewer than 3 non-reference results to tamper with")
        return False
    victims[0]["fp"]["vsum"] *= 1 + 1e-6  # a value off in the 7th digit
    victims[1]["fp"]["hsum"] = hex(int(victims[1]["fp"]["hsum"], 16) ^ 1)  # one index differs
    victims[2]["fp"]["count"] += 1  # one entry too many
    _, bad, _ = check(t, verbose=False)
    caught = bad - base_bad
    print(
        f"teeth: {caught} of 3 planted mismatches caught "
        f"({victims[0]['lib']}, {victims[1]['lib']}, {victims[2]['lib']})"
    )
    return caught == 3


def machine_lines(rows):
    out = []
    for r in rows:
        if r.get("kind") != "machine":
            continue
        cpu, g, v = r.get("cpu", {}), r.get("graphblas", {}), r.get("versions", {})
        cores = f"{cpu.get('logical_cores')} cores"
        if cpu.get("core_types"):
            cores += " (" + ", ".join(f"{n} {k}" for k, n in cpu["core_types"].items()) + ")"
        vers = ", ".join(f"{k} {val}" for k, val in v.items())
        out.append(
            f"- **{r.get('host')}**, {r.get('time')}: {cpu.get('model')}, {cores}, "
            f"{cpu.get('memory_gb')} GB; {r.get('os')}. "
            f"SuiteSparse:GraphBLAS {g.get('suitesparse')}, "
            f"JIT {g.get('jit_c_control')} ({g.get('jit_c_compiler_name')}), Numba "
            f"{'yes' if g.get('has_numba') else 'no'}; threads {r.get('threads')}; load at start "
            f"{r.get('load_avg')}; commit {r.get('git_commit')}. {vers}."
        )
    return out


def tables(rows, host):
    rs = [r for r in timed(rows) if r.get("host") == host]
    best = {}
    for r in rs:
        k = (r["group"], r["dataset"], r["case"], col(r))
        if k not in best or r["best_ms"] < best[k]["best_ms"]:
            best[k] = r
    groups = sorted(
        {k[0] for k in best}, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99
    )
    for g in groups:
        cols = sorted(
            {k[3] for k in best if k[0] == g},
            key=lambda c: (LIB_ORDER.index(c[0]) if c[0] in LIB_ORDER else 99, -(c[1] or 0)),
        )
        keys = sorted({(k[1], k[2]) for k in best if k[0] == g})
        print(f"\n### {g} ({host}; best of N, ms; bold = fastest)\n")
        print("| dataset | case | " + " | ".join(colname(c) for c in cols) + " | load |")
        print("|---|---|" + "---|" * len(cols) + "---|")
        for ds, case in keys:
            cells = [best.get((g, ds, case, c)) for c in cols]
            times = [c["best_ms"] for c in cells if c]
            lo = min(times)
            loads = [c.get("load1") for c in cells if c and c.get("load1") is not None]
            row = [
                (
                    "**" + fmt(c["best_ms"]) + "**"
                    if c and c["best_ms"] == lo
                    else fmt(c and c["best_ms"])
                )
                for c in cells
            ]
            ld = f"{min(loads):.1f}-{max(loads):.1f}" if loads else ""
            print(f"| {ds} | {case} | " + " | ".join(row) + f" | {ld} |")
    # GraphBLAS over the fastest other library, geometric mean per dataset
    ratios = defaultdict(list)
    for (g, ds, case, c), r in best.items():
        # export to COO is a copy for the Tensor and free for COO libraries: not a kernel
        if c[0] != "graphblas" or case == "to_coo":
            continue
        others = [
            v["best_ms"]
            for (g2, d2, c2, cc), v in best.items()
            if (g2, d2, c2) == (g, ds, case)
            and not cc[0].startswith("graphblas")
            and cc[0] != "numpy"
        ]
        # (an export that returns the library's own arrays can time at 0)
        if others and min(others) > 0 and r["best_ms"] > 0:
            ratios[(ds, c[1])].append(min(others) / r["best_ms"])
    if ratios:
        print(
            f"\n### Speedup of graphblas over the fastest other library ({host}; "
            "geometric mean, >1 = graphblas faster; export to COO left out)\n"
        )
        print("| dataset | threads | cases | geomean | min | max |")
        print("|---|---|---|---|---|---|")
        for (ds, nt), xs in sorted(ratios.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0))):
            gm = math.exp(sum(map(math.log, xs)) / len(xs))
            print(f"| {ds} | {nt} | {len(xs)} | {gm:.2f}x | {min(xs):.2f}x | {max(xs):.2f}x |")


def write_csv(rows, path):
    keys = [
        "host",
        "dataset",
        "lib",
        "group",
        "case",
        "nthreads",
        "best_ms",
        "median_ms",
        "n",
        "first_ms",
        "prep_ms",
        "compile_ms",
        "load1",
        "time",
        "fp_count",
        "fp_hsum",
        "fp_vsum",
        "note",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in timed(rows):
            fp = r.get("fp") or {}
            w.writerow(
                dict(r, fp_count=fp.get("count"), fp_hsum=fp.get("hsum"), fp_vsum=fp.get("vsum"))
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", nargs="+")
    ap.add_argument("--host", default=None, help="only this machine's tables")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--teeth", action="store_true")
    args = ap.parse_args()
    rows = load(args.results)
    print("# Tensor benchmark results\n")
    for line in machine_lines(rows):
        print(line)
    hosts = sorted({r.get("host") for r in timed(rows)}, key=str)
    for host in [args.host] if args.host else hosts:
        tables(rows, host)
    print("\n### Fingerprint check\n")
    ok, bad, _ = check(timed(rows))
    errors = [r for r in rows if r.get("status") in ("error", "crash", "timeout")]
    print(
        f"\n{ok} results match their reference, {bad} mismatch; "
        f"{len(errors)} errors, crashes or timeouts."
    )
    for r in errors:
        print(
            f"- {r['status']}: {r.get('lib')} {r.get('dataset')} {r.get('group')}/{r.get('case')} "
            f"nt={r.get('nthreads')}: {str(r.get('note'))[:200]}"
        )
    if args.csv:
        write_csv(rows, args.csv)
    status = 1 if bad else 0
    if args.teeth and not teeth(timed(rows)):
        status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()

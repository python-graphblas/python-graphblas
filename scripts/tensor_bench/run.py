"""Run the Tensor benchmarks against other sparse-tensor libraries, one process at a time.

    python run.py                                  # u6 and u7, every installed library
    python run.py --datasets u7,uber --libs graphblas,pydata,scipy
    python run.py --datasets nell2 --groups mttkrp --libs graphblas,splatt,taco

Each (library, dataset, group) runs in a fresh process (``harness/runner.py``), so caches,
JIT state and memory high-water marks do not leak between libraries. The first line of the
results file records the machine (CPU, cores, OS, versions, the SuiteSparse JIT); every
result carries its time, the machine load, and a fingerprint of the output that
``report.py`` compares across libraries.

Libraries (``--libs``):
  graphblas            the Tensor, with the engines it picks
  graphblas-numba      the Tensor with its Numba kernels forced for TTM and MTTKRP
  graphblas-udt        ... its array-UDT engine (needs the SuiteSparse JIT)
  graphblas-products   ... its GraphBLAS products (what runs without Numba or the JIT)
  pydata               pydata/sparse COO          gcxs      pydata/sparse GCXS (reduce, permute)
  scipy                scipy.sparse n-d coo_array torch     torch.sparse COO
  numba                hand-written Numba kernels for TTV, TTM and MTTKRP over per-mode copies
  numpy                dense numpy, the reference on the small datasets
  splatt, taco, finch  compiled comparators, if built (see compiled/README.md)
The default is graphblas plus whichever of pydata, scipy and torch import.

Datasets (``--datasets``): synthetic ones are generated on first use (small, small_skew, u6,
u7, skew, hyper, zipf, bmm10, bmm100, bmm1000); FROSTT ones (nell2, enron, uber, nips,
chicago) need ``python fetch_data.py frostt-<name>`` first.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness"
COMPILED = HERE / "compiled"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HARNESS))


GROUPS = [
    "construct",
    "ewise",
    "bcast",
    "reduce",
    "permute",
    "ttv",
    "ttm",
    "mttkrp",
    "contract",
    "slice",
]
# groups each library implements (default: all)
LIB_GROUPS = {
    "numba": {"ttv", "ttm", "mttkrp"},
    "gcxs": {"reduce", "permute"},
    "splatt": {"mttkrp"},
    "taco": {"ewise", "reduce", "ttv", "ttm", "mttkrp"},
    "finch": {"ewise", "reduce", "ttv", "ttm", "mttkrp"},
}
# groups with no operands for a dataset
SKIP = {
    "nell2": {"contract"},
    "hyper": {"contract"},
    "enron": {"contract", "bcast"},
    "uber": {"contract", "bcast"},
    "nips": {"contract", "bcast"},
    "chicago": {"contract", "bcast"},
}
BMM = {"bmm10", "bmm100", "bmm1000", "bmm_small"}
DEFAULT_OPTIONAL = ["pydata", "scipy", "torch"]
IMPORT_OF = {
    "pydata": "sparse",
    "gcxs": "sparse",
    "scipy": "scipy",
    "torch": "torch",
    "numba": "numba",
}
GB_ENGINES = {"graphblas-numba": "numba", "graphblas-udt": "udt", "graphblas-products": "products"}
COMPILED_LIBS = {"splatt", "taco", "finch"}


def installed(lib):
    mod = IMPORT_OF.get(lib)
    if mod is None:
        return True
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True)
    return r.returncode == 0


def plan(datasets, libs, groups):
    for ds in datasets:
        for lib in libs:
            base = "graphblas" if lib in GB_ENGINES else lib
            if ds in BMM:
                gs = ["bmm"] if base not in LIB_GROUPS else []
            else:
                gs = [
                    g
                    for g in groups
                    if g not in SKIP.get(ds, ()) and g in LIB_GROUPS.get(base, GROUPS)
                ]
            for g in gs:
                yield ds, lib, g


def load1():
    return round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None


def append(out, rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_python(ds, lib, group, out, args, env):
    adapter = "graphblas" if lib in GB_ENGINES else lib
    cmd = [
        sys.executable,
        str(HARNESS / "runner.py"),
        "--lib",
        adapter,
        "--dataset",
        ds,
        "--groups",
        group,
        "--out",
        str(out),
        "--threads",
        args.threads,
    ]
    if args.cases:
        cmd += ["--cases", args.cases]
    penv = dict(env)
    if lib in GB_ENGINES:
        penv["TENSOR_BENCH_GB_ENGINE"] = GB_ENGINES[lib]
        penv["TENSOR_BENCH_LIB_LABEL"] = lib
    try:
        p = subprocess.run(
            cmd, cwd=HARNESS, env=penv, timeout=args.timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        append(
            out,
            {
                "lib": lib,
                "dataset": ds,
                "group": group,
                "case": "*",
                "status": "timeout",
                "note": f"process timeout {args.timeout:.0f} s",
            },
        )
        return "TIMEOUT"
    if p.returncode:
        tail = " / ".join((p.stderr or "").strip().splitlines()[-3:])
        append(
            out,
            {
                "lib": lib,
                "dataset": ds,
                "group": group,
                "case": "*",
                "status": "crash",
                "note": tail[:500],
            },
        )
        return f"exit {p.returncode} | {tail}"
    return "ok"


def run_compiled(ds, lib, group, out, args, env):
    import compiled_runner

    return compiled_runner.run(lib, ds, group, out, args.threads, env, args.timeout)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datasets", default="u6,u7")
    ap.add_argument(
        "--libs",
        default=None,
        help="comma list (default: graphblas + installed pydata, scipy, torch)",
    )
    ap.add_argument("--groups", default="all")
    ap.add_argument("--cases", default=None, help="comma list: only these cases")
    ap.add_argument(
        "--threads",
        default=None,
        help="thread counts for multithreaded libraries (default: min(8, cores),1)",
    )
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ./data)"
    )
    ap.add_argument("--out", default=None, help="results file (JSON lines; appended to)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    if args.threads is None:
        n = min(8, os.cpu_count() or 1)
        args.threads = f"{n},1" if n > 1 else "1"
    if args.libs is None:
        libs = ["graphblas"] + [lib for lib in DEFAULT_OPTIONAL if installed(lib)]
    else:
        libs = args.libs.split(",")
        missing = [lib for lib in libs if lib not in COMPILED_LIBS and not installed(lib)]
        if missing:
            ap.error(f"not installed: {', '.join(missing)}")
    groups = GROUPS if args.groups == "all" else args.groups.split(",")
    out = (
        Path(args.out)
        if args.out
        else paths.results_dir() / f"{platform.node()}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    from machine import machine_info

    repo = HERE.parent.parent if (HERE.parent.parent / ".git").exists() else None
    info = machine_info(repo)
    info.update(
        kind="machine",
        libs=libs,
        datasets=args.datasets.split(","),
        groups=groups,
        threads=args.threads,
        data_dir=str(paths.data_dir()),
    )
    append(out, info)
    cpu = info["cpu"]
    print(
        f"machine: {cpu.get('model')}, {cpu.get('logical_cores')} cores "
        f"{cpu.get('core_types', '')}, {info['os']}; "
        f"SuiteSparse {info['graphblas'].get('suitesparse')}, "
        f"JIT {info['graphblas'].get('jit_c_control')}"
    )
    print(f"results: {out}", flush=True)
    for ds, lib, g in plan(args.datasets.split(","), libs, groups):
        t0 = time.time()
        print(
            f"[{time.strftime('%H:%M:%S')}] {ds:8s} {lib:18s} {g:10s} load {load1()}",
            end=" ",
            flush=True,
        )
        status = (run_compiled if lib in COMPILED_LIBS else run_python)(ds, lib, g, out, args, env)
        print(f"{time.time() - t0:7.1f} s  {status}", flush=True)
    print(f"done; summarize with: python {Path(__file__).with_name('report.py')} {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(COMPILED))
    main()

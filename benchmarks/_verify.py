#!/usr/bin/env python
"""Standalone sanity check for the asv benchmark suite (not an asv benchmark).

Imports every benchmark module the way asv does (benchmark_dir on sys.path),
then for each benchmark class runs setup() and calls each benchmark method once
across all parameter combinations. timeraw_* methods return a code string, which
is executed in a fresh subprocess. Prints PASS/FAIL per benchmark and a timing so
we can spot anything accidentally slow.

Usage:  python _verify.py
"""

import importlib
import inspect
import itertools
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))  # flat import mode, as asv does

MODULES = [
    "scalar_access",
    "index_parse",
    "small_ops",
    "large_kernels",
    "conversions",
    "imports",
    "repr_bench",
]
PREFIXES = ("time_", "timeraw_", "peakmem_", "mem_", "track_")


def param_combos(cls):
    params = getattr(cls, "params", None)
    if not params:
        return [()]
    # asv: a flat list is a single parameter; a list of lists is a product.
    if params and isinstance(params[0], (list, tuple)):
        return list(itertools.product(*params))
    return [(p,) for p in params]


def run_one(cls, combo):
    results = []
    obj = cls()
    if hasattr(obj, "setup"):
        obj.setup(*combo)
    for name in sorted(dir(obj)):
        if not name.startswith(PREFIXES):
            continue
        method = getattr(obj, name)
        if not callable(method):
            continue
        label = f"{cls.__module__}.{cls.__name__}.{name}{combo or ''}"
        t0 = time.perf_counter()
        try:
            if name.startswith("timeraw_"):
                code = method()
                setup_code = ""
                if isinstance(code, tuple):
                    code, setup_code = code
                script = (setup_code + "\n" + code) if setup_code else code
                subprocess.run(
                    [sys.executable, "-c", script], check=True, capture_output=True, timeout=180
                )
            else:
                method(*combo)
            dt = time.perf_counter() - t0
            results.append((label, dt, None))
        except Exception as exc:
            dt = time.perf_counter() - t0
            results.append((label, dt, repr(exc)))
    if hasattr(obj, "teardown"):
        obj.teardown(*combo)
    return results


def main():
    all_results = []
    for modname in MODULES:
        mod = importlib.import_module(modname)
        classes = [
            c for _, c in inspect.getmembers(mod, inspect.isclass) if c.__module__ == modname
        ]
        for cls in classes:
            for combo in param_combos(cls):
                all_results.extend(run_one(cls, combo))

    fails = [r for r in all_results if r[2] is not None]
    for label, dt, err in all_results:
        status = "PASS" if err is None else "FAIL"
        line = f"[{status}] {dt:7.3f}s  {label}"
        if err is not None:
            line += f"   -> {err}"
        print(line)
    print(
        f"\n{len(all_results) - len(fails)}/{len(all_results)} benchmarks ran; {len(fails)} failed"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

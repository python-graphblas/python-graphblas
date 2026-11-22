#!/usr/bin/env python
"""Run every benchmark of the asv suite once, without asv (not an asv benchmark).

Imports each benchmark module as asv does (with this directory on sys.path), then for
each benchmark class runs setup() and calls each benchmark method once for every
combination of its parameters. A timeraw_* method returns code, which runs in a fresh
process. Prints PASS or FAIL and the time of each benchmark, so a slow one shows.

Usage:  python benchmarks/_verify.py
"""

import importlib
import inspect
import itertools
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))  # import the modules flat, as asv does

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

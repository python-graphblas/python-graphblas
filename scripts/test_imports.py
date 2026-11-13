#!/usr/bin/env python
"""Import every module of graphblas, each as the first import of a fresh Python process.

Importing all of them in one process is not enough. Once one module has loaded, so has
everything it imports, and an import cycle that breaks only when a module is the first
one imported goes unnoticed: ``import graphblas.core.dtypes`` failed that way. So each
module, and each statement in ``STATEMENTS``, runs in its own process, several at a time.

Run it from the root of the repository, as ``python scripts/test_imports.py``: like
``python -c``, it imports graphblas from the current directory when it is there.
The times it reports measure import performance too.
"""

import argparse
import importlib.util
import os
import pkgutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The test modules are imported by pytest, which they need; only their package is checked
NO_DESCEND = {"graphblas.tests"}

# Modules that need an optional dependency, which only code that checks for it imports
NEEDS = {"graphblas.core.operator.udf": "numba"}

# Statements that import names and reach lazily created attributes
_NAMESPACE_ATTRS = {
    "agg": "count",
    "binary": "plus",
    "indexunary": "tril",
    "monoid": "plus",
    "op": "plus",
    "select": "tril",
    "semiring": "plus_times",
    "unary": "exp",
}
# Names imported from graphblas and from graphblas.core
# fmt: off
_TOP_LEVEL_NAMES = [
    "Matrix", "Scalar", "Vector", "Recorder", "agg", "binary", "dtypes", "exceptions", "init", "io",
    "monoid", "op", "select", "semiring", "tests", "unary", "ss", "viz", "MAX_SIZE",
]
_CORE_NAMES = [
    "base", "descriptor", "dtypes", "expr", "formatting", "ffi", "infix", "lib", "mask", "matrix",
    "operator", "scalar", "vector", "recorder", "automethods", "infixmethods", "slice", "ss",
]
# fmt: on
STATEMENTS = [
    "from graphblas import * ; Matrix",
    "from graphblas.core.operator import agg",
    *(f"from graphblas import {name}" for name in _TOP_LEVEL_NAMES),
    *(f"from graphblas.core import {name}" for name in _CORE_NAMES),
    *(f"from graphblas.{ns} import {attr}" for ns, attr in _NAMESPACE_ATTRS.items()),
    *(f"import graphblas as gb ; gb.{ns}.{attr}" for ns, attr in _NAMESPACE_ATTRS.items()),
    *(f"from graphblas import {ns} ; {ns}.{attr}" for ns, attr in _NAMESPACE_ATTRS.items()),
]


def find_submodules(package):
    """Yield ``package`` and the names of the modules under it, without importing them."""
    spec = importlib.util.find_spec(package)
    if spec is None:
        raise ModuleNotFoundError(f"No module named {package!r}", name=package)
    yield from _walk(package, spec.submodule_search_locations)


def _walk(name, locations):
    yield name
    if not locations or name in NO_DESCEND:
        return
    for info in pkgutil.iter_modules(locations):
        subname = f"{name}.{info.name}"
        if info.ispkg:
            yield from _walk(subname, [str(Path(info.module_finder.path) / info.name)])
        else:
            yield subname


def run(statement):
    """Run ``statement`` in a fresh process; return its exit code, stderr and duration."""
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stderr, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count(), help="processes to run at once"
    )
    args = parser.parse_args()

    # Find modules as the child processes will: from the current directory first
    sys.path[0] = str(Path.cwd())
    modules = list(find_submodules("graphblas"))
    print(f"Testing {len(modules)} modules of {importlib.util.find_spec('graphblas').origin}")

    missing = {dep for dep in set(NEEDS.values()) if run(f"import {dep}")[0] != 0}
    skipped = sorted(mod for mod, dep in NEEDS.items() if dep in missing)
    checks = [f"import {mod}" for mod in modules if mod not in skipped] + STATEMENTS

    failures = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run, check): check for check in checks}
        for future in as_completed(futures):
            check = futures[future]
            returncode, stderr, seconds = future.result()
            if returncode != 0:
                failures.append((check, stderr))
            status = "FAIL" if returncode != 0 else "ok"
            print(f"{status:4} {seconds:5.2f}s  python -c {check!r}", flush=True)
    elapsed = time.perf_counter() - start

    for check, stderr in failures:
        print(f"\n===== FAIL: python -c {check!r}\n{stderr}", end="")
    for mod in skipped:
        print(f"skipped import {mod}: needs {NEEDS[mod]}")
    print(
        f"\n{len(checks) - len(failures)} of {len(checks)} passed, {len(skipped)} skipped, "
        f"in {elapsed:.1f}s with {args.jobs} jobs"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

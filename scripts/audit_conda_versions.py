#!/usr/bin/env python
"""Audit the CI dependency pools in `ci_pick_versions.py` against conda-forge.

CI installs its dependencies from conda-forge, so this is the authoritative companion
to `audit_pypi_versions.py`. It can see two things PyPI cannot:

* which Pythons a version was actually built for. conda-forge builds more than PyPI
  ships (it has a py311 build of pyyaml 5.4.1 where PyPI's newest wheel is cp39), and
  sometimes fewer, so "no wheel" and "no build" are different questions.
* `constrains` metadata (conda's run_constrained). Several rules in the picker exist
  only because a package constrains something it does not depend on, which is what
  makes a pinned pair fail to solve.

Usage:
    python scripts/audit_conda_versions.py [--samples N]

Requires `conda` on PATH and network access. Slower than the PyPI audit: one
`conda search` per package, a few seconds each. Exits non-zero if any check fails.
See also scripts/check_versions.sh for the by-hand version of the same questions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

_PICKER = Path(__file__).with_name("ci_pick_versions.py")
_spec = importlib.util.spec_from_file_location("ci_pick_versions", _PICKER)
picker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(picker)

# Our internal names to conda-forge package names
CONDA_NAMES = {"psg": "python-suitesparse-graphblas", "matplotlib": "matplotlib-base"}

PER_PYTHON_POOLS = {
    "numpy": picker.NUMPY_VERSIONS,
    "networkx": picker.NETWORKX_VERSIONS,
    "pyyaml": picker.PYYAML_VERSIONS,
    "sparse": picker.SPARSE_VERSIONS,
}

SAMPLED_DEPS = ("scipy", "pandas", "awkward", "numba", "networkx", "sparse", "pyyaml", "psg")

# Constraints the picker hard-codes because they come from conda metadata alone. If one
# of these drifts, the matching rule in `apply_constraints` needs another look.
# Each entry is (dep, version pin, field, expected constraint).
DOCUMENTED_METADATA = [
    ("networkx", "3.6", "constrains", "numpy >=1.25"),
    ("networkx", "3.6", "constrains", "scipy >=1.11.2"),
    ("networkx", "3.6", "constrains", "pandas >=2.0"),
    ("networkx", "3.5", "constrains", "numpy >=1.25"),
    ("pandas", "2.3", "constrains", "scipy >=1.10.0"),
    # matplotlib 3.11.0's metadata understates its numpy floor; 3.11.1 fixed it. The
    # picker pins matplotlib <3.11 for old numpy because of the 3.11.0 builds.
    ("matplotlib", "3.11.0", "depends", "numpy >=1.23"),
    ("matplotlib", "3.11.1", "depends", "numpy >=1.25"),
]


# ---------------------------------------------------------------------------
# conda access
# ---------------------------------------------------------------------------

_cache: dict[tuple[str, str], list] = {}


def conda_search(dep, floor):
    """All conda-forge builds of `dep` at or above `floor`, cached per process."""
    if (dep, floor) not in _cache:
        name = CONDA_NAMES.get(dep, dep)
        spec = f"{name}[channel=conda-forge]>={floor}"
        result = subprocess.run(
            [_CONDA, "search", "--json", spec],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise SystemExit(f"conda search failed for {spec}:\n{result.stderr.strip()}") from None
        if isinstance(payload, dict) and payload.get("error"):
            raise SystemExit(f"conda search failed for {spec}: {payload['error']}")
        _cache[dep, floor] = [build for builds in payload.values() for build in builds]
    return _cache[dep, floor]


def builds_for(dep, pin, floor):
    """Builds of `dep` matching a pin ("" means every build at or above `floor`)."""
    return [
        b
        for b in conda_search(dep, floor)
        if not pin or b["version"] == pin or b["version"].startswith(f"{pin}.")
    ]


def build_supports(build, pyver):
    """Whether a conda build can be installed on `pyver`."""
    minor = int(pyver.split(".")[1])
    if match := re.search(r"py3(\d\d)", build["build"]):
        return int(match.group(1)) == minor
    constraints = []
    for dep in build.get("depends", []):
        tokens = dep.split()
        if tokens[0] != "python" or len(tokens) < 2:
            continue
        constraints.append(f"=={tokens[1]}" if tokens[1].endswith(".*") else tokens[1])
    if not constraints:
        return True  # noarch with a bare `python` dep
    try:
        return Version(pyver) in SpecifierSet(",".join(constraints))
    except (InvalidSpecifier, InvalidVersion):
        return True


def supports_python(dep, pin, pyver, floor):
    """Whether any build matching the pin supports `pyver`."""
    return any(build_supports(b, pyver) for b in builds_for(dep, pin, floor))


def numpy_cap(dep, pin, floor):
    """The tightest `numpy <X` upper bound the newest build of a pin declares.

    Reads `constrains` as well as `depends`: conda-forge puts the ABI cap in depends
    (numba 0.57 says `numpy <2.0a0`) and the package's real runtime cap in constrains
    (`numpy ...,<1.25`), and it is the latter that makes the solve fail.
    """
    matching = builds_for(dep, pin, floor)
    if not matching:
        return None
    newest = max(matching, key=lambda b: (_version_key(b["version"]), b["build_number"]))
    caps = []
    for entry in newest.get("depends", []) + newest.get("constrains", []):
        tokens = entry.split(None, 1)
        if tokens[0] != "numpy" or len(tokens) < 2:
            continue
        for part in tokens[1].split(","):
            part = part.strip()
            if part.startswith("<") and not part.startswith("<="):
                release = Version(part[1:]).release
                caps.append((release[0], release[1] if len(release) > 1 else 0))
    return min(caps) if caps else None


def _version_key(version):
    try:
        return Version(version)
    except InvalidVersion:
        return Version("0")


def _oldest_pin(dep):
    """The oldest pin we test for a dep, used to bound the conda search."""
    pools = {
        "numpy": [v for pool in picker.NUMPY_VERSIONS.values() for v in pool],
        "scipy": [v for pool in picker.SCIPY_VERSIONS["1.x"].values() for v in pool]
        + picker.SCIPY_VERSIONS["2.x"],
        "pandas": [v for pool in picker.PANDAS_VERSIONS["1.x"].values() for v in pool]
        + picker.PANDAS_VERSIONS["2.x"],
        "awkward": [v for pool in picker.AWKWARD_VERSIONS["1.x"].values() for v in pool]
        + picker.AWKWARD_VERSIONS["2.x"],
        "numba": picker.NUMBA_VERSIONS["1.x"] + picker.NUMBA_VERSIONS["2.x"],
        "networkx": [v for pool in picker.NETWORKX_VERSIONS.values() for v in pool],
        "pyyaml": [v for pool in picker.PYYAML_VERSIONS.values() for v in pool],
        "sparse": [v for pool in picker.SPARSE_VERSIONS.values() for v in pool],
        "psg": [v for pool in picker.PSG_VERSIONS_NP1.values() for p in pool.values() for v in p]
        + picker.PSG_VERSIONS_NP2,
        "matplotlib": ["3.11.0"],
    }
    pins = [v for v in pools[dep] if v and v != "NA"]
    return min(pins, key=picker._ver)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_per_python_pools():
    """Every pin in a per-Python pool must have a conda-forge build for that Python."""
    problems = []
    for dep, pools in PER_PYTHON_POOLS.items():
        floor = _oldest_pin(dep)
        for pyver, pins in pools.items():
            for pin in pins:
                if pin == "NA":
                    continue
                if not builds_for(dep, pin, floor):
                    problems.append(f"{dep} {pin or 'latest'} has no conda-forge build at all")
                elif not supports_python(dep, pin, pyver, floor):
                    problems.append(
                        f"{dep} {pin or 'latest'} is in the py{pyver} pool, "
                        f"but conda-forge has no py{pyver} build of it"
                    )
    return problems


def check_sampled_picks(samples):
    """Every (Python, pin) pair the picker can emit must have a conda-forge build."""
    pairs = set()
    for pyver in picker.NUMPY_VERSIONS:
        for source in ("conda-forge", "wheel", "source", "upstream"):
            for _ in range(samples):
                picked = picker.pick_versions(pyver, source)
                pairs.add(("numpy", picked["numpy"], pyver))
                for dep in SAMPLED_DEPS:
                    pin = picked[dep].lstrip("=") if dep == "psg" else picked[dep]
                    if pin != "NA" and not (dep == "psg" and source != "conda-forge"):
                        pairs.add((dep, pin, pyver))
    problems = []
    for dep, pin, pyver in sorted(pairs):
        if not supports_python(dep, pin, pyver, _oldest_pin(dep)):
            problems.append(f"{dep} {pin or 'latest'} is picked for py{pyver} with no such build")
    return problems


def check_numpy_ceilings():
    """A ceiling table row must never sit above what conda-forge's builds allow.

    The table is what the picker treats as incompatible, so a row above the real cap
    lets through a pairing conda refuses. A row below it is merely conservative, which
    costs coverage rather than correctness, so that is reported separately.
    """
    problems = []
    for dep, table in (("scipy", picker.SCIPY_MAX_NUMPY), ("numba", picker.NUMBA_MAX_NUMPY)):
        floor = _oldest_pin(dep)
        for pin, ceiling in table.items():
            declared = numpy_cap(dep, pin, floor)
            if declared is None:
                continue  # no cap at all: the table can say what it likes
            if ceiling > declared:
                problems.append(
                    f"{dep} {pin} caps numpy at {declared} on conda-forge, "
                    f"but the table allows up to {ceiling}"
                )
    return problems


def note_conservative_ceilings():
    """Ceiling rows stricter than conda-forge's cap. Advisory: coverage left on the table."""
    notes = []
    for dep, table in (("scipy", picker.SCIPY_MAX_NUMPY), ("numba", picker.NUMBA_MAX_NUMPY)):
        floor = _oldest_pin(dep)
        for pin, ceiling in table.items():
            declared = numpy_cap(dep, pin, floor)
            if declared is not None and ceiling < declared:
                notes.append(
                    f"{dep} {pin} is capped at {ceiling} in the table, "
                    f"but conda-forge allows numpy up to {declared}"
                )
    return notes


def check_psg_python_floors():
    """PSG_MIN_PYTHON's conda-forge floor must be the oldest psg built for that Python."""
    problems = []
    floor = _oldest_pin("psg")
    for pyver, sources in picker.PSG_MIN_PYTHON.items():
        expected = sources.get("conda-forge")
        built = [p for p in picker.PSG_VERSIONS_NP2 if supports_python("psg", p, pyver, floor)]
        if built and min(built, key=picker._ver) != expected:
            problems.append(
                f"PSG_MIN_PYTHON says conda-forge py{pyver} starts at psg {expected}, "
                f"but {min(built, key=picker._ver)} has a py{pyver} build"
            )
    return problems


def check_documented_metadata():
    """Constraints the picker's rules quote must still be what conda-forge publishes."""
    problems = []
    for dep, pin, field, expected in DOCUMENTED_METADATA:
        matching = builds_for(dep, pin, _oldest_pin(dep))
        if not matching:
            problems.append(f"{dep} {pin} has no conda-forge build; cannot confirm {expected!r}")
            continue
        if not any(expected in build.get(field, []) for build in matching):
            problems.append(f"no {dep} {pin} build declares {field} {expected!r} any more")
    return problems


def report_newer_releases():
    """conda-forge versions newer than our newest pin. Informational, never a failure."""
    news = []
    for dep in (
        "numpy",
        "scipy",
        "pandas",
        "awkward",
        "numba",
        "networkx",
        "pyyaml",
        "sparse",
        "psg",
    ):
        floor = _oldest_pin(dep)
        newest_pin = max(
            (b["version"] for b in builds_for(dep, "", floor)), key=_version_key, default=None
        )
        tested = _newest_tested(dep)
        if not newest_pin or not tested:
            continue
        # A pin names a series, so "2.5" already covers conda-forge's 2.5.3
        in_tested_series = newest_pin == tested or newest_pin.startswith(f"{tested}.")
        if not in_tested_series and _version_key(newest_pin) > _version_key(tested):
            news.append(f"{dep}: newest pin is {tested}, conda-forge has {newest_pin}")
    return news


def _newest_tested(dep):
    pools = {
        "numpy": picker.NUMPY_VERSIONS,
        "scipy": {"": picker.SCIPY_VERSIONS["2.x"]},
        "pandas": {"": picker.PANDAS_VERSIONS["2.x"]},
        "awkward": {"": picker.AWKWARD_VERSIONS["2.x"]},
        "numba": {"": picker.NUMBA_VERSIONS["2.x"]},
        "networkx": picker.NETWORKX_VERSIONS,
        "pyyaml": picker.PYYAML_VERSIONS,
        "sparse": picker.SPARSE_VERSIONS,
        "psg": {"": picker.PSG_VERSIONS_NP2},
    }[dep]
    pins = {p for group in pools.values() for p in group if p and p != "NA"}
    return max(pins, key=picker._ver) if pins else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CONDA = shutil.which("conda")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="random picks per Python/source combination (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    args = parser.parse_args()
    if _CONDA is None:
        print("conda is not on PATH; this audit needs it", file=sys.stderr)
        return 2
    if args.seed is not None:
        random.seed(args.seed)

    checks = (
        ("per-Python pools", check_per_python_pools),
        ("sampled picks", lambda: check_sampled_picks(args.samples)),
        ("numpy ceiling tables", check_numpy_ceilings),
        ("psg Python floors", check_psg_python_floors),
        ("documented conda metadata", check_documented_metadata),
    )
    failures = 0
    for name, check in checks:
        problems = check()
        if problems:
            failures += len(problems)
            print(f"FAIL {name}:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"ok   {name}")

    if notes := note_conservative_ceilings():
        print("\nNotes (stricter than conda-forge needs; often right, since conda's depends")
        print("carry ABI caps while a package's real runtime cap lives upstream):")
        for note in notes:
            print(f"  - {note}")

    if news := report_newer_releases():
        print("\nNewer versions exist (not a failure; update the pools when convenient):")
        for line in news:
            print(f"  - {line}")

    print(f"\n{failures} problem(s) found", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

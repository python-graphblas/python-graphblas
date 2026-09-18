#!/usr/bin/env python
"""Audit the CI dependency pools in `ci_pick_versions.py` against PyPI.

`ci_pick_versions.py --validate` proves the picker agrees with the constraints it
documents. It cannot notice when a constraint has drifted from reality: a pin that
never had a build for a Python we test, or a ceiling table that a new release moved.
This script checks the pools and tables against what the packages actually declare.

The picker's rules are written for conda-forge, which is what CI installs from, so
treat PyPI as a fast proxy and `audit_conda_versions.py` as the authority. Anything
this script reports is worth looking at in both places.

Usage:
    python scripts/audit_pypi_versions.py [--samples N]

Requires network access and `packaging`. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Import the picker as a module so the pools stay the single source of truth
_PICKER = Path(__file__).with_name("ci_pick_versions.py")
_spec = importlib.util.spec_from_file_location("ci_pick_versions", _PICKER)
picker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(picker)

# Our internal names to PyPI project names (conda-forge calls psg python-suitesparse-graphblas)
PYPI_NAMES = {"pyyaml": "PyYAML", "psg": "suitesparse-graphblas"}

# Pools keyed by Python version, where every pin is offered on that Python and nowhere else
PER_PYTHON_POOLS = {
    "numpy": picker.NUMPY_VERSIONS,
    "networkx": picker.NETWORKX_VERSIONS,
    "pyyaml": picker.PYYAML_VERSIONS,
    "sparse": picker.SPARSE_VERSIONS,
}

# Deps a pick can carry, in the order the picker fills them in
SAMPLED_DEPS = ("scipy", "pandas", "awkward", "numba", "networkx", "sparse", "pyyaml", "psg")


# ---------------------------------------------------------------------------
# PyPI access
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}


def _pypi(path):
    """GET a PyPI JSON endpoint, cached for the life of the process."""
    if path not in _cache:
        url = f"https://pypi.org/pypi/{path}/json"
        request = urllib.request.Request(url, headers={"User-Agent": "python-graphblas-audit"})
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            _cache[path] = json.load(response)
    return _cache[path]


def releases(dep):
    """Every non-prerelease, non-yanked release of `dep`: {Version: {...}}."""
    key = f"releases:{dep}"
    if key not in _cache:
        data = _pypi(PYPI_NAMES.get(dep, dep))
        out = {}
        for raw, files in data["releases"].items():
            files = [f for f in files if not f["yanked"]]
            if not files:
                continue
            try:
                version = Version(raw)
            except InvalidVersion:
                continue
            if version.is_prerelease:
                continue
            out[version] = {
                "requires_python": files[0].get("requires_python"),
                "cpython": {int(m) for f in files for m in re.findall(r"cp3(\d+)", f["filename"])},
            }
        _cache[key] = out
    return _cache[key]


def in_series(dep, pin):
    """Every release matching a pin. An empty pin means "latest", so every release."""
    return [v for v in releases(dep) if not pin or str(v) == pin or str(v).startswith(f"{pin}.")]


def newest_in_series(dep, pin, pyver=None):
    """The release a pin resolves to, preferring one that `pyver` can install.

    Returns (version, installable). When no release in the series supports `pyver` the
    newest one is returned with installable=False, so callers can still read its
    metadata while reporting the gap.
    """
    found = in_series(dep, pin)
    if not found:
        return None, False
    if pyver is not None:
        usable = [v for v in found if supports_python(dep, v, pyver)]
        if usable:
            return max(usable), True
        return max(found), False
    return max(found), True


def supports_python(dep, version, pyver):
    """Whether `version` of `dep` can be installed on `pyver`.

    Projects that ship compiled wheels are judged by their CPython tags, which is the
    only signal that catches a release whose requires-python is wide but that simply
    never built for this interpreter. Pure-Python projects are judged by requires-python.
    """
    info = releases(dep)[version]
    if info["cpython"]:
        return int(pyver.split(".")[1]) in info["cpython"]
    requires = info["requires_python"]
    return not requires or Version(pyver) in SpecifierSet(requires)


def numpy_bound(dep, version, pyver):
    """The numpy requirement `version` of `dep` declares for `pyver`, if any.

    A project may list numpy more than once (numba 0.65 declares both `numpy>=1.22` and
    `numpy<2.5,>=1.22`), so combine every applicable clause rather than taking the first.
    """
    data = _pypi(f"{PYPI_NAMES.get(dep, dep)}/{version}")
    combined = None
    for raw in data["info"].get("requires_dist") or []:
        requirement = Requirement(raw)
        if requirement.name.lower() != "numpy":
            continue
        marker = requirement.marker
        if marker is None or marker.evaluate({"python_version": pyver, "extra": ""}):
            combined = (
                requirement.specifier if combined is None else combined & requirement.specifier
            )
    return combined


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_per_python_pools():
    """Every pin in a per-Python pool must exist on PyPI at all."""
    problems = []
    for dep, pools in PER_PYTHON_POOLS.items():
        for pyver, pins in pools.items():
            for pin in pins:
                if pin == "NA":
                    continue
                version, _ = newest_in_series(dep, pin)
                if version is None:
                    problems.append(f"{dep} {pin} in the py{pyver} pool has no release on PyPI")
    return problems


def note_pypi_python_gaps():
    """Pins with no PyPI build for a Python we offer them on. Advisory, not a failure.

    CI installs these from conda-forge, which builds more than PyPI ships: conda-forge
    has a py311 build of pyyaml 5.4.1 where PyPI's newest wheel is cp39. Treat a note
    here as a prompt to check `audit_conda_versions.py`, not as a broken pool.
    """
    notes = []
    for dep, pools in PER_PYTHON_POOLS.items():
        for pyver, pins in pools.items():
            for pin in pins:
                if pin == "NA":
                    continue
                version, installable = newest_in_series(dep, pin, pyver)
                if version is not None and not installable:
                    notes.append(
                        f"{dep} {pin or 'latest'} is offered on py{pyver}, "
                        f"but PyPI's {version} has no py{pyver} build"
                    )
    return notes


def check_ceiling_tables():
    """A ceiling row must never sit above the cap the release actually declares.

    A row above the real cap lets through a pairing that breaks; a row below it is
    conservative, which is often deliberate, so that is reported as a note instead.
    """
    problems = []
    tables = (("scipy", picker.SCIPY_MAX_NUMPY), ("numba", picker.NUMBA_MAX_NUMPY))
    for dep, table in tables:
        for pin, ceiling in table.items():
            version, _ = newest_in_series(dep, pin)
            if version is None:
                problems.append(f"{dep} {pin} has a ceiling row but no release on PyPI")
                continue
            # Read the bound on the oldest Python we test, where markers are least likely
            # to narrow it, then compare against the "< X.Y" the table records.
            specifier = numpy_bound(dep, version, min(picker.NUMPY_VERSIONS, key=picker._ver))
            declared = _upper_bound(specifier)
            if declared is not None and ceiling > declared:
                problems.append(
                    f"{dep} {pin} ({version}) caps numpy at {declared}, "
                    f"but the table allows up to {ceiling}"
                )
    return problems


def note_conservative_ceilings():
    """Ceiling rows stricter than PyPI's cap. Advisory: usually a conda-forge cap."""
    notes = []
    for dep, table in (("scipy", picker.SCIPY_MAX_NUMPY), ("numba", picker.NUMBA_MAX_NUMPY)):
        for pin, ceiling in table.items():
            version, _ = newest_in_series(dep, pin)
            if version is None:
                continue
            declared = _upper_bound(
                numpy_bound(dep, version, min(picker.NUMPY_VERSIONS, key=picker._ver))
            )
            if declared is not None and ceiling < declared:
                notes.append(
                    f"{dep} {pin} is capped at {ceiling} in the table, "
                    f"but PyPI's {version} allows numpy up to {declared}"
                )
    return notes


def _upper_bound(specifier):
    """The `<X.Y` in a specifier as a tuple, rounded up to the next minor for `<=`.

    The ceiling tables are exclusive bounds ("this pin breaks at numpy X.Y"), so a
    `<=X.Y` cap becomes X.(Y+1). Dropping `<=` instead would silently switch the check
    off for any release that writes its cap that way.
    """
    if specifier is None:
        return None
    for spec in specifier:
        if spec.operator in ("<", "<="):
            parts = Version(spec.version).release
            major, minor = parts[0], parts[1] if len(parts) > 1 else 0
            return (major, minor + 1) if spec.operator == "<=" else (major, minor)
    return None


def check_python_floor_tables():
    """SCIPY_MIN_PYTHON and NUMBA_MIN_PYTHON must match what the releases support."""
    problems = []
    pythons = sorted(picker.NUMPY_VERSIONS, key=picker._ver)

    for pin, floor in picker.SCIPY_MIN_PYTHON.items():
        version, _ = newest_in_series("scipy", pin)
        if version is None:
            problems.append(
                f"SCIPY_MIN_PYTHON has a row for scipy {pin}, which PyPI has no release of"
            )
            continue
        oldest = [p for p in pythons if supports_python("scipy", version, p)]
        recorded = ".".join(str(part) for part in floor)
        if oldest and oldest[0] != recorded:
            problems.append(
                f"SCIPY_MIN_PYTHON says scipy {pin} needs py{recorded}, "
                f"but {version} supports py{oldest[0]}"
            )

    for pyver, floor in picker.NUMBA_MIN_PYTHON.items():
        usable = [
            pin
            for pin in picker.NUMBA_MAX_NUMPY
            if (version := newest_in_series("numba", pin)[0])
            and supports_python("numba", version, pyver)
        ]
        recorded = ".".join(str(part) for part in floor)
        if usable and min(usable, key=picker._ver) != recorded:
            problems.append(
                f"NUMBA_MIN_PYTHON says py{pyver} needs numba {recorded}, "
                f"but {min(usable, key=picker._ver)} has a py{pyver} build"
            )
    return problems


def check_sampled_picks(samples):
    """Run the picker and check what it emits against each release's own metadata.

    Only two pins can truly conflict: conda backs an unpinned package off to something
    compatible, so an unpinned dep beside a pinned numpy is not a finding.
    """
    problems = set()
    for pyver in picker.NUMPY_VERSIONS:
        for source in ("conda-forge", "wheel", "source", "upstream"):
            for _ in range(samples):
                picked = picker.pick_versions(pyver, source)
                problems.update(_audit_pick(picked, pyver, source))
    return sorted(problems)


def _audit_pick(picked, pyver, source):
    """Check one pick. psg is the only dependency CI installs from PyPI."""
    problems = []
    numpy_version, _ = newest_in_series("numpy", picked["numpy"], pyver)
    if numpy_version is None:
        return [f"numpy {picked['numpy'] or 'latest'} has no release on PyPI"]
    for dep in SAMPLED_DEPS:
        pin = picked[dep].lstrip("=") if dep == "psg" else picked[dep]
        if pin == "NA" or (dep == "psg" and source in ("conda-forge", "upstream")):
            continue
        version, installable = newest_in_series(dep, pin, pyver)
        if version is None:
            problems.append(f"{dep} {pin or 'latest'} has no release on PyPI")
            continue
        if not installable and dep == "psg":
            # wheel builds pip-install psg, so a missing wheel is a hard failure
            problems.append(
                f"psg {pin} is picked for py{pyver} {source}, but {version} has no py{pyver} wheel"
            )
            continue
        if not pin or not picked["numpy"]:
            continue
        specifier = numpy_bound(dep, version, pyver)
        if specifier is not None and numpy_version not in specifier:
            problems.append(
                f"{dep} {version} requires numpy{specifier}, "
                f"got numpy {numpy_version} (pin {picked['numpy']})"
            )
    return problems


def report_newer_releases():
    """List releases newer than the newest pin we test. Informational, never a failure."""
    news = []
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
    }
    for dep, pool in pools.items():
        pins = {p for group in pool.values() for p in group if p and p != "NA"}
        newest_pin = max(pins, key=picker._ver)
        resolved, _ = newest_in_series(dep, newest_pin)
        latest = max(releases(dep))
        if resolved is not None and latest > resolved and not str(latest).startswith(newest_pin):
            news.append(f"{dep}: newest pin is {newest_pin} ({resolved}), PyPI has {latest}")
    return news


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    if args.seed is not None:
        random.seed(args.seed)

    checks = (
        ("per-Python pools", check_per_python_pools),
        ("numpy ceiling tables", check_ceiling_tables),
        ("Python floor tables", check_python_floor_tables),
        ("sampled picks", lambda: check_sampled_picks(args.samples)),
    )
    failures = 0
    for name, check in checks:
        try:
            problems = check()
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"SKIP {name}: PyPI unreachable ({exc})", file=sys.stderr)
            continue
        if problems:
            failures += len(problems)
            print(f"FAIL {name}:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"ok   {name}")

    try:
        notes = note_conservative_ceilings() + note_pypi_python_gaps()
        news = report_newer_releases()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"SKIP notes: PyPI unreachable ({exc})", file=sys.stderr)
        notes = news = []

    if notes:
        print("\nNotes (PyPI only; conda-forge builds more, so check the conda audit):")
        for note in notes:
            print(f"  - {note}")

    if news:
        print("\nNewer releases exist (not a failure; update the pools when convenient):")
        for line in news:
            print(f"  - {line}")

    print(f"\n{failures} problem(s) found", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

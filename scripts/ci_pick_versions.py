#!/usr/bin/env python
"""Pick random dependency versions for CI testing.

Randomly selects compatible dependency versions for python-graphblas CI.
Replaces the bash-based version selection in test_and_build.yml.

Usage (in GitHub Actions workflow):
    eval "$(python scripts/ci_pick_versions.py --python 3.12 --source conda-forge)"

Output: bash-eval-safe key=value lines using the same variable names as the workflow:
    npver='=2.2'     # conda pin
    spver='=1.15'
    npver=''          # empty = latest (no pin)
    sparsever='NA'    # NA = skip this package
"""

import argparse
import random
import sys

# ---------------------------------------------------------------------------
# Version pools: which versions we want to test per package.
# "" means "latest" (no pin). "NA" means "don't install".
#
# When updating versions here, also update scripts/check_versions.sh
# ---------------------------------------------------------------------------

NUMPY_VERSIONS = {
    "3.11": ["1.24", "1.25", "1.26", "2.0", "2.1", "2.2", "2.3", "2.4", ""],
    "3.12": ["1.26", "2.0", "2.1", "2.2", "2.3", "2.4", ""],
    "3.13": ["2.1", "2.2", "2.3", "2.4", ""],
    "3.14": ["2.3", "2.4", ""],
}

# Deps that depend on numpy version (1.x vs 2.x path).
# Per-Python sublists narrow to versions with available conda builds.
SCIPY_VERSIONS = {
    "1.x": {
        "3.11": ["1.9", "1.10", "1.11", "1.12", "1.13", "1.14", ""],
        "3.12": ["1.11", "1.12", "1.13", "1.14", ""],
    },
    "2.x": ["1.13", "1.14", "1.15", "1.16", "1.17", ""],
}

PANDAS_VERSIONS = {
    "1.x": {
        "3.11": ["1.5", "2.0", "2.1", "2.2", "2.3", ""],
        "3.12": ["2.1", "2.2", "2.3", ""],
    },
    "2.x": ["2.2", "2.3", "3.0", ""],
}

AWKWARD_VERSIONS = {
    "1.x": {
        "3.11": ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", ""],
        "3.12": ["2.4", "2.5", "2.6", "2.7", "2.8", "2.9", ""],
    },
    "2.x": ["2.6", "2.7", "2.8", "2.9", ""],
}

NUMBA_VERSIONS = {
    "1.x": ["0.57", "0.58", "0.59", "0.60", "0.61", ""],
    "2.x": ["0.62", "0.63", "0.64", ""],
}

# Deps that only depend on Python version (not numpy)
NETWORKX_VERSIONS = {
    "3.11": ["2.8", "3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", ""],
    "3.12": ["3.2", "3.3", "3.4", "3.5", "3.6", ""],
    "3.13": ["3.4", "3.5", "3.6", ""],
    "3.14": ["3.6", ""],
}

PYYAML_VERSIONS = {
    "3.11": ["5.4", "6.0", ""],
    "3.12": ["6.0", ""],
    "3.13": ["6.0", ""],
    "3.14": ["6.0", ""],
}

SPARSE_VERSIONS = {
    "3.11": ["0.14", "0.15", ""],
    "3.12": ["0.14", "0.15", ""],
    "3.13": "NA",
    "3.14": "NA",
}

# PSG versions by numpy branch and source type.
# conda-forge uses "=" prefix, wheel/source use "==".
PSG_VERSIONS_NP1 = {
    "conda-forge": {
        "3.11": [
            "7.4.0",
            "7.4.1",
            "7.4.2",
            "7.4.3.0",
            "7.4.3.1",
            "7.4.3.2",
            "8.0.2.1",
            "8.2.0.1",
            "8.2.1.0",
        ],
        "3.12": ["8.2.0.1", "8.2.1.0"],
    },
    "wheel": {
        "3.11": ["7.4.3.2", "8.0.2.1", "8.2.0.1", "8.2.1.0"],
        "3.12": ["8.2.0.1", "8.2.1.0"],
    },
    "source": {
        "3.11": [
            "7.4.0.0",
            "7.4.1.0",
            "7.4.2.0",
            "7.4.3.0",
            "7.4.3.1",
            "7.4.3.2",
            "8.0.2.1",
            "8.2.0.1",
            "8.2.1.0",
        ],
        "3.12": ["8.2.0.1", "8.2.1.0"],
    },
}

PSG_VERSIONS_NP2 = {
    "no_py314": ["9.3.1.0", "9.4.5.0", "10.0.1.1", "10.1.1.0", "10.3.1.0"],
    "py314_only": ["10.0.1.1", "10.1.1.0", "10.3.1.0"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ver(s):
    """Parse version string to tuple for comparison. "" means latest (very large)."""
    if s in ("", "NA"):
        return (9999,)
    return tuple(int(x) for x in s.split("."))


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def apply_constraints(v, pyver, scipy_pool, numba_pool):
    """Mutate version dict to satisfy all known compatibility constraints.

    Each constraint comment documents the real-world requirement it encodes.
    Order matters: numpy/scipy constraints first, then pandas (which may bump scipy/numba),
    then Python-version constraints, then numba/numpy constraints.
    """
    # --- scipy / numpy constraints ---

    # scipy >=1.15 requires numpy >=1.26.4
    if v["numpy"] in ("1.24", "1.25") and _ver(v["scipy"]) >= (1, 15):
        candidates = [s for s in scipy_pool if s and _ver(s) < (1, 15)]
        v["scipy"] = random.choice(candidates) if candidates else "1.14"

    # scipy <1.13 doesn't support numpy 2.x (safety net)
    np_is_1x = v["numpy"].startswith("1.") if v["numpy"] else False
    if not np_is_1x and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 13):
        v["scipy"] = random.choice([s for s in scipy_pool if _ver(s) >= (1, 13)])

    # scipy <1.15.1 requires numpy <2.3; scipy <1.16 requires numpy <2.5
    if _ver(v["numpy"]) >= (2, 3):
        if v["scipy"] in ("1.13", "1.14"):
            v["scipy"] = random.choice(["1.16", "1.17", ""])
        elif v["scipy"] == "1.15":
            v["scipy"] = random.choice(["1.15", "1.16", "1.17", ""])

    # numpy 1.26 + scipy 1.9 conflict
    if v["numpy"] == "1.26" and v["scipy"] == "1.9":
        v["scipy"] = random.choice(["1.10", "1.11", ""])

    # --- scipy / Python version availability ---

    # scipy <1.14 has no py3.13 builds; scipy <1.16 has no py3.14 builds
    if pyver == "3.14" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        v["scipy"] = random.choice(["1.16", "1.17", ""])
    elif pyver == "3.13" and v["scipy"] == "1.13":
        v["scipy"] = random.choice(["1.14", "1.15", "1.16", "1.17", ""])

    # --- pandas constraints ---

    # pandas <2.3 has no py3.14 builds
    if pyver == "3.14" and v["pandas"] == "2.2":
        v["pandas"] = random.choice(["2.3", "3.0", ""])

    # pandas 3.0 requires numba >=0.60 and scipy >=1.14.1
    if v["pandas"] == "3.0":
        if v["numba"] not in ("", "NA") and _ver(v["numba"]) < (0, 60):
            v["numba"] = "0.60"
        if v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
            v["scipy"] = random.choice(["1.15", "1.16", "1.17", ""])

    # --- awkward / Python version availability ---

    # awkward <2.7 has no py3.13 builds; awkward <2.8 has no py3.14 builds
    if pyver == "3.14" and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
        v["awkward"] = random.choice(["2.8", "2.9", ""])
    elif pyver == "3.13" and v["awkward"] == "2.6":
        v["awkward"] = random.choice(["2.7", "2.8", "2.9", ""])

    # --- numba constraints ---

    # numba minimum by Python version: 0.59 for 3.12, 0.61 for 3.13, 0.63 for 3.14
    numba_min = {"3.11": (0, 57), "3.12": (0, 59), "3.13": (0, 61), "3.14": (0, 63)}
    if v["numba"] not in ("", "NA"):
        min_ver = numba_min[pyver]
        if _ver(v["numba"]) < min_ver:
            pool = [n for n in numba_pool if _ver(n) >= min_ver]
            v["numba"] = random.choice(pool) if pool else ""

    # numba <0.64 requires numpy <2.4
    if _ver(v["numpy"]) >= (2, 4) and v["numba"] in ("0.62", "0.63"):
        v["numba"] = "0.64"

    # numba <0.62 doesn't support numpy 2.x
    if not np_is_1x and v["numba"] not in ("", "NA") and _ver(v["numba"]) < (0, 62):
        v["numba"] = "NA"

    # --- sparse ---

    # sparse doesn't support Python 3.13+
    if pyver in ("3.13", "3.14"):
        v["sparse"] = "NA"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def pick_versions(pyver, source_type):
    """Pick random compatible dependency versions.

    Parameters
    ----------
    pyver : str
        Python version like "3.12"
    source_type : str
        One of "conda-forge", "wheel", "source", "upstream"

    Returns
    -------
    dict
        Package name -> version string ("" = latest, "NA" = skip)
    """
    # Step 1: Pick numpy
    numpy_pool = NUMPY_VERSIONS[pyver]
    npver = random.choice(numpy_pool)

    # Upstream needs numpy 2
    if source_type == "upstream" and npver.startswith("1."):
        npver = random.choice([v for v in numpy_pool if not v.startswith("1.")] or [""])

    np_is_1x = npver.startswith("1.") if npver else False

    # Step 2: Pick numpy-dependent deps
    if np_is_1x:
        scipy_pool = SCIPY_VERSIONS["1.x"].get(pyver, SCIPY_VERSIONS["1.x"]["3.11"])
        pandas_pool = PANDAS_VERSIONS["1.x"].get(pyver, PANDAS_VERSIONS["1.x"]["3.11"])
        awkward_pool = AWKWARD_VERSIONS["1.x"].get(pyver, AWKWARD_VERSIONS["1.x"]["3.11"])
        numba_pool = NUMBA_VERSIONS["1.x"]
    else:
        scipy_pool = SCIPY_VERSIONS["2.x"]
        pandas_pool = PANDAS_VERSIONS["2.x"]
        awkward_pool = AWKWARD_VERSIONS["2.x"]
        numba_pool = NUMBA_VERSIONS["2.x"]

    v = {
        "numpy": npver,
        "scipy": random.choice(scipy_pool),
        "pandas": random.choice(pandas_pool),
        "awkward": random.choice(awkward_pool),
        "numba": random.choice(numba_pool),
        "networkx": random.choice(NETWORKX_VERSIONS[pyver]),
        "pyyaml": random.choice(PYYAML_VERSIONS[pyver]),
        "sparse": _pick_scalar_or_list(SPARSE_VERSIONS[pyver]),
    }

    # Source builds have issues with some numpy/scipy/pandas versions;
    # blank them before constraints so numba/etc constraints see the right numpy.
    if source_type == "source":
        v["numpy"] = ""
        v["scipy"] = ""
        v["pandas"] = ""

    # Step 3: Apply compatibility constraints
    apply_constraints(v, pyver, scipy_pool, numba_pool)

    # Step 4: Pick psg version
    v["psg"] = _pick_psg(v["numpy"], pyver, source_type)

    return v


def _pick_scalar_or_list(pool):
    """Handle pools that are either "NA" (string) or a list of choices."""
    return pool if isinstance(pool, str) else random.choice(pool)


def _pick_psg(npver, pyver, source_type):
    """Pick python-suitesparse-graphblas version."""
    if source_type == "upstream":
        return ""

    np_is_1x = npver.startswith("1.") if npver else False
    eq = "=" if source_type == "conda-forge" else "=="

    if np_is_1x:
        pool = PSG_VERSIONS_NP1.get(source_type, {}).get(pyver, [])
        if not pool:
            return ""
        return f"{eq}{random.choice(pool)}"
    pool = PSG_VERSIONS_NP2["py314_only"] if pyver == "3.14" else PSG_VERSIONS_NP2["no_py314"]
    return random.choice([f"{eq}{ver}" for ver in pool] + [""])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# Map internal names to the short variable names used in the workflow
_VAR_NAMES = {
    "numpy": "npver",
    "scipy": "spver",
    "pandas": "pdver",
    "awkward": "akver",
    "networkx": "nxver",
    "pyyaml": "yamlver",
    "sparse": "sparsever",
    "numba": "numbaver",
    "psg": "psgver",
}

_SUMMARY_NAMES = {
    "numpy": "np",
    "scipy": "sp",
    "pandas": "pd",
    "awkward": "ak",
    "networkx": "nx",
    "pyyaml": "yaml",
    "sparse": "sparse",
    "numba": "numba",
    "psg": "psg",
}


def format_output(v):
    """Format version dict as bash-eval-safe key=value lines.

    Values use '=X.Y' prefix for conda install (e.g., npver='=2.2').
    Empty means latest, NA means skip.
    psg already has its prefix baked in.
    """
    lines = []
    for key in ("numpy", "scipy", "pandas", "awkward", "networkx", "pyyaml", "sparse", "numba"):
        var = _VAR_NAMES[key]
        val = v[key]
        if val in ("NA", ""):
            lines.append(f"{var}='{val}'")
        else:
            lines.append(f"{var}='={val}'")

    # psg already has = or == prefix
    lines.append(f"psgver='{v['psg']}'")
    return "\n".join(lines)


def format_summary(v):
    """One-line summary for CI log."""
    parts = []
    for key in (
        "numpy",
        "scipy",
        "pandas",
        "awkward",
        "networkx",
        "numba",
        "pyyaml",
        "sparse",
        "psg",
    ):
        name = _SUMMARY_NAMES[key]
        val = v[key]
        if val == "NA":
            parts.append(f"{name}=NA")
        elif val == "":
            parts.append(f"{name}=latest")
        else:
            parts.append(f"{name}={val}")
    return "versions: " + " ".join(parts)


# ---------------------------------------------------------------------------
# Validation (for testing the script itself)
# ---------------------------------------------------------------------------


def validate(v, pyver):
    """Check that a version combination satisfies all constraints. Returns list of errors."""
    errors = []
    np_is_1x = v["numpy"].startswith("1.") if v["numpy"] else False

    # scipy >=1.15 requires numpy >=1.26.4
    if v["numpy"] in ("1.24", "1.25") and _ver(v["scipy"]) >= (1, 15):
        errors.append(f"scipy {v['scipy']} requires numpy >=1.26.4, got {v['numpy']}")

    # scipy <1.13 requires numpy 1.x
    if not np_is_1x and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 13):
        errors.append(f"scipy {v['scipy']} doesn't support numpy 2.x")

    # scipy <1.15.1 requires numpy <2.3
    if _ver(v["numpy"]) >= (2, 3) and v["scipy"] in ("1.13", "1.14"):
        errors.append(f"scipy {v['scipy']} requires numpy <2.3, got {v['numpy']}")

    # numpy 1.26 + scipy 1.9
    if v["numpy"] == "1.26" and v["scipy"] == "1.9":
        errors.append("numpy 1.26 + scipy 1.9 conflict")

    # scipy Python availability
    if pyver == "3.14" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        errors.append(f"scipy {v['scipy']} has no py3.14 build")
    if pyver == "3.13" and v["scipy"] == "1.13":
        errors.append("scipy 1.13 has no py3.13 build")

    # pandas Python availability
    if pyver == "3.14" and v["pandas"] == "2.2":
        errors.append("pandas 2.2 has no py3.14 build")

    # pandas 3.0 requirements
    if v["pandas"] == "3.0":
        if v["numba"] not in ("", "NA") and _ver(v["numba"]) < (0, 60):
            errors.append(f"pandas 3.0 requires numba >=0.60, got {v['numba']}")
        if v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
            errors.append(f"pandas 3.0 requires scipy >=1.14.1, got {v['scipy']}")

    # awkward Python availability
    if pyver == "3.14" and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
        errors.append(f"awkward {v['awkward']} has no py3.14 build")
    if pyver == "3.13" and v["awkward"] == "2.6":
        errors.append("awkward 2.6 has no py3.13 build")

    # numba Python minimums
    numba_min = {"3.11": (0, 57), "3.12": (0, 59), "3.13": (0, 61), "3.14": (0, 63)}
    if v["numba"] not in ("", "NA") and _ver(v["numba"]) < numba_min[pyver]:
        errors.append(f"numba {v['numba']} doesn't support Python {pyver}")

    # numba <0.64 requires numpy <2.4
    if v["numba"] in ("0.62", "0.63") and _ver(v["numpy"]) >= (2, 4):
        errors.append(f"numba {v['numba']} requires numpy <2.4, got {v['numpy']}")

    # numba <0.62 requires numpy 1.x
    if not np_is_1x and v["numba"] not in ("", "NA") and _ver(v["numba"]) < (0, 62):
        errors.append(f"numba {v['numba']} doesn't support numpy 2.x")

    # sparse Python availability
    if pyver in ("3.13", "3.14") and v["sparse"] != "NA":
        errors.append(f"sparse doesn't support Python {pyver}")

    return errors


def stress_test(n=10000):
    """Run n random picks for each Python/source combo and validate all."""
    total = 0
    failures = 0
    for pyver in NUMPY_VERSIONS:
        for source in ("conda-forge", "wheel", "source", "upstream"):
            for _ in range(n):
                v = pick_versions(pyver, source)
                errs = validate(v, pyver)
                total += 1
                if errs:
                    failures += 1
                    print(f"FAIL py{pyver} {source}: {v}", file=sys.stderr)
                    for e in errs:
                        print(f"  - {e}", file=sys.stderr)
    print(f"Stress test: {total} combos, {failures} failures", file=sys.stderr)
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Pick random CI dependency versions")
    parser.add_argument("--python", help="Python version (e.g. 3.12)")
    parser.add_argument(
        "--source",
        choices=["conda-forge", "wheel", "source", "upstream"],
        help="Package source type",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--validate", action="store_true", help="Run stress test to validate all constraints"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.validate:
        failures = stress_test()
        sys.exit(1 if failures else 0)

    if not args.python or not args.source:
        parser.error("--python and --source are required (unless --validate)")

    pyver = args.python
    if pyver not in NUMPY_VERSIONS:
        print(f"Error: unsupported Python version {pyver}", file=sys.stderr)
        print(f"Supported: {', '.join(NUMPY_VERSIONS.keys())}", file=sys.stderr)
        sys.exit(1)

    v = pick_versions(pyver, args.source)

    # Print summary to stderr (visible in CI logs)
    print(format_summary(v), file=sys.stderr)

    # Print bash-eval-safe key=value to stdout
    print(format_output(v))


if __name__ == "__main__":
    main()

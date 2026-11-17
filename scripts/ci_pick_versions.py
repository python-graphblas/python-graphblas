#!/usr/bin/env python
"""Pick random, compatible dependency versions for python-graphblas CI.

Usage (as in test_and_build.yml, which checks the exit status before the eval):
    picks=$(python scripts/ci_pick_versions.py --python 3.12 --source conda-forge)
    eval "$picks"

Output: key=value lines that bash can eval, with the variable names the workflow uses:
    npver='=2.2'      # a conda pin
    pdver='<3.1'      # latest, below a cap (LATEST_CEILINGS)
    numbaver=''       # latest (no pin)
    sparsever='NA'    # do not install

``--validate`` checks 10,000 random picks for each Python and source against the rules.
"""

import argparse
import random
import sys

# ---------------------------------------------------------------------------
# Version pools: which versions we want to test per package.
# "" means "latest" (no pin). "NA" means "don't install".
#
# When updating versions here, also update scripts/check_versions.sh
#
# To add a Python version (3.16, say): add it to `choices` in the "RNG for Python
# version" step of test_and_build.yml, give it a row in every table keyed by Python
# below (`check_python_pools` fails the run if you miss one), and revisit the floors
# in `apply_constraints`. Those floors are written as `_ver(pyver) >= (3, N)`, so a
# new Python inherits the previous one's floors instead of escaping them. That is
# enough to keep CI honest, but usually still too loose to install, so check each dep.
# ---------------------------------------------------------------------------

# conda-forge's first py3.15 builds are numpy 2.5.2, which the "2.5" pin resolves to.
NUMPY_VERSIONS = {
    "3.12": ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5", ""],
    "3.13": ["2.1", "2.2", "2.3", "2.4", "2.5", ""],
    "3.14": ["2.3", "2.4", "2.5", ""],
    "3.15": ["2.5", ""],
}

# scipy 1.13 is the first release built for numpy 2. One pool serves every Python, and
# `apply_constraints` re-picks the pins a Python has no build for.
SCIPY_VERSIONS = ["1.13", "1.14", "1.15", "1.16", "1.17", "1.18", ""]

# The numpy ceiling each scipy pin declares, the same idea as NUMBA_MAX_NUMPY below:
# scipy caps `numpy <X`, so a pinned pair that violates it fails the conda solve.
# Rows hold the stricter of what PyPI and conda-forge declare, since either one can be
# the binding constraint. Every pin in SCIPY_VERSIONS needs a row here (`check_ceilings`
# enforces it).
# MAINT: 2026-09-16 scripts/audit_pypi_versions.py and audit_conda_versions.py check these
SCIPY_MAX_NUMPY = {
    "1.13": (2, 3),
    "1.14": (2, 3),
    "1.15": (2, 5),
    "1.16": (2, 6),
    "1.17": (2, 7),
    "1.18": (2, 8),
}

# pandas 2.2.2 is the first release built for numpy 2; the "2.2" pin resolves to 2.2.3.
PANDAS_VERSIONS = ["2.2", "2.3", "3.0", ""]

# Caps on an unpinned ("" = latest) pick, as conda specs. Pins in the pools are not
# affected; a capped package's newest release stays out of CI until the cap goes.
# MAINT: 2026-10-22 pandas 3.1.0 fits fewer columns in a repr's auto-width
# (display.max_columns=0), and warns (Pandas4Warning) when the pandas-based repr renderer
# appends its "..." row to a numeric column; both fail test_formatting.py. Remove the cap
# once reprs render without pandas
# (https://github.com/python-graphblas/python-graphblas/pull/605).
LATEST_CEILINGS = {"pandas": "<3.1"}

# The pool starts at 2.6 because awkward <2.6 still uses numpy.AxisError, which numpy 2.0
# removed. Only >=2.10 pins survive numpy >=2.5; see the awkward constraints in
# `apply_constraints`.
AWKWARD_VERSIONS = ["2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14", ""]

NUMBA_VERSIONS = ["0.61", "0.62", "0.63", "0.64", "0.65", "0.66", "0.67", "0.68", ""]

# Oldest numba pin that supports each Python.
NUMBA_MIN_PYTHON = {
    "3.12": (0, 61),
    "3.13": (0, 61),
    "3.14": (0, 63),
    "3.15": (0, 68),
}

# The numpy ceiling each numba pin declares. numba caps `numpy <X` and refuses to import
# against anything newer. conda-forge carries that cap in `constrains` rather than
# `depends`, whose cap is only the numpy ABI, so a pinned pair that violates it fails
# the conda solve outright. Unpinned numba ("") installs the newest release, so it takes
# the newest row: numba is the dependency that most often lags a new numpy, and a pinned
# numpy that has outrun every numba would otherwise fail the solve. Every pin in
# NUMBA_VERSIONS needs a row here (`check_ceilings` fails the run if one is missing).
# MAINT: 2026-09-16 scripts/audit_pypi_versions.py and audit_conda_versions.py check these
NUMBA_MAX_NUMPY = {
    "0.61": (2, 3),
    "0.62": (2, 4),
    "0.63": (2, 4),
    "0.64": (2, 5),
    "0.65": (2, 5),
    "0.66": (2, 5),
    "0.67": (2, 6),
    "0.68": (2, 6),
}

# Deps that only depend on Python version (not numpy)
NETWORKX_VERSIONS = {
    "3.12": ["3.2", "3.3", "3.4", "3.5", "3.6", "3.7", ""],
    "3.13": ["3.4", "3.5", "3.6", "3.7", ""],
    "3.14": ["3.6", "3.7", ""],
    "3.15": ["3.6", "3.7", ""],
}

PYYAML_VERSIONS = {
    "3.12": ["6.0", ""],
    "3.13": ["6.0", ""],
    "3.14": ["6.0", ""],
    "3.15": ["6.0", ""],
}

# sparse is noarch and caps neither Python nor numpy: every release here declares numba
# >=0.49, numpy >=1.17 and a Python floor at or below our oldest (>=3.8 for 0.14/0.15,
# >=3.10 for 0.16/0.17, >=3.11 for 0.18/0.19), so every pin installs on every Python and
# the rows are all the same. It depends on numba, so the workflow drops sparse wherever it
# skips numba. To skip sparse on a Python, give it a ["NA"] pool: a bare "NA" would be
# indexed as a string by `random.choice`.
# MAINT: 2026-09-26 sparse 0.19.2 is latest. Each pin, as conda resolves it, passes the
# pydata-sparse tests on py3.13 and py3.14 with the oldest and newest numpy/numba pair
# each one draws.
SPARSE_VERSIONS = {
    "3.12": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
    "3.13": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
    "3.14": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
    "3.15": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
}

# psg versions, oldest first. 9.3.1.0 is the first release built for numpy 2 (every psg
# 8.x build requires numpy <2). Every source type except "upstream" (psg from git) draws
# from this one pool; Pythons too new for the older releases draw from a tail of it (see
# PSG_MIN_PYTHON).
PSG_VERSIONS = [
    "9.3.1.0",
    "9.4.5.0",
    "10.0.1.1",
    "10.1.1.0",
    "10.3.1.0",
    "10.4.0.0",
    "10.4.1.0",
    "10.5.0.0",
    "10.5.1.1",
    "10.5.2.0",
]

# Oldest psg release that installs on a given Python, by where psg comes from.
# Pythons older than every entry here (3.12 and 3.13) can use the whole pool; a Python
# newer than every entry inherits the newest one until it gets a row of its own.
#
# py3.14 needs a different floor per source: conda-forge has py314 builds from
# 10.0.1.1 on, but PyPI's first cp314 wheels are in 10.3.1.0, so the "wheel" path
# can't reach back as far. "source" builds the sdist, and the sdists before 10.3.1.0
# predate psg's py3.14 support (they still declare requires-python >=3.9), so hold
# them to the wheel floor rather than test a build psg never claimed works.
#
# py3.15 starts at 10.5.1.1 everywhere: it is psg's first release with py315 builds on
# conda-forge and cp315 wheels on PyPI, and its sdist is the first from after psg added
# Python 3.15 support.
# MAINT: 2026-10-28 checked conda-forge build strings and PyPI wheel tags for psg 9-10
PSG_MIN_PYTHON = {
    "3.14": {"conda-forge": "10.0.1.1", "wheel": "10.3.1.0", "source": "10.3.1.0"},
    "3.15": {"conda-forge": "10.5.1.1", "wheel": "10.5.1.1", "source": "10.5.1.1"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ver(s):
    """Parse version string to tuple for comparison. "" means latest (very large)."""
    if s in ("", "NA"):
        return (9999,)
    return tuple(int(x) for x in s.split("."))


def _numba_ceiling(pin):
    """The numpy ceiling a numba pin declares, or None when numba is skipped."""
    if pin == "NA":
        return None
    return max(NUMBA_MAX_NUMPY.values()) if pin == "" else NUMBA_MAX_NUMPY[pin]


def _pick_scipy(npver, floor=(0,)):
    """Pick a scipy pin at or above `floor` that this numpy can use.

    Filters the pool by the pin's numpy ceiling (SCIPY_MAX_NUMPY), so every re-pick site
    gets the same answer no matter where it sits in the constraint order. "" (latest) is
    always a candidate: an unpinned scipy is one conda resolves itself, so it satisfies
    every rule here by construction.
    """
    return random.choice(
        [s for s in SCIPY_VERSIONS if s and _ver(s) >= floor and SCIPY_MAX_NUMPY[s] > npver] + [""]
    )


def _numpy_ver(v, pyver):
    """Version of numpy to compare against, resolving "" (latest) to what it installs.

    An unpinned numpy is taken to be the newest pin in this Python's pool (2.5 today), not
    something infinitely new, so a numba that caps numpy just above it (0.67, at <2.6)
    still qualifies. Deriving it from the pool gives each Python its own answer and leaves
    nothing extra to bump.
    """
    if v["numpy"]:
        return _ver(v["numpy"])
    return max(_ver(n) for n in NUMPY_VERSIONS[pyver] if n)


def _min_for_python(table, pyver):
    """Look up `pyver` in a {python version: minimum} table, or None if it predates it.

    A Python newer than every row inherits the newest row: it needs at least as new a
    build as the newest Python we have characterized. That guess is conservative rather
    than correct, so give a new Python its own row once its real floors are known.
    """
    if pyver in table:
        return table[pyver]
    newest = max(table, key=_ver)
    return table[newest] if _ver(pyver) > _ver(newest) else None


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def apply_constraints(v, pyver):
    """Mutate version dict to satisfy all known compatibility constraints.

    Order matters: numpy/scipy constraints first, then pandas (which may bump scipy), then
    awkward, then numba/numpy constraints. Within the awkward section, the
    Python-availability picks run first, and the awkward >=2.10 rule runs last so it wins.
    """
    # Nothing below changes the numpy pick, so derive this once.
    npver = _numpy_ver(v, pyver)

    # --- scipy / numpy constraints ---

    # scipy's numpy ceiling (see SCIPY_MAX_NUMPY): each pin against the numpy that
    # outgrew it.
    if (ceiling := SCIPY_MAX_NUMPY.get(v["scipy"])) is not None and npver >= ceiling:
        v["scipy"] = _pick_scipy(npver)

    # --- scipy / Python version availability ---

    # scipy <1.14 has no py3.13 builds, <1.16 none for py3.14, and <1.18 none for py3.15
    if _ver(pyver) >= (3, 15) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 18):
        v["scipy"] = _pick_scipy(npver, floor=(1, 18))
    elif _ver(pyver) >= (3, 14) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        v["scipy"] = _pick_scipy(npver, floor=(1, 16))
    elif _ver(pyver) >= (3, 13) and v["scipy"] == "1.13":
        v["scipy"] = _pick_scipy(npver, floor=(1, 14))

    # --- pandas constraints ---

    # pandas <2.3 has no py3.14 builds, and <3.0 none for py3.15
    if _ver(pyver) >= (3, 15) and v["pandas"] not in ("", "NA") and _ver(v["pandas"]) < (3, 0):
        v["pandas"] = random.choice(["3.0", ""])
    elif _ver(pyver) >= (3, 14) and v["pandas"] == "2.2":
        v["pandas"] = random.choice(["2.3", "3.0", ""])

    # pandas 3.0 requires scipy >=1.14.1. It also requires numba >=0.60, which needs no
    # rule: the numba pool starts at 0.61.
    if v["pandas"] == "3.0" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
        v["scipy"] = _pick_scipy(npver, floor=(1, 15))

    # --- awkward / Python version availability ---

    # awkward <2.7 has no py3.13 builds, <2.8 none for py3.14, and <2.13 none for py3.15
    # (awkward is noarch, but each release pins one awkward-cpp, and awkward-cpp 56 is the
    # first with py315 builds)
    if _ver(pyver) >= (3, 15) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 13):
        v["awkward"] = random.choice([a for a in AWKWARD_VERSIONS if _ver(a) >= (2, 13)])
    elif _ver(pyver) >= (3, 14) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
        v["awkward"] = random.choice(["2.8", "2.9", ""])
    elif _ver(pyver) >= (3, 13) and v["awkward"] == "2.6":
        v["awkward"] = random.choice(["2.7", "2.8", "2.9", ""])

    # --- awkward / numpy 2.5 deprecations ---

    # awkward <2.10 builds numpy.datetime64("NaT") with a generic unit at import time,
    # which numpy 2.5 deprecates; `filterwarnings = error` in pyproject.toml then turns the
    # import into a collection error. Unpinned numpy is latest, which is >=2.5.
    # MAINT: 2026-08-04 confirmed with awkward 2.8.12 and 2.9.0 + numpy 2.5
    if (
        _ver(v["numpy"]) >= (2, 5)
        and v["awkward"] not in ("", "NA")
        and _ver(v["awkward"]) < (2, 10)
    ):
        v["awkward"] = random.choice([a for a in AWKWARD_VERSIONS if _ver(a) >= (2, 10)])

    # --- numba constraints ---

    # numba minimum by Python version (see NUMBA_MIN_PYTHON)
    if v["numba"] not in ("", "NA"):
        min_ver = _min_for_python(NUMBA_MIN_PYTHON, pyver)
        if _ver(v["numba"]) < min_ver:
            pool = [n for n in NUMBA_VERSIONS if _ver(n) >= min_ver]
            v["numba"] = random.choice(pool) if pool else ""

    # numba's numpy ceiling (see NUMBA_MAX_NUMPY): each pin against the numpy that
    # outgrew it. Re-pick from the pins that clear both this numpy and the Python
    # minimum, or skip numba.
    ceiling = _numba_ceiling(v["numba"])
    if ceiling is not None and npver >= ceiling:
        min_ver = _min_for_python(NUMBA_MIN_PYTHON, pyver)
        pool = [
            n for n in NUMBA_VERSIONS if n and NUMBA_MAX_NUMPY[n] > npver and _ver(n) >= min_ver
        ]
        v["numba"] = random.choice(pool) if pool else "NA"

    # --- sparse ---
    # No rules needed: sparse caps neither Python nor numpy, and the workflow drops it
    # whenever numba is skipped or NA.


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
    v = {
        "numpy": random.choice(NUMPY_VERSIONS[pyver]),
        "scipy": random.choice(SCIPY_VERSIONS),
        "pandas": random.choice(PANDAS_VERSIONS),
        "awkward": random.choice(AWKWARD_VERSIONS),
        "numba": random.choice(NUMBA_VERSIONS),
        "networkx": random.choice(NETWORKX_VERSIONS[pyver]),
        "pyyaml": random.choice(PYYAML_VERSIONS[pyver]),
        "sparse": random.choice(SPARSE_VERSIONS[pyver]),
    }

    # Source builds have issues with some numpy/scipy/pandas versions, so they take the
    # latest. Blank them before the constraints, so the numba rules see that numpy.
    if source_type == "source":
        v["numpy"] = ""
        v["scipy"] = ""
        v["pandas"] = ""

    apply_constraints(v, pyver)
    v["psg"] = _pick_psg(pyver, source_type)
    return v


def _pick_psg(pyver, source_type):
    """Pick python-suitesparse-graphblas version."""
    if source_type == "upstream":
        return ""
    eq = "=" if source_type == "conda-forge" else "=="
    pool = PSG_VERSIONS
    if (floors := _min_for_python(PSG_MIN_PYTHON, pyver)) is not None:
        floor = _ver(floors[source_type])
        pool = [ver for ver in pool if _ver(ver) >= floor]
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
    Empty means latest, NA means skip. Latest is capped by LATEST_CEILINGS
    (e.g., pdver='<3.1').
    psg already has its prefix baked in.
    """
    lines = []
    for key in ("numpy", "scipy", "pandas", "awkward", "networkx", "pyyaml", "sparse", "numba"):
        var = _VAR_NAMES[key]
        val = v[key]
        if val == "" and key in LATEST_CEILINGS:
            lines.append(f"{var}='{LATEST_CEILINGS[key]}'")
        elif val in ("NA", ""):
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
        # psg carries its own "=" or "==" pin prefix; strip it so the summary reads
        # "psg=10.5.0.0" like every other entry instead of "psg==10.5.0.0".
        val = v[key].lstrip("=") if key == "psg" else v[key]
        if val == "NA":
            parts.append(f"{name}=NA")
        elif val == "":
            parts.append(f"{name}=latest{LATEST_CEILINGS.get(key, '')}")
        else:
            parts.append(f"{name}={val}")
    return "versions: " + " ".join(parts)


# ---------------------------------------------------------------------------
# Validation (for testing the script itself)
# ---------------------------------------------------------------------------


def check_ceilings():
    """Check that every scipy and numba pin has a numpy-ceiling row. Returns errors.

    A pin with no row would silently skip the numpy-ceiling rule and be paired with a
    numpy it refuses to import against.
    """
    tables = [
        ("SCIPY_MAX_NUMPY", "scipy", SCIPY_MAX_NUMPY, SCIPY_VERSIONS),
        ("NUMBA_MAX_NUMPY", "numba", NUMBA_MAX_NUMPY, NUMBA_VERSIONS),
    ]
    errors = []
    for name, dep, table, pool in tables:
        pins = {p for p in pool if p}
        if missing := sorted(pins - table.keys(), key=_ver):
            errors.append(f"{name} has no numpy ceiling for {dep} {', '.join(missing)}")
        if extra := sorted(table.keys() - pins, key=_ver):
            errors.append(f"{name} has rows for untested {dep} {', '.join(extra)}")
    return errors


def check_python_pools():
    """Check that every per-Python table covers the same Pythons. Returns list of errors.

    Adding a Python to some tables but not others would otherwise surface as a KeyError
    from somewhere deep in the picking, or (for the tables consulted only by a
    constraint) as a silently wrong pick.
    """
    tables = {
        "NETWORKX_VERSIONS": NETWORKX_VERSIONS,
        "PYYAML_VERSIONS": PYYAML_VERSIONS,
        "SPARSE_VERSIONS": SPARSE_VERSIONS,
        "NUMBA_MIN_PYTHON": NUMBA_MIN_PYTHON,
    }
    errors = []
    for name, table in tables.items():
        missing = sorted(NUMPY_VERSIONS.keys() - table.keys(), key=_ver)
        extra = sorted(table.keys() - NUMPY_VERSIONS.keys(), key=_ver)
        if missing:
            errors.append(f"{name} has no entry for Python {', '.join(missing)}")
        if extra:
            errors.append(f"{name} has entries for unsupported Python {', '.join(extra)}")
    return errors


def validate(v, pyver, source_type=None):
    """Check that a version combination satisfies all constraints. Returns list of errors."""
    errors = []

    # scipy's declared numpy ceiling (see SCIPY_MAX_NUMPY)
    ceiling = SCIPY_MAX_NUMPY.get(v["scipy"])
    if ceiling is not None and _numpy_ver(v, pyver) >= ceiling:
        cap = ".".join(str(x) for x in ceiling)
        errors.append(f"scipy {v['scipy']} requires numpy <{cap}, got {v['numpy'] or 'latest'}")

    # scipy Python availability
    if _ver(pyver) >= (3, 15) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 18):
        errors.append(f"scipy {v['scipy']} has no py{pyver} build (needs >=1.18)")
    elif _ver(pyver) >= (3, 14) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        errors.append(f"scipy {v['scipy']} has no py{pyver} build (needs >=1.16)")
    elif _ver(pyver) >= (3, 13) and v["scipy"] == "1.13":
        errors.append(f"scipy 1.13 has no py{pyver} build")

    # pandas Python availability
    if _ver(pyver) >= (3, 15) and v["pandas"] not in ("", "NA") and _ver(v["pandas"]) < (3, 0):
        errors.append(f"pandas {v['pandas']} has no py{pyver} build")
    elif _ver(pyver) >= (3, 14) and v["pandas"] == "2.2":
        errors.append(f"pandas 2.2 has no py{pyver} build")

    # pandas 3.0 requirements (numba >=0.60 is unreachable; see `apply_constraints`)
    if v["pandas"] == "3.0" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
        errors.append(f"pandas 3.0 requires scipy >=1.14.1, got {v['scipy']}")

    # awkward Python availability
    if _ver(pyver) >= (3, 15) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 13):
        errors.append(f"awkward {v['awkward']} has no py{pyver} build (needs >=2.13)")
    elif _ver(pyver) >= (3, 14) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
        errors.append(f"awkward {v['awkward']} has no py{pyver} build (needs >=2.8)")
    elif _ver(pyver) >= (3, 13) and v["awkward"] == "2.6":
        errors.append(f"awkward 2.6 has no py{pyver} build")

    # awkward <2.10 requires numpy <2.5 (generic-unit datetime64("NaT") at import)
    if (
        _ver(v["numpy"]) >= (2, 5)
        and v["awkward"] not in ("", "NA")
        and _ver(v["awkward"]) < (2, 10)
    ):
        errors.append(f"awkward {v['awkward']} requires numpy <2.5, got {v['numpy'] or 'latest'}")

    # numba Python minimums
    numba_min = _min_for_python(NUMBA_MIN_PYTHON, pyver)
    if v["numba"] not in ("", "NA") and _ver(v["numba"]) < numba_min:
        errors.append(f"numba {v['numba']} doesn't support Python {pyver}")

    # numba's declared numpy ceiling (see NUMBA_MAX_NUMPY)
    ceiling = _numba_ceiling(v["numba"])
    if ceiling is not None and _numpy_ver(v, pyver) >= ceiling:
        cap = ".".join(str(x) for x in ceiling)
        errors.append(
            f"numba {v['numba'] or 'latest'} requires numpy <{cap}, "
            f"got {v['numpy'] or 'latest'}"
        )

    # sparse: the per-Python pool is the whole rule, so nothing may leave it
    if v["sparse"] not in SPARSE_VERSIONS[pyver]:
        errors.append(f"sparse {v['sparse']} isn't in the py{pyver} pool")

    # psg build availability (the floor depends on the Python and where psg comes from)
    floors = _min_for_python(PSG_MIN_PYTHON, pyver)
    if floors is not None and source_type in floors and v.get("psg"):
        psgver = v["psg"].lstrip("=")
        floor = floors[source_type]
        if _ver(psgver) < _ver(floor):
            errors.append(f"psg {psgver} has no py{pyver} {source_type} build (needs >={floor})")

    return errors


def stress_test(n=10000):
    """Run n random picks for each Python/source combo and validate all."""
    total = 0
    failures = 0
    for pyver in NUMPY_VERSIONS:
        for source in ("conda-forge", "wheel", "source", "upstream"):
            for _ in range(n):
                v = pick_versions(pyver, source)
                errs = validate(v, pyver, source)
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

    if pool_errors := check_python_pools() + check_ceilings():
        print("Error: version pools are inconsistent:", file=sys.stderr)
        for e in pool_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

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

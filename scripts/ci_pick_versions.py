#!/usr/bin/env python
"""Pick random, compatible dependency versions for python-graphblas CI.

Replaces the bash-based version selection that used to live in
test_and_build.yml.

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
#
# To add a Python version (3.15, say): add it to `choices` in the "RNG for Python
# version" step of test_and_build.yml, give it a row in every table keyed by Python
# below (`check_python_pools` fails the run if you miss one), and revisit the floors
# in `apply_constraints`. Those floors are written as `_ver(pyver) >= (3, N)`, so a
# new Python inherits the previous one's floors instead of escaping them. That is
# enough to keep CI honest, but usually still too loose to install, so check each dep.
# ---------------------------------------------------------------------------

# numpy 2.5 requires Python >=3.12, so py3.11 stops at 2.4.
NUMPY_VERSIONS = {
    "3.11": ["1.24", "1.25", "1.26", "2.0", "2.1", "2.2", "2.3", "2.4", ""],
    "3.12": ["1.26", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", ""],
    "3.13": ["2.1", "2.2", "2.3", "2.4", "2.5", ""],
    "3.14": ["2.3", "2.4", "2.5", ""],
}

# Deps that depend on numpy version (1.x vs 2.x path).
# Per-Python sublists narrow to versions with available conda builds.
SCIPY_VERSIONS = {
    "1.x": {
        "3.11": ["1.9", "1.10", "1.11", "1.12", "1.13", "1.14", ""],
        "3.12": ["1.11", "1.12", "1.13", "1.14", ""],
    },
    "2.x": ["1.13", "1.14", "1.15", "1.16", "1.17", "1.18", ""],
}

# Oldest Python each scipy pin supports (scipy's requires-python). The "2.x" pool is
# shared by every Python, so re-picks run through `_pick_scipy` to drop pins this
# Python has no build for.
# MAINT: 2026-09-16 scipy 1.18 requires Python >=3.12
SCIPY_MIN_PYTHON = {"1.18": (3, 12)}

# The numpy ceiling each scipy pin declares, the same idea as NUMBA_MAX_NUMPY below:
# scipy caps `numpy <X`, so a pinned pair that violates it fails the conda solve.
# Rows hold the stricter of what PyPI and conda-forge declare, since either one can be
# the binding constraint (conda-forge caps scipy 1.12 at numpy <1.28, PyPI at <1.29).
# Every pin in SCIPY_VERSIONS needs a row here (`check_ceilings` enforces it).
# MAINT: 2026-09-16 scripts/audit_pypi_versions.py and audit_conda_versions.py check these
SCIPY_MAX_NUMPY = {
    "1.9": (1, 26),
    "1.10": (1, 27),
    "1.11": (1, 28),
    "1.12": (1, 28),
    "1.13": (2, 3),
    "1.14": (2, 3),
    "1.15": (2, 5),
    "1.16": (2, 6),
    "1.17": (2, 7),
    "1.18": (2, 8),
}

PANDAS_VERSIONS = {
    "1.x": {
        "3.11": ["1.5", "2.0", "2.1", "2.2", "2.3", ""],
        "3.12": ["2.1", "2.2", "2.3", ""],
    },
    "2.x": ["2.2", "2.3", "3.0", ""],
}

# The "2.x" pool starts at 2.6 because awkward <2.6 still uses numpy.AxisError,
# which numpy 2.0 removed. Only >=2.10 pins survive numpy >=2.5; see the awkward
# constraints in `apply_constraints`.
AWKWARD_VERSIONS = {
    "1.x": {
        "3.11": ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", ""],
        "3.12": ["2.4", "2.5", "2.6", "2.7", "2.8", "2.9", ""],
    },
    "2.x": ["2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", ""],
}

NUMBA_VERSIONS = {
    "1.x": ["0.57", "0.58", "0.59", "0.60", "0.61", ""],
    "2.x": ["0.62", "0.63", "0.64", "0.65", "0.66", "0.67", ""],
}

# Oldest numba that supports each Python.
NUMBA_MIN_PYTHON = {"3.11": (0, 57), "3.12": (0, 59), "3.13": (0, 61), "3.14": (0, 63)}

# The numpy ceiling each numba pin declares. numba caps `numpy <X` and refuses to import
# against anything newer. conda-forge carries that cap in `constrains` rather than
# `depends`, whose cap is only the numpy ABI, so a pinned pair that violates it fails
# the conda solve outright. Unpinned numba ("") installs the newest release, so it takes
# the newest row: numba is the dependency that most often lags a new numpy, and a pinned
# numpy that has outrun every numba would otherwise fail the solve. Every pin in
# NUMBA_VERSIONS needs a row here (`check_ceilings` fails the run if one is missing).
# MAINT: 2026-09-16 scripts/audit_pypi_versions.py and audit_conda_versions.py check these
NUMBA_MAX_NUMPY = {
    "0.57": (1, 25),
    "0.58": (1, 27),
    "0.59": (1, 27),
    "0.60": (2, 1),
    "0.61": (2, 3),
    "0.62": (2, 4),
    "0.63": (2, 4),
    "0.64": (2, 5),
    "0.65": (2, 5),
    "0.66": (2, 5),
    "0.67": (2, 6),
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

# sparse is noarch with no numpy ceiling, so only its Python floors matter: 0.16/0.17
# declare python >=3.10 and 0.18/0.19 declare >=3.11. On py3.13/3.14 the pools start at
# 0.16 because 0.14/0.15 predate those Pythons (the old "NA" here was written in the
# 0.15 era, when numba could not run there at all). To skip sparse on a Python, give it
# a ["NA"] pool: a bare "NA" would be indexed as a string by `random.choice`.
# MAINT: 2026-09-16 sparse 0.19.2 is latest; 0.16.0, 0.18.0 and 0.19.2 verified on py3.14
SPARSE_VERSIONS = {
    "3.11": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
    "3.12": ["0.14", "0.15", "0.16", "0.17", "0.18", "0.19", ""],
    "3.13": ["0.16", "0.17", "0.18", "0.19", ""],
    "3.14": ["0.16", "0.17", "0.18", "0.19", ""],
}

# PSG versions to pair with numpy 1.x (only reachable on py3.11/py3.12, since
# py3.13+ have no numpy 1.x in their pools). Only "conda-forge" and "wheel"
# builds get here: "source" builds blank numpy before picking psg (so they use
# PSG_VERSIONS_NP2), and "upstream" builds always use psg from git.
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
}

# PSG versions to pair with numpy 2.x, oldest first. Every source type except
# "upstream" draws from this one pool; Pythons too new for the older releases draw
# from a tail of it (see PSG_MIN_PYTHON).
PSG_VERSIONS_NP2 = [
    "9.3.1.0",
    "9.4.5.0",
    "10.0.1.1",
    "10.1.1.0",
    "10.3.1.0",
    "10.4.0.0",
    "10.4.1.0",
    "10.5.0.0",
]

# Oldest psg release that installs on a given Python, by where psg comes from.
# Pythons older than every entry here (3.11-3.13) can use the whole pool; a Python
# newer than every entry inherits the newest one until it gets a row of its own.
#
# py3.14 needs a different floor per source: conda-forge has py314 builds from
# 10.0.1.1 on, but PyPI's first cp314 wheels are in 10.3.1.0, so the "wheel" path
# can't reach back as far. "source" builds the sdist, and the sdists before 10.3.1.0
# predate psg's py3.14 support (they still declare requires-python >=3.9), so hold
# them to the wheel floor rather than test a build psg never claimed works.
# MAINT: 2026-09-11 checked conda-forge build strings and PyPI wheel tags for psg 9-10
PSG_MIN_PYTHON = {
    "3.14": {"conda-forge": "10.0.1.1", "wheel": "10.3.1.0", "source": "10.3.1.0"},
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


def _pick_scipy(pyver, npver, pool, floor=(0,)):
    """Pick a scipy pin at or above `floor` that this Python and this numpy can use.

    Filters the pool by the pin's requires-python (SCIPY_MIN_PYTHON), its numpy ceiling
    (SCIPY_MAX_NUMPY) and the one numpy floor scipy imposes (>=1.15 needs numpy
    >=1.26.4), so every re-pick site gets the same answer no matter where it sits in the
    constraint order. "" (latest) is always a candidate: an unpinned scipy is one conda
    resolves itself, so it satisfies every rule here by construction.
    """
    return random.choice(
        [
            s
            for s in pool
            if s
            and _ver(s) >= floor
            and SCIPY_MAX_NUMPY[s] > npver
            and (_ver(s) < (1, 15) or npver >= (1, 26))
            and _ver(pyver) >= SCIPY_MIN_PYTHON.get(s, (0,))
        ]
        + [""]
    )


def _numpy_ver(v, pyver):
    """Version of numpy to compare against, resolving "" (latest) to what it installs.

    An unpinned numpy is the newest release this Python can actually get, which is the
    newest pin in its pool, not something infinitely new: numba 0.67 caps numpy at <2.6
    and still qualifies today. Deriving it from the pool keeps it honest per Python
    (py3.11 tops out at numpy 2.4) and leaves nothing extra to bump.
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


def apply_constraints(v, pyver, scipy_pool, numba_pool):
    """Mutate version dict to satisfy all known compatibility constraints.

    Each constraint comment documents the real-world requirement it encodes.
    Order matters: numpy/scipy constraints first, then pandas (which may bump scipy/numba),
    then Python-version constraints, then numba/numpy constraints. Within the awkward
    section, the numpy 2.x floor runs before the Python-availability picks (which may
    narrow the choices further), and the awkward >=2.10 rule runs last so it wins.
    The networkx section reads numpy, scipy, and pandas, so it runs after all three
    are final.
    """
    # Nothing below changes the numpy pick, so derive these once.
    npver = _numpy_ver(v, pyver)
    np_is_1x = v["numpy"].startswith("1.") if v["numpy"] else False

    # --- scipy / numpy constraints ---

    # scipy >=1.15 requires numpy >=1.26.4
    if v["numpy"] in ("1.24", "1.25") and _ver(v["scipy"]) >= (1, 15):
        candidates = [s for s in scipy_pool if s and _ver(s) < (1, 15)]
        v["scipy"] = random.choice(candidates) if candidates else "1.14"

    # scipy's numpy ceiling (see SCIPY_MAX_NUMPY). This covers every scipy/numpy
    # pairing at once: the pre-1.13 pins against numpy 2.x, 1.9 against numpy 1.26,
    # and each later pin against the numpy that outgrew it.
    if (ceiling := SCIPY_MAX_NUMPY.get(v["scipy"])) is not None and npver >= ceiling:
        v["scipy"] = _pick_scipy(pyver, npver, scipy_pool)

    # --- scipy / Python version availability ---

    # scipy <1.14 has no py3.13 builds; scipy <1.16 has no py3.14 builds
    if _ver(pyver) >= (3, 14) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        v["scipy"] = _pick_scipy(pyver, npver, scipy_pool, floor=(1, 16))
    elif _ver(pyver) >= (3, 13) and v["scipy"] == "1.13":
        v["scipy"] = _pick_scipy(pyver, npver, scipy_pool, floor=(1, 14))

    # A scipy pin whose requires-python is newer than this Python has no build for it
    if v["scipy"] and _ver(pyver) < SCIPY_MIN_PYTHON.get(v["scipy"], (0,)):
        v["scipy"] = _pick_scipy(pyver, npver, scipy_pool)

    # --- pandas constraints ---

    # pandas <2.3 has no py3.14 builds
    if _ver(pyver) >= (3, 14) and v["pandas"] == "2.2":
        v["pandas"] = random.choice(["2.3", "3.0", ""])

    # pandas 3.0 requires scipy >=1.14.1. It also requires numba >=0.60, which needs no
    # rule: pandas 3.0 only appears in the numpy 2.x pool, whose numba pins start at 0.62.
    if v["pandas"] == "3.0" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
        v["scipy"] = _pick_scipy(pyver, npver, scipy_pool, floor=(1, 15))

    # conda-forge pandas >=2.2 carries run_constrained "scipy >=1.10.0" (latest 2.2.3
    # build and all 2.3 builds; 2.0/2.1 declare no scipy constraint), so a pandas
    # 2.2/2.3 pin alongside a scipy 1.9 pin fails the conda solve outright. Unpinned
    # pandas backs off on its own, so only pins trigger. A scipy 1.9 pick can only
    # reach here with numpy 1.24/1.25 (the 1.26 conflict above already re-picked),
    # so keep the replacement pinned below the scipy 1.15 ceiling those numpys impose.
    # MAINT: 2026-08-06 verified with conda search --info against conda-forge
    if (
        v["pandas"] not in ("", "NA")
        and _ver(v["pandas"]) >= (2, 2)
        and v["scipy"] not in ("", "NA")
        and _ver(v["scipy"]) < (1, 10)
    ):
        v["scipy"] = random.choice([s for s in scipy_pool if s and (1, 10) <= _ver(s) < (1, 15)])

    # --- awkward / numpy 2.x support ---

    # awkward <2.6 uses numpy.AxisError, which numpy 2.0 removed (test_io.py then fails
    # at collection). awkward is picked from the numpy 1.x pool before source builds blank
    # numpy to latest, so a numpy 1.x pick can still end up here with numpy 2.x.
    # MAINT: 2026-08-04 confirmed with awkward 2.0.10 + numpy 2.4
    if not np_is_1x and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 6):
        v["awkward"] = random.choice(AWKWARD_VERSIONS["2.x"])

    # --- awkward / Python version availability ---

    # awkward <2.7 has no py3.13 builds; awkward <2.8 has no py3.14 builds
    if _ver(pyver) >= (3, 14) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
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
        v["awkward"] = random.choice([a for a in AWKWARD_VERSIONS["2.x"] if _ver(a) >= (2, 10)])

    # --- networkx / numpy, scipy, pandas floors ---

    # conda-forge networkx 3.5 and 3.6 declare `constrains: numpy >=1.25, scipy >=1.11.2,
    # pandas >=2.0`. conda enforces those against our pins, and since networkx is pinned too
    # the solve fails outright instead of backing off. Older pins are safe because a pin like
    # "3.4" can resolve down to 3.4.0, whose floors (numpy >=1.22, scipy >=1.9, pandas >=1.4)
    # sit at or below everything in our pools. Unpinned networkx is latest, so it gets the
    # 3.5+ floors. Compare scipy at the minor level: the "1.11" pin resolves to 1.11.4, which
    # clears the 1.11.2 floor.
    # MAINT: 2026-08-04 conda-forge networkx 3.6.1 constrains numpy >=1.25, scipy >=1.11.2
    if _ver(v["networkx"]) >= (3, 5) and (
        _ver(v["numpy"]) < (1, 25) or _ver(v["scipy"]) < (1, 11) or _ver(v["pandas"]) < (2, 0)
    ):
        # Only the numpy 1.x pools reach here, and those Pythons all have pre-3.5 networkx.
        v["networkx"] = random.choice([n for n in NETWORKX_VERSIONS[pyver] if _ver(n) < (3, 5)])

    # --- matplotlib / numpy floor (notebooks job) ---

    # matplotlib is installed unpinned for the notebooks task. matplotlib 3.11's runtime
    # check requires numpy >=1.25. conda-forge's 3.11.1 builds declare that floor, so the
    # solver backs off on its own, but the 3.11.0 builds still say only "numpy >=1.23":
    # the solver pairs 3.11.0 with a numpy 1.24 pin and the notebooks die at
    # `import matplotlib`. Pin below 3.11 whenever the numpy pick sits under that floor;
    # empty means latest is fine.
    # MAINT: 2026-09-16 conda-forge matplotlib-base 3.11.0 depends on "numpy >=1.23",
    # 3.11.1 on "numpy >=1.25"
    if v["numpy"] and _ver(v["numpy"]) < (1, 25):
        v["matplotlib"] = "<3.11"
    else:
        v["matplotlib"] = ""

    # --- numba constraints ---

    # numba minimum by Python version (see NUMBA_MIN_PYTHON)
    if v["numba"] not in ("", "NA"):
        min_ver = _min_for_python(NUMBA_MIN_PYTHON, pyver)
        if _ver(v["numba"]) < min_ver:
            pool = [n for n in numba_pool if _ver(n) >= min_ver]
            v["numba"] = random.choice(pool) if pool else ""

    # numba's numpy ceiling (see NUMBA_MAX_NUMPY). This covers every numba/numpy
    # pairing rule at once: 0.57 against numpy >=1.25, the pre-0.62 pins against
    # numpy 2.x, and each later pin against the numpy that outgrew it. Re-pick from
    # the pins that clear both this numpy and the Python minimum, or skip numba.
    ceiling = _numba_ceiling(v["numba"])
    if ceiling is not None and npver >= ceiling:
        min_ver = _min_for_python(NUMBA_MIN_PYTHON, pyver)
        pool = [n for n in numba_pool if n and NUMBA_MAX_NUMPY[n] > npver and _ver(n) >= min_ver]
        v["numba"] = random.choice(pool) if pool else "NA"

    # --- sparse ---
    # No rules needed: the per-Python pools already respect sparse's Python floors,
    # and the workflow skips sparse whenever numba is skipped or NA.


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
        "sparse": random.choice(SPARSE_VERSIONS[pyver]),
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
    pool = PSG_VERSIONS_NP2
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
    "matplotlib": "mpl",
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
    # matplotlib carries a bare ceiling (e.g. "<3.11") or is empty for latest
    lines.append(f"mplver='{v['matplotlib']}'")
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
        "matplotlib",
    ):
        name = _SUMMARY_NAMES[key]
        # psg carries its own "=" or "==" pin prefix; strip it so the summary reads
        # "psg=10.5.0.0" like every other entry instead of "psg==10.5.0.0".
        val = v[key].lstrip("=") if key == "psg" else v[key]
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


def check_ceilings():
    """Check that every scipy and numba pin has a numpy-ceiling row. Returns errors.

    A pin with no row would silently skip the numpy-ceiling rule and be paired with a
    numpy it refuses to import against.
    """
    tables = [
        (
            "SCIPY_MAX_NUMPY",
            "scipy",
            SCIPY_MAX_NUMPY,
            [SCIPY_VERSIONS["2.x"], *SCIPY_VERSIONS["1.x"].values()],
        ),
        ("NUMBA_MAX_NUMPY", "numba", NUMBA_MAX_NUMPY, list(NUMBA_VERSIONS.values())),
    ]
    errors = []
    for name, dep, table, pools in tables:
        pins = {p for pool in pools for p in pool if p}
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
    np_is_1x = v["numpy"].startswith("1.") if v["numpy"] else False

    # scipy >=1.15 requires numpy >=1.26.4
    if v["numpy"] in ("1.24", "1.25") and _ver(v["scipy"]) >= (1, 15):
        errors.append(f"scipy {v['scipy']} requires numpy >=1.26.4, got {v['numpy']}")

    # scipy's declared numpy ceiling (see SCIPY_MAX_NUMPY)
    ceiling = SCIPY_MAX_NUMPY.get(v["scipy"])
    if ceiling is not None and _numpy_ver(v, pyver) >= ceiling:
        cap = ".".join(str(x) for x in ceiling)
        errors.append(f"scipy {v['scipy']} requires numpy <{cap}, got {v['numpy'] or 'latest'}")

    # scipy Python availability
    if _ver(pyver) >= (3, 14) and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 16):
        errors.append(f"scipy {v['scipy']} has no py{pyver} build")
    elif _ver(pyver) >= (3, 13) and v["scipy"] == "1.13":
        errors.append(f"scipy 1.13 has no py{pyver} build")
    if v["scipy"] and _ver(pyver) < (floor := SCIPY_MIN_PYTHON.get(v["scipy"], (0,))):
        need = ".".join(str(x) for x in floor)
        errors.append(f"scipy {v['scipy']} requires Python >={need}, has no py{pyver} build")

    # pandas Python availability
    if _ver(pyver) >= (3, 14) and v["pandas"] == "2.2":
        errors.append(f"pandas 2.2 has no py{pyver} build")

    # pandas 3.0 requirements (numba >=0.60 is unreachable; see `apply_constraints`)
    if v["pandas"] == "3.0" and v["scipy"] not in ("", "NA") and _ver(v["scipy"]) < (1, 15):
        errors.append(f"pandas 3.0 requires scipy >=1.14.1, got {v['scipy']}")

    # conda-forge pandas >=2.2 constrains scipy >=1.10.0 (fails the conda solve)
    if (
        v["pandas"] not in ("", "NA")
        and _ver(v["pandas"]) >= (2, 2)
        and v["scipy"] not in ("", "NA")
        and _ver(v["scipy"]) < (1, 10)
    ):
        errors.append(f"pandas {v['pandas']} constrains scipy >=1.10.0, got {v['scipy']}")

    # awkward Python availability
    if _ver(pyver) >= (3, 14) and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 8):
        errors.append(f"awkward {v['awkward']} has no py{pyver} build")
    elif _ver(pyver) >= (3, 13) and v["awkward"] == "2.6":
        errors.append(f"awkward 2.6 has no py{pyver} build")

    # awkward <2.6 requires numpy 1.x (numpy.AxisError)
    if not np_is_1x and v["awkward"] not in ("", "NA") and _ver(v["awkward"]) < (2, 6):
        errors.append(f"awkward {v['awkward']} doesn't support numpy 2.x")

    # matplotlib 3.11 requires numpy >=1.25 at runtime (notebooks job)
    if v["numpy"] and _ver(v["numpy"]) < (1, 25) and v.get("matplotlib", "") != "<3.11":
        errors.append(f"matplotlib unpinned alongside numpy {v['numpy']} (needs <3.11)")

    # awkward <2.10 requires numpy <2.5 (generic-unit datetime64("NaT") at import)
    if (
        _ver(v["numpy"]) >= (2, 5)
        and v["awkward"] not in ("", "NA")
        and _ver(v["awkward"]) < (2, 10)
    ):
        errors.append(f"awkward {v['awkward']} requires numpy <2.5, got {v['numpy'] or 'latest'}")

    # networkx >=3.5 constrains numpy >=1.25, scipy >=1.11.2 (the "1.11" pin clears it),
    # and pandas >=2.0
    if _ver(v["networkx"]) >= (3, 5):
        nx = v["networkx"] or "latest"
        if _ver(v["numpy"]) < (1, 25):
            errors.append(f"networkx {nx} requires numpy >=1.25, got {v['numpy']}")
        if _ver(v["scipy"]) < (1, 11):
            errors.append(f"networkx {nx} requires scipy >=1.11.2, got {v['scipy']}")
        if _ver(v["pandas"]) < (2, 0):
            errors.append(f"networkx {nx} requires pandas >=2.0, got {v['pandas']}")

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

import atexit
import contextlib
import functools
import itertools
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import graphblas as gb
from graphblas.core import _supports_udfs as supports_udfs

# Under ``pytest -n`` (xdist), give each worker its own ``GRAPHBLAS_CACHE_PATH``
# so concurrent JIT compiles within a single session don't race on the same
# ``.c``/``.dylib`` paths. SuiteSparse reads this env var at first GraphBLAS
# init; it must be set before that init runs, or SS takes the unset
# (HOME-derived) default. ``import graphblas`` above does not trigger
# init (it's lazy), so setting it here is in time.
#
# The path is stable per worker id (``gw0``, ``gw1``, ...) and persistent
# across pytest invocations (not session-scoped). This way the JIT cache
# builds up over successive runs and most kernels are dlopen'd instead of
# recompiled. Total disk footprint is bounded by the number of unique kernels
# the suite touches across all worker assignments (roughly a few hundred
# ``.dylib`` files, low tens of MB).
#
# Caveat: two ``pytest -n`` invocations running simultaneously would both
# write to ``gw0/`` etc. and can race on shared kernels. We accept that
# occasional concurrent-run failure rather than reintroducing per-session
# isolation, which would defeat cache reuse.
if (
    _xdist_worker := os.environ.get("PYTEST_XDIST_WORKER")
) and "GRAPHBLAS_CACHE_PATH" not in os.environ:
    _home = os.environ.get("HOME") or os.environ.get("LOCALAPPDATA") or str(Path.cwd())
    os.environ["GRAPHBLAS_CACHE_PATH"] = str(Path(_home) / ".SuiteSparse_xdist" / _xdist_worker)

# Several ``test_formatting.py`` tests assert verbatim against a Matrix /
# Vector ``repr()``, and those reprs are produced via pandas. With pandas'
# default ``display.max_columns = 0`` (terminal mode), pandas auto-fits to
# ``shutil.get_terminal_size()``, which reads the ``COLUMNS`` env var. When
# pytest is invoked from a wide terminal the workers inherit ``COLUMNS=<wide>``
# and pandas renders extra columns that the tests expected to be elided to
# ``...``. Drop ``COLUMNS`` here so terminal-size detection falls back to the
# 80x24 default the test expectations were written for. The pop has to run at
# conftest module load, before any worker imports pandas and before xdist
# forks. Setting ``display.width`` alone doesn't help; the terminal-size
# branch is taken whenever ``max_columns == 0``.
os.environ.pop("COLUMNS", None)


orig_binaryops = set()
orig_semirings = set()

pypy = platform.python_implementation() == "PyPy"


def pytest_configure(config):
    # Under ``pytest -n`` (xdist), every worker imports this conftest and
    # calls ``pytest_configure`` independently. A fresh ``default_rng()`` in
    # each worker would pick different backend / blocking / mapnumpy values,
    # so the workers collect different tests (vanilla skips ``test_ssjit``
    # etc.) and xdist aborts with "Different tests were collected between
    # gw0 and gw2." Seed once on the controller and propagate to workers via
    # an env var; workers inherit it at spawn and derive identical choices.
    # Set the env var unconditionally on the controller so reproducing a
    # specific session is just ``GRAPHBLAS_TEST_SEED=<n> pytest ...``.
    seed_str = os.environ.get("GRAPHBLAS_TEST_SEED")
    if seed_str is None:
        seed = int.from_bytes(os.urandom(8), "big")
        os.environ["GRAPHBLAS_TEST_SEED"] = str(seed)
    else:
        seed = int(seed_str)
    rng = np.random.default_rng(seed)
    randomly = config.getoption("--randomly", None)
    if randomly is None:  # pragma: no cover
        options_unavailable = True
        randomly = True
        config.addinivalue_line("markers", "slow: Skipped unless --runslow passed")
    else:
        options_unavailable = False
    backend = config.getoption("--backend", None)
    if backend is None:
        if randomly:
            backend = "suitesparse" if rng.random() < 0.5 else "suitesparse-vanilla"
        else:
            backend = "suitesparse"
    blocking = config.getoption("--blocking", None)
    if blocking is None:  # pragma: no branch
        blocking = rng.random() < 0.5 if randomly else True
    record = config.getoption("--record", False)
    if record is None:  # pragma: no branch
        record = rng.random() < 0.5 if randomly else False
    mapnumpy = config.getoption("--mapnumpy", None)
    if mapnumpy is None:
        mapnumpy = rng.random() < 0.5 if randomly else False
    runslow = config.getoption("--runslow", None)
    if runslow is None:
        runslow = options_unavailable
    config.runslow = runslow

    gb.config.set(autocompute=False, mapnumpy=mapnumpy)

    gb.init(backend, blocking=blocking)
    print(
        f"Running tests with {backend!r} backend, blocking={blocking}, "
        f"record={record}, mapnumpy={mapnumpy}, runslow={runslow}, seed={seed}"
    )
    if record:
        rec = gb.Recorder()
        rec.start()

        def save_records():
            with Path("record.txt").open("w") as f:  # pragma: no cover (???)
                f.write("\n".join(rec.data))

        # I'm sure there's a `pytest` way to do this...
        atexit.register(save_records)
    orig_semirings.update(
        key
        for key in dir(gb.semiring)
        if key != "ss"
        and isinstance(
            (
                getattr(gb.semiring, key)
                if key not in gb.semiring._deprecated
                else gb.semiring._deprecated[key]
            ),
            (gb.core.operator.Semiring, gb.core.operator.ParameterizedSemiring),
        )
    )
    orig_binaryops.update(
        key
        for key in dir(gb.binary)
        if key != "ss"
        and isinstance(
            (
                getattr(gb.binary, key)
                if key not in gb.binary._deprecated
                else gb.binary._deprecated[key]
            ),
            (gb.core.operator.BinaryOp, gb.core.operator.ParameterizedBinaryOp),
        )
    )
    for mod in [gb.unary, gb.binary, gb.monoid, gb.semiring, gb.op]:
        for name in list(mod._delayed):
            getattr(mod, name)


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.runslow:
        pytest.skip("need --runslow option to run")


@pytest.fixture(autouse=True)
def _reset_name_counters():
    """Reset automatic names for each test for easier comparison of record.txt."""
    gb.Matrix._name_counter = itertools.count()
    gb.Vector._name_counter = itertools.count()
    gb.Scalar._name_counter = itertools.count()


@pytest.fixture(scope="session", autouse=True)
def ic():  # pragma: no cover (debug)
    """Make `ic` available everywhere during testing for easier debugging."""
    try:
        import icecream
    except ImportError:
        return
    icecream.install()
    # icecream.ic.disable()  # This disables icecream; do ic.enable() to re-enable
    return icecream.ic


@contextlib.contextmanager
def burble():  # pragma: no cover (debug)
    """Show the burble diagnostics within a context."""
    if gb.backend != "suitesparse":
        yield
        return
    prev = gb.ss.config["burble"]
    gb.ss.config["burble"] = True
    try:
        yield
    finally:
        gb.ss.config["burble"] = prev


@pytest.fixture(scope="session")
def burble_all():  # pragma: no cover (debug)
    """Show the burble diagnostics for the entire test."""
    with burble():
        yield burble


def autocompute(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        with gb.config.set(autocompute=True):
            return func(*args, **kwargs)

    return inner


def compute(x):
    return x


def shouldhave(module, opname):
    """Whether an "operator" module should have the given operator."""
    return supports_udfs or hasattr(module, opname)


def dprint(*args, **kwargs):  # pragma: no cover (debug)
    """Print to stderr for debugging purposes."""
    kwargs["file"] = sys.stderr
    kwargs["flush"] = True
    print(*args, **kwargs)

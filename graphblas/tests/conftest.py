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

try:
    from numpy._core._multiarray_umath import _set_numpy_warn_if_no_mem_policy
except ImportError:  # pragma: no cover (it is private, so it may go away)
    _set_numpy_warn_if_no_mem_policy = None

# Under ``pytest -n`` (xdist), give each worker its own JIT cache, so workers
# compiling the same kernel at once do not write the same files. SuiteSparse
# reads ``GRAPHBLAS_CACHE_PATH`` when GraphBLAS initializes, and
# ``import graphblas`` does not initialize it, so setting it here is in time.
#
# The path depends only on the SuiteSparse version and the worker id, as in
# ``~/.SuiteSparse_xdist/GrB10.5.3/gw0``, so the cache persists across runs and
# later runs load most kernels instead of compiling them. The version keeps
# environments on different SuiteSparse versions apart, as it does in
# SuiteSparse's own default, ``~/.SuiteSparse/GrB10.5.3``. Two ``pytest -n``
# runs at once share these directories and can race on a kernel; a directory
# per run would avoid that but compile every kernel anew.
if (
    _xdist_worker := os.environ.get("PYTEST_XDIST_WORKER")
) and "GRAPHBLAS_CACHE_PATH" not in os.environ:
    # Importing suitesparse_graphblas does not initialize GraphBLAS either
    from suitesparse_graphblas import lib as _lib

    _home = os.environ.get("HOME") or os.environ.get("LOCALAPPDATA") or str(Path.cwd())
    _version = (
        f"GrB{_lib.GxB_IMPLEMENTATION_MAJOR}.{_lib.GxB_IMPLEMENTATION_MINOR}"
        f".{_lib.GxB_IMPLEMENTATION_SUB}"
    )
    os.environ["GRAPHBLAS_CACHE_PATH"] = str(
        Path(_home) / ".SuiteSparse_xdist" / _version / _xdist_worker
    )

# Matrix and Vector reprs come from pandas. With ``display.max_columns`` at 0
# (its default in a terminal), pandas fits a repr to the terminal width, which
# it reads from ``COLUMNS`` and not from ``display.width``. Tests that compare
# reprs expect 80 columns, the width pandas falls back to when ``COLUMNS`` is
# unset and pytest captures output, so drop a wide terminal's ``COLUMNS``.
# test_formatting.py also sets 80 itself; this covers any other test that
# renders a repr.
os.environ.pop("COLUMNS", None)


orig_binaryops = set()
orig_semirings = set()


def op_items(module):
    """Yield ``(name, value)`` for each name in an operator namespace such as ``gb.binary``.

    With the suitesparse backend, this includes the built-in operators in the ``ss``
    namespace, with names such as ``"ss.firsti"``. Operators that tests register in
    ``ss``, such as JIT operators, are left out. Resolve a name with :func:`getop`.
    """
    for key in dir(module):
        if key != "ss":
            yield key, getattr(module, key)
    if gb.backend == "suitesparse":
        from graphblas.core.operator.base import _SS_OPERATORS

        for key in sorted(vars(module.ss).keys() & _SS_OPERATORS):
            yield f"ss.{key}", getattr(module.ss, key)


def getop(module, name):
    """Get an operator by a name from :func:`op_items`, such as ``"ss.firsti"``."""
    return functools.reduce(getattr, name.split("."), module)


pypy = platform.python_implementation() == "PyPy"


def pytest_configure(config):
    # Under ``pytest -n`` (xdist), each worker runs ``pytest_configure``. The
    # workers must make the same random choices (backend, blocking, mapnumpy),
    # or they collect different tests and xdist aborts with "Different tests
    # were collected between gw0 and gw2". So the controller picks the seed and
    # passes it to the workers in ``GRAPHBLAS_TEST_SEED``. Set that variable to
    # reproduce a run: ``GRAPHBLAS_TEST_SEED=<n> pytest ...``.
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
        for key, val in op_items(gb.semiring)
        if isinstance(val, (gb.core.operator.Semiring, gb.core.operator.ParameterizedSemiring))
    )
    orig_binaryops.update(
        key
        for key, val in op_items(gb.binary)
        if isinstance(val, (gb.core.operator.BinaryOp, gb.core.operator.ParameterizedBinaryOp))
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


@pytest.fixture(autouse=True)
def _stop_a_leaked_recorder():
    """Fail a test that leaves a Recorder running, and stop that Recorder.

    A running Recorder records every later call, and the fast paths for element
    access are off while one runs, so every test after a leak would test the
    full path.
    """
    from graphblas.core import base

    def running():
        return base._recorder.get(base._prev_recorder)

    before = running()  # the session's own Recorder under --record, else None
    yield
    rec = running()
    if rec is not None and rec is not before:  # pragma: no cover (only on failure)
        while rec is not None and rec is not before:
            rec.stop()
            if running() is rec:
                break
            rec = running()
        pytest.fail("The test left a Recorder running; stop it, or use `with Recorder():`")


@pytest.fixture(scope="session", autouse=True)
def _warn_if_no_memory_handler():
    """Fail a test that frees an array that owns its data but has no memory handler.

    NumPy frees such data with libc ``free``, which is right only if libc's
    ``malloc`` made it; GraphBLAS's allocator is NumPy's. NumPy warns about
    this only when asked, and ``filterwarnings = error`` makes the warning fail
    the test. suitesparse-graphblas before 10.5.1.1 (which lacks
    ``can_unclaim_buffer``) exports arrays with no handler, so it would fail
    every export.
    """
    from suitesparse_graphblas import utils

    if _set_numpy_warn_if_no_mem_policy is None or not hasattr(utils, "can_unclaim_buffer"):
        yield
        return
    prev = _set_numpy_warn_if_no_mem_policy(True)
    yield
    _set_numpy_warn_if_no_mem_policy(prev)


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

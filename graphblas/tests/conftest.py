import atexit
import contextlib
import functools
import itertools
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import graphblas as gb
from graphblas.core import _supports_udfs as supports_udfs

orig_binaryops = set()
orig_semirings = set()

pypy = platform.python_implementation() == "PyPy"


def pytest_configure(config):
    rng = np.random.default_rng()
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
        f"record={record}, mapnumpy={mapnumpy}, runslow={runslow}"
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


def dprint(*args, **kwargs):
    """Print to stderr for debugging purposes."""
    kwargs["file"] = sys.stderr
    kwargs["flush"] = True
    print(*args, **kwargs)

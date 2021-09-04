import atexit
import itertools

import pytest

import grblas


def pytest_configure(config):
    backend = config.getoption("--backend", "suitesparse")
    blocking = config.getoption("--blocking", True)
    record = config.getoption("--record", False)

    grblas.init(backend, blocking=blocking)
    print(f'Running tests with "{backend}" backend, blocking={blocking}, record={record}')
    if record:
        rec = grblas.Recorder()
        rec.start()

        def save_records():
            with open("record.txt", "w") as f:  # pragma: no cover
                f.write("\n".join(rec.data))

        # I'm sure there's a `pytest` way to do this...
        atexit.register(save_records)


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow", True):  # pragma: no cover
        pytest.skip("need --runslow option to run")


@pytest.fixture(autouse=True, scope="function")
def reset_name_counters():
    """Reset automatic names for each test for easier comparison of record.txt"""
    grblas.Matrix._name_counter = itertools.count()
    grblas.Vector._name_counter = itertools.count()
    grblas.Scalar._name_counter = itertools.count()

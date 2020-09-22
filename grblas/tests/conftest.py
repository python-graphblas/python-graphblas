import atexit
import pytest


def pytest_configure(config):
    backend = config.getoption("--backend", "suitesparse")
    blocking = config.getoption("--blocking", True)
    record = config.getoption("--record", False)
    import grblas

    grblas.init(backend, blocking=blocking)
    print(f'Running tests with "{backend}" backend, blocking={blocking}, record={record}')
    if record:
        rec = grblas.Recorder()
        rec.start()

        def save_records():
            with open("record.txt", "w") as f:
                f.write("\n".join(rec.data))

        # I'm sure there's a `pytest` way to do this...
        atexit.register(save_records)


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow", True):  # pragma: no cover
        pytest.skip("need --runslow option to run")

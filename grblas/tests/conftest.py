import pytest


def pytest_configure(config):
    backend = config.getoption("--backend", "suitesparse")
    blocking = config.getoption("--blocking", True)
    import grblas

    grblas.init(backend, blocking=blocking)
    print(f'Running tests with "{backend}" backend, blocking={blocking}')


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow", True):
        pytest.skip("need --runslow option to run")

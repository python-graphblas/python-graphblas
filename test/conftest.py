import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default="suitesparse", help="name of a backend in grblas.backends"
    )
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_configure(config):
    backend = config.getoption('--backend')
    import grblas
    grblas.init(backend)
    print(f'Running tests with "{backend}" backend')


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")

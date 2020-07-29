import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default="suitesparse", help="name of a backend in grblas.backends"
    )
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--blocking", dest='blocking', default=True, action="store_true", help="run in blocking mode")
    parser.addoption("--nonblocking", "--no-blocking", "--non-blocking", dest='blocking', action="store_false", help="run in non-blocking mode")


def pytest_configure(config):
    backend = config.getoption('--backend')
    blocking = config.getoption('--blocking')
    import grblas
    grblas.init(backend, blocking=blocking)
    print(f'Running tests with "{backend}" backend, blocking={blocking}')


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")

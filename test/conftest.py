def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default="suitesparse", help="name of a backend in grblas.backends"
    )


def pytest_configure(config):
    backend = config.getoption('--backend')
    import grblas
    grblas.init(backend)
    print(f'Running tests with "{backend}" backend')

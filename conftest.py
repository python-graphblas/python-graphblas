def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="suitesparse",
        help="name of a backend in grblas.backends",
    )
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption(
        "--blocking",
        dest="blocking",
        default=True,
        action="store_true",
        help="run in blocking mode",
    )
    parser.addoption(
        "--nonblocking",
        "--no-blocking",
        "--non-blocking",
        dest="blocking",
        action="store_false",
        help="run in non-blocking mode",
    )

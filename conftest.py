def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="name of a graphblas backend",
    )
    parser.addoption("--runslow", default=None, action="store_true", help="run slow tests")
    parser.addoption(
        "--blocking",
        dest="blocking",
        default=None,
        action="store_true",
        help="run in blocking mode",
    )
    parser.addoption(
        "--nonblocking",
        "--no-blocking",
        "--noblocking",
        "--non-blocking",
        dest="blocking",
        action="store_false",
        help="run in non-blocking mode",
    )
    parser.addoption(
        "--record",
        dest="record",
        default=None,
        action="store_true",
        help="Record GraphBLAS C calls and save to 'record.txt'",
    )
    parser.addoption(
        "--mapnumpy", action="store_true", default=None, help="map numpy ops to GraphBLAS ops"
    )
    parser.addoption(
        "--nomapnumpy",
        "--no-mapnumpy",
        dest="mapnumpy",
        action="store_false",
        help="don't map numpy ops to GraphBLAS ops",
    )
    parser.addoption("--randomly", action="store_true", help="run random test config")

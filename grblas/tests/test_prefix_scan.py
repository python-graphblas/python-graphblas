import numpy as np
import pytest

import grblas as gb


@pytest.mark.parametrize("method", ["scan_rows", "scan_columns"])
@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_matrix(method, length, do_random):
    if do_random:
        A = np.random.randint(10, size=2 * length).reshape((2, length))
        mask = (A % 2).astype(bool)
        A[~mask] = 0
        M = gb.Matrix.ss.import_bitmapr(values=A, bitmap=mask, name="A")
        expected = A.cumsum(axis=1)
        expected[~mask] = 0
    else:
        A = np.arange(2 * length).reshape((2, length))
        M = gb.Matrix.ss.import_fullr(values=A, name="A")
        expected = A.cumsum(axis=1)

    if method == "scan_rows":
        R = M.ss.scan_rows()
    else:
        M = M.T.new(name="A")
        R = M.ss.scan_columns().T.new()

    result = gb.io.to_numpy(R)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover
        print(M)
        raise


@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_vector(length, do_random):
    if do_random:
        a = np.random.randint(10, size=length)
        mask = (a % 2).astype(bool)
        a[~mask] = 0
        v = gb.Vector.ss.import_bitmap(values=a, bitmap=mask)
        expected = a.cumsum()
        expected[~mask] = 0
    else:
        a = np.arange(length)
        v = gb.Vector.ss.import_full(values=a)
        expected = a.cumsum()
    r = v.ss.scan()
    result = gb.io.to_numpy(r)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover
        print(v)
        raise

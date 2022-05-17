import numpy as np
import pytest

import graphblas as gb
from graphblas import binary, monoid

from graphblas import Matrix, Vector  # isort:skip (for dask-graphblas)

try:
    # gb.io.to_numpy currently requires scipy
    import scipy.sparse as ss
except ImportError:  # pragma: no cover
    ss = None


@pytest.mark.skipif("not ss")
@pytest.mark.parametrize("method", ["scan_rowwise", "scan_columnwise"])
@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_matrix(method, length, do_random):
    if do_random:
        A = np.random.randint(10, size=2 * length).reshape((2, length))
        mask = (A % 2).astype(bool)
        A[~mask] = 0
        M = Matrix.ss.import_bitmapr(values=A, bitmap=mask, name="A")
        expected = A.cumsum(axis=1)
        expected[~mask] = 0
    else:
        A = np.arange(2 * length).reshape((2, length))
        M = Matrix.ss.import_fullr(values=A, name="A")
        expected = A.cumsum(axis=1)

    if method == "scan_rowwise":
        R = M.ss.scan_rowwise()
    else:
        M = M.T.new(name="A")
        R = M.ss.scan_columnwise(binary.plus).T.new()

    result = gb.io.to_numpy(R)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover
        print(M)
        raise


@pytest.mark.skipif("not ss")
@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_vector(length, do_random):
    if do_random:
        a = np.random.randint(10, size=length)
        mask = (a % 2).astype(bool)
        a[~mask] = 0
        v = Vector.ss.import_bitmap(values=a, bitmap=mask)
        expected = a.cumsum()
        expected[~mask] = 0
    else:
        a = np.arange(length)
        v = Vector.ss.import_full(values=a)
        expected = a.cumsum()
    r = v.ss.scan()
    result = gb.io.to_numpy(r)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover
        print(v)
        raise


def test_cumprod():
    v = Vector.from_values([1, 3, 4, 6], [2, 3, 4, 5])
    expected = Vector.from_values([1, 3, 4, 6], [2, 6, 24, 120])
    r = v.ss.scan(monoid.times)
    assert r.isequal(expected)


def test_bad_scan():
    v = Vector.from_values(range(10), range(10))
    with pytest.raises(TypeError, match="Bad type for argument `op`"):
        v.ss.scan(op=binary.first)

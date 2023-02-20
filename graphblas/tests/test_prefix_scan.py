import numpy as np
import pytest

from graphblas import backend, binary, monoid

from graphblas import Matrix, Vector  # isort:skip (for dask-graphblas)

suitesparse = backend == "suitesparse"


@pytest.mark.skipif("not suitesparse")
@pytest.mark.parametrize("method", ["scan_rowwise", "scan_columnwise"])
@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_matrix(method, length, do_random):
    if do_random:
        rng = np.random.default_rng()
        A = rng.integers(10, size=2 * length).reshape((2, length))
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
        R = M.ss.scan()
    else:
        M = M.T.new(name="A")
        R = M.ss.scan(binary.plus, order="col").T.new()

    result = R.to_dense(0)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover (debug)
        print(M)
        raise


@pytest.mark.skipif("not suitesparse")
@pytest.mark.parametrize("length", list(range(34)))
@pytest.mark.parametrize("do_random", [False, True])
def test_scan_vector(length, do_random):
    if do_random:
        rng = np.random.default_rng()
        a = rng.integers(10, size=length)
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
    result = r.to_dense(0)
    try:
        np.testing.assert_array_equal(result, expected)
    except Exception:  # pragma: no cover (debug)
        print(v)
        raise


@pytest.mark.skipif("not suitesparse")
def test_cumprod():
    v = Vector.from_coo([1, 3, 4, 6], [2, 3, 4, 5])
    expected = Vector.from_coo([1, 3, 4, 6], [2, 6, 24, 120])
    r = v.ss.scan(monoid.times)
    assert r.isequal(expected)


@pytest.mark.skipif("not suitesparse")
def test_bad_scan():
    v = Vector.from_coo(range(10), range(10))
    with pytest.raises(TypeError, match="Bad type for argument `op`"):
        v.ss.scan(op=binary.first)

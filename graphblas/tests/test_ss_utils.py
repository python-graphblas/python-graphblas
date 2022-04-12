import numpy as np
import pytest
from numpy.testing import assert_array_equal

from graphblas import Matrix, Vector


@pytest.mark.parametrize("do_iso", [False, True])
def test_vector_head(do_iso):
    v0 = Vector(int, 5)
    if do_iso:
        values1 = values2 = values3 = [1, 1, 1]
    else:
        values1 = [10, 20, 30]
        values2 = [2, 4, 6]
        values3 = [1, 2, 3]
    v1 = Vector.from_values([0, 1, 2], values1)  # full
    v2 = Vector.from_values([1, 3, 5], values2)  # bitmap
    v3 = Vector.from_values([100, 200, 300], values3)  # sparse
    assert v1.ss.export()["format"] == "full"
    assert v2.ss.export()["format"] == "bitmap"
    assert v3.ss.export()["format"] == "sparse"
    assert v1.ss.is_iso is do_iso
    assert v2.ss.is_iso is do_iso
    assert v3.ss.is_iso is do_iso
    for dtype in [None, np.float64]:
        expected_dtype = np.int64 if dtype is None else dtype
        for _ in range(2):
            indices, vals = v0.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [])
            assert_array_equal(vals, [])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v1.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [0, 1])
            assert_array_equal(vals, values1[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [1, 3])
            assert_array_equal(vals, values2[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [100, 200])
            assert_array_equal(vals, values3[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype


@pytest.mark.parametrize("do_iso", [False, True])
def test_matrix_head(do_iso):
    A0 = Matrix(int, 5, 5)
    if do_iso:
        values1 = [1, 1, 1, 1]
        values2 = values3 = values4 = [1, 1, 1]
    else:
        values1 = [1, 2, 3, 4]
        values2 = [1, 2, 4]
        values3 = values4 = [1, 2, 3]

    A1 = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], values1)  # fullr
    A2 = Matrix.from_values([0, 0, 1], [0, 1, 1], values2)  # Bitmap
    A3 = Matrix.from_values([5, 5, 10], [4, 5, 10], values3)  # CSR
    A4 = Matrix.from_values([500, 500, 1000], [400, 500, 1000], values4)  # HyperCSR
    d = A1.ss.export(raw=True)
    assert d["format"] == "fullr"
    d["format"] = "fullc"
    A5 = Matrix.ss.import_any(**d)  # fullc

    d = A2.ss.export(raw=True)
    assert d["format"] == "bitmapr"
    d["format"] = "bitmapc"
    A6 = Matrix.ss.import_any(**d)  # bitmapc

    d = A3.ss.export(raw=True)
    assert d["format"] == "csr"
    d["format"] = "csc"
    d["row_indices"] = d["col_indices"]
    del d["col_indices"]
    A7 = Matrix.ss.import_any(**d)  # csc

    d = A4.ss.export(raw=True)
    assert d["format"] == "hypercsr"
    d["format"] = "hypercsc"
    d["row_indices"] = d["col_indices"]
    del d["col_indices"]
    d["cols"] = d["rows"]
    del d["rows"]
    A8 = Matrix.ss.import_any(**d)  # hypercsc

    assert A1.ss.is_iso is do_iso
    assert A2.ss.is_iso is do_iso
    assert A3.ss.is_iso is do_iso
    assert A4.ss.is_iso is do_iso
    assert A5.ss.is_iso is do_iso
    assert A6.ss.is_iso is do_iso
    assert A7.ss.is_iso is do_iso
    assert A8.ss.is_iso is do_iso

    for dtype in [None, np.float64]:
        expected_dtype = np.int64 if dtype is None else dtype
        for _ in range(2):
            rows, cols, vals = A0.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [])
            assert_array_equal(cols, [])
            assert_array_equal(vals, [])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A1.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 0])
            assert_array_equal(cols, [0, 1])
            assert_array_equal(vals, values1[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 0])
            assert_array_equal(cols, [0, 1])
            assert_array_equal(vals, values2[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [5, 5])
            assert_array_equal(cols, [4, 5])
            assert_array_equal(vals, values3[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A4.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [500, 500])
            assert_array_equal(cols, [400, 500])
            assert_array_equal(vals, values4[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A5.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, values1[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A6.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, values2[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A7.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [4, 5])
            assert_array_equal(cols, [5, 5])
            assert_array_equal(vals, values3[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A8.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [400, 500])
            assert_array_equal(cols, [500, 500])
            assert_array_equal(vals, values4[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

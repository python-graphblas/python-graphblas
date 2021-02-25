import numpy as np
from numpy.testing import assert_array_equal
from grblas import Matrix, Vector


def test_vector_head():
    v0 = Vector.new(int, 5)
    v1 = Vector.from_values([0, 1, 2], [10, 20, 30])  # full
    v2 = Vector.from_values([1, 3, 5], [2, 4, 6])  # bitmap
    v3 = Vector.from_values([100, 200, 300], [1, 2, 3])  # sparse
    assert v1.ss.export()["format"] == "full"
    assert v2.ss.export()["format"] == "bitmap"
    assert v3.ss.export()["format"] == "sparse"

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
            assert_array_equal(vals, [10, 20])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [1, 3])
            assert_array_equal(vals, [2, 4])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [100, 200])
            assert_array_equal(vals, [1, 2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype


def test_matrix_head():
    A0 = Matrix.new(int, 5, 5)
    A1 = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])  # fullr
    A2 = Matrix.from_values([0, 0, 1], [0, 1, 1], [1, 2, 4])  # Bitmap
    A3 = Matrix.from_values([5, 5, 10], [4, 5, 10], [1, 2, 3])  # CSR
    A4 = Matrix.from_values([500, 500, 1000], [400, 500, 1000], [1, 2, 3])  # HyperCSR

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
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 0])
            assert_array_equal(cols, [0, 1])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [5, 5])
            assert_array_equal(cols, [4, 5])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A4.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [500, 500])
            assert_array_equal(cols, [400, 500])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A5.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A6.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A7.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [4, 5])
            assert_array_equal(cols, [5, 5])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A8.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [400, 500])
            assert_array_equal(cols, [500, 500])
            assert_array_equal(vals, [1, 2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

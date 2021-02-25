from numpy.testing import assert_array_equal
from grblas import Matrix, Vector
from grblas._ss_utils import vector_head, matrix_head


def test_vector_head():
    v0 = Vector.new(int, 5)
    v1 = Vector.from_values([0, 1, 2], [10, 20, 30])  # full
    v2 = Vector.from_values([1, 3, 5], [2, 4, 6])  # bitmap
    v3 = Vector.from_values([100, 200, 300], [1, 2, 3])  # sparse
    assert v1.ss.export()["format"] == "full"
    assert v2.ss.export()["format"] == "bitmap"
    assert v3.ss.export()["format"] == "sparse"

    for _ in range(2):
        indices, vals = vector_head(v0, 2, sort=True)
        assert_array_equal(indices, [])
        assert_array_equal(vals, [])

        indices, vals = vector_head(v1, 2, sort=True)
        assert_array_equal(indices, [0, 1])
        assert_array_equal(vals, [10, 20])

        indices, vals = vector_head(v2, 2, sort=True)
        assert_array_equal(indices, [1, 3])
        assert_array_equal(vals, [2, 4])

        indices, vals = vector_head(v3, 2, sort=True)
        assert_array_equal(indices, [100, 200])
        assert_array_equal(vals, [1, 2])


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

    for _ in range(2):
        rows, cols, vals = matrix_head(A0, 2, sort=True)
        assert_array_equal(rows, [])
        assert_array_equal(cols, [])
        assert_array_equal(vals, [])

        rows, cols, vals = matrix_head(A1, 2, sort=True)
        assert_array_equal(rows, [0, 0])
        assert_array_equal(cols, [0, 1])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A2, 2, sort=True)
        assert_array_equal(rows, [0, 0])
        assert_array_equal(cols, [0, 1])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A3, 2, sort=True)
        assert_array_equal(rows, [5, 5])
        assert_array_equal(cols, [4, 5])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A4, 2, sort=True)
        assert_array_equal(rows, [500, 500])
        assert_array_equal(cols, [400, 500])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A5, 2, sort=True)
        assert_array_equal(rows, [0, 1])
        assert_array_equal(cols, [0, 0])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A6, 2, sort=True)
        assert_array_equal(rows, [0, 1])
        assert_array_equal(cols, [0, 0])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A7, 2, sort=True)
        assert_array_equal(rows, [4, 5])
        assert_array_equal(cols, [5, 5])
        assert_array_equal(vals, [1, 2])

        rows, cols, vals = matrix_head(A8, 2, sort=True)
        assert_array_equal(rows, [400, 500])
        assert_array_equal(cols, [500, 500])
        assert_array_equal(vals, [1, 2])

"""Import/export conversions between graphblas objects and numpy/scipy formats.

Sparse conversions (from_coo/to_coo, scipy interop) run on ~1e6 nonzeros. Dense
conversions use a separate fully dense 1000x1000 matrix so nothing materializes a
1e6 x 1e6 dense array.
"""

import numpy as np

from graphblas import Matrix, Vector, io

try:
    from . import common
except ImportError:
    import common


class MatrixCoo:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.rows, self.cols, self.vals = common.make_coo(common.LARGE_N, common.LARGE_NNZ, seed=40)
        self.M = common.make_matrix(seed=40)

    def time_from_coo(self):
        Matrix.from_coo(
            self.rows,
            self.cols,
            self.vals,
            nrows=common.LARGE_N,
            ncols=common.LARGE_N,
            dup_op="plus",
        )

    def time_to_coo(self):
        self.M.to_coo()


class MatrixDense:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.dense = np.random.default_rng(41).random((common.DENSE_DIM, common.DENSE_DIM))
        self.M = Matrix.from_dense(self.dense)

    def time_from_dense(self):
        Matrix.from_dense(self.dense)

    def time_to_dense(self):
        self.M.to_dense()


class MatrixScipy:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.sp = common.make_scipy(seed=42)
        self.M = io.from_scipy_sparse(self.sp)

    def time_from_scipy_sparse(self):
        io.from_scipy_sparse(self.sp)

    def time_to_scipy_sparse(self):
        io.to_scipy_sparse(self.M, format="csr")


class VectorConvert:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.v = common.make_vector(seed=43)
        self.idx, self.vals = self.v.to_coo()
        self.dense_arr = np.random.default_rng(44).random(common.LARGE_N)
        self.dv = Vector.from_dense(self.dense_arr)

    def time_from_coo(self):
        Vector.from_coo(self.idx, self.vals, size=common.LARGE_N, dup_op="plus")

    def time_to_coo(self):
        self.v.to_coo()

    def time_from_dense(self):
        Vector.from_dense(self.dense_arr)

    def time_to_dense(self):
        self.dv.to_dense()

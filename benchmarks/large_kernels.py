"""Large kernels (~1e6 nonzeros): dominated by the C library, so these catch
suitesparse-graphblas-side regressions (and any python-graphblas overhead that
scales with data).

Each op runs once per sample (``number = 1``) with no warmup, because the inputs
are large enough that a single call is well above timer resolution and we do not
want asv auto-tuning ``number`` up into multi-second batches. ``repeat`` and the
timeout keep total time bounded. Average degree ~1 keeps ``mxm`` output near the
input size instead of exploding into quadratic fill-in.
"""

from graphblas import binary, monoid, semiring, unary

try:
    from . import common
except ImportError:
    import common


class VectorLarge:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.v = common.make_vector(seed=30)
        self.u = common.make_vector(seed=31)

    def time_ewise_mult(self):
        self.v.ewise_mult(self.u, binary.times).new()

    def time_ewise_add(self):
        self.v.ewise_add(self.u, monoid.plus).new()

    def time_apply(self):
        self.v.apply(unary.abs).new()

    def time_reduce(self):
        self.v.reduce(monoid.plus).new()

    def peakmem_ewise_add(self):
        self.v.ewise_add(self.u, monoid.plus).new()


class MatrixLarge:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300

    def setup(self):
        self.A = common.make_matrix(seed=32)
        self.B = common.make_matrix(seed=33)
        self.x = common.make_dense_vector(common.LARGE_N, seed=34)

    def time_mxv(self):
        self.A.mxv(self.x, semiring.plus_times).new()

    def time_mxm(self):
        self.A.mxm(self.B, semiring.plus_times).new()

    def time_ewise_add(self):
        self.A.ewise_add(self.B, monoid.plus).new()

    def time_reduce_rowwise(self):
        self.A.reduce_rowwise(monoid.plus).new()

    def time_reduce_scalar(self):
        self.A.reduce_scalar(monoid.plus).new()

    def peakmem_mxm(self):
        self.A.mxm(self.B, semiring.plus_times).new()

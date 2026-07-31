"""Small-operand op overhead.

On ~100-element objects the C kernels are trivially fast, so these times isolate
the Python-side cost of building an expression and resolving operators. The
``time_build_*`` benchmarks stop at the expression object (no ``.new()``) to
separate expression construction from evaluation.
"""

from graphblas import binary, monoid, semiring, unary

try:
    from . import common
except ImportError:
    import common


class SmallVector:
    params = [10, 100, 1000]
    param_names = ["size"]

    def setup(self, size):
        # Dense vectors so every op touches ``size`` elements.
        self.v = common.make_dense_vector(size, seed=20)
        self.u = common.make_dense_vector(size, seed=21)

    def time_build_ewise_mult(self, size):
        # Expression only, not evaluated: pure construction overhead.
        self.v.ewise_mult(self.u, binary.times)

    def time_ewise_mult(self, size):
        self.v.ewise_mult(self.u, binary.times).new()

    def time_ewise_add(self, size):
        self.v.ewise_add(self.u, monoid.plus).new()

    def time_apply(self, size):
        self.v.apply(unary.abs).new()

    def time_apply_bind_scalar(self, size):
        self.v.apply(binary.plus, right=1.0).new()

    def time_reduce(self, size):
        self.v.reduce(monoid.plus).new()

    def time_assign_into(self, size):
        # The `<<` update path: evaluate into an existing object with no mask/accum.
        self.v << self.v.ewise_mult(self.u, binary.times)


class SmallMatrix:
    params = [10, 100]
    param_names = ["dim"]

    def setup(self, dim):
        # Dense dim x dim (dim**2 entries): 100 or 10000 nonzeros.
        self.A = common.make_dense_matrix(dim=dim, seed=22)
        self.B = common.make_dense_matrix(dim=dim, seed=23)

    def time_ewise_mult(self, dim):
        self.A.ewise_mult(self.B, binary.times).new()

    def time_apply(self, dim):
        self.A.apply(unary.abs).new()

    def time_reduce_rowwise(self, dim):
        self.A.reduce_rowwise(monoid.plus).new()

    def time_reduce_scalar(self, dim):
        self.A.reduce_scalar(monoid.plus).new()

    def time_mxm(self, dim):
        self.A.mxm(self.B, semiring.plus_times).new()

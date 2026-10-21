"""The experimental n-d ``Tensor``: regression tracking for its main operation families.

All inputs are synthetic and seeded (``common.make_tensor``): a uniform 3-d tensor of ~1e6
entries in 200 x 300 x 400, in the default layout (rows (0, 1), columns (2,)), and a 4-d one of
~5e5 entries for the n-d kernels. Each benchmark is one call, materialized with ``.new()`` or
``.wait()``, so lazy work and relayouts are inside the timing.

The Tensor has several engines for TTM, MTTKRP and gathers (Numba kernels, an array-UDT
engine that needs the SuiteSparse JIT, and plain GraphBLAS products); these benchmarks run
whatever the Tensor picks in the benchmark environment. Numba kernels compile on first use
(cached on disk afterwards), so the classes that use them call each operation once in
``setup``. The cross-library comparisons are not here: they live in ``scripts/tensor_bench/``,
outside the directory asv imports, so asv never loads the other libraries.
"""

import numpy as np

from graphblas import Matrix, Vector, binary, monoid, select, semiring

try:
    from . import common
except ImportError:
    import common


class _TensorBench:
    number = 1
    repeat = 5
    warmup_time = 0
    timeout = 300


class TensorEwise(_TensorBench):
    def setup(self):
        self.X = common.make_tensor(seed=50)
        self.Y = common.make_tensor(seed=51)
        # Y's values stored with the axes rolled, seen through a permuted view, so that the
        # element-wise op meets operands in different layouts
        Z = self.Y.permute_dims((1, 2, 0)).relayout(((0, 1), (2,)))
        self.Yp = Z.permute_dims((2, 0, 1))
        rng = np.random.default_rng(52)
        shape = common.TENSOR_SHAPE
        self.v0 = rng.random((shape[0], 1, 1))
        self.v1 = rng.random((shape[1], 1))
        self.v2 = rng.random(shape[2])
        self.M12 = rng.random(shape[1:])

    def time_add(self):
        (self.X + self.Y).new().wait()

    def time_mult(self):
        (self.X * self.Y).new().wait()

    def time_add_permuted_operand(self):
        (self.X + self.Yp).new().wait()

    def time_ewise_union(self):
        self.X.ewise_union(self.Y, binary.minus, 0.0, 0.0).new().wait()

    def time_broadcast_vector_axis0(self):
        (self.X * self.v0).new().wait()

    def time_broadcast_vector_axis1(self):
        (self.X * self.v1).new().wait()

    def time_broadcast_vector_axis2(self):
        (self.X * self.v2).new().wait()

    def time_broadcast_matrix(self):
        (self.X * self.M12).new().wait()

    def time_apply_select(self):
        self.X.apply(binary.times, 2.0).new().select(select.valuegt, 1.0).new().wait()


class TensorReduce(_TensorBench):
    params = [[0, 1, 2, (0, 1), (1, 2), (0, 2), None]]
    param_names = ["axis"]

    def setup(self, axis):
        self.X = common.make_tensor(seed=53)

    def time_sum(self, axis):
        r = self.X.sum(axis=axis).new()
        if axis is not None:
            r.wait()

    def time_max(self, axis):
        r = self.X.reduce(monoid.max, axis=axis).new()
        if axis is not None:
            r.wait()


class TensorPermute(_TensorBench):
    params = [[(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]]
    param_names = ["axes"]

    def setup(self, axes):
        self.X = common.make_tensor(seed=54)

    def time_permute_view(self, axes):
        self.X.permute_dims(axes)

    def time_permute_materialize(self, axes):
        # permute_dims is an O(1) view; storing it in the identity axis order moves the data
        self.X.permute_dims(axes).relayout(((0, 1), (2,)))


class TensorReshape(_TensorBench):
    def setup(self):
        self.X = common.make_tensor(seed=55)
        s0, s1, s2 = common.TENSOR_SHAPE
        self.fused = (s0 * s1, s2)
        self.split = (s0, s1, 20, s2 // 20)
        self.across = (s0, s1 * s2)

    def time_reshape_view(self):
        # (0, 1) are the row group: fusing them is a copy-on-write view
        self.X.reshape(self.fused).wait()

    def time_reshape_split_view(self):
        self.X.reshape(self.split).wait()

    def time_reshape_move(self):
        # fusing axes 1 and 2 crosses the row/column split, so the data moves
        self.X.reshape(self.across).wait()


class TensorConstruct(_TensorBench):
    def setup(self):
        from graphblas import Tensor

        self.Tensor = Tensor
        self.coords, self.vals = common.make_tensor_coo(seed=56)
        order = np.random.default_rng(57).permutation(self.vals.size)
        self.shuffled = [c[order] for c in self.coords], self.vals[order]
        self.X = common.make_tensor(seed=56)

    def time_from_coo_sorted(self):
        self.Tensor.from_coo(self.coords, self.vals, shape=common.TENSOR_SHAPE).wait()

    def time_from_coo_shuffled(self):
        c, v = self.shuffled
        self.Tensor.from_coo(c, v, shape=common.TENSOR_SHAPE).wait()

    def time_to_coo(self):
        self.X.to_coo()

    def peakmem_from_coo_shuffled(self):
        c, v = self.shuffled
        self.Tensor.from_coo(c, v, shape=common.TENSOR_SHAPE).wait()


class TensorContract(_TensorBench):
    params = [[0, 1, 2]]
    param_names = ["mode"]

    def setup(self, mode):
        self.X = common.make_tensor(seed=58)
        shape = common.TENSOR_SHAPE
        self.v = Vector.from_dense(np.random.default_rng(59).random(shape[mode]))
        self.U = Matrix.from_dense(common.make_factor(shape[mode], 16, seed=60))
        self.F = [common.make_factor(n, 16, seed=61 + a) for a, n in enumerate(shape)]
        # Numba kernels compile on first use in a process
        self.time_ttm(mode)
        self.time_mttkrp(mode)

    def time_ttv(self, mode):
        self.X.tensordot(self.v, ([mode], [0])).new().wait()

    def time_ttm(self, mode):
        self.X.tensordot(self.U, ([mode], [0])).new().wait()

    def time_mttkrp(self, mode):
        self.X.mttkrp(self.F, mode).new().wait()


class TensorContract4d(_TensorBench):
    params = [[0, 3]]
    param_names = ["mode"]

    def setup(self, mode):
        shape = common.TENSOR4_SHAPE
        self.X = common.make_tensor(shape, common.TENSOR4_NNZ, seed=62)
        self.F = [common.make_factor(n, 16, seed=63 + a) for a, n in enumerate(shape)]
        self.time_mttkrp(mode)

    def time_mttkrp(self, mode):
        self.X.mttkrp(self.F, mode).new().wait()


class TensorProducts(_TensorBench):
    def setup(self):
        from graphblas import Tensor

        self.X = common.make_tensor(seed=64)
        shape = common.TENSOR_SHAPE
        rng = np.random.default_rng(65)
        # 'ijk,kl->ijl' with a sparse (400 x 100) matrix, two entries per row
        rows = np.repeat(np.arange(shape[2]), 2)
        cols = rng.integers(0, 100, rows.size)
        self.S = Matrix.from_coo(rows, cols, rng.random(rows.size), nrows=shape[2], ncols=100,
                                 dup_op=binary.plus)  # fmt: skip
        # 'ijk,jkl->il' with a sparse (300, 400, 50) tensor of ~1e5 entries
        self.Y = common.make_tensor((shape[1], shape[2], 50), 100_000, seed=66)
        # batched 'bij,bjk->bik': 100 batches of 200 x 200 at 5 entries per row
        self.A = common.make_tensor((100, 200, 200), 100_000, seed=67)
        self.B = common.make_tensor((100, 200, 200), 100_000, seed=68)
        self.Tensor = Tensor

    def time_tensordot_matrix(self):
        self.X.tensordot(self.S, ([2], [0])).new().wait()

    def time_tensordot_two_axes(self):
        self.X.tensordot(self.Y, ([1, 2], [0, 1])).new().wait()

    def time_batched_matmul(self):
        (self.A @ self.B).new().wait()


class TensorEinsum(_TensorBench):
    def setup(self):
        from graphblas import einsum

        self.einsum = einsum
        self.X = common.make_tensor(seed=69)
        _, s1, s2 = common.TENSOR_SHAPE
        self.W = Matrix.from_dense(common.make_factor(s2, 8, seed=70))
        self.B = common.make_factor(s1, 16, seed=71)
        self.C = common.make_factor(s2, 16, seed=72)
        self.u = np.random.default_rng(73).random(s1)
        self.v = np.random.default_rng(74).random(s2)
        self.time_einsum_mttkrp()

    def time_einsum_contract(self):
        self.einsum("ijk,kl->ijl", self.X, self.W).new().wait()

    def time_einsum_mttkrp(self):
        self.einsum("ijk,jr,kr->ir", self.X, self.B, self.C).new().wait()

    def time_einsum_three_operands(self):
        self.einsum("ijk,j,k->i", self.X, self.u, self.v).new().wait()


class TensorSmall:
    """A 2 x 3 x 4 Tensor: the time is Python-side overhead (parsing, planning, expressions)."""

    def setup(self):
        from graphblas import Tensor, einsum

        self.einsum = einsum
        self.X = Tensor.from_dense(np.arange(24.0).reshape(2, 3, 4))
        self.Y = Tensor.from_dense(np.arange(24.0).reshape(2, 3, 4) % 5, missing_value=0)
        self.v = np.ones(4)

    def time_einsum(self):
        self.einsum("ijk,k->ij", self.X, self.v).new()

    def time_tensordot(self):
        self.X.tensordot(self.v, ([2], [0])).new()

    def time_ewise_add(self):
        (self.X + self.Y).new()

    def time_sum_axis(self):
        self.X.sum(axis=1).new()

    def time_permute_view(self):
        self.X.permute_dims((2, 0, 1))

    def time_extract_element(self):
        self.X[1, 2, 3].new()


class TensorExtract(_TensorBench):
    def setup(self):
        self.X = common.make_tensor(seed=75)
        s0, s1, s2 = common.TENSOR_SHAPE
        self.mid = (s0 // 2, s1 // 2, s2 // 2)
        self.ranges = (slice(s0 // 10, s0 // 5), slice(s1 // 10, s1 // 5), slice(s2 // 10, s2 // 5))
        self.lists = [
            np.random.default_rng(76).choice(n, n // 10, replace=False) for n in (s0, s1, s2)
        ]

    def time_slab_axis0(self):
        self.X[self.mid[0]].new().wait()

    def time_slab_axis1(self):
        self.X[:, self.mid[1]].new().wait()

    def time_slab_axis2(self):
        self.X[:, :, self.mid[2]].new().wait()

    def time_range_axis0(self):
        self.X[self.ranges[0]].new().wait()

    def time_range_axis2(self):
        self.X[:, :, self.ranges[2]].new().wait()

    def time_index_lists(self):
        self.X[self.lists[0], self.lists[1], self.lists[2]].new().wait()

    def time_element(self):
        self.X[self.mid].new()


class TensorAssign(_TensorBench):
    def setup(self):
        self.X = common.make_tensor(seed=77)
        s0, s1, s2 = common.TENSOR_SHAPE
        self.region = (slice(s0 // 4, s0 // 2), slice(None), slice(s2 // 4, s2 // 2))
        # a value covering the region, ~2% dense
        shape = (s0 // 4, s1, s2 // 4)
        self.value = common.make_tensor(shape, int(np.prod(shape) * 0.02), seed=78)
        self.Y = common.make_tensor(seed=79)
        self.mask = self.Y.S

    # In place on the Tensor that setup made (one call per setup: number = 1)

    def time_assign_tensor_region(self):
        self.X[self.region] << self.value
        self.X.wait()

    def time_assign_scalar_slab(self):
        self.X[:, 7, :] << 1.0
        self.X.wait()

    def time_masked_update(self):
        self.X(self.mask, accum=binary.plus) << self.Y
        self.X.wait()

    def time_delete_slab(self):
        del self.X[3]
        self.X.wait()


class TensorGather(_TensorBench):
    params = [[10_000, 1_000_000]]
    param_names = ["npoints"]

    def setup(self, npoints):
        self.X = common.make_tensor(seed=80)
        rng = np.random.default_rng(81)
        shape = common.TENSOR_SHAPE
        self.points = [rng.integers(0, n, npoints) for n in shape]
        self.flat = rng.integers(0, int(np.prod(shape)), npoints)
        self.along = rng.integers(0, shape[2], (shape[0], shape[1], 4))
        self.time_vindex(npoints)

    def time_vindex(self, npoints):
        self.X.vindex[self.points[0], self.points[1], self.points[2]].wait()

    def time_take_flat(self, npoints):
        self.X.take(self.flat).wait()

    def time_take_axis(self, npoints):
        self.X.take(self.points[2][:100], axis=2).wait()

    def time_take_along_axis(self, npoints):
        self.X.take_along_axis(self.along, axis=2).wait()


class TensorSemiring(_TensorBench):
    """Contractions with a semiring other than plus_times (the general GraphBLAS path)."""

    def setup(self):
        self.X = common.make_tensor(seed=82)
        s2 = common.TENSOR_SHAPE[2]
        self.U = Matrix.from_dense(common.make_factor(s2, 16, seed=83))

    def time_ttm_min_plus(self):
        self.X.tensordot(self.U, ([2], [0]), op=semiring.min_plus).new().wait()

"""`grblas.agg` is an experimental module for exploring Aggregators.

Aggregators may be used in reduce methods:
    - Matrix.reduce_rows
    - Matrix.reduce_columns
    - Matrix.reduce_scalar
    - Vector.reduce

Aggregators are implemented as recipes that often use monoids or semirings.

Monoid-only aggregators:
    - sum
    - prod
    - all
    - any
    - min
    - max
    - any_value
    - bitwise_all
    - bitwise_any

Semiring aggregators with O(1) dense vector:
    - count
    - count_nonzero
    - count_zero
    - sum_of_squares
    - sum_of_inverses

Semiring aggregators with UnaryOp applied to the final result:
    - hypot
    - logaddexp
    - logaddexp2

Composite aggregators (require multiple aggregation steps):
    - mean
    - ptp, peak-to-peak
    - varp, population variance
    - vars, sample variance
    - stdp, population standard deviation
    - stds, sample standard deviation
    - geometric_mean
    - harmonic_mean

Custom recipes:
    - first
    - last
    - argmin, doesn't work with Matrix.reduce_scalar
    - argmini
    - argminj, doesn't work with Vector.reduce
    - argmax, doesn't work with Matrix.reduce_scalar
    - argmaxi
    - argmaxj, doesn't work with Vector.reduce
"""
from functools import partial

import numpy as np

import grblas as gb


class Aggregator:
    opclass = "Aggregator"

    def __init__(
        self,
        name,
        *,
        initval=False,
        monoid=None,
        semiring=None,
        switch=False,
        semiring2=None,
        finalize=None,
        composite=None,
        custom=None,
    ):
        self.name = name
        self._initval = initval
        self._initdtype = gb.dtypes.lookup_dtype(type(initval))
        self._monoid = monoid
        self._semiring = semiring
        self._semiring2 = semiring2
        self._switch = switch
        self._finalize = finalize
        self._composite = composite
        self._custom = custom

    def __getitem__(self, dtype):
        return TypedAggregator(self, dtype)


class TypedAggregator:
    opclass = "Aggregator"

    def __init__(self, agg, dtype):
        self.name = agg.name
        self._agg = agg
        self._dtype = dtype
        # TODO: should we specialize the monoid or semiring based on dtype?
        if agg._monoid is not None:
            self.return_type = agg._monoid[dtype].return_type
        elif agg._semiring is not None:
            self.return_type = agg._semiring[dtype].return_type
        elif agg._composite is not None:
            self.return_type = "FP64"  # TODO
        elif agg._custom is not None:
            self.return_type = "INT64"  # TODO
        else:
            raise NotImplementedError()

    def __repr__(self):
        return f"{self.name}[{self._dtype}]"

    def _new(self, updater, expr, *, in_composite=False):
        agg = self._agg
        if agg._monoid is not None:
            updater << getattr(expr.args[0], expr.method_name)(agg._monoid)
            if in_composite:
                parent = updater.parent
                if not parent._is_scalar:
                    return parent
                rv = gb.Vector.new(parent.dtype, size=1)
                rv[0] = parent
                return rv
        elif agg._composite is not None:
            results = []
            mask = updater.kwargs.get("mask")
            for cur_agg in agg._composite:
                arg = expr.construct_output(dtype=updater.parent.dtype)
                results.append(cur_agg[self._dtype]._new(arg(mask=mask), expr, in_composite=True))
            final_expr = agg._finalize(*results)
            if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
                updater << final_expr
            elif expr.cfunc_name.startswith("GrB_Vector_reduce") or expr.cfunc_name.startswith(
                "GrB_Matrix_reduce"
            ):
                final = final_expr.new()
                updater << final.reduce(gb.monoid.any)
            else:
                raise NotImplementedError()
            if in_composite:
                1 / 0
                return updater.parent
        elif agg._custom is not None:
            return agg._custom(self, updater, expr, in_composite=in_composite)
        elif expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
            # Matrix -> Vector
            A = expr.args[0]
            orig_updater = updater
            if agg._finalize is not None:
                step1 = expr.construct_output(dtype=updater.parent.dtype)
                updater = step1(mask=updater.kwargs.get("mask"))
            if expr.method_name == "reduce_columns":
                A = A.T
            size = A._ncols
            init = gb.Vector.new(agg._initdtype, size=size)
            init[:] = agg._initval  # O(1) dense vector in SuiteSparse 5
            if agg._switch:
                updater << agg._semiring(init @ A.T)
            else:
                updater << agg._semiring(A @ init)
            if agg._finalize is not None:
                orig_updater << agg._finalize(step1)
            if in_composite:
                return orig_updater.parent
        elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
            # Vector -> Scalar
            v = expr.args[0]
            step1 = gb.Vector.new(updater.parent.dtype, size=1)
            init = gb.Matrix.new(agg._initdtype, nrows=v._size, ncols=1)
            init[:, :] = agg._initval  # O(1) dense column vector in SuiteSparse 5
            if agg._switch:
                step1 << agg._semiring(init.T @ v)
            else:
                step1 << agg._semiring[float](v @ init)
            if agg._finalize is not None:
                step1 << agg._finalize(step1)
            if in_composite:
                return step1
            updater << step1.reduce(gb.monoid.any)
        elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
            # Matrix -> Scalar
            A = expr.args[0]
            # We need to compute in two steps: Matrix -> Vector -> Scalar.
            # This has not been benchmarked or optimized.
            # We may be able to intelligently choose the faster path.
            init1 = gb.Vector.new(agg._initdtype, size=A._ncols)
            init1[:] = agg._initval  # O(1) dense vector in SuiteSparse 5
            step1 = gb.Vector.new(updater.parent.dtype, size=A._nrows)
            if agg._switch:
                step1 << agg._semiring(init1 @ A.T)
            else:
                step1 << agg._semiring(A @ init1)
            init2 = gb.Matrix.new(agg._initdtype, nrows=A._nrows, ncols=1)
            init2[:, :] = agg._initval  # O(1) dense vector in SuiteSparse 5
            step2 = gb.Vector.new(updater.parent.dtype, size=1)
            step2 << agg._semiring2(step1 @ init2)
            if agg._finalize is not None:
                step2 << agg._finalize(step2)
            if in_composite:
                return step2
            updater << step2.reduce(gb.monoid.any)
        else:
            raise NotImplementedError()


# Monoid-only
sum = Aggregator("sum", monoid=gb.monoid.plus)
prod = Aggregator("prod", monoid=gb.monoid.times)
all = Aggregator("all", monoid=gb.monoid.land)
any = Aggregator("any", monoid=gb.monoid.lor)
min = Aggregator("min", monoid=gb.monoid.min)
max = Aggregator("max", monoid=gb.monoid.max)
any_value = Aggregator("any_value", monoid=gb.monoid.any)
bitwise_all = Aggregator("bitwise_all", monoid=gb.monoid.band)
bitwise_any = Aggregator("bitwise_any", monoid=gb.monoid.bor)
# Other monoids: bxnor bxor eq lxnor lxor

# Semiring-only
count = Aggregator("count", semiring=gb.semiring.plus_pair, semiring2=gb.semiring.plus_first)
count_nonzero = Aggregator(
    "count_nonzero", semiring=gb.semiring.plus_isne, semiring2=gb.semiring.plus_first
)
count_zero = Aggregator(
    "count_zero", semiring=gb.semiring.plus_iseq, semiring2=gb.semiring.plus_first
)
sum_of_squares = Aggregator(
    "sum_of_squares", initval=2, semiring=gb.semiring.plus_pow, semiring2=gb.semiring.plus_first
)
sum_of_inverses = Aggregator(
    "sum_of_inverses", initval=-1, semiring=gb.semiring.plus_pow, semiring2=gb.semiring.plus_first
)

# Semiring and finalize
hypot = Aggregator(
    "hypot",
    initval=2,
    semiring=gb.semiring.plus_pow,
    semiring2=gb.semiring.plus_first,
    finalize=gb.unary.sqrt,
)
logaddexp = Aggregator(
    "logaddexp",
    initval=np.e,
    semiring=gb.semiring.plus_pow,
    switch=True,
    semiring2=gb.semiring.plus_first,
    finalize=gb.unary.log,
)
logaddexp2 = Aggregator(
    "logaddexp2",
    initval=2,
    semiring=gb.semiring.plus_pow,
    switch=True,
    semiring2=gb.semiring.plus_first,
    finalize=gb.unary.log2,
)
# Alternatives
# hypot = Aggregator('hypot', monoid=gb.semiring.numpy.hypot)
# logaddexp = Aggregator('logaddexp', monoid=gb.semiring.numpy.logaddexp)
# logaddexp2 = Aggregator('logaddexp2', monoid=gb.semiring.numpy.logaddexp2)


# Composite
def _mean_finalize(c, x):
    return gb.binary.truediv(x & c)


def _ptp_finalize(max, min):
    return gb.binary.minus(max & min)


def _varp_finalize(c, x, x2):
    # <x2> / n - (<x> / n)**2
    x2 << gb.binary.truediv(x2 & c)
    x << gb.binary.truediv(x & c)
    x << gb.binary.pow(x, 2)
    return gb.binary.minus(x2 & x)


def _vars_finalize(c, x, x2):
    # <x2> / (n-1) - <x>**2 / (n * (n-1))
    x << gb.binary.pow(x, 2)
    x << gb.binary.truediv(x & c)
    c << gb.binary.minus(c, 1)
    x << gb.binary.truediv(x & c)
    x2 << gb.binary.truediv(x2 & c)
    return gb.binary.minus(x2 & x)


def _stdp_finalize(c, x, x2):
    x << _varp_finalize(c, x, x2)
    return gb.unary.sqrt(x)


def _stds_finalize(c, x, x2):
    x << _vars_finalize(c, x, x2)
    return gb.unary.sqrt(x)


def _geometric_mean_finalize(c, x):
    c << gb.unary.minv(c)  # XXX: should be float
    return gb.binary.pow(x & c)


def _harmonic_mean_finalize(c, x):
    return gb.binary.truediv(c & x)


mean = Aggregator(
    "mean",
    composite=[count, sum],
    finalize=_mean_finalize,
)
ptp = Aggregator(
    "ptp",
    composite=[max, min],
    finalize=_ptp_finalize,
)
varp = Aggregator(
    "varp",
    composite=[count, sum, sum_of_squares],
    finalize=_varp_finalize,
)
vars = Aggregator(
    "vars",
    composite=[count, sum, sum_of_squares],
    finalize=_vars_finalize,
)
stdp = Aggregator(
    "stdp",
    composite=[count, sum, sum_of_squares],
    finalize=_stdp_finalize,
)
stds = Aggregator(
    "stds",
    composite=[count, sum, sum_of_squares],
    finalize=_stds_finalize,
)
geometric_mean = Aggregator(
    "geometric_mean",
    composite=[count, prod],
    finalize=_geometric_mean_finalize,
)
harmonic_mean = Aggregator(
    "harmonic_mean",
    composite=[count, sum_of_inverses],
    finalize=_harmonic_mean_finalize,
)


# Special recipes
def _argminmaxij(
    agg,
    updater,
    expr,
    *,
    in_composite,
    monoid,
    col_semiring,
    row_semiring,
    allow_vector=True,
):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_rows":
            step1 = A.reduce_rows(monoid).new()

            # TODO: use diag
            i, j = step1.to_values()
            D = gb.Matrix.from_values(i, i, j, nrows=A._nrows, ncols=A._nrows)

            masked = gb.semiring.any_eq(D @ A).new()
            masked(mask=masked.V, replace=True) << masked  # Could use select
            init = gb.Vector.new(bool, size=A._ncols)
            init[:] = False  # O(1) dense vector in SuiteSparse 5
            updater << row_semiring(masked @ init)
            if in_composite:
                1 / 0
                return updater.parent
        else:
            step1 = A.reduce_columns(monoid).new()

            # TODO: use diag
            i, j = step1.to_values()
            D = gb.Matrix.from_values(i, i, j, nrows=A._ncols, ncols=A._ncols)

            masked = gb.semiring.any_eq(A @ D).new()
            masked(mask=masked.V, replace=True) << masked  # Could use select
            init = gb.Vector.new(bool, size=A._nrows)
            init[:] = False  # O(1) dense vector in SuiteSparse 5
            updater << col_semiring(init @ masked)
            if in_composite:
                1 / 0
                return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        if not allow_vector:
            # XXX: it would be best to raise upon creation of the expression
            raise ValueError(
                f"Aggregator {agg.name} may not be used with Vector.reduce; "
                f"use {agg.name[:-1]} instead."
            )
        v = expr.args[0]
        step1 = v.reduce(monoid).new()
        masked = gb.binary.eq(v, step1).new()
        masked(mask=masked.V, replace=True) << masked  # Could use select
        init = gb.Matrix.new(bool, nrows=v._size, ncols=1)
        init[:, :] = False  # O(1) dense column vector in SuiteSparse 5
        step2 = gb.Vector.new(updater.parent.dtype, size=1)
        step2 << col_semiring(masked @ init)
        if in_composite:
            1 / 0
            return step2
        updater << step2.reduce(gb.monoid.any)
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        A = expr.args[0]
        step1 = A.reduce_scalar(monoid).new()

        masked = gb.binary.eq(A, step1).new()
        masked(mask=masked.V, replace=True) << masked  # Could use select
        init = gb.Vector.new(bool, size=A._nrows)
        init[:] = False  # O(1) dense vector in SuiteSparse 5

        # Always choose the one with smallest i
        step2 = gb.semiring.min_secondi(init @ masked).new()
        step3 = step2.reduce(gb.monoid.min)
        if agg.name in {"argmini", "argmaxi"}:
            # We're done!
            if in_composite:
                1 / 0
                # TODO: put scalar in vector
                raise NotImplementedError()
            updater << step3
            return
        i = step3.value
        step4 = gb.Vector.from_values([i], [False], size=A._nrows)
        step5 = col_semiring(step4 @ masked).new()
        step6 = step5.reduce(gb.monoid.min)
        if in_composite:
            1 / 0
            # TODO: put scalar in vector
            raise NotImplementedError()
        updater << step6
    else:
        raise NotImplementedError()


def _argminmax(agg, updater, expr, *, in_composite, monoid):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        if expr.method_name == "reduce_rows":
            return _argminmaxij(
                agg,
                updater,
                expr,
                in_composite=in_composite,
                monoid=monoid,
                row_semiring=gb.semiring.min_firstj,
                col_semiring=gb.semiring.min_secondj,
            )
        return _argminmaxij(
            agg,
            updater,
            expr,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=gb.semiring.min_firsti,
            col_semiring=gb.semiring.min_secondi,
        )
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        return _argminmaxij(
            agg,
            updater,
            expr,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=gb.semiring.min_firsti,
            col_semiring=gb.semiring.min_secondi,
        )
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        # XXX: it would be best to raise upon creation of the expression
        raise ValueError(
            f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar; "
            f"use {agg.name}i or {agg.name}j instead."
        )
    else:
        raise NotImplementedError()


# argmini (argmaxi) is the same as argmin (argmax) for vectors
argmini = Aggregator(
    "argmini",
    custom=partial(
        _argminmaxij,
        monoid=gb.monoid.min,
        row_semiring=gb.semiring.min_firsti,
        col_semiring=gb.semiring.min_secondi,
    ),
)
argmaxi = Aggregator(
    "argmaxi",
    custom=partial(
        _argminmaxij,
        monoid=gb.monoid.max,
        row_semiring=gb.semiring.min_firsti,
        col_semiring=gb.semiring.min_secondi,
    ),
)
# argminj, argmaxj don't work on vectors
argminj = Aggregator(
    "argminj",
    custom=partial(
        _argminmaxij,
        monoid=gb.monoid.min,
        row_semiring=gb.semiring.min_firstj,
        col_semiring=gb.semiring.min_secondj,
        allow_vector=False,
    ),
)
argmaxj = Aggregator(
    "argmaxj",
    custom=partial(
        _argminmaxij,
        monoid=gb.monoid.max,
        row_semiring=gb.semiring.min_firstj,
        col_semiring=gb.semiring.min_secondj,
        allow_vector=False,
    ),
)
# These "do the right thing", but don't work with `reduce_scalar`
argmin = Aggregator("argmin", custom=partial(_argminmax, monoid=gb.monoid.min))
argmax = Aggregator("argmax", custom=partial(_argminmax, monoid=gb.monoid.max))


def _first_last(agg, updater, expr, *, in_composite, semiring):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_columns":
            A = A.T
        init = gb.Vector.new(bool, size=A._ncols)
        init[:] = False
        step1 = semiring(A @ init).new()
        Is, Js = step1.to_values()
        # TODO: perform these loops in e.g. Cython
        # Populate numpy array
        vals = np.empty(Is.size, dtype=A.dtype.np_type)
        for index, (i, j) in enumerate(zip(Is, Js)):
            vals[index] = A[i, j].value
        # or Vector
        # v = gb.Vector.new(A.dtype, size=A._nrows)
        # for i, j in zip(Is, Js):
        #     v[i] = A[i, j].value
        result = gb.Vector.from_values(Is, vals)
        updater << result
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        init = gb.Matrix.new(bool, nrows=v._size, ncols=1)
        init[:, :] = False
        step1 = semiring(v @ init).new()
        index = step1[0].value
        if index is None:
            index = 0
        updater << v[index]
    else:
        raise NotImplementedError()


first = Aggregator("first", custom=partial(_first_last, semiring=gb.op.min_secondi))
last = Aggregator("last", custom=partial(_first_last, semiring=gb.op.max_secondi))

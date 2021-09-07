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

Vector norms
    - L0norm (=count_nonzero), sum(x != 0).  Not a proper norm.
    - L1norm, sum(abs(x))
    - L2norm (=hypot), sum(x**2)**0.5
    - Linfnorm, max(abs(x))

Composite aggregators (require multiple aggregation steps):
    - mean
    - peak_to_peak, max - min
    - varp, population variance
    - vars, sample variance
    - stdp, population standard deviation
    - stds, sample standard deviation
    - geometric_mean
    - harmonic_mean
    - root_mean_square

Custom recipes:
    - first
    - last
    - argmini
    - argmaxi
    # These don't work with Vector.reduce
    - argminj
    - argmaxj
    # These don't work with Matrix.reduce_scalar
    - first_index  (argfirst?)
    - last_index (arglast?)
    - argmin
    - argmax
    # Misc.
    # absolute_deviation, sum(abs(x - mean(x))),  sum_absminus(x, mean(x))
    # mean_absolute_deviation, absolute_deviation / count
    # firsti, firstj, lasti, lastj

"""
from functools import partial as _partial

import numpy as _np

from . import binary as _binary
from . import monoid as _monoid
from . import semiring as _semiring
from . import unary as _unary
from .dtypes import lookup_dtype as _lookup_dtype
from .dtypes import unify as _unify
from .matrix import Matrix as _Matrix
from .monoid import any as _any
from .operator import _normalize_type
from .scalar import Scalar as _Scalar
from .ss import diag as _diag
from .vector import Vector as _Vector


def _get_types(ops, initdtype):
    if initdtype is None:
        prev = dict(ops[0].types)
    else:
        initdtype = _lookup_dtype(initdtype)
        prev = {
            key: _unify(_lookup_dtype(val), initdtype).name for key, val in ops[0].types.items()
        }
    for op in ops[1:]:
        cur = {}
        types = op.types
        for in_type, out_type in prev.items():
            if out_type not in types:  # pragma: no cover
                continue
            cur[in_type] = types[out_type]
        prev = cur
    return prev


class Aggregator:
    opclass = "Aggregator"

    def __init__(
        self,
        name,
        *,
        initval=None,
        monoid=None,
        semiring=None,
        switch=False,
        semiring2=None,
        finalize=None,
        composite=None,
        custom=None,
        types=None,
    ):
        self.name = name
        self._initval = False if initval is None else initval
        self._initdtype = _lookup_dtype(type(self._initval))
        self._monoid = monoid
        self._semiring = semiring
        self._semiring2 = semiring2
        self._switch = switch
        self._finalize = finalize
        self._composite = composite
        self._custom = custom
        if types is None:
            if monoid is not None:
                types = [monoid]
            elif semiring is not None:
                types = [semiring, semiring2]
                if finalize is not None:
                    types.append(finalize)
                initval = self._initval
            else:  # pragma: no cover
                raise TypeError("types must be provided for composite and custom aggregators")
        self.types = _get_types(types, None if initval is None else self._initdtype)
        self._typed_ops = {}

    def __getitem__(self, dtype):
        dtype = _normalize_type(dtype)
        if dtype not in self.types:
            1 / 0
            raise KeyError(f"{self.name} does not work with {dtype}")
        if dtype not in self._typed_ops:
            self._typed_ops[dtype] = TypedAggregator(self, dtype)
        return self._typed_ops[dtype]


class TypedAggregator:
    opclass = "Aggregator"

    def __init__(self, agg, dtype):
        self.name = agg.name
        self.parent = agg
        self.type = dtype
        self.return_type = agg.types[dtype]

    def __repr__(self):
        return f"{self.name}[{self.type}]"

    def _new(self, updater, expr, *, in_composite=False):
        agg = self.parent
        if agg._monoid is not None:
            x = expr.args[0]
            expr = getattr(x, expr.method_name)(agg._monoid[self.type])
            if expr.output_type is _Scalar and x._nvals == 0:
                # Don't set scalar output to monoid identity if empty
                expr = _Scalar.new(expr.dtype)
            updater << expr
            if in_composite:
                parent = updater.parent
                if not parent._is_scalar:
                    return parent
                rv = _Vector.new(parent.dtype, size=1)
                if parent._nvals != 0:
                    rv[0] = parent
                return rv
            return

        if agg._composite is not None:
            results = []
            mask = updater.kwargs.get("mask")
            for cur_agg in agg._composite:
                cur_agg = cur_agg[self.type]  # Hopefully works well enough
                arg = expr.construct_output(dtype=cur_agg.return_type)
                results.append(cur_agg._new(arg(mask=mask), expr, in_composite=True))
            final_expr = agg._finalize(*results)
            if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
                updater << final_expr
            elif expr.cfunc_name.startswith("GrB_Vector_reduce") or expr.cfunc_name.startswith(
                "GrB_Matrix_reduce"
            ):
                final = final_expr.new()
                expr = final.reduce(_any)
                if final._nvals == 0:
                    expr = _Scalar.new(expr.dtype)
                updater << expr
            else:
                raise NotImplementedError()
            if in_composite:
                1 / 0
                return updater.parent
            return

        if agg._custom is not None:
            return agg._custom(self, updater, expr, in_composite=in_composite)

        dtype = _unify(_lookup_dtype(self.type), _lookup_dtype(agg._initdtype))
        semiring = agg._semiring[dtype]
        if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
            # Matrix -> Vector
            A = expr.args[0]
            orig_updater = updater
            if agg._finalize is not None:
                step1 = expr.construct_output(dtype=semiring.return_type)
                updater = step1(mask=updater.kwargs.get("mask"))
            if expr.method_name == "reduce_columns":
                A = A.T
            size = A._ncols
            init = _Vector.new(agg._initdtype, size=size)
            init[:] = agg._initval  # O(1) dense vector in SuiteSparse 5
            if agg._switch:
                updater << semiring(init @ A.T)
            else:
                updater << semiring(A @ init)
            if agg._finalize is not None:
                orig_updater << agg._finalize[semiring.return_type](step1)
            if in_composite:
                return orig_updater.parent
        elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
            # Vector -> Scalar
            v = expr.args[0]
            step1 = _Vector.new(semiring.return_type, size=1)
            init = _Matrix.new(agg._initdtype, nrows=v._size, ncols=1)
            init[:, :] = agg._initval  # O(1) dense column vector in SuiteSparse 5
            if agg._switch:
                step1 << semiring(init.T @ v)
            else:
                step1 << semiring(v @ init)
            if agg._finalize is not None:
                finalize = agg._finalize[semiring.return_type]
                if step1.dtype == finalize.return_type:
                    step1 << finalize(step1)
                else:
                    step1 = finalize(step1).new(dtype=finalize.return_type)
            if in_composite:
                return step1
            expr = step1.reduce(_any)
            if step1._nvals == 0:
                expr = _Scalar.new(expr.dtype)
            updater << expr
        elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
            # Matrix -> Scalar
            A = expr.args[0]
            # We need to compute in two steps: Matrix -> Vector -> Scalar.
            # This has not been benchmarked or optimized.
            # We may be able to intelligently choose the faster path.
            init1 = _Vector.new(agg._initdtype, size=A._ncols)
            init1[:] = agg._initval  # O(1) dense vector in SuiteSparse 5
            step1 = _Vector.new(semiring.return_type, size=A._nrows)
            if agg._switch:
                step1 << semiring(init1 @ A.T)
            else:
                step1 << semiring(A @ init1)
            init2 = _Matrix.new(agg._initdtype, nrows=A._nrows, ncols=1)
            init2[:, :] = agg._initval  # O(1) dense vector in SuiteSparse 5
            semiring2 = agg._semiring2[semiring.return_type]
            step2 = _Vector.new(semiring2.return_type, size=1)
            step2 << semiring2(step1 @ init2)
            if agg._finalize is not None:
                finalize = agg._finalize[semiring2.return_type]
                if step2.dtype == finalize.return_type:
                    step2 << finalize(step2)
                else:
                    step2 = finalize(step2).new(dtype=finalize.return_type)
            if in_composite:
                return step2
            expr = step2.reduce(_any)
            if step2._nvals == 0:
                expr = _Scalar.new(expr.dtype)
            updater << expr
        else:
            raise NotImplementedError()


# Monoid-only
sum = Aggregator("sum", monoid=_monoid.plus)
prod = Aggregator("prod", monoid=_monoid.times)
all = Aggregator("all", monoid=_monoid.land)
any = Aggregator("any", monoid=_monoid.lor)
min = Aggregator("min", monoid=_monoid.min)
max = Aggregator("max", monoid=_monoid.max)
any_value = Aggregator("any_value", monoid=_monoid.any)
bitwise_all = Aggregator("bitwise_all", monoid=_monoid.band)
bitwise_any = Aggregator("bitwise_any", monoid=_monoid.bor)
# Other monoids: bxnor bxor eq lxnor lxor

# Semiring-only
count = Aggregator("count", semiring=_semiring.plus_pair, semiring2=_semiring.plus_first)
count_nonzero = Aggregator(
    "count_nonzero", semiring=_semiring.plus_isne, semiring2=_semiring.plus_first
)
count_zero = Aggregator("count_zero", semiring=_semiring.plus_iseq, semiring2=_semiring.plus_first)
sum_of_squares = Aggregator(
    "sum_of_squares", initval=2, semiring=_semiring.plus_pow, semiring2=_semiring.plus_first
)
sum_of_inverses = Aggregator(
    "sum_of_inverses",
    initval=-1.0,
    semiring=_semiring.plus_pow,
    semiring2=_semiring.plus_first,
)

# Semiring and finalize
hypot = Aggregator(
    "hypot",
    initval=2,
    semiring=_semiring.plus_pow,
    semiring2=_semiring.plus_first,
    finalize=_unary.sqrt,
)
logaddexp = Aggregator(
    "logaddexp",
    initval=_np.e,
    semiring=_semiring.plus_pow,
    switch=True,
    semiring2=_semiring.plus_first,
    finalize=_unary.log,
)
logaddexp2 = Aggregator(
    "logaddexp2",
    initval=2,
    semiring=_semiring.plus_pow,
    switch=True,
    semiring2=_semiring.plus_first,
    finalize=_unary.log2,
)
# Alternatives
# hypot as monoid doesn't work if single negative element!
# hypot = Aggregator('hypot', monoid=_semiring.numpy.hypot)
# logaddexp = Aggregator('logaddexp', monoid=_semiring.numpy.logaddexp)
# logaddexp2 = Aggregator('logaddexp2', monoid=_semiring.numpy.logaddexp2)

L0norm = count_nonzero
L1norm = Aggregator("L1norm", semiring=_semiring.plus_absfirst, semiring2=_semiring.plus_first)
L2norm = hypot
Linfnorm = Aggregator("Linfnorm", semiring=_semiring.max_absfirst, semiring2=_semiring.max_first)


# Composite
def _mean_finalize(c, x):
    return _binary.truediv(x & c)


def _ptp_finalize(max, min):
    return _binary.minus(max & min)


def _varp_finalize(c, x, x2):
    # <x2> / n - (<x> / n)**2
    left = _binary.truediv(x2 & c).new()
    right = _binary.truediv(x & c).new()
    right << _binary.pow(right, 2)
    return _binary.minus(left & right)


def _vars_finalize(c, x, x2):
    # <x2> / (n-1) - <x>**2 / (n * (n-1))
    x << _binary.pow(x, 2)
    right = _binary.truediv(x & c).new()
    c << _binary.minus(c, 1)
    right << _binary.truediv(right & c)
    left = _binary.truediv(x2 & c).new()
    return _binary.minus(left & right)


def _stdp_finalize(c, x, x2):
    val = _varp_finalize(c, x, x2).new()
    return _unary.sqrt(val)


def _stds_finalize(c, x, x2):
    val = _vars_finalize(c, x, x2).new()
    return _unary.sqrt(val)


def _geometric_mean_finalize(c, x):
    right = _unary.minv["FP64"](c).new()
    return _binary.pow(x & right)


def _harmonic_mean_finalize(c, x):
    return _binary.truediv(c & x)


def _root_mean_square_finalize(c, x2):
    val = _binary.truediv(x2 & c).new()
    return _unary.sqrt(val)


mean = Aggregator(
    "mean",
    composite=[count, sum],
    finalize=_mean_finalize,
    types=[_binary.truediv],
)
peak_to_peak = Aggregator(
    "peak_to_peak",
    composite=[max, min],
    finalize=_ptp_finalize,
    types=[_monoid.min],
)
varp = Aggregator(
    "varp",
    composite=[count, sum, sum_of_squares],
    finalize=_varp_finalize,
    types=[_binary.truediv],
)
vars = Aggregator(
    "vars",
    composite=[count, sum, sum_of_squares],
    finalize=_vars_finalize,
    types=[_binary.truediv],
)
stdp = Aggregator(
    "stdp",
    composite=[count, sum, sum_of_squares],
    finalize=_stdp_finalize,
    types=[_binary.truediv, _unary.sqrt],
)
stds = Aggregator(
    "stds",
    composite=[count, sum, sum_of_squares],
    finalize=_stds_finalize,
    types=[_binary.truediv, _unary.sqrt],
)
geometric_mean = Aggregator(
    "geometric_mean",
    composite=[count, prod],
    finalize=_geometric_mean_finalize,
    types=[_binary.truediv],  # XXX
)
harmonic_mean = Aggregator(
    "harmonic_mean",
    composite=[count, sum_of_inverses],
    finalize=_harmonic_mean_finalize,
    types=[sum_of_inverses, _binary.truediv],
)
root_mean_square = Aggregator(
    "root_mean_square",
    composite=[count, sum_of_squares],
    finalize=_root_mean_square_finalize,
    types=[_binary.truediv, _unary.sqrt],
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

            # i, j = step1.to_values()
            # D = _Matrix.from_values(i, i, j, nrows=A._nrows, ncols=A._nrows)
            D = _diag(step1)

            masked = _semiring.any_eq(D @ A).new()
            masked(mask=masked.V, replace=True) << masked  # Could use select
            init = _Vector.new(bool, size=A._ncols)
            init[:] = False  # O(1) dense vector in SuiteSparse 5
            updater << row_semiring(masked @ init)
            if in_composite:
                1 / 0
                return updater.parent
        else:
            step1 = A.reduce_columns(monoid).new()

            # i, j = step1.to_values()
            # D = _Matrix.from_values(i, i, j, nrows=A._ncols, ncols=A._ncols)
            D = _diag(step1)

            masked = _semiring.any_eq(A @ D).new()
            masked(mask=masked.V, replace=True) << masked  # Could use select
            init = _Vector.new(bool, size=A._nrows)
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
        masked = _binary.eq(v, step1).new()
        masked(mask=masked.V, replace=True) << masked  # Could use select
        init = _Matrix.new(bool, nrows=v._size, ncols=1)
        init[:, :] = False  # O(1) dense column vector in SuiteSparse 5
        step2 = _Vector.new(updater.parent.dtype, size=1)
        step2 << col_semiring(masked @ init)
        if in_composite:
            1 / 0
            return step2
        expr = step2.reduce(_any)
        if step2._nvals == 0:
            expr = _Scalar.new(expr.dtype)
        updater << expr
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        A = expr.args[0]
        if A._nvals == 0:
            if in_composite:
                1 / 0
                return _Vector.new(updater.parent.dtype, size=1)
            updater << _Scalar.new(updater.parent.dtype)
            return
        step1 = A.reduce_scalar(monoid).new()

        masked = _binary.eq(A, step1).new()
        masked(mask=masked.V, replace=True) << masked  # Could use select
        init = _Vector.new(bool, size=A._nrows)
        init[:] = False  # O(1) dense vector in SuiteSparse 5

        # Always choose the one with smallest i
        step2 = _semiring.min_secondi(init @ masked).new()
        step3 = step2.reduce(_monoid.min)
        if agg.name in {"argmini", "argmaxi"}:
            # We're done!
            if in_composite:
                1 / 0
                rv = _Vector.new(step3.dtype, size=1)
                rv[0] = step3.value
                return rv
            updater << step3
            return
        i = step3.value
        step4 = _Vector.from_values([i], [False], size=A._nrows)
        step5 = col_semiring(step4 @ masked).new()
        step6 = step5.reduce(_monoid.min)
        if in_composite:
            1 / 0
            rv = _Vector.new(step6.dtype, size=1)
            rv[0] = step6.value
            return rv
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
                row_semiring=_semiring.min_firstj,
                col_semiring=_semiring.min_secondj,
            )
        return _argminmaxij(
            agg,
            updater,
            expr,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=_semiring.min_firsti,
            col_semiring=_semiring.min_secondi,
        )
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        return _argminmaxij(
            agg,
            updater,
            expr,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=_semiring.min_firsti,
            col_semiring=_semiring.min_secondi,
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
    custom=_partial(
        _argminmaxij,
        monoid=_monoid.min,
        row_semiring=_semiring.min_firsti,
        col_semiring=_semiring.min_secondi,
    ),
    types=[_semiring.min_firsti],
)
argmaxi = Aggregator(
    "argmaxi",
    custom=_partial(
        _argminmaxij,
        monoid=_monoid.max,
        row_semiring=_semiring.min_firsti,
        col_semiring=_semiring.min_secondi,
    ),
    types=[_semiring.min_firsti],
)
# argminj, argmaxj don't work on vectors
argminj = Aggregator(
    "argminj",
    custom=_partial(
        _argminmaxij,
        monoid=_monoid.min,
        row_semiring=_semiring.min_firstj,
        col_semiring=_semiring.min_secondj,
        allow_vector=False,
    ),
    types=[_semiring.min_firstj],
)
argmaxj = Aggregator(
    "argmaxj",
    custom=_partial(
        _argminmaxij,
        monoid=_monoid.max,
        row_semiring=_semiring.min_firstj,
        col_semiring=_semiring.min_secondj,
        allow_vector=False,
    ),
    types=[_semiring.min_firstj],
)
# These "do the right thing", but don't work with `reduce_scalar`
argmin = Aggregator(
    "argmin",
    custom=_partial(_argminmax, monoid=_monoid.min),
    types=[_semiring.min_firsti],
)
argmax = Aggregator(
    "argmax",
    custom=_partial(_argminmax, monoid=_monoid.max),
    types=[_semiring.min_firsti],
)


def _first_last(agg, updater, expr, *, in_composite, semiring):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_columns":
            A = A.T
        init = _Vector.new(bool, size=A._ncols)
        init[:] = False
        step1 = semiring(A @ init).new()
        Is, Js = step1.to_values()
        # TODO: perform these loops in e.g. Cython
        # Populate numpy array
        vals = _np.empty(Is.size, dtype=A.dtype.np_type)
        for index, (i, j) in enumerate(zip(Is, Js)):
            vals[index] = A[i, j].value
        # or Vector
        # v = _Vector.new(A.dtype, size=A._nrows)
        # for i, j in zip(Is, Js):
        #     v[i] = A[i, j].value
        result = _Vector.from_values(Is, vals, size=A._nrows)
        updater << result
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        init = _Matrix.new(bool, nrows=v._size, ncols=1)
        init[:, :] = False
        step1 = semiring(v @ init).new()
        index = step1[0].value
        if index is None:
            index = 0
        updater << v[index]
    else:  # GrB_Matrix_reduce
        A = expr.args[0]
        init1 = _Matrix.new(bool, nrows=A._ncols, ncols=1)
        init1[:, :] = False
        step1 = semiring(A @ init1).new()
        init2 = _Vector.new(bool, size=A._nrows)
        init2[:] = False
        step2 = semiring(step1.T @ init2).new()
        i = step2[0].value
        if i is None:
            i = j = 0
        else:
            j = step1[i, 0].value
        updater << A[i, j]


first = Aggregator(
    "first",
    custom=_partial(_first_last, semiring=_semiring.min_secondi),
    types=[_binary.first],
)
last = Aggregator(
    "last",
    custom=_partial(_first_last, semiring=_semiring.max_secondi),
    types=[_binary.second],
)


def _first_last_index(agg, updater, expr, *, in_composite, semiring):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_columns":
            A = A.T
        init = _Vector.new(bool, size=A._ncols)
        init[:] = False
        updater << semiring(A @ init)
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        init = _Matrix.new(bool, nrows=v._size, ncols=1)
        init[:, :] = False
        step1 = semiring(v @ init).new()
        updater << step1[0]
    else:  # GrB_Matrix_reduce
        # XXX: it would be best to raise upon creation of the expression
        raise ValueError(f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar")
        # To get the first/last j index:
        # A = expr.args[0]
        # init1 = _Matrix.new(bool, nrows=A._ncols, ncols=1)
        # init1[:, :] = False
        # step1 = semiring(A @ init1).new()
        # init2 = _Vector.new(bool, size=A._nrows)
        # init2[:] = False
        # step2 = semiring(step1.T @ init2).new()
        # i = step2[0].value
        # if i is None:
        #     i = 0
        # updater << step1[i, 0]


first_index = Aggregator(
    "first_index",
    custom=_partial(_first_last_index, semiring=_semiring.min_secondi),
    types=[_semiring.min_secondi],
)
last_index = Aggregator(
    "last_index",
    custom=_partial(_first_last_index, semiring=_semiring.max_secondi),
    types=[_semiring.min_secondi],
)

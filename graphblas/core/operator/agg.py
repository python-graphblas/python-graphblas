from functools import partial
from operator import getitem

import numpy as np

from ... import agg, backend, binary, monoid, semiring, unary
from ...dtypes import INT64, lookup_dtype
from .. import _supports_udfs
from ..utils import output_type


def _get_types(ops, initdtype):
    """Determine the input and output types of an aggregator based on a list of ops."""
    if initdtype is None:
        prev = dict(ops[0].types)
    else:
        op = ops[0]
        prev = {key: get_typed_op(op, key, initdtype).return_type for key in op.types}
    for op in ops[1:]:
        cur = {}
        types = op.types
        for in_type, out_type in prev.items():
            if out_type not in types:  # pragma: no cover (safety)
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
        applybegin=None,
        finalize=None,
        composite=None,
        custom=None,
        types=None,
        any_dtype=None,
    ):
        self.name = name
        self._initval_orig = initval
        self._initval = False if initval is None else initval
        self._initdtype = lookup_dtype(type(self._initval), self._initval)
        self._monoid = monoid
        self._semiring = semiring
        self._semiring2 = semiring2
        self._switch = switch
        self._applybegin = applybegin
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
            else:  # pragma: no cover (sanity)
                raise TypeError("types must be provided for composite and custom aggregators")
        self._types_orig = types
        self._types = None
        self._typed_ops = {}
        self._any_dtype = any_dtype

    @property
    def types(self):
        if self._types is None:
            if type(self._semiring) is str:
                self._semiring = semiring.from_string(self._semiring)
                if type(self._types_orig[0]) is str:  # pragma: no branch
                    self._types_orig[0] = semiring.from_string(self._types_orig[0])
            self._types = _get_types(
                self._types_orig, None if self._initval_orig is None else self._initdtype
            )
        return self._types

    def __getitem__(self, dtype):
        dtype = lookup_dtype(dtype)
        if not self._any_dtype and dtype not in self.types:
            raise KeyError(f"{self.name} does not work with {dtype}")
        if dtype not in self._typed_ops:
            self._typed_ops[dtype] = TypedAggregator(self, dtype)
        return self._typed_ops[dtype]

    def __contains__(self, dtype):
        dtype = lookup_dtype(dtype)
        return self._any_dtype or dtype in self.types

    def __repr__(self):
        if self.name in agg._deprecated:
            return f"agg.ss.{self.name}"
        return f"agg.{self.name}"

    def __reduce__(self):
        if self.name in agg._deprecated:
            return f"agg.ss.{self.name}"
        return f"agg.{self.name}"

    def __call__(self, val, *, rowwise=False, columnwise=False):
        # Should we expose `allow_empty=` keyword when reducing to Scalar?
        from ..matrix import Matrix, TransposedMatrix
        from ..vector import Vector

        typ = output_type(val)
        if typ is Vector:
            if rowwise or columnwise:
                raise ValueError(
                    "rowwise and columnwise arguments should not be used with Vector input"
                )
            return val.reduce(self)
        if typ in {Matrix, TransposedMatrix}:
            if rowwise:
                if columnwise:
                    raise ValueError("rowwise and columnwise arguments cannot both be True")
                return val.reduce_rowwise(self)
            if columnwise:
                return val.reduce_columnwise(self)
            return val.reduce_scalar(self)
        raise TypeError(
            f"Bad type when calling {self!r}.\n"
            "    - Expected type: Vector, Matrix, TransposedMatrix.\n"
            f"    - Got: {type(val)}.\n"
            "Calling an Aggregator is syntactic sugar for calling reduce methods.  "
            f"For example, `A.reduce_scalar({self!r})` is the same as `{self!r}(A)`."
        )


class TypedAggregator:
    opclass = "Aggregator"

    def __init__(self, agg, dtype):
        self.name = agg.name
        self.parent = agg
        self.type = dtype
        if dtype in agg.types:
            self.return_type = agg.types[dtype]
        elif agg._any_dtype is True:
            self.return_type = dtype
        else:
            self.return_type = agg._any_dtype

    def __repr__(self):
        return f"agg.{self.name}[{self.type}]"

    def _new(self, updater, expr, *, in_composite=False):
        agg = self.parent
        opts = updater.opts
        if agg._monoid is not None:
            x = expr.args[0]
            if agg._applybegin is not None:  # pragma: no cover (unused)
                x = agg._applybegin(x).new(**opts)
            method = getattr(x, expr.method_name)
            if expr.output_type.__name__ == "Scalar":
                expr = method(agg._monoid[self.type], allow_empty=not expr._is_cscalar)
            else:
                expr = method(agg._monoid[self.type])
            updater << expr
            if in_composite:
                parent = updater.parent
                if not parent._is_scalar:
                    return parent
                return parent._as_vector()
            return

        if agg._composite is not None:
            # Masks are applied throughout the aggregation, including composite aggregations.
            # Aggregations done while `in_composite is True` should return the updater parent
            # if the result is not a Scalar.  If the result is a Scalar, then there can be no
            # output mask, and a Vector of size 1 should be returned instead.
            results = []
            mask = updater.kwargs.get("mask")
            for cur_agg in agg._composite:
                cur_agg = cur_agg[self.type]  # Hopefully works well enough
                arg = expr.construct_output(cur_agg.return_type)
                results.append(cur_agg._new(arg(mask=mask, **opts), expr, in_composite=True))
            final_expr = agg._finalize(*results, opts)
            if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
                updater << final_expr
            elif expr.cfunc_name.startswith("GrB_Vector_reduce") or expr.cfunc_name.startswith(
                "GrB_Matrix_reduce"
            ):
                final = final_expr.new(**opts)
                updater << final[0]
            else:
                raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")
            if in_composite:
                parent = updater.parent
                if not parent._is_scalar:
                    return parent
                return parent._as_vector()
            return

        if agg._custom is not None:
            return agg._custom(self, updater, expr, opts, in_composite=in_composite)

        semiring = get_typed_op(agg._semiring, self.type, agg._initdtype)
        if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
            # Matrix -> Vector
            A = expr.args[0]
            if agg._applybegin is not None:
                A = agg._applybegin(A).new(**opts)
            orig_updater = updater
            if agg._finalize is not None:
                step1 = expr.construct_output(semiring.return_type)
                updater = step1(mask=updater.kwargs.get("mask"), **opts)
            if expr.method_name == "reduce_columnwise":
                A = A.T
            size = A._ncols
            init = expr._new_vector(agg._initdtype, size=size)
            init(**opts)[...] = agg._initval  # O(1) dense vector in SuiteSparse 5
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
            if agg._applybegin is not None:
                v = agg._applybegin(v).new(**opts)
            step1 = expr._new_vector(semiring.return_type, size=1)
            init = expr._new_matrix(agg._initdtype, nrows=v._size, ncols=1)
            init(**opts)[...] = agg._initval  # O(1) dense column vector in SuiteSparse 5
            if agg._switch:
                step1(**opts) << semiring(init.T @ v)
            else:
                step1(**opts) << semiring(v @ init)
            if agg._finalize is not None:
                finalize = agg._finalize[semiring.return_type]
                if step1.dtype == finalize.return_type:
                    step1(**opts) << finalize(step1)
                else:
                    step1 = finalize(step1).new(finalize.return_type, **opts)
            if in_composite:
                return step1
            updater << step1[0]
        elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
            # Matrix -> Scalar
            A = expr.args[0]
            if agg._applybegin is not None:
                A = agg._applybegin(A).new(**opts)
            # We need to compute in two steps: Matrix -> Vector -> Scalar.
            # This has not been benchmarked or optimized.
            # We may be able to intelligently choose the faster path.
            init1 = expr._new_vector(agg._initdtype, size=A._ncols)
            init1(**opts)[...] = agg._initval  # O(1) dense vector in SuiteSparse 5
            step1 = expr._new_vector(semiring.return_type, size=A._nrows)
            if agg._switch:
                step1(**opts) << semiring(init1 @ A.T)
            else:
                step1(**opts) << semiring(A @ init1)
            init2 = expr._new_matrix(agg._initdtype, nrows=A._nrows, ncols=1)
            init2(**opts)[...] = agg._initval  # O(1) dense vector in SuiteSparse 5
            semiring2 = agg._semiring2[semiring.return_type]
            step2 = expr._new_vector(semiring2.return_type, size=1)
            step2(**opts) << semiring2(step1 @ init2)
            if agg._finalize is not None:
                finalize = agg._finalize[semiring2.return_type]
                if step2.dtype == finalize.return_type:
                    step2 << finalize(step2)
                else:
                    step2 = finalize(step2).new(finalize.return_type, **opts)
            if in_composite:
                return step2
            updater << step2[0]
        else:
            raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")

    def __reduce__(self):
        return (getitem, (self.parent, self.type))

    __call__ = Aggregator.__call__


# Monoid-only
agg.sum = Aggregator("sum", monoid=monoid.plus)
agg.prod = Aggregator("prod", monoid=monoid.times)
agg.all = Aggregator("all", monoid=monoid.land)
agg.any = Aggregator("any", monoid=monoid.lor)
agg.min = Aggregator("min", monoid=monoid.min)
agg.max = Aggregator("max", monoid=monoid.max)
agg.any_value = Aggregator("any_value", monoid=monoid.any, any_dtype=True)
agg.bitwise_all = Aggregator("bitwise_all", monoid=monoid.band)
agg.bitwise_any = Aggregator("bitwise_any", monoid=monoid.bor)
# Other monoids: bxnor bxor eq lxnor lxor

# Semiring-only
agg.count = Aggregator(
    "count", semiring=semiring.plus_pair, semiring2=semiring.plus_first, any_dtype=INT64
)
agg.count_nonzero = Aggregator(
    "count_nonzero", semiring=semiring.plus_isne, semiring2=semiring.plus_first
)
agg.count_zero = Aggregator(
    "count_zero", semiring=semiring.plus_iseq, semiring2=semiring.plus_first
)
agg.sum_of_squares = Aggregator(
    "sum_of_squares", initval=2, semiring=semiring.plus_pow, semiring2=semiring.plus_first
)
agg.sum_of_inverses = Aggregator(
    "sum_of_inverses",
    initval=-1.0,
    semiring=semiring.plus_pow,
    semiring2=semiring.plus_first,
)
agg.exists = Aggregator(
    "exists", semiring=semiring.any_pair, semiring2=semiring.any_pair, any_dtype=INT64
)

# Semiring and finalize
agg.hypot = Aggregator(
    "hypot",
    initval=2,
    semiring=semiring.plus_pow,
    semiring2=semiring.plus_first,
    finalize=unary.sqrt,
)
agg.logaddexp = Aggregator(
    "logaddexp",
    initval=np.e,
    semiring=semiring.plus_pow,
    switch=True,
    semiring2=semiring.plus_first,
    finalize=unary.log,
)
agg.logaddexp2 = Aggregator(
    "logaddexp2",
    initval=2,
    semiring=semiring.plus_pow,
    switch=True,
    semiring2=semiring.plus_first,
    finalize=unary.log2,
)
# Alternatives
# logaddexp = Aggregator('logaddexp', monoid=semiring.numpy.logaddexp)
# logaddexp2 = Aggregator('logaddexp2', monoid=semiring.numpy.logaddexp2)
# hypot as monoid doesn't work if single negative element!
# hypot = Aggregator('hypot', monoid=semiring.numpy.hypot)
# hypot = Aggregator('hypot', applybegin=unary.abs, monoid=semiring.numpy.hypot)

agg.L0norm = agg.count_nonzero
agg.L2norm = agg.hypot
if _supports_udfs:
    agg.L1norm = Aggregator("L1norm", semiring="plus_absfirst", semiring2=semiring.plus_first)
    agg.Linfnorm = Aggregator("Linfnorm", semiring="max_absfirst", semiring2=semiring.max_first)
else:
    # Are these always better?
    agg.L1norm = Aggregator(
        "L1norm", applybegin=unary.abs, semiring=semiring.plus_first, semiring2=semiring.plus_first
    )
    agg.Linfnorm = Aggregator(
        "Linfnorm", applybegin=unary.abs, semiring=semiring.max_first, semiring2=semiring.max_first
    )


# Composite
def _mean_finalize(c, x, opts):
    return binary.truediv(x & c)


def _ptp_finalize(max, min, opts):
    return binary.minus(max & min)


def _varp_finalize(c, x, x2, opts):
    # <x2> / n - (<x> / n)**2
    left = binary.truediv(x2 & c).new(**opts)
    right = binary.truediv(x & c).new(**opts)
    right(**opts) << binary.pow(right, 2)
    return binary.minus(left & right)


def _vars_finalize(c, x, x2, opts):
    # <x2> / (n-1) - <x>**2 / (n * (n-1))
    x(**opts) << binary.pow(x, 2)
    right = binary.truediv(x & c).new(**opts)
    c(**opts) << binary.minus(c, 1)
    right(**opts) << binary.truediv(right & c)
    left = binary.truediv(x2 & c).new(**opts)
    return binary.minus(left & right)


def _stdp_finalize(c, x, x2, opts):
    val = _varp_finalize(c, x, x2, opts).new(**opts)
    return unary.sqrt(val)


def _stds_finalize(c, x, x2, opts):
    val = _vars_finalize(c, x, x2, opts).new(**opts)
    return unary.sqrt(val)


def _geometric_mean_finalize(c, x, opts):
    right = unary.minv["FP64"](c).new(**opts)
    return binary.pow(x & right)


def _harmonic_mean_finalize(c, x, opts):
    return binary.truediv(c & x)


def _root_mean_square_finalize(c, x2, opts):
    val = binary.truediv(x2 & c).new(**opts)
    return unary.sqrt(val)


agg.mean = Aggregator(
    "mean",
    composite=[agg.count, agg.sum],
    finalize=_mean_finalize,
    types=[binary.truediv],
)
agg.peak_to_peak = Aggregator(
    "peak_to_peak",
    composite=[agg.max, agg.min],
    finalize=_ptp_finalize,
    types=[monoid.min],
)
agg.varp = Aggregator(
    "varp",
    composite=[agg.count, agg.sum, agg.sum_of_squares],
    finalize=_varp_finalize,
    types=[binary.truediv],
)
agg.vars = Aggregator(
    "vars",
    composite=[agg.count, agg.sum, agg.sum_of_squares],
    finalize=_vars_finalize,
    types=[binary.truediv],
)
agg.stdp = Aggregator(
    "stdp",
    composite=[agg.count, agg.sum, agg.sum_of_squares],
    finalize=_stdp_finalize,
    types=[binary.truediv, unary.sqrt],
)
agg.stds = Aggregator(
    "stds",
    composite=[agg.count, agg.sum, agg.sum_of_squares],
    finalize=_stds_finalize,
    types=[binary.truediv, unary.sqrt],
)
agg.geometric_mean = Aggregator(
    "geometric_mean",
    composite=[agg.count, agg.prod],
    finalize=_geometric_mean_finalize,
    types=[binary.truediv],
)
agg.harmonic_mean = Aggregator(
    "harmonic_mean",
    composite=[agg.count, agg.sum_of_inverses],
    finalize=_harmonic_mean_finalize,
    types=[agg.sum_of_inverses, binary.truediv],
)
agg.root_mean_square = Aggregator(
    "root_mean_square",
    composite=[agg.count, agg.sum_of_squares],
    finalize=_root_mean_square_finalize,
    types=[binary.truediv, unary.sqrt],
)


# Special recipes
def _argminmaxij(
    agg,
    updater,
    expr,
    opts,
    *,
    in_composite,
    monoid,
    col_semiring,
    row_semiring,
):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_rowwise":
            step1 = A.reduce_rowwise(monoid).new(**opts)

            D = step1.diag()

            masked = semiring.any_eq(D @ A).new(**opts)
            masked(mask=masked.V, replace=True, **opts) << masked  # Could use select
            init = expr._new_vector(bool, size=A._ncols)
            init(**opts)[...] = False  # O(1) dense vector in SuiteSparse 5
            updater << row_semiring(masked @ init)
            if in_composite:
                return updater.parent
        else:
            step1 = A.reduce_columnwise(monoid).new(**opts)

            D = step1.diag()

            masked = semiring.any_eq(A @ D).new(**opts)
            masked(mask=masked.V, replace=True, **opts) << masked  # Could use select
            init = expr._new_vector(bool, size=A._nrows)
            init(**opts)[...] = False  # O(1) dense vector in SuiteSparse 5
            updater << col_semiring(init @ masked)
            if in_composite:
                return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        step1 = v.reduce(monoid, allow_empty=False).new(**opts)
        masked = binary.eq(v, step1).new(**opts)
        masked(mask=masked.V, replace=True, **opts) << masked  # Could use select
        init = expr._new_matrix(bool, nrows=v._size, ncols=1)
        init(**opts)[...] = False  # O(1) dense column vector in SuiteSparse 5
        step2 = col_semiring(masked @ init).new(**opts)
        if in_composite:
            return step2
        updater << step2[0]
    else:
        raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")


def _argminmax(agg, updater, expr, opts, *, in_composite, monoid):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        if expr.method_name == "reduce_rowwise":
            return _argminmaxij(
                agg,
                updater,
                expr,
                opts,
                in_composite=in_composite,
                monoid=monoid,
                row_semiring=semiring._deprecated["min_firstj"],
                col_semiring=semiring._deprecated["min_secondj"],
            )
        return _argminmaxij(
            agg,
            updater,
            expr,
            opts,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=semiring._deprecated["min_firsti"],
            col_semiring=semiring._deprecated["min_secondi"],
        )
    if expr.cfunc_name.startswith("GrB_Vector_reduce"):
        return _argminmaxij(
            agg,
            updater,
            expr,
            opts,
            in_composite=in_composite,
            monoid=monoid,
            row_semiring=semiring._deprecated["min_firsti"],
            col_semiring=semiring._deprecated["min_secondi"],
        )
    if expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        raise ValueError(f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar.")
    raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")


# These "do the right thing", but don't work with `reduce_scalar`
_argmin = Aggregator(
    "argmin",
    custom=partial(_argminmax, monoid=monoid.min),
    types=[semiring._deprecated["min_firsti"]],
)
_argmax = Aggregator(
    "argmax",
    custom=partial(_argminmax, monoid=monoid.max),
    types=[semiring._deprecated["min_firsti"]],
)


def _first_last(agg, updater, expr, opts, *, in_composite, semiring_):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_columnwise":
            A = A.T
        init = expr._new_vector(bool, size=A._ncols)
        init(**opts)[...] = False  # O(1) dense vector in SuiteSparse 5
        step1 = semiring_(A @ init).new(**opts)
        Is, Js = step1.to_coo()

        Matrix_ = type(expr._new_matrix(bool))
        P = Matrix_.from_coo(Js, Is, 1, nrows=A._ncols, ncols=A._nrows)
        mask = step1.diag()
        result = semiring.any_first(A @ P).new(mask=mask.S, **opts).diag(**opts)

        updater << result
        if in_composite:
            return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        init = expr._new_matrix(bool, nrows=v._size, ncols=1)
        init(**opts)[...] = False  # O(1) dense matrix in SuiteSparse 5
        step1 = semiring_(v @ init).new(**opts)
        index = step1[0].new().value
        # `==` instead of `is` automatically triggers index.compute() in dask-graphblas:
        if index == None:  # noqa: E711
            index = 0
        if in_composite:
            return v[[index]].new(**opts)
        updater << v[index]
    else:  # GrB_Matrix_reduce
        A = expr.args[0]
        init1 = expr._new_matrix(bool, nrows=A._ncols, ncols=1)
        init1(**opts)[...] = False  # O(1) dense matrix in SuiteSparse 5
        step1 = semiring_(A @ init1).new(**opts)
        init2 = expr._new_vector(bool, size=A._nrows)
        init2(**opts)[...] = False  # O(1) dense vector in SuiteSparse 5
        step2 = semiring_(step1.T @ init2).new(**opts)
        i = step2[0].new().value
        # `==` instead of `is` automatically triggers i.compute() in dask-graphblas:
        if i == None:  # noqa: E711
            i = j = 0
        else:
            j = step1[i, 0].new().value
        if in_composite:
            return A[i, [j]].new(**opts)
        updater << A[i, j]


_first = Aggregator(
    "first",
    custom=partial(_first_last, semiring_=semiring._deprecated["min_secondi"]),
    types=[binary.first],
    any_dtype=True,
)
_last = Aggregator(
    "last",
    custom=partial(_first_last, semiring_=semiring._deprecated["max_secondi"]),
    types=[binary.second],
    any_dtype=True,
)


def _first_last_index(agg, updater, expr, opts, *, in_composite, semiring):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        if expr.method_name == "reduce_columnwise":
            A = A.T
        init = expr._new_vector(bool, size=A._ncols)
        init(**opts)[...] = False  # O(1) dense vector in SuiteSparse 5
        expr = semiring(A @ init)
        updater << expr
        if in_composite:
            return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        init = expr._new_matrix(bool, nrows=v._size, ncols=1)
        init(**opts)[...] = False  # O(1) dense matrix in SuiteSparse 5
        step1 = semiring(v @ init).new(**opts)
        if in_composite:
            return step1
        updater << step1[0]
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        raise ValueError(f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar.")
    else:
        raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")


_first_index = Aggregator(
    "first_index",
    custom=partial(_first_last_index, semiring=semiring._deprecated["min_secondi"]),
    types=[semiring._deprecated["min_secondi"]],
    any_dtype=INT64,
)
_last_index = Aggregator(
    "last_index",
    custom=partial(_first_last_index, semiring=semiring._deprecated["max_secondi"]),
    types=[semiring._deprecated["min_secondi"]],
    any_dtype=INT64,
)
agg._deprecated = {
    "argmin": _argmin,
    "argmax": _argmax,
    "first": _first,
    "last": _last,
    "first_index": _first_index,
    "last_index": _last_index,
}
if backend == "suitesparse":
    agg.ss.argmin = _argmin
    agg.ss.argmax = _argmax
    agg.ss.first = _first
    agg.ss.last = _last
    agg.ss.first_index = _first_index
    agg.ss.last_index = _last_index

agg.Aggregator = Aggregator
agg.TypedAggregator = TypedAggregator

from .utils import get_typed_op  # noqa: E402 isort:skip

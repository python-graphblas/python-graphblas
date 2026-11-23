from functools import partial
from operator import getitem

import numpy as np

from ... import agg, backend, binary, indexunary, monoid, semiring, unary
from ...dtypes import BOOL, INT64, lookup_dtype
from ...exceptions import UdfParseError
from .. import _supports_udfs
from ..utils import output_type


def _get_types(ops, initdtype):
    """Determine the input and output types of an aggregator based on a list of ops.

    The first item may instead be a dict of input types to output types.
    """
    if isinstance(ops[0], dict):
        prev = dict(ops[0])
    elif initdtype is None:
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


def _from_nvals_vector(expr, agg, x, dtype):
    """Vector of size 1 with ``agg._from_nvals(x.nvals)``, or empty when ``x`` is empty."""
    rv = expr._new_vector(dtype, size=1)
    if nvals := x._nvals:
        rv[0] = agg._from_nvals(nvals)
    return rv


def _store_element(updater, element, opts):
    """``updater << element``, for a Scalar ``updater`` and an element such as ``v[0]``."""
    if updater.kwargs.get("accum") is not None:
        # With autocompute off, `s(accum=op) << v[0]` raises TypeError, so extract first
        updater << element.new(**opts)
    else:
        # Store the element itself so GraphBLAS casts it to the output type: 763 into an
        # INT8 C scalar wraps to -5, where storing `element.new()` raises OverflowError.
        updater << element


def _by_kind(agg, updater, expr, opts, *, in_composite, aggs):
    """Reduce with the Aggregator in ``aggs`` for the kind of ``agg.type``.

    ``aggs`` maps an ``np.dtype.kind`` character, or ``"other"`` for the rest, to an Aggregator.
    """
    sub = aggs.get(agg.type.np_type.kind, aggs["other"])
    return sub[agg.type]._new(updater, expr, in_composite=in_composite)


class Aggregator:
    """A reduction operator that collapses the values of a Matrix or Vector.

    An Aggregator is used with the ``reduce`` family of methods: ``Vector.reduce``,
    ``Matrix.reduce_rowwise``, ``Matrix.reduce_columnwise``, and ``Matrix.reduce_scalar``.
    Some aggregators return a summary value (``sum``, ``mean``, ``max``); others return a
    position (``argmin``, ``argmax``, ``first_index``).

    Built-in aggregators live in the ``graphblas.agg`` namespace, such as ``agg.sum``,
    ``agg.mean``, and ``agg.count``. Unlike the other operators, an Aggregator is not a
    single GraphBLAS object. Most are built from a monoid or semiring, with an optional
    finalize step, for each dtype on first use.

    Examples
    --------
    >>> import graphblas as gb
    >>> v = gb.Vector.from_coo([0, 1, 2], [1, 2, 3])
    >>> int(v.reduce(gb.agg.sum).new())
    6
    """

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
        from_nvals=None,
        vector_apply=False,
        stable=None,
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
        # `from_nvals(n)` is the result on a Vector or a whole Matrix with n > 0 entries, for
        # a semiring aggregator such as `agg.count` whose result depends on nothing else
        self._from_nvals = from_nvals
        # If True, a semiring aggregator reduces a Vector by applying its multiplier and
        # then reducing with its monoid, instead of with one product (see
        # ``TypedAggregator._new``).  Leave it False when the multiplier can return NaN
        # under a `max` monoid, or -0.0 under `plus`: a reduce of bitmap data to a Scalar
        # loses both (see the Matrix -> Scalar step there).
        self._vector_apply = vector_apply
        # A custom function used instead of the semiring for real values, where the semiring
        # overflows (``agg.logaddexp``); a Tensor reduce still uses the semiring
        self._stable = stable
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
            if isinstance(self._semiring, str):
                self._semiring = semiring.from_string(self._semiring)
                if isinstance(self._types_orig[0], str):  # pragma: no branch
                    self._types_orig[0] = semiring.from_string(self._types_orig[0])
            self._types = _get_types(
                self._types_orig, None if self._initval_orig is None else self._initdtype
            )
        return self._types

    def __getitem__(self, dtype):
        dtype = lookup_dtype(dtype)
        if not self._any_dtype and dtype not in self.types:
            # An aggregator made of a monoid alone works on a UDT when the monoid
            # lifts to it (``agg.sum``, ``agg.min``). The others (``agg.hypot``,
            # ``agg.mean``) do not lift.
            if (
                dtype._is_udt
                and self._monoid is not None
                and self._semiring is None
                and self._composite is None
                and self._custom is None
                and self._finalize is None
                and self._applybegin is None
            ):
                # A monoid that does not lift to this UDT raises one of these;
                # each becomes the KeyError below
                try:
                    typed_monoid = self._monoid[dtype]
                except (KeyError, TypeError, UdfParseError):
                    pass
                else:
                    self.types[dtype] = typed_monoid.return_type
            if dtype not in self.types:
                raise KeyError(f"{self.name} does not work with {dtype}")
        if dtype not in self._typed_ops:
            self._typed_ops[dtype] = TypedAggregator(self, dtype)
        return self._typed_ops[dtype]

    def __contains__(self, dtype):
        # Through ``__getitem__``, which lifts the monoid to a UDT when it can
        try:
            self[dtype]
        except (KeyError, TypeError, UdfParseError):
            return False
        return True

    def __repr__(self):
        return f"agg.{self.name}"

    def __reduce__(self):
        # Aggregator has no register_anonymous/register_new, so every
        # instance is a module-level singleton; pickle by name.
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
        return f"{self.parent!r}[{self.type}]"

    @property
    def jit_c_source(self):
        """The C definition of this aggregator's monoid on its type, or ``None``.

        Aggregators not made of a monoid alone (``agg.hypot``, ``agg.mean``,
        ``agg.first``) have none.
        """
        mon = getattr(self.parent, "_monoid", None)
        if mon is None or self.type not in mon:
            return None
        return mon[self.type].jit_c_source

    @property
    def jit_c_name(self):
        """The C name of this aggregator's monoid on its type, or ``None``."""
        mon = getattr(self.parent, "_monoid", None)
        if mon is None or self.type not in mon:
            return None
        return mon[self.type].jit_c_name

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
                _store_element(updater, final[0], opts)
            else:
                raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")
            if in_composite:
                parent = updater.parent
                if not parent._is_scalar:
                    return parent
                return parent._as_vector()
            return

        if agg._stable is not None and self.type.np_type.kind in "iuf":
            return agg._stable(self, updater, expr, opts, in_composite=in_composite)

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
            if agg._from_nvals is not None or not v._nvals:
                # With no entries the result is empty, so skip the product: in SuiteSparse, a
                # product over an inner dimension of 0 with iso operands can return an entry
                step1 = _from_nvals_vector(expr, agg, v, semiring.return_type)
            elif agg._vector_apply:
                # SuiteSparse computes `v @ init` below as one dot product, on one thread.  For
                # an expensive multiplier, such as `pow`, an apply and a reduce on every
                # thread are faster: 8 ms instead of 83 ms for `sum_of_squares` on 1e7
                # entries.  For a cheap one, such as `isne`, the one pass of `v @ init` is as
                # fast.
                binop = semiring.binaryop
                x = binop(agg._initval, v) if agg._switch else binop(v, agg._initval)
                step1 = x.new(**opts).reduce(semiring.monoid).new(**opts)._as_vector()
            else:
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
            _store_element(updater, step1[0], opts)
        elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
            # Matrix -> Scalar
            A = expr.args[0]
            if agg._applybegin is not None:
                A = agg._applybegin(A).new(**opts)
            semiring2 = agg._semiring2[semiring.return_type]
            if agg._from_nvals is not None or not A._nvals:
                # With no entries, skip the product, as for a Vector above
                step2 = _from_nvals_vector(expr, agg, A, semiring2.return_type)
            else:
                # In two steps: Matrix -> Vector -> Scalar
                init1 = expr._new_vector(agg._initdtype, size=A._ncols)
                init1(**opts)[...] = agg._initval  # O(1) dense vector in SuiteSparse 5
                step1 = expr._new_vector(semiring.return_type, size=A._nrows)
                if agg._switch:
                    step1(**opts) << semiring(init1 @ A.T)
                else:
                    step1(**opts) << semiring(A @ init1)
                # A dot product, not `step1.reduce(semiring2.monoid)`: SuiteSparse reduces
                # bitmap data to a Scalar from the monoid's identity, so the max of only
                # NaN would be -inf and the sum of only -0.0 would be +0.0.  (When `step1`
                # is sparse, the dot product also returns +0.0 for that sum.)
                init2 = expr._new_matrix(agg._initdtype, nrows=A._nrows, ncols=1)
                init2(**opts)[...] = agg._initval  # O(1) dense vector in SuiteSparse 5
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
            _store_element(updater, step2[0], opts)
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
    "count",
    semiring=semiring.plus_pair,
    semiring2=semiring.plus_first,
    any_dtype=INT64,
    from_nvals=lambda n: n,
)
agg.count_nonzero = Aggregator(
    "count_nonzero", semiring=semiring.plus_isne, semiring2=semiring.plus_first
)
agg.count_zero = Aggregator(
    "count_zero", semiring=semiring.plus_iseq, semiring2=semiring.plus_first
)
agg.sum_of_squares = Aggregator(
    "sum_of_squares",
    initval=2,
    semiring=semiring.plus_pow,
    semiring2=semiring.plus_first,
    vector_apply=True,
)
# On real types, `rdiv(x, 1.0)` is `1.0 / x`: correctly rounded, and 8 times faster rowwise
# than `pow(x, -1.0)`.  Complex types use `pow`, which returns inf+0j at 0 and nan+0j at
# nan+0j, where complex division returns inf+nanj and nan+nanj.
_sum_of_inverses_by_kind = {
    "c": Aggregator(
        "sum_of_inverses",
        initval=-1.0,
        semiring=semiring.plus_pow,
        semiring2=semiring.plus_first,
        vector_apply=True,
    ),
    "other": Aggregator(
        "sum_of_inverses",
        initval=1.0,
        semiring=semiring.plus_rdiv,
        semiring2=semiring.plus_first,
    ),
}
agg.sum_of_inverses = Aggregator(
    "sum_of_inverses",
    custom=partial(_by_kind, aggs=_sum_of_inverses_by_kind),
    types=[_sum_of_inverses_by_kind["other"]],
)
agg.exists = Aggregator(
    "exists",
    semiring=semiring.any_pair,
    semiring2=semiring.any_pair,
    any_dtype=INT64,
    from_nvals=lambda n: 1,
)


def _logaddexp(agg, updater, expr, opts, *, in_composite, fast, exp, log, np_log, low):
    """Reduce with ``fast``, the sum of ``exp(x)`` then its log, and where a result is not
    finite or is below ``low``, reduce again as ``m + log(sum(exp(x - m)))``, with ``m`` the
    largest value (0 when that is not finite), as numpy's ``logaddexp`` reduces. The sum
    overflows to infinity past 709 (1023 for ``exp2``), and below -708 (-1022) it loses
    precision until it rounds to 0.
    """
    dtype = agg.return_type
    result = fast[agg.type]._new(expr.construct_output(dtype)(**opts), expr, in_composite=True)
    # A NaN result comes only from a NaN value, as numpy's would, so it is not checked
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        # Two reduces, not a copy of a value per row (max and min skip NaN)
        top = result.reduce(monoid.max).new().value
        redo = top is not None and (top == np.inf or result.reduce(monoid.min).new().value < low)
    else:
        values = result.to_coo(indices=False)[1]  # (one value at most)
        redo = values.size > 0 and (values[0] == np.inf or values[0] < low)
    if redo:
        result = _logaddexp_shifted(expr, dtype, opts, exp, log, np_log)
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        updater << result
        return updater.parent if in_composite else None
    if in_composite:
        return result
    _store_element(updater, result[0], opts)
    return None


def _logaddexp_shifted(expr, dtype, opts, exp, log, np_log):
    """``m + log(sum(exp(x - m)))`` of each group of values, as a Vector (of size 1 for a
    Vector or a whole Matrix), with ``m`` the group's largest value (0 when not finite).
    """
    x = expr.args[0]
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        rowwise = expr.method_name == "reduce_rowwise"
        reduce = x.reduce_rowwise if rowwise else x.reduce_columnwise
        shift = reduce(monoid.max).new(dtype, **opts)
        finite = unary.isfinite(shift).new(**opts)
        shift(mask=~finite.V, **opts) << 0.0  # (an empty group stays empty below)
        D = _scaling_diag(expr, unary.ainv(shift).new(**opts), opts)
        y = semiring.any_plus(D @ x if rowwise else x @ D).new(dtype, **opts)
        y << exp(y)
        total = (y.reduce_rowwise if rowwise else y.reduce_columnwise)(monoid.plus)
        return binary.plus(log(total.new(**opts)).new(**opts) & shift).new(**opts)
    whole = x.reduce if expr.cfunc_name.startswith("GrB_Vector_reduce") else x.reduce_scalar
    m = whole(monoid.max).new(dtype).value
    m = m if np.isfinite(m) else 0.0
    y = x.apply(binary.minus, right=m).new(dtype, **opts)
    y << exp(y)
    total = (y.reduce if y.ndim == 1 else y.reduce_scalar)(monoid.plus).new().value
    result = expr._new_vector(dtype, size=1)
    with np.errstate(divide="ignore"):  # (the log of 0 is -inf, as numpy's is)
        result[0] = np_log(total) + m
    return result


# Semiring and finalize
agg.hypot = Aggregator(
    "hypot",
    initval=2,
    semiring=semiring.plus_pow,
    semiring2=semiring.plus_first,
    finalize=unary.sqrt,
    vector_apply=True,
)


def _logaddexp_aggregator(name, base, log, **stable):
    """``log(sum(base ** x))``: the semiring computes it, and ``_logaddexp`` checks it."""
    kwargs = {
        "initval": base,
        "semiring": semiring.plus_pow,
        "switch": True,
        "semiring2": semiring.plus_first,
        "finalize": log,
        "vector_apply": True,
    }
    fast = Aggregator(name, **kwargs)
    return Aggregator(name, **kwargs, stable=partial(_logaddexp, fast=fast, log=log, **stable))


agg.logaddexp = _logaddexp_aggregator(
    "logaddexp", np.e, unary.log, exp=unary.exp, np_log=np.log, low=-700.0
)
# 2.0, not 2, so that `pow` is FP64 on integers too: `pow(2, -1)` in INT64 is 0
agg.logaddexp2 = _logaddexp_aggregator(
    "logaddexp2", 2.0, unary.log2, exp=unary.exp2, np_log=np.log2, low=-1000.0
)
# Alternatives
# logaddexp = Aggregator('logaddexp', monoid=semiring.numpy.logaddexp)
# logaddexp2 = Aggregator('logaddexp2', monoid=semiring.numpy.logaddexp2)
# hypot as monoid doesn't work if single negative element!
# hypot = Aggregator('hypot', monoid=semiring.numpy.hypot)
# hypot = Aggregator('hypot', applybegin=unary.abs, monoid=semiring.numpy.hypot)

agg.L0norm = agg.count_nonzero
agg.L2norm = agg.hypot

# On floats, `copysign(x, True)` is `fabs(x)`.  A BOOL `True`, unlike 1.0, keeps the semiring
# typed on FP32 for FP32 input.  As a built-in op, the JIT compiles `copysign` into the
# reduction: 0.9 ms rowwise on 1e7 entries, against 2.9 ms with the `absfirst` UDF.  `abs`
# changes neither unsigned integers nor BOOL, so L1norm on unsigned integers is `sum`, and
# Linfnorm on both is `max`.  Signed integers and complex numbers use `absfirst`, or, without
# Numba, an apply of `abs` and then the reduction.
_L1norm_by_kind = {
    "f": Aggregator(
        "L1norm", initval=True, semiring="plus.copysign", semiring2=semiring.plus_first
    ),
    "u": agg.sum,
}
_Linfnorm_by_kind = {
    "f": Aggregator(
        "Linfnorm", initval=True, semiring="max.copysign", semiring2=semiring.max_first
    ),
    "u": agg.max,
    "b": agg.max,
}
if _supports_udfs:
    _L1norm_by_kind["other"] = Aggregator(
        "L1norm", semiring="plus_absfirst", semiring2=semiring.plus_first, vector_apply=True
    )
    _Linfnorm_by_kind["other"] = Aggregator(
        "Linfnorm", semiring="max_absfirst", semiring2=semiring.max_first
    )
    # The types of `absfirst`, read from `abs` so that listing them does not compile the UDF.
    # L1norm leaves out BOOL, which the plus monoid lacks.
    _L1norm_types = [{k: v for k, v in unary.abs.types.items() if k != BOOL}]
    _Linfnorm_types = [unary.abs]
else:
    _L1norm_by_kind["other"] = Aggregator(
        "L1norm",
        applybegin=unary.abs,
        semiring=semiring.plus_first,
        semiring2=semiring.plus_first,
        vector_apply=True,
    )
    _Linfnorm_by_kind["other"] = Aggregator(
        "Linfnorm",
        applybegin=unary.abs,
        semiring=semiring.max_first,
        semiring2=semiring.max_first,
    )
    _L1norm_types = [_L1norm_by_kind["other"]]
    _Linfnorm_types = [_Linfnorm_by_kind["other"]]
agg.L1norm = Aggregator(
    "L1norm", custom=partial(_by_kind, aggs=_L1norm_by_kind), types=_L1norm_types
)
agg.Linfnorm = Aggregator(
    "Linfnorm", custom=partial(_by_kind, aggs=_Linfnorm_by_kind), types=_Linfnorm_types
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


# Position recipes, for argmin, argmax, first, last, first_index and last_index.
# The positional ops (`min_secondi`, `rowindex` and the others) are typed on INT64 whatever
# the input type: typed on INT32, they would return INT32 positions, which wrap at 2**31.
def _end_index(expr, A, opts, *, last, rowwise):
    """Expression of the index of the first (or last) entry in each row or column of ``A``."""
    if backend == "suitesparse":
        # SuiteSparse reads these from its sorted indices, which for the rows of a row-major
        # `A` is much faster than the pass over every entry below.  `secondi` gives the
        # column of `A` for `A @ init` and the row of `A` for `init @ A`.
        sr = (semiring.ss.max_secondi if last else semiring.ss.min_secondi)[INT64]
        init = expr._new_vector(bool, size=A._ncols if rowwise else A._nrows)
        init(**opts)[...] = False  # O(1) dense vector in SuiteSparse
        return sr(A @ init) if rowwise else sr(init @ A)
    mon = monoid.max if last else monoid.min
    if rowwise:
        return A.apply(indexunary.colindex[INT64]).new(**opts).reduce_rowwise(mon)
    return A.apply(indexunary.rowindex[INT64]).new(**opts).reduce_columnwise(mon)


def _end_index_vector(expr, v, opts, *, last):
    """The index of the first (or last) entry of ``v``, as a new Vector of size 1."""
    if backend == "suitesparse":
        sr = (semiring.ss.max_secondi if last else semiring.ss.min_secondi)[INT64]
        init = expr._new_matrix(bool, nrows=v._size, ncols=1)
        init(**opts)[...] = False  # O(1) dense column vector in SuiteSparse
        return sr(v @ init).new(**opts)
    mon = monoid.max if last else monoid.min
    return v.apply(indexunary.rowindex[INT64]).new(**opts).reduce(mon).new(**opts)._as_vector()


def _scaling_diag(expr, x, opts):
    """Return ``x.diag()``, with the missing diagonal entries filled when that is cheap.

    ``x`` is the min (or max) of each row or column of the matrix ``A`` being scaled, so it
    lacks an entry only where that row or column of ``A`` is empty. A filled entry scales
    only an empty row or column, so the product is the same.

    The fill is an optimization for SuiteSparse:GraphBLAS, the library under both the
    "suitesparse" and "suitesparse-vanilla" backends. SuiteSparse uses its rowscale or
    colscale kernel for a diagonal matrix only when every diagonal entry is present
    (``GB_is_diagonal``). With 1e7 entries in 1e6 columns, that made ``A @ D`` 10x faster.
    Another GraphBLAS library may gain nothing from the fill and pay more for it, such as
    O(size) memory for ``full[...] = False``, which SuiteSparse stores as one iso value.

    On any library, the guard ``size <= 2 * nvals`` keeps the filled vector to at most twice
    as many entries as ``x``. The diagonal for a 2**60 x 2**60 matrix is filled only when
    ``x`` has at least 2**59 entries. Whoever adds another backend should revisit this
    function.
    """
    if x._nvals < x._size <= 2 * x._nvals:
        full = expr._new_vector(x.dtype, size=x._size)
        full(**opts)[...] = False
        full(mask=x.S, **opts) << x
        x = full
    return x.diag()


def _argminmax(agg, updater, expr, opts, *, in_composite, monoid):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        rowwise = expr.method_name == "reduce_rowwise"
        # True where an entry equals the min (or max) of its row or column
        if rowwise:
            step1 = A.reduce_rowwise(monoid).new(**opts)
            masked = semiring.any_eq(_scaling_diag(expr, step1, opts) @ A).new(**opts)
        else:
            step1 = A.reduce_columnwise(monoid).new(**opts)
            masked = semiring.any_eq(A @ _scaling_diag(expr, step1, opts)).new(**opts)
        masked(mask=masked.V, replace=True, **opts) << masked  # Could use select
        updater << _end_index(expr, masked, opts, last=False, rowwise=rowwise)
        if in_composite:
            return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        step1 = v.reduce(monoid, allow_empty=False).new(**opts)
        masked = binary.eq(v, step1).new(**opts)
        masked(mask=masked.V, replace=True, **opts) << masked  # Could use select
        step2 = _end_index_vector(expr, masked, opts, last=False)
        if in_composite:
            return step2
        _store_element(updater, step2[0], opts)
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        raise ValueError(f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar.")
    else:
        raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")


def _first_last(agg, updater, expr, opts, *, in_composite, last):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        rowwise = expr.method_name == "reduce_rowwise"
        step1 = _end_index(expr, A, opts, last=last, rowwise=rowwise).new(**opts)
        # `P` is True at the first (or last) entry of each nonempty row (or column) of `A`
        indices, positions = step1.to_coo()
        if rowwise:
            rows, cols = indices, positions
        else:
            rows, cols = positions, indices
        Matrix_ = type(expr._new_matrix(bool))
        P = Matrix_.from_coo(rows, cols, True, nrows=A._nrows, ncols=A._ncols)
        step2 = binary.first(A & P).new(**opts)
        if rowwise:
            updater << step2.reduce_rowwise(monoid.any)
        else:
            updater << step2.reduce_columnwise(monoid.any)
        if in_composite:
            return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        step1 = _end_index_vector(expr, v, opts, last=last)
        index = step1[0].new().value
        # `==` instead of `is` automatically triggers index.compute() in dask-graphblas:
        if index == None:  # noqa: E711
            # `v` is empty
            result = expr._new_vector(v.dtype, size=1)
            if in_composite:
                return result
            _store_element(updater, result[0], opts)
        elif in_composite:
            return v[[index]].new(**opts)
        else:
            _store_element(updater, v[index], opts)
    else:  # GrB_Matrix_reduce, in row-major order
        A = expr.args[0]
        step1 = _end_index(expr, A, opts, last=last, rowwise=True).new(**opts)
        step2 = _end_index_vector(expr, step1, opts, last=last)
        i = step2[0].new().value
        # `==` instead of `is` automatically triggers i.compute() in dask-graphblas:
        if i == None:  # noqa: E711
            # `A` is empty
            result = expr._new_vector(A.dtype, size=1)
            if in_composite:
                return result
            _store_element(updater, result[0], opts)
        else:
            j = step1[i].new().value
            if in_composite:
                return A[i, [j]].new(**opts)
            _store_element(updater, A[i, j], opts)


def _first_last_index(agg, updater, expr, opts, *, in_composite, last):
    if expr.cfunc_name == "GrB_Matrix_reduce_Aggregator":
        A = expr.args[0]
        updater << _end_index(
            expr, A, opts, last=last, rowwise=expr.method_name == "reduce_rowwise"
        )
        if in_composite:
            return updater.parent
    elif expr.cfunc_name.startswith("GrB_Vector_reduce"):
        v = expr.args[0]
        step1 = _end_index_vector(expr, v, opts, last=last)
        if in_composite:
            return step1
        _store_element(updater, step1[0], opts)
    elif expr.cfunc_name.startswith("GrB_Matrix_reduce"):
        raise ValueError(f"Aggregator {agg.name} may not be used with Matrix.reduce_scalar.")
    else:
        raise NotImplementedError(f"{agg.name} with {expr.cfunc_name}")


_INDEX_TYPES = dict.fromkeys(binary.first.types, INT64)
# argmin, argmax, first_index and last_index don't work with `reduce_scalar`
agg.argmin = Aggregator(
    "argmin",
    custom=partial(_argminmax, monoid=monoid.min),
    types=[dict.fromkeys(monoid.min.types, INT64)],
)
agg.argmax = Aggregator(
    "argmax",
    custom=partial(_argminmax, monoid=monoid.max),
    types=[dict.fromkeys(monoid.max.types, INT64)],
)
agg.first = Aggregator(
    "first",
    custom=partial(_first_last, last=False),
    types=[binary.first],
    any_dtype=True,
)
agg.last = Aggregator(
    "last",
    custom=partial(_first_last, last=True),
    types=[binary.second],
    any_dtype=True,
)
agg.first_index = Aggregator(
    "first_index",
    custom=partial(_first_last_index, last=False),
    types=[_INDEX_TYPES],
    any_dtype=INT64,
)
agg.last_index = Aggregator(
    "last_index",
    custom=partial(_first_last_index, last=True),
    types=[_INDEX_TYPES],
    any_dtype=INT64,
)
if backend == "suitesparse":
    # Also in `agg.ss`, for code that uses those names
    for _name in ["argmin", "argmax", "first", "last", "first_index", "last_index"]:
        setattr(agg.ss, _name, getattr(agg, _name))
    del _name

agg.Aggregator = Aggregator
agg.TypedAggregator = TypedAggregator

from .utils import get_typed_op  # noqa: E402 isort:skip

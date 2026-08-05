import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import graphblas as gb
from graphblas import (
    agg,
    backend,
    binary,
    config,
    dtypes,
    indexunary,
    monoid,
    op,
    select,
    semiring,
    unary,
)
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core import lib, operator
from graphblas.core.operator import (
    BinaryOp,
    IndexUnaryOp,
    Monoid,
    SelectOp,
    Semiring,
    UnaryOp,
    get_semiring,
)
from graphblas.core.ss import version_major as ss_version_major  # noqa: F401 (used in skipif)
from graphblas.dtypes import (
    BOOL,
    FP32,
    FP64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)
from graphblas.exceptions import DimensionMismatch, DomainMismatch, InvalidValue, UdfParseError

from .conftest import shouldhave

if dtypes._supports_complex:
    from graphblas.dtypes import FC32, FC64

from graphblas import Matrix, Vector  # isort:skip (for dask-graphblas)

suitesparse = backend == "suitesparse"


def orig_types(op):
    return op.types.keys() - op.coercions.keys()


def test_operator_initialized():
    assert operator.UnaryOp._initialized
    assert operator.BinaryOp._initialized
    assert operator.Monoid._initialized
    assert operator.Semiring._initialized


def test_op_repr():
    assert repr(unary.ainv) == "unary.ainv"
    assert repr(binary.plus) == "binary.plus"
    assert repr(monoid.times) == "monoid.times"
    assert repr(semiring.plus_times) == "semiring.plus_times"


def test_unaryop():
    assert unary.ainv["INT32"].gb_obj == lib.GrB_AINV_INT32
    assert unary.ainv[dtypes.UINT16].gb_obj == lib.GrB_AINV_UINT16
    if suitesparse:
        assert orig_types(unary.ss.positioni) == {INT32, INT64}
        assert orig_types(unary.ss.positionj1) == {INT32, INT64}


def test_binaryop():
    assert binary.plus["INT32"].gb_obj == lib.GrB_PLUS_INT32
    assert binary.plus[dtypes.UINT16].gb_obj == lib.GrB_PLUS_UINT16
    if suitesparse:
        assert orig_types(binary.ss.firsti) == {INT32, INT64}
        assert orig_types(binary.ss.secondj1) == {INT32, INT64}


def test_monoid():
    assert monoid.max["INT32"].gb_obj == lib.GrB_MAX_MONOID_INT32
    assert monoid.max[dtypes.UINT16].gb_obj == lib.GrB_MAX_MONOID_UINT16


def test_semiring():
    assert semiring.min_plus["INT32"].gb_obj == lib.GrB_MIN_PLUS_SEMIRING_INT32
    assert semiring.min_plus[dtypes.UINT16].gb_obj == lib.GrB_MIN_PLUS_SEMIRING_UINT16
    if suitesparse:
        assert orig_types(semiring.ss.min_firsti) == {INT32, INT64}


def test_agg():
    assert repr(agg.count) == "agg.count"
    assert repr(agg.count["INT32"]) == "agg.count[INT32]"
    if suitesparse:
        assert repr(agg.ss.first) == "agg.ss.first"
    assert "INT64" in agg.sum_of_inverses
    assert agg.sum_of_inverses["INT64"].return_type == FP64
    assert "BOOL" not in agg.sum_of_inverses
    with pytest.raises(KeyError, match="BOOL"):
        agg.sum_of_inverses["BOOL"]
    assert agg.varp["INT64"].return_type == "FP64"
    assert set(dir(agg)).issuperset({"count", "mean", "ss"})


def test_find_opclass_unaryop():
    assert operator.find_opclass(unary.minv)[1] == "UnaryOp"
    # assert operator.find_opclass(lib.GrB_MINV_INT64)[1] == 'UnaryOp'


def test_find_opclass_binaryop():
    assert operator.find_opclass(binary.times)[1] == "BinaryOp"
    # assert operator.find_opclass(lib.GrB_TIMES_INT64)[1] == 'BinaryOp'


def test_find_opclass_monoid():
    assert operator.find_opclass(monoid.max)[1] == "Monoid"
    # assert operator.find_opclass(lib.GxB_MAX_INT64_MONOID)[1] == 'Monoid'


def test_find_opclass_semiring():
    assert operator.find_opclass(semiring.plus_plus)[1] == "Semiring"
    # assert operator.find_opclass(lib.GxB_PLUS_PLUS_INT64)[1] == 'Semiring'


def test_find_opclass_invalid():
    assert operator.find_opclass("foobar")[1] == operator.UNKNOWN_OPCLASS
    # assert operator.find_opclass(lib.GrB_INP0)[1] == operator.UNKNOWN_OPCLASS


def test_get_typed_op():
    assert operator.get_typed_op(binary.bor, dtypes.INT64) is binary.bor[dtypes.INT64]
    with pytest.raises(KeyError, match="bor does not work with FP64"):
        operator.get_typed_op(binary.bor, dtypes.FP64)
    with pytest.raises(TypeError, match="Unable to get typed operator"):
        operator.get_typed_op(object(), dtypes.INT64)
    assert operator.get_typed_op("<", dtypes.INT64, kind="binary") is binary.lt["INT64"]
    assert operator.get_typed_op("-", dtypes.INT64, kind="unary") is unary.ainv["INT64"]
    assert operator.get_typed_op("+", dtypes.FP64, kind="monoid") is monoid.plus["FP64"]
    assert operator.get_typed_op("+[int64]", dtypes.FP64, kind="monoid") is monoid.plus["INT64"]
    assert operator.get_typed_op("+.*", dtypes.FP64, kind="semiring") is semiring.plus_times["FP64"]
    assert operator.get_typed_op("row<=", dtypes.INT64, kind="select") is select.rowle["INT64"]
    with pytest.raises(ValueError, match="Unable to get op from string"):
        operator.get_typed_op("+", dtypes.FP64)
    assert (
        operator.get_typed_op("+", dtypes.INT64, kind="binary|aggregator") is binary.plus["INT64"]
    )
    assert (
        operator.get_typed_op("count", dtypes.INT64, kind="binary|aggregator") is agg.count["INT64"]
    )
    with pytest.raises(ValueError, match="Unknown binary or aggregator"):
        operator.get_typed_op("bad_op_name", dtypes.INT64, kind="binary|aggregator")
    with pytest.raises(AttributeError):
        # get_typed_op expects dtypes to already be dtypes
        operator.get_typed_op(binary.plus, dtypes.INT64, "bad dtype")


@pytest.mark.skipif("supports_udfs")
def test_udf_mentions_numba():
    with pytest.raises(AttributeError, match="install numba"):
        binary.rfloordiv
    assert "rfloordiv" not in dir(binary)
    with pytest.raises(AttributeError, match="install numba"):
        semiring.any_rfloordiv
    assert "any_rfloordiv" not in dir(semiring)
    with pytest.raises(AttributeError, match="install numba"):
        op.absfirst
    assert "absfirst" not in dir(op)
    with pytest.raises(AttributeError, match="install numba"):
        op.plus_rpow
    assert "plus_rpow" not in dir(op)
    with pytest.raises(AttributeError, match="install numba"):
        binary.numpy.gcd
    assert "gcd" not in dir(binary.numpy)
    assert "gcd" not in dir(op.numpy)


@pytest.mark.skipif("supports_udfs")
def test_unaryop_udf_no_support():
    def plus_one(x):  # pragma: no cover (numba)
        return x + 1

    with pytest.raises(RuntimeError, match="UnaryOp.register_new.* unavailable"):
        unary.register_new("plus_one", plus_one)


@pytest.mark.skipif("not supports_udfs")
def test_unaryop_udf():
    def plus_one(x):
        return x + 1  # pragma: no cover (numba)

    unary.register_new("plus_one", plus_one)
    assert hasattr(unary, "plus_one")
    assert unary.plus_one.orig_func is plus_one
    assert unary.plus_one[int].orig_func is plus_one
    assert unary.plus_one[int]._numba_func(1) == 2
    comp_set = {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
        BOOL,
    }
    if dtypes._supports_complex:
        comp_set.update({FC32, FC64})
    assert set(unary.plus_one.types) == comp_set
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.plus_one)
    result = Vector.from_coo([0, 1, 3], [2, 3, -3], dtype=dtypes.INT32)
    assert v.isequal(result)
    assert "INT8" in unary.plus_one
    assert INT8 in unary.plus_one.types
    del unary.plus_one["INT8"]
    assert "INT8" not in unary.plus_one
    assert INT8 not in unary.plus_one.types
    with pytest.raises(TypeError, match="UDF argument must be a function"):
        UnaryOp.register_new("bad", object())
    assert not hasattr(unary, "bad")
    with pytest.raises(UdfParseError, match="Unable to parse function using Numba"):
        UnaryOp.register_new("bad", lambda x: v)  # pragma: no branch (numba)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_unaryop_parameterized():
    def plus_x(x=0):
        def inner(val):
            return val + x  # pragma: no cover (numba)

        return inner

    op = UnaryOp.register_anonymous(plus_x, parameterized=True)
    assert not op.is_positional
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.apply(op).new()
    assert v.isequal(v0, check_dtype=True)
    v0 = v.apply(op(0)).new()
    assert v.isequal(v0, check_dtype=True)
    v10 = v.apply(op(x=10)).new()
    r10 = Vector.from_coo([0, 1, 3], [11, 12, 6], dtype=dtypes.INT32)
    assert r10.isequal(v10, check_dtype=True)
    UnaryOp._initialize()  # no-op
    UnaryOp.register_new("plus_x_parameterized", plus_x, parameterized=True)
    op = unary.plus_x_parameterized
    v11 = v.apply(op(x=10)["INT32"]).new()
    assert r10.isequal(v11, check_dtype=True)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_binaryop_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x  # pragma: no cover (numba)

        return inner

    op = binary.register_anonymous(plus_plus_x, parameterized=True)
    assert not op.is_positional
    assert op.monoid is None
    assert op(1).monoid is None
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.ewise_mult(v, op).new()
    r0 = Vector.from_coo([0, 1, 3], [2, 4, -8], dtype=dtypes.INT32)
    assert v0.isequal(r0, check_dtype=True)
    v1 = v.ewise_add(v, op(1)).new()
    r1 = Vector.from_coo([0, 1, 3], [3, 5, -7], dtype=dtypes.INT32)
    assert v1.isequal(r1, check_dtype=True)

    w = Vector.from_coo([0, 0, 1, 3], [1, 0, 2, -4], dtype=dtypes.INT32, dup_op=op)
    assert v.isequal(w, check_dtype=True)
    with pytest.raises(TypeError, match="Monoid"):
        assert v.reduce(op).new() == -1

    v(op) << v
    assert v.isequal(r0)
    v(accum=op) << v
    x = r0.ewise_mult(r0, op).new()
    assert v.isequal(x)
    v(op(1)) << v
    x = x.ewise_mult(x, op(1)).new()
    assert v.isequal(x)
    v(accum=op(1)) << v
    x = x.ewise_mult(x, op(1)).new()
    assert v.isequal(x)

    assert v.isequal(Vector.from_coo([0, 1, 3], [19, 35, -61], dtype=dtypes.INT32))
    v11 = v.apply(op(1), left=10).new()
    r11 = Vector.from_coo([0, 1, 3], [30, 46, -50], dtype=dtypes.INT32)
    # Should we check for dtype here?
    # Is it okay if the literal scalar is an INT64, which causes the output to default to INT64?
    assert v11.isequal(r11, check_dtype=False)

    with pytest.raises(TypeError, match="UDF argument must be a function"):
        BinaryOp.register_new("bad", object())
    assert not hasattr(binary, "bad")

    def bad(x, y):  # pragma: no cover (numba)
        return v

    with pytest.raises(UdfParseError, match="Unable to parse function using Numba"):
        BinaryOp.register_new("bad", bad)

    def my_add(x, y):
        return x + y  # pragma: no cover (numba)

    op = BinaryOp.register_anonymous(my_add)
    assert op.name == "my_add"


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_monoid_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x  # pragma: no cover (numba)

        return inner

    bin_op = BinaryOp.register_anonymous(plus_plus_x, parameterized=True)

    # signatures must match
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, lambda x: -x)  # pragma: no branch (numba)
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, lambda y=0: -y)  # pragma: no branch (numba)
    with pytest.raises(TypeError, match="binaryop must be parameterized"):
        operator.ParameterizedMonoid("bad_monoid", binary.plus, 0)

    def plus_plus_x_identity(x=0):
        return -x

    assert bin_op.monoid is None
    bin_op1 = bin_op(1)
    assert bin_op1.monoid is None
    monoid = Monoid.register_anonymous(bin_op, plus_plus_x_identity, name="my_monoid")
    assert not monoid.is_positional
    assert bin_op.monoid is monoid
    assert bin_op(1).monoid is monoid(1)
    assert monoid(2) is bin_op(2).monoid
    assert not monoid.is_idempotent
    assert not monoid(1).is_idempotent
    # However, this still fails.
    # For this to work, we would need `bin_op1` to know it was created from a
    # ParameterizedBinaryOp. It would then need to check to see if the parameterized
    # parent has been associated with a monoid since the creation of `bin_op1`.
    assert bin_op1.monoid is None

    assert monoid.name == "my_monoid"
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.ewise_add(v, monoid).new()
    r0 = Vector.from_coo([0, 1, 3], [2, 4, -8], dtype=dtypes.INT32)
    assert v0.isequal(r0, check_dtype=True)
    v1 = v.ewise_mult(v, monoid(1)).new()
    r1 = Vector.from_coo([0, 1, 3], [3, 5, -7], dtype=dtypes.INT32)
    assert v1.isequal(r1, check_dtype=True)

    assert v.reduce(monoid).new() == -1
    assert v.reduce(monoid(1)).new() == 1
    # with pytest.raises(TypeError, match="BinaryOp"):  # NOW OKAY
    w1 = Vector.from_coo([0, 0, 1, 3], [1, 0, 2, -4], dtype=dtypes.INT32, dup_op=monoid)
    w2 = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    assert w1.isequal(w2)

    # identity may be a value
    def logaddexp(base):
        def inner(x, y):
            return np.log(base**x + base**y) / np.log(base)  # pragma: no cover (numba)

        return inner

    fv = v.apply(unary.identity).new(dtype=dtypes.FP64)
    bin_op = BinaryOp.register_anonymous(logaddexp, parameterized=True)
    Monoid.register_new("_user_defined_monoid", bin_op, -np.inf)
    monoid = gb.monoid._user_defined_monoid
    fv2 = fv.ewise_mult(fv, monoid(2)).new()

    def plus1(x):  # pragma: no cover (numba)
        return x + 1

    plus1 = UnaryOp.register_anonymous(plus1)
    expected = fv.apply(plus1).new()
    assert fv2.isclose(expected, check_dtype=True)
    with pytest.raises(TypeError, match="must be a BinaryOp"):
        Monoid.register_anonymous(monoid, 0)

    def plus_times_x(x=0):
        def inner(left, right):
            return (left + right) * x  # pragma: no cover (numba)

        return inner

    bin_op = BinaryOp.register_anonymous(plus_times_x, parameterized=True)

    def bad_identity(x=0):
        raise ValueError("hahaha!")

    assert bin_op.monoid is None
    monoid = Monoid.register_anonymous(
        bin_op, bad_identity, is_idempotent=True, name="broken_monoid"
    )
    assert bin_op.monoid is monoid
    assert bin_op(1).monoid is None
    assert monoid.is_idempotent


def test_monoid_rejects_builtin_binaryop():
    """A built-in binaryop cannot back a user monoid.

    Built-ins already own their monoid association (binary.max resolves to
    monoid.max), and SuiteSparse silently ignores the identity passed to
    GrB_Monoid_new for them, so accepting one would produce a monoid whose
    Python-side identity disagrees with what GraphBLAS computes with.
    """
    with pytest.raises(TypeError, match="must be a user-defined BinaryOp"):
        Monoid.register_new("_bad_builtin_monoid", binary.max, 0)
    with pytest.raises(TypeError, match="must be a user-defined BinaryOp"):
        Monoid.register_anonymous(binary.plus, 1)
    # The rejection happens before any object is created, so the built-in's
    # own monoid association is untouched.
    assert binary.max.monoid is monoid.max
    assert not hasattr(gb.monoid, "_bad_builtin_monoid")


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_semiring_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x  # pragma: no cover (numba)

        return inner

    def plus_plus_x_identity(x=0):
        return -x

    assert semiring.register_anonymous(monoid.min, binary.plus).name == "min_plus"

    bin_op = BinaryOp.register_anonymous(plus_plus_x, parameterized=True)
    mymonoid = monoid.register_anonymous(bin_op, plus_plus_x_identity)
    # monoid and binaryop are both parameterized
    mysemiring = Semiring.register_anonymous(mymonoid, bin_op, name="my_semiring")
    assert not mysemiring.is_positional
    assert mysemiring.name == "my_semiring"

    A = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    x = Vector.from_coo([0, 1], [10, 20])

    y = A.mxv(x, mysemiring).new()
    assert y.isequal(A.mxv(x, semiring.plus_plus).new())
    assert y.isequal(x.vxm(A.T, semiring.plus_plus).new())
    assert y.isequal(Vector.from_coo([0, 1], [33, 37]))

    y = A.mxv(x, mysemiring(1)).new()
    assert y.isequal(Vector.from_coo([0, 1], [36, 40]))  # three extra pluses

    y = x.vxm(A.T, mysemiring(1)).new()  # same as previous
    assert y.isequal(Vector.from_coo([0, 1], [36, 40]))

    y = x.vxm(A.T, mysemiring).new()
    assert y.isequal(Vector.from_coo([0, 1], [33, 37]))

    B = A.mxm(A, mysemiring).new()
    assert B.isequal(A.mxm(A, semiring.plus_plus).new())
    assert B.isequal(Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [7, 9, 11, 13]))

    B = A.mxm(A, mysemiring(1)).new()  # three extra pluses
    assert B.isequal(Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [10, 12, 14, 16]))

    with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
        A.ewise_add(A, mysemiring)

    # mismatched signatures.
    def other_binary(y=0):  # pragma: no cover (numba)
        def inner(left, right):
            return left + right - y

        return inner

    def other_identity(y=0):
        return x  # pragma: no cover (numba)

    other_op = BinaryOp.register_anonymous(other_binary, parameterized=True)
    other_monoid = Monoid.register_anonymous(other_op, other_identity)
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(other_op, plus_plus_x_identity)
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, other_identity)
    with pytest.raises(ValueError, match="Signatures"):
        Semiring.register_anonymous(other_monoid, bin_op)
    with pytest.raises(ValueError, match="Signatures"):
        Semiring.register_anonymous(mymonoid, other_op)

    # only monoid is parameterized
    Semiring.register_new("my_special_semiring", mymonoid, binary.plus)
    mysemiring = semiring.my_special_semiring
    B0 = A.mxm(A, semiring.plus_plus).new()
    B1 = A.mxm(A, mysemiring).new()
    B2 = A.mxm(A, mysemiring(0)).new()
    assert B0.isequal(B1)
    assert B0.isequal(B2)

    # only binaryop is parameterized
    mysemiring = Semiring.register_anonymous(monoid.plus, bin_op)
    B0 = A.mxm(A, semiring.plus_plus).new()
    B1 = A.mxm(A, mysemiring).new()
    B2 = A.mxm(A, mysemiring(0)).new()
    assert B0.isequal(B1)
    assert B0.isequal(B2)

    with pytest.raises(TypeError, match="must be a Monoid"):
        Semiring.register_anonymous(binary.plus, binary.plus)
    with pytest.raises(TypeError, match="must be a BinaryOp"):
        Semiring.register_anonymous(monoid.plus, monoid.plus)
    with pytest.raises(TypeError, match="At least one of"):
        operator.ParameterizedSemiring("bad_semiring", monoid.plus, binary.plus)
    with pytest.raises(TypeError, match="monoid must be of type"):
        operator.ParameterizedSemiring("bad_semiring", binary.plus, binary.plus)
    with pytest.raises(TypeError, match="binaryop must be of"):
        operator.ParameterizedSemiring("bad_semiring", monoid.plus, monoid.plus)

    # While we're here, let's check misc Matrix operations
    Adup = Matrix.from_coo([0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [100, 1, 2, 3, 4], dup_op=bin_op)
    Adup2 = Matrix.from_coo([0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [100, 1, 2, 3, 4], dup_op=binary.plus)
    assert Adup.isequal(Adup2)

    def plus_x(x=0):
        def inner(y):
            return x + y  # pragma: no cover (numba)

        return inner

    unaryop = UnaryOp.register_anonymous(plus_x, parameterized=True)
    B = A.apply(unaryop).new()
    assert B.isequal(A)

    # SuiteSparse 4.0.1 no longer supports reduce with user-defined binary op
    # But, we can associate this to a monoid!
    x = A.reduce_rowwise(bin_op).new()
    assert x.isequal(A.reduce_rowwise(binary.plus).new())
    x = A.reduce_columnwise(bin_op).new()
    assert x.isequal(A.reduce_columnwise(binary.plus).new())

    s = A.reduce_scalar(mymonoid).new()
    assert s.value == A.reduce_scalar(monoid.plus).new()

    assert A.reduce_scalar(bin_op).new() == A.reduce_scalar(binary.plus).new()

    B = A.kronecker(A, bin_op).new()
    assert B.isequal(A.kronecker(A, binary.plus).new())


@pytest.mark.skipif("not supports_udfs")
def test_unaryop_udf_bool_result():
    # numba has trouble compiling this, but we have a work-around
    def is_positive(x):
        return x > 0  # pragma: no cover (numba)

    UnaryOp.register_new("is_positive", is_positive)
    assert hasattr(unary, "is_positive")
    assert set(unary.is_positive.types) == {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
        BOOL,
    }
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    w = v.apply(unary.is_positive).new()
    result = Vector.from_coo([0, 1, 3], [True, True, False], dtype=dtypes.BOOL)
    assert w.isequal(result)


@pytest.mark.skipif("not supports_udfs")
def test_binaryop_udf():
    def times_minus_sum(x, y):
        return x * y - (x + y)  # pragma: no cover (numba)

    BinaryOp.register_new("bin_test_func", times_minus_sum)
    assert hasattr(binary, "bin_test_func")
    assert binary.bin_test_func[int].orig_func is times_minus_sum
    comp_set = {
        BOOL,  # goes to INT64
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
    }
    if dtypes._supports_complex:
        comp_set.update({FC32, FC64})
    assert set(binary.bin_test_func.types) == comp_set
    v1 = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.from_coo([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, binary.bin_test_func).new()
    result = Vector.from_coo([0, 1, 2, 3], [-1, 2, 3, -31], dtype=dtypes.INT32)
    assert w.isequal(result)


@pytest.mark.skipif("not supports_udfs")
def test_monoid_udf():
    def plus_plus_one(x, y):
        return x + y + 1  # pragma: no cover (numba)

    BinaryOp.register_new("plus_plus_one", plus_plus_one)
    Monoid.register_new("plus_plus_one", binary.plus_plus_one, -1)
    assert hasattr(monoid, "plus_plus_one")
    comp_set = {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
    }
    if dtypes._supports_complex:
        comp_set.update({FC32, FC64})
    assert set(monoid.plus_plus_one.types) == comp_set
    v1 = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.from_coo([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, monoid.plus_plus_one).new()
    result = Vector.from_coo([0, 1, 2, 3], [4, 2, 3, 4], dtype=dtypes.INT32)
    assert w.isequal(result)

    with pytest.raises(DomainMismatch):
        Monoid.register_anonymous(binary.plus_plus_one, {"BOOL": True})
    with pytest.raises(DomainMismatch):
        Monoid.register_anonymous(binary.plus_plus_one, {"BOOL": -1})


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_semiring_udf():
    def plus_plus_two(x, y):
        return x + y + 2  # pragma: no cover (numba)

    BinaryOp.register_new("plus_plus_two", plus_plus_two)
    Semiring.register_new("extra_twos", monoid.plus, binary.plus_plus_two)
    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    A = Matrix.from_coo(
        [0, 0, 0, 0, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [2, 3, 4, 5, 6, 7, 8, 9],
        dtype=dtypes.INT32,
    )
    w = v.vxm(A, semiring.extra_twos).new()
    result = Vector.from_coo([0, 1, 2, 3], [9, 11, 13, 15], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_binary_updates():
    assert not hasattr(binary, "div")
    assert binary.cdiv["INT64"].gb_obj == lib.GrB_DIV_INT64
    vec1 = Vector.from_coo([0], [1], dtype=dtypes.INT64)
    vec2 = Vector.from_coo([0], [2], dtype=dtypes.INT64)
    result = vec1.ewise_mult(vec2, binary.truediv).new()
    assert result.isclose(Vector.from_coo([0], [0.5], dtype=dtypes.FP64), check_dtype=True)
    vec4 = Vector.from_coo([0], [-3], dtype=dtypes.INT64)
    result2 = vec4.ewise_mult(vec2, binary.cdiv).new()
    assert result2.isequal(Vector.from_coo([0], [-1], dtype=dtypes.INT64), check_dtype=True)
    if shouldhave(binary, "floordiv"):
        result3 = vec4.ewise_mult(vec2, binary.floordiv).new()
        assert result3.isequal(Vector.from_coo([0], [-2], dtype=dtypes.INT64), check_dtype=True)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_nested_names():
    def plus_three(x):
        return x + 3  # pragma: no cover (numba)

    UnaryOp.register_new("incrementers.plus_three", plus_three)
    assert hasattr(unary, "incrementers")
    assert type(unary.incrementers) is operator.OpPath
    assert hasattr(unary.incrementers, "plus_three")
    comp_set = {
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FP32,
        FP64,
        BOOL,
    }
    if dtypes._supports_complex:
        comp_set.update({FC32, FC64})
    assert set(unary.incrementers.plus_three.types) == comp_set

    v = Vector.from_coo([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.incrementers.plus_three)
    result = Vector.from_coo([0, 1, 3], [4, 5, -1], dtype=dtypes.INT32)
    assert v.isequal(result), v

    def plus_four(x):
        return x + 4  # pragma: no cover (numba)

    UnaryOp.register_new("incrementers.plus_four", plus_four)
    assert hasattr(unary.incrementers, "plus_four")
    assert hasattr(op.incrementers, "plus_four")  # Also save it to `graphblas.op`!
    v << v.apply(unary.incrementers.plus_four)  # this is in addition to the plus_three earlier
    result2 = Vector.from_coo([0, 1, 3], [8, 9, 3], dtype=dtypes.INT32)
    assert v.isequal(result2), v

    def bad_will_overwrite_path(x):
        return x + 7  # pragma: no cover (numba)

    with pytest.raises(AttributeError):
        UnaryOp.register_new("incrementers", bad_will_overwrite_path)
    with pytest.raises(AttributeError, match="already defined"):
        UnaryOp.register_new("identity.newfunc", bad_will_overwrite_path)
    with pytest.raises(AttributeError, match="already defined"):
        UnaryOp.register_new("incrementers.plus_four", bad_will_overwrite_path)


@pytest.mark.slow
def test_op_namespace():
    assert op.abs is unary.abs
    assert op.minus is binary.minus
    assert op.plus is binary.plus
    assert op.plus_times is semiring.plus_times

    if shouldhave(unary.numpy, "fabs"):
        assert op.numpy.fabs is unary.numpy.fabs
    if shouldhave(binary.numpy, "subtract"):
        assert op.numpy.subtract is binary.numpy.subtract
    if shouldhave(binary.numpy, "add"):
        assert op.numpy.add is binary.numpy.add
    if shouldhave(semiring.numpy, "add_add"):
        assert op.numpy.add_add is semiring.numpy.add_add
    assert len(dir(op)) > 300
    if supports_udfs:
        assert len(dir(op.numpy)) > 500

    with pytest.raises(
        AttributeError, match="module 'graphblas.op.numpy' has no attribute 'bad_attr'"
    ):
        op.numpy.bad_attr

    # Make sure all have been initialized so `vars` below works
    for key in list(op._delayed):  # pragma: no cover (safety)
        getattr(op, key)
    opnames = {
        key
        for key, val in vars(op).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    unarynames = {
        key
        for key, val in vars(unary).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    binarynames = {
        key
        for key, val in vars(binary).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    monoidnames = {
        key
        for key, val in vars(monoid).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    semiringnames = {
        key
        for key, val in vars(semiring).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    indexunarynames = {
        key
        for key, val in vars(indexunary).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    selectnames = {
        key
        for key, val in vars(select).items()
        if isinstance(val, (operator.OpBase, operator.ParameterizedUdf))
    }
    extra_unary = unarynames - opnames - unary._deprecated.keys()
    assert not extra_unary
    extra_binary = binarynames - opnames - binary._deprecated.keys()
    assert not extra_binary
    assert not monoidnames - opnames, monoidnames - opnames
    extra_semiring = semiringnames - opnames - semiring._deprecated.keys()
    assert not extra_semiring
    extra_ops = (
        opnames - (unarynames | binarynames | monoidnames | semiringnames) - op._deprecated.keys()
    )
    assert not extra_ops
    # These are not part of the `op` namespace
    assert indexunarynames - opnames == indexunarynames, indexunarynames - opnames
    assert selectnames - opnames == selectnames, selectnames - opnames


@pytest.mark.slow
def test_binaryop_attributes_numpy():
    # Some coverage from this test depends on order of tests
    if shouldhave(monoid.numpy, "add"):
        assert binary.numpy.add[int].monoid is monoid.numpy.add[int]
        assert binary.numpy.add.monoid is monoid.numpy.add
    if shouldhave(binary.numpy, "subtract"):
        assert binary.numpy.subtract[int].monoid is None
        assert binary.numpy.subtract.monoid is None


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_binaryop_monoid_numpy():
    assert gb.binary.numpy.minimum[int].monoid is gb.monoid.numpy.minimum[int]


@pytest.mark.slow
def test_binaryop_attributes():
    assert binary.plus[int].monoid is monoid.plus[int]
    assert binary.minus[int].monoid is None
    assert binary.plus.monoid is monoid.plus
    assert binary.minus.monoid is None

    def plus(x, y):
        return x + y  # pragma: no cover (numba)

    if supports_udfs:
        op = BinaryOp.register_anonymous(plus, name="plus")
        assert op.monoid is None
        assert op[int].monoid is None
        assert op[int].parent is op

    assert binary.plus[int].parent is binary.plus
    if shouldhave(binary.numpy, "add"):
        assert binary.numpy.add[int].parent is binary.numpy.add

    # bad type
    assert binary.plus[bool].monoid is None
    if shouldhave(binary.numpy, "equal"):
        assert binary.numpy.equal[int].monoid is None
        assert binary.numpy.equal[bool].monoid is monoid.numpy.equal[bool]  # sanity

    for attr, val in vars(binary).items():
        if not isinstance(val, BinaryOp):
            continue
        print(attr)
        if hasattr(monoid, attr):
            assert val.monoid is not None
            assert any(val[type_].monoid is not None for type_ in val.types)
        else:
            assert val.monoid is None or val.monoid.name != attr
            assert all(
                val[type_].monoid is None or val[type_].monoid.name != attr for type_ in val.types
            )


@pytest.mark.slow
def test_monoid_attributes():
    assert monoid.plus[int].binaryop is binary.plus[int]
    assert monoid.plus[int].identity == 0
    assert monoid.plus.binaryop is binary.plus
    assert monoid.plus.identities == dict.fromkeys(monoid.plus.types, 0)

    if shouldhave(monoid.numpy, "add"):
        assert monoid.numpy.add[int].binaryop is binary.numpy.add[int]
        assert monoid.numpy.add[int].identity == 0
        assert monoid.numpy.add.binaryop is binary.numpy.add
        assert monoid.numpy.add.identities == dict.fromkeys(monoid.numpy.add.types, 0)

    def plus(x, y):  # pragma: no cover (numba)
        return x + y

    if supports_udfs:
        binop = BinaryOp.register_anonymous(plus, name="plus")
        op = Monoid.register_anonymous(binop, 0, name="plus")
        assert op.binaryop is binop
        assert op[int].binaryop is binop[int]
        assert op[int].parent is op

    assert monoid.plus[int].parent is monoid.plus
    if shouldhave(monoid.numpy, "add"):
        assert monoid.numpy.add[int].parent is monoid.numpy.add

    for attr, val in vars(monoid).items():
        if not isinstance(val, Monoid):
            continue
        print(attr)
        assert val.binaryop is not None
        assert val.identities is not None
        for type_ in val.types:
            x = val[type_]
            assert x.binaryop is not None
            assert x.identity is not None


@pytest.mark.slow
def test_semiring_attributes():
    assert semiring.min_plus[int].monoid is monoid.min[int]
    assert semiring.min_plus[int].binaryop is binary.plus[int]
    assert semiring.min_plus.monoid is monoid.min
    assert semiring.min_plus.binaryop is binary.plus

    if shouldhave(semiring.numpy, "add_subtract"):
        assert semiring.numpy.add_subtract[int].monoid is monoid.numpy.add[int]
        assert semiring.numpy.add_subtract[int].binaryop is binary.numpy.subtract[int]
        assert semiring.numpy.add_subtract.monoid is monoid.numpy.add
        assert semiring.numpy.add_subtract.binaryop is binary.numpy.subtract
        assert semiring.numpy.add_subtract[int].parent is semiring.numpy.add_subtract

    def plus(x, y):
        return x + y  # pragma: no cover (numba)

    if supports_udfs:
        binop = BinaryOp.register_anonymous(plus, name="plus")
        mymonoid = Monoid.register_anonymous(binop, 0, name="plus")
        op = Semiring.register_anonymous(mymonoid, binop, name="plus_plus")
        assert op.binaryop is binop
        assert op.binaryop[int] is binop[int]
        assert op.monoid is mymonoid
        assert op.monoid[int] is mymonoid[int]
        assert op[int].parent is op

    assert semiring.min_plus[int].parent is semiring.min_plus

    for attr, val in vars(semiring).items():
        if not isinstance(val, Semiring):
            continue
        print(attr)
        assert val.binaryop is not None
        assert val.monoid is not None
        for type_ in val.types:
            x = val[type_]
            assert x.binaryop is not None
            assert x.monoid is not None


def test_binaryop_superset_monoids():
    ignore = {"udt_any", "lazy2", "monoid_pickle", "monoid_pickle_par"}
    monoid_names = {x for x in dir(monoid) if not x.startswith("_")} - ignore
    binary_names = {x for x in dir(binary) if not x.startswith("_")} - ignore
    diff = monoid_names - binary_names
    assert not diff
    extras = {x for x in set(dir(monoid.numpy)) - set(dir(binary.numpy)) if not x.startswith("_")}
    extras -= ignore
    assert not extras, ", ".join(sorted(extras))


def test_div_semirings():
    assert not hasattr(semiring, "plus_div")
    A1 = Matrix.from_coo([0, 1], [0, 0], [-1, -3])
    A2 = Matrix.from_coo([0, 1], [0, 0], [2, 2])
    result = A1.T.mxm(A2, semiring.plus_cdiv).new()
    assert result[0, 0].new() == -1
    assert result.dtype == dtypes.INT64

    result = A1.T.mxm(A2, semiring.plus_truediv).new()
    assert result[0, 0].new() == -2
    assert result.dtype == dtypes.FP64

    if shouldhave(semiring, "plus_floordiv"):
        result = A1.T.mxm(A2, semiring.plus_floordiv).new()
        assert result[0, 0].new() == -3
        assert result.dtype == dtypes.INT64


@pytest.mark.slow
def test_get_semiring():
    sr = get_semiring(monoid.plus, binary.times)
    assert sr is semiring.plus_times
    # Be somewhat forgiving
    sr = get_semiring(monoid.plus, monoid.times)
    assert sr is semiring.plus_times
    sr = get_semiring(binary.plus, binary.times)
    assert sr is semiring.plus_times
    # But not if switched
    with pytest.raises(TypeError, match="switch"):
        get_semiring(binary.plus, monoid.times)

    def myplus(x, y):
        return x + y  # pragma: no cover (numba)

    if supports_udfs:
        binop = BinaryOp.register_anonymous(myplus, name="myplus")
        st = get_semiring(monoid.plus, binop)
        assert st.monoid is monoid.plus
        assert st.binaryop is binop

        binop = BinaryOp.register_new("myplus", myplus)
        assert binop is binary.myplus
        st = get_semiring(monoid.plus, binop)
        assert st.monoid is monoid.plus
        assert st.binaryop is binop

    with pytest.raises(TypeError, match="Monoid"):
        get_semiring(None, binary.times)
    with pytest.raises(TypeError, match="Binary"):
        get_semiring(monoid.plus, None)

    if shouldhave(binary.numpy, "copysign"):
        sr = get_semiring(monoid.plus, binary.numpy.copysign)
        assert sr.monoid is monoid.plus
        assert sr.binaryop is binary.numpy.copysign


def test_create_semiring():
    # stress test / sanity check
    monoid_names = {x for x in dir(monoid) if not x.startswith("_") and x != "ss"}
    binary_names = {x for x in dir(binary) if not x.startswith("_") and x != "ss"}
    for monoid_name, binary_name in itertools.product(monoid_names, binary_names):
        cur_monoid = getattr(monoid, monoid_name)
        if not isinstance(cur_monoid, Monoid):
            continue
        cur_binary = (
            getattr(binary, binary_name)
            if binary_name not in binary._deprecated
            else binary._deprecated[binary_name]
        )
        if not isinstance(cur_binary, BinaryOp):
            continue
        Semiring.register_anonymous(cur_monoid, cur_binary)


@pytest.mark.slow
def test_commutes():
    # Untyped
    assert binary.plus.commutes_to is binary.plus
    assert binary.plus.is_commutative
    assert binary.first.commutes_to is binary.second
    assert not binary.first.is_commutative
    assert monoid.plus.commutes_to is monoid.plus
    assert monoid.plus.is_commutative
    assert binary.atan2.commutes_to is None
    assert not binary.atan2.is_commutative
    assert semiring.plus_times.commutes_to is semiring.plus_times
    assert semiring.plus_times.is_commutative
    assert semiring.any_first.commutes_to is semiring.any_second
    assert semiring.plus_times.is_commutative
    if suitesparse:
        assert semiring.ss.min_secondi.commutes_to is semiring.ss.min_firstj
    if shouldhave(semiring, "plus_pow") and shouldhave(semiring, "plus_rpow"):
        assert semiring.plus_pow.commutes_to is semiring.plus_rpow
    assert not semiring.plus_pow.is_commutative
    if shouldhave(binary, "isclose"):
        assert binary.isclose.commutes_to is binary.isclose
        assert binary.isclose.is_commutative
        assert binary.isclose(0.1).commutes_to is binary.isclose(0.1)
    if shouldhave(binary, "floordiv") and shouldhave(binary, "rfloordiv"):
        assert binary.floordiv.commutes_to is binary.rfloordiv
        assert not binary.floordiv.is_commutative
    if shouldhave(binary.numpy, "add"):
        assert binary.numpy.add.commutes_to is binary.numpy.add
        assert binary.numpy.add.is_commutative
    if shouldhave(binary.numpy, "less") and shouldhave(binary.numpy, "greater"):
        assert binary.numpy.less.commutes_to is binary.numpy.greater
        assert not binary.numpy.less.is_commutative

    # Typed
    assert binary.plus[int].commutes_to is binary.plus[int]
    assert binary.plus[int].is_commutative
    assert binary.first[int].commutes_to is binary.second[int]
    assert not binary.first[int].is_commutative
    assert monoid.plus[int].commutes_to is monoid.plus[int]
    assert monoid.plus[int].is_commutative
    assert binary.atan2[int].commutes_to is None
    assert not binary.atan2[int].is_commutative
    assert semiring.plus_times[int].commutes_to is semiring.plus_times[int]
    assert semiring.plus_times[int].is_commutative
    assert semiring.any_first[int].commutes_to is semiring.any_second[int]
    assert semiring.plus_times[int].is_commutative
    if suitesparse:
        assert semiring.ss.min_secondi[int].commutes_to is semiring.ss.min_firstj[int]
    if shouldhave(semiring, "plus_rpow"):
        assert semiring.plus_pow[int].commutes_to is semiring.plus_rpow[int]
    assert not semiring.plus_pow[int].is_commutative
    if shouldhave(binary, "isclose"):
        assert binary.isclose(0.1)[int].commutes_to is binary.isclose(0.1)[int]
    if shouldhave(binary, "floordiv") and shouldhave(binary, "rfloordiv"):
        assert binary.floordiv[int].commutes_to is binary.rfloordiv[int]
        assert not binary.floordiv[int].is_commutative
    if shouldhave(binary.numpy, "add"):
        assert binary.numpy.add[int].commutes_to is binary.numpy.add[int]
        assert binary.numpy.add[int].is_commutative
    if shouldhave(binary.numpy, "less") and shouldhave(binary.numpy, "greater"):
        assert binary.numpy.less[int].commutes_to is binary.numpy.greater[int]
        assert not binary.numpy.less[int].is_commutative

    # Stress test (this can create extra semirings)
    names = dir(semiring)
    for name in names:
        if name in semiring._deprecated:
            val = semiring._deprecated[name]
        elif name == "ss":
            continue
        else:
            val = getattr(semiring, name)
        if not hasattr(val, "commutes_to"):
            continue
        assert val.commutes_to is None or isinstance(val.commutes_to, type(val))


def test_from_string():
    assert unary.from_string("-") is unary.ainv
    assert unary.from_string("abs[float]") is unary.abs[float]
    assert binary.from_string("+") is binary.plus
    assert binary.from_string("-[int]") is binary.minus[int]
    if config["mapnumpy"] or shouldhave(binary.numpy, "true_divide"):
        assert binary.from_string("true_divide") is binary.numpy.true_divide
    if shouldhave(binary, "floordiv"):
        assert binary.from_string("//") is binary.floordiv
    if shouldhave(binary.numpy, "mod"):
        assert binary.from_string("%") is binary.numpy.mod
    assert monoid.from_string("*[FP64]") is monoid.times["FP64"]
    assert semiring.from_string("min.plus") is semiring.min_plus
    assert semiring.from_string("min.+") is semiring.min_plus
    assert semiring.from_string("min_plus") is semiring.min_plus

    with pytest.raises(ValueError, match="does not end with"):
        assert binary.from_string("plus[int")
    with pytest.raises(ValueError, match="too many"):
        assert binary.from_string("plus[int][float]")
    with pytest.raises(ValueError, match="not matched by"):
        assert binary.from_string("plus][int]")
    with pytest.raises(ValueError, match="does not end with"):
        assert binary.from_string("plus[int]extra")
    with pytest.raises(ValueError, match="Unknown binary string"):
        assert binary.from_string("")
    with pytest.raises(ValueError, match="Unknown binary string"):
        assert binary.from_string("badname")
    with pytest.raises(ValueError, match="Bad semiring string"):
        assert semiring.from_string("badname")
    with pytest.raises(ValueError, match="Bad semiring string"):
        semiring.from_string("min.plus.times")

    assert op.from_string("-") is unary.ainv
    assert op.from_string("+") is binary.plus
    assert op.from_string("min.plus") is semiring.min_plus
    with pytest.raises(ValueError, match="Unknown op string"):
        op.from_string("min.plus.times")
    assert op.from_string("count") is agg.count

    assert agg.from_string("count") is agg.count
    assert agg.from_string("|") is agg.any
    assert agg.from_string("+[int]") is agg.sum[int]
    with pytest.raises(ValueError, match="Unknown agg string"):
        agg.from_string("bad_agg")

    assert select.from_string("tril") is select.tril
    assert select.from_string(">=") is select.valuege
    assert indexunary.from_string("rowindex") is indexunary.rowindex
    assert indexunary.from_string("rowindex[int]") is indexunary.rowindex[int]

    # Every namespace's from_string carries a docstring (GH #513)
    for ns in [unary, binary, monoid, semiring, select, indexunary, agg, op]:
        assert ns.from_string.__doc__


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_lazy_op():
    UnaryOp.register_new("lazy", lambda x: x, lazy=True)  # pragma: no branch (numba)
    assert isinstance(op.lazy, UnaryOp)
    assert isinstance(unary.lazy, UnaryOp)
    BinaryOp.register_new("lazy", lambda x, y: x + y, lazy=True)  # pragma: no branch (numba)
    Monoid.register_new("lazy", "lazy", 0, lazy=True)
    assert isinstance(monoid.lazy, Monoid)
    assert isinstance(binary.lazy, BinaryOp)
    Monoid.register_new("lazy2", binary.lazy, 0, lazy=True)
    assert isinstance(op.lazy2, Monoid)
    assert isinstance(monoid.lazy2, Monoid)
    Semiring.register_new("lazy", "lazy", "lazy", lazy=True)
    assert isinstance(semiring.lazy, Semiring)
    Semiring.register_new("lazy_lazy", monoid.lazy, binary.lazy, lazy=True)
    assert isinstance(semiring.lazy_lazy, Semiring)
    # numpy
    UnaryOp.register_new("numpy.lazy", lambda x: x, lazy=True)  # pragma: no branch (numba)
    assert isinstance(unary.numpy.lazy, UnaryOp)
    BinaryOp.register_new("numpy.lazy", lambda x, y: x + y, lazy=True)  # pragma: no branch (numba)
    Monoid.register_new("numpy.lazy", "numpy.lazy", 0, lazy=True)
    assert isinstance(monoid.numpy.lazy, Monoid)
    assert isinstance(binary.numpy.lazy, BinaryOp)
    Monoid.register_new("numpy.lazy2", binary.numpy.lazy, 0, lazy=True)
    assert isinstance(operator.get_semiring(monoid.numpy.lazy2, binary.numpy.lazy), Semiring)
    assert isinstance(op.numpy.lazy2, Monoid)
    assert isinstance(monoid.numpy.lazy2, Monoid)
    Semiring.register_new("numpy.lazy", "numpy.lazy", "numpy.lazy", lazy=True)
    assert isinstance(semiring.numpy.lazy, Semiring)
    Semiring.register_new("numpy.lazy_lazy", monoid.numpy.lazy, binary.numpy.lazy, lazy=True)
    assert isinstance(semiring.numpy.lazy_lazy, Semiring)
    # misc
    UnaryOp.register_new("misc.lazy", lambda x: x, lazy=True)  # pragma: no branch (numba)
    assert isinstance(unary.misc.lazy, UnaryOp)
    with pytest.raises(AttributeError):
        unary.misc.bad
    with pytest.raises(ValueError, match="Unknown unary string:"):
        unary.from_string("misc.lazy.badpath")
    assert op.from_string("lazy") is unary.lazy
    assert op.from_string("numpy.lazy") is unary.numpy.lazy


def test_positional():
    assert not unary.exp.is_positional
    assert not unary.abs[bool].is_positional
    assert not binary.plus.is_positional
    assert not binary.minus[float].is_positional
    assert not monoid.plus.is_positional
    assert not monoid.plus[int].is_positional
    assert not semiring.any_first.is_positional
    assert not semiring.any_second[int].is_positional
    if suitesparse:
        assert unary.ss.positioni.is_positional
        assert unary.ss.positioni1[int].is_positional
        assert unary.ss.positionj1.is_positional
        assert unary.ss.positionj[float].is_positional
        assert binary.ss.firsti.is_positional
        assert binary.ss.secondj1[int].is_positional
        assert semiring.ss.any_firsti.is_positional
        assert semiring.ss.any_secondj[int].is_positional


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt():
    record_dtype = np.dtype([("x", np.bool_), ("y", np.float64)], align=True)
    udt = dtypes.register_new("TestUDT", record_dtype)
    assert not udt._is_anonymous
    v = Vector(udt, size=3)
    w = Vector(udt, size=3)
    v[:] = 0
    w[:] = 1

    def _udt_identity(val):
        return val  # pragma: no cover (numba)

    udt_identity = UnaryOp.register_new("udt_identity", _udt_identity, is_udt=True)
    assert udt in udt_identity
    assert udt in binary.eq
    result = v.apply(udt_identity).new()
    assert result.isequal(v)
    assert dtypes.UINT8 in udt_identity
    assert udt in udt_identity
    assert int in udt_identity
    assert operator.get_typed_op(udt_identity, udt) is udt_identity[udt]
    with pytest.raises(ValueError, match="Unknown dtype:"):
        assert "badname" in binary.eq
    with pytest.raises(ValueError, match="Unknown dtype:"):
        assert "badname" in udt_identity

    def _udt_getx(val):
        return val["x"]  # pragma: no cover (numba)

    udt_getx = UnaryOp.register_anonymous(_udt_getx, "udt_getx", is_udt=True)
    assert udt in udt_getx
    result = v.apply(udt_getx).new()
    expected = Vector.from_coo([0, 1, 2], 0)
    assert result.isequal(expected)

    def _udt_index(val, idx, _, thunk):  # pragma: no cover (numba)
        if idx == 0:
            return thunk["y"]
        return -thunk["y"]

    _udt_index = IndexUnaryOp.register_anonymous(_udt_index, "_udt_index", is_udt=True)
    assert udt in _udt_index
    result = v.apply(_udt_index, (False, 3)).new()
    expected = Vector.from_coo([0, 1, 2], [3, -3, -3])
    assert result.isequal(expected)
    # A number fills every field, and an int is not a bool: 3 does not fit.
    with pytest.raises(ValueError, match="3 does not fit"):
        v.apply(_udt_index, 3)

    def _udt_first(x, y):
        return x  # pragma: no cover (numba)

    udt_first = BinaryOp.register_anonymous(_udt_first, "udt_first", is_udt=True)
    assert udt in udt_first
    assert operator.get_typed_op(udt_first, udt) is udt_first[udt]
    assert udt_first(v & w).new().isequal(v)
    # A literal fills every field, by type: an int does not fit the bool field,
    # True does. 1 still converts, being exact, but that is deprecated.
    assert udt_first(v, True).new().isequal(v)
    with pytest.warns(DeprecationWarning, match="1 does not fit TestUDT"):
        assert udt_first(v, 1).new().isequal(v)
    assert udt_first[udt, dtypes.INT64].return_type == udt
    assert udt_first[dtypes.INT64, udt].return_type == dtypes.INT64
    assert udt_first[udt, dtypes.BOOL].return_type == udt
    assert udt_first[dtypes.BOOL, udt].return_type == dtypes.BOOL
    udt_dup = dtypes.register_anonymous(record_dtype)
    assert udt_first[udt, udt_dup].return_type == udt
    # assert udt_first[udt_dup, udt].return_type == udt ?

    udt_any = Monoid.register_new("udt_any", udt_first, (0, 0))
    assert udt in udt_any
    assert (udt, udt) in udt_any
    assert (udt, dtypes.INT8) not in udt_any
    assert operator.get_typed_op(udt_any, udt) is udt_any[udt]
    assert udt_any(v | w).new().isequal(v)

    udt_semiring = Semiring.register_new("udt_semiring", udt_any, udt_first)
    assert udt in udt_semiring
    assert operator.get_typed_op(udt_semiring, udt) is udt_semiring[udt]
    assert udt_semiring(v @ v).new() == (0, 0)

    result = v.apply(gb.unary.identity).new()
    assert result.isequal(v)
    result = v.apply(gb.unary.one).new()
    assert result.dtype == dtypes.INT64
    expected = Vector(int, size=v.size)
    expected(result.S) << 1
    assert result.isequal(expected)
    if suitesparse:
        result = v.apply(gb.unary.ss.positioni).new()
        expected = expected.apply(gb.unary.ss.positioni).new()
        assert result.isequal(expected)

    result = indexunary.rowindex(v).new()
    assert result.isequal(Vector.from_coo([0, 1, 2], [0, 1, 2]))
    result = select.rowle(v, 2).new()
    assert result.isequal(v)

    class BreakCompile:
        pass

    def badfunc(x):  # pragma: no cover (numba)
        return BreakCompile(x)

    badunary = UnaryOp.register_anonymous(badfunc, is_udt=True)
    assert udt not in badunary
    assert int not in badunary

    def badfunc2(x, y):  # pragma: no cover (numba)
        return BreakCompile(x)

    badbinary = BinaryOp.register_anonymous(badfunc2, is_udt=True)
    assert udt not in badbinary
    assert int not in badbinary

    assert binary.first[udt].return_type is udt
    assert binary.first[udt].commutes_to is binary.second[udt]
    if suitesparse:
        assert semiring.ss.any_firsti[int].commutes_to is semiring.ss.any_secondj[int]
        assert semiring.ss.any_firsti[udt].commutes_to is semiring.ss.any_secondj[udt]

    assert binary.second[udt].type is udt
    assert binary.second[udt].type2 is udt
    assert binary.second[udt, dtypes.INT8].type is udt
    assert binary.second[udt, dtypes.INT8].type2 is dtypes.INT8
    assert semiring.any_second[udt, dtypes.INT8].type is udt
    assert semiring.any_second[udt, dtypes.INT8].type2 is dtypes.INT8
    assert binary.first[udt, dtypes.INT8].type is udt
    assert binary.first[udt, dtypes.INT8].type2 is dtypes.INT8
    assert monoid.any[udt].type2 is udt

    def _this_or_that(val, idx, _, thunk):  # pragma: no cover (numba)
        return val["x"]

    sel = SelectOp.register_anonymous(_this_or_that, is_udt=True)
    sel[udt]
    assert udt in sel
    # The thunk becomes an element of the record, whose bool field takes False, not 0.
    result = v.select(sel, False).new()
    assert result.nvals == 0
    assert result.dtype == v.dtype
    result = w.select(sel, False).new()
    assert result.nvals == 3
    assert result.isequal(w)


@pytest.mark.skipif("not supports_udfs")
def test_udf_division_by_zero_follows_numpy():
    """Dividing by zero in a UDF returns numpy's answer instead of losing the element.

    Under Numba's default error model the division raises ZeroDivisionError
    inside the cfunc, where Numba prints the traceback and returns, so
    GraphBLAS keeps whatever was in the output element (in practice the
    previous element's value). ``error_model="numpy"`` fixes that, but only if
    it is set on the ``njit`` Dispatcher: a Dispatcher holds one compilation
    per signature, and ``_build`` calls ``.compile(sig)`` before the wrapper
    exists, so setting it on the ``cfunc`` alone comes too late to matter.
    """

    def _idiv(x, y):  # pragma: no cover (numba)
        return x // y

    op = BinaryOp.register_anonymous(_idiv, "_udf_zero_idiv")
    v = Vector.from_coo([0, 1], [10, 20], dtype=dtypes.INT64)
    w = Vector.from_coo([0, 1], [2, 0], dtype=dtypes.INT64)
    assert op(v & w).new().to_coo()[1].tolist() == [5, 0]

    def _tdiv(x, y):  # pragma: no cover (numba)
        return x / y

    op = BinaryOp.register_anonymous(_tdiv, "_udf_zero_tdiv")
    v = Vector.from_coo([0, 1], [1.0, 1.0], dtype=dtypes.FP64)
    w = Vector.from_coo([0, 1], [2.0, 0.0], dtype=dtypes.FP64)
    assert op(v & w).new().to_coo()[1].tolist() == [0.5, float("inf")]

    # Same guarantee for a UDT UDF, which reaches the cfunc by another route.
    udt = dtypes.register_anonymous(
        np.dtype([("dz_a", np.int64), ("dz_b", np.int64)], align=True), "_UdfDivZeroRec"
    )

    def _rec_idiv(x, y):  # pragma: no cover (numba)
        return (x["dz_a"] // y["dz_a"], x["dz_b"])

    op = BinaryOp.register_anonymous(_rec_idiv, "_udf_zero_rec", is_udt=True)
    v = Vector(udt, size=1)
    v[0] = (10, 5)
    w = Vector(udt, size=1)
    w[0] = (0, 1)
    got = v.ewise_mult(w, op).new()[0].new().value
    assert (got["dz_a"], got["dz_b"]) == (0, 5)


@pytest.mark.skipif("not supports_udfs")
def test_select_op_outlives_source_indexunary():
    """A SelectOp keeps alive the IndexUnaryOp whose GraphBLAS handle it borrows.

    ``SelectOp._from_indexunary`` reuses the IndexUnaryOp's ``gb_obj`` rather
    than allocating a second one, and ``register_anonymous`` drops that
    IndexUnaryOp on the way out. Without an explicit reference the handle is
    freed as soon as it is collected, and every use of the SelectOp raises
    ``UninitializedObject``.
    """
    import gc

    def _ne_thunk(x, i, j, thunk):  # pragma: no cover (numba)
        return x != thunk

    sel = SelectOp.register_anonymous(_ne_thunk)
    gc.collect()
    v = Vector.from_coo([0, 1, 2], [1, 5, 9])
    assert v.select(sel, 5).new().isequal(Vector.from_coo([0, 2], [1, 9], size=3))


_jit_can_compile_cache = []


def _jit_can_compile():
    """True when SuiteSparse has a C compiler it can actually use.

    Without one it falls back to the Numba cfunc and says nothing, so the
    ``jit`` parameter would run the cfunc and report a pass.

    ``jit_compiler_is_usable`` alone is not enough: it only checks that the
    configured compiler path exists on disk, and a runner can have the file
    yet fail every compile (broken toolchain, missing headers).
    ``_enable_jit_for_udt`` is the same call the UDT auto-lift path makes,
    and it does a real compile once per process, so its answer is the
    honest one here; ``test_ssjit`` keys its skips on the same signal. The
    fixture calls this before it mutates the control, and the cache keeps
    later per-test mutations from flipping it.
    """
    if not _jit_can_compile_cache:
        from graphblas.core.ss.jit_config import _enable_jit_for_udt

        _jit_can_compile_cache.append(gb.ss.jit_compiler_is_usable() and _enable_jit_for_udt())
    return _jit_can_compile_cache[0]


@pytest.fixture(params=["jit", "cfunc"])
def udt_op_path(request):
    """Pin SuiteSparse to one execution path for built-in UDT operators.

    Each auto-lifted UDT op carries both a C JIT definition and a Numba
    cfunc, and SuiteSparse chooses between them per call depending on
    whether a C compiler is available. A machine with one and a machine
    without therefore run different code, so results have to hold on both.
    """
    path = request.param
    if backend != "suitesparse" or "jit_c_control" not in gb.ss.config:
        if path == "jit":
            pytest.skip("no SuiteSparse C JIT on this backend")
        yield path
        return
    previous = gb.ss.config["jit_c_control"]
    if path == "jit":
        if not _jit_can_compile():
            pytest.skip("C JIT compilation not available (probe failed or compiler missing)")
        # Set it rather than assume it. SuiteSparse demotes ``on`` to ``load``
        # after a failed compile, and a demoted control routes to the cfunc
        # silently, so this parameter would pass while running the other path.
        gb.ss.config["jit_c_control"] = "on"
    else:
        gb.ss.config["jit_c_control"] = "off"
    try:
        yield path
    finally:
        # Read before restoring: a demotion during the test is the signal that
        # the kernel never compiled, which no assertion in the test can see.
        demoted = path == "jit" and gb.ss.config["jit_c_control"] != "on"
        gb.ss.config["jit_c_control"] = previous
        if demoted:
            pytest.fail("SuiteSparse demoted jit_c_control; the JIT path did not run")


def _udt_vectors(udt, xs, ys=None):
    """Build one or two dense UDT vectors whose leaves all hold the given values.

    Values that repeat down the whole vector make it iso-valued, and
    SuiteSparse answers those from a single element without reaching for a
    C JIT kernel, so callers pass varied data.
    """
    names = udt.np_type.names
    out = []
    for vals in (xs, ys):
        if vals is None:
            continue
        v = Vector(udt, size=len(vals))
        for i, val in enumerate(vals):
            v[i] = tuple(val for _ in names) if names else np.full(udt.np_type.subdtype[1], val)
        out.append(v)
    return out


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_mixed_record_dtypes_use_each_operands_own_dtype(udt_op_path):
    """Two records sharing field names but not field types promote to the wider one.

    ``_check_udt_pair`` matches record operands on field names only, so their
    leaf dtypes can differ. Offering the return-type resolver just the left
    operand left it no choice but that record, so an int record over a float
    record came back as the int record: ``7 / 2.0`` landed as 3 and ``6 / 0.0``
    as INT64_MAX. Swapping the operands changed the answer for the same pair.
    """
    int_udt = dtypes.register_anonymous(np.dtype([("mxd_a", np.int64)], align=True), "_MixedRecInt")
    float_udt = dtypes.register_anonymous(
        np.dtype([("mxd_a", np.float64)], align=True), "_MixedRecFloat"
    )
    v = Vector(int_udt, size=2)
    v[0] = (6,)
    v[1] = (7,)
    w = Vector(float_udt, size=2)
    w[0] = (0.0,)
    w[1] = (2.0,)

    result = v.ewise_mult(w, binary.truediv).new()
    assert result.dtype == float_udt, "result should promote to the float record"
    assert result[0].new().value["mxd_a"] == float("inf")
    assert result[1].new().value["mxd_a"] == 3.5

    # The same pair the other way round must agree, which it did not when the
    # resolver only ever saw the left operand.
    swapped = w.ewise_mult(v, binary.truediv).new()
    assert swapped.dtype == float_udt
    assert swapped[1].new().value["mxd_a"] == 2.0 / 7.0


def _bitwise_eq(got, want):
    """Compare two floats by bit pattern, treating any two NaNs as equal.

    Bit patterns rather than ``==`` because ``-0.0 == 0.0``, and the sign of
    a zero is exactly what a min/max tie-break decides. NaNs are exempted
    because ``fmin`` may hand back either operand's NaN payload.
    """
    if np.isnan(got) and np.isnan(want):
        return True
    return got.tobytes() == want.tobytes()


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
@pytest.mark.parametrize("np_dtype", [np.float64, np.float32])
def test_udt_min_max_answer_what_the_builtin_dtype_answers(udt_op_path, np_dtype):
    """``binary.min[udt]`` must give what ``binary.min[FP64]`` gives, bit for bit.

    An operator that means one thing on FP64 and another on a record of
    FP64 is not one operator. SuiteSparse's ``GrB_MIN_FP64`` is C99 ``fmin``,
    so that is what the UDT kernels have to be, and this compares them
    directly against the built-in rather than against a convention chosen on
    the Python side. The grid is every ordered pair drawn from NaN, both
    infinities, both zeros and two ordinary values, so it covers a NaN on
    either side, two NaNs, and a signed-zero tie either way round.

    The signed-zero tie itself is compared by value only. C99 leaves
    ``fmin(-0.0, 0.0)`` unspecified and the built-in answers differently per
    platform (left operand on macOS x86, right operand on Linux x86, IEEE
    minNum on arm64 and Windows), so bit-for-bit agreement on that one pair
    is not something any implementation can promise. Everything else,
    including which zero a mixed zero/nonzero pair keeps, stays bit-exact.

    What this catches, in the two spellings it replaces: Python's builtin
    ``min``, which the generated code reached through the exec namespace,
    ordered NaN by position, and ``np.fmin`` under Numba gets the NaN rule
    right but keeps the left operand on a signed-zero tie, so it drifts from
    the C JIT kernel on ``min(0.0, -0.0)``. Both execution paths are checked
    because SuiteSparse picks between them without telling anyone.
    """
    nan, inf = float("nan"), float("inf")
    values = [nan, inf, -inf, -0.0, 0.0, 1.5, -2.5]
    xs = [x for x in values for _ in values]
    ys = list(values) * len(values)

    udt = dtypes.register_anonymous(
        np.dtype([("mmb_a", np_dtype)], align=True), f"_MinMaxBuiltin{np.dtype(np_dtype).name}"
    )
    v, w = _udt_vectors(udt, xs, ys)
    ref_v = Vector.from_dense(np.array(xs, dtype=np_dtype))
    ref_w = Vector.from_dense(np.array(ys, dtype=np_dtype))

    for gb_op in (binary.min, binary.max):
        expected = gb_op(ref_v & ref_w).new().to_dense()
        result = gb_op(v & w).new()
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            got = result[i].new().value[0]
            if x == 0 and y == 0 and np.signbit(x) != np.signbit(y):
                # The one unspecified cell of the grid: either signed zero is
                # a correct answer from either implementation, so only agree
                # that both produced a zero.
                msg = (
                    f"{udt_op_path} {gb_op.name}({x}, {y}) on {udt.name}: "
                    f"got {got!r}, built-in {np.dtype(np_dtype).name} gives {expected[i]!r}"
                )
                assert got == 0, msg
                assert expected[i] == 0, msg
                continue
            assert _bitwise_eq(got, expected[i]), (
                f"{udt_op_path} {gb_op.name}({x}, {y}) on {udt.name}: "
                f"got {got!r}, built-in {np.dtype(np_dtype).name} gives {expected[i]!r}"
            )

    # A NaN anywhere in the input must not change where a reduce lands. Under
    # the Python-builtin semantics this same multiset reduced to 1.0 or to nan
    # depending on which index the NaN sat at.
    for data in ([1.0, 2.0, 3.0, nan], [nan, 1.0, 2.0, 3.0], [1.0, nan, 3.0, 2.0]):
        (u,) = _udt_vectors(udt, data)
        assert u.reduce(monoid.min[udt]).new().value[0] == 1.0, f"{udt_op_path} {data}"
        assert u.reduce(monoid.max[udt]).new().value[0] == 3.0, f"{udt_op_path} {data}"


@pytest.mark.skipif("not supports_udfs")
def test_udt_truediv_divides_in_floating_point(udt_op_path):
    """``binary.truediv`` on integer fields divides in float64 and keeps the quotient.

    Regression: the C JIT kernel emitted C ``/``, which is integer division.
    ``10**18 / 3`` came out as 333333333333333333 under the C JIT and
    333333333333333312 (float64, like numpy) through the cfunc, so the same
    program gave different answers depending on whether a C compiler was
    installed. The result fields are now float64, as ``truediv`` on two INT64
    vectors is FP64, so the quotient is not truncated either.
    """
    udt = dtypes.register_anonymous(
        np.dtype([("tdv_i", np.int64), ("tdv_j", np.int64)], align=True), "_TrueDivIntUDT"
    )
    xs = [10**18, 10**18 + 1, 7, 22]
    ys = [3, 3, 2, 7]
    expected = np.array(xs, np.int64) / np.array(ys, np.int64)
    v, w = _udt_vectors(udt, xs, ys)
    result = binary.truediv(v & w).new()
    assert [result.dtype.np_type[name] for name in ("tdv_i", "tdv_j")] == [np.dtype(np.float64)] * 2
    got = [result[i].new().value[0] for i in range(len(xs))]
    assert got == list(expected), udt_op_path


@pytest.mark.skipif("not supports_udfs")
def test_udt_floordiv_matches_numpy_on_floats(udt_op_path):
    """``binary.floordiv`` on float fields is not ``floor(a / b)``.

    Regression: the C JIT kernel computed ``floor(a / b)``, which rounds
    differently from the remainder-based algorithm numpy and CPython use and
    treats infinities as ordinary values. ``1.0 // 0.1`` came out as 10.0
    instead of 9.0, and ``inf // 2.0`` as ``inf`` instead of NaN, while the
    cfunc agreed with numpy all along.
    """
    nan = float("nan")
    inf = float("inf")
    # ``floor(a / b)`` disagrees with numpy on the first four pairs: inf and
    # -inf where numpy gives NaN, 10.0 rather than 9.0 for 1.0 // 0.1, and
    # -0.0 rather than -1.0 for -2.0 // inf.
    xs = [inf, -inf, 1.0, -2.0, 2.0, -2.0, 0.0, nan, -7.0, 7.0, -0.0, 7.5]
    ys = [2.0, 2.0, 0.1, inf, 0.0, 0.0, 0.0, 2.0, 2.0, -2.0, 4.0, 2.5]
    for np_dtype, name in ((np.float64, "_FloorDivF64UDT"), (np.float32, "_FloorDivF32UDT")):
        udt = dtypes.register_anonymous(
            np.dtype([("fdv_a", np_dtype), ("fdv_b", np_dtype)], align=True), name
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.floor_divide(np.array(xs, np_dtype), np.array(ys, np_dtype))
        v, w = _udt_vectors(udt, xs, ys)
        result = binary.floordiv(v & w).new()
        got = np.array([result[i].new().value[0] for i in range(len(xs))], np_dtype)
        np.testing.assert_array_equal(got, expected, err_msg=f"{udt_op_path} {np_dtype.__name__}")
        # ``assert_array_equal`` reads -0.0 and 0.0 as equal, so the sign of a
        # zero quotient needs its own assertion. It is the whole job of the
        # ``copysign`` branch the C JIT kernel emits for an exact-zero result.
        np.testing.assert_array_equal(
            np.signbit(got),
            np.signbit(expected),
            err_msg=f"{udt_op_path} {np_dtype.__name__} sign of zero",
        )


@pytest.mark.skipif("not supports_udfs")
def test_udt_integer_division_by_zero_is_defined(udt_op_path):
    """Integer division by zero must return a value rather than trap.

    The C JIT kernel divided in integers, where a zero divisor is undefined
    behaviour: on x86-64 ``idiv`` raises #DE, which is SIGFPE and process
    death rather than an exception. AArch64's ``sdiv`` returns 0 and does not
    trap, so this cannot be exhibited on an arm64 machine. The same trap
    fires on ``INT_MIN / -1``, whose quotient is not representable.

    ``truediv`` now gives float64 fields, as on built-in integer vectors, so a
    zero divisor gives an infinity there. ``floordiv`` stays in integers, where
    the values are a choice: a zero divisor gives 0, as ``np.floor_divide``
    does, and ``INT_MIN // -1`` wraps to ``INT_MIN``, as numpy does.
    """
    signed = dtypes.register_anonymous(
        np.dtype([("dvz_a", np.int32), ("dvz_b", np.int8)], align=True), "_DivZeroSignedUDT"
    )
    unsigned = dtypes.register_anonymous(
        np.dtype([("dvz_c", np.uint32), ("dvz_d", np.uint64)], align=True), "_DivZeroUnsignedUDT"
    )
    inf = float("inf")
    v, w = _udt_vectors(signed, [7, -7, 100, -128], [0, 0, 3, -1])
    result = binary.truediv(v & w).new()
    got = [result[i].new().value[0] for i in range(4)]
    assert got == [inf, -inf, 100 / 3, 128.0], udt_op_path
    result = binary.floordiv(v & w).new()
    got = [result[i].new().value[0] for i in range(4)]
    assert got[:3] == [0, 0, 33], udt_op_path
    # ``-128 // -1`` is the second trapping case; numpy wraps it to INT8_MIN.
    result = binary.floordiv(v & w).new()
    assert result[3].new().value[1] == np.iinfo(np.int8).min, udt_op_path

    v, w = _udt_vectors(unsigned, [7, 9, 100, 5], [0, 0, 3, 2])
    result = binary.truediv(v & w).new()
    got = [result[i].new().value[0] for i in range(4)]
    assert got == [inf, inf, 100 / 3, 2.5], udt_op_path
    result = binary.floordiv(v & w).new()
    got = [result[i].new().value[0] for i in range(4)]
    assert got == [0, 0, 33, 2], udt_op_path

    # Floor division still floors for signed operands of mixed sign.
    v, w = _udt_vectors(signed, [-7, 7, -9, 11], [2, -2, 2, 3])
    result = binary.floordiv(v & w).new()
    got = [result[i].new().value[0] for i in range(4)]
    assert got == [-4, -4, -5, 3], udt_op_path


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.skipif("not dtypes._supports_complex")
def test_udt_complex_truediv_by_zero(udt_op_path):
    """``binary.truediv`` on a complex field survives a zero divisor.

    Numba's complex division raises ``ZeroDivisionError`` unconditionally,
    outside the error model's control, so the cfunc left the element unwritten
    while the C JIT kernel returned numpy's infinities. Reading back the
    abandoned element gave uninitialized memory, or the previous element's
    answer, either of which looks like a plausible value.
    """
    udt = dtypes.register_anonymous(
        np.dtype([("cxz_a", np.complex128)], align=True), "_ComplexDivZeroUDT"
    )
    xs = [3 + 4j, 0j, 1 + 1j, 2 - 2j]
    ys = [0j, 0j, 2 + 0j, 0j]
    v, w = _udt_vectors(udt, xs, ys)
    result = binary.truediv(v & w).new()
    got = np.array([result[i].new().value[0] for i in range(len(xs))])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.array(xs) / np.array(ys)
    np.testing.assert_array_equal(got, expected, err_msg=udt_op_path)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.skipif("not dtypes._supports_complex")
def test_udt_complex_times_truediv_with_infinities_as_numpy(udt_op_path):
    """Complex ``*`` and ``/`` on a UDT field answer as numpy does at infinities.

    From Python 3.14, CPython recovers infinities that the textbook formulas
    leave as ``nan`` (C99 Annex G), so ``1j * (inf+infj)`` is ``-inf+infj``
    there and ``nan+nanj`` in numpy. Numba 0.68 follows CPython, so a cfunc
    built on Numba's operators answered as Python and the C JIT kernel as
    numpy. Both now spell numpy's formulas out, on any Python and Numba.
    """
    inf, nan = np.inf, np.nan
    # The last pair is finite, with an exact product and quotient: numpy
    # divides by multiplying with ``1 / denominator``, which can round the last
    # bit differently from CPython's division, which the kernels use.
    xs = [1j, -2.5j, complex(nan, inf), complex(inf, nan), 1 + 1j, complex(inf, 0.0), -1 + 7j]
    ys = [complex(inf, inf), complex(inf, -inf), 2 + 1j, 1 + 1j, complex(inf, inf), 1j, 1 + 1j]
    rec = dtypes.register_anonymous(
        np.dtype([("cxi_s", np.complex64), ("cxi_d", np.complex128)], align=True), "_CxInfRec"
    )
    arr = dtypes.register_anonymous(np.dtype((np.complex64, (3,))), "_CxInfArr3")
    for udt, leaves in [(rec, [0, 1]), (arr, [0, 2])]:
        v, w = _udt_vectors(udt, xs, ys)
        for gb_op, reference in [(binary.times, np.multiply), (binary.truediv, np.true_divide)]:
            result = gb_op(v & w).new()
            for leaf in leaves:
                got = np.array([result[i].new().value[leaf] for i in range(len(xs))])
                with np.errstate(invalid="ignore"):
                    expected = reference(np.array(xs, got.dtype), np.array(ys, got.dtype))
                msg = f"{gb_op.name} {got.dtype} {udt_op_path}"
                np.testing.assert_array_equal(got, expected, err_msg=msg)


@pytest.mark.skipif("not supports_udfs")
# 136-byte UDT, which SS < 9 rejects; see test_udt_large_array.
@pytest.mark.skipif(
    "ss_version_major < 9",
    reason="SuiteSparse < 9 rejects a 136-byte UDT on builds without VLA support",
)
def test_udt_float_truediv_by_zero_is_infinite(udt_op_path):
    """A zero divisor on a float field gives numpy's infinity, not a lost element.

    Unlike the integer case, nothing here is guarded: the generated code
    divides and lets IEEE produce the infinity. That only holds because the
    generated wrapper is compiled under Numba's numpy error model, which
    nothing else in the suite pins down.
    """
    udt = dtypes.register_anonymous(np.dtype((np.float64, (17,))), "_FloatDivZeroArr17")
    xs = [1.0, -1.0, 0.0, 6.0]
    ys = [0.0, 0.0, 0.0, 3.0]
    v, w = _udt_vectors(udt, xs, ys)
    result = binary.truediv(v & w).new()
    got = np.array([result[i].new().value[0] for i in range(len(xs))])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.array(xs) / np.array(ys)
    np.testing.assert_array_equal(got, expected, err_msg=udt_op_path)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_ops_match_record_ops(udt_op_path):
    """The array-UDT codegen carries the same division and NaN fixes as records.

    Records and flat arrays go through separate branches in both the Numba
    and the C JIT generators, so each fix has to land in both.
    """
    nan = float("nan")
    inf = float("inf")
    float_udt = dtypes.register_anonymous(np.dtype((np.float64, (13,))), "_ArrOpsF64")
    xs = [inf, 1.0, -2.0, nan, -7.0, 2.0]
    ys = [2.0, 0.1, inf, 2.0, 2.0, 1.0]
    v, w = _udt_vectors(float_udt, xs, ys)
    # The reference computations touch inf and nan, and numpy raises the FP
    # invalid flag for them on some platforms (Linux and Windows, via fmod)
    # but not others; pyproject promotes the RuntimeWarning to an error.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_floordiv = np.floor_divide(np.array(xs), np.array(ys))
        expected_min = np.fmin(np.array(xs), np.array(ys))
    np.testing.assert_array_equal(
        [binary.floordiv(v & w).new()[i].new().value[0] for i in range(len(xs))],
        expected_floordiv,
        err_msg=udt_op_path,
    )
    # ``np.fmin``, not ``np.minimum``: ``binary.min`` is SuiteSparse's
    # ``GrB_MIN_FP64``, which ignores a NaN operand rather than propagating it.
    np.testing.assert_array_equal(
        [binary.min(v & w).new()[i].new().value[0] for i in range(len(xs))],
        expected_min,
        err_msg=udt_op_path,
    )

    int_udt = dtypes.register_anonymous(np.dtype((np.int64, (6,))), "_ArrOpsI64")
    ixs = [10**18, 7, -7, 100, -9, 5]
    iys = [3, 0, 0, 3, 2, 2]
    v, w = _udt_vectors(int_udt, ixs, iys)
    result = binary.truediv(v & w).new()
    got = [result[i].new().value[0] for i in range(len(ixs))]
    with np.errstate(divide="ignore"):
        assert got == list(np.array(ixs) / np.array(iys)), udt_op_path
    result = binary.floordiv(v & w).new()
    got = [result[i].new().value[0] for i in range(len(ixs))]
    assert got == [333333333333333333, 0, 0, 33, -5, 2], udt_op_path


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_multidim_array_ops_match_numpy(udt_op_path):
    """Built-in ops on a multi-dimensional array UDT agree with numpy on both paths.

    The C JIT typedef flattens any rank to ``double v [N]`` and the Numba
    wrapper walks the same flat run, so a 2-D UDT covers codegen that the 1-D
    cases reach only by accident of both being contiguous.
    """
    udt = dtypes.register_anonymous(np.dtype((np.float64, (3, 2))), "_ArrOps2D")
    xs = [1.0, -7.0, float("inf"), 2.0]
    ys = [0.1, 2.0, 2.0, 0.0]
    v, w = _udt_vectors(udt, xs, ys)
    for gb_op, reference in (
        (binary.floordiv, np.floor_divide),
        (binary.truediv, np.true_divide),
        # ``fmin`` rather than ``minimum``: these inputs carry no NaN, so the
        # two agree here, but ``binary.min`` is the NaN-ignoring one.
        (binary.min, np.fmin),
    ):
        result = gb_op(v & w).new()
        element = result[0].new().value
        assert element.shape == (3, 2)
        got = np.array([result[i].new().value[1, 1] for i in range(len(xs))])
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = reference(np.array(xs), np.array(ys))
        np.testing.assert_array_equal(got, expected, err_msg=f"{udt_op_path} {gb_op.name}")


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_binaryop(record_udt):
    v, w = _record_pair(record_udt)

    def _add_udt(x, y):
        return (x["a"] + y["a"], x["b"] + y["b"])  # pragma: no cover (numba)

    add_udt = BinaryOp.register_anonymous(_add_udt, "test_add_udt_b", is_udt=True)
    result = add_udt(v & w).new()
    assert result.dtype == record_udt
    expected = _record_expected(record_udt, [(11, 22.0), (33, 44.0), (55, 66.0)])
    assert result.isequal(expected)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_unaryop_vector(record_udt):
    v, _ = _record_pair(record_udt)

    def _double_udt(val):
        return (val["a"] * 2, val["b"] * 2.0)  # pragma: no cover (numba)

    double_udt = UnaryOp.register_anonymous(_double_udt, "test_double_udt_v", is_udt=True)
    result = v.apply(double_udt).new()
    assert result.dtype == record_udt
    expected = _record_expected(record_udt, [(2, 4.0), (6, 8.0), (10, 12.0)])
    assert result.isequal(expected)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_unaryop_matrix(record_udt):

    def _double_udt(val):
        return (val["a"] * 2, val["b"] * 2.0)  # pragma: no cover (numba)

    double_udt = UnaryOp.register_anonymous(_double_udt, "test_double_udt_m", is_udt=True)
    M = Matrix(record_udt, nrows=2, ncols=2)
    M[0, 0] = (1, 2.0)
    M[0, 1] = (3, 4.0)
    M[1, 0] = (5, 6.0)
    M[1, 1] = (7, 8.0)
    result = M.apply(double_udt).new()
    assert result[0, 0].new() == (2, 4.0)
    assert result[1, 1].new() == (14, 16.0)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_monoid(record_udt):
    """A Monoid built from a tuple-returning BinaryOp reduces field-by-field via ewise_add."""
    v, w = _record_pair(record_udt)

    def _add_udt(x, y):
        return (x["a"] + y["a"], x["b"] + y["b"])  # pragma: no cover (numba)

    add_udt = BinaryOp.register_anonymous(_add_udt, "test_add_udt_mon", is_udt=True)
    add_monoid = Monoid.register_anonymous(add_udt, (0, 0.0))
    result = add_monoid(v | w).new()
    expected = _record_expected(record_udt, [(11, 22.0), (33, 44.0), (55, 66.0)])
    assert result.isequal(expected)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_semiring(record_udt):
    """A Semiring built from tuple-returning ops drives mxm correctly."""
    v, _ = _record_pair(record_udt)

    def _add_udt(x, y):
        return (x["a"] + y["a"], x["b"] + y["b"])  # pragma: no cover (numba)

    def _first_udt(x, y):
        return x  # pragma: no cover (numba)

    add_udt = BinaryOp.register_anonymous(_add_udt, "test_add_udt_sr", is_udt=True)
    add_monoid = Monoid.register_anonymous(add_udt, (0, 0.0))
    first_udt = BinaryOp.register_anonymous(_first_udt, "test_first_udt_sr", is_udt=True)
    sr = Semiring.register_anonymous(add_monoid, first_udt)
    assert sr(v @ v).new() == (9, 12.0)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_indexunary(record_udt):
    v, _ = _record_pair(record_udt)

    def _idx_udt(val, idx, _col, thunk):
        return (val["a"] * (idx + 1), val["b"] + thunk["b"])  # pragma: no cover (numba)

    idx_op = IndexUnaryOp.register_anonymous(_idx_udt, "test_idx_udt", is_udt=True)
    result = v.apply(idx_op, (0, 100.0)).new()
    assert result.dtype == record_udt
    expected = _record_expected(record_udt, [(1, 102.0), (6, 104.0), (15, 106.0)])
    assert result.isequal(expected)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_tuple_return_3field():
    """Tuple-return works for records with more than two fields."""
    dtype3 = np.dtype([("x", np.int32), ("y", np.float64), ("z", np.int64)], align=True)
    udt3 = dtypes.register_anonymous(dtype3)
    v3 = Vector(udt3, size=2)
    v3[0] = (1, 2.0, 3)
    v3[1] = (4, 5.0, 6)
    w3 = Vector(udt3, size=2)
    w3[0] = (10, 20.0, 30)
    w3[1] = (40, 50.0, 60)

    def _add3(x, y):
        return (x["x"] + y["x"], x["y"] + y["y"], x["z"] + y["z"])  # pragma: no cover (numba)

    add3 = BinaryOp.register_anonymous(_add3, "test_add3", is_udt=True)
    result = add3(v3 & w3).new()
    expected3 = Vector(udt3, size=2)
    expected3[0] = (11, 22.0, 33)
    expected3[1] = (44, 55.0, 66)
    assert result.isequal(expected3)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_input_with_scalar_return(record_udt):
    """A UDF that reads a UDT but returns a scalar still works (no tuple unpacking)."""
    v, w = _record_pair(record_udt)

    def _sum_fields(x, y):
        return x["a"] + y["b"]  # pragma: no cover (numba)

    sum_op = BinaryOp.register_anonymous(_sum_fields, "test_sum_fields", is_udt=True)
    result = sum_op(v & w).new()
    assert result[0].new() == 21.0  # 1 + 20.0
    assert result[1].new() == 43.0  # 3 + 40.0
    assert result[2].new() == 65.0  # 5 + 60.0


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_return_type_errors():
    """Friendly error when a UDT UDF returns a shape that doesn't match the input."""
    record_dtype = np.dtype([("a", np.int64), ("b", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_RetErrUDT")

    # Wrong-arity tuple return: UDT has 2 fields, UDF returns 3.
    def _three(x, y):  # pragma: no cover (numba; raises before execution)
        return (x["a"] + y["a"], x["b"] + y["b"], 0)

    op_three = BinaryOp.register_anonymous(_three, is_udt=True)
    with pytest.raises(UdfParseError, match="tuple of length 3.*expected 2"):
        op_three[udt]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_return_type_errors_array_udt():
    """Tuple return against an array UDT input should suggest a numpy array,
    not "record UDT's fields" (which would be misleading).
    """
    arr_dtype = np.dtype((np.float64, (4,)))
    audt = dtypes.register_anonymous(arr_dtype, "_RetErrArrUDT")

    def _bad_tuple(x, y):  # pragma: no cover (numba; raises before execution)
        return (x[0] + y[0], x[1] + y[1], x[2] + y[2])

    op = BinaryOp.register_anonymous(_bad_tuple, is_udt=True)
    with pytest.raises(UdfParseError, match="array UDTs of shape.*numpy array"):
        op[audt]


@pytest.fixture(scope="module")
def record_udt():
    return dtypes.register_anonymous(
        np.dtype([("a", np.int64), ("b", np.float64)], align=True),
        "_BuiltinOpsRec",
    )


@pytest.fixture(scope="module")
def array_udt():
    return dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_BuiltinOpsArr")


def _record_pair(udt):
    """Return ``(v, w)`` with overlapping entries used by the record-UDT ops tests."""
    v = Vector(udt, size=3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)
    w = Vector(udt, size=3)
    w[0] = (10, 20.0)
    w[1] = (30, 40.0)
    w[2] = (50, 60.0)
    return v, w


def _record_expected(udt, rows):
    out = Vector(udt, size=len(rows))
    for i, row in enumerate(rows):
        out[i] = row
    return out


@pytest.mark.parametrize(
    ("op_name", "expected_rows"),
    [
        ("plus", [(11, 22.0), (33, 44.0), (55, 66.0)]),
        ("minus", [(-9, -18.0), (-27, -36.0), (-45, -54.0)]),
        ("times", [(10, 40.0), (90, 160.0), (250, 360.0)]),
        ("truediv", [(0.1, 0.1), (0.1, 0.1), (0.1, 0.1)]),
    ],
)
@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_builtin_binary_record(record_udt, op_name, expected_rows):
    """Per-field arithmetic on a record UDT matches the scalar definition."""
    v, w = _record_pair(record_udt)
    result = getattr(binary, op_name)(v & w).new()
    if op_name == "truediv":
        # The int64 field divides to float64, as INT64 does, so the result is
        # the record with both fields float64.
        expected_type = dtypes.lookup_dtype(
            np.dtype([("a", np.float64), ("b", np.float64)], align=True)
        )
    else:
        expected_type = record_udt
    assert result.dtype == expected_type
    assert result.isequal(_record_expected(expected_type, expected_rows))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_builtin_floordiv_record(record_udt):
    """``binary.floordiv`` on a record UDT applies ``//`` per field."""
    v, w = _record_pair(record_udt)
    result = binary.floordiv(w & v).new()
    assert result.isequal(_record_expected(record_udt, [(10, 10), (10, 10), (10, 10)]))


@pytest.mark.parametrize(
    ("op_name", "expected_rows"),
    [
        ("min", [(3, 1.0), (2, 6.0)]),
        ("max", [(5, 4.0), (7, 8.0)]),
    ],
)
@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_builtin_minmax_record(record_udt, op_name, expected_rows):
    """``binary.min`` / ``binary.max`` pick winners per field independently."""
    v = Vector(record_udt, size=2)
    v[0] = (5, 1.0)
    v[1] = (2, 8.0)
    w = Vector(record_udt, size=2)
    w[0] = (3, 4.0)
    w[1] = (7, 6.0)
    result = getattr(binary, op_name)(v & w).new()
    assert result.isequal(_record_expected(record_udt, expected_rows))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_unary_ainv_abs_record(record_udt):
    v = Vector(record_udt, size=3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)
    neg = v.apply(unary.ainv).new()
    assert neg.isequal(_record_expected(record_udt, [(-1, -2.0), (-3, -4.0), (-5, -6.0)]))

    mixed = Vector(record_udt, size=3)
    mixed[0] = (-1, -2.0)
    mixed[1] = (3, -4.0)
    mixed[2] = (-5, 6.0)
    assert (
        mixed.apply(unary.abs)
        .new()
        .isequal(_record_expected(record_udt, [(1, 2.0), (3, 4.0), (5, 6.0)]))
    )


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_matrix_apply_unary(record_udt):
    M = Matrix(record_udt, nrows=2, ncols=2)
    M[0, 0] = (1, 2.0)
    M[0, 1] = (3, 4.0)
    M[1, 0] = (5, 6.0)
    M[1, 1] = (7, 8.0)
    result = M.apply(unary.ainv).new()
    assert result[0, 0].new() == (-1, -2.0)
    assert result[0, 1].new() == (-3, -4.0)
    assert result[1, 0].new() == (-5, -6.0)
    assert result[1, 1].new() == (-7, -8.0)


def _array_pair(udt):
    a = Vector(udt, size=2)
    a[0] = [1.0, 2.0, 3.0]
    a[1] = [4.0, 5.0, 6.0]
    b = Vector(udt, size=2)
    b[0] = [10.0, 20.0, 30.0]
    b[1] = [40.0, 50.0, 60.0]
    return a, b


@pytest.mark.parametrize(
    ("op_name", "expected"),
    [
        ("plus", [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]),
        ("times", [[10.0, 40.0, 90.0], [160.0, 250.0, 360.0]]),
        ("minus", [[-9.0, -18.0, -27.0], [-36.0, -45.0, -54.0]]),
    ],
)
@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_builtin_binary_array(array_udt, op_name, expected):
    """Per-element arithmetic on a fixed-shape array UDT matches the scalar definition."""
    a, b = _array_pair(array_udt)
    result = getattr(binary, op_name)(a & b).new()
    for i, row in enumerate(expected):
        np.testing.assert_array_equal(result[i].new().value, row)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_unary_ainv_abs_array(array_udt):
    a, _b = _array_pair(array_udt)
    np.testing.assert_array_equal(a.apply(unary.ainv).new()[0].new().value, [-1.0, -2.0, -3.0])
    c = Vector(array_udt, size=2)
    c[0] = [-1.0, 2.0, -3.0]
    c[1] = [4.0, -5.0, 6.0]
    abs_c = c.apply(unary.abs).new()
    np.testing.assert_array_equal(abs_c[0].new().value, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(abs_c[1].new().value, [4.0, 5.0, 6.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_jit_typedef():
    """Registering a UDT sets GxB_JIT_C_NAME and GxB_JIT_C_DEFINITION."""
    from graphblas.core import ffi, lib
    from graphblas.core.operator.udt_utils import _has_jit_set

    if not _has_jit_set:
        pytest.skip("JIT not available")

    # Record UDT with valid identifier name. Use unique field names to avoid
    # collisions with UDTs registered by other tests (the registry caches by dtype).
    record_dtype = np.dtype([("jx", np.int64), ("jy", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "JitTypeTest")
    buf = ffi.new("char[512]")
    lib.GrB_Type_get_String(udt._carg, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "int64_t jx" in defn
    assert "double jy" in defn
    assert "JitTypeTest" in defn

    # Array UDT, of a length no other test uses: a literal can register float64
    # arrays of common lengths under another name first, in any test order.
    arr_dtype = np.dtype((np.float64, (31,)))
    arr_udt = dtypes.register_anonymous(arr_dtype, "Vec31")
    lib.GrB_Type_get_String(arr_udt._carg, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "double v [31]" in defn
    assert "Vec31" in defn

    # 2D array UDT
    mat_dtype = np.dtype((np.int32, (5, 5)))
    mat_udt = dtypes.register_anonymous(mat_dtype, "Mat5x5")
    lib.GrB_Type_get_String(mat_udt._carg, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "int32_t" in defn
    assert "Mat5x5" in defn


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_jit_op_definitions():
    """Auto-compiled UDT ops carry the JIT C name and source."""
    from graphblas.core import ffi, lib
    from graphblas.core.operator.udt_utils import _has_jit_set

    if not _has_jit_set:
        pytest.skip("JIT not available")

    record_dtype = np.dtype([("jp", np.int64), ("jq", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "JitOpTest")
    buf = ffi.new("char[1024]")

    # Binary op JIT definitions
    for op_name, expected_c_op in [("plus", "+"), ("minus", "-"), ("times", "*")]:
        typed = getattr(binary, op_name)[udt]
        lib.GrB_BinaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
        defn = ffi.string(buf).decode()
        assert f"{op_name}_JitOpTest" in defn
        assert "jp" in defn
        assert "jq" in defn
        assert expected_c_op in defn

    # Unary op JIT definitions
    typed = unary.ainv[udt]
    lib.GrB_UnaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "ainv_JitOpTest" in defn
    assert "jp" in defn

    # Array UDT JIT definitions, of a length no other test uses (see test_udt_jit_typedef)
    arr_dtype = np.dtype((np.float64, (37,)))
    arr_udt = dtypes.register_anonymous(arr_dtype, "Vec37Jit")
    typed = binary.plus[arr_udt]
    lib.GrB_BinaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "plus_Vec37Jit" in defn
    assert "z->v[i] = (x->v[i]) + (y->v[i])" in defn
    assert "i < 37" in defn

    typed = unary.ainv[arr_udt]
    lib.GrB_UnaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "ainv_Vec37Jit" in defn


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_auto_monoid():
    """Built-in monoids auto-lift to UDTs with the right identity per field."""
    record_dtype = np.dtype([("p", np.int64), ("q", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype)

    v = Vector(udt, size=3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)
    w = Vector(udt, size=3)
    w[0] = (10, 20.0)
    w[1] = (30, 40.0)
    w[2] = (50, 60.0)

    # monoid.plus: reduce and ewise_add
    result = v.reduce(monoid.plus).new()
    assert result == (9, 12.0)
    result = monoid.plus(v | w).new()
    expected = Vector(udt, size=3)
    expected[0] = (11, 22.0)
    expected[1] = (33, 44.0)
    expected[2] = (55, 66.0)
    assert result.isequal(expected)

    # monoid.times: reduce
    result = v.reduce(monoid.times).new()
    assert result == (15, 48.0)

    # monoid.min: reduce
    result = v.reduce(monoid.min).new()
    assert result == (1, 2.0)

    # monoid.max: reduce
    result = v.reduce(monoid.max).new()
    assert result == (5, 6.0)

    # Identity correctness: reduce of single element returns element
    single = Vector(udt, size=1)
    single[0] = (42, 99.5)
    for mon in [monoid.plus, monoid.times, monoid.min, monoid.max]:
        assert single.reduce(mon).new() == (42, 99.5)

    # __contains__
    assert udt in monoid.plus
    assert udt in monoid.times
    assert udt in monoid.min
    assert udt in monoid.max

    # ---- Array UDT ----
    arr_dtype = np.dtype((np.float64, (4,)))
    arr_udt = dtypes.register_anonymous(arr_dtype)

    a = Vector(arr_udt, size=3)
    a[0] = [1.0, 2.0, 3.0, 4.0]
    a[1] = [5.0, 6.0, 7.0, 8.0]
    a[2] = [9.0, 10.0, 11.0, 12.0]

    result = a.reduce(monoid.plus).new()
    np.testing.assert_array_equal(result.value, [15.0, 18.0, 21.0, 24.0])

    result = a.reduce(monoid.min).new()
    np.testing.assert_array_equal(result.value, [1.0, 2.0, 3.0, 4.0])

    result = a.reduce(monoid.max).new()
    np.testing.assert_array_equal(result.value, [9.0, 10.0, 11.0, 12.0])

    # monoid.any on UDTs must return an actual input value, never the identity.
    # Regression: previously ``binary.any._numba_func`` used ``_first`` semantics,
    # so the UDT-reduce fold ``acc = first(acc, v_i) = acc`` always left the
    # accumulator at the (zero) identity. Now ``_second`` semantics, so the fold
    # captures an actual value.
    arr_any = a.reduce(monoid.any).new()
    np.testing.assert_array_equal(arr_any.value, [9.0, 10.0, 11.0, 12.0])

    rec_any = Vector(udt, size=2)
    rec_any[0] = (7, 8.0)
    rec_any[1] = (11, 12.0)
    any_res = rec_any.reduce(monoid.any).new()
    # Result must be one of the input tuples; reject the (0, 0.0) identity.
    # Compare via Scalar.__eq__ over a tuple of candidates (a set would require
    # Scalar to be hashable, which it isn't).
    assert any_res in ((7, 8.0), (11, 12.0))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.parametrize("shape", [(4,), (3, 2)])
def test_udt_array_wrapper_stays_within_element(shape):
    """The array-UDT cfunc wrapper writes one element's bytes and nothing more.

    The wrapper used to load and store each operand as a ``NestedArray``
    *value*, which Numba models as its full array descriptor (meminfo, parent,
    nitems, itemsize, data, shape, strides): 56 bytes on 64-bit for a 1-D
    element and 72 for a 2-D one, whatever the payload size. Both elements here
    are smaller, and the old wrapper overran each by 24 bytes. SuiteSparse's
    generic reduce keeps a UDT accumulator in a stack array sized to the
    element, so ``monoid.any`` overflowed it on every fold; depending on the
    build that clobbered a spilled pointer (segfault or SIGBUS) or silently
    produced a wrong answer.

    ``binary.any`` compiles through the generic ``_numba_func`` branch of
    ``BinaryOp._compile_udt``, the same one a user's ``is_udt=True`` op takes,
    so its wrapper stands in for both. Driving it over heap buffers with slack
    makes a regression trip an assert instead of corrupting a stack frame. The
    two sentinels must differ: the old load/store copied the source element
    plus its trailing bytes, so if ``y``'s slack held ``z``'s guard value, the
    overrun rewrote the guard with identical bytes and went unseen.
    """
    import ctypes

    import numba

    from graphblas.core.operator.base import _get_udt_wrapper

    udt = dtypes.register_anonymous(np.dtype((np.float64, shape)), f"_ArrWrapPin{len(shape)}D")

    # Mirror the generic ``_numba_func`` branch of ``BinaryOp._compile_udt``.
    numba_func = binary.any._numba_func
    sig = (udt.numba_type, udt.numba_type)
    numba_func.compile(sig)
    numba_ret_type = numba_func.overloads[sig].signature.return_type
    wrapper, wrapper_sig = _get_udt_wrapper(
        numba_func, udt, udt, udt, numba_ret_type=numba_ret_type
    )
    cfunc = numba.cfunc(wrapper_sig, nopython=True, error_model="numpy")(wrapper)
    call = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(cfunc.address)

    itemsize = udt.np_type.itemsize
    slack = 128  # the old wrapper overran by 24 bytes; leave generous headroom
    z = np.full(itemsize + slack, 0xAB, dtype=np.uint8)
    x = np.full(itemsize + slack, 0xCD, dtype=np.uint8)
    y = np.full(itemsize + slack, 0xCD, dtype=np.uint8)
    xvals = np.arange(1.0, itemsize // 8 + 1).reshape(shape)
    yvals = 10 * xvals
    x[:itemsize] = xvals.ravel().view(np.uint8)
    y[:itemsize] = yvals.ravel().view(np.uint8)
    call(z.ctypes.data, x.ctypes.data, y.ctypes.data)

    # ``any`` uses ``_second`` semantics, so the payload must be ``y``'s.
    np.testing.assert_array_equal(z[:itemsize].view(np.float64).reshape(shape), yvals)
    overrun = np.flatnonzero(z[itemsize:] != 0xAB)
    assert overrun.size == 0, f"wrote {overrun.size} bytes past the element at offsets {overrun}"

    # Public-path smoke: the reduce whose stack accumulator the old wrapper
    # overflowed. Kept after the byte-level checks so a regression fails the
    # assert above instead of reaching code that may crash the process.
    v = Vector(udt, size=3)
    rows = [xvals, yvals, xvals + yvals]
    for i, row in enumerate(rows):
        v[i] = row
    res = v.reduce(monoid.any).new()
    assert any(np.array_equal(res.value, row) for row in rows)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_udf_returns_new_array():
    """An array-UDT UDF may build its result instead of returning an operand.

    The wrapper hands the UDF a numpy view of each element in the UDT's
    declared shape, so ordinary array expressions work and the result is copied
    back element-wise.
    """
    udt = dtypes.register_anonymous(np.dtype((np.float64, (8,))), "_ArrRetUDT")
    v = Vector(udt, size=2)
    v[0] = np.arange(8.0)
    v[1] = np.arange(8.0, 16.0)
    w = Vector(udt, size=2)
    w[0] = np.full(8, 100.0)
    w[1] = np.full(8, 200.0)

    def _add(x, y):
        return x + y  # pragma: no cover (numba)

    add_op = BinaryOp.register_anonymous(_add, "_arr_ret_add", is_udt=True)
    result = add_op(v & w).new()
    np.testing.assert_array_equal(result[0].new().value, np.arange(8.0) + 100.0)
    np.testing.assert_array_equal(result[1].new().value, np.arange(8.0, 16.0) + 200.0)

    def _double(x):
        return x * 2  # pragma: no cover (numba)

    double_op = UnaryOp.register_anonymous(_double, "_arr_ret_double", is_udt=True)
    np.testing.assert_array_equal(double_op(v).new()[0].new().value, np.arange(8.0) * 2)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_multidim_array_keeps_shape_in_udf():
    """A multi-dimensional array UDT reaches the UDF in its declared shape.

    The wrapper builds the operand with ``numba.carray(ptr, shape)``, so 2-D
    indexing and ``.shape`` work. Addressing it as a flat run of elements
    would still be memory-safe but would silently drop that metadata.
    """
    udt2d = dtypes.register_anonymous(np.dtype((np.float64, (2, 3))), "_ArrRet2D")
    m = Vector(udt2d, size=1)
    m[0] = np.arange(6.0).reshape(2, 3)
    n = Vector(udt2d, size=1)
    n[0] = np.full((2, 3), 10.0)

    def _add_corner(x, y):
        # Fails to compile unless `x` really is 2-D with shape metadata.
        return x + y[0, 0] + x.shape[1]  # pragma: no cover (numba)

    add_2d = BinaryOp.register_anonymous(_add_corner, "_arr_ret_add_2d", is_udt=True)
    np.testing.assert_array_equal(
        add_2d(m & n).new()[0].new().value, np.arange(6.0).reshape(2, 3) + 10.0 + 3
    )


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_udf_shape_errors():
    """An array-UDT UDF whose result can't fill the element is rejected when typed.

    Numba's ``Array`` type records ``ndim`` but not extents, so this is not a
    type error. It used to reach the cfunc, where the shape mismatch raises in
    a context that swallows the exception, handing the caller an uninitialized
    element and no error.
    """
    # Shapes unique to this test: ``register_anonymous`` caches by dtype and
    # freezes the JIT C name at first registration, so sharing a shape with
    # another test makes both order-dependent.
    udt9 = dtypes.register_anonymous(np.dtype((np.float64, (9,))), "_ShapeErr9")

    def _truncate(x):  # pragma: no cover (numba)
        return x[:2]

    op = UnaryOp.register_anonymous(_truncate, "_shape_err_trunc", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) when run on sample values"):
        op[udt9]

    # IndexUnaryOp passes the row and column between x and the thunk, and so
    # must the probe: with the thunk's array in an index slot, ``i + j`` would
    # not type as a slice bound. The probe's indices are 1, so this is (2,).
    def _head(x, i, j, thunk):  # pragma: no cover (numba)
        return (x + thunk)[: i + j]

    op = IndexUnaryOp.register_anonymous(_head, "_shape_err_head", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) when run on sample values"):
        op[udt9]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_array_leaf_shape_errors():
    """A record UDF that under-fills an array leaf, or can return None, is rejected.

    The wrapper slice-assigns array leaves, so a short return raises inside
    the cfunc and abandons the write part-way: leaves after it keep whatever
    SuiteSparse had in the buffer, scalar leaves included. A ``None`` leaf
    raises there too.
    """
    spec = np.dtype([("rl_vec", np.float64, (3,)), ("rl_tag", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec, "_RecLeafShape")

    def _short(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"][:2], x["rl_tag"])

    op = BinaryOp.register_anonymous(_short, "_rec_leaf_short", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) for field .* holds \(3,\)"):
        op[udt]

    # ``v if cond else None`` types as Optional. Numba unwraps it where the
    # wrapper assigns, and a ``None`` raises there like a short array does, so
    # the type alone rejects it: no probe, and no dependence on which branch
    # the sample values take. A scalar leaf fails the same way.
    def _maybe_none(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"] + y["rl_vec"] if x["rl_tag"] > 0 else None, x["rl_tag"])

    op = BinaryOp.register_anonymous(_maybe_none, "_rec_leaf_maybe_none", is_udt=True)
    with pytest.raises(UdfParseError, match=r"can return None for field \['rl_vec'\]"):
        op[udt]

    def _maybe_no_tag(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"], x["rl_tag"] if x["rl_tag"] > 0 else None)

    op = BinaryOp.register_anonymous(_maybe_no_tag, "_rec_leaf_maybe_no_tag", is_udt=True)
    with pytest.raises(UdfParseError, match=r"can return None for field \['rl_tag'\]"):
        op[udt]

    # A leaf that is always ``None`` would otherwise fail the wrapper compile
    # with Numba's full traceback; it gets the same one-line diagnostic.
    def _no_tag(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"], None)

    op = BinaryOp.register_anonymous(_no_tag, "_rec_leaf_no_tag", is_udt=True)
    with pytest.raises(UdfParseError, match=r"can return None for field \['rl_tag'\]"):
        op[udt]

    # An array for a scalar leaf would fail the wrapper compile the same way
    # (Numba has no record setitem for it), whether it is an operand's field
    # or built.
    def _arr_for_tag(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"], x["rl_vec"])

    op = BinaryOp.register_anonymous(_arr_for_tag, "_rec_leaf_arr_for_tag", is_udt=True)
    with pytest.raises(UdfParseError, match=r"returned an array for field \['rl_tag'\]"):
        op[udt]

    def _built_for_tag(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"], x["rl_vec"] + y["rl_vec"])

    op = BinaryOp.register_anonymous(_built_for_tag, "_rec_leaf_built_for_tag", is_udt=True)
    with pytest.raises(UdfParseError, match=r"returned an array for field \['rl_tag'\]"):
        op[udt]

    def _full(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"] + y["rl_vec"], x["rl_tag"] + y["rl_tag"])

    op = BinaryOp.register_anonymous(_full, "_rec_leaf_full", is_udt=True)
    v = Vector(udt, size=1)
    v[0] = ([1.0, 2.0, 3.0], 7)
    w = Vector(udt, size=1)
    w[0] = ([4.0, 5.0, 6.0], 8)
    got = v.ewise_mult(w, op).new()[0].new().value
    np.testing.assert_array_equal(got["rl_vec"], [5.0, 7.0, 9.0])
    assert got["rl_tag"] == 15


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_udf_broadcast_return():
    """A return that broadcasts to the element fills it, and is not rejected.

    The wrapper slice-assigns and numpy broadcasts on assignment, so a ``(1,)``
    return legitimately fills every slot of a ``(6,)`` element. Requiring an
    exact shape would refuse this, which works.
    """
    udt6 = dtypes.register_anonymous(np.dtype((np.float64, (6,))), "_BCast6")

    def _fill(x):  # pragma: no cover (numba)
        return x[:1] + 10.0

    op1 = UnaryOp.register_anonymous(_fill, "_bcast_fill", is_udt=True)
    assert op1[udt6].return_type is udt6
    v = Vector(udt6, size=1)
    v[0] = np.arange(1.0, 7.0)
    np.testing.assert_array_equal(v.apply(op1).new()[0].new().value, [11.0] * 6)

    # A row broadcast across a 2-D element: the same rule one rank up.
    udt42 = dtypes.register_anonymous(np.dtype((np.float64, (4, 2))), "_BCast42")

    def _fill_rows(x):  # pragma: no cover (numba)
        return x[:1, :] + 100.0

    op2 = UnaryOp.register_anonymous(_fill_rows, "_bcast_fill_rows", is_udt=True)
    assert op2[udt42].return_type is udt42
    v2 = Vector(udt42, size=1)
    v2[0] = np.arange(8.0).reshape(4, 2)
    np.testing.assert_array_equal(
        v2.apply(op2).new()[0].new().value, np.tile([100.0, 101.0], (4, 1))
    )

    # The other side of the boundary: (2,) does not broadcast to (6,), Numba's
    # slice-assign raises on it, and it stays rejected.
    def _short(x):  # pragma: no cover (numba)
        return x[:2] + 10.0

    op3 = UnaryOp.register_anonymous(_short, "_bcast_short", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) when run on sample values"):
        op3[udt6]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_leaf_broadcast_return():
    """A broadcastable array leaf fills its field, and later leaves still land.

    Same boundary as the array case, and
    ``test_udt_record_array_leaf_shape_errors`` holds the rejecting side. The
    scalar leaf is worth asserting because a leaf that raises in the cfunc
    abandons the write, leaving every leaf after it as SuiteSparse had it.
    """
    spec = np.dtype([("bc_vec", np.float64, (11,)), ("bc_tag", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec, "_RecLeafBCast")

    def _fill_leaf(x, y):  # pragma: no cover (numba)
        return (x["bc_vec"][:1] + y["bc_vec"][:1], x["bc_tag"] + y["bc_tag"])

    op1 = BinaryOp.register_anonymous(_fill_leaf, "_rec_leaf_bcast", is_udt=True)
    v = Vector(udt, size=1)
    v[0] = (np.arange(11.0), 7)
    w = Vector(udt, size=1)
    w[0] = (np.arange(11.0) + 1.0, 8)
    got = v.ewise_mult(w, op1).new()[0].new().value
    np.testing.assert_array_equal(got["bc_vec"], [1.0] * 11)
    assert got["bc_tag"] == 15

    # A scalar fills the leaf the same way. Its type says it is no array, so
    # the check lets it through without running the UDF.
    def _fill_leaf_scalar(x, y):  # pragma: no cover (numba)
        return (x["bc_vec"][0] + y["bc_vec"][0], x["bc_tag"] + y["bc_tag"])

    op2 = BinaryOp.register_anonymous(_fill_leaf_scalar, "_rec_leaf_bcast_scalar", is_udt=True)
    got = v.ewise_mult(w, op2).new()[0].new().value
    np.testing.assert_array_equal(got["bc_vec"], [1.0] * 11)
    assert got["bc_tag"] == 15


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_leaf_sequence_return():
    """A tuple or list fills an array leaf only at its exact length.

    Numba copies a sequence into the field element by element, and unlike an
    array it does not broadcast one. A wrong length raises inside the cfunc,
    and the element comes back holding whatever was in the buffer. A tuple's
    length is in its type; a list's is learned by running the UDF.
    """
    spec = np.dtype([("sq_v", np.float64, (3,)), ("sq_t", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec, "_RecLeafSeq")
    v = Vector(udt, size=1)
    v[0] = ([1.0, 2.0, 3.0], 7)

    def _tuple3(x, y):  # pragma: no cover (numba)
        return ((x["sq_v"][0], x["sq_v"][1], 9.0), x["sq_t"])

    def _list3(x, y):  # pragma: no cover (numba)
        return ([x["sq_v"][0], x["sq_v"][1], 9.0], x["sq_t"])

    for func in [_tuple3, _list3]:
        op = BinaryOp.register_anonymous(func, f"_rec_leaf_seq{func.__name__}", is_udt=True)
        got = v.ewise_mult(v, op).new()[0].new().value
        np.testing.assert_array_equal(got["sq_v"], [1.0, 2.0, 9.0])
        assert got["sq_t"] == 7

    def _tuple2(x, y):  # pragma: no cover (numba)
        return ((x["sq_v"][0], 9.0), x["sq_t"])

    op = BinaryOp.register_anonymous(_tuple2, "_rec_leaf_seq_tuple2", is_udt=True)
    with pytest.raises(UdfParseError, match=r"tuple of length 2 for field \['sq_v'\] of _RecL"):
        op[udt]

    # As a (1,) array this would broadcast; as a tuple it does not.
    def _tuple1(x, y):  # pragma: no cover (numba)
        return ((9.0,), x["sq_t"])

    op = BinaryOp.register_anonymous(_tuple1, "_rec_leaf_seq_tuple1", is_udt=True)
    with pytest.raises(UdfParseError, match=r"tuple of length 1 .* holds \(3,\)\. Return 3 values"):
        op[udt]

    def _list2(x, y):  # pragma: no cover (numba)
        return ([x["sq_v"][0], 9.0], x["sq_t"])

    op = BinaryOp.register_anonymous(_list2, "_rec_leaf_seq_list2", is_udt=True)
    with pytest.raises(UdfParseError, match=r"list of length 2 .* when run on sample values"):
        op[udt]

    # A sequence never fills a field of higher rank; Numba will not compile it.
    spec2 = np.dtype([("sq_m", np.float64, (2, 2)), ("sq_n", np.int64)], align=True)
    udt2 = dtypes.register_anonymous(spec2, "_RecLeafSeq2D")

    def _flat4(x, y):  # pragma: no cover (numba)
        return ((1.0, 2.0, 3.0, 4.0), x["sq_n"])

    op = BinaryOp.register_anonymous(_flat4, "_rec_leaf_seq_flat4", is_udt=True)
    with pytest.raises(UdfParseError, match=r"only fills a one-dimensional field"):
        op[udt2]

    # Nor a scalar field, like an array (test_udt_record_array_leaf_shape_errors).
    def _tuple_for_t(x, y):  # pragma: no cover (numba)
        return (x["sq_v"], (x["sq_t"], 1))

    op = BinaryOp.register_anonymous(_tuple_for_t, "_rec_leaf_seq_tuple_for_t", is_udt=True)
    with pytest.raises(UdfParseError, match=r"returned a tuple for field \['sq_t'\] .* a scalar"):
        op[udt]

    def _list_for_t(x, y):  # pragma: no cover (numba)
        return (x["sq_v"], [x["sq_t"], 1])

    op = BinaryOp.register_anonymous(_list_for_t, "_rec_leaf_seq_list_for_t", is_udt=True)
    with pytest.raises(UdfParseError, match=r"returned a list for field \['sq_t'\] .* a scalar"):
        op[udt]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_broadcast_matches_numba_slice_assign():
    """The shape check accepts exactly what the wrapper's slice-assign accepts.

    The check turns a silent cfunc failure into a registration error, so a
    shape it rejects that Numba would have assigned is a false rejection, and
    one it accepts that Numba raises on is the failure it exists to catch. Pin
    both directions against Numba itself, including the two ranks where
    broadcasting alone gives the wrong answer: ``(1, 6)`` fills a ``(6,)``
    destination because assignment drops leading ones, ``(6, 1)`` does not.
    """
    import numba

    from graphblas.core.operator.base import _fits_by_broadcast

    @numba.njit
    def _assign(z, src):  # pragma: no cover (numba)
        z[:] = src

    for dst, src in [
        ((6,), ()),
        ((6,), (1,)),
        ((6,), (6,)),
        ((6,), (2,)),
        ((6,), (12,)),
        ((6,), (1, 6)),
        ((6,), (6, 1)),
        ((2, 3), (1, 3)),
        ((2, 3), (2, 1)),
        ((2, 3), (1, 1)),
        ((2, 3), (3,)),
        ((2, 3), (2, 3)),
        ((2, 3), (6,)),
        ((2, 3), (3, 2)),
        # Every leading one is dropped, not just the first.
        ((6,), (1, 1, 6)),
        ((2, 3), (1, 1, 3)),
        ((2, 3), (2, 1, 3)),
    ]:
        try:
            _assign(np.zeros(dst), np.ones(src))
        except ValueError:
            numba_assigns = False
        else:
            numba_assigns = True
        assert _fits_by_broadcast(src, dst) is numba_assigns, (src, dst)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
# _ProbeWhenRec's numpy repr is 142 chars; see test_udt_eq_nested_record_with_nan_leaf.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_udf_shape_check_runs_udf_only_for_built_arrays(monkeypatch):
    """The shape check runs the UDF only when Numba's type lacks the extents.

    An operand, or operand field, returned as-is keeps its extents in Numba's
    ``NestedArray`` type, so it is checked without running user code. That
    includes rejecting one returned into a field of another shape. A tuple is
    checked the same way, since its length is in its type. Only an array the
    UDF builds, typed as a plain ``Array``, needs the probe; so does a list,
    whose type has no length (``test_udt_record_leaf_sequence_return``).
    """
    from graphblas.core.operator import base as _base

    probed = []
    real_probe = _base._run_udf_probe

    def spy(numba_func, arg_dtypes):
        probed.append(numba_func.py_func.__name__)
        return real_probe(numba_func, arg_dtypes)

    monkeypatch.setattr(_base, "_run_udf_probe", spy)
    udt = dtypes.register_anonymous(np.dtype((np.float32, (18,))), "_ProbeWhen18")
    rec = dtypes.register_anonymous(
        np.dtype(
            [("pw_a", np.float64, (3,)), ("pw_b", np.float64, (2,)), ("pw_n", np.int64)],
            align=True,
        ),
        "_ProbeWhenRec",
    )

    def _second(x, y):  # pragma: no cover (numba)
        return y

    op = BinaryOp.register_anonymous(_second, "_probe_when_second", is_udt=True)
    assert op[udt].return_type is udt

    def _fields(x, y):  # pragma: no cover (numba)
        return (x["pw_a"], y["pw_b"], x["pw_n"])

    op = BinaryOp.register_anonymous(_fields, "_probe_when_fields", is_udt=True)
    assert op[rec].return_type is rec

    def _swapped(x, y):  # pragma: no cover (numba)
        return (x["pw_b"], x["pw_a"], x["pw_n"])

    op = BinaryOp.register_anonymous(_swapped, "_probe_when_swapped", is_udt=True)
    with pytest.raises(
        UdfParseError, match=r"shape \(2,\) for field \['pw_a'\] of _ProbeWhenRec, but"
    ):
        op[rec]

    # A scalar for an array leaf broadcasts into it; its type says it is no
    # array, so there is nothing to probe.
    def _scalars(x, y):  # pragma: no cover (numba)
        return (x["pw_n"], y["pw_n"], x["pw_n"] + y["pw_n"])

    op = BinaryOp.register_anonymous(_scalars, "_probe_when_scalars", is_udt=True)
    assert op[rec].return_type is rec

    # A tuple carries its length in its type, so it is checked without a run.
    def _tuple_leaf(x, y):  # pragma: no cover (numba)
        return ((x["pw_a"][0], x["pw_a"][1], 9.0), y["pw_b"], x["pw_n"])

    op = BinaryOp.register_anonymous(_tuple_leaf, "_probe_when_tuple_leaf", is_udt=True)
    assert op[rec].return_type is rec
    assert probed == []

    def _sum(x, y):  # pragma: no cover (numba)
        return x + y

    op = BinaryOp.register_anonymous(_sum, "_probe_when_sum", is_udt=True)
    assert op[udt].return_type is udt
    assert probed == ["_sum"]

    # A UDF that raises on the probe's values is not checked, and still types.
    def _no_ones(x, y):  # pragma: no cover (numba)
        if x[0] == 1:
            raise ValueError("the probe's values")
        return x + y

    op = BinaryOp.register_anonymous(_no_ones, "_probe_when_raises", is_udt=True)
    assert op[udt].return_type is udt
    assert probed == ["_sum", "_no_ones"]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_array_field_roundtrip():
    """A record UDT with an array field writes exactly that field's extent.

    Numba's record-field setitem copies the *destination* extent whatever the
    source's length, so a short return used to read past the end of the source
    array. The wrapper slice-assigns array leaves to make that a shape error.
    """
    spec = np.dtype([("count", np.int64), ("vec", np.float64, (3,))], align=True)
    udt = dtypes.register_anonymous(spec, "_RecArrField")
    v = Vector(udt, size=2)
    v[0] = (1, [1.0, 2.0, 3.0])
    v[1] = (2, [4.0, 5.0, 6.0])

    def _combine(x, y):  # pragma: no cover (numba)
        return (x["count"] + y["count"], x["vec"] + y["vec"])

    op = BinaryOp.register_anonymous(_combine, "_rec_arr_field", is_udt=True)
    result = v.ewise_mult(v, op).new()
    assert result[0].new().value["count"] == 2
    np.testing.assert_array_equal(result[0].new().value["vec"], [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result[1].new().value["vec"], [8.0, 10.0, 12.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_udf_returning_operand_keeps_its_type():
    """A UDF that returns an array-UDT operand as-is produces that operand's UDT.

    A layered ``FP64[5][2]`` and the flat ``FP64[2, 5]`` share the Numba type
    ``nestedarray(float64, (2, 5))``, and ``lookup_dtype`` used to key UDTs on
    it, so ``return y`` named whichever of the two had registered last.
    SuiteSparse rejects the other one as a domain mismatch in a monoid or on
    assignment into the operand's vector. The two now register as one UDT, and
    a Numba type resolves through its numpy dtype, not registration order.
    """
    flat = dtypes.register_anonymous(np.dtype((np.float64, (2, 5))), "_RetOperandFlat")
    assert dtypes.register_anonymous(np.dtype((np.dtype((np.float64, (5,))), (2,)))) is flat

    def _second(x, y):  # pragma: no cover (numba)
        return y

    op = BinaryOp.register_anonymous(_second, "_ret_operand_second", is_udt=True)
    assert op[flat].return_type is flat

    v = Vector(flat, size=2)
    v[0] = np.arange(10.0).reshape(2, 5)
    v[1] = np.arange(10.0, 20.0).reshape(2, 5)
    w = Vector(flat, size=2)
    w << op(v & v)
    np.testing.assert_array_equal(w[1].new().value, np.arange(10.0, 20.0).reshape(2, 5))
    # ``binary.any`` resolves its return type the same way.
    res = v.reduce(monoid.any).new().value
    assert any(np.array_equal(res, v[i].new().value) for i in range(2))

    # A record's array field returned as-is carries its extents too, so it
    # resolves to the array UDT of that shape, whether or not one existed.
    # FP32 keeps each UDT under 128 bytes, the most SuiteSparse < 9 accepts on
    # builds without variable-length arrays (Windows).
    rec = dtypes.register_anonymous(
        np.dtype([("ro_n", np.int64), ("ro_vec", np.float32, (19,))], align=True),
        "_RetOperandRec",
    )
    udt14 = dtypes.register_anonymous(np.dtype((np.float32, (14,))), "_RetOperand14")

    def _field(x, y):  # pragma: no cover (numba)
        return x["ro_vec"]

    op2 = BinaryOp.register_anonymous(_field, "_ret_operand_field", is_udt=True)
    assert op2[rec, udt14].return_type is dtypes.lookup_dtype(np.dtype((np.float32, (19,))))
    r = Vector(rec, size=1)
    r[0] = (1, np.arange(19.0))
    u = Vector(udt14, size=1)
    u[0] = np.zeros(14)
    np.testing.assert_array_equal(r.ewise_mult(u, op2).new()[0].new().value, np.arange(19.0))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_udf_building_array_takes_operand_shape():
    """An array the UDF builds takes the type of the one operand it can match.

    Numba types ``x + 1.0`` as a plain ``Array`` with an element type and a
    rank but no extents, so only an operand can say how long the result is.
    Two same-rank candidates, or none, is an error rather than a guess.
    """
    udt15 = dtypes.register_anonymous(np.dtype((np.float64, (15,))), "_BuildArr15")
    udt16 = dtypes.register_anonymous(np.dtype((np.float64, (16,))), "_BuildArr16")

    def _shift(x, y):  # pragma: no cover (numba)
        return x + 1.0

    op = BinaryOp.register_anonymous(_shift, "_build_arr_shift", is_udt=True)
    assert op[udt15].return_type is udt15
    with pytest.raises(UdfParseError, match="matches more than one input array UDT"):
        op[udt15, udt16]

    def _fold(x, y):  # pragma: no cover (numba)
        return x.reshape(3, 5)

    op2 = BinaryOp.register_anonymous(_fold, "_build_arr_fold", is_udt=True)
    with pytest.raises(UdfParseError, match="matches no input array UDT"):
        op2[udt15]


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_udf_known_extents_never_match_by_rank(monkeypatch):
    """A return that kept its extents is not matched to a same-rank operand.

    ``lookup_dtype`` names a UDT for known extents, registering one if no UDT
    has that layout yet, so the rank matcher normally sees only an ``Array``
    the UDF built, which carries no extents. Registration can fail, though:
    SuiteSparse builds without variable-length arrays reject a UDT over 128
    bytes before 9.0 and over 1024 from then on. Matching known extents to a
    same-rank operand would then hand SuiteSparse an element of the wrong
    size, so refuse instead. Simulated here, since this build registers the
    type.
    """
    import numba

    from graphblas.core.operator import base as _base

    # FP32 keeps this UDT under that 128-byte cap, so it registers everywhere.
    udt21 = dtypes.register_anonymous(np.dtype((np.float32, (21,))), "_RankGuard21")
    ret = numba.typeof(np.dtype((np.float32, (23,)))).dtype
    # Same element type and rank, different length: what the rank matcher sees.
    assert ret.dtype == udt21.numba_type.dtype
    assert ret.ndim == udt21.numba_type.ndim

    real_lookup = _base.lookup_dtype

    def refuse_nested(key, value=None):
        if isinstance(key, numba.core.types.NestedArray):
            raise ValueError("simulated: SuiteSparse refused to register this UDT")
        return real_lookup(key, value)

    monkeypatch.setattr(_base, "lookup_dtype", refuse_nested)
    with pytest.raises(UdfParseError, match="matches no input array UDT"):
        _base._resolve_udt_return_type(ret, udt21)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_declared_with_nested_subarrays_gets_builtin_ops():
    """Built-in ops work on UDTs declared with nested subarray dtypes.

    numpy reports such a dtype's ``subdtype`` as ``(inner array dtype, outer
    shape)``, which the auto-lift codegen took for ``(scalar, shape)``:
    ``plus``, ``eq``, and ``ainv`` failed in Numba lowering, ``monoid.plus`` and
    ``agg.sum`` could not build an identity, and a UDF returning a record
    operand could come back as its flat-field twin. Registration flattens these
    dtypes now, so they take the ordinary path.
    """
    udt = dtypes.register_anonymous(np.dtype((np.dtype((np.int64, (3,))), (4,))), "_NestedDeclArr")
    assert udt.np_type == np.dtype((np.int64, (4, 3)))
    vals = np.arange(12).reshape(4, 3)
    v = Vector(udt, size=2)
    v[0] = vals
    v[1] = vals * 10
    np.testing.assert_array_equal(binary.plus(v & v).new()[1].new().value, vals * 20)
    assert binary.eq(v & v).new().to_coo()[1].all()
    np.testing.assert_array_equal(unary.ainv(v).new()[0].new().value, -vals)
    np.testing.assert_array_equal(v.reduce(monoid.plus).new().value, vals * 11)
    np.testing.assert_array_equal(v.reduce(agg.sum).new().value, vals * 11)

    point = np.dtype((np.float64, (3,)))
    rec = dtypes.register_anonymous(
        np.dtype([("nd_n", np.int64), ("nd_tri", point, (2,))], align=True), "_NestedDeclRec"
    )
    flat_twin = np.dtype([("nd_n", np.int64), ("nd_tri", np.float64, (2, 3))], align=True)
    assert dtypes.register_anonymous(flat_twin) is rec
    r = Vector(rec, size=2)
    r[0] = (1, np.ones((2, 3)))
    r[1] = (2, np.full((2, 3), 5.0))
    res = r.reduce(monoid.plus).new().value
    assert res["nd_n"] == 3
    np.testing.assert_array_equal(res["nd_tri"], np.full((2, 3), 6.0))

    def _first(x, y):  # pragma: no cover (numba)
        return x

    op = BinaryOp.register_anonymous(_first, "_nested_decl_first", is_udt=True)
    assert op[rec].return_type is rec


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_auto_semiring():
    """Built-in semirings auto-lift to UDTs and drive ``mxm``/``mxv``/``vxm``."""
    record_dtype = np.dtype([("r", np.float64), ("s", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype)

    # Matrix-vector multiply with plus_times
    A = Matrix(udt, nrows=2, ncols=2)
    A[0, 0] = (1.0, 2.0)
    A[0, 1] = (3.0, 4.0)
    A[1, 0] = (5.0, 6.0)
    A[1, 1] = (7.0, 8.0)
    x = Vector(udt, size=2)
    x[0] = (1.0, 1.0)
    x[1] = (1.0, 1.0)

    result = semiring.plus_times(A @ x).new()
    # [0] = (1*1 + 3*1, 2*1 + 4*1) = (4, 6)
    # [1] = (5*1 + 7*1, 6*1 + 8*1) = (12, 14)
    assert result[0].new() == (4.0, 6.0)
    assert result[1].new() == (12.0, 14.0)

    # __contains__
    assert udt in semiring.plus_times

    # vxm
    result = semiring.plus_times(x @ A).new()
    # [0] = (1*1 + 1*5, 1*2 + 1*6) = (6, 8)
    # [1] = (1*3 + 1*7, 1*4 + 1*8) = (10, 12)
    assert result[0].new() == (6.0, 8.0)
    assert result[1].new() == (10.0, 12.0)

    # mxm
    eye = Matrix(udt, nrows=2, ncols=2)
    eye[0, 0] = (1.0, 1.0)
    eye[1, 1] = (1.0, 1.0)
    result = semiring.plus_times(A @ eye).new()
    assert result.isequal(A)

    # Array UDT semiring (the use case from GH discussion #298)
    arr_dtype = np.dtype((np.float64, (3,)))
    arr_udt = dtypes.register_anonymous(arr_dtype)

    M = Matrix(arr_udt, nrows=2, ncols=2)
    M[0, 0] = [1.0, 2.0, 3.0]
    M[0, 1] = [4.0, 5.0, 6.0]
    M[1, 0] = [7.0, 8.0, 9.0]
    M[1, 1] = [10.0, 11.0, 12.0]
    ones = Vector(arr_udt, size=2)
    ones[0] = [1.0, 1.0, 1.0]
    ones[1] = [1.0, 1.0, 1.0]

    result = semiring.plus_times(M @ ones).new()
    np.testing.assert_array_equal(result[0].new().value, [5.0, 7.0, 9.0])
    np.testing.assert_array_equal(result[1].new().value, [17.0, 19.0, 21.0])

    # min_plus semiring on array UDT
    result = semiring.min_plus(M @ ones).new()
    np.testing.assert_array_equal(result[0].new().value, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result[1].new().value, [8.0, 9.0, 10.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_single_field_record():
    single_dtype = np.dtype([("val", np.float64)], align=True)
    single_udt = dtypes.register_anonymous(single_dtype)
    v = Vector(single_udt, 2)
    v[0] = (3.0,)
    v[1] = (7.0,)
    w = Vector(single_udt, 2)
    w[0] = (10.0,)
    w[1] = (20.0,)
    result = binary.plus(v & w).new()
    assert result[0].new() == (13.0,)
    assert result[1].new() == (27.0,)
    assert v.reduce(monoid.plus).new() == (10.0,)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_bool_field_record():
    """A record UDT with a bool field works for plus (bool + bool yields int in Numba)."""
    bool_dtype = np.dtype([("flag", np.bool_), ("count", np.int64)], align=True)
    bool_udt = dtypes.register_anonymous(bool_dtype)
    bv = Vector(bool_udt, 2)
    bv[0] = (True, 1)
    bv[1] = (False, 2)
    bw = Vector(bool_udt, 2)
    bw[0] = (True, 10)
    bw[1] = (True, 20)
    result = binary.plus(bv & bw).new()
    # count field sums normally; flag is ``bool(a + b)``, so it's False only
    # when both inputs are False.
    assert result[0].new().value.tolist() == (True, 11)
    assert result[1].new().value.tolist() == (True, 22)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_op_compilation_is_lazy():
    """Registering a UDT does not compile any ops for it; the first use does."""
    lazy_dtype = np.dtype([("lazy_a", np.int64), ("lazy_b", np.float64)], align=True)
    lazy_udt = dtypes.register_anonymous(lazy_dtype, "LazyCheck")
    assert (lazy_udt, lazy_udt) not in binary.plus._udt_ops
    assert lazy_udt not in monoid.plus._udt_ops
    binary.plus[lazy_udt]
    assert (lazy_udt, lazy_udt) in binary.plus._udt_ops
    # The monoid is independent of the binary op cache.
    assert lazy_udt not in monoid.plus._udt_ops


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
# A SuiteSparse compiled without variable-length arrays (MSVC, so the Windows
# builds) rejects a user-defined type larger than GB_VLA_MAXSIZE with
# GrB_INVALID_VALUE. That ceiling was 128 bytes through SS 8.x and is 1024 as
# of 9.0, so skipping on SS < 9 covers the affected builds and costs one test
# on the rest.
@pytest.mark.skipif(
    "ss_version_major < 9",
    reason="SuiteSparse < 9 rejects an 800-byte UDT on builds without VLA support",
)
def test_udt_large_array():
    big_dtype = np.dtype((np.float64, (100,)))
    big_udt = dtypes.register_anonymous(big_dtype)
    a = Vector(big_udt, 2)
    a[0] = list(range(100))
    a[1] = list(range(100, 200))
    b = Vector(big_udt, 2)
    b[0] = [1.0] * 100
    b[1] = [1.0] * 100
    result = binary.plus(a & b).new()
    np.testing.assert_array_equal(result[0].new().value, np.arange(1.0, 101.0))
    np.testing.assert_array_equal(result[1].new().value, np.arange(101.0, 201.0))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_int_array():
    int_arr_dtype = np.dtype((np.int32, (4,)))
    int_arr_udt = dtypes.register_anonymous(int_arr_dtype)
    iv = Vector(int_arr_udt, 2)
    iv[0] = [1, 2, 3, 4]
    iv[1] = [5, 6, 7, 8]
    iw = Vector(int_arr_udt, 2)
    iw[0] = [10, 20, 30, 40]
    iw[1] = [50, 60, 70, 80]
    result = binary.times(iv & iw).new()
    np.testing.assert_array_equal(result[0].new().value, [10, 40, 90, 160])
    np.testing.assert_array_equal(result[1].new().value, [250, 360, 490, 640])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_expr_repr_does_not_crash():
    """``repr`` of UDT expressions returns a non-empty string mentioning the UDT.

    Regression pin: the expression types used to lack ``_expr_name`` for UDT
    pointer return types, so ``repr`` raised. Pin both that the call returns
    a non-empty string and that the UDT's dtype name appears in it, so a
    future regression returning ``""`` or a generic placeholder still fails.
    """
    record_dtype2 = np.dtype([("rx", np.float64), ("ry", np.float64)], align=True)
    repr_udt = dtypes.register_anonymous(record_dtype2, "_ReprPinUdt")
    rv = Vector(repr_udt, 2)
    rv[:] = (1.0, 2.0)
    rw = Vector(repr_udt, 2)
    rw[:] = (10.0, 20.0)
    M = Matrix(repr_udt, 2, 2)
    M[:, :] = (1.0, 2.0)
    for expr in (rv + 1, 1 + rv, rv * 2, -rv, rv + rw, M + 1):
        text = repr(expr)
        assert text, f"repr returned empty string for {expr!r}"
        assert (
            repr_udt.name in text
        ), f"repr did not mention the UDT name {repr_udt.name!r}: {text!r}"


@pytest.mark.skipif("not supports_udfs")
def test_udt_eq_ne_nan_simple_record():
    """Simple float-field record: NaN-bearing entries compare unequal under eq.

    Regression: the original implementation byte-compared records (with a
    padding-byte mask) so two records whose float fields both held NaN
    compared *equal*. The cfunc now reads each leaf and applies scalar
    ``==`` / ``!=``, matching ``binary.eq[FP64](nan, nan) == False``.
    """
    spec = np.dtype([("eq_a", np.float64), ("eq_b", np.float64)], align=True)
    udt = dtypes.register_anonymous(spec, "_NaNEqSimple")
    v1 = Vector(udt, size=2)
    v2 = Vector(udt, size=2)
    v1[0] = (1.0, np.nan)
    v2[0] = (1.0, np.nan)
    v1[1] = (1.0, 2.0)
    v2[1] = (1.0, 2.0)
    eq = v1.ewise_mult(v2, binary.eq[udt]).new()
    ne = v1.ewise_mult(v2, binary.ne[udt]).new()
    assert eq[0].new().value is False
    assert eq[1].new().value is True
    assert ne[0].new().value is True
    assert ne[1].new().value is False


@pytest.mark.skipif("not supports_udfs")
def test_udt_eq_packed_mixed_width():
    """Packed (non-aligned) records compare by leaf, so padding bytes don't matter."""
    spec_packed = np.dtype([("pk_a", np.int32), ("pk_b", np.float64)])
    assert spec_packed.itemsize == 12  # packed: int32 + float64, no padding
    udt_pk = dtypes.register_anonymous(spec_packed, "_NaNEqPacked")
    vp1 = Vector(udt_pk, size=1)
    vp2 = Vector(udt_pk, size=1)
    vp1[0] = (1, 2.5)
    vp2[0] = (1, 2.5)
    assert vp1.ewise_mult(vp2, binary.eq[udt_pk]).new()[0].new().value is True


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
# SS < 9 has no GrB_NAME setter, so registration falls back to storing the
# numpy repr in the type name and warns when it does not fit in 128 chars.
# This dtype's repr is 133; how it serializes is not what the test is about.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_eq_nested_record_with_nan_leaf():
    """A NaN in a nested-record leaf still makes the outer record compare unequal."""
    nested = np.dtype(
        [("n_id", np.int32), ("n_pt", [("n_x", np.float64), ("n_y", np.float64)])],
        align=True,
    )
    udt_n = dtypes.register_anonymous(nested, "_NaNEqNested")
    vn1 = Vector(udt_n, size=2)
    vn2 = Vector(udt_n, size=2)
    vn1[0] = (1, (np.nan, 2.0))
    vn2[0] = (1, (np.nan, 2.0))
    vn1[1] = (1, (3.0, 4.0))
    vn2[1] = (1, (3.0, 4.0))
    eq_n = vn1.ewise_mult(vn2, binary.eq[udt_n]).new()
    assert eq_n[0].new().value is False
    assert eq_n[1].new().value is True


@pytest.mark.skipif("not supports_udfs")
def test_udt_eq_ne_array_with_nan_element():
    """Array UDT with a NaN element compares unequal under eq, equal under ne."""
    arr = np.dtype((np.float64, (3,)))
    udt_a = dtypes.register_anonymous(arr, "_NaNEqArr")
    va1 = Vector(udt_a, size=2)
    va2 = Vector(udt_a, size=2)
    va1[0] = [1.0, np.nan, 3.0]
    va2[0] = [1.0, np.nan, 3.0]
    va1[1] = [1.0, 2.0, 3.0]
    va2[1] = [1.0, 2.0, 3.0]
    eq_a = va1.ewise_mult(va2, binary.eq[udt_a]).new()
    ne_a = va1.ewise_mult(va2, binary.ne[udt_a]).new()
    assert eq_a[0].new().value is False
    assert eq_a[1].new().value is True
    assert ne_a[0].new().value is True
    assert ne_a[1].new().value is False


@pytest.fixture(scope="module")
def broadcast_record_udt():
    return dtypes.register_anonymous(
        np.dtype([("u", np.float64), ("v", np.float64)], align=True),
        "_BroadcastRecUdt",
    )


@pytest.fixture(scope="module")
def broadcast_array_udt():
    return dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_BroadcastArrUdt")


def _broadcast_record_vec(udt):
    v = Vector(udt, size=3)
    v[0] = (1.0, 2.0)
    v[1] = (3.0, 4.0)
    v[2] = (5.0, 6.0)
    return v


@pytest.mark.parametrize(
    ("op_name", "scalar_dtype", "scalar_values", "expected_rows"),
    [
        # commutative ops applied with UDT on the left
        ("plus", "int", [10, 20, 30], [(11.0, 12.0), (23.0, 24.0), (35.0, 36.0)]),
        ("times", "float", [2.0, 0.5, 10.0], [(2.0, 4.0), (1.5, 2.0), (50.0, 60.0)]),
        ("min", "int", [10, 20, 30], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        ("max", "int", [10, 20, 30], [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]),
    ],
)
@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_scalar_broadcast_udt_lhs(
    broadcast_record_udt, op_name, scalar_dtype, scalar_values, expected_rows
):
    """Scalar broadcasts to every field of a record UDT (UDT on the left)."""
    udt = broadcast_record_udt
    vec_udt = _broadcast_record_vec(udt)
    vec_s = Vector.from_coo([0, 1, 2], scalar_values, dtype=scalar_dtype)
    result = getattr(binary, op_name)(vec_udt & vec_s).new()
    expected = _record_expected(udt, expected_rows)
    assert result.isequal(expected)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_scalar_broadcast_commutativity(broadcast_record_udt):
    """``plus`` is commutative across UDT/scalar broadcast; ``minus`` is not."""
    udt = broadcast_record_udt
    vec_udt = _broadcast_record_vec(udt)
    vec_int = Vector.from_coo([0, 1, 2], [10, 20, 30])

    expected_plus = _record_expected(udt, [(11.0, 12.0), (23.0, 24.0), (35.0, 36.0)])
    assert binary.plus(vec_udt & vec_int).new().isequal(expected_plus)
    assert binary.plus(vec_int & vec_udt).new().isequal(expected_plus)

    expected_minus_udt_lhs = _record_expected(udt, [(-9.0, -8.0), (-17.0, -16.0), (-25.0, -24.0)])
    expected_minus_int_lhs = _record_expected(udt, [(9.0, 8.0), (17.0, 16.0), (25.0, 24.0)])
    assert binary.minus(vec_udt & vec_int).new().isequal(expected_minus_udt_lhs)
    assert binary.minus(vec_int & vec_udt).new().isequal(expected_minus_int_lhs)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_scalar_broadcast_plus_times(broadcast_array_udt):
    """Scalar broadcasts to every element of an array UDT for commutative ops."""
    arr_udt = broadcast_array_udt
    vec_arr = Vector(arr_udt, size=2)
    vec_arr[0] = [1.0, 2.0, 3.0]
    vec_arr[1] = [4.0, 5.0, 6.0]
    vec_s = Vector.from_coo([0, 1], [10.0, 100.0])

    # plus is commutative; both directions yield the same per-element broadcast.
    res_lhs = binary.plus(vec_arr & vec_s).new()
    res_rhs = binary.plus(vec_s & vec_arr).new()
    np.testing.assert_array_equal(res_lhs[0].new().value, [11.0, 12.0, 13.0])
    np.testing.assert_array_equal(res_lhs[1].new().value, [104.0, 105.0, 106.0])
    np.testing.assert_array_equal(res_rhs[0].new().value, [11.0, 12.0, 13.0])
    np.testing.assert_array_equal(res_rhs[1].new().value, [104.0, 105.0, 106.0])

    res_times = binary.times(vec_arr & vec_s).new()
    np.testing.assert_array_equal(res_times[0].new().value, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(res_times[1].new().value, [400.0, 500.0, 600.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_array_scalar_broadcast_minus_direction(broadcast_array_udt):
    """For non-commutative ops on array UDTs, operand order is respected."""
    arr_udt = broadcast_array_udt
    vec_arr = Vector(arr_udt, size=2)
    vec_arr[0] = [1.0, 2.0, 3.0]
    vec_arr[1] = [4.0, 5.0, 6.0]
    vec_s = Vector.from_coo([0, 1], [10.0, 100.0])

    res_scalar_lhs = binary.minus(vec_s & vec_arr).new()
    np.testing.assert_array_equal(res_scalar_lhs[0].new().value, [9.0, 8.0, 7.0])

    res_udt_lhs = binary.minus(vec_arr & vec_s).new()
    np.testing.assert_array_equal(res_udt_lhs[0].new().value, [-9.0, -8.0, -7.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_matrix_scalar_broadcast(broadcast_record_udt):
    """Scalar/UDT broadcast also works on matrix-shaped operands."""
    udt = broadcast_record_udt
    mat = Matrix(udt, nrows=2, ncols=2)
    mat[:, :] = (1.0, 2.0)
    mat_int = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [10, 20, 30, 40], nrows=2, ncols=2)
    result = binary.plus(mat & mat_int).new()
    assert result[0, 0].new() == (11.0, 12.0)
    assert result[1, 1].new() == (41.0, 42.0)


# eq/ne broadcasting between a UDT and a scalar type.
#
# Before this fix, ``binary.eq(udt_vec & int_vec)`` silently reinterpreted
# the int cell as a UDT struct (reading past the cell) and produced
# byte-comparison nonsense that happened to look like plausible False/True.
# Now the scalar broadcasts to every leaf, so ``eq`` is true only when all
# leaves equal the scalar.


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast_record():
    record = dtypes.register_anonymous(
        np.dtype([("u", np.float64), ("v", np.float64)], align=True),
        name="_EqBcastUV",
    )
    v_udt = Vector(record, size=3)
    v_udt[0] = (10.0, 10.0)
    v_udt[1] = (10.0, 20.0)  # partial match
    v_udt[2] = (1.0, 2.0)  # no match
    v_int = Vector.from_coo([0, 1, 2], [10, 10, 10])

    eq_result = binary.eq(v_udt & v_int).new()
    assert eq_result.dtype == dtypes.BOOL
    expected_eq = Vector.from_coo([0, 1, 2], [True, False, False])
    assert eq_result.isequal(expected_eq)

    ne_result = binary.ne(v_udt & v_int).new()
    assert ne_result.isequal(Vector.from_coo([0, 1, 2], [False, True, True]))

    # Reverse direction (scalar on left).
    eq_rev = binary.eq(v_int & v_udt).new()
    assert eq_rev.isequal(expected_eq)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast_nan_propagates():
    """A NaN leaf never equals anything, even another NaN."""
    # Distinct field names from the sibling record test: ``register_anonymous``
    # keys on ``np.dtype``, so reusing ``("u", "v")`` would alias the cached
    # DataType across tests (see CLAUDE.md).
    record = dtypes.register_anonymous(
        np.dtype([("nan_u", np.float64), ("nan_v", np.float64)], align=True),
        name="_EqBcastUVNan",
    )
    v_nan = Vector(record, size=3)
    v_nan[0] = (np.nan, 5.0)
    v_nan[1] = (5.0, 5.0)
    v_nan[2] = (np.nan, np.nan)
    v_five = Vector.from_coo([0, 1, 2], [5.0, 5.0, 5.0])
    eq_nan = binary.eq(v_nan & v_five).new()
    assert eq_nan.isequal(Vector.from_coo([0, 1, 2], [False, True, False]))


@pytest.mark.skipif("not supports_udfs")
def test_udt_eq_ne_type_a_literal_as_builtin_comparisons_do():
    """``eq`` and ``ne`` beside a UDT type a Python number as a built-in comparison does.

    The number is weak, leaf by leaf, so ``fp32_udt == 0.1`` compares in
    float32 and is True, as ``fp32_vec == 0.1`` and numpy are; it was strong,
    and compared float32 0.1 with float64 0.1. A comparison keeps no result
    type, so a literal with an int out of a leaf's range equals no element,
    even past int64, where it raised. Beside an int64 leaf, it is not rounded
    to equal one (in float64, the int64 ``2**63 - 1`` and ``2**63`` are equal).
    numpy scalars stay strong.
    """
    f32 = dtypes.register_anonymous(np.dtype((np.float32, (7,))), "_EqWeakF32")
    i8 = dtypes.register_anonymous(np.dtype((np.int8, (7,))), "_EqWeakI8")
    v = Vector(f32, size=1)
    v[0] = np.full(7, 0.1)
    assert (v == 0.1).new()[0].new().value
    assert not (v != 0.1).new()[0].new().value
    assert (v == (0.1,) * 7).new()[0].new().value
    assert not (v == np.float64(0.1)).new()[0].new().value
    builtin = Vector.from_coo([0], [0.1], dtype=dtypes.FP32)
    assert (builtin == 0.1).new()[0].new().value
    w = Vector(i8, size=1)
    w[0] = np.arange(7)
    for number in [300, -(2**63) - 1, 2**63, 2**64]:
        assert not (w == number).new()[0].new().value
        assert (w != number).new()[0].new().value
    assert (w == tuple(range(7))).new()[0].new().value
    assert not (w == (0.5, *range(1, 7))).new()[0].new().value
    i64 = dtypes.register_anonymous(np.dtype((np.int64, (2,))), "_EqWeakI64")
    x = Vector(i64, size=2)
    x[0] = [2**63 - 1, 2**63 - 1]
    x[1] = [-(2**63), 0]
    for literal in [2**63, (2**63, 2**63), [2**63 - 1, 2**63], -(2**63) - 1, 10**400]:
        assert (x == literal).new().to_coo()[1].tolist() == [False, False]
        assert (x != literal).new().to_coo()[1].tolist() == [True, True]
    assert (x == (2**63 - 1, 2**63 - 1)).new().to_coo()[1].tolist() == [True, False]
    rec = dtypes.register_anonymous(
        np.dtype([("a", np.int64), ("b", np.uint64)], align=True), "_EqWeakRec64"
    )
    r = Vector(rec, size=1)
    r[0] = (2**63 - 1, 2**64 - 1)
    for literal in [(2**63, 2**64 - 1), {"a": 2**63 - 1, "b": 2**64}, (2**63 - 1, -1)]:
        assert not (r == literal).new()[0].new().value
        assert (r != literal).new()[0].new().value
    assert (r == {"a": 2**63 - 1, "b": 2**64 - 1}).new()[0].new().value


def test_udt_float32_overflow_warns_at_the_callers_line():
    """A float too large for a float32 leaf is infinity, with a warning at the caller's line.

    numpy warned from inside python-graphblas, so the default filter showed it
    once per process; a built-in FP32 warns at the caller's line, as numpy does.
    """
    f32 = dtypes.register_anonymous(np.dtype((np.float32, (2,))), "_OverflowF32")
    v = Vector(f32, size=1)
    with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as record:
        v[0] = [1e300, 1]
    assert [w.filename for w in record] == [__file__]
    assert v[0].new().value.tolist() == [np.inf, 1]
    if supports_udfs:  # UDT eq and plus need Numba where there is no C JIT
        with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as record:
            (v == 1e300).new()
        assert [w.filename for w in record] == [__file__]
        with pytest.warns(RuntimeWarning, match="overflow encountered in cast") as record:
            (v + 1e300).new()
        assert [w.filename for w in record] == [__file__]
    with np.errstate(over="ignore"):
        v[0] = [1e300, 1]  # no warning, as numpy


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast_array_1d():
    arr1d = dtypes.register_anonymous(np.dtype((np.float64, (3,))), name="_EqBcastA3")
    v_a = Vector(arr1d, size=2)
    v_a[0] = [1.0, 1.0, 1.0]
    v_a[1] = [1.0, 2.0, 1.0]
    v_one = Vector.from_coo([0, 1], [1.0, 1.0])
    eq_a = binary.eq(v_a & v_one).new()
    assert eq_a.isequal(Vector.from_coo([0, 1], [True, False]))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast_array_2d():
    arr2d = dtypes.register_anonymous(np.dtype((np.float64, (2, 2))), name="_EqBcastA22")
    v_a22 = Vector(arr2d, size=2)
    v_a22[0] = [[3.0, 3.0], [3.0, 3.0]]
    v_a22[1] = [[3.0, 3.0], [4.0, 3.0]]
    v_three = Vector.from_coo([0, 1], [3.0, 3.0])
    eq_a22 = binary.eq(v_a22 & v_three).new()
    assert eq_a22.isequal(Vector.from_coo([0, 1], [True, False]))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
# 131-char numpy repr; see test_udt_eq_nested_record_with_nan_leaf.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_eq_ne_scalar_broadcast_nested_record():
    inner = np.dtype([("a", np.float64), ("b", np.float64)], align=True)
    nested = dtypes.register_anonymous(
        np.dtype([("outer", np.float64), ("inner", inner)], align=True),
        name="_EqBcastNest",
    )
    v_n = Vector(nested, size=3)
    v_n[0] = (5.0, (5.0, 5.0))
    v_n[1] = (5.0, (5.0, 6.0))
    v_n[2] = (1.0, (1.0, 1.0))
    v_5 = Vector.from_coo([0, 1, 2], [5.0, 5.0, 5.0])
    eq_n = binary.eq(v_n & v_5).new()
    assert eq_n.isequal(Vector.from_coo([0, 1, 2], [True, False, False]))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast_record_with_array_subfield():
    rec_arr = dtypes.register_anonymous(
        np.dtype([("a", np.float64), ("v", np.float64, (3,))], align=True),
        name="_EqBcastRecArr",
    )
    v_ra = Vector(rec_arr, size=2)
    v_ra[0] = (7.0, [7.0, 7.0, 7.0])
    v_ra[1] = (7.0, [7.0, 8.0, 7.0])
    v_7 = Vector.from_coo([0, 1], [7.0, 7.0])
    eq_ra = binary.eq(v_ra & v_7).new()
    assert eq_ra.isequal(Vector.from_coo([0, 1], [True, False]))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_rejects_incompatible_pairs():
    """eq/ne between two UDTs must reject mismatched structure rather than
    silently byte-compare.

    The old code only consulted the first dtype when generating the leaf
    chain, so two records with different field names (but identical
    byte layout) compared as equal, and a record-vs-array pair compared
    by reinterpreting one side as the other.
    """
    uv = dtypes.register_anonymous(
        np.dtype([("u", np.float64), ("v", np.float64)], align=True),
        name="_EqRejUV",
    )
    uw = dtypes.register_anonymous(
        np.dtype([("u", np.float64), ("w", np.float64)], align=True),
        name="_EqRejUW",
    )
    arr = dtypes.register_anonymous(np.dtype((np.float64, (2,))), name="_EqRejA")

    v_uv = Vector(uv, size=1)
    v_uv[0] = (1.0, 2.0)
    v_uw = Vector(uw, size=1)
    v_uw[0] = (1.0, 2.0)
    v_arr = Vector(arr, size=1)
    v_arr[0] = [1.0, 2.0]

    with pytest.raises(KeyError, match="record UDTs must share field names"):
        binary.eq(v_uv & v_uw).new()
    with pytest.raises(KeyError, match="record UDTs must share field names"):
        binary.ne(v_uv & v_uw).new()
    with pytest.raises(KeyError, match="cannot mix record and array UDTs"):
        binary.eq(v_uv & v_arr).new()


@pytest.mark.skipif("not supports_udfs")
# SS < 9 has no GrB_NAME setter, so registration falls back to storing the
# numpy repr in the type name and warns when it does not fit in 128 chars.
# Each nested record's repr here is 141 or 142; how they serialize is not what
# the test is about.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
@pytest.mark.parametrize("op", ["plus", "eq"])
@pytest.mark.parametrize("case", ["leaf_count", "nesting", "inner_names", "inner_order"])
def test_udt_record_nesting_mismatch_is_a_keyerror(case, op):
    """Records sharing top-level field names but nesting differently are a KeyError.

    ``_check_udt_pair`` matched on top-level names only, but the codegen reads
    both operands at the left one's leaf paths. A field that is a sub-record on
    one side and a scalar on the other, or sub-records with different names,
    reached Numba, whose typing failure was a ``UdfParseError`` for ``plus``
    and a raw ``TypingError`` for ``eq``: compile errors for what is really the
    shape disagreement its sibling checks raise ``KeyError`` for. Sub-record
    fields in a different order are rejected as top-level ones are, so that
    leaf ``i`` of one operand is leaf ``i`` of the other.
    """
    f8 = np.float64
    sub = [("nst_n1", f8), ("nst_n2", f8)]
    left, right = {
        # A scalar field on the left is a sub-record on the right.
        "leaf_count": ([("nst_a", f8), ("nst_b", f8)], [("nst_a", sub), ("nst_b", f8)]),
        # Three leaves each, but nested under different fields.
        "nesting": ([("nst_a", sub), ("nst_b", f8)], [("nst_a", f8), ("nst_b", sub)]),
        "inner_names": (
            [("nst_a", sub), ("nst_b", f8)],
            [("nst_a", [("nst_m1", f8), ("nst_m2", f8)]), ("nst_b", f8)],
        ),
        "inner_order": ([("nst_a", sub), ("nst_b", f8)], [("nst_a", sub[::-1]), ("nst_b", f8)]),
    }[case]
    v = Vector(dtypes.register_anonymous(np.dtype(left, align=True), "_NestLeft"), size=1)
    w = Vector(dtypes.register_anonymous(np.dtype(right, align=True), "_NestRight"), size=1)
    with pytest.raises(KeyError, match="must nest the same way, with the same field names"):
        v.ewise_mult(w, getattr(binary, op)).new()


def _record_with_field_shape(shape, name, b_dtype=np.float64):
    """Register a record UDT whose ``fsh_a`` field has ``shape`` (a scalar for ``()``)."""
    field = ("fsh_a", np.float64, shape) if shape else ("fsh_a", np.float64)
    return dtypes.register_anonymous(np.dtype([field, ("fsh_b", b_dtype)]), name)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.parametrize("op", ["plus", "eq"])
@pytest.mark.parametrize(
    ("left", "right"),
    [((3,), (4,)), ((2, 3), (3, 2)), ((3,), ()), ((3,), (1,)), ((3,), (1, 3))],
)
def test_udt_record_field_shape_mismatch_is_a_keyerror(left, right, op):
    """Record fields whose shapes differ are a KeyError, whichever operand is on the left.

    The codegen combines array fields as numpy arrays, so a ``(3,)`` field with
    a ``(4,)`` one raised inside the cfunc, where the caller never saw it, and
    the result element kept whatever was in the buffer (``eq`` read it as its
    bool). Shapes that broadcast worked only one way round: ``plus`` writes
    into a field shaped like the left operand's, so a ``(3,)`` field took a
    ``(1,)`` one on its right but failed with it on its left, and a scalar
    field on the left of an array one never compiled. numpy will not promote
    or compare two record dtypes whose fields differ in shape either.
    """
    X = _record_with_field_shape(left, "_FshLeft")
    Y = _record_with_field_shape(right, "_FshRight")
    for a, b in [(X, Y), (Y, X)]:
        v = Vector(a, size=1)
        w = Vector(b, size=1)
        with pytest.raises(KeyError, match="must have the same shape for each field"):
            v.ewise_mult(w, getattr(binary, op)).new()


@pytest.mark.skipif("not supports_udfs")
def test_udt_record_fields_of_equal_shape_combine():
    """Two record UDTs with the same field shapes combine, though a field's dtype differs.

    ``eq`` and ``ne`` used to fail on this pair with GrB_DOMAIN_MISMATCH:
    ``get_typed_op`` unified the two records to one of them, since numpy
    promotes their field dtypes, so GraphBLAS saw an operand of the other
    type. They now compile the pair, as the arithmetic ops do.
    """
    X = _record_with_field_shape((2, 3), "_FshLeft")
    v = Vector(X, size=2)
    v[0] = (np.arange(6.0).reshape(2, 3), 1.0)
    v[1] = (np.ones((2, 3)), 2.0)
    w = Vector(_record_with_field_shape((2, 3), "_FshRight", b_dtype=np.float32), size=2)
    w[0] = (np.full((2, 3), 10.0), 0.5)
    w[1] = (np.ones((2, 3)), 2.0)
    total = v.ewise_mult(w, binary.plus).new()
    assert total.dtype == X
    value = total[0].new().value
    np.testing.assert_array_equal(value["fsh_a"], np.arange(6.0).reshape(2, 3) + 10.0)
    assert value["fsh_b"] == 1.5
    assert binary.eq(v & w).new().isequal(Vector.from_coo([0, 1], [False, True]))
    assert binary.ne(v & w).new().isequal(Vector.from_coo([0, 1], [True, False]))
    # A semiring dispatches through its BinaryOp, so ``lor_eq`` takes the pair too.
    assert v.inner(w, semiring.lor_eq).new().value
    assert not v.inner(w, semiring.land_eq).new().value


@pytest.mark.skipif("not supports_udfs")
def test_udt_positional_ops_take_two_different_udts():
    """``first``, ``second``, ``any`` and ``pair`` take any two UDTs; they never read a field.

    ``any`` was the odd one out: it had no ``_custom_dtype``, so ``get_typed_op``
    unified two records of the same layout to one of them and GraphBLAS raised
    GrB_DOMAIN_MISMATCH on the other operand. ``get_typed_op`` now never unifies
    two different UDTs.
    """
    # Own field names: the helper's scalar-field record is ``_FshRight``'s dtype,
    # and ``register_anonymous`` returns one cached DataType per dtype.
    X = dtypes.register_anonymous(
        np.dtype([("pos_a", np.float64), ("pos_b", np.float64)]), "_PosLeft"
    )
    Y = dtypes.register_anonymous(
        np.dtype([("pos_a", np.float64), ("pos_b", np.int64)]), "_PosRight"
    )
    v = Vector(X, size=1)
    v[0] = (1.0, 3.0)
    w = Vector(Y, size=1)
    w[0] = (4.0, 6)
    assert binary.first(v & w).new().isequal(v)
    assert binary.second(v & w).new().isequal(w)
    assert binary.any(v & w).new().isequal(w)
    assert binary.pair(v & w).new().isequal(Vector.from_coo([0], [1]))


@pytest.mark.skipif("not supports_udfs")
def test_udt_record_pair_promotes_each_field():
    """Two records whose field dtypes differ give each field the dtype the built-in op gives.

    The result was always the left record, so ``x + y`` truncated an int64 field
    that ``y + x`` kept as float64, and a complex field on the right of ``min``
    slipped past the check that rejects complex ``min`` and failed in Numba.
    When one operand's record holds every promoted field, the result is that
    record, on either side; otherwise it is the record with the promoted
    fields, which is the UDT registered with that layout if there is one. Each
    record also has an array field, which no Numba return type can be matched
    against, so the result type cannot be recovered from what the generated
    function returns.
    """

    def record(a_dtype, b_dtype, name):
        fields = [("wid_a", a_dtype), ("wid_b", b_dtype), ("wid_v", np.float64, (2,))]
        return dtypes.register_anonymous(np.dtype(fields), name)

    f8 = np.float64
    narrow = record(np.int64, f8, "_WidN")
    wide = record(f8, f8, "_WidW")
    crossed = record(f8, np.int64, "_WidX")
    cplx = record(np.complex128, f8, "_WidC")
    v = Vector(narrow, size=1)
    v[0] = (1, 2.0, [1.0, 2.0])
    w = Vector(wide, size=1)
    w[0] = (0.5, 0.25, [10.0, 20.0])
    for a, b in [(v, w), (w, v)]:
        result = binary.plus(a & b).new()
        assert result.dtype == wide
        value = result[0].new().value
        assert (value["wid_a"], value["wid_b"]) == (1.5, 2.25)
        np.testing.assert_array_equal(value["wid_v"], [11.0, 22.0])
    # Each of ``narrow`` and ``crossed`` is wider in one field, so neither holds
    # both. The promoted record has ``wide``'s layout, so it is ``wide``.
    x = Vector(crossed, size=1)
    x[0] = (1.0, 2, [1.0, 2.0])
    for a, b in [(v, x), (x, v)]:
        result = binary.times(a & b).new()
        assert result.dtype == wide
        value = result[0].new().value
        assert (value["wid_a"], value["wid_b"]) == (1.0, 4.0)
        np.testing.assert_array_equal(value["wid_v"], [1.0, 4.0])
    assert binary.eq(v & x).new().isequal(Vector.from_coo([0], [True]))
    c = Vector(cplx, size=1)
    c[0] = (1 + 1j, 2.0, [0.0, 0.0])
    for a, b in [(w, c), (c, w)]:
        assert binary.plus(a & b).new().dtype == cplx
        with pytest.raises(KeyError, match="not defined on complex fields"):
            binary.min(a & b).new()
    # The result is ``wide`` on either side. Stored in ``narrow``, its float
    # field is cast to int64 as a built-in store would cast it.
    v << binary.plus(v & w)
    assert v.dtype == narrow
    value = v[0].new().value
    assert (value["wid_a"], value["wid_b"]) == (1, 2.25)


@pytest.mark.skipif("not supports_udfs")
# SS < 9 names a UDT by its numpy repr and warns when that is over 128 chars,
# as these nested records' reprs are.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_record_pair_promotes_to_a_third_record():
    """Records that are each wider in some field give a third record, as numpy would.

    Each field is promoted as the built-in op promotes two Vectors, so int16
    with float32 is float32, and the result keeps the operands' names, nesting
    and alignment. This pair used to be a KeyError, since the result had to be
    one of the two operands.
    """
    i2, i8, f4, f8 = np.int16, np.int64, np.float32, np.float64
    X = dtypes.register_anonymous(
        np.dtype([("pro_a", [("pro_i", i2), ("pro_j", f8)]), ("pro_b", f4)], align=True), "_ProX"
    )
    Y = dtypes.register_anonymous(
        np.dtype([("pro_a", [("pro_i", f4), ("pro_j", i8)]), ("pro_b", i2)], align=True), "_ProY"
    )
    expected = np.dtype([("pro_a", [("pro_i", f4), ("pro_j", f8)]), ("pro_b", f4)], align=True)
    v = Vector(X, size=1)
    v[0] = ((3, 0.5), 1.5)
    w = Vector(Y, size=1)
    w[0] = ((0.25, 4), 2)
    for a, b in [(v, w), (w, v)]:
        result = binary.plus(a & b).new()
        assert result.dtype.np_type == expected
        assert result[0].new().value.tolist() == ((3.25, 4.5), 3.5)


@pytest.mark.skipif("not supports_udfs")
def test_udt_record_scalar_promotes_each_field():
    """A number beside a record UDT keeps its dtype, and each field promotes as a Vector's would.

    The number used to be converted to the record first, so ``int_record + 0.5``
    added 0 to each field.
    """
    ints = dtypes.register_anonymous(
        np.dtype([("rsp_x", np.int64), ("rsp_y", np.int64)]), "_RspInts"
    )
    floats = dtypes.register_anonymous(
        np.dtype([("rsp_f", np.float64), ("rsp_g", np.float64)]), "_RspFloats"
    )
    promoted = np.dtype([("rsp_x", np.float64), ("rsp_y", np.float64)])
    v = Vector(ints, size=1)
    v[0] = (2, 3)
    for expr, expected in [
        (v + 0.5, (2.5, 3.5)),
        (1.5 * v, (3.0, 4.5)),
        (v.apply(binary.truediv, right=4), (0.5, 0.75)),
    ]:
        result = expr.new()
        assert result.dtype.np_type == promoted
        assert result[0].new().value.tolist() == expected
    assert (v + 1).new().dtype == ints
    w = Vector(floats, size=1)
    w[0] = (0.5, 1.5)
    result = (w + 1).new()
    assert result.dtype == floats
    assert result[0].new().value.tolist() == (1.5, 2.5)


@pytest.mark.skipif("not supports_udfs")
def test_udt_packed_record_with_aligned_takes_an_operand_type():
    """A packed and an aligned record with the same leaves give the left operand's type.

    Their fields are the same dtypes at different offsets, so neither holds the
    other's layout, but each holds every result field. Taking the left one lets
    ``x << x + y`` update ``x`` in place, whichever of the two ``x`` is.
    """
    fields = [("pak_a", np.int8), ("pak_b", np.float64)]
    aligned = dtypes.register_anonymous(np.dtype(fields, align=True), "_PakAligned")
    packed = dtypes.register_anonymous(np.dtype(fields), "_PakPacked")
    assert aligned.np_type.itemsize != packed.np_type.itemsize
    x = Vector(aligned, size=1)
    x[0] = (1, 0.5)
    y = Vector(packed, size=1)
    y[0] = (2, 0.25)
    for a, b in [(x, y), (y, x)]:
        assert binary.plus(a & b).new().dtype == a.dtype
        a << binary.plus(a & b)
    assert x[0].new().value.tolist() == (3, 0.75)
    assert y[0].new().value.tolist() == (5, 1.0)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.parametrize("op", ["plus", "eq"])
@pytest.mark.parametrize(("left", "right"), [((2, 3), (3, 2)), ((6,), (2, 3)), ((3,), (4,))])
def test_udt_array_pair_shape_mismatch_is_a_keyerror(left, right, op):
    """Array UDTs whose shapes do not broadcast are a KeyError, as numpy refuses them.

    Two of these with the same size used to combine over their flat buffers,
    so ``(2, 3)`` plus ``(3, 2)`` added elements numpy never pairs. The base
    dtypes differ, which is allowed, so only the shapes are at fault.
    """
    # int16 and int32: other tests register float64 arrays of these shapes, and
    # ``register_anonymous`` returns one cached DataType per dtype.
    X = dtypes.register_anonymous(np.dtype((np.int16, left)), "_PairArrL")
    Y = dtypes.register_anonymous(np.dtype((np.int32, right)), "_PairArrR")
    for a, b in [(X, Y), (Y, X)]:
        with pytest.raises(KeyError, match="shapes that broadcast together"):
            getattr(binary, op)(Vector(a, size=1) & Vector(b, size=1)).new()


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.parametrize(
    ("x_type", "y_type"),
    [
        ((np.uint16, (3, 1)), (np.int32, (1, 4))),
        ((np.uint16, (2, 1, 3)), (np.float32, (4, 1))),
        ((np.uint16, (1,)), (np.int16, (2, 2))),
        ((np.int16, (3, 1)), (np.int16, (2, 1, 4))),
    ],
)
def test_udt_array_pairs_broadcast_as_numpy(x_type, y_type):
    """Two array UDTs broadcast as numpy arrays do, in either order.

    The result has numpy's broadcast shape and the promoted element dtype. It
    is an operand's type when one has both, else the structural array type.
    ``eq`` is True when every pair of elements numpy compares is equal.
    """
    X = dtypes.register_anonymous(np.dtype(x_type), "_BcastX")
    Y = dtypes.register_anonymous(np.dtype(y_type), "_BcastY")
    xv = np.arange(1, 1 + np.prod(x_type[1])).reshape(x_type[1]).astype(x_type[0])
    yv = 10 * np.arange(1, 1 + np.prod(y_type[1])).reshape(y_type[1]).astype(y_type[0])
    v = Vector(X, size=3)
    v[0] = xv
    v[1] = xv
    w = Vector(Y, size=3)
    w[0] = yv
    w[2] = yv
    M = Matrix(X, nrows=1, ncols=1)
    M[0, 0] = xv
    N = Matrix(Y, nrows=1, ncols=1)
    N[0, 0] = yv
    for a, b, A, B, av, bv in [(v, w, M, N, xv, yv), (w, v, N, M, yv, xv)]:
        expected = np.subtract(av, bv)
        result_type = dtypes.lookup_dtype(np.dtype((expected.dtype, expected.shape)))
        for result in [binary.minus(a & b).new(), binary.minus(A & B).new()]:
            assert result.dtype == result_type
            assert result.nvals == 1
            value = result[0].new().value if result.ndim == 1 else result[0, 0].new().value
            np.testing.assert_array_equal(value, expected)
        assert not binary.eq(a & b).new()[0].new().value
        assert binary.ne(a & b).new()[0].new().value
        # ewise_add would copy an entry without a partner as the result type, a
        # cast GraphBLAS cannot make for a UDT; ewise_union computes it instead,
        # so it broadcasts to the result's shape.
        with pytest.raises(DomainMismatch, match="ewise_add cannot use binary.minus"):
            a.ewise_add(b, binary.minus)
        added = a.ewise_union(b, binary.minus, 0, 0).new()
        assert added.dtype == result_type
        np.testing.assert_array_equal(added[0].new().value, expected)
        lone = 1 if a is v else 2
        np.testing.assert_array_equal(added[lone].new().value, np.broadcast_to(av, expected.shape))
    # Equal values: every compared pair is equal, as ``(xv == yv).all()``.
    w[0] = np.broadcast_to(xv[(0,) * xv.ndim], y_type[1])
    v[0] = np.broadcast_to(xv[(0,) * xv.ndim], x_type[1])
    assert binary.eq(v & w).new()[0].new().value
    assert binary.eq(w & v).new()[0].new().value
    assert not binary.ne(v & w).new()[0].new().value


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_broadcast_semiring_and_reduce():
    """A semiring on two array UDTs that broadcast sums the broadcast products."""
    X = dtypes.register_anonymous(np.dtype((np.int16, (3, 1))), "_BcastSemiX")
    Y = dtypes.register_anonymous(np.dtype((np.int16, (1, 5))), "_BcastSemiY")
    a = [np.arange(3).reshape(3, 1) + k for k in range(2)]
    b = [10 * np.arange(5).reshape(1, 5) - k for k in range(2)]
    A = Matrix(X, nrows=1, ncols=2)
    A[0, 0] = a[0]
    A[0, 1] = a[1]
    B = Matrix(Y, nrows=2, ncols=1)
    B[0, 0] = b[0]
    B[1, 0] = b[1]
    C = A.mxm(B, semiring.plus_times).new()
    assert C.dtype.np_type == np.dtype((np.int16, (3, 5)))
    expected = a[0] * b[0] + a[1] * b[1]
    np.testing.assert_array_equal(C[0, 0].new().value, expected)
    s = A.reduce_scalar(monoid.plus).new()
    np.testing.assert_array_equal(s.value, a[0] + a[1])


@pytest.mark.skipif("not supports_udfs")
def test_udt_record_2d_field_compares_with_a_number():
    """``eq`` and ``ne`` of a record with a 2-D array field and a number compare every element.

    The generated code read the field at flat positions, which in a 2-D field
    are rows, so it failed to compile once a number was compared as it is
    rather than converted into the record first.
    """
    T = dtypes.register_anonymous(
        np.dtype([("cf_grid", np.float64, (2, 3)), ("cf_n", np.int64)], align=True), "_CmpField2D"
    )
    v = Vector(T, size=2)
    v[0] = (np.full((2, 3), 2.0), 2)
    v[1] = (np.array([[2.0, 2, 2], [2, 2, np.nan]]), 2)
    assert binary.eq(v, 2.0).new().isequal(Vector.from_coo([0, 1], [True, False]))
    assert binary.ne(2.0, v).new().isequal(Vector.from_coo([0, 1], [False, True]))


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_ops_on_large_elements():
    """Lifted ops on a 64 by 64 array compile in a loop, not one statement per element.

    Unrolled, Numba took about two minutes to type ``minus`` on this element.
    """
    try:
        T = dtypes.register_anonymous(np.dtype((np.float64, (64, 64))), "_LoopBig")
    except InvalidValue:
        if sys.platform != "win32":
            raise
        # SuiteSparse:GraphBLAS built by MSVC (no variable-length arrays) refuses a
        # UDT larger than 1024 bytes (GB_VLA_MAXSIZE); this one is 32768.
        pytest.skip("SuiteSparse:GraphBLAS limits a UDT to 1024 bytes on Windows")
    S = dtypes.register_anonymous(np.dtype((np.float64, (64, 1))), "_LoopBigCol")
    x = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    col = np.arange(64, dtype=np.float64).reshape(64, 1)
    v = Vector(T, size=1)
    v[0] = x
    u = Vector(S, size=1)
    u[0] = col
    np.testing.assert_array_equal((v - u).new()[0].new().value, x - col)
    np.testing.assert_array_equal(unary.ainv(v).new()[0].new().value, -x)
    assert binary.eq(v & v).new()[0].new().value
    assert not binary.ne(v & v).new()[0].new().value
    w = v.dup()
    w[0] = np.where(x == 4095, np.nan, x)
    assert not binary.eq(w & w).new()[0].new().value
    assert binary.ne(w & w).new()[0].new().value


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_pair_leading_ones():
    """Array UDTs that differ only in leading axes of length 1 combine, as numpy broadcasts them.

    This is what a ``Matrix.to_csr`` round trip produces: values of shape
    ``(1, 3)`` for a ``(3,)`` UDT. The result takes the operand with more axes,
    numpy's broadcast shape, whichever side it is on.
    """
    short = dtypes.register_anonymous(np.dtype((np.int16, (3,))), "_PairArrShort")
    long = dtypes.register_anonymous(np.dtype((np.int16, (1, 3))), "_PairArrLong")
    v = Vector(short, size=1)
    v[0] = [1, 2, 3]
    w = Vector(long, size=1)
    w[0] = [[10, 20, 30]]
    for a, b in [(v, w), (w, v)]:
        result = binary.plus(a & b).new()
        assert result.dtype == long
        np.testing.assert_array_equal(result[0].new().value, [[11, 22, 33]])
        assert not binary.eq(a & b).new()[0].new().value


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_pair_promotes_the_base():
    """Array UDTs with different base dtypes combine, each element promoted as the built-in op does.

    The pair used to be a KeyError. The result is the operand whose type holds
    the promoted elements, on either side, so ``int64`` with ``float64`` gives
    the ``float64`` one. ``eq`` and ``ne`` must read each operand as its own
    dtype: reading both as the left one's compares the bits of a float64 as an
    int64.
    """
    ints = dtypes.register_anonymous(np.dtype((np.int64, (2, 2, 2))), "_PromI222")
    floats = dtypes.register_anonymous(np.dtype((np.float64, (2, 2, 2))), "_PromF222")
    base = np.arange(8).reshape(2, 2, 2)
    v = Vector(ints, size=2)
    v[0] = base
    v[1] = base
    w = Vector(floats, size=2)
    w[0] = base + 0.5
    w[1] = base
    for a, b in [(v, w), (w, v)]:
        result = binary.plus(a & b).new()
        assert result.dtype == floats
        np.testing.assert_array_equal(result[0].new().value, 2 * base + 0.5)
        assert binary.eq(a & b).new().isequal(Vector.from_coo([0, 1], [False, True]))
        assert binary.ne(a & b).new().isequal(Vector.from_coo([0, 1], [True, False]))
    # A semiring's monoid is typed by its multiplier's result.
    assert v.inner(w, semiring.plus_times).new().dtype == floats
    # GraphBLAS would cast an entry of ``v`` alone to the result type, which a
    # UDT cannot be; ewise_union computes that entry with the op instead, and
    # infix ``+`` uses it, with 0 for the missing value.
    with pytest.raises(DomainMismatch, match="ewise_add cannot use binary.plus"):
        v.ewise_add(w)
    assert v.ewise_union(w, binary.plus, 0, 0.0).new().dtype == floats
    assert (v + w).new().isequal(v.ewise_union(w, binary.plus, 0, 0.0).new())


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_scalar_promotes_like_builtin():
    """A number beside an array UDT keeps its dtype, and each element promotes as the op does.

    ``apply`` converted a Python number to the UDT first (``v + 0.5`` is an
    ``apply``), and the result took the UDT's base whatever the other operand
    was, so ``int_udt + 0.5`` added 0 and ``int_udt / 4`` gave integers. Each
    element now has the dtype the op gives on the built-in dtypes, and the
    result is the operand's UDT only when that holds it.
    """
    ints = dtypes.register_anonymous(np.dtype((np.int64, (2, 4))), "_UpcI24")
    int8s = dtypes.register_anonymous(np.dtype((np.int8, (2, 4))), "_UpcB24")
    fp32s = dtypes.register_anonymous(np.dtype((np.float32, (2, 4))), "_UpcF24")
    fp64s = np.dtype((np.float64, (2, 4)))
    base = np.arange(8).reshape(2, 4)
    v = Vector(ints, size=1)
    v[0] = base
    M = Matrix(ints, nrows=1, ncols=1)
    M[0, 0] = base
    for expr, expected in [
        (v + 0.5, base + 0.5),
        (0.5 + v, base + 0.5),
        (v.apply(binary.times, right=2.5), base * 2.5),
        (v.apply(binary.truediv, right=4), base / 4),
        (binary.minus(10.5, v), 10.5 - base),
        (v.apply(binary.plus, right=gb.Scalar.from_value(0.5)), base + 0.5),
        (v.ewise_mult(Vector.from_coo([0], [0.5]), binary.times), base * 0.5),
        (M * 0.5, base * 0.5),
        (0.5 * M, base * 0.5),
        # A 0-d numpy array is a number too, not a whole element.
        (v + np.array(0.5, np.float32), base + 0.5),
        (v / np.array(4, np.int8), base / 4),
    ]:
        result = expr.new()
        assert result.dtype.np_type == fp64s
        value = result[0].new().value if result.ndim == 1 else result[0, 0].new().value
        np.testing.assert_array_equal(value, expected)
    # Same element type: the result keeps the UDT, as does a literal that is a
    # whole element rather than a number.
    for expr in [v + 1, v + [[1] * 4] * 2]:
        result = expr.new()
        assert result.dtype == ints
        np.testing.assert_array_equal(result[0].new().value, base + 1)
    # A Python number is weak, as in numpy 2: it takes the elements' dtype when
    # their kind holds it, so INT8 elements plus 1 stay INT8 and FP32 elements
    # plus 0.5 stay FP32, and an in-place update can store the result. A typed
    # number is strong, and an int out of the elements' range is an error.
    w = Vector(int8s, size=1)
    w[0] = base
    assert (w + 1).new().dtype == int8s
    w += 1
    np.testing.assert_array_equal(w[0].new().value, base + 1)
    assert (w + np.int64(1)).new().dtype == ints  # the UDT registered with that layout
    with pytest.raises(OverflowError, match="300 out of bounds for int8"):
        w + 300
    # truediv gives float elements whatever the int, as numpy divides in float64.
    result = (w / 300).new()
    assert result.dtype.np_type.base == np.float64
    np.testing.assert_array_equal(result[0].new().value, (base + 1) / 300)
    x = Vector(fp32s, size=1)
    x[0] = base
    result = (x + 0.5).new()
    assert result.dtype == fp32s
    np.testing.assert_array_equal(result[0].new().value, base + 0.5)
    x *= 2
    np.testing.assert_array_equal(x[0].new().value, base * 2)
    assert (x + np.float64(0.5)).new().dtype.np_type == fp64s
    if dtypes._supports_complex:
        for expr in [lambda: binary.min(x, 1j), lambda: binary.min(1j, x)]:
            with pytest.raises(KeyError, match="not defined on complex fields"):
                expr()
    # eq and ne compare with the number itself; converted to the UDT, 0.5 was 0.
    zeros = Vector(ints, size=1)
    zeros[0] = np.zeros((2, 4))
    assert not (zeros == 0.5).new()[0].new().value
    assert (zeros != 0.5).new()[0].new().value
    assert (zeros == 0).new()[0].new().value


@pytest.mark.skipif("not supports_udfs")
def test_udt_uint64_with_signed_ints_computes_like_builtin():
    """UINT64 elements with signed ones are computed in FP64, as the built-in ops compute them.

    The result type was right, but the generated code let Numba do the
    arithmetic, and Numba computes uint64 with int64 in int64: ``2**63 + 1``
    came out as ``-9.22e18`` and ``2**63 * 2`` as 0. Each operand is now
    converted to the type the built-in op computes in first.
    """
    big = 2**63
    uints = dtypes.register_anonymous(np.dtype((np.uint64, (2, 2))), "_MixU22")
    ints = dtypes.register_anonymous(np.dtype((np.int64, (2, 2))), "_MixI22")
    u = Vector(uints, size=1)
    u[0] = [[big, 7], [0, 1]]
    i = Vector(ints, size=1)
    i[0] = [[1, -1], [0, 0]]
    for expr, expected in [
        (binary.plus(u & i), [[big + 1.0, 6.0], [0.0, 1.0]]),
        (binary.plus(i & u), [[big + 1.0, 6.0], [0.0, 1.0]]),
        (binary.floordiv(i & u), [[0.0, -1.0], [np.nan, 0.0]]),
        # A numpy int64 is strong, so it is a signed operand like ``i``.
        (u + np.int64(1), [[big + 1.0, 8.0], [1.0, 2.0]]),
        (binary.plus(np.int64(1), u), [[big + 1.0, 8.0], [1.0, 2.0]]),
        (u * np.int64(2), [[2.0 * big, 14.0], [0.0, 2.0]]),
    ]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype((np.float64, (2, 2)))
        np.testing.assert_array_equal(result[0].new().value, expected)
    # A Python int is weak, so it takes the uint64 elements' type, exactly.
    result = (u + 1).new()
    assert result.dtype == uints
    np.testing.assert_array_equal(result[0].new().value, [[big + 1, 8], [1, 2]])
    # BOOL with UINT64 is UINT64, so True // 2**63 is 0, not -1 through int64.
    result = binary.floordiv(True, u).new()
    assert result.dtype == uints
    np.testing.assert_array_equal(result[0].new().value[0], [0, 0])
    record = dtypes.register_anonymous(
        np.dtype([("mxs_u", np.uint64), ("mxs_f", np.float64)]), "_MixRec"
    )
    other = dtypes.register_anonymous(
        np.dtype([("mxs_u", np.int64), ("mxs_f", np.float64)]), "_MixRecI"
    )
    r = Vector(record, size=1)
    r[0] = (big, 1.5)
    s = Vector(other, size=1)
    s[0] = (-1, 0.5)
    assert (r * np.int64(2)).new()[0].new().value.tolist() == (2.0 * big, 3.0)
    assert binary.times(np.int64(2), r).new()[0].new().value.tolist() == (2.0 * big, 3.0)
    # A weak 2 keeps the uint64 field, which wraps as numpy 2's uint64 does.
    assert (r * 2).new()[0].new().value.tolist() == (0, 3.0)
    assert binary.plus(r & s).new()[0].new().value.tolist() == (big - 1.0, 2.0)
    assert binary.floordiv(s & r).new()[0].new().value.tolist() == (-1.0, 0.0)


@pytest.mark.skipif("not supports_udfs")
def test_udt_user_op_named_like_a_lifted_op():
    """A UDF registered under a lifted op's name runs its own function, and literals become UDTs.

    ``_compile_udt`` matched built-in ops by name only, so a UDF registered as
    ``"plus"`` or ``"max"`` was compiled as the built-in, its function ignored,
    and on a built-in dtype it raised KeyError. A literal beside a UDT is
    converted to the UDT for it, since nothing says how a UDF combines a scalar
    with an element, but only when its type fits: ``0.5`` became an element of
    zeros.
    """
    udt = dtypes.register_anonymous(np.dtype((np.int64, (2, 3))), "_NamedLikeLifted")

    def scaled_plus(x, y):
        return x * 10 + y

    def difference(x, y):
        return x - y

    plus_like = BinaryOp.register_anonymous(scaled_plus, "plus", is_udt=True)
    max_like = BinaryOp.register_anonymous(difference, "max", is_udt=True)
    v = Vector(udt, size=1)
    v[0] = np.ones((2, 3), dtype=np.int64)
    for udf_op, right, expected in [
        (plus_like, 3, 13),
        (plus_like, np.int8(2), 12),
        (max_like, 3, -2),
    ]:
        result = v.apply(udf_op, right=right).new()
        assert result.dtype == udt
        np.testing.assert_array_equal(result[0].new().value, np.full((2, 3), expected))
        assert udf_op[udt].jit_c_source is None
    # A float is not an int, whatever its value, as numpy refuses ``ints += 2.0``;
    # 2.0 still converts, being exact, but that is deprecated.
    with pytest.raises(ValueError, match="0.5 does not fit"):
        v.apply(plus_like, right=0.5)
    with pytest.warns(DeprecationWarning, match="2.0 does not fit"):
        v.apply(plus_like, right=2.0)
    assert plus_like(Vector.from_coo([0], [5]), 2).new().isequal(Vector.from_coo([0], [52]))


@pytest.mark.skipif("not supports_udfs")
def test_udt_python_number_is_weak_per_leaf():
    """A Python number beside a UDT is typed leaf by leaf, as numpy 2 types it.

    Each leaf keeps its dtype when its kind holds the number (an int beside
    int8, a float beside float32) and takes the number's default dtype when it
    does not (a float beside int8 is float64). So an in-place update with a
    number of the leaf's kind keeps the UDT, where typing every Python int as
    INT64 and every float as FP64 made ``int8_udt += 1`` and ``f32_udt *= 2``
    unstorable. numpy scalars are strong, and so are typed Scalars.
    """

    # Aligned, so the same-type ops get a JIT kernel and warn nothing.
    def record(i_dtype, f_dtype):
        return np.dtype([("wk_i", i_dtype), ("wk_f", f_dtype)], align=True)

    rec = dtypes.register_anonymous(record(np.int8, np.float32), "_WeakRec")
    v = Vector(rec, size=1)
    v[0] = (1, 2.0)
    v += 1
    v *= 2
    assert v.dtype == rec
    assert v[0].new().value.tolist() == (4, 6.0)
    result = (v + 0.5).new()
    assert result.dtype.np_type == record(np.float64, np.float32)
    assert result[0].new().value.tolist() == (4.5, 6.5)
    assert (v + np.float32(0.5)).new().dtype.np_type == record(np.float32, np.float32)
    # The result has a float64 leaf, which is cast back to int8 to be stored.
    v += 0.5
    assert v.dtype == rec
    assert v[0].new().value.tolist() == (4, 6.5)
    with pytest.raises(OverflowError, match="200 out of bounds for int8"):
        v + 200
    # A UDT Scalar follows the same rule, through infix and apply alike.
    arr = dtypes.register_anonymous(np.dtype((np.int16, (5,))), "_WeakArr")
    s = gb.Scalar(arr)
    s.value = [1, 2, 3, 4, 5]
    for expr in [s * 2.5, 2.5 * s, s.apply(binary.times, right=2.5)]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype((np.float64, (5,)))
        np.testing.assert_array_equal(result.value, np.arange(1, 6) * 2.5)
    assert (s + 1).new().dtype == arr


@pytest.mark.skipif("not supports_udfs")
def test_udt_ewise_add_needs_operands_of_the_result_type():
    """``ewise_add`` on UDT operands of another type than the result's: Scalars only.

    GraphBLAS copies an entry present in only one input into the result, cast
    to the result type, which a UDT cannot be. A Scalar is one element, so its
    operands are converted to the result type first (which changes no value,
    not even ``-0.0``). Vectors and Matrices would need converted copies of
    whole operands, so they raise, and ``ewise_union`` computes the same sum
    with the op instead.
    """
    int8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_UniPlusI8")
    fp32s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_UniPlusF32")
    f64s = dtypes.lookup_dtype(np.dtype((np.float64, (3,))))
    s = gb.Scalar.from_value([1, 2, 3], dtype=int8s)
    for expr in [s + 0.5, 0.5 + s]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype((np.float64, (3,)))
        np.testing.assert_array_equal(result.value, [1.5, 2.5, 3.5])
    result = (gb.Scalar(int8s) + 0.5).new()  # an empty Scalar adds as 0
    np.testing.assert_array_equal(result.value, [0.5, 0.5, 0.5])
    neg = gb.Scalar.from_value([-0.0, 0.0, -1.5], dtype=fp32s)
    result = (neg + gb.Scalar(f64s)).new()
    assert result.dtype == f64s
    np.testing.assert_array_equal(np.signbit(result.value), [True, False, True])
    v = Vector(int8s, size=3)
    v[0] = [1, 2, 3]
    v[1] = [4, 5, 6]
    w = Vector(fp32s, size=3)
    w[1] = [0.5, 0.5, 0.5]
    w[2] = [7, 8, 9]
    A = Matrix(int8s, nrows=1, ncols=2)
    A[0, 0] = [1, 2, 3]
    B = Matrix(fp32s, nrows=1, ncols=2)
    B[0, 1] = [0.5, 0.5, 0.5]
    for expr in [lambda: v.ewise_add(w), lambda: w.ewise_add(v), lambda: binary.plus(v | w)]:
        with pytest.raises(DomainMismatch, match="ewise_add cannot use .* converted copies"):
            expr()
    # Infix ``+`` takes the union with each operand's zero for a missing value
    # instead, as ``-`` takes 0.
    for left, right in [(v, w), (w, v)]:
        result = (left + right).new()
        assert result.dtype == fp32s
        assert result.isequal(left.ewise_union(right, binary.plus, False, False).new())
        assert result.to_coo()[1].tolist() == [[1, 2, 3], [4.5, 5.5, 6.5], [7, 8, 9]]
    result = (A + B).new()
    assert result.dtype == fp32s
    assert result.to_coo()[2].tolist() == [[1, 2, 3], [0.5, 0.5, 0.5]]
    assert (A + B.T.T).new().isequal(result)
    result = (A + Vector.from_coo([1], [[0.5, 0.5, 0.5]], dtype=fp32s, size=2)).new()
    assert result.to_coo()[2].tolist() == [[1, 2, 3], [0.5, 0.5, 0.5]]
    # That zero is -0.0 in a float field, so an entry of one operand alone keeps
    # its value, as ewise_add would copy it; beside an int field, 0 turns -0.0
    # into 0.0.
    zeros = Vector(fp32s, size=3)
    zeros[2] = [-0.0, 0.0, -1.5]
    for left, right in [(zeros, Vector(f64s, size=3)), (Vector(f64s, size=3), zeros)]:
        result = (left + right).new()
        assert result.dtype == f64s
        np.testing.assert_array_equal(np.signbit(result[2].new().value), [True, False, True])
    result = (v + zeros).new()
    np.testing.assert_array_equal(np.signbit(result[2].new().value), [False, False, True])
    x = v.dup()
    x += w  # an accumulating store: cast, then added, as for built-in types
    assert x.dtype == int8s
    assert x.to_coo()[1].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # The same type in and out still goes through ewise_add itself...
    assert (v + v).new().dtype == int8s
    # ...but not int truediv, whose result is float.
    with pytest.raises(DomainMismatch, match="binary.truediv"):
        v.ewise_add(v, binary.truediv)
    udf = BinaryOp.register_anonymous(lambda x, y: x, "_uni_first", is_udt=True)
    with pytest.raises(DomainMismatch, match="ewise_add cannot use .* on _UniPlusI8 and"):
        v.ewise_add(Vector(f64s, size=3), udf)
    # A size or broadcast error is reported first, as for a same-type pair.
    with pytest.raises(DimensionMismatch):
        v.ewise_add(Vector(fp32s, size=2), binary.plus)
    with pytest.raises(DimensionMismatch, match="Matrix.nrows"):
        v.ewise_add(B, binary.plus)
    with pytest.raises(DimensionMismatch, match="Matrix.ncols"):
        B.ewise_add(v, binary.plus)
    # ewise_union computes an entry alone with the op, so it needs no cast.
    result = v.ewise_union(w, binary.plus, 0, 0).new()
    assert result.dtype == fp32s
    assert result.to_coo()[1].tolist() == [[1, 2, 3], [4.5, 5.5, 6.5], [7, 8, 9]]
    # Accumulating it into the int8 object casts as it stores, as built-ins do.
    x = v.dup()
    x(accum=binary.plus) << x.ewise_union(w, binary.plus, 0, 0)
    assert x.dtype == int8s
    assert x.to_coo()[1].tolist() == [[2, 4, 6], [8, 10, 12], [7, 8, 9]]


@pytest.mark.skipif("not supports_udfs")
def test_udt_literal_converts_only_when_exact():
    """A literal that has to become an element of a UDT must be of that type.

    A literal beside a user-defined op, an IndexUnaryOp thunk and an
    ``ewise_union`` default are converted into the UDT, which numpy does
    silently: ``(0.5, 1.5, 2.5)`` added ``(0, 1, 2)`` to an int8 array UDT, and
    a default of ``0.5`` stood in as ``0``. The literal's type decides, as
    numpy's same_kind rule decides ``ints += x``: a Python number or sequence
    is weak, so ``1`` and ``(1, 2, 3)`` are int8 and ``0.5`` and ``2.0`` are
    float64; a numpy value is strong and must cast safely. A Monoid in
    ``apply`` is typed as its BinaryOp, so its literal promotes instead.
    """
    import re

    arr = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_ExactArr")
    rec = dtypes.register_anonymous(
        np.dtype([("ex_i", np.int16), ("ex_f", np.float32)], align=True), "_ExactRec"
    )
    v = Vector(arr, size=3)
    v[0] = [1, 2, 3]
    w = Vector(arr, size=3)
    w[1] = [4, 5, 6]
    r = Vector(rec, size=1)
    r[0] = (1, 1.5)
    udf = BinaryOp.register_anonymous(lambda x, y: x + y, "_exact_plus", is_udt=True)
    second = BinaryOp.register_anonymous(lambda x, y: y, "_exact_second", is_udt=True)
    # Literals of the UDT's type convert, a float rounding into a float32 field.
    assert v.apply(udf, right=(1, 2, 3)).new()[0].new().value.tolist() == [2, 4, 6]
    assert v.apply(udf, right=[True, 2, 3]).new().dtype == arr
    assert v.apply(udf, right=np.ones(3, dtype=np.int8)).new().dtype == arr
    assert v.apply(udf, right=np.int8(1)).new().dtype == arr
    assert r.apply(second, right=(1, 0.1)).new()[0].new().value.tolist() == (1, np.float32(0.1))
    assert r.apply(second, right={"ex_i": 2, "ex_f": 0.5}).new()[0].new().value.tolist() == (2, 0.5)
    assert r.apply(second, right=np.int8(2)).new()[0].new().value.tolist() == (2, 2.0)
    # A sequence holding numpy values is strong, as the array numpy makes of it
    # (int64, or int32 on Windows with numpy 1).
    made = np.asarray((np.int8(1), 2, 3)).dtype
    assert binary.plus(v, (np.int8(1), 2, 3)).new().dtype.np_type == np.dtype((made, (3,)))
    # The error names the literal's type; a layout's name is whatever UDT
    # registered it first, so only the built-in scalar types are pinned here.
    for expr, typed_as in [
        (lambda: v.apply(udf, right=(0.5, 1.5, 2.5)), "_ExactArr"),
        (lambda: v.apply(udf, left=np.array([0.5, 1, 1])), "_ExactArr"),
        (lambda: v.apply(udf, right=(np.float64(0.5), 1, 1)), "_ExactArr"),
        (lambda: r.apply(second, left=(0.5, 1)), "_ExactRec"),
        (lambda: v.ewise_union(w, udf, 0.5, 0), "_ExactArr"),
        (lambda: v.ewise_union(w, udf, 0, 0.5), "_ExactArr"),
        (
            lambda: gb.Scalar.from_value([1, 2, 3], arr).ewise_union(gb.Scalar(arr), udf, 0, 0.5),
            "_ExactArr",
        ),
    ]:
        with pytest.raises(ValueError, match=re.escape(f"does not fit {typed_as}")):
            expr()
    # One that does not fit by type but converts exactly still converts, as it
    # did on main, with a DeprecationWarning.
    for expr, typed_as in [
        (lambda: v.apply(udf, right=[2.0, 2.0, 2.0]), "_ExactArr"),
        (lambda: v.apply(udf, right=np.ones(3, dtype=np.int64)), "_ExactArr"),
        (lambda: v.apply(udf, right=np.int64(1)), "_ExactArr: it is typed as INT64,"),
        (lambda: v.apply(udf, right=(np.int8(1), 2, 3)), "_ExactArr"),
        (lambda: r.apply(second, right=np.int32(2)), "_ExactRec: it is typed as INT32,"),
    ]:
        with pytest.warns(DeprecationWarning, match=re.escape(f"does not fit {typed_as}")):
            expr()
    with pytest.raises(OverflowError, match="300 out of bounds for int8"):
        v.apply(udf, right=(1, 2, 300))
    # Defaults that fit still work, and fill in for the missing side.
    result = v.ewise_union(w, udf, 10, 20).new()
    assert result.to_coo()[1].tolist() == [[21, 22, 23], [14, 15, 16]]
    # apply with a Monoid promotes as its BinaryOp does, on either side.
    for expr in [v.apply(monoid.plus, right=0.5), v.apply(monoid.plus, left=0.5)]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype((np.float64, (3,)))
        np.testing.assert_array_equal(result[0].new().value, [1.5, 2.5, 3.5])
    assert v.apply(monoid.max, right=2).new().dtype == arr


# _SeqNest's numpy repr is over 128 chars on numpy 1; see test_udt_eq_nested_record_with_nan_leaf.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
@pytest.mark.skipif("not supports_udfs")
def test_udt_sequence_and_array_literals_type_like_numbers():
    """A tuple, list or array beside a UDT under a lifted op is typed, not converted into the UDT.

    A tuple or list of Python numbers is weak leaf by leaf, as a Python number
    is: ``int8_udt + (1, 2, 3)`` stays int8 and ``+ (0.5, 1.5, 2.5)`` is
    float64, where converting it raised (and before that added ``(0, 1, 2)``).
    A numpy array is strong, as its own array type. ``eq`` and ``ne`` compare
    the values. A literal beside a user-defined op is still converted
    (``test_udt_literal_converts_only_when_exact``).
    """
    int8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_SeqI8")
    fp32s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_SeqF32")
    nested = dtypes.register_anonymous(
        np.dtype(
            [("sq_a", np.int16), ("sq_in", [("sq_b", np.int8), ("sq_c", np.float32)])],
            align=True,
        ),
        "_SeqNest",
    )
    f64s = np.dtype((np.float64, (3,)))
    v = Vector(int8s, size=2)
    v[0] = [1, 2, 3]
    v[1] = [4, 5, 6]
    # Weak: the leaf's dtype when its kind holds the numbers, of the widest kind among them.
    for literal in [(1, 2, 3), [1, 2, 3], (True, 2, 3), (1,)]:
        result = (v + literal).new()
        assert result.dtype == int8s
        np.testing.assert_array_equal(result[0].new().value, np.add([1, 2, 3], literal))
    halves = (0.5, 1.5, 2.5)
    for expr in [v + halves, halves + v, v.apply(binary.plus, right=list(halves))]:
        result = expr.new()
        assert result.dtype.np_type == f64s
        np.testing.assert_array_equal(result[0].new().value, [1.5, 3.5, 5.5])
    result = binary.plus(v, (1, 0.5, 3)).new()  # one float anywhere makes the leaf float
    assert result.dtype.np_type == f64s
    np.testing.assert_array_equal(result[0].new().value, [2, 2.5, 6])
    result = binary.times(v, (1j, 1, 1)).new()
    assert result.dtype.np_type == np.dtype((np.complex128, (3,)))
    np.testing.assert_array_equal(result[0].new().value, [1j, 2, 3])
    with pytest.raises(OverflowError, match="300 out of bounds for int8"):
        binary.plus(v, (1, 2, 300))
    v += (1, 1, 1)
    np.testing.assert_array_equal(v[0].new().value, [2, 3, 4])
    # The float64 result is cast as it is stored, as a built-in int8 vector's would be.
    v += (0.5, 1, 1)
    np.testing.assert_array_equal(v[0].new().value, [2, 4, 5])
    # A float list beside float32 stays float32, as 0.5 does.
    x = Vector(fp32s, size=1)
    x[0] = [1, 2, 3]
    assert binary.plus(x, [0.5, 0.5, 0.5]).new().dtype == fp32s
    # Strong: an array is its own dtype, in its shape, or broadcast to the UDT's.
    for literal, expected in [
        (np.array([0.5, 1, 2]), f64s),
        (np.array([1, 2, 3]), np.dtype((np.array([1]).dtype, (3,)))),  # int32 on numpy 1 Windows
        (np.array([1], np.int8), int8s.np_type),
        (np.array([0.5]), f64s),
        (np.ones((1, 3)), np.dtype((np.float64, (1, 3)))),
    ]:
        result = (v + literal).new()
        assert result.dtype.np_type == expected
        np.testing.assert_array_equal(result[0].new().value, np.add([2, 4, 5], literal))
    # Records: by position, nested for a nested record, a dict by name, or a
    # 0-d structured array of another record type.
    n = Vector(nested, size=1)
    n[0] = (1, (2, 3.5))
    assert binary.plus(n, (1, (1, 1))).new().dtype == nested
    result = binary.plus(n, (1, (0.5, 1))).new()
    assert result.dtype.np_type == np.dtype(
        [("sq_a", np.int16), ("sq_in", [("sq_b", np.float64), ("sq_c", np.float32)])],
        align=True,
    )
    assert result[0].new().value.tolist() == (2, (2.5, 4.5))
    assert (n + {"sq_a": 0.5, "sq_in": {"sq_b": 1, "sq_c": 1}}).new()[0].new().value.tolist() == (
        1.5,
        (3, 4.5),
    )
    strong = np.array(
        (1, (0.5, 1)),
        dtype=[("sq_a", np.int8), ("sq_in", [("sq_b", np.float64), ("sq_c", np.int8)])],
    )
    assert (n + strong).new()[0].new().value.tolist() == (2, (2.5, 4.5))
    # eq and ne take the literal as it is, like numpy, instead of raising.
    assert (v == (0.5, 4, 5)).new().isequal(Vector.from_coo([0, 1], [False, False]))
    assert (v == (2, 4, 5)).new().isequal(Vector.from_coo([0, 1], [True, False]))
    assert (v != (2, 4, 500)).new().isequal(Vector.from_coo([0, 1], [True, True]))
    assert (v == np.array([2.0, 4, 5])).new()[0].new().value
    # Matrix and Scalar take the same path.
    A = Matrix(int8s, nrows=1, ncols=1)
    A[0, 0] = [1, 2, 3]
    ones = [1, 1, 1]
    assert (A + halves).new().dtype.np_type == f64s
    assert (A + ones).new().dtype == int8s
    s = gb.Scalar.from_value([1, 2, 3], int8s)
    assert (s + halves).new().dtype.np_type == f64s
    assert (s * ones).new().dtype == int8s


@pytest.mark.skipif("not supports_udfs")
def test_udt_ewise_union_defaults_are_typed_not_converted():
    """An ``ewise_union`` default beside a UDT is typed as a literal and must fit the operand.

    GraphBLAS casts a default to the op's input type, which for a UDT operand
    is its own type, and converting the operands instead would copy them. So
    a default is typed by the literal rules and accepted only when that type
    fits: ``1`` and ``(1, 1, 1)`` fit an int8 UDT, ``0.5``, ``np.int64(1)`` and
    a float64 Scalar do not. A Scalar (one element) of a type that casts
    safely is converted.
    """
    import re

    int8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_UnionI8")
    fp32s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_UnionF32")
    v = Vector(int8s, size=3)
    v[0] = [0, 1, 2]
    v[1] = [4, 5, 6]
    w = Vector(int8s, size=3)
    w[1] = [1, 1, 1]
    w[2] = [7, 8, 9]
    plus = binary.plus
    expected = [[1, 2, 3], [5, 6, 7], [7, 8, 9]]
    for left, right in [
        (0, 1),
        ((0, 0, 0), [1, 1, 1]),
        (np.int8(0), np.array([1, 1, 1], np.int8)),
        (gb.Scalar.from_value(0, "INT8"), gb.Scalar.from_value([1, 1, 1], int8s)),
        (False, True),
    ]:
        for expr in [
            v.ewise_union(w, plus, left, right),
            v.ewise_union(w, monoid.plus, left, right),
        ]:
            result = expr.new()
            assert result.dtype == int8s
            assert result.to_coo()[1].tolist() == expected

    def union_forms(d):
        return [
            lambda: v.ewise_union(w, plus, d, 0),
            lambda: v.ewise_union(w, plus, 0, d),
            lambda: plus(v | w, left_default=d, right_default=0),
            lambda: v.ewise_union(w, monoid.plus, d, 0),
        ]

    for default, message in [
        (0.5, "0.5 does not fit _UnionI8"),
        ((0.5, 0.5, 0.5), "(0.5, 0.5, 0.5) does not fit _UnionI8"),
        (np.float32(0.5), "does not fit _UnionI8: it is typed as FP32,"),
        (gb.Scalar.from_value(0.5), "Scalar of type FP64 does not fit _UnionI8"),
    ]:
        for expr in union_forms(default):
            with pytest.raises(ValueError, match=re.escape(message)):
                expr()
    # Exact values of another type still convert, as on main, with a
    # DeprecationWarning.
    for default, message in [
        (2.0, "2.0 does not fit _UnionI8"),
        (np.int64(1), "does not fit _UnionI8: it is typed as INT64,"),
        (np.array([1, 1, 1]), "array([1, 1, 1]) does not fit _UnionI8"),
        (gb.Scalar.from_value(1), "Scalar of type INT64 does not fit _UnionI8"),
    ]:
        for expr in union_forms(default):
            with pytest.warns(DeprecationWarning, match=re.escape(message)):
                expr()
    with pytest.raises(OverflowError, match="300 out of bounds for int8"):
        v.ewise_union(w, plus, 300, 0)
    # Each side is typed beside its own operand, so a mixed pair takes a
    # default of each type, and an int8 Scalar casts safely into float32.
    x = Vector(fp32s, size=3)
    x[2] = [0.5, 0.5, 0.5]
    result = v.ewise_union(x, plus, 0, 0.5).new()
    assert result.dtype == fp32s
    assert result.to_coo()[1].tolist() == [[0.5, 1.5, 2.5], [4.5, 5.5, 6.5], [0.5, 0.5, 0.5]]
    result = v.ewise_union(x, plus, 0, gb.Scalar.from_value([1, 1, 1], int8s)).new()
    assert result.to_coo()[1].tolist() == [[1, 2, 3], [5, 6, 7], [0.5, 0.5, 0.5]]
    with pytest.raises(ValueError, match="0.5 does not fit _UnionI8"):
        v.ewise_union(x, plus, 0.5, 0)
    # Matrix and Scalar check the same way.
    A = Matrix(int8s, nrows=2, ncols=1)
    A[0, 0] = [0, 1, 2]
    B = Matrix(int8s, nrows=2, ncols=1)
    B[1, 0] = [7, 8, 9]
    assert A.ewise_union(B, plus, 1, 1).new().to_coo()[2].tolist() == [[1, 2, 3], [8, 9, 10]]
    s = gb.Scalar.from_value([1, 2, 3], int8s)
    assert s.ewise_union(gb.Scalar(int8s), plus, 0, 1).new().value.tolist() == [2, 3, 4]
    for expr in [
        lambda: A.ewise_union(B, plus, 0.5, 0),
        lambda: A.ewise_union(v, plus, 0, 0.5),
        lambda: s.ewise_union(gb.Scalar(int8s), plus, 0, 0.5),
    ]:
        with pytest.raises(ValueError, match="0.5 does not fit _UnionI8"):
            expr()


@pytest.mark.skipif("not supports_udfs")
# SS < 9 names a UDT by its numpy repr and warns when that is over 128 chars,
# as the records with a field per dtype pair are.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_store_casts_each_element_as_builtin_types_do():
    """A UDT result stored in an object of another UDT type is cast element by element.

    GraphBLAS cannot cast a UDT, so storing across UDT types raised, and
    setting one element from a Scalar of another UDT type read its bytes as the
    object's type. Each array element or record leaf is now cast as GraphBLAS
    casts built-in types on store (compared below), without a temporary copy:
    an object stores through the cast op (``<<``, a mask, an accumulator,
    ``replace``, ``dup``), an element-wise expression of a lifted op casts as
    it computes, and a Scalar or one element is converted. Anything that would
    need a converted copy of a whole object raises, as do layouts that do not
    correspond and a built-in type on either side.
    """
    # One record casts every pair of leaf dtypes, so it compiles once.
    pairs = [
        ("f8", "i1"),
        ("f8", "u1"),
        ("f8", "i8"),
        ("f8", "u8"),
        ("f8", "?"),
        ("f8", "f4"),
        ("f4", "u4"),
        ("i8", "i1"),
        ("u8", "i8"),
        ("i1", "u8"),
        ("?", "f8"),
    ]
    if dtypes._supports_complex:
        pairs += [("c16", "f8"), ("c16", "i2"), ("c16", "?"), ("i4", "c8")]
    samples = {
        "f": [np.nan, np.inf, -np.inf, 300.7, -1.7, 2.0**63, -(2.0**64)],
        "c": [complex(np.nan, 1), 1 + 2j, -1.7 - 3j, 300.5j, 2.0**64, 0j, -2.5],
        "i": [2**63 - 1, -(2**63), 128, -129, -1, 0, 255],
        "u": [2**64 - 1, 2**63, 255, 256, 1, 0, 65535],
        "b": [True, False, True, True, False, True, False],
    }
    src_type = np.dtype([(f"sc{i}", src) for i, (src, _dst) in enumerate(pairs)])
    dst_type = np.dtype([(f"sc{i}", dst) for i, (_src, dst) in enumerate(pairs)])
    values = np.zeros(7, dtype=src_type)
    for i, (src, _dst) in enumerate(pairs):
        kind = np.dtype(src).kind
        values[f"sc{i}"] = np.array(samples[kind], np.uint64 if kind == "u" else None).astype(src)
    x = Vector.from_coo(np.arange(7), values, dtype=dtypes.register_anonymous(src_type))
    z = Vector(dtypes.register_anonymous(dst_type), size=7)
    z << x
    cast = z.to_coo()[1]
    for i, (src, dst) in enumerate(pairs):
        expected = Vector(dst, size=7)
        expected << Vector.from_coo(np.arange(7), values[f"sc{i}"], dtype=src)
        np.testing.assert_array_equal(cast[f"sc{i}"], expected.to_coo()[1], err_msg=f"{src} {dst}")

    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3, 2))), "_StoreCastI8")
    f64s = dtypes.register_anonymous(np.dtype((np.float64, (3, 2))), "_StoreCastF64")
    v = Vector(f64s, size=3)
    v[0] = [[1.7, -1.7], [300.0, np.nan], [-np.inf, 2.5]]
    v[2] = np.full((3, 2), 9.5)
    first = [[1, -1], [127, 0], [-128, 2]]
    nines = [[9, 9]] * 3
    w = Vector(i8s, size=3)
    w << v
    assert w.to_coo()[1].tolist() == [first, nines]
    assert (v * 1).new(dtype=i8s).isequal(w)
    # The mask, accumulator and replace act on the cast values, as for built-in
    # types: ``1 + 127`` wraps in int8.
    mask = Vector.from_coo([0, 1], [True, True], size=3)
    ones = [[1, 1]] * 3
    w = Vector.from_coo([0, 1, 2], [ones] * 3, dtype=i8s)
    w(mask.S, accum=binary.plus) << v
    assert w.to_coo()[1].tolist() == [[[2, 0], [-128, 1], [-127, 3]], ones, ones]
    w(mask.S) << v * 1  # an expression of a lifted op casts as it computes
    assert w.to_coo()[1].tolist() == [first, ones]
    w(mask.S, replace=True) << v
    assert w.to_coo()[1].tolist() == [first]
    # Scalars, as GraphBLAS scalars or not, and assignment of one element or many.
    for is_cscalar in [True, False]:
        s = gb.Scalar(i8s, is_cscalar=is_cscalar)
        s << v[0].new()
        assert s.value.tolist() == first
        s << v.reduce(monoid.plus)  # summed in float64, then cast
        assert s.value.tolist() == [[11, 7], [127, 0], [-128, 12]]
        assert v[0].new(is_cscalar=is_cscalar).dup(i8s).value.tolist() == first
    assert v.dup(i8s).isequal((v * 1).new(dtype=i8s))
    w = Vector(i8s, size=3)
    w[1] = v[0].new()  # read as int8 bytes before
    w[[0, 2]] = v[2].new()
    assert w.to_coo()[1].tolist() == [nines, first, nines]
    A = Matrix.from_coo([0], [1], [v[0].new().value], dtype=f64s, nrows=2, ncols=2)
    C = Matrix(i8s, nrows=2, ncols=2)
    C << A.T
    assert C.to_coo()[2].tolist() == [first]
    # A whole Vector of another type would need a converted copy to assign.
    with pytest.raises(DomainMismatch, match="converted copy of the whole value"):
        C[0, :] = v[[0, 1]].new()
    C[0, :] = v[[0, 1]].new().dup(dtype=i8s)
    assert C[0, 0].new().value.tolist() == first
    # So would a computation that cannot write another type as it computes.
    with pytest.raises(DomainMismatch, match="would need a temporary copy"):
        C << A.mxm(A, semiring.plus_times)
    with pytest.raises(DomainMismatch, match="would need a temporary copy"):
        w << v.ewise_add(v, binary.plus)
    with pytest.raises(DomainMismatch, match="extraction cannot"):
        w << v[[0, 1, 2]]
    # The computed result, stored as an existing object, casts in one pass.
    C << A.mxm(A, semiring.plus_times).new()
    # power raises before it computes any product; n of 0 and 1 store as above.
    with pytest.raises(DomainMismatch, match="mxm with plus_times cannot"):
        C << A.power(3)
    C << A.power(1)
    assert C.to_coo()[2].tolist() == [first]
    # Leading axes of length 1 do not change the layout.
    lead = dtypes.register_anonymous(np.dtype((np.float32, (1, 3, 2))), "_StoreCastLead")
    u = Vector(lead, size=3)
    u << w
    assert u.to_coo()[1].tolist() == [[nines], [first], [nines]]

    other_shape = Vector(dtypes.register_anonymous(np.dtype((np.int16, (6,)))), size=3)
    with pytest.raises(DomainMismatch, match="same shape, apart from leading axes"):
        other_shape << v
    renamed = np.dtype([(f"sr{i}", dst) for i, (_src, dst) in enumerate(pairs)])
    with pytest.raises(DomainMismatch, match="same field names"):
        Vector(dtypes.register_anonymous(renamed), size=7) << x
    raw = Vector(dtypes.register_anonymous(np.dtype("S48"), "_StoreCastBytes"), size=3)
    with pytest.raises(DomainMismatch, match="only record and array UDTs cast"):
        raw << v
    for expr in [
        lambda: Vector(f64s, size=3) << Vector.from_coo([0], [1.5], size=3),
        lambda: Vector(dtypes.FP64, size=3) << v,
        lambda: w.__setitem__(0, gb.Scalar.from_value(1.5)),
    ]:
        with pytest.raises(DomainMismatch, match="a UDT casts only to another UDT"):
            expr()


@pytest.mark.skipif("not supports_udfs")
def test_udt_extracted_element_of_another_type_is_cast():
    """One element extracted into another UDT type is cast, not copied as raw bytes.

    GraphBLAS's extractElement for a UDT copies the bytes of the object's type
    into the destination, so ``s << v[0]`` with ``s`` of another UDT read
    garbage, and wrote past the end of a smaller element (also on main).
    """
    f64s = dtypes.register_anonymous(np.dtype((np.float64, (2,))), "_ExtractF64")
    i16s = dtypes.register_anonymous(np.dtype((np.int16, (2,))), "_ExtractI16")
    rec = dtypes.register_anonymous(np.dtype([("ex_a", np.int8), ("ex_b", np.int8)]), "_ExtractRec")
    v = Vector(f64s, size=3)
    v[0] = [1.7, -300.2]
    A = Matrix(f64s, nrows=2, ncols=3)
    A[1, 2] = [2.5, -1e30]
    s = gb.Scalar(i16s)
    s << v[0]
    assert s.value.tolist() == [1, -300]
    assert v[0].new(dtype=i16s).value.tolist() == [1, -300]
    assert A.T[2, 1].new(dtype=i16s).value.tolist() == [2, -32768]  # saturates
    w = Vector(i16s, size=3)
    C = Matrix(i16s, nrows=2, ncols=2)
    with gb.config.set(autocompute=True):
        w[1] = v[0]
        C[1, 0] = A[1, 2]
    assert w[1].new().value.tolist() == [1, -300]
    assert C[1, 0].new().value.tolist() == [2, -32768]
    # A larger destination, and a missing element.
    x = Vector(i16s, size=2)
    x[0] = [7, -8]
    assert x[0].new(dtype=f64s).value.tolist() == [7.0, -8.0]
    assert x[1].new(dtype=f64s).is_empty
    for dtype in [rec, dtypes.FP64]:
        with pytest.raises(DomainMismatch, match="cannot store"):
            v[0].new(dtype=dtype)


@pytest.mark.skipif("not supports_udfs")
def test_udt_fused_store_matches_compute_then_cast():
    """Storing an element-wise result into another UDT type casts as the op computes.

    The op computes each element as it would for its own result type and casts
    it into the object's type as it writes it, so the values are those of
    computing the result and then casting it (``.new().dup(dtype=)``), with no
    temporary of the result's type: special values, records with array
    fields, literals on either side, unary ops, masks and accumulators.
    """
    i16s = dtypes.register_anonymous(np.dtype((np.int16, (2, 2))), "_FuseI16")
    f32s = dtypes.register_anonymous(np.dtype((np.float32, (2, 2))), "_FuseF32")
    rec_i = dtypes.register_anonymous(
        np.dtype([("fz_a", np.int8), ("fz_b", np.uint16, (3,))]), "_FuseRecI"
    )
    rec_f = dtypes.register_anonymous(
        np.dtype([("fz_a", np.float64), ("fz_b", np.float64, (3,))]), "_FuseRecF"
    )
    i = Vector(i16s, size=4)
    i[0] = [[1, -2], [32767, -32768]]
    i[2] = [[7, 0], [-7, 3]]
    f = Vector(f32s, size=4)
    f[0] = [[np.nan, -np.inf], [1e10, -0.5]]
    f[1] = [[2.5, -2.5], [65535.9, 1.5]]
    f[2] = [[0.25, 3.75], [-1.0, 2.0]]
    r = Vector(rec_f, size=4)
    r[0] = (300.7, [-1.5, 70000.2, np.nan])
    r[1] = (-129.9, [2.5, 3.5, 1e300])
    cases = [
        (lambda: i * 2.5, i16s),
        (lambda: 2.5 - i, i16s),
        (lambda: i / 4, i16s),
        (lambda: i.ewise_mult(f, binary.times), i16s),
        (lambda: f.ewise_mult(i, binary.minus), i16s),
        (lambda: i.ewise_union(f, binary.plus, 0, 0), i16s),
        (lambda: f.apply(unary.ainv), i16s),
        (lambda: f.apply(monoid.max, right=3), i16s),
        (lambda: f * f, i16s),
        (lambda: f.ewise_mult(f, monoid.plus), i16s),
        (lambda: r * 1.5, rec_i),
        (lambda: r.apply(unary.abs), rec_i),
    ]
    mask = Vector.from_coo([0, 2], [True, True], size=4)
    for make, out in cases:
        expected = make().new().dup(dtype=out)
        fused = Vector(out, size=expected.size)
        fused << make()
        assert fused.isequal(expected), make
        # A mask and an accumulator act on the cast values, as they would on
        # values that were already of the object's type.
        start = expected.dup()
        explicit = start.dup()
        explicit(mask.S, accum=binary.plus) << make().new().dup(dtype=out)
        start(mask.S, accum=binary.plus) << make()
        assert start.isequal(explicit), make
    A = Matrix(f32s, nrows=2, ncols=2)
    A[0, 1] = [[1.5, -2.5], [np.nan, 1e9]]
    C = Matrix(i16s, nrows=2, ncols=2)
    C << A.T.apply(binary.times, right=2)
    assert C.isequal(A.T.apply(binary.times, right=2).new().dup(dtype=i16s))


def _assert_same_floats(got, expected, msg):
    """Equal, NaN equal to NaN, and with the same sign on every zero (of each complex part)."""
    got, expected = np.asarray(got), np.asarray(expected)
    np.testing.assert_array_equal(got, expected, err_msg=msg)
    for part in (np.real, np.imag) if got.dtype.kind == "c" else (np.asarray,):
        np.testing.assert_array_equal(
            np.signbit(part(got)), np.signbit(part(expected)), err_msg=msg
        )


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_record_array_fields_compute_element_by_element():
    """An array-valued record field computes each element as a scalar field does.

    As a whole-array expression it ran through Numba's ufuncs: ``min`` and
    ``max`` did not compile (and the scalar NaN rule, applied to an array, left
    garbage), a zero complex divisor raised inside the cfunc and left the
    element unwritten, and ``int8 / int8`` ran in float32. Each element now
    takes the scalar expression, so a field agrees with numpy's ``fmin``,
    ``fmax``, ``floor_divide`` and ``true_divide`` element for element, with a
    record or a number on the other side. ``INT64_MIN // -1`` is the one
    integer quotient Numba gets wrong by itself (0, where numpy wraps).
    """
    nan, inf = float("nan"), float("inf")
    rec = dtypes.register_anonymous(
        np.dtype(
            [
                ("aef_s", np.float64),
                ("aef_f", np.float64, (2, 3)),
                ("aef_i", np.int8, (4,)),
                ("aef_l", np.int64, (3,)),
                ("aef_m", np.int64),
            ],
            align=True,
        ),
        "_ArrElemFields",
    )
    fx = np.array([[nan, 1.0, -0.0], [inf, 1.0, 7.5]])
    fy = np.array([[1.0, nan, 2.0], [2.0, 0.1, -2.5]])
    ix = np.array([-128, 7, -7, 100], np.int8)
    iy = np.array([-1, 0, 2, 3], np.int8)
    int64_min = np.iinfo(np.int64).min
    lx = np.array([int64_min, -7, 9], np.int64)
    ly = np.array([-1, 2, 0], np.int64)
    v = Vector(rec, size=1)
    v[0] = (nan, fx, ix, lx, int64_min)
    w = Vector(rec, size=1)
    w[0] = (1.0, fy, iy, ly, -1)
    # The references overflow (``-128 // -1``) and divide by zero on purpose.
    with np.errstate(all="ignore"):
        for gb_op, reference in [
            (binary.min, np.fmin),
            (binary.max, np.fmax),
            (binary.floordiv, np.floor_divide),
            (binary.truediv, np.true_divide),
        ]:
            got = gb_op(v & w).new()[0].new().value
            _assert_same_floats(got["aef_s"], reference(nan, 1.0), gb_op.name)
            _assert_same_floats(got["aef_f"], reference(fx, fy), gb_op.name)
            np.testing.assert_array_equal(got["aef_i"], reference(ix, iy), err_msg=gb_op.name)
            np.testing.assert_array_equal(got["aef_l"], reference(lx, ly), err_msg=gb_op.name)
            np.testing.assert_array_equal(
                got["aef_m"], reference(np.int64(int64_min), np.int64(-1)), err_msg=gb_op.name
            )
            # A number on the other side reaches every element of the field.
            for other in [np.float64(0.5), np.float64(nan)]:
                got = v.apply(gb_op, right=other).new()[0].new().value
                _assert_same_floats(got["aef_f"], reference(fx, other), f"{gb_op.name} {other}")
            got = v.apply(gb_op, right=np.int8(-1)).new()[0].new().value
            np.testing.assert_array_equal(
                got["aef_i"], reference(ix, np.int8(-1)), err_msg=gb_op.name
            )
            np.testing.assert_array_equal(
                got["aef_l"], reference(lx, np.int8(-1)), err_msg=gb_op.name
            )

    cplx = dtypes.register_anonymous(
        np.dtype([("aef_c", np.complex128, (3,))], align=True), "_ArrElemComplex"
    )
    cx = np.array([3 + 4j, 1j, 2 - 2j])
    cy = np.array([0j, 2, 0j])
    v = Vector(cplx, size=1)
    v[0] = (cx,)
    w = Vector(cplx, size=1)
    w[0] = (cy,)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = cx / cy
    np.testing.assert_array_equal(binary.truediv(v & w).new()[0].new().value["aef_c"], expected)


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_min_max_and_division_match_numpy(udt_op_path):
    """Array UDT elements take ``fmin``, ``fmax`` and numpy's divisions on both paths.

    NaN sits on each side in turn, and ``INT64_MIN // -1`` is the integer
    quotient Numba gets wrong by itself.
    """
    nan, inf = float("nan"), float("inf")
    f64 = dtypes.register_anonymous(np.dtype((np.float64, (2, 3))), "_ArrNumpyF64")
    i64 = dtypes.register_anonymous(np.dtype((np.int64, (3,))), "_ArrNumpyI64")
    fx = np.array([[nan, 1.0, -7.5], [inf, 1.0, 0.0]])
    fy = np.array([[1.0, nan, 2.0], [2.0, 0.1, 0.0]])
    int64_min = np.iinfo(np.int64).min
    ix = np.array([int64_min, -7, 9], np.int64)
    iy = np.array([-1, 2, 0], np.int64)
    # Two entries each, with the values swapped in the second, so neither
    # vector is iso-valued (SuiteSparse answers those without a kernel).
    v = Vector(f64, size=2)
    w = Vector(f64, size=2)
    v[0], w[0], v[1], w[1] = fx, fy, fy, fx
    a = Vector(i64, size=2)
    b = Vector(i64, size=2)
    a[0], b[0], a[1], b[1] = ix, iy, ix[::-1], iy[::-1]
    # The references divide by zero and overflow on purpose.
    with np.errstate(all="ignore"):
        for gb_op, reference in [
            (binary.min, np.fmin),
            (binary.max, np.fmax),
            (binary.floordiv, np.floor_divide),
            (binary.truediv, np.true_divide),
        ]:
            got = gb_op(v & w).new()
            for k, (x, y) in enumerate([(fx, fy), (fy, fx)]):
                msg = f"{udt_op_path} {gb_op.name} [{k}]"
                _assert_same_floats(got[k].new().value, reference(x, y), msg)
            got = gb_op(a & b).new()
            for k, (x, y) in enumerate([(ix, iy), (ix[::-1], iy[::-1])]):
                msg = f"{udt_op_path} {gb_op.name} [{k}]"
                np.testing.assert_array_equal(got[k].new().value, reference(x, y), err_msg=msg)


@pytest.mark.skipif("not supports_udfs")
def test_udt_fused_store_keeps_min_max_and_division_semantics(udt_op_path):
    """A fused store computes ``min``, ``max`` and the divisions as the op does, then casts.

    The fused op is cfunc only, so with the C JIT on, the plain op runs the C
    kernel and the two paths are compared here directly: NaN in ``min`` and
    ``max``, zero divisors, ``INT_MIN // -1``, a complex zero divisor, and
    array-valued record fields.
    """
    nan, inf = float("nan"), float("inf")
    f64 = dtypes.register_anonymous(np.dtype((np.float64, (2, 2))), "_FuseSemF64")
    f32 = dtypes.register_anonymous(np.dtype((np.float32, (2, 2))), "_FuseSemF32")
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (4,))), "_FuseSemI8")
    i32s = dtypes.register_anonymous(np.dtype((np.int32, (4,))), "_FuseSemI32")
    rec64 = dtypes.register_anonymous(
        np.dtype([("fss_a", np.float64), ("fss_b", np.float64, (3,))], align=True), "_FuseSemR64"
    )
    rec32 = dtypes.register_anonymous(
        np.dtype([("fss_a", np.float32), ("fss_b", np.float32, (3,))], align=True), "_FuseSemR32"
    )
    c128 = dtypes.register_anonymous(
        np.dtype([("fss_c", np.complex128)], align=True), "_FuseSemC128"
    )
    c64 = dtypes.register_anonymous(np.dtype([("fss_c", np.complex64)], align=True), "_FuseSemC64")
    f, g = _udt_vectors(f64, [nan, 1.0, -7.0, inf, 0.0], [1.0, nan, 2.0, 2.0, 0.0])
    i, j = _udt_vectors(i8s, [-128, 7, -7, 100, 5], [-1, 0, 2, 3, -2])
    r = Vector(rec64, size=3)
    s = Vector(rec64, size=3)
    for k, (x, y) in enumerate([(nan, 1.0), (1.0, nan), (-7.5, 2.0)]):
        r[k] = (x, [x, y, x * 2])
        s[k] = (y, [y, x, 0.0])
    c = Vector(c128, size=2)
    d = Vector(c128, size=2)
    c[0], d[0] = (3 + 4j,), (0j,)
    c[1], d[1] = (1j,), (2 + 0j,)
    cases = [(f, g, f32, op) for op in ("min", "max", "floordiv", "truediv")]
    cases += [(i, j, i32s, "floordiv"), (r, s, rec32, "min"), (r, s, rec32, "max")]
    cases += [(r, s, rec32, "floordiv"), (c, d, c64, "truediv")]
    for x, y, out, name in cases:
        gb_op = getattr(binary, name)
        expected = gb_op(x & y).new().dup(dtype=out)
        fused = Vector(out, size=x.size)
        fused << gb_op(x & y)
        assert fused.nvals == expected.nvals, (udt_op_path, name, out)
        for k in range(x.size):
            got, want = fused[k].new().value, expected[k].new().value
            msg = f"{udt_op_path} {name} into {out} at {k}"
            if out.np_type.names is None:
                _assert_same_floats(got, want, msg)
            else:
                for field in out.np_type.names:
                    _assert_same_floats(got[field], want[field], msg)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_records_differing_only_in_layout_combine():
    """A packed and an aligned record with the same leaves combine, either way round.

    The result is the left operand's type, since it holds every leaf. Converting
    the other operand to it for a Scalar's ``ewise_add`` subtracted a zero,
    which came back as the other operand's own type, so the conversion recursed
    forever; Vectors raised RecursionError the same way before ``ewise_add``
    refused operands of another type than the result's.
    """
    fields = [("lay_f", np.float64), ("lay_i", np.int16), ("lay_n", [("lay_k", np.int32)])]
    packed = dtypes.register_anonymous(np.dtype(fields), "_LayPacked")
    aligned = dtypes.register_anonymous(np.dtype(fields, align=True), "_LayAligned")
    x = Vector(packed, size=3)
    x[0] = (1.5, 2, (3,))
    y = Vector(aligned, size=3)
    y[0] = (-0.0, 1, (1,))
    y[2] = (10.0, 20, (30,))
    for a, b in [(x, y), (y, x)]:
        result = a.ewise_union(b, binary.plus, 0, 0).new()
        assert result.dtype == a.dtype
        assert result.to_coo()[1].tolist() == [(1.5, 3, (4,)), (10.0, 20, (30,))]
        assert (a * b).new().dtype == a.dtype
        with pytest.raises(DomainMismatch, match="ewise_add cannot use"):
            a.ewise_add(b)
        assert (a + b).new().isequal(result)
    s = gb.Scalar.from_value((1.5, 2, (3,)), dtype=packed)
    t = gb.Scalar.from_value((1.0, 1, (1,)), dtype=aligned)
    assert (s + t).new().value.tolist() == (2.5, 3, (4,))
    # Converting keeps a negative zero where there is no partner.
    z = gb.Scalar.from_value((-0.0, 0, (0,)), dtype=aligned)
    assert np.signbit((gb.Scalar(packed) + z).new().value["lay_f"])
    # Either layout stores into the other through the cast op.
    x << y
    assert x.to_coo()[1].tolist() == [(-0.0, 1, (1,)), (10.0, 20, (30,))]


@pytest.mark.skipif("not supports_udfs")
def test_udt_record_array_field_truediv_is_float64():
    """``truediv`` on a small integer array field divides in float64, as a scalar field does.

    Numba runs an array field through a ufunc whose loop it picks itself, and
    for int8 or int16 it picked float32, so ``1 / 46`` came back as
    0.021739130839705467 in an FP64 field.
    """
    for np_type in [np.int8, np.uint8, np.int16, np.uint16]:
        name = f"_TdArr{np.dtype(np_type).name}"
        rec = dtypes.register_anonymous(
            np.dtype([(f"td_a_{name}", np_type, (2,)), (f"td_s_{name}", np_type)]), name
        )
        v = Vector(rec, size=1)
        v[0] = ([1, 46], 46)
        w = Vector(rec, size=1)
        w[0] = ([46, 7], 3)
        result = (v / w).new()[0].new().value
        np.testing.assert_array_equal(result[0], np.array([1, 46]) / np.array([46, 7]))
        assert result[1] == 46 / 3


@pytest.mark.skipif("not supports_udfs")
def test_udt_typed_scalar_operands_and_defaults():
    """Typed Scalars keep their values beside a UDT, as literals do.

    ``Scalar.ewise_union`` converted its other operand into the left UDT
    (``s8 - sf`` was int8), and an ``ewise_union`` default given as a typed
    Scalar was converted into the operand's UDT however much that changed it.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_TsI8")
    f8s = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_TsF8")
    s8 = gb.Scalar.from_value([1, 2, 3], dtype=i8s)
    sf = gb.Scalar.from_value([0.5, 0.5, 0.5], dtype=f8s)
    for expr, expected in [(s8 - sf, [0.5, 1.5, 2.5]), (sf - s8, [-0.5, -1.5, -2.5])]:
        result = expr.new()
        assert result.dtype == f8s
        np.testing.assert_array_equal(result.value, expected)
    v = Vector(i8s, size=2)
    v[0] = [1, 2, 3]
    w = Vector(i8s, size=2)
    w[1] = [4, 5, 6]
    A = Matrix(i8s, nrows=1, ncols=2)
    A[0, 0] = [1, 2, 3]
    B = Matrix(i8s, nrows=1, ncols=2)
    B[0, 1] = [4, 5, 6]
    for a, b in [(v, w), (A, B)]:
        # A default must have the operand's type, and FP64 does not cast safely to
        # int8; 2.0 still converts, being exact, but that is deprecated.
        with pytest.raises(ValueError, match="Scalar of type FP64 does not fit _TsI8"):
            a.ewise_union(b, binary.plus, gb.Scalar.from_value(0.5), 0)
        with pytest.warns(DeprecationWarning, match="Scalar of type FP64 does not fit _TsI8"):
            a.ewise_union(b, binary.plus, gb.Scalar.from_value(2.0), 0)
        with pytest.raises(ValueError, match="Scalar of type _TsF8 does not fit _TsI8"):
            a.ewise_union(b, binary.plus, 0, sf)
        assert a.ewise_union(b, binary.plus, gb.Scalar.from_value(2, "INT8"), 0).new().dtype == i8s
    # Matrix.apply with a Monoid promotes as Vector.apply does.
    result = A.apply(monoid.plus, right=0.5).new()
    assert result.dtype.np_type == np.dtype((np.float64, (3,)))


@pytest.mark.skipif("not supports_udfs")
def test_udt_errors_name_the_real_problem():
    """Errors that the UDT rules had made misleading.

    Infix on two UDTs of different sizes said ``ewise_add cannot use
    binary.first``, the op it uses to provoke the size error, and an op that
    does not take UDTs at all reported the literal check's ValueError.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_ErrI8")
    f8s = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_ErrF8")
    v = Vector(i8s, size=2)
    w = Vector(f8s, size=3)
    for expr in [lambda: binary.plus(v | w), lambda: binary.first(v | w), lambda: v + w]:
        with pytest.raises(DimensionMismatch):
            expr().new()
    with pytest.raises(KeyError, match="lt does not work with _ErrI8"):
        v.apply(binary.lt, right=2.5)


@pytest.mark.skipif("not supports_udfs")
def test_udt_literal_rules_edge_cases():
    """Edges of the weak rule, each pinned on its own.

    numpy 2 raises OverflowError itself when ``300`` is converted to int8, which
    hid that the weak rule's own range check matters on numpy 1, where the
    conversion wraps. A ``Fraction`` or ``Decimal`` is typed as the int or
    float it is worth; numpy would have truncated either into an int field.
    """
    from decimal import Decimal
    from fractions import Fraction

    from graphblas.core.operator.udt_utils import _weak_literal_udt

    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_EdgeI8")
    f4s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_EdgeF4")
    c8s = dtypes.register_anonymous(np.dtype((np.complex64, (3,))), "_EdgeC8")
    with pytest.raises(OverflowError, match="300 out of bounds for int8"):
        _weak_literal_udt(i8s, 300)
    assert _weak_literal_udt(i8s, 1) is i8s
    assert _weak_literal_udt(f4s, 1j).np_type == np.dtype((np.complex64, (3,)))
    v = Vector(i8s, size=1)
    v[0] = [1, 2, 3]
    for value in [Fraction(1, 2), Decimal("0.5")]:
        result = (v + value).new()
        assert result.dtype.np_type == np.dtype((np.float64, (3,)))
        assert result[0].new().value.tolist() == [1.5, 2.5, 3.5]
    for value in [Fraction(2), Decimal(2)]:
        result = (v + value).new()
        assert result.dtype == i8s
        assert result[0].new().value.tolist() == [3, 4, 5]
    udf = BinaryOp.register_anonymous(lambda x, y: x + y, "_edge_plus", is_udt=True)
    with pytest.raises(ValueError, match=r"Fraction\(1, 2\) does not fit _EdgeI8"):
        v.apply(udf, right=Fraction(1, 2))
    c = Vector(c8s, size=1)
    c[0] = [1, 2, 3]
    assert c.apply(binary.plus, right=(1 + 2j, 0, 0)).new()[0].new().value[0] == 2 + 2j


@pytest.mark.skipif("not supports_udfs")
def test_udt_store_casts_reach_every_store_path():
    """Each way of storing into a UDT object casts as ``<<`` does, or says why it cannot.

    ``agg.first`` into a Scalar of another UDT handed GraphBLAS the output as it
    was, which read the int8 bytes as float64 (24 bytes from a 3-byte
    element). Indexed assignment of an expression skipped the cast, a UDT
    Scalar's ``s += 0.5`` converted 0.5 into the UDT where a Vector casts the
    sum, and ``dup`` failed for a dtype given as a numpy dtype or a string.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_PathI8")
    f8s = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_PathF8")
    v = Vector(i8s, size=3)
    v[0] = [1, 2, 3]
    v[2] = [4, 5, 6]
    aggregators = [(agg.sum, [5.0, 7.0, 9.0])]
    if gb.backend == "suitesparse":
        aggregators += [(agg.ss.first, [1.0, 2.0, 3.0]), (agg.ss.last, [4.0, 5.0, 6.0])]
    for aggregator, expected in aggregators:
        s = gb.Scalar(f8s)
        s << v.reduce(aggregator)
        np.testing.assert_array_equal(s.value, expected)
    with pytest.raises(DomainMismatch, match="a UDT casts only to another UDT"):
        gb.Scalar(FP64) << v.reduce(agg.sum)
    w = Vector(f8s, size=3)
    w[0] = [0.5, 0.5, 0.5]
    w[1] = [1.5, 1.5, 1.5]
    # Assigning part of an object from a whole object of another type would
    # need a converted copy, an expression's (extract) as well; so would
    # ewise_add, which copies an entry without a partner as it is.
    x = v.dup()
    for expr in [lambda: x[[0, 1]] << w[[0, 1]], lambda: x[[0, 2]] << (v + 0.5).new()[[0, 2]]]:
        with pytest.raises(DomainMismatch, match="converted copy of the whole value"):
            expr()
    x[[0, 1]] << w[[0, 1]].new().dup(dtype=i8s)
    assert x.to_coo()[1].tolist() == [[0, 0, 0], [1, 1, 1], [4, 5, 6]]
    s = gb.Scalar.from_value([1, 2, 3], dtype=i8s)
    s += 0.5
    np.testing.assert_array_equal(s.value, [1, 2, 3])
    s += 1
    np.testing.assert_array_equal(s.value, [2, 3, 4])
    result = s.dup(dtype=np.dtype((np.float64, (3,))))
    assert result.dtype == f8s
    np.testing.assert_array_equal(result.value, [2, 3, 4])
    with pytest.raises(DomainMismatch, match="a UDT casts only to another UDT"):
        s.dup(dtype="FP64")


@pytest.mark.skipif("not supports_udfs")
def test_udt_literal_fits_in_select_bool_arrays_and_overflow():
    """Three edges of the literal rules found by review.

    A user IndexUnaryOp's thunk in ``select`` skipped the must-fit check (1.5
    became 1), a bool numpy array beside an array UDT failed to look up a
    ``BOOL[3]`` type (main converted it), and a float literal too large for a
    float32 field became infinity where it must become an element.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_EdgeSelI8")
    f4s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_EdgeSelF4")

    def ge_first(x, i, j, t):  # pragma: no cover (numba)
        return x[0] >= t[0]

    ge = IndexUnaryOp.register_anonymous(ge_first, "_edge_sel_ge", is_udt=True)
    v = Vector(i8s, size=3)
    v[0] = [1, 2, 3]
    v[1] = [2, 3, 4]
    assert v.select(ge, 2).new().to_coo()[0].tolist() == [1]
    for thunk in [1.5, (1.5, 0, 0), np.float64(2.5)]:
        with pytest.raises(ValueError, match="does not fit _EdgeSelI8"):
            v.select(ge, thunk)
    A = Matrix(i8s, nrows=1, ncols=2)
    A[0, 0] = [1, 2, 3]
    with pytest.raises(ValueError, match="does not fit _EdgeSelI8"):
        A.select(ge, 1.5)
    # A bool array converts into any numeric element, as it did, and bools in a
    # sequence compare as 0 and 1 (a BOOL[3] UDT cannot be registered).
    result = v.apply(binary.plus, right=np.array([True, False, True])).new()
    assert result.dtype == i8s
    assert result[0].new().value.tolist() == [2, 2, 4]
    ones = Vector(i8s, size=2)
    ones[0] = [1, 0, 1]
    assert (ones == (True, False, True)).new()[0].new().value
    assert not (ones != [True, False, True]).new()[0].new().value
    # A float too large for a float32 field raises where it must become an element.
    f = Vector(f4s, size=2)
    f[0] = [1, 2, 3]
    with pytest.raises(OverflowError, match="overflows _EdgeSelF4"):
        f.ewise_union(f, binary.plus, 1e300, 0)
    with pytest.raises(OverflowError, match="overflows _EdgeSelF4"):
        f.apply(binary.register_anonymous(lambda x, y: x, "_edge_sel_first", is_udt=True), 1e300)


@pytest.mark.skipif("not supports_udfs")
def test_udt_exact_literal_of_another_type_is_deprecated():
    """A must-fit value that fits only by value converts, with a DeprecationWarning.

    On main any literal, thunk or default was converted into the UDT, and what
    did not fit was lost. Now it must fit by type; one whose values convert
    exactly, such as 0 for a bool field, still converts for the deprecation
    period, and the warning points at the caller's line.
    """
    rec = dtypes.register_anonymous(
        np.dtype([("dp_b", np.bool_), ("dp_f", np.float64), ("dp_i", np.int8)]), "_DeprecRec"
    )
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_DeprecI8")
    r = Vector(rec, size=2)
    r[0] = (True, 1.5, 2)
    s = Vector(rec, size=2)
    s[1] = (False, 2.5, 3)
    expected = r.ewise_union(s, binary.plus, False, False).new()
    for default in [0, (0, 0.0, 0), {"dp_b": 0, "dp_f": 0.0, "dp_i": 0}]:
        with pytest.warns(DeprecationWarning, match="does not fit _DeprecRec") as record:
            result = r.ewise_union(s, binary.plus, default, False).new()
        assert result.isequal(expected)
        assert {w.filename for w in record} == {__file__}
    with pytest.raises(ValueError, match="3 does not fit _DeprecRec"):
        r.ewise_union(s, binary.plus, 3, False)

    def ge_first(x, i, j, t):  # pragma: no cover (numba)
        return x[0] >= t[0]

    ge = IndexUnaryOp.register_anonymous(ge_first, "_deprec_ge", is_udt=True)
    v = Vector(i8s, size=3)
    v[0] = [1, 2, 3]
    v[1] = [2, 3, 4]
    A = Matrix(i8s, nrows=1, ncols=2)
    A[0, 1] = [2, 3, 4]
    for x in [v, A]:
        with pytest.warns(DeprecationWarning, match="2.0 does not fit _DeprecI8"):
            assert x.select(ge, 2.0).new().isequal(x.select(ge, 2).new())
        for thunk in [1.5, float("nan"), np.int64(300)]:
            with pytest.raises(ValueError, match="does not fit _DeprecI8"):
                x.select(ge, thunk)


@pytest.mark.skipif("not supports_udfs")
def test_udt_isequal_is_array_equal_and_eq_broadcasts():
    """Isequal compares array elements as np.array_equal; == broadcasts, on Scalars too.

    eq on array UDTs broadcasts, as numpy's == does, and isequal used it, so an
    FP64[3] of [1, 1, 1] was isequal to an FP64[1] of [1]. A Scalar's == was
    isequal itself, and a literal was converted into the UDT first, so an
    INT8[3] of [0, 0, 0] equaled 0.5 (truncated to 0) by either spelling.
    """
    f3 = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_IeqF3")
    f1 = dtypes.register_anonymous(np.dtype((np.float64, (1,))), "_IeqF1")
    f13 = dtypes.register_anonymous(np.dtype((np.float64, (1, 3))), "_IeqF13")
    i3 = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_IeqI3")
    rec = dtypes.register_anonymous(np.dtype([("ieq_a", np.int8), ("ieq_b", np.float32)]), "_IeqR")

    def vec(dtype, value):
        v = Vector(dtype, size=2)
        v[1] = value
        return v

    ones = vec(f3, [1, 1, 1])
    assert not ones.isequal(vec(f1, [1]))
    assert (ones == vec(f1, [1])).new().reduce(monoid.land).new().value  # == broadcasts
    assert ones.isequal(vec(f13, [[1, 1, 1]]))  # leading axes of length 1 do not count
    assert ones.isequal(vec(i3, [1, 1, 1]))
    assert not ones.isequal(Vector.from_coo([1], [1.0], size=2))
    assert vec(f1, [5]).isequal(Vector.from_coo([1], [5.0], size=2))
    A = Matrix.from_coo([0], [1], [[1, 1, 1]], dtype=f3, nrows=1, ncols=2)
    assert not A.isequal(Matrix.from_coo([0], [1], [[1]], dtype=f1, nrows=1, ncols=2))
    assert A.isequal(Matrix.from_coo([0], [1], [[1, 1, 1]], dtype=i3, nrows=1, ncols=2))

    u = gb.Scalar.from_value([1, 1, 1], i3)
    zeros = gb.Scalar.from_value([0, 0, 0], i3)
    assert u == 1
    assert u != 0
    assert u == gb.Scalar.from_value(1)
    assert gb.Scalar.from_value(1) == u
    assert u == gb.Scalar.from_value([1.0], f1)
    assert not u.isequal(1)
    assert not u.isequal([1])
    assert not u.isequal(gb.Scalar.from_value([1.0], f1))
    assert u.isequal((1, 1, 1))
    assert u.isequal(np.ones((1, 3), dtype=np.int8))
    assert gb.Scalar.from_value([5], f1).isequal(5)
    # A literal is compared as given, not converted into the UDT first.
    assert zeros != 0.5
    assert not zeros.isequal(0.5)
    assert not zeros.isequal((0, 0, 0.5))
    r = gb.Scalar.from_value((0, 0.1), rec)
    assert r == (0, 0.1)  # 0.1 beside a float32 field is float32, as == types it
    assert r.isequal((0, 0.1))
    assert r != (0.5, 0.1)
    assert not r.isequal((0.5, 0.1))
    # Empty Scalars are equal to each other, and to None.
    assert gb.Scalar(i3) == gb.Scalar(f3)
    assert gb.Scalar(i3) == None  # noqa: E711
    assert gb.Scalar(i3) != u
    assert u != gb.Scalar(i3)


@pytest.mark.skipif("not supports_udfs")
def test_udt_monoid_literal_is_typed_as_its_binaryop():
    """A literal beside a Monoid is typed as beside its BinaryOp, everywhere.

    ``apply`` already used the BinaryOp, but a Scalar's ``ewise_add`` and
    ``ewise_mult`` with a Monoid converted the literal into the UDT (so ``0.5``
    had to fit), where the same call with ``binary.plus`` promoted.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_MonoLitI8")
    s = gb.Scalar.from_value([1, 2, 3], dtype=i8s)
    for expr in [s.ewise_add(0.5, monoid.plus), s.ewise_add(0.5, binary.plus)]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype((np.float64, (3,)))
        np.testing.assert_array_equal(result.value, [1.5, 2.5, 3.5])
    assert s.ewise_mult(2, monoid.times).new().dtype == i8s
    # ewise_union defaults still have to fit the op's input types.
    v = Vector(i8s, size=2)
    v[0] = [1, 2, 3]
    with pytest.raises(ValueError, match="does not fit _MonoLitI8"):
        v.ewise_union(v, monoid.plus, 0.5, 0)


@pytest.mark.skipif("not supports_udfs")
def test_udt_outer_on_mixed_types():
    """``outer`` builds its semiring from the op, typed on both input types.

    ``get_semiring`` typed the semiring by the op's first input type only, so
    ``outer`` on two different UDTs, or a UDT and a built-in dtype, was a bare
    GrB_DOMAIN_MISMATCH.
    """
    i8s = dtypes.register_anonymous(np.dtype((np.int8, (3,))), "_OuterI8")
    f4s = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_OuterF4")
    v = Vector(i8s, size=2)
    v[0] = [1, 2, 3]
    u = Vector(f4s, size=2)
    u[1] = [0.5, 0.5, 0.5]
    for a, b in [(v, u), (u, v)]:
        result = a.outer(b).new()
        assert result.dtype == f4s
        assert result.to_coo()[2].tolist() == [[0.5, 1.0, 1.5]]
    result = v.outer(Vector.from_coo([0], [2.0], size=2)).new()
    assert result.dtype.np_type == np.dtype((np.float64, (3,)))
    assert result.to_coo()[2].tolist() == [[2.0, 4.0, 6.0]]


@pytest.mark.skipif("not supports_udfs")
def test_udt_jit_kernel_only_for_a_same_type_result():
    """A same-type pair gets the arithmetic JIT kernel only when its result is that type too.

    The kernel declares its result with the operands' type, so for ``truediv``
    on an integer UDT, whose result is float64, it would write integer
    quotients into a float64 buffer. That op runs through the cfunc instead.
    64 entries keep SuiteSparse off its iso and short-vector shortcuts, so a
    JIT kernel would run.
    """
    from graphblas.core.operator.udt_utils import _has_jit_set

    ints = dtypes.register_anonymous(np.dtype((np.int32, (11,))), "_JitDivI11")
    N = 64
    v = Vector(ints, size=N)
    w = Vector(ints, size=N)
    for i in range(N):
        v[i] = np.arange(11) + i
        w[i] = np.full(11, 4)
    result = v.ewise_mult(w, binary.truediv).new()
    assert result.dtype.np_type == np.dtype((np.float64, (11,)))
    for i in [0, 1, N - 1]:
        np.testing.assert_array_equal(result[i].new().value, (np.arange(11) + i) / 4)
    if suitesparse and _has_jit_set:
        assert binary.truediv[ints].jit_c_source is None
        assert binary.plus[ints].jit_c_source is not None


@pytest.mark.skipif("not supports_udfs")
def test_udt_eq_compile_failure_is_a_udfparseerror():
    """``eq`` reports a pair the codegen cannot type as a UdfParseError, as ``plus`` does.

    Its cfunc was compiled without the wrapper that turns Numba's errors into
    a one-line UdfParseError, so a bytes field against a float field surfaced
    as Numba's full TypingError. ``plus`` rejects the pair sooner: bytes and
    float64 have no common dtype for the result field.
    """
    text = dtypes.register_anonymous(np.dtype([("ueq_a", "S4"), ("ueq_b", np.float64)]), "_UeqS")
    num = dtypes.register_anonymous(
        np.dtype([("ueq_a", np.float64), ("ueq_b", np.float64)]), "_UeqF"
    )
    v = Vector(text, size=1)
    v[0] = (b"ab", 1.0)
    w = Vector(num, size=1)
    w[0] = (1.0, 1.0)
    with pytest.raises(UdfParseError, match="binary.eq does not work with"):
        binary.eq(v & w).new()
    with pytest.raises(KeyError, match=r"elements of \|S4 and float64 have no common type"):
        binary.plus(v & w).new()


@pytest.mark.skipif("not supports_udfs")
def test_udt_pair_is_never_unified():
    """Two different UDTs reach the op as they are, never unified to one type.

    numpy promotes two records of the same layout to one of them or to a third,
    which ``unify`` registered as a new type. An op that does not lift to UDTs
    then named that third type in its KeyError, and ``ewise_add`` with a monoid
    passed GraphBLAS two operands it rejected with GrB_DOMAIN_MISMATCH. A
    Monoid takes one type, so on a mixed pair it runs as its BinaryOp.
    """
    X = dtypes.register_anonymous(np.dtype([("uni_a", np.float32), ("uni_b", np.int64)]), "_UniX")
    Y = dtypes.register_anonymous(np.dtype([("uni_a", np.int64), ("uni_b", np.float32)]), "_UniY")
    with pytest.raises(KeyError, match=r"lt does not work with \(_UniX, _UniY\)"):
        binary.lt[X, Y]
    v = Vector(X, size=1)
    v[0] = (1.0, 2)
    w = Vector(Y, size=1)
    w[0] = (1, 2.0)
    for expr in [v.ewise_mult(w, monoid.plus), v.ewise_union(w, monoid.plus, 0, 0)]:
        result = expr.new()
        assert result.dtype.np_type == np.dtype([("uni_a", np.float64), ("uni_b", np.float64)])
        assert result[0].new().value.tolist() == (2.0, 4.0)
    assert v.ewise_mult(w, monoid.max).new()[0].new().value.tolist() == (1.0, 2.0)
    # ewise_add, the Monoid's default, would need converted copies of the operands.
    for expr in [lambda: v.ewise_add(w), lambda: monoid.plus(v | w)]:
        with pytest.raises(DomainMismatch, match="ewise_add cannot use binary.plus"):
            expr()
    with pytest.raises(TypeError, match="Monoid inputs must be the same dtype"):
        monoid.plus[X, Y]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_aggregators():
    """Monoid-based aggregators must auto-extend to UDTs the underlying monoid supports.

    Before: ``v.reduce(agg.sum)`` on a UDT vector failed with
    ``KeyError: 'sum does not work with <udt>'``. Now ``Aggregator.__getitem__``
    triggers UDT compilation of the underlying monoid for monoid-based aggs.
    """
    record = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record, "_AggUdt")
    v = Vector(udt, size=3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)

    assert v.reduce(agg.sum).new() == (9, 12.0)
    assert v.reduce(agg.prod).new() == (15, 48.0)
    assert v.reduce(agg.min).new() == (1, 2.0)
    assert v.reduce(agg.max).new() == (5, 6.0)
    # any_value uses any_dtype=True and works on any input
    assert v.reduce(agg.any_value).new() in [(1, 2.0), (3, 4.0), (5, 6.0)]
    # count is dtype-agnostic
    assert v.reduce(agg.count).new() == 3

    # __contains__ should agree
    assert udt in agg.sum
    assert udt in agg.prod
    assert udt in agg.min
    assert udt in agg.max
    # Composite/semiring-based aggregators aren't auto-lifted
    assert udt not in agg.hypot

    if suitesparse:
        # agg.ss.first / agg.ss.last are positional aggregators (any_dtype=True):
        # they pick an existing entry rather than combining values, so they
        # work on any dtype, UDT included, without per-UDT compilation.
        assert tuple(v.reduce(agg.ss.first).new().value) == (1, 2.0)
        assert tuple(v.reduce(agg.ss.last).new().value) == (5, 6.0)

    # Array UDT path
    adt = np.dtype((np.float64, (3,)))
    audt = dtypes.register_anonymous(adt, "_AggArrUdt")
    a = Vector(audt, size=2)
    a[0] = [1.0, 2.0, 3.0]
    a[1] = [4.0, 5.0, 6.0]
    np.testing.assert_array_equal(a.reduce(agg.sum).new().value, [5.0, 7.0, 9.0])
    np.testing.assert_array_equal(a.reduce(agg.min).new().value, [1.0, 2.0, 3.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_lazy_registration():
    """``lazy=True`` must preserve ``is_udt`` so registration succeeds when fired.

    Regression: the ``module._delayed[funcname]`` kwargs dict didn't include
    ``is_udt``, so the delayed callback compiled the function for standard
    types only and failed with ``UdfParseError``.
    """
    from graphblas.core.operator import BinaryOp, IndexUnaryOp, UnaryOp

    record = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record, "_LazyUdt")

    BinaryOp.register_new("_lazy_udt_add", _pkl_udt_add, is_udt=True, lazy=True)
    UnaryOp.register_new("_lazy_udt_neg", _pkl_udt_neg, is_udt=True, lazy=True)
    IndexUnaryOp.register_new("_lazy_udt_iu", _pkl_udt_get_a, is_udt=True, lazy=True)
    try:
        # Trigger the delayed registration by attribute access; the failure
        # mode was UdfParseError raised inside the delayed callback.
        add_op = binary._lazy_udt_add
        neg_op = unary._lazy_udt_neg
        iu_op = indexunary._lazy_udt_iu
        assert add_op._is_udt
        assert neg_op._is_udt
        assert iu_op._is_udt

        v = Vector(udt, 2)
        v[0] = (1, 2.0)
        v[1] = (3, 4.0)
        assert add_op(v & v).new()[0].new() == (2, 4.0)
        assert v.apply(neg_op).new()[0].new() == (-1, -2.0)
    finally:
        # Remove these UDT-only ops from every namespace they leaked into,
        # including the combined ``op`` (binary and unary register there too).
        # ``test_dir`` enumerates module names, and test_op_namespace iterates
        # ``op._delayed`` and would fail to resolve an op whose per-type entry
        # is gone. Pop ``__dict__`` and ``_delayed`` directly to avoid
        # re-triggering ``__getattr__``.
        for module, name in [
            (binary, "_lazy_udt_add"),
            (unary, "_lazy_udt_neg"),
            (indexunary, "_lazy_udt_iu"),
            (op, "_lazy_udt_add"),
            (op, "_lazy_udt_neg"),
        ]:
            vars(module).pop(name, None)
            module._delayed.pop(name, None)


def _pkl_udt_add(x, y):  # pragma: no cover (numba)
    return (x["a"] + y["a"], x["b"] + y["b"])


def _pkl_udt_neg(x):  # pragma: no cover (numba)
    return (-x["a"], -x["b"])


def _pkl_udt_get_a(x, ix, jx, t):  # pragma: no cover (numba)
    return x["a"]


def _pkl_udt_big_a(x, ix, jx, t):  # pragma: no cover (numba)
    return x["a"] > t["a"]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_op_pickle():
    """Pickle round-trip for typed and untyped UDT operators.

    Catches regressions like:
    - TypedUserMonoid / TypedUserSemiring being built with the cffi ``GrB_*``
      pointer as ``parent`` instead of the Monoid / Semiring Python object.
    - Anonymous reduce paths losing ``is_udt`` when re-registering.
    """
    import pickle

    from graphblas.core.operator import (
        BinaryOp,
        IndexUnaryOp,
        Monoid,
        SelectOp,
        Semiring,
        UnaryOp,
    )

    record = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record, "_PickleUdt")

    bin_op = BinaryOp.register_anonymous(_pkl_udt_add, "_pkl_b", is_udt=True)
    un_op = UnaryOp.register_anonymous(_pkl_udt_neg, "_pkl_u", is_udt=True)
    iu_op = IndexUnaryOp.register_anonymous(_pkl_udt_get_a, "_pkl_iu", is_udt=True)
    sel_op = SelectOp.register_anonymous(_pkl_udt_big_a, "_pkl_s", is_udt=True)
    mon_op = Monoid.register_anonymous(bin_op, (0, 0.0), "_pkl_m")
    sr_op = Semiring.register_anonymous(mon_op, bin_op, "_pkl_sr")

    # Anonymous-op round-trip; verifies `is_udt` flows through `__reduce__`.
    for anon_op in [bin_op, un_op, iu_op, sel_op, mon_op, sr_op]:
        op2 = pickle.loads(pickle.dumps(anon_op))
        assert op2._is_udt is True, f"is_udt lost on {anon_op.name}"

    # Typed UDT instances on user-defined parents.
    for anon_op in [bin_op, un_op, iu_op, sel_op, mon_op, sr_op]:
        typed = anon_op[udt]
        typed2 = pickle.loads(pickle.dumps(typed))
        assert typed2.name == typed.name
        # parent must be the Python op object, not a cffi pointer
        assert isinstance(
            typed2.parent, type(anon_op)
        ), f"{anon_op.name} typed parent had wrong type: {type(typed2.parent).__name__}"

    # Typed UDT instances on built-in monoid or semiring used to fail with
    # ``cannot pickle '_cffi_backend.__CDataOwn'`` because the parent slot
    # held the raw GrB pointer instead of the Python Monoid object.
    pickle.loads(pickle.dumps(monoid.plus[udt]))
    pickle.loads(pickle.dumps(monoid.times[udt]))
    pickle.loads(pickle.dumps(semiring.plus_times[udt]))
    pickle.loads(pickle.dumps(semiring.min_plus[udt]))

    # Vector round-trip with UDT (was already working; this is a sanity check).
    v = Vector(udt, 2)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v2 = pickle.loads(pickle.dumps(v))
    assert v.isequal(v2)


def test_dir():
    for mod in [unary, binary, monoid, semiring, op]:
        assert not set(mod._delayed) - set(dir(mod))


def test_semiring_commute_exists():
    from .conftest import orig_semirings

    vals = {
        semiring._deprecated[key] if key in semiring._deprecated else getattr(semiring, key)
        for key in orig_semirings
    }
    missing = set()
    for key in orig_semirings:
        val = semiring._deprecated[key] if key in semiring._deprecated else getattr(semiring, key)
        commutes_to = val.commutes_to
        if commutes_to is not None and commutes_to not in vals:  # pragma: no cover (debug)
            missing.add(commutes_to.name)
    if missing:
        raise AssertionError("Missing semirings: " + ", ".join(sorted(missing)))


def test_binaryop_commute_exists():
    from .conftest import orig_binaryops

    vals = {
        binary._deprecated[key] if key in binary._deprecated else getattr(binary, key)
        for key in orig_binaryops
    }
    missing = set()
    for key in orig_binaryops:
        val = binary._deprecated[key] if key in binary._deprecated else getattr(binary, key)
        commutes_to = val.commutes_to
        if commutes_to is not None and commutes_to not in vals:  # pragma: no cover (debug)
            missing.add(commutes_to.name)
    if missing:
        raise AssertionError("Missing binaryops: " + ", ".join(sorted(missing)))


@pytest.mark.skipif("not supports_udfs")
def test_binom():
    v = Vector.from_coo([0, 1, 2], [3, 4, 5])
    result = v.apply(binary.binom, 2).new()
    expected = Vector.from_coo([0, 1, 2], [3, 6, 10])
    assert result.isequal(expected)
    assert op.binom is binary.binom


def test_builtins():
    v1 = Vector.from_coo([0, 1, 2], [1, 2, 3])
    v2 = Vector.from_coo([0, 1, 2], [3, 2, 1])
    result = v1.ewise_mult(v2, min).new()
    expected = Vector.from_coo([0, 1, 2], [1, 2, 1])
    assert result.isequal(expected)
    v1(max) << v2
    expected = Vector.from_coo([0, 1, 2], [3, 2, 3])
    assert v1.isequal(expected)


def test_op_ss():
    if suitesparse:
        gb.unary.ss.positioni
        gb.binary.ss.firsti
        gb.semiring.ss.max_secondj
        gb.op.ss.positionj
        gb.agg.ss.argmin
    else:
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.unary.ss
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.binary.ss
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.semiring.ss
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.op.ss
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.agg.ss


def test_deprecated():
    with pytest.warns(DeprecationWarning, match="please use"):
        gb.unary.erf
    with pytest.warns(DeprecationWarning, match="please use `gb.indexunary.rowindex`"):
        gb.unary.positioni
    with pytest.warns(DeprecationWarning, match="please use"):
        gb.binary.firsti
    with pytest.warns(DeprecationWarning, match="please use"):
        gb.semiring.min_firsti
    with pytest.warns(DeprecationWarning, match="please use"):
        gb.op.secondj
    with pytest.warns(DeprecationWarning, match="please use"):
        gb.agg.argmin


@pytest.mark.slow
def test_is_idempotent():
    assert monoid.min.is_idempotent
    assert monoid.max[int].is_idempotent
    assert monoid.lor.is_idempotent
    assert monoid.band.is_idempotent
    if shouldhave(monoid.numpy, "gcd"):
        assert monoid.numpy.gcd.is_idempotent
    assert not monoid.plus.is_idempotent
    assert not monoid.times[float].is_idempotent
    if config["mapnumpy"] or shouldhave(monoid.numpy, "equal"):
        assert not monoid.numpy.equal.is_idempotent
    with pytest.raises(AttributeError):
        binary.min.is_idempotent


def _isidem_factory(scale):  # pragma: no cover (called by Numba)
    # Plain arithmetic so the binop compiles for every builtin type that
    # ``BinaryOp._build`` samples, including complex (no ``>`` lowering).
    def inner(x, y):
        return (x + y) * scale

    return inner


def _isidem_pick_first(x, y):  # pragma: no cover (called by Numba)
    # Returning ``x`` is trivially idempotent (``op(x, x) == x``) and
    # compiles for every builtin type, including complex where ``max``
    # has no Numba lowering.
    return x


@pytest.mark.skipif("not supports_udfs")
def test_monoid_pickle_preserves_is_idempotent():
    """Regression: anonymous ``Monoid`` and ``ParameterizedMonoid`` both
    dropped ``is_idempotent`` from the pickle round-trip, silently turning
    a known-idempotent op into a non-idempotent one. Both
    ``Monoid.__reduce__`` and ``ParameterizedMonoid.__reduce__`` now carry
    the flag explicitly so ``_deserialize`` can pass it to
    ``register_anonymous`` / ``register_new``.
    """
    import pickle

    # Plain Monoid (non-parameterized) over an anonymous BinaryOp.
    bin_op = BinaryOp.register_anonymous(_isidem_pick_first, "isidem_pick_first")
    mon = Monoid.register_anonymous(bin_op, 0, "isidem_monoid", is_idempotent=True)
    assert mon.is_idempotent is True
    assert pickle.loads(pickle.dumps(mon)).is_idempotent is True

    # ParameterizedMonoid wrapping a ParameterizedBinaryOp.
    pbin = BinaryOp.register_anonymous(_isidem_factory, parameterized=True)
    pmon = Monoid.register_anonymous(pbin, 0, "isidem_param_monoid", is_idempotent=True)
    assert pmon.is_idempotent is True
    assert pickle.loads(pickle.dumps(pmon)).is_idempotent is True


def _parameterized_is_udt_factory(scale):  # pragma: no cover (called by Numba inside the op)
    def inner(x, y):
        return x * scale + y * scale

    return inner


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.parametrize(
    "module_name",
    ["unary", "binary", "indexunary", "select", "indexbinary"],
)
def test_parameterized_is_udt_pickle_roundtrip(module_name):
    """Parameterized + ``is_udt=True`` propagates through ``__reduce__``.

    Regression: ``Parameterized{Unary,Binary,IndexUnary,Select,IndexBinary}Op.__reduce__``
    used to emit ``(name, func, anonymous)``, so ``_deserialize`` invoked
    ``register_*(..., parameterized=True)`` without ``is_udt``. Cross-process
    re-register lost the flag, then dispatch took the non-UDT compile path
    and failed at first use.
    """
    import pickle

    module = getattr(gb, module_name)
    op = module.register_anonymous(_parameterized_is_udt_factory, parameterized=True, is_udt=True)
    assert op._is_udt is True
    op2 = pickle.loads(pickle.dumps(op))
    assert op2._is_udt is True


def test_ops_have_ss():
    modules = [unary, binary, monoid, semiring, indexunary, select, op]
    if suitesparse:
        for mod in modules:
            assert mod.ss is not None
    else:
        for mod in modules:
            with pytest.raises(AttributeError):
                mod.ss


@pytest.mark.skipif("not supports_udfs")
def test_compile_codegen_helper():
    """The ``_compile_codegen`` helper validates source and surfaces typos clearly.

    Codegen bugs used to surface as a cryptic ``SyntaxError`` from ``exec``
    or, worse, as a Numba ``TypingError`` at first use of the generated
    function. The helper catches them at the call site with the offending
    source attached, and registers each generated function with
    ``linecache`` so any later traceback shows real lines instead of
    ``<string>``.
    """
    import linecache

    from graphblas.core.operator.udt_utils import _compile_codegen

    fn = _compile_codegen(
        "def _op(x, y):\n    return x + y\n",
        func_name="_op",
        source_label="<gb-udt-helper-test plus>",
    )
    assert fn(2, 3) == 5
    # The synthetic filename is registered with linecache so a traceback
    # raised from inside the generated function points at real source.
    co_filename = fn.__code__.co_filename
    assert co_filename.startswith("<gb-udt-helper-test plus> #")
    assert "x + y" in "".join(linecache.cache[co_filename][2])

    # A bad source surfaces as RuntimeError with the offending source attached
    # and the underlying SyntaxError as ``__cause__``.
    bad_src = "def _op(x, y):\n    return (x + y\n"  # missing close paren
    with pytest.raises(RuntimeError) as exc_info:
        _compile_codegen(
            bad_src,
            func_name="_op",
            source_label="<gb-udt-helper-test typo>",
        )
    msg = str(exc_info.value)
    assert "<gb-udt-helper-test typo>" in msg
    assert "not valid Python" in msg
    assert "Source:" in msg
    assert bad_src in msg
    assert isinstance(exc_info.value.__cause__, SyntaxError)


def test_operator_namespace_typo_suggestions():
    # A typo in an operator namespace should suggest close matches (via difflib),
    # drawn from __dir__() so lazily-registered operators are offered without
    # forcing them to build.
    with pytest.raises(AttributeError, match="has no attribute 'pluss'.*Did you mean 'plus'"):
        binary.pluss
    with pytest.raises(AttributeError, match="Did you mean 'plus'"):
        monoid.pluss
    with pytest.raises(AttributeError, match="plus_times"):
        semiring.plus_time
    with pytest.raises(AttributeError, match="Did you mean 'sum'"):
        agg.summ
    with pytest.raises(AttributeError, match="Did you mean"):
        unary.expp
    with pytest.raises(AttributeError, match="rowindex"):
        indexunary.rowindexx
    with pytest.raises(AttributeError, match="triu"):
        select.triu_typo
    with pytest.raises(AttributeError, match="Did you mean 'plus'"):
        op.pluss

    # No close match -> plain message, no suggestion appended
    with pytest.raises(AttributeError) as exc_info:
        binary.zzzzzz
    assert "has no attribute 'zzzzzz'" in str(exc_info.value)
    assert "Did you mean" not in str(exc_info.value)

    # Building suggestions must not force lazy operators to compile
    before = set(binary._delayed)
    with pytest.raises(AttributeError):
        binary.pluss
    assert set(binary._delayed) == before


# Touching an operator namespace must not compile any lazily-registered UDF.
# Run in a subprocess: BinaryOp._initialize runs once per process, so by the
# time any test executes, the import-time behavior under test is long past.
_LAZY_UDF_PROBE = """
import graphblas as gb

gb.binary.plus  # forces BinaryOp._initialize

lazy_udfs = ("floordiv", "rfloordiv", "absfirst", "abssecond", "rpow")
print("package: " + gb.__file__)
print("still lazy: " + " ".join(n for n in lazy_udfs if n in gb.binary._delayed))
"""


@pytest.mark.skipif("not supports_udfs")
def test_initialize_does_not_build_lazy_udfs():
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(
        [sys.executable, "-c", _LAZY_UDF_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report

    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    # Assert the child probed this tree before trusting what it reports about it.
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report

    still_lazy = lines.get("still lazy", "").split()
    assert still_lazy == ["floordiv", "rfloordiv", "absfirst", "abssecond", "rpow"], report


@pytest.mark.skipif("not supports_udfs")
def test_builtin_udfs_are_disk_cached():
    # The built-in UDF binops are module-level functions, so numba can persist
    # their compilation across processes. Anything a user registers cannot be:
    # numba keys a cache entry on a stable on-disk source, which a lambda or an
    # interactively defined function does not have.
    from numba.core.caching import NullCache

    for name in ["floordiv", "rfloordiv", "absfirst", "abssecond", "rpow"]:
        numba_func = getattr(binary, name)._numba_func
        assert not isinstance(numba_func._cache, NullCache), name

    def _uncached_probe(x, y):
        return x + y

    user_op = BinaryOp.register_anonymous(_uncached_probe)
    assert isinstance(user_op._numba_func._cache, NullCache)


# The built-in UDF binops advertise every dtype in ``.types`` up front but
# compile none of them until asked. Run in a subprocess: any earlier test may
# already have materialized them in this process.
_DEFERRED_BUILD_PROBE = """
import graphblas as gb

op = gb.binary.floordiv
print("package: " + gb.__file__)
print("types: %d" % len(op.types))
print("compiled before: %d" % len(op._typed_ops))
op[gb.dtypes.INT64]
print("compiled after: %d" % len(op._typed_ops))
"""


@pytest.mark.skipif("not supports_udfs")
def test_builtin_udf_types_precede_compilation():
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(
        [sys.executable, "-c", _DEFERRED_BUILD_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report

    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report

    assert int(lines["types"]) == len(binary.floordiv.types), report
    assert int(lines["compiled before"]) == 0, report
    assert int(lines["compiled after"]) == 1, report


_DEFERRED_COMMUTES_PROBE = """
import graphblas as gb

print("package: " + gb.__file__)
ct = gb.binary.floordiv[gb.dtypes.INT64].commutes_to
print("floordiv_ok: " + str(ct is gb.binary.rfloordiv[gb.dtypes.INT64]))
ct = gb.binary.absfirst[gb.dtypes.INT64].commutes_to
print("absfirst_ok: " + str(ct is gb.binary.abssecond[gb.dtypes.INT64]))
"""


@pytest.mark.skipif("not supports_udfs")
def test_deferred_commutes_to():
    # A deferred partner op must still answer commutes_to. The membership
    # test consults .types, not ._typed_ops: with the latter, a fresh process
    # answered None for floordiv[INT64].commutes_to until rfloordiv happened
    # to be compiled, so the answer depended on access order. Subprocess for
    # the same reason as test_initialize_does_not_build_lazy_udfs: in this
    # process the partners may already be built.
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(
        [sys.executable, "-c", _DEFERRED_COMMUTES_PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report
    assert lines.get("floordiv_ok") == "True", report
    assert lines.get("absfirst_ok") == "True", report


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_ret_dtype_names_an_output_udt():
    """``ret_dtype`` names an output UDT that is not one of the operands.

    Without it the return type is inferred from what the UDF builds, and the
    only names in scope are the input dtypes. That makes an op that shortens
    its element, such as FP64[10] -> FP64[3], unreachable: the inferred type
    is the input's, and the shape check then rejects the shorter array the
    UDF returns.
    """
    # The input shape is unique to this test; see the note in
    # test_udt_array_udf_shape_errors.
    ten = dtypes.register_anonymous(np.dtype((np.float64, (10,))), "_RetD10")
    three = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "_RetD3")

    def _first_three(x):  # pragma: no cover (numba)
        return x[:3]

    without = UnaryOp.register_anonymous(_first_three, "_ret_dtype_without", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(3,\) when run on sample values"):
        without[ten]

    op_ = UnaryOp.register_anonymous(_first_three, "_ret_dtype_with", is_udt=True, ret_dtype=three)
    assert op_[ten].return_type is three

    # lazy=True must hand ret_dtype to the registration it defers, or the op
    # that fires on first attribute access is built without it.
    UnaryOp.register_new("_ret_dtype_lazy", _first_three, is_udt=True, lazy=True, ret_dtype=three)
    try:
        assert unary._ret_dtype_lazy[ten].return_type is three
    finally:
        # Same cleanup as test_udt_lazy_registration: the op lands in ``unary``
        # and ``op``, and both keep a ``_delayed`` entry until it fires.
        for module in (unary, op):
            vars(module).pop("_ret_dtype_lazy", None)
            module._delayed.pop("_ret_dtype_lazy", None)

    v = Vector(ten, size=2)
    v[0] = np.arange(10.0)
    w = op_(v).new()
    assert w.dtype is three
    np.testing.assert_array_equal(w[0].new().value, np.arange(3.0))


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_ret_dtype_binary_and_record():
    """``ret_dtype`` on a binary op, including a type that is neither operand."""
    a4 = dtypes.register_anonymous(np.dtype((np.float64, (4,))), "_RetDBinA4")
    b6 = dtypes.register_anonymous(np.dtype((np.float64, (6,))), "_RetDBinB6")
    out2 = dtypes.register_anonymous(np.dtype((np.float64, (2,))), "_RetDBinOut2")

    def _head_two(x, y):  # pragma: no cover (numba)
        return x[:2] + y[:2]

    op_ = BinaryOp.register_anonymous(_head_two, "_ret_dtype_bin", is_udt=True, ret_dtype=out2)
    assert op_[a4, b6].return_type is out2

    # ret_dtype equal to an operand's dtype agrees with what inference picks.
    def _plus(x, y):  # pragma: no cover (numba)
        return x + y

    same = BinaryOp.register_anonymous(_plus, "_ret_dtype_same", is_udt=True, ret_dtype=a4)
    inferred = BinaryOp.register_anonymous(_plus, "_ret_dtype_inferred", is_udt=True)
    assert same[a4, a4].return_type is inferred[a4, a4].return_type is a4

    # A record UDT output that appears in no operand.
    rec = dtypes.register_anonymous(
        np.dtype([("lo", np.float64), ("hi", np.float64)], align=True), "_RetDRec"
    )

    def _bounds(x, y):  # pragma: no cover (numba)
        return (min(x, y), max(x, y))

    recop = BinaryOp.register_anonymous(_bounds, "_ret_dtype_rec", is_udt=True, ret_dtype=rec)
    assert recop[FP64, FP64].return_type is rec

    w = Vector(FP64, size=2)
    w[0] = 3.0
    u = Vector(FP64, size=2)
    u[0] = 1.0
    res = recop(w & u).new()
    assert res.dtype is rec
    assert tuple(res[0].new().value.tolist()) == (1.0, 3.0)


@pytest.mark.skipif("not supports_udfs")
def test_udt_ret_dtype_errors():
    """``ret_dtype`` is rejected outside the UDT path and for unrecognized dtypes."""
    with pytest.raises(ValueError, match="not a recognized dtype"):
        UnaryOp.register_anonymous(lambda x: x, "_ret_dtype_junk", is_udt=True, ret_dtype="NOPE")
    with pytest.raises(ValueError, match="not a recognized dtype"):
        UnaryOp.register_anonymous(lambda x: x, "_ret_dtype_junk2", is_udt=True, ret_dtype=object())

    # The builtin path derives a return type per input dtype, so one fixed
    # dtype cannot describe it.
    with pytest.raises(ValueError, match="ret_dtype requires is_udt=True"):
        UnaryOp.register_anonymous(lambda x: x, "_ret_dtype_builtin", ret_dtype=FP64)
    with pytest.raises(ValueError, match="ret_dtype requires is_udt=True"):
        BinaryOp.register_anonymous(lambda x, y: x + y, "_ret_dtype_builtin2", ret_dtype=FP64)

    # lazy=True must not defer the validation: a bad combination fails at the
    # registration site, not at first attribute touch of the delayed op.
    with pytest.raises(ValueError, match="parameterized=True"):
        UnaryOp.register_new(
            "_ret_dtype_lazy_param",
            lambda x: x,
            parameterized=True,
            is_udt=True,
            lazy=True,
            ret_dtype=FP64,
        )

    # A parameterized operator builds its function when called; ret_dtype
    # belongs to the register call for that function.
    with pytest.raises(ValueError, match="does not work with parameterized=True"):
        UnaryOp.register_anonymous(
            lambda t=1: (lambda x: x + t),
            "_ret_dtype_param",
            parameterized=True,
            is_udt=True,
            ret_dtype=FP64,
        )

    # SelectOp takes no ret_dtype: GraphBLAS fixes its return type to BOOL.
    with pytest.raises(TypeError, match="ret_dtype"):
        SelectOp.register_anonymous(
            lambda x, i, j, t: x > t, "_ret_dtype_select", is_udt=True, ret_dtype=BOOL
        )


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_ret_dtype_index_ops():
    """``ret_dtype`` reaches the IndexUnaryOp and IndexBinaryOp compile paths too."""
    # Shape (11,) is used nowhere else: anonymous UDTs share one DataType per
    # np.dtype, so a shape reused across tests would inherit whichever JIT C
    # state was frozen first under random test ordering.
    in11 = dtypes.register_anonymous(np.dtype((np.float64, (11,))), "_RetDIdx11")
    out2 = dtypes.register_anonymous(np.dtype((np.float64, (2,))), "_RetDIdx2")

    def _head_plus_row(x, i, j, t):  # pragma: no cover (numba)
        return x[:2] + i

    iu = IndexUnaryOp.register_anonymous(
        _head_plus_row, "_ret_dtype_indexunary", is_udt=True, ret_dtype=out2
    )
    assert iu[in11, INT64].return_type is out2

    if lib.__dict__.get("GxB_IndexBinaryOp_new") is not None:
        from graphblas.core.operator import IndexBinaryOp

        def _head_sum(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
            return x[:2] + y[:2]

        ib = IndexBinaryOp.register_anonymous(
            _head_sum, "_ret_dtype_indexbinary", is_udt=True, ret_dtype=out2
        )
        assert ib[in11, in11].return_type is out2

    # register_new also adds an IndexUnaryOp to ``select`` when it returns BOOL.
    # A UDT op has no types until it is compiled, so a declared non-BOOL
    # ret_dtype is what keeps this one out.
    IndexUnaryOp.register_new("_ret_dtype_iu_named", _head_plus_row, is_udt=True, ret_dtype=out2)
    try:
        assert "_ret_dtype_iu_named" not in vars(select)
    finally:
        vars(indexunary).pop("_ret_dtype_iu_named", None)
        vars(select).pop("_ret_dtype_iu_named", None)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_ret_dtype_still_shape_checked():
    """The registration-time shape probe checks against the declared ``ret_dtype``.

    Short-circuiting the return-type inference must not take the probe with
    it. Declaring the output type makes the check worth more, not less: the
    inferred case can only ever compare a UDF against a type derived from
    what it returned, while a declared type is an independent claim the UDF
    can contradict.
    """
    # Shape (14,) is used nowhere else: anonymous UDTs share one DataType per
    # np.dtype, so a shape reused across tests would inherit whichever JIT C
    # state was frozen first under random test ordering.
    in14 = dtypes.register_anonymous(np.dtype((np.float64, (14,))), "_RetDPrb14")
    out2 = dtypes.register_anonymous(np.dtype((np.float64, (2,))), "_RetDPrb2")

    def _three(x):  # pragma: no cover (numba)
        return x[:3]

    # Three elements cannot fill a declared two-element output.
    op_ = UnaryOp.register_anonymous(_three, "_ret_dtype_prb_bad", is_udt=True, ret_dtype=out2)
    with pytest.raises(UdfParseError, match=r"shape \(3,\).*_RetDPrb2 elements are \(2,\)"):
        op_[in14]

    # The check is fit-by-broadcast, not equality: a one-element return
    # legitimately fills every slot of the declared element.
    def _one(x):  # pragma: no cover (numba)
        return x[:1]

    ok = UnaryOp.register_anonymous(_one, "_ret_dtype_prb_bcast", is_udt=True, ret_dtype=out2)
    assert ok[in14].return_type is out2

    # An operand returned as-is keeps its extents in its Numba type, so the
    # declared shape is checked without running the UDF (no "sample values").
    def _as_is(x):  # pragma: no cover (numba)
        return x

    as_is = UnaryOp.register_anonymous(_as_is, "_ret_dtype_prb_as_is", is_udt=True, ret_dtype=out2)
    with pytest.raises(UdfParseError, match=r"shape \(14,\), but _RetDPrb2 elements are \(2,\)"):
        as_is[in14]

    # The record half of the probe checks the declared type's array leaves.
    rec = dtypes.register_anonymous(
        np.dtype([("v", np.float64, (4,)), ("n", np.int64)], align=True), "_RetDPrbRec"
    )

    def _short_leaf(x, y):  # pragma: no cover (numba)
        return (x[:2], 1)

    recop = BinaryOp.register_anonymous(
        _short_leaf, "_ret_dtype_prb_rec", is_udt=True, ret_dtype=rec
    )
    with pytest.raises(UdfParseError, match=r"shape \(2,\) for field \['v'\] of _RetDPrbRec"):
        recop[in14, in14]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
# _RetDMisNested's numpy repr is 141 chars; see test_udt_eq_nested_record_with_nan_leaf.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_ret_dtype_rejects_a_return_it_cannot_hold():
    """A declared ``ret_dtype`` is checked against what the UDF returns.

    Inference only names a type the return can be written as, and a declared
    type skips it. Unchecked, a mismatch reached the wrapper, which either
    failed to compile with a raw Numba error or, worse, ran and gave a wrong
    answer with no error: a longer tuple lost its extra values, an array
    written to a scalar kept its first element, and a tuple written to an
    array element left it uninitialized.
    """
    # Names and shapes unique to this test; see the note in
    # test_udt_array_udf_shape_errors.
    rec2 = dtypes.register_anonymous(
        np.dtype([("rdm_lo", np.float64), ("rdm_hi", np.float64)], align=True), "_RetDMisRec2"
    )
    rec3 = dtypes.register_anonymous(
        np.dtype([("rdm_a", np.int64), ("rdm_b", np.int64), ("rdm_c", np.int64)], align=True),
        "_RetDMisRec3",
    )
    arr2 = dtypes.register_anonymous(np.dtype((np.float32, (2,))), "_RetDMisArr2")
    arr3 = dtypes.register_anonymous(np.dtype((np.float32, (3,))), "_RetDMisArr3")

    def binary_rejects(func, ret_dtype, match):
        op_ = BinaryOp.register_anonymous(func, is_udt=True, ret_dtype=ret_dtype)
        with pytest.raises(UdfParseError, match=match):
            op_[FP64, FP64]

    binary_rejects(
        lambda x, y: (x, y, x + y), rec2, "tuple of length 3, but ret_dtype=_RetDMisRec2 has 2"
    )
    binary_rejects(lambda x, y: (x,), rec2, "tuple of length 1, but ret_dtype=_RetDMisRec2 has 2")
    binary_rejects(
        lambda x, y: x + y, rec2, "returned float64, which cannot be written as ret_dtype"
    )

    # A nested record takes a flat tuple, one value per leaf, and the count in
    # the message says so when it differs from the number of top-level fields.
    nested = dtypes.register_anonymous(
        np.dtype(
            [("rdm_in", [("rdm_p", np.float64), ("rdm_q", np.float64)]), ("rdm_r", np.float64)],
            align=True,
        ),
        "_RetDMisNested",
    )
    binary_rejects(
        lambda x, y: ((x, y), x + y),
        nested,
        "tuple of length 2, but ret_dtype=_RetDMisNested has 3 fields once nested records",
    )
    flat = BinaryOp.register_anonymous(lambda x, y: (x, y, x + y), is_udt=True, ret_dtype=nested)
    assert flat[FP64, FP64].return_type is nested
    binary_rejects(lambda x, y: (x, y, x + y), arr2, "tuple of length 3, which cannot be written")
    binary_rejects(lambda x, y: (x, y), FP64, "tuple of length 2, which cannot be written")

    def unary_rejects(func, ret_dtype, operand, match):
        op_ = UnaryOp.register_anonymous(func, is_udt=True, ret_dtype=ret_dtype)
        with pytest.raises(UdfParseError, match=match):
            op_[operand]

    unary_rejects(lambda x: x * 1.0, FP64, arr3, "returned an array, which cannot be written")
    unary_rejects(lambda x: x, rec3, rec2, r"record with fields \['rdm_lo', 'rdm_hi'\]")
    unary_rejects(lambda x: x, arr3, rec2, "record with fields")
    unary_rejects(lambda x: x, rec2, arr3, "returned an array, which cannot be written")

    # What a declared type can hold still works: one value per field, a
    # scalar that fills an array element, and a scalar cast to the declared type.
    ok = BinaryOp.register_anonymous(lambda x, y: (x, y), is_udt=True, ret_dtype=rec2)
    fill = BinaryOp.register_anonymous(lambda x, y: x + y, is_udt=True, ret_dtype=arr2)
    cast = BinaryOp.register_anonymous(lambda x, y: x + y, is_udt=True, ret_dtype=INT64)
    w = Vector(FP64, size=1)
    w[0] = 3.0
    u = Vector(FP64, size=1)
    u[0] = 1.0
    assert tuple(w.ewise_mult(u, ok).new()[0].new().value.tolist()) == (3.0, 1.0)
    np.testing.assert_array_equal(w.ewise_mult(u, fill).new()[0].new().value, [4.0, 4.0])
    assert w.ewise_mult(u, cast).new()[0].new().value == 4


def _ret_dtype_pickle_udf(x):  # pragma: no cover (numba)
    return x[:5]


_RET_DTYPE_PICKLE_WRITER = """
import pickle
import sys

import numpy as np

import graphblas as gb
from graphblas.core.operator.unary import UnaryOp
from graphblas.tests.test_op import _ret_dtype_pickle_udf

print("package: " + gb.__file__)
five = gb.dtypes.register_anonymous(np.dtype((np.float32, (5,))), "_RetPickle5")
UnaryOp.register_new("_ret_dtype_pickled", _ret_dtype_pickle_udf, is_udt=True, ret_dtype=five)
anon = UnaryOp.register_anonymous(
    _ret_dtype_pickle_udf, "_ret_dtype_pickled_anon", is_udt=True, ret_dtype=five
)
with open(sys.argv[1], "wb") as f:
    pickle.dump((gb.unary._ret_dtype_pickled, anon), f)
print("wrote: ok")
"""


_RET_DTYPE_PICKLE_READER = """
import pickle
import sys

import numpy as np

import graphblas as gb

print("package: " + gb.__file__)
with open(sys.argv[1], "rb") as f:
    named, anon = pickle.load(f)
ten = gb.dtypes.register_anonymous(np.dtype((np.float32, (10,))), "_RetPickle10")
print("named shape: " + str(named[ten].return_type.np_type.subdtype[1]))
print("anon shape: " + str(anon[ten].return_type.np_type.subdtype[1]))
"""


@pytest.mark.skipif("not supports_udfs")
def test_udt_ret_dtype_survives_pickle(tmp_path):
    # In-process unpickling takes the _find shortcut and returns the very
    # same object, so only a cross-process round trip exercises what pickle
    # exists for: the reduce tuple must carry ret_dtype, or the reconstructed
    # op re-registers without it and fails its own shape probe. Both sides
    # run in subprocesses so the named registration never lands in this
    # process's operator namespace (test_operator_types enumerates it).
    payload = tmp_path / "ops.pkl"
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    for probe, checks in (
        (_RET_DTYPE_PICKLE_WRITER, {"wrote": "ok"}),
        (_RET_DTYPE_PICKLE_READER, {"named shape": "(5,)", "anon shape": "(5,)"}),
    ):
        result = subprocess.run(
            [sys.executable, "-c", probe, str(payload)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert result.returncode == 0, report
        lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
        assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report
        for key, want in checks.items():
            assert lines.get(key) == want, report

    # An op without ret_dtype pickles as the 3-tuple it always did, which
    # versions without ret_dtype can still read.
    plain = UnaryOp.register_anonymous(_ret_dtype_pickle_udf, "_ret_dtype_plain", is_udt=True)
    assert len(plain.__reduce__()[1]) == 3

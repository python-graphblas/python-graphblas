import itertools

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
from graphblas.exceptions import DomainMismatch, UdfParseError

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
    result = v.apply(_udt_index, 3).new()
    expected = Vector.from_coo([0, 1, 2], [3, -3, -3])
    assert result.isequal(expected)

    def _udt_first(x, y):
        return x  # pragma: no cover (numba)

    udt_first = BinaryOp.register_anonymous(_udt_first, "udt_first", is_udt=True)
    assert udt in udt_first
    assert operator.get_typed_op(udt_first, udt) is udt_first[udt]
    assert udt_first(v & w).new().isequal(v)
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
    result = v.select(sel, 0).new()
    assert result.nvals == 0
    assert result.dtype == v.dtype
    result = w.select(sel, 0).new()
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
    ``jit`` parameter would run the cfunc and report a pass. Any repair of
    conda-baked compiler paths has already happened in
    ``_auto_fix_jit_at_import``; calling ``fix_jit_config`` again from here
    would rewrite process-wide compiler settings that no fixture restores.

    ``jit_compiler_is_usable`` alone is not enough: it only checks that the
    configured compiler path exists on disk, and a runner can have the file
    yet fail every compile (broken toolchain, missing headers). The
    import-time probe already did a real compile, and SuiteSparse demotes
    ``jit_c_control`` from ``'on'`` when that compile fails, so a control
    still ``'on'`` here is the probe's success flag; ``test_ssjit`` keys its
    skips on the same signal. The fixture calls this before it mutates the
    control, and the cache keeps later per-test mutations from flipping it.
    """
    if not _jit_can_compile_cache:
        _jit_can_compile_cache.append(
            gb.ss.jit_compiler_is_usable() and gb.ss.config["jit_c_control"] == "on"
        )
    return _jit_can_compile_cache[0]


@pytest.fixture(params=["jit", "cfunc"])
def udt_op_path(request):
    """Pin SuiteSparse to one execution path for built-in UDT operators.

    Each auto-lifted UDT op carries both a JIT C definition and a Numba
    cfunc, and SuiteSparse chooses between them per call depending on
    whether a C compiler is available. A machine with one and a machine
    without therefore run different code, so results have to hold on both.
    """
    path = request.param
    if backend != "suitesparse" or "jit_c_control" not in gb.ss.config:
        if path == "jit":
            pytest.skip("no SuiteSparse JIT on this backend")
        yield path
        return
    previous = gb.ss.config["jit_c_control"]
    if path == "jit":
        if not _jit_can_compile():
            pytest.skip("JIT compilation not available (probe failed or compiler missing)")
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
    JIT kernel, so callers pass varied data.
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
    the JIT C kernel on ``min(0.0, -0.0)``. Both execution paths are checked
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
    """``binary.truediv`` on integer fields must divide the way Python does.

    Regression: the JIT kernel emitted C ``/``, which is integer division.
    ``10**18 / 3`` came out as 333333333333333333 under the JIT and
    333333333333333312 (float64, like numpy) through the cfunc, so the same
    program gave different answers depending on whether a C compiler was
    installed.
    """
    udt = dtypes.register_anonymous(
        np.dtype([("tdv_i", np.int64), ("tdv_j", np.int64)], align=True), "_TrueDivIntUDT"
    )
    xs = [10**18, 10**18 + 1, 7, 22]
    ys = [3, 3, 2, 7]
    expected = (np.array(xs, np.int64) / np.array(ys, np.int64)).astype(np.int64)
    assert expected[0] == 333333333333333312  # not 333333333333333333
    v, w = _udt_vectors(udt, xs, ys)
    result = binary.truediv(v & w).new()
    got = [result[i].new().value[0] for i in range(len(xs))]
    assert got == list(expected), udt_op_path


@pytest.mark.skipif("not supports_udfs")
def test_udt_floordiv_matches_numpy_on_floats(udt_op_path):
    """``binary.floordiv`` on float fields is not ``floor(a / b)``.

    Regression: the JIT kernel computed ``floor(a / b)``, which rounds
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
        # ``copysign`` branch the JIT kernel emits for an exact-zero result.
        np.testing.assert_array_equal(
            np.signbit(got),
            np.signbit(expected),
            err_msg=f"{udt_op_path} {np_dtype.__name__} sign of zero",
        )


@pytest.mark.skipif("not supports_udfs")
def test_udt_integer_division_by_zero_is_defined(udt_op_path):
    """Integer division by zero must return a value rather than trap.

    The JIT kernel divided in integers, where a zero divisor is undefined
    behaviour: on x86-64 ``idiv`` raises #DE, which is SIGFPE and process
    death rather than an exception. AArch64's ``sdiv`` returns 0 and does not
    trap, so this cannot be exhibited on an arm64 machine. The same trap
    fires on ``INT_MIN / -1``, whose quotient is not representable.

    The values below are a choice, not a discovery. A quotient that doesn't
    fit the field is an undefined conversion in both C and LLVM, and the two
    answered differently per field width, so both paths now rule those cases
    out: a zero divisor gives 0, as ``np.floor_divide`` does, and
    ``INT_MIN // -1`` wraps to ``INT_MIN``, as numpy does.
    """
    signed = dtypes.register_anonymous(
        np.dtype([("dvz_a", np.int32), ("dvz_b", np.int8)], align=True), "_DivZeroSignedUDT"
    )
    unsigned = dtypes.register_anonymous(
        np.dtype([("dvz_c", np.uint32), ("dvz_d", np.uint64)], align=True), "_DivZeroUnsignedUDT"
    )
    v, w = _udt_vectors(signed, [7, -7, 100, -128], [0, 0, 3, -1])
    for gb_op in (binary.truediv, binary.floordiv):
        result = gb_op(v & w).new()
        got = [result[i].new().value[0] for i in range(4)]
        assert got[:2] == [0, 0], f"{udt_op_path} {gb_op.name}"
        assert got[2] == 33, f"{udt_op_path} {gb_op.name}"  # 100 / 3 truncates either way
    # ``-128 // -1`` is the second trapping case; numpy wraps it to INT8_MIN.
    result = binary.floordiv(v & w).new()
    assert result[3].new().value[1] == np.iinfo(np.int8).min, udt_op_path

    v, w = _udt_vectors(unsigned, [7, 9, 100, 5], [0, 0, 3, 2])
    for gb_op in (binary.truediv, binary.floordiv):
        result = gb_op(v & w).new()
        got = [result[i].new().value[0] for i in range(4)]
        assert got == [0, 0, 33, 2], f"{udt_op_path} {gb_op.name}"

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
    while the JIT kernel returned numpy's infinities. Reading back the
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
    and the JIT C generators, so each fix has to land in both.
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
    assert got == [333333333333333312, 0, 0, 33, -4, 2], udt_op_path
    result = binary.floordiv(v & w).new()
    got = [result[i].new().value[0] for i in range(len(ixs))]
    assert got == [333333333333333333, 0, 0, 33, -5, 2], udt_op_path


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_multidim_array_ops_match_numpy(udt_op_path):
    """Built-in ops on a multi-dimensional array UDT agree with numpy on both paths.

    The JIT C typedef flattens any rank to ``double v [N]`` and the Numba
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
        ("truediv", [(0, 0.1), (0, 0.1), (0, 0.1)]),
    ],
)
@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_builtin_binary_record(record_udt, op_name, expected_rows):
    """Per-field arithmetic on a record UDT matches the scalar definition."""
    v, w = _record_pair(record_udt)
    result = getattr(binary, op_name)(v & w).new()
    assert result.isequal(_record_expected(record_udt, expected_rows))


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

    # Array UDT
    arr_dtype = np.dtype((np.float64, (7,)))
    arr_udt = dtypes.register_anonymous(arr_dtype, "Vec7")
    lib.GrB_Type_get_String(arr_udt._carg, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "double v [7]" in defn
    assert "Vec7" in defn

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

    # Array UDT JIT definitions
    arr_dtype = np.dtype((np.float64, (5,)))
    arr_udt = dtypes.register_anonymous(arr_dtype, "Vec5Jit")
    typed = binary.plus[arr_udt]
    lib.GrB_BinaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "plus_Vec5Jit" in defn
    assert "v[0]" in defn
    assert "v[4]" in defn

    typed = unary.ainv[arr_udt]
    lib.GrB_UnaryOp_get_String(typed.gb_obj, buf, lib.GxB_JIT_C_DEFINITION)
    defn = ffi.string(buf).decode()
    assert "ainv_Vec5Jit" in defn


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
def test_udt_array_wrapper_stays_within_element():
    """The array-UDT cfunc wrapper writes the element payload and nothing more.

    Numba represents a ``NestedArray`` *value* as an array descriptor (data
    pointer, shape, strides, ...), so the wrapper used to store that descriptor
    where GraphBLAS expected only the elements: 56 bytes into the 48 an
    ``FP64[6]`` element gets. GraphBLAS owns the buffer it hands the cfunc, so
    the extra bytes land on memory the library allocated for something else.

    Pinned at the codegen level rather than end-to-end, and deliberately so:
    nothing in the suite was ever observed to fault on the old codegen, so an
    end-to-end test would not catch a regression. Measured directly instead,
    driving the wrapper through ctypes over a guard-filled buffer: shape
    ``(6,)`` wrote 8 bytes past a 48-byte element, shape ``(2, 3)`` wrote 24.
    """
    import ctypes

    import numba

    from graphblas.core.operator.base import _get_udt_wrapper

    size = 6
    udt = dtypes.register_anonymous(np.dtype((np.float64, (size,))), "_ArrWrapPin")

    @numba.njit
    def _second(x, y):  # pragma: no cover (numba)
        return y

    wrapper, wrapper_sig = _get_udt_wrapper(_second, udt, udt, udt)
    cfunc = numba.cfunc(wrapper_sig, nopython=True, error_model="numpy")(wrapper)
    call = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(cfunc.address)

    guard = -1.0
    xvals = [float(i) for i in range(size)]
    yvals = [100.0 + i for i in range(size)]
    z = (ctypes.c_double * (4 * size))(*([guard] * (4 * size)))
    x = (ctypes.c_double * size)(*xvals)
    y = (ctypes.c_double * size)(*yvals)
    call(ctypes.byref(z), ctypes.byref(x), ctypes.byref(y))

    assert list(z)[:size] == yvals
    assert all(val == guard for val in list(z)[size:]), f"wrote past the UDT element: {list(z)}"


@pytest.mark.skipif("not supports_udfs")
def test_udt_array_any_wrapper_stays_within_element():
    """``binary.any`` on an array UDT must write one element, not Numba's array descriptor.

    ``any``, ``first``, and ``second`` are not in ``_BUILTIN_UDT_BINARY_OPS``,
    so they compile through the generic ``_numba_func`` branch of
    ``BinaryOp._compile_udt``. The wrapper there used to load and store the
    operand as a ``NestedArray`` *value*, which Numba models as its full array
    descriptor (meminfo, parent, nitems, itemsize, data, shape, strides): 56
    bytes on 64-bit for a 1-D element, regardless of payload size. SuiteSparse's
    generic reduce keeps a UDT accumulator in a stack array sized to the element
    (32 bytes here), so each fold overflowed it by 24 bytes; depending on the
    build that clobbered a spilled pointer (segfault or SIGBUS) or silently
    produced a wrong answer.

    Drive the compiled wrapper directly, over heap buffers with slack, so a
    regression trips an assert instead of corrupting a stack frame. The two
    sentinels must differ: the descriptor load/store is a byte-preserving copy
    of the source element plus its trailing bytes, so if ``y``'s slack held the
    same sentinel as ``z``'s guard, the overflow would rewrite ``z``'s guard
    bytes with identical values and the check would be blind to it.
    """
    import ctypes

    import numba

    from graphblas.core.operator.base import _get_udt_wrapper

    # ``register_anonymous`` caches per np.dtype, so this may return the same
    # DataType as other float64[4] tests, renamed. That is fine here: the
    # wrapper below is compiled fresh and nothing asserts on cached JIT state.
    udt = dtypes.register_anonymous(np.dtype((np.float64, (4,))), "_AnyOverflowArr")

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
    slack = 128  # the descriptor overran by 24 bytes; leave generous headroom
    z = np.full(itemsize + slack, 0xAB, dtype=np.uint8)
    x = np.full(itemsize + slack, 0xCD, dtype=np.uint8)
    y = np.full(itemsize + slack, 0xCD, dtype=np.uint8)
    xvals = np.array([1.0, 2.0, 3.0, 4.0])
    yvals = np.array([10.0, 20.0, 30.0, 40.0])
    x[:itemsize] = xvals.view(np.uint8)
    y[:itemsize] = yvals.view(np.uint8)
    call(z.ctypes.data, x.ctypes.data, y.ctypes.data)

    # ``any`` uses ``_second`` semantics, so the payload must be ``y``'s.
    np.testing.assert_array_equal(z[:itemsize].view(np.float64), yvals)
    overrun = np.flatnonzero(z[itemsize:] != 0xAB)
    assert overrun.size == 0, f"wrote {overrun.size} bytes past the element at offsets {overrun}"

    # Public-path smoke: the reduce whose stack accumulator the old wrapper
    # overflowed. Kept after the byte-level checks so a regression fails the
    # assert above instead of reaching code that may crash the process.
    v = Vector(udt, size=3)
    rows = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
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
    """Array-UDT UDFs that can't fill the element are rejected at registration.

    Numba's ``Array`` type records ``ndim`` but not extents, so neither case
    below is a type error. Both used to reach the cfunc, where the shape
    mismatch raises in a context that swallows the exception, handing the
    caller an uninitialized element and no error.
    """
    # Shapes unique to this test: ``register_anonymous`` caches by dtype and
    # freezes the JIT C name at first registration, so sharing a shape with
    # another test makes both order-dependent.
    udt9 = dtypes.register_anonymous(np.dtype((np.float64, (9,))), "_ShapeErr9")
    udt10 = dtypes.register_anonymous(np.dtype((np.float64, (10,))), "_ShapeErr10")

    def _truncate(x):  # pragma: no cover (numba)
        return x[:2]

    op = UnaryOp.register_anonymous(_truncate, "_shape_err_trunc", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) when run on sample values"):
        op[udt9]

    # Two array UDTs sharing a base dtype and rank are indistinguishable once a
    # UDF builds its result, so refuse to guess which one it meant.
    def _built(x, y):  # pragma: no cover (numba)
        return y + 0.0

    op2 = BinaryOp.register_anonymous(_built, "_shape_err_ambiguous", is_udt=True)
    with pytest.raises(UdfParseError, match="matches more than one input array UDT"):
        op2[udt9, udt10]

    # Ambiguity is decided on the UDTs, not their Numba shapes: these two are
    # separate DataTypes with separate GraphBLAS handles, but Numba collapses
    # the layered dtype to the flat one's ``nestedarray(float64, (2, 3))``.
    flat = dtypes.register_anonymous(np.dtype((np.float64, (3, 4))), "_ShapeErrFlat")
    layered = dtypes.register_anonymous(
        np.dtype((np.dtype((np.float64, (4,))), (3,))), "_ShapeErrLayered"
    )
    assert flat.numba_type == layered.numba_type
    with pytest.raises(UdfParseError, match="matches more than one input array UDT"):
        op2[flat, layered]
    assert op2[flat, flat].return_type is flat  # a same-type pair is not ambiguous

    # An array UDF whose result matches no input names the mismatch rather
    # than telling the user to return an array, which is what they did.
    def _recast(x):  # pragma: no cover (numba)
        return x.astype(np.float32)

    op3 = UnaryOp.register_anonymous(_recast, "_shape_err_recast", is_udt=True)
    with pytest.raises(UdfParseError, match="matches no input array UDT"):
        op3[udt9]


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_record_array_leaf_shape_errors():
    """A record UDF that under-fills an array-typed leaf is rejected at registration.

    The wrapper slice-assigns array leaves, so a short return raises inside
    the cfunc and abandons the write part-way: leaves after it keep whatever
    SuiteSparse had in the buffer, scalar leaves included.
    """
    spec = np.dtype([("rl_vec", np.float64, (3,)), ("rl_tag", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec, "_RecLeafShape")

    def _short(x, y):  # pragma: no cover (numba)
        return (x["rl_vec"][:2], x["rl_tag"])

    op = BinaryOp.register_anonymous(_short, "_rec_leaf_short", is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) for field .* holds \(3,\)"):
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
# _NestDeep's repr is 142; how it serializes is not what the test is about.
@pytest.mark.filterwarnings("ignore:UDT repr is too large")
def test_udt_record_nesting_mismatch_is_a_keyerror():
    """Records sharing field names but not nesting depth are rejected as a KeyError.

    ``_check_udt_pair`` matched on top-level names only, but the codegen pairs
    operands leaf by leaf, and a field that is a sub-record on one side and a
    scalar on the other contributes a different number of leaves. Without the
    guard the pair reaches Numba, whose typing failure arrives as a
    ``UdfParseError``: a compile error reported for what is really the same
    shape disagreement its sibling checks raise ``KeyError`` for.
    """
    flat = dtypes.register_anonymous(
        np.dtype([("nst_a", np.float64), ("nst_b", np.float64)], align=True), "_NestFlat"
    )
    nested = dtypes.register_anonymous(
        np.dtype(
            [
                ("nst_a", np.dtype([("nst_n1", np.float64), ("nst_n2", np.float64)])),
                ("nst_b", np.float64),
            ],
            align=True,
        ),
        "_NestDeep",
    )
    v = Vector(flat, size=1)
    v[0] = (1.0, 2.0)
    w = Vector(nested, size=1)
    w[0] = ((3.0, 4.0), 5.0)
    with pytest.raises(KeyError, match="same number of leaf fields"):
        v.ewise_mult(w, binary.plus).new()


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

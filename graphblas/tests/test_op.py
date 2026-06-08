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
@pytest.mark.slow
def test_udt_tuple_return_binaryop(record_udt):
    """A BinaryOp UDF returning a tuple builds the result UDT field-by-field."""
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
    """A UnaryOp UDF returning a tuple builds the result UDT field-by-field on a Vector."""
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
    """A tuple-returning UnaryOp also works on a Matrix.apply path."""

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
    """An IndexUnaryOp UDF returning a tuple builds the result UDT field-by-field."""
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
    """Tuple-return scales beyond the standard 2-field record dtype."""
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
    """``unary.ainv`` and ``unary.abs`` apply per field on a record UDT."""
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
    """``Matrix.apply`` works on a record-UDT-valued matrix."""
    M = Matrix(record_udt, nrows=2, ncols=2)
    M[0, 0] = (1, 2.0)
    M[0, 1] = (3, 4.0)
    M[1, 0] = (5, 6.0)
    M[1, 1] = (7, 8.0)
    result = M.apply(unary.ainv).new()
    assert result[0, 0].new() == (-1, -2.0)
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
    """``unary.ainv`` / ``unary.abs`` apply per element on an array UDT."""
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
    assert "double v [7]" in defn or "double v[7]" in defn
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
    """A single-field record UDT works for binary and reduce paths."""
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
    assert result[0].new().value[1] == 11  # count field


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
def test_udt_large_array():
    """A 100-element array UDT compiles and runs."""
    big_dtype = np.dtype((np.float64, (100,)))
    big_udt = dtypes.register_anonymous(big_dtype)
    a = Vector(big_udt, 2)
    a[0] = list(range(100))
    a[1] = list(range(100, 200))
    b = Vector(big_udt, 2)
    b[0] = [1.0] * 100
    b[1] = [1.0] * 100
    result = binary.plus(a & b).new()
    np.testing.assert_array_equal(result[0].new().value[:3], [1.0, 2.0, 3.0])


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_int_array():
    """An integer array UDT applies binary ops element-by-element."""
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


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_eq_ne_scalar_broadcast():
    """eq/ne broadcasting between a UDT and a scalar type.

    Before this fix, ``binary.eq(udt_vec & int_vec)`` silently
    reinterpreted the int cell as a UDT struct (reading past the cell)
    and produced byte-comparison nonsense that happened to look like
    plausible False/True. Now the scalar broadcasts to every leaf, so
    ``eq`` is true only when all leaves equal the scalar.
    """
    # ---- Record UDT vs scalar ----
    record = dtypes.register_anonymous(
        np.dtype([("u", np.float64), ("v", np.float64)], align=True),
        name="_EqBcastUV",
    )
    v_udt = Vector(record, size=3)
    v_udt[0] = (10.0, 10.0)  # all equal to 10 -> eq True
    v_udt[1] = (10.0, 20.0)  # partial match -> eq False
    v_udt[2] = (1.0, 2.0)  # no match -> eq False
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

    # NaN propagation through the broadcast: a NaN leaf never equals
    # anything, even another NaN.
    v_nan = Vector(record, size=3)
    v_nan[0] = (np.nan, 5.0)
    v_nan[1] = (5.0, 5.0)
    v_nan[2] = (np.nan, np.nan)
    v_five = Vector.from_coo([0, 1, 2], [5.0, 5.0, 5.0])
    eq_nan = binary.eq(v_nan & v_five).new()
    assert eq_nan.isequal(Vector.from_coo([0, 1, 2], [False, True, False]))

    # ---- Array UDT (1D and 2D) vs scalar ----
    arr1d = dtypes.register_anonymous(np.dtype((np.float64, (3,))), name="_EqBcastA3")
    v_a = Vector(arr1d, size=2)
    v_a[0] = [1.0, 1.0, 1.0]
    v_a[1] = [1.0, 2.0, 1.0]
    v_one = Vector.from_coo([0, 1], [1.0, 1.0])
    eq_a = binary.eq(v_a & v_one).new()
    assert eq_a.isequal(Vector.from_coo([0, 1], [True, False]))

    arr2d = dtypes.register_anonymous(np.dtype((np.float64, (2, 2))), name="_EqBcastA22")
    v_a22 = Vector(arr2d, size=2)
    v_a22[0] = [[3.0, 3.0], [3.0, 3.0]]
    v_a22[1] = [[3.0, 3.0], [4.0, 3.0]]
    v_three = Vector.from_coo([0, 1], [3.0, 3.0])
    eq_a22 = binary.eq(v_a22 & v_three).new()
    assert eq_a22.isequal(Vector.from_coo([0, 1], [True, False]))

    # ---- Nested record vs scalar ----
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

    # ---- Record with array sub-field vs scalar ----
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
        # Clean up the namespace so test_operator_types' enumeration of
        # binary/unary/indexunary doesn't see these UDT-only ops.
        for module, name in [
            (binary, "_lazy_udt_add"),
            (unary, "_lazy_udt_neg"),
            (indexunary, "_lazy_udt_iu"),
        ]:
            if hasattr(module, name):
                delattr(module, name)
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

    # A bad source surfaces as RuntimeError with the offending source attached.
    bad_src = "def _op(x, y):\n    return (x + y\n"  # missing close paren
    with pytest.raises(RuntimeError, match=r"(?s)not valid Python.*x \+ y"):
        _compile_codegen(
            bad_src,
            func_name="_op",
            source_label="<gb-udt-helper-test typo>",
        )

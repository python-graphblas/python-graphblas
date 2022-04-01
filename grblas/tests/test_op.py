import itertools

import numpy as np
import pytest

import grblas as gb
from grblas import agg, binary, dtypes, lib, monoid, op, operator, semiring, unary
from grblas.exceptions import DomainMismatch, UdfParseError
from grblas.operator import BinaryOp, Monoid, Semiring, UnaryOp, get_semiring

from grblas import Matrix, Vector  # isort:skip


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
    assert orig_types(unary.positioni) == {"INT32", "INT64"}
    assert orig_types(unary.positionj1) == {"INT32", "INT64"}


def test_binaryop():
    assert binary.plus["INT32"].gb_obj == lib.GrB_PLUS_INT32
    assert binary.plus[dtypes.UINT16].gb_obj == lib.GrB_PLUS_UINT16
    assert orig_types(binary.firsti) == {"INT32", "INT64"}
    assert orig_types(binary.secondj1) == {"INT32", "INT64"}


def test_monoid():
    assert monoid.max["INT32"].gb_obj == lib.GrB_MAX_MONOID_INT32
    assert monoid.max[dtypes.UINT16].gb_obj == lib.GrB_MAX_MONOID_UINT16


def test_semiring():
    assert semiring.min_plus["INT32"].gb_obj == lib.GrB_MIN_PLUS_SEMIRING_INT32
    assert semiring.min_plus[dtypes.UINT16].gb_obj == lib.GrB_MIN_PLUS_SEMIRING_UINT16
    assert orig_types(semiring.min_firsti) == {"INT32", "INT64"}


def test_agg():
    assert repr(agg.count) == "agg.count"
    assert repr(agg.count["INT32"]) == "agg.count[INT32]"
    assert "INT64" in agg.sum_of_inverses
    assert agg.sum_of_inverses["INT64"].return_type == "FP64"
    assert "BOOL" not in agg.sum_of_inverses
    with pytest.raises(KeyError, match="BOOL"):
        agg.sum_of_inverses["BOOL"]
    assert agg.varp["INT64"].return_type == "FP64"


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


def test_unaryop_udf():
    def plus_one(x):
        return x + 1

    UnaryOp.register_new("plus_one", plus_one)
    assert hasattr(unary, "plus_one")
    comp_set = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP32",
        "FP64",
        "BOOL",
    }
    if dtypes._supports_complex:
        comp_set.update({"FC32", "FC64"})
    assert set(unary.plus_one.types) == comp_set
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.plus_one)
    result = Vector.from_values([0, 1, 3], [2, 3, -3], dtype=dtypes.INT32)
    assert v.isequal(result)
    assert "INT8" in unary.plus_one
    assert "INT8" in unary.plus_one.types
    del unary.plus_one["INT8"]
    assert "INT8" not in unary.plus_one
    assert "INT8" not in unary.plus_one.types
    with pytest.raises(TypeError, match="UDF argument must be a function"):
        UnaryOp.register_new("bad", object())
    assert not hasattr(unary, "bad")
    with pytest.raises(UdfParseError, match="Unable to parse function using Numba"):
        UnaryOp.register_new("bad", lambda x: v)


@pytest.mark.slow
def test_unaryop_parameterized():
    def plus_x(x=0):
        def inner(val):
            return val + x

        return inner

    op = UnaryOp.register_anonymous(plus_x, parameterized=True)
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.apply(op).new()
    assert v.isequal(v0, check_dtype=True)
    v0 = v.apply(op(0)).new()
    assert v.isequal(v0, check_dtype=True)
    v10 = v.apply(op(x=10)).new()
    r10 = Vector.from_values([0, 1, 3], [11, 12, 6], dtype=dtypes.INT32)
    assert r10.isequal(v10, check_dtype=True)
    UnaryOp._initialize()  # no-op
    UnaryOp.register_new("plus_x_parameterized", plus_x, parameterized=True)
    op = unary.plus_x_parameterized
    v11 = v.apply(op(x=10)["INT32"]).new()
    assert r10.isequal(v11, check_dtype=True)


@pytest.mark.slow
def test_binaryop_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x

        return inner

    op = BinaryOp.register_anonymous(plus_plus_x, parameterized=True)
    assert op.monoid is None
    assert op(1).monoid is None
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.ewise_mult(v, op).new()
    r0 = Vector.from_values([0, 1, 3], [2, 4, -8], dtype=dtypes.INT32)
    assert v0.isequal(r0, check_dtype=True)
    v1 = v.ewise_add(v, op(1), require_monoid=False).new()
    r1 = Vector.from_values([0, 1, 3], [3, 5, -7], dtype=dtypes.INT32)
    assert v1.isequal(r1, check_dtype=True)

    w = Vector.from_values([0, 0, 1, 3], [1, 0, 2, -4], dtype=dtypes.INT32, dup_op=op)
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

    assert v.isequal(Vector.from_values([0, 1, 3], [19, 35, -61], dtype=dtypes.INT32))
    v11 = v.apply(op(1), left=10).new()
    r11 = Vector.from_values([0, 1, 3], [30, 46, -50], dtype=dtypes.INT32)
    # Should we check for dtype here?
    # Is it okay if the literal scalar is an INT64, which causes the output to default to INT64?
    assert v11.isequal(r11, check_dtype=False)

    with pytest.raises(TypeError, match="UDF argument must be a function"):
        BinaryOp.register_new("bad", object())
    assert not hasattr(binary, "bad")
    with pytest.raises(UdfParseError, match="Unable to parse function using Numba"):
        BinaryOp.register_new("bad", lambda x, y: v)

    def my_add(x, y):
        return x + y

    op = BinaryOp.register_anonymous(my_add)
    assert op.name == "my_add"


@pytest.mark.slow
def test_monoid_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x

        return inner

    bin_op = BinaryOp.register_anonymous(plus_plus_x, parameterized=True)

    # signatures must match
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, lambda x: -x)  # pragma: no cover
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, lambda y=0: -y)  # pragma: no cover
    with pytest.raises(TypeError, match="binaryop must be parameterized"):
        operator.ParameterizedMonoid("bad_monoid", binary.plus, 0)

    def plus_plus_x_identity(x=0):
        return -x

    assert bin_op.monoid is None
    bin_op1 = bin_op(1)
    assert bin_op1.monoid is None
    monoid = Monoid.register_anonymous(bin_op, plus_plus_x_identity, name="my_monoid")
    assert bin_op.monoid is monoid
    assert bin_op(1).monoid is monoid(1)
    assert monoid(2) is bin_op(2).monoid
    # However, this still fails.
    # For this to work, we would need `bin_op1` to know it was created from a
    # ParameterizedBinaryOp. It would then need to check to see if the parameterized
    # parent has been associated with a monoid since the creation of `bin_op1`.
    assert bin_op1.monoid is None

    assert monoid.name == "my_monoid"
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.ewise_add(v, monoid).new()
    r0 = Vector.from_values([0, 1, 3], [2, 4, -8], dtype=dtypes.INT32)
    assert v0.isequal(r0, check_dtype=True)
    v1 = v.ewise_mult(v, monoid(1)).new()
    r1 = Vector.from_values([0, 1, 3], [3, 5, -7], dtype=dtypes.INT32)
    assert v1.isequal(r1, check_dtype=True)

    assert v.reduce(monoid).new() == -1
    assert v.reduce(monoid(1)).new() == 1
    # with pytest.raises(TypeError, match="BinaryOp"):  # NOW OKAY
    w1 = Vector.from_values([0, 0, 1, 3], [1, 0, 2, -4], dtype=dtypes.INT32, dup_op=monoid)
    w2 = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    assert w1.isequal(w2)

    # identity may be a value
    def logaddexp(base):
        def inner(x, y):
            return np.log(base**x + base**y) / np.log(base)

        return inner

    fv = v.apply(unary.identity).new(dtype=dtypes.FP64)
    bin_op = BinaryOp.register_anonymous(logaddexp, parameterized=True)
    Monoid.register_new("_user_defined_monoid", bin_op, -np.inf)
    monoid = gb.monoid._user_defined_monoid
    fv2 = fv.ewise_mult(fv, monoid(2)).new()
    plus1 = UnaryOp.register_anonymous(lambda x: x + 1)
    expected = fv.apply(plus1).new()
    assert fv2.isclose(expected, check_dtype=True)
    with pytest.raises(TypeError, match="must be a BinaryOp"):
        Monoid.register_anonymous(monoid, 0)

    def plus_times_x(x=0):
        def inner(left, right):
            return (left + right) * x

        return inner

    bin_op = BinaryOp.register_anonymous(plus_times_x, parameterized=True)

    def bad_identity(x=0):
        raise ValueError("hahaha!")

    assert bin_op.monoid is None
    monoid = Monoid.register_anonymous(bin_op, bad_identity, name="broken_monoid")
    assert bin_op.monoid is monoid
    assert bin_op(1).monoid is None


@pytest.mark.slow
def test_semiring_parameterized():
    def plus_plus_x(x=0):
        def inner(left, right):
            return left + right + x

        return inner

    def plus_plus_x_identity(x=0):
        return -x

    assert Semiring.register_anonymous(monoid.min, binary.plus).name == "min_plus"

    bin_op = BinaryOp.register_anonymous(plus_plus_x, parameterized=True)
    mymonoid = Monoid.register_anonymous(bin_op, plus_plus_x_identity)
    # monoid and binaryop are both parameterized
    mysemiring = Semiring.register_anonymous(mymonoid, bin_op, name="my_semiring")
    assert mysemiring.name == "my_semiring"

    A = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    x = Vector.from_values([0, 1], [10, 20])

    y = A.mxv(x, mysemiring).new()
    assert y.isequal(A.mxv(x, semiring.plus_plus).new())
    assert y.isequal(x.vxm(A.T, semiring.plus_plus).new())
    assert y.isequal(Vector.from_values([0, 1], [33, 37]))

    y = A.mxv(x, mysemiring(1)).new()
    assert y.isequal(Vector.from_values([0, 1], [36, 40]))  # three extra pluses

    y = x.vxm(A.T, mysemiring(1)).new()  # same as previous
    assert y.isequal(Vector.from_values([0, 1], [36, 40]))

    y = x.vxm(A.T, mysemiring).new()
    assert y.isequal(Vector.from_values([0, 1], [33, 37]))

    B = A.mxm(A, mysemiring).new()
    assert B.isequal(A.mxm(A, semiring.plus_plus).new())
    assert B.isequal(Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [7, 9, 11, 13]))

    B = A.mxm(A, mysemiring(1)).new()  # three extra pluses
    assert B.isequal(Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [10, 12, 14, 16]))

    with pytest.raises(TypeError, match="Expected type: Monoid"):
        A.ewise_add(A, mysemiring)

    # mismatched signatures.
    def other_binary(y=0):  # pragma: no cover
        def inner(left, right):
            return left + right - y

        return inner

    def other_identity(y=0):
        return x  # pragma: no cover

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
    Adup = Matrix.from_values([0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [100, 1, 2, 3, 4], dup_op=bin_op)
    Adup2 = Matrix.from_values(
        [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [100, 1, 2, 3, 4], dup_op=binary.plus
    )
    assert Adup.isequal(Adup2)

    def plus_x(x=0):
        def inner(y):
            return x + y

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


def test_unaryop_udf_bool_result():
    # numba has trouble compiling this, but we have a work-around
    def is_positive(x):
        return x > 0

    UnaryOp.register_new("is_positive", is_positive)
    assert hasattr(unary, "is_positive")
    assert set(unary.is_positive.types) == {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP32",
        "FP64",
        "BOOL",
    }
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    w = v.apply(unary.is_positive).new()
    result = Vector.from_values([0, 1, 3], [True, True, False], dtype=dtypes.BOOL)
    assert w.isequal(result)


def test_binaryop_udf():
    def times_minus_sum(x, y):
        return x * y - (x + y)

    BinaryOp.register_new("bin_test_func", times_minus_sum)
    assert hasattr(binary, "bin_test_func")
    comp_set = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP32",
        "FP64",
    }
    if dtypes._supports_complex:
        comp_set.update({"FC32", "FC64"})
    assert set(binary.bin_test_func.types) == comp_set
    v1 = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.from_values([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, binary.bin_test_func, require_monoid=False).new()
    result = Vector.from_values([0, 1, 2, 3], [-1, 2, 3, -31], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_monoid_udf():
    def plus_plus_one(x, y):
        return x + y + 1

    BinaryOp.register_new("plus_plus_one", plus_plus_one)
    Monoid.register_new("plus_plus_one", binary.plus_plus_one, -1)
    assert hasattr(monoid, "plus_plus_one")
    assert set(monoid.plus_plus_one.types) == {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP32",
        "FP64",
    }
    v1 = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v2 = Vector.from_values([0, 2, 3], [2, 3, 7], dtype=dtypes.INT32)
    w = v1.ewise_add(v2, monoid.plus_plus_one).new()
    result = Vector.from_values([0, 1, 2, 3], [4, 2, 3, 4], dtype=dtypes.INT32)
    assert w.isequal(result)

    with pytest.raises(DomainMismatch):
        Monoid.register_anonymous(binary.plus_plus_one, {"BOOL": True})
    with pytest.raises(DomainMismatch):
        Monoid.register_anonymous(binary.plus_plus_one, {"BOOL": -1})


def test_semiring_udf():
    def plus_plus_two(x, y):
        return x + y + 2

    BinaryOp.register_new("plus_plus_two", plus_plus_two)
    Semiring.register_new("extra_twos", monoid.plus, binary.plus_plus_two)
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    A = Matrix.from_values(
        [0, 0, 0, 0, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [2, 3, 4, 5, 6, 7, 8, 9],
        dtype=dtypes.INT32,
    )
    w = v.vxm(A, semiring.extra_twos).new()
    result = Vector.from_values([0, 1, 2, 3], [9, 11, 13, 15], dtype=dtypes.INT32)
    assert w.isequal(result)


def test_binary_updates():
    assert not hasattr(binary, "div")
    assert binary.cdiv["INT64"].gb_obj == lib.GrB_DIV_INT64
    vec1 = Vector.from_values([0], [1], dtype=dtypes.INT64)
    vec2 = Vector.from_values([0], [2], dtype=dtypes.INT64)
    result = vec1.ewise_mult(vec2, binary.truediv).new()
    assert result.isclose(Vector.from_values([0], [0.5], dtype=dtypes.FP64), check_dtype=True)
    vec4 = Vector.from_values([0], [-3], dtype=dtypes.INT64)
    result2 = vec4.ewise_mult(vec2, binary.cdiv).new()
    assert result2.isequal(Vector.from_values([0], [-1], dtype=dtypes.INT64), check_dtype=True)
    result3 = vec4.ewise_mult(vec2, binary.floordiv).new()
    assert result3.isequal(Vector.from_values([0], [-2], dtype=dtypes.INT64), check_dtype=True)


@pytest.mark.slow
def test_nested_names():
    def plus_three(x):
        return x + 3

    UnaryOp.register_new("incrementers.plus_three", plus_three)
    assert hasattr(unary, "incrementers")
    assert type(unary.incrementers) is operator.OpPath
    assert hasattr(unary.incrementers, "plus_three")
    comp_set = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP32",
        "FP64",
        "BOOL",
    }
    if dtypes._supports_complex:
        comp_set.update({"FC32", "FC64"})
    assert set(unary.incrementers.plus_three.types) == comp_set

    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.incrementers.plus_three)
    result = Vector.from_values([0, 1, 3], [4, 5, -1], dtype=dtypes.INT32)
    assert v.isequal(result), v

    def plus_four(x):
        return x + 4

    UnaryOp.register_new("incrementers.plus_four", plus_four)
    assert hasattr(unary.incrementers, "plus_four")
    assert hasattr(op.incrementers, "plus_four")  # Also save it to `grblas.op`!
    v << v.apply(unary.incrementers.plus_four)  # this is in addition to the plus_three earlier
    result2 = Vector.from_values([0, 1, 3], [8, 9, 3], dtype=dtypes.INT32)
    assert v.isequal(result2), v

    def bad_will_overwrite_path(x):
        return x + 7  # pragma: no cover

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

    assert op.numpy.fabs is unary.numpy.fabs
    assert op.numpy.subtract is binary.numpy.subtract
    assert op.numpy.add is binary.numpy.add
    assert op.numpy.add_add is semiring.numpy.add_add
    assert len(dir(op)) > 300
    assert len(dir(op.numpy)) > 500

    with pytest.raises(
        AttributeError, match="module 'grblas.op.numpy' has no attribute 'bad_attr'"
    ):
        op.numpy.bad_attr

    # Make sure all have been initialized so `vars` below works
    for key in list(op._delayed):  # pragma: no cover
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
    assert not unarynames - opnames, unarynames - opnames
    assert not binarynames - opnames, binarynames - opnames
    assert not monoidnames - opnames, monoidnames - opnames
    assert not semiringnames - opnames, semiringnames - opnames
    assert not opnames - (unarynames | binarynames | monoidnames | semiringnames)


@pytest.mark.slow
def test_binaryop_attributes():
    assert binary.plus[int].monoid is monoid.plus[int]
    assert binary.minus[int].monoid is None
    assert binary.plus.monoid is monoid.plus
    assert binary.minus.monoid is None

    assert binary.numpy.add[int].monoid is monoid.numpy.add[int]
    assert binary.numpy.subtract[int].monoid is None
    assert binary.numpy.add.monoid is monoid.numpy.add
    assert binary.numpy.subtract.monoid is None

    op = BinaryOp.register_anonymous(lambda x, y: x + y, name="plus")
    assert op.monoid is None
    assert op[int].monoid is None

    assert binary.plus[int].parent is binary.plus
    assert binary.numpy.add[int].parent is binary.numpy.add
    assert op[int].parent is op

    # bad type
    assert binary.plus[bool].monoid is None
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
            assert val.monoid is None
            assert all(val[type_].monoid is None for type_ in val.types)


@pytest.mark.slow
def test_monoid_attributes():
    assert monoid.plus[int].binaryop is binary.plus[int]
    assert monoid.plus[int].identity == 0
    assert monoid.plus.binaryop is binary.plus
    assert monoid.plus.identities == {typ: 0 for typ in monoid.plus.types}

    assert monoid.numpy.add[int].binaryop is binary.numpy.add[int]
    assert monoid.numpy.add[int].identity == 0
    assert monoid.numpy.add.binaryop is binary.numpy.add
    assert monoid.numpy.add.identities == {typ: 0 for typ in monoid.numpy.add.types}

    binop = BinaryOp.register_anonymous(lambda x, y: x + y, name="plus")
    op = Monoid.register_anonymous(binop, 0, name="plus")
    assert op.binaryop is binop
    assert op[int].binaryop is binop[int]

    assert monoid.plus[int].parent is monoid.plus
    assert monoid.numpy.add[int].parent is monoid.numpy.add
    assert op[int].parent is op

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

    assert semiring.numpy.add_subtract[int].monoid is monoid.numpy.add[int]
    assert semiring.numpy.add_subtract[int].binaryop is binary.numpy.subtract[int]
    assert semiring.numpy.add_subtract.monoid is monoid.numpy.add
    assert semiring.numpy.add_subtract.binaryop is binary.numpy.subtract

    binop = BinaryOp.register_anonymous(lambda x, y: x + y, name="plus")
    mymonoid = Monoid.register_anonymous(binop, 0, name="plus")
    op = Semiring.register_anonymous(mymonoid, binop, name="plus_plus")
    assert op.binaryop is binop
    assert op.binaryop[int] is binop[int]
    assert op.monoid is mymonoid
    assert op.monoid[int] is mymonoid[int]

    assert semiring.min_plus[int].parent is semiring.min_plus
    assert semiring.numpy.add_subtract[int].parent is semiring.numpy.add_subtract
    assert op[int].parent is op

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
    monoid_names = {x for x in dir(monoid) if not x.startswith("_")}
    binary_names = {x for x in dir(binary) if not x.startswith("_")}
    diff = monoid_names - binary_names
    assert not diff
    extras = {x for x in set(dir(monoid.numpy)) - set(dir(binary.numpy)) if not x.startswith("_")}
    assert not extras, ", ".join(sorted(extras))


def test_div_semirings():
    assert not hasattr(semiring, "plus_div")
    A1 = Matrix.from_values([0, 1], [0, 0], [-1, -3])
    A2 = Matrix.from_values([0, 1], [0, 0], [2, 2])
    result = A1.T.mxm(A2, semiring.plus_cdiv).new()
    assert result[0, 0].new() == -1
    assert result.dtype == dtypes.INT64

    result = A1.T.mxm(A2, semiring.plus_truediv).new()
    assert result[0, 0].new() == -2
    assert result.dtype == dtypes.FP64

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
        return x + y

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

    sr = get_semiring(monoid.plus, binary.numpy.copysign)
    assert sr.monoid is monoid.plus
    assert sr.binaryop is binary.numpy.copysign


def test_create_semiring():
    # stress test / sanity check
    monoid_names = {x for x in dir(monoid) if not x.startswith("_")}
    binary_names = {x for x in dir(binary) if not x.startswith("_")}
    for monoid_name, binary_name in itertools.product(monoid_names, binary_names):
        cur_monoid = getattr(monoid, monoid_name)
        if not isinstance(cur_monoid, Monoid):
            continue
        cur_binary = getattr(binary, binary_name)
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
    assert semiring.min_secondi.commutes_to is semiring.min_firstj
    assert semiring.plus_pow.commutes_to is semiring.plus_rpow
    assert not semiring.plus_pow.is_commutative
    assert binary.isclose.commutes_to is binary.isclose
    assert binary.isclose.is_commutative
    assert binary.isclose(0.1).commutes_to is binary.isclose(0.1)
    assert binary.floordiv.commutes_to is binary.rfloordiv
    assert not binary.floordiv.is_commutative
    assert binary.numpy.add.commutes_to is binary.numpy.add
    assert binary.numpy.add.is_commutative
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
    assert semiring.min_secondi[int].commutes_to is semiring.min_firstj[int]
    assert semiring.plus_pow[int].commutes_to is semiring.plus_rpow[int]
    assert not semiring.plus_pow[int].is_commutative
    assert binary.isclose(0.1)[int].commutes_to is binary.isclose(0.1)[int]
    assert binary.floordiv[int].commutes_to is binary.rfloordiv[int]
    assert not binary.floordiv[int].is_commutative
    assert binary.numpy.add[int].commutes_to is binary.numpy.add[int]
    assert binary.numpy.add[int].is_commutative
    assert binary.numpy.less[int].commutes_to is binary.numpy.greater[int]
    assert not binary.numpy.less[int].is_commutative

    # Stress test (this can create extra semirings)
    names = dir(semiring)
    for name in names:
        val = getattr(semiring, name)
        if not hasattr(val, "commutes_to"):
            continue
        assert val.commutes_to is None or isinstance(val.commutes_to, type(val))


def test_from_string():
    assert unary.from_string("-") is unary.ainv
    assert unary.from_string("abs[float]") is unary.abs[float]
    assert binary.from_string("+") is binary.plus
    assert binary.from_string("-[int]") is binary.minus[int]
    assert binary.from_string("true_divide") is binary.numpy.true_divide
    assert binary.from_string("//") is binary.floordiv
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

    assert agg.from_string("count") is agg.count
    assert agg.from_string("|") is agg.any
    assert agg.from_string("+[int]") is agg.sum[int]
    with pytest.raises(ValueError, match="Unknown agg string"):
        agg.from_string("bad_agg")


@pytest.mark.slow
def test_lazy_op():
    UnaryOp.register_new("lazy", lambda x: x, lazy=True)
    assert isinstance(op.lazy, UnaryOp)
    assert isinstance(unary.lazy, UnaryOp)
    BinaryOp.register_new("lazy", lambda x, y: x + y, lazy=True)
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
    UnaryOp.register_new("numpy.lazy", lambda x: x, lazy=True)
    assert isinstance(unary.numpy.lazy, UnaryOp)
    BinaryOp.register_new("numpy.lazy", lambda x, y: x + y, lazy=True)
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
    UnaryOp.register_new("misc.lazy", lambda x: x, lazy=True)
    assert isinstance(unary.misc.lazy, UnaryOp)
    with pytest.raises(AttributeError):
        unary.misc.bad
    with pytest.raises(ValueError):
        unary.from_string("misc.lazy.badpath")
    assert op.from_string("lazy") is unary.lazy
    assert op.from_string("numpy.lazy") is unary.numpy.lazy


def test_positional():
    assert unary.positioni.is_positional
    assert unary.positioni1[int].is_positional
    assert unary.positionj1.is_positional
    assert unary.positionj[float].is_positional
    assert not unary.exp.is_positional
    assert not unary.abs[bool].is_positional

    assert binary.firsti.is_positional
    assert binary.secondj1[int].is_positional
    assert not binary.plus.is_positional
    assert not binary.minus[float].is_positional

    assert not monoid.plus.is_positional
    assert not monoid.plus[int].is_positional

    assert semiring.any_firsti.is_positional
    assert semiring.any_secondj[int].is_positional
    assert not semiring.any_first.is_positional
    assert not semiring.any_second[int].is_positional


def test_dir():
    for mod in [unary, binary, monoid, semiring, op]:
        assert not set(mod._delayed) - set(dir(mod))


def test_semiring_commute_exists():
    from .conftest import orig_semirings

    vals = {getattr(semiring, key) for key in orig_semirings}
    missing = set()
    for key in orig_semirings:
        val = getattr(semiring, key)
        commutes_to = val.commutes_to
        if commutes_to is not None and commutes_to not in vals:  # pragma: no cover
            missing.add(commutes_to.name)
    if missing:
        raise AssertionError("Missing semirings: " + ", ".join(sorted(missing)))


def test_binaryop_commute_exists():
    from .conftest import orig_binaryops

    vals = {getattr(binary, key) for key in orig_binaryops}
    missing = set()
    for key in orig_binaryops:
        val = getattr(binary, key)
        commutes_to = val.commutes_to
        if commutes_to is not None and commutes_to not in vals:  # pragma: no cover
            missing.add(commutes_to.name)
    if missing:
        raise AssertionError("Missing binaryops: " + ", ".join(sorted(missing)))

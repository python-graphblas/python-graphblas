import pytest
import numpy as np
import grblas
from grblas import lib
from grblas import unary, binary, monoid, semiring
from grblas import dtypes, ops, exceptions
from grblas import Vector, Matrix
from grblas.ops import UnaryOp, BinaryOp, Monoid, Semiring


def test_unaryop():
    assert unary.ainv["INT32"].gb_obj == lib.GrB_AINV_INT32
    assert unary.ainv[dtypes.UINT16].gb_obj == lib.GrB_AINV_UINT16


def test_binaryop():
    assert binary.plus["INT32"].gb_obj == lib.GrB_PLUS_INT32
    assert binary.plus[dtypes.UINT16].gb_obj == lib.GrB_PLUS_UINT16


def test_monoid():
    assert monoid.max["INT32"].gb_obj == lib.GxB_MAX_INT32_MONOID
    assert monoid.max[dtypes.UINT16].gb_obj == lib.GxB_MAX_UINT16_MONOID


def test_semiring():
    assert semiring.min_plus["INT32"].gb_obj == lib.GxB_MIN_PLUS_INT32
    assert semiring.min_plus[dtypes.UINT16].gb_obj == lib.GxB_MIN_PLUS_UINT16


def test_find_opclass_unaryop():
    assert ops.find_opclass(unary.minv)[1] == "UnaryOp"
    # assert ops.find_opclass(lib.GrB_MINV_INT64)[1] == 'UnaryOp'


def test_find_opclass_binaryop():
    assert ops.find_opclass(binary.times)[1] == "BinaryOp"
    # assert ops.find_opclass(lib.GrB_TIMES_INT64)[1] == 'BinaryOp'


def test_find_opclass_monoid():
    assert ops.find_opclass(monoid.max)[1] == "Monoid"
    # assert ops.find_opclass(lib.GxB_MAX_INT64_MONOID)[1] == 'Monoid'


def test_find_opclass_semiring():
    assert ops.find_opclass(semiring.plus_plus)[1] == "Semiring"
    # assert ops.find_opclass(lib.GxB_PLUS_PLUS_INT64)[1] == 'Semiring'


def test_find_opclass_invalid():
    assert ops.find_opclass("foobar")[1] == ops.UNKNOWN_OPCLASS
    # assert ops.find_opclass(lib.GrB_INP0)[1] == ops.UNKNOWN_OPCLASS


def test_get_typed_op():
    assert ops.get_typed_op(binary.bor, dtypes.INT64) is binary.bor[dtypes.INT64]
    with pytest.raises(KeyError, match="bor does not work with FP64"):
        ops.get_typed_op(binary.bor, dtypes.FP64)
    with pytest.raises(TypeError, match="Unable to get typed operator"):
        ops.get_typed_op(object(), dtypes.INT64)


def test_unaryop_udf():
    def plus_one(x):
        return x + 1

    UnaryOp.register_new("plus_one", plus_one)
    assert hasattr(unary, "plus_one")
    assert set(unary.plus_one.types) == {
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
        "FC32",
        "FC64",
    }
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
    with pytest.raises(exceptions.UdfParseError, match="Unable to parse function using Numba"):
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
        assert v.reduce(op).value == -1

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
    with pytest.raises(exceptions.UdfParseError, match="Unable to parse function using Numba"):
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
        Monoid.register_anonymous(bin_op, lambda x: -x)
    with pytest.raises(ValueError, match="Signatures"):
        Monoid.register_anonymous(bin_op, lambda y=0: -y)
    with pytest.raises(TypeError, match="binaryop must be parameterized"):
        ops.ParameterizedMonoid("bad_monoid", binary.plus, 0)

    def plus_plus_x_identity(x=0):
        return -x

    monoid = Monoid.register_anonymous(bin_op, plus_plus_x_identity, name="my_monoid")
    assert monoid.name == "my_monoid"
    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v0 = v.ewise_add(v, monoid).new()
    r0 = Vector.from_values([0, 1, 3], [2, 4, -8], dtype=dtypes.INT32)
    assert v0.isequal(r0, check_dtype=True)
    v1 = v.ewise_mult(v, monoid(1)).new()
    r1 = Vector.from_values([0, 1, 3], [3, 5, -7], dtype=dtypes.INT32)
    assert v1.isequal(r1, check_dtype=True)

    assert v.reduce(monoid).value == -1
    assert v.reduce(monoid(1)).value == 1
    with pytest.raises(TypeError, match="BinaryOp"):
        Vector.from_values([0, 0, 1, 3], [1, 0, 2, -4], dtype=dtypes.INT32, dup_op=monoid)

    # identity may be a value
    def logaddexp(base):
        def inner(x, y):
            return np.log(base ** x + base ** y) / np.log(base)

        return inner

    fv = v.apply(unary.identity).new(dtype=dtypes.FP64)
    bin_op = BinaryOp.register_anonymous(logaddexp, parameterized=True)
    Monoid.register_new("user_defined_monoid", bin_op, -np.inf)
    monoid = grblas.monoid.user_defined_monoid
    fv2 = fv.ewise_mult(fv, monoid(2)).new()
    plus1 = UnaryOp.register_anonymous(lambda x: x + 1)
    expected = fv.apply(plus1).new()
    assert fv2.isclose(expected, check_dtype=True)
    with pytest.raises(TypeError, match="must be a BinaryOp"):
        Monoid.register_anonymous(monoid, 0)


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

    B = A.ewise_add(A, mysemiring).new()
    assert B.isequal(A.ewise_add(A, semiring.plus_plus).new())
    assert B.isequal(A.ewise_mult(A, mysemiring).new())

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
        ops.ParameterizedSemiring("bad_semiring", monoid.plus, binary.plus)
    with pytest.raises(TypeError, match="monoid must be of type"):
        ops.ParameterizedSemiring("bad_semiring", binary.plus, binary.plus)
    with pytest.raises(TypeError, match="binaryop must be of"):
        ops.ParameterizedSemiring("bad_semiring", monoid.plus, monoid.plus)

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

    x = A.reduce_rows(bin_op).new()
    assert x.isequal(A.reduce_rows(binary.plus).new())

    x = A.reduce_columns(bin_op).new()
    assert x.isequal(A.reduce_columns(binary.plus).new())

    s = A.reduce_scalar(mymonoid).new()
    assert s.value == A.reduce_scalar(monoid.plus).value

    with pytest.raises(TypeError, match="Monoid"):
        A.reduce_scalar(bin_op).new()
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
    assert set(binary.bin_test_func.types) == {
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
        "FC32",
        "FC64",
    }
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

    with pytest.raises(exceptions.DomainMismatch):
        Monoid.register_anonymous(binary.plus_plus_one, {"BOOL": True})
    with pytest.raises(exceptions.DomainMismatch):
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
    assert type(unary.incrementers) is ops.OpPath
    assert hasattr(unary.incrementers, "plus_three")
    assert set(unary.incrementers.plus_three.types) == {
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
        "FC32",
        "FC64",
    }

    v = Vector.from_values([0, 1, 3], [1, 2, -4], dtype=dtypes.INT32)
    v << v.apply(unary.incrementers.plus_three)
    result = Vector.from_values([0, 1, 3], [4, 5, -1], dtype=dtypes.INT32)
    assert v.isequal(result), v

    def plus_four(x):
        return x + 4

    UnaryOp.register_new("incrementers.plus_four", plus_four)
    assert hasattr(unary.incrementers, "plus_four")
    v << v.apply(unary.incrementers.plus_four)  # this is in addition to the plus_three earlier
    result2 = Vector.from_values([0, 1, 3], [8, 9, 3], dtype=dtypes.INT32)
    assert v.isequal(result2), v

    def bad_will_overwrite_path(x):
        return x + 7  # pragma: no cover

    with pytest.raises(AttributeError):
        UnaryOp.register_new("incrementers", bad_will_overwrite_path)
    with pytest.raises(AttributeError, match="already defined"):
        UnaryOp.register_new("identity.newfunc", bad_will_overwrite_path)

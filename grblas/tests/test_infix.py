from pytest import fixture, raises

from grblas import monoid, op
from grblas.exceptions import DimensionMismatch

from .conftest import autocompute

from grblas import Matrix, Scalar, Vector  # isort:skip


@fixture
def v1():
    return Vector.from_values([0, 2], [2.0, 5.0], name="v_1")


@fixture
def v2():
    return Vector.from_values([1, 2], [3.0, 7.0], name="v_2")


@fixture
def A1():
    return Matrix.from_values([0, 0], [0, 1], [0.0, 4.0], ncols=3, name="A_1")


@fixture
def A2():
    return Matrix.from_values([0, 2], [0, 0], [6.0, 8.0], name="A_2")


@fixture
def s1():
    return Scalar.from_value(3, name="s_1")


@autocompute
def test_ewise(v1, v2, A1, A2):
    v1 = v1.dup(dtype=bool)
    v2 = v2.dup(dtype=bool)
    A1 = A1.dup(dtype=bool)
    A2 = A2.dup(dtype=bool)
    for left, right in [
        (v1, v2),
        (A1, A2.T),
        (A1.T, A1.T),
        (A1, A1),
    ]:
        expected = left.ewise_mult(right, monoid.land).new()
        expr = left & right
        assert expr.nvals == expected.nvals
        val = expr.new(name="val")
        assert val.name == "val"
        assert expected.isequal((left & right).new(dtype=float))
        assert expected.isequal(monoid.land(left & right).new())
        assert expected.isequal(monoid.land[float](left & right).new())
        assert expected.isequal((left & right).new())  # use `left.ewise_mult` default op
        if isinstance(left, Vector):
            assert (left & right).size == left.size
            assert (left | right).size == left.size
        else:
            assert (left & right).nrows == left.nrows
            assert (left | right).nrows == left.nrows
            assert (left & right).ncols == left.ncols
            assert (left | right).ncols == left.ncols

        expected = left.ewise_add(right, op.lor).new()
        assert expected.isequal(op.lor(left | right).new())
        assert expected.isequal(op.lor[float](left | right).new())
        assert expected.isequal((left | right).new())  # use `left.ewise_add` default op

        expected = left.ewise_mult(right, op.minus).new()
        assert expected.isequal(op.minus(left & right).new())

        expected = left.ewise_add(right, op.minus, require_monoid=False).new()
        assert expected.isequal(op.minus(left | right, require_monoid=False).new())
        expr = left | right
        assert expr.nvals == expr.nvals  # tests caching of ._value


def test_matmul(v1, v2, A1, A2):
    for method, left, right in [
        ("vxm", v2, A2),
        ("vxm", v2, A1.T),
        ("mxv", A1, v1),
        ("mxv", A2.T, v1),
        ("mxm", A1, A2),
        ("mxm", A1.T, A2.T),
        ("mxm", A1, A1.T),
        ("mxm", A2.T, A2),
        ("inner", v1, v2),
        ("inner", v1, v1),
    ]:
        expected = getattr(left, method)(right, op.plus_times).new()
        assert expected.isequal(op.plus_times(left @ right).new())
        assert expected.isequal(op.plus_times[float](left @ right).new())
        assert expected.isequal((left @ right).new())  # use default semiring
        if isinstance(left, Vector):
            if not isinstance(right, Vector):
                assert (left @ right).size == right.ncols
                assert op.plus_times(left @ right).size == right.ncols
        elif isinstance(right, Vector):
            assert (left @ right).size == left.nrows
            assert op.plus_times(left @ right).size == left.nrows
        else:
            assert (left @ right).nrows == left.nrows
            assert (left @ right).ncols == right.ncols


@autocompute
def test_bad_ewise(s1, v1, A1, A2):
    for left, right in [
        (v1, s1),
        (s1, v1),
        (v1, 1),
        (1, v1),
        (v1, A1),
        (A1, v1),
        (v1, A1.T),
        (A1.T, v1),
        (A1, s1),
        (s1, A1),
        (A1.T, s1),
        (s1, A1.T),
        (A1, 1),
        (1, A1),
    ]:
        with raises(TypeError, match="Bad type for argument"):
            left | right
        with raises(TypeError, match="Bad type for argument"):
            left & right

    w = v1[: v1.size - 1].new()
    with raises(DimensionMismatch):
        v1 | w
    with raises(DimensionMismatch):
        v1 & w
    with raises(DimensionMismatch):
        A2 | A1
    with raises(DimensionMismatch):
        A2 & A1
    with raises(DimensionMismatch):
        A1.T | A1
    with raises(DimensionMismatch):
        A1.T & A1

    with raises(TypeError):
        s1 | 1
    with raises(TypeError):
        1 | s1
    with raises(TypeError):
        s1 & 1
    with raises(TypeError):
        1 & s1

    with raises(TypeError, match="not supported for FP64"):
        v1 |= v1
    with raises(TypeError, match="not supported for FP64"):
        A1 |= A1
    with raises(TypeError, match="not supported for FP64"):
        v1 &= v1
    with raises(TypeError, match="not supported for FP64"):
        A1 &= A1

    with raises(TypeError, match="require_monoid"):
        op.minus(v1 | v1)
    with raises(TypeError):
        op.minus(v1 & v1, require_monoid=False)
    with raises(TypeError, match="Bad dtype"):
        op.plus(v1 & v1, 1)


def test_bad_matmul(s1, v1, A1, A2):
    for left, right in [
        (v1, s1),
        (s1, v1),
        (v1, 1),
        (1, v1),
        (A1, s1),
        (s1, A1),
        (A1.T, s1),
        (s1, A1.T),
        (A1, 1),
        (1, A1),
    ]:
        with raises(TypeError, match="Bad type for argument"):
            left @ right

    with raises(DimensionMismatch):
        v1 @ A1
    with raises(DimensionMismatch):
        A1.T @ v1
    with raises(DimensionMismatch):
        A2 @ v1
    with raises(DimensionMismatch):
        v1 @ A2.T
    with raises(DimensionMismatch):
        A1 @ A1
    with raises(DimensionMismatch):
        A1.T @ A1.T
    with raises(DimensionMismatch):
        A1 @= A1
    with raises(TypeError):
        s1 @ 1
    with raises(TypeError):
        1 @ s1

    w = v1[:1].new()
    with raises(DimensionMismatch):
        w @ v1
    with raises(TypeError):
        v1 @= v1

    with raises(TypeError, match="Bad type when calling semiring.plus_times"):
        op.plus_times(A1)
    with raises(TypeError, match="Bad types when calling semiring.plus_times."):
        op.plus_times(A1, A2)
    with raises(TypeError, match="Bad types when calling semiring.plus_times."):
        op.plus_times(A1 @ A2, 1)


def test_apply_unary(v1, A1):
    expected = v1.apply(op.exp).new()
    assert expected.isequal(op.exp(v1).new())
    assert expected.isequal(op.exp[float](v1).new())

    expected = A1.apply(op.exp).new()
    assert expected.isequal(op.exp(A1).new())


@autocompute
def test_apply_unary_bad(s1, v1):
    with raises(TypeError, match="__call__"):
        op.exp(v1, 1)
    with raises(TypeError, match="__call__"):
        op.exp(1, v1)
    with raises(TypeError, match="Bad type when calling unary.exp"):
        op.exp(s1)
    with raises(TypeError, match="Bad type when calling unary.exp"):
        op.exp(1)
    with raises(TypeError, match="Bad dtype"):
        op.exp(v1 | v1)


def test_apply_binary(v1, A1):
    expected = v1.apply(monoid.plus, right=2).new()
    assert expected.isequal(monoid.plus(v1, 2).new())
    assert expected.isequal(monoid.plus[float](v1, 2).new())

    expected = v1.apply(op.minus, right=2).new()
    assert expected.isequal(op.minus(v1, 2).new())
    assert expected.isequal(op.minus[float](v1, 2).new())

    expected = v1.apply(op.minus, left=2).new()
    assert expected.isequal(op.minus(2, v1).new())

    expected = A1.apply(op.minus, right=2).new()
    assert expected.isequal(op.minus(A1, 2).new())

    expected = A1.apply(op.minus, left=2).new()
    assert expected.isequal(op.minus(2, A1).new())


def test_apply_binary_bad(s1, v1):
    with raises(TypeError, match="Bad types when calling binary.plus"):
        op.plus(1, 1)
    with raises(TypeError, match="Bad type when calling binary.plus"):
        op.plus(v1)
    with raises(TypeError, match="Bad type for keyword argument `right="):
        op.plus(v1, v1)
    with raises(TypeError, match="may only be used when performing an ewise_add"):
        op.plus(v1, 1, require_monoid=False)


def test_infix_nonscalars(v1, v2):
    # with raises(TypeError, match="refuse to guess"):
    assert (v1 + v2).new().isequal(op.plus(v1 | v2).new())
    # with raises(TypeError, match="refuse to guess"):
    assert (v1 - v2).new().isequal(v1.ewise_union(v2, "-", 0, 0).new())


@autocompute
def test_inplace_infix(s1, v1, v2, A1, A2):
    A = Matrix.new(float, nrows=3, ncols=3)
    A[:, :] = 1
    x = v1.dup()
    x @= A
    assert isinstance(x, Vector)
    assert x.isequal(v1 @ A)
    with raises(TypeError, match="not supported for FP64"):
        v1 |= v2
    with raises(TypeError, match="not supported for FP64"):
        A1 &= A2.T

    v1 = v1.dup(bool)
    v2 = v2.dup(bool)
    A1 = A1.dup(bool)
    A2 = A2.dup(bool)
    x = v1.dup()
    x |= v2
    assert isinstance(x, Vector)
    assert x.isequal(v1 | v2)
    x = v1.dup()
    x &= v2
    assert isinstance(x, Vector)
    assert x.isequal(v1 & v2)
    x = A1.dup()
    x |= A2.T
    assert isinstance(x, Matrix)
    assert x.isequal(A1 | A2.T)
    x = A1.dup()
    x &= A2.T
    assert isinstance(x, Matrix)
    assert x.isequal(A1 & A2.T)

    expr = v1 | v2
    with raises(TypeError, match="not supported"):
        expr |= v1
    with raises(TypeError, match="not supported"):
        expr &= v1
    with raises(TypeError, match="not supported"):
        expr @= A
    with raises(TypeError, match="not supported"):
        s1 @= v1

import pytest

from graphblas import binary, monoid, op
from graphblas.exceptions import DimensionMismatch

from .conftest import autocompute

from graphblas import Matrix, Scalar, Vector  # isort:skip (for dask-graphblas)


@pytest.fixture
def v1():
    return Vector.from_coo([0, 2], [2.0, 5.0], name="v_1")


@pytest.fixture
def v2():
    return Vector.from_coo([1, 2], [3.0, 7.0], name="v_2")


@pytest.fixture
def A1():
    return Matrix.from_coo([0, 0], [0, 1], [0.0, 4.0], ncols=3, name="A_1")


@pytest.fixture
def A2():
    return Matrix.from_coo([0, 2], [0, 0], [6.0, 8.0], name="A_2")


@pytest.fixture
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

        expected = left.ewise_add(right, op.minus).new()
        assert expected.isequal(op.minus(left | right).new())
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
        (A1, s1),
        (s1, A1),
        (A1.T, s1),
        (s1, A1.T),
        (A1, 1),
        (1, A1),
    ]:
        with pytest.raises(TypeError, match="Bad type for argument"):
            left | right
        with pytest.raises(TypeError, match="Bad type for argument"):
            left & right
    # These are okay now
    for left, right in [
        (A1, v1),
        (v1, A1.T),
    ]:
        left.ewise_add(right)
        left | right
        left.ewise_mult(right)
        left & right
        left.ewise_union(right, op.plus, 0, 0)
    # Wrong dimension; can't broadcast
    for left, right in [
        (v1, A1),
        (A1.T, v1),
    ]:
        with pytest.raises(DimensionMismatch):
            left | right
        with pytest.raises(DimensionMismatch):
            left & right
        with pytest.raises(DimensionMismatch):
            left.ewise_add(right)
        with pytest.raises(DimensionMismatch):
            left.ewise_mult(right)
        with pytest.raises(DimensionMismatch):
            left.ewise_union(right, op.plus, 0, 0)

    w = v1[: v1.size - 1].new()
    with pytest.raises(DimensionMismatch):
        v1 | w
    with pytest.raises(DimensionMismatch):
        v1 & w
    with pytest.raises(DimensionMismatch):
        A2 | A1
    with pytest.raises(DimensionMismatch):
        A2 & A1
    with pytest.raises(DimensionMismatch):
        A1.T | A1
    with pytest.raises(DimensionMismatch):
        A1.T & A1

    # These are okay now
    # with pytest.raises(TypeError):
    s1 | 1
    # with pytest.raises(TypeError):
    1 | s1
    # with pytest.raises(TypeError):
    s1 & 1
    # with pytest.raises(TypeError):
    1 & s1

    with pytest.raises(TypeError, match="not supported for FP64"):
        v1 |= v1
    with pytest.raises(TypeError, match="not supported for FP64"):
        A1 |= A1
    with pytest.raises(TypeError, match="not supported for FP64"):
        v1 &= v1
    with pytest.raises(TypeError, match="not supported for FP64"):
        A1 &= A1

    op.minus(v1 | v1)  # ok now
    with pytest.raises(TypeError, match="unexpected keyword argument 'require_monoid'"):
        op.minus(v1 & v1, require_monoid=False)
    with pytest.raises(TypeError, match="Bad dtype"):
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
        with pytest.raises(TypeError, match="Bad type for argument"):
            left @ right

    with pytest.raises(DimensionMismatch):
        v1 @ A1
    with pytest.raises(DimensionMismatch):
        A1.T @ v1
    with pytest.raises(DimensionMismatch):
        A2 @ v1
    with pytest.raises(DimensionMismatch):
        v1 @ A2.T
    with pytest.raises(DimensionMismatch):
        A1 @ A1
    with pytest.raises(DimensionMismatch):
        A1.T @ A1.T
    with pytest.raises(DimensionMismatch):
        A1 @= A1
    with pytest.raises(TypeError):
        s1 @ 1
    with pytest.raises(TypeError):
        1 @ s1

    w = v1[:1].new()
    with pytest.raises(DimensionMismatch):
        w @ v1
    with pytest.raises(TypeError):
        v1 @= v1

    with pytest.raises(TypeError, match="Bad type when calling semiring.plus_times"):
        op.plus_times(A1)
    with pytest.raises(TypeError, match="Bad types when calling semiring.plus_times."):
        op.plus_times(A1, A2)
    with pytest.raises(TypeError, match="Bad types when calling semiring.plus_times."):
        op.plus_times(A1 @ A2, 1)


def test_apply_unary(v1, A1):
    expected = v1.apply(op.exp).new()
    assert expected.isequal(op.exp(v1).new())
    assert expected.isequal(op.exp[float](v1).new())

    expected = A1.apply(op.exp).new()
    assert expected.isequal(op.exp(A1).new())


@autocompute
def test_apply_unary_bad(s1, v1):
    with pytest.raises(TypeError, match="__call__"):
        op.exp(v1, 1)
    with pytest.raises(TypeError, match="__call__"):
        op.exp(1, v1)
    # with pytest.raises(TypeError, match="Bad type when calling unary.exp"):
    op.exp(s1)  # Okay now
    # with pytest.raises(TypeError, match="Bad type when calling unary.exp"):
    op.exp(1)  # Okay now
    with pytest.raises(TypeError, match="Bad dtype"):
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


def test_apply_binary_bad(v1):
    # with pytest.raises(TypeError, match="Bad types when calling binary.plus"):
    op.plus(1, 1)  # Okay now
    with pytest.raises(TypeError, match="Bad type when calling binary.plus"):
        op.plus(v1)
    with pytest.raises(TypeError, match="Bad type for keyword argument `right="):
        op.plus(v1, v1)
    with pytest.raises(TypeError, match="unexpected keyword argument 'require_monoid'"):
        op.plus(v1, 1, require_monoid=False)


def test_infix_nonscalars(v1, v2):
    # with pytest.raises(TypeError, match="refuse to guess"):
    assert (v1 + v2).new().isequal(op.plus(v1 | v2).new())
    # with pytest.raises(TypeError, match="refuse to guess"):
    assert (v1 - v2).new().isequal(v1.ewise_union(v2, "-", 0, 0).new())


@autocompute
def test_inplace_infix(s1, v1, v2, A1, A2):
    A = Matrix(float, nrows=3, ncols=3)
    A[:, :] = 1
    x = v1.dup()
    x @= A
    assert isinstance(x, Vector)
    assert x.isequal(v1 @ A)
    with pytest.raises(TypeError, match="not supported for FP64"):
        v1 |= v2
    with pytest.raises(TypeError, match="not supported for FP64"):
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
    with pytest.raises(TypeError, match="not supported"):
        expr |= v1
    with pytest.raises(TypeError, match="not supported"):
        expr &= v1
    with pytest.raises(TypeError, match="not supported"):
        expr @= A
    with pytest.raises(TypeError, match="not supported"):
        s1 @= v1


@autocompute
def test_infix_expr_value_types():
    """Test bug where `infix_expr._value` was used as MatrixExpression or Matrix."""
    from graphblas.core.matrix import MatrixExpression

    A = Matrix(int, 3, 3)
    A << 1
    expr = A @ A.T
    assert expr._expr is None
    assert expr._value is None
    assert type(expr._get_value()) is Matrix
    assert type(expr._expr) is MatrixExpression
    assert type(expr.new()) is Matrix
    assert expr._expr is not None
    assert expr._value is None
    assert type(expr.new()) is Matrix
    assert type(expr._get_value()) is Matrix
    assert expr._expr is not None
    assert expr._value is not None
    assert expr._expr._value is not None
    expr._value = None
    assert expr._value is None
    assert expr._expr._value is None


def test_multi_infix_vector():
    D0 = Vector.from_scalar(0, 3).diag()
    v1 = Vector.from_coo([0, 1], [1, 2], size=3)  # 1 2 .
    v2 = Vector.from_coo([1, 2], [1, 2], size=3)  # . 1 2
    v3 = Vector.from_coo([2, 0], [1, 2], size=3)  # 2 . 1
    # ewise_add
    result = binary.plus((v1 | v2) | v3).new()
    expected = Vector.from_scalar(3, size=3)
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | v3)).new()
    assert result.isequal(expected)
    result = monoid.min(v1 | v2 | v3).new()
    expected = Vector.from_scalar(1, size=3)
    assert result.isequal(expected)
    # ewise_mult
    result = monoid.max((v1 & v2) & v3).new()
    expected = Vector(int, size=3)
    assert result.isequal(expected)
    result = monoid.max(v1 & (v2 & v3)).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & v2) & v1).new()
    expected = Vector.from_coo([1], [1], size=3)
    assert result.isequal(expected)
    # ewise_union
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10).new()
    expected = Vector.from_scalar(13, size=3)
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10.0).new()
    expected = Vector.from_scalar(13.0, size=3)
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)
    # inner
    assert op.plus_plus(v1 @ v1).new().value == 6
    assert op.plus_plus(v1 @ (v1 @ D0)).new().value == 6
    assert op.plus_plus((D0 @ v1) @ v1).new().value == 6
    # matrix-vector ewise_add
    result = binary.plus((D0 | v1) | v2).new()
    expected = binary.plus(binary.plus(D0 | v1).new() | v2).new()
    assert result.isequal(expected)
    result = binary.plus(D0 | (v1 | v2)).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | D0).new()
    assert result.isequal(expected.T)
    result = binary.plus(v1 | (v2 | D0)).new()
    assert result.isequal(expected.T)
    # matrix-vector ewise_mult
    result = binary.plus((D0 & v1) & v2).new()
    expected = binary.plus(binary.plus(D0 & v1).new() & v2).new()
    assert result.isequal(expected)
    assert result.nvals > 0
    result = binary.plus(D0 & (v1 & v2)).new()
    assert result.isequal(expected)
    result = binary.plus((v1 & v2) & D0).new()
    assert result.isequal(expected.T)
    result = binary.plus(v1 & (v2 & D0)).new()
    assert result.isequal(expected.T)
    # matrix-vector ewise_union
    kwargs = {"left_default": 10, "right_default": 20}
    result = binary.plus((D0 | v1) | v2, **kwargs).new()
    expected = binary.plus(binary.plus(D0 | v1, **kwargs).new() | v2, **kwargs).new()
    assert result.isequal(expected)
    result = binary.plus(D0 | (v1 | v2), **kwargs).new()
    expected = binary.plus(D0 | binary.plus(v1 | v2, **kwargs).new(), **kwargs).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | D0, **kwargs).new()
    expected = binary.plus(binary.plus(v1 | v2, **kwargs).new() | D0, **kwargs).new()
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | D0), **kwargs).new()
    expected = binary.plus(v1 | binary.plus(v2 | D0, **kwargs).new(), **kwargs).new()
    assert result.isequal(expected)
    # vxm, mxv
    result = op.plus_plus((D0 @ v1) @ D0).new()
    assert result.isequal(v1)
    result = op.plus_plus(D0 @ (v1 @ D0)).new()
    assert result.isequal(v1)
    result = op.plus_plus(v1 @ (D0 @ D0)).new()
    assert result.isequal(v1)
    result = op.plus_plus((D0 @ D0) @ v1).new()
    assert result.isequal(v1)
    result = op.plus_plus((v1 @ D0) @ D0).new()
    assert result.isequal(v1)
    result = op.plus_plus(D0 @ (D0 @ v1)).new()
    assert result.isequal(v1)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2).__ror__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__ror__(v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) | (v2 & v3)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__rand__(v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2).__rand__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 & v3)

    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 | v2).ewise_mult(v3)


@autocompute
def test_multi_infix_vector_auto():
    v1 = Vector.from_coo([0, 1], [1, 2], size=3)  # 1 2 .
    v2 = Vector.from_coo([1, 2], [1, 2], size=3)  # . 1 2
    v3 = Vector.from_coo([2, 0], [1, 2], size=3)  # 2 . 1
    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 | v2).ewise_mult(v3)


def test_multi_infix_matrix():
    # Adapted from test_multi_infix_vector
    D0 = Vector.from_scalar(0, 3).diag()
    v1 = Matrix.from_coo([0, 1], [0, 0], [1, 2], nrows=3)  # 1 2 .
    v2 = Matrix.from_coo([1, 2], [0, 0], [1, 2], nrows=3)  # . 1 2
    v3 = Matrix.from_coo([2, 0], [0, 0], [1, 2], nrows=3)  # 2 . 1
    # ewise_add
    result = binary.plus((v1 | v2) | v3).new()
    expected = Matrix.from_scalar(3, 3, 1)
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | v3)).new()
    assert result.isequal(expected)
    result = monoid.min(v1 | v2 | v3).new()
    expected = Matrix.from_scalar(1, 3, 1)
    assert result.isequal(expected)
    result = binary.plus(v1 | v1 | v1 | v1 | v1).new()
    expected = (5 * v1).new()
    assert result.isequal(expected)
    # ewise_mult
    result = monoid.max((v1 & v2) & v3).new()
    expected = Matrix(int, 3, 1)
    assert result.isequal(expected)
    result = monoid.max(v1 & (v2 & v3)).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & v2) & v1).new()
    expected = Matrix.from_coo([1], [0], [1], nrows=3)
    assert result.isequal(expected)
    result = binary.plus(v1 & v1 & v1 & v1 & v1).new()
    expected = (5 * v1).new()
    assert result.isequal(expected)
    # ewise_union
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10).new()
    expected = Matrix.from_scalar(13, 3, 1)
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10.0).new()
    expected = Matrix.from_scalar(13.0, 3, 1)
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)
    # mxm
    assert op.plus_plus(v1.T @ v1).new()[0, 0].new().value == 6
    assert op.plus_plus(v1 @ (v1.T @ D0)).new()[0, 0].new().value == 2
    assert op.plus_plus((v1.T @ D0) @ v1).new()[0, 0].new().value == 6
    assert op.plus_plus(D0 @ D0 @ D0 @ D0 @ D0).new().isequal(D0)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2).__ror__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__ror__(v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) | (v2 & v3)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__rand__(v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2).__rand__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 & v3)

    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 | v2).ewise_mult(v3)


@autocompute
def test_multi_infix_matrix_auto():
    v1 = Matrix.from_coo([0, 1], [0, 0], [1, 2], nrows=3)  # 1 2 .
    v2 = Matrix.from_coo([1, 2], [0, 0], [1, 2], nrows=3)  # . 1 2
    v3 = Matrix.from_coo([2, 0], [0, 0], [1, 2], nrows=3)  # 2 . 1
    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 | v2).ewise_mult(v3)


def test_multi_infix_scalar():
    # Adapted from test_multi_infix_vector
    v1 = Scalar.from_value(1)
    v2 = Scalar.from_value(2)
    v3 = Scalar(int)
    # ewise_add
    result = binary.plus((v1 | v2) | v3).new()
    expected = 3
    assert result.isequal(expected)
    result = binary.plus((1 | v2) | v3).new()
    assert result.isequal(expected)
    result = binary.plus((1 | v2) | 0).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | 2) | v3).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | 2) | 0).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | 0).new()
    assert result.isequal(expected)

    result = binary.plus(v1 | (v2 | v3)).new()
    assert result.isequal(expected)
    result = binary.plus(1 | (v2 | v3)).new()
    assert result.isequal(expected)
    result = binary.plus(1 | (2 | v3)).new()
    assert result.isequal(expected)
    result = binary.plus(1 | (v2 | 0)).new()
    assert result.isequal(expected)
    result = binary.plus(v1 | (2 | v3)).new()
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | 0)).new()
    assert result.isequal(expected)

    result = monoid.min(v1 | v2 | v3).new()
    expected = 1
    assert result.isequal(expected)
    # ewise_mult
    result = monoid.max((v1 & v2) & v3).new()
    expected = None
    assert result.isequal(expected)
    result = monoid.max(v1 & (v2 & v3)).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & v2) & v1).new()
    expected = 1
    assert result.isequal(expected)

    result = monoid.min((1 & v2) & v1).new()
    assert result.isequal(expected)
    result = monoid.min((1 & v2) & 1).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & 2) & v1).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & 2) & 1).new()
    assert result.isequal(expected)
    result = monoid.min((v1 & v2) & 1).new()
    assert result.isequal(expected)

    result = monoid.min(1 & (v2 & v1)).new()
    assert result.isequal(expected)
    result = monoid.min(1 & (2 & v1)).new()
    assert result.isequal(expected)
    result = monoid.min(1 & (v2 & 1)).new()
    assert result.isequal(expected)
    result = monoid.min(v1 & (2 & v1)).new()
    assert result.isequal(expected)
    result = monoid.min(v1 & (v2 & 1)).new()
    assert result.isequal(expected)

    # ewise_union
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10).new()
    expected = 13
    assert result.isequal(expected)
    result = binary.plus((1 | v2) | v3, left_default=10, right_default=10).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | 2) | v3, left_default=10, right_default=10).new()
    assert result.isequal(expected)
    result = binary.plus((v1 | v2) | v3, left_default=10, right_default=10.0).new()
    assert result.isequal(expected)
    result = binary.plus(v1 | (v2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)
    result = binary.plus(1 | (v2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)
    result = binary.plus(1 | (2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)
    result = binary.plus(v1 | (2 | v3), left_default=10, right_default=10).new()
    assert result.isequal(expected)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2).__ror__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) | (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 | (v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__ror__(v2 & v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) | (v2 & v3)

    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1 & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        v1.__rand__(v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 & v2) & (v2 | v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & v3
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2).__rand__(v3)
    with pytest.raises(TypeError, match="XXX"):  # TODO
        (v1 | v2) & (v2 & v3)

    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="to automatically compute"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="Automatic computation"):
        (v1 | v2).ewise_mult(v3)


@autocompute
def test_multi_infix_scalar_auto():
    v1 = Scalar.from_value(1)
    v2 = Scalar.from_value(2)
    v3 = Scalar(int)
    # We differentiate between infix and methods
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_add(v2 & v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_add(v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_union(v2 & v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 & v2).ewise_union(v3, binary.plus, left_default=1, right_default=1)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        v1.ewise_mult(v2 | v3)
    with pytest.raises(TypeError, match="only valid for BOOL"):
        (v1 | v2).ewise_mult(v3)

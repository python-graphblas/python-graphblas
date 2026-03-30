import pickle

import pytest

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, dtypes, indexbinary
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.operator.indexbinary import _has_idxbinop
from graphblas.exceptions import UdfParseError

pytestmark = [
    pytest.mark.skipif(not supports_udfs, reason="requires numba"),
    pytest.mark.skipif(not _has_idxbinop, reason="requires SuiteSparse:GraphBLAS 9.4+"),
]


def test_register_anonymous():
    def add_with_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_with_theta)
    assert op is not None
    assert "add_with_theta" in op.name
    assert int in op.types or dtypes.INT64 in op.types


def test_register_new():
    def my_idxbin(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x * y + theta

    result = indexbinary.register_new("my_idxbin", my_idxbin)
    assert result is not None
    assert hasattr(indexbinary, "my_idxbin")

    A = Matrix.from_coo([0, 1], [1, 0], [3, 7])
    B = Matrix.from_coo([0, 1], [1, 0], [5, 2])
    binop = indexbinary.my_idxbin(100)
    C = A.ewise_mult(B, binop).new()
    assert list(C.to_coo()[2]) == [115, 114]  # 3*5+100, 7*2+100

    delattr(indexbinary, "my_idxbin")


def test_register_new_lazy():
    def lazy_op(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y

    result = indexbinary.register_new("lazy_op", lazy_op, lazy=True)
    assert result is None
    assert "lazy_op" in dir(indexbinary)

    op = indexbinary.lazy_op
    assert op is not None
    delattr(indexbinary, "lazy_op")


def test_typed_call():
    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    typed = op[int]
    binop = typed(10)
    assert binop.opclass == "BinaryOp"

    A = Matrix.from_coo([0, 1], [1, 0], [3, 7])
    B = Matrix.from_coo([0, 1], [1, 0], [5, 2])
    C = A.ewise_mult(B, binop).new()
    assert list(C.to_coo()[2]) == [18, 19]  # 3+5+10, 7+2+10


def test_untyped_call():
    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    binop = op(10)
    assert binop.opclass == "BinaryOp"

    A = Matrix.from_coo([0, 1], [1, 0], [3, 7])
    B = Matrix.from_coo([0, 1], [1, 0], [5, 2])
    C = A.ewise_mult(B, binop).new()
    assert list(C.to_coo()[2]) == [18, 19]


def test_index_aware():
    """Test that indices are correctly passed to the function."""

    def index_sum(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return ix + jx + iy + jy + theta

    op = indexbinary.register_anonymous(index_sum)
    # For ewise_mult, both operands have the same indices: ix==iy and jx==jy
    A = Matrix.from_coo([0, 1, 2], [0, 1, 2], [100, 200, 300])
    B = Matrix.from_coo([0, 1, 2], [0, 1, 2], [1, 1, 1])
    binop = op[int](0)
    C = A.ewise_mult(B, binop).new()
    # (0,0): 0+0+0+0+0=0, (1,1): 1+1+1+1+0=4, (2,2): 2+2+2+2+0=8
    assert list(C.to_coo()[2]) == [0, 4, 8]


def test_floating_point():
    def fp_add(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta * 0.5

    op = indexbinary.register_anonymous(fp_add)
    A = Matrix.from_coo([0], [0], [1.5])
    B = Matrix.from_coo([0], [0], [2.5])
    binop = op(4.0)
    C = A.ewise_mult(B, binop).new()
    assert abs(C.to_coo()[2][0] - 6.0) < 1e-10  # 1.5 + 2.5 + 4.0*0.5 = 6.0


def test_vector_ewise():
    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    v1 = Vector.from_coo([0, 1, 2], [10, 20, 30])
    v2 = Vector.from_coo([0, 1, 2], [1, 2, 3])
    binop = op(0)
    v3 = v1.ewise_mult(v2, binop).new()
    assert list(v3.to_coo()[1]) == [11, 22, 33]


def test_ewise_add():
    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    A = Matrix.from_coo([0, 1], [0, 1], [3, 7])
    B = Matrix.from_coo([0, 1], [0, 1], [5, 2])
    binop = op(10)
    C = A.ewise_add(B, binop).new()
    assert list(C.to_coo()[2]) == [18, 19]


def test_default_theta():
    """Test that theta=0 works correctly."""

    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    binop = op(0)  # theta=0 as int
    A = Matrix.from_coo([0], [0], [3])
    B = Matrix.from_coo([0], [0], [5])
    C = A.ewise_mult(B, binop).new()
    assert C.to_coo()[2][0] == 8  # 3 + 5 + 0


def test_bool_return():
    def is_close(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return abs(x - y) <= theta

    op = indexbinary.register_anonymous(is_close)
    A = Matrix.from_coo([0, 1], [0, 1], [10, 20])
    B = Matrix.from_coo([0, 1], [0, 1], [11, 25])
    binop = op[int](2)
    C = A.ewise_mult(B, binop).new()
    assert list(C.to_coo()[2]) == [True, False]  # |10-11|<=2, |20-25|>2


def test_scalar_theta():
    """Test passing a graphblas Scalar as theta."""

    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_theta)
    theta = Scalar.from_value(42)
    binop = op[int](theta)
    A = Matrix.from_coo([0], [0], [3])
    B = Matrix.from_coo([0], [0], [5])
    C = A.ewise_mult(B, binop).new()
    assert C.to_coo()[2][0] == 50  # 3 + 5 + 42


def test_parameterized():
    def make_op(scale):
        def inner(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
            return (x + y) * scale + theta

        return inner

    op = indexbinary.register_anonymous(make_op, parameterized=True)
    scaled_op = op(2)  # scale=2
    binop = scaled_op(10)  # theta=10
    A = Matrix.from_coo([0], [0], [3])
    B = Matrix.from_coo([0], [0], [5])
    C = A.ewise_mult(B, binop).new()
    assert C.to_coo()[2][0] == 26  # (3+5)*2 + 10 = 26


def test_pickle_registered():
    def add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    indexbinary.register_new("pickle_test_op", add_theta)
    op = indexbinary.pickle_test_op
    op2 = pickle.loads(pickle.dumps(op))
    assert op2.name == op.name

    typed = op[int]
    typed2 = pickle.loads(pickle.dumps(typed))
    assert typed2.name == typed.name

    delattr(indexbinary, "pickle_test_op")


def test_bad_udf():
    with pytest.raises(UdfParseError, match="Unable to parse function using Numba"):
        indexbinary.register_anonymous(lambda x, ix, jx, y, iy, jy, theta: result)  # noqa: F821


def test_bad_type():
    with pytest.raises(TypeError, match="UDF argument must be a function"):
        indexbinary.register_anonymous(42)


def test_with_mask():
    def add_vals(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_vals)
    A = Matrix.from_coo([0, 1, 2], [0, 1, 2], [3, 7, 11])
    B = Matrix.from_coo([0, 1, 2], [0, 1, 2], [5, 2, 1])
    mask = Matrix.from_coo([0, 2], [0, 2], [True, True], nrows=3, ncols=3)
    C = Matrix(int, nrows=3, ncols=3)
    binop = op(10)
    C(mask=mask.S) << A.ewise_mult(B, binop)
    rows, _, vals = C.to_coo()
    assert list(rows) == [0, 2]
    assert list(vals) == [18, 22]  # 3+5+10, 11+1+10


def test_with_accumulator():
    def add_vals(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_vals)
    A = Matrix.from_coo([0], [0], [3])
    B = Matrix.from_coo([0], [0], [5])
    C = Matrix.from_coo([0], [0], [100])
    binop = op(10)
    C(accum=gb.binary.plus) << A.ewise_mult(B, binop)
    assert C.to_coo()[2][0] == 118  # 100 + (3+5+10)


def test_ewise_with_bound_binop():
    """Confirm bound IndexBinaryOp works in all ewise operations."""

    def mul_plus(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x * y + theta

    op = indexbinary.register_anonymous(mul_plus)
    binop = op[int](0)
    A = Matrix.from_coo([0, 0], [0, 1], [2, 3])
    B = Matrix.from_coo([0, 0], [0, 1], [4, 5])
    C = A.ewise_mult(B, binop).new()
    assert list(C.to_coo()[2]) == [8, 15]  # 2*4+0, 3*5+0


def test_find_opclass():
    from graphblas.core.operator import find_opclass

    def add_vals(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y

    op = indexbinary.register_anonymous(add_vals)
    _, opclass = find_opclass(op)
    assert opclass == "IndexBinaryOp"

    typed = op[int]
    assert typed.opclass == "IndexBinaryOp"

    bound = typed(0)
    assert bound.opclass == "BinaryOp"


def test_dir_and_module():
    assert "register_new" in dir(indexbinary)
    assert "register_anonymous" in dir(indexbinary)
    assert "ss" in dir(indexbinary)
    # Actually access the ss module to verify it exists (not just in __dir__)
    ss = indexbinary.ss
    assert hasattr(ss, "_delayed")
    assert hasattr(ss, "register_new")

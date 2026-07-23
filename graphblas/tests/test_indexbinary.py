import pickle

import numpy as np
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
    assert "add_with_theta" in op.name
    assert int in op.types or dtypes.INT64 in op.types


def test_register_new():
    def my_idxbin(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x * y + theta

    result = indexbinary.register_new("my_idxbin", my_idxbin)
    assert hasattr(indexbinary, "my_idxbin")
    assert result is indexbinary.my_idxbin

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
    assert op.name == "lazy_op"
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
    """The wrapper passes ``(ix, jx, iy, jy)`` through to the user function."""

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
    """``theta=0`` is a valid bind value and does not get treated as missing."""

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
    """A graphblas ``Scalar`` is accepted as a theta bind value."""

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


# Module-level so pickle can serialize the function reference. IBO
# ``__reduce__`` for user-registered ops returns a tuple containing the
# original function; pickling needs the function to be globally importable
# (i.e., not a closure / local function).
def _ibo_add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
    return x + y + theta


def _ibo_return_x(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
    return x


def test_pickle_registered():
    indexbinary.register_new("pickle_test_op", _ibo_add_theta)
    try:
        op = indexbinary.pickle_test_op
        op2 = pickle.loads(pickle.dumps(op))
        assert op2.name == op.name

        typed = op[int]
        typed2 = pickle.loads(pickle.dumps(typed))
        assert typed2.name == typed.name

        # Unpickled op must actually run, not just carry the right name.
        bound = op2[int](7)
        A = Matrix.from_coo([0, 1], [0, 1], [3, 11])
        B = Matrix.from_coo([0, 1], [0, 1], [5, 1])
        C = A.ewise_mult(B, bound).new()
        # (3+5)+7 = 15; (11+1)+7 = 19
        assert C.isequal(Matrix.from_coo([0, 1], [0, 1], [15, 19]))
    finally:
        delattr(indexbinary, "pickle_test_op")


def test_pickle_bound():
    indexbinary.register_new("pickle_bound_test_op", _ibo_add_theta)
    try:
        bound = indexbinary.pickle_bound_test_op[int](7)
        bound2 = pickle.loads(pickle.dumps(bound))
        assert bound2._theta == 7
        A = Matrix.from_coo([0, 1], [0, 1], [3, 11])
        B = Matrix.from_coo([0, 1], [0, 1], [5, 1])
        C1 = A.ewise_mult(B, bound).new()
        C2 = A.ewise_mult(B, bound2).new()
        assert C1.isequal(C2)
    finally:
        delattr(indexbinary, "pickle_bound_test_op")


def test_pickle_bound_array_udt_theta():
    """Bound IBO with an array-UDT theta value round-trips through pickle.

    Regression: ``_BoundIndexBinaryOp.__reduce__`` stored ``_theta`` as the
    raw numpy array. On unpickle, ``_rebind_indexbinaryop`` passed it back to
    ``TypedBuiltinIndexBinaryOp.__call__``, which called
    ``Scalar.from_value(theta)`` with no dtype; that crashed inside the
    multi-dim ndarray ``Scalar.value`` setter. The fix wraps the value as a
    typed Scalar in ``_rebind_indexbinaryop`` when the thunk type is a UDT.
    """
    from graphblas import dtypes

    arr_udt = dtypes.register_anonymous(np.dtype((np.int64, (3,))), "_BoundIboArr3")
    indexbinary.register_new("pickle_bound_array_udt_op", _ibo_return_x, is_udt=True)
    try:
        op = indexbinary.pickle_bound_array_udt_op
        theta_scalar = gb.Scalar(arr_udt)
        theta_scalar.value = np.array([1, 2, 3], dtype=np.int64)
        bound = op[arr_udt](theta_scalar)
        bound2 = pickle.loads(pickle.dumps(bound))
        # ``_theta`` round-trips bit-identically.
        assert np.array_equal(bound2._theta, np.array([1, 2, 3], dtype=np.int64))
        # The unpickled bound op is a fresh ``_BoundIndexBinaryOp`` wrapping a
        # different ``GrB_BinaryOp`` pointing at the same kernel; pin the
        # introspection state that previously crashed.
        assert bound2.parent is bound.parent
        assert bound2.type is arr_udt
    finally:
        delattr(indexbinary, "pickle_bound_array_udt_op")


def test_bind_raw_array_udt_theta():
    """A raw (non-Scalar) array-UDT theta is bound using the known thunk dtype.

    Regression: the ``not isinstance(theta, Scalar)`` branch of the IBO
    ``__call__`` passed the raw value to ``Scalar.from_value`` with no dtype,
    which inferred a scalar element type and crashed on the multi-dim ndarray.
    The typed op now supplies its thunk dtype; the untyped op uses the explicit
    ``dtype=`` argument.
    """
    from graphblas import dtypes

    arr_udt = dtypes.register_anonymous(np.dtype((np.int64, (3,))), "_RawThetaArr3")
    indexbinary.register_new("raw_array_udt_op", _ibo_return_x, is_udt=True)
    try:
        op = indexbinary.raw_array_udt_op
        raw = np.array([1, 2, 3], dtype=np.int64)
        # Typed op: ``op[arr_udt]`` knows the thunk dtype.
        bound = op[arr_udt](raw)
        assert bound.type is arr_udt
        assert np.array_equal(bound._theta, raw)
        # Untyped op: the explicit ``dtype=`` supplies the UDT.
        bound2 = op(raw, dtype=arr_udt)
        assert bound2.type is arr_udt
        assert np.array_equal(bound2._theta, raw)
    finally:
        delattr(indexbinary, "raw_array_udt_op")


def test_bind_raw_udt_theta_without_dtype_errors():
    """A raw UDT theta with no dtype can't be inferred; the error is clear and actionable."""
    indexbinary.register_new("raw_no_dtype_op", _ibo_return_x, is_udt=True)
    try:
        op = indexbinary.raw_no_dtype_op
        with pytest.raises(TypeError, match="Cannot infer a dtype for theta"):
            op(np.array([1, 2, 3], dtype=np.int64))
        with pytest.raises(TypeError, match="Cannot infer a dtype for theta"):
            op((1, 2, 3))
    finally:
        delattr(indexbinary, "raw_no_dtype_op")


def test_pickle_bound_record_udt_theta():
    """Bound IBO with a record-UDT theta value round-trips through pickle.

    Companion to :func:`test_pickle_bound_array_udt_theta` for the record
    flavor of UDT; the array case was the one that crashed before the
    ``_rebind_indexbinaryop`` Scalar-wrap fix, and the record case worked
    only because ``_theta`` happened to be a 0-d ``np.void`` that
    ``Scalar.value =`` accepted. Pin the contract for both shapes so a
    regression in either is loud.
    """
    from graphblas import dtypes

    rec_udt = dtypes.register_anonymous(
        np.dtype([("a", np.int64), ("b", np.int64)], align=True),
        "_BoundIboRec",
    )
    indexbinary.register_new("pickle_bound_record_udt_op", _ibo_return_x, is_udt=True)
    try:
        op = indexbinary.pickle_bound_record_udt_op
        theta_scalar = gb.Scalar(rec_udt)
        theta_scalar.value = (5, 7)
        bound = op[rec_udt](theta_scalar)
        bound2 = pickle.loads(pickle.dumps(bound))
        assert tuple(bound2._theta) == (5, 7)
        assert bound2.parent is bound.parent
        assert bound2.type is rec_udt
    finally:
        delattr(indexbinary, "pickle_bound_record_udt_op")


@pytest.mark.slow
def test_bound_ibo_memory_bounded():
    """Regression: ``iop(theta)`` used to leak one ``GrB_BinaryOp`` per call.

    Bound IBOs are never cached, so a loop that re-binds with different
    theta values leaks unboundedly without the ``TypedOpBase.__del__`` free.
    50k bound IBOs are well above the noise floor for the leak (each is
    several hundred bytes of SS state plus a python-level wrapper); RSS
    growth should stay below a few MB after GC drops them.
    """
    import gc
    import resource
    import sys

    if hasattr(indexbinary, "leak_test_bound_ibo_op"):
        delattr(indexbinary, "leak_test_bound_ibo_op")
    indexbinary.register_new("leak_test_bound_ibo_op", _ibo_add_theta)
    op = indexbinary.leak_test_bound_ibo_op
    try:
        # ``ru_maxrss`` is bytes on macOS, KB on Linux.
        scale = 1024 if sys.platform.startswith("linux") else 1024 * 1024

        def rss_mb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

        # Warm up to populate typed-op cache; first call also touches the
        # numba sample-types path.
        for _ in range(100):
            _ = op[int](0)
        gc.collect()
        before = rss_mb()
        for i in range(50_000):
            _ = op[int](i)
        gc.collect()
        after = rss_mb()
        # 50k leaked GrB_BinaryOps were on the order of tens of MB before the
        # fix; under the fix we expect single-digit MB.
        assert (
            after - before
        ) < 25.0, f"bound-IBO leak: RSS grew {after - before:.1f} MB over 50k binds"
    finally:
        delattr(indexbinary, "leak_test_bound_ibo_op")


def test_bound_ibo_jit_introspection_returns_none():
    """Bound IBOs are user-defined and don't go through the built-in JIT C
    codegen path, so the introspection properties must report ``None``
    without crashing. Regression for an earlier construction path that
    used ``__new__`` directly and skipped the inherited ``_jit_c_info``
    slot's default-None initialization. Also checks the unpickled instance,
    which goes through a different construction path (``_rebind_indexbinaryop``
    -> ``TypedBuiltinIndexBinaryOp.__call__``).
    """
    indexbinary.register_new("_jit_introspect_test_op", _ibo_add_theta)
    try:
        op = indexbinary._jit_introspect_test_op
        bound = op[int](3)
        # User IBOs don't go through the built-in JIT C codegen path, so these
        # are expected to be ``None``; the introspection properties must not
        # crash either way.
        assert bound.jit_c_name is None
        assert bound.jit_c_source is None
        bound2 = pickle.loads(pickle.dumps(bound))
        assert bound2.jit_c_name is None
        assert bound2.jit_c_source is None
    finally:
        delattr(indexbinary, "_jit_introspect_test_op")


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


def test_udt_tuple_return():
    """An IndexBinaryOp UDF can return a tuple matching the output UDT's fields."""
    record_dtype = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype)

    v = Vector(udt, size=2)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    w = Vector(udt, size=2)
    w[0] = (10, 20.0)
    w[1] = (30, 40.0)

    theta = Scalar.from_value(np.array((0, 0.0), dtype=record_dtype))

    def _add_with_idx(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return (x["a"] + y["a"] + theta["a"], x["b"] + y["b"] + ix)

    op = indexbinary.register_anonymous(_add_with_idx, is_udt=True)
    binop = op[udt](theta)
    result = v.ewise_mult(w, binop).new()
    assert result.dtype == udt
    expected = Vector(udt, size=2)
    expected[0] = (11, 22.0)  # a: 1+10+0, b: 2+20+0
    expected[1] = (33, 45.0)  # a: 3+30+0, b: 4+40+1
    assert result.isequal(expected)

    # UDT input, scalar output (no tuple unpacking needed)
    def _sum_ab(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x["a"] + y["b"] + theta["a"]

    theta2 = Scalar.from_value(np.array((100, 0.0), dtype=record_dtype))
    op2 = indexbinary.register_anonymous(_sum_ab, is_udt=True)
    binop2 = op2[udt](theta2)
    result2 = v.ewise_mult(w, binop2).new()
    assert result2[0].new() == 121.0  # 1 + 20.0 + 100
    assert result2[1].new() == 143.0  # 3 + 40.0 + 100


def test_udt_matrix_ewise_with_bound_ibo():
    """End-to-end: register an IBO on a record UDT, bind a theta, and use it
    as the binary op of ``Matrix.ewise_mult`` and ``Matrix.ewise_add``.

    The matrix path is more interesting than the vector path because the
    IBO receives both row and column indices for each operand. This also
    exercises the full ``is_udt`` codegen with two-field-tuple returns on
    Matrix-shaped inputs.
    """
    record_dtype = np.dtype([("a", np.int64), ("b", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "ibo_mxm_udt")

    def weighted_add(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        # Field a: x.a * theta.a + y.a; Field b: x.b + y.b + ix + jy
        return (x["a"] * theta["a"] + y["a"], x["b"] + y["b"] + ix + jy)

    ibo = indexbinary.register_anonymous(weighted_add, is_udt=True)
    theta = Scalar.from_value(np.array((10, 0), dtype=record_dtype))
    binop = ibo[udt](theta)
    assert binop.opclass == "BinaryOp"

    A = Matrix(udt, nrows=3, ncols=3)
    A[0, 0] = (1, 2)
    A[1, 1] = (3, 4)
    A[2, 0] = (5, 6)
    B = Matrix(udt, nrows=3, ncols=3)
    B[0, 0] = (1, 1)
    B[1, 1] = (2, 2)
    B[2, 0] = (3, 3)

    C = A.ewise_mult(B, binop).new()
    assert C.dtype == udt
    expected = Matrix(udt, nrows=3, ncols=3)
    # (0,0): a = 1*10 + 1 = 11, b = 2 + 1 + 0 + 0 = 3
    # (1,1): a = 3*10 + 2 = 32, b = 4 + 2 + 1 + 1 = 8
    # (2,0): a = 5*10 + 3 = 53, b = 6 + 3 + 2 + 0 = 11
    expected[0, 0] = (11, 3)
    expected[1, 1] = (32, 8)
    expected[2, 0] = (53, 11)
    assert C.isequal(expected)

    # ewise_add path with the same bound IBO. ewise_add applies the op only where
    # both sides are present; pattern is the same as ewise_mult here.
    D = A.ewise_add(B, binop).new()
    assert D.isequal(expected)


def _semiring_with_bound_ibo_factory(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
    # Module-level so pickle can find it; used by both the semiring and the
    # cross-pickle test below.
    return x * y + theta * (ix + jy)


def test_semiring_from_bound_ibo_mxm():
    """A bound IndexBinaryOp may serve as the multiplier of a Semiring.

    Per SS:GraphBLAS, ``GxB_BinaryOp_new_IndexOp`` turns an IBO + theta into
    a regular ``GrB_BinaryOp``, which any Semiring can use as its multiplier.
    Verify the end-to-end path: build a Semiring with ``monoid.plus`` plus a
    bound IBO, and use it as ``A.mxm(B, sr)``.
    """
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    ibo = indexbinary.register_anonymous(_semiring_with_bound_ibo_factory)
    bound = ibo[dtypes.INT64](theta=2)
    sr = semiring_module.register_anonymous(monoid_module.plus, bound)

    # Diagonal-ish A and B so we can compute the result by hand.
    A = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4], dtype=dtypes.INT64)
    B = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [5, 6, 7, 8], dtype=dtypes.INT64)
    C = A.mxm(B, sr).new()

    # mxm(A, B)[i, j] = sum_k f(A[i,k], i, k, B[k,j], k, j, theta=2)
    #                = sum_k A[i,k]*B[k,j] + 2*(i + j)
    # For [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    # plus 2*(i + j) per inner-product term (two terms per cell):
    #   C[0, 0] = 5 + 14 + 2*(0+0)*2 = 19      (theta term: 0 per k)
    #   C[0, 1] = 6 + 16 + 2*(0+1)*2 = 22 + 4  = 26
    #   C[1, 0] = 15 + 28 + 2*(1+0)*2 = 43 + 4 = 47
    #   C[1, 1] = 18 + 32 + 2*(1+1)*2 = 50 + 8 = 58
    assert C.dtype == dtypes.INT64
    expected = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [19, 26, 47, 58], dtype=dtypes.INT64)
    assert C.isequal(expected)


def test_semiring_from_bound_ibo_pickle():
    """A Semiring built from a bound IBO survives pickle round-trip.

    The bound IBO already pickles (via its module-level parent op); the
    Semiring inherits the same path.
    """
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    indexbinary.register_new("semiring_pickle_ibo", _semiring_with_bound_ibo_factory)
    bound = indexbinary.semiring_pickle_ibo[dtypes.INT64](theta=2)
    sr = semiring_module.register_anonymous(monoid_module.plus, bound)
    sr2 = pickle.loads(pickle.dumps(sr))

    A = Matrix.from_coo([0, 1], [0, 1], [1, 3], dtype=dtypes.INT64)
    B = Matrix.from_coo([0, 1], [0, 1], [5, 7], dtype=dtypes.INT64)
    assert A.mxm(B, sr).new().isequal(A.mxm(B, sr2).new())


def test_semiring_from_bound_ibo_type_mismatch_errors():
    """Monoid must accept the bound IBO's return type."""
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    ibo = indexbinary.register_anonymous(_semiring_with_bound_ibo_factory)
    bound = ibo[dtypes.INT64](theta=0)
    with pytest.raises(TypeError, match="band.*INT64"):
        # ``band`` is bitwise-AND on unsigned ints; INT64 is not in its types.
        semiring_module.register_anonymous(monoid_module.band, bound)


def test_semiring_register_anonymous_rejects_bad_binaryop():
    """The error message points users at the bound-IBO syntax."""
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    with pytest.raises(TypeError, match=r"bound IndexBinaryOp.*ibo\[dtype\]\(theta\)"):
        semiring_module.register_anonymous(monoid_module.plus, 42)


def test_semiring_from_bound_ibo_over_udt_mxm():
    """A bound IBO over a UDT also composes into a Semiring via the
    auto-lifted ``monoid.plus[udt]`` path.
    """
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    record_dtype = np.dtype([("a", np.int64), ("b", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "ibo_mxm_sr_udt")

    def weighted(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return (x["a"] * y["a"] + theta["a"], x["b"] + y["b"])

    ibo = indexbinary.register_anonymous(weighted, is_udt=True)
    theta = Scalar.from_value(np.array((1, 0), dtype=record_dtype))
    bound = ibo[udt](theta)

    # ``monoid.plus[udt]`` is auto-lifted (field-wise add); the semiring
    # composes the bound IBO multiplier with that monoid.
    sr = semiring_module.register_anonymous(monoid_module.plus, bound)

    A = Matrix(udt, nrows=2, ncols=2)
    A[0, 0] = (2, 3)
    A[0, 1] = (1, 1)
    A[1, 1] = (4, 5)
    B = Matrix(udt, nrows=2, ncols=2)
    B[0, 0] = (5, 1)
    B[1, 1] = (6, 2)

    C = A.mxm(B, sr).new()
    # mxm(A, B)[i, j] = reduce_plus_k(weighted(A[i, k], i, k, B[k, j], k, j, theta=(1, 0)))
    # weighted has fields a = A.a * B.a + theta.a (=1), b = A.b + B.b
    # C[0, 0]: only k=0 contributes -> (2*5 + 1, 3 + 1) = (11, 4)
    # C[0, 1]: only k=1 contributes -> (1*6 + 1, 1 + 2) = (7, 3)
    # C[1, 1]: only k=1 contributes -> (4*6 + 1, 5 + 2) = (25, 7)
    assert C.dtype == udt
    assert C[0, 0].new() == (11, 4)
    assert C[0, 1].new() == (7, 3)
    assert C[1, 1].new() == (25, 7)


def test_dir_and_module():
    assert "register_new" in dir(indexbinary)
    assert "register_anonymous" in dir(indexbinary)
    assert "ss" in dir(indexbinary)
    # Actually access the ss module to verify it exists (not just in __dir__)
    ss = indexbinary.ss
    assert hasattr(ss, "_delayed")
    assert hasattr(ss, "register_new")

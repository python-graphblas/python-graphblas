import pickle

import numpy as np
import pytest

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, dtypes, indexbinary
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.operator.indexbinary import _has_idxbinop
from graphblas.exceptions import EmptyObject, UdfParseError

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


def test_untyped_call_is_typed_when_it_runs():
    """``op(theta)`` takes its type from both inputs and theta each time it runs.

    Theta is typed as an IndexUnaryOp's thunk is: a Python int is INT64 and a
    float FP64, a numpy scalar or a Scalar keeps its type, and the default,
    False, changes no type. Before, the op had theta's type alone: ``op()``
    was BOOL, so ints became True; ``op(2)`` truncated floats to INT64; and
    ``op(np.int8(2))`` computed in INT8, where 300.0 became 127 and wrapped.
    """

    def upper_from_x(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x if ix <= jx else y

    pick = indexbinary.register_anonymous(upper_from_x)
    A = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [10, 20, 30, 40])
    B = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [-1, -2, -3, -4])
    expected = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], [10, 20, -3, 40])
    for binop in [pick(), pick(0), pick(False)]:
        assert A.ewise_mult(B, binop).new().isequal(expected, check_dtype=True)
    C = A.ewise_mult(B, pick(0.0)).new()
    assert C.isequal(expected.dup("FP64"), check_dtype=True)

    def weighted(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + theta * y

    wsum = indexbinary.register_anonymous(weighted)
    u = Vector.from_coo([0, 1], [1.5, 300.0])
    w = Vector.from_coo([0, 1], [2.25, 1.0])
    expected = Vector.from_coo([0, 1], [6.0, 302.0])
    for theta in [2, 2.0, np.int8(2), Scalar.from_value(2, "INT8")]:
        binop = wsum(theta)
        assert u.ewise_mult(w, binop).new().isequal(expected, check_dtype=True)
        assert binop(u & w).new().isequal(expected, check_dtype=True)
    # One op runs at each type it meets
    i8 = Vector.from_coo([0, 1], [1, 100], dtype="INT8")
    binop = wsum(1)
    assert u.ewise_mult(w, binop).new().isequal(Vector.from_coo([0, 1], [3.75, 301.0]))
    result = i8.ewise_mult(i8, binop).new()
    assert result.isequal(Vector.from_coo([0, 1], [2, 200], dtype="INT64"), check_dtype=True)
    for binop in [wsum(), wsum(np.int8(1))]:
        assert i8.ewise_mult(i8, binop).new().dtype == dtypes.INT8
    # A named type is the op's type, as before
    for binop in [
        wsum["INT64"](2),
        wsum(2, dtype="INT64"),
        wsum(Scalar.from_value(2.0), dtype="INT64"),
        wsum(2)["INT64"],
    ]:
        assert binop.type == dtypes.INT64
        result = u.ewise_mult(w, binop).new()
        assert result.isequal(Vector.from_coo([0, 1], [5, 302]), check_dtype=True)
    assert repr(wsum(2)) == "indexbinary.weighted(2)"
    rec_udt = dtypes.register_anonymous(np.dtype([("theta_a", np.int64)]), "_UntypedThetaRec")
    with pytest.raises(KeyError, match="weighted does not work with _UntypedThetaRec"):
        wsum(Scalar.from_value((1,), rec_udt))
    # Binding when it runs uses theta's value from when it was given
    theta = Scalar.from_value(2.0)
    binop = wsum(theta)
    theta.value = 100.0
    assert u.ewise_mult(w, binop).new().isequal(expected)


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


def test_array_udt_udf_shape_is_checked():
    """An IBO whose UDF builds a wrong-shape array is rejected when typed.

    The same guard the binary and unary paths get. Numba's ``Array`` type
    carries ``ndim`` but not extents, so the mismatch is not a type error; left
    to the wrapper it raises inside the cfunc, where Numba prints the traceback
    and returns, leaving the element as SuiteSparse found it.
    """
    # A shape of its own: ``register_anonymous`` caches by numpy dtype, so
    # sharing one with another test would hand back that test's DataType.
    arr_udt = dtypes.register_anonymous(np.dtype((np.float64, (12,))), "_IboShapeArr12")

    def _truncate(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return (x + y)[:2]

    op = indexbinary.register_anonymous(_truncate, is_udt=True)
    with pytest.raises(UdfParseError, match=r"shape \(2,\) when run on sample values"):
        op[arr_udt]

    def _combine(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    good = indexbinary.register_anonymous(_combine, is_udt=True)
    assert good[arr_udt].return_type is arr_udt


def test_empty_scalar_theta():
    # SuiteSparse would read the empty Scalar's value: a crash, or a value it never held
    op = indexbinary.register_anonymous(_ibo_add_theta)
    opu = indexbinary.register_anonymous(_ibo_add_theta, is_udt=True)
    cleared = Scalar.from_value(5)
    cleared.clear()
    for theta in [Scalar(int), Scalar(int, is_cscalar=True), cleared]:
        for bind in [op, opu, op[int], opu[int], lambda t: op(t, dtype=int)]:
            with pytest.raises(EmptyObject, match="theta is an empty Scalar"):
                bind(theta)


def test_bind_raw_udt_theta_without_dtype():
    """A numpy array theta is typed as its array layout, as other array literals are.

    With ``is_udt=True``, theta keeps that type, as any theta does, so it is
    never converted to y's type. A tuple has no type of its own: give it as a
    Scalar of its UDT, or name the op's type. An op without ``is_udt`` does
    not take a UDT theta.
    """

    def theta_only(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return theta

    arr_udt = dtypes.register_anonymous(np.dtype((np.int32, (3,))), "_RawThetaArr3i32")
    indexbinary.register_new("raw_no_dtype_op", theta_only, is_udt=True)
    try:
        op = indexbinary.raw_no_dtype_op
        v = Vector.from_coo([0], np.array([[1, 2, 3]], dtype=np.int32), dtype=arr_udt)
        for theta in [np.array([10, 20, 30], np.int32), Scalar.from_value((10, 20, 30), arr_udt)]:
            result = v.ewise_mult(v, op(theta)).new()
            assert result.dtype == arr_udt
            assert result[0].new().value.tolist() == [10, 20, 30]
        # An int64 or float64 array keeps its type, where it would have become int32
        result = v.ewise_mult(v, op(np.array([10, 20, 2**40]))).new()
        assert result.dtype.np_type == np.dtype((np.int64, (3,)))
        assert result[0].new().value.tolist() == [10, 20, 2**40]
        result = v.ewise_mult(v, op(np.array([10, 20.5, 30]))).new()
        assert result.dtype.np_type == np.dtype((np.float64, (3,)))
        assert result[0].new().value.tolist() == [10, 20.5, 30]
        with pytest.raises(TypeError, match=r"Cannot infer a dtype for theta from \(10, 20, 30\)"):
            op((10, 20, 30))
        with pytest.raises(TypeError, match="Cannot infer a dtype for theta"):
            op(np.array([True, False]))
        # Its array UDT is shared, under the name it got first (another test's)
        with pytest.raises(KeyError, match="_ibo_add_theta does not work with "):
            indexbinary.register_anonymous(_ibo_add_theta)(np.array([1, 2, 3]))
    finally:
        delattr(indexbinary, "raw_no_dtype_op")


def test_untyped_bind_theta_keeps_its_type_with_is_udt():
    """With ``is_udt=True``, ``op(theta)`` compiles for x, y and theta as they are typed.

    SuiteSparse gives theta a type of its own (``GxB_IndexBinaryOp_new`` takes
    ``theta_type``), so theta is not converted: 0.5 stays 0.5 beside INT64
    inputs, and Numba types the result as it would for those three types.
    Before, theta took y's type when the op ran. Naming two types,
    ``op[X, Y](theta)``, still gives theta y's type and casts it, and
    ``op[X, Y, T](theta)`` names all three.
    """
    op = indexbinary.register_anonymous(_ibo_add_theta, is_udt=True)
    v = Vector.from_coo([0, 1], [1, 2])
    expected = Vector.from_coo([0, 1], [4, 6])
    for theta in [2, np.int8(2), Scalar.from_value(2, "INT16")]:
        assert v.ewise_mult(v, op(theta)).new().isequal(expected, check_dtype=True)
    for theta in [2.0, np.float64(2), Scalar.from_value(2.0)]:
        result = v.ewise_mult(v, op(theta)).new()
        assert result.isequal(expected.dup(float), check_dtype=True)
    result = v.ewise_mult(v, op(0.5)).new()
    assert result.isequal(Vector.from_coo([0, 1], [2.5, 4.5]), check_dtype=True)
    i8 = Vector.from_coo([0], [1], dtype="INT8")
    result = i8.ewise_mult(i8, op(300)).new()
    assert result.isequal(Vector.from_coo([0], [302]), check_dtype=True)
    assert {key for key in op._udt_types if key[0] == dtypes.INT64} == {
        (dtypes.INT64, dtypes.INT64, dtypes.INT64),
        (dtypes.INT64, dtypes.INT64, dtypes.INT8),
        (dtypes.INT64, dtypes.INT64, dtypes.INT16),
        (dtypes.INT64, dtypes.INT64, dtypes.FP64),
    }
    # Named types
    result = v.ewise_mult(v, op["INT64", "INT64"](0.5)).new()
    assert result.isequal(Vector.from_coo([0, 1], [2, 4]), check_dtype=True)
    typed = op["INT64", "INT64", "FP64"]
    assert typed is op._udt_ops[dtypes.INT64, dtypes.INT64, dtypes.FP64]
    assert typed.thunk_type == dtypes.FP64
    assert repr(typed) == "indexbinary._ibo_add_theta[INT64, INT64, FP64]"
    assert repr(op["INT64", "INT64"]) == "indexbinary._ibo_add_theta[INT64, INT64]"
    result = v.ewise_mult(v, typed(0.5)).new()
    assert result.isequal(Vector.from_coo([0, 1], [2.5, 4.5]), check_dtype=True)
    # Pickle keeps all three types, for the typed op and for the op it binds
    typed2 = pickle.loads(pickle.dumps(typed))  # re-registers the anonymous op
    assert repr(typed2) == repr(typed)
    assert v.ewise_mult(v, typed2(0.5)).new().isequal(result, check_dtype=True)
    bound = v.ewise_mult(v, op(0.5)).op
    assert bound.type == bound._type2 == dtypes.INT64
    bound2 = pickle.loads(pickle.dumps(bound))
    assert bound2._theta_type == dtypes.FP64
    assert v.ewise_mult(v, bound2).new().isequal(result, check_dtype=True)
    bound2 = pickle.loads(pickle.dumps(op["INT64", "FP64"](1)))  # was op["INT64"](1)
    assert (bound2.type, bound2._type2, bound2._theta_type) == (
        dtypes.INT64,
        dtypes.FP64,
        dtypes.FP64,
    )
    # Without is_udt, the three types are one, as when the op runs
    wsum = indexbinary.register_anonymous(_ibo_add_theta)
    assert wsum["INT8", "INT8", "INT64"] is wsum["INT64"]
    rec_udt = dtypes.register_anonymous(np.dtype([("theta_b", np.int64)]), "_UntypedThetaRec2")
    with pytest.raises(KeyError, match="does not work with .INT64, INT64, _UntypedThetaRec2"):
        wsum["INT64", "INT64", rec_udt]


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
    import sys

    pytest.importorskip("resource")
    import resource

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


def test_type_the_op_does_not_take():
    """``R in op`` is False and ``op[R]`` raises KeyError, as for an IndexUnaryOp.

    Both raised AttributeError: IndexBinaryOp had no ``is_positional``, which
    ``op[R]`` reads for a type the op does not have.
    """

    def add_vals(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_vals)
    rec = dtypes.register_anonymous(
        np.dtype([("ibt_a", np.float64), ("ibt_b", np.int64)]), "_IbtRec"
    )
    assert dtypes.INT64 in op
    assert rec not in op
    for expr in [lambda: op[rec], lambda: op(Scalar.from_value((1.0, 2), rec), dtype=rec)]:
        with pytest.raises(KeyError, match="does not work with _IbtRec"):
            expr()


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


def test_reduce_refuses_a_bound_op():
    """A bound IndexBinaryOp gets the TypeError reduce gives a BinaryOp that is in no Monoid.

    It raised AttributeError: neither the bound op nor its IndexBinaryOp had
    ``monoid``, which reduce and its error message read.
    """

    def add_vals(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return x + y + theta

    op = indexbinary.register_anonymous(add_vals)
    v = Vector.from_coo([0, 1], [1, 2])
    for binop in [op(1), op[int](1)]:
        with pytest.raises(TypeError, match="is not known to be part of a Monoid"):
            v.reduce(binop)


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
    assert find_opclass(op(0))[1] == "BinaryOp"


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


def _ibo_decay(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
    d = ix - jx if ix >= jx else jx - ix
    return x * y * theta**d


def test_semiring_from_untyped_bound_ibo():
    """A Semiring from ``ibo(theta)`` is typed when it runs, by the inputs and theta.

    Before, ``ibo(0.5)`` was an FP64 op, so its semiring refused INT64
    matrices with KeyError, and ``ibo(1)`` was INT64 and refused FP64 ones.
    """
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    decay = indexbinary.register_anonymous(_ibo_decay)
    A = Matrix.from_coo([0, 0, 1, 2], [0, 2, 1, 0], [1, 1, 1, 1], nrows=3, ncols=3)
    v = Vector.from_coo([0, 1, 2], [4, 4, 4])
    half = semiring_module.register_anonymous(monoid_module.plus, decay(0.5))
    one = semiring_module.register_anonymous(monoid_module.plus, decay(1))
    expected = Vector.from_coo([0, 1, 2], [5.0, 4.0, 1.0])
    assert A.mxv(v, half).new().isequal(expected, check_dtype=True)
    assert A.dup("FP32").mxv(v.dup("FP32"), half).new().isequal(expected, check_dtype=True)
    assert A.mxv(v, one).new().isequal(Vector.from_coo([0, 1, 2], [8, 4, 4]), check_dtype=True)
    result = A.dup(float).mxv(v.dup(float), one).new()
    assert result.isequal(Vector.from_coo([0, 1, 2], [8.0, 4.0, 4.0]), check_dtype=True)
    sr2 = pickle.loads(pickle.dumps(half))
    assert A.mxv(v, sr2).new().isequal(expected, check_dtype=True)
    # Theta keeps its type through pickle: FP32 data stays FP32 with an FP32 theta
    bound2 = pickle.loads(pickle.dumps(decay(np.float32(0.5))))
    sr2 = semiring_module.register_anonymous(monoid_module.plus, bound2)
    result = A.dup("FP32").mxv(v.dup("FP32"), sr2).new()
    assert result.isequal(expected.dup("FP32"), check_dtype=True)


def test_argmax_semiring_from_untyped_bound_ibo():
    """The SuiteSparse user guide's row argmax, with theta unused, bound with ``iop()``.

    The multiplier turns ``A(i, k)`` into the pair ``(k, A(i, k))`` and a max
    monoid on pairs keeps the larger value (the smaller column on a tie). The
    guide binds a BOOL theta because theta cannot be NULL. Here the matrix
    holds pairs, so the UDF's tuple return takes their type. Before, ``iop()``
    and ``iop(0)`` were ops on BOOL and INT64 values, not the matrix's pairs.
    Theta keeps its own type, so each bind compiles the op for it.
    """
    from graphblas import binary
    from graphblas import monoid as monoid_module
    from graphblas import semiring as semiring_module

    pair = dtypes.register_anonymous(np.dtype([("k", np.int64), ("v", np.float64)]), "IboPair")

    def make_pair(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
        return (np.int64(jx), x["v"])

    def max_pair(a, b):  # pragma: no cover (numba)
        if a[1] > b[1] or (a[1] == b[1] and a[0] < b[0]):
            return a
        return b

    iop = indexbinary.register_anonymous(make_pair, is_udt=True)
    max_monoid = monoid_module.register_anonymous(
        binary.register_anonymous(max_pair, is_udt=True), (np.iinfo(np.int64).max, -np.inf)
    )
    values = np.array([(0, 3.0), (0, 7.0), (0, 7.0), (0, 5.0), (0, -1.0)], dtype=pair.np_type)
    A = Matrix.from_coo([0, 0, 0, 1, 1], [0, 1, 2, 0, 2], values, dtype=pair)
    ones = Vector.from_scalar(True, A.ncols)
    for binop in [iop(), iop(False), iop(0), iop(0.0)]:
        argmax = semiring_module.register_anonymous(max_monoid, binop)
        c = A.mxv(ones, argmax).new()
        assert c.dtype == pair
        assert c[0].new() == (1, 7.0)
        assert c[1].new() == (0, 5.0)
    assert set(iop._udt_types) == {
        (pair, dtypes.BOOL, dtypes.BOOL),
        (pair, dtypes.BOOL, dtypes.INT64),
        (pair, dtypes.BOOL, dtypes.FP64),
    }


def test_dir_and_module():
    assert "register_new" in dir(indexbinary)
    assert "register_anonymous" in dir(indexbinary)
    assert "ss" in dir(indexbinary)
    # Actually access the ss module to verify it exists (not just in __dir__)
    ss = indexbinary.ss
    assert hasattr(ss, "_delayed")
    assert hasattr(ss, "register_new")

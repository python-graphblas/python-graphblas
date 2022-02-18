import numpy as np
import pytest

from grblas import binary, dtypes, replace, unary
from grblas.expr import Updater

from grblas import Matrix, Scalar, Vector  # isort:skip


def test_from_values_dtype_resolving():
    u = Vector.from_values([0, 1, 2], [1, 2, 3], dtype=dtypes.INT32)
    assert u.dtype == "INT32"
    u = Vector.from_values([0, 1, 2], [1, 2, 3], dtype="INT32")
    assert u.dtype == dtypes.INT32
    M = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=dtypes.UINT8)
    assert M.dtype == "UINT8"
    M = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=float)
    assert M.dtype == dtypes.FP64


def test_from_values_invalid_dtype():
    # We now rely on numpy to coerce the data
    A = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=dtypes.BOOL)
    expected = Matrix.from_values([0, 1, 2], [2, 0, 1], [False, True, True], dtype=dtypes.BOOL)
    assert A.isequal(expected)
    with pytest.raises(ValueError, match="object dtype for values is not allowed"):
        Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, object()])


def test_resolve_ops_using_common_dtype():
    # C << A.ewise_mult(B, binary.plus) <-- PLUS should use FP64 because unify(INT64, FP64) -> FP64
    u = Vector.from_values([0, 1, 3], [1, 2, 3], dtype=dtypes.INT64)
    v = Vector.from_values([0, 1, 3], [0.1, 0.1, 0.1], dtype="FP64")
    w = Vector.new("FP32", u.size)
    w << u.ewise_mult(v, binary.plus)
    result = Vector.from_values([0, 1, 3], [1.1, 2.1, 3.1], dtype="FP32")
    assert w.isclose(result, check_dtype=True)


def test_order_of_updater_params_does_not_matter():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    mask = Vector.from_values([0, 3], [True, True])
    accum = binary.plus
    result = Vector.from_values([0, 3], [5, 10])
    # mask, accum, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask.V, accum, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # accum, mask, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask.V, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # accum, mask=, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask=mask.V, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # mask, accum=, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask.V, accum=accum, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # replace=, mask=, accum=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(replace=True, mask=mask.V, accum=accum) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # replace, mask=, accum=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(replace, mask=mask.V, accum=accum) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)


def test_updater_replace_no_mask():
    u = Vector.from_values([0, 1, 2], [1, 2, 3])
    with pytest.raises(
        TypeError, match="'replace' argument may only be True if a mask is provided"
    ):
        u(replace=True)
    with pytest.raises(
        TypeError, match="'replace' argument may only be True if a mask is provided"
    ):
        u(replace)


def test_replace_repr():
    assert repr(replace) == "replace"
    assert str(replace) == "replace"


def test_updater_repeat_argument_types():
    mask = Vector.from_values([0, 3], [True, True])
    accum = binary.plus
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    with pytest.raises(TypeError, match="multiple"):
        v(mask.S, mask.S)
    with pytest.raises(TypeError, match="multiple"):
        v(mask.S, mask=mask.S)
    with pytest.raises(TypeError, match="multiple"):
        v(accum, accum)
    with pytest.raises(TypeError, match="multiple"):
        v(accum, accum=accum)


def test_updater_bad_types():
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    M = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=dtypes.UINT8)
    with pytest.raises(TypeError, match="Invalid mask"):
        v(mask=object())
    with pytest.raises(TypeError, match="Invalid mask"):
        v[[1, 2]].new(mask=object())
    with pytest.raises(TypeError, match="Mask object must be type Vector"):
        v.ewise_mult(v).new(mask=M.S)
    with pytest.raises(TypeError, match="Invalid"):
        v(object())
    with pytest.raises(TypeError, match="Expected type: BinaryOp"):
        v(unary.one)


def test_already_resolved_ops_allowed_in_updater():
    # C(binary.plus['FP64']) << ...
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    u(binary.plus["INT64"]) << u.ewise_mult(u, binary.times["INT64"])
    result = Vector.from_values([0, 1, 3], [2, 6, 12])
    assert u.isequal(result)


def test_updater_returns_updater():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    y = u(accum=binary.times)
    assert isinstance(y, Updater)
    z = y << u.apply(unary.ainv)
    assert z is None
    assert isinstance(y, Updater)
    final_result = Vector.from_values([0, 1, 3], [-1, -4, -9])
    assert u.isequal(final_result)


def test_updater_only_once():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    with pytest.raises(TypeError, match="'Assigner' object is not callable"):
        u()[0]()
    with pytest.raises(TypeError, match="'Assigner' object is not callable"):
        u(mask=u.S)[0]()
    with pytest.raises(TypeError, match="'Assigner' object is not callable"):
        u(accum=binary.plus)[0]()
    with pytest.raises(TypeError, match="not callable"):
        u()()
    with pytest.raises(TypeError, match="'Assigner' object is not callable"):
        u[[0, 1]]()()
    # While we're at it...
    with pytest.raises(TypeError, match="is not subscriptable"):
        u[[0, 1]]()[0]
    with pytest.raises(TypeError, match="is not subscriptable"):
        u()[[0, 1]][0]
    with pytest.raises(TypeError, match="autocompute"):
        u[[0, 1]][0]


def test_bad_extract_with_updater():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    assert u[0].new() == 1
    with pytest.raises(AttributeError, match="'Assigner' object has no attribute 'value'"):
        u(mask=u.S)[0].value
    with pytest.raises(AttributeError, match="has no attribute"):
        u[[0, 1]].value
    with pytest.raises(AttributeError, match="'Assigner' object has no attribute 'new'"):
        u(mask=u.S)[0].new()
    with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
        u << u(mask=u.S)[[1, 2]]
    with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
        u << u()[[1, 2]]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        u[0].new(mask=u.S)
    s = Scalar.from_value(10)
    with pytest.raises(TypeError, match="Indexing not supported for Scalars"):
        del s()[0]
    with pytest.raises(TypeError, match="Indexing not supported for Scalars"):
        s()[0] = 1
    with pytest.raises(TypeError, match="Indexing not supported for Scalars"):
        s()[0]


def test_updater_on_rhs():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
        u << u(mask=u.S)
    with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
        u << u()
    with pytest.raises(TypeError, match="Bad type for argument `value`"):
        u[:] << u()


def test_py_indices():
    v = Vector.from_values(np.arange(5), np.arange(5))

    idx = v[:].resolved_indexes.py_indices
    assert idx == slice(None)
    w = v[idx].new()
    assert w.isequal(v)

    idx = v[3].resolved_indexes.py_indices
    assert idx == 3

    idx = v[[0, 2]].resolved_indexes.py_indices
    np.testing.assert_array_equal(idx, [0, 2])

    idx = v[0:0].resolved_indexes.py_indices
    assert idx == slice(1, 1)  # strange, but expected
    w = v[idx].new()
    assert w.size == 0

    idx = v[2:0].resolved_indexes.py_indices
    assert idx == slice(2, 1)  # strange, but expected
    w = v[idx].new()
    assert w.size == 0

    idx = v[::-1].resolved_indexes.py_indices
    assert idx == slice(None, None, -1)
    w = v[idx].new()
    expected = Vector.from_values(np.arange(5), np.arange(5)[::-1])
    assert w.isequal(expected)

    idx = v[4:-3:-1].resolved_indexes.py_indices
    assert idx == slice(None, -v.size - 1 + 3, -1)
    w = v[idx].new()
    expected = Vector.from_values(np.arange(2), np.arange(5)[4:-3:-1])
    assert w.isequal(expected)

    idx = v[1::2].resolved_indexes.py_indices
    assert idx == slice(1, None, 2)
    w = v[idx].new()
    expected = Vector.from_values(np.arange(2), np.arange(5)[1::2])
    assert w.isequal(expected)

    A = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], nrows=10, ncols=10)
    idx = A[1:6, 8:-8:-2].resolved_indexes.py_indices
    assert idx == (slice(1, 6), slice(8, -8, -2))

    # start=0 -> None
    idx = v[0:2].resolved_indexes.py_indices
    assert idx == slice(None, 2)
    # stop=size -> None
    idx = v[1 : v.size].resolved_indexes.py_indices
    assert idx == slice(1, None)
    # step=1 -> None
    idx = v[1:3:1].resolved_indexes.py_indices
    assert idx == slice(1, 3)
    # All together now!
    idx = v[0 : v.size : 1].resolved_indexes.py_indices
    assert idx == slice(None)

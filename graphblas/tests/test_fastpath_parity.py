"""Parity tests for the scalar-access fast paths.

Several hot scalar-access operations grew fast paths that bypass the general
expression machinery: ``Vector.get`` / ``Matrix.get``, ``index in obj``
(``__contains__``), integer-key ``obj[i] = value`` (``__setitem__``),
``obj[i].new()`` on a scalar index expression, and the plain-int lane in
``parse_index``. Each fast path is only correct if it produces exactly what the
slower general path would. These tests pin that equivalence: for every case we
run the shipped fast path and an inlined copy of the pre-fast-path reference,
then require the results (values, dtypes, and raised exceptions with their
messages) to match.

The references here are deliberately self-contained rather than imported from
the throwaway benchmark scripts they were distilled from, so a refactor of the
production code can never silently drag the reference along with it.
"""

import numpy as np
import pytest

import graphblas as gb
from graphblas import Matrix, Vector, binary, dtypes
from graphblas.core.expr import Updater
from graphblas.core.recorder import Recorder
from graphblas.core.scalar import Scalar

# Builtin dtypes to sweep. Complex is SuiteSparse-only, so gate it the same way
# the dtype tests do; on the vanilla backend the parametrization simply omits it.
BUILTIN_DTYPES = [
    dtypes.BOOL,
    dtypes.INT8,
    dtypes.INT16,
    dtypes.INT32,
    dtypes.INT64,
    dtypes.UINT8,
    dtypes.UINT16,
    dtypes.UINT32,
    dtypes.UINT64,
    dtypes.FP32,
    dtypes.FP64,
]
if dtypes._supports_complex:
    BUILTIN_DTYPES += [dtypes.FC32, dtypes.FC64]

DTYPE_IDS = [dt.name for dt in BUILTIN_DTYPES]


def _hit_value(dt):
    """A value that fits every builtin dtype (matches the source harnesses)."""
    if "FC" in dt.name:
        return 3 + 4j
    if dt == dtypes.BOOL:
        return True
    return 3


def _capture(fn):
    """Return ``("val", result)`` or ``("exc", type_name, message)``."""
    try:
        return ("val", fn())
    except Exception as e:
        return ("exc", type(e).__name__, str(e))


# ---------------------------------------------------------------------------
# Vector.get / Matrix.get
# ---------------------------------------------------------------------------
def _get_vector_slow(v, index, default=None):
    expr = v[index]
    if not expr._is_scalar:
        raise ValueError("Bad index in Vector.get(...)")
    rv = expr.new().value
    return default if rv is None else rv


def _get_matrix_slow(A, r, c, default=None):
    rv = A[r, c].new().value
    return default if rv is None else rv


def _assert_get_equal(got, expected):
    msg = f"got {got!r} ({type(got).__name__}), expected {expected!r} ({type(expected).__name__})"
    assert got == expected, msg
    assert type(got) is type(expected), msg


@pytest.mark.parametrize("dt", BUILTIN_DTYPES, ids=DTYPE_IDS)
def test_get_dtype_parity(dt):
    val = _hit_value(dt)
    v = Vector(dt, 10)
    v[2] = val
    _assert_get_equal(v.get(2), _get_vector_slow(v, 2))
    _assert_get_equal(v.get(3), _get_vector_slow(v, 3))
    _assert_get_equal(v.get(3, -1), _get_vector_slow(v, 3, -1))
    A = Matrix(dt, 5, 7)
    A[1, 2] = val
    _assert_get_equal(A.get(1, 2), _get_matrix_slow(A, 1, 2))
    _assert_get_equal(A.get(0, 0, 99), _get_matrix_slow(A, 0, 0, 99))


def test_get_index_types():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    _assert_get_equal(v.get(np.int64(5)), 2.5)
    _assert_get_equal(v.get(np.uint8(5)), 2.5)
    _assert_get_equal(v.get(-5), 2.5)
    _assert_get_equal(v.get(-2, "d"), "d")
    # empty-scalar semantics: a miss with no default is None
    assert v.get(1) is None


@pytest.mark.parametrize("idx", [10, -11, 1000])
def test_get_out_of_range(idx):
    """Out-of-range raises IndexError with the same message on both paths."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    with pytest.raises(IndexError) as fast:
        v.get(idx)
    with pytest.raises(IndexError) as slow:
        _get_vector_slow(v, idx)
    assert str(fast.value) == str(slow.value)


@pytest.mark.parametrize("bad", [1.5, "x", None, [1, 2], slice(None), 2.0, True])
def test_get_non_integer_index(bad):
    """Non-integer indices raise the same exception type as the reference path."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    fast = _capture(lambda: v.get(bad))
    slow = _capture(lambda: _get_vector_slow(v, bad))
    assert fast[:2] == slow[:2], f"fast={fast} slow={slow}"


def test_get_matrix_negative_and_range():
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10, 20, 30], nrows=5, ncols=7)
    _assert_get_equal(A.get(-1, -1, "d"), 30)
    _assert_get_equal(A.get(-1, 6), 30)
    _assert_get_equal(A.get(-2, -2, "d"), "d")
    with pytest.raises(IndexError):
        A.get(5, 0)
    with pytest.raises(IndexError):
        A.get(0, 7)


def test_get_transposed_matrix():
    """TransposedMatrix reuses Matrix.get and must extract the mirrored element."""
    # Non-square and non-symmetric, so a row/column swap cannot go unnoticed.
    A = Matrix.from_coo([0, 0, 1, 2], [1, 3, 0, 2], [10, 30, 20, 40], nrows=3, ncols=4)
    AT = A.T
    assert AT.shape == (4, 3)
    for r in range(4):
        for c in range(3):
            _assert_get_equal(AT.get(r, c, "d"), _get_matrix_slow(AT, r, c, "d"))
            # The mirrored element is the same one A sees with the axes swapped.
            _assert_get_equal(AT.get(r, c, "d"), A.get(c, r, "d"))
    # Out-of-range is judged against the transposed dimensions.
    with pytest.raises(IndexError):
        AT.get(4, 0)
    with pytest.raises(IndexError):
        AT.get(0, 3)
    # The setitem fast path is unreachable through a transposed view.
    assert not hasattr(type(AT), "__setitem__")


def test_get_udt_fallback():
    udt = gb.dtypes.register_anonymous(
        np.dtype([("fx", np.int64), ("fy", np.float64)]), "GetFastPathProbe"
    )
    u = Vector(udt, 3)
    u[1] = (7, 2.5)
    r = u.get(1)
    assert r["fx"] == 7, r
    assert r["fy"] == 2.5, r
    assert u.get(0, "dflt") == "dflt"


def test_get_recorder_fallback():
    """An active Recorder must see the extract call, i.e. the fast path defers."""
    v = Vector.from_coo([0], [1.5], size=4)
    with Recorder() as rec:
        assert v.get(0) == 1.5
    data = "".join(rec.data)
    assert "extractElement" in data, f"recorder missed call: {data!r}"


def test_get_pending_value():
    """A value set in non-blocking mode is visible before an explicit wait."""
    v = Vector(float, 100)
    v[3] = 2.25  # pending setElement in non-blocking mode
    _assert_get_equal(v.get(3), 2.25)


# ---------------------------------------------------------------------------
# __contains__ (index in obj)
# ---------------------------------------------------------------------------
def _contains_vector_slow(v, index):
    extractor = v[index]
    if not extractor._is_scalar:
        raise TypeError(
            f"Invalid index to Vector contains: {index!r}.  An integer is expected.  "
            "Doing `index in my_vector` checks whether a value is present at that index."
        )
    scalar = extractor.new(name="s_contains")
    return not scalar._is_empty


def _contains_matrix_slow(A, index):
    extractor = A[index]
    if not extractor._is_scalar:
        raise TypeError(
            f"Invalid index to Matrix contains: {index!r}.  A 2-tuple of ints is expected.  "
            "Doing `(i, j) in my_matrix` checks whether a value is present at that index."
        )
    scalar = extractor.new(name="s_contains")
    return not scalar._is_empty


@pytest.mark.parametrize("dt", BUILTIN_DTYPES, ids=DTYPE_IDS)
def test_contains_dtype_parity(dt):
    val = _hit_value(dt)
    v = Vector(dt, 10)
    v[2] = val
    assert (2 in v) == _contains_vector_slow(v, 2)
    assert (3 in v) == _contains_vector_slow(v, 3)
    # a zero / false value is still "present"
    v[4] = False if dt == dtypes.BOOL else 0
    assert (4 in v) == _contains_vector_slow(v, 4)
    A = Matrix(dt, 5, 7)
    A[1, 2] = val
    assert ((1, 2) in A) == _contains_matrix_slow(A, (1, 2))
    assert ((0, 0) in A) == _contains_matrix_slow(A, (0, 0))


def test_contains_index_types():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    assert (np.int64(5) in v) == _contains_vector_slow(v, np.int64(5))
    assert (np.uint8(5) in v) == _contains_vector_slow(v, np.uint8(5))
    assert (np.int64(6) in v) == _contains_vector_slow(v, np.int64(6))
    assert (-5 in v) == _contains_vector_slow(v, -5)
    assert (-2 in v) == _contains_vector_slow(v, -2)
    assert (-1 in v) == _contains_vector_slow(v, -1)


@pytest.mark.parametrize("idx", [10, -11, 1000, -1000])
def test_contains_out_of_range(idx):
    """Out-of-range raises the identical IndexError through either lane."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    fast = _capture(lambda: idx in v)
    slow = _capture(lambda: _contains_vector_slow(v, idx))
    assert fast == slow, f"fast={fast} slow={slow}"
    assert fast[:2] == ("exc", "IndexError"), fast


@pytest.mark.parametrize("b", [True, False])
def test_contains_bool_index(b):
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    fast = _capture(lambda: b in v)
    slow = _capture(lambda: _contains_vector_slow(v, b))
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("bad", [1.5, "x", None, [1, 2], slice(None), 2.0, (1, 2)])
def test_contains_non_integer_index(bad):
    """Non-integer indices raise the same exception type AND message."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    fast = _capture(lambda: bad in v)
    slow = _capture(lambda: _contains_vector_slow(v, bad))
    assert fast == slow, f"fast={fast} slow={slow}"


def test_contains_matrix_negative_and_range():
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10, 20, 30], nrows=5, ncols=7)
    assert ((-1, -1) in A) == _contains_matrix_slow(A, (-1, -1))
    assert ((-1, 6) in A) == _contains_matrix_slow(A, (-1, 6))
    assert ((-2, -2) in A) == _contains_matrix_slow(A, (-2, -2))
    for idx in [(5, 0), (0, 7), (-6, 0), (0, -8), (100, 100)]:
        fast = _capture(lambda idx=idx: idx in A)
        slow = _capture(lambda idx=idx: _contains_matrix_slow(A, idx))
        assert fast == slow, f"{idx}: fast={fast} slow={slow}"
        assert fast[:2] == ("exc", "IndexError"), (idx, fast)


def test_contains_transposed_matrix():
    """TransposedMatrix shares Matrix.__contains__; mirrored indices must agree."""
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10, 20, 30], nrows=5, ncols=7)
    AT = A.T
    for idx in [(0, 0), (6, 0), (2, 1), (6, 4), (-1, -1)]:
        assert (idx in AT) == _contains_matrix_slow(AT, idx), idx
    fast = _capture(lambda: (0, 6) in AT)
    slow = _capture(lambda: _contains_matrix_slow(AT, (0, 6)))
    assert fast == slow, f"fast={fast} slow={slow}"
    assert fast[:2] == ("exc", "IndexError"), fast


@pytest.mark.parametrize("bad", [5, (1, 2, 3), (1.5, 2), ("a", "b"), np.int64(3)])
def test_contains_matrix_bad_index(bad):
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10, 20, 30], nrows=5, ncols=7)
    fast = _capture(lambda: bad in A)
    slow = _capture(lambda: _contains_matrix_slow(A, bad))
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("pair", [(True, 0), (0, True), (True, True)])
def test_contains_matrix_bool_pair(pair):
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10, 20, 30], nrows=5, ncols=7)
    fast = _capture(lambda: pair in A)
    slow = _capture(lambda: _contains_matrix_slow(A, pair))
    assert fast == slow, f"fast={fast} slow={slow}"


def test_contains_udt_fallback():
    udt = gb.dtypes.register_anonymous(
        np.dtype([("cx", np.int64), ("cy", np.float64)]), "ContainsFastPathProbe"
    )
    u = Vector(udt, 3)
    u[1] = (7, 2.5)
    assert (1 in u) == _contains_vector_slow(u, 1)
    assert (0 in u) == _contains_vector_slow(u, 0)
    fast = _capture(lambda: 5 in u)
    slow = _capture(lambda: _contains_vector_slow(u, 5))
    assert fast == slow, f"fast={fast} slow={slow}"
    assert fast[:2] == ("exc", "IndexError"), fast
    Mu = Matrix(udt, 3, 3)
    Mu[1, 1] = (7, 2.5)
    assert ((1, 1) in Mu) == _contains_matrix_slow(Mu, (1, 1))
    assert ((0, 0) in Mu) == _contains_matrix_slow(Mu, (0, 0))


def test_contains_recorder_fallback():
    v = Vector.from_coo([0], [1.5], size=4)
    with Recorder() as rec:
        assert 0 in v
    assert "extractElement" in "".join(rec.data)
    A = Matrix.from_coo([0], [0], [1.5], nrows=4, ncols=4)
    with Recorder() as rec:
        assert (0, 0) in A
    assert "extractElement" in "".join(rec.data)


def test_contains_pending_value():
    v = Vector(float, 100)
    v[3] = 2.25
    assert (3 in v) is True
    assert (4 in v) is False


# ---------------------------------------------------------------------------
# __setitem__ with an integer key
# ---------------------------------------------------------------------------
def _set_vector_slow(v, key, val):
    Updater(v, opts={})[key] = val


def _set_matrix_slow(A, key, val):
    Updater(A, opts={})[key] = val


def _assert_setitem_vector(dt, key, val, size=10):
    a = Vector(dt, size)
    b = Vector(dt, size)
    ea = _capture(lambda: a.__setitem__(key, val))
    eb = _capture(lambda: _set_vector_slow(b, key, val))
    # On success both capture ("val", None); on error both capture the full
    # ("exc", type, message), so equality pins exception type AND message.
    assert ea == eb, f"exc fast={ea} slow={eb}"
    if ea[0] == "val":
        assert a.isequal(b, check_dtype=True), f"value fast={a.to_coo()} slow={b.to_coo()}"


def _assert_setitem_matrix(dt, key, val, nrows=5, ncols=7):
    a = Matrix(dt, nrows, ncols)
    b = Matrix(dt, nrows, ncols)
    ea = _capture(lambda: a.__setitem__(key, val))
    eb = _capture(lambda: _set_matrix_slow(b, key, val))
    assert ea == eb, f"exc fast={ea} slow={eb}"
    if ea[0] == "val":
        assert a.isequal(b, check_dtype=True), f"value fast={a.to_coo()} slow={b.to_coo()}"


def _setitem_natural_value(dt):
    if "FC" in dt.name:
        return 3 + 4j
    if dt == dtypes.BOOL:
        return True
    if "FP" in dt.name:
        return 3.5
    return 3


# Cross-type coercion: the value's Python type differs from the container dtype.
# The point is that whatever SuiteSparse does on the cast, the fast path does it
# too. Complex-target cases are gated on complex support.
CROSS_COERCION = [
    (dtypes.FP64, 3),  # int -> FP64
    (dtypes.FP32, 3),  # int -> FP32
    (dtypes.INT64, 3.9),  # float -> INT64 (truncation via SS cast)
    (dtypes.INT32, -2.5),  # float -> INT32
    (dtypes.INT8, True),  # bool -> INT8
    (dtypes.UINT8, True),  # bool -> UINT8
    (dtypes.FP64, True),  # bool -> FP64
    (dtypes.UINT8, -1),  # negative int -> unsigned (wrap per SS)
    (dtypes.UINT8, 256),  # int overflow of the container (SS casts)
    (dtypes.INT8, 200),  # int overflow of the container
    (dtypes.INT64, 2**62),  # large int that fits int64
    (dtypes.INT64, 2**63),  # overflow int64 -> OverflowError on both
    (dtypes.UINT64, 2**63),  # fits uint64
    (dtypes.FP64, 1e308),  # large float
    (dtypes.FP32, 1e308),  # overflow FP32 -> inf via SS cast
]
if dtypes._supports_complex:
    CROSS_COERCION += [
        (dtypes.FP64, 3 + 0j),  # complex -> real (whatever SS does)
        (dtypes.FC64, 3),  # int -> FC64
        (dtypes.FC64, 3.5),  # float -> FC64
    ]

CROSS_IDS = [f"{dt.name}<-{val!r}" for dt, val in CROSS_COERCION]


@pytest.mark.parametrize("dt", BUILTIN_DTYPES, ids=DTYPE_IDS)
def test_setitem_dtype_natural(dt):
    val = _setitem_natural_value(dt)
    _assert_setitem_vector(dt, 2, val)
    _assert_setitem_matrix(dt, (1, 2), val)


@pytest.mark.parametrize(("dt", "val"), CROSS_COERCION, ids=CROSS_IDS)
def test_setitem_cross_coercion(dt, val):
    _assert_setitem_vector(dt, 4, val)
    _assert_setitem_matrix(dt, (2, 3), val)


@pytest.mark.parametrize("dt", [dtypes.FP64, dtypes.INT64, dtypes.BOOL], ids=lambda dt: dt.name)
def test_setitem_overwrite(dt):
    a = Vector(dt, 6)
    b = Vector(dt, 6)
    a[1] = 1
    _set_vector_slow(b, 1, 1)
    a[1] = 5  # overwrite
    _set_vector_slow(b, 1, 5)
    assert a.isequal(b, check_dtype=True)


def test_setitem_numpy_index():
    for key in [np.int64(3), np.uint8(3), np.int32(3)]:
        _assert_setitem_vector(dtypes.FP64, key, 2.5)
    _assert_setitem_matrix(dtypes.FP64, (np.int64(1), np.int32(2)), 2.5)


def test_setitem_negative_index():
    _assert_setitem_vector(dtypes.FP64, -1, 9.0)
    _assert_setitem_vector(dtypes.FP64, -10, 9.0)
    _assert_setitem_matrix(dtypes.FP64, (-1, -1), 9.0)
    _assert_setitem_matrix(dtypes.FP64, (-2, 3), 9.0)


@pytest.mark.parametrize("key", [10, -11, 1000, -1000])
def test_setitem_vector_out_of_range(key):
    _assert_setitem_vector(dtypes.FP64, key, 1.0)


@pytest.mark.parametrize("key", [(5, 0), (0, 7), (-6, 0), (0, -8), (100, 3), (2, 100)])
def test_setitem_matrix_out_of_range(key):
    _assert_setitem_matrix(dtypes.FP64, key, 1.0)


@pytest.mark.parametrize("key", [1.5, "x", None, slice(None), [1, 2], (1, 2)])
def test_setitem_vector_bad_key(key):
    _assert_setitem_vector(dtypes.FP64, key, 1.0)


@pytest.mark.parametrize("key", [5, (1, 2, 3), (1.5, 2), ("a", "b"), 1.5, slice(None)])
def test_setitem_matrix_bad_key(key):
    _assert_setitem_matrix(dtypes.FP64, key, 1.0)


def test_setitem_bool_key():
    # bool is treated as an int index by parse_index; the fast path falls back.
    _assert_setitem_vector(dtypes.FP64, True, 1.0)
    _assert_setitem_vector(dtypes.FP64, False, 1.0)
    _assert_setitem_matrix(dtypes.FP64, (True, 0), 1.0)


def test_setitem_value_fallbacks():
    # numpy scalar, Scalar, None, and str all route through the slow path.
    _assert_setitem_vector(dtypes.FP64, 2, np.int32(7))
    _assert_setitem_vector(dtypes.INT64, 2, np.float64(7.9))
    _assert_setitem_vector(dtypes.FP64, 2, Scalar.from_value(7.5))
    _assert_setitem_vector(dtypes.FP64, 2, None)
    _assert_setitem_vector(dtypes.FP64, 2, "bad")
    _assert_setitem_matrix(dtypes.FP64, (1, 1), np.int32(7))
    _assert_setitem_matrix(dtypes.FP64, (1, 1), None)


def test_setitem_udt_fallback():
    udt = gb.dtypes.register_anonymous(
        np.dtype([("sx", np.int64), ("sy", np.float64)]), "SetitemFastPathProbe"
    )
    _assert_setitem_vector(udt, 1, None)  # None clears; both fall back
    au = Vector(udt, 3)
    bu = Vector(udt, 3)
    au[1] = (7, 2.5)
    _set_vector_slow(bu, 1, (7, 2.5))
    assert au.isequal(bu, check_dtype=True)


def test_setitem_recorder_fallback():
    v = Vector(dtypes.FP64, 4)
    with Recorder() as rec:
        v[0] = 1.5
    assert "setElement" in "".join(rec.data)
    A = Matrix(dtypes.FP64, 4, 4)
    with Recorder() as rec:
        A[0, 0] = 1.5
    assert "setElement" in "".join(rec.data)


def test_setitem_update_forms():
    """``obj[i] << x`` and masked / accum forms stay on the Updater path."""
    w = Vector(dtypes.FP64, 5)
    w[2] << 3.5
    assert w.get(2) == 3.5
    w(accum=binary.plus)[2] << 1.5
    assert w.get(2) == 5.0
    A = Matrix(dtypes.FP64, 4, 4)
    A[1, 1] << 3.5
    assert A.get(1, 1) == 3.5
    m = Vector(dtypes.FP64, 5)
    m[:] = 1.0
    mask = Vector(dtypes.BOOL, 5)
    mask[0] = True
    mask[2] = True
    m(mask.V)[:] = 9.0
    assert m.get(0) == 9.0
    assert m.get(1) == 1.0
    assert m.get(2) == 9.0


def test_setitem_empty_vector():
    e = Vector(dtypes.INT32, 3)
    e[1] = 42
    assert e.get(1) == 42
    assert e.nvals == 1


# ---------------------------------------------------------------------------
# ScalarIndexExpr.new()
# ---------------------------------------------------------------------------
def _new_slow(expr, dtype=None, is_cscalar=None, name=None, **opts):
    if is_cscalar is None:
        is_cscalar = False
    return expr.parent._extract_element(
        expr.resolved_indexes, dtype, opts, is_cscalar=is_cscalar, name=name
    )


def _assert_scalars_equal(fast, slow):
    assert fast.dtype == slow.dtype, f"dtype {fast.dtype} vs {slow.dtype}"
    assert fast.is_cscalar == slow.is_cscalar, f"is_cscalar {fast.is_cscalar} vs {slow.is_cscalar}"
    assert fast.is_empty == slow.is_empty, f"is_empty {fast.is_empty} vs {slow.is_empty}"
    fv, sv = fast.value, slow.value
    assert fv == sv or (fv is None and sv is None), f"value {fv!r} vs {sv!r}"
    assert type(fv) is type(sv), f"value type {type(fv).__name__} vs {type(sv).__name__}"


@pytest.mark.parametrize("dt", BUILTIN_DTYPES, ids=DTYPE_IDS)
def test_scalarnew_dtype_parity(dt):
    val = _hit_value(dt)
    v = Vector(dt, 10)
    v[2] = val
    _assert_scalars_equal(v[2].new(), _new_slow(v[2]))
    _assert_scalars_equal(v[3].new(), _new_slow(v[3]))
    A = Matrix(dt, 5, 7)
    A[1, 2] = val
    _assert_scalars_equal(A[1, 2].new(), _new_slow(A[1, 2]))
    _assert_scalars_equal(A[0, 0].new(), _new_slow(A[0, 0]))


def test_scalarnew_negative_and_numpy():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    _assert_scalars_equal(v[-1].new(), _new_slow(v[-1]))
    _assert_scalars_equal(v[-5].new(), _new_slow(v[-5]))
    _assert_scalars_equal(v[-2].new(), _new_slow(v[-2]))
    _assert_scalars_equal(v[np.int64(5)].new(), _new_slow(v[np.int64(5)]))


def test_scalarnew_transposed_matrix():
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10.0, 20.0, 30.0], nrows=5, ncols=7)
    AT = A.T
    _assert_scalars_equal(AT[1, 0].new(), _new_slow(AT[1, 0]))
    _assert_scalars_equal(AT[6, 4].new(), _new_slow(AT[6, 4]))
    _assert_scalars_equal(AT[0, 0].new(), _new_slow(AT[0, 0]))
    _assert_scalars_equal(AT[-1, -1].new(), _new_slow(AT[-1, -1]))


def test_scalarnew_name_honored():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    s = v[5].new(name="myscalar")
    assert s.name == "myscalar"


def test_scalarnew_dtype_cast_fallback():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    _assert_scalars_equal(v[5].new(dtype=dtypes.INT32), _new_slow(v[5], dtype=dtypes.INT32))
    _assert_scalars_equal(v[3].new(dtype=dtypes.INT32), _new_slow(v[3], dtype=dtypes.INT32))


def test_scalarnew_is_cscalar_fallback():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    f_cs = v[5].new(is_cscalar=True)
    s_cs = _new_slow(v[5], is_cscalar=True)
    assert f_cs.is_cscalar
    assert s_cs.is_cscalar
    assert f_cs.value == s_cs.value == 2.5


def test_scalarnew_udt_fallback():
    udt = gb.dtypes.register_anonymous(
        np.dtype([("nx", np.int64), ("ny", np.float64)]), "NewFastPathProbe"
    )
    u = Vector(udt, 3)
    u[1] = (7, 2.5)
    fu = u[1].new()
    su = _new_slow(u[1])
    assert fu.dtype == su.dtype
    assert fu.value["nx"] == su.value["nx"] == 7
    assert u[0].new().is_empty
    assert _new_slow(u[0]).is_empty


def test_scalarnew_recorder_fallback():
    v = Vector.from_coo([0], [1.5], size=4)
    with Recorder() as rec:
        assert v[0].new().value == 1.5
    assert "extractElement" in "".join(rec.data)


def test_scalarnew_autocompute_value_path():
    """With autocompute on, ``v[i].value`` / ``float(v[i])`` still match ``get``."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    with gb.config.set(autocompute=True):
        for idx in [0, 5, 9, 3]:  # 3 is empty
            want = v.get(idx)
            got_val = v[idx].value
            got_get = v[idx].get()
            if want is None:
                assert got_val is None, (idx, got_val, got_get)
                assert got_get is None, (idx, got_val, got_get)
            else:
                assert got_val == want, (idx, got_val, got_get, want)
                assert got_get == want, (idx, got_val, got_get, want)
        assert float(v[5]) == 2.5
        assert int(v[0]) == 1  # 1.5 -> int truncates via Scalar.__int__


def test_scalarnew_autocompute_false_raises():
    """With autocompute off, the value path raises but ``.new()`` still works."""
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    with gb.config.set(autocompute=False):
        with pytest.raises(TypeError):
            v[5].value
        with pytest.raises(TypeError):
            float(v[5])
        assert v[5].new().value == 2.5


# ---------------------------------------------------------------------------
# ScalarIndexExpr value reads (.value / float / int / ...): the single-extract
# fast path in automethods._get_value via ScalarIndexExpr._extract_fast.
# ---------------------------------------------------------------------------
# The reference is the pre-fast-path resolution: _get_value used to resolve the
# expression to a GrB_Scalar (is_cscalar=False) via `.new()`, then read that
# scalar's attribute. `_new_slow` reproduces that GrB_Scalar exactly, so
# `getattr(_new_slow(expr), attr)` is the old behavior for every read attr.
_VALUE_READ_ATTRS = [
    "value",
    "is_empty",
    "_is_empty",
    "__float__",
    "__int__",
    "__complex__",
    "__bool__",
    "__index__",
    "__array__",
]


def _read_attr(scalar, attr):
    """Fingerprint of a value-read attr on a Scalar (or index expr).

    Captures the whole access (``__index__`` raises via its property on
    non-integral dtypes) and reduces the result to ``(repr, typename)`` so NaN
    from an empty float scalar and numpy arrays compare structurally, not by
    value.
    """

    def go():
        getter = getattr(scalar, attr)
        if attr in ("__float__", "__int__", "__complex__", "__bool__", "__index__", "__array__"):
            return getter()
        return getter

    cap = _capture(go)
    if cap[0] == "val":
        return ("val", repr(cap[1]), type(cap[1]).__name__)
    return cap


@pytest.mark.parametrize("dt", BUILTIN_DTYPES, ids=DTYPE_IDS)
def test_scalarvalue_dtype_parity(dt):
    """Every read attr matches the pre-fast-path GrB_Scalar result, hit and miss."""
    val = _hit_value(dt)
    v = Vector(dt, 10)
    v[2] = val
    A = Matrix(dt, 5, 7)
    A[1, 2] = val
    with gb.config.set(autocompute=True):
        for probe in [v[2], v[3], A[1, 2], A[0, 0]]:
            slow = _new_slow(probe)  # GrB_Scalar, old path
            for attr in _VALUE_READ_ATTRS:
                assert _read_attr(probe, attr) == _read_attr(slow, attr), (dt.name, attr)


def test_scalarvalue_transposed_matrix():
    A = Matrix.from_coo([0, 1, 4], [1, 2, 6], [10.0, 20.0, 30.0], nrows=5, ncols=7)
    AT = A.T
    with gb.config.set(autocompute=True):
        for r, c in [(1, 0), (6, 4), (0, 0), (-1, -1)]:
            for attr in _VALUE_READ_ATTRS:
                got = _read_attr(AT[r, c], attr)
                want = _read_attr(_new_slow(AT[r, c]), attr)
                assert got == want, (r, c, attr, got, want)


def test_scalarvalue_negative_and_numpy_index():
    v = Vector.from_coo([0, 5, 9], [1.5, 2.5, 3.5], size=10)
    with gb.config.set(autocompute=True):
        for idx in [-1, -5, -2, np.int64(5)]:
            assert repr(v[idx].value) == repr(_new_slow(v[idx]).value)
            got = _capture(lambda: float(v[idx]))  # noqa: B023
            want = _capture(lambda: float(_new_slow(v[idx])))  # noqa: B023
            assert got == want, (idx, got, want)


def test_scalarvalue_udt_fallback():
    """UDT reads defer to `.new()`; Scalar.value's numpy conversion still runs."""
    udt = gb.dtypes.register_anonymous(
        np.dtype([("vx", np.int64), ("vy", np.float64)]), "ValueFastPathProbe"
    )
    u = Vector(udt, 3)
    u[1] = (7, 2.5)
    with gb.config.set(autocompute=True):
        got = u[1].value
        want = _new_slow(u[1]).value
        assert got["vx"] == want["vx"] == 7
        assert got["vy"] == want["vy"] == 2.5
        assert u[0].value is None
        assert u[0].is_empty is True


def test_scalarvalue_recorder_fallback():
    """With a Recorder active the value path still records an extractElement."""
    v = Vector.from_coo([0], [1.5], size=4)
    with gb.config.set(autocompute=True), Recorder() as rec:
        assert v[0].value == 1.5
    assert "extractElement" in "".join(rec.data)


def test_scalarvalue_pending_value():
    """A value written in non-blocking mode is read back correctly (iso/pending)."""
    v = Vector(dtypes.FP64, 100)
    v[3] = 2.25
    with gb.config.set(autocompute=True):
        assert v[3].value == 2.25
        assert float(v[3]) == 2.25


def test_scalarvalue_index_integral_only():
    """__index__ is available on integral dtypes and absent on float, via the fast path."""
    vi = Vector.from_coo([2], [7], dtype=dtypes.INT64, size=5)
    vf = Vector.from_coo([2], [7.5], dtype=dtypes.FP64, size=5)
    with gb.config.set(autocompute=True):
        assert vi[2].__index__() == 7
        with pytest.raises(AttributeError):
            vf[2].__index__


# ---------------------------------------------------------------------------
# parse_index plain-int lane
# ---------------------------------------------------------------------------
def _resolved_fingerprint(expr):
    """A comparable fingerprint of a resolved scalar index expression."""
    ax = expr.resolved_indexes.indices
    return tuple((a.size, a.index.value, a.dimsize, a._carg.__class__.__name__) for a in ax)


def _index_probe_vector(v, i):
    return _capture(lambda: (v[i].new().value, _resolved_fingerprint(v[i])))


def _index_probe_matrix(A, r, c):
    return _capture(lambda: (A[r, c].new().value, _resolved_fingerprint(A[r, c])))


@pytest.mark.parametrize("i", [0, 2, 4, 9, -1, -10, -6])
def test_indexparse_vector_valid(i):
    """The plain-int fast lane matches the numpy-int lane for valid indices."""
    v = Vector.from_coo([0, 2, 4, 9], [10.0, 20.0, 30.0, 90.0], size=10)
    fast = _index_probe_vector(v, i)
    slow = _index_probe_vector(v, np.int64(i))
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("i", [10, 11, 1000, -11, -100])
def test_indexparse_vector_out_of_range(i):
    """Both lanes raise the identical IndexError for out-of-range indices."""
    v = Vector.from_coo([0, 2, 4, 9], [10.0, 20.0, 30.0, 90.0], size=10)
    fast = _capture(lambda: v[i])
    slow = _capture(lambda: v[np.int64(i)])
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("b", [True, False])
def test_indexparse_vector_bool(b):
    """A bool must not take the int lane; it keeps the pre-existing TypeError."""
    v = Vector.from_coo([0, 2, 4, 9], [10.0, 20.0, 30.0, 90.0], size=10)
    r = _capture(lambda: v[b])
    assert r[0] == "exc", r
    assert r[1] == "TypeError", r


@pytest.mark.parametrize("bad", [2.0, "x", None])
def test_indexparse_vector_non_integer(bad):
    v = Vector.from_coo([0, 2, 4, 9], [10.0, 20.0, 30.0, 90.0], size=10)
    r = _capture(lambda: v[bad])
    assert r[0] == "exc", r


@pytest.mark.parametrize("r", [0, 1, 4, -1, -5])
@pytest.mark.parametrize("c", [0, 2, 6, -1, -7])
def test_indexparse_matrix_valid(r, c):
    A = Matrix.from_coo([0, 1, 4], [0, 2, 6], [1.0, 2.0, 3.0], nrows=5, ncols=7)
    fast = _index_probe_matrix(A, r, c)
    slow = _capture(
        lambda: (
            A[np.int64(r), np.int64(c)].new().value,
            _resolved_fingerprint(A[np.int64(r), np.int64(c)]),
        )
    )
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("r", [1, -1])
@pytest.mark.parametrize("c", [2, -2])
def test_indexparse_matrix_mixed_lane(r, c):
    """Plain int on one axis, numpy int on the other, must still agree."""
    A = Matrix.from_coo([0, 1, 4], [0, 2, 6], [1.0, 2.0, 3.0], nrows=5, ncols=7)
    a = _capture(lambda: (A[r, np.int64(c)].new().value, _resolved_fingerprint(A[r, np.int64(c)])))
    b = _capture(lambda: (A[np.int64(r), c].new().value, _resolved_fingerprint(A[np.int64(r), c])))
    assert a == b, f"a={a} b={b}"


@pytest.mark.parametrize(("r", "c"), [(5, 0), (100, 0), (-6, 0), (0, 7), (0, 1000), (0, -8)])
def test_indexparse_matrix_out_of_range(r, c):
    A = Matrix.from_coo([0, 1, 4], [0, 2, 6], [1.0, 2.0, 3.0], nrows=5, ncols=7)
    fast = _capture(lambda: A[r, c])
    slow = _capture(lambda: A[np.int64(r), np.int64(c)])
    assert fast == slow, f"fast={fast} slow={slow}"


@pytest.mark.parametrize("probe", ["row", "col", "false"])
def test_indexparse_matrix_bool(probe):
    A = Matrix.from_coo([0, 1, 4], [0, 2, 6], [1.0, 2.0, 3.0], nrows=5, ncols=7)
    probes = {
        "row": lambda: A[True, 0],
        "col": lambda: A[0, True],
        "false": lambda: A[False, 1],
    }
    r = _capture(probes[probe])
    assert r[0] == "exc", r
    assert r[1] == "TypeError", r


def test_indexparse_fancy_unaffected():
    """List / ndarray fancy indexing is untouched by the plain-int lane."""
    v = Vector.from_coo([0, 2, 4, 9], [10.0, 20.0, 30.0, 90.0], size=10)
    fast = _capture(lambda: v[[1, 3, 5]].new().to_coo())
    slow = _capture(lambda: v[np.array([1, 3, 5])].new().to_coo())
    assert fast[0] == slow[0] == "val"
    # to_coo returns numpy arrays; compare structurally
    (fi, fx), (si, sx) = fast[1], slow[1]
    assert np.array_equal(fi, si)
    assert np.array_equal(fx, sx)

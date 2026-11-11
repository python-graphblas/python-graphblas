"""graphblas.array_api element-wise functions against numpy, position by position.

The helpers here (``make``, ``assert_array_equal``, ``inputs``) are shared by the other
``test_array_api_*`` modules.
"""

import math
import operator
import warnings

import numpy as np
import pytest

import graphblas.array_api as xp
from graphblas import backend
from graphblas.core.tensor import _has_container as tensor_has_container

if backend != "suitesparse":
    pytest.skip("graphblas.array_api needs the suitesparse backend", allow_module_level=True)
if not tensor_has_container():
    pytest.skip("graphblas.array_api needs SuiteSparse:GraphBLAS 10", allow_module_level=True)


DTYPES = [
    np.dtype(name)
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
]
FLOAT_SPECIALS = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0, 0.5, -2.5, 3.0, 1e30]


def random_values(rng, dtype, shape):
    """Dense values of ``dtype`` with the special values of the kind."""
    if dtype.kind == "b":
        return rng.random(shape) < 0.5
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        vals = rng.integers(-4 if dtype.kind == "i" else 0, 5, shape).astype(dtype)
        edge = rng.random(shape) < 0.1
        extremes = np.where(rng.random(shape) < 0.5, np.array(info.min, dtype), info.max)
        return np.where(edge, extremes, vals)
    real = rng.choice(np.array(FLOAT_SPECIALS), size=shape)
    real = np.where(rng.random(shape) < 0.5, real, np.round(rng.normal(size=shape) * 3, 2))
    if dtype.kind == "c":
        imag = rng.choice(np.array(FLOAT_SPECIALS), size=shape)
        imag = np.where(rng.random(shape) < 0.6, 0.0, imag)
        out = np.empty(shape, dtype)
        with np.errstate(all="ignore"):
            out.real = real
            out.imag = imag
        return out
    with np.errstate(all="ignore"):
        return real.astype(dtype)


def fills_for(dtype):
    if dtype.kind == "b":
        return [False, True]
    if dtype.kind in "iu":
        return [0, 3]
    if dtype.kind == "c":
        return [0, complex(1.5, -2), complex(np.nan, 0)]
    return [0.0, -0.0, 2.5, np.nan, -np.inf]


def same_value(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "c":
        return same_value(a.real, b.real) and same_value(a.imag, b.imag)
    if a.dtype.kind == "f":
        if np.isnan(a) or np.isnan(b):
            return bool(np.isnan(a) and np.isnan(b))
        return bool(a == b and np.signbit(a) == np.signbit(b))
    return bool(np.equal(a, b))


def make(dense, fill, rng=None, *, explicit=0.3):
    """An Array whose value is ``dense`` at every position: the positions that differ from
    ``fill`` are stored, and (with ``rng``) some that equal it are stored too.
    """
    dense = np.asarray(dense)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        fill = np.asarray(fill).astype(dense.dtype)[()]
    if dense.ndim == 0:
        return xp.asarray(dense)
    flat = dense.reshape(-1)
    keep = np.array([not same_value(v, fill) for v in flat], dtype=bool)
    if rng is not None:
        keep |= rng.random(flat.size) < explicit
    idx = np.nonzero(keep.reshape(dense.shape))
    out = xp.from_coo(idx, dense[idx], dense.shape, fill_value=fill, dtype=dense.dtype)
    assert_array_equal(out, dense)
    return out


def check_invariants(x):
    assert isinstance(x, xp.Array)
    assert np.asarray(x.fill_value).dtype == x.dtype
    T = x._T()
    assert T.shape == x.shape
    assert np.dtype(T.dtype.np_type) == x.dtype
    if x.ndim == 0:
        assert T.nvals == 0
    T._check_invariants()


def assert_array_equal(x, expected):
    """``x`` holds ``expected`` exactly: dtype, shape, NaN, signed zeros."""
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape, (x.shape, expected.shape)
    dense = x.todense()
    assert dense.dtype == expected.dtype
    _assert_identical(dense, expected)


def _assert_identical(a, b):
    if a.dtype.kind == "c":
        _assert_identical(a.real, b.real)
        _assert_identical(a.imag, b.imag)
        return
    if a.dtype.kind == "f":
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        bad = (nan_a != nan_b) | (~nan_a & ((a != b) | (np.signbit(a) != np.signbit(b))))
    else:
        bad = a != b
    if bad.any():
        idx = np.argwhere(bad)[:5]
        detail = [(tuple(int(i) for i in ix), a[tuple(ix)], b[tuple(ix)]) for ix in idx]
        raise AssertionError(f"{bad.sum()} positions differ (position, got, expected): {detail}")


def np_call(func, *args, **kwargs):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def inputs(dtype, shape, seed=0):
    """(dense, Array) pairs over the fills of the dtype, some with explicit fills."""
    rng = np.random.default_rng(seed)
    for i, fill in enumerate(fills_for(dtype)):
        dense = random_values(rng, dtype, shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fillv = np.asarray(fill).astype(dtype)[()]
        if dense.ndim:
            sparse = rng.random(shape) < 0.5
            dense = np.where(sparse, fillv, dense).astype(dtype)
        yield dense, make(dense, fillv, rng if i % 2 else None)


# (name, numpy function)
UNARY = [
    ("abs", np.abs),
    ("acos", np.arccos),
    ("acosh", np.arccosh),
    ("asin", np.arcsin),
    ("asinh", np.arcsinh),
    ("atan", np.arctan),
    ("atanh", np.arctanh),
    ("bitwise_invert", np.invert),
    ("ceil", np.ceil),
    ("conj", np.conj),
    ("cos", np.cos),
    ("cosh", np.cosh),
    ("exp", np.exp),
    ("expm1", np.expm1),
    ("floor", np.floor),
    ("imag", np.imag),
    ("isfinite", np.isfinite),
    ("isinf", np.isinf),
    ("isnan", np.isnan),
    ("log", np.log),
    ("log1p", np.log1p),
    ("log2", np.log2),
    ("log10", np.log10),
    ("logical_not", np.logical_not),
    ("negative", np.negative),
    ("positive", np.positive),
    ("real", np.real),
    ("reciprocal", np.reciprocal),
    ("round", np.round),
    ("sign", np.sign),
    ("signbit", np.signbit),
    ("sin", np.sin),
    ("sinh", np.sinh),
    ("sqrt", np.sqrt),
    ("square", np.square),
    ("tan", np.tan),
    ("tanh", np.tanh),
    ("trunc", np.trunc),
]

BINARY = [
    ("add", np.add),
    ("atan2", np.arctan2),
    ("bitwise_and", np.bitwise_and),
    ("bitwise_left_shift", np.left_shift),
    ("bitwise_or", np.bitwise_or),
    ("bitwise_right_shift", np.right_shift),
    ("bitwise_xor", np.bitwise_xor),
    ("copysign", np.copysign),
    ("divide", np.divide),
    ("equal", np.equal),
    ("floor_divide", np.floor_divide),
    ("greater", np.greater),
    ("greater_equal", np.greater_equal),
    ("hypot", np.hypot),
    ("less", np.less),
    ("less_equal", np.less_equal),
    ("logaddexp", np.logaddexp),
    ("logical_and", np.logical_and),
    ("logical_or", np.logical_or),
    ("logical_xor", np.logical_xor),
    ("maximum", np.maximum),
    ("minimum", np.minimum),
    ("multiply", np.multiply),
    ("nextafter", np.nextafter),
    ("not_equal", np.not_equal),
    ("pow", np.power),
    ("remainder", np.remainder),
    ("subtract", np.subtract),
]

SUPPORTED = set(DTYPES)


def expected_or_error(func, *args):
    """Numpy's result, or the exception class numpy raises; None when numpy's result dtype is
    one graphblas.array_api lacks (float16).
    """
    try:
        out = np_call(func, *args)
    except (TypeError, ValueError, OverflowError) as exc:
        return type(exc)
    out = np.asarray(out)
    if out.dtype not in SUPPORTED:
        return None
    return out


def test_all_elementwise_functions_are_tested():
    names = {name for name, _ in UNARY + BINARY} | {"clip"}
    assert names == set(xp._elementwise.__all__)


@pytest.mark.parametrize(("name", "func"), UNARY, ids=[n for n, _ in UNARY])
@pytest.mark.parametrize("shape", [(), (7,), (3, 4), (2, 3, 4)])
def test_unary(name, func, shape):
    for dtype in DTYPES:
        for dense, x in inputs(dtype, shape, seed=hash(name) % 1000):
            expected = expected_or_error(func, dense)
            if expected is None:
                with pytest.raises(TypeError):
                    getattr(xp, name)(x)
                continue
            if isinstance(expected, type):
                with pytest.raises(expected):
                    getattr(xp, name)(x)
                continue
            out = getattr(xp, name)(x)
            try:
                assert_array_equal(out, expected)
            except AssertionError as exc:
                raise AssertionError(f"{name} of {dtype} fill {x.fill_value!r}: {exc}") from None


BINARY_SHAPES = [
    ((5,), (5,)),
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (3, 1)),
    ((), (3, 4)),
    ((3, 4), ()),
    ((), ()),
]
DTYPE_PAIRS = [(dt, dt) for dt in DTYPES] + [
    (np.dtype("int8"), np.dtype("int64")),
    (np.dtype("uint8"), np.dtype("int8")),
    (np.dtype("int32"), np.dtype("float32")),
    (np.dtype("int64"), np.dtype("float64")),
    (np.dtype("float32"), np.dtype("float64")),
    (np.dtype("float32"), np.dtype("complex128")),
    (np.dtype("bool"), np.dtype("int16")),
    (np.dtype("uint64"), np.dtype("int64")),
]


def _binary_cases(dt1, dt2, shapes, seed):
    for i, (dense1, x1) in enumerate(inputs(dt1, shapes[0], seed=seed)):
        for j, (dense2, x2) in enumerate(inputs(dt2, shapes[1], seed=seed + 1)):
            if (i + j) % 2 == 0 or i == j:
                yield dense1, x1, dense2, x2


@pytest.mark.parametrize(("name", "func"), BINARY, ids=[n for n, _ in BINARY])
@pytest.mark.parametrize("shapes", BINARY_SHAPES, ids=str)
def test_binary(name, func, shapes):
    for dt1, dt2 in DTYPE_PAIRS:
        for dense1, x1, dense2, x2 in _binary_cases(dt1, dt2, shapes, hash(name) % 1000):
            expected = expected_or_error(func, dense1, dense2)
            if expected is None:
                with pytest.raises(TypeError):
                    getattr(xp, name)(x1, x2)
                continue
            if isinstance(expected, type):
                with pytest.raises(expected):
                    np_call(getattr(xp, name), x1, x2)
                continue
            out = getattr(xp, name)(x1, x2)
            try:
                assert_array_equal(out, expected)
            except AssertionError as exc:
                raise AssertionError(
                    f"{name} of {dt1} fill {x1.fill_value!r} and {dt2} fill {x2.fill_value!r}: "
                    f"{exc}"
                ) from None


PY_SCALARS = [0, 1, 2, -3, 200, 2**40, True, False, 0.0, -0.0, 0.5, -1.5, np.nan, np.inf, 1j]


@pytest.mark.parametrize(("name", "func"), BINARY, ids=[n for n, _ in BINARY])
@pytest.mark.parametrize("shape", [(), (3, 4)])
def test_binary_python_scalar(name, func, shape):
    for dtype in DTYPES:
        dense, x = next(iter(inputs(dtype, shape, seed=7)))
        for s in PY_SCALARS:
            for args, dargs in (((x, s), (dense, s)), ((s, x), (s, dense))):
                expected = expected_or_error(func, *dargs)
                if expected is None:
                    with pytest.raises(TypeError):
                        getattr(xp, name)(*args)
                    continue
                if isinstance(expected, type):
                    with pytest.raises(expected):
                        np_call(getattr(xp, name), *args)
                    continue
                out = np_call(getattr(xp, name), *args)
                try:
                    assert_array_equal(out, expected)
                except AssertionError as exc:
                    raise AssertionError(f"{name}{tuple(map(repr, dargs))}: {exc}") from None


OPERATORS = [
    (operator.add, "add"),
    (operator.sub, "subtract"),
    (operator.mul, "multiply"),
    (operator.truediv, "divide"),
    (operator.floordiv, "floor_divide"),
    (operator.mod, "remainder"),
    (operator.pow, "pow"),
    (operator.and_, "bitwise_and"),
    (operator.or_, "bitwise_or"),
    (operator.xor, "bitwise_xor"),
    (operator.lshift, "bitwise_left_shift"),
    (operator.rshift, "bitwise_right_shift"),
    (operator.eq, "equal"),
    (operator.ne, "not_equal"),
    (operator.lt, "less"),
    (operator.le, "less_equal"),
    (operator.gt, "greater"),
    (operator.ge, "greater_equal"),
]


@pytest.mark.parametrize(("op", "name"), OPERATORS, ids=[n for _, n in OPERATORS])
def test_operators(op, name):
    rng = np.random.default_rng(1)
    a = rng.integers(1, 5, (3, 4))
    b = rng.integers(1, 5, (4,))
    x, y = make(a, 0), make(b, 2)
    assert_array_equal(op(x, y), np_call(op, a, b))
    assert_array_equal(op(x, 3), np_call(op, a, 3))
    assert_array_equal(op(3, y), np_call(op, 3, b))
    # numpy arrays defer to the Array
    out = op(a, y)
    assert isinstance(out, xp.Array)
    assert_array_equal(out, np_call(op, a, b))


def test_unary_operators():
    a = np.array([1, -2, 0, 3])
    x = make(a, 0)
    assert_array_equal(-x, -a)
    assert_array_equal(+x, +a)
    assert_array_equal(abs(x), abs(a))
    assert_array_equal(~x, ~a)


@pytest.mark.parametrize(
    ("op", "name"),
    [(operator.iadd, "add"), (operator.imul, "multiply"), (operator.isub, "subtract")],
)
def test_inplace_operators(op, name):
    a = np.array([[1.0, 0.0, 2.0], [0.0, -0.0, np.nan]])
    x = make(a, 0.0)
    orig = x
    x = op(x, make(np.array([1.0, 2.0, 0.0]), 0.0))
    assert x is orig
    assert_array_equal(x, np_call(op, a.copy(), np.array([1.0, 2.0, 0.0])))
    x = op(x, 2)
    assert x is orig
    with pytest.raises(TypeError):
        op(make(np.array([1, 2]), 0), 1.5)
    with pytest.raises(ValueError, match="shape"):
        op(make(np.array([1.0, 2.0]), 0.0), make(np.ones((3, 2)), 0.0))


@pytest.mark.parametrize("dtype", ["int8", "uint8", "float32", "float64", "int64"])
def test_clip(dtype):
    rng = np.random.default_rng(3)
    dt = np.dtype(dtype)
    for dense, x in inputs(dt, (3, 4), seed=4):
        for lo, hi in [(1, None), (None, 2), (0, 3), (-1, 1)]:
            if dt.kind == "u" and lo == -1:
                lo = 0
            expected = np.clip(dense, lo, hi) if (lo, hi) != (None, None) else dense
            out = xp.clip(x, min=lo, max=hi)
            assert_array_equal(out, np.asarray(expected).astype(dt))
        lo_dense = random_values(rng, dt, (4,))
        lo_arr = make(lo_dense, 0)
        with np.errstate(all="ignore"):
            expected = np.maximum(dense, lo_dense).astype(dt)
        assert_array_equal(xp.clip(x, min=lo_arr), expected)
    x = make(np.array([1, 2, 3], dtype=np.int8), 0)
    assert_array_equal(xp.clip(x, min=-1000, max=1000), np.array([1, 2, 3], dtype=np.int8))
    assert_array_equal(xp.clip(x), np.array([1, 2, 3], dtype=np.int8))


def test_signed_zero_union():
    # 0.0 fill: an unstored position is +0.0, a stored -0.0 differs
    x = make(np.array([-0.0, 0.0, 1.0, -1.0]), 0.0)
    y = make(np.array([0.0, -0.0, 0.0, 0.0]), 0.0)
    assert x.nnz == 3
    assert y.nnz == 1
    a, b = x.todense(), y.todense()
    for name, func in [("add", np.add), ("multiply", np.multiply), ("subtract", np.subtract)]:
        assert_array_equal(getattr(xp, name)(x, y), func(a, b))
        assert_array_equal(getattr(xp, name)(y, x), func(b, a))


def test_integer_multiply_intersects():
    # Both fills 0: integer products are 0 where either side is unstored, so the result
    # stores only the intersection of entries.
    x = make(np.array([1, 0, 2, 0, 3]), 0)
    y = make(np.array([0, 4, 5, 0, 6]), 0)
    out = x * y
    assert out.nnz == 2
    assert_array_equal(out, np.array([0, 0, 10, 0, 18]))


def test_float_multiply_of_nonnegative_values_intersects():
    x = make(np.array([1.0, 0.0, 2.0, 0.0, 3.0]), 0.0)
    y = make(np.array([0.0, 4.0, 5.0, 0.0, 6.0]), 0.0)
    out = x * y
    assert out.nnz == 2
    assert_array_equal(out, x.todense() * y.todense())
    # a negative value makes -0.0 where the other operand is unstored
    x *= -1
    out = x * y
    assert_array_equal(out, x.todense() * y.todense())
    assert out.nnz > 2
    z = make(np.array([0.0, np.inf, 1.0, 0.0, 0.0]), 0.0)
    assert_array_equal(z * y, np_call(np.multiply, z.todense(), y.todense()))


def test_float_multiply_keeps_union():
    x = make(np.array([-1.0, 0.0, 2.0, np.inf]), 0.0)
    y = make(np.array([0.0, 4.0, 5.0, 0.0]), 0.0)
    assert_array_equal(x * y, np_call(np.multiply, x.todense(), y.todense()))


def test_broadcast_expansion_is_guarded():
    import graphblas as gb

    x = make(np.arange(1, 101, dtype=np.float64).reshape(100, 1), 0.0)
    y = make(np.arange(1, 101, dtype=np.float64).reshape(1, 100), 0.0)
    with gb.config.set(array_api_densify_limit=1000), pytest.raises(xp.DensifyError):
        x + y
    assert_array_equal(x + y, x.todense() + y.todense())


def test_values_mapped_on_every_storage_format():
    # numpy maps the stored values through the Container in place: check bitmap, full,
    # hypersparse and iso storage
    import graphblas as gb

    rng = np.random.default_rng(5)
    for density in [0.0, 0.05, 0.5, 1.0]:
        for shape in [(40,), (30, 20), (6, 5, 4)]:
            a = np.round(rng.normal(size=shape), 2)
            a[rng.random(shape) >= density] = 0.0
            x = make(a, 0.0)
            assert_array_equal(xp.sin(x), np.sin(a))
            assert_array_equal(xp.pow(x, x), np_call(np.power, a, a))
            for fmt in ["bitmap", "sparse", "hypersparse", "full"]:
                M = x._T()._M
                if fmt == "full" and M.nvals != M.nrows * M.ncols:
                    continue
                M.ss.config["sparsity_control"] = fmt
                assert_array_equal(xp.exp(x), np.exp(a))
                assert_array_equal(xp.floor_divide(x, 0.3), np_call(np.floor_divide, a, 0.3))
    iso = xp.from_coo(([0, 3, 5],), 2.5, (8,), fill_value=0.0)
    assert iso._T().ss.is_iso
    assert_array_equal(xp.log(iso), np_call(np.log, iso.todense()))
    other = make(np.arange(8.0), 0.0)
    assert_array_equal(xp.atan2(iso, other), np.arctan2(iso.todense(), other.todense()))
    assert_array_equal(xp.atan2(other, iso), np.arctan2(other.todense(), iso.todense()))
    del gb


def test_zero_d_results_store_nothing():
    x = xp.asarray(2.5)
    y = xp.add(x, 1)
    check_invariants(y)
    assert float(y) == 3.5
    z = make(np.array([1.0, 2.0]), 0.0)[1]
    check_invariants(z)
    assert float(z) == 2.0
    assert z._T().nvals == 0


def test_results_do_not_share_writes():
    x = make(np.array([1.0, 0.0, 2.0]), 0.0)
    y = +x
    y += 1
    assert_array_equal(x, np.array([1.0, 0.0, 2.0]))
    assert_array_equal(y, np.array([2.0, 1.0, 3.0]))


def test_math_consistency_of_fill():
    # An entry equal to the fill maps to the new fill, bit for bit
    for name, _ in UNARY:
        if name in {"logical_not", "bitwise_invert"}:
            continue
        x = xp.from_coo(([0, 1],), [0.7, 0.7], (4,), fill_value=0.7)
        out = getattr(xp, name)(x)
        dense = out.todense()
        assert all(same_value(v, out.fill_value) for v in dense), name


def test_isnan_of_integers_stores_nothing():
    x = make(np.array([1, 0, 2]), 0)
    for name, value in [("isnan", False), ("isinf", False), ("isfinite", True)]:
        out = getattr(xp, name)(x)
        assert out.nnz == 0
        assert out.fill_value == value


def test_math_constants():
    assert xp.e == math.e
    assert xp.pi == math.pi
    assert math.isinf(xp.inf)
    assert math.isnan(xp.nan)
    assert xp.newaxis is None


def test_unsupported_operands():
    x = make(np.array([1.0, 2.0]), 0.0)
    with pytest.raises(TypeError):
        xp.add(x, "a")
    assert (x == "a") is False
    with pytest.raises(TypeError):
        xp.sin(make(np.array([True, False]), False))  # numpy would give float16

"""graphblas.array_api element-wise functions against numpy, position by position.

The helpers here (``make``, ``assert_array_equal``, ``inputs``) are shared by the other
``test_array_api_*`` modules.
"""

import math
import operator
import warnings
import zlib

import numpy as np
import pytest

import graphblas as gb
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


def assert_array_equal(x, expected, *, zero_signs=True):
    """``x`` holds ``expected`` exactly (dtype, shape, NaN, signed zeros), but for the sign of
    a zero at a position ``x`` does not store, which the contract leaves unspecified (DESIGN
    D3b). A 0-d array's value is exact. ``zero_signs=False`` ignores every zero's sign (as
    for max and min, where the Array API leaves the order of -0.0 and 0.0 unspecified).
    """
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape, (x.shape, expected.shape)
    dense = x.todense()
    assert dense.dtype == expected.dtype
    unstored = np.full(x.shape, not zero_signs)
    if x.ndim and zero_signs:
        unstored[...] = True
        unstored[x.to_coo()[0]] = False
    _assert_identical(dense, expected, unstored)


def _assert_identical(a, b, unstored):
    if a.dtype.kind == "c":
        _assert_identical(a.real, b.real, unstored)
        _assert_identical(a.imag, b.imag, unstored)
        return
    if a.dtype.kind == "f":
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        sign = (np.signbit(a) != np.signbit(b)) & ~(unstored & (a == 0) & (b == 0))
        bad = (nan_a != nan_b) | (~nan_a & ((a != b) | sign))
    else:
        bad = a != b
    if bad.any():
        idx = np.argwhere(bad)[:5]
        detail = [(tuple(int(i) for i in ix), a[tuple(ix)], b[tuple(ix)]) for ix in idx]
        raise AssertionError(f"{bad.sum()} positions differ (position, got, expected): {detail}")


# Functions computed by libm (SuiteSparse's unary ops) for floating dtypes: special values
# (NaN, inf, zeros and their signs) exactly as numpy has them, other values within MAXULP units
# in the last place. libm and numpy's own loops are each within 1 to 2 ulp of the exact value
# for these (glibc's libm-test-ulps lists at most 2 for the real ones in double; numpy's float32
# SIMD loops document up to 1.5), so two of them differ by up to 4; on macOS arm64 the largest
# difference seen was 1 (tanh, float32 sin and cos, complex log2 and log10).
LIBM = {"acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh", "exp", "expm1", "log",
        "log1p", "log2", "log10", "sin", "sinh", "tan", "tanh"}  # fmt: skip
MAXULP = 4


def assert_libm_close(x, expected):
    """``x`` holds ``expected`` but for libm's rounding: special values exact, the rest within
    MAXULP (complex: each part's special values exact, the rest within MAXULP * eps of the
    magnitude).
    """
    expected = np.asarray(expected)
    check_invariants(x)
    assert x.dtype == expected.dtype, (x.dtype, expected.dtype)
    assert x.shape == expected.shape, (x.shape, expected.shape)
    got = x.todense()
    if got.dtype.kind not in "fc":
        _assert_identical(got, expected, np.zeros(got.shape, bool))
        return
    if got.dtype.kind == "f":
        parts = [(got, expected)]
    else:
        parts = [(got.real, expected.real), (got.imag, expected.imag)]
    with np.errstate(all="ignore"):
        scale = np.abs(expected)
    for g, e in parts:
        special = ~np.isfinite(g) | ~np.isfinite(e) | (g == 0) | (e == 0)
        _assert_identical(g[special], e[special], np.zeros(special.sum(), bool))
        g, e = g[~special], e[~special]
        if got.dtype.kind == "f":
            np.testing.assert_array_max_ulp(g, e, maxulp=MAXULP)
        else:
            # (a part much smaller than the whole is only as exact as the whole)
            s = scale[~special]
            s = np.where(np.isfinite(s), s, np.abs(e))
            assert (np.abs(g - e) <= MAXULP * np.finfo(e.dtype).eps * s).all(), (g, e)


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
        for dense, x in inputs(dtype, shape, seed=zlib.crc32(name.encode()) % 1000):
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
                (assert_libm_close if name in LIBM else assert_array_equal)(out, expected)
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
        for dense1, x1, dense2, x2 in _binary_cases(
            dt1, dt2, shapes, zlib.crc32(name.encode()) % 1000
        ):
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
                assert_array_equal(out, expected, zero_signs=name not in MAX_MIN)
            except AssertionError as exc:
                raise AssertionError(
                    f"{name} of {dt1} fill {x1.fill_value!r} and {dt2} fill {x2.fill_value!r}: "
                    f"{exc}"
                ) from None


MAX_MIN = {"maximum", "minimum"}
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
                    assert_array_equal(out, expected, zero_signs=name not in MAX_MIN)
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


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_float_maximum_minimum(name):
    # Without NaN GraphBLAS's max and min are numpy's; with it, numpy computes. The order of
    # -0.0 and 0.0 is unspecified.
    rng = np.random.default_rng(8)
    func = getattr(np, name)

    def check(x, expected):
        assert_array_equal(x, expected, zero_signs=False)

    for fill1, fill2 in [(0.0, 0.0), (2.5, -1.0), (np.inf, 0.0), (0.0, -0.0), (np.nan, 1.0)]:
        a = np.where(rng.random((4, 5)) < 0.5, fill1, np.round(rng.normal(size=(4, 5)), 1))
        b = np.where(rng.random(5) < 0.5, fill2, np.round(rng.normal(size=5), 1))
        x, y = make(a, fill1), make(b, fill2)
        check(getattr(xp, name)(x, y), func(a, b))
        for s in [0.0, -0.0, 1.5, np.nan, -np.inf, 3]:
            check(getattr(xp, name)(x, s), func(a, s))
            check(getattr(xp, name)(s, y), func(s, b))
        a[0, 0] = -0.0
        a[1, 1] = np.nan
        x = make(a, fill1)
        check(getattr(xp, name)(x, y), func(a, b))
        check(getattr(xp, name)(y, x), func(b, a))
    x = make(np.array([1.0, 0.0, 2.0]), 0.0)
    y = make(np.array([0.5, 0.5, 0.5]), 0.0)
    check(getattr(xp, name)(x, y), func(x.todense(), y.todense()))
    x *= -0.0  # (the cached facts about x's values must not survive)
    check(getattr(xp, name)(x, y), func(x.todense(), y.todense()))


def test_float_multiply_keeps_union():
    x = make(np.array([-1.0, 0.0, 2.0, np.inf]), 0.0)
    y = make(np.array([0.0, 4.0, 5.0, 0.0]), 0.0)
    assert_array_equal(x * y, np_call(np.multiply, x.todense(), y.todense()))


def test_broadcast_expansion_is_guarded():
    x = make(np.arange(1, 101, dtype=np.float64).reshape(100, 1), 0.0)
    y = make(np.arange(1, 101, dtype=np.float64).reshape(1, 100), 0.0)
    with gb.config.set(array_api_densify_limit=1000), pytest.raises(xp.DensifyError):
        x + y
    assert_array_equal(x + y, x.todense() + y.todense())


def test_values_mapped_on_every_storage_format():
    # numpy maps the stored values through the Container in place: check bitmap, full,
    # hypersparse and iso storage
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


def test_unseen_fill_does_not_raise():
    # x stores every position, so its fill (-1) never meets y's fill as an exponent
    x = xp.from_coo(([0, 1, 2],), [2, 3, 4], (3,), fill_value=-1)
    y = xp.full((3,), -2)
    assert_array_equal(xp.pow(y, x), np.power(np.full(3, -2), [2, 3, 4]))
    with pytest.raises(ValueError, match="negative"):
        xp.pow(make(np.array([2, 0, 3]), 0), -1)


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


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_libm_functions_on_wide_values(dtype):
    # Values over the whole range, the edges of each domain and the special values
    rng = np.random.default_rng(11)
    dt = np.dtype(dtype)
    info = np.finfo(dt)
    edges = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, info.tiny, -info.tiny, info.tiny / 8,
             info.max, -info.max, 88.0, 89.0, 709.0, 711.0, -745.0, np.pi, np.nan, np.inf,
             -np.inf]  # fmt: skip
    n = 3000
    v = np.concatenate([
        np.array(edges),
        rng.normal(size=n) * 3,
        np.exp(rng.uniform(-60, 60, n)) * rng.choice([-1.0, 1.0], n),
        rng.uniform(-1.2, 1.2, n),
    ])  # fmt: skip
    with np.errstate(all="ignore"):
        if dt.kind == "c":
            w = np.empty(v.size, dt)
            w.real, w.imag = v, rng.permutation(v)
            v = w
        v = v.astype(dt)
    x = make(v, 0)
    for name in sorted(LIBM):
        func = dict(UNARY)[name]
        try:
            assert_libm_close(getattr(xp, name)(x), np_call(func, v))
        except AssertionError as exc:
            raise AssertionError(f"{name} of {dtype}: {exc}") from None


def test_math_consistency_of_fill():
    # An entry equal to the fill maps to the new fill, bit for bit (DESIGN D10: no value
    # depends on which positions are stored), also where libm computes the entries
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


def test_intersection_of_broadcast_operands_is_guarded():
    # an intersection of two broadcast operands is an outer product (review finding M4)
    x = make(np.arange(1, 101).reshape(100, 1), 0)
    y = make(np.arange(1, 101).reshape(1, 100), 0)
    b = x > 0
    for func, args in [
        (xp.multiply, (x, y)),
        (xp.bitwise_and, (x, y)),
        (xp.logical_and, (b, b.mT)),
    ]:
        with gb.config.set(array_api_densify_limit=1000), pytest.raises(xp.DensifyError):
            func(*args)
        assert func(*args).nnz == 10000


def test_clip_with_array_bounds_ties():
    # numpy's clip gives the bound on a tie, so the sign of a zero is the bound's (review M6)
    x = make(np.array([0.0, -0.0, 0.0, 1.0, np.nan]), 0.0)
    lo = make(np.array([-0.0, 0.0, 0.0, -0.0, 0.0]), 0.0)
    hi = make(np.array([-0.0, 0.0, 2.0, 0.5, 1.0]), 0.0)
    a, b, c = x.todense(), lo.todense(), hi.todense()
    assert_array_equal(xp.clip(x, min=lo, max=hi), np_call(np.clip, a, b, c))
    assert_array_equal(xp.clip(x, min=lo), np_call(np.clip, a, b, None))
    assert_array_equal(xp.clip(x, max=hi), np_call(np.clip, a, None, c))
    assert_array_equal(xp.clip(x, min=lo, max=0.5), np_call(np.clip, a, b, 0.5))
    # the bounds may broadcast x, as in numpy
    lo2 = make(np.array([[0.5] * 5, [-1.0] * 5]), 0.0)
    assert_array_equal(xp.clip(x, min=lo2), np_call(np.clip, a, lo2.todense(), None))
    s = xp.asarray(np.uint8(0))
    assert_array_equal(xp.clip(s, max=xp.zeros((0,), dtype=xp.uint8)), np.zeros(0, np.uint8))
    assert_array_equal(xp.clip(xp.asarray(2.0), min=make(np.array([1.0, 3.0]), 0.0)), [2.0, 3.0])


def test_signed_zeros_give_way():
    # (review finding H1, DESIGN D3a, D3b) a product with an unstored zero is the fill
    # whatever its sign: a row scaling stores only A's entries
    rng = np.random.default_rng(0)
    A = make(np.where(rng.random((50, 50)) < 0.05, rng.normal(size=(50, 50)), 0.0), 0.0)
    d = make(rng.normal(size=(50, 1)), 0.0)
    out = d * A
    assert out.nnz == A.nnz
    assert_array_equal(out, d.todense() * A.todense())
    # inf and NaN stay exact
    inf = make(np.array([[np.inf, 0.0]]), 0.0)
    assert_array_equal(inf * make(np.array([[0.0, 1.0]]), 0.0), np.array([[np.nan, 0.0]]))
    # divide: the numerator's entries, when no value of the denominator is 0 or NaN
    a = np.where(rng.random((4, 5)) < 0.5, 0.0, rng.normal(size=(4, 5)))
    b = rng.normal(size=5) + 3.0
    for fb in [1.0, -2.0]:
        q = make(a, 0.0) / make(b, fb)
        assert q.nnz == np.count_nonzero(a)
        assert_array_equal(q, a / b)
    q = make(a, 0.0) / make(np.r_[b[:4], 0.0], 1.0)
    assert_array_equal(q, np_call(np.divide, a, np.r_[b[:4], 0.0]))


@pytest.mark.parametrize("name", ["multiply", "divide"])
def test_products_match_numpy_but_for_unstored_zero_signs(name):
    func = getattr(np, name)
    for shapes in [((5,), (5,)), ((3, 4), (4,)), ((3, 1), (1, 4))]:
        for dt1, dt2 in [(np.dtype("float64"),) * 2, (np.dtype("float32"), np.dtype("int8"))]:
            for dense1, x1, dense2, x2 in _binary_cases(dt1, dt2, shapes, 5):
                out = getattr(xp, name)(x1, x2)
                assert_array_equal(out, np_call(func, dense1, dense2))


def test_operands_storing_every_position_keep_products_sparse():
    # An operand that stores every position never shows its fill (review finding N1): A / d
    # and A * (1 / deg)[:, None] store A's entries, whatever d's fill
    rng = np.random.default_rng(3)
    A = make(np.where(rng.random((40, 40)) < 0.05, rng.normal(size=(40, 40)), 0.0), 0.0)
    d = xp.asarray(rng.normal(size=40) + 5.0)
    deg = xp.asarray(rng.integers(1, 9, 40).astype(float))
    q = A / d
    assert q.nnz == A.nnz
    assert_array_equal(q, A.todense() / d.todense())
    inv = 1 / deg
    assert inv.fill_value == np.inf
    p = A * xp.reshape(inv, (40, 1))
    assert p.nnz == A.nnz
    assert p.fill_value == 0.0
    assert_array_equal(p, A.todense() * inv.todense()[:, None])
    # integers too: a full operand's fill takes no part
    i = make(np.where(rng.random((5, 6)) < 0.5, 0, rng.integers(1, 5, (5, 6))), 0)
    full = xp.asarray(rng.integers(1, 5, 6), fill_value=7)
    assert (i * full).nnz == i.nnz
    assert_array_equal(i * full, i.todense() * full.todense())
    assert (A / xp.exp(d)).nnz == A.nnz
    assert_array_equal(A / xp.exp(d), A.todense() / np.exp(d.todense()))
    signed = xp.asarray(rng.normal(size=40))
    assert (A / signed).nnz == A.nnz
    assert_array_equal(A / signed, A.todense() / signed.todense())


def test_one_sided_products():
    # one operand's fill is 0, the other's is not: the result lies within the first's entries
    # (a broadcast second operand is stored in full first, not replicated)
    rng = np.random.default_rng(4)
    A = make(np.where(rng.random((30, 30)) < 0.05, rng.normal(size=(30, 30)), 0.0), 0.0)
    v = make(np.where(rng.random(30) < 0.5, 0.0, rng.normal(size=30)), 0.0)
    e = xp.exp(v)  # (fill 1, some positions unstored)
    w = make(np.where(rng.random(30) < 0.3, 1.0, rng.normal(size=30) + 4), 1.0)
    for out, expected in [
        (A * xp.reshape(e, (1, 30)), A.todense() * e.todense()[None, :]),
        (xp.reshape(e, (30, 1)) * A, e.todense()[:, None] * A.todense()),
        (A / xp.reshape(w, (30, 1)), A.todense() / w.todense()[:, None]),
        (A / e, A.todense() / e.todense()),
    ]:
        assert out.nnz <= A.nnz
        assert_array_equal(out, expected)
    # a second operand as large as the result: the masked union, no full copy
    B = make(np.where(rng.random((60, 60)) < 0.05, rng.normal(size=(60, 60)), 0.0), 0.0)
    big = make(np.where(rng.random((60, 60)) < 0.5, 2.0, rng.normal(size=(60, 60)) + 5), 2.0)
    out = B / big
    assert out.nnz == B.nnz
    assert_array_equal(out, B.todense() / big.todense())


def test_unstored_zero_signs_in_the_oracle():
    # The tests' oracle: a stored -0.0 must match, an unstored one may read 0.0
    x = make(np.array([-0.0, 1.0]), 0.0)
    assert x.nnz == 2
    with pytest.raises(AssertionError, match="differ"):
        assert_array_equal(x, np.array([0.0, 1.0]))
    y = make(np.array([0.0, 1.0]), 0.0)
    assert_array_equal(y, np.array([-0.0, 1.0]))
    with pytest.raises(AssertionError, match="differ"):
        assert_array_equal(xp.asarray(-0.0), np.array(0.0))


def test_one_sided_broadcast_operand_is_stored_in_full():
    # (review finding M9) a broadcast operand that does not store every position is stored in
    # full, not replicated over the axes it broadcasts along: X of shape (1000, 1e6) with 4
    # entries times exp(v)[None, :] (fill 1) stores 4 entries, under a densify limit of 2e6
    rng = np.random.default_rng(12)
    n = 10**6
    rows, cols = [0, 10, 500, 999], [5, 77, n - 1, 123_456]
    X = xp.from_coo((rows, cols), [1.0, 2.0, 3.0, 4.0], (1000, n))
    v = xp.from_coo((np.sort(rng.choice(n, n // 2, replace=False)),), rng.normal(size=n // 2), (n,))
    w = make(np.where(rng.random(n) < 0.5, 2.0, rng.random(n) + 1), 2.0)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    with gb.config.set(array_api_densify_limit=2 * n):
        e = xp.exp(v)
        ed, wd = e.todense()[cols], w.todense()[cols]
        for out, expected in [
            (X * e[None, :], vals * ed),
            (e[None, :] * X, ed * vals),
            (X / w, vals / wd),
        ]:
            assert out.nnz == 4
            assert out.fill_value == 0.0
            coords, values = out.to_coo()
            assert coords[0].tolist() == rows
            np.testing.assert_array_equal(values, expected)
    # numpy's answer at every position, on a small one
    A = make(np.where(rng.random((6, 50)) < 0.05, rng.normal(size=(6, 50)), 0.0), 0.0)
    u = make(np.where(rng.random(50) < 0.5, 1.0, rng.normal(size=50)), 1.0)
    with gb.config.set(array_api_densify_limit=60):
        out = A * u[None, :]
    assert out.nnz == A.nnz
    assert_array_equal(out, A.todense() * u.todense()[None, :])


def test_clip_fill_uses_numpys_array_loop():
    # numpy's 0-d clip breaks a tie differently from its array loop (review finding N3)
    x = xp.full((3,), 0.0)
    lo = xp.full((3,), -0.0)
    a, b = np.zeros(3), np.full(3, -0.0)
    assert_array_equal(xp.clip(x, min=lo, max=0.5), np.clip(a, b, 0.5))
    assert_array_equal(xp.clip(x, min=-0.0, max=0.5), np.clip(a, -0.0, 0.5))


def test_fill_follows_numpys_loop_choice():
    # numpy's power of an array by a 0-d 0.5 is sqrt (pow(-inf, 0.5) is NaN), and between two
    # arrays it is pow (inf): an array's fill takes the loop the dense array would
    x = make(np.array([[-np.inf, 2.0], [-np.inf, -np.inf]], np.float32), np.float32(-np.inf))
    half = xp.asarray(np.float32(0.5))
    halves = make(np.full((2, 2), 0.5, np.float32), np.float32(0.5))
    assert_array_equal(np_call(xp.pow, x, half), np_call(np.power, x.todense(), np.float32(0.5)))
    assert_array_equal(np_call(xp.pow, x, halves), np_call(np.power, x.todense(), halves.todense()))

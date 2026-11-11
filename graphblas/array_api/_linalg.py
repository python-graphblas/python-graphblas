"""Linear algebra in the main namespace: matmul, matrix_transpose, tensordot and vecdot, and
the extension einsum.

Contractions treat a fill of zero as a structural zero (DESIGN D6), as scipy.sparse and
pydata/sparse do: a product with an unstored position is skipped, so 0 * inf there is not NaN,
unlike numpy. Sums may round differently from numpy's (the order of the terms differs). A
nonzero fill is the fill times a matrix of ones plus an array with fill zero, which turns into
sums of the operands broadcast over the result: the result is then dense in general (under the
densify limit). A fill that is not finite, or True, makes the result go through numpy densely.

Semirings (an extension, DESIGN D11): ``semiring=`` takes a ``graphblas`` Semiring or its
name ("min_plus", "max_min", ...), whose monoid takes the place of the sum and whose binary
operator that of the product. A contraction skips the unstored positions, which is right when
every operand's fill is the identity ``e`` of the semiring's monoid and ``e`` annihilates the
multiply from both sides, ``mult(e, y) == mult(y, e) == e``: inf for min_plus, -inf for
max_plus, False for lor_land, 0 for plus_times. The result's fill is then ``e``; other fills
raise ValueError. As in D6 the skipping is structural: an infinity meeting an unstored
position makes no NaN (inf + -inf in min_plus). The annihilator is checked on a set of finite
probe values of each operand type, once per typed semiring. The default (None) is plus_times,
or lor_land for bool, under the rules of the first paragraph.

A contraction that skips the unstored positions is deferred: its result computes on first use,
or only where a consumer needs it (``multiply`` by a zero-filled array, a slice; see
``_lazy``).
"""

import math

import numpy as np

from .. import semiring as semirings
from ..core.operator.semiring import Semiring, TypedBuiltinSemiring, TypedUserSemiring
from ..core.scalar import Scalar
from ..core.tensor import Tensor
from ..core.vector import Vector
from . import _lazy, _utils
from ._array import Array
from ._dtypes import check_supported
from ._utils import gb_dtype, np_dtype

__all__ = ["einsum", "matmul", "matrix_transpose", "tensordot", "vecdot"]


def _check(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs Arrays, not {type(x).__name__}")


def _operand(x):
    """A numpy array as an Array (``ndarray @ Array`` comes here through ``__rmatmul__``)."""
    if isinstance(x, np.ndarray):
        from ._creation import asarray

        return asarray(x)
    return x


def _semiring(dt):
    return (semirings.lor_land if dt.kind == "b" else semirings.plus_times)[gb_dtype(dt)]


def _resolve(semiring, dt, name):
    """``(typed semiring, whether it is the default)`` for the ``semiring=`` argument, typed
    for ``dt``, the operands' promoted dtype.
    """
    default = _semiring(dt)
    if semiring is None:
        return default, True
    if isinstance(semiring, str):
        try:
            sr = semirings.from_string(semiring)
        except ValueError:
            raise ValueError(f"{name}: unknown semiring {semiring!r}") from None
    elif isinstance(semiring, (Semiring, TypedBuiltinSemiring, TypedUserSemiring)):
        sr = semiring
    else:
        raise TypeError(
            f"{name}'s semiring must be a graphblas Semiring or its name, not "
            f"{type(semiring).__name__}"
        )
    if isinstance(sr, Semiring):
        if dt.kind == "b" and sr.name == "plus_times":
            return default, True  # (numpy's sum of bools is their lor)
        try:
            sr = sr[gb_dtype(dt)]
        except KeyError:
            raise TypeError(f"{name}: the semiring {sr.name} does not work with {dt}") from None
    same = sr is default or (
        sr.name == default.name and sr.type == default.type and sr.type2 == default.type2
    )
    return sr, same


def _probes(dt):
    """Values of the numpy dtype ``dt`` to check an annihilator on: both signs, zeros and the
    extremes, all finite (an infinity meets an unstored position only structurally, as in D6).
    """
    if dt.kind == "b":
        return np.array([False, True])
    if dt.kind in "iu":
        info = np.iinfo(dt)
        values = [0, 1, 2, 7, info.max, info.max // 3]
        if dt.kind == "i":
            values += [-1, -5, info.min, info.min // 3]
        return np.array(values, dtype=dt)
    big = np.finfo(dt).max
    values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3e9, big, -big, np.finfo(dt).tiny]
    if dt.kind == "c":
        values += [1j, -1j, 1 + 2j, -3.5 + 0.5j, complex(0, big)]
    return np.array(values, dtype=dt)


_ANNIHILATES = {}


def _annihilates(typed):
    """Whether the identity of the typed semiring's monoid annihilates its multiply from both
    sides on the probe values (cached).
    """
    key = (typed.name, typed.type.name, typed.type2.name)
    if (ok := _ANNIHILATES.get(key)) is None:
        ok = _ANNIHILATES[key] = _check_annihilates(typed)
    return ok


def _check_annihilates(typed):
    if typed.is_positional:
        return False
    mult = typed.binaryop
    e = typed.monoid.identity
    for left in (True, False):
        e_type, y_type = (typed.type, typed.type2) if left else (typed.type2, typed.type)
        ys = _probes(np_dtype(y_type))
        v = Vector.from_coo(np.arange(ys.size), ys, dtype=y_type, size=ys.size)
        s = Scalar.from_value(_utils.scalar(e, np_dtype(e_type)), dtype=e_type)
        r = (v.apply(mult, left=s) if left else v.apply(mult, right=s)).new()
        if r.nvals != ys.size or not bool(np.all(r.to_coo()[1] == e)):
            return False
    return True


def _operand_name(name, i):
    return f"operand {i}" if name == "einsum" else f"x{i + 1}"


def _spell(value):
    """``value`` as the namespace spells it (``xp.inf`` for inf)."""
    value = np.asarray(value).item()
    if isinstance(value, float) and np.isinf(value):
        return "xp.inf" if value > 0 else "-xp.inf"
    return repr(value)


def _sound(typed, arrays, name, *, multiplies=True):
    """The result dtype and fill of a contraction of ``arrays`` with a typed semiring other
    than the default, after checking that it may skip their unstored positions (ValueError
    if not). ``multiplies``: whether values meet the multiply (einsum of one operand only
    sums).
    """
    e = typed.monoid.identity
    out_dt = check_supported(np_dtype(typed.return_type), name)
    if multiplies and not _annihilates(typed):
        mult = typed.binaryop.name
        raise ValueError(
            f"{name}: the semiring {typed.name} on {np_dtype(typed.type)} cannot skip unstored "
            f"positions: the identity {e!r} of its monoid {typed.monoid.name} does not "
            f"annihilate {mult} ({mult}(e, y) and {mult}(y, e) must be e for every y), so no "
            "fill works. Use it on Tensors (x.tensor), which skip missing entries."
        )
    for i, x in enumerate(arrays):
        in_dt = np_dtype(typed.type if i == 0 else typed.type2)
        fill = _utils.scalar(x._fill, in_dt)
        if fill != e:  # (a NaN fill too)
            spelled = _spell(e)
            raise ValueError(
                f"{name} with the semiring {typed.name} needs operands whose fill is {e!r}, "
                f"the identity of {typed.monoid.name}"
                + (f", which annihilates {typed.binaryop.name}" if multiplies else "")
                + f"; {_operand_name(name, i)} has the fill {x._fill.item()!r}. Make the array "
                f"with fill_value={spelled} (xp.asarray(x.tensor, fill_value={spelled}) reads "
                f"x's unstored positions as {e!r})."
            )
    return out_dt, _utils.scalar(e, out_dt)


def _is_zero(fill):
    return not bool(fill)


def _dense_ok(*arrays):
    """Whether the decomposition A = A0 + fill works: finite fills, not bool."""
    for x in arrays:
        if _is_zero(x._fill):
            continue
        if x.dtype.kind == "b" or (x.dtype.kind in "fc" and not np.isfinite(x._fill)):
            return False
    return True


def _dense(func, *arrays):
    from ._creation import _from_numpy

    for x in arrays:
        _utils.check_dense(x.size, "a contraction with a fill that is not finite or is True")
    out = np.asarray(func(*[x.todense() for x in arrays]))
    return _from_numpy(out)


def _zero_fill(x):
    """``x`` minus its fill (exact where the fill is 0), and the fill."""
    from ._elementwise import subtract

    if _is_zero(x._fill):
        return x, None
    return subtract(x, x._fill), x._fill


def _rdt(name, func, x1, x2):
    probe = func(
        np.ones((1,) * max(x1.ndim, 1), x1.dtype), np.ones((1,) * max(x2.ndim, 1), x2.dtype)
    )
    return check_supported(np.asarray(probe).dtype, name)


def matmul(x1, x2, /, *, semiring=None):
    """The matrix product (numpy's matmul rules). ``semiring`` (an extension): a graphblas
    Semiring or its name, under the module's rules for fills.
    """
    from ._elementwise import add, multiply
    from ._manipulation import expand_dims
    from ._statistical import sum as sum_

    x1, x2 = _operand(x1), _operand(x2)
    _check(x1, "matmul")
    _check(x2, "matmul")
    if x1.ndim == 0 or x2.ndim == 0:
        raise ValueError("matmul needs arrays of at least one dimension")
    K = x1.shape[-1]
    if x2.shape[-2 if x2.ndim > 1 else 0] != K:
        raise ValueError(f"matmul: shapes {x1.shape} and {x2.shape} do not align")
    b1 = x1.shape[:-2]
    b2 = x2.shape[:-2]
    try:
        np.broadcast_shapes(b1, b2)
    except ValueError:
        raise ValueError(f"matmul: shapes {x1.shape} and {x2.shape} do not broadcast") from None
    rdt = _rdt("matmul", np.matmul, x1, x2)
    typed, default = _resolve(semiring, rdt, "matmul")
    if not default:
        out_dt, fill = _sound(typed, (x1, x2), "matmul")
        return _product(x1, x2, typed, out_dt, fill)
    if not _dense_ok(x1, x2):
        return _dense(np.matmul, x1, x2)
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    out = _product(a0, b0, typed, rdt, _utils.scalar(0, rdt))
    if fa is None and fb is None:
        return out
    terms = []
    if fa is not None:
        # fa * (ones @ b0): b0 summed over its rows, one row of the result
        s = sum_(b0, axis=-2 if x2.ndim > 1 else 0, dtype=rdt)
        if x1.ndim > 1 and x2.ndim > 1:
            s = expand_dims(s, axis=-2)
        terms.append(multiply(fa, s))
    if fb is not None:
        r = sum_(a0, axis=-1, dtype=rdt)
        if x2.ndim > 1 and x1.ndim > 1:
            r = expand_dims(r, axis=-1)
        terms.append(multiply(fb, r))
    if fa is not None and fb is not None:
        terms.append(multiply(multiply(fa, fb), K))
    for t in terms:
        out = add(out, t)
    from ._elementwise import cast

    return out if out.dtype == rdt else cast(out, rdt)


def _product(a, b, typed, out_dt, fill):
    """``a @ b`` with the typed semiring, skipping unstored positions (deferred, see
    ``_lazy``); the result's fill is ``fill``.
    """
    Ta, Tb = a._T(), b._T()
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    shape = shape + (a.shape[-2:-1] if a.ndim > 1 else ()) + (b.shape[-1:] if b.ndim > 1 else ())
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)

    def build(Ta, Tb):
        return Ta.matmul(Tb, typed)

    spec = _matmul_terms(a.shape, b.shape)
    spec = None if spec is None else (*spec, typed)
    return _lazy.contraction((a, b), build, shape, out_dt, fill, spec)


# Index letters for the einsum that computes a contraction (for slicing a pending result)
_LETTERS = "abcdefghlmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _matmul_terms(sa, sb):
    """``(terms, output)`` of the einsum equal to matmul of shapes ``sa`` and ``sb``, or None
    when a batch axis of size 1 broadcasts (einsum has no broadcasting).
    """
    ba, bb = sa[:-2], sb[:-2]
    nb = max(len(ba), len(bb))
    if nb > len(_LETTERS):
        return None  # pragma: no cover (safety)
    batch = _LETTERS[:nb]
    for b in (ba, bb):
        for n, m in zip(b[::-1], np.broadcast_shapes(ba, bb)[::-1], strict=False):
            if n != m:
                return None
    ta = batch[nb - len(ba) :] + ("ik" if len(sa) > 1 else "k")
    tb = batch[nb - len(bb) :] + ("kj" if len(sb) > 1 else "k")
    out = batch + ("i" if len(sa) > 1 else "") + ("j" if len(sb) > 1 else "")
    return (ta, tb), out


def _tensordot_terms(na, nb, a_axes, b_axes):
    """``(terms, output)`` of the einsum equal to tensordot of arrays of ``na`` and ``nb``
    axes contracting ``a_axes`` with ``b_axes``.
    """
    if na + nb > len(_LETTERS):
        return None  # pragma: no cover (safety)
    ta = _LETTERS[:na]
    tb = list(_LETTERS[na : na + nb])
    for i, j in zip(a_axes, b_axes, strict=True):
        tb[j] = ta[i]
    tb = "".join(tb)
    out = "".join(c for c in ta if c not in tb) + "".join(c for c in tb if c not in ta)
    return (ta, tb), out


def matrix_transpose(x, /):
    _check(x, "matrix_transpose")
    return x.mT


def _tensordot_axes(x1, x2, axes):
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError("tensordot's axes must be nonnegative")
        if axes > min(x1.ndim, x2.ndim):
            raise ValueError("tensordot: more axes than the arrays have")
        return tuple(range(x1.ndim - axes, x1.ndim)), tuple(range(axes))
    a, b = axes
    a = tuple(int(i) % x1.ndim for i in (a if isinstance(a, (list, tuple)) else (a,)))
    b = tuple(int(i) % x2.ndim for i in (b if isinstance(b, (list, tuple)) else (b,)))
    if len(a) != len(b):
        raise ValueError("tensordot needs as many axes of each array")
    return a, b


def tensordot(x1, x2, /, *, axes=2, semiring=None):
    """The tensor dot product (numpy's tensordot). ``semiring`` (an extension): a graphblas
    Semiring or its name, under the module's rules for fills.
    """
    from ._elementwise import add, cast, multiply
    from ._manipulation import reshape
    from ._statistical import sum as sum_

    _check(x1, "tensordot")
    _check(x2, "tensordot")
    a_axes, b_axes = _tensordot_axes(x1, x2, axes)
    for a, b in zip(a_axes, b_axes, strict=True):
        if x1.shape[a] != x2.shape[b]:
            raise ValueError(f"tensordot: axis {a} of x1 and axis {b} of x2 differ in size")
    rdt = check_supported(np.result_type(x1.dtype, x2.dtype), "tensordot")
    typed, default = _resolve(semiring, rdt, "tensordot")
    free1 = tuple(n for i, n in enumerate(x1.shape) if i not in a_axes)
    free2 = tuple(n for i, n in enumerate(x2.shape) if i not in b_axes)
    shape = free1 + free2
    if not default:
        out_dt, fill = _sound(typed, (x1, x2), "tensordot")
        return _tensordot(x1, x2, a_axes, b_axes, shape, typed, out_dt, fill)
    if not _dense_ok(x1, x2):
        return _dense(lambda a, b: np.tensordot(a, b, axes=(a_axes, b_axes)), x1, x2)
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    out = _tensordot(a0, b0, a_axes, b_axes, shape, typed, rdt, _utils.scalar(0, rdt))
    if fa is not None:
        s = sum_(b0, axis=b_axes, dtype=rdt) if b_axes else cast(b0, rdt)
        out = add(out, multiply(fa, reshape(s, (1,) * len(free1) + free2)))
    if fb is not None:
        r = sum_(a0, axis=a_axes, dtype=rdt) if a_axes else cast(a0, rdt)
        out = add(out, multiply(fb, reshape(r, free1 + (1,) * len(free2))))
    if fa is not None and fb is not None:
        K = math.prod(x1.shape[a] for a in a_axes)
        out = add(out, multiply(multiply(fa, fb), K))
    return out if out.dtype == rdt else cast(out, rdt)


def _tensordot(a, b, a_axes, b_axes, shape, typed, out_dt, fill):
    Ta, Tb = a._T(), b._T()
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)
    axes = (list(a_axes), list(b_axes))

    def build(Ta, Tb):
        return Ta.tensordot(Tb, axes, typed)

    spec = _tensordot_terms(a.ndim, b.ndim, a_axes, b_axes)
    spec = None if spec is None else (*spec, typed)
    return _lazy.contraction((a, b), build, shape, out_dt, fill, spec)


def vecdot(x1, x2, /, *, axis=-1, semiring=None):
    """Sum over ``axis`` of ``conj(x1) * x2`` (broadcast), with zero fills structural.

    ``semiring`` (an extension): a graphblas Semiring or its name, whose monoid reduces
    ``mult(x1, x2)`` over ``axis`` (no conjugate), under the module's rules for fills.
    """
    from ._elementwise import add, cast, conj, multiply
    from ._statistical import sum as sum_

    _check(x1, "vecdot")
    _check(x2, "vecdot")
    ndim = min(x1.ndim, x2.ndim)
    if ndim == 0:
        raise ValueError("vecdot needs arrays of at least one dimension")
    if not -ndim <= axis < 0:
        if 0 <= axis < ndim and x1.ndim == x2.ndim:
            axis -= x1.ndim
        else:
            raise ValueError(f"vecdot's axis must be in [-{ndim}, -1], not {axis}")
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("vecdot: the axis differs in size")
    shape = np.broadcast_shapes(x1.shape, x2.shape)
    rdt = check_supported(np.result_type(x1.dtype, x2.dtype), "vecdot")
    typed, default = _resolve(semiring, rdt, "vecdot")
    out_shape = tuple(n for i, n in enumerate(shape) if i != len(shape) + axis)
    if not default:
        out_dt, fill = _sound(typed, (x1, x2), "vecdot")
        return _vecdot(x1, x2, axis, shape, out_shape, typed, out_dt, fill)
    if not _dense_ok(x1, x2):
        return _dense(lambda a, b: np.vecdot(a, b, axis=axis), x1, x2)
    if x1.dtype.kind == "c":
        x1 = conj(x1)
    a0, fa = _zero_fill(x1)
    b0, fb = _zero_fill(x2)
    out = _vecdot(a0, b0, axis, shape, out_shape, typed, rdt, _utils.scalar(0, rdt))
    if fa is not None:
        out = add(out, multiply(fa, sum_(b0, axis=axis, dtype=rdt)))
    if fb is not None:
        out = add(out, multiply(fb, sum_(a0, axis=axis, dtype=rdt)))
    if fa is not None and fb is not None:
        out = add(out, multiply(multiply(fa, fb), shape[axis]))
    return out if out.dtype == rdt else cast(out, rdt)


def _vecdot(a, b, axis, shape, out_shape, typed, out_dt, fill):
    """The semiring's monoid over ``axis`` of ``mult(a, b)`` (broadcast to ``shape``)."""
    Ta, Tb = a._T(), b._T()
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), out_shape), fill)
    if _both_broadcast(Ta.shape, Tb.shape, shape):
        # (an outer product: ewise_mult replicates one operand)
        _utils.expansion_check(Ta, Tb, shape)
    axis += len(shape)

    def build(Ta, Tb):
        return Ta.ewise_mult(Tb, typed.binaryop).new().reduce(typed.monoid, axis=axis)

    return _lazy.contraction((a, b), build, out_shape, out_dt, fill)


def _both_broadcast(sa, sb, shape):
    from ..core.tensor_ewise import _broadcast_factor

    return _broadcast_factor(sa, shape) > 1 and _broadcast_factor(sb, shape) > 1


def einsum(subscripts, /, *operands, semiring=None):
    """Einstein summation (an extension), with numpy's subscripts: ``einsum("ij,jk->ik", a,
    b)`` is ``a @ b``. Each index appears at most once per operand (no diagonals), and
    ``...`` is not supported.

    ``semiring``: a graphblas Semiring or its name; the default is plus_times, or lor_land
    for bool. Its multiply combines the operands of each term, and its monoid sums the terms
    over the indices not in the output. Fills follow the module's rules; with the default
    semiring an operand whose fill is not zero makes the computation go through numpy
    densely (under the densify limit).
    """
    from ..core.tensor_einsum import _einsum_plan
    from ..core.tensor_einsum import einsum as tensor_einsum

    if not isinstance(subscripts, str):
        raise TypeError(f"einsum's subscripts must be a str, not {type(subscripts).__name__}")
    if not operands:
        raise ValueError("einsum needs at least one operand")
    ops = [_operand(x) for x in operands]
    for x in ops:
        _check(x, "einsum")
    rdt = check_supported(np.result_type(*(x.dtype for x in ops)), "einsum")
    typed, default = _resolve(semiring, rdt, "einsum")
    shapes = tuple(x.shape for x in ops)
    shape = _einsum_plan(subscripts, shapes, "greedy")[2]
    if default:
        if not all(_is_zero(x._fill) for x in ops):
            _utils.check_dense(math.prod(shape), "einsum with a nonzero fill")
            return _dense(lambda *a: np.einsum(subscripts, *a), *ops)
        out_dt, fill = rdt, _utils.scalar(0, rdt)
    else:
        out_dt, fill = _sound(typed, ops, "einsum", multiplies=len(ops) > 1)
    if math.prod(shape) == 0 or any(x._T().nvals == 0 for x in ops):
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)

    def build(*tensors):
        return tensor_einsum(subscripts, *tensors, op=typed)

    terms, output = _einsum_plan(subscripts, shapes, "greedy")[:2]
    return _lazy.contraction(ops, build, shape, out_dt, fill, (terms, output, typed))

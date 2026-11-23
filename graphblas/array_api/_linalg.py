"""Linear algebra in the main namespace: matmul, matrix_transpose, tensordot and vecdot, and
the extension einsum.

The rules for fills, semirings and deferred results are in the Technical Notes page
``docs/technical_notes/array_api_contractions.rst``. Where they live in the code:

- a fill of zero is a structural zero: the default semiring's contractions skip unstored
  positions; ``_zero_fills`` relabels or stores the other finite fills, and ``_dense``
  computes with numpy when it cannot;
- ``_sound`` checks the fills for another semiring, and ``_annihilates`` whether its
  monoid's identity annihilates its multiply;
- ``_lazy.contraction`` drops stored entries equal to the fill when an operand stores inf
  or NaN, handles NaN under a float min or max monoid, and defers the result.
"""

import math
import weakref

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

__all__ = ["compute", "einsum", "matmul", "matrix_transpose", "tensordot", "vecdot"]


def _check(x, name):
    if not isinstance(x, Array):
        raise TypeError(f"{name} needs Arrays, not {type(x).__name__}")


def _operand(x):
    """``x`` as an Array when it is a numpy array (``ndarray @ Array`` calls matmul through
    ``Array.__rmatmul__``).
    """
    if isinstance(x, np.ndarray):
        from ._creation import asarray

        return asarray(x)
    return x


def compute(x, /):
    """Compute ``x`` now if it is deferred, and return it (an extension).

    matmul, tensordot, vecdot and einsum may return an Array whose value is computed on first
    use, or only in part when a consumer needs only part of it
    (:ref:`tn_array_api_deferred`). ``compute`` computes all of it at once, for timing or
    before handing it on. Any other Array is returned unchanged.
    """
    _check(x, "compute")
    x._T()
    return x


def _semiring(dt):
    return (semirings.lor_land if dt.kind == "b" else semirings.plus_times)[gb_dtype(dt)]


def _resolve(semiring, dt, name):
    """``(typed semiring, whether it equals the default)`` for the ``semiring=`` argument, typed
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
            return default, True  # (numpy contracts bools with logical or and logical and)
        try:
            sr = sr[gb_dtype(dt)]
        except KeyError:
            raise TypeError(f"{name}: the semiring {sr.name} does not work with {dt}") from None
    same = sr is default or (
        sr.name == default.name and sr.type == default.type and sr.type2 == default.type2
    )
    return sr, same


def _probes(dt):
    """Finite values of the numpy dtype ``dt`` to check an annihilator on: both signs, zeros
    and the extremes. Infinities are left out because an infinity is never multiplied by the
    identity: a contraction skips unstored positions, and when an operand stores an infinity,
    the contraction first drops the operands' stored entries equal to their fills
    (``_lazy.contraction``).
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


_ANNIHILATES = weakref.WeakKeyDictionary()


def _annihilates(typed):
    """Whether the identity of the typed semiring's monoid annihilates its multiply from both
    sides on the ``_probes`` values. Cached per typed semiring object, not per name: two user
    semirings may share a name.
    """
    if (ok := _ANNIHILATES.get(typed)) is None:
        ok = _ANNIHILATES[typed] = _check_annihilates(typed)
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
    """``(result dtype, result fill, arrays)`` for a contraction of ``arrays`` with ``typed``,
    a semiring other than the default, after checking that it may skip their unstored
    positions.

    Raise ValueError when the identity of the monoid does not annihilate the multiply, or
    when an operand's fill is not that identity. An array that stores every position never
    shows its fill, so it is relabeled with the identity (O(1)) when its dtype represents the
    identity exactly. ``multiplies`` is False for an einsum of one operand, which only sums,
    so the multiply is not checked.
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
    out = []
    for i, x in enumerate(arrays):
        in_dt = np_dtype(typed.type if i == 0 else typed.type2)
        fill = _utils.scalar(x._fill, in_dt)
        if fill != e and (relabeled := _relabel_full(x, e)) is not None:
            x, fill = relabeled, e
        out.append(x)
        if fill != e:  # (a NaN fill raises too)
            spelled = _spell(e)
            raise ValueError(
                f"{name} with the semiring {typed.name} needs operands whose fill is {e!r}, "
                f"the identity of {typed.monoid.name}"
                + (f", which annihilates {typed.binaryop.name}" if multiplies else "")
                + f"; {_operand_name(name, i)} has the fill {x._fill.item()!r}. Make the array "
                f"with fill_value={spelled}, or use xp.refill(x, {spelled}), which keeps its "
                "values (it stores the positions that held the old fill)."
            )
    return out_dt, _utils.scalar(e, out_dt), out


def _relabel_full(x, e):
    """``x`` with fill ``e`` when it stores every position (its fill is never seen) and its
    dtype represents ``e`` exactly; else None.
    """
    if x._T().nvals != x.size:
        return None
    with np.errstate(all="ignore"):
        fill = np.asarray(e).astype(x.dtype)[()]  # (inf cast to an integer is not inf)
    if not _utils.same_value(np.asarray(fill).astype(np.asarray(e).dtype)[()], e):
        return None
    return Array._new(x._T(), fill)


def _is_zero(fill):
    return not bool(fill)


def _zero_fills(*arrays):
    """The arrays relabeled to fill 0, or None when one of them cannot be.

    An array that stores every position never shows its fill, so its fill becomes 0 (O(1)).
    An array with another finite fill and a size under the densify limit (a vector or matrix
    of ones times a sparse matrix) first stores its fill at every position it does not store,
    O(its size). Those positions then store a finite value, and a finite value times an
    unstored zero of another operand is 0, so skipping those terms changes nothing. With any
    other nonzero fill it returns None, and the caller computes with numpy densely: a
    non-finite fill times an unstored zero of another operand is NaN.
    """
    out = []
    for x in arrays:
        if not _is_zero(x._fill):
            T = x._T()
            if T.nvals != x.size:
                small = x.size <= _utils.densify_limit()
                if x.ndim == 0 or not small or not np.isfinite(x._fill):
                    return None
                T = _utils.with_fill(x, _utils.scalar(0, x.dtype))._T()
            x = Array._new(T, _utils.scalar(0, x.dtype))
        out.append(x)
    return out


def _dense(func, shape, *arrays):
    """``func`` of dense numpy copies of the arrays, under the densify limit on each operand
    and on the result, whose shape is ``shape``.
    """
    from ._creation import _from_numpy

    what = "a contraction with a nonzero fill that does not store every position"
    for x in arrays:
        _utils.check_dense(x.size, what)
    _utils.check_dense(math.prod(shape), what)
    # Silence floating-point warnings, as the namespace's other numpy paths do. The GraphBLAS
    # paths never warn, so a warning here would depend on the fills. A 0 * inf term here is
    # NaN, as in numpy, and some BLAS kernels set the invalid flag even without one.
    with np.errstate(all="ignore"):
        out = np.asarray(func(*[x.todense() for x in arrays]))
    return _from_numpy(out)


def _rdt(name, func, x1, x2):
    probe = func(
        np.ones((1,) * max(x1.ndim, 1), x1.dtype), np.ones((1,) * max(x2.ndim, 1), x2.dtype)
    )
    return check_supported(np.asarray(probe).dtype, name)


def matmul(x1, x2, /, *, semiring=None):
    """The matrix product, with numpy's matmul rules for shapes.

    ``semiring`` (an extension) is a graphblas Semiring or its name, such as ``"min_plus"``.
    The unstored positions of an operand whose fill is 0 contribute no terms, so ``0 * inf``
    there is not NaN, unlike numpy. :ref:`tn_array_api_contractions` gives the rules for
    other fills, for semirings, and for results computed on first use.
    """
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
        out_dt, fill, (x1, x2) = _sound(typed, (x1, x2), "matmul")
        return _product(x1, x2, typed, out_dt, fill)
    if (ops := _zero_fills(x1, x2)) is None:
        return _dense(np.matmul, _matmul_shape(x1, x2), x1, x2)
    return _product(*ops, typed, rdt, _utils.scalar(0, rdt))


def _matmul_shape(a, b):
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return shape + (a.shape[-2:-1] if a.ndim > 1 else ()) + (b.shape[-1:] if b.ndim > 1 else ())


def _product(a, b, typed, out_dt, fill):
    """``a @ b`` with the typed semiring, skipping unstored positions (deferred, see
    ``_lazy``); the result's fill is ``fill``.
    """
    Ta, Tb = a._T(), b._T()
    shape = _matmul_shape(a, b)
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)

    def build(Ta, Tb, op=typed):
        return Ta.matmul(Tb, op)

    spec = _matmul_terms(a.shape, b.shape)
    spec = None if spec is None else (*spec, typed)
    return _lazy.contraction((a, b), build, shape, out_dt, fill, spec, a.shape[-1], True, typed)


# Index letters for the einsum that computes a contraction (for slicing a pending result).
# There is no i, j or k: _matmul_terms uses them for matmul's own axes.
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
    a = tuple(_axis(i, x1.ndim) for i in (a if isinstance(a, (list, tuple)) else (a,)))
    b = tuple(_axis(i, x2.ndim) for i in (b if isinstance(b, (list, tuple)) else (b,)))
    if len(a) != len(b):
        raise ValueError("tensordot needs as many axes of each array")
    return a, b


def _axis(i, ndim):
    i = int(i)
    if not -ndim <= i < ndim:
        raise IndexError(f"tensordot: axis {i} is out of bounds for an array of {ndim} dimensions")
    return i % ndim


def tensordot(x1, x2, /, *, axes=2, semiring=None):
    """The tensor dot product, as numpy's tensordot.

    ``semiring`` (an extension) is a graphblas Semiring or its name. Fills follow the rules
    for ``matmul`` (:ref:`tn_array_api_contractions`).
    """
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
        out_dt, fill, (x1, x2) = _sound(typed, (x1, x2), "tensordot")
        return _tensordot(x1, x2, a_axes, b_axes, shape, typed, out_dt, fill)
    if (ops := _zero_fills(x1, x2)) is None:
        func = lambda a, b: np.tensordot(a, b, axes=(a_axes, b_axes))  # noqa: E731
        return _dense(func, shape, x1, x2)
    return _tensordot(*ops, a_axes, b_axes, shape, typed, rdt, _utils.scalar(0, rdt))


def _tensordot(a, b, a_axes, b_axes, shape, typed, out_dt, fill):
    Ta, Tb = a._T(), b._T()
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)
    axes = (list(a_axes), list(b_axes))

    def build(Ta, Tb, op=typed):
        return Ta.tensordot(Tb, axes, op)

    spec = _tensordot_terms(a.ndim, b.ndim, a_axes, b_axes)
    spec = None if spec is None else (*spec, typed)
    nterms = math.prod(a.shape[i] for i in a_axes)
    return _lazy.contraction((a, b), build, shape, out_dt, fill, spec, nterms, True, typed)


def vecdot(x1, x2, /, *, axis=-1, semiring=None):
    """The sum over ``axis`` of ``conj(x1) * x2``, broadcast.

    ``semiring`` (an extension) is a graphblas Semiring or its name, whose monoid reduces
    ``mult(x1, x2)`` over ``axis``. The default semiring, plus_times (lor_land for bool),
    conjugates ``x1`` whether it is given or not; other semirings do not. Fills follow the
    rules for ``matmul`` (:ref:`tn_array_api_contractions`).
    """
    from ._elementwise import conj

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
        out_dt, fill, (x1, x2) = _sound(typed, (x1, x2), "vecdot")
        return _vecdot(x1, x2, axis, shape, out_shape, typed, out_dt, fill)
    if (ops := _zero_fills(x1, x2)) is None:
        return _dense(lambda a, b: np.vecdot(a, b, axis=axis), out_shape, x1, x2)
    a0, b0 = ops
    if a0.dtype.kind == "c":
        a0 = conj(a0)
    return _vecdot(a0, b0, axis, shape, out_shape, typed, rdt, _utils.scalar(0, rdt))


def _vecdot(a, b, axis, shape, out_shape, typed, out_dt, fill):
    """The semiring's monoid over ``axis`` of ``mult(a, b)`` (broadcast to ``shape``)."""
    Ta, Tb = a._T(), b._T()
    if Ta.nvals == 0 or Tb.nvals == 0 or math.prod(shape) == 0:
        return Array._new(Tensor(gb_dtype(out_dt), out_shape), fill)
    _utils.mult_expansion_check(Ta, Tb, shape)
    axis += len(shape)

    def build(Ta, Tb, op=typed):
        return Ta.ewise_mult(Tb, op.binaryop).new().reduce(op.monoid, axis=axis)

    return _lazy.contraction(
        (a, b), build, out_shape, out_dt, fill, nterms=shape[axis], typed=typed
    )


def einsum(subscripts, /, *operands, semiring=None):
    """Einstein summation (an extension), with numpy's subscripts: ``einsum("ij,jk->ik", a,
    b)`` is ``a @ b``.

    An index may appear at most once in each operand (no diagonals), and ``...`` is not
    supported.

    ``semiring`` is a graphblas Semiring or its name; the default is plus_times, or lor_land
    for bool. Its multiply combines the operands of each term, and its monoid sums the terms
    over the indices not in the output. Fills follow the rules for ``matmul``
    (:ref:`tn_array_api_contractions`).
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
    scalars = [x._fill for x in ops if x.ndim == 0]
    if default and any(map(bool, scalars)) and all(np.isfinite(scalars)):
        # A finite 0-d operand is a factor of every term, so it factors out of the sum. A
        # non-finite one makes the contraction compute densely with numpy, since inf times an
        # unstored zero is NaN
        return _einsum_scalars(subscripts, ops)
    if default:
        if (zeroed := _zero_fills(*ops)) is None:
            return _dense(lambda *a: np.einsum(subscripts, *a), shape, *ops)
        ops = zeroed
        out_dt, fill = rdt, _utils.scalar(0, rdt)
    else:
        out_dt, fill, ops = _sound(typed, ops, "einsum", multiplies=len(ops) > 1)
    if math.prod(shape) == 0 or any(x._T().nvals == 0 for x in ops):
        return Array._new(Tensor(gb_dtype(out_dt), shape), fill)

    def build(*tensors, op=typed):
        return tensor_einsum(subscripts, *tensors, op=op)

    terms, output = _einsum_plan(subscripts, shapes, "greedy")[:2]
    if len(ops) == 3:
        masked = _masked_three(terms, output, ops, typed, out_dt, fill)
        if masked is not None:
            return masked
    sizes = {
        c: n for term, s in zip(terms, shapes, strict=True) for c, n in zip(term, s, strict=True)
    }
    nterms = math.prod(n for c, n in sizes.items() if c not in output)
    product = _lazy._one_product(terms, output)
    return _lazy.contraction(
        ops, build, shape, out_dt, fill, (terms, output, typed), nterms, product, typed
    )


# (monoid, multiply) pairs where the multiply distributes over the monoid, so that a term's
# third factor can multiply the sum of the other two's products: s * (a + b) = s*a + s*b
_DISTRIBUTES = {
    ("plus", "times"),
    ("lor", "land"),
    ("min", "plus"),
    ("max", "plus"),
    ("min", "max"),
    ("max", "min"),
}


def _masked_three(terms, output, ops, typed, out_dt, fill):
    """An einsum of three operands computed as a masked product, or None when that does not
    apply.

    It applies when one operand's indices are exactly those the other two keep after
    contracting their shared indices: triangle counting's ``"ij,jk,ik->"``, or an SDDMM's
    ``"ik,ij,jk->ik"``. The pair's product is computed only at the third operand's stored
    positions, multiplied by the third, and summed into the output. This has the same terms
    as the full einsum, since a position the third operand does not store contributes no
    term either way. Only the order of the sums differs, which can change rounding and where a
    sum overflows.

    When an operand stores inf or NaN, stored entries equal to the fill are dropped first, as
    in every contraction (see ``_lazy.contraction``). With a NaN under a float min or max
    monoid it returns None, so that the full einsum propagates the NaN. With plus_times the
    third operand must be finite: ``inf * (x + y)`` is not ``inf * x + inf * y`` when x and
    y differ in sign.
    """
    if (typed.monoid.name, typed.binaryop.name) not in _DISTRIBUTES or not all(x.ndim for x in ops):
        return None
    finite = [x.dtype.kind not in "fc" or _utils.fact(x, "finite") for x in ops]
    if not all(finite):
        if typed.monoid.name in {"min", "max"} and any(_utils.fact(x, "nan") for x in ops):
            return None
        if any(x.dtype.kind == "c" for x in ops):
            return None  # (complex infinities do not distribute)
        from ._manipulation import prune

        ops = [prune(x) for x in ops]
    for m, tm in enumerate(terms):
        a, b = (t for i, t in enumerate(terms) if i != m)
        contracted = (set(a) & set(b)) - set(tm)
        # (the pair's other indices, batch ones included, are exactly the third's)
        if not contracted or set(a) | set(b) != set(tm) | contracted or not set(output) <= set(tm):
            continue
        if typed.monoid.name == "plus" and not finite[m]:
            continue
        A, B = (x for i, x in enumerate(ops) if i != m)
        P = einsum(f"{a},{b}->{tm}", A, B, semiring=typed)
        TS = ops[m]._T()
        if (node := P._pending) is not None:
            TP = node.compute(mask=TS.S, **_lazy._dot_hint(node, TS.nvals, P.shape))
        else:
            TP = P._T()  # (computed already: deferral is off, or the product is empty)
        R = Array._new(TS.ewise_mult(TP, typed.binaryop).new(dtype=gb_dtype(out_dt)), fill)
        return R if output == tm else einsum(f"{tm}->{output}", R, semiring=typed)
    return None


def _einsum_scalars(subscripts, ops):
    """Einsum with the 0-d operands taken out as a product of scalars: the einsum of the
    others, times that product.
    """
    from ._elementwise import multiply

    inputs, arrow, output = subscripts.replace(" ", "").partition("->")
    terms = inputs.split(",")
    keep = [i for i, x in enumerate(ops) if x.ndim > 0]
    factor = None
    for x in ops:
        if x.ndim == 0:
            factor = x if factor is None else multiply(factor, x)
    if not keep:
        return _dense(lambda *a: np.einsum(subscripts, *a), (), *ops)
    rest = ",".join(terms[i] for i in keep) + arrow + output
    if not arrow:
        # (implicit output: the letters that appear once, sorted, as numpy orders them)
        counts = {c: "".join(terms).count(c) for c in "".join(terms)}
        rest += "->" + "".join(sorted(c for c, k in counts.items() if k == 1))
    return multiply(einsum(rest, *(ops[i] for i in keep)), factor)

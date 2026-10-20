"""How python-graphblas types a literal operand, such as the ``1`` in ``v + 1``.

A Python number is weak, as numpy 2 types it (NEP 50): it takes the dtype of
the operand beside it when that dtype's kind holds it, so ``int8_vec + 1`` is
INT8 and ``int8_vec * 0.5`` is FP64. A numpy value keeps its own dtype. Beside
a UDT, the same rules apply leaf by leaf. :func:`_literal_operand` decides for
each operand; a literal that must become an element of a UDT is checked by
:func:`_check_literal_fits`.

The module also types a literal stored into an object
(:func:`_literal_store_dtype`) and a fill value
(:func:`_weak_builtin_literal_dtype` with no op). A literal that converts into
a UDT without fitting it is deprecated in :func:`_refuse_unless_exact` (MAINT).
"""

import datetime
import math
import numbers
import struct
import warnings
from functools import lru_cache

import numpy as np

from ..dtypes import FP64, lookup_dtype
from . import utils
from .dtypes import _view_if_same_layout
from .operator import IndexUnaryOp, SelectOp, TypedOpBase, binary_from_string
from .operator.binary import BinaryOp
from .operator.monoid import Monoid
from .udt import (
    BUILTIN_UDT_BINARY_OPS,
    _casts_safely_into,
    _dict_to_record,
    _leaf_arrays,
    _promoted_record,
    _strip_leading_ones,
)


def _literal_operand(dtype, value, op):
    """Return literal ``value`` and the dtype to make it, as ``op``'s operand beside ``dtype``.

    A dtype of ``None`` means the value's own dtype. This is where
    python-graphblas decides how a literal operand is typed; beside a built-in
    dtype, see :func:`_weak_builtin_literal_dtype`.

    Beside a UDT, a literal becomes an element of that UDT, as it must for a
    user's op, when its type allows it (:func:`_check_literal_fits`). With a
    built-in op that lifts to UDTs, a literal is an operand in its own right,
    typed as numpy 2 types it (NEP 50), by :func:`_literal_type`:

    - a numpy scalar or array keeps its dtype (strong); an array's type is the
      array layout of its dtype (:func:`_array_literal_udt`);
    - a Python ``bool``, ``int``, ``float`` or ``complex`` is weak: in each
      element or leaf it takes that element's dtype when it fits the kind, so
      ``int8_udt + 1`` stays ``int8_udt`` and ``int8_udt += 1`` can be stored,
      while ``int8_udt * 2.5`` has float64 elements (see
      :func:`_weak_literal_udt`), where converting it to the UDT would
      multiply by 2;
    - a tuple, list or dict of Python numbers, laid out like an element, is
      weak in the same way, leaf by leaf: ``int8_udt + (1, 2, 3)`` stays
      ``int8_udt`` and ``+ (0.5, 1.5, 2.5)`` has float64 elements.

    ``eq`` and ``ne`` type a literal the same way, as numpy compares, so
    ``fp32_udt == 0.1`` compares in float32, as ``fp32_vec == 0.1`` does.

    A comparison keeps no result type, so a Python int beyond an integer
    element's range is compared exactly, as numpy 2 compares it, as infinity
    of its sign: every element compares with that infinity as with the int,
    while a type that held both would round them (in FP64, the INT64
    ``2**63 - 1`` and the int ``2**63`` are equal). So ``int64_vec < 2**63`` is
    all True and ``int8_vec == 300`` all False. Beside a UDT, where only ``eq``
    and ``ne`` compare, a literal with such an int in an integer leaf equals no
    element: ``int8_udt == 300`` is False, not an error.
    """
    if getattr(op, "is_positional", False):
        return value, None
    if not dtype._is_udt:
        literal_dtype = _weak_builtin_literal_dtype(dtype, value, op)
        if (
            literal_dtype is not dtype
            and literal_dtype is not None
            and type(value) is int
            and dtype.np_type.kind in "iu"
        ):
            # Only an int beyond dtype's range takes another type here, typed so
            # by a comparison or by an op that gives another type (truediv).
            if isinstance(op, str):
                op = binary_from_string(op)
            if op.name in _COMPARISONS:
                return (math.inf if value > 0 else -math.inf), FP64
        return value, literal_dtype
    if getattr(op, "_udt_types", True) is None:
        # An op that does not take UDTs at all (``lt``) reports that itself.
        return value, dtype
    if isinstance(op, Monoid) and not op._anonymous:
        # Element-wise, a Monoid is its BinaryOp (``get_typed_op`` types it so).
        op = op.binaryop
    if type(op) is BinaryOp and not op._anonymous:
        is_comparison = op.name in ("eq", "ne")
        if is_comparison or op.name in BUILTIN_UDT_BINARY_OPS:
            # truediv gives float leaves whatever the int, so, as a comparison, it
            # types an int out of an integer leaf's range exactly instead of raising.
            exact = is_comparison or op.name == "truediv"
            literal_type = _literal_type(dtype, value, exact=exact)
            if literal_type is not None:
                if is_comparison and not isinstance(value, (np.generic, np.ndarray)):
                    np_type = dtype.np_type
                    parts = _dict_to_record(np_type, value) if isinstance(value, dict) else value
                    if _int_beyond_leaf(np_type, parts) is not None:
                        # As infinity in its leaf, it still equals no element.
                        value = _beyond_as_infinity(np_type, parts)
                        literal_type = _literal_type(dtype, value)
                return value, literal_type
    _check_literal_fits(dtype, value)
    return value, dtype


# Built-in ops that compare their operands, and so keep no result type.
_COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge"}
_COMPARISONS.update([f"value{name}" for name in list(_COMPARISONS)])


# The types a Python float or complex can overflow.
_SINGLE_PRECISION = {np.dtype(np.float32), np.dtype(np.complex64)}


def _weak_builtin_literal_dtype(dtype, value, op=None):
    """Return the dtype of Python number ``value`` beside built-in ``dtype``, or ``None``.

    A Python ``bool``, ``int``, ``float`` or ``complex`` is weak, as numpy 2
    types it (NEP 50), by the table UDT leaves use
    (:func:`_weak_literal_leaf`): ``int8_vec + 1`` is INT8,
    ``fp32_vec * 0.5`` is FP32, ``int8_vec * 0.5`` is FP64, and
    ``int8_vec + 300`` raises ``OverflowError``. python-graphblas applies the
    table itself, so numpy 1 gives the same answers. ``None`` keeps the value's
    own dtype (strong): for anything but a Python number, for a typed op, which
    fixes the operand types itself, and for the thunk of an index op that does
    not compare it with the values, such as a row index.

    A comparison keeps no result type, so an int beyond ``dtype``'s range
    takes a type that holds it rather than raise (:func:`_literal_operand` then
    compares it as infinity). ``op`` of ``None``, for a fill value, does the
    same, so the value is not lost.
    """
    kind = _LITERAL_KINDS.get(type(value))
    if kind is None:
        return None
    if isinstance(op, str):
        try:
            op = binary_from_string(op)
        except Exception:
            return None
    exact = op is None
    if op is not None:
        if isinstance(op, TypedOpBase):
            return None
        exact = getattr(op, "name", None) in _COMPARISONS and not op._anonymous
        if isinstance(op, (IndexUnaryOp, SelectOp)) and not exact:
            return None
    try:
        np_type = _weak_literal_leaf([value], kind, dtype.np_type, exact=exact)
    except OverflowError:
        if exact or not _gives_another_type(op, dtype):
            raise
        # An op whose result is not the operand's type has no narrow result to
        # keep, so an int out of range is typed exactly, as for a comparison:
        # int8_vec / 300 is FP64, as numpy divides integers in float64.
        np_type = _weak_literal_leaf([value], kind, dtype.np_type, exact=True)
    if np_type == dtype.np_type:
        rv = dtype
    else:
        try:
            rv = lookup_dtype(np_type)
        except ValueError:  # A complex literal where the backend has no complex types
            return None
    if kind != "b" and rv.np_type in _SINGLE_PRECISION:
        parts = (value.real, value.imag) if kind == "c" else (float(value),)
        try:
            for part in parts:
                struct.pack("<f", part)  # raises when a finite part rounds to infinity
        except OverflowError:
            if op is None:
                # A fill value widens rather than lose its value, as an int out of
                # range does: FP32's to_dense(fill_value=1e300) is float64.
                return lookup_dtype(np.complex128 if kind == "c" else np.float64)
            # As numpy 2 and a UDT warn: fp32_vec + 1e300 (or + 2**200) is inf, and
            # so is the 1e300 of fp32_vec == 1e300, which numpy compares with inf too.
            utils._warn_from_caller(
                f"overflow encountered in cast: {value!r} is infinity as {rv}", RuntimeWarning
            )
    return rv


def _gives_another_type(op, dtype):
    """Whether built-in ``op`` on two operands of built-in ``dtype`` gives another type.

    ``truediv`` on integers gives FP64, and the numpy ufuncs that compute in
    float give a float type, as numpy computes them in a float loop.
    """
    if op is None or getattr(op, "_anonymous", True) or not hasattr(op, "__getitem__"):
        return False
    try:
        return op[dtype].return_type != dtype
    except Exception:  # An op that does not take this dtype reports it itself, later.
        return False


def _literal_store_dtype(dtype, value):
    """Return the dtype to make literal ``value`` a Scalar of, to store it as ``dtype``.

    A UDT types the literal itself. A Python int is INT64 by itself, so one
    beyond INT64 (the top half of UINT64) is made ``dtype``:
    ``uint64_vec[0] = 2**64 - 1`` works, and a type that cannot hold it raises.
    ``None`` lets ``Scalar.from_value`` choose.
    """
    if dtype._is_udt or type(value) is int and not -(2**63) <= value < 2**63:
        return dtype
    return None


def _literal_type(dtype, value, *, exact=False):
    """Return the type of literal ``value`` beside UDT ``dtype``, or ``None`` when it has none.

    A Python number (a ``Fraction`` or ``Decimal`` counts as the int or float
    it is worth) or a tuple, list or dict of them is weak, leaf by leaf
    (:func:`_weak_literal_udt`); with ``exact``, for a comparison, an int
    out of a leaf's range takes a type that holds it. A numpy value is
    strong: its dtype, in the array layout of its shape for an array
    (:func:`_array_literal_udt`); a sequence holding numpy values is the
    array numpy makes of it. Anything else has no type here, and
    ``Scalar.from_value`` reports it.
    """
    value = _as_python_number(value)
    np_type = dtype.np_type
    if isinstance(value, dict) and np_type.names is not None:
        value = _dict_to_record(np_type, value)
    if type(value) in _LITERAL_KINDS or isinstance(value, (tuple, list)):
        literal_type = _weak_literal_udt(dtype, value, exact=exact)
        if literal_type is not None or type(value) in _LITERAL_KINDS:
            return literal_type
        try:
            value = np.asarray(value)
        except (TypeError, ValueError):  # ragged
            return None
    if isinstance(value, (np.bool_, np.number)):
        return lookup_dtype(value.dtype)
    if isinstance(value, (np.ndarray, np.void)):
        if value.ndim == 0 and value.dtype.kind in "biufc":
            return lookup_dtype(value.dtype)
        return _array_literal_udt(dtype, value)
    return None


def _check_literal_fits(dtype, value):
    """Raise unless literal ``value`` may become an element of UDT ``dtype``, by its type.

    A literal becomes an element of the UDT when nothing else types it: beside
    a user-defined op, as an IndexUnaryOp thunk, or as an ``ewise_union``
    default, which GraphBLAS casts to the op's input type, and a UDT casts only
    to itself. Its type decides, not its values (:func:`_literal_type`): a
    Python number or a sequence of them is weak, so ``1`` and ``(1, 2, 3)``
    beside ``INT8[3]`` are ``INT8[3]`` and fit, ``0.5`` is ``FP64[3]`` and does
    not, and ``300`` is ``OverflowError``; a numpy value is strong and fits when
    each leaf casts safely into the UDT's (``np.can_cast``), so ``np.int8(1)``
    fits ``INT8[3]`` and ``np.int64(1)`` does not. ``Scalar.from_value`` then
    converts it, which changes no value. A literal that does not fit by type but
    converts exactly still converts, with a DeprecationWarning
    (:func:`_refuse_unless_exact`).
    """
    literal_type = _literal_type(dtype, value)
    if literal_type is not None and not _casts_safely_into(literal_type, dtype):
        _refuse_unless_exact(
            dtype,
            value,
            f"{value!r} does not fit {dtype}: it is typed as {literal_type}, and a literal "
            "becomes an element of a UDT only when its type is that UDT or casts into it "
            "safely, field by field (a Python number or sequence takes a field's dtype when "
            "the field's kind holds it).",
            "Write it in the element's types, such as False rather than 0 for a bool field "
            "and 2 rather than 2.0 for an integer field.",
        )
        return
    # A weak float fits a float32 field by type, but a finite one too large for
    # it would become infinity; numpy warns as it converts, on numpy 1 and 2.
    np_type = dtype.np_type
    if np_type.names is not None:
        target = np.zeros(1, dtype=np_type)
        if isinstance(value, dict):
            value = _dict_to_record(np_type, value)
    else:
        # An array's elements, or the one value of a UDT that is neither (bytes).
        base, shape = np_type.subdtype or (np_type, ())
        target = np.zeros(shape, dtype=base)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            target[...] = value
        except RuntimeWarning:
            raise OverflowError(
                f"{value!r} overflows {dtype}: a literal that becomes an element of a UDT "
                "must not turn a finite number into infinity."
            ) from None
        except (TypeError, ValueError):
            pass  # Scalar.from_value reports these.


def _check_scalar_fits(dtype, scalar):
    """Raise unless a typed Scalar casts safely into UDT ``dtype``, as a literal must.

    An ``ewise_union`` default must have the op's input type, so it is
    converted; the same rule as for a literal (:func:`_check_literal_fits`).
    """
    if (
        dtype is not None
        and scalar.dtype != dtype
        and not scalar._is_empty
        and not _casts_safely_into(scalar.dtype, dtype)
    ):
        _refuse_unless_exact(
            dtype,
            scalar.value,
            f"a Scalar of type {scalar.dtype} does not fit {dtype}: it is used where only "
            "that UDT can go, and casts into it only when every field casts safely.",
            "Convert it first with .dup(dtype=...).",
        )


def _refuse_unless_exact(dtype, value, reason, advice):
    """Raise ValueError ``reason``, or only warn if ``value`` converts into ``dtype`` exactly.

    MAINT 2026-10-07: a literal or Scalar that must become an element of a UDT
    must fit it by type since #591. Before, any value was converted and what did
    not fit was lost; one that converts exactly, such as ``0`` for a bool field
    or ``2.0`` for an integer field, still converts, with a DeprecationWarning.
    Eight months after the release that adds the warning (the deprecation policy
    in docs/getting_started/faq.rst), always raise here and delete
    ``_converts_exactly`` and ``_object_layout``.
    """
    if not _converts_exactly(dtype, value):
        raise ValueError(reason)
    utils._warn_from_caller(
        f"{reason} Its values convert exactly, so it is converted for now; this is "
        f"deprecated and will raise ValueError in a future version. {advice}",
        DeprecationWarning,
    )


def _converts_exactly(dtype, value):
    """Whether converting ``value`` into UDT ``dtype`` keeps every number in it.

    A float may round to a narrower float, as numpy 2 rounds a Python number
    beside float32, but may not become infinite or lose an imaginary part. A
    leaf that holds no number (bytes, a date, a duration) keeps a value only
    if it converts back to it: not a number (``1.5`` would become ``b"1.5"``),
    a string cut short, or a duration in a coarser unit (5 ms is 0 s).
    """
    np_type = dtype.np_type
    try:
        if np_type.names is not None:
            if isinstance(value, dict):
                value = _dict_to_record(np_type, value)
            given = np.zeros(1, dtype=_object_layout(np_type))
            converted = np.zeros(1, dtype=np_type)
        else:
            base, shape = np_type.subdtype or (np_type, ())
            given = np.empty(shape, dtype=object)
            converted = np.zeros(shape, dtype=base)
        given[...] = value
        with warnings.catch_warnings():
            # numpy warns for some lossy conversions; the comparison below finds each.
            warnings.simplefilter("ignore")
            converted[...] = _view_if_same_layout(value, converted.dtype)
    except (TypeError, ValueError, OverflowError):
        return False
    for given_leaf, converted_leaf in zip(
        _leaf_arrays(given), _leaf_arrays(converted), strict=True
    ):
        kind = converted_leaf.dtype.kind
        if kind not in "biufc":
            for x, y in zip(given_leaf.ravel().tolist(), converted_leaf.ravel(), strict=True):
                if _is_number(x) or not _converts_back(x, y):
                    return False
            continue
        for x, y in zip(given_leaf.ravel().tolist(), converted_leaf.ravel().tolist(), strict=True):
            if kind in "fc":
                z = complex(x)
                if (kind == "f" and z.imag != 0) or (np.isfinite(z) and not np.isfinite(y)):
                    return False
            elif x != y:
                return False
    return True


def _is_number(x):
    """A Python or numpy number or bool; not a timedelta, which numpy counts as an integer."""
    return isinstance(x, (numbers.Number, np.bool_)) and not isinstance(x, np.timedelta64)


def _keeps_value(value, dtype):
    """Whether ``value`` converts into ``dtype`` and back unchanged (:func:`_converts_back`)."""
    try:
        converted = np.array(value, dtype=dtype)[()]
    except (TypeError, ValueError, OverflowError):
        return False
    return _converts_back(value, converted)


def _converts_back(given, converted):
    """Whether ``converted``, ``given`` as text, a date or a duration, converts back to ``given``.

    numpy compares dates and durations across units exactly, so 0 s is not 5 ms.
    A date given as text is compared as a date.
    """
    given, converted = np.asarray(given), np.asarray(converted)
    try:
        back = converted.astype(given.dtype)
    except (TypeError, ValueError, RuntimeError):
        try:
            back, given = converted, given.astype(converted.dtype)
        except (TypeError, ValueError, RuntimeError):
            return False
    if given.dtype.kind in "mM" and np.isnat(given):
        return bool(np.isnat(back))
    return bool(np.all(back == given))


def _object_layout(np_type):
    """Return ``np_type`` with every leaf an object, keeping field names, nesting and shapes."""
    if np_type.names is not None:
        return np.dtype([(name, _object_layout(np_type.fields[name][0])) for name in np_type.names])
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        return np.dtype((_object_layout(base), shape))
    return np.dtype(object)


# The kinds of element a weak Python literal fits without changing kind, and
# the dtype it takes otherwise: numpy 2's rules for Python scalars (NEP 50).
_WEAK_LITERAL_FITS = {"b": "biufc", "i": "iufc", "f": "fc", "c": "c"}
_WEAK_LITERAL_DEFAULT = {
    "b": np.dtype(np.bool_),
    "i": np.dtype(np.int64),
    "f": np.dtype(np.float64),
    "c": np.dtype(np.complex128),
}


@lru_cache
def _int_bounds(elem):
    info = np.iinfo(elem)
    return info.min, info.max


def _weak_literal_element(value, kind, elem):
    """Return the dtype a Python literal of ``kind`` takes beside an element of ``elem``.

    It is ``elem`` when that kind of element holds the literal (``1`` beside
    int8, ``0.5`` beside float32), else the literal's default dtype (``0.5``
    beside int8 is float64), and complex of ``elem``'s precision for a complex
    literal beside a float. An int literal outside an integer element's range
    raises ``OverflowError``, as numpy 2 does, rather than wrapping.
    """
    if elem.kind in _WEAK_LITERAL_FITS[kind]:
        if kind == "i" and elem.kind in "iu":
            lo, hi = _int_bounds(elem)
            if not lo <= value <= hi:
                raise OverflowError(f"Python integer {value} out of bounds for {elem}")
        return elem
    if kind == "c" and elem.kind == "f":
        return np.result_type(elem, np.complex64)
    rv = _WEAK_LITERAL_DEFAULT[kind]
    if kind == "i":  # beside a bool, an int is INT64, and must fit it
        lo, hi = _int_bounds(rv)
        if not lo <= value <= hi:
            raise OverflowError(f"Python integer {value} out of bounds for {rv}")
    return rv


def _weak_literal_leaf(values, kind, elem, *, exact=False):
    """Return the one dtype Python numbers ``values`` of widest kind ``kind`` take beside ``elem``.

    Each is typed by :func:`_weak_literal_element`, so an int out of an integer
    ``elem``'s range raises ``OverflowError``. With ``exact``, for a comparison,
    which keeps no result type, a fill value or truediv, such an int takes
    int64, uint64 or float64, whichever holds the values (a comparison then
    compares it as infinity: :func:`_literal_operand`). Built-in dtypes and
    UDT leaves both use this.
    """
    if exact and kind == "i" and elem.kind in "biu":
        # The integer type an int would take: ``elem``, or INT64 beside a bool.
        lo, hi = _int_bounds(elem if elem.kind != "b" else _WEAK_LITERAL_DEFAULT["i"])
        for x in values:
            if not lo <= x <= hi:
                if all(-(2**63) <= x < 2**63 for x in values):
                    return np.dtype(np.int64)
                if all(0 <= x < 2**64 for x in values):
                    return np.dtype(np.uint64)
                return np.dtype(np.float64)
    for x in values:
        elem_type = _weak_literal_element(x, kind, elem)
    return elem_type


_LITERAL_KINDS = {bool: "b", int: "i", float: "f", complex: "c"}


class _LiteralDoesNotFit(Exception):
    """A tuple or list literal is not laid out like an element of the UDT."""


def _as_python_number(value):
    """Return a ``Fraction`` or ``Decimal`` as the int or float it stands for; anything else as is.

    numpy would convert either into an integer field by truncating it (``Fraction(1, 2)``
    becomes 0), so they are typed as the Python number they are worth instead.
    """
    from decimal import Decimal
    from fractions import Fraction

    if isinstance(value, (Fraction, Decimal)):
        try:
            return int(value) if value == int(value) else float(value)
        except (OverflowError, ValueError):  # infinite or NaN
            return float(value)
    return value


def _literal_leaf_values(value, shape, *, numpy_scalars=False):
    """Return the Python numbers in literal ``value`` read as an array of ``shape``, or ``None``.

    ``shape`` is ``()`` for a scalar leaf. A number fills any shape, as numpy
    broadcasts it. A sequence that does not fill the shape, or holds anything
    that is not a Python number (or, with ``numpy_scalars``, a numpy number or
    bool), gives ``None``.
    """
    if shape:
        arr = np.empty(shape, dtype=object)
        try:
            arr[...] = value
        except (TypeError, ValueError):
            return None
        values = [_as_python_number(x) for x in arr.ravel().tolist()]
    else:
        values = [_as_python_number(value)]
    numpy_types = (np.number, np.bool_) if numpy_scalars else ()
    if all(type(x) in _LITERAL_KINDS or isinstance(x, numpy_types) for x in values):
        return values
    return None


def _weak_literal_udt(dtype, value, *, exact=False):
    """Return the UDT a Python number, or a tuple or list of them, becomes beside UDT ``dtype``.

    The literal is read as an element of ``dtype``: a number stands in every
    leaf, a sequence fills an array UDT or an array field, and a tuple or list
    fills a record by position, nested for a nested record. Each leaf then takes
    the dtype numpy 2 gives a Python scalar beside that leaf's element
    (:func:`_weak_literal_element`), of the widest kind among the leaf's
    numbers, so the pair promotes as two UDTs do: ``int8_udt + 1`` and
    ``int8_udt + (1, 2, 3)`` stay ``int8_udt``, while ``int8_udt * 2.5`` and
    ``int8_udt + (0.5, 1.5, 2.5)`` give float64 elements. A leaf that is not a
    number (bytes, datetime) takes the literal's default dtype and fails to
    pair. With ``exact``, for ``eq`` and ``ne``, an int out of a leaf's range
    takes a type that holds it instead of raising (:func:`_weak_literal_leaf`),
    so ``int8_udt == 300`` is False. In a record, a numpy number or bool is
    strong, as beside a built-in type: ``int8_record + (np.float64(1.5), 1)``
    has a float64 field. (A sequence for an array UDT is the array numpy makes
    of it instead: :func:`_literal_type`.) A sequence not laid out like an
    element gives ``None``.
    """

    def leaf_dtype(leaf, part, *, in_record=False):
        base, shape = leaf.subdtype or (leaf, ())
        if in_record and not shape and base.kind in "SUMm":
            # Text or a date in such a field is typed too, so the number fields
            # beside it are. It takes the field's own type when it converts into
            # the field and back unchanged, so ops pair the field with itself;
            # otherwise its own type, which does not fit: text as the field's
            # kind at its own length, a numpy date or duration in its unit.
            # Other dates (datetime, str) take the field's type.
            if isinstance(part, (str, bytes)) and base.kind in "SU":
                own = np.dtype(f"{base.kind}{max(len(part), 1)}")
            elif isinstance(part, (np.datetime64, np.timedelta64)):
                own = part.dtype
                if own.kind != base.kind:
                    return own
            elif isinstance(part, (str, bytes, datetime.date, datetime.timedelta)):
                return base
            else:
                own = None
            if own is not None:
                return base if _keeps_value(part, base) else own
        values = _literal_leaf_values(part, shape, numpy_scalars=in_record)
        if values is None:
            raise _LiteralDoesNotFit
        weak = [x for x in values if type(x) in _LITERAL_KINDS]
        strong = [x.dtype for x in values if type(x) not in _LITERAL_KINDS]
        elem = None
        if weak:
            kind = max((_LITERAL_KINDS[type(x)] for x in weak), key="bifc".index)
            if base.kind in "biufc":
                elem = _weak_literal_leaf(weak, kind, base, exact=exact)
            else:
                elem = _WEAK_LITERAL_DEFAULT[kind]
        if strong:
            elem = np.result_type(*strong, *([] if elem is None else [elem]))
        return np.dtype((elem, shape)) if shape else elem

    def record_leaves(np_type, part):
        if type(_as_python_number(part)) in _LITERAL_KINDS or isinstance(
            part, (np.number, np.bool_)
        ):
            parts = [part] * len(np_type.names)
        elif isinstance(part, (tuple, list)) and len(part) == len(np_type.names):
            parts = part
        else:
            raise _LiteralDoesNotFit
        for name, item in zip(np_type.names, parts, strict=True):
            field = np_type.fields[name][0]
            if field.names is not None:
                yield from record_leaves(field, item)
            else:
                yield leaf_dtype(field, item, in_record=True)

    np_type = dtype.np_type
    try:
        if np_type.names is not None:
            literal_type = _promoted_record([np_type], record_leaves(np_type, value))
        else:
            literal_type = leaf_dtype(np_type, value)
    except _LiteralDoesNotFit:
        return None
    return dtype if literal_type == np_type else lookup_dtype(literal_type)


def _int_beyond_leaf(np_type, value):
    """Return a Python int in literal ``value`` beyond its integer leaf's range, or ``None``.

    ``value`` is laid out like an element of ``np_type``, as
    :func:`_weak_literal_udt` reads it (a record's dict already a tuple). No
    element equals a literal with such an int, whatever its other leaves.
    """
    if np_type.names is not None:
        parts = value if isinstance(value, (tuple, list)) else [value] * len(np_type.names)
        for name, part in zip(np_type.names, parts, strict=True):
            found = _int_beyond_leaf(np_type.fields[name][0], part)
            if found is not None:
                return found
        return None
    base, shape = np_type.subdtype or (np_type, ())
    if base.kind in "iu":
        lo, hi = _int_bounds(base)
        for x in _literal_leaf_values(value, shape) or ():
            if type(x) is int and not lo <= x <= hi:
                return x
    return None


def _beyond_as_infinity(np_type, value):
    """Return literal ``value`` with each int beyond its integer leaf's range as infinity.

    ``value`` is laid out as :func:`_int_beyond_leaf` reads it. Infinity of the
    int's sign equals no element, as the int does; every other value is kept,
    so a text leaf still pairs with its field.
    """
    if np_type.names is not None:
        parts = value if isinstance(value, (tuple, list)) else [value] * len(np_type.names)
        return tuple(
            _beyond_as_infinity(np_type.fields[name][0], part)
            for name, part in zip(np_type.names, parts, strict=True)
        )
    base, shape = np_type.subdtype or (np_type, ())
    values = _literal_leaf_values(value, shape) if base.kind in "iu" else None
    if values is None:
        return value
    lo, hi = _int_bounds(base)
    values = [
        (math.inf if x > 0 else -math.inf) if type(x) is int and not lo <= x <= hi else x
        for x in values
    ]
    return np.array(values, dtype=object).reshape(shape).tolist() if shape else values[0]


def _array_literal_udt(dtype, value):
    """Return the UDT a numpy array ``value`` is beside UDT ``dtype``, or ``None`` to convert it.

    An array has a dtype, so it is strong, as numpy 2 treats it:
    ``int8_udt + np.array([0.5, 1, 2])`` has float64 elements and
    ``+ np.array([1, 2, 3])`` int64 ones. Its type is the array layout of its
    dtype, in its own shape when that pairs with the UDT's and in the UDT's
    shape when it broadcasts to that (``np.array([1])`` beside ``INT8[3]``).
    A 0-d structured array or ``np.void`` beside a record is its record dtype.
    Anything else, such as a numeric array beside a record, is converted into
    the UDT.
    """
    np_type = dtype.np_type
    if value.dtype.names is not None:
        return lookup_dtype(value.dtype) if value.ndim == 0 else None
    # A bool array converts as it is (a BOOL array UDT cannot be registered),
    # and bool casts safely into every numeric element.
    if np_type.names is not None or value.dtype.kind not in "iufc":
        return None
    shape = np_type.subdtype[1]
    if _strip_leading_ones(value.shape) == _strip_leading_ones(shape):
        shape = value.shape
    else:
        try:
            np.broadcast_to(value, shape)
        except ValueError:
            return None
    return lookup_dtype(np.dtype((value.dtype, shape)))

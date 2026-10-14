"""The rules for user-defined types (UDTs), apart from generating code for them.

GraphBLAS knows a UDT only by its size: it cannot cast one to or from any other
type, and it copies an element of one as raw bytes. The rules here fill that
gap, in this order:

- which built-in ops lift to UDTs (``BUILTIN_UDT_BINARY_OPS``), and the layout
  of an element (:func:`_iter_record_leaves`);
- which operands an op on UDTs takes and the type of its result
  (:func:`_check_udt_pair`, :func:`_udt_op_types`);
- what may be stored in an object of another type, and how it is converted on
  the way (:func:`_store_cast_op`);
- when ``ewise_add`` would need GraphBLAS to cast a UDT;
- how ``isequal``, and ``==`` without Numba, compare elements;
- how an element's value becomes bytes and back, for ``Scalar.value``.

``operator.udt_codegen`` generates the operators that compute on UDTs,
``operator.udf`` wraps a user's function on them, and ``literals`` types a
literal operand beside one.

``operator.udt_codegen`` imports this module while the operator package is still
loading, so a function here that needs an operator, ``base`` or ``scalar``
imports it inside the function, once it knows a UDT is involved.
"""

import copy
from functools import lru_cache, reduce
from operator import mul

import numpy as np

from ..exceptions import DomainMismatch
from . import _has_numba, ffi, utils
from .dtypes import _registry, _view_if_same_layout, lookup_dtype
from .utils import output_type

# The built-in ops that lift to UDTs, each with the Python operator or function
# the generated code applies.
BUILTIN_UDT_BINARY_OPS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "truediv": "/",
    "floordiv": "//",
    "min": "min",
    "max": "max",
}
BUILTIN_UDT_UNARY_OPS = {
    "ainv": "-",
    "abs": "abs",
}


# Ops with no meaning on complex fields: no ordering for min and max, and no
# integer mod for floordiv.
_OPS_NOT_FOR_COMPLEX = frozenset({"min", "max", "floordiv"})


# The layout of an element.


def _get_udt_info(dtype):
    """Return ('record', field_names) or ('array', (base_dtype, flat_size)) or None."""
    np_type = dtype.np_type
    if np_type.names is not None:
        return "record", np_type.names
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        return "array", (base, reduce(mul, shape))
    return None


def _iter_record_leaves(np_type, python_prefix="", c_prefix=""):
    """Yield ``(python_access, c_access, leaf_dtype)`` for each leaf in a record.

    A non-nested record yields one entry per field, with paths like
    ``"['a']"`` (Python) and ``"a"`` (C struct). A nested record recurses
    into structured fields: a top-level field ``"outer"`` whose dtype is
    itself a struct with field ``"inner_a"`` yields
    ``("['outer']['inner_a']", "outer.inner_a", float64_dtype)``.

    Only record-in-record nesting is recognised. Array-typed fields are
    yielded as a single leaf, which the Numba path handles: the generated
    expression operates on the whole sub-array and the wrapper slice-assigns
    it back. The JIT path never sees such a record, because a subarray dtype
    has no entry in ``udt_codegen.NP_TO_C_TYPES`` and ``udt_codegen._udt_c_typedef``
    bails out, so the op keeps the cfunc for every call.
    """
    for name in np_type.names:
        field_dtype = np_type.fields[name][0]
        py_path = f"{python_prefix}[{name!r}]"
        c_path = f"{c_prefix}{name}" if not c_prefix else f"{c_prefix}.{name}"
        if field_dtype.names is not None:
            yield from _iter_record_leaves(field_dtype, py_path, c_path)
        else:
            yield py_path, c_path, field_dtype


def _strip_leading_ones(shape):
    i = 0
    while i < len(shape) and shape[i] == 1:
        i += 1
    return shape[i:]


# Which operands an op on UDTs takes, and the type of its result.


def _check_udt_pair(op_name, dtype, dtype2, info_x, info_y):
    """Raise ``KeyError`` if two UDT operands disagree on shape.

    ``info_x`` and ``info_y`` are ``_get_udt_info`` results. When either is
    ``None`` (one side is a scalar broadcast), no check fires. Without this
    gate the generated wrapper code raises a cryptic Numba ``TypingError``
    on the first field access, or, for array fields that do not broadcast,
    fails inside the cfunc without telling the caller. Element dtypes may
    differ; :func:`_udt_op_types` promotes them.
    """
    if info_x is None or info_y is None:
        return
    kind_x, detail_x = info_x
    kind_y, detail_y = info_y
    if kind_x != kind_y:
        raise KeyError(
            f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
            f"cannot mix record and array UDTs in a single element-wise op."
        )
    if kind_x == "record":
        if detail_x != detail_y:
            raise KeyError(
                f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                f"record UDTs must share field names, in the same order; got "
                f"{list(detail_x)} vs {list(detail_y)}."
            )
        # Matching top-level names is not enough. The codegen reads both
        # operands at the left one's leaf paths, so a field that is a
        # sub-record on one side and a scalar on the other, or sub-records
        # with different names, fails Numba's typing: a compile error
        # reported for what is really the same shape disagreement the check
        # above reports as a KeyError. The paths must also come in the same
        # order, as the top-level names must, so that leaf ``i`` of one
        # operand is leaf ``i`` of the other.
        leaves_x = list(_iter_record_leaves(dtype.np_type))
        leaves_y = list(_iter_record_leaves(dtype2.np_type))
        if [py for py, _c, _d in leaves_x] != [py for py, _c, _d in leaves_y]:
            raise KeyError(
                f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                f"record UDTs must nest the same way, with the same field names in the "
                f"same order at every level; got {[c for _py, c, _d in leaves_x]} vs "
                f"{[c for _py, c, _d in leaves_y]}."
            )
        # Each field must also have the same shape in both (field dtypes may
        # differ), as numpy requires before it promotes or compares two record
        # dtypes. Shapes that do not broadcast would raise inside the cfunc,
        # unseen, and leave whatever was in the output buffer. Shapes that do
        # broadcast would work one way round only, since the arithmetic ops
        # write into a field shaped like the left operand's: ``(3,)`` plus
        # ``(1,)`` works, ``(1,)`` plus ``(3,)`` does not.
        for (_py, c, leaf_x), (_py_y, _c_y, leaf_y) in zip(leaves_x, leaves_y, strict=True):
            if leaf_x.shape != leaf_y.shape:
                raise KeyError(
                    f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                    f"record UDTs must have the same shape for each field; field {c} is "
                    f"{leaf_x.shape} vs {leaf_y.shape}."
                )
        return
    # Array UDTs broadcast as numpy arrays do, so ``(3, 1)`` with ``(1, 4)``
    # gives ``(3, 4)``; the generated code reads each operand through a flat
    # index map (``udt_codegen._array_operand_source``). Shapes such as ``(2, 3)`` with
    # ``(3, 2)`` do not broadcast, as numpy refuses them.
    (base_x, shape_x), (base_y, shape_y) = dtype.np_type.subdtype, dtype2.np_type.subdtype
    try:
        np.broadcast_shapes(shape_x, shape_y)
    except ValueError:
        raise KeyError(
            f"binary.{op_name} does not work with ({dtype}, {dtype2}): array UDTs must "
            f"have shapes that broadcast together; got {base_x}{list(shape_x)} vs "
            f"{base_y}{list(shape_y)}."
        ) from None


def _builtin_dtype(np_type):
    """Return the built-in ``DataType`` for ``np_type``, or ``None`` if it is not one."""
    dt = _registry.get(np_type)
    return None if dt is None or dt._is_udt else dt


def _complex_op_error(op_name, dtype, dtype2):
    return KeyError(
        f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
        f"this op is not defined on complex fields. Use ``binary.plus``, "
        f"``minus``, ``times``, or ``truediv`` for complex element-wise "
        f"arithmetic, or register a custom binary op."
    )


def _element_types(op_name, dtype, dtype2, elem_x, elem_y):
    """Return ``(in_x, in_y, out)``, the numpy dtypes ``binary.{op_name}`` uses on one element.

    They come from the op's typing on the built-in dtypes, so each element of an
    array UDT, or each record leaf, is promoted as a Vector of that dtype would
    be, and ``truediv`` on integers gives floats. ``in_x`` and ``in_y`` are the
    types GraphBLAS casts the operands to before it applies the op. The
    generated code casts to them too, because Numba promotes differently: it
    adds uint64 and int64 in int64, where GraphBLAS uses float64. ``dtype`` and
    ``dtype2`` are the operands, named in errors. Two elements that are not
    both built-in dtypes (bytes, datetime, records inside an array field) have
    no promotion rule, so they must be the same dtype, which the result keeps.
    """
    # ``min``/``max``/``floordiv`` have no defined semantics on complex
    # operands (no ordering, no integer mod). Reject early with a clear
    # message; otherwise Numba's cfunc compile blows up several frames
    # down with ``NotImplementedError: No definition for lowering lt``.
    if op_name in _OPS_NOT_FOR_COMPLEX and "c" in (elem_x.kind, elem_y.kind):
        raise _complex_op_error(op_name, dtype, dtype2)
    builtin_x = _builtin_dtype(elem_x)
    builtin_y = _builtin_dtype(elem_y)
    if builtin_x is None or builtin_y is None:
        if elem_x == elem_y:
            return elem_x, elem_y, elem_x
        # Builds without complex dtypes (the vanilla backend) still lift complex
        # fields, so promote them as the built-in typing does where it has them.
        if "c" in (elem_x.kind, elem_y.kind) and {elem_x.kind, elem_y.kind} <= set("biufc"):
            promoted = np.promote_types(elem_x, elem_y)
            return promoted, promoted, promoted
        raise KeyError(
            f"binary.{op_name} does not work with ({dtype}, {dtype2}): elements of "
            f"{elem_x} and {elem_y} have no common type."
        )
    from .. import binary
    from .operator import get_typed_op

    typed = get_typed_op(getattr(binary, op_name), builtin_x, builtin_y)
    return typed.type.np_type, typed.type2.np_type, typed.return_type.np_type


def _udt_op_types(op_name, dtype, dtype2):
    """Return ``(ret_type, input_types)`` for ``binary.{op_name}`` on ``dtype`` and ``dtype2``.

    At least one operand is a record or array UDT, and ``_check_udt_pair`` has
    accepted the pair; the other may be a plain scalar type. Each element, or
    record leaf, follows :func:`_element_types`, so the result never narrows:
    an int64 array UDT plus ``0.5`` is ``FP64[3]``. ``ret_type`` is an
    operand's own type when that holds the result, so same-type arithmetic
    keeps its UDT. Otherwise it is the promoted layout, which is the UDT a user
    registered with that layout if there is one, else an anonymous one.
    ``input_types`` holds the ``(in_x, in_y)`` pair of each record leaf, in
    leaf order, or the single pair for an array UDT's elements.
    """
    np_x, np_y = dtype.np_type, dtype2.np_type
    if np_x.names is not None or np_y.names is not None:
        leaves_x = list(_iter_record_leaves(np_x)) if np_x.names is not None else None
        leaves_y = list(_iter_record_leaves(np_y)) if np_y.names is not None else None
        input_types = []
        result_leaves = []
        for i, (_py, _c, leaf) in enumerate(leaves_x or leaves_y):
            elem_x = np_x if leaves_x is None else leaves_x[i][2].base
            elem_y = np_y if leaves_y is None else leaves_y[i][2].base
            in_x, in_y, elem = _element_types(op_name, dtype, dtype2, elem_x, elem_y)
            input_types.append((in_x, in_y))
            result_leaves.append(np.dtype((elem, leaf.shape)) if leaf.shape else elem)
        for operand, leaves in [(dtype, leaves_x), (dtype2, leaves_y)]:
            if leaves is not None and all(
                result_leaf == leaf
                for result_leaf, (_py, _c, leaf) in zip(result_leaves, leaves, strict=True)
            ):
                return operand, input_types
        records = [np_type for np_type in (np_x, np_y) if np_type.names is not None]
        return lookup_dtype(_promoted_record(records, iter(result_leaves))), input_types
    base_x, shape_x = np_x.subdtype or (np_x, ())
    base_y, shape_y = np_y.subdtype or (np_y, ())
    in_x, in_y, elem = _element_types(op_name, dtype, dtype2, base_x, base_y)
    result = np.dtype((elem, np.broadcast_shapes(shape_x, shape_y)))
    for operand in (dtype, dtype2):
        if operand.np_type == result:
            return operand, [(in_x, in_y)]
    return lookup_dtype(result), [(in_x, in_y)]


def _promoted_record(records, result_leaves):
    """Build a record laid out like ``records`` with ``result_leaves`` for its leaves.

    ``records`` holds one or two record dtypes with the same field names and
    nesting. The new record aligns its fields, at each level, if any of them
    does there, so a C-compatible operand gives a C-compatible result.
    """
    first = records[0]
    fields = []
    for name in first.names:
        field = first.fields[name][0]
        if field.names is not None:
            field = _promoted_record([r.fields[name][0] for r in records], result_leaves)
        else:
            field = next(result_leaves)
        fields.append((name, field))
    return np.dtype(fields, align=any(r.isalignedstruct for r in records))


# Storing a value of one type in an object of another.


def _corresponding_leaves(np_x, np_z):
    """Return ``(reason, pairs)`` for putting an element of ``np_x`` into one of ``np_z``.

    ``reason`` says why the two layouts do not correspond, or is ``None``, and
    then ``pairs`` holds the base dtypes of each pair of leaves. Records must
    line up as the operands of a lifted op do (:func:`_check_udt_pair`): the
    same field names, order and nesting and the same shape for each field.
    Arrays must have the same shape apart from leading axes of length 1;
    operands broadcast, but a stored element keeps every value in its place.
    numpy casts records by position whatever their names, and repeats or
    drops elements of an array field of another length, which would put
    values in places nobody chose.
    """
    if (np_x.names is None) != (np_z.names is None):
        return "a record UDT and an array UDT do not correspond", None
    if np_x.names is None:
        if np_x.subdtype is None or np_z.subdtype is None:
            return "only record and array UDTs cast", None
        (base_x, shape_x), (base_z, shape_z) = np_x.subdtype, np_z.subdtype
        if _strip_leading_ones(shape_x) != _strip_leading_ones(shape_z):
            return "array UDTs must have the same shape, apart from leading axes of length 1", None
        return None, [(base_x, base_z)]
    leaves_x = list(_iter_record_leaves(np_x))
    leaves_z = list(_iter_record_leaves(np_z))
    if [py for py, _c, _d in leaves_x] != [py for py, _c, _d in leaves_z]:
        return "records must have the same field names, in the same order, at every level", None
    pairs = [(x, z) for (_, _, x), (_, _, z) in zip(leaves_x, leaves_z, strict=True)]
    if any(x.shape != z.shape for x, z in pairs):
        return "records must have the same shape for each field", None
    return None, [(x.base, z.base) for x, z in pairs]


def _udt_cast_error(dtype, to_dtype):
    """Return why an element of UDT ``dtype`` cannot be stored as UDT ``to_dtype``, or ``None``.

    The layouts must correspond (:func:`_corresponding_leaves`), and each pair
    of elements must be built-in numbers, or the same type.
    """
    reason, pairs = _corresponding_leaves(dtype.np_type, to_dtype.np_type)
    for x, z in pairs or ():
        if x != z and not (x.kind in "biufc" and z.kind in "biufc"):
            return f"elements of {x} do not cast to {z}"
    return reason


def _leaf_dtypes(np_type):
    """Yield the base dtype of each leaf of ``np_type``: a record's fields, an array's element."""
    if np_type.names is not None:
        for _py, _c, leaf in _iter_record_leaves(np_type):
            yield leaf.base
    else:
        yield np_type.base


def _casts_safely_into(literal_type, dtype):
    """Whether each leaf of ``literal_type`` casts safely (``np.can_cast``) into UDT ``dtype``'s.

    A scalar type fills every leaf; otherwise the layouts must correspond
    (:func:`_corresponding_leaves`).
    """
    if literal_type == dtype:
        return True
    np_from, np_to = literal_type.np_type, dtype.np_type
    if np_from.names is None and np_from.subdtype is None:
        return all(np.can_cast(np_from, leaf, "safe") for leaf in _leaf_dtypes(np_to))
    reason, pairs = _corresponding_leaves(np_from, np_to)
    return reason is None and all(np.can_cast(x, z, "safe") for x, z in pairs)


def _store_cast_op(result_dtype, output_dtype):
    """Return the UnaryOp that converts a result for an object of ``output_dtype``, or ``None``.

    GraphBLAS casts between built-in types on store (C's casts, with a float
    saturating into an integer type), but it cannot cast to or from a UDT;
    left to GraphBLAS, that is a bare GrB_DOMAIN_MISMATCH or, for a single
    element, bytes read as the wrong type. So when a UDT is involved, the
    result is stored by applying the op returned here, which casts each
    element or record leaf the way GraphBLAS casts built-in types, between
    UDTs whose layouts correspond (:func:`_udt_cast_error`). Any other
    pair raises ``DomainMismatch``. ``None`` means GraphBLAS stores the result
    as it is. This is the one place that decides what may be stored where.
    """
    if result_dtype is output_dtype or not (result_dtype._is_udt or output_dtype._is_udt):
        return None
    reason = "UDTs do not cast"
    if _has_numba and result_dtype._is_udt and output_dtype._is_udt:
        if (reason := _udt_cast_error(result_dtype, output_dtype)) is None:
            from .operator.unary import _udt_cast_op

            return _udt_cast_op(result_dtype, output_dtype)
    elif _has_numba:
        reason = "a UDT casts only to another UDT"
    raise DomainMismatch(
        f"cannot store a result of type {result_dtype} in an object of type "
        f"{output_dtype}: {reason}. Use .new() for an object of the result's type, or "
        "compute the result in the type you want to store."
    )


def _fused_store_expr(expr, output_dtype):
    """Return ``expr`` computing straight into ``output_dtype``, or raise ``DomainMismatch``.

    A result of another UDT type is stored by the op that computes it, which
    casts each element as it writes it (``udt_codegen.fused_store_op``), so no
    temporary of the result's own type is made: data GraphBLAS works on is
    often too large to hold twice. That works for an element-wise
    computation by a built-in op that lifts to UDTs (``apply``,
    ``ewise_mult``, ``ewise_union``). ``ewise_add`` copies an entry present in
    only one input without the op, and a semiring or reduction would add in
    the narrower type, so those and any other computation raise.
    """
    from .operator.udt_codegen import fused_store_op  # needs Numba

    op, cfunc_name = expr.op, expr.cfunc_name
    if op is not None and op.opclass == "Monoid":
        # Element-wise, a Monoid is its BinaryOp.
        op, cfunc_name = op.binaryop, cfunc_name.replace("_Monoid", "_BinaryOp")
    fused = None
    if any(k in cfunc_name for k in ("_apply", "eWiseMult", "eWiseUnion")):
        fused = fused_store_op(op, output_dtype)
    if fused is None:
        method = "extraction" if expr.method_name == "__getitem__" else expr.method_name
        what = method if op is None else f"{method} with {op.name}"
        raise DomainMismatch(
            f"cannot store a result of type {expr.dtype} in an object of type "
            f"{output_dtype}: GraphBLAS cannot cast a UDT, and only a built-in op that lifts "
            "to UDTs, in apply, ewise_mult or ewise_union, can write another type as it "
            f"computes ({what} cannot), so this would need a temporary copy of the whole "
            "result. Make it explicitly by storing the computed result, as in "
            "`obj << expr.new()`, which casts it in one more pass."
        )
    rv = copy.copy(expr)
    rv.op, rv.cfunc_name, rv.dtype = fused, cfunc_name, output_dtype
    return rv


def _scalar_accum_op(scalar, accum):
    """Return the op to accumulate into ``scalar`` with, as ``scalar.ewise_add`` runs it.

    ``__call__`` types ``accum`` on the object's dtype. For a UDT Scalar, that
    would convert a literal into the UDT (``s += 0.5``), where a Vector
    computes ``s + 0.5`` in its own type and casts as it stores; the untyped
    op does the same here.
    """
    return accum.parent if scalar.dtype._is_udt else accum


def _cast_for_store(value, dtype):
    """Return Scalar, Vector or Matrix ``value``, or its expression, to assign as ``dtype``.

    GraphBLAS's assign and setElement take a UDT value as raw bytes, so one of
    another type would be read as the wrong type; :func:`_store_cast_op`
    casts it or raises. Assign cannot cast as it writes, so a value of
    another UDT type needs a converted copy: one element for a Scalar, which
    is made here, but the whole value for a Vector or Matrix, which raises.
    """
    if (cast_op := _store_cast_op(value.dtype, dtype)) is None:
        return value
    from .base import BaseType
    from .scalar import Scalar

    if output_type(value) is not Scalar:
        raise DomainMismatch(
            f"cannot assign a value of type {value.dtype} into an object of type {dtype}: "
            "assign cannot cast a UDT as it writes, so this would need a converted copy of "
            "the whole value. Make it explicitly with .dup(dtype=...) first."
        )
    if not isinstance(value, BaseType):
        value = value.new(name="")
    return value.apply(cast_op).new(name=value.name)


# ewise_add, which copies an entry present in only one input without the op.


def _is_lifted_op(op):
    """Whether ``op`` is a built-in BinaryOp that lifts to UDTs, or a Monoid of one."""
    from .operator.binary import BinaryOp
    from .operator.monoid import Monoid

    if isinstance(op, Monoid):
        op = op.binaryop
    return type(op) is BinaryOp and not op._anonymous and op.name in BUILTIN_UDT_BINARY_OPS


def _ewise_add_needs_cast(op, left_dtype, right_dtype):
    """Whether ``ewise_add`` with ``op`` would have GraphBLAS cast to or from a UDT.

    eWiseAdd copies an entry present in only one input into the result, cast
    to the op's output type, and GraphBLAS checks that cast before it looks at
    the data. A UDT casts only to itself, so such a pair is a bare
    GrB_DOMAIN_MISMATCH even when every entry has a partner.
    """
    out = op.return_type
    if not (out._is_udt or left_dtype._is_udt or right_dtype._is_udt):
        return False
    return left_dtype != out or right_dtype != out


def _ewise_add_cast_error(op, left_dtype, right_dtype):
    """Return the ``DomainMismatch`` for an ``ewise_add`` that would need a UDT cast."""
    return DomainMismatch(
        f"ewise_add cannot use {op.parent!r} on {left_dtype} and {right_dtype}: an entry "
        f"present in only one input is copied into the result as {op.return_type}, and "
        "GraphBLAS cannot cast a UDT, so this would need converted copies of the operands. "
        "Use ewise_union, which takes a default for each side (infix + does, with zeros), or "
        "ewise_mult; or convert an operand first with .dup(dtype=...)."
    )


def _ewise_add_casts_udt(left, right, op):
    """Whether ``left.ewise_add(right, op)`` would need GraphBLAS to cast a UDT."""
    try:
        left_dtype, right_dtype = left.dtype, right.dtype
    except TypeError:
        # An infix operand that has no dtype here (``x & y`` of non-BOOL);
        # ewise_add reports it as before.
        return False
    if not (left_dtype._is_udt or right_dtype._is_udt):
        return False
    from .operator import get_typed_op

    typed_op = get_typed_op(op, left_dtype, right_dtype, kind="binary")
    return _ewise_add_needs_cast(typed_op, left_dtype, right_dtype)


def _scalar_ewise_add_converted(left, right, op):
    """Return ``left``, ``right`` and ``op`` for an ``ewise_add`` of two Scalars that needs a cast.

    ``op`` is typed on the two, and :func:`_ewise_add_needs_cast` holds:
    GraphBLAS would cast an unpaired entry to the result type, which a UDT
    cannot be. For a built-in op that lifts to UDTs, both are converted to the
    result type first, which changes no value, and the op applies in that one
    type: an int8 UDT plus 0.5 is float64, and an empty Scalar still adds as 0.
    A Scalar is one element, so the converted copies cost nothing; Vector and
    Matrix raise instead, as any other op does here (:func:`_ewise_add_cast_error`).
    """
    if not _is_lifted_op(op.parent):
        raise _ewise_add_cast_error(op, left.dtype, right.dtype)
    if left.dtype != op.return_type:
        left = _converted_to(left, op.return_type, name="")
    if right.dtype != op.return_type:
        right = _converted_to(right, op.return_type, name="")
    from .operator import get_typed_op

    return left, right, get_typed_op(op.parent, left.dtype, right.dtype, kind="binary")


def _plus_zero(dtype):
    """Return a Scalar of ``dtype`` that ``plus`` adds to any value without changing it.

    Its float and complex leaves are ``-0.0``, because ``-0.0 + x`` is ``x``
    for every float, where ``0.0 + -0.0`` is ``0.0``.
    """
    from .scalar import Scalar

    np_type = dtype.np_type
    if np_type.subdtype is not None:
        zero = np.zeros(np_type.subdtype[1], dtype=np_type.subdtype[0])
    else:
        zero = np.zeros(1, dtype=np_type)
    for leaf in _leaf_arrays(zero):
        if leaf.dtype.kind in "fc":
            leaf[...] = complex(-0.0, -0.0) if leaf.dtype.kind == "c" else -0.0
    return Scalar.from_value(zero if np_type.subdtype is not None else zero[0], dtype, name="")


def _converted_to(operand, dtype, **new_kwargs):
    """Return ``operand`` converted to UDT ``dtype``, which holds its elements, value for value.

    It adds ``dtype``'s zero on the left with the lifted ``plus``. With the
    zero on the left, the result is ``dtype`` itself even when the operand has
    the same elements in another layout (a packed and an aligned record),
    since a left operand that holds the promoted elements is the result type.
    """
    from .. import binary

    rv = operand.apply(binary.plus, left=_plus_zero(dtype)).new(**new_kwargs)
    if rv.dtype != dtype:  # pragma: no cover (safety)
        raise DomainMismatch(f"cannot convert {operand.dtype} to {dtype}")
    return rv


# isequal, and == without Numba.


def _element_shapes_differ(dtype1, dtype2):
    """Whether elements of ``dtype1`` and ``dtype2`` differ in shape, for ``isequal``.

    ``==`` on array UDTs broadcasts, as numpy's does, but ``isequal`` asks
    whether two values are the same array, as ``np.array_equal`` does: the
    shapes must match, apart from leading axes of length 1 (the rule a store
    uses), and an element of a built-in dtype has shape ``()``. Records pair
    field by field instead (:func:`_check_udt_pair`).
    """
    if dtype1 is dtype2 or not (dtype1._is_udt or dtype2._is_udt):
        return False
    shapes = []
    for np_type in (dtype1.np_type, dtype2.np_type):
        if np_type.names is not None:
            return False
        shapes.append(_strip_leading_ones(np_type.subdtype[1]) if np_type.subdtype else ())
    return shapes[0] != shapes[1]


def _literal_shape_differs(dtype, value):
    """Whether literal ``value`` differs in shape from an element of array UDT ``dtype``.

    As :func:`_element_shapes_differ`, for ``Scalar.isequal`` with a literal: a
    number has shape ``()``, so it equals an ``FP64[1]`` element but not an
    ``FP64[3]`` one, as in ``np.array_equal``. ``False`` for a record UDT.
    """
    np_type = dtype.np_type
    if np_type.subdtype is None or isinstance(value, dict):
        return False
    try:
        shape = np.shape(value)
    except ValueError:  # ragged; Scalar.from_value reports it
        return False
    return _strip_leading_ones(shape) != _strip_leading_ones(np_type.subdtype[1])


def _leaves_equal(x, x_type, y, y_type):
    """Whether values ``x`` and ``y`` are equal in every leaf, as ``binary.eq`` compares them.

    For ``==`` on a UDT Scalar without Numba. Records pair their fields in
    order (the caller checks, by ``_check_udt_pair``, that two records pair as
    for ``binary.eq``), and a value that is not a record stands in every
    field. Each pair of leaves compares in the type their dtypes promote to,
    whatever the values (numpy 1 types a scalar by its value), and broadcasts.
    """
    if x_type.names is not None or y_type.names is not None:
        x_names = x_type.names or [None] * len(y_type.names)
        y_names = y_type.names or [None] * len(x_type.names)
        return all(
            _leaves_equal(
                x if x_name is None else x[x_name],
                x_type if x_name is None else x_type.fields[x_name][0],
                y if y_name is None else y[y_name],
                y_type if y_name is None else y_type.fields[y_name][0],
            )
            for x_name, y_name in zip(x_names, y_names, strict=True)
        )
    x = np.asarray(x, dtype=x_type.base)
    y = np.asarray(y, dtype=y_type.base)
    common = np.result_type(x.dtype, y.dtype)
    return bool(np.all(x.astype(common) == y.astype(common)))


def _udt_eq_without_numba(scalar, other):
    """Return ``scalar == other`` as a bool, in numpy, for a UDT when Numba is not installed.

    ``binary.eq`` on a UDT needs Numba. ``scalar`` is nonempty, as is ``other``
    when it is a Scalar, and one of them is a UDT. A literal ``other`` is typed
    as ``binary.eq`` types it, and the values compare leaf by leaf as
    ``binary.eq`` compares them (:func:`_leaves_equal`).
    """
    from .. import binary
    from .literals import _literal_operand
    from .scalar import Scalar

    if type(other) is not Scalar:
        value, dtype = _literal_operand(scalar.dtype, other, binary.eq)
        other = Scalar.from_value(value, dtype, is_cscalar=None, name="s_eq_other")
    # Two UDTs pair as for binary.eq, or raise its KeyError (records with other
    # field names, array shapes that do not broadcast, a record and an array).
    info = _get_udt_info(scalar.dtype), _get_udt_info(other.dtype)
    _check_udt_pair("eq", scalar.dtype, other.dtype, *info)
    return _leaves_equal(scalar.value, scalar.dtype.np_type, other.value, other.dtype.np_type)


# An element's value as bytes and back, for Scalar.value.


def _element_bytes(dtype, value):
    """Return ``value`` as the bytes of an element of UDT ``dtype``, padding zeroed.

    ``value`` is anything numpy assigns to one element: a tuple or dict for a
    record, an array or sequence for an array UDT, or a number for every leaf.
    """
    np_type = dtype.np_type
    # ``_udt_bytes`` zeros a record's alignment padding, which numpy may leave
    # as junk however it writes the record.
    if np_type.subdtype is None:
        arr = np.zeros(1, dtype=np_type)
        if isinstance(value, dict) and np_type.names is not None:
            value = _dict_to_record(np_type, value)
    else:
        arr = np.zeros(np_type.subdtype[1], dtype=np_type.subdtype[0])
    value = _view_if_same_layout(value, arr.dtype)
    if _has_narrow_float_leaf(np_type) and np.geterr()["over"] == "warn":
        # A float too large for a float32 leaf is infinity, and numpy warns;
        # warn at the caller's line instead, as a built-in FP32 does, so the
        # default filter shows each place it happens.
        try:
            with np.errstate(over="raise"):
                arr[:] = value
        except FloatingPointError:
            utils._warn_from_caller(
                f"overflow encountered in cast: {value!r} as {dtype}", RuntimeWarning
            )
            with np.errstate(over="ignore"):
                arr[:] = value
    else:
        arr[:] = value
    return _udt_bytes(np_type, arr)


def _element_value(dtype, data):
    """Return the element of UDT ``dtype`` that cffi ``data`` holds: a record or an array."""
    np_type = dtype.np_type
    rv = np.array(ffi.buffer(data[0 : np_type.itemsize]))
    if np_type.subdtype is None:
        return rv.view(np_type)[0]
    base, shape = np_type.subdtype
    return rv.view(base).reshape(shape)


def _dict_to_record(np_type, d):
    """Converts e.g. ``{"x": 1, "y": 2.3}`` to ``(1, 2.3)``."""
    rv = []
    for name, (dtype, _) in np_type.fields.items():
        val = d[name]
        if dtype.names is not None and isinstance(val, dict):
            rv.append(_dict_to_record(dtype, val))
        else:
            rv.append(val)
    return tuple(rv)


def _leaf_arrays(arr):
    """Yield the array of each leaf of a structured array, in field order."""
    if arr.dtype.names is None:
        yield arr
    else:
        for name in arr.dtype.names:
            yield from _leaf_arrays(arr[name])


@lru_cache
def _has_narrow_float_leaf(np_type):
    """Whether ``np_type`` has a float leaf narrower than float64, which a float may overflow."""
    if np_type.subdtype is not None:
        return _has_narrow_float_leaf(np_type.subdtype[0])
    if np_type.names is not None:
        return any(_has_narrow_float_leaf(np_type.fields[name][0]) for name in np_type.names)
    kind, size = np_type.kind, np_type.itemsize
    return (kind == "f" and size < 8) or (kind == "c" and size < 16)


@lru_cache
def _padding_ranges(np_type):
    """The ``(start, stop)`` byte ranges of ``np_type`` that are alignment padding.

    Walks records, nested records, and array dtypes, and returns the gaps
    between the field data.
    """
    data = []
    _collect_data_ranges(np_type, 0, data)
    data.sort()
    ranges = []
    pos = 0
    for start, stop in data:
        if start > pos:
            ranges.append((pos, start))
        pos = max(pos, stop)
    if pos < np_type.itemsize:
        ranges.append((pos, np_type.itemsize))
    return tuple(ranges)


def _collect_data_ranges(np_type, offset, out):
    """Append the ``(start, stop)`` byte ranges of ``np_type`` that hold field data."""
    if np_type.subdtype is not None:
        base = np_type.subdtype[0]
        for start in range(offset, offset + np_type.itemsize, base.itemsize):
            _collect_data_ranges(base, start, out)
    elif np_type.names is not None:
        for field_type, field_offset in (np_type.fields[name][:2] for name in np_type.names):
            _collect_data_ranges(field_type, offset + field_offset, out)
    else:
        out.append((offset, offset + np_type.itemsize))


def _udt_bytes(np_type, arr):
    """The bytes of a UDT buffer, with any alignment padding zeroed.

    ``tobytes()`` flattens any-rank numpy buffers (in particular, multi-dim
    array UDTs like ``FP64[2, 3]``) to a 1-D byte buffer that cffi can copy
    into GrB_Scalar storage.

    numpy promises nothing about the padding bytes it leaves behind, and which
    way it goes has changed across releases: numpy 2.5 made ``arr[:] = record``
    a whole-item copy out of an uninitialized temporary, so even a buffer from
    ``np.zeros`` comes back with junk between the fields. A caller's own array
    can carry junk padding in too. Since the whole record is copied into the
    GrB_Scalar and read back out of ``Scalar.value``, anything that compares or
    hashes those raw bytes would otherwise see an equal-valued scalar as
    unequal.
    """
    raw = arr.tobytes()
    ranges = _padding_ranges(np_type)
    if not ranges:
        return raw
    buf = bytearray(raw)
    for start, stop in ranges:
        buf[start:stop] = bytes(stop - start)
    return bytes(buf)

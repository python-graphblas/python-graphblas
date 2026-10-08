"""Shared utilities for auto-generating UDT operator implementations.

Provides Numba wrapper generators and JIT C code generators for element-wise
operations on record UDTs (struct types) and array UDTs (e.g., FP64[3]).

Compilation is lazy: functions here are only called from ``_compile_udt``
methods when an operator is first used with a particular UDT.
"""

import ast
import itertools
import linecache
from functools import lru_cache, reduce
from operator import mul

import numpy as np

from ... import backend
from .. import _has_numba, ffi, lib

if _has_numba:
    import numba


_codegen_counter = itertools.count(1)


def _compile_codegen(src, *, func_name, source_label, extra_ns=None):
    """Compile a generated Python source string and return the named function.

    Centralizes the small amount of ``exec``-based code generation that the
    UDT operator wrappers need (Numba has to see a literal function body,
    not a closure, to type-check shape-specialized arithmetic). Three things
    this helper does that a bare ``exec`` doesn't:

    1. ``ast.parse`` runs first so a codegen typo raises a clear
       ``RuntimeError`` with the offending source attached, at the call
       site, rather than as a cryptic ``SyntaxError`` from ``exec`` or a
       confusing ``TypingError`` from Numba's first compile.
    2. ``compile(..., filename, "exec")`` uses a human-readable synthetic
       filename like ``"<gb-udt plus record nleaves=2> #7"``, and the
       generated source is registered with ``linecache`` so any later
       traceback shows real lines instead of ``<string>:??``.
    3. The execution namespace is constructed here, so the names visible
       to the generated code are auditable in one place. Extra entries
       (e.g., the user's compiled ``numba_func`` for wrapper bodies) come
       in via ``extra_ns``.

    Returns ``namespace[func_name]``.
    """
    try:
        ast.parse(src)
    except SyntaxError as exc:
        # Codegen bug, not user input; surface the source so a future
        # maintainer can see exactly what was generated.
        raise RuntimeError(
            f"Generated code for {source_label!r} is not valid Python "
            f"(parse error: {exc}). Source:\n{src}"
        ) from exc
    # Counter suffix so two codegens with the same label still get distinct
    # cache keys (e.g., the same op compiled for two different UDTs that
    # happen to print the same shape summary).
    filename = f"{source_label} #{next(_codegen_counter)}"
    linecache.cache[filename] = (
        len(src),
        None,
        src.splitlines(keepends=True),
        filename,
    )
    code = compile(src, filename, "exec")
    # Generated code never calls Python's ``min`` / ``max`` (still reachable as
    # builtins): they order a NaN by argument position, where SuiteSparse's
    # ``fmin`` ignores it, so :func:`_minmax_expr` spells the comparison out,
    # with ``signbit`` for the signed-zero tie.
    namespace = {"abs": abs, "np": np, "signbit": np.signbit}
    if _has_numba:
        namespace["numba"] = numba
    if extra_ns:
        namespace.update(extra_ns)
    exec(code, namespace)
    return namespace[func_name]


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

# Ops that use function-call syntax rather than infix (e.g., min(a, b), not "a min b")
_FUNC_BINARY_OPS = {"min", "max"}
_FUNC_UNARY_OPS = {"abs"}

NP_TO_C_TYPES = {
    np.dtype(np.bool_): "_Bool",
    np.dtype(np.int8): "int8_t",
    np.dtype(np.int16): "int16_t",
    np.dtype(np.int32): "int32_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.uint8): "uint8_t",
    np.dtype(np.uint16): "uint16_t",
    np.dtype(np.uint32): "uint32_t",
    np.dtype(np.uint64): "uint64_t",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    # Complex types are layout-compatible with numpy's; SS's GraphBLAS.h
    # typedefs ``GxB_FC32_t`` / ``GxB_FC64_t`` to C99 ``float _Complex`` /
    # ``double _Complex`` (or MSVC's ``_Fcomplex`` / ``_Dcomplex``). The
    # JIT include chain pulls in GraphBLAS.h, so these names are in scope.
    # Plus, minus, times, truediv, ainv compile natively on _Complex; abs
    # uses cabs / cabsf (see _c_expr_unary). min, max, floordiv don't make
    # sense on complex; ``_element_types`` rejects them before any codegen.
    np.dtype(np.complex64): "GxB_FC32_t",
    np.dtype(np.complex128): "GxB_FC64_t",
}

# Ops with no meaning on complex fields (no ordering for min/max, no integer
# mod for floordiv); ``_element_types`` rejects them on both paths.
_OPS_NOT_FOR_COMPLEX = frozenset({"min", "max", "floordiv"})

# ``<type>_MIN`` spellings from <stdint.h>, keyed by itemsize. Signed integer
# division traps on ``MIN / -1``; the generated C tests for it explicitly.
_C_INT_MIN = {1: "INT8_MIN", 2: "INT16_MIN", 4: "INT32_MIN", 8: "INT64_MIN"}


def _is_float(np_dtype):
    return np_dtype.kind == "f"


def _is_signed_int(np_dtype):
    return np_dtype.kind == "i"


# Vanilla strips GxB callables but keeps GxB constants, so the bare
# ``hasattr`` would lie; gate on the backend too.
_has_jit_set = backend == "suitesparse" and hasattr(lib, "GxB_JIT_C_NAME")

# Names that can't be used as UDT type names or record field names in the
# JIT C source. Covers two failure modes from the JIT include chain
# (``GraphBLAS.h`` transitively pulls in ``<math.h>``, ``<complex.h>``,
# ``<stddef.h>``, ``<stdio.h>``, ``<errno.h>``, ``<stdint.h>``):
#
#   1. **Macro names**: the preprocessor expands them inside any declarator
#      or expression they appear in, so a field declared ``double M_PI ;``
#      becomes ``double 3.14159... ;`` and won't compile. The C-standard
#      separate namespace for struct members doesn't help here.
#   2. **Typedef names** at the *outer* type-name position: emitting
#      ``typedef struct { ... } FILE ;`` collides with the existing
#      ``FILE`` typedef in scope (most compilers reject as a redefinition).
#      Struct member names of the same spelling are fine (members live in
#      their own namespace), but our gate runs the same check at both
#      positions for simplicity.
#
# Either way the JIT compile fails, SuiteSparse swallows the failure
# silently, and the op runs through the slower Numba cfunc path. Block
# these eagerly so the user sees the "without JIT" warning instead.
_C_RESERVED = frozenset(
    {
        # C keywords
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "_Alignas",
        "_Alignof",
        "_Atomic",
        "_Bool",
        "_Complex",
        "_Generic",
        "_Imaginary",
        "_Noreturn",
        "_Static_assert",
        "_Thread_local",
        # C++ keywords that some compilers also reserve; cheap insurance
        "bool",
        "class",
        "new",
        "delete",
        "template",
        "namespace",
        "this",
        "true",
        "false",
        "nullptr",
        # <stddef.h> / <stdio.h>: macros + the FILE typedef
        "NULL",
        "EOF",
        "offsetof",
        "stdin",
        "stdout",
        "stderr",
        "FILE",
        # <stddef.h> / <stdint.h>: typedefs at risk in the outer position
        "size_t",
        "ptrdiff_t",
        "wchar_t",
        "intptr_t",
        "uintptr_t",
        "intmax_t",
        "uintmax_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        # <math.h> numeric macros
        "INFINITY",
        "NAN",
        "HUGE_VAL",
        "HUGE_VALF",
        "HUGE_VALL",
        "M_E",
        "M_LOG2E",
        "M_LOG10E",
        "M_LN2",
        "M_LN10",
        "M_PI",
        "M_PI_2",
        "M_PI_4",
        "M_1_PI",
        "M_2_PI",
        "M_2_SQRTPI",
        "M_SQRT2",
        "M_SQRT1_2",
        # <complex.h>: ``complex`` and ``I`` in particular expand inside
        # any field declarator.
        "complex",
        "imaginary",
        "I",
        "_Complex_I",
        "_Imaginary_I",
        "CMPLX",
        "CMPLXF",
        "CMPLXL",
        # <errno.h>
        "errno",
    }
)


def _is_valid_c_identifier(name):
    """True if ``name`` is a valid C identifier and not a reserved word."""
    return isinstance(name, str) and name.isidentifier() and name not in _C_RESERVED


# Process-global counter for synthesizing C names when the user-supplied
# Python name isn't a valid C identifier (or is a C reserved word, or the
# UDT was registered without ``name=`` at all). The Python-side ``DataType.name``
# is left alone; this only affects ``GxB_JIT_C_NAME``. ``itertools.count`` is
# atomic enough for our purposes; collisions across processes are harmless
# because the SS JIT cache files are keyed by ``(c_name, content_hash)`` per
# process state.
_synthetic_udt_counter = itertools.count(1)


def _pick_c_type_name(python_name):
    """Return a valid C identifier for use as ``GxB_JIT_C_NAME`` on a UDT.

    When ``python_name`` is already a valid C identifier (and not reserved),
    return it unchanged so introspection and JIT cache filenames remain
    readable. Otherwise mint a fresh ``_gbudt_NNN`` name. This lets UDTs
    that were anonymous in the Python sense (no ``name=`` passed, name with
    dots / special chars, name that collides with a C keyword) still take
    the JIT path; users never see the synthetic name unless they read
    ``DataType.jit_c_name``.
    """
    if _is_valid_c_identifier(python_name):
        return python_name
    return f"_gbudt_{next(_synthetic_udt_counter)}"


def _get_udt_info(dtype):
    """Return ('record', field_names) or ('array', (base_dtype, flat_size)) or None."""
    np_type = dtype.np_type
    if np_type.names is not None:
        return "record", np_type.names
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        return "array", (base, reduce(mul, shape))
    return None


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
    # index map (``_array_operand_source``). Shapes such as ``(2, 3)`` with
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


def _strip_leading_ones(shape):
    i = 0
    while i < len(shape) and shape[i] == 1:
        i += 1
    return shape[i:]


def _builtin_dtype(np_type):
    """Return the built-in ``DataType`` for ``np_type``, or ``None`` if it is not one."""
    from ..dtypes import _registry

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
    from ... import binary
    from .utils import get_typed_op

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
    from ..dtypes import lookup_dtype

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
    which keeps no result type, such an int takes int64, uint64 or float64,
    whichever holds the values, so it compares exactly, as numpy compares it:
    ``int8_vec < 300`` is all True. Built-in dtypes and UDT leaves both use this.
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


def _literal_leaf_values(value, shape):
    """Return the Python numbers in literal ``value`` read as an array of ``shape``, or ``None``.

    ``shape`` is ``()`` for a scalar leaf. A number fills any shape, as numpy
    broadcasts it. A sequence that does not fill the shape, or holds anything
    that is not a Python number, gives ``None``.
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
    return values if all(type(x) in _LITERAL_KINDS for x in values) else None


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
    so ``int8_udt == 300`` is False. A sequence not laid out like an element
    gives ``None``.
    """
    from ..dtypes import lookup_dtype

    def leaf_dtype(leaf, part):
        base, shape = leaf.subdtype or (leaf, ())
        values = _literal_leaf_values(part, shape)
        if values is None:
            raise _LiteralDoesNotFit
        kind = max((_LITERAL_KINDS[type(x)] for x in values), key="bifc".index)
        if base.kind in "biufc":
            elem = _weak_literal_leaf(values, kind, base, exact=exact)
        else:
            elem = _WEAK_LITERAL_DEFAULT[kind]
        return np.dtype((elem, shape)) if shape else elem

    def record_leaves(np_type, part):
        if type(_as_python_number(part)) in _LITERAL_KINDS:
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
                yield leaf_dtype(field, item)

    np_type = dtype.np_type
    try:
        if np_type.names is not None:
            literal_type = _promoted_record([np_type], record_leaves(np_type, value))
        else:
            literal_type = leaf_dtype(np_type, value)
    except _LiteralDoesNotFit:
        return None
    return dtype if literal_type == np_type else lookup_dtype(literal_type)


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
    from ..dtypes import lookup_dtype

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
    has no entry in ``NP_TO_C_TYPES`` and ``_udt_c_typedef`` bails out, so
    the op keeps the cfunc for every call.
    """
    for name in np_type.names:
        field_dtype = np_type.fields[name][0]
        py_path = f"{python_prefix}[{name!r}]"
        c_path = f"{c_prefix}{name}" if not c_prefix else f"{c_prefix}.{name}"
        if field_dtype.names is not None:
            yield from _iter_record_leaves(field_dtype, py_path, c_path)
        else:
            yield py_path, c_path, field_dtype


def _udt_c_typedef(python_name, np_type):
    """Build the JIT C typedef for a record or array UDT, or return ``None``.

    Returns ``(type_name, typedef)`` on success, where ``type_name`` is the
    C identifier we register with SuiteSparse (may differ from
    ``python_name`` if that wasn't a valid C identifier; see
    ``_pick_c_type_name``). Returns ``None`` when the UDT still can't be
    expressed in C after synthesizing the top-level name (a field name
    collides with a C reserved word, or a leaf field type isn't in
    ``NP_TO_C_TYPES``, e.g. object or datetime).

    Nested record UDTs (structured field whose dtype is itself a record)
    are supported: each inner struct is emitted with a synthesized
    ``_gbnest_NNN`` C name before the outer typedef in the same
    ``GxB_JIT_C_DEFINITION`` string, so SuiteSparse's JIT C file ends up
    with the full chain of typedefs in scope. Array UDTs are flattened to
    a single-dimension C array so that the generated per-element operator
    code can use flat indices like ``x->v[i]``, matching the
    (row-major contiguous) numpy memory layout.
    """
    # If the numpy layout disagrees with what a C compiler would produce,
    # the JIT kernel would read fields at the wrong offsets. Skip rather
    # than register a broken typedef; the cfunc path is still correct.
    if np_type.names is not None and not _is_c_compatible_layout(np_type):
        return None
    type_name = _pick_c_type_name(python_name)
    if np_type.names is not None:
        # Inner typedefs (for nested structured fields) are prepended to the
        # final definition string so the outer struct's field types are in
        # scope when SuiteSparse compiles the kernel.
        inner_typedefs = []
        fields = []
        for field_name in np_type.names:
            # Field names still have to be valid C; synthesizing per-field
            # names would mean tracking a name map for the codegen and would
            # break the human-readable JIT source (``z->_f0 = ...``). Not
            # worth it for the corner case where a user named a field
            # ``"class"``; fall back to cfunc there.
            if not _is_valid_c_identifier(field_name):
                return None
            field_dtype = np_type.fields[field_name][0]
            if field_dtype.names is not None:
                # Nested struct field. Recurse to build the inner typedef.
                # The inner C name is synthesized to avoid collisions with
                # user-supplied type names elsewhere in the process.
                inner_name = f"_gbnest_{next(_synthetic_udt_counter)}"
                inner_info = _udt_c_typedef(inner_name, field_dtype)
                if inner_info is None:
                    return None
                inner_resolved_name, inner_typedef = inner_info
                inner_typedefs.append(inner_typedef)
                fields.append(f"{inner_resolved_name} {field_name}")
                continue
            c_type = NP_TO_C_TYPES.get(field_dtype)
            if c_type is None:
                return None
            fields.append(f"{c_type} {field_name}")
        outer_typedef = f"typedef struct {{ {' ; '.join(fields)} ; }} {type_name} ;"
        if inner_typedefs:
            typedef = " ".join([*inner_typedefs, outer_typedef])
        else:
            typedef = outer_typedef
        return type_name, typedef
    if np_type.subdtype is not None:
        base_dtype, shape = np_type.subdtype
        base_c_type = NP_TO_C_TYPES.get(base_dtype)
        if base_c_type is None:
            return None
        size = reduce(mul, shape)
        typedef = f"typedef struct {{ {base_c_type} v [{size}] ; }} {type_name} ;"
        return type_name, typedef
    return None


def _c_aligned_version(np_type):
    """Return an ``align=True`` version of ``np_type`` with every nested
    record also aligned recursively.

    ``np.dtype([..., (name, inner_struct)], align=True)`` respects the
    *inner_struct*'s own declared alignment, so a packed inner struct stays
    packed at its natural-1-byte alignment and the outer's ``coord`` field
    lands at the outer's next-after-flag offset (4 for ``int32``), not at
    the 8-byte boundary a C compiler would pick. To compare against what C
    would emit, we need to rebuild the inner as ``align=True`` first, then
    use that rebuilt type as the outer's field type.
    """
    if np_type.names is None:
        return np_type
    fields = []
    for name in np_type.names:
        f = np_type.fields[name][0]
        if f.names is not None:
            f = _c_aligned_version(f)
        fields.append((name, f))
    return np.dtype(fields, align=True)


def _is_c_compatible_layout(np_type):
    """Return True iff ``np_type``'s layout matches what a C compiler will
    pick for the corresponding ``typedef struct``.

    A user-supplied ``np.dtype([(name, dtype), ...])`` is *packed* by default
    (no padding between fields). A C struct *aligns* fields to their natural
    boundary. For ``[(int32, float64)]`` numpy uses offsets ``0, 4`` and
    itemsize 12; C uses offsets ``0, 8`` and itemsize 16. The JIT-compiled
    kernel reads fields at C offsets but the numpy buffer holds them at the
    packed offsets, so the JIT would read garbage. Detect and refuse JIT in
    that case; the user can either re-register with ``align=True``, use the
    dict / dataclass form (which already does), or accept the cfunc path.

    Array UDTs are always C-compatible (single contiguous run of one type).
    """
    if np_type.names is None:
        return True
    aligned = _c_aligned_version(np_type)
    if aligned.itemsize != np_type.itemsize:
        return False
    for name in np_type.names:
        if np_type.fields[name][1] != aligned.fields[name][1]:
            return False
        if not _is_c_compatible_layout(np_type.fields[name][0]):
            return False
    return True


# Numba function generators, called lazily from each op's ``_compile_udt``.
if _has_numba:

    def _expr_binary(py_op, x_expr, y_expr, x_dtype, y_dtype):
        """Python-source builder; sibling of :func:`_c_expr_binary` for the C JIT.

        ``x_expr`` and ``y_expr`` are scalars, never arrays. ``x_dtype`` and
        ``y_dtype`` are the numpy dtypes the op computes in for each operand:
        the built-in op's input types, which the operands are converted to
        first. Three ops consult them:

        - ``min`` / ``max`` follow SuiteSparse's own ``GrB_MIN_FP64``, which
          is C99 ``fmin`` and ignores a NaN operand rather than ordering or
          propagating it; see :func:`_minmax_expr`. Only floating-point
          operands carry the NaN and signed-zero rules, so the dtype decides
          which of the two forms is emitted.
        - ``//`` on two signed integers. Numba deliberately returns 0 for
          ``INT64_MIN // -1`` to dodge the SIGFPE that x86 raises on the
          unrepresentable quotient, while numpy wraps to ``INT64_MIN``. Since
          ``a // -1`` is exactly ``-a``, routing that divisor through
          negation (which wraps) reaches numpy's answer without the trap.
        - ``/`` where either side is complex. Numba's complex division raises
          ``ZeroDivisionError`` unconditionally, outside the error model's
          control, so a zero divisor left the element unwritten while the C
          kernel returned numpy's infinities. Spelling out the zero case here
          makes both paths match numpy.

        ``/`` on two integers needs nothing: the built-in typing makes its
        result float64, as ``truediv`` on two INT64 vectors is FP64, so
        ``x / 0`` is an infinity, as there.
        """
        if py_op in _FUNC_BINARY_OPS:
            return _minmax_expr(py_op, x_expr, y_expr, x_dtype, y_dtype)
        if py_op == "//" and _is_signed_int(x_dtype) and _is_signed_int(y_dtype):
            return f"(-{x_expr} if {y_expr} == -1 else {x_expr} // {y_expr})"
        if py_op == "/" and "c" in (x_dtype.kind, y_dtype.kind):
            return (
                f"({x_expr} / {y_expr} if {y_expr} != 0 "
                f"else complex({x_expr}.real / 0.0, {x_expr}.imag / 0.0))"
            )
        return f"{x_expr} {py_op} {y_expr}"

    def _minmax_expr(py_op, x_expr, y_expr, x_dtype, y_dtype):
        """Return a Python expression for ``min`` / ``max`` matching C ``fmin``.

        ``GrB_MIN_FP64`` is C99 ``fmin``, so this builder and
        :func:`_c_minmax_expr` both reproduce it or ``binary.min`` means one
        thing on FP64 and another on a UDT: a NaN operand is ignored unless
        both are NaN. C99 leaves the tie between ``-0.0`` and ``0.0`` to the
        C library, which differs by platform, so both builders fix it: it
        resolves to ``-0.0`` for ``min``, ``0.0`` for ``max``, whichever side
        each sits on (Apple's libm answers so; glibc's x86-64 ``fmin`` answers
        by position).

        Neither ``min`` nor ``np.fmin`` gets both rules right under Numba.
        Python's builtin returns its first argument whenever the comparison
        is false, so it keeps a NaN on the left and drops one on the right.
        ``np.fmin`` fixes the NaN rule but Numba lowers it without the
        signed-zero tie-break. Spelling the comparison out, here and in the C
        kernel alike, is what keeps the two execution paths equal on every
        platform.

        Integers have neither NaN nor a signed zero, so they take the same
        plain comparison the C side emits.
        """
        cmp_op = "<" if py_op == "min" else ">"
        if not _is_float(x_dtype) and not _is_float(y_dtype):
            return f"({x_expr} if {x_expr} {cmp_op} {y_expr} else {y_expr})"
        # ``y != y`` is the NaN test on the right operand: when it holds, the
        # left one wins whatever it is, which is fmin's "ignore the NaN" rule
        # and also (correctly) returns NaN when both are NaN.
        tie = f"signbit({x_expr})" if py_op == "min" else f"not signbit({x_expr})"
        return (
            f"({x_expr} if ({x_expr} {cmp_op} {y_expr} or {y_expr} != {y_expr}"
            f" or ({x_expr} == {y_expr} and {tie})) else {y_expr})"
        )

    def _expr_unary(py_op, operand):
        """Python-source builder; sibling of :func:`_c_expr_unary` for JIT C."""
        if py_op in _FUNC_UNARY_OPS or py_op == "np.abs":
            return f"{py_op}({operand})"
        return f"{py_op}{operand}"

    def _cast_expr(expr, to_dtype, ns):
        """Return Numba source converting scalar ``expr`` to numpy dtype ``to_dtype``.

        ``to_dtype`` of ``None`` leaves ``expr`` alone. The conversion function
        is added to ``ns``, the generated code's namespace.
        """
        if to_dtype is None:
            return expr
        name = f"_to_{to_dtype.name}"
        ns[name] = numba.from_dtype(to_dtype)
        return f"{name}({expr})"

    def _make_record_func(
        leaf_paths, arity, py_op, *, x_is_scalar=False, y_is_scalar=False, leaf_types=None
    ):
        """Build a Numba njit function for a record UDT.

        ``leaf_paths`` is a sequence of Python access strings ``"['a']"`` or,
        for nested records, ``"['outer']['inner_a']"``. The generated function
        always returns a *flat* tuple of leaf values regardless of nesting
        depth; the wrapper (in base.py) walks the same leaf paths when
        writing the result back, so nested-record outputs land at the
        correct depth without nested tuple construction (which Numba can't
        ``setitem``-assign to a record field).

        When ``x_is_scalar`` or ``y_is_scalar`` is True, that argument is a
        plain scalar (not a record), so it is used directly for all leaves.
        A binary op needs ``leaf_types``, one ``(x_cast, y_cast, x_dtype,
        y_dtype, result)`` tuple per leaf: the numpy dtype to convert each
        operand to before the op (``None`` for none), the dtypes the op then
        computes in, which :func:`_expr_binary` keys its semantics on, and the
        result leaf's dtype.

        An array-valued leaf is computed element by element, with the same
        scalar expression as a scalar leaf or an array UDT element, into an
        array of the result leaf's dtype. As a whole-array expression it ran
        through Numba's ufuncs, which pick their own loop (``int8 / int8`` in
        float32), raise on a complex zero divisor, and take none of the scalar
        forms of ``min``, ``max`` and ``//``.
        """
        ns = {}
        lines = []
        if arity == 2:
            parts = []
            for i, path in enumerate(leaf_paths):
                x_cast, y_cast, x_dtype, y_dtype, result = leaf_types[i]
                if not result.shape:
                    x_expr = _cast_expr("x" if x_is_scalar else f"x{path}", x_cast, ns)
                    y_expr = _cast_expr("y" if y_is_scalar else f"y{path}", y_cast, ns)
                    parts.append(_expr_binary(py_op, x_expr, y_expr, x_dtype, y_dtype))
                    continue
                # Two array leaves have the same shape (``_check_udt_pair``).
                ns[f"_r{i}"] = result.base
                x_expr = _cast_expr("x" if x_is_scalar else f"x{path}[k]", x_cast, ns)
                y_expr = _cast_expr("y" if y_is_scalar else f"y{path}[k]", y_cast, ns)
                lines += [
                    f"    r{i} = np.empty({result.shape}, _r{i})",
                    f"    for k in np.ndindex({result.shape}):",
                    f"        xv = {x_expr}",
                    f"        yv = {y_expr}",
                    f"        r{i}[k] = {_expr_binary(py_op, 'xv', 'yv', x_dtype, y_dtype)}",
                ]
                parts.append(f"r{i}")
            sig = "x, y"
        else:
            # Python's ``abs`` does not take an array field in Numba; ``np.abs``
            # takes both.
            unary_op = "np.abs" if py_op == "abs" else py_op
            parts = [_expr_unary(unary_op, f"x{path}") for path in leaf_paths]
            sig = "x"
        body = ", ".join(parts)
        # Single-leaf tuple needs the trailing comma to remain a tuple.
        ret = f"({body},)" if len(leaf_paths) == 1 else f"({body})"
        lines.append(f"    return {ret}")
        src = f"def _op({sig}):\n" + "\n".join(lines) + "\n"
        op_func = _compile_codegen(
            src,
            func_name="_op",
            source_label=f"<gb-udt {py_op!r} record nleaves={len(leaf_paths)} arity={arity}>",
            extra_ns=ns,
        )
        return numba.njit(op_func, error_model="numpy")

    def _array_operand_source(side, operand_shape, shape, ns):
        """Return ``(setup, ref)``, Numba source that reads operand ``side`` at element ``i``.

        ``i`` is a flat position in a result of ``shape``. ``operand_shape`` is
        the operand's shape, or ``None`` for a plain scalar, read for every
        element. An operand that broadcasts to ``shape`` reads through a
        constant flat index map, added to ``ns``.
        """
        if operand_shape is None:
            return "", f"{side}_ptr[0]"
        size = reduce(mul, operand_shape)
        setup = f"    {side} = numba.carray({side}_ptr, {size})\n"
        # Broadcasting repeats no element when the sizes agree, so the shapes
        # differ at most by leading axes of length 1 and the flat order is the same.
        if size == reduce(mul, shape):
            return setup, f"{side}[i]"
        ns[f"{side}_index"] = np.broadcast_to(np.arange(size).reshape(operand_shape), shape).ravel()
        return setup, f"{side}[{side}_index[i]]"

    def _make_array_wrapper(
        shape,
        base_numba_type,
        arity,
        py_op,
        *,
        x_type,
        y_type=None,
        x_shape=None,
        y_shape=None,
        x_cast=None,
        y_cast=None,
        store_cast=None,
    ):
        """Build a cfunc-ready wrapper for an array UDT (element-by-element).

        ``shape`` and ``base_numba_type`` are the shape and element type of the
        result. ``x_type`` and ``y_type`` are the operands' element types. For
        a binary op, ``x_shape`` and ``y_shape`` are the operands' shapes, which
        broadcast to the result's; ``None`` makes that side a plain scalar of
        its type, broadcast to all elements.
        ``x_cast`` and ``y_cast`` are numpy dtypes to convert each operand to
        before the op, if any. ``store_cast``, a ``(computed, stored)`` pair of
        numpy dtypes, makes each element computed as ``computed`` and stored as
        ``stored``, cast as a store casts (:func:`_cast_leaf_source`); then
        ``base_numba_type`` is the stored element type.

        Returns (wrapper_func, wrapper_sig).
        """
        nt = numba.types
        ns = {}

        def store(expr):
            if store_cast is None:
                return expr
            computed, stored = store_cast
            return _cast_leaf_source(computed, stored, _cast_expr(expr, computed, ns), ns)

        size = reduce(mul, shape)
        # A loop, not one statement per element: Numba's typing time grows
        # faster than linearly with the number of statements, so unrolling took
        # seconds for a 32 by 32 array and minutes for a 64 by 64 one.
        arrays = f"    z = numba.carray(z_ptr, {size})\n"
        if arity == 2:
            # Each operand is read once per element, into ``xv`` and ``yv``,
            # so an op that names an operand more than once reads it once.
            body = ""
            computed = []
            for side, operand_shape, operand_type, cast in [
                ("x", x_shape, x_type, x_cast),
                ("y", y_shape, y_type, y_cast),
            ]:
                setup, ref = _array_operand_source(side, operand_shape, shape, ns)
                arrays += setup
                body += f"        {side}v = {_cast_expr(ref, cast, ns)}\n"
                computed.append(
                    numba.np.numpy_support.as_dtype(operand_type) if cast is None else cast
                )
            body += f"        z[i] = {store(_expr_binary(py_op, 'xv', 'yv', *computed))}\n"
            params = "z_ptr, x_ptr, y_ptr"
            sig = nt.void(nt.CPointer(base_numba_type), nt.CPointer(x_type), nt.CPointer(y_type))
        else:
            body = f"        z[i] = {store(_expr_unary(py_op, 'x[i]'))}\n"
            params = "z_ptr, x_ptr"
            arrays += f"    x = numba.carray(x_ptr, {size})\n"
            sig = nt.void(nt.CPointer(base_numba_type), nt.CPointer(x_type))
        src = f"def _op({params}):\n{arrays}    for i in range({size}):\n{body}"
        op_func = _compile_codegen(
            src,
            func_name="_op",
            source_label=f"<gb-udt {py_op!r} array size={size} arity={arity}>",
            extra_ns=ns,
        )
        return op_func, sig

    # MAINT 2026-05-21: this function, ``compile_udt_unary_wrapper`` below,
    # and ``_make_jit_c_definition`` all branch on ``kind == "record"`` vs the
    # array path with near-identical scaffolding. When a third op family
    # (ternary, indexbinary, ...) needs the same treatment, fold the three
    # into one shape-parametrized helper instead of pasting a third copy.
    def compile_udt_binary_wrapper(op_name, py_op, dtype, dtype2, out=None):
        """Compile a built-in element-wise binary op for a UDT.

        Handles two cases:

        - Both sides are UDTs that pair (:func:`_check_udt_pair`): a
          field-by-field (record) or element-by-element (array) op.
        - One side is a UDT and the other is a scalar type: the scalar is
          broadcast to all fields or elements (e.g., ``Point + int`` adds
          the int to every field).

        Each element is computed as the op computes it on the operands'
        element dtypes (see :func:`_udt_op_types`).

        ``out``, a UDT of the result's layout, makes the op store each element
        into that type instead, cast as a store casts (:func:`fused_store_op`).

        Returns ``(wrapper_func, wrapper_sig, ret_type)``. Raises ``KeyError``
        when the dtype combination is not supported, with a clear message for
        common mistakes like passing two record UDTs with different field
        names.
        """
        from .base import _get_udt_wrapper

        info_x = _get_udt_info(dtype)
        info_y = _get_udt_info(dtype2)
        if info_x is None and info_y is None:
            raise KeyError(
                f"binary.{op_name} does not work with ({dtype}, {dtype2}). "
                f"Element-wise UDT ops require a record dtype (named fields) "
                f"or an array dtype (e.g., FP64[3])."
            )
        _check_udt_pair(op_name, dtype, dtype2, info_x, info_y)
        ret_type, input_types = _udt_op_types(op_name, dtype, dtype2)

        x_is_scalar = info_x is None  # left side is a plain scalar type
        y_is_scalar = info_y is None  # right side is a plain scalar type
        udt_dtype = dtype2 if x_is_scalar else dtype

        # Numba promotes mixed operands its own way (uint64 with int64 is
        # int64, where GraphBLAS uses float64) and only then stores the value
        # into the result's type, so each operand is converted first to the
        # type the built-in op computes in. Same-type operands are left alone.
        def cast(in_dtype, elem):
            return None if in_dtype == elem.base else in_dtype

        if udt_dtype.np_type.names is not None:
            from .base import _compile_udf_for_udt

            # Use leaf paths so the same codegen handles nested-record UDTs
            # uniformly. A non-nested record's leaves are its top-level
            # fields, with paths like ``"['a']"``.
            leaves = list(_iter_record_leaves(udt_dtype.np_type))
            leaves_x = None if x_is_scalar else list(_iter_record_leaves(dtype.np_type))
            leaves_y = None if y_is_scalar else list(_iter_record_leaves(dtype2.np_type))
            ret_leaves = list(_iter_record_leaves(ret_type.np_type))
            leaf_types = [
                (
                    cast(in_x, dtype.np_type if x_is_scalar else leaves_x[i][2]),
                    cast(in_y, dtype2.np_type if y_is_scalar else leaves_y[i][2]),
                    in_x,
                    in_y,
                    ret_leaves[i][2],
                )
                for i, (in_x, in_y) in enumerate(input_types)
            ]
            func = _make_record_func(
                [py for py, _c, _d in leaves],
                2,
                py_op,
                x_is_scalar=x_is_scalar,
                y_is_scalar=y_is_scalar,
                leaf_types=leaf_types,
            )
            sig = (dtype.numba_type, dtype2.numba_type)
            _compile_udf_for_udt(
                func, sig, op_kind="binary", op_name=op_name, dtypes=(dtype, dtype2)
            )
            if out is not None:
                wrapper, wrapper_sig = _make_record_store_wrapper(
                    func, ret_type, out, dtype, dtype2
                )
                return wrapper, wrapper_sig, out
            numba_ret_type = func.overloads[sig].signature.return_type
            # Numba can still return a leaf wider than ``ret_type``'s (int8
            # plus int8 is int64 in Numba, INT8 here); the wrapper casts each
            # leaf as it stores it, which for those ops gives the same value.
            wrapper, wrapper_sig = _get_udt_wrapper(
                func, ret_type, dtype, dtype2, numba_ret_type=numba_ret_type
            )
        else:
            # Each side has its own element type, since the bases may differ.
            x_type = dtype.np_type.base
            y_type = dtype2.np_type.base
            ((in_x, in_y),) = input_types
            base, shape = ret_type.np_type.subdtype
            stored = base if out is None else out.np_type.subdtype[0]
            wrapper, wrapper_sig = _make_array_wrapper(
                shape,
                numba.from_dtype(stored),
                2,
                py_op,
                x_type=numba.from_dtype(x_type),
                y_type=numba.from_dtype(y_type),
                x_shape=None if x_is_scalar else dtype.np_type.shape,
                y_shape=None if y_is_scalar else dtype2.np_type.shape,
                x_cast=cast(in_x, x_type),
                y_cast=cast(in_y, y_type),
                store_cast=None if out is None else (base, stored),
            )
        return wrapper, wrapper_sig, ret_type if out is None else out

    def _make_record_store_wrapper(func, result_type, out_type, dtype, dtype2=None):
        """Build a cfunc-ready wrapper storing ``func``'s result as UDT ``out_type``.

        ``func`` returns one value per leaf of ``result_type``, the type the op
        computes in. Each value becomes that leaf's dtype and is then cast into
        the matching leaf of ``out_type`` as a store casts it
        (:func:`_cast_leaf_source`), so storing into an object of another type
        needs no temporary of the result's own type.

        Returns (wrapper_func, wrapper_sig).
        """
        from .base import _input_operand

        nt = numba.types
        ns = {"numba_func": func}
        lines = ["    z = numba.carray(z_ptr, 1)"]
        params, args, sig_args = ["z_ptr"], [], [nt.CPointer(out_type.numba_type)]
        for var, operand in [("x", dtype), ("y", dtype2)]:
            if operand is None:
                continue
            setup, deref, ptr_type = _input_operand(operand, var)
            if setup:
                lines.append(setup.rstrip("\n"))
            params.append(f"{var}_ptr")
            args.append(deref)
            sig_args.append(ptr_type)
        lines.append(f"    r = numba_func({', '.join(args)})")
        leaves = zip(
            _iter_record_leaves(result_type.np_type),
            _iter_record_leaves(out_type.np_type),
            strict=True,
        )
        for k, ((path, _c, computed), (_p, _cz, stored)) in enumerate(leaves):
            if computed.shape:
                value = _cast_expr(f"r[{k}][i]", computed.base, ns)
                value = _cast_leaf_source(computed.base, stored.base, value, ns)
                lines.append(f"    for i in np.ndindex({computed.shape}):")
                lines.append(f"        z[0]{path}[i] = {value}")
            else:
                value = _cast_leaf_source(computed, stored, _cast_expr(f"r[{k}]", computed, ns), ns)
                lines.append(f"    z[0]{path} = {value}")
        src = f"def _store({', '.join(params)}):\n" + "\n".join(lines) + "\n"
        wrapper = _compile_codegen(
            src,
            func_name="_store",
            source_label=f"<gb-udt store {result_type} as {out_type}>",
            extra_ns=ns,
        )
        return wrapper, nt.void(*sig_args)

    @lru_cache
    def _saturating_cast(dst):
        """Return a Numba function converting a float to integer dtype ``dst`` as GraphBLAS does.

        NaN is 0, a value beyond the type's range is its nearest bound, and
        anything else truncates toward zero. Numba's own conversion is LLVM's,
        which leaves the first two undefined, and numpy's differs by platform.
        """
        info = np.iinfo(dst)
        lo = "0.0" if dst.kind == "u" else repr(float(info.min))
        src = (
            "def _sat(x):\n"
            "    x = _f64(x)\n"
            f"    if x != x or x <= {lo}:\n"
            f"        return _to(0 if x != x else {info.min})\n"
            f"    if x >= {float(info.max)!r}:\n"
            f"        return _to({info.max})\n"
            "    return _to(x)\n"
        )
        func = _compile_codegen(
            src,
            func_name="_sat",
            source_label=f"<gb-udt saturating cast to {dst}>",
            extra_ns={"_f64": numba.float64, "_to": numba.from_dtype(dst)},
        )
        return numba.njit(func)

    def _cast_leaf_source(src, dst, expr, ns):
        """Return Numba source converting ``expr`` from numpy dtype ``src`` to ``dst``.

        It converts as GraphBLAS casts built-in types on store: anything
        nonzero is True, a complex number loses its imaginary part, a float
        stored as an integer saturates (:func:`_saturating_cast`), and
        integers wrap. Helpers are added to ``ns``, the generated code's
        namespace.
        """
        if src == dst:
            return expr
        if dst.kind == "b":
            return f"({expr} != 0)"
        if src.kind == "c" and dst.kind != "c":
            expr = f"({expr}).real"
            src = np.dtype(f"f{src.itemsize // 2}")
        if src.kind == "f" and dst.kind in "iu":
            ns[f"_sat_{dst.name}"] = _saturating_cast(dst)
            return f"_sat_{dst.name}({expr})"
        ns[f"_to_{dst.name}"] = numba.from_dtype(dst)
        return f"_to_{dst.name}({expr})"

    def compile_udt_cast_wrapper(dtype, to_dtype):
        """Compile a cfunc wrapper that stores an element of UDT ``dtype`` as UDT ``to_dtype``.

        :func:`_udt_cast_error` has accepted the pair. Each element of an array
        UDT, and each record leaf, converts by :func:`_cast_leaf_source`.

        Returns (wrapper_func, wrapper_sig).
        """
        nt = numba.types
        ns = {}
        np_x, np_z = dtype.np_type, to_dtype.np_type
        if np_x.names is not None:
            body = ["    x = numba.carray(x_ptr, 1)", "    z = numba.carray(z_ptr, 1)"]
            for (path, _c, leaf_x), (_path, _cz, leaf_z) in zip(
                _iter_record_leaves(np_x), _iter_record_leaves(np_z), strict=True
            ):
                if leaf_x.shape:
                    value = _cast_leaf_source(leaf_x.base, leaf_z.base, f"x[0]{path}[i]", ns)
                    body.append(f"    for i in np.ndindex({leaf_x.shape}):")
                    body.append(f"        z[0]{path}[i] = {value}")
                else:
                    value = _cast_leaf_source(leaf_x, leaf_z, f"x[0]{path}", ns)
                    body.append(f"    z[0]{path} = {value}")
            sig = nt.void(nt.CPointer(to_dtype.numba_type), nt.CPointer(dtype.numba_type))
        else:
            (base_x, shape), base_z = np_x.subdtype, np_z.subdtype[0]
            size = reduce(mul, shape)
            body = [
                f"    x = numba.carray(x_ptr, {size})",
                f"    z = numba.carray(z_ptr, {size})",
                f"    for i in range({size}):",
                f"        z[i] = {_cast_leaf_source(base_x, base_z, 'x[i]', ns)}",
            ]
            sig = nt.void(
                nt.CPointer(numba.from_dtype(base_z)), nt.CPointer(numba.from_dtype(base_x))
            )
        src = "def _cast(z_ptr, x_ptr):\n" + "\n".join(body) + "\n"
        wrapper = _compile_codegen(
            src,
            func_name="_cast",
            source_label=f"<gb-udt cast {dtype} to {to_dtype}>",
            extra_ns=ns,
        )
        return wrapper, sig

    def compile_udt_unary_wrapper(op_name, py_op, dtype, out=None):
        """Compile a built-in element-wise unary op for a UDT.

        ``out`` stores the result as another UDT, as in
        :func:`compile_udt_binary_wrapper`.

        Returns (wrapper_func, wrapper_sig, ret_type).
        Raises KeyError if the dtype is not supported.
        """
        from .base import _get_udt_wrapper, _resolve_udt_return_type

        info = _get_udt_info(dtype)
        if info is None:
            raise KeyError(
                f"unary.{op_name} does not work with {dtype}. "
                f"Element-wise UDT ops require a record dtype (named fields) "
                f"or an array dtype (e.g., FP64[3])."
            )

        kind, _detail = info
        if kind == "record":
            from .base import _compile_udf_for_udt

            leaf_paths = [py for py, _c, _d in _iter_record_leaves(dtype.np_type)]
            func = _make_record_func(leaf_paths, 1, py_op)
            sig = (dtype.numba_type,)
            _compile_udf_for_udt(func, sig, op_kind="unary", op_name=op_name, dtypes=(dtype,))
            numba_ret_type = func.overloads[sig].signature.return_type
            ret_type = _resolve_udt_return_type(numba_ret_type, dtype)
            if out is not None:
                wrapper, wrapper_sig = _make_record_store_wrapper(func, ret_type, out, dtype)
                return wrapper, wrapper_sig, out
            wrapper, wrapper_sig = _get_udt_wrapper(
                func, ret_type, dtype, numba_ret_type=numba_ret_type
            )
        else:
            base_dtype, shape = dtype.np_type.subdtype
            ret_type = dtype
            stored = base_dtype if out is None else out.np_type.subdtype[0]
            wrapper, wrapper_sig = _make_array_wrapper(
                shape,
                numba.from_dtype(stored),
                1,
                py_op,
                x_type=numba.from_dtype(base_dtype),
                store_cast=None if out is None else (base_dtype, stored),
            )
        return wrapper, wrapper_sig, ret_type if out is None else out

    _fused_store_parents = {}

    def fused_store_op(typed_op, out_type):
        """Return ``typed_op`` storing its result as ``out_type``, or ``None`` if it cannot.

        The returned op computes what ``typed_op`` computes, in the same type,
        and casts each element or leaf into ``out_type`` as it writes it, as a
        store casts (``base._store_cast_op``). So a result goes into an object
        of another UDT type in the one GraphBLAS call that computes it, with no
        temporary. Only the built-in ops that lift to UDTs have such a form;
        ``base._store_cast_op`` has already accepted the pair of types.
        """
        from .base import _finalize_udt_op

        parent = getattr(typed_op, "parent", None)
        if parent is None or parent._anonymous:
            return None
        if typed_op.opclass == "BinaryOp" and parent.name in BUILTIN_UDT_BINARY_OPS:
            from .binary import BinaryOp, TypedUserBinaryOp

            cls, typed_cls = BinaryOp, TypedUserBinaryOp
            dtype2 = typed_op.type2
            py_op = BUILTIN_UDT_BINARY_OPS[parent.name]
        elif typed_op.opclass == "UnaryOp" and parent.name in BUILTIN_UDT_UNARY_OPS:
            from .unary import TypedUserUnaryOp, UnaryOp

            cls, typed_cls = UnaryOp, TypedUserUnaryOp
            dtype2 = None
            py_op = BUILTIN_UDT_UNARY_OPS[parent.name]
        else:
            return None
        key = (cls, parent.name, out_type)
        fused = _fused_store_parents.get(key)
        if fused is None:
            fused = _fused_store_parents[key] = cls(
                f"{parent.name}_as_{out_type.name}", anonymous=True, is_udt=True
            )
        dtype = typed_op.type
        if (types := dtype if dtype2 is None else (dtype, dtype2)) in fused._udt_ops:
            return fused._udt_ops[types]
        if dtype2 is None:
            wrapper, sig, _ = compile_udt_unary_wrapper(parent.name, py_op, dtype, out=out_type)
        else:
            wrapper, sig, _ = compile_udt_binary_wrapper(
                parent.name, py_op, dtype, dtype2, out=out_type
            )
        return _finalize_udt_op(fused, dtype, dtype2, out_type, wrapper, sig, typed_cls)


# JIT C code generators below.


def _c_assign_binary(py_op, target, lhs, rhs, field_dtype):
    """Return a C *statement* assigning ``lhs py_op rhs`` to ``target``.

    Almost every op is a single expression, but floating-point ``//`` and
    complex ``/`` and ``*`` need temporaries (see :func:`_c_float_floordiv_stmt`,
    :func:`_c_complex_truediv_stmt` and :func:`_c_complex_times_stmt`), so this
    is the entry point the kernel builder uses rather than :func:`_c_expr_binary`.
    """
    if py_op == "//" and _is_float(field_dtype):
        return _c_float_floordiv_stmt(target, lhs, rhs, field_dtype)
    if py_op == "/" and field_dtype.kind == "c":
        return _c_complex_truediv_stmt(target, lhs, rhs, field_dtype)
    if py_op == "*" and field_dtype.kind == "c":
        return _c_complex_times_stmt(target, lhs, rhs, field_dtype)
    return f"{target} = {_c_expr_binary(py_op, lhs, rhs, field_dtype)} ;"


def _c_expr_binary(py_op, lhs, rhs, field_dtype):
    """Return a C expression for a binary op: e.g., ``(x->a) + (y->a)``.

    ``field_dtype`` is the numpy dtype of the operands and of the element
    being written: a C kernel exists only for an op whose result has the
    operands' type. Two of the ops need it because the C spelling that looks
    obvious disagrees with what the Numba cfunc computes: ``//`` is not C
    ``/``, and ``min`` / ``max`` have to ignore NaN and break a signed-zero tie
    as :func:`_minmax_expr` does. It is required rather than optional so a new
    caller can't silently get the wrong kernel. :func:`_expr_binary` is the
    sibling for the Numba cfunc; SuiteSparse picks between the two on its
    own, so they must agree.

    Floating-point ``//`` and complex ``/`` and ``*`` have no single-expression
    form; go through :func:`_c_assign_binary`.
    """
    if py_op in _FUNC_BINARY_OPS:
        return _c_minmax_expr(py_op, lhs, rhs, field_dtype)
    if py_op == "/":
        return _c_truediv_expr(lhs, rhs, field_dtype)
    if py_op == "//":
        return _c_int_floordiv_expr(lhs, rhs, field_dtype)
    # Everything left (``+``, ``-``, and ``*`` on real fields) spells the same
    # in C as in Python and needs no dtype-specific handling.
    return f"({lhs}) {py_op} ({rhs})"


def _c_minmax_expr(py_op, lhs, rhs, field_dtype):
    """Return a C expression for ``min`` / ``max`` on a field, as the Numba cfunc computes it.

    ``GrB_MIN_FP64`` is C99 ``fmin``: it ignores a NaN operand rather than
    ordering it, and a UDT field does the same. C99 leaves the tie between
    ``-0.0`` and ``0.0`` to the implementation, though: Apple's libm returns
    ``-0.0`` for ``fmin``, while glibc's x86-64 ``fmin`` (one ``minsd``)
    returns an operand by position. So the comparison is spelled out, the same
    as :func:`_minmax_expr` spells it for the cfunc: a NaN on the right loses,
    and a tie resolves to ``-0.0`` for ``min`` and ``0.0`` for ``max``,
    whichever side each sits on. The C kernel and the cfunc then agree on
    every platform; only SuiteSparse's built-in FP64 ``min`` may break that
    tie the other way, where the C library does. ``signbit`` comes from
    ``<math.h>``, which reaches the C JIT kernel through ``GraphBLAS.h``, the
    same route :func:`_c_float_floordiv_stmt` relies on for ``floor`` and
    ``fmod``.

    Integers have neither NaN nor a signed zero, so a plain comparison is
    exact for them.
    """
    cmp_op = "<" if py_op == "min" else ">"
    if not _is_float(field_dtype):
        return f"(({lhs}) {cmp_op} ({rhs}) ? ({lhs}) : ({rhs}))"
    tie = f"signbit({lhs})" if py_op == "min" else f"!signbit({lhs})"
    return (
        f"((({lhs}) {cmp_op} ({rhs}) || ({rhs}) != ({rhs}) "
        f"|| (({lhs}) == ({rhs}) && {tie})) ? ({lhs}) : ({rhs}))"
    )


def _c_truediv_expr(lhs, rhs, field_dtype):
    """Return a C expression for true division on a float field.

    Only float and complex fields reach a C kernel. True division of integers
    or bools gives float64, as on built-in vectors, so its result never has
    the operands' type, which a C kernel needs (``BinaryOp._compile_udt``).
    Complex division is a statement (:func:`_c_complex_truediv_stmt`).
    """
    if not _is_float(field_dtype):
        raise ValueError(f"no C expression divides {field_dtype}")
    return f"({lhs}) / ({rhs})"


def _c_complex_truediv_stmt(target, lhs, rhs, field_dtype):
    """Return a C block dividing two complex fields into ``target``, as the Numba cfunc does.

    C's own ``/`` on ``_Complex`` cannot be used. SuiteSparse's C JIT compiles
    with ``-fcx-limited-range`` under GCC, which replaces C99 Annex G division
    with the naive formula: it squares the divisor's parts, so ``1e20 / 1e20``
    in complex64 gave ``nan`` and ``(1+1j) / (1e160+1e160j)`` gave ``0`` on
    Linux. This block is Numba's complex division, which is CPython's
    ``_Py_c_quot`` (Smith's method: scale by the larger part of the divisor),
    computed in the field's own precision with real arithmetic only, so no
    flag changes it. Each product is its own statement: clang contracts a
    multiply and an add within one expression into a fused multiply-add,
    which rounds once where Numba rounds twice, and GCC under ``-std=c11``
    does not contract at all. A zero divisor divides the parts by ``0.0`` as
    reals, for numpy's infinities, as ``_expr_binary`` spells it for the
    cfunc; ``rhs == 0`` on a ``_Complex`` compares both parts, mirroring the
    cfunc's ``y != 0``.

    The constructor is SuiteSparse's ``GB_CMPLX32``/``GB_CMPLX64`` rather
    than C11's ``CMPLXF``/``CMPLX``, which macOS does not define.
    ``GB_complex.h`` is on the C JIT's include path, names that exact gap, and
    its fallback builds the value through a two-element array, so it keeps
    infinities and NaNs as given.
    """
    ctype, cmplx, creal, cimag, fabs, zero = _c_complex_names(field_dtype)
    return (
        f"{{ {ctype} gb_ar = {creal} ({lhs}) ; {ctype} gb_ai = {cimag} ({lhs}) ; "
        f"{ctype} gb_br = {creal} ({rhs}) ; {ctype} gb_bi = {cimag} ({rhs}) ; "
        f"if (gb_br == {zero} && gb_bi == {zero}) "
        f"{{ {target} = {cmplx} (gb_ar / {zero}, gb_ai / {zero}) ; }} "
        f"else if ({fabs} (gb_br) >= {fabs} (gb_bi)) {{ "
        f"{ctype} gb_r = gb_bi / gb_br ; {ctype} gb_t = gb_bi * gb_r ; "
        f"{ctype} gb_d = gb_br + gb_t ; {ctype} gb_u = gb_ai * gb_r ; "
        f"{ctype} gb_v = gb_ar * gb_r ; "
        f"{target} = {cmplx} ((gb_ar + gb_u) / gb_d, (gb_ai - gb_v) / gb_d) ; }} "
        f"else if (gb_bi == {zero}) {{ {target} = {cmplx} (NAN, NAN) ; }} "
        f"else {{ {ctype} gb_r = gb_br / gb_bi ; {ctype} gb_t = gb_br * gb_r ; "
        f"{ctype} gb_d = gb_t + gb_bi ; {ctype} gb_u = gb_ar * gb_r ; "
        f"{ctype} gb_v = gb_ai * gb_r ; "
        f"{target} = {cmplx} ((gb_u + gb_ai) / gb_d, (gb_v - gb_ar) / gb_d) ; }} }}"
    )


def _c_complex_times_stmt(target, lhs, rhs, field_dtype):
    """Return a C block multiplying two complex fields into ``target``, as the Numba cfunc does.

    Numba and numpy multiply complex numbers by the textbook formula. C99
    Annex G adds a recovery step that turns some of its ``nan`` results into
    infinities, and whether C's ``*`` on ``_Complex`` takes it depends on the
    compiler: GCC under SuiteSparse's ``-fcx-limited-range`` does not, clang
    does. So ``1j * (inf+infj)`` was ``nan+nanj`` on Linux and the cfunc but
    ``-inf+infj`` from a C kernel on macOS. This block is the textbook formula
    in real arithmetic, one product per statement so clang cannot contract it
    into a fused multiply-add (see :func:`_c_complex_truediv_stmt`).
    """
    ctype, cmplx, creal, cimag, _, _ = _c_complex_names(field_dtype)
    return (
        f"{{ {ctype} gb_ar = {creal} ({lhs}) ; {ctype} gb_ai = {cimag} ({lhs}) ; "
        f"{ctype} gb_br = {creal} ({rhs}) ; {ctype} gb_bi = {cimag} ({rhs}) ; "
        f"{ctype} gb_rr = gb_ar * gb_br ; {ctype} gb_ii = gb_ai * gb_bi ; "
        f"{ctype} gb_ri = gb_ar * gb_bi ; {ctype} gb_ir = gb_ai * gb_br ; "
        f"{target} = {cmplx} (gb_rr - gb_ii, gb_ri + gb_ir) ; }}"
    )


def _c_complex_names(field_dtype):
    """Return the C type, constructor and part functions for a complex field."""
    if field_dtype.itemsize == 8:
        return "float", "GB_CMPLX32", "crealf", "cimagf", "fabsf", "0.0f"
    return "double", "GB_CMPLX64", "creal", "cimag", "fabs", "0.0"


def _c_int_floordiv_expr(lhs, rhs, field_dtype):
    """Return a C expression for integer floor division.

    Three ways C ``/`` differs from what the cfunc computes:

    - It truncates toward zero; Python ``//`` floors. Subtract one when the
      operands have different signs and the remainder is non-zero.
    - It raises SIGFPE (process death, not an exception) when the divisor is
      zero. Numba's numpy error model returns 0, as does ``np.floor_divide``.
    - It raises SIGFPE on ``INT_MIN / -1``, whose true quotient is not
      representable. numpy wraps to ``INT_MIN``, which is what the cfunc
      reaches through negation (see ``_expr_binary``).
    """
    if field_dtype.kind in ("u", "b"):
        # Both operands are non-negative, so truncation already floors.
        return f"(({rhs}) == 0 ? 0 : (({lhs}) / ({rhs})))"
    type_min = _C_INT_MIN[field_dtype.itemsize]
    floored = f"({lhs}) / ({rhs}) - ((({lhs}) % ({rhs}) != 0) && ((({lhs}) < 0) != (({rhs}) < 0)))"
    return (
        f"(({rhs}) == 0 ? 0 : "
        f"((({lhs}) == {type_min} && ({rhs}) == -1) ? {type_min} : ({floored})))"
    )


def _c_float_floordiv_stmt(target, lhs, rhs, field_dtype):
    """Return a C block computing floating-point ``//`` into ``target``.

    ``floor(a / b)`` is not floor division. numpy and CPython both compute
    the quotient from the remainder (``(a - fmod(a, b)) / b``) and snap it,
    which is more accurate and handles infinities differently: ``1.0 // 0.1``
    is 9.0 but ``floor(1.0 / 0.1)`` is 10.0, and ``inf // 2.0`` is NaN but
    ``floor(inf / 2.0)`` is ``inf``. This mirrors numpy's ``npy_divmod``
    (which CPython's ``float_divmod`` matches), so the C JIT kernel, the
    Numba cfunc, and ``np.floor_divide`` all agree.

    The sign fix-up needs the remainder twice and the quotient twice, so
    this emits a statement with temporaries rather than one expression.
    """
    is_f32 = field_dtype.itemsize == 4
    ctype = "float" if is_f32 else "double"
    sfx = "f" if is_f32 else ""
    half = "0.5f" if is_f32 else "0.5"
    one = "1.0f" if is_f32 else "1.0"
    zero = "0.0f" if is_f32 else "0.0"
    return (
        f"{{ {ctype} gb_a = ({lhs}) ; {ctype} gb_b = ({rhs}) ; {ctype} gb_q ; "
        # A zero divisor is the one case numpy answers straight from the
        # division: +-inf, or NaN for 0/0.
        f"if (gb_b == {zero}) {{ gb_q = gb_a / gb_b ; }} "
        f"else {{ {ctype} gb_m = fmod{sfx} (gb_a, gb_b) ; "
        f"{ctype} gb_d = (gb_a - gb_m) / gb_b ; "
        # NaN compares false both ways, so a NaN remainder skips the
        # adjustment and carries through to the quotient.
        f"if (gb_m != {zero} && ((gb_b < {zero}) != (gb_m < {zero}))) {{ gb_d -= {one} ; }} "
        f"if (gb_d != {zero}) {{ gb_q = floor{sfx} (gb_d) ; "
        f"if (gb_d - gb_q > {half}) {{ gb_q += {one} ; }} }} "
        f"else {{ gb_q = copysign{sfx} ({zero}, gb_a / gb_b) ; }} }} "
        f"{target} = gb_q ; }}"
    )


def _c_expr_unary(py_op, operand, field_dtype=None):
    """Return a C expression for a unary op: e.g., ``-(x->a)``.

    ``field_dtype`` is only consulted for ``abs``:

    - Float fields use ``fabs`` / ``fabsf`` so ``abs(-0.0)`` returns
      ``+0.0`` (matching Python). The naive ternary preserves the sign bit.
    - Complex fields use ``cabs`` / ``cabsf``: the magnitude (a real). It's
      assigned to a complex field via implicit ``double -> _Complex``
      conversion (``imag = 0``), matching Numba's behavior when Python
      ``abs`` on a complex value is written back to a complex record field.
    - Integer fields keep the ternary; ``abs`` of unsigned is a no-op and
      ``abs`` of signed int wraps on INT_MIN, both matching the cfunc.
    """
    if py_op == "abs":
        if field_dtype is not None:
            if field_dtype.kind == "f":
                fn = "fabsf" if field_dtype.itemsize == 4 else "fabs"
                return f"{fn}({operand})"
            if field_dtype.kind == "c":
                fn = "cabsf" if field_dtype.itemsize == 8 else "cabs"
                return f"{fn}({operand})"
        return f"(({operand}) < 0 ? -({operand}) : ({operand}))"
    if py_op == "-":
        return f"-({operand})"
    raise ValueError(f"Unknown unary C op: {py_op}")


def _make_jit_c_definition(op_name, py_op, dtype, arity):
    """Generate a JIT C function definition for a UDT operator.

    Returns ``(c_name, c_defn)``, or ``None`` if the dtype can't be expressed
    in C (unsupported field types, or names that collide with C reserved words).
    An op not defined on the field types (``min`` on a complex field) never
    gets here: ``_element_types`` rejects it when the op is typed.
    """
    np_type = dtype.np_type
    # Prefer the C name SuiteSparse already has for this type. ``GxB_JIT_C_NAME``
    # on a ``GrB_Type`` is one-shot, so a later ``dtype.name`` rename does not
    # propagate to SS. If the op's signature referenced ``dtype.name`` it would
    # use an undefined struct name, and SS would silently fall back to the
    # Numba cfunc instead of JIT-compiling. ``jit_c_name`` is also where the
    # synthetic ``_gbudt_NNN`` name (for UDTs whose Python name isn't a valid
    # C identifier) is recorded; using it keeps op codegen consistent.
    pinned_name = dtype.jit_c_name
    typedef_info = _udt_c_typedef(pinned_name or dtype.name, dtype.np_type)
    if typedef_info is None:
        return None
    type_name, _typedef = typedef_info
    c_name = f"{op_name}_{type_name}"

    if arity == 2:
        params = f"{type_name} *z, const {type_name} *x, const {type_name} *y"
    else:
        params = f"{type_name} *z, const {type_name} *x"

    if np_type.names is not None:
        # Use leaf C paths so nested records emit ``z->outer.inner_a = ...``
        # alongside the inner+outer typedefs. Non-nested records degenerate
        # to plain ``z->name``.
        leaves = list(_iter_record_leaves(np_type))
        if arity == 2:
            # Pass the leaf dtype to the binary statement builder so
            # type-sensitive ops (min, max, and the divisions) emit correct C.
            assigns = " ".join(
                _c_assign_binary(py_op, f"z->{c}", f"x->{c}", f"y->{c}", leaf_dtype)
                for _py, c, leaf_dtype in leaves
            )
        else:
            assigns = " ".join(
                f"z->{c} = {_c_expr_unary(py_op, f'x->{c}', leaf_dtype)} ;"
                for _py, c, leaf_dtype in leaves
            )
    else:  # array UDT, flattened to v[size]
        base_dtype, shape = np_type.subdtype
        size = reduce(mul, shape)
        # A loop, as in the Numba wrapper: one statement per element made the
        # C compiler take seconds on a 64 by 64 array, once per JIT cache.
        if arity == 2:
            # A statement, not an expression: floating-point ``//`` needs a
            # block with temporaries (see ``_c_assign_binary``).
            assign = _c_assign_binary(py_op, "z->v[i]", "x->v[i]", "y->v[i]", base_dtype)
        else:
            assign = f"z->v[i] = {_c_expr_unary(py_op, 'x->v[i]', base_dtype)} ;"
        assigns = f"for (int64_t i = 0 ; i < {size} ; i++) {{ {assign} }}"
    return c_name, f"void {c_name} ({params}) {{ {assigns} }}"


def _make_jit_c_comparison_definition(op_name, dtype, *, is_eq):
    """Generate a JIT C function definition for ``binary.eq[udt]`` /
    ``binary.ne[udt]``.

    Returns ``(c_name, c_defn)``, or ``None`` when the dtype can't be
    expressed in C. The kernel signature is
    ``void op(_Bool *z, const Udt *x, const Udt *y)``: each leaf field
    contributes a scalar ``==`` (or ``!=``) comparison; record-UDT leaves
    are chained with ``&&`` (eq) or ``||`` (ne). Top-level array UDTs loop
    over their elements.

    Array-valued sub-fields inside a record (e.g.
    ``[("weights", (float64, 3))]``) aren't supported: ``_udt_c_typedef``
    rejects records with array sub-fields (``NP_TO_C_TYPES`` has no entry
    for sub-array dtypes), so this function returns ``None`` before
    iterating leaves in that case. Adding support would mean extending
    ``_udt_c_typedef`` to emit ``double name[N]`` and ``_make_jit_c_definition``
    to unroll array sub-fields in arithmetic codegen too; both legs need
    to land together or the eq/ne path is alone in supporting it.

    IEEE NaN propagation comes for free: C ``a == b`` is false when either
    side is NaN, so two records both carrying NaN compare unequal under
    ``eq`` (and equal under ``ne``), matching the cfunc path's leaf-wise
    semantic comparison.
    """
    np_type = dtype.np_type
    pinned_name = dtype.jit_c_name
    # ``_udt_c_typedef`` returns None when the numpy layout doesn't match a
    # C compiler's layout for the same struct.
    typedef_info = _udt_c_typedef(pinned_name or dtype.name, np_type)
    if typedef_info is None:
        return None
    type_name, _typedef = typedef_info
    c_name = f"{op_name}_{type_name}"
    op = "==" if is_eq else "!="
    join = " && " if is_eq else " || "
    if np_type.subdtype is not None:
        # Array UDT: a loop over every element (see ``_make_jit_c_definition``
        # and the Numba wrapper in ``binary._make_udt_comparison``).
        _base, shape = np_type.subdtype
        size = reduce(mul, shape)
        accumulate = "&=" if is_eq else "|="
        body = (
            f"_Bool r = {int(is_eq)} ; for (int64_t i = 0 ; i < {size} ; i++) "
            f"{{ r {accumulate} ((x->v[i]) {op} (y->v[i])) ; }} *z = r ;"
        )
    elif np_type.names is not None:
        terms = [f"((x->{c}) {op} (y->{c}))" for _py, c, _d in _iter_record_leaves(np_type)]
        body = f"*z = {join.join(terms) if terms else int(is_eq)} ;"
    else:
        return None
    params = f"_Bool *z, const {type_name} *x, const {type_name} *y"
    return c_name, f"void {c_name} ({params}) {{ {body} }}"


def _maybe_warn_jit_skipped(jit_info, op_name, dtype_name):
    """Emit a ``NoJITWarning`` when JIT setup was skipped despite SS supporting JIT.

    Called by op-class ``_compile_udt`` paths after ``set_jit_c_*_on_op``
    returns ``None``. The ``_has_jit_set`` gate keeps the warning silent on
    SS < 8 where JIT is fundamentally absent.
    """
    if jit_info is None and _has_jit_set:
        from ..ss.jit_config import _maybe_warn_no_jit

        _maybe_warn_no_jit(op_name=op_name, dtype_name=dtype_name)


def _set_jit_c_strings(gb_obj, c_name, c_defn, set_string_func):
    """Pin ``GxB_JIT_C_NAME`` + ``GxB_JIT_C_DEFINITION`` on a fresh op handle.

    Both strings are one-shot on SuiteSparse: a second set returns
    ``GrB_ALREADY_SET`` silently. The return is not checked here because
    callers always pass a freshly-allocated handle.

    Arming an op with C source is the point at which this process has
    committed to wanting a JIT compiler, so it is also where SuiteSparse's
    non-compiling default gets raised.
    """
    from ..ss.jit_config import _enable_jit_for_udt

    _enable_jit_for_udt()
    set_string_func(gb_obj, ffi.new("char[]", c_name.encode()), lib.GxB_JIT_C_NAME)
    set_string_func(gb_obj, ffi.new("char[]", c_defn.encode()), lib.GxB_JIT_C_DEFINITION)


def set_jit_c_comparison_on_op(gb_obj, op_name, dtype, set_string_func, *, is_eq):
    """Generate and set JIT C name+definition for ``eq``/``ne`` on a UDT.

    Sibling of :func:`set_jit_c_on_op` for the structurally different
    comparison kernel (BOOL output, chained leaf comparisons). Returns
    ``(c_name, c_defn)`` on success, ``None`` when JIT isn't available or
    the dtype can't be expressed in C; callers cache the returned strings
    on the op for introspection (see ``TypedUserBinaryOp.jit_c_source``).
    """
    if not _has_jit_set:
        return None
    result = _make_jit_c_comparison_definition(op_name, dtype, is_eq=is_eq)
    if result is None:
        return None
    _set_jit_c_strings(gb_obj, *result, set_string_func)
    return result


def set_jit_c_on_op(gb_obj, op_name, py_op, dtype, set_string_func, arity=2):
    """Generate and set JIT C name+definition on a GrB operator, if possible.

    Returns ``(c_name, c_defn)`` when the JIT setters are available and the
    dtype is expressible in C, ``None`` otherwise. Callers may cache the
    returned strings for introspection (see ``TypedUserUnaryOp.jit_c_source``
    and ``TypedUserBinaryOp.jit_c_source``).
    """
    if not _has_jit_set:
        return None
    result = _make_jit_c_definition(op_name, py_op, dtype, arity)
    if result is None:
        return None
    _set_jit_c_strings(gb_obj, *result, set_string_func)
    return result

"""Shared utilities for auto-generating UDT operator implementations.

Provides Numba wrapper generators and JIT C code generators for element-wise
operations on record UDTs (struct types) and array UDTs (e.g., FP64[3]).

Compilation is lazy: functions here are only called from ``_compile_udt``
methods when an operator is first used with a particular UDT.
"""

import ast
import itertools
import linecache
from functools import reduce
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
    namespace = {"min": min, "max": max, "abs": abs}
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
    # sense on complex and are skipped via _op_supports_field_dtype.
    np.dtype(np.complex64): "GxB_FC32_t",
    np.dtype(np.complex128): "GxB_FC64_t",
}

# Ops whose JIT C codegen would produce uncompilable kernels on complex fields
# (no ordering for min/max, no integer-mod for floordiv).
_OPS_NOT_FOR_COMPLEX = frozenset({"min", "max", "floordiv"})

# C operator equivalents for JIT code generation
_C_INFIX_OPS = {"+": "+", "-": "-", "*": "*", "/": "/", "//": "/"}

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


def _iter_record_leaves(np_type, python_prefix="", c_prefix=""):
    """Yield ``(python_access, c_access, leaf_dtype)`` for each leaf in a record.

    A non-nested record yields one entry per field, with paths like
    ``"['a']"`` (Python) and ``"a"`` (C struct). A nested record recurses
    into structured fields: a top-level field ``"outer"`` whose dtype is
    itself a struct with field ``"inner_a"`` yields
    ``("['outer']['inner_a']", "outer.inner_a", float64_dtype)``.

    Only record-in-record nesting is recognised. Array-typed fields are
    yielded as a single leaf (the codegen treats them opaquely; arithmetic
    on an array-typed sub-field isn't supported by either the cfunc or the
    JIT path today).
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


def _op_supports_field_dtypes(op_name, np_type):
    """Return True if the JIT codegen for ``op_name`` can handle every field
    type in ``np_type``.

    Currently the only restriction is complex fields: ``min`` / ``max`` /
    ``floordiv`` have no defined semantics on C99 ``_Complex`` (no ordering,
    no integer mod), so we skip JIT and let SuiteSparse use the Numba cfunc
    instead. The cfunc itself errors on these combinations, so this gate
    only moves the failure earlier and gives a clearer message.

    Recurses into nested record fields so a complex field nested inside an
    outer record is still detected.
    """
    if op_name not in _OPS_NOT_FOR_COMPLEX:
        return True
    if np_type.names is not None:
        return not any(leaf.kind == "c" for _, _, leaf in _iter_record_leaves(np_type))
    if np_type.subdtype is not None:
        return np_type.subdtype[0].kind != "c"
    return True


# Numba function generators, called lazily from each op's ``_compile_udt``.
if _has_numba:

    def _expr_binary(py_op, x_expr, y_expr):
        """Return a Python expression string for a binary op on two expressions."""
        if py_op in _FUNC_BINARY_OPS:
            return f"{py_op}({x_expr}, {y_expr})"
        return f"{x_expr} {py_op} {y_expr}"

    def _expr_unary(py_op, operand):
        """Return a Python expression string for a unary op on a fully-qualified operand."""
        if py_op in _FUNC_UNARY_OPS:
            return f"{py_op}({operand})"
        return f"{py_op}{operand}"

    def _make_record_func(leaf_paths, arity, py_op, *, x_is_scalar=False, y_is_scalar=False):
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
        """
        if arity == 2:
            parts = []
            for path in leaf_paths:
                x_expr = "x" if x_is_scalar else f"x{path}"
                y_expr = "y" if y_is_scalar else f"y{path}"
                parts.append(_expr_binary(py_op, x_expr, y_expr))
            sig = "x, y"
        else:
            parts = [_expr_unary(py_op, f"x{path}") for path in leaf_paths]
            sig = "x"
        body = ", ".join(parts)
        # Single-leaf tuple needs the trailing comma to remain a tuple.
        ret = f"({body},)" if len(leaf_paths) == 1 else f"({body})"
        src = f"def _op({sig}):\n    return {ret}\n"
        op_func = _compile_codegen(
            src,
            func_name="_op",
            source_label=f"<gb-udt {py_op!r} record nleaves={len(leaf_paths)} arity={arity}>",
        )
        return numba.njit(op_func)

    def _make_array_wrapper(
        size,
        base_numba_type,
        arity,
        py_op,
        *,
        x_scalar_type=None,
        y_scalar_type=None,
    ):
        """Build a cfunc-ready wrapper for an array UDT (element-by-element).

        When ``x_scalar_type`` or ``y_scalar_type`` is set, that side is a plain
        scalar pointer (broadcast to all elements).

        Returns (wrapper_func, wrapper_sig).
        """
        nt = numba.types
        if arity == 2:
            x_ref = "x_ptr[0]" if x_scalar_type else "x[{i}]"
            y_ref = "y_ptr[0]" if y_scalar_type else "y[{i}]"
            assigns = "\n".join(
                f"    z[{i}] = {_expr_binary(py_op, x_ref.format(i=i), y_ref.format(i=i))}"
                for i in range(size)
            )
            params = "z_ptr, x_ptr, y_ptr"
            arrays = f"    z = numba.carray(z_ptr, {size})\n"
            if not x_scalar_type:
                arrays += f"    x = numba.carray(x_ptr, {size})\n"
            if not y_scalar_type:
                arrays += f"    y = numba.carray(y_ptr, {size})\n"
            x_numba = nt.CPointer(x_scalar_type) if x_scalar_type else nt.CPointer(base_numba_type)
            y_numba = nt.CPointer(y_scalar_type) if y_scalar_type else nt.CPointer(base_numba_type)
            sig = nt.void(nt.CPointer(base_numba_type), x_numba, y_numba)
        else:
            assigns = "\n".join(
                f"    z[{i}] = {_expr_unary(py_op, f'x[{i}]')}" for i in range(size)
            )
            params = "z_ptr, x_ptr"
            arrays = f"    z = numba.carray(z_ptr, {size})\n    x = numba.carray(x_ptr, {size})\n"
            sig = nt.void(nt.CPointer(base_numba_type), nt.CPointer(base_numba_type))
        src = f"def _op({params}):\n{arrays}{assigns}\n"
        op_func = _compile_codegen(
            src,
            func_name="_op",
            source_label=f"<gb-udt {py_op!r} array size={size} arity={arity}>",
        )
        return op_func, sig

    # MAINT: this function, ``compile_udt_unary_wrapper`` below, and
    # ``_make_jit_c_definition`` all branch on ``kind == "record"`` vs the
    # array path with near-identical scaffolding. When a third op family
    # (ternary, indexbinary, ...) needs the same treatment, fold the three
    # into one shape-parametrized helper instead of pasting a third copy.
    def compile_udt_binary_wrapper(op_name, py_op, dtype, dtype2):
        """Compile a built-in element-wise binary op for a UDT.

        Handles two cases:

        - Both sides are the same UDT: a field-by-field (record) or
          element-by-element (array) op.
        - One side is a UDT and the other is a scalar type: the scalar is
          broadcast to all fields or elements (e.g., ``Point + int`` adds
          the int to every field).

        Returns ``(wrapper_func, wrapper_sig, ret_type)``. Raises ``KeyError``
        when the dtype combination is not supported, with a clear message for
        common mistakes like passing two record UDTs with different field
        names.
        """
        from .base import _get_udt_wrapper, _resolve_udt_return_type

        info_x = _get_udt_info(dtype)
        info_y = _get_udt_info(dtype2)

        # When both sides are UDTs, they must agree on shape. Without this
        # pre-check, generated code like ``x['a'] + y['a']`` fails inside
        # Numba with a cryptic TypingError mentioning record-field internals.
        if info_x is not None and info_y is not None:
            kind_x, detail_x = info_x
            kind_y, detail_y = info_y
            if kind_x != kind_y:
                raise KeyError(
                    f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                    f"cannot mix record and array UDTs in a single element-wise op."
                )
            if kind_x == "record" and detail_x != detail_y:
                raise KeyError(
                    f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                    f"record UDTs must share field names; got {list(detail_x)} vs "
                    f"{list(detail_y)}."
                )
            if kind_x == "array" and detail_x != detail_y:
                raise KeyError(
                    f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                    f"array UDTs must share base dtype and flat size; "
                    f"got {detail_x} vs {detail_y}."
                )

        # Pick the UDT side. Both sides may be UDTs; the pre-check above
        # ensures they share a shape in that case.
        if info_x is not None:
            udt_dtype = dtype
            udt_info = info_x
        elif info_y is not None:
            udt_dtype = dtype2
            udt_info = info_y
        else:
            raise KeyError(
                f"binary.{op_name} does not work with ({dtype}, {dtype2}). "
                f"Element-wise UDT ops require a record dtype (named fields) "
                f"or an array dtype (e.g., FP64[3])."
            )

        # ``min``/``max``/``floordiv`` have no defined semantics on complex
        # operands (no ordering, no integer mod). Reject early with a clear
        # message; otherwise Numba's cfunc compile blows up several frames
        # down with ``NotImplementedError: No definition for lowering lt``.
        if not _op_supports_field_dtypes(op_name, udt_dtype.np_type):
            raise KeyError(
                f"binary.{op_name} does not work with ({dtype}, {dtype2}): "
                f"this op is not defined on complex fields. Use ``binary.plus``, "
                f"``minus``, ``times``, or ``truediv`` for complex element-wise "
                f"arithmetic, or register a custom binary op."
            )

        x_is_scalar = info_x is None  # left side is a plain scalar type
        y_is_scalar = info_y is None  # right side is a plain scalar type
        kind, detail = udt_info

        if kind == "record":
            from .base import _compile_udf_for_udt

            # Use leaf paths so the same codegen handles nested-record UDTs
            # uniformly. A non-nested record's leaves are its top-level
            # fields, with paths like ``"['a']"``.
            leaf_paths = [py for py, _c, _d in _iter_record_leaves(udt_dtype.np_type)]
            func = _make_record_func(
                leaf_paths,
                2,
                py_op,
                x_is_scalar=x_is_scalar,
                y_is_scalar=y_is_scalar,
            )
            sig = (dtype.numba_type, dtype2.numba_type)
            _compile_udf_for_udt(
                func, sig, op_kind="binary", op_name=op_name, dtypes=(dtype, dtype2)
            )
            numba_ret_type = func.overloads[sig].signature.return_type
            ret_type = _resolve_udt_return_type(numba_ret_type, udt_dtype)
            wrapper, wrapper_sig = _get_udt_wrapper(
                func, ret_type, dtype, dtype2, numba_ret_type=numba_ret_type
            )
        else:
            base_dtype, size = detail
            ret_type = udt_dtype
            wrapper, wrapper_sig = _make_array_wrapper(
                size,
                numba.from_dtype(base_dtype),
                2,
                py_op,
                x_scalar_type=numba.from_dtype(dtype.np_type) if x_is_scalar else None,
                y_scalar_type=numba.from_dtype(dtype2.np_type) if y_is_scalar else None,
            )
        return wrapper, wrapper_sig, ret_type

    def compile_udt_unary_wrapper(op_name, py_op, dtype):
        """Compile a built-in element-wise unary op for a UDT.

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

        kind, detail = info
        if kind == "record":
            from .base import _compile_udf_for_udt

            leaf_paths = [py for py, _c, _d in _iter_record_leaves(dtype.np_type)]
            func = _make_record_func(leaf_paths, 1, py_op)
            sig = (dtype.numba_type,)
            _compile_udf_for_udt(func, sig, op_kind="unary", op_name=op_name, dtypes=(dtype,))
            numba_ret_type = func.overloads[sig].signature.return_type
            ret_type = _resolve_udt_return_type(numba_ret_type, dtype)
            wrapper, wrapper_sig = _get_udt_wrapper(
                func, ret_type, dtype, numba_ret_type=numba_ret_type
            )
        else:
            base_dtype, size = detail
            ret_type = dtype
            wrapper, wrapper_sig = _make_array_wrapper(size, numba.from_dtype(base_dtype), 1, py_op)
        return wrapper, wrapper_sig, ret_type


# JIT C code generators below.


def _c_expr_binary(py_op, lhs, rhs, field_dtype=None):
    """Return a C expression for a binary op: e.g., ``(x->a) + (y->a)``.

    ``field_dtype`` is the numpy dtype of the *result* element. It is only
    consulted for ``floordiv`` (``//``), which needs Python ``//`` semantics
    rather than C ``/`` (trunc toward zero for ints, true division for
    floats). Other ops are type-agnostic at the C level.
    """
    if py_op == "min":
        # Match Python ``min(a, b) = b if b < a else a`` so NaN propagates
        # from the first operand (cfunc / numba follows the same rule).
        # The naive ``(a < b ? a : b)`` would silently swallow NaN to the
        # right-hand side and disagree with the cfunc path.
        return f"(({rhs}) < ({lhs}) ? ({rhs}) : ({lhs}))"
    if py_op == "max":
        return f"(({rhs}) > ({lhs}) ? ({rhs}) : ({lhs}))"
    if py_op == "//":
        return _c_floordiv_expr(lhs, rhs, field_dtype)
    c_op = _C_INFIX_OPS.get(py_op, py_op)
    return f"({lhs}) {c_op} ({rhs})"


def _c_floordiv_expr(lhs, rhs, field_dtype):
    """Return a C expression for Python-semantics floor division.

    Python ``//`` is floor (rounds toward negative infinity); C ``/`` is
    trunc toward zero for ints and true division for floats. The two only
    agree for non-negative integer operands; for everything else the JIT
    path silently disagreed with the Numba cfunc path before this helper.

    Float fields use ``floor()`` / ``floorf()`` from ``<math.h>``, which is
    available in the JIT kernel via SuiteSparse's include chain
    (``GraphBLAS.h`` -> ``<math.h>``). Signed integer fields use the
    standard trunc-to-floor adjustment. Unsigned integers don't need
    adjusting because both operands are non-negative.
    """
    if field_dtype is None:
        # Caller didn't pass dtype info. The C ``/`` semantics match Python
        # ``//`` for non-negative integer operands only.
        return f"({lhs}) / ({rhs})"
    kind = field_dtype.kind
    if kind == "f":
        if field_dtype.itemsize == 4:
            return f"floorf((float)({lhs}) / (float)({rhs}))"
        return f"floor((double)({lhs}) / (double)({rhs}))"
    if kind in ("u", "b"):
        return f"({lhs}) / ({rhs})"
    # Signed integer: trunc-toward-zero is one greater than floor when the
    # signs of ``a`` and ``b`` differ and the division has a non-zero
    # remainder; subtract 1 in that case.
    return f"(({lhs}) / ({rhs}) - ((({lhs}) % ({rhs}) != 0) && ((({lhs}) < 0) != (({rhs}) < 0))))"


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
    in C (unsupported field types, or names that collide with C reserved words),
    or if the op isn't defined on the dtype's field types (e.g. ``min`` on a
    complex field).
    """
    np_type = dtype.np_type
    if not _op_supports_field_dtypes(op_name, np_type):
        return None
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
            # Pass the leaf dtype to the binary expression builder so
            # type-sensitive ops (currently floordiv) can emit correct C.
            assigns = " ".join(
                f"z->{c} = {_c_expr_binary(py_op, f'x->{c}', f'y->{c}', leaf_dtype)} ;"
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
        if arity == 2:
            assigns = " ".join(
                f"z->v[{i}] = {_c_expr_binary(py_op, f'x->v[{i}]', f'y->v[{i}]', base_dtype)} ;"
                for i in range(size)
            )
        else:
            assigns = " ".join(
                f"z->v[{i}] = {_c_expr_unary(py_op, f'x->v[{i}]', base_dtype)} ;"
                for i in range(size)
            )
    return c_name, f"void {c_name} ({params}) {{ {assigns} }}"


def _make_jit_c_comparison_definition(op_name, dtype, *, is_eq):
    """Generate a JIT C function definition for ``binary.eq[udt]`` /
    ``binary.ne[udt]``.

    Returns ``(c_name, c_defn)``, or ``None`` when the dtype can't be
    expressed in C. The kernel signature is
    ``void op(_Bool *z, const Udt *x, const Udt *y)``: each leaf field
    contributes a scalar ``==`` (or ``!=``) comparison; record-UDT leaves
    are chained with ``&&`` (eq) or ``||`` (ne). Array UDTs and array
    sub-fields unroll their elements at codegen time.

    IEEE NaN propagation comes for free: C ``a == b`` is false when either
    side is NaN, so two records both carrying NaN compare unequal under
    ``eq`` (and equal under ``ne``), matching the cfunc path's leaf-wise
    semantic comparison.
    """
    np_type = dtype.np_type
    if np_type.names is not None and not _is_c_compatible_layout(np_type):
        return None
    pinned_name = dtype.jit_c_name
    typedef_info = _udt_c_typedef(pinned_name or dtype.name, np_type)
    if typedef_info is None:
        return None
    type_name, _typedef = typedef_info
    c_name = f"{op_name}_{type_name}"
    op = "==" if is_eq else "!="
    join = " && " if is_eq else " || "
    terms = []
    if np_type.subdtype is not None:
        # Array UDT: unroll all elements.
        _base, shape = np_type.subdtype
        size = reduce(mul, shape)
        terms = [f"((x->v[{i}]) {op} (y->v[{i}]))" for i in range(size)]
    elif np_type.names is not None:
        for _py, c_path, leaf_dtype in _iter_record_leaves(np_type):
            if leaf_dtype.subdtype is not None:
                # Array-valued sub-field inside a record: unroll its
                # elements and combine with the same connective.
                base_dtype, shape = leaf_dtype.subdtype
                if base_dtype not in NP_TO_C_TYPES:
                    return None
                size = reduce(mul, shape)
                arr_terms = [f"((x->{c_path}[{i}]) {op} (y->{c_path}[{i}]))" for i in range(size)]
                terms.append("(" + join.join(arr_terms) + ")")
            else:
                terms.append(f"((x->{c_path}) {op} (y->{c_path}))")
    else:
        return None
    body = join.join(terms) if terms else ("1" if is_eq else "0")
    params = f"_Bool *z, const {type_name} *x, const {type_name} *y"
    return c_name, f"void {c_name} ({params}) {{ *z = {body} ; }}"


def set_jit_c_comparison_on_op(gb_obj, op_name, dtype, set_string_func, *, is_eq):
    """Generate and set JIT C name+definition for ``eq``/``ne`` on a UDT.

    Sibling of :func:`set_jit_c_on_op` for the structurally different
    comparison kernel (BOOL output, chained leaf comparisons). Returns
    ``(c_name, c_defn)`` on success, ``None`` when JIT isn't available or
    the dtype can't be expressed in C; callers cache the returned strings
    on the op for introspection (see ``TypedUserBinaryOp.jit_c_source``).

    ``gb_obj`` must be a freshly-created ``GrB_BinaryOp``: ``GxB_JIT_C_NAME``
    and ``GxB_JIT_C_DEFINITION`` are one-shot on SuiteSparse, so a second
    set returns ``GrB_ALREADY_SET`` silently (the return is not checked
    here).
    """
    if not _has_jit_set:
        return None
    result = _make_jit_c_comparison_definition(op_name, dtype, is_eq=is_eq)
    if result is None:
        return None
    c_name, c_defn = result
    set_string_func(gb_obj, ffi.new("char[]", c_name.encode()), lib.GxB_JIT_C_NAME)
    set_string_func(gb_obj, ffi.new("char[]", c_defn.encode()), lib.GxB_JIT_C_DEFINITION)
    return c_name, c_defn


def set_jit_c_on_op(gb_obj, op_name, py_op, dtype, set_string_func, arity=2):
    """Generate and set JIT C name+definition on a GrB operator, if possible.

    Returns ``(c_name, c_defn)`` when the JIT setters are available and the
    dtype is expressible in C, ``None`` otherwise. Callers may cache the
    returned strings for introspection (see ``TypedUserUnaryOp.jit_c_source``
    and ``TypedUserBinaryOp.jit_c_source``).

    ``gb_obj`` must be a freshly-created ``GrB_UnaryOp`` / ``GrB_BinaryOp``:
    ``GxB_JIT_C_NAME`` and ``GxB_JIT_C_DEFINITION`` are one-shot on
    SuiteSparse, so a second set returns ``GrB_ALREADY_SET`` silently
    (the return is not checked here).
    """
    if not _has_jit_set:
        return None
    result = _make_jit_c_definition(op_name, py_op, dtype, arity)
    if result is None:
        return None
    c_name, c_defn = result
    set_string_func(gb_obj, ffi.new("char[]", c_name.encode()), lib.GxB_JIT_C_NAME)
    set_string_func(gb_obj, ffi.new("char[]", c_defn.encode()), lib.GxB_JIT_C_DEFINITION)
    return c_name, c_defn

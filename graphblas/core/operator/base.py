import re
from functools import lru_cache
from operator import getitem
from types import BuiltinFunctionType, ModuleType

from ... import _STANDARD_OPERATOR_NAMES, backend, op
from ...dtypes import BOOL, INT8, UINT64, _supports_complex, lookup_dtype
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, _supports_udfs, ffi, lib
from ..expr import InfixExprBase
from ..utils import output_type

if _has_numba:
    import numba
    from numba import NumbaError
else:
    NumbaError = TypeError

UNKNOWN_OPCLASS = "UnknownOpClass"

# These now live as e.g. `gb.unary.ss.positioni`
# Deprecations such as `gb.unary.positioni` will be removed in 2023.9.0 or later.
_SS_OPERATORS = {
    # unary
    "erf",  # scipy.special.erf
    "erfc",  # scipy.special.erfc
    "frexpe",  # np.frexp[1]
    "frexpx",  # np.frexp[0]
    "lgamma",  # scipy.special.loggamma
    "tgamma",  # scipy.special.gamma
    # Positional
    # unary
    "positioni",
    "positioni1",
    "positionj",
    "positionj1",
    # binary
    "firsti",
    "firsti1",
    "firstj",
    "firstj1",
    "secondi",
    "secondi1",
    "secondj",
    "secondj1",
    # semiring
    "any_firsti",
    "any_firsti1",
    "any_firstj",
    "any_firstj1",
    "any_secondi",
    "any_secondi1",
    "any_secondj",
    "any_secondj1",
    "max_firsti",
    "max_firsti1",
    "max_firstj",
    "max_firstj1",
    "max_secondi",
    "max_secondi1",
    "max_secondj",
    "max_secondj1",
    "min_firsti",
    "min_firsti1",
    "min_firstj",
    "min_firstj1",
    "min_secondi",
    "min_secondi1",
    "min_secondj",
    "min_secondj1",
    "plus_firsti",
    "plus_firsti1",
    "plus_firstj",
    "plus_firstj1",
    "plus_secondi",
    "plus_secondi1",
    "plus_secondj",
    "plus_secondj1",
    "times_firsti",
    "times_firsti1",
    "times_firstj",
    "times_firstj1",
    "times_secondi",
    "times_secondi1",
    "times_secondj",
    "times_secondj1",
}


def _hasop(module, name):
    return (
        name in module.__dict__
        or name in module._delayed
        or name in getattr(module, "_deprecated", ())
    )


def _bool_to_int8(dtype):
    """Return INT8 if ``dtype`` is BOOL, else the dtype unchanged.

    Numba can't compile cfuncs that read or write ``CPointer(boolean)``
    (errors like ``cannot store i1 to i8*`` / ``cond is not i1: i8``); see
    numba/numba#5395. Routing BOOL through INT8 sidesteps that; GraphBLAS
    coerces back at the cfunc boundary.

    MAINT 2026-05-24: still hits on Numba 0.65. Re-test periodically and
    drop the INT8 routing when upstream is fixed.
    """
    return INT8 if dtype == BOOL else dtype


class OpPath:
    def __init__(self, parent, name):
        self._parent = parent
        self._name = name
        self._delayed = {}
        self._delayed_commutes_to = {}

    def __getattr__(self, key):
        if key in self._delayed:
            func, kwargs = self._delayed.pop(key)
            return func(**kwargs)
        self.__getattribute__(key)  # raises


def _call_op(op, left, right=None, thunk=None, **kwargs):
    if right is None and thunk is None:
        if isinstance(left, InfixExprBase):
            # op(A & B), op(A | B), op(A @ B)
            return getattr(left.left, f"_{left.method_name}")(
                left.right, op, is_infix=True, **kwargs
            )
        if find_opclass(op)[1] == "Semiring":
            raise TypeError(
                f"Bad type when calling {op!r}.  Got type: {type(left)}.\n"
                f"Expected an infix expression, such as: {op!r}(A @ B)"
            )
        raise TypeError(
            f"Bad type when calling {op!r}.  Got type: {type(left)}.\n"
            "Expected an infix expression or an apply with a Vector or Matrix and a scalar:\n"
            f"    - {op!r}(A & B)\n"
            f"    - {op!r}(A, 1)\n"
            f"    - {op!r}(1, A)"
        )

    # op(A, 1) -> apply (or select if thunk provided)
    from ..matrix import Matrix, TransposedMatrix
    from ..vector import Vector

    if (left_type := output_type(left)) in {Vector, Matrix, TransposedMatrix}:
        if thunk is not None:
            return left.select(op, thunk=thunk, **kwargs)
        return left.apply(op, right=right, **kwargs)
    if (right_type := output_type(right)) in {Vector, Matrix, TransposedMatrix}:
        return right.apply(op, left=left, **kwargs)

    from ..scalar import Scalar, _as_scalar

    if left_type is Scalar:
        if thunk is not None:
            return left.select(op, thunk=thunk, **kwargs)
        return left.apply(op, right=right, **kwargs)
    if right_type is Scalar:
        return right.apply(op, left=left, **kwargs)
    try:
        left_scalar = _as_scalar(left, is_cscalar=False)
    except Exception:
        pass
    else:
        if thunk is not None:
            return left_scalar.select(op, thunk=thunk, **kwargs)
        return left_scalar.apply(op, right=right, **kwargs)
    raise TypeError(
        f"Bad types when calling {op!r}.  Got types: {type(left)}, {type(right)}.\n"
        "Expected an infix expression or an apply with a Vector or Matrix and a scalar:\n"
        f"    - {op!r}(A & B)\n"
        f"    - {op!r}(A, 1)\n"
        f"    - {op!r}(1, A)"
    )


if _has_numba:

    def _finalize_udt_op(parent_op, dtype, dtype2, ret_type, wrapper, wrapper_sig, typed_user_cls):
        """Compile the cfunc, allocate the ``GrB`` op, wrap it, and cache it.

        Shared tail for ``_compile_udt`` in UnaryOp / BinaryOp / IndexUnaryOp /
        SelectOp. Looks up the SuiteSparse handle type and ``_new`` symbol
        from ``typed_user_cls.opclass``. ``dtype2`` is ``None`` for unary
        ops; the rest pass both. Returns the cached ``TypedUser*Op``.
        """
        wrapper = numba.cfunc(wrapper_sig, nopython=True)(wrapper)
        c_typename = _GB_OBJ_C_TYPENAME[typed_user_cls.opclass]
        error_label = c_typename.removeprefix("GrB_").removeprefix("GxB_")
        gb_obj = ffi.new(f"{c_typename}*")
        new_func = getattr(lib, f"{c_typename}_new")
        if dtype2 is None:
            check_status_carg(
                new_func(gb_obj, wrapper.cffi, ret_type._carg, dtype._carg),
                error_label,
                gb_obj[0],
            )
            op = typed_user_cls(parent_op, parent_op.name, dtype, ret_type, gb_obj[0])
            key = dtype
        else:
            check_status_carg(
                new_func(gb_obj, wrapper.cffi, ret_type._carg, dtype._carg, dtype2._carg),
                error_label,
                gb_obj[0],
            )
            op = typed_user_cls(
                parent_op, parent_op.name, dtype, ret_type, gb_obj[0], dtype2=dtype2
            )
            key = (dtype, dtype2)
        parent_op._udt_types[key] = ret_type
        parent_op._udt_ops[key] = op
        return op

    def _compile_udf_for_udt(numba_func, sig, *, op_kind, op_name, dtypes):
        """Compile ``sig`` and re-raise Numba compilation errors as ``UdfParseError``.

        Catches the full ``NumbaError`` hierarchy (TypingError, LoweringError,
        UnsupportedError, ...) so any compilation failure produces a
        single-line UDT diagnostic.
        """
        try:
            numba_func.compile(sig)
        except NumbaError as exc:
            dtypes_str = ", ".join(str(d) for d in dtypes)
            snippet = _summarize_numba_typing_error(exc)
            raise UdfParseError(
                f"{op_kind}.{op_name} does not work with ({dtypes_str}): {snippet}"
            ) from exc

    # Numba prefixes its actionable diagnostic line with one of these.
    _NUMBA_DIAG_PREFIXES = (
        "No implementation of function",
        "No conversion from",
        "Field ",
        "Cannot infer",
        "Operator Overload",
        "Invalid use of",
        "use of undeclared",
        "Untyped global",
    )

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _summarize_numba_typing_error(exc):
        """Pull the most actionable lines out of a Numba TypingError.

        Numba's "No implementation of function ... found for signature:"
        diagnostic puts the signature on the line *below* the prefix; the
        signature is the actionable part. When the matched line ends with a
        ``:``, append the next non-empty line so the user sees both.
        """
        lines = [_ANSI_RE.sub("", line).strip() for line in str(exc).splitlines()]
        for i, line in enumerate(lines):
            if line.startswith(_NUMBA_DIAG_PREFIXES):
                if line.endswith(":"):
                    for follow in lines[i + 1 :]:
                        if follow:
                            return f"{line} {follow}"
                return line
        for line in lines:
            if line:
                return line
        return "Numba could not compile the function for these input types"

    def _resolve_udt_return_type(numba_ret_type, *dtypes):
        """Resolve a Numba return type to a DataType, matching Tuple returns to an input UDT.

        When a UDF returns a tuple, Numba infers ``Tuple(...)`` rather than a
        Record type. Match by field count, preferring a candidate whose field
        types align with the Tuple's element types.
        """
        try:
            return lookup_dtype(numba_ret_type)
        except (ValueError, TypeError):
            pass
        if isinstance(numba_ret_type, numba.core.types.BaseTuple):
            from .udt_utils import _iter_record_leaves

            n = len(numba_ret_type.types)

            def _leaves(d):
                """Yield ``(python_path, c_path, leaf_dtype)`` for a record UDT."""
                return list(_iter_record_leaves(d.np_type))

            # Match by total leaf count (flat for shallow records, total
            # leaves across nesting for nested records).
            same_arity = [
                d
                for d in dtypes
                if d._is_udt and d.np_type.names is not None and len(_leaves(d)) == n
            ]
            # Prefer a UDT whose leaf types match the tuple elements element-wise.
            for d in same_arity:
                leaf_dtypes = [leaf for _py, _c, leaf in _leaves(d)]
                try:
                    expected = [lookup_dtype(t).numba_type for t in leaf_dtypes]
                except (ValueError, TypeError):
                    continue
                if all(et == tt for et, tt in zip(expected, numba_ret_type.types, strict=True)):
                    return d
            # No perfect match; fall back to the first arity-compatible UDT.
            if same_arity:
                return same_arity[0]
            # Tuple return whose arity matches no input UDT: most likely the
            # user is returning the wrong number of fields. List the candidate
            # arities so the fix is obvious.
            record_arities = sorted(
                {len(_leaves(d)) for d in dtypes if d._is_udt and d.np_type.names is not None}
            )
            if record_arities:
                expected = " or ".join(str(a) for a in record_arities)
                raise UdfParseError(
                    f"UDT UDF returned a tuple of length {n}; expected {expected} "
                    f"to match one of the input record UDTs."
                )
            # All UDT inputs are array UDTs. Tuples don't map to those: the
            # function should return a numpy array of the right shape (or a
            # scalar) instead.
            array_inputs = [d for d in dtypes if d._is_udt and d.np_type.subdtype is not None]
            if array_inputs:
                shape = array_inputs[0].np_type.subdtype[1]
                raise UdfParseError(
                    f"UDT UDF returned a tuple of length {n}, but inputs are array UDTs of "
                    f"shape {shape}. Return a numpy array (e.g., ``np.array(...)``) or a "
                    f"scalar; tuple returns are only matched to record UDTs."
                )
        raise UdfParseError(
            f"UDT UDF returned an unsupported type {numba_ret_type!r}. "
            f"Return a scalar, a tuple matching a record UDT's fields, or a numpy array "
            f"matching an array UDT's shape."
        )

    def _input_operand(dtype, var):
        """Return ``(setup_line, deref_expr, ptr_arg_type)`` for one input operand.

        - ``setup_line`` is the optional ``var = numba.carray(var_ptr, 1)`` line
          to add to the wrapper body (empty for non-record cases).
        - ``deref_expr`` is the value to pass to ``numba_func`` for this operand.
        - ``ptr_arg_type`` is the Numba ``CPointer(...)`` type for the wrapper signature.
        """
        nt = numba.types
        if dtype._is_udt:
            if dtype.np_type.subdtype is None:
                return (
                    f"    {var} = numba.carray({var}_ptr, 1)\n",
                    f"{var}[0]",
                    nt.CPointer(dtype.numba_type),
                )
            # Array UDT: pass the raw pointer. The UDF carrays it if it wants.
            return "", f"{var}_ptr", nt.CPointer(dtype.numba_type)
        if dtype == BOOL:
            # Numba can't compile bool ptrs (numba/numba#5395); expose them
            # as int8 and cast on deref.
            # MAINT 2026-05-24: still hits on Numba 0.65; re-test periodically.
            return "", f"bool({var}_ptr[0])", nt.CPointer(INT8.numba_type)
        return "", f"{var}_ptr[0]", nt.CPointer(dtype.numba_type)

    def _output_handler(return_type, numba_ret_type):
        """Return ``(setup_line, ret_ptr_type, write_kind, write_info)``.

        ``write_kind`` is ``"record_fields"`` when the UDF returns a Tuple to be
        unpacked into a record output, otherwise ``"direct"``.
        ``write_info`` is the tuple-unpack field tuple for ``"record_fields"``,
        or a ``(BL, BR, zname)`` 3-tuple for ``"direct"``.
        """
        nt = numba.types
        ztype = INT8 if return_type == BOOL else return_type
        ret_ptr_type = nt.CPointer(ztype.numba_type)

        if (
            numba_ret_type is not None
            and isinstance(numba_ret_type, numba.core.types.BaseTuple)
            and return_type._is_udt
            and return_type.np_type.names is not None
        ):
            # ``write_info`` is the list of Python access paths to each leaf
            # field (``"['a']"`` for flat, ``"['outer']['inner_a']"`` for
            # nested). The wrapper iterates these to write the flat tuple
            # ``_result`` back leaf-by-leaf.
            from .udt_utils import _iter_record_leaves

            leaf_python_paths = tuple(py for py, _c, _d in _iter_record_leaves(return_type.np_type))
            return (
                "    z = numba.carray(z_ptr, 1)\n",
                ret_ptr_type,
                "record_fields",
                leaf_python_paths,
            )
        if return_type._is_udt:
            if return_type.np_type.subdtype is None:
                return (
                    "    z = numba.carray(z_ptr, 1)\n",
                    ret_ptr_type,
                    "direct",
                    ("", "", "z[0]"),
                )
            # Array UDT: write via z_ptr[0]. The UDF receives the carray and
            # writes in place.
            return "", ret_ptr_type, "direct", ("", "[0]", "z_ptr[0]")
        if return_type == BOOL:
            return "", ret_ptr_type, "direct", ("bool(", ")", "z_ptr[0]")
        return "", ret_ptr_type, "direct", ("", "", "z_ptr[0]")

    def _compose_wrapper_body(zkind, zinfo, signature_line, body_setup, call_expr):
        """Assemble the Python source for a UDT cfunc wrapper.

        For record returns, the wrapper writes leaf-by-leaf; Numba does not
        compile a nested-tuple assignment to an outer record field.
        """
        if zkind == "record_fields":
            field_assigns = "".join(
                f"    z[0]{path} = _result[{i}]\n" for i, path in enumerate(zinfo)
            )
            return (
                f"{signature_line}\n"
                f"{body_setup}"
                f"    _result = {call_expr}\n"
                f"{field_assigns}"
            )
        BL, BR, zname = zinfo
        return f"{signature_line}\n{body_setup}    {zname} = {BL}{call_expr}{BR}\n"

    def _get_udt_wrapper(
        numba_func, return_type, dtype, dtype2=None, *, include_indexes=False, numba_ret_type=None
    ):
        """Build a Numba cfunc wrapper for unary, binary, indexunary, or select UDFs on UDTs.

        ``include_indexes=True`` inserts ``(row, col)`` between ``x`` and
        ``y`` in both the wrapper signature and the call to ``numba_func``,
        matching the IndexUnaryOp and SelectOp shape. The IndexBinaryOp path
        (four indices plus a theta operand) uses
        :func:`_get_udt_wrapper_indexbinary`.
        """
        nt = numba.types
        zsetup, zptr_type, zkind, zinfo = _output_handler(return_type, numba_ret_type)
        xsetup, xderef, xptr_type = _input_operand(dtype, "x")
        wrapper_args = [zptr_type, xptr_type]
        if include_indexes:
            wrapper_args.extend([UINT64.numba_type, UINT64.numba_type])
        ysetup, yderef_expr, yarg = "", "", ""
        if dtype2 is not None:
            ysetup, yderef, yptr_type = _input_operand(dtype2, "y")
            wrapper_args.append(yptr_type)
            yarg = ", y_ptr"
            yderef_expr = f", {yderef}"
        wrapper_sig = nt.void(*wrapper_args)

        rcidx = ", row, col" if include_indexes else ""
        signature_line = f"def wrapper(z_ptr, x_ptr{rcidx}{yarg}):"
        body_setup = f"{zsetup}{xsetup}{ysetup}"
        call_expr = f"numba_func({xderef}{rcidx}{yderef_expr})"

        text = _compose_wrapper_body(zkind, zinfo, signature_line, body_setup, call_expr)
        from .udt_utils import _compile_codegen

        kind = "indexunary" if include_indexes else ("binary" if dtype2 is not None else "unary")
        wrapper = _compile_codegen(
            text,
            func_name="wrapper",
            source_label=f"<gb-udt-wrapper {kind} dtype={dtype} ret={return_type}>",
            extra_ns={"numba_func": numba_func},
        )
        return wrapper, wrapper_sig

    def _get_udt_wrapper_indexbinary(
        numba_func, return_type, dtype, dtype2, *, numba_ret_type=None
    ):
        """Build a Numba cfunc wrapper for IndexBinaryOp UDFs on UDTs.

        Signature: ``f(x, ix, jx, y, iy, jy, theta) -> z``. ``dtype2`` is the
        shared type of ``y`` and ``theta``.
        """
        nt = numba.types
        zsetup, zptr_type, zkind, zinfo = _output_handler(return_type, numba_ret_type)
        xsetup, xderef, xptr_type = _input_operand(dtype, "x")
        ysetup, yderef, yptr_type = _input_operand(dtype2, "y")
        tsetup, tderef, tptr_type = _input_operand(dtype2, "t")
        wrapper_sig = nt.void(
            zptr_type,
            xptr_type,
            UINT64.numba_type,
            UINT64.numba_type,
            yptr_type,
            UINT64.numba_type,
            UINT64.numba_type,
            tptr_type,
        )

        signature_line = "def wrapper(z_ptr, x_ptr, ix, jx, y_ptr, iy, jy, t_ptr):"
        body_setup = f"{zsetup}{xsetup}{ysetup}{tsetup}"
        call_expr = f"numba_func({xderef}, ix, jx, {yderef}, iy, jy, {tderef})"

        text = _compose_wrapper_body(zkind, zinfo, signature_line, body_setup, call_expr)
        from .udt_utils import _compile_codegen

        wrapper = _compile_codegen(
            text,
            func_name="wrapper",
            source_label=f"<gb-udt-wrapper indexbinary dtype={dtype} dtype2={dtype2}>",
            extra_ns={"numba_func": numba_func},
        )
        return wrapper, wrapper_sig


# Maps ``opclass`` to the SuiteSparse C type name SS allocates for it.
# ``IndexBinaryOp`` is in the GxB_ namespace (SS-specific, added in 9.4);
# the rest are GrB_. ``SelectOp`` is implemented on top of
# ``GrB_IndexUnaryOp`` (BOOL-returning), so its handle frees through the
# same ``GrB_IndexUnaryOp_free``. Used by ``TypedOpBase.__del__`` to
# synthesize the pointer cell for ``<C type>_free``.
_GB_OBJ_C_TYPENAME = {
    "UnaryOp": "GrB_UnaryOp",
    "BinaryOp": "GrB_BinaryOp",
    "IndexUnaryOp": "GrB_IndexUnaryOp",
    "SelectOp": "GrB_IndexUnaryOp",
    "IndexBinaryOp": "GxB_IndexBinaryOp",
    "Monoid": "GrB_Monoid",
    "Semiring": "GrB_Semiring",
}


class TypedOpBase:
    __slots__ = (
        "parent",
        "name",
        "type",
        "return_type",
        "gb_obj",
        "gb_name",
        "_type2",
        "_jit_c_info",
        "_owns_gb_obj_inst",
        "__weakref__",
    )
    # Subclasses whose ``gb_obj`` was allocated via ``GrB_<Type>_new`` /
    # ``GxB_<Type>_new`` (TypedUser*Op, _BoundIndexBinaryOp) override this so
    # ``__del__`` frees the SuiteSparse handle. Built-in typed ops point at
    # SuiteSparse's permanent built-in singletons and must never free.
    # Specific instances can override via ``_owns_gb_obj_inst`` (set by
    # the constructor); ``SelectOp._from_indexunary`` aliases an existing
    # ``GrB_IndexUnaryOp`` and must clear ownership to avoid a double free.
    _owns_gb_obj = False

    def __init__(self, parent, name, type_, return_type, gb_obj, gb_name, dtype2=None):
        self.parent = parent
        self.name = name
        self.type = type_
        self.return_type = return_type
        self.gb_obj = gb_obj
        self.gb_name = gb_name
        self._type2 = dtype2
        # ``(c_name, c_definition)`` when SuiteSparse JIT-compiled a kernel
        # for this typed op; ``None`` for built-in ops and for UDT ops with
        # no JIT path.
        self._jit_c_info = None
        # Per-instance ownership override; defaults to the class attribute.
        # ``SelectOp._from_indexunary`` flips this to ``False`` on aliasing
        # TypedUserSelectOps so only the IndexUnaryOp frees the handle.
        self._owns_gb_obj_inst = type(self)._owns_gb_obj

    @property
    def jit_c_name(self):
        """The C symbol name SuiteSparse uses for this op's JIT kernel, or ``None``."""
        return self._jit_c_info[0] if self._jit_c_info is not None else None

    @property
    def jit_c_source(self):
        """C source SuiteSparse JIT-compiles for this op, or ``None`` when no JIT kernel exists."""
        return self._jit_c_info[1] if self._jit_c_info is not None else None

    def __repr__(self):
        classname = self.opclass.lower()
        classname = classname.removesuffix("op")
        dtype2 = "" if self._type2 is None else f", {self._type2.name}"
        return f"{classname}.{self.name}[{self.type.name}{dtype2}]"

    @property
    def _carg(self):
        return self.gb_obj

    @property
    def is_positional(self):
        return self.parent.is_positional

    def __reduce__(self):
        if self._type2 is None or self.type == self._type2:
            return (getitem, (self.parent, self.type))
        return (getitem, (self.parent, (self.type, self._type2)))

    def __del__(self):
        # Free the SuiteSparse handle we allocated. Built-in typed ops alias
        # SuiteSparse's permanent built-in singletons and must never free, so
        # gate on the per-instance owns flag (defaults to the class
        # attribute; the alias case overrides to False). Mirrors the
        # ``Matrix.__del__`` / ``Vector.__del__`` pattern.
        if not getattr(self, "_owns_gb_obj_inst", False):
            return
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is None or lib is None or ffi is None:
            # Interpreter shutdown can clear ``lib`` / ``ffi`` before
            # finalizers run; SS will clean up the handles at process exit.
            return
        c_type_name = _GB_OBJ_C_TYPENAME.get(self.opclass)
        if c_type_name is None:  # pragma: no cover (defensive)
            return
        free_fn = getattr(lib, f"{c_type_name}_free", None)
        if free_fn is None:
            # ``GxB_IndexBinaryOp_free`` is absent on SS < 9.4; that build
            # also can't allocate one in the first place, so this path is
            # unreachable in practice but guarded for safety.
            return
        # ``GrB_<Type>_free`` takes a pointer-to-pointer (sets ``*p = NULL``
        # after free). Synthesize a cell pointing at our handle and call it.
        free_fn(ffi.new(f"{c_type_name}*", gb_obj))


class _BinaryopJitDelegate:
    """Mixin for ops that don't own a JIT kernel; defer introspection to ``binaryop``.

    Monoids and semirings reuse the binary op's JIT kernel. The setter
    accepts ``None`` so ``TypedOpBase.__init__``'s slot init is a no-op.
    """

    __slots__ = ()

    @property
    def _jit_c_info(self):
        return self.binaryop._jit_c_info

    @_jit_c_info.setter
    def _jit_c_info(self, value):
        # No-op so ``TypedOpBase.__init__``'s ``self._jit_c_info = None``
        # slot-init succeeds. The kernel lives on ``binaryop``.
        pass


def _deserialize_parameterized(parameterized_op, args, kwargs):
    return parameterized_op(*args, **kwargs)


class ParameterizedUdf:
    __slots__ = "name", "__call__", "_anonymous", "__weakref__"
    is_positional = False
    _custom_dtype = None
    # Subclasses set this to the OpBase subclass they parameterize (e.g.,
    # ``ParameterizedUnaryOp._op_class = UnaryOp``). Assigned after the
    # OpBase subclass is defined to avoid an import-order cycle.
    _op_class = None

    def __init__(self, name, anonymous):
        self.name = name
        self._anonymous = anonymous
        # lru_cache per instance
        method = self._call.__get__(self, type(self))
        self.__call__ = lru_cache(maxsize=1024)(method)

    def _call(self, *args, **kwargs):
        raise NotImplementedError

    def __reduce__(self):
        # The namespace prefix (``unary``, ``binary``, ...) comes from the
        # OpBase subclass each parameterized op wraps. Standard ops pickle by
        # name; user-registered ones pickle the reduce tuple and re-register
        # on load via ``_deserialize`` (which dispatches through ``_op_class``).
        name = f"{self._op_class._modname}.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous, self._is_udt))

    @classmethod
    def _deserialize(cls, name, func, anonymous, is_udt=False):
        """Re-register a parameterized UDF on unpickle, or reuse if already present.

        Shared by the five ``Parameterized*Op`` subclasses; each sets
        ``_op_class`` to the matching OpBase subclass for the dispatch below.
        """
        op_cls = cls._op_class
        if anonymous:
            return op_cls.register_anonymous(func, name, parameterized=True, is_udt=is_udt)
        if (rv := op_cls._find(name)) is not None:
            return rv
        return op_cls.register_new(name, func, parameterized=True, is_udt=is_udt)


_VARNAMES = tuple(x for x in dir(lib) if x[0] != "_")


class OpBase:
    __slots__ = (
        "name",
        "_typed_ops",
        "types",
        "coercions",
        "_anonymous",
        "_udt_types",
        "_udt_ops",
        "__weakref__",
    )
    _parse_config = None
    _initialized = False
    _module = None
    _positional = None

    def __init__(self, name, *, anonymous=False):
        self.name = name
        self._typed_ops = {}
        self.types = {}
        self.coercions = {}
        self._anonymous = anonymous
        self._udt_types = None
        self._udt_ops = None

    def __repr__(self):
        return f"{self._modname}.{self.name}"

    def __getitem__(self, type_):
        if type(type_) is tuple:
            from .utils import get_typed_op

            dtype1, dtype2 = type_
            dtype1 = lookup_dtype(dtype1)
            dtype2 = lookup_dtype(dtype2)
            return get_typed_op(self, dtype1, dtype2)
        if not self._is_udt:
            type_ = lookup_dtype(type_)
            if type_ not in self._typed_ops:
                if self._udt_types is None:
                    if self.is_positional:
                        return self._typed_ops[UINT64]
                    raise KeyError(f"{self.name} does not work with {type_}")
            else:
                return self._typed_ops[type_]
        # This is a UDT or is able to operate on UDTs such as `first` any `any`
        dtype = lookup_dtype(type_)
        return self._compile_udt(dtype, dtype)

    def _add(self, op, *, is_jit=False):
        if is_jit:
            if hasattr(op, "type2") or hasattr(op, "thunk_type"):
                dtypes = (op.type, op._type2)
            else:
                dtypes = op.type
            self.types[dtypes] = op.return_type  # This is a different use of .types
            self._udt_types[dtypes] = op.return_type
            self._udt_ops[dtypes] = op
        else:
            self._typed_ops[op.type] = op
            self.types[op.type] = op.return_type

    def __delitem__(self, type_):
        type_ = lookup_dtype(type_)
        del self._typed_ops[type_]
        del self.types[type_]

    def __contains__(self, type_):
        try:
            self[type_]
        except (TypeError, KeyError, NumbaError, UdfParseError):
            return False
        return True

    @classmethod
    def _remove_nesting(cls, funcname, *, module=None, modname=None, strict=True):
        if module is None:
            module = cls._module
        if modname is None:
            modname = cls._modname
        if "." not in funcname:
            if strict and _hasop(module, funcname):
                raise AttributeError(f"{modname}.{funcname} is already defined")
        else:
            path, funcname = funcname.rsplit(".", 1)
            for folder in path.split("."):
                if not _hasop(module, folder):
                    setattr(module, folder, OpPath(module, folder))
                module = getattr(module, folder)
                modname = f"{modname}.{folder}"
                if not isinstance(module, (OpPath, ModuleType)):
                    raise AttributeError(
                        f"{modname} is already defined. Cannot use as a nested path."
                    )
            if strict and _hasop(module, funcname):
                raise AttributeError(f"{path}.{funcname} is already defined")
        return module, funcname

    @classmethod
    def _find(cls, funcname):
        rv = cls._module
        for attr in funcname.split("."):
            if attr in getattr(rv, "_deprecated", ()):
                rv = rv._deprecated[attr]
            else:
                rv = getattr(rv, attr, None)
            if rv is None:
                break
        return rv

    @classmethod
    def _initialize(cls, include_in_ops=True):
        """Initialize operators for this operator type.

        include_in_ops determines whether the operators are included in the
        ``gb.ops`` namespace in addition to the defined module.
        """
        if cls._initialized:  # pragma: no cover (safety)
            return
        # Read in the parse configs
        trim_from_front = cls._parse_config.get("trim_from_front", 0)
        delete_exact = cls._parse_config.get("delete_exact")
        num_underscores = cls._parse_config["num_underscores"]

        for re_str, return_prefix in [
            ("re_exprs", None),
            ("re_exprs_return_bool", "BOOL"),
            ("re_exprs_return_float", "FP"),
            ("re_exprs_return_complex", "FC"),
        ]:
            if re_str not in cls._parse_config:
                continue
            if "complex" in re_str and not _supports_complex:
                continue
            for r in reversed(cls._parse_config[re_str]):
                for varname in _VARNAMES:
                    m = r.match(varname)
                    if m:
                        # Parse function into name and datatype
                        gb_name = m.string
                        splitname = gb_name[trim_from_front:].split("_")
                        if delete_exact and delete_exact in splitname:
                            splitname.remove(delete_exact)
                        if len(splitname) == num_underscores + 1:
                            *splitname, type_ = splitname
                        else:
                            type_ = None
                        name = "_".join(splitname).lower()
                        # Create object for name unless it already exists
                        if not _hasop(cls._module, name):
                            if backend == "suitesparse" and name in _SS_OPERATORS:
                                fullname = f"ss.{name}"
                            else:
                                fullname = name
                            if cls._positional is None:
                                obj = cls(fullname)
                            else:
                                obj = cls(fullname, is_positional=name in cls._positional)
                            if name in _SS_OPERATORS:
                                if backend == "suitesparse":
                                    setattr(cls._module.ss, name, obj)
                                cls._module._deprecated[name] = obj
                                if include_in_ops and not _hasop(op, name):  # pragma: no branch
                                    op._deprecated[name] = obj
                                    if backend == "suitesparse":
                                        setattr(op.ss, name, obj)
                            else:
                                setattr(cls._module, name, obj)
                                if include_in_ops and not _hasop(op, name):
                                    setattr(op, name, obj)
                            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{fullname}")
                        elif name in _SS_OPERATORS:
                            obj = cls._module._deprecated[name]
                        else:
                            obj = getattr(cls._module, name)
                        gb_obj = getattr(lib, varname)
                        # Determine return type
                        if return_prefix == "BOOL":
                            return_type = BOOL
                            if type_ is None:
                                type_ = BOOL
                        else:
                            if type_ is None:  # pragma: no cover (safety)
                                raise TypeError(f"Unable to determine return type for {varname}")
                            if return_prefix is None:
                                return_type = type_
                            else:
                                # Grab the number of bits from type_
                                num_bits = type_[-2:]
                                if num_bits not in {"32", "64"}:  # pragma: no cover (safety)
                                    raise TypeError(f"Unexpected number of bits: {num_bits}")
                                return_type = f"{return_prefix}{num_bits}"
                        builtin_op = cls._typed_class(
                            obj,
                            name,
                            lookup_dtype(type_),
                            lookup_dtype(return_type),
                            gb_obj,
                            gb_name,
                        )
                        obj._add(builtin_op)

    @classmethod
    def _deserialize(cls, name, *args):
        if (rv := cls._find(name)) is not None:
            return rv  # Should we verify this is what the user expects?
        return cls.register_new(name, *args)

    @classmethod
    def _deserialize_udf(cls, name, orig_func, is_udt):
        """Re-register a named UDF on unpickle, or reuse if already present.

        Shared by the five UDF-capable subclasses (UnaryOp, BinaryOp,
        IndexUnaryOp, SelectOp, IndexBinaryOp), all of which use the
        default ``__reduce__`` below.
        """
        if (rv := cls._find(name)) is not None:
            return rv
        return cls.register_new(name, orig_func, is_udt=is_udt)

    @classmethod
    def _deserialize_anon_udf(cls, func, name, is_udt):
        """Re-register an anonymous UDF on unpickle."""
        return cls.register_anonymous(func, name, is_udt=is_udt)

    def __reduce__(self):
        """Default ``__reduce__`` for UDF-capable subclasses.

        Assumes the instance has ``orig_func`` and ``_is_udt`` attributes (all
        five UDF-capable subclasses do). ``Monoid``, ``Semiring``, and
        ``Aggregator`` define their own ``__reduce__`` because their pickle
        shape differs (they hold a binary op + identity, etc.).
        """
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (type(self)._deserialize_anon_udf, (self.orig_func, self.name, self._is_udt))
        if (name := f"{self._modname}.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (type(self)._deserialize_udf, (self.name, self.orig_func, self._is_udt))

    @classmethod
    def _check_supports_udf(cls, method_name):
        if not _supports_udfs:
            raise RuntimeError(
                f"{cls.__name__}.{method_name}(...) unavailable; install numba for UDF support"
            )


_builtin_to_op = {}  # Populated in .utils


def find_opclass(gb_op):
    if isinstance(gb_op, OpBase):
        opclass = type(gb_op).__name__
    elif isinstance(gb_op, TypedOpBase):
        opclass = gb_op.opclass
    elif isinstance(gb_op, ParameterizedUdf):
        gb_op = gb_op()  # Use default parameters of parameterized UDFs
        gb_op, opclass = find_opclass(gb_op)
    elif isinstance(gb_op, BuiltinFunctionType) and gb_op in _builtin_to_op:
        gb_op, opclass = find_opclass(_builtin_to_op[gb_op])
    else:
        opclass = UNKNOWN_OPCLASS
    return gb_op, opclass

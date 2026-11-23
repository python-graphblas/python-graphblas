import inspect
import re
import threading
from functools import lru_cache, reduce
from operator import mul
from types import FunctionType

import numpy as np

from ... import _STANDARD_OPERATOR_NAMES, backend, binary, monoid, op
from ...dtypes import (
    BOOL,
    FP32,
    FP64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    _supports_complex,
    lookup_dtype,
)
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, _supports_udfs, ffi, lib
from ..dtypes import _sample_values
from ..expr import InfixExprBase

# Not under ``if _has_numba``: ``_compile_udt`` reads it without Numba too
from ..udt import BUILTIN_UDT_BINARY_OPS as _BUILTIN_UDT_BINARY_OPS
from .base import (
    _SS_OPERATORS,
    OpBase,
    ParameterizedUdf,
    TypedOpBase,
    _call_op,
    _hasop,
)

if _has_numba:
    import numba

    from .udf import (
        _bool_to_int8,
        _compile_for_dtype,
        _compile_udf_for_udt,
        _finalize_udt_op,
        _get_udt_wrapper,
        _resolve_udt_return_type,
    )

    try:
        # ``_infer_ret_types_typing_only`` learns the built-in UDFs' result types from
        # Numba's typing pass alone, so ``.types`` is complete before any dtype compiles.
        # Numba has no public API for a return type without a compile, so this uses
        # ``numba.core`` internals, which can change in any Numba release. If the import
        # fails, or the call returns something unexpected, the function returns None and
        # ``BinaryOp._build`` compiles every dtype up front.
        from numba.core import compiler as _numba_compiler
        from numba.core import typed_passes as _numba_typed_passes
        from numba.core.registry import cpu_target as _numba_cpu_target

        _HAS_TYPING_ONLY = True
    except Exception:  # pragma: no cover (numba internals moved)
        _HAS_TYPING_ONLY = False
else:
    _HAS_TYPING_ONLY = False
if _supports_complex:
    from ...dtypes import FC32, FC64

ffi_new = ffi.new

if _has_numba:

    def _make_udt_comparison(dtype, dtype2, *, is_eq):
        """Return ``(wrapper, signature)`` for a cfunc that compares UDT elements with eq or ne.

        Leaves compare as numpy compares them, so NaN is unequal even to itself. A built-in
        operand is compared with every leaf. Types that do not pair raise ``KeyError``
        (``_check_udt_pair``). A UDT that is neither a record nor an array, such as
        text, is compared as one value.
        """
        from ..udt import (
            _check_leaves_compare,
            _check_udt_pair,
            _get_udt_info,
            _iter_record_leaves,
        )
        from .udt_codegen import _array_operand_source

        info_x = _get_udt_info(dtype)
        info_y = _get_udt_info(dtype2)
        op_name = "eq" if is_eq else "ne"
        _check_udt_pair(op_name, dtype, dtype2, info_x, info_y)
        _check_leaves_compare(op_name, dtype, dtype2)

        nt = numba.types
        cmp_op = "==" if is_eq else "!="
        join = " and " if is_eq else " or "

        if info_x is None and info_y is None:
            # Text, a date, or a complex number on a backend without complex types.
            # The element is one value, compared as numpy compares it (as a record
            # field of that type is). ``_check_udt_pair`` has checked that both
            # operands have the same UDT, or are both text of one kind, both dates
            # or both durations.
            wrapper = _compile_codegen(
                f"def wrapper(z_ptr, x_ptr, y_ptr):\n    z_ptr[0] = x_ptr[0] {cmp_op} y_ptr[0]\n",
                func_name="wrapper",
                source_label=f"<gb-udt {op_name} {dtype.np_type} {dtype2.np_type}>",
            )
            x_ptr_type = nt.CPointer(dtype.numba_type)
            y_ptr_type = nt.CPointer(dtype2.numba_type)
            return wrapper, nt.void(nt.CPointer(INT8.numba_type), x_ptr_type, y_ptr_type)

        x_is_scalar = info_x is None
        y_is_scalar = info_y is None
        udt_dtype = dtype2 if x_is_scalar else dtype
        udt_info = info_y if x_is_scalar else info_x

        if udt_info[0] == "array":
            # Array UDT. An array operand is a flat pointer to its base type, not to
            # the UDT's numba type, with which ``numba.carray`` would read N
            # elements of the UDT's size. Each operand gets its own base type, since
            # two array UDTs can differ in it. A built-in operand is read through
            # its pointer. The shapes broadcast, as for the arithmetic ops, so eq is
            # True when every pair of elements numpy would compare is equal.
            shape_x = None if x_is_scalar else dtype.np_type.subdtype[1]
            shape_y = None if y_is_scalar else dtype2.np_type.subdtype[1]
            shape = np.broadcast_shapes(*(s for s in (shape_x, shape_y) if s is not None))
            N = reduce(mul, shape)
            x_ptr_type = nt.CPointer(numba.from_dtype(dtype.np_type.base))
            y_ptr_type = nt.CPointer(numba.from_dtype(dtype2.np_type.base))
            wrapper_sig = nt.void(nt.CPointer(INT8.numba_type), x_ptr_type, y_ptr_type)
            ns = {}
            x_setup, x_ref = _array_operand_source("x", shape_x, shape, ns)
            y_setup, y_ref = _array_operand_source("y", shape_y, shape, ns)
            # A loop, not a chain of N terms, whose typing time grows faster than N
            # (see ``_make_array_wrapper``). It reads every element instead of
            # stopping at the first that decides, so the compiler can vectorize it;
            # equal values need every element read anyway.
            accumulate = "&=" if is_eq else "|="
            src = (
                f"def wrapper(z_ptr, x_ptr, y_ptr):\n{x_setup}{y_setup}"
                f"    z = {is_eq}\n"
                f"    for i in range({N}):\n"
                f"        z {accumulate} {x_ref} {cmp_op} {y_ref}\n"
                f"    z_ptr[0] = z\n"
            )
            wrapper = _compile_codegen(
                src,
                func_name="wrapper",
                source_label=f"<gb-udt {op_name} array N={N}>",
                extra_ns=ns,
            )
            return wrapper, wrapper_sig

        # Record UDT, possibly nested: join the leaf comparisons. An array field is
        # compared element by element with the other operand's field or number, and
        # reduced to one bool with ``all`` (eq) or ``any`` (ne).
        np_type = udt_dtype.np_type
        terms = []
        for py_path, _c, leaf_dtype in _iter_record_leaves(np_type):
            x_expr = "x_ptr[0]" if x_is_scalar else f"x[0]{py_path}"
            y_expr = "y_ptr[0]" if y_is_scalar else f"y[0]{py_path}"
            if leaf_dtype.subdtype is not None:
                reducer = ".all()" if is_eq else ".any()"
                terms.append(f"({x_expr} {cmp_op} {y_expr}){reducer}")
            else:
                terms.append(f"({x_expr} {cmp_op} {y_expr})")
        expr = join.join(terms) if terms else ("True" if is_eq else "False")
        lines = ["def wrapper(z_ptr, x_ptr, y_ptr):"]
        if not x_is_scalar:
            lines.append("    x = numba.carray(x_ptr, 1)")
        if not y_is_scalar:
            lines.append("    y = numba.carray(y_ptr, 1)")
        lines.append(f"    z_ptr[0] = {expr}")
        src = "\n".join(lines) + "\n"
        wrapper = _compile_codegen(
            src,
            func_name="wrapper",
            source_label=f"<gb-udt {op_name} record nleaves={len(terms)}>",
        )
        x_ptr_type = (
            nt.CPointer(numba.from_dtype(dtype.np_type))
            if x_is_scalar
            else nt.CPointer(dtype.numba_type)
        )
        y_ptr_type = (
            nt.CPointer(numba.from_dtype(dtype2.np_type))
            if y_is_scalar
            else nt.CPointer(dtype2.numba_type)
        )
        wrapper_sig = nt.void(nt.CPointer(INT8.numba_type), x_ptr_type, y_ptr_type)
        return wrapper, wrapper_sig


class TypedBuiltinBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "BinaryOp"

    def __call__(self, left, right=None, *, left_default=None, right_default=None):
        if left_default is not None or right_default is not None:
            if (
                left_default is None
                or right_default is None
                or right is not None
                or not isinstance(left, InfixExprBase)
                or left.method_name != "ewise_add"
            ):
                raise TypeError(
                    "Specifying `left_default` or `right_default` keyword arguments implies "
                    "performing `ewise_union` operation with infix notation.\n"
                    "There is only one valid way to do this:\n\n"
                    f">>> {self}(x | y, left_default=0, right_default=0)\n\nwhere x and y "
                    "are Vectors or Matrices, and left_default and right_default are scalars."
                )
            return left._call_left("_ewise_union", self, left_default, right_default, is_infix=True)
        return _call_op(self, left, right)

    @property
    def monoid(self):
        rv = getattr(monoid, self.name, None)
        if rv is not None and self.type in rv._typed_ops:
            return rv[self.type]

    @property
    def commutes_to(self):
        commutes_to = self.parent.commutes_to
        if commutes_to is not None and (self.type in commutes_to.types or self.type._is_udt):
            return commutes_to[self.type]

    @property
    def _semiring_commutes_to(self):
        commutes_to = self.parent._semiring_commutes_to
        if commutes_to is not None and (self.type in commutes_to.types or self.type._is_udt):
            return commutes_to[self.type]

    @property
    def is_commutative(self):
        return self.commutes_to is self

    @property
    def type2(self):
        return self.type if self._type2 is None else self._type2


class TypedUserBinaryOp(TypedOpBase):
    __slots__ = "_monoid"
    opclass = "BinaryOp"
    _owns_gb_obj = True

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)
        self._monoid = None

    @property
    def monoid(self):
        if self._monoid is None:
            monoid = self.parent.monoid
            if monoid is not None and self.type in monoid:
                self._monoid = monoid[self.type]
        return self._monoid

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    commutes_to = TypedBuiltinBinaryOp.commutes_to
    _semiring_commutes_to = TypedBuiltinBinaryOp._semiring_commutes_to
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    type2 = TypedBuiltinBinaryOp.type2
    __call__ = TypedBuiltinBinaryOp.__call__


class ParameterizedBinaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_monoid", "_cached_call", "_commutes_to", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._monoid = None
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)
        method = self._call_to_cache.__get__(self, type(self))
        self._cached_call = lru_cache(maxsize=1024)(method)
        self.__call__ = self._call
        self._commutes_to = None

    def _call_to_cache(self, *args, **kwargs):
        binary = self.func(*args, **kwargs)
        binary._parameterized_info = (self, args, kwargs)
        return BinaryOp.register_anonymous(binary, self.name, is_udt=self._is_udt)

    def _call(self, *args, **kwargs):
        binop = self._cached_call(*args, **kwargs)
        if self._monoid is not None and binop._monoid is None:
            # This is all a bit funky.  We try our best to associate a binaryop
            # to a monoid.  So, if we made a ParameterizedMonoid using this object,
            # then try to create a monoid with the given arguments.
            binop._monoid = binop  # temporary!
            try:
                # If this call is successful, then it will set `binop._monoid`
                self._monoid(*args, **kwargs)  # pylint: disable=not-callable
            except Exception:
                binop._monoid = None
            # assert binop._monoid is not binop
        if self.is_commutative:
            binop._commutes_to = binop
        # Don't bother yet with creating `binop.commutes_to` (but we could!)
        return binop

    @property
    def monoid(self):
        return self._monoid

    @property
    def commutes_to(self):
        if isinstance(self._commutes_to, str):
            self._commutes_to = BinaryOp._find(self._commutes_to)
        return self._commutes_to

    is_commutative = TypedBuiltinBinaryOp.is_commutative


def _floordiv(x, y):
    return x // y  # pragma: no cover (numba)


def _rfloordiv(x, y):
    return y // x  # pragma: no cover (numba)


def _absfirst(x, y):
    return np.abs(x)  # pragma: no cover (numba)


def _abssecond(x, y):
    return np.abs(y)  # pragma: no cover (numba)


def _rpow(x, y):
    return y**x  # pragma: no cover (numba)


def _isclose(rel_tol=1e-7, abs_tol=0.0):
    def inner(x, y):  # pragma: no cover (numba)
        if x == y:
            return True
        # As in math.isclose, an infinity is close only to itself
        if abs(x) == np.inf or abs(y) == np.inf:
            return False
        return _abs_diff(x, y) <= max(rel_tol * max(_magnitude(x), _magnitude(y)), abs_tol)

    return inner


# Integer arithmetic in Numba wraps: ``x - y`` in int64 or uint64 (Numba widens smaller
# unsigned types to uint64, so 5 - 6 is 2**64 - 1), and ``abs`` of a signed type's
# minimum. These helpers are plain Python, which is right for Python ints; the overloads
# below give Numba's integers versions that do not wrap.
def _abs_diff(x, y):
    return abs(x - y)


def _magnitude(x):
    return abs(x)


if _has_numba:
    from numba.extending import overload

    @overload(_abs_diff)
    def _abs_diff_overload(x, y):
        if isinstance(x, numba.types.Integer) and x == y:

            def abs_diff(x, y):  # pragma: no cover (numba)
                # The difference fits in uint64, and both operands convert to it modulo 2**64
                if x >= y:
                    return np.uint64(x) - np.uint64(y)
                return np.uint64(y) - np.uint64(x)

            return abs_diff
        return lambda x, y: abs(x - y)

    @overload(_magnitude)
    def _magnitude_overload(x):
        if isinstance(x, numba.types.Integer):
            return lambda x: abs(np.float64(x))
        return lambda x: abs(x)  # noqa: PLW0108 (an overload returns a Python function)


_MAX_INT64 = np.iinfo(np.int64).max


def _binom(N, k):  # pragma: no cover (numba)
    # Returns 0 if overflow or out-of-bounds
    if k > N or k < 0:
        return 0
    val = np.int64(1)
    for i in range(min(k, N - k)):
        if val > _MAX_INT64 // (N - i):  # Overflow
            return 0
        val *= N - i
        val //= i + 1
    return val


# Kinda complicated, but works for now
def _register_binom():
    # "Fake" UDT so we only compile once for INT64
    op = BinaryOp.register_new("binom", _binom, is_udt=True)
    typed_op = op[INT64, INT64]
    # Make this look like a normal operator
    for dtype in [UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64]:
        op.types[dtype] = INT64
        op._typed_ops[dtype] = typed_op
        if dtype != INT64:
            op.coercions[dtype] = typed_op
    # And make it not look like it operates on UDTs
    typed_op._type2 = None
    op._is_udt = False
    op._udt_types = None
    op._udt_ops = None
    return op


def _first(x, y):
    return x  # pragma: no cover (numba)


def _second(x, y):
    return y  # pragma: no cover (numba)


def _pair(x, y):
    return 1  # pragma: no cover (numba)


def _udt_dtype(op, dtype, dtype2):
    """Custom dtype handler that compiles an operator for UDTs on first use."""
    if dtype._is_udt or dtype2._is_udt:
        return op._compile_udt(dtype, dtype2)


def _pair_dtype(op, dtype, dtype2):
    return op[INT64]


def _adjust_ret_type(type_, ret_type, return_types):
    """Return the result type of a UDF on ``type_``, given the type Numba inferred for it.

    An integer result for integer input is the input type, and so is a float result for
    float input and a complex result for complex input. A UINT64 input with an FP64 result
    gets UINT64 if INT64 kept its type, and a BOOL input with an INT64 result gets INT8 if
    INT8 kept its type. ``return_types`` holds the results for the sample dtypes done so
    far, which include INT64 and INT8. The eager loop and the typing-only inference both
    call this, so they give the same ``.types``.
    """
    if ret_type != type_ and (
        ("INT" in ret_type.name and "INT" in type_.name)
        or ("FP" in ret_type.name and "FP" in type_.name)
        or ("FC" in ret_type.name and "FC" in type_.name)
        or (type_ == UINT64 and ret_type == FP64 and return_types.get(INT64) == INT64)
    ):
        # This is what users want most of the time, but we can't make a perfect
        # rule. There should be a way for users to be explicit.
        return type_
    if type_ == BOOL and ret_type == INT64 and return_types.get(INT8) == INT8:
        return INT8
    return ret_type


_deferred_build_lock = threading.Lock()  # see BinaryOp._build_deferred


def _finalize_typed_binaryop(parent, binary_udf, name, type_, ret_type):
    """Compile ``binary_udf`` for ``type_`` in a cfunc wrapper; add the typed op to ``parent``.

    Return the typed op. ``ret_type`` is the result type from ``_adjust_ret_type``. The
    eager loop in ``BinaryOp._build`` calls this for every dtype. For a built-in UDF,
    ``BinaryOp._build_deferred`` calls it when a dtype is first used, so that is when the
    dtype compiles.
    """
    nt = numba.types
    input_type = _bool_to_int8(type_)
    return_type = _bool_to_int8(ret_type)

    # Build wrapper because GraphBLAS wants pointers and void return
    wrapper_sig = nt.void(
        nt.CPointer(return_type.numba_type),
        nt.CPointer(input_type.numba_type),
        nt.CPointer(input_type.numba_type),
    )

    if type_ == BOOL:
        if ret_type == BOOL:

            def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                z[0] = bool(binary_udf(bool(x[0]), bool(y[0])))

        else:

            def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                z[0] = binary_udf(bool(x[0]), bool(y[0]))

    elif ret_type == BOOL:

        def binary_wrapper(z, x, y):  # pragma: no cover (numba)
            z[0] = bool(binary_udf(x[0], y[0]))

    else:

        def binary_wrapper(z, x, y):  # pragma: no cover (numba)
            z[0] = binary_udf(x[0], y[0])

    binary_wrapper = numba.cfunc(wrapper_sig, nopython=True, error_model="numpy")(binary_wrapper)
    new_binary = ffi_new("GrB_BinaryOp*")
    check_status_carg(
        lib.GrB_BinaryOp_new(
            new_binary,
            binary_wrapper.cffi,
            ret_type.gb_obj,
            type_.gb_obj,
            type_.gb_obj,
        ),
        "BinaryOp",
        new_binary[0],
    )
    op = TypedUserBinaryOp(parent, name, type_, ret_type, new_binary[0])
    parent._add(op)
    return op


def _infer_ret_types_typing_only(func):
    """Return ``{DataType: result DataType}`` for the sample dtypes from typing alone, or None.

    ``func`` is not compiled. None means that typing raised something other than a
    TypingError for some dtype, or that Numba's internals have moved, and the caller then
    compiles every dtype. This is only for the built-in UDFs: typing alone can accept a dtype
    that does not compile, which only compiling would catch. For the built-ins,
    test_builtin_udf_types_without_typing_only checks that compiling gives the same types.
    """
    if not _HAS_TYPING_ONLY:
        return None
    try:
        typingctx = _numba_cpu_target.typing_context
        targetctx = _numba_cpu_target.target_context
        typingctx.refresh()
        targetctx.refresh()
        return_types = {}
        for type_ in _sample_values:
            try:
                interp = _numba_compiler.run_frontend(func)
                result = _numba_typed_passes.type_inference_stage(
                    typingctx, targetctx, interp, [type_.numba_type, type_.numba_type], None
                )
                numba_ret_type = result.return_type
            except numba.TypingError:
                # This dtype does not type-check; skip it, as the eager loop does.
                continue
            ret_type = _adjust_ret_type(type_, lookup_dtype(numba_ret_type), return_types)
            return_types[type_] = ret_type
    except Exception:  # pragma: no cover (numba internals moved)
        return None
    else:
        return return_types


if _has_numba:
    from .udt_codegen import (
        _compile_codegen,
        _has_jit_set,
        _maybe_warn_jit_skipped,
        compile_udt_binary_wrapper,
        set_jit_c_mixed_on_op,
        set_jit_c_on_op,
    )


class BinaryOp(OpBase):
    """Takes two inputs and returns one output, possibly of a different data type.

    Built-in and registered BinaryOps are located in the ``graphblas.binary`` namespace
    as well as in the ``graphblas.op`` combined namespace.
    """

    __slots__ = (
        "_monoid",
        "_commutes_to",
        "_semiring_commutes_to",
        "orig_func",
        "is_positional",
        "_is_udt",
        "_numba_func",
        "_custom_dtype",
        "_defer_builds",
    )
    _module = binary
    _modname = "binary"
    _typed_class = TypedBuiltinBinaryOp
    _parse_config = {
        "trim_from_front": 4,
        "num_underscores": 1,
        "re_exprs": [
            re.compile(
                "^GrB_(FIRST|SECOND|PLUS|MINUS|TIMES|DIV|MIN|MAX)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile(
                "GrB_(BOR|BAND|BXOR|BXNOR)_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
            re.compile(
                "^GxB_(POW|RMINUS|RDIV|PAIR|ANY|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile("^GxB_(FIRST|SECOND|PLUS|MINUS|TIMES|DIV)_(FC32|FC64)$"),
            re.compile("^GxB_(ATAN2|HYPOT|FMOD|REMAINDER|LDEXP|COPYSIGN)_(FP32|FP64)$"),
            re.compile(
                "GxB_(BGET|BSET|BCLR|BSHIFT|FIRSTI1|FIRSTI|FIRSTJ1|FIRSTJ"
                "|SECONDI1|SECONDI|SECONDJ1|SECONDJ)"
                "_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
            # These are coerced to 0 or 1, but don't return BOOL
            re.compile(
                "^GxB_(LOR|LAND|LXOR|LXNOR)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)$"),
            re.compile(
                "^GrB_(EQ|NE|GT|LT|GE|LE)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile("^GxB_(EQ|NE)_(FC32|FC64)$"),
        ],
        "re_exprs_return_complex": [re.compile("^GxB_(CMPLX)_(FP32|FP64)$")],
    }
    _commutes = {
        # builtins
        "cdiv": "rdiv",
        "first": "second",
        "ge": "le",
        "gt": "lt",
        "isge": "isle",
        "isgt": "islt",
        "minus": "rminus",
        "pow": "rpow",
        # special
        "firsti": "secondi",
        "firsti1": "secondi1",
        "firstj": "secondj",
        "firstj1": "secondj1",
        # custom
        # "absfirst": "abssecond",  # handled in graphblas.binary
        # "floordiv": "rfloordiv",
        "truediv": "rtruediv",
    }
    _commutes_to_in_semiring = {
        "firsti": "secondj",
        "firsti1": "secondj1",
        "firstj": "secondi",
        "firstj1": "secondi1",
    }
    _commutative = {
        # monoids
        "any",
        "band",
        "bor",
        "bxnor",
        "bxor",
        "eq",
        "land",
        "lor",
        "lxnor",
        "lxor",
        "max",
        "min",
        "plus",
        "times",
        # other
        "hypot",
        "isclose",
        "iseq",
        "isne",
        "ne",
        "pair",
    }
    # Don't commute: atan2, bclr, bget, bset, bshift, cmplx, copysign, fmod, ldexp, remainder
    _positional = {
        "firsti",
        "firsti1",
        "firstj",
        "firstj1",
        "secondi",
        "secondi1",
        "secondj",
        "secondj1",
    }

    @classmethod
    def _build(cls, name, func, *, is_udt=False, anonymous=False, cache=False):
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_binary>")
        success = False
        # The error model has to be set here, not only on the cfunc wrapper in
        # ``_finalize_typed_binaryop``. The eager loop's ``.compile(sig)`` below builds
        # the specialization the wrapper then reuses, and a Dispatcher keeps one
        # compilation per signature, so the first compile fixes the model. Set only on
        # the wrapper, ``x // 0`` raises ZeroDivisionError inside the cfunc, where Numba
        # prints the traceback and returns, handing GraphBLAS an element it never wrote.
        # The other op classes' ``_build`` point here.
        #
        # ``cache`` is True only for the built-in UDFs, which are module-level
        # functions; Numba cannot cache a lambda or an interactively defined function
        # (docs/technical_notes/udfs.rst, "The Numba disk cache").
        if cache:
            try:
                binary_udf = numba.njit(func, error_model="numpy", cache=True)
            except RuntimeError:
                # Numba has nowhere to write the cache ("no locator available"), as in
                # a read-only install whose user has no writable cache directory
                binary_udf = numba.njit(func, error_model="numpy")
        else:
            binary_udf = numba.njit(func, error_model="numpy")
        new_type_obj = cls(name, func, anonymous=anonymous, is_udt=is_udt, numba_func=binary_udf)
        return_types = {}
        if not is_udt:
            # A built-in UDF (``cache=True``) gets its result types from typing alone,
            # and compiles each dtype when it is first used (``_build_deferred``). A
            # user's function compiles every sample dtype here, which also checks that
            # each one compiles. So does a built-in when the typing returns None.
            inferred = _infer_ret_types_typing_only(func) if cache else None
            if inferred is not None:
                new_type_obj.types.update(inferred)
                return_types.update(inferred)
                new_type_obj._defer_builds = True
                success = bool(inferred)
            else:
                for type_ in _sample_values:
                    sig = (type_.numba_type, type_.numba_type)
                    if not _compile_for_dtype(binary_udf, sig):
                        continue
                    ret_type = lookup_dtype(binary_udf.overloads[sig].signature.return_type)
                    ret_type = _adjust_ret_type(type_, ret_type, return_types)
                    _finalize_typed_binaryop(new_type_obj, binary_udf, name, type_, ret_type)
                    success = True
                    return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]
        from ..udt import _refuse_half

        _refuse_half(self, dtype, dtype2)

        # Built-in ops lift to UDTs by name. A UDF registered under one of those
        # names (``BinaryOp.register_anonymous(func, "plus", is_udt=True)``) is
        # anonymous, and is compiled from its own function below.
        is_lifted = not self._anonymous and (
            self.name in _BUILTIN_UDT_BINARY_OPS or self.name in ("eq", "ne")
        )

        # A lifted op has ``_udt_types``, so a built-in type it lacks (``min["FC64"]``)
        # comes here too. Raise the KeyError ``OpBase.__getitem__`` raises for one.
        if (
            is_lifted
            and self.name in _BUILTIN_UDT_BINARY_OPS
            and not dtype._is_udt
            and not dtype2._is_udt
        ):
            raise KeyError(f"{self.name} does not work with {dtype}")

        if is_lifted and self.name in ("eq", "ne") and _has_numba:
            binary_wrapper, wrapper_sig = _make_udt_comparison(
                dtype, dtype2, is_eq=(self.name == "eq")
            )
            ret_type = BOOL

        elif _has_numba and is_lifted:
            # Generate a wrapper that computes leaf by leaf. This comes before the
            # ``_numba_func`` branch because ``floordiv`` has a ``_numba_func``, which
            # Numba cannot compile for records.
            py_op = _BUILTIN_UDT_BINARY_OPS[self.name]
            binary_wrapper, wrapper_sig, ret_type = compile_udt_binary_wrapper(
                self.name, py_op, dtype, dtype2
            )
        elif self._numba_func is None:
            raise KeyError(f"{self.name} does not work with {dtypes} types")
        else:
            numba_func = self._numba_func
            sig = (dtype.numba_type, dtype2.numba_type)
            _compile_udf_for_udt(
                numba_func, sig, op_kind="binary", op_name=self.name, dtypes=(dtype, dtype2)
            )
            numba_ret_type = numba_func.overloads[sig].signature.return_type
            ret_type = _resolve_udt_return_type(numba_ret_type, dtype, dtype2)
            binary_wrapper, wrapper_sig = _get_udt_wrapper(
                numba_func, ret_type, dtype, dtype2, numba_ret_type=numba_ret_type
            )

        op = _finalize_udt_op(
            self, dtype, dtype2, ret_type, binary_wrapper, wrapper_sig, TypedUserBinaryOp
        )
        # Give a lifted op a C definition for the JIT. When x and y have one type, and
        # an arithmetic op's result has it too, the definition uses that type
        # (``set_jit_c_on_op``, or ``set_jit_c_comparison_on_op`` for eq and ne).
        # Otherwise an arithmetic op gets a definition with a C type for each of z, x
        # and y, when there is one (``set_jit_c_mixed_on_op``), and eq and ne get none.
        # ``_has_jit_set`` is False on the vanilla backend.
        if is_lifted and self.name in _BUILTIN_UDT_BINARY_OPS:
            is_jittable = ret_type == dtype
        else:
            is_jittable = is_lifted
        if _has_numba and _has_jit_set and dtype == dtype2 and is_jittable:
            if self.name in _BUILTIN_UDT_BINARY_OPS:
                op._jit_c_info = set_jit_c_on_op(
                    op.gb_obj,
                    self.name,
                    _BUILTIN_UDT_BINARY_OPS[self.name],
                    dtype,
                    lib.GrB_BinaryOp_set_String,
                    arity=2,
                )
            else:
                # eq and ne return BOOL, not the UDT; their definition compares leaf by leaf
                from .udt_codegen import set_jit_c_comparison_on_op

                op._jit_c_info = set_jit_c_comparison_on_op(
                    op.gb_obj,
                    self.name,
                    dtype,
                    lib.GrB_BinaryOp_set_String,
                    is_eq=(self.name == "eq"),
                )
            _maybe_warn_jit_skipped(op._jit_c_info, self.name, dtype.name)
        elif _has_numba and _has_jit_set and is_lifted and self.name in _BUILTIN_UDT_BINARY_OPS:
            # No NoJITWarning if this gives no definition, as documented in
            # docs/technical_notes/jit.rst ("When a lifted operator runs without the JIT")
            from ..udt import _udt_op_types

            _, input_types = _udt_op_types(self.name, dtype, dtype2)
            op._jit_c_info = set_jit_c_mixed_on_op(
                op.gb_obj,
                self.name,
                _BUILTIN_UDT_BINARY_OPS[self.name],
                dtype,
                dtype2,
                ret_type,
                input_types,
            )
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register a BinaryOp without registering it in the ``graphblas.binary`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes two input parameters of any dtype and returns any dtype.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.binary.{name}``.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
            For example, ``gb.binary.isclose`` is a parameterized function that
            optionally accepts ``rel_tol`` and ``abs_tol`` parameters, and it
            can be used as: ``A.ewise_mult(B, gb.binary.isclose(rel_tol=1e-5))``.
            When creating a parameterized user-defined operator, the ``func``
            parameter must be a callable that *returns* a function that will
            then get compiled.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.
            If True, then the function will not be automatically compiled for
            builtin types, and it will be compiled "just in time" when used.
            Setting ``is_udt=True`` is also helpful when the left and right
            dtypes need to be different.

        Returns
        -------
        BinaryOp or ParameterizedBinaryOp

        """
        cls._check_supports_udf("register_anonymous")
        if parameterized:
            return ParameterizedBinaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(
        cls, name, func, *, parameterized=False, is_udt=False, lazy=False, _cache=False
    ):
        """Register a new BinaryOp and save it to ``graphblas.binary`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.binary.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.binary.x.y.z`` for name ``"x.y.z"``.
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes two input parameters of any dtype and returns any dtype.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
            For example, ``gb.binary.isclose`` is a parameterized function that
            optionally accepts ``rel_tol`` and ``abs_tol`` parameters, and it
            can be used as: ``A.ewise_mult(B, gb.binary.isclose(rel_tol=1e-5))``.
            When creating a parameterized user-defined operator, the ``func``
            parameter must be a callable that *returns* a function that will
            then get compiled. See the ``user_isclose`` example below.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.
            If True, then the function will not be automatically compiled for
            builtin types, and it will be compiled "just in time" when used.
            Setting ``is_udt=True`` is also helpful when the left and right
            dtypes need to be different.
        lazy : bool, default False
            If False (the default), then the function will be automatically
            compiled for builtin types (unless ``is_udt`` is True).
            Compiling functions can be slow, however, so you may want to
            delay compilation and only compile when the operator is used,
            which is done by setting ``lazy=True``.

        Examples
        --------
        >>> def max_zero(x, y):
                r = 0
                if x > r:
                    r = x
                if y > r:
                    r = y
                return r
        >>> gb.core.operator.BinaryOp.register_new("max_zero", max_zero)
        >>> dir(gb.binary)
        [..., 'max_zero', ...]

        ``gb.binary.isclose`` is defined like this, except that it computes ``x - y``
        and ``abs(x)`` for integers without wrapping:

        >>> def user_isclose(rel_tol=1e-7, abs_tol=0.0):
        >>>     def inner(x, y):
        >>>         if x == y:
        >>>             return True
        >>>         if abs(x) == np.inf or abs(y) == np.inf:
        >>>             return False
        >>>         return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
        >>>     return inner
        >>> gb.binary.register_new("user_isclose", user_isclose, parameterized=True)

        """
        cls._check_supports_udf("register_new")
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {
                    "name": name,
                    "func": func,
                    "parameterized": parameterized,
                    "is_udt": is_udt,
                    "_cache": _cache,
                },
            )
        elif parameterized:
            binary_op = ParameterizedBinaryOp(name, func, is_udt=is_udt)
            setattr(module, funcname, binary_op)
        else:
            binary_op = cls._build(name, func, is_udt=is_udt, cache=_cache)
            setattr(module, funcname, binary_op)
        # Also save it to `graphblas.op` if not yet defined
        opmodule, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not _hasop(opmodule, funcname):
            if lazy:
                opmodule._delayed[funcname] = module
            else:
                setattr(opmodule, funcname, binary_op)
        if not cls._initialized:
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return binary_op

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover (safety)
            return
        super()._initialize()
        # Rename div to cdiv
        cdiv = binary.cdiv = op.cdiv = BinaryOp("cdiv")
        for dtype, ret_type in binary.div.types.items():
            orig_op = binary.div[dtype]
            cur_op = TypedBuiltinBinaryOp(
                cdiv, "cdiv", dtype, ret_type, orig_op.gb_obj, orig_op.gb_name
            )
            cdiv._add(cur_op)
        del binary.div
        del op.div
        # Add truediv which always points to floating point cdiv
        # We are effectively hacking cdiv to always return floating point values
        # If the inputs are FP32, we use DIV_FP32; use DIV_FP64 for all other input dtypes
        truediv = binary.truediv = op.truediv = BinaryOp("truediv")
        rtruediv = binary.rtruediv = op.rtruediv = BinaryOp("rtruediv")
        for new_op, builtin_op in [(truediv, binary.cdiv), (rtruediv, binary.rdiv)]:
            for dtype in builtin_op.types:
                if dtype.name in {"FP32", "FC32", "FC64"}:
                    orig_dtype = dtype
                else:
                    orig_dtype = FP64
                orig_op = builtin_op[orig_dtype]
                cur_op = TypedBuiltinBinaryOp(
                    new_op,
                    new_op.name,
                    dtype,
                    builtin_op.types[orig_dtype],
                    orig_op.gb_obj,
                    orig_op.gb_name,
                )
                new_op._add(cur_op)
        if _supports_udfs:
            # Add floordiv
            # cdiv truncates towards 0, while floordiv truncates towards -inf
            # ``_cache=True`` saves Numba's compiled code for these UDFs on disk, so
            # later processes load it instead of compiling. A warm cache cuts about 20%
            # from the first use of one op, and about 45% when several are used; the
            # per-process LLVM setup and linking are not cached. See ``_build``.
            BinaryOp.register_new("floordiv", _floordiv, lazy=True, _cache=True)  # cast to integer
            BinaryOp.register_new(
                "rfloordiv", _rfloordiv, lazy=True, _cache=True
            )  # cast to integer

            # For aggregators
            BinaryOp.register_new("absfirst", _absfirst, lazy=True, _cache=True)
            BinaryOp.register_new("abssecond", _abssecond, lazy=True, _cache=True)
            BinaryOp.register_new("rpow", _rpow, lazy=True, _cache=True)

            # For algorithms
            binary._delayed["binom"] = (_register_binom, {})  # Lazy with custom creation
            op._delayed["binom"] = binary

            BinaryOp.register_new("isclose", _isclose, parameterized=True)

        # Update type information with sane coercion
        position_dtypes = [
            BOOL,
            FP32,
            FP64,
            INT8,
            INT16,
            UINT8,
            UINT16,
            UINT32,
            UINT64,
        ]
        if _supports_complex:
            position_dtypes.extend([FC32, FC64])
        name_types = [
            # fmt: off
            (
                ("atan2", "copysign", "fmod", "hypot", "ldexp", "remainder"),
                ((BOOL, INT8, INT16, UINT8, UINT16), FP32),
                ((INT32, INT64, UINT32, UINT64), FP64),
            ),
            (
                (
                    "firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1",
                    "secondj", "secondj1"),
                (
                    position_dtypes,
                    INT64,
                ),
            ),
            (
                ["lxnor"],
                (
                    (
                        FP32, FP64, INT8, INT16, INT32, INT64,
                        UINT8, UINT16, UINT32, UINT64,
                    ),
                    BOOL,
                ),
            ),
            # fmt: on
        ]
        if _supports_complex:
            name_types.append(
                (
                    ["cmplx"],
                    ((BOOL, INT8, INT16, UINT8, UINT16), FP32),
                    ((INT32, INT64, UINT32, UINT64), FP64),
                )
            )
        for names, *types in name_types:
            for name in names:
                if name not in _SS_OPERATORS:
                    cur_op = getattr(binary, name)
                elif backend == "suitesparse":
                    cur_op = getattr(binary.ss, name)
                else:
                    continue
                for input_types, target_type in types:
                    typed_op = cur_op._typed_ops[target_type]
                    output_type = cur_op.types[target_type]
                    for dtype in input_types:
                        if dtype not in cur_op.types:  # pragma: no branch (safety)
                            cur_op.types[dtype] = output_type
                            cur_op._typed_ops[dtype] = typed_op
                            cur_op.coercions[dtype] = target_type
        # Not valid input dtypes
        del binary.ldexp[FP32]
        del binary.ldexp[FP64]
        # Fill in commutes info
        for left_name, right_name in cls._commutes.items():
            if left_name in _SS_OPERATORS:
                if backend != "suitesparse":
                    continue
                left = getattr(binary.ss, left_name)
            else:
                left = getattr(binary, left_name)
            if right_name in _SS_OPERATORS:
                left._commutes_to = f"ss.{right_name}"
            else:
                left._commutes_to = right_name
            if right_name not in binary._delayed:
                if right_name in _SS_OPERATORS:
                    right = getattr(binary.ss, right_name)
                elif _supports_udfs:
                    right = getattr(binary, right_name)
                else:
                    right = getattr(binary, right_name, None)
                    if right is None:
                        continue
                if left_name in _SS_OPERATORS:
                    right._commutes_to = f"ss.{left_name}"
                else:
                    right._commutes_to = left_name
        for name in cls._commutative:
            if _supports_udfs:
                cur_op = getattr(binary, name)
            else:
                cur_op = getattr(binary, name, None)
                if cur_op is None:
                    continue
            cur_op._commutes_to = name
        # These are all SuiteSparse-specific positional ops
        if backend == "suitesparse":
            for left_name, right_name in cls._commutes_to_in_semiring.items():
                left = getattr(binary.ss, left_name)
                right = getattr(binary.ss, right_name)
                left._semiring_commutes_to = right
                right._semiring_commutes_to = left
        # ``any`` runs ``_second``, so a reduce with ``monoid.any`` on a UDT returns one
        # of the elements. With ``_first`` it would return the identity it starts from.
        for binop, func in [
            (binary.first, _first),
            (binary.second, _second),
            (binary.pair, _pair),
            (binary.any, _second),
        ]:
            binop.orig_func = func
            binop._numba_func = numba.njit(func, error_model="numpy") if _has_numba else None
            binop._udt_types = {}
            binop._udt_ops = {}
        binary.any._numba_func = binary.second._numba_func
        binary.first._custom_dtype = _udt_dtype
        binary.second._custom_dtype = _udt_dtype
        binary.pair._custom_dtype = _pair_dtype
        # eq and ne compare a UDT leaf by leaf (``_make_udt_comparison``), so NaN is
        # unequal to itself, where a comparison of the bytes would find it equal
        binary.eq._udt_types = {}
        binary.eq._udt_ops = {}
        binary.ne._udt_types = {}
        binary.ne._udt_ops = {}
        # The arithmetic ops that lift to UDTs are set up in ``__init__``, not here
        cls._initialized = True

    def __init__(
        self,
        name,
        func=None,
        *,
        anonymous=False,
        is_positional=False,
        is_udt=False,
        numba_func=None,
    ):
        super().__init__(name, anonymous=anonymous)
        self._monoid = None
        self._commutes_to = None
        self._semiring_commutes_to = None
        self.orig_func = func
        self._numba_func = numba_func
        self._is_udt = is_udt
        self.is_positional = is_positional
        self._custom_dtype = None
        # True for a built-in UDF whose ``.types`` came from typing alone (see ``_build``)
        self._defer_builds = False
        if is_udt:
            self._udt_types = {}  # {(dtype, dtype): DataType}
            self._udt_ops = {}  # {(dtype, dtype): TypedUserBinaryOp}
        elif _has_numba and not anonymous and name in _BUILTIN_UDT_BINARY_OPS:
            # A built-in arithmetic op lifts to UDTs. This is set up here, not in
            # ``_initialize``, where ``getattr(binary, name)`` would make the lazily
            # registered UDFs, such as ``floordiv``, compile with Numba at import
            # (about a second). An anonymous op with a built-in's name is a user's op,
            # which does not lift.
            self._udt_types = {}
            self._udt_ops = {}
            self._custom_dtype = _udt_dtype

    def _build_deferred(self, type_):
        # The result type comes from ``.types``. ``_adjust_ret_type`` cannot be redone for
        # one dtype, since UINT64 and BOOL read the results for INT64 and INT8.
        if not self._defer_builds or type_ not in self.types:
            return None
        # Under the lock, threads that first use one dtype at once build it once. A second
        # build would replace the first op in ``_typed_ops``, and the first is freed once
        # nothing refers to it, though a Monoid built from it meanwhile may point at it.
        with _deferred_build_lock:
            op = self._typed_ops.get(type_)
            if op is None:
                op = _finalize_typed_binaryop(
                    self, self._numba_func, self.name, type_, self.types[type_]
                )
        return op

    __call__ = TypedBuiltinBinaryOp.__call__
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    commutes_to = ParameterizedBinaryOp.commutes_to

    @property
    def monoid(self):
        if self._monoid is None and not self._anonymous:
            from .monoid import Monoid

            self._monoid = Monoid._find(self.name)
        return self._monoid


ParameterizedBinaryOp._op_class = BinaryOp

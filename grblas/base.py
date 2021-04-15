from contextvars import ContextVar
from . import ffi, replace as replace_singleton
from .descriptor import lookup as descriptor_lookup
from .dtypes import lookup_dtype
from .exceptions import check_status
from .expr import AmbiguousAssignOrExtract, Updater, _ewise_infix_expr, _matmul_infix_expr
from .mask import Mask
from .operator import UNKNOWN_OPCLASS, find_opclass, get_typed_op
from .unary import identity
from .utils import libget, _Pointer

NULL = ffi.NULL
CData = ffi.CData
_recorder = ContextVar("recorder")
_prev_recorder = None


def call(cfunc_name, args):
    call_args = [getattr(x, "_carg", x) if x is not None else NULL for x in args]
    cfunc = libget(cfunc_name)
    try:
        err_code = cfunc(*call_args)
    except TypeError as exc:
        # We should strive to not encounter this during normal usage
        from .recorder import gbstr

        callstr = f'{cfunc.__name__}({", ".join(gbstr(x) for x in args)})'
        lines = cfunc.__doc__.splitlines()
        sig = lines[0] if lines else ""
        raise TypeError(
            f"Error calling {cfunc.__name__}:\n"
            f" - Call objects: {callstr}\n"
            f" - C signature: {sig}\n"
            f" - Error: {exc}"
        )
    rv = check_status(err_code, args)
    rec = _recorder.get(_prev_recorder)
    if rec is not None:
        rec.record(cfunc_name, args)
    return rv


def _expect_type_message(
    self, x, types, *, within, argname=None, keyword_name=None, extra_message=""
):
    if type(types) is tuple:
        if type(x) in types:
            return
    elif type(x) is types:
        return
    if argname:
        argmsg = f"for argument `{argname}` "
    elif keyword_name:
        argmsg = f"for keyword argument `{keyword_name}=` "
    else:
        argmsg = ""
    if type(types) is tuple:
        expected = ", ".join(typ.__name__ for typ in types)
    else:
        expected = types.__name__
    if extra_message:
        extra_message = f"\n{extra_message}"
    return (
        f"Bad type {argmsg}in {type(self).__name__}.{within}(...).\n"
        f"    - Expected type: {expected}.\n"
        f"    - Got: {type(x)}."
        f"{extra_message}"
    )


def _expect_type(self, x, types, **kwargs):
    message = _expect_type_message(self, x, types, **kwargs)
    if message is not None:
        raise TypeError(message) from None


def _expect_op_message(
    self, op, values, *, within, argname=None, keyword_name=None, extra_message=""
):
    if type(values) is tuple:
        if op.opclass in values:
            return
    elif op.opclass == values:
        return
    if argname:
        argmsg = f"for argument `{argname}` "
    elif keyword_name:
        argmsg = f"for keyword argument `{keyword_name}=` "
    else:  # pragma: no cover
        argmsg = ""
    if type(values) is tuple:
        expected = ", ".join(values)
    else:
        expected = values
    special_message = ""
    if op.opclass == "Semiring":
        if "BinaryOp" in values:
            if "Monoid" in values:
                special_message = (
                    f"\nYou may do `{op.name}.binaryop` or `{op.name}.monoid` "
                    "to get the BinaryOp or Monoid."
                )
            else:
                special_message = f"\nYou may do `{op.name}.binaryop` to get the BinaryOp."
        elif "Monoid" in values:
            special_message = f"\nYou may do `{op.name}.monoid` to get the Monoid."
    elif op.opclass == "BinaryOp" and op.monoid is None and "Monoid" in values:
        special_message = "\nThe BinaryOp {op.name} is not known to be part of a Monoid."
    if extra_message:
        extra_message = f"\n{extra_message}"
    return (
        f"Bad type {argmsg}in {type(self).__name__}.{within}(...).\n"
        f"    - Expected type: {expected}.\n"
        f"    - Got: {op.opclass} ({op.name})."
        f"{extra_message}"
        f"{special_message}"
    )


def _expect_op(self, op, values, **kwargs):
    message = _expect_op_message(self, op, values, **kwargs)
    if message is not None:
        raise TypeError(message) from None


def _check_mask(mask, output=None):
    if isinstance(mask, BaseType) or type(mask) is Mask:
        raise TypeError("Mask must indicate values (M.V) or structure (M.S)")
    if not isinstance(mask, Mask):
        raise TypeError(f"Invalid mask: {type(mask)}")
    if output is not None:
        from .vector import Vector

        if type(output) is Vector and type(mask.mask) is not Vector:
            raise TypeError(f"Mask object must be type Vector; got {type(mask.mask)}")


class BaseType:
    __slots__ = "gb_obj", "dtype", "name", "__weakref__"
    # Flag for operations which depend on scalar vs vector/matrix
    _is_scalar = False

    def __init__(self, gb_obj, dtype, name):
        if not isinstance(gb_obj, CData):
            raise TypeError("Object passed to __init__ must be CData type")
        self.gb_obj = gb_obj
        self.dtype = lookup_dtype(dtype)
        self.name = name

    def __call__(
        self, *optional_mask_accum_replace, mask=None, accum=None, replace=False, input_mask=None
    ):
        # Pick out mask and accum from positional arguments
        mask_arg = None
        accum_arg = None
        for arg in optional_mask_accum_replace:
            if arg is replace_singleton:
                replace = True
            elif isinstance(arg, (BaseType, Mask)):
                if self._is_scalar:
                    raise TypeError("Mask not allowed for Scalars")
                if mask_arg is not None:
                    raise TypeError("Got multiple values for argument 'mask'")
                mask_arg = arg
            else:
                if accum_arg is not None:
                    raise TypeError("Got multiple values for argument 'accum'")
                accum_arg, opclass = find_opclass(arg)
                if opclass == UNKNOWN_OPCLASS:
                    raise TypeError(f"Invalid item found in output params: {type(arg)}")
        # Merge positional and keyword arguments
        if mask_arg is not None and mask is not None:
            raise TypeError("Got multiple values for argument 'mask'")
        if mask_arg is not None:
            mask = mask_arg
        if mask is None:
            if input_mask is None:
                if replace:
                    if self._is_scalar:
                        raise TypeError(
                            "'replace' argument may not be True for Scalar (replace may only be "
                            "True when a mask is provided, and masks aren't allowed for Scalars)."
                        )
                    raise TypeError("'replace' argument may only be True if a mask is provided")
            elif self._is_scalar:
                raise TypeError("input_mask not allowed for Scalars")
            else:
                _check_mask(input_mask)
        elif self._is_scalar:
            raise TypeError("Mask not allowed for Scalars")
        elif input_mask is not None:
            raise TypeError("mask and input_mask arguments cannot both be given")
        else:
            _check_mask(mask)
        if accum_arg is not None:
            if accum is not None:
                raise TypeError("Got multiple values for argument 'accum'")
            accum = accum_arg
        if accum is not None:
            # Normalize accumulator
            accum = get_typed_op(accum, self.dtype)
            if accum.opclass == "Monoid":
                accum = accum.binaryop
            else:
                self._expect_op(accum, "BinaryOp", within="__call__", keyword_name="accum")
        return Updater(self, mask=mask, accum=accum, replace=replace, input_mask=input_mask)

    def __or__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _ewise_infix_expr(self, other, method="ewise_add", within="__or__")

    def __ror__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _ewise_infix_expr(other, self, method="ewise_add", within="__ror__")

    def __ior__(self, other):
        raise TypeError(
            f"Using __ior__ (e.g., x |= y) is not supported for type {type(self).__name__}."
            "  To create an ewise_add expression using infix notation, do e.g. `op(x | y)`."
        )

    def __and__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _ewise_infix_expr(self, other, method="ewise_mult", within="__and__")

    def __rand__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _ewise_infix_expr(self, other, method="ewise_mult", within="__rand__")

    def __iand__(self, other):
        raise TypeError(
            f"Using __iand__ (e.g., x &= y) is not supported for type {type(self).__name__}."
            "  To create an ewise_mult expression using infix notation, do e.g. `op(x & y)`."
        )

    def __matmul__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _matmul_infix_expr(self, other, within="__matmul__")

    def __rmatmul__(self, other):
        if self._is_scalar:
            return NotImplemented
        return _matmul_infix_expr(other, self, within="__rmatmul__")

    def __imatmul__(self, other):
        raise TypeError(
            f"Using __imatmul__ (e.g., x @= y) is not supported for type {type(self).__name__}."
            "  To create an matmul expression using infix notation, do e.g. `semiring(x @ y)`."
        )

    def __eq__(self, other):
        raise TypeError(
            f"__eq__ not defined for objects of type {type(self)}.  Use `.isequal` method instead."
        )

    def __bool__(self):
        raise TypeError(
            f"__bool__ not defined for objects of type {type(self)}.  "
            "Perhaps use .nvals attribute instead."
        )

    def __lshift__(self, delayed):
        return self._update(delayed)

    def update(self, delayed):
        """
        Convenience function when no output arguments (mask, accum, replace) are used
        """
        return self._update(delayed)

    def _update(self, delayed, mask=None, accum=None, replace=False, input_mask=None):
        # TODO: check expected output type (now included in Expression object)
        if not isinstance(delayed, BaseExpression):
            if type(delayed) is AmbiguousAssignOrExtract:
                if delayed.resolved_indexes.is_single_element and self._is_scalar:
                    # Extract element (s << v[1])
                    if accum is not None:
                        raise TypeError(
                            "Scalar accumulation with extract element"
                            "--such as `s(accum=accum) << v[0]`--is not supported"
                        )
                    self.value = delayed.new(dtype=self.dtype, name="s_extract").value
                    return

                # Extract (C << A[rows, cols])
                if input_mask is not None:
                    if mask is not None:
                        raise TypeError("mask and input_mask arguments cannot both be given")
                    _check_mask(input_mask, output=delayed.parent)
                    mask = delayed._input_mask_to_mask(input_mask)
                    input_mask = None
                delayed = delayed._extract_delayed()
            elif type(delayed) is type(self):
                # Simple assignment (w << v)
                if self._is_scalar:
                    if accum is not None:
                        raise TypeError(
                            "Scalar update with accumulation--such as `s(accum=accum) << t`"
                            "--is not supported"
                        )
                    self.value = delayed.value
                    return

                delayed = delayed.apply(identity)
            elif self._is_scalar:
                if accum is not None:
                    raise TypeError(
                        "Scalar update with accumulation--such as `s(accum=accum) << t`"
                        "--is not supported"
                    )
                self.value = delayed
                return

            else:
                from .matrix import Matrix, TransposedMatrix, MatrixExpression

                if type(delayed) is TransposedMatrix and type(self) is Matrix:
                    # Transpose (C << A.T)
                    delayed = MatrixExpression(
                        "transpose",
                        "GrB_transpose",
                        [delayed],
                        expr_repr="{0}",
                        dtype=delayed.dtype,
                        nrows=delayed._nrows,
                        ncols=delayed._ncols,
                    )
                else:
                    from .scalar import Scalar

                    if type(delayed) is Scalar:
                        scalar = delayed
                    else:
                        try:
                            scalar = Scalar.from_value(delayed, name="")
                        except TypeError:
                            raise TypeError(
                                "Assignment value must be a valid expression type, not "
                                f"{type(delayed)}.\n\nValid expression types include "
                                f"{type(self).__name__}, {type(self).__name__}Expression, "
                                "AmbiguousAssignOrExtract, and scalars."
                            )
                    updater = self(mask=mask, accum=accum, replace=replace, input_mask=input_mask)
                    if type(self) is Matrix:
                        if mask is None:
                            raise TypeError(
                                "Warning: updating a Matrix with a scalar without a mask will "
                                "make the Matrix dense.  This may use a lot of memory and probably "
                                "isn't what you want.  Perhaps you meant:"
                                "\n\n    M(M.S) << s\n\n"
                                "If you do wish to make a dense matrix, then please be explicit:"
                                "\n\n    M[:, :] = s"
                            )
                        updater[:, :] = scalar
                    else:  # Vector
                        updater[:] = scalar
                    return

        if input_mask is not None:
            raise TypeError("`input_mask` argument may only be used for extract")
        # Normalize mask and separate out complement and structural flags
        if mask is None:
            complement = False
            structure = False
        else:
            _check_mask(mask, self)
            complement = mask.complement
            structure = mask.structure

        # Get descriptor based on flags
        desc = descriptor_lookup(
            transpose_first=delayed.at,
            transpose_second=delayed.bt,
            mask_complement=complement,
            mask_structure=structure,
            output_replace=replace,
        )
        if self._is_scalar:
            args = [_Pointer(self), accum]
            cfunc_name = delayed.cfunc_name.format(output_dtype=self.dtype)
        else:
            args = [self, mask, accum]
            cfunc_name = delayed.cfunc_name
        if delayed.op is not None:
            args.append(delayed.op)
        args.extend(delayed.args)
        args.append(desc)
        # Make the GraphBLAS call
        call(cfunc_name, args)
        if self._is_scalar:
            self._is_empty = False

    @property
    def _name_html(self):
        """Treat characters after _ as subscript"""
        split = self.name.split("_", 1)
        if len(split) == 1:
            return self.name
        return f"{split[0]}<sub>{split[1]}</sub>"

    _expect_type = _expect_type
    _expect_op = _expect_op

    # Don't let objects be coerced to numpy arrays
    def __array__(self, *args, **kwargs):
        raise TypeError(
            f"{type(self).__name__} can't be directly converted to a numpy array; "
            f"perhaps use `{self.name}.to_values()` method instead."
        )

    __array_struct__ = property(__array__)


class BaseExpression:
    __slots__ = (
        "method_name",
        "cfunc_name",
        "args",
        "at",
        "bt",
        "op",
        "expr_repr",
        "dtype",
        "__weakref__",
    )
    output_type = None

    def __init__(
        self,
        method_name,
        cfunc_name,
        args,
        *,
        at=False,
        bt=False,
        op=None,
        dtype=None,
        expr_repr=None,
    ):
        self.method_name = method_name
        self.cfunc_name = cfunc_name
        self.args = args
        self.at = at
        self.bt = bt
        self.op = op
        if expr_repr is None:
            if len(args) == 1:
                expr_repr = "{0.name}.{method_name}({op})"
            elif len(args) == 2:
                expr_repr = "{0.name}.{method_name}({1.name}, op={op})"
            else:  # pragma: no cover
                raise ValueError(f"No default expr_repr for len(args) == {len(args)}")
        self.expr_repr = expr_repr
        if dtype is None:
            self.dtype = op.return_type
        else:
            self.dtype = dtype

    def new(self, *, dtype=None, mask=None, name=None):
        output = self.construct_output(dtype=dtype, name=name)
        if mask is None:
            output.update(self)
        else:
            _check_mask(mask, output)
            output(mask=mask).update(self)
        return output

    def __eq__(self, other):
        raise TypeError(
            f"__eq__ not defined for objects of type {type(self)}.  "
            f"Use `.new()` to create a new {self.output_type.__name__}, then use `.isequal` method."
        )

    def __bool__(self):
        raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")

    def _format_expr(self):
        return self.expr_repr.format(*self.args, method_name=self.method_name, op=self.op)

    def _format_expr_html(self):
        expr_repr = self.expr_repr.replace(".name", "._name_html")
        return expr_repr.format(*self.args, method_name=self.method_name, op=self.op)

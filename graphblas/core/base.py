from contextvars import ContextVar

from .. import backend, config
from .. import replace as replace_singleton
from ..dtypes import BOOL
from ..exceptions import check_status
from . import NULL
from .descriptor import lookup as descriptor_lookup
from .expr import AmbiguousAssignOrExtract, Updater
from .mask import Mask
from .operator import UNKNOWN_OPCLASS, binary_from_string, find_opclass, get_typed_op
from .utils import _Pointer, libget, output_type

_recorder = ContextVar("recorder")
_prev_recorder = None


def record_raw(text):
    if (rec := _recorder.get(_prev_recorder)) is not None:
        rec.record_raw(text)


def call(cfunc_name, args):
    call_args = [getattr(x, "_carg", x) if x is not None else NULL for x in args]
    cfunc = libget(cfunc_name)
    try:
        err_code = cfunc(*call_args)
    except TypeError as exc:
        # We should strive to not encounter this during normal usage
        from .recorder import gbstr

        callstr = f'{cfunc.__name__}({", ".join(gbstr(x) for x in args)})'
        # calltypes = f'{cfunc.__name__}({", ".join(str(x) for x in call_args)})'
        lines = cfunc.__doc__.splitlines()
        sig = lines[0] if lines else ""
        raise TypeError(
            f"Error calling {cfunc.__name__}:\n"
            f" - Call objects: {callstr}\n"
            # f" - Call types: {calltypes}\n"  # Useful during development
            f" - C signature: {sig}\n"
            f" - Error: {exc}"
        ) from None
    try:
        rv = check_status(err_code, args)
    except Exception as exc:
        # Record calls that fail for easier debugging
        rec = _recorder.get(_prev_recorder)
        if rec is not None:
            rec.record(cfunc_name, args, exc=exc)
        raise
    rec = _recorder.get(_prev_recorder)
    if rec is not None:
        rec.record(cfunc_name, args)
    return rv


def _expect_type_message(
    self, x, types, *, within, argname=None, keyword_name=None, op=None, extra_message=""
):
    if type(types) is tuple:
        if type(x) in types:
            return x, None
        if output_type(x) in types:
            if config.get("autocompute"):
                return x.new(), None
            extra_message = f"{extra_message}\n\n" if extra_message else ""
            extra_message += (
                "Hint: use `graphblas.config.set(autocompute=True)` to automatically "
                "compute arguments that are expressions."
            )
    elif type(x) is types:
        return x, None
    elif output_type(x) is types:
        if config.get("autocompute"):
            return x.new(), None
        extra_message = f"{extra_message}\n\n" if extra_message else ""
        extra_message += (
            "Hint: use `graphblas.config.set(autocompute=True)` to automatically "
            "compute arguments that are expressions."
        )
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
    op, opclass = find_opclass(op)
    if opclass == UNKNOWN_OPCLASS:
        argstr = "..."
    else:
        argstr = f"op={op}"
    return x, (
        f"Bad type {argmsg}in {type(self).__name__}.{within}({argstr}).\n"
        f"    - Expected type: {expected}.\n"
        f"    - Got: {type(x)}."
        f"{extra_message}"
    )


def _expect_type(self, x, types, **kwargs):
    x, message = _expect_type_message(self, x, types, **kwargs)
    if message is not None:
        raise TypeError(message) from None
    return x


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
    else:  # pragma: no cover (safety)
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
        special_message = f"\nThe BinaryOp {op.name} is not known to be part of a Monoid"
        if op.parent.monoid is not None:
            special_message += f" for {op.type} datatype."
        else:
            special_message += "."
    if extra_message:
        extra_message = f"\n{extra_message}"
    return (
        f"Bad type {argmsg}in {type(self).__name__}.{within}(...).\n"
        f"    - Expected type: {expected}.\n"
        f"    - Got: {op.opclass} ({op})."
        f"{extra_message}"
        f"{special_message}"
    )


def _expect_op(self, op, values, *, within, **kwargs):
    if (message := _expect_op_message(self, op, values, within=within, **kwargs)) is not None:
        raise TypeError(message) from None


AmbiguousAssignOrExtract._expect_op = _expect_op
AmbiguousAssignOrExtract._expect_type = _expect_type


def _check_mask(mask, output=None):
    if not isinstance(mask, Mask):
        # Convert bool objects to value masks
        if output_type(mask).__name__ in {"Vector", "Matrix"}:
            if mask.dtype != BOOL:
                raise TypeError(
                    f"Mask must be boolean objects (got {mask.dtype}) "
                    "or indicate values (M.V) or structure (M.S)"
                )
            mask = mask.V  # auto-compute (will raise if disabled)
        else:
            raise TypeError(f"Invalid mask: {type(mask)}")
    if output is not None and output.ndim == 1 and mask.parent.ndim != 1:
        raise TypeError(f"Mask object must be type Vector; got {type(mask.parent)}")
    return mask


class BaseType:
    # pylint: disable=assigning-non-slot
    __slots__ = "gb_obj", "dtype", "name", "__weakref__"
    # Flag for operations which depend on scalar vs vector/matrix
    _is_scalar = False

    def __call__(
        self,
        *optional_mask_accum_replace,
        mask=None,
        accum=None,
        replace=False,
        input_mask=None,
        **opts,
    ):
        # Pick out mask and accum from positional arguments
        mask_arg = None
        accum_arg = None
        for arg in optional_mask_accum_replace:
            if arg is replace_singleton:
                replace = True
            elif isinstance(arg, (BaseType, Mask)) or output_type(arg).__name__ in {
                "Vector",
                "Matrix",
                "TransposedMatrix",  # Included here so we can give a better error message
            }:
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
                    if isinstance(accum_arg, str):
                        accum_arg = binary_from_string(accum_arg)
                    else:
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
                input_mask = _check_mask(input_mask)
        elif self._is_scalar:
            raise TypeError("Mask not allowed for Scalars")
        elif input_mask is not None:
            raise TypeError("mask and input_mask arguments cannot both be given")
        else:
            mask = _check_mask(mask)
        if accum_arg is not None:
            if accum is not None:
                raise TypeError("Got multiple values for argument 'accum'")
            accum = accum_arg
        if accum is not None:
            # Normalize accumulator
            accum = get_typed_op(accum, self.dtype, kind="binary")
            if accum.opclass == "Monoid":
                accum = accum.binaryop
            else:
                self._expect_op(accum, "BinaryOp", within="__call__", keyword_name="accum")
        return Updater(
            self, mask=mask, accum=accum, replace=replace, input_mask=input_mask, opts=opts
        )

    def __or__(self, other):
        from .infix import _ewise_infix_expr, _ewise_mult_expr_types

        if isinstance(other, _ewise_mult_expr_types):
            raise TypeError("XXX")
        return _ewise_infix_expr(self, other, method="ewise_add", within="__or__")

    def __ror__(self, other):
        from .infix import _ewise_infix_expr, _ewise_mult_expr_types

        if isinstance(other, _ewise_mult_expr_types):
            raise TypeError("XXX")
        return _ewise_infix_expr(other, self, method="ewise_add", within="__ror__")

    def __and__(self, other):
        from .infix import _ewise_add_expr_types, _ewise_infix_expr

        if isinstance(other, _ewise_add_expr_types):
            raise TypeError("XXX")
        return _ewise_infix_expr(self, other, method="ewise_mult", within="__and__")

    def __rand__(self, other):
        from .infix import _ewise_add_expr_types, _ewise_infix_expr

        if isinstance(other, _ewise_add_expr_types):
            raise TypeError("XXX")
        return _ewise_infix_expr(other, self, method="ewise_mult", within="__rand__")

    def __matmul__(self, other):
        if self._is_scalar:
            return NotImplemented
        from .infix import _matmul_infix_expr

        return _matmul_infix_expr(self, other, within="__matmul__")

    def __rmatmul__(self, other):
        if self._is_scalar:
            return NotImplemented
        from .infix import _matmul_infix_expr

        return _matmul_infix_expr(other, self, within="__rmatmul__")

    def __imatmul__(self, other):
        if self._is_scalar:
            raise TypeError("'__imatmul__' not supported for Scalar")
        self << self @ other
        return self

    def __bool__(self):
        raise TypeError(
            f"__bool__ not defined for objects of type {type(self)}.  "
            "Perhaps use .nvals attribute instead."
        )

    def __lshift__(self, expr, **opts):
        return self._update(expr, opts=opts)

    def update(self, expr, **opts):
        """Convenience function when no output arguments (mask, accum, replace) are used."""
        return self._update(expr, opts=opts)

    def _update(self, expr, mask=None, accum=None, replace=False, input_mask=None, *, opts):
        if not isinstance(expr, BaseExpression):
            if isinstance(expr, AmbiguousAssignOrExtract):
                if expr._is_scalar and self._is_scalar:
                    # Extract element (s << v[1])
                    if accum is not None:
                        self(**opts) << self.ewise_add(expr, accum)
                        return
                    expr.parent._extract_element(
                        expr.resolved_indexes,
                        self.dtype,
                        opts,
                        is_cscalar=self._is_cscalar,
                        result=self,
                    )
                    return

                # Extract (C << A[rows, cols])
                if input_mask is not None:
                    if mask is not None:  # pragma: no cover (can this be covered?!)
                        raise TypeError("mask and input_mask arguments cannot both be given")
                    input_mask = _check_mask(input_mask, expr.parent)
                    mask = expr._input_mask_to_mask(input_mask)
                    input_mask = None
                expr = expr._extract_delayed()
            elif type(expr) is type(self):
                # Simple assignment (w << v)
                if self._is_scalar:
                    if accum is not None:
                        self(**opts) << self.ewise_add(expr, accum)
                        return
                    if opts:
                        # Ignore opts for now
                        desc = descriptor_lookup(**opts)
                    self.value = expr
                    return

                # Two choices here: apply identity `expr = expr.apply(identity)`, or assign.
                # Choose assign for now, since it works better for iso-valued objects.
                # Perhaps we should benchmark to see which is faster and has less Python overhead.
                self(mask=mask, accum=accum, replace=replace, input_mask=input_mask, **opts)[
                    ...
                ] = expr
                return
            elif self._is_scalar:
                from .infix import InfixExprBase

                if isinstance(expr, InfixExprBase):
                    # s << (v @ v)
                    expr = expr._to_expr()
                elif accum is not None:
                    self(**opts) << self.ewise_add(expr, accum)
                    return
                else:
                    if opts:
                        # Ignore opts for now
                        desc = descriptor_lookup(**opts)
                    self.value = expr
                    return
            else:
                from .infix import InfixExprBase
                from .matrix import Matrix, MatrixExpression, TransposedMatrix

                if type(expr) is TransposedMatrix and type(self) is Matrix:
                    # Transpose (C << A.T)
                    expr = MatrixExpression(
                        "transpose",
                        "GrB_transpose",
                        [expr],
                        expr_repr="{0}",
                        dtype=expr.dtype,
                        nrows=expr._nrows,
                        ncols=expr._ncols,
                    )
                elif isinstance(expr, InfixExprBase):
                    # w << (v & v)
                    expr = expr._to_expr()
                else:
                    from .scalar import Scalar

                    if type(expr) is Scalar:
                        scalar = expr
                    else:
                        dtype = self.dtype if self.dtype._is_udt else None
                        try:
                            scalar = Scalar.from_value(expr, dtype, is_cscalar=None, name="")
                        except TypeError:
                            raise TypeError(
                                "Assignment value must be a valid expression type, not "
                                f"{type(expr)}.\n\nValid expression types include "
                                f"{type(self).__name__}, {type(self).__name__}Expression, "
                                "AmbiguousAssignOrExtract, and scalars."
                            ) from None
                    updater = self(
                        mask=mask, accum=accum, replace=replace, input_mask=input_mask, **opts
                    )
                    updater[...] = scalar
                    return

        if type(self) is not expr.output_type:
            if expr.output_type._is_scalar and config.get("autocompute"):
                self._update(expr.new(), mask, accum, replace, input_mask, opts=opts)
                return
            from .scalar import Scalar

            valid = (Scalar, type(self), type(type(self).__name__ + "Expression", (), {}))
            valid = valid[1:] if self._is_scalar else valid
            self._expect_type(expr, valid, within="update")

        if input_mask is not None:
            raise TypeError("`input_mask` argument may only be used for extract")
        if expr.op is not None and expr.op.opclass == "Aggregator":
            updater = self(mask=mask, accum=accum, replace=replace, **opts)
            expr.op._new(updater, expr)
            return
        if expr.cfunc_name is None:  # Custom recipe
            updater = self(mask=mask, accum=accum, replace=replace, **opts)
            expr.args[-2](updater, *expr.args[-1])
            return

        # Normalize mask and separate out complement and structural flags
        if mask is None:
            complement = False
            structure = False
        else:
            mask = _check_mask(mask, self)
            complement = mask.complement
            structure = mask.structure

        # Get descriptor based on flags
        desc = descriptor_lookup(
            transpose_first=expr.at,
            transpose_second=expr.bt,
            mask_complement=complement,
            mask_structure=structure,
            output_replace=replace,
            **opts,
        )
        if self._is_scalar:
            if expr._scalar_as_vector:
                fake_self = self._as_vector()
                cfunc_name = expr.cfunc_name
                args = [fake_self, mask, accum]
            else:
                is_temp_scalar = expr._is_cscalar != self._is_cscalar
                if is_temp_scalar:
                    temp_scalar = expr.construct_output(self.dtype, name="s_temp")
                    if accum is not None and not self._is_empty:
                        temp_scalar.value = self
                else:
                    temp_scalar = self
                if expr._is_cscalar:
                    cfunc_name = expr.cfunc_name.format(output_dtype=self.dtype)
                    args = [_Pointer(temp_scalar), accum]
                else:
                    cfunc_name = expr.cfunc_name
                    args = [temp_scalar, accum]
        else:
            args = [self, mask, accum]
            cfunc_name = expr.cfunc_name
        if expr.op is not None:
            args.append(expr.op)
        args.extend(expr.args)
        args.append(desc)
        # Make the GraphBLAS call
        call(cfunc_name, args)
        if self._is_scalar:
            if expr._scalar_as_vector:
                if self._is_cscalar or backend != "suitesparse":
                    self.value = fake_self[0].new(is_cscalar=True, name="")
                # SS: this assumes GrB_Scalar was cast to Vector
            elif is_temp_scalar:
                if temp_scalar._is_cscalar:
                    temp_scalar._empty = False
                self.value = temp_scalar
            elif self._is_cscalar:
                self._empty = False

    @property
    def _name_html(self):
        """Treat characters after _ as subscript."""
        split = self.name.split("_", 1)
        if len(split) == 1:
            return self.name
        return f"{split[0]}<sub>{split[1]}</sub>"

    _expect_type = _expect_type
    _expect_op = _expect_op

    # Don't let non-scalars be coerced to numpy arrays
    def __array__(self, dtype=None, *, copy=None):
        raise TypeError(
            f"{type(self).__name__} can't be directly converted to a numpy array; "
            f"perhaps use `{self.name}.to_coo()` method instead."
        )


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
        "_value",
        "__weakref__",
    )
    output_type = None
    _is_scalar = False

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
            if len(args) == 1 or cfunc_name is None and len(args) == 3:
                expr_repr = "{0.name}.{method_name}({op})"
            elif len(args) == 2 or cfunc_name is None and len(args) == 4:
                expr_repr = "{0.name}.{method_name}({1.name}, op={op})"
            else:  # pragma: no cover (sanity)
                raise ValueError(f"No default expr_repr for len(args) == {len(args)}")
        self.expr_repr = expr_repr
        if dtype is None:
            self.dtype = op.return_type
        else:
            self.dtype = dtype
        self._value = None

    def new(self, dtype=None, *, mask=None, name=None, **opts):
        return self._new(dtype, mask, name, **opts)

    def _new(self, dtype, mask, name, is_cscalar=None, **opts):
        if (
            mask is None
            and self._value is not None
            and (dtype is None or self._value.dtype == dtype)
        ):
            if opts:
                # Ignore opts for now
                desc = descriptor_lookup(**opts)  # noqa: F841 (keep desc in scope for context)
            if self._is_scalar and self._value._is_cscalar != is_cscalar:
                return self._value.dup(is_cscalar=is_cscalar, name=name)
            rv = self._value
            if name is not None:
                rv.name = name
            self._value = None
            return rv
        if is_cscalar is None:
            output = self.construct_output(dtype, name=name)
        else:
            output = self.construct_output(dtype, name=name, is_cscalar=is_cscalar)
        if self.op is not None and self.op.opclass == "Aggregator":
            updater = output(mask=mask, **opts)
            self.op._new(updater, self)
        elif self.cfunc_name is None:  # Custom recipe
            self.args[-2](output(mask=mask, **opts), *self.args[-1])
        elif mask is None:
            output.update(self, **opts)
        else:
            mask = _check_mask(mask, output)
            output(mask=mask, **opts).update(self)
        return output

    def _format_expr(self):
        args = self.args[:-2] if self.cfunc_name is None else self.args
        return self.expr_repr.format(*args, method_name=self.method_name, op=self.op)

    def _format_expr_html(self):
        expr_repr = self.expr_repr.replace(".name", "._name_html").replace(
            "._expr_name", "._expr_name_html"
        )
        args = self.args[:-2] if self.cfunc_name is None else self.args
        return expr_repr.format(*args, method_name=self.method_name, op=self.op)

    _expect_type = _expect_type
    _expect_op = _expect_op

    def _new_scalar(self, dtype, *, is_cscalar=False, name=None):
        """Create a new empty Scalar.

        This is useful for supporting other graphblas-compatible APIs in recipes.
        """
        from .scalar import Scalar

        return Scalar(dtype, is_cscalar=is_cscalar, name=name)

    def _new_vector(self, dtype, size=0, *, name=None):
        """Create a new empty Vector.

        This is useful for supporting other graphblas-compatible APIs in recipes.
        """
        from .vector import Vector

        return Vector(dtype, size, name=name)

    def _new_matrix(self, dtype, nrows=0, ncols=0, *, name=None):
        """Create a new empty Matrix.

        This is useful for supporting other graphblas-compatible APIs in recipes.
        """
        from .matrix import Matrix

        return Matrix(dtype, nrows, ncols, name=name)

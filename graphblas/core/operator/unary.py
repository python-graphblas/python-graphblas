import inspect
import re
from types import FunctionType

from ... import _STANDARD_OPERATOR_NAMES, op, unary
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
    _sample_values,
    _supports_complex,
    lookup_dtype,
)
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, ffi, lib
from ..utils import output_type
from .base import (
    _SS_OPERATORS,
    OpBase,
    ParameterizedUdf,
    TypedOpBase,
    _deserialize_parameterized,
    _hasop,
)

if _supports_complex:
    from ...dtypes import FC32, FC64
if _has_numba:
    import numba

    from .base import _get_udt_wrapper

ffi_new = ffi.new


class TypedBuiltinUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "UnaryOp"

    def __call__(self, val):
        from ..matrix import Matrix, TransposedMatrix
        from ..vector import Vector

        if (typ := output_type(val)) in {Vector, Matrix, TransposedMatrix}:
            return val.apply(self)
        from ..scalar import Scalar, _as_scalar

        if typ is Scalar:
            return val.apply(self)
        try:
            scalar = _as_scalar(val, is_cscalar=False)
        except Exception:
            pass
        else:
            return scalar.apply(self)
        raise TypeError(
            f"Bad type when calling {self!r}.\n"
            "    - Expected type: Scalar, Vector, Matrix, TransposedMatrix.\n"
            f"    - Got: {type(val)}.\n"
            "Calling a UnaryOp is syntactic sugar for calling apply.  "
            f"For example, `A.apply({self!r})` is the same as `{self!r}(A)`."
        )


class TypedUserUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "UnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    __call__ = TypedBuiltinUnaryOp.__call__


class ParameterizedUnaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        unary = self.func(*args, **kwargs)
        unary._parameterized_info = (self, args, kwargs)
        return UnaryOp.register_anonymous(unary, self.name, is_udt=self._is_udt)

    def __reduce__(self):
        name = f"unary.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:  # pragma: no cover
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
        if anonymous:
            return UnaryOp.register_anonymous(func, name, parameterized=True)
        if (rv := UnaryOp._find(name)) is not None:
            return rv
        return UnaryOp.register_new(name, func, parameterized=True)


def _identity(x):
    return x  # pragma: no cover (numba)


def _one(x):
    return 1  # pragma: no cover (numba)


class UnaryOp(OpBase):
    """Takes one input and returns one output, possibly of a different data type.

    Built-in and registered UnaryOps are located in the ``graphblas.unary`` namespace
    as well as in the ``graphblas.ops`` combined namespace.
    """

    __slots__ = "orig_func", "is_positional", "_is_udt", "_numba_func"
    _custom_dtype = None
    _module = unary
    _modname = "unary"
    _typed_class = TypedBuiltinUnaryOp
    _parse_config = {
        "trim_from_front": 4,
        "num_underscores": 1,
        "re_exprs": [
            re.compile(
                "^GrB_(IDENTITY|AINV|MINV|ABS|BNOT)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile(
                "^GxB_(LNOT|ONE|POSITIONI1|POSITIONI|POSITIONJ1|POSITIONJ)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(SQRT|LOG|EXP|LOG2|SIN|COS|TAN|ACOS|ASIN|ATAN|SINH|COSH|TANH|ACOSH"
                "|ASINH|ATANH|SIGNUM|CEIL|FLOOR|ROUND|TRUNC|EXP2|EXPM1|LOG10|LOG1P)"
                "_(FP32|FP64|FC32|FC64)$"
            ),
            re.compile("^GxB_(LGAMMA|TGAMMA|ERF|ERFC|FREXPX|FREXPE|CBRT)_(FP32|FP64)$"),
            re.compile("^GxB_(IDENTITY|AINV|MINV|ONE|CONJ)_(FC32|FC64)$"),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_LNOT$"),
            re.compile("^GxB_(ISINF|ISNAN|ISFINITE)_(FP32|FP64|FC32|FC64)$"),
        ],
        "re_exprs_return_float": [re.compile("^GxB_(CREAL|CIMAG|CARG|ABS)_(FC32|FC64)$")],
    }
    _positional = {"positioni", "positioni1", "positionj", "positionj1"}

    @classmethod
    def _build(cls, name, func, *, anonymous=False, is_udt=False):
        if type(func) is not FunctionType:
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_unary>")
        success = False
        unary_udf = numba.njit(func)
        new_type_obj = cls(name, func, anonymous=anonymous, is_udt=is_udt, numba_func=unary_udf)
        return_types = {}
        nt = numba.types
        if not is_udt:
            for type_ in _sample_values:
                sig = (type_.numba_type,)
                try:
                    unary_udf.compile(sig)
                except numba.TypingError:
                    continue
                ret_type = lookup_dtype(unary_udf.overloads[sig].signature.return_type)
                if ret_type != type_ and (
                    ("INT" in ret_type.name and "INT" in type_.name)
                    or ("FP" in ret_type.name and "FP" in type_.name)
                    or ("FC" in ret_type.name and "FC" in type_.name)
                    or (type_ == UINT64 and ret_type == FP64 and return_types.get(INT64) == INT64)
                ):
                    # Downcast `ret_type` to `type_`.
                    # This is what users want most of the time, but we can't make a perfect rule.
                    # There should be a way for users to be explicit.
                    ret_type = type_
                elif type_ == BOOL and ret_type == INT64 and return_types.get(INT8) == INT8:
                    ret_type = INT8

                # Numba is unable to handle BOOL correctly right now, but we have a workaround
                # See: https://github.com/numba/numba/issues/5395
                # We're relying on coercion behaving correctly here
                input_type = INT8 if type_ == BOOL else type_
                return_type = INT8 if ret_type == BOOL else ret_type

                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(
                    nt.CPointer(return_type.numba_type),
                    nt.CPointer(input_type.numba_type),
                )

                if type_ == BOOL:
                    if ret_type == BOOL:

                        def unary_wrapper(z, x):
                            z[0] = bool(unary_udf(bool(x[0])))  # pragma: no cover (numba)

                    else:

                        def unary_wrapper(z, x):
                            z[0] = unary_udf(bool(x[0]))  # pragma: no cover (numba)

                elif ret_type == BOOL:

                    def unary_wrapper(z, x):
                        z[0] = bool(unary_udf(x[0]))  # pragma: no cover (numba)

                else:

                    def unary_wrapper(z, x):
                        z[0] = unary_udf(x[0])  # pragma: no cover (numba)

                unary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(unary_wrapper)
                new_unary = ffi_new("GrB_UnaryOp*")
                check_status_carg(
                    lib.GrB_UnaryOp_new(
                        new_unary, unary_wrapper.cffi, ret_type.gb_obj, type_.gb_obj
                    ),
                    "UnaryOp",
                    new_unary,
                )
                op = TypedUserUnaryOp(new_type_obj, name, type_, ret_type, new_unary[0])
                new_type_obj._add(op)
                success = True
                return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if dtype in self._udt_types:
            return self._udt_ops[dtype]

        numba_func = self._numba_func
        sig = (dtype.numba_type,)
        numba_func.compile(sig)  # Should we catch and give additional error message?
        ret_type = lookup_dtype(numba_func.overloads[sig].signature.return_type)

        unary_wrapper, wrapper_sig = _get_udt_wrapper(numba_func, ret_type, dtype)
        unary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(unary_wrapper)
        new_unary = ffi_new("GrB_UnaryOp*")
        check_status_carg(
            lib.GrB_UnaryOp_new(new_unary, unary_wrapper.cffi, ret_type._carg, dtype._carg),
            "UnaryOp",
            new_unary,
        )
        op = TypedUserUnaryOp(self, self.name, dtype, ret_type, new_unary[0])
        self._udt_types[dtype] = ret_type
        self._udt_ops[dtype] = op
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register a UnaryOp without registering it in the ``graphblas.unary`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes one input parameters of any dtype and returns any dtype.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.unary.{name}``.
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

        Returns
        -------
        UnaryOp or ParameterizedUnaryOp
        """
        cls._check_supports_udf("register_anonymous")
        if parameterized:
            return ParameterizedUnaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register a new UnaryOp and save it to ``graphblas.unary`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.unary.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.unary.x.y.z`` for name ``"x.y.z"``.
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes one input parameters of any dtype and returns any dtype.
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
        lazy : bool, default False
            If False (the default), then the function will be automatically
            compiled for builtin types (unless ``is_udt`` is True).
            Compiling functions can be slow, however, so you may want to
            delay compilation and only compile when the operator is used,
            which is done by setting ``lazy=True``.

        Examples
        --------
        >>> gb.core.operator.UnaryOp.register_new("plus_one", lambda x: x + 1)
        >>> dir(gb.unary)
        [..., 'plus_one', ...]
        """
        cls._check_supports_udf("register_new")
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "func": func, "parameterized": parameterized},
            )
        elif parameterized:
            unary_op = ParameterizedUnaryOp(name, func, is_udt=is_udt)
            setattr(module, funcname, unary_op)
        else:
            unary_op = cls._build(name, func, is_udt=is_udt)
            setattr(module, funcname, unary_op)
        # Also save it to `graphblas.op` if not yet defined
        opmodule, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not _hasop(opmodule, funcname):
            if lazy:
                opmodule._delayed[funcname] = module
            else:
                setattr(opmodule, funcname, unary_op)
        if not cls._initialized:  # pragma: no cover
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return unary_op

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        super()._initialize()
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
        for names, *types in [
            # fmt: off
            (
                (
                    "erf", "erfc", "lgamma", "tgamma", "acos", "acosh", "asin", "asinh",
                    "atan", "atanh", "ceil", "cos", "cosh", "exp", "exp2", "expm1", "floor",
                    "log", "log10", "log1p", "log2", "round", "signum", "sin", "sinh", "sqrt",
                    "tan", "tanh", "trunc", "cbrt",
                ),
                ((BOOL, INT8, INT16, UINT8, UINT16), FP32),
                ((INT32, INT64, UINT32, UINT64), FP64),
            ),
            (
                ("positioni", "positioni1", "positionj", "positionj1"),
                (
                    position_dtypes,
                    INT64,
                ),
            ),
            # fmt: on
        ]:
            for name in names:
                if name in _SS_OPERATORS:
                    op = unary._deprecated[name]
                else:
                    op = getattr(unary, name)
                for input_types, target_type in types:
                    typed_op = op._typed_ops[target_type]
                    output_type = op.types[target_type]
                    for dtype in input_types:
                        if dtype not in op.types:  # pragma: no branch (safety)
                            op.types[dtype] = output_type
                            op._typed_ops[dtype] = typed_op
                            op.coercions[dtype] = target_type
        # Allow some functions to work on UDTs
        for unop, func in [
            (unary.identity, _identity),
            (unary.one, _one),
        ]:
            unop.orig_func = func
            if _has_numba:
                unop._numba_func = numba.njit(func)
            else:
                unop._numba_func = None
            unop._udt_types = {}
            unop._udt_ops = {}
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
        self.orig_func = func
        self._numba_func = numba_func
        self.is_positional = is_positional
        self._is_udt = is_udt
        if is_udt:
            self._udt_types = {}  # {dtype: DataType}
            self._udt_ops = {}  # {dtype: TypedUserUnaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"unary.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinUnaryOp.__call__

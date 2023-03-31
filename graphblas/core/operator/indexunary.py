import inspect
import re
from types import FunctionType

import numba

from ... import _STANDARD_OPERATOR_NAMES, indexunary, select
from ...dtypes import BOOL, FP64, INT8, INT64, UINT64, _sample_values, lookup_dtype
from ...exceptions import UdfParseError, check_status_carg
from .. import ffi, lib
from .base import (
    OpBase,
    ParameterizedUdf,
    TypedOpBase,
    _call_op,
    _deserialize_parameterized,
    _get_udt_wrapper,
)

ffi_new = ffi.new


class TypedBuiltinIndexUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexUnaryOp"

    def __call__(self, val, thunk=None):
        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        return _call_op(self, val, right=thunk)


class TypedUserIndexUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexUnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    __call__ = TypedBuiltinIndexUnaryOp.__call__


class ParameterizedIndexUnaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        indexunary = self.func(*args, **kwargs)
        indexunary._parameterized_info = (self, args, kwargs)
        return IndexUnaryOp.register_anonymous(indexunary, self.name, is_udt=self._is_udt)

    def __reduce__(self):
        name = f"indexunary.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
        if anonymous:
            return IndexUnaryOp.register_anonymous(func, name, parameterized=True)
        if (rv := IndexUnaryOp._find(name)) is not None:
            return rv
        return IndexUnaryOp.register_new(name, func, parameterized=True)


class IndexUnaryOp(OpBase):
    """Takes one input and a thunk and returns one output, possibly of a different data type.
    Along with the input value, the index(es) of the element are given to the function.

    This is an advanced form of a unary operation that allows, for example, converting
    elements of a Vector to their index position to build a ramp structure. Another use
    case is returning a boolean value indicating whether the element is part of the upper
    triangular structure of a Matrix.

    Built-in and registered IndexUnaryOps are located in the ``graphblas.indexunary`` namespace.
    """

    __slots__ = "orig_func", "is_positional", "_is_udt", "_numba_func"
    _module = indexunary
    _modname = "indexunary"
    _custom_dtype = None
    _typed_class = TypedBuiltinIndexUnaryOp
    _typed_user_class = TypedUserIndexUnaryOp
    _parse_config = {
        "trim_from_front": 4,
        "num_underscores": 1,
        "re_exprs": [
            re.compile("^GrB_(ROWINDEX|COLINDEX|DIAGINDEX)_(INT32|INT64)$"),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_(TRIL|TRIU|DIAG|OFFDIAG|COLLE|COLGT|ROWLE|ROWGT)$"),
            re.compile(
                "^GrB_(VALUEEQ|VALUENE|VALUEGT|VALUEGE|VALUELT|VALUELE)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile("^GxB_(VALUEEQ|VALUENE)_(FC32|FC64)$"),
        ],
    }
    _positional = {"tril", "triu", "diag", "offdiag", "colle", "colgt", "rowle", "rowgt",
                   "rowindex", "colindex"}  # fmt: skip

    @classmethod
    def _build(cls, name, func, *, is_udt=False, anonymous=False):
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_binary>")
        success = False
        indexunary_udf = numba.njit(func)
        new_type_obj = cls(
            name, func, anonymous=anonymous, is_udt=is_udt, numba_func=indexunary_udf
        )
        return_types = {}
        nt = numba.types
        if not is_udt:
            for type_ in _sample_values:
                sig = (type_.numba_type, UINT64.numba_type, UINT64.numba_type, type_.numba_type)
                try:
                    indexunary_udf.compile(sig)
                except numba.TypingError:
                    continue
                ret_type = lookup_dtype(indexunary_udf.overloads[sig].signature.return_type)
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
                    UINT64.numba_type,
                    UINT64.numba_type,
                    nt.CPointer(input_type.numba_type),
                )

                if type_ == BOOL:
                    if ret_type == BOOL:

                        def indexunary_wrapper(z, x, row, col, y):  # pragma: no cover (numba)
                            z[0] = bool(indexunary_udf(bool(x[0]), row, col, bool(y[0])))

                    else:

                        def indexunary_wrapper(z, x, row, col, y):  # pragma: no cover (numba)
                            z[0] = indexunary_udf(bool(x[0]), row, col, bool(y[0]))

                elif ret_type == BOOL:

                    def indexunary_wrapper(z, x, row, col, y):  # pragma: no cover (numba)
                        z[0] = bool(indexunary_udf(x[0], row, col, y[0]))

                else:

                    def indexunary_wrapper(z, x, row, col, y):  # pragma: no cover (numba)
                        z[0] = indexunary_udf(x[0], row, col, y[0])

                indexunary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(indexunary_wrapper)
                new_indexunary = ffi_new("GrB_IndexUnaryOp*")
                check_status_carg(
                    lib.GrB_IndexUnaryOp_new(
                        new_indexunary,
                        indexunary_wrapper.cffi,
                        ret_type.gb_obj,
                        type_.gb_obj,
                        type_.gb_obj,
                    ),
                    "IndexUnaryOp",
                    new_indexunary,
                )
                op = cls._typed_user_class(new_type_obj, name, type_, ret_type, new_indexunary[0])
                new_type_obj._add(op)
                success = True
                return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:  # pragma: no cover
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]

        numba_func = self._numba_func
        sig = (dtype.numba_type, UINT64.numba_type, UINT64.numba_type, dtype2.numba_type)
        numba_func.compile(sig)  # Should we catch and give additional error message?
        ret_type = lookup_dtype(numba_func.overloads[sig].signature.return_type)
        indexunary_wrapper, wrapper_sig = _get_udt_wrapper(
            numba_func, ret_type, dtype, dtype2, include_indexes=True
        )

        indexunary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(indexunary_wrapper)
        new_indexunary = ffi_new("GrB_IndexUnaryOp*")
        check_status_carg(
            lib.GrB_IndexUnaryOp_new(
                new_indexunary, indexunary_wrapper.cffi, ret_type._carg, dtype._carg, dtype2._carg
            ),
            "IndexUnaryOp",
            new_indexunary,
        )
        op = TypedUserIndexUnaryOp(
            self,
            self.name,
            dtype,
            ret_type,
            new_indexunary[0],
            dtype2=dtype2,
        )
        self._udt_types[dtypes] = ret_type
        self._udt_ops[dtypes] = op
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register an IndexUnaryOp without registering it in the
        ``graphblas.indexunary`` namespace.

        Because it is not registered in the namespace, the name is optional.
        """
        if parameterized:
            return ParameterizedIndexUnaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register an IndexUnaryOp. The name will be used to identify the IndexUnaryOp in the
        ``graphblas.indexunary`` namespace.

        If the return type is Boolean, the function will also be registered as a SelectOp
        with the same name.

            >>> gb.indexunary.register_new("row_mod", lambda x, i, j, thunk: i % max(thunk, 2))
            >>> dir(gb.indexunary)
            [..., 'row_mod', ...]
        """
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "func": func, "parameterized": parameterized},
            )
        elif parameterized:
            indexunary_op = ParameterizedIndexUnaryOp(name, func, is_udt=is_udt)
            setattr(module, funcname, indexunary_op)
        else:
            indexunary_op = cls._build(name, func, is_udt=is_udt)
            setattr(module, funcname, indexunary_op)
            # If return type is BOOL, register additionally as a SelectOp
            if all(x == BOOL for x in indexunary_op.types.values()):
                from .select import SelectOp

                setattr(select, funcname, SelectOp._from_indexunary(indexunary_op))

        if not cls._initialized:
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return indexunary_op

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        from .select import SelectOp

        super()._initialize(include_in_ops=False)
        # Update type information to include UINT64 for positional ops
        for name in ["tril", "triu", "diag", "offdiag", "colle", "colgt", "rowle", "rowgt"]:
            op = getattr(indexunary, name)
            typed_op = op._typed_ops[BOOL]
            output_type = op.types[BOOL]
            if UINT64 not in op.types:  # pragma: no branch (safety)
                op.types[UINT64] = output_type
                op._typed_ops[UINT64] = typed_op
                op.coercions[UINT64] = BOOL
        for name in ["rowindex", "colindex"]:
            op = getattr(indexunary, name)
            typed_op = op._typed_ops[INT64]
            output_type = op.types[INT64]
            if UINT64 not in op.types:  # pragma: no branch (safety)
                op.types[UINT64] = output_type
                op._typed_ops[UINT64] = typed_op
                op.coercions[UINT64] = INT64
        # Add index->row alias to make it more intuitive which to use for vectors
        indexunary.indexle = indexunary.rowle
        indexunary.indexgt = indexunary.rowgt
        indexunary.index = indexunary.rowindex
        # fmt: off
        # Add SelectOp when it makes sense
        for name in ["tril", "triu", "diag", "offdiag",
                     "colle", "colgt", "rowle", "rowgt", "indexle", "indexgt",
                     "valueeq", "valuene", "valuegt", "valuege", "valuelt", "valuele"]:
            iop = getattr(indexunary, name)
            setattr(select, name, SelectOp._from_indexunary(iop))
        # fmt: on
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
            self._udt_ops = {}  # {dtype: TypedUserIndexUnaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"indexunary.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinIndexUnaryOp.__call__

import inspect

from ... import _STANDARD_OPERATOR_NAMES, select
from ...dtypes import BOOL, UINT64
from ...exceptions import check_status_carg
from .. import _has_numba, ffi, lib
from .base import OpBase, ParameterizedUdf, TypedOpBase, _call_op, _deserialize_parameterized
from .indexunary import IndexUnaryOp, TypedBuiltinIndexUnaryOp

if _has_numba:
    import numba

    from .base import _get_udt_wrapper
ffi_new = ffi.new


class TypedBuiltinSelectOp(TypedOpBase):
    __slots__ = ()
    opclass = "SelectOp"

    def __call__(self, val, thunk=None):
        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        return _call_op(self, val, thunk=thunk)

    thunk_type = TypedBuiltinIndexUnaryOp.thunk_type


class TypedUserSelectOp(TypedOpBase):
    __slots__ = ()
    opclass = "SelectOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    thunk_type = TypedBuiltinSelectOp.thunk_type
    __call__ = TypedBuiltinSelectOp.__call__


class ParameterizedSelectOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        sel = self.func(*args, **kwargs)
        sel._parameterized_info = (self, args, kwargs)
        return SelectOp.register_anonymous(sel, self.name, is_udt=self._is_udt)

    def __reduce__(self):
        # NOT COVERED
        name = f"select.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
        # NOT COVERED
        if anonymous:
            return SelectOp.register_anonymous(func, name, parameterized=True)
        if (rv := SelectOp._find(name)) is not None:
            return rv
        return SelectOp.register_new(name, func, parameterized=True)


class SelectOp(OpBase):
    """Identical to an :class:`IndexUnaryOp <graphblas.core.operator.IndexUnaryOp>`,
    but must have a Boolean return type.

    A SelectOp is used exclusively to select a subset of values from a collection where
    the function returns True.

    Built-in and registered SelectOps are located in the ``graphblas.select`` namespace.
    """

    __slots__ = "orig_func", "is_positional", "_is_udt", "_numba_func"
    _module = select
    _modname = "select"
    _custom_dtype = None
    _typed_class = TypedBuiltinSelectOp
    _typed_user_class = TypedUserSelectOp

    @classmethod
    def _from_indexunary(cls, iop):
        obj = cls(
            iop.name,
            iop.orig_func,
            anonymous=iop._anonymous,
            is_positional=iop.is_positional,
            is_udt=iop._is_udt,
            numba_func=iop._numba_func,
        )
        if not all(x == BOOL for x in iop.types.values()):
            raise ValueError("SelectOp must have BOOL return type")
        for type_, t in iop._typed_ops.items():
            if iop.orig_func is not None:
                op = cls._typed_user_class(
                    obj,
                    iop.name,
                    t.type,
                    t.return_type,
                    t.gb_obj,
                )
            else:
                op = cls._typed_class(
                    obj,
                    iop.name,
                    t.type,
                    t.return_type,
                    t.gb_obj,
                    t.gb_name,
                )
            # type is not always equal to t.type, so can't use op._add
            # but otherwise perform the same logic
            obj._typed_ops[type_] = op
            obj.types[type_] = op.return_type
        return obj

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:  # pragma: no cover
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]
        if self._numba_func is None:
            raise KeyError(f"{self.name} does not work with {dtypes} types")

        # It would be nice if we could reuse compiling done for IndexUnaryOp
        numba_func = self._numba_func
        sig = (dtype.numba_type, UINT64.numba_type, UINT64.numba_type, dtype2.numba_type)
        numba_func.compile(sig)  # Should we catch and give additional error message?
        select_wrapper, wrapper_sig = _get_udt_wrapper(
            numba_func, BOOL, dtype, dtype2, include_indexes=True
        )

        select_wrapper = numba.cfunc(wrapper_sig, nopython=True)(select_wrapper)
        new_select = ffi_new("GrB_IndexUnaryOp*")
        check_status_carg(
            lib.GrB_IndexUnaryOp_new(
                new_select, select_wrapper.cffi, BOOL._carg, dtype._carg, dtype2._carg
            ),
            "IndexUnaryOp",
            new_select[0],
        )
        op = TypedUserSelectOp(
            self,
            self.name,
            dtype,
            BOOL,
            new_select[0],
            dtype2=dtype2,
        )
        self._udt_types[dtypes] = BOOL
        self._udt_ops[dtypes] = op
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register a SelectOp without registering it in the ``graphblas.select`` namespace.

        Because it is not registered in the namespace, the name is optional.
        The return type must be Boolean.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes four input parameters--any dtype, int64, int64,
            any dtype and returns boolean. The first argument (any dtype) is
            the value of the input Matrix or Vector, the second argument (int64)
            is the row index of the Matrix or the index of the Vector, the third
            argument (int64) is the column index of the Matrix or 0 for a Vector,
            and the fourth argument (any dtype) is the value of the input Scalar.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.select.{name}``.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
            For example, ``gb.binary.isclose`` is a parameterized BinaryOp that
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
        SelectOp or ParameterizedSelectOp

        """
        cls._check_supports_udf("register_anonymous")
        if parameterized:
            return ParameterizedSelectOp(name, func, anonymous=True, is_udt=is_udt)
        iop = IndexUnaryOp._build(name, func, anonymous=True, is_udt=is_udt)
        return SelectOp._from_indexunary(iop)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register a new SelectOp and save it to ``graphblas.select`` namespace.

        The function will also be registered as a IndexUnaryOp with the same name.
        The return type must be Boolean.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.select.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.select.x.y.z`` for name ``"x.y.z"``.
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes four input parameters--any dtype, int64, int64,
            any dtype and returns boolean. The first argument (any dtype) is
            the value of the input Matrix or Vector, the second argument (int64)
            is the row index of the Matrix or the index of the Vector, the third
            argument (int64) is the column index of the Matrix or 0 for a Vector,
            and the fourth argument (any dtype) is the value of the input Scalar.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
            For example, ``gb.binary.isclose`` is a parameterized BinaryOp that
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
        lazy : bool, default False
            If False (the default), then the function will be automatically
            compiled for builtin types (unless ``is_udt`` is True).
            Compiling functions can be slow, however, so you may want to
            delay compilation and only compile when the operator is used,
            which is done by setting ``lazy=True``.

        Examples
        --------
        >>> gb.select.register_new("upper_left_triangle", lambda x, i, j, thunk: i + j <= thunk)
        >>> dir(gb.select)
        [..., 'upper_left_triangle', ...]

        """
        cls._check_supports_udf("register_new")
        iop = IndexUnaryOp.register_new(
            name, func, parameterized=parameterized, is_udt=is_udt, lazy=lazy
        )
        module, funcname = cls._remove_nesting(name, strict=False)
        if lazy:
            module._delayed[funcname] = (
                cls._get_delayed,
                {"name": name},
            )
        elif parameterized:
            op = ParameterizedSelectOp(funcname, func, is_udt=is_udt)
            setattr(module, funcname, op)
            return op
        elif not all(x == BOOL for x in iop.types.values()):
            # Undo registration of indexunaryop
            imodule, funcname = IndexUnaryOp._remove_nesting(name, strict=False)
            delattr(imodule, funcname)
            raise ValueError("SelectOp must have BOOL return type")
        else:
            return getattr(module, funcname)

    @classmethod
    def _get_delayed(cls, name):
        imodule, funcname = IndexUnaryOp._remove_nesting(name, strict=False)
        iop = getattr(imodule, name)
        if not all(x == BOOL for x in iop.types.values()):
            raise ValueError("SelectOp must have BOOL return type")
        module, funcname = cls._remove_nesting(name, strict=False)
        return getattr(module, funcname)

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover (safety)
            return
        # IndexUnaryOp adds it boolean-returning objects to SelectOp
        IndexUnaryOp._initialize()
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
            # NOT COVERED
            self._udt_types = {}  # {dtype: DataType}
            self._udt_ops = {}  # {dtype: TypedUserIndexUnaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                # NOT COVERED
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"select.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        # NOT COVERED
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinSelectOp.__call__

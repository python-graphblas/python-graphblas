import inspect

from ... import _STANDARD_OPERATOR_NAMES, select
from ...dtypes import BOOL
from .base import OpBase, ParameterizedUdf, TypedOpBase, _call_op, _deserialize_parameterized
from .indexunary import IndexUnaryOp


class TypedBuiltinSelectOp(TypedOpBase):
    __slots__ = ()
    opclass = "SelectOp"

    def __call__(self, val, thunk=None):
        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        return _call_op(self, val, thunk=thunk)


class TypedUserSelectOp(TypedOpBase):
    __slots__ = ()
    opclass = "SelectOp"

    def __init__(self, parent, name, type_, return_type, gb_obj):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

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
        name = f"select.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
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

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register a SelectOp without registering it in the ``graphblas.select`` namespace.

        Because it is not registered in the namespace, the name is optional.
        """
        if parameterized:
            return ParameterizedSelectOp(name, func, anonymous=True, is_udt=is_udt)
        iop = IndexUnaryOp._build(name, func, anonymous=True, is_udt=is_udt)
        return SelectOp._from_indexunary(iop)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register a SelectOp. The name will be used to identify the SelectOp in the
        ``graphblas.select`` namespace.

        The function will also be registered as a IndexUnaryOp with the same name.

            >>> gb.select.register_new("upper_left_triangle", lambda x, i, j, thunk: i + j <= thunk)
            >>> dir(gb.select)
            [..., 'upper_left_triangle', ...]
        """
        iop = IndexUnaryOp.register_new(
            name, func, parameterized=parameterized, is_udt=is_udt, lazy=lazy
        )
        if not all(x == BOOL for x in iop.types.values()):
            raise ValueError("SelectOp must have BOOL return type")
        if lazy:
            return getattr(select, iop.name)

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
            self._udt_types = {}  # {dtype: DataType}
            self._udt_ops = {}  # {dtype: TypedUserIndexUnaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"select.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinSelectOp.__call__

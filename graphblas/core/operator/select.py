import inspect

from ... import select
from ...dtypes import BOOL, UINT64
from .. import _has_numba
from .base import OpBase, ParameterizedUdf, TypedOpBase, _call_op
from .indexunary import IndexUnaryOp, TypedBuiltinIndexUnaryOp

if _has_numba:
    from .base import _compile_udf_for_udt, _finalize_udt_op, _get_udt_wrapper


_VALUE_OPS = {"valueeq", "valuene", "valuelt", "valuele", "valuegt", "valuege"}


def _value_select_in_own_type(op, dtype, thunk, given):
    """Return ``op`` and ``thunk`` for ``select`` on built-in ``dtype``, as a value op of ``dtype``.

    SuiteSparse:GraphBLAS (8.0 to 10.5, at least) runs ``select`` with a value
    op typed for another type than the input's by reading the thunk's bytes as
    the input's type, where the spec casts both to the op's type: on an INT8
    vector, ``valueeq`` with ``1.0`` compared with 0, and ``valuelt`` with
    ``300`` compared with 44. ``apply`` with the same op is right. So the
    comparison is made in the input's own type, with a thunk that selects the
    same elements: for integers, ``x < 2.5`` is ``x < 3``, and a thunk no
    element can equal, or that every element passes, selects nothing or
    everything; for floats, the thunk is rounded toward the side that keeps
    the comparison exact. ``given`` is the thunk as the caller gave it, whose
    value is exact where ``thunk`` may have rounded it (an int beyond 64 bits
    is FP64). Other ops and types are returned unchanged.
    """
    # MAINT 2026-10-07: works around the SuiteSparse:GraphBLAS bug above (seen in
    # 8.0.2 through 10.6.0; fix proposed in DrTimothyAldenDavis/GraphBLAS#463).
    # Once every supported version has the fix, pass the typed op and thunk
    # through and delete this function.
    if op.type is dtype or dtype._is_udt or op.parent.name not in _VALUE_OPS:
        return op, thunk
    import math

    import numpy as np

    from ... import indexunary
    from ..scalar import Scalar

    parent = op.parent
    name = parent.name
    if (
        parent is not getattr(select, name)
        and parent is not getattr(indexunary, name)
        or dtype.np_type.kind not in "biuf"
        or thunk._is_empty
    ):
        return op, thunk
    value = given.value if type(given) is Scalar else given
    if isinstance(value, (np.generic, np.ndarray)):
        value = value.item()
    if isinstance(value, complex):
        if value.imag != 0:
            if name not in {"valueeq", "valuene"}:  # pragma: no cover (no complex order)
                return op, thunk
            value = math.nan  # equal to nothing, as a complex number is to a real one
        else:
            value = value.real
    np_type = dtype.np_type
    if np_type.kind == "f":
        nan, ftype = np_type.type(np.nan), np_type.type
        if value != value:  # NaN: only ``!=`` holds, for every element
            new_name, new_value = name, nan
        else:
            try:
                near = ftype(value)
            except OverflowError:  # an int too large for any float
                near = ftype(math.copysign(math.inf, value))
            above = near if float(near) >= value else np.nextafter(near, ftype(math.inf))
            below = near if float(near) <= value else np.nextafter(near, ftype(-math.inf))
            new_name = name
            if name in {"valueeq", "valuene"}:
                # No element equals NaN: == selects nothing and != everything.
                new_value = near if float(near) == value else nan
            else:
                new_value = above if name in {"valuelt", "valuege"} else below
    else:
        if np_type.kind == "b":
            lo, hi = 0, 1
        else:
            info = np.iinfo(np_type)
            lo, hi = int(info.min), int(info.max)
        keep_all, keep_none = ("valuege", lo), ("valuelt", lo)
        if value != value:
            new_name, new_value = keep_all if name == "valuene" else keep_none
        elif name in {"valueeq", "valuene"}:
            exact = value == math.floor(value) if math.isfinite(value) else False
            if exact and lo <= value <= hi:
                new_name, new_value = name, int(value)
            else:
                new_name, new_value = keep_all if name == "valuene" else keep_none
        else:
            # Integers: x < t is x < ceil(t), x <= t is x <= floor(t), and so on.
            if math.isinf(value):
                bound = value
            elif name in {"valuelt", "valuege"}:
                bound = math.ceil(value)
            else:
                bound = math.floor(value)
            passes_all = {
                "valuelt": bound > hi,
                "valuele": bound >= hi,
                "valuegt": bound < lo,
                "valuege": bound <= lo,
            }[name]
            passes_none = {
                "valuelt": bound <= lo,
                "valuele": bound < lo,
                "valuegt": bound >= hi,
                "valuege": bound > hi,
            }[name]
            if passes_all:
                new_name, new_value = keep_all
            elif passes_none:
                new_name, new_value = keep_none
            else:
                new_name, new_value = name, int(bound)
        if np_type.kind == "b":
            new_value = bool(new_value)
    module = select if isinstance(parent, SelectOp) else indexunary
    new_op = getattr(module, new_name)[dtype]
    new_thunk = Scalar.from_value(new_value, dtype, is_cscalar=thunk._is_cscalar, name="")
    return new_op, new_thunk


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
    # The underlying object is a ``GrB_IndexUnaryOp``. Instances built by
    # ``_from_indexunary`` borrow that handle and set ``_gb_obj_owner``, which
    # suppresses the free here. Freeing from both sides is a use-after-free that
    # no test catches: the freed handle usually still reads as valid.
    _owns_gb_obj = True

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
                # Borrow the IndexUnaryOp's allocation instead of making a
                # second one. Holding ``t`` keeps that handle alive for as long
                # as this SelectOp can use it: ``iop`` is a temporary in
                # ``register_anonymous``, so without this the handle is freed
                # the moment it is collected and every call raises
                # UninitializedObject.
                op._gb_obj_owner = t
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
        _compile_udf_for_udt(
            numba_func, sig, op_kind="select", op_name=self.name, dtypes=(dtype, dtype2)
        )
        select_wrapper, wrapper_sig = _get_udt_wrapper(
            numba_func, BOOL, dtype, dtype2, include_indexes=True
        )
        return _finalize_udt_op(
            self, dtype, dtype2, BOOL, select_wrapper, wrapper_sig, TypedUserSelectOp
        )

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
            ``func`` takes four input parameters (any dtype, int64, int64,
            any dtype) and returns boolean. The first argument (any dtype) is
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
            ``func`` takes four input parameters (any dtype, int64, int64,
            any dtype) and returns boolean. The first argument (any dtype) is
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
            self._udt_types = {}  # {dtype: DataType}
            self._udt_ops = {}  # {dtype: TypedUserIndexUnaryOp}

    __call__ = TypedBuiltinSelectOp.__call__


ParameterizedSelectOp._op_class = SelectOp

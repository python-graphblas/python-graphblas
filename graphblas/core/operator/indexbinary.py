import inspect
from types import FunctionType

from ... import _STANDARD_OPERATOR_NAMES, indexbinary
from ...dtypes import BOOL, INT8, UINT64, lookup_dtype
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, ffi, lib
from ..dtypes import _sample_values
from .base import OpBase, ParameterizedUdf, TypedOpBase, _deserialize_parameterized

_has_idxbinop = hasattr(lib, "GxB_IndexBinaryOp_new")

if _has_numba:
    import numba

ffi_new = ffi.new


class _BoundIndexBinaryOp(TypedOpBase):
    """A BinaryOp created by binding a theta value to an IndexBinaryOp."""

    __slots__ = ()
    opclass = "BinaryOp"


class TypedBuiltinIndexBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexBinaryOp"

    def __call__(self, theta=None):
        """Bind a theta value to create a BinaryOp that can be used in operations.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Default is 0 (False).

        Returns
        -------
        TypedOpBase
            A BinaryOp created from this IndexBinaryOp with the given theta.

        """
        from ..scalar import Scalar

        if theta is None:
            theta = False
        if not isinstance(theta, Scalar):
            theta = Scalar.from_value(theta, is_cscalar=False, name="")  # pragma: is_grbscalar
        elif theta._is_cscalar:
            # fmt: off
            val = theta.value
            theta = Scalar.from_value(val, is_cscalar=False, name="")  # pragma: is_grbscalar
            # fmt: on
        new_binop = ffi_new("GrB_BinaryOp*")
        check_status_carg(
            lib.GxB_BinaryOp_new_IndexOp(new_binop, self.gb_obj, theta._carg),
            "BinaryOp",
            new_binop[0],
        )
        rv = _BoundIndexBinaryOp.__new__(_BoundIndexBinaryOp)
        rv.parent = self.parent
        rv.name = self.name
        rv.type = self.type
        rv.return_type = self.return_type
        rv.gb_obj = new_binop[0]
        rv.gb_name = f"{self.name}_bound"
        rv._type2 = self._type2
        return rv

    @property
    def thunk_type(self):
        return self.type if self._type2 is None else self._type2


class TypedUserIndexBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexBinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    thunk_type = TypedBuiltinIndexBinaryOp.thunk_type

    def __call__(self, theta=None):
        return TypedBuiltinIndexBinaryOp.__call__(self, theta)

    __call__.__doc__ = TypedBuiltinIndexBinaryOp.__call__.__doc__


class ParameterizedIndexBinaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        idxbinop = self.func(*args, **kwargs)
        idxbinop._parameterized_info = (self, args, kwargs)
        return IndexBinaryOp.register_anonymous(idxbinop, self.name, is_udt=self._is_udt)

    def __reduce__(self):
        name = f"indexbinary.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
        if anonymous:
            return IndexBinaryOp.register_anonymous(func, name, parameterized=True)
        if (rv := IndexBinaryOp._find(name)) is not None:
            return rv
        return IndexBinaryOp.register_new(name, func, parameterized=True)


class IndexBinaryOp(OpBase):
    """Takes two inputs with their indices, plus a thunk, and returns one output.

    The function has the signature ``f(x, ix, jx, y, iy, jy, theta) -> z``,
    where ``ix, jx`` are the row/column indices of ``x`` and ``iy, jy`` are
    the row/column indices of ``y``.

    An IndexBinaryOp can be converted to a BinaryOp by binding a theta value,
    which makes it usable in any operation that accepts a BinaryOp (eWiseMult,
    eWiseAdd, mxm, etc.).

    IndexBinaryOps are located in the ``graphblas.indexbinary`` namespace.

    There are no built-in IndexBinaryOps; all are user-defined.
    """

    __slots__ = "orig_func", "_is_udt", "_numba_func"
    _module = indexbinary
    _modname = "indexbinary"
    _custom_dtype = None
    _typed_class = TypedBuiltinIndexBinaryOp
    _typed_user_class = TypedUserIndexBinaryOp
    # No built-in IndexBinaryOps; no parse config needed
    _parse_config = {
        "trim_from_front": 4,
        "num_underscores": 1,
        "re_exprs": [],
    }

    @classmethod
    def _build(cls, name, func, *, is_udt=False, anonymous=False):
        if not _has_idxbinop:
            raise RuntimeError(
                "IndexBinaryOp requires SuiteSparse:GraphBLAS 9.4+ "
                "(python-suitesparse-graphblas 9.3.1+)"
            )
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_indexbinary>")
        success = False
        indexbinary_udf = numba.njit(func)
        new_type_obj = cls(
            name, func, anonymous=anonymous, is_udt=is_udt, numba_func=indexbinary_udf
        )
        return_types = {}
        nt = numba.types
        if not is_udt:
            for type_ in _sample_values:
                sig = (
                    type_.numba_type,
                    UINT64.numba_type,
                    UINT64.numba_type,
                    type_.numba_type,
                    UINT64.numba_type,
                    UINT64.numba_type,
                    type_.numba_type,
                )
                try:
                    indexbinary_udf.compile(sig)
                except numba.TypingError:
                    continue
                ret_type = lookup_dtype(indexbinary_udf.overloads[sig].signature.return_type)
                if ret_type != type_ and (
                    ("INT" in ret_type.name and "INT" in type_.name)
                    or ("FP" in ret_type.name and "FP" in type_.name)
                    or ("FC" in ret_type.name and "FC" in type_.name)
                    or (
                        type_ == UINT64
                        and ret_type.name == "FP64"
                        and return_types.get(lookup_dtype("INT64")) == lookup_dtype("INT64")
                    )
                ):
                    ret_type = type_
                elif type_ == BOOL and ret_type.name == "INT64" and return_types.get(INT8) == INT8:
                    ret_type = INT8

                # Numba is unable to handle BOOL correctly right now
                input_type = INT8 if type_ == BOOL else type_
                return_type = INT8 if ret_type == BOOL else ret_type

                # Build wrapper: z = f(x, ix, jx, y, iy, jy, theta)
                # C signature: void(z*, x*, ix, jx, y*, iy, jy, theta*)
                wrapper_sig = nt.void(
                    nt.CPointer(return_type.numba_type),
                    nt.CPointer(input_type.numba_type),
                    UINT64.numba_type,
                    UINT64.numba_type,
                    nt.CPointer(input_type.numba_type),
                    UINT64.numba_type,
                    UINT64.numba_type,
                    nt.CPointer(input_type.numba_type),
                )

                if type_ == BOOL:
                    if ret_type == BOOL:

                        def indexbinary_wrapper(
                            z, x, ix, jx, y, iy, jy, theta
                        ):  # pragma: no cover (numba)
                            z[0] = bool(
                                indexbinary_udf(
                                    bool(x[0]), ix, jx, bool(y[0]), iy, jy, bool(theta[0])
                                )
                            )

                    else:

                        def indexbinary_wrapper(
                            z, x, ix, jx, y, iy, jy, theta
                        ):  # pragma: no cover (numba)
                            z[0] = indexbinary_udf(
                                bool(x[0]), ix, jx, bool(y[0]), iy, jy, bool(theta[0])
                            )

                elif ret_type == BOOL:

                    def indexbinary_wrapper(
                        z, x, ix, jx, y, iy, jy, theta
                    ):  # pragma: no cover (numba)
                        z[0] = bool(indexbinary_udf(x[0], ix, jx, y[0], iy, jy, theta[0]))

                else:

                    def indexbinary_wrapper(
                        z, x, ix, jx, y, iy, jy, theta
                    ):  # pragma: no cover (numba)
                        z[0] = indexbinary_udf(x[0], ix, jx, y[0], iy, jy, theta[0])

                indexbinary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(indexbinary_wrapper)
                new_idxbinop = ffi_new("GxB_IndexBinaryOp*")
                check_status_carg(
                    lib.GxB_IndexBinaryOp_new(
                        new_idxbinop,
                        indexbinary_wrapper.cffi,
                        ret_type.gb_obj,
                        type_.gb_obj,
                        type_.gb_obj,
                        type_.gb_obj,
                        ffi_new("char[]", name.encode()),
                        ffi.NULL,
                    ),
                    "IndexBinaryOp",
                    new_idxbinop[0],
                )
                op = cls._typed_user_class(new_type_obj, name, type_, ret_type, new_idxbinop[0])
                new_type_obj._add(op)
                success = True
                return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if not _has_idxbinop:
            raise RuntimeError(
                "IndexBinaryOp requires SuiteSparse:GraphBLAS 9.4+ "
                "(python-suitesparse-graphblas 9.3.1+)"
            )
        if dtype2 is None:
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]
        if self._numba_func is None:
            raise KeyError(f"{self.name} does not work with {dtypes} types")

        numba_func = self._numba_func
        sig = (
            dtype.numba_type,
            UINT64.numba_type,
            UINT64.numba_type,
            dtype2.numba_type,
            UINT64.numba_type,
            UINT64.numba_type,
            dtype2.numba_type,
        )
        numba_func.compile(sig)
        ret_type = lookup_dtype(numba_func.overloads[sig].signature.return_type)
        indexbinary_wrapper, wrapper_sig = _get_udt_wrapper_indexbinary(
            numba_func, ret_type, dtype, dtype2
        )

        indexbinary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(indexbinary_wrapper)
        new_idxbinop = ffi_new("GxB_IndexBinaryOp*")
        check_status_carg(
            lib.GxB_IndexBinaryOp_new(
                new_idxbinop,
                indexbinary_wrapper.cffi,
                ret_type._carg,
                dtype._carg,
                dtype2._carg,
                dtype2._carg,
                ffi_new("char[]", self.name.encode()),
                ffi.NULL,
            ),
            "IndexBinaryOp",
            new_idxbinop[0],
        )
        op = TypedUserIndexBinaryOp(
            self,
            self.name,
            dtype,
            ret_type,
            new_idxbinop[0],
            dtype2=dtype2,
        )
        self._udt_types[dtypes] = ret_type
        self._udt_ops[dtypes] = op
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register an IndexBinaryOp without registering it in the ``indexbinary`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes seven input parameters--``(x, ix, jx, y, iy, jy, theta)``--
            where ``x`` and ``y`` are element values, ``ix, jx`` and ``iy, jy``
            are their row/column indices (int64), and ``theta`` is a scalar parameter.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.indexbinary.{name}``.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.

        Returns
        -------
        IndexBinaryOp or ParameterizedIndexBinaryOp

        """
        cls._check_supports_udf("register_anonymous")
        if parameterized:
            return ParameterizedIndexBinaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register a new IndexBinaryOp and save it to ``graphblas.indexbinary`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.indexbinary.{name}``.
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes seven input parameters--``(x, ix, jx, y, iy, jy, theta)``--
            where ``x`` and ``y`` are element values, ``ix, jx`` and ``iy, jy``
            are their row/column indices (int64), and ``theta`` is a scalar parameter.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.
        lazy : bool, default False
            If True, delay compilation until the operator is used.

        Examples
        --------
        >>> gb.indexbinary.register_new("index_dist", lambda x, ix, jx, y, iy, jy, t: abs(ix - iy))

        """
        cls._check_supports_udf("register_new")
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "func": func, "parameterized": parameterized},
            )
        elif parameterized:
            idxbinop = ParameterizedIndexBinaryOp(name, func, is_udt=is_udt)
            setattr(module, funcname, idxbinop)
        else:
            idxbinop = cls._build(name, func, is_udt=is_udt)
            setattr(module, funcname, idxbinop)

        if not cls._initialized:  # pragma: no cover (safety)
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return idxbinop

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        super()._initialize(include_in_ops=False)
        # No built-in IndexBinaryOps to register
        cls._initialized = True

    def __init__(self, name, func=None, *, anonymous=False, is_udt=False, numba_func=None):
        super().__init__(name, anonymous=anonymous)
        self.orig_func = func
        self._numba_func = numba_func
        self._is_udt = is_udt
        if is_udt:
            self._udt_types = {}  # {(dtype, dtype2): DataType}
            self._udt_ops = {}  # {(dtype, dtype2): TypedUserIndexBinaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"indexbinary.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    def __call__(self, theta=None, dtype=None):
        """Bind a theta value to create a BinaryOp.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Default is 0 (False).
        dtype : dtype, optional
            The dtype to use. If not provided, it will be inferred from theta.

        Returns
        -------
        TypedOpBase
            A BinaryOp created from this IndexBinaryOp with the given theta.

        """
        from ...dtypes import lookup_dtype as _lookup_dtype
        from ..scalar import Scalar

        if theta is None:
            theta = False
        if not isinstance(theta, Scalar):
            theta = Scalar.from_value(theta, is_cscalar=False, name="")  # pragma: is_grbscalar
        elif theta._is_cscalar:
            # fmt: off
            val = theta.value
            theta = Scalar.from_value(val, is_cscalar=False, name="")  # pragma: is_grbscalar
            # fmt: on
        if dtype is None:
            dtype = theta.dtype
        else:
            dtype = _lookup_dtype(dtype)
        typed_op = self[dtype]
        return typed_op(theta)


def _get_udt_wrapper_indexbinary(numba_func, return_type, dtype, dtype2):
    """Build a wrapper function for UDT IndexBinaryOp: z = f(x, ix, jx, y, iy, jy, theta)."""
    nt = numba.types
    ztype = INT8 if return_type == BOOL else return_type
    xtype = INT8 if dtype == BOOL else dtype
    ytype = INT8 if dtype2 == BOOL else dtype2

    wrapper_sig = nt.void(
        nt.CPointer(ztype.numba_type),
        nt.CPointer(xtype.numba_type),
        UINT64.numba_type,
        UINT64.numba_type,
        nt.CPointer(ytype.numba_type),
        UINT64.numba_type,
        UINT64.numba_type,
        nt.CPointer(ytype.numba_type),
    )

    d = {"numba": numba, "numba_func": numba_func}
    xderef = "bool(x_ptr[0])" if dtype == BOOL else "x_ptr[0]"
    yderef = "bool(y_ptr[0])" if dtype2 == BOOL else "y_ptr[0]"
    tderef = "bool(t_ptr[0])" if dtype2 == BOOL else "t_ptr[0]"
    call = f"numba_func({xderef}, ix, jx, {yderef}, iy, jy, {tderef})"
    if return_type == BOOL:
        call = f"bool({call})"
    text = f"def wrapper(z_ptr, x_ptr, ix, jx, y_ptr, iy, jy, t_ptr):\n    z_ptr[0] = {call}\n"
    exec(text, d)  # noqa: S102
    return d["wrapper"], wrapper_sig

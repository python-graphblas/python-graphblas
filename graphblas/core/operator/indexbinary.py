import inspect
from types import FunctionType

from ... import _STANDARD_OPERATOR_NAMES, indexbinary
from ...dtypes import BOOL, INT8, UINT64, lookup_dtype
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, ffi, lib
from ..dtypes import _sample_values
from .base import OpBase, ParameterizedUdf, TypedOpBase

_has_idxbinop = hasattr(lib, "GxB_IndexBinaryOp_new")

if _has_numba:
    import numba

    from .base import (
        _bool_to_int8,
        _compile_udf_for_udt,
        _get_udt_wrapper_indexbinary,
        _resolve_udt_return_type,
    )

ffi_new = ffi.new


def _rebind_indexbinaryop(parent, type_, theta):
    # ``theta`` was stored as a raw Python/numpy value. For UDT thunks,
    # ``TypedBuiltinIndexBinaryOp.__call__`` would route a raw value through
    # ``Scalar.from_value(theta)`` with no dtype, which then fails inside the
    # multi-dim ndarray ``Scalar.value`` setter for array UDT values. Wrap
    # as a typed Scalar here so the bind uses an explicit dtype.
    typed = parent[type_]
    thunk_type = typed.thunk_type
    if thunk_type._is_udt and not isinstance(theta, (int, float, bool, complex)):
        from ..scalar import Scalar

        theta = Scalar.from_value(theta, dtype=thunk_type, is_cscalar=False, name="")
    return typed(theta)


def _theta_to_scalar(theta, dtype):
    """Wrap a raw (non-Scalar) theta as a GrB Scalar for binding.

    ``dtype`` is the thunk dtype when known (a UDT thunk or an explicit
    ``dtype=``), else ``None`` to infer from the value. A raw user-defined-type
    value (array/tuple) can't be inferred, so raise a clear error that names the
    value and the working alternatives.
    """
    from ..scalar import Scalar

    try:
        return Scalar.from_value(theta, dtype, is_cscalar=False, name="")  # pragma: is_grbscalar
    except TypeError as exc:
        if dtype is None:
            raise TypeError(
                f"Cannot infer a dtype for theta from {theta!r}; pass an explicit dtype "
                "(e.g., op(value, dtype=...)) or a typed Scalar. Required for user-defined types."
            ) from exc
        raise


class _BoundIndexBinaryOp(TypedOpBase):
    """A BinaryOp produced by binding a theta value to an IndexBinaryOp.

    Under the hood this is a single typed ``GrB_BinaryOp`` (constructed via
    ``GxB_BinaryOp_new_IndexOp``), so SuiteSparse accepts it anywhere a
    typed ``GrB_BinaryOp`` is expected: ``ewise_mult`` / ``ewise_add`` and,
    via :func:`Semiring.register_anonymous`, as the multiplier of a
    Semiring that then works in ``mxm`` / ``mxv`` / ``vxm``.
    """

    __slots__ = ("_theta",)
    opclass = "BinaryOp"
    # Each ``iop(theta)`` call allocates a fresh ``GrB_BinaryOp`` via
    # ``GxB_BinaryOp_new_IndexOp`` and bound IBOs are never cached, so this
    # is the per-call allocation that ``TypedOpBase.__del__`` must release.
    # Without this, every ``iop(theta)`` (including each pickle round-trip)
    # leaks one ``GrB_BinaryOp`` for the life of the process.
    _owns_gb_obj = True
    # A bound IBO is monomorphic in its operand types (one ``(x, y, z,
    # theta)`` combination). Expose the polymorphism / commutativity
    # markers that ``Semiring.__init__``, ``Semiring.is_positional``,
    # ``Semiring._custom_dtype``, and ``TypedUserSemiring`` probe as
    # class-level defaults, so they get sane answers (and no
    # ``AttributeError``) when a bound IBO is passed in place of a
    # ``BinaryOp`` parent.
    _is_udt = False
    _udt_types = None
    _udt_ops = None
    _semiring_commutes_to = None
    commutes_to = None
    is_commutative = False
    # ``is_positional`` and ``_custom_dtype`` are BinaryOp-specific dispatch
    # hints: when a polymorphic BinaryOp is queried for a type it doesn't
    # support, the resolver in ``utils.get_typed_op`` falls back through
    # them. A bound IBO is already type-monomorphic (one ``GrB_BinaryOp``
    # for one (x, y, z, theta) combination), so neither fallback applies.
    is_positional = False
    _custom_dtype = None

    def __reduce__(self):
        return (_rebind_indexbinaryop, (self.parent, self.type, self._theta))


class TypedBuiltinIndexBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexBinaryOp"

    def __call__(self, theta=None):
        """Bind a theta value, returning a BinaryOp.

        The result is usable directly in ``ewise_mult`` and ``ewise_add``,
        or as the multiplier of a Semiring (see
        :meth:`Semiring.register_anonymous`) for ``mxm`` / ``mxv`` / ``vxm``.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Defaults to 0 (False).

        Returns
        -------
        TypedOpBase
            A BinaryOp built from this IndexBinaryOp with the given theta.

        """
        from ..scalar import Scalar

        if theta is None:
            theta = False
        theta_value = theta.value if isinstance(theta, Scalar) else theta
        if not isinstance(theta, Scalar):
            # A raw value for a UDT thunk carries the thunk dtype; otherwise the
            # value is inferred (and a raw UDT value raises a clear error).
            tt = self.thunk_type
            theta = _theta_to_scalar(theta, tt if tt._is_udt else None)
        elif theta._is_cscalar:
            # fmt: off
            # Pass dtype explicitly; an array-UDT value would otherwise infer wrong.
            val, dt = theta.value, theta.dtype
            theta = Scalar.from_value(val, dt, is_cscalar=False, name="")  # pragma: is_grbscalar
            # fmt: on
        new_binop = ffi_new("GrB_BinaryOp*")
        check_status_carg(
            lib.GxB_BinaryOp_new_IndexOp(new_binop, self.gb_obj, theta._carg),
            "BinaryOp",
            new_binop[0],
        )
        rv = _BoundIndexBinaryOp(
            self.parent,
            self.name,
            self.type,
            self.return_type,
            new_binop[0],
            f"{self.name}_bound",
            dtype2=self._type2,
        )
        rv._theta = theta_value
        return rv

    @property
    def thunk_type(self):
        return self.type if self._type2 is None else self._type2


class TypedUserIndexBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexBinaryOp"
    _owns_gb_obj = True

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


class IndexBinaryOp(OpBase):
    """Takes two inputs with their indices plus a thunk, and returns one output.

    The function has the signature ``f(x, ix, jx, y, iy, jy, theta) -> z``,
    where ``ix, jx`` are the row and column indices of ``x``, and ``iy, jy``
    are the row and column indices of ``y``.

    Binding a theta value (``ibo[dtype](theta)``) produces a BinaryOp usable
    directly in ``ewise_mult`` and ``ewise_add``. To use it as a multiplier in
    ``mxm`` / ``mxv`` / ``vxm``, wrap it in a Semiring via
    :meth:`Semiring.register_anonymous`; per SuiteSparse, the additive monoid
    itself cannot be IndexBinaryOp-based.

    IndexBinaryOps live in the ``graphblas.indexbinary`` namespace. There are
    no built-ins; all IndexBinaryOps are user-defined.
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

                input_type = _bool_to_int8(type_)
                return_type = _bool_to_int8(ret_type)

                # Build a wrapper that calls z = f(x, ix, jx, y, iy, jy, theta).
                # C signature: void(z*, x*, ix, jx, y*, iy, jy, theta*).
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
            # KeyError (not RuntimeError) so ``udt in op`` and the resolver
            # chain in ``__contains__`` / ``[dtype]`` lookups return cleanly.
            raise KeyError(
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
        _compile_udf_for_udt(
            numba_func, sig, op_kind="indexbinary", op_name=self.name, dtypes=(dtype, dtype2)
        )
        numba_ret_type = numba_func.overloads[sig].signature.return_type
        ret_type = _resolve_udt_return_type(numba_ret_type, dtype, dtype2)
        indexbinary_wrapper, wrapper_sig = _get_udt_wrapper_indexbinary(
            numba_func, ret_type, dtype, dtype2, numba_ret_type=numba_ret_type
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
        """Register an IndexBinaryOp without adding it to the ``indexbinary`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be
            compilable with ``numba.njit``. The function takes seven inputs
            ``(x, ix, jx, y, iy, jy, theta)``: ``x`` and ``y`` are element
            values, ``ix, jx`` and ``iy, jy`` are their row and column indices
            (int64), and ``theta`` is a scalar parameter.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.indexbinary.{name}``.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, so that
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
        """Register a new IndexBinaryOp under the ``graphblas.indexbinary`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. Available afterwards as ``gb.indexbinary.{name}``.
        func : FunctionType
            The function to compile. For all current backends, this must be
            compilable with ``numba.njit``. The function takes seven inputs
            ``(x, ix, jx, y, iy, jy, theta)``: ``x`` and ``y`` are element
            values, ``ix, jx`` and ``iy, jy`` are their row and column indices
            (int64), and ``theta`` is a scalar parameter.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.
        lazy : bool, default False
            When True, defer compilation until the operator is first used.

        Examples
        --------
        >>> gb.indexbinary.register_new("index_dist", lambda x, ix, jx, y, iy, jy, t: abs(ix - iy))

        """
        cls._check_supports_udf("register_new")
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "func": func, "parameterized": parameterized, "is_udt": is_udt},
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
        # No built-in IndexBinaryOps to register.
        cls._initialized = True

    def __init__(self, name, func=None, *, anonymous=False, is_udt=False, numba_func=None):
        super().__init__(name, anonymous=anonymous)
        self.orig_func = func
        self._numba_func = numba_func
        self._is_udt = is_udt
        if is_udt:
            self._udt_types = {}  # {(dtype, dtype2): DataType}
            self._udt_ops = {}  # {(dtype, dtype2): TypedUserIndexBinaryOp}

    def __call__(self, theta=None, dtype=None):
        """Bind a theta value to produce a BinaryOp.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Defaults to 0 (False).
        dtype : dtype, optional
            The dtype to use. Inferred from ``theta`` when not provided.

        Returns
        -------
        TypedOpBase
            A BinaryOp built from this IndexBinaryOp with the given theta.

        """
        from ...dtypes import lookup_dtype as _lookup_dtype
        from ..scalar import Scalar

        if theta is None:
            theta = False
        if not isinstance(theta, Scalar):
            # An explicit dtype supplies a UDT; with no dtype the value is
            # inferred (and a raw UDT value raises a clear error).
            theta = _theta_to_scalar(theta, _lookup_dtype(dtype) if dtype is not None else None)
        elif theta._is_cscalar:
            # fmt: off
            # Pass dtype explicitly; an array-UDT value would otherwise infer wrong.
            val, dt = theta.value, theta.dtype
            theta = Scalar.from_value(val, dt, is_cscalar=False, name="")  # pragma: is_grbscalar
            # fmt: on
        if dtype is None:
            dtype = theta.dtype
        else:
            dtype = _lookup_dtype(dtype)
        typed_op = self[dtype]
        return typed_op(theta)


ParameterizedIndexBinaryOp._op_class = IndexBinaryOp

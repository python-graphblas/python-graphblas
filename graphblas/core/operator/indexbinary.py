import inspect
from operator import getitem
from types import FunctionType

import numpy as np

from ... import _STANDARD_OPERATOR_NAMES, indexbinary
from ...dtypes import BOOL, INT8, UINT64, lookup_dtype
from ...exceptions import EmptyObject, UdfParseError, check_status_carg
from .. import _has_numba, ffi, lib
from ..dtypes import _sample_values, unify
from .base import OpBase, ParameterizedUdf, TypedOpBase
from .binary import BinaryOp, TypedBuiltinBinaryOp

_has_idxbinop = hasattr(lib, "GxB_IndexBinaryOp_new")

if _has_numba:
    import numba

    from .udf import (
        _bool_to_int8,
        _compile_udf_for_udt,
        _get_udt_wrapper_indexbinary,
        _resolve_udt_return_type,
    )

ffi_new = ffi.new


def _rebind_indexbinaryop(parent, type_, theta):
    # ``type_`` is the key from ``_ibo_types_key``, and ``theta`` the value that was
    # bound. A value for a UDT theta becomes a Scalar of that UDT here, but ``__call__``
    # does the same, so this is redundant.
    typed = parent[type_]
    thunk_type = typed.thunk_type
    if thunk_type._is_udt and not isinstance(theta, (int, float, bool, complex)):
        from ..scalar import Scalar

        theta = Scalar.from_value(theta, dtype=thunk_type, is_cscalar=False, name="")
    return typed(theta)


def _theta_to_scalar(theta, dtype):
    """Return ``theta``, which is not a Scalar, as a GraphBLAS Scalar to bind.

    ``dtype`` is theta's type when it is known (a UDT ``thunk_type``, or ``dtype=``). With
    None, theta has the type of its value: a Python number its default type, a numpy
    scalar or ``np.void`` its dtype, and a numpy array its array UDT (``INT32[3]`` for
    an int32 array of shape ``(3,)``). A tuple, list or dict has no type of its own, and
    raises TypeError.
    """
    from ..scalar import Scalar

    if dtype is None and isinstance(theta, np.ndarray) and theta.ndim > 0:
        try:
            dtype = lookup_dtype(np.dtype((theta.dtype, theta.shape)))
        except (TypeError, ValueError):
            pass  # not a valid UDT (a bool array, for one); ``from_value`` raises
    try:
        return Scalar.from_value(theta, dtype, is_cscalar=False, name="")  # pragma: is_grbscalar
    except TypeError as exc:
        if dtype is None:
            raise TypeError(
                f"Cannot infer a dtype for theta from {theta!r}; give theta as a Scalar of "
                "its type (Scalar.from_value(theta, dtype)), or name the op's type "
                "(op[dtype](theta), or op(theta, dtype=...))."
            ) from exc
        raise


def _refuse_empty_theta(theta):
    # SuiteSparse binds theta without checking that it has a value: binding a new empty
    # Scalar crashes, and binding a copy of one uses whatever its value array contains.
    # Raise EmptyObject, as ``apply`` does for an empty IndexUnaryOp thunk.
    if theta._is_empty:
        raise EmptyObject("theta is an empty Scalar; bind a Scalar that holds a value")


class _BoundIndexBinaryOp(TypedOpBase):
    """A BinaryOp made by binding theta to a typed IndexBinaryOp.

    ``ibo[dtype](theta)`` returns one, and ``ibo(theta)`` makes one for each pair of
    types it runs at (:class:`_UntypedBoundIndexBinaryOp`). It is one typed
    ``GrB_BinaryOp`` (from ``GxB_BinaryOp_new_IndexOp``). SuiteSparse accepts it in
    ``ewise_mult``, ``ewise_add``, ``ewise_union`` and ``kronecker``, and as the
    multiplier of a Semiring, but not in ``apply`` or as an accumulator.
    """

    __slots__ = "_theta", "_theta_type"
    opclass = "BinaryOp"
    # Each bind allocates a new GrB_BinaryOp, and bound ops are not cached, so
    # ``__del__`` must free it
    _owns_gb_obj = True
    # In its ``__init__``, a Semiring made from this op reads these BinaryOp attributes:
    # ``is_positional``, ``_custom_dtype``, ``_is_udt`` and ``commutes_to``. A bound op
    # has one set of types, and is not positional or commutative.
    _is_udt = False
    _udt_types = None
    _udt_ops = None
    _semiring_commutes_to = None
    commutes_to = None
    is_commutative = False
    monoid = None  # so ``reduce`` refuses it, as it refuses a BinaryOp without a Monoid
    is_positional = False
    _custom_dtype = None

    # ``ewise_union`` reads ``type2`` for the type of its right default
    type2 = TypedBuiltinBinaryOp.type2

    def __reduce__(self):
        return (
            _rebind_indexbinaryop,
            (self.parent, _ibo_types_key(self), self._theta),
        )


class _UntypedBoundIndexBinaryOp(BinaryOp):
    """A BinaryOp made by binding theta to an IndexBinaryOp, typed each time it runs.

    ``ibo(theta)`` returns one. For each pair of input types it runs the typed op
    ``ibo[...](theta)``, built once (``IndexBinaryOp._get_typed``). The typing rules are
    in docs/technical_notes/udfs.rst ("IndexBinaryOp types and theta").
    """

    __slots__ = "_ibo", "_theta"
    # For ``find_opclass``, where the class name would not say BinaryOp.
    opclass = "BinaryOp"

    def __init__(self, ibo, theta):
        # ``is_udt=True`` sends every lookup (``get_typed_op``, a Semiring's) to
        # ``_compile_udt`` below, UDT or not.
        super().__init__(ibo.name, anonymous=True, is_udt=True)
        self._ibo = ibo
        self._theta = theta

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:  # pragma: no cover (safety)
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_ops:
            return self._udt_ops[dtypes]
        op = self._ibo._get_typed(dtype, dtype2, self._theta.dtype)(self._theta)
        self._udt_types[dtypes] = op.return_type
        self._udt_ops[dtypes] = op
        return op

    def __getitem__(self, type_):
        # A type named here is the op's type, as in ``ibo[type_](theta)``.
        return self._ibo[type_](self._theta)

    def __reduce__(self):
        return (self._ibo, (self._theta,))

    def __repr__(self):
        return f"{self._ibo!r}({self._theta.value!r})"


def _ibo_types_key(op):
    """Return ``op``'s key in its parent: its type, or the (x, y, theta) types if they differ."""
    dtype2 = op.type if op._type2 is None else op._type2
    if op.type == dtype2 == op._theta_type:
        return op.type
    return (op.type, dtype2, op._theta_type)


class _TypedIndexBinaryOpMixin:
    """Theta type, repr and pickle for a typed IndexBinaryOp.

    Its theta type can differ from its input types.
    """

    __slots__ = ()

    @property
    def thunk_type(self):
        return self._theta_type

    def __repr__(self):
        dtype2 = self.type if self._type2 is None else self._type2
        if self._theta_type == dtype2:
            return super().__repr__()
        return f"indexbinary.{self.name}[{self.type.name}, {dtype2.name}, {self._theta_type.name}]"

    def __reduce__(self):
        return (getitem, (self.parent, _ibo_types_key(self)))


class TypedBuiltinIndexBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexBinaryOp"

    def __call__(self, theta=None):
        """Bind a theta value, returning a BinaryOp.

        The BinaryOp works in ``ewise_mult``, ``ewise_add``, ``ewise_union`` and
        ``kronecker``. For ``mxm``, ``mxv`` or ``vxm``, make it the multiplier of a
        Semiring with ``Semiring.register_anonymous``.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Defaults to 0 (False). Its value is bound
            here, so changing a Scalar theta afterwards does not change the BinaryOp.

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
            # A value for a UDT theta becomes an element of that UDT; any other value
            # has its own type
            tt = self.thunk_type
            theta = _theta_to_scalar(theta, tt if tt._is_udt else None)
        elif theta._is_cscalar:
            # fmt: off
            # Pass the dtype: ``from_value`` cannot infer it from an array UDT's value
            val, dt = theta.value, theta.dtype
            theta = Scalar.from_value(val, dt, is_cscalar=False, name="")  # pragma: is_grbscalar
            # fmt: on
        _refuse_empty_theta(theta)
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
        rv._theta_type = self.thunk_type
        return rv


class TypedUserIndexBinaryOp(_TypedIndexBinaryOpMixin, TypedOpBase):
    __slots__ = ("_theta_type",)
    opclass = "IndexBinaryOp"
    _owns_gb_obj = True

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None, theta_type=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)
        if theta_type is None:
            theta_type = type_ if dtype2 is None else dtype2
        self._theta_type = theta_type

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

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
    """Takes two inputs with their indices and a scalar theta, and returns one output.

    The function has the signature ``f(x, ix, jx, y, iy, jy, theta) -> z``,
    where ``ix, jx`` are the row and column indices of ``x``, and ``iy, jy``
    are the row and column indices of ``y``.

    Binding theta, ``ibo(theta)``, makes a BinaryOp for ``ewise_mult``, ``ewise_add``,
    ``ewise_union`` and ``kronecker``. The BinaryOp is typed when it runs, from both
    inputs and theta; ``ibo[dtype](theta)`` binds at one type. With ``is_udt=True``,
    x, y and theta each keep their own type, and ``ibo[X, Y, T](theta)`` names all
    three. To use it in ``mxm``, ``mxv`` or ``vxm``, make it the multiplier of a
    Semiring with :meth:`Semiring.register_anonymous`. A Monoid cannot be made from
    an IndexBinaryOp. See :ref:`udf_indexbinary`.

    IndexBinaryOps live in the ``graphblas.indexbinary`` namespace. There are
    no built-ins; all IndexBinaryOps are user-defined. They need
    SuiteSparse:GraphBLAS 9.4 or later.
    """

    __slots__ = "orig_func", "_is_udt", "_numba_func"
    _module = indexbinary
    _modname = "indexbinary"
    _custom_dtype = None
    # Read by ``OpBase.__getitem__`` for a type the op does not have; no
    # IndexBinaryOp is positional.
    is_positional = False
    # Read (as ``op.parent.monoid``) by the TypeError ``reduce`` raises for a bound op
    monoid = None
    _typed_class = TypedBuiltinIndexBinaryOp
    _typed_user_class = TypedUserIndexBinaryOp
    # No built-in IndexBinaryOps, so there is nothing to parse
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
        # Set on the Dispatcher too, not only on the cfunc wrapper; see the note in
        # ``BinaryOp._build``.
        indexbinary_udf = numba.njit(func, error_model="numpy")
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

                indexbinary_wrapper = numba.cfunc(wrapper_sig, nopython=True, error_model="numpy")(
                    indexbinary_wrapper
                )
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

    def _compile_udt(self, dtype, dtype2, theta_type=None):
        if not _has_idxbinop:
            # KeyError, not RuntimeError, so ``dtype in op`` returns False
            raise KeyError(
                "IndexBinaryOp requires SuiteSparse:GraphBLAS 9.4+ "
                "(python-suitesparse-graphblas 9.3.1+)"
            )
        if dtype2 is None:
            dtype2 = dtype
        if theta_type is None:
            # With ``ibo[X, Y]`` or ``ibo[T]``, theta has y's type and is cast to it
            theta_type = dtype2
        dtypes = (dtype, dtype2, theta_type)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]
        from ..udt import _refuse_half

        _refuse_half(self, *dtypes)
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
            theta_type.numba_type,
        )
        _compile_udf_for_udt(
            numba_func, sig, op_kind="indexbinary", op_name=self.name, dtypes=dtypes
        )
        numba_ret_type = numba_func.overloads[sig].signature.return_type
        ret_type = _resolve_udt_return_type(numba_ret_type, *dtypes)
        indexbinary_wrapper, wrapper_sig = _get_udt_wrapper_indexbinary(
            numba_func, ret_type, dtype, dtype2, theta_type, numba_ret_type=numba_ret_type
        )

        indexbinary_wrapper = numba.cfunc(wrapper_sig, nopython=True, error_model="numpy")(
            indexbinary_wrapper
        )
        new_idxbinop = ffi_new("GxB_IndexBinaryOp*")
        check_status_carg(
            lib.GxB_IndexBinaryOp_new(
                new_idxbinop,
                indexbinary_wrapper.cffi,
                ret_type._carg,
                dtype._carg,
                dtype2._carg,
                theta_type._carg,
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
            theta_type=theta_type,
        )
        self._udt_types[dtypes] = ret_type
        self._udt_ops[dtypes] = op
        return op

    def _get_typed(self, dtype, dtype2, theta_type):
        """Return the typed op that ``ibo(theta)`` runs for x, y and theta of these types.

        With ``is_udt``, it is compiled for the three types as they are. Otherwise x,
        y and theta share the one type ``unify`` gives for all three.
        """
        if self._is_udt:
            return self._compile_udt(dtype, dtype2, theta_type)
        if dtype._is_udt or dtype2._is_udt or theta_type._is_udt:
            raise KeyError(f"{self.name} does not work with ({dtype}, {dtype2}, {theta_type})")
        return self[unify(unify(dtype, dtype2), theta_type)]

    def __getitem__(self, type_):
        if type(type_) is tuple and len(type_) == 3:
            return self._get_typed(*(lookup_dtype(dtype) for dtype in type_))
        return super().__getitem__(type_)

    def _add(self, op, *, is_jit=False):
        if not is_jit:
            return super()._add(op)
        # A JIT op is registered for its x, y and theta types
        dtypes = (op.type, op._type2, op.thunk_type)
        self.types[dtypes] = op.return_type
        self._udt_types[dtypes] = op.return_type
        self._udt_ops[dtypes] = op

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
            (uint64), and ``theta`` is a scalar parameter. A difference of two indices
            wraps when it would be negative; see :ref:`unsigned indices <udf_indices>`.
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
            (uint64), and ``theta`` is a scalar parameter. A difference of two indices
            wraps when it would be negative; see :ref:`unsigned indices <udf_indices>`.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator.
        is_udt : bool, default False
            Whether the operator is intended to operate on user-defined types.
        lazy : bool, default False
            When True, register and compile the operator when its name is first accessed.

        Examples
        --------
        >>> def index_dist(x, ix, jx, y, iy, jy, theta):
        ...     return max(ix, iy) - min(ix, iy)
        >>> gb.indexbinary.register_new("index_dist", index_dist)

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
            self._udt_types = {}  # {(dtype, dtype2, theta_type): DataType}
            self._udt_ops = {}  # {(dtype, dtype2, theta_type): TypedUserIndexBinaryOp}

    def __call__(self, theta=None, dtype=None):
        """Bind a theta value to produce a BinaryOp.

        Parameters
        ----------
        theta : scalar, optional
            The theta parameter to bind. Defaults to 0 (False). Its value is bound
            here, so changing a Scalar theta afterwards does not change the BinaryOp.
        dtype : dtype, optional
            The type of the BinaryOp, as in ``op[dtype](theta)``. Without it, the
            BinaryOp is typed each time it runs, from both inputs and theta, and theta
            has the type of its value (a Python int is INT64). The Technical Notes give
            the rules, in "IndexBinaryOp types and theta".

        Returns
        -------
        BinaryOp
            A BinaryOp built from this IndexBinaryOp with the given theta.

        """
        from ...dtypes import lookup_dtype as _lookup_dtype
        from ..scalar import Scalar

        if theta is None:
            theta = False
        if dtype is None:
            if isinstance(theta, Scalar):
                _refuse_empty_theta(theta)
                # Bound when it runs, so keep the value theta has now.
                theta = theta.dup(name="")
            else:
                theta = _theta_to_scalar(theta, None)
            if not self._is_udt and theta.dtype._is_udt:
                raise KeyError(f"{self.name} does not work with {theta.dtype}")
            return _UntypedBoundIndexBinaryOp(self, theta)
        dtype = _lookup_dtype(dtype)
        if not isinstance(theta, Scalar):
            # theta becomes a value of ``dtype``, a UDT included
            theta = _theta_to_scalar(theta, dtype)
        return self[dtype](theta)


ParameterizedIndexBinaryOp._op_class = IndexBinaryOp

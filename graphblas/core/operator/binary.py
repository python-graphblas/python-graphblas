import inspect
import re
from functools import lru_cache, reduce
from operator import mul
from types import FunctionType

import numpy as np

from ... import _STANDARD_OPERATOR_NAMES, backend, binary, monoid, op
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
    _supports_complex,
    lookup_dtype,
)
from ...exceptions import UdfParseError, check_status_carg
from .. import _has_numba, _supports_udfs, ffi, lib
from ..dtypes import _sample_values
from ..expr import InfixExprBase
from .base import (
    _SS_OPERATORS,
    OpBase,
    ParameterizedUdf,
    TypedOpBase,
    _call_op,
    _deserialize_parameterized,
    _hasop,
)

if _has_numba:
    import numba

    from .base import _get_udt_wrapper
if _supports_complex:
    from ...dtypes import FC32, FC64

ffi_new = ffi.new

if _has_numba:
    _udt_mask_cache = {}

    def _udt_mask(dtype):
        """Create mask to determine which bytes of UDTs to use for equality check."""
        if dtype in _udt_mask_cache:
            return _udt_mask_cache[dtype]
        if dtype.subdtype is not None:
            mask = _udt_mask(dtype.subdtype[0])
            N = reduce(mul, dtype.subdtype[1])
            rv = np.concatenate([mask] * N)
        elif dtype.names is not None:
            prev_offset = mask = None
            masks = []
            for name in dtype.names:
                dtype2, offset = dtype.fields[name]
                if mask is not None:
                    masks.append(np.pad(mask, (0, offset - prev_offset - mask.size)))
                mask = _udt_mask(dtype2)
                prev_offset = offset
            masks.append(np.pad(mask, (0, dtype.itemsize - prev_offset - mask.size)))
            rv = np.concatenate(masks)
        else:
            rv = np.ones(dtype.itemsize, dtype=bool)
        # assert rv.size == dtype.itemsize
        _udt_mask_cache[dtype] = rv
        return rv


class TypedBuiltinBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "BinaryOp"

    def __call__(self, left, right=None, *, left_default=None, right_default=None):
        if left_default is not None or right_default is not None:
            if (
                left_default is None
                or right_default is None
                or right is not None
                or not isinstance(left, InfixExprBase)
                or left.method_name != "ewise_add"
            ):
                raise TypeError(
                    "Specifying `left_default` or `right_default` keyword arguments implies "
                    "performing `ewise_union` operation with infix notation.\n"
                    "There is only one valid way to do this:\n\n"
                    f">>> {self}(x | y, left_default=0, right_default=0)\n\nwhere x and y "
                    "are Vectors or Matrices, and left_default and right_default are scalars."
                )
            return left.left._ewise_union(
                left.right, self, left_default, right_default, is_infix=True
            )
        return _call_op(self, left, right)

    @property
    def monoid(self):
        rv = getattr(monoid, self.name, None)
        if rv is not None and self.type in rv._typed_ops:
            return rv[self.type]

    @property
    def commutes_to(self):
        commutes_to = self.parent.commutes_to
        if commutes_to is not None and (self.type in commutes_to._typed_ops or self.type._is_udt):
            return commutes_to[self.type]

    @property
    def _semiring_commutes_to(self):
        commutes_to = self.parent._semiring_commutes_to
        if commutes_to is not None and (self.type in commutes_to._typed_ops or self.type._is_udt):
            return commutes_to[self.type]

    @property
    def is_commutative(self):
        return self.commutes_to is self

    @property
    def type2(self):
        return self.type if self._type2 is None else self._type2


class TypedUserBinaryOp(TypedOpBase):
    __slots__ = "_monoid"
    opclass = "BinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)
        self._monoid = None

    @property
    def monoid(self):
        if self._monoid is None:
            monoid = self.parent.monoid
            if monoid is not None and self.type in monoid:
                self._monoid = monoid[self.type]
        return self._monoid

    @property
    def orig_func(self):
        return self.parent.orig_func

    @property
    def _numba_func(self):
        return self.parent._numba_func

    commutes_to = TypedBuiltinBinaryOp.commutes_to
    _semiring_commutes_to = TypedBuiltinBinaryOp._semiring_commutes_to
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    type2 = TypedBuiltinBinaryOp.type2
    __call__ = TypedBuiltinBinaryOp.__call__


class ParameterizedBinaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_monoid", "_cached_call", "_commutes_to", "_is_udt"

    def __init__(self, name, func, *, anonymous=False, is_udt=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._monoid = None
        self._is_udt = is_udt
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)
        method = self._call_to_cache.__get__(self, type(self))
        self._cached_call = lru_cache(maxsize=1024)(method)
        self.__call__ = self._call
        self._commutes_to = None

    def _call_to_cache(self, *args, **kwargs):
        binary = self.func(*args, **kwargs)
        binary._parameterized_info = (self, args, kwargs)
        return BinaryOp.register_anonymous(binary, self.name, is_udt=self._is_udt)

    def _call(self, *args, **kwargs):
        binop = self._cached_call(*args, **kwargs)
        if self._monoid is not None and binop._monoid is None:
            # This is all a bit funky.  We try our best to associate a binaryop
            # to a monoid.  So, if we made a ParameterizedMonoid using this object,
            # then try to create a monoid with the given arguments.
            binop._monoid = binop  # temporary!
            try:
                # If this call is successful, then it will set `binop._monoid`
                self._monoid(*args, **kwargs)  # pylint: disable=not-callable
            except Exception:
                binop._monoid = None
            # assert binop._monoid is not binop
        if self.is_commutative:
            binop._commutes_to = binop
        # Don't bother yet with creating `binop.commutes_to` (but we could!)
        return binop

    @property
    def monoid(self):
        return self._monoid

    @property
    def commutes_to(self):
        if isinstance(self._commutes_to, str):
            self._commutes_to = BinaryOp._find(self._commutes_to)
        return self._commutes_to

    is_commutative = TypedBuiltinBinaryOp.is_commutative

    def __reduce__(self):
        name = f"binary.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.func, self._anonymous))

    @staticmethod
    def _deserialize(name, func, anonymous):
        if anonymous:
            return BinaryOp.register_anonymous(func, name, parameterized=True)
        if (rv := BinaryOp._find(name)) is not None:
            return rv
        return BinaryOp.register_new(name, func, parameterized=True)


def _floordiv(x, y):
    return x // y  # pragma: no cover (numba)


def _rfloordiv(x, y):
    return y // x  # pragma: no cover (numba)


def _absfirst(x, y):
    return np.abs(x)  # pragma: no cover (numba)


def _abssecond(x, y):
    return np.abs(y)  # pragma: no cover (numba)


def _rpow(x, y):
    return y**x  # pragma: no cover (numba)


def _isclose(rel_tol=1e-7, abs_tol=0.0):
    def inner(x, y):  # pragma: no cover (numba)
        return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)

    return inner


_MAX_INT64 = np.iinfo(np.int64).max


def _binom(N, k):  # pragma: no cover (numba)
    # Returns 0 if overflow or out-of-bounds
    if k > N or k < 0:
        return 0
    val = np.int64(1)
    for i in range(min(k, N - k)):
        if val > _MAX_INT64 // (N - i):  # Overflow
            return 0
        val *= N - i
        val //= i + 1
    return val


# Kinda complicated, but works for now
def _register_binom():
    # "Fake" UDT so we only compile once for INT64
    op = BinaryOp.register_new("binom", _binom, is_udt=True)
    typed_op = op[INT64, INT64]
    # Make this look like a normal operator
    for dtype in [UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64]:
        op.types[dtype] = INT64
        op._typed_ops[dtype] = typed_op
        if dtype != INT64:
            op.coercions[dtype] = typed_op
    # And make it not look like it operates on UDTs
    typed_op._type2 = None
    op._is_udt = False
    op._udt_types = None
    op._udt_ops = None
    return op


def _first(x, y):
    return x  # pragma: no cover (numba)


def _second(x, y):
    return y  # pragma: no cover (numba)


def _pair(x, y):
    return 1  # pragma: no cover (numba)


def _first_dtype(op, dtype, dtype2):
    if dtype._is_udt or dtype2._is_udt:
        return op._compile_udt(dtype, dtype2)


def _second_dtype(op, dtype, dtype2):
    if dtype._is_udt or dtype2._is_udt:
        return op._compile_udt(dtype, dtype2)


def _pair_dtype(op, dtype, dtype2):
    return op[INT64]


class BinaryOp(OpBase):
    """Takes two inputs and returns one output, possibly of a different data type.

    Built-in and registered BinaryOps are located in the ``graphblas.binary`` namespace
    as well as in the ``graphblas.ops`` combined namespace.
    """

    __slots__ = (
        "_monoid",
        "_commutes_to",
        "_semiring_commutes_to",
        "orig_func",
        "is_positional",
        "_is_udt",
        "_numba_func",
        "_custom_dtype",
    )
    _module = binary
    _modname = "binary"
    _typed_class = TypedBuiltinBinaryOp
    _parse_config = {
        "trim_from_front": 4,
        "num_underscores": 1,
        "re_exprs": [
            re.compile(
                "^GrB_(FIRST|SECOND|PLUS|MINUS|TIMES|DIV|MIN|MAX)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile(
                "GrB_(BOR|BAND|BXOR|BXNOR)_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
            re.compile(
                "^GxB_(POW|RMINUS|RDIV|PAIR|ANY|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile("^GxB_(FIRST|SECOND|PLUS|MINUS|TIMES|DIV)_(FC32|FC64)$"),
            re.compile("^GxB_(ATAN2|HYPOT|FMOD|REMAINDER|LDEXP|COPYSIGN)_(FP32|FP64)$"),
            re.compile(
                "GxB_(BGET|BSET|BCLR|BSHIFT|FIRSTI1|FIRSTI|FIRSTJ1|FIRSTJ"
                "|SECONDI1|SECONDI|SECONDJ1|SECONDJ)"
                "_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
            # These are coerced to 0 or 1, but don't return BOOL
            re.compile(
                "^GxB_(LOR|LAND|LXOR|LXNOR)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)$"),
            re.compile(
                "^GrB_(EQ|NE|GT|LT|GE|LE)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile("^GxB_(EQ|NE)_(FC32|FC64)$"),
        ],
        "re_exprs_return_complex": [re.compile("^GxB_(CMPLX)_(FP32|FP64)$")],
    }
    _commutes = {
        # builtins
        "cdiv": "rdiv",
        "first": "second",
        "ge": "le",
        "gt": "lt",
        "isge": "isle",
        "isgt": "islt",
        "minus": "rminus",
        "pow": "rpow",
        # special
        "firsti": "secondi",
        "firsti1": "secondi1",
        "firstj": "secondj",
        "firstj1": "secondj1",
        # custom
        # "absfirst": "abssecond",  # handled in graphblas.binary
        # "floordiv": "rfloordiv",
        "truediv": "rtruediv",
    }
    _commutes_to_in_semiring = {
        "firsti": "secondj",
        "firsti1": "secondj1",
        "firstj": "secondi",
        "firstj1": "secondi1",
    }
    _commutative = {
        # monoids
        "any",
        "band",
        "bor",
        "bxnor",
        "bxor",
        "eq",
        "land",
        "lor",
        "lxnor",
        "lxor",
        "max",
        "min",
        "plus",
        "times",
        # other
        "hypot",
        "isclose",
        "iseq",
        "isne",
        "ne",
        "pair",
    }
    # Don't commute: atan2, bclr, bget, bset, bshift, cmplx, copysign, fmod, ldexp, remainder
    _positional = {
        "firsti",
        "firsti1",
        "firstj",
        "firstj1",
        "secondi",
        "secondi1",
        "secondj",
        "secondj1",
    }

    @classmethod
    def _build(cls, name, func, *, is_udt=False, anonymous=False):
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_binary>")
        success = False
        binary_udf = numba.njit(func)
        new_type_obj = cls(name, func, anonymous=anonymous, is_udt=is_udt, numba_func=binary_udf)
        return_types = {}
        nt = numba.types
        if not is_udt:
            for type_ in _sample_values:
                sig = (type_.numba_type, type_.numba_type)
                try:
                    binary_udf.compile(sig)
                except numba.TypingError:
                    continue
                ret_type = lookup_dtype(binary_udf.overloads[sig].signature.return_type)
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
                    nt.CPointer(input_type.numba_type),
                )

                if type_ == BOOL:
                    if ret_type == BOOL:

                        def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                            z[0] = bool(binary_udf(bool(x[0]), bool(y[0])))

                    else:

                        def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                            z[0] = binary_udf(bool(x[0]), bool(y[0]))

                elif ret_type == BOOL:

                    def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                        z[0] = bool(binary_udf(x[0], y[0]))

                else:

                    def binary_wrapper(z, x, y):  # pragma: no cover (numba)
                        z[0] = binary_udf(x[0], y[0])

                binary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(binary_wrapper)
                new_binary = ffi_new("GrB_BinaryOp*")
                check_status_carg(
                    lib.GrB_BinaryOp_new(
                        new_binary,
                        binary_wrapper.cffi,
                        ret_type.gb_obj,
                        type_.gb_obj,
                        type_.gb_obj,
                    ),
                    "BinaryOp",
                    new_binary[0],
                )
                op = TypedUserBinaryOp(new_type_obj, name, type_, ret_type, new_binary[0])
                new_type_obj._add(op)
                success = True
                return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]

        if self.name == "eq" and not self._anonymous and _has_numba:
            nt = numba.types
            # assert dtype.np_type == dtype2.np_type
            itemsize = dtype.np_type.itemsize
            mask = _udt_mask(dtype.np_type)
            ret_type = BOOL
            wrapper_sig = nt.void(
                nt.CPointer(INT8.numba_type),
                nt.CPointer(UINT8.numba_type),
                nt.CPointer(UINT8.numba_type),
            )
            # PERF: we can probably make this faster
            if mask.all():

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover (numba)
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if x[i] != y[i]:
                    #         z_ptr[0] = False
                    #         break
                    # else:
                    #     z_ptr[0] = True
                    z_ptr[0] = (x == y).all()

            else:

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover (numba)
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if mask[i] and x[i] != y[i]:
                    #         z_ptr[0] = False
                    #         break
                    # else:
                    #     z_ptr[0] = True
                    z_ptr[0] = (x[mask] == y[mask]).all()

        elif self.name == "ne" and not self._anonymous and _has_numba:
            nt = numba.types
            # assert dtype.np_type == dtype2.np_type
            itemsize = dtype.np_type.itemsize
            mask = _udt_mask(dtype.np_type)
            ret_type = BOOL
            wrapper_sig = nt.void(
                nt.CPointer(INT8.numba_type),
                nt.CPointer(UINT8.numba_type),
                nt.CPointer(UINT8.numba_type),
            )
            if mask.all():

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover (numba)
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if x[i] != y[i]:
                    #         z_ptr[0] = True
                    #         break
                    # else:
                    #     z_ptr[0] = False
                    z_ptr[0] = (x != y).any()

            else:

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover (numba)
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if mask[i] and x[i] != y[i]:
                    #         z_ptr[0] = True
                    #         break
                    # else:
                    #     z_ptr[0] = False
                    z_ptr[0] = (x[mask] != y[mask]).any()

        elif self._numba_func is None:
            raise KeyError(f"{self.name} does not work with {dtypes} types")
        else:
            numba_func = self._numba_func
            sig = (dtype.numba_type, dtype2.numba_type)
            numba_func.compile(sig)  # Should we catch and give additional error message?
            ret_type = lookup_dtype(numba_func.overloads[sig].signature.return_type)
            binary_wrapper, wrapper_sig = _get_udt_wrapper(numba_func, ret_type, dtype, dtype2)

        binary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(binary_wrapper)
        new_binary = ffi_new("GrB_BinaryOp*")
        check_status_carg(
            lib.GrB_BinaryOp_new(
                new_binary, binary_wrapper.cffi, ret_type._carg, dtype._carg, dtype2._carg
            ),
            "BinaryOp",
            new_binary[0],
        )
        op = TypedUserBinaryOp(
            self,
            self.name,
            dtype,
            ret_type,
            new_binary[0],
            dtype2=dtype2,
        )
        self._udt_types[dtypes] = ret_type
        self._udt_ops[dtypes] = op
        return op

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        """Register a BinaryOp without registering it in the ``graphblas.binary`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes two input parameters of any dtype and returns any dtype.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.binary.{name}``.
        parameterized : bool, default False
            When True, create a parameterized user-defined operator, which means
            additional parameters can be "baked into" the operator when used.
            For example, ``gb.binary.isclose`` is a parameterized function that
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
        BinaryOp or ParameterizedBinaryOp

        """
        cls._check_supports_udf("register_anonymous")
        if parameterized:
            return ParameterizedBinaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        """Register a new BinaryOp and save it to ``graphblas.binary`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.binary.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.binary.x.y.z`` for name ``"x.y.z"``.
        func : FunctionType
            The function to compile. For all current backends, this must be able
            to be compiled with ``numba.njit``.
            ``func`` takes two input parameters of any dtype and returns any dtype.
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
        >>> def max_zero(x, y):
                r = 0
                if x > r:
                    r = x
                if y > r:
                    r = y
                return r
        >>> gb.core.operator.BinaryOp.register_new("max_zero", max_zero)
        >>> dir(gb.binary)
        [..., 'max_zero', ...]

        This is how ``gb.binary.isclose`` is defined:

        >>> def user_isclose(rel_tol=1e-7, abs_tol=0.0):
        >>>     def inner(x, y):
        >>>         return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
        >>>     return inner
        >>> gb.binary.register_new("user_isclose", user_isclose, parameterized=True)

        """
        cls._check_supports_udf("register_new")
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "func": func, "parameterized": parameterized},
            )
        elif parameterized:
            binary_op = ParameterizedBinaryOp(name, func, is_udt=is_udt)
            setattr(module, funcname, binary_op)
        else:
            binary_op = cls._build(name, func, is_udt=is_udt)
            setattr(module, funcname, binary_op)
        # Also save it to `graphblas.op` if not yet defined
        opmodule, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not _hasop(opmodule, funcname):
            if lazy:
                opmodule._delayed[funcname] = module
            else:
                setattr(opmodule, funcname, binary_op)
        if not cls._initialized:
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return binary_op

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover (safety)
            return
        super()._initialize()
        # Rename div to cdiv
        cdiv = binary.cdiv = op.cdiv = BinaryOp("cdiv")
        for dtype, ret_type in binary.div.types.items():
            orig_op = binary.div[dtype]
            cur_op = TypedBuiltinBinaryOp(
                cdiv, "cdiv", dtype, ret_type, orig_op.gb_obj, orig_op.gb_name
            )
            cdiv._add(cur_op)
        del binary.div
        del op.div
        # Add truediv which always points to floating point cdiv
        # We are effectively hacking cdiv to always return floating point values
        # If the inputs are FP32, we use DIV_FP32; use DIV_FP64 for all other input dtypes
        truediv = binary.truediv = op.truediv = BinaryOp("truediv")
        rtruediv = binary.rtruediv = op.rtruediv = BinaryOp("rtruediv")
        for new_op, builtin_op in [(truediv, binary.cdiv), (rtruediv, binary.rdiv)]:
            for dtype in builtin_op.types:
                if dtype.name in {"FP32", "FC32", "FC64"}:
                    orig_dtype = dtype
                else:
                    orig_dtype = FP64
                orig_op = builtin_op[orig_dtype]
                cur_op = TypedBuiltinBinaryOp(
                    new_op,
                    new_op.name,
                    dtype,
                    builtin_op.types[orig_dtype],
                    orig_op.gb_obj,
                    orig_op.gb_name,
                )
                new_op._add(cur_op)
        if _supports_udfs:
            # Add floordiv
            # cdiv truncates towards 0, while floordiv truncates towards -inf
            BinaryOp.register_new("floordiv", _floordiv, lazy=True)  # cast to integer
            BinaryOp.register_new("rfloordiv", _rfloordiv, lazy=True)  # cast to integer

            # For aggregators
            BinaryOp.register_new("absfirst", _absfirst, lazy=True)
            BinaryOp.register_new("abssecond", _abssecond, lazy=True)
            BinaryOp.register_new("rpow", _rpow, lazy=True)

            # For algorithms
            binary._delayed["binom"] = (_register_binom, {})  # Lazy with custom creation
            op._delayed["binom"] = binary

            BinaryOp.register_new("isclose", _isclose, parameterized=True)

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
        name_types = [
            # fmt: off
            (
                ("atan2", "copysign", "fmod", "hypot", "ldexp", "remainder"),
                ((BOOL, INT8, INT16, UINT8, UINT16), FP32),
                ((INT32, INT64, UINT32, UINT64), FP64),
            ),
            (
                (
                    "firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1",
                    "secondj", "secondj1"),
                (
                    position_dtypes,
                    INT64,
                ),
            ),
            (
                ["lxnor"],
                (
                    (
                        FP32, FP64, INT8, INT16, INT32, INT64,
                        UINT8, UINT16, UINT32, UINT64,
                    ),
                    BOOL,
                ),
            ),
            # fmt: on
        ]
        if _supports_complex:
            name_types.append(
                (
                    ["cmplx"],
                    ((BOOL, INT8, INT16, UINT8, UINT16), FP32),
                    ((INT32, INT64, UINT32, UINT64), FP64),
                )
            )
        for names, *types in name_types:
            for name in names:
                if name in _SS_OPERATORS:
                    cur_op = binary._deprecated[name]
                else:
                    cur_op = getattr(binary, name)
                for input_types, target_type in types:
                    typed_op = cur_op._typed_ops[target_type]
                    output_type = cur_op.types[target_type]
                    for dtype in input_types:
                        if dtype not in cur_op.types:  # pragma: no branch (safety)
                            cur_op.types[dtype] = output_type
                            cur_op._typed_ops[dtype] = typed_op
                            cur_op.coercions[dtype] = target_type
        # Not valid input dtypes
        del binary.ldexp[FP32]
        del binary.ldexp[FP64]
        # Fill in commutes info
        for left_name, right_name in cls._commutes.items():
            if left_name in _SS_OPERATORS:
                left = binary._deprecated[left_name]
            else:
                left = getattr(binary, left_name)
            if backend == "suitesparse" and right_name in _SS_OPERATORS:
                left._commutes_to = f"ss.{right_name}"
            else:
                left._commutes_to = right_name
            if right_name not in binary._delayed:
                if right_name in _SS_OPERATORS:
                    right = binary._deprecated[right_name]
                elif _supports_udfs:
                    right = getattr(binary, right_name)
                else:
                    right = getattr(binary, right_name, None)
                    if right is None:
                        continue
                if backend == "suitesparse" and left_name in _SS_OPERATORS:
                    right._commutes_to = f"ss.{left_name}"
                else:
                    right._commutes_to = left_name
        for name in cls._commutative:
            if _supports_udfs:
                cur_op = getattr(binary, name)
            else:
                cur_op = getattr(binary, name, None)
                if cur_op is None:
                    continue
            cur_op._commutes_to = name
        for left_name, right_name in cls._commutes_to_in_semiring.items():
            if left_name in _SS_OPERATORS:
                left = binary._deprecated[left_name]
            else:  # pragma: no cover (safety)
                left = getattr(binary, left_name)
            if right_name in _SS_OPERATORS:
                right = binary._deprecated[right_name]
            else:  # pragma: no cover (safety)
                right = getattr(binary, right_name)
            left._semiring_commutes_to = right
            right._semiring_commutes_to = left
        # Allow some functions to work on UDTs
        for binop, func in [
            (binary.first, _first),
            (binary.second, _second),
            (binary.pair, _pair),
            (binary.any, _first),
        ]:
            binop.orig_func = func
            if _has_numba:
                binop._numba_func = numba.njit(func)
            else:
                binop._numba_func = None
            binop._udt_types = {}
            binop._udt_ops = {}
        binary.any._numba_func = binary.first._numba_func
        binary.eq._udt_types = {}
        binary.eq._udt_ops = {}
        binary.ne._udt_types = {}
        binary.ne._udt_ops = {}
        # Set custom dtype handling
        binary.first._custom_dtype = _first_dtype
        binary.second._custom_dtype = _second_dtype
        binary.pair._custom_dtype = _pair_dtype
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
        self._monoid = None
        self._commutes_to = None
        self._semiring_commutes_to = None
        self.orig_func = func
        self._numba_func = numba_func
        self._is_udt = is_udt
        self.is_positional = is_positional
        self._custom_dtype = None
        if is_udt:
            self._udt_types = {}  # {(dtype, dtype): DataType}
            self._udt_ops = {}  # {(dtype, dtype): TypedUserBinaryOp}

    def __reduce__(self):
        if self._anonymous:
            if hasattr(self.orig_func, "_parameterized_info"):
                return (_deserialize_parameterized, self.orig_func._parameterized_info)
            return (self.register_anonymous, (self.orig_func, self.name))
        if (name := f"binary.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinBinaryOp.__call__
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    commutes_to = ParameterizedBinaryOp.commutes_to

    @property
    def monoid(self):
        if self._monoid is None and not self._anonymous:
            from .monoid import Monoid

            self._monoid = Monoid._find(self.name)
        return self._monoid

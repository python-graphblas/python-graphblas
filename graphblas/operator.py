import inspect
import itertools
import re
from collections.abc import Mapping
from functools import lru_cache, reduce
from operator import getitem, mul
from types import FunctionType, ModuleType

import numba
import numpy as np

from . import (
    _STANDARD_OPERATOR_NAMES,
    binary,
    config,
    ffi,
    indexunary,
    lib,
    monoid,
    op,
    select,
    semiring,
    unary,
)
from .dtypes import (
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
    unify,
)
from .exceptions import UdfParseError, check_status_carg
from .expr import InfixExprBase
from .utils import libget, output_type

if _supports_complex:
    from .dtypes import FC32, FC64

ffi_new = ffi.new
UNKNOWN_OPCLASS = "UnknownOpClass"


def _hasop(module, name):
    return name in module.__dict__ or name in module._delayed


class OpPath:
    def __init__(self, parent, name):
        self._parent = parent
        self._name = name
        self._delayed = {}
        self._delayed_commutes_to = {}

    def __getattr__(self, key):
        if key in self._delayed:
            func, kwargs = self._delayed.pop(key)
            return func(**kwargs)
        self.__getattribute__(key)  # raises


def _call_op(op, left, right=None, thunk=None, **kwargs):
    if right is None and thunk is None:
        if isinstance(left, InfixExprBase):
            # op(A & B), op(A | B), op(A @ B)
            return getattr(left.left, left.method_name)(left.right, op, **kwargs)
        if find_opclass(op)[1] == "Semiring":
            raise TypeError(
                f"Bad type when calling {op!r}.  Got type: {type(left)}.\n"
                f"Expected an infix expression, such as: {op!r}(A @ B)"
            )
        raise TypeError(
            f"Bad type when calling {op!r}.  Got type: {type(left)}.\n"
            "Expected an infix expression or an apply with a Vector or Matrix and a scalar:\n"
            f"    - {op!r}(A & B)\n"
            f"    - {op!r}(A, 1)\n"
            f"    - {op!r}(1, A)"
        )

    # op(A, 1) -> apply (or select if thunk provided)
    from .matrix import Matrix, TransposedMatrix
    from .vector import Vector

    if output_type(left) in {Vector, Matrix, TransposedMatrix}:
        if thunk is not None:
            return left.select(op, thunk=thunk, **kwargs)
        else:
            return left.apply(op, right=right, **kwargs)
    elif output_type(right) in {Vector, Matrix, TransposedMatrix}:
        return right.apply(op, left=left, **kwargs)
    raise TypeError(
        f"Bad types when calling {op!r}.  Got types: {type(left)}, {type(right)}.\n"
        "Expected an infix expression or an apply with a Vector or Matrix and a scalar:\n"
        f"    - {op!r}(A & B)\n"
        f"    - {op!r}(A, 1)\n"
        f"    - {op!r}(1, A)"
    )


_udt_mask_cache = {}


def _udt_mask(dtype):
    """Create mask to determine which bytes of UDTs to use for equality check"""
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


class TypedOpBase:
    __slots__ = (
        "parent",
        "name",
        "type",
        "return_type",
        "gb_obj",
        "gb_name",
        "_type2",
        "__weakref__",
    )

    def __init__(self, parent, name, type_, return_type, gb_obj, gb_name, dtype2=None):
        self.parent = parent
        self.name = name
        self.type = type_
        self.return_type = return_type
        self.gb_obj = gb_obj
        self.gb_name = gb_name
        self._type2 = dtype2

    def __repr__(self):
        classname = self.opclass.lower()
        if classname.endswith("op"):
            classname = classname[:-2]
        dtype2 = "" if self._type2 is None else f", {self._type2.name}"
        return f"{classname}.{self.name}[{self.type.name}{dtype2}]"

    @property
    def _carg(self):
        return self.gb_obj

    @property
    def is_positional(self):
        return self.parent.is_positional

    def __reduce__(self):
        if self._type2 is None or self.type == self._type2:
            return (getitem, (self.parent, self.type))
        else:
            return (getitem, (self.parent, (self.type, self._type2)))


class TypedBuiltinUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "UnaryOp"

    def __call__(self, val):
        from .matrix import Matrix, TransposedMatrix
        from .vector import Vector

        if output_type(val) in {Vector, Matrix, TransposedMatrix}:
            return val.apply(self)
        raise TypeError(
            f"Bad type when calling {self!r}.\n"
            "    - Expected type: Vector, Matrix, TransposedMatrix.\n"
            f"    - Got: {type(val)}.\n"
            "Calling a UnaryOp is syntactic sugar for calling apply.  "
            f"For example, `A.apply({self!r})` is the same as `{self!r}(A)`."
        )


class TypedBuiltinIndexUnaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "IndexUnaryOp"

    def __call__(self, val, thunk=None):
        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        return _call_op(self, val, right=thunk)


class TypedBuiltinSelectOp(TypedOpBase):
    __slots__ = ()
    opclass = "SelectOp"

    def __call__(self, val, thunk=None):
        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        return _call_op(self, val, thunk=thunk)


class TypedBuiltinBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "BinaryOp"

    def __call__(
        self, left, right=None, *, require_monoid=None, left_default=None, right_default=None
    ):
        if left_default is not None or right_default is not None:
            if (
                left_default is None
                or right_default is None
                or require_monoid is not None
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
            return left.left.ewise_union(left.right, self, left_default, right_default)
        if require_monoid is not None:
            if right is not None:
                raise TypeError(
                    f"Bad keyword argument `require_monoid=` when calling {self!r}.\n"
                    "require_monoid keyword may only be used when performing an ewise_add.\n"
                    f"For example: {self!r}(A | B, require_monoid=False)."
                )
            return _call_op(self, left, require_monoid=require_monoid)
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


class TypedBuiltinMonoid(TypedOpBase):
    __slots__ = "_identity"
    opclass = "Monoid"
    is_commutative = True

    def __init__(self, parent, name, type_, return_type, gb_obj, gb_name):
        super().__init__(parent, name, type_, return_type, gb_obj, gb_name)
        self._identity = None

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
            return left.left.ewise_union(left.right, self, left_default, right_default)
        return _call_op(self, left, right)

    @property
    def identity(self):
        if self._identity is None:
            from .recorder import skip_record
            from .vector import Vector

            with skip_record:
                self._identity = (
                    Vector(self.type, size=1, name="").reduce(self, allow_empty=False).new().value
                )
        return self._identity

    @property
    def binaryop(self):
        return getattr(binary, self.name)[self.type]

    @property
    def commutes_to(self):
        return self

    @property
    def type2(self):
        return self.type


class TypedBuiltinSemiring(TypedOpBase):
    __slots__ = ()
    opclass = "Semiring"

    def __call__(self, left, right=None):
        if right is not None:
            raise TypeError(
                f"Bad types when calling {self!r}.  Got types: {type(left)}, {type(right)}.\n"
                f"Expected an infix expression, such as: {self!r}(A @ B)"
            )
        return _call_op(self, left)

    @property
    def binaryop(self):
        return getattr(binary, self.name.split("_", 1)[1])[self.type]

    @property
    def monoid(self):
        monoid_name, binary_name = self.name.split("_", 1)
        binop = getattr(binary, binary_name)[self.type]
        val = getattr(monoid, monoid_name)
        return val[binop.return_type]

    @property
    def commutes_to(self):
        binop = self.binaryop
        commutes_to = binop._semiring_commutes_to or binop.commutes_to
        if commutes_to is None:
            return
        if commutes_to is binop:
            return self
        return get_semiring(self.monoid, commutes_to)

    @property
    def is_commutative(self):
        return self.binaryop.is_commutative

    type2 = TypedBuiltinBinaryOp.type2


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

    commutes_to = TypedBuiltinBinaryOp.commutes_to
    _semiring_commutes_to = TypedBuiltinBinaryOp._semiring_commutes_to
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    orig_func = TypedUserUnaryOp.orig_func
    _numba_func = TypedUserUnaryOp._numba_func
    type2 = TypedBuiltinBinaryOp.type2
    __call__ = TypedBuiltinBinaryOp.__call__


class TypedUserMonoid(TypedOpBase):
    __slots__ = "binaryop", "identity"
    opclass = "Monoid"
    is_commutative = True

    def __init__(self, parent, name, type_, return_type, gb_obj, binaryop, identity):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")
        self.binaryop = binaryop
        self.identity = identity
        binaryop._monoid = self

    commutes_to = TypedBuiltinMonoid.commutes_to
    type2 = TypedBuiltinMonoid.type2
    __call__ = TypedBuiltinMonoid.__call__


class TypedUserSemiring(TypedOpBase):
    __slots__ = "monoid", "binaryop"
    opclass = "Semiring"

    def __init__(self, parent, name, type_, return_type, gb_obj, monoid, binaryop, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)
        self.monoid = monoid
        self.binaryop = binaryop

    commutes_to = TypedBuiltinSemiring.commutes_to
    is_commutative = TypedBuiltinSemiring.is_commutative
    type2 = TypedBuiltinBinaryOp.type2
    __call__ = TypedBuiltinSemiring.__call__


def _deserialize_parameterized(parameterized_op, args, kwargs):
    return parameterized_op(*args, **kwargs)


class ParameterizedUdf:
    __slots__ = "name", "__call__", "_anonymous", "__weakref__"
    is_positional = False
    _custom_dtype = None

    def __init__(self, name, anonymous):
        self.name = name
        self._anonymous = anonymous
        # lru_cache per instance
        method = self._call.__get__(self, type(self))
        self.__call__ = lru_cache(maxsize=1024)(method)

    def _call(self, *args, **kwargs):
        raise NotImplementedError()


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
        rv = UnaryOp._find(name)
        if rv is not None:
            return rv
        return UnaryOp.register_new(name, func, parameterized=True)


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
        rv = IndexUnaryOp._find(name)
        if rv is not None:
            return rv
        return IndexUnaryOp.register_new(name, func, parameterized=True)


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
        rv = SelectOp._find(name)
        if rv is not None:
            return rv
        return SelectOp.register_new(name, func, parameterized=True)


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
                self._monoid(*args, **kwargs)
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
        if type(self._commutes_to) is str:
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
        rv = BinaryOp._find(name)
        if rv is not None:
            return rv
        return BinaryOp.register_new(name, func, parameterized=True)


class ParameterizedMonoid(ParameterizedUdf):
    __slots__ = "binaryop", "identity", "__signature__"
    is_commutative = True

    def __init__(self, name, binaryop, identity, *, anonymous=False):
        if not type(binaryop) is ParameterizedBinaryOp:
            raise TypeError("binaryop must be parameterized")
        self.binaryop = binaryop
        self.__signature__ = binaryop.__signature__
        if callable(identity):
            # assume it must be parameterized as well, so signature must match
            sig = inspect.signature(identity)
            if sig != self.__signature__:
                raise ValueError(
                    f"Signatures of binaryop and identity passed to "
                    f"{type(self).__name__} must be the same.  Got:\n"
                    f"    binaryop{self.__signature__}\n"
                    f"    !=\n"
                    f"    identity{sig}"
                )
        self.identity = identity
        if name is None:
            name = binaryop.name
        super().__init__(name, anonymous)
        binaryop._monoid = self
        # clear binaryop cache so it can be associated with this monoid
        binaryop._cached_call.cache_clear()

    def _call(self, *args, **kwargs):
        binary = self.binaryop(*args, **kwargs)
        identity = self.identity
        if callable(identity):
            identity = identity(*args, **kwargs)
        return Monoid.register_anonymous(binary, identity, self.name)

    commutes_to = TypedBuiltinMonoid.commutes_to

    def __reduce__(self):
        name = f"monoid.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:  # pragma: no cover
            return name
        return (self._deserialize, (self.name, self.binaryop, self.identity, self._anonymous))

    @staticmethod
    def _deserialize(name, binaryop, identity, anonymous):
        if anonymous:
            return Monoid.register_anonymous(binaryop, identity, name)
        rv = Monoid._find(name)
        if rv is not None:
            return rv
        return Monoid.register_new(name, binaryop, identity)


class ParameterizedSemiring(ParameterizedUdf):
    __slots__ = "monoid", "binaryop", "__signature__"

    def __init__(self, name, monoid, binaryop, *, anonymous=False):
        if type(monoid) not in {ParameterizedMonoid, Monoid}:
            raise TypeError("monoid must be of type Monoid or ParameterizedMonoid")
        if type(binaryop) is ParameterizedBinaryOp:
            self.__signature__ = binaryop.__signature__
            if type(monoid) is ParameterizedMonoid and monoid.__signature__ != self.__signature__:
                raise ValueError(
                    f"Signatures of monoid and binaryop passed to "
                    f"{type(self).__name__} must be the same.  Got:\n"
                    f"    monoid{monoid.__signature__}\n"
                    f"    !=\n"
                    f"    binaryop{self.__signature__}\n\n"
                    "Perhaps call monoid or binaryop with parameters before creating the semiring."
                )
        elif type(binaryop) is BinaryOp:
            if type(monoid) is Monoid:
                raise TypeError("At least one of monoid or binaryop must be parameterized")
            self.__signature__ = monoid.__signature__
        else:
            raise TypeError("binaryop must be of type BinaryOp or ParameterizedBinaryOp")
        self.monoid = monoid
        self.binaryop = binaryop
        if name is None:
            name = f"{monoid.name}_{binaryop.name}"
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        monoid = self.monoid
        if type(monoid) is ParameterizedMonoid:
            monoid = monoid(*args, **kwargs)
        binary = self.binaryop
        if type(binary) is ParameterizedBinaryOp:
            binary = binary(*args, **kwargs)
        return Semiring.register_anonymous(monoid, binary, self.name)

    commutes_to = TypedBuiltinSemiring.commutes_to
    is_commutative = TypedBuiltinSemiring.is_commutative

    def __reduce__(self):
        name = f"semiring.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:  # pragma: no cover
            return name
        return (self._deserialize, (self.name, self.monoid, self.binaryop, self._anonymous))

    @staticmethod
    def _deserialize(name, monoid, binaryop, anonymous):
        if anonymous:
            return Semiring.register_anonymous(monoid, binaryop, name)
        rv = Semiring._find(name)
        if rv is not None:
            return rv
        return Semiring.register_new(name, monoid, binaryop)


_VARNAMES = tuple(x for x in dir(lib) if x[0] != "_")


class OpBase:
    __slots__ = (
        "name",
        "_typed_ops",
        "types",
        "coercions",
        "_anonymous",
        "_udt_types",
        "_udt_ops",
        "__weakref__",
    )
    _parse_config = None
    _initialized = False
    _module = None
    _positional = None

    def __init__(self, name, *, anonymous=False):
        self.name = name
        self._typed_ops = {}
        self.types = {}
        self.coercions = {}
        self._anonymous = anonymous
        self._udt_types = None
        self._udt_ops = None

    def __repr__(self):
        return f"{self._modname}.{self.name}"

    def __getitem__(self, type_):
        if type(type_) is tuple:
            dtype1, dtype2 = type_
            dtype1 = lookup_dtype(dtype1)
            dtype2 = lookup_dtype(dtype2)
            return get_typed_op(self, dtype1, dtype2)
        elif not self._is_udt:
            type_ = lookup_dtype(type_)
            if type_ not in self._typed_ops:
                if self._udt_types is None:
                    if self.is_positional:
                        return self._typed_ops[UINT64]
                    raise KeyError(f"{self.name} does not work with {type_}")
            else:
                return self._typed_ops[type_]
        # This is a UDT or is able to operate on UDTs such as `first` any `any`
        dtype = lookup_dtype(type_)
        return self._compile_udt(dtype, dtype)

    def _add(self, op):
        self._typed_ops[op.type] = op
        self.types[op.type] = op.return_type

    def __delitem__(self, type_):
        type_ = lookup_dtype(type_)
        del self._typed_ops[type_]
        del self.types[type_]

    def __contains__(self, type_):
        try:
            self[type_]
        except (TypeError, KeyError, numba.NumbaError):
            return False
        else:
            return True

    @classmethod
    def _remove_nesting(cls, funcname, *, module=None, modname=None, strict=True):
        if module is None:
            module = cls._module
        if modname is None:
            modname = cls._modname
        if "." not in funcname:
            if strict and hasattr(module, funcname):
                raise AttributeError(f"{modname}.{funcname} is already defined")
        else:
            path, funcname = funcname.rsplit(".", 1)
            for folder in path.split("."):
                if not hasattr(module, folder):
                    setattr(module, folder, OpPath(module, folder))
                module = getattr(module, folder)
                modname = f"{modname}.{folder}"
                if not isinstance(module, (OpPath, ModuleType)):
                    raise AttributeError(
                        f"{modname} is already defined. Cannot use as a nested path."
                    )
            if strict and _hasop(module, funcname):
                raise AttributeError(f"{path}.{funcname} is already defined")
        return module, funcname

    @classmethod
    def _find(cls, funcname):
        rv = cls._module
        for attr in funcname.split("."):
            rv = getattr(rv, attr, None)
            if rv is None:
                break
        return rv

    @classmethod
    def _initialize(cls, include_in_ops=True):
        """
        include_in_ops determines whether the operators are included in the
        `gb.ops` namespace in addition to the defined module.
        """
        if cls._initialized:  # pragma: no cover
            return
        # Read in the parse configs
        trim_from_front = cls._parse_config.get("trim_from_front", 0)
        delete_exact = cls._parse_config.get("delete_exact", None)
        num_underscores = cls._parse_config["num_underscores"]

        for re_str, return_prefix in (
            ("re_exprs", None),
            ("re_exprs_return_bool", "BOOL"),
            ("re_exprs_return_float", "FP"),
            ("re_exprs_return_complex", "FC"),
        ):
            if re_str not in cls._parse_config:
                continue
            if "complex" in re_str and not _supports_complex:
                continue
            for r in reversed(cls._parse_config[re_str]):
                for varname in _VARNAMES:
                    m = r.match(varname)
                    if m:
                        # Parse function into name and datatype
                        gb_name = m.string
                        splitname = gb_name[trim_from_front:].split("_")
                        if delete_exact and delete_exact in splitname:
                            splitname.remove(delete_exact)
                        if len(splitname) == num_underscores + 1:
                            *splitname, type_ = splitname
                        else:
                            type_ = None
                        name = "_".join(splitname).lower()
                        # Create object for name unless it already exists
                        if not hasattr(cls._module, name):
                            if cls._positional is None:
                                obj = cls(name)
                            else:
                                obj = cls(name, is_positional=name in cls._positional)
                            setattr(cls._module, name, obj)
                            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
                            if include_in_ops and not hasattr(op, name):
                                setattr(op, name, obj)
                        else:
                            obj = getattr(cls._module, name)
                        gb_obj = getattr(lib, varname)
                        # Determine return type
                        if return_prefix == "BOOL":
                            return_type = BOOL
                            if type_ is None:
                                type_ = BOOL
                        else:
                            if type_ is None:  # pragma: no cover
                                raise TypeError(f"Unable to determine return type for {varname}")
                            if return_prefix is None:
                                return_type = type_
                            else:
                                # Grab the number of bits from type_
                                num_bits = type_[-2:]
                                if num_bits not in {"32", "64"}:  # pragma: no cover
                                    raise TypeError(f"Unexpected number of bits: {num_bits}")
                                return_type = f"{return_prefix}{num_bits}"
                        builtin_op = cls._typed_class(
                            obj,
                            name,
                            lookup_dtype(type_),
                            lookup_dtype(return_type),
                            gb_obj,
                            gb_name,
                        )
                        obj._add(builtin_op)

    @classmethod
    def _deserialize(cls, name, *args):
        rv = cls._find(name)
        if rv is not None:
            return rv  # Should we verify this is what the user expects?
        return cls.register_new(name, *args)


def _identity(x):
    return x  # pragma: no cover


def _one(x):
    return 1  # pragma: no cover


class UnaryOp(OpBase):
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
                            z[0] = bool(unary_udf(bool(x[0])))  # pragma: no cover

                    else:

                        def unary_wrapper(z, x):
                            z[0] = unary_udf(bool(x[0]))  # pragma: no cover

                elif ret_type == BOOL:

                    def unary_wrapper(z, x):
                        z[0] = bool(unary_udf(x[0]))  # pragma: no cover

                else:

                    def unary_wrapper(z, x):
                        z[0] = unary_udf(x[0])  # pragma: no cover

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
        else:
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
        if parameterized:
            return ParameterizedUnaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
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
        for names, *types in (
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
        ):
            for name in names:
                op = getattr(unary, name)
                for input_types, target_type in types:
                    typed_op = op._typed_ops[target_type]
                    output_type = op.types[target_type]
                    for dtype in input_types:
                        if dtype not in op.types:  # pragma: no branch
                            op.types[dtype] = output_type
                            op._typed_ops[dtype] = typed_op
                            op.coercions[dtype] = target_type
        # Allow some functions to work on UDTs
        for (unop, func) in [
            (unary.identity, _identity),
            (unary.one, _one),
        ]:
            unop.orig_func = func
            unop._numba_func = numba.njit(func)
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
        name = f"unary.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinUnaryOp.__call__


class IndexUnaryOp(OpBase):
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
            re.compile("^GrB_(ROWINDEX|COLINDEX)_(INT32|INT64)$"),
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
    # fmt: off
    _positional = {"tril", "triu", "diag", "offdiag", "colle", "colgt", "rowle", "rowgt",
                   "rowindex", "colindex"}
    # fmt: on

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

                        def indexunary_wrapper(z, x, row, col, y):
                            z[0] = bool(
                                indexunary_udf(bool(x[0]), row, col, bool(y[0]))
                            )  # pragma: no cover

                    else:

                        def indexunary_wrapper(z, x, row, col, y):
                            z[0] = indexunary_udf(
                                bool(x[0]), row, col, bool(y[0])
                            )  # pragma: no cover

                elif ret_type == BOOL:

                    def indexunary_wrapper(z, x, row, col, y):
                        z[0] = bool(indexunary_udf(x[0], row, col, y[0]))  # pragma: no cover

                else:

                    def indexunary_wrapper(z, x, row, col, y):
                        z[0] = indexunary_udf(x[0], row, col, y[0])  # pragma: no cover

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
        else:
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
        if parameterized:
            return ParameterizedIndexUnaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
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
                setattr(select, funcname, SelectOp._from_indexunary(indexunary_op))

        if not cls._initialized:
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return indexunary_op

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        super()._initialize()
        # Update type information to include UINT64 for positional ops
        for name in ("tril", "triu", "diag", "offdiag", "colle", "colgt", "rowle", "rowgt"):
            op = getattr(indexunary, name)
            typed_op = op._typed_ops[BOOL]
            output_type = op.types[BOOL]
            if UINT64 not in op.types:  # pragma: no cover
                op.types[UINT64] = output_type
                op._typed_ops[UINT64] = typed_op
                op.coercions[UINT64] = BOOL
        for name in ("rowindex", "colindex"):
            op = getattr(indexunary, name)
            typed_op = op._typed_ops[INT64]
            output_type = op.types[INT64]
            if UINT64 not in op.types:  # pragma: no cover
                op.types[UINT64] = output_type
                op._typed_ops[UINT64] = typed_op
                op.coercions[UINT64] = INT64
        # Add index->row alias to make it more intuitive which to use for vectors
        indexunary.indexle = indexunary.rowle
        indexunary.indexgt = indexunary.rowgt
        indexunary.index = indexunary.rowindex
        # fmt: off
        # Add SelectOp when it makes sense
        for name in ("tril", "triu", "diag", "offdiag",
                     "colle", "colgt", "rowle", "rowgt", "indexle", "indexgt",
                     "valueeq", "valuene", "valuegt", "valuege", "valuelt", "valuele"):
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
        name = f"indexunary.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinIndexUnaryOp.__call__


class SelectOp(OpBase):
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
        for type, t in iop._typed_ops.items():
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
            obj._typed_ops[type] = op
            obj.types[type] = op.return_type
        return obj

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False, is_udt=False):
        if parameterized:
            return ParameterizedSelectOp(name, func, anonymous=True, is_udt=is_udt)
        iop = IndexUnaryOp._build(name, func, anonymous=True, is_udt=is_udt)
        return SelectOp._from_indexunary(iop)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
        iop = IndexUnaryOp.register_new(
            name, func, parameterized=parameterized, is_udt=is_udt, lazy=lazy
        )
        if not all(x == BOOL for x in iop.types.values()):
            raise ValueError("SelectOp must have BOOL return type")
        if lazy:
            return getattr(select, iop.name)

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover
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
        name = f"select.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinSelectOp.__call__


def _floordiv(x, y):
    return x // y  # pragma: no cover


def _rfloordiv(x, y):
    return y // x  # pragma: no cover


def _absfirst(x, y):
    return np.abs(x)  # pragma: no cover


def _abssecond(x, y):
    return np.abs(y)  # pragma: no cover


def _rpow(x, y):
    return y**x  # pragma: no cover


def _isclose(rel_tol=1e-7, abs_tol=0.0):
    def inner(x, y):  # pragma: no cover
        return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)

    return inner


_MAX_INT64 = np.iinfo(np.int64).max


def _binom(N, k):  # pragma: no cover
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
    return x  # pragma: no cover


def _second(x, y):
    return y  # pragma: no cover


def _pair(x, y):
    return 1  # pragma: no cover


def _first_dtype(op, dtype, dtype2):
    if dtype._is_udt:
        return op._compile_udt(dtype, dtype2)
    else:
        return op[dtype]


def _second_dtype(op, dtype, dtype2):
    if dtype2._is_udt:
        return op._compile_udt(dtype, dtype2)
    else:
        return op[dtype2]


def _pair_dtype(op, dtype, dtype2):
    return op[INT64]


def _get_udt_wrapper(numba_func, return_type, dtype, dtype2=None, *, include_indexes=False):
    ztype = INT8 if return_type == BOOL else return_type
    xtype = INT8 if dtype == BOOL else dtype
    nt = numba.types
    wrapper_args = [nt.CPointer(ztype.numba_type), nt.CPointer(xtype.numba_type)]
    if include_indexes:
        wrapper_args.extend([UINT64.numba_type, UINT64.numba_type])
    if dtype2 is not None:
        ytype = INT8 if dtype2 == BOOL else dtype2
        wrapper_args.append(nt.CPointer(ytype.numba_type))
    wrapper_sig = nt.void(*wrapper_args)

    zarray = xarray = yarray = BL = BR = yarg = yname = rcidx = ""
    if return_type._is_udt:
        if return_type.np_type.subdtype is None:
            zarray = "    z = numba.carray(z_ptr, 1)\n"
            zname = "z[0]"
        else:
            zname = "z_ptr[0]"
            BR = "[0]"
    else:
        zname = "z_ptr[0]"
        if return_type == BOOL:
            BL = "bool("
            BR = ")"

    if dtype._is_udt:
        if dtype.np_type.subdtype is None:
            xarray = "    x = numba.carray(x_ptr, 1)\n"
            xname = "x[0]"
        else:
            xname = "x_ptr"
    elif dtype == BOOL:
        xname = "bool(x_ptr[0])"
    else:
        xname = "x_ptr[0]"

    if dtype2 is not None:
        yarg = ", y_ptr"
        if dtype2._is_udt:
            if dtype2.np_type.subdtype is None:
                yarray = "    y = numba.carray(y_ptr, 1)\n"
                yname = ", y[0]"
            else:
                yname = ", y_ptr"
        elif dtype2 == BOOL:
            yname = ", bool(y_ptr[0])"
        else:
            yname = ", y_ptr[0]"

    if include_indexes:
        rcidx = ", row, col"

    d = {"numba": numba, "numba_func": numba_func}
    text = (
        f"def wrapper(z_ptr, x_ptr{rcidx}{yarg}):\n"
        f"{zarray}{xarray}{yarray}"
        f"    {zname} = {BL}numba_func({xname}{rcidx}{yname}){BR}\n"
    )
    exec(text, d)
    return d["wrapper"], wrapper_sig


class BinaryOp(OpBase):
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
                "GrB_(BOR|BAND|BXOR|BXNOR)" "_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
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

                        def binary_wrapper(z, x, y):
                            z[0] = bool(binary_udf(bool(x[0]), bool(y[0])))  # pragma: no cover

                    else:

                        def binary_wrapper(z, x, y):
                            z[0] = binary_udf(bool(x[0]), bool(y[0]))  # pragma: no cover

                elif ret_type == BOOL:

                    def binary_wrapper(z, x, y):
                        z[0] = bool(binary_udf(x[0], y[0]))  # pragma: no cover

                else:

                    def binary_wrapper(z, x, y):
                        z[0] = binary_udf(x[0], y[0])  # pragma: no cover

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
                    new_binary,
                )
                op = TypedUserBinaryOp(new_type_obj, name, type_, ret_type, new_binary[0])
                new_type_obj._add(op)
                success = True
                return_types[type_] = ret_type
        if success or is_udt:
            return new_type_obj
        else:
            raise UdfParseError("Unable to parse function using Numba")

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]

        nt = numba.types
        if self.name == "eq" and not self._anonymous:
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

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover
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

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if mask[i] and x[i] != y[i]:
                    #         z_ptr[0] = False
                    #         break
                    # else:
                    #     z_ptr[0] = True
                    z_ptr[0] = (x[mask] == y[mask]).all()

        elif self.name == "ne" and not self._anonymous:
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

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover
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

                def binary_wrapper(z_ptr, x_ptr, y_ptr):  # pragma: no cover
                    x = numba.carray(x_ptr, itemsize)
                    y = numba.carray(y_ptr, itemsize)
                    # for i in range(itemsize):
                    #     if mask[i] and x[i] != y[i]:
                    #         z_ptr[0] = True
                    #         break
                    # else:
                    #     z_ptr[0] = False
                    z_ptr[0] = (x[mask] != y[mask]).any()

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
            new_binary,
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
        if parameterized:
            return ParameterizedBinaryOp(name, func, anonymous=True, is_udt=is_udt)
        return cls._build(name, func, anonymous=True, is_udt=is_udt)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False, is_udt=False, lazy=False):
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
        if cls._initialized:  # pragma: no cover
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
        for (new_op, builtin_op) in [(truediv, binary.cdiv), (rtruediv, binary.rdiv)]:
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
                    "secondj", "secondj1",),
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
                cur_op = getattr(binary, name)
                for input_types, target_type in types:
                    typed_op = cur_op._typed_ops[target_type]
                    output_type = cur_op.types[target_type]
                    for dtype in input_types:
                        if dtype not in cur_op.types:  # pragma: no branch
                            cur_op.types[dtype] = output_type
                            cur_op._typed_ops[dtype] = typed_op
                            cur_op.coercions[dtype] = target_type
        # Not valid input dtypes
        del binary.ldexp[FP32]
        del binary.ldexp[FP64]
        # Fill in commutes info
        for left_name, right_name in cls._commutes.items():
            left = getattr(binary, left_name)
            left._commutes_to = right_name
            if right_name not in binary._delayed:
                right = getattr(binary, right_name)
                right._commutes_to = left_name
        for name in cls._commutative:
            cur_op = getattr(binary, name)
            cur_op._commutes_to = name
        for left, right in cls._commutes_to_in_semiring.items():
            left = getattr(binary, left)
            right = getattr(binary, right)
            left._semiring_commutes_to = right
            right._semiring_commutes_to = left
        # Allow some functions to work on UDTs
        for (binop, func) in [
            (binary.first, _first),
            (binary.second, _second),
            (binary.pair, _pair),
            (binary.any, _first),
        ]:
            binop.orig_func = func
            binop._numba_func = numba.njit(func)
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
        name = f"binary.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self.orig_func))

    __call__ = TypedBuiltinBinaryOp.__call__
    is_commutative = TypedBuiltinBinaryOp.is_commutative
    commutes_to = ParameterizedBinaryOp.commutes_to

    @property
    def monoid(self):
        if self._monoid is None and not self._anonymous:
            self._monoid = Monoid._find(self.name)
        return self._monoid


class Monoid(OpBase):
    __slots__ = "_binaryop", "_identity"
    is_commutative = True
    is_positional = False
    _custom_dtype = None
    _module = monoid
    _modname = "monoid"
    _typed_class = TypedBuiltinMonoid
    _parse_config = {
        "trim_from_front": 4,
        "delete_exact": "MONOID",
        "num_underscores": 1,
        "re_exprs": [
            re.compile(
                "^GrB_(MIN|MAX|PLUS|TIMES|LOR|LAND|LXOR|LXNOR)_MONOID"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(ANY)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)_MONOID$"
            ),
            re.compile("^GxB_(PLUS|TIMES|ANY)_(FC32|FC64)_MONOID$"),
            re.compile("^GxB_(EQ|ANY)_BOOL_MONOID$"),
            re.compile("^GxB_(BOR|BAND|BXOR|BXNOR)_(UINT8|UINT16|UINT32|UINT64)_MONOID$"),
        ],
    }

    @classmethod
    def _build(cls, name, binaryop, identity, *, anonymous=False):
        if type(binaryop) is not BinaryOp:
            raise TypeError(f"binaryop must be a BinaryOp, not {type(binaryop)}")
        if name is None:
            name = binaryop.name
        new_type_obj = cls(name, binaryop, identity, anonymous=anonymous)
        if not binaryop._is_udt:
            if not isinstance(identity, Mapping):
                identities = dict.fromkeys(binaryop.types, identity)
                explicit_identities = False
            else:
                identities = {lookup_dtype(key): val for key, val in identity.items()}
                explicit_identities = True
            for type_, identity in identities.items():
                ret_type = binaryop[type_].return_type
                # If there is a domain mismatch, then DomainMismatch will be raised
                # below if identities were explicitly given.
                if type_ != ret_type and not explicit_identities:
                    continue
                new_monoid = ffi_new("GrB_Monoid*")
                func = libget(f"GrB_Monoid_new_{type_.name}")
                zcast = ffi.cast(type_.c_type, identity)
                check_status_carg(
                    func(new_monoid, binaryop[type_].gb_obj, zcast), "Monoid", new_monoid[0]
                )
                op = TypedUserMonoid(
                    new_type_obj,
                    name,
                    type_,
                    ret_type,
                    new_monoid[0],
                    binaryop[type_],
                    identity,
                )
                new_type_obj._add(op)
        return new_type_obj

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:
            dtype2 = dtype
        elif dtype != dtype2:
            raise TypeError(
                "Monoid inputs must be the same dtype (got {dtype} and {dtype2}); "
                "unable to coerce when using UDTs."
            )
        if dtype in self._udt_types:
            return self._udt_ops[dtype]
        binaryop = self.binaryop._compile_udt(dtype, dtype2)
        from .scalar import Scalar

        ret_type = binaryop.return_type
        identity = Scalar.from_value(self._identity, dtype=ret_type, is_cscalar=True)
        new_monoid = ffi_new("GrB_Monoid*")
        status = lib.GrB_Monoid_new_UDT(new_monoid, binaryop.gb_obj, identity.gb_obj)
        check_status_carg(status, "Monoid", new_monoid[0])
        op = TypedUserMonoid(
            new_monoid,
            self.name,
            dtype,
            ret_type,
            new_monoid[0],
            binaryop,
            identity,
        )
        self._udt_types[dtype] = dtype
        self._udt_ops[dtype] = op
        return op

    @classmethod
    def register_anonymous(cls, binaryop, identity, name=None):
        if type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedMonoid(name, binaryop, identity, anonymous=True)
        return cls._build(name, binaryop, identity, anonymous=True)

    @classmethod
    def register_new(cls, name, binaryop, identity, *, lazy=False):
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "binaryop": binaryop, "identity": identity},
            )
        elif type(binaryop) is ParameterizedBinaryOp:
            monoid = ParameterizedMonoid(name, binaryop, identity)
            setattr(module, funcname, monoid)
        else:
            monoid = cls._build(name, binaryop, identity)
            setattr(module, funcname, monoid)
        # Also save it to `graphblas.op` if not yet defined
        opmodule, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not _hasop(opmodule, funcname):
            if lazy:
                opmodule._delayed[funcname] = module
            else:
                setattr(opmodule, funcname, monoid)
        if not cls._initialized:  # pragma: no cover
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return monoid

    def __init__(self, name, binaryop=None, identity=None, *, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._binaryop = binaryop
        self._identity = identity
        if binaryop is not None:
            binaryop._monoid = self
            if binaryop._is_udt:
                self._udt_types = {}  # {dtype: DataType}
                self._udt_ops = {}  # {dtype: TypedUserMonoid}

    def __reduce__(self):
        if self._anonymous:
            return (self.register_anonymous, (self._binaryop, self._identity, self.name))
        name = f"monoid.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self._binaryop, self._identity))

    @property
    def binaryop(self):
        if self._binaryop is not None:
            return self._binaryop
        # Must be builtin
        return getattr(binary, self.name)

    @property
    def identities(self):
        return {dtype: val.identity for dtype, val in self._typed_ops.items()}

    @property
    def _is_udt(self):
        return self._binaryop is not None and self._binaryop._is_udt

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover
            return
        super()._initialize()
        lor = monoid.lor._typed_ops[BOOL]
        land = monoid.land._typed_ops[BOOL]
        for cur_op, typed_op in [
            (monoid.max, lor),
            (monoid.min, land),
            # (monoid.plus, lor),  # two choices: lor, or plus[int]
            (monoid.times, land),
        ]:
            if BOOL not in cur_op.types:  # pragma: no branch
                cur_op.types[BOOL] = BOOL
                cur_op.coercions[BOOL] = BOOL
                cur_op._typed_ops[BOOL] = typed_op

        for cur_op in (monoid.lor, monoid.land, monoid.lxnor, monoid.lxor):
            bool_op = cur_op._typed_ops[BOOL]
            for dtype in (
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
            ):
                if dtype in cur_op.types:  # pragma: no cover
                    continue
                cur_op.types[dtype] = BOOL
                cur_op.coercions[dtype] = BOOL
                cur_op._typed_ops[dtype] = bool_op
        # Allow some functions to work on UDTs
        any_ = monoid.any
        any_._identity = 0
        any_._udt_types = {}
        any_._udt_ops = {}
        cls._initialized = True

    commutes_to = TypedBuiltinMonoid.commutes_to
    __call__ = TypedBuiltinMonoid.__call__


class Semiring(OpBase):
    __slots__ = "_monoid", "_binaryop"
    _module = semiring
    _modname = "semiring"
    _typed_class = TypedBuiltinSemiring
    _parse_config = {
        "trim_from_front": 4,
        "delete_exact": "SEMIRING",
        "num_underscores": 2,
        "re_exprs": [
            re.compile(
                "^GrB_(PLUS|MIN|MAX)_(PLUS|TIMES|FIRST|SECOND|MIN|MAX)_SEMIRING"
                "_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(MIN|MAX|PLUS|TIMES|ANY)"
                "_(FIRST|SECOND|PAIR|MIN|MAX|PLUS|MINUS|RMINUS|TIMES"
                "|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR"
                "|FIRSTI1|FIRSTI|FIRSTJ1|FIRSTJ|SECONDI1|SECONDI|SECONDJ1|SECONDJ)"
                "_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(PLUS|TIMES|ANY)_(FIRST|SECOND|PAIR|PLUS|MINUS|TIMES|DIV|RDIV|RMINUS)"
                "_(FC32|FC64)$"
            ),
            re.compile(
                "^GxB_(BOR|BAND|BXOR|BXNOR)_(BOR|BAND|BXOR|BXNOR)_(UINT8|UINT16|UINT32|UINT64)$"
            ),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)_(LOR|LAND)_SEMIRING_BOOL$"),
            re.compile(
                "^GxB_(LOR|LAND|LXOR|EQ|ANY)_(EQ|NE|GT|LT|GE|LE)"
                "_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(LOR|LAND|LXOR|EQ|ANY)_(FIRST|SECOND|PAIR|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_BOOL$"
            ),
        ],
    }

    @classmethod
    def _build(cls, name, monoid, binaryop, *, anonymous=False):
        if type(monoid) is not Monoid:
            raise TypeError(f"monoid must be a Monoid, not {type(monoid)}")
        if type(binaryop) is not BinaryOp:
            raise TypeError(f"binaryop must be a BinaryOp, not {type(binaryop)}")
        if name is None:
            name = f"{monoid.name}_{binaryop.name}".replace(".", "_")
        new_type_obj = cls(name, monoid, binaryop, anonymous=anonymous)
        if binaryop._is_udt:
            return new_type_obj
        for binary_in, binary_func in binaryop._typed_ops.items():
            binary_out = binary_func.return_type
            # Unfortunately, we can't have user-defined monoids over bools yet
            # because numba can't compile correctly.
            if (
                binary_out not in monoid.types
                # Are all coercions bad, or just to bool?
                or monoid.coercions.get(binary_out, binary_out) != binary_out
            ):
                continue
            new_semiring = ffi_new("GrB_Semiring*")
            check_status_carg(
                lib.GrB_Semiring_new(new_semiring, monoid[binary_out].gb_obj, binary_func.gb_obj),
                "Semiring",
                new_semiring,
            )
            ret_type = monoid[binary_out].return_type
            op = TypedUserSemiring(
                new_type_obj,
                name,
                binary_in,
                ret_type,
                new_semiring[0],
                monoid[binary_out],
                binary_func,
            )
            new_type_obj._add(op)
        return new_type_obj

    def _compile_udt(self, dtype, dtype2):
        if dtype2 is None:
            dtype2 = dtype
        dtypes = (dtype, dtype2)
        if dtypes in self._udt_types:
            return self._udt_ops[dtypes]
        binaryop = self.binaryop._compile_udt(dtype, dtype2)
        monoid = self.monoid[binaryop.return_type]
        ret_type = monoid.return_type
        new_semiring = ffi_new("GrB_Semiring*")
        status = lib.GrB_Semiring_new(new_semiring, monoid.gb_obj, binaryop.gb_obj)
        check_status_carg(status, "Semiring", new_semiring)
        op = TypedUserSemiring(
            new_semiring,
            self.name,
            dtype,
            ret_type,
            new_semiring[0],
            monoid,
            binaryop,
            dtype2=dtype2,
        )
        self._udt_types[dtypes] = dtype
        self._udt_ops[dtypes] = op
        return op

    @classmethod
    def register_anonymous(cls, monoid, binaryop, name=None):
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedSemiring(name, monoid, binaryop, anonymous=True)
        return cls._build(name, monoid, binaryop, anonymous=True)

    @classmethod
    def register_new(cls, name, monoid, binaryop, *, lazy=False):
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "monoid": monoid, "binaryop": binaryop},
            )
        elif type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            semiring = ParameterizedSemiring(name, monoid, binaryop)
            setattr(module, funcname, semiring)
        else:
            semiring = cls._build(name, monoid, binaryop)
            setattr(module, funcname, semiring)
        # Also save it to `graphblas.op` if not yet defined
        opmodule, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not _hasop(opmodule, funcname):
            if lazy:
                opmodule._delayed[funcname] = module
            else:
                setattr(opmodule, funcname, semiring)
        if not cls._initialized:
            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{name}")
        if not lazy:
            return semiring

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover
            return
        super()._initialize()
        # Rename div to cdiv (truncate towards 0)
        div_semirings = {
            attr: val
            for attr, val in vars(semiring).items()
            if type(val) is Semiring and attr.endswith("_div")
        }
        for orig_name, orig in div_semirings.items():
            name = f"{orig_name[:-3]}cdiv"
            cdiv_semiring = Semiring(name)
            setattr(semiring, name, cdiv_semiring)
            setattr(op, name, cdiv_semiring)
            delattr(semiring, orig_name)
            delattr(op, orig_name)
            for dtype, ret_type in orig.types.items():
                orig_semiring = orig[dtype]
                new_semiring = TypedBuiltinSemiring(
                    cdiv_semiring,
                    name,
                    dtype,
                    ret_type,
                    orig_semiring.gb_obj,
                    orig_semiring.gb_name,
                )
                cdiv_semiring._add(new_semiring)
        # Also add truediv (always floating point) and floordiv (truncate towards -inf)
        for orig_name, orig in div_semirings.items():
            cls.register_new(f"{orig_name[:-3]}truediv", orig.monoid, binary.truediv, lazy=True)
            cls.register_new(f"{orig_name[:-3]}rtruediv", orig.monoid, "rtruediv", lazy=True)
            cls.register_new(f"{orig_name[:-3]}floordiv", orig.monoid, "floordiv", lazy=True)
            cls.register_new(f"{orig_name[:-3]}rfloordiv", orig.monoid, "rfloordiv", lazy=True)
        # For aggregators
        cls.register_new("plus_pow", monoid.plus, binary.pow)
        cls.register_new("plus_rpow", monoid.plus, "rpow", lazy=True)
        cls.register_new("plus_absfirst", monoid.plus, "absfirst", lazy=True)
        cls.register_new("max_absfirst", monoid.max, "absfirst", lazy=True)
        cls.register_new("plus_abssecond", monoid.plus, "abssecond", lazy=True)
        cls.register_new("max_abssecond", monoid.max, "abssecond", lazy=True)

        # Update type information with sane coercion
        for lname in ("any", "eq", "land", "lor", "lxnor", "lxor"):
            target_name = f"{lname}_ne"
            source_name = f"{lname}_lxor"
            if not hasattr(semiring, target_name):
                continue
            target_op = getattr(semiring, target_name)
            if BOOL not in target_op.types:  # pragma: no branch
                source_op = getattr(semiring, source_name)
                typed_op = source_op._typed_ops[BOOL]
                target_op.types[BOOL] = BOOL
                target_op._typed_ops[BOOL] = typed_op
                target_op.coercions[dtype] = BOOL

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
        notbool_dtypes = [
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
        ]
        if _supports_complex:
            position_dtypes.extend([FC32, FC64])
            notbool_dtypes.extend([FC32, FC64])
        for lnames, rnames, *types in (
            # fmt: off
            (
                ("any", "max", "min", "plus", "times"),
                (
                    "firsti", "firsti1", "firstj", "firstj1",
                    "secondi", "secondi1", "secondj", "secondj1",
                ),
                (
                    position_dtypes,
                    INT64,
                ),
            ),
            (
                ("eq", "land", "lor", "lxnor", "lxor"),
                ("first", "pair", "second"),
                # TODO: check if FC coercion works here
                (
                    notbool_dtypes,
                    BOOL,
                ),
            ),
            (
                ("band", "bor", "bxnor", "bxor"),
                ("band", "bor", "bxnor", "bxor"),
                ([INT8], UINT16),
                ([INT16], UINT32),
                ([INT32], UINT64),
                ([INT64], UINT64),
            ),
            (
                ("any", "eq", "land", "lor", "lxnor", "lxor"),
                ("eq", "land", "lor", "lxnor", "lxor", "ne"),
                (
                    (
                        FP32, FP64, INT8, INT16, INT32, INT64,
                        UINT8, UINT16, UINT32, UINT64,
                    ),
                    BOOL,
                ),
            ),
            # fmt: on
        ):
            for left, right in itertools.product(lnames, rnames):
                name = f"{left}_{right}"
                if not hasattr(semiring, name):
                    continue
                cur_op = getattr(semiring, name)
                for input_types, target_type in types:
                    typed_op = cur_op._typed_ops[target_type]
                    output_type = cur_op.types[target_type]
                    for dtype in input_types:
                        if dtype not in cur_op.types:
                            cur_op.types[dtype] = output_type
                            cur_op._typed_ops[dtype] = typed_op
                            cur_op.coercions[dtype] = target_type

        # Handle a few boolean cases
        for opname, targetname in (
            ("max_first", "lor_first"),
            ("max_second", "lor_second"),
            ("max_land", "lor_land"),
            ("max_lor", "lor_lor"),
            ("max_lxor", "lor_lxor"),
            ("min_first", "land_first"),
            ("min_second", "land_second"),
            ("min_land", "land_land"),
            ("min_lor", "land_lor"),
            ("min_lxor", "land_lxor"),
        ):
            cur_op = getattr(semiring, opname)
            target = getattr(semiring, targetname)
            if BOOL in cur_op.types or BOOL not in target.types:  # pragma: no cover
                continue
            cur_op.types[BOOL] = target.types[BOOL]
            cur_op._typed_ops[BOOL] = target._typed_ops[BOOL]
            cur_op.coercions[BOOL] = BOOL
        cls._initialized = True

    def __init__(self, name, monoid=None, binaryop=None, *, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._monoid = monoid
        self._binaryop = binaryop
        try:
            if self.binaryop._udt_types is not None:
                self._udt_types = {}  # {(dtype, dtype): DataType}
                self._udt_ops = {}  # {(dtype, dtype): TypedUserSemiring}
        except AttributeError:
            # `*_div` semirings raise here, but don't need `_udt_types`
            pass

    def __reduce__(self):
        if self._anonymous:
            return (self.register_anonymous, (self._monoid, self._binaryop, self.name))
        name = f"semiring.{self.name}"
        if name in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self._monoid, self._binaryop))

    @property
    def binaryop(self):
        if self._binaryop is not None:
            return self._binaryop
        # Must be builtin
        return getattr(binary, self.name.split("_")[1])

    @property
    def monoid(self):
        if self._monoid is not None:
            return self._monoid
        # Must be builtin
        return getattr(monoid, self.name.split("_")[0])

    @property
    def is_positional(self):
        return self.binaryop.is_positional

    @property
    def _is_udt(self):
        return self._binaryop is not None and self._binaryop._is_udt

    @property
    def _custom_dtype(self):
        return self.binaryop._custom_dtype

    commutes_to = TypedBuiltinSemiring.commutes_to
    is_commutative = TypedBuiltinSemiring.is_commutative
    __call__ = TypedBuiltinSemiring.__call__


def get_typed_op(op, dtype, dtype2=None, *, is_left_scalar=False, is_right_scalar=False, kind=None):
    if isinstance(op, OpBase):
        # UDTs always get compiled
        if op._is_udt:
            return op._compile_udt(dtype, dtype2)
        # Single dtype is simple lookup
        elif dtype2 is None:
            return op[dtype]
        # Handle special cases such as first and second (may have UDTs)
        elif op._custom_dtype is not None:
            return op._custom_dtype(op, dtype, dtype2)
        # Generic case: try to unify the two dtypes
        try:
            return op[
                unify(dtype, dtype2, is_left_scalar=is_left_scalar, is_right_scalar=is_right_scalar)
            ]
        except (TypeError, AttributeError):
            # Failure to unify implies a dtype is UDT; some builtin operators can handle UDTs
            if op.is_positional:
                return op[UINT64]
            elif op._udt_types is None:
                raise
            else:
                return op._compile_udt(dtype, dtype2)
    elif isinstance(op, ParameterizedUdf):
        op = op()  # Use default parameters of parameterized UDFs
        return get_typed_op(
            op,
            dtype,
            dtype2,
            is_left_scalar=is_left_scalar,
            is_right_scalar=is_right_scalar,
            kind=kind,
        )
    elif isinstance(op, TypedOpBase):
        return op

    from ._agg import Aggregator, TypedAggregator

    if isinstance(op, Aggregator):
        return op[dtype]
    elif isinstance(op, TypedAggregator):
        return op
    elif isinstance(op, str):
        if kind == "unary":
            op = unary_from_string(op)
        elif kind == "select":
            op = select_from_string(op)
        elif kind == "binary":
            op = binary_from_string(op)
        elif kind == "monoid":
            op = monoid_from_string(op)
        elif kind == "semiring":
            op = semiring_from_string(op)
        elif kind == "binary|aggregator":
            try:
                op = binary_from_string(op)
            except ValueError:
                try:
                    op = aggregator_from_string(op)
                except ValueError:
                    raise ValueError(
                        f"Unknown binary or aggregator string: {op!r}.  Example usage: '+[int]'"
                    ) from None

        else:
            raise ValueError(
                f"Unable to get op from string {op!r}.  `kind=` argument must be provided as "
                '"unary", "binary", "monoid", "semiring", "indexunary", "select", '
                'or "binary|aggregator".'
            )
        return get_typed_op(
            op,
            dtype,
            dtype2,
            is_left_scalar=is_left_scalar,
            is_right_scalar=is_right_scalar,
            kind=kind,
        )
    elif isinstance(op, FunctionType):
        if kind == "unary":
            op = UnaryOp.register_anonymous(op, is_udt=True)
            return op._compile_udt(dtype, dtype2)
        elif kind.startswith("binary"):
            op = BinaryOp.register_anonymous(op, is_udt=True)
            return op._compile_udt(dtype, dtype2)
    raise TypeError(f"Unable to get typed operator from object with type {type(op)}")


def find_opclass(gb_op):
    if isinstance(gb_op, OpBase):
        opclass = type(gb_op).__name__
    elif isinstance(gb_op, TypedOpBase):
        opclass = gb_op.opclass
    elif isinstance(gb_op, ParameterizedUdf):
        gb_op = gb_op()  # Use default parameters of parameterized UDFs
        gb_op, opclass = find_opclass(gb_op)
    else:
        opclass = UNKNOWN_OPCLASS
    return gb_op, opclass


def get_semiring(monoid, binaryop, name=None):
    """Get or create a Semiring object from a monoid and binaryop.

    If either are typed, then the returned semiring will also be typed.

    See Also
    --------
    Semiring.register_anonymous
    Semiring.register_new
    """
    monoid, opclass = find_opclass(monoid)
    switched = False
    if opclass == "BinaryOp" and monoid.monoid is not None:
        switched = True
        monoid = monoid.monoid
    elif opclass != "Monoid":
        raise TypeError(f"Expected a Monoid for the monoid argument.  Got type: {type(monoid)}")
    binaryop, opclass = find_opclass(binaryop)
    if opclass == "Monoid":
        if switched:
            raise TypeError(
                "Got a BinaryOp for the monoid argument and a Monoid for the binaryop argument.  "
                "Are the arguments switched?  Hint: you can do `mymonoid.binaryop` to get the "
                "binaryop from a monoid."
            )
        binaryop = binaryop.binaryop
    elif opclass != "BinaryOp":
        raise TypeError(
            f"Expected a BinaryOp for the binaryop argument.  Got type: {type(binaryop)}"
        )
    if isinstance(monoid, Monoid):
        monoid_type = None
    else:
        monoid_type = monoid.type
        monoid = monoid.parent
    if isinstance(binaryop, BinaryOp):
        binary_type = None
    else:
        binary_type = binaryop.type
        binaryop = binaryop.parent
    if monoid._anonymous or binaryop._anonymous:
        rv = Semiring.register_anonymous(monoid, binaryop, name=name)
    else:
        *monoid_prefix, monoid_name = monoid.name.rsplit(".", 1)
        *binary_prefix, binary_name = binaryop.name.rsplit(".", 1)
        if (
            monoid_prefix
            and binary_prefix
            and monoid_prefix == binary_prefix
            or config.get("mapnumpy")
            and (
                monoid_prefix == ["numpy"]
                and not binary_prefix
                or binary_prefix == ["numpy"]
                and not monoid_prefix
            )
        ):
            canonical_name = (
                ".".join(monoid_prefix or binary_prefix) + f".{monoid_name}_{binary_name}"
            )
        else:
            canonical_name = f"{monoid.name}_{binaryop.name}".replace(".", "_")
        if name is None:
            name = canonical_name

        module, funcname = Semiring._remove_nesting(canonical_name, strict=False)
        rv = (
            getattr(module, funcname)
            if funcname in module.__dict__ or funcname in module._delayed
            else None
        )
        if rv is None and name != canonical_name:
            module, funcname = Semiring._remove_nesting(name, strict=False)
            rv = (
                getattr(module, funcname)
                if funcname in module.__dict__ or funcname in module._delayed
                else None
            )
        if rv is None:
            rv = Semiring.register_new(canonical_name, monoid, binaryop)
        elif rv.monoid is not monoid or rv.binaryop is not binaryop:  # pragma: no cover
            # It's not the object we expect (can this happen?)
            rv = Semiring.register_anonymous(monoid, binaryop, name=name)
        if name != canonical_name:
            module, funcname = Semiring._remove_nesting(name, strict=False)
            if not _hasop(module, funcname):  # pragma: no branch
                setattr(module, funcname, rv)

    if binary_type is not None:
        return rv[binary_type]
    elif monoid_type is not None:
        return rv[monoid_type]
    else:
        return rv


# Now initialize all the things!
try:
    UnaryOp._initialize()
    IndexUnaryOp._initialize()
    SelectOp._initialize()
    BinaryOp._initialize()
    Monoid._initialize()
    Semiring._initialize()
except Exception:  # pragma: no cover
    # Exceptions here can often get ignored by Python
    import traceback

    traceback.print_exc()
    raise

unary.register_new = UnaryOp.register_new
unary.register_anonymous = UnaryOp.register_anonymous
indexunary.register_new = IndexUnaryOp.register_new
indexunary.register_anonymous = IndexUnaryOp.register_anonymous
select.register_new = SelectOp.register_new
select.register_anonymous = SelectOp.register_anonymous
binary.register_new = BinaryOp.register_new
binary.register_anonymous = BinaryOp.register_anonymous
monoid.register_new = Monoid.register_new
monoid.register_anonymous = Monoid.register_anonymous
semiring.register_new = Semiring.register_new
semiring.register_anonymous = Semiring.register_anonymous

select._binary_to_select.update(
    {
        binary.eq: select.valueeq,
        binary.ne: select.valuene,
        binary.le: select.valuele,
        binary.lt: select.valuelt,
        binary.ge: select.valuege,
        binary.gt: select.valuegt,
        binary.iseq: select.valueeq,
        binary.isne: select.valuene,
        binary.isle: select.valuele,
        binary.islt: select.valuelt,
        binary.isge: select.valuege,
        binary.isgt: select.valuegt,
    }
)

_str_to_unary = {
    "-": unary.ainv,
    "~": unary.lnot,
}
_str_to_select = {
    "<": select.valuelt,
    ">": select.valuegt,
    "<=": select.valuele,
    ">=": select.valuege,
    "!=": select.valuene,
    "==": select.valueeq,
    "col<=": select.colle,
    "col>": select.colgt,
    "row<=": select.rowle,
    "row>": select.rowgt,
    "index<=": select.indexle,
    "index>": select.indexgt,
}
_str_to_binary = {
    "<": binary.lt,
    ">": binary.gt,
    "<=": binary.le,
    ">=": binary.ge,
    "!=": binary.ne,
    "==": binary.eq,
    "+": binary.plus,
    "-": binary.minus,
    "*": binary.times,
    "/": binary.truediv,
    "//": "floordiv",
    "%": "numpy.mod",
    "**": binary.pow,
    "&": binary.land,
    "|": binary.lor,
    "^": binary.lxor,
}
_str_to_monoid = {
    "==": monoid.eq,
    "+": monoid.plus,
    "*": monoid.times,
    "&": monoid.land,
    "|": monoid.lor,
    "^": monoid.lxor,
}


def _from_string(string, module, mapping, example):
    s = string.lower().strip()
    base, *dtype = s.split("[")
    if len(dtype) > 1:
        name = module.__name__.split(".")[-1]
        raise ValueError(
            f'Bad {name} string: {string!r}.  Contains too many "[".  Example usage: {example!r}'
        )
    elif dtype:
        dtype = dtype[0]
        if not dtype.endswith("]"):
            name = module.__name__.split(".")[-1]
            raise ValueError(
                f'Bad {name} string: {string!r}.  Datatype specification does not end with "]".  '
                f"Example usage: {example!r}"
            )
        dtype = lookup_dtype(dtype[:-1])
    if "]" in base:
        name = module.__name__.split(".")[-1]
        raise ValueError(
            f'Bad {name} string: {string!r}.  "]" not matched by "[".  Example usage: {example!r}'
        )
    if base in mapping:
        op = mapping[base]
        if type(op) is str:
            op = mapping[base] = module.from_string(op)
    elif hasattr(module, base):
        op = getattr(module, base)
    elif hasattr(module, "numpy") and hasattr(module.numpy, base):
        op = getattr(module.numpy, base)
    else:
        *paths, attr = base.split(".")
        op = None
        cur = module
        for path in paths:
            cur = getattr(cur, path, None)
            if not isinstance(cur, (OpPath, ModuleType)):
                cur = None
                break
        op = getattr(cur, attr, None)
        if op is None:
            name = module.__name__.split(".")[-1]
            raise ValueError(f"Unknown {name} string: {string!r}.  Example usage: {example!r}")
    if dtype:
        op = op[dtype]
    return op


def unary_from_string(string):
    return _from_string(string, unary, _str_to_unary, "abs[int]")


def indexunary_from_string(string):
    # "select" is a variant of IndexUnary, so the string abbreviations in
    # _str_to_select are appropriate to reuse here
    return _from_string(string, indexunary, _str_to_select, "row_index")


def select_from_string(string):
    return _from_string(string, select, _str_to_select, "tril")


def binary_from_string(string):
    return _from_string(string, binary, _str_to_binary, "+[int]")


def monoid_from_string(string):
    return _from_string(string, monoid, _str_to_monoid, "+[int]")


def semiring_from_string(string):
    split = string.split(".")
    if len(split) == 1:
        try:
            return _from_string(string, semiring, {}, "min.+[int]")
        except Exception:
            pass
    if len(split) != 2:
        raise ValueError(
            f"Bad semiring string: {string!r}.  "
            'The monoid and binaryop should be separated by exactly one period, ".".  '
            "Example usage: min.+[int]"
        )
    cur_monoid = monoid_from_string(split[0])
    cur_binary = binary_from_string(split[1])
    return get_semiring(cur_monoid, cur_binary)


def op_from_string(string):
    for func in [
        # Note: order matters here
        unary_from_string,
        binary_from_string,
        monoid_from_string,
        semiring_from_string,
        indexunary_from_string,
        select_from_string,
        aggregator_from_string,
    ]:
        try:
            return func(string)
        except Exception:
            pass
    raise ValueError(f"Unknown op string: {string!r}.  Example usage: 'abs[int]'")


unary.from_string = unary_from_string
indexunary.from_string = indexunary_from_string
select.from_string = select_from_string
binary.from_string = binary_from_string
monoid.from_string = monoid_from_string
semiring.from_string = semiring_from_string
op.from_string = op_from_string

_str_to_agg = {
    "+": "sum",
    "*": "prod",
    "&": "all",
    "|": "any",
}


def aggregator_from_string(string):
    return _from_string(string, agg, _str_to_agg, "sum[int]")


from . import agg  # noqa isort:skip

agg.from_string = aggregator_from_string

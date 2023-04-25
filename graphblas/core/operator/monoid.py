import inspect
import re
from collections.abc import Mapping

from ... import _STANDARD_OPERATOR_NAMES, binary, monoid, op
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
    lookup_dtype,
)
from ...exceptions import check_status_carg
from .. import ffi, lib
from ..expr import InfixExprBase
from ..utils import libget
from .base import OpBase, ParameterizedUdf, TypedOpBase, _call_op, _hasop
from .binary import BinaryOp, ParameterizedBinaryOp

ffi_new = ffi.new


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
            from ..recorder import skip_record
            from ..vector import Vector

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

    @property
    def is_idempotent(self):
        """True if ``monoid(x, x) == x`` for any x."""
        return self.parent.is_idempotent


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
    is_idempotent = TypedBuiltinMonoid.is_idempotent
    __call__ = TypedBuiltinMonoid.__call__


class ParameterizedMonoid(ParameterizedUdf):
    __slots__ = "binaryop", "identity", "_is_idempotent", "__signature__"
    is_commutative = True

    def __init__(self, name, binaryop, identity, *, is_idempotent=False, anonymous=False):
        if type(binaryop) is not ParameterizedBinaryOp:
            raise TypeError("binaryop must be parameterized")
        self.binaryop = binaryop
        self.__signature__ = binaryop.__signature__
        if callable(identity):
            # assume it must be parameterized as well, so signature must match
            sig = inspect.signature(identity)
            if sig != self.__signature__:
                raise ValueError(
                    "Signatures of binaryop and identity passed to "
                    f"{type(self).__name__} must be the same.  Got:\n"
                    f"    binaryop{self.__signature__}\n"
                    "    !=\n"
                    f"    identity{sig}"
                )
        self.identity = identity
        self._is_idempotent = is_idempotent
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
        return Monoid.register_anonymous(
            binary, identity, self.name, is_idempotent=self._is_idempotent
        )

    commutes_to = TypedBuiltinMonoid.commutes_to

    @property
    def is_idempotent(self):
        """True if ``monoid(x, x) == x`` for any x."""
        return self._is_idempotent

    def __reduce__(self):
        name = f"monoid.{self.name}"
        if not self._anonymous and name in _STANDARD_OPERATOR_NAMES:  # pragma: no cover
            return name
        return (self._deserialize, (self.name, self.binaryop, self.identity, self._anonymous))

    @staticmethod
    def _deserialize(name, binaryop, identity, anonymous):
        if anonymous:
            return Monoid.register_anonymous(binaryop, identity, name)
        if (rv := Monoid._find(name)) is not None:
            return rv
        return Monoid.register_new(name, binaryop, identity)


class Monoid(OpBase):
    """Takes two inputs and returns one output, all of the same data type.

    Built-in and registered Monoids are located in the ``graphblas.monoid`` namespace
    as well as in the ``graphblas.ops`` combined namespace.
    """

    __slots__ = "_binaryop", "_identity", "_is_idempotent"
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
    def _build(cls, name, binaryop, identity, *, is_idempotent=False, anonymous=False):
        if type(binaryop) is not BinaryOp:
            raise TypeError(f"binaryop must be a BinaryOp, not {type(binaryop)}")
        if name is None:
            name = binaryop.name
        new_type_obj = cls(
            name, binaryop, identity, is_idempotent=is_idempotent, anonymous=anonymous
        )
        if not binaryop._is_udt:
            if not isinstance(identity, Mapping):
                identities = dict.fromkeys(binaryop.types, identity)
                explicit_identities = False
            else:
                identities = {lookup_dtype(key): val for key, val in identity.items()}
                explicit_identities = True
            for type_, ident in identities.items():
                ret_type = binaryop[type_].return_type
                # If there is a domain mismatch, then DomainMismatch will be raised
                # below if identities were explicitly given.
                if type_ != ret_type and not explicit_identities:
                    continue
                new_monoid = ffi_new("GrB_Monoid*")
                func = libget(f"GrB_Monoid_new_{type_.name}")
                zcast = ffi.cast(type_.c_type, ident)
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
                    ident,
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
        from ..scalar import Scalar

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
    def register_anonymous(cls, binaryop, identity, name=None, *, is_idempotent=False):
        """Register a Monoid without registering it in the ``graphblas.monoid`` namespace.

        A monoid is a binary operator whose inputs and output are the same dtype.
        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        binaryop: BinaryOp or ParameterizedBinaryOp
            The binary operator of the monoid, which should be able to use the same
            dtype for both inputs and the output.
        identity: scalar or Mapping
            The identity of the monoid such that ``op(x, identity) == x`` for any x.
            ``identity`` may also be a mapping from dtype to scalar.
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.monoid.{name}``.
        is_idempotent : bool, default False
            Does ``op(x, x) == x`` for any x?

        Returns
        -------
        Monoid or ParameterizedMonoid
        """
        if type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedMonoid(
                name, binaryop, identity, is_idempotent=is_idempotent, anonymous=True
            )
        return cls._build(name, binaryop, identity, is_idempotent=is_idempotent, anonymous=True)

    @classmethod
    def register_new(cls, name, binaryop, identity, *, is_idempotent=False, lazy=False):
        """Register a new Monoid and save it to ``graphblas.monoid`` namespace.

        A monoid is a binary operator whose inputs and output are the same dtype.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.monoid.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.monoid.x.y.z`` for name ``"x.y.z"``.
        binaryop: BinaryOp or ParameterizedBinaryOp
            The binary operator of the monoid, which should be able to use the same
            dtype for both inputs and the output.
        identity: scalar or Mapping
            The identity of the monoid such that ``op(x, identity) == x`` for any x.
            ``identity`` may also be a mapping from dtype to scalar.
        is_idempotent : bool, default False
            Does ``op(x, x) == x`` for any x?
        lazy : bool, default False
            If False (the default), then the function will be automatically
            compiled for builtin types (unless ``is_udt`` was True for the binaryop).
            Compiling functions can be slow, however, so you may want to
            delay compilation and only compile when the operator is used,
            which is done by setting ``lazy=True``.

        Examples
        --------
        >>> gb.core.operator.Monoid.register_new("max_zero", gb.binary.max_zero, 0)
        >>> dir(gb.monoid)
        [..., 'max_zero', ...]
        """
        module, funcname = cls._remove_nesting(name)
        if lazy:
            module._delayed[funcname] = (
                cls.register_new,
                {"name": name, "binaryop": binaryop, "identity": identity},
            )
        elif type(binaryop) is ParameterizedBinaryOp:
            monoid = ParameterizedMonoid(name, binaryop, identity, is_idempotent=is_idempotent)
            setattr(module, funcname, monoid)
        else:
            monoid = cls._build(name, binaryop, identity, is_idempotent=is_idempotent)
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

    def __init__(self, name, binaryop=None, identity=None, *, is_idempotent=False, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._binaryop = binaryop
        self._identity = identity
        self._is_idempotent = is_idempotent
        if binaryop is not None:
            binaryop._monoid = self
            if binaryop._is_udt:
                self._udt_types = {}  # {dtype: DataType}
                self._udt_ops = {}  # {dtype: TypedUserMonoid}

    def __reduce__(self):
        if self._anonymous:
            return (self.register_anonymous, (self._binaryop, self._identity, self.name))
        if (name := f"monoid.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self._binaryop, self._identity))

    @property
    def binaryop(self):
        """The :class:`BinaryOp` associated with the Monoid."""
        if self._binaryop is not None:
            return self._binaryop
        # Must be builtin
        return getattr(binary, self.name)

    @property
    def identities(self):
        """The per-dtype identity values for the Monoid."""
        return {dtype: val.identity for dtype, val in self._typed_ops.items()}

    @property
    def is_idempotent(self):
        """True if ``monoid(x, x) == x`` for any x."""
        return self._is_idempotent

    @property
    def _is_udt(self):
        return self._binaryop is not None and self._binaryop._is_udt

    @classmethod
    def _initialize(cls):
        if cls._initialized:  # pragma: no cover (safety)
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
            if BOOL not in cur_op.types:  # pragma: no branch (safety)
                cur_op.types[BOOL] = BOOL
                cur_op.coercions[BOOL] = BOOL
                cur_op._typed_ops[BOOL] = typed_op

        for cur_op in [monoid.lor, monoid.land, monoid.lxnor, monoid.lxor]:
            bool_op = cur_op._typed_ops[BOOL]
            for dtype in [
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
            ]:
                if dtype in cur_op.types:  # pragma: no cover (safety)
                    continue
                cur_op.types[dtype] = BOOL
                cur_op.coercions[dtype] = BOOL
                cur_op._typed_ops[dtype] = bool_op

        # Builtin monoids that are idempotent; i.e., `op(x, x) == x` for any x
        for name in ["any", "band", "bor", "land", "lor", "max", "min"]:
            getattr(monoid, name)._is_idempotent = True
        # Allow some functions to work on UDTs
        any_ = monoid.any
        any_._identity = 0
        any_._udt_types = {}
        any_._udt_ops = {}
        cls._initialized = True

    commutes_to = TypedBuiltinMonoid.commutes_to
    __call__ = TypedBuiltinMonoid.__call__

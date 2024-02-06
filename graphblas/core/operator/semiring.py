import itertools
import re

from ... import _STANDARD_OPERATOR_NAMES, binary, monoid, op, semiring
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
)
from ...exceptions import check_status_carg
from .. import _supports_udfs, ffi, lib
from .base import _SS_OPERATORS, OpBase, ParameterizedUdf, TypedOpBase, _call_op, _hasop
from .binary import BinaryOp, ParameterizedBinaryOp
from .monoid import Monoid, ParameterizedMonoid

if _supports_complex:
    from ...dtypes import FC32, FC64

ffi_new = ffi.new


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
        name = self.name.split("_", 1)[1]
        if name in _SS_OPERATORS:
            binop = binary._deprecated[name]
        else:
            binop = getattr(binary, name)
        return binop[self.type]

    @property
    def monoid(self):
        monoid_name, binary_name = self.name.split("_", 1)
        if binary_name in _SS_OPERATORS:
            binop = binary._deprecated[binary_name]
        else:
            binop = getattr(binary, binary_name)
        binop = binop[self.type]
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
        from .utils import get_semiring

        return get_semiring(self.monoid, commutes_to)

    @property
    def is_commutative(self):
        return self.binaryop.is_commutative

    @property
    def type2(self):
        return self.type if self._type2 is None else self._type2


class TypedUserSemiring(TypedOpBase):
    __slots__ = "monoid", "binaryop"
    opclass = "Semiring"

    def __init__(self, parent, name, type_, return_type, gb_obj, monoid, binaryop, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}", dtype2=dtype2)
        self.monoid = monoid
        self.binaryop = binaryop

    commutes_to = TypedBuiltinSemiring.commutes_to
    is_commutative = TypedBuiltinSemiring.is_commutative
    type2 = TypedBuiltinSemiring.type2
    __call__ = TypedBuiltinSemiring.__call__


class ParameterizedSemiring(ParameterizedUdf):
    __slots__ = "monoid", "binaryop", "__signature__"

    def __init__(self, name, monoid, binaryop, *, anonymous=False):
        if type(monoid) not in {ParameterizedMonoid, Monoid}:
            raise TypeError("monoid must be of type Monoid or ParameterizedMonoid")
        if type(binaryop) is ParameterizedBinaryOp:
            self.__signature__ = binaryop.__signature__
            if type(monoid) is ParameterizedMonoid and monoid.__signature__ != self.__signature__:
                raise ValueError(
                    "Signatures of monoid and binaryop passed to "
                    f"{type(self).__name__} must be the same.  Got:\n"
                    f"    monoid{monoid.__signature__}\n"
                    "    !=\n"
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
        if (rv := Semiring._find(name)) is not None:
            return rv
        return Semiring.register_new(name, monoid, binaryop)


class Semiring(OpBase):
    """Combination of a :class:`Monoid` and a :class:`BinaryOp`.

    Semirings are most commonly used for performing matrix multiplication,
    with the BinaryOp taking the place of the standard multiplication operator
    and the Monoid taking the place of the standard addition operator.

    Built-in and registered Semirings are located in the ``graphblas.semiring`` namespace
    as well as in the ``graphblas.ops`` combined namespace.
    """

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
                new_semiring[0],
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
        check_status_carg(status, "Semiring", new_semiring[0])
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
        """Register a Semiring without registering it in the ``graphblas.semiring`` namespace.

        Because it is not registered in the namespace, the name is optional.

        Parameters
        ----------
        monoid : Monoid or ParameterizedMonoid
            The monoid of the semiring (like "plus" in the default "plus_times" semiring).
        binaryop : BinaryOp or ParameterizedBinaryOp
            The binaryop of the semiring (like "times" in the default "plus_times" semiring).
        name : str, optional
            The name of the operator. This *does not* show up as ``gb.semiring.{name}``.

        Returns
        -------
        Semiring or ParameterizedSemiring

        """
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedSemiring(name, monoid, binaryop, anonymous=True)
        return cls._build(name, monoid, binaryop, anonymous=True)

    @classmethod
    def register_new(cls, name, monoid, binaryop, *, lazy=False):
        """Register a new Semiring and save it to ``graphblas.semiring`` namespace.

        Parameters
        ----------
        name : str
            The name of the operator. This will show up as ``gb.semiring.{name}``.
            The name may contain periods, ".", which will result in nested objects
            such as ``gb.semiring.x.y.z`` for name ``"x.y.z"``.
        monoid : Monoid or ParameterizedMonoid
            The monoid of the semiring (like "plus" in the default "plus_times" semiring).
        binaryop : BinaryOp or ParameterizedBinaryOp
            The binaryop of the semiring (like "times" in the default "plus_times" semiring).
        lazy : bool, default False
            If False (the default), then the function will be automatically
            compiled for builtin types (unless ``is_udt`` is True).
            Compiling functions can be slow, however, so you may want to
            delay compilation and only compile when the operator is used,
            which is done by setting ``lazy=True``.

        Examples
        --------
        >>> gb.core.operator.Semiring.register_new("max_max", gb.monoid.max, gb.binary.max)
        >>> dir(gb.semiring)
        [..., 'max_max', ...]

        """
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
        if cls._initialized:  # pragma: no cover (safety)
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
            if _supports_udfs:
                cls.register_new(f"{orig_name[:-3]}floordiv", orig.monoid, "floordiv", lazy=True)
                cls.register_new(f"{orig_name[:-3]}rfloordiv", orig.monoid, "rfloordiv", lazy=True)
        # For aggregators
        cls.register_new("plus_pow", monoid.plus, binary.pow)
        if _supports_udfs:
            cls.register_new("plus_rpow", monoid.plus, "rpow", lazy=True)
            cls.register_new("plus_absfirst", monoid.plus, "absfirst", lazy=True)
            cls.register_new("max_absfirst", monoid.max, "absfirst", lazy=True)
            cls.register_new("plus_abssecond", monoid.plus, "abssecond", lazy=True)
            cls.register_new("max_abssecond", monoid.max, "abssecond", lazy=True)

        # Update type information with sane coercion
        for lname in ["any", "eq", "land", "lor", "lxnor", "lxor"]:
            target_name = f"{lname}_ne"
            source_name = f"{lname}_lxor"
            if not _hasop(semiring, target_name):
                continue
            target_op = getattr(semiring, target_name)
            if BOOL not in target_op.types:  # pragma: no branch (safety)
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
        for lnames, rnames, *types in [
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
        ]:
            for left, right in itertools.product(lnames, rnames):
                name = f"{left}_{right}"
                if not _hasop(semiring, name):
                    continue
                if name in _SS_OPERATORS:
                    cur_op = semiring._deprecated[name]
                else:
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
        for opname, targetname in [
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
        ]:
            cur_op = getattr(semiring, opname)
            target = getattr(semiring, targetname)
            if BOOL in cur_op.types or BOOL not in target.types:  # pragma: no cover (safety)
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
        if (name := f"semiring.{self.name}") in _STANDARD_OPERATOR_NAMES:
            return name
        return (self._deserialize, (self.name, self._monoid, self._binaryop))

    @property
    def binaryop(self):
        """The :class:`BinaryOp` associated with the Semiring."""
        if self._binaryop is not None:
            return self._binaryop
        # Must be builtin
        name = self.name.split("_")[1]
        if name in _SS_OPERATORS:
            return binary._deprecated[name]
        return getattr(binary, name)

    @property
    def monoid(self):
        """The :class:`Monoid` associated with the Semiring."""
        if self._monoid is not None:
            return self._monoid
        # Must be builtin
        return getattr(monoid, self.name.split("_")[0].split(".")[-1])

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

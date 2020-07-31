import inspect
import re
import numpy as np
import numba
from collections.abc import Mapping
from functools import lru_cache
from types import FunctionType, ModuleType
from . import ffi, lib, unary, binary, monoid, semiring
from .dtypes import libget, lookup_dtype, unify, INT8, _sample_values
from .exceptions import UdfParseError, check_status

ffi_new = ffi.new
UNKNOWN_OPCLASS = "UnknownOpClass"


def _normalize_type(type_):
    return lookup_dtype(type_).name


class OpPath:
    def __init__(self, parent, name):
        self._parent = parent
        self._name = name


class TypedOpBase:
    def __init__(self, name, type_, return_type, gb_obj):
        self.name = name
        self.type = _normalize_type(type_)
        self.return_type = _normalize_type(return_type)
        self.gb_obj = gb_obj

    def __repr__(self):
        classname = self.opclass.lower()
        if classname.endswith("op"):
            classname = classname[:-2]
        return f"{classname}.{self.name}[{self.type}]"

    @property
    def _carg(self):
        return self.gb_obj


class TypedBuiltinUnaryOp(TypedOpBase):
    opclass = "UnaryOp"


class TypedBuiltinBinaryOp(TypedOpBase):
    opclass = "BinaryOp"


class TypedBuiltinMonoid(TypedOpBase):
    opclass = "Monoid"


class TypedBuiltinSemiring(TypedOpBase):
    opclass = "Semiring"


class TypedUserUnaryOp(TypedOpBase):
    opclass = "UnaryOp"

    def __init__(self, name, type_, return_type, gb_obj, orig_func, numba_func):
        super().__init__(name, type_, return_type, gb_obj)
        self.orig_func = orig_func
        self.numba_func = numba_func


class TypedUserBinaryOp(TypedOpBase):
    opclass = "BinaryOp"

    def __init__(self, name, type_, return_type, gb_obj, orig_func, numba_func):
        super().__init__(name, type_, return_type, gb_obj)
        self.orig_func = orig_func
        self.numba_func = numba_func


class TypedUserMonoid(TypedOpBase):
    opclass = "Monoid"

    def __init__(self, name, type_, return_type, gb_obj, binaryop, identity):
        super().__init__(name, type_, return_type, gb_obj)
        self.binaryop = binaryop
        self.identity = identity


class TypedUserSemiring(TypedOpBase):
    opclass = "Semiring"

    def __init__(self, name, type_, return_type, gb_obj, monoid, binaryop):
        super().__init__(name, type_, return_type, gb_obj)
        self.monoid = monoid
        self.binaryop = binaryop


class ParameterizedUdf:
    def __init__(self, name):
        self.name = name
        # lru_cache per instance
        method = self.__call__.__get__(self, type(self))
        self.__call__ = lru_cache(maxsize=1024)(method)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ParameterizedUnaryOp(ParameterizedUdf):
    def __init__(self, name, func):
        self.func = func
        self.__signature__ = inspect.signature(func)
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        unary = self.func(*args, **kwargs)
        return UnaryOp.register_anonymous(unary, self.name)


class ParameterizedBinaryOp(ParameterizedUdf):
    def __init__(self, name, func):
        self.func = func
        self.__signature__ = inspect.signature(func)
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        binary = self.func(*args, **kwargs)
        return BinaryOp.register_anonymous(binary, self.name)


class ParameterizedMonoid(ParameterizedUdf):
    def __init__(self, name, binaryop, identity):
        if not type(binaryop) is ParameterizedBinaryOp:
            raise TypeError("binaryop must be parameterized")
        self.binaryop = binaryop
        self.__signature__ = binaryop.__signature__
        if callable(identity):
            # assume it must be parameterized as well, so signature must match
            sig = inspect.signature(identity)
            if sig != self.__signature__:
                raise ValueError(
                    f"Signatures of binarop and identity passed to "
                    f"{type(self).__name__} must be the same.  Got:\n"
                    f"    binaryop{self.__signature__}\n"
                    f"    !=\n"
                    f"    identity{sig}"
                )
        self.identity = identity
        if name is None:
            name = binaryop.name
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        binary = self.binaryop
        binary = binary(*args, **kwargs)
        identity = self.identity
        if callable(identity):
            identity = identity(*args, **kwargs)
        return Monoid.register_anonymous(binary, identity, self.name)


class ParameterizedSemiring(ParameterizedUdf):
    def __init__(self, name, monoid, binaryop):
        if type(monoid) not in {ParameterizedMonoid, Monoid}:
            raise TypeError("monoid must be of type Monoid or ParameterizedMonoid")
        if type(binaryop) is ParameterizedBinaryOp:
            self.__signature__ = binaryop.__signature__
            if type(monoid) is ParameterizedMonoid and monoid.__signature__ != self.__signature__:
                raise ValueError(
                    f"Signatures of monoid and binarop passed to "
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
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        monoid = self.monoid
        if type(monoid) is ParameterizedMonoid:
            monoid = monoid(*args, **kwargs)
        binary = self.binaryop
        if type(binary) is ParameterizedBinaryOp:
            binary = binary(*args, **kwargs)
        return Semiring.register_anonymous(monoid, binary, self.name)


class OpBase:
    _parse_config = None
    _initialized = False
    _module = None

    def __init__(self, name):
        self.name = name
        self._typed_ops = {}
        self.types = {}

    def __repr__(self):
        return f"{self._modname}.{self.name}"

    def __getitem__(self, type_):
        type_ = _normalize_type(type_)
        if type_ not in self._typed_ops:
            raise KeyError(f"{self.name} does not work with {type_}")
        return self._typed_ops[type_]

    def _add(self, op):
        self._typed_ops[op.type] = op
        self.types[op.type] = op.return_type

    def __delitem__(self, type_):
        type_ = _normalize_type(type_)
        del self._typed_ops[type_]
        del self.types[type_]

    def __contains__(self, type_):
        type_ = _normalize_type(type_)
        return type_ in self._typed_ops

    @classmethod
    def _remove_nesting(cls, funcname):
        module = cls._module
        modname = cls._modname
        if "." not in funcname:
            if hasattr(module, funcname):
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
        return module, funcname

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        # Read in the parse configs
        trim_from_front = cls._parse_config.get("trim_from_front", 0)
        delete_exact = cls._parse_config.get("delete_exact", None)
        num_underscores = cls._parse_config["num_underscores"]

        varnames = tuple(x for x in dir(lib) if x[0] != "_")
        for re_str, return_prefix in (
            ("re_exprs", None),
            ("re_exprs_return_bool", "BOOL"),
            ("re_exprs_return_float", "FP"),
            ("re_exprs_return_complex", "FC"),
        ):
            if re_str not in cls._parse_config:
                continue
            for r in cls._parse_config[re_str]:
                for varname in varnames:
                    m = r.match(varname)
                    if m:
                        # Parse function into name and datatype
                        splitname = m.string[trim_from_front:].split("_")
                        if delete_exact and delete_exact in splitname:
                            splitname.remove(delete_exact)
                        if len(splitname) == num_underscores + 1:
                            *splitname, type_ = splitname
                        else:
                            type_ = None
                        name = "_".join(splitname).lower()
                        # Create object for name unless it already exists
                        if not hasattr(cls._module, name):
                            setattr(cls._module, name, cls(name))
                        obj = getattr(cls._module, name)
                        gb_obj = getattr(lib, varname)
                        # Determine return type
                        if return_prefix == "BOOL":
                            return_type = "BOOL"
                            if type_ is None:
                                type_ = "BOOL"
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
                        op = cls._typed_class(name, type_, return_type, gb_obj)
                        obj._add(op)
        cls._initialized = True


class UnaryOp(OpBase):
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
                "^GxB_(LNOT|ONE)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(SQRT|LOG|EXP|LOG2|SIN|COS|TAN|ACOS|ASIN|ATAN|SINH|COSH|TANH|ACOSH"
                "|ASINH|ATANH|SIGNUM|CEIL|FLOOR|ROUND|TRUNC|EXP2|EXPM1|LOG10|LOG1P)"
                "_(FP32|FP64|FC32|FC64)$"
            ),
            re.compile("^GxB_(LGAMMA|TGAMMA|ERF|ERFC|FREXPX|FREXPE)_(FP32|FP64)$"),
            re.compile("^GxB_(IDENTITY|AINV|MINV|ONE|CONJ)_(FC32|FC64)$"),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_LNOT$"),
            re.compile("^GxB_(ISINF|ISNAN|ISFINITE)_(FP32|FP64|FC32|FC64)$"),
        ],
        "re_exprs_return_float": [re.compile("^GxB_(CREAL|CIMAG|CARG|ABS)_(FC32|FC64)$"),],
    }

    @classmethod
    def _build(cls, name, func):
        if type(func) is not FunctionType:
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_unary>")
        success = False
        new_type_obj = cls(name)
        return_types = {}
        nt = numba.types
        for type_, sample_val in _sample_values.items():
            type_ = lookup_dtype(type_)
            # Check if func can handle this data type
            try:
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    ret = func(sample_val)
                ret_type = lookup_dtype(type(ret))
                if ret_type != type_ and (
                    "INT" in ret_type.name
                    and "INT" in type_.name
                    or "FP" in ret_type.name
                    and "FP" in type_.name
                    or "FC" in ret_type.name
                    and "FC" in type_.name
                    or type_ == "UINT64"
                    and ret_type == "FP64"
                    and return_types.get("INT64") == "INT64"
                ):
                    # Downcast `ret_type` to `type_`.
                    # This is what users want most of the time, but we can't make a perfect rule.
                    # There should be a way for users to be explicit.
                    ret_type = type_
                elif type_ == "BOOL" and ret_type == "INT64" and return_types.get("INT8") == "INT8":
                    ret_type = INT8

                # Numba is unable to handle BOOL correctly right now, but we have a workaround
                # See: https://github.com/numba/numba/issues/5395
                # We're relying on coercion behaving correctly here
                input_type = INT8 if type_ == "BOOL" else type_
                return_type = INT8 if ret_type == "BOOL" else ret_type

                # JIT the func so it can be used from a cfunc
                unary_udf = numba.njit(func)
                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(
                    nt.CPointer(return_type.numba_type), nt.CPointer(input_type.numba_type),
                )

                if type_ == "BOOL":
                    if ret_type == "BOOL":

                        def unary_wrapper(z, x):
                            z[0] = bool(unary_udf(bool(x[0])))  # pragma: no cover

                    else:

                        def unary_wrapper(z, x):
                            z[0] = unary_udf(bool(x[0]))  # pragma: no cover

                elif ret_type == "BOOL":

                    def unary_wrapper(z, x):
                        z[0] = bool(unary_udf(x[0]))  # pragma: no cover

                else:

                    def unary_wrapper(z, x):
                        z[0] = unary_udf(x[0])  # pragma: no cover

                unary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(unary_wrapper)
                new_unary = ffi_new("GrB_UnaryOp*")
                check_status(
                    lib.GrB_UnaryOp_new(
                        new_unary, unary_wrapper.cffi, ret_type.gb_type, type_.gb_type
                    )
                )
                op = TypedUserUnaryOp(
                    name, type_.name, ret_type.name, new_unary[0], func, unary_udf
                )
                new_type_obj._add(op)
                success = True
                return_types[type_.name] = ret_type.name
            except Exception:
                continue
        if success:
            return new_type_obj
        else:
            raise UdfParseError("Unable to parse function using Numba")

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False):
        if parameterized:
            return ParameterizedUnaryOp(name, func)
        return cls._build(name, func)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            unary_op = ParameterizedUnaryOp(name, func)
        else:
            unary_op = cls._build(name, func)
        setattr(module, funcname, unary_op)


class BinaryOp(OpBase):
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
                "^GxB_(POW|RMINUS|RDIV|PAIR|ANY|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)"
                "_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64|FC32|FC64)$"
            ),
            re.compile("^GxB_(FIRST|SECOND|PLUS|MINUS|TIMES|DIV)_(FC32|FC64)$"),
            re.compile("^GxB_(ATAN2|HYPOT|FMOD|REMAINDER|LDEXP|COPYSIGN)_(FP32|FP64)$"),
            re.compile(
                "GrB_(BOR|BAND|BXOR|BXNOR)" "_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
            re.compile(
                "GxB_(BGET|BSET|BCLR|BSHIFT)"
                "_(INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64)$"
            ),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)$"),
            re.compile(
                "^GrB_(EQ|NE|GT|LT|GE|LE)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile(
                "^GxB_(LOR|LAND|LXOR)_"
                "(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$"
            ),
            re.compile("^GxB_(EQ|NE)_(FC32|FC64)$"),
        ],
        "re_exprs_return_complex": [re.compile("^GxB_(CMPLX)_(FP32|FP64)$"),],
    }

    @classmethod
    def _build(cls, name, func):
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_binary>")
        success = False
        new_type_obj = cls(name)
        return_types = {}
        nt = numba.types
        for type_, sample_val in _sample_values.items():
            type_ = lookup_dtype(type_)
            # Check if func can handle this data type
            try:
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    ret = func(sample_val, sample_val)
                ret_type = lookup_dtype(type(ret))
                if ret_type != type_ and (
                    "INT" in ret_type.name
                    and "INT" in type_.name
                    or "FP" in ret_type.name
                    and "FP" in type_.name
                    or "FC" in ret_type.name
                    and "FC" in type_.name
                    or type_ == "UINT64"
                    and ret_type == "FP64"
                    and return_types.get("INT64") == "INT64"
                ):
                    # Downcast `ret_type` to `type_`.
                    # This is what users want most of the time, but we can't make a perfect rule.
                    # There should be a way for users to be explicit.
                    ret_type = type_
                elif type_ == "BOOL" and ret_type == "INT64" and return_types.get("INT8") == "INT8":
                    ret_type = INT8

                # Numba is unable to handle BOOL correctly right now, but we have a workaround
                # See: https://github.com/numba/numba/issues/5395
                # We're relying on coercion behaving correctly here
                input_type = INT8 if type_ == "BOOL" else type_
                return_type = INT8 if ret_type == "BOOL" else ret_type

                # JIT the func so it can be used from a cfunc
                binary_udf = numba.njit(func)

                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(
                    nt.CPointer(return_type.numba_type),
                    nt.CPointer(input_type.numba_type),
                    nt.CPointer(input_type.numba_type),
                )

                if type_ == "BOOL":
                    if ret_type == "BOOL":

                        def binary_wrapper(z, x, y):
                            z[0] = bool(binary_udf(bool(x[0]), bool(y[0])))  # pragma: no cover

                    else:

                        def binary_wrapper(z, x, y):
                            z[0] = binary_udf(bool(x[0]), bool(y[0]))  # pragma: no cover

                elif ret_type == "BOOL":

                    def binary_wrapper(z, x, y):
                        z[0] = bool(binary_udf(x[0], y[0]))  # pragma: no cover

                else:

                    def binary_wrapper(z, x, y):
                        z[0] = binary_udf(x[0], y[0])  # pragma: no cover

                binary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(binary_wrapper)
                new_binary = ffi_new("GrB_BinaryOp*")
                check_status(
                    lib.GrB_BinaryOp_new(
                        new_binary,
                        binary_wrapper.cffi,
                        ret_type.gb_type,
                        type_.gb_type,
                        type_.gb_type,
                    )
                )
                op = TypedUserBinaryOp(
                    name, type_.name, ret_type.name, new_binary[0], func, binary_udf
                )
                new_type_obj._add(op)
                success = True
                return_types[type_.name] = ret_type.name
            except Exception:
                continue
        if success:
            return new_type_obj
        else:
            raise UdfParseError("Unable to parse function using Numba")

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False):
        if parameterized:
            return ParameterizedBinaryOp(name, func)
        return cls._build(name, func)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            binary_op = ParameterizedBinaryOp(name, func)
        else:
            binary_op = cls._build(name, func)
        setattr(module, funcname, binary_op)

    @classmethod
    def _initialize(cls):
        super()._initialize()
        # Rename div to cdiv
        binary.cdiv = BinaryOp("cdiv")
        for dtype, ret_type in binary.div.types.items():
            op = TypedBuiltinBinaryOp("cdiv", dtype, ret_type, binary.div[dtype].gb_obj)
            binary.cdiv._add(op)
        del binary.div
        # Add truediv which always points to floating point cdiv
        # We are effectively hacking cdiv to always return floating point values
        # If the inputs are FP32, we use DIV_FP32; use DIV_FP64 for all other input dtypes
        binary.truediv = BinaryOp("truediv")
        for dtype in binary.cdiv.types:
            float_type = "FP32" if dtype == "FP32" else "FP64"
            op = TypedBuiltinBinaryOp(
                "truediv", dtype, binary.cdiv.types[float_type], binary.cdiv[float_type].gb_obj,
            )
            binary.truediv._add(op)
        # Add floordiv
        # cdiv truncates towards 0, while floordiv truncates towards -inf
        BinaryOp.register_new("floordiv", lambda x, y: x // y)

        def isclose(rel_tol=1e-7, abs_tol=0.0):
            def inner(x, y):
                return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)

            return inner

        BinaryOp.register_new("isclose", isclose, parameterized=True)


class Monoid(OpBase):
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
    def _build(cls, name, binaryop, identity):
        if type(binaryop) is not BinaryOp:
            raise TypeError(f"binaryop must be a BinaryOp, not {type(binaryop)}")
        if name is None:
            name = binaryop.name
        new_type_obj = cls(name)
        if not isinstance(identity, Mapping):
            identities = dict.fromkeys(binaryop.types, identity)
            explicit_identities = False
        else:
            identities = identity
            explicit_identities = True
        for type_, identity in identities.items():
            type_ = lookup_dtype(type_)
            ret_type = binaryop[type_].return_type
            # If there is a domain mismatch, then DomainMismatch will be raised
            # below if identities were explicitly given.
            if type_ != ret_type and not explicit_identities or "FC" in type_.name:
                continue
            new_monoid = ffi_new("GrB_Monoid*")
            func = libget(f"GrB_Monoid_new_{type_.name}")
            zcast = ffi.cast(type_.c_type, identity)
            check_status(func(new_monoid, binaryop[type_].gb_obj, zcast))
            op = TypedUserMonoid(
                name, type_.name, ret_type, new_monoid[0], binaryop[type_], identity
            )
            new_type_obj._add(op)
        return new_type_obj

    @classmethod
    def register_anonymous(cls, binaryop, identity, name=None):
        if type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedMonoid(name, binaryop, identity)
        return cls._build(name, binaryop, identity)

    @classmethod
    def register_new(cls, name, binaryop, identity):
        module, funcname = cls._remove_nesting(name)
        if type(binaryop) is ParameterizedBinaryOp:
            monoid = ParameterizedMonoid(name, binaryop, identity)
        else:
            monoid = cls._build(name, binaryop, identity)
        setattr(module, funcname, monoid)


class Semiring(OpBase):
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
                "|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)"
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
    def _build(cls, name, monoid, binaryop):
        if type(monoid) is not Monoid:
            raise TypeError(f"monoid must be a Monoid, not {type(monoid)}")
        if type(binaryop) is not BinaryOp:
            raise TypeError(f"binaryop must be a BinaryOp, not {type(binaryop)}")
        if name is None:
            name = f"{monoid.name}_{binaryop.name}"
        new_type_obj = cls(name)
        for binary_in, binary_func in binaryop._typed_ops.items():
            binary_out = binary_func.return_type
            # Unfortunately, we can't have user-defined monoids over bools yet
            # because numba can't compile correctly.
            if binary_out not in monoid.types:
                continue
            binary_out = lookup_dtype(binary_out)
            new_semiring = ffi_new("GrB_Semiring*")
            check_status(
                lib.GrB_Semiring_new(new_semiring, monoid[binary_out].gb_obj, binary_func.gb_obj)
            )
            ret_type = monoid[binary_out].return_type
            op = TypedUserSemiring(
                name, binary_in, ret_type, new_semiring[0], monoid[binary_out], binary_func,
            )
            new_type_obj._add(op)
        return new_type_obj

    @classmethod
    def register_anonymous(cls, monoid, binaryop, name=None):
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedSemiring(name, monoid, binaryop)
        return cls._build(name, monoid, binaryop)

    @classmethod
    def register_new(cls, name, monoid, binaryop):
        module, funcname = cls._remove_nesting(name)
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            semiring = ParameterizedSemiring(name, monoid, binaryop)
        else:
            semiring = cls._build(name, monoid, binaryop)
        setattr(module, funcname, semiring)


def get_typed_op(op, dtype, dtype2=None):
    if isinstance(op, OpBase):
        if dtype2 is not None:
            dtype = unify(dtype, dtype2)
        return op[dtype]
    elif isinstance(op, ParameterizedUdf):
        op = op()  # Use default parameters of parameterized UDFs
        return get_typed_op(op, dtype, dtype2)
    elif isinstance(op, TypedOpBase):
        return op
    else:
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


# Now initialize all the things!
UnaryOp._initialize()
BinaryOp._initialize()
Monoid._initialize()
Semiring._initialize()

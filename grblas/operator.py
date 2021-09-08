import inspect
import itertools
import re
from collections.abc import Mapping
from functools import lru_cache
from types import FunctionType, ModuleType

import numba
import numpy as np

from . import binary, ffi, lib, monoid, op, semiring, unary
from .dtypes import INT8, _sample_values, _supports_complex, lookup_dtype, unify
from .exceptions import UdfParseError, check_status_carg
from .expr import InfixExprBase
from .utils import libget, output_type

ffi_new = ffi.new
UNKNOWN_OPCLASS = "UnknownOpClass"


def _normalize_type(type_):
    return lookup_dtype(type_).name


class OpPath:
    def __init__(self, parent, name):
        self._parent = parent
        self._name = name


def _call_op(op, left, right=None, **kwargs):
    if right is None:
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

    # op(A, 1) -> apply (or select once available)
    from .matrix import Matrix, TransposedMatrix
    from .vector import Vector

    if output_type(left) in {Vector, Matrix, TransposedMatrix}:
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


class TypedOpBase:
    __slots__ = "parent", "name", "type", "return_type", "gb_obj", "gb_name", "__weakref__"

    def __init__(self, parent, name, type_, return_type, gb_obj, gb_name):
        self.parent = parent
        self.name = name
        self.type = _normalize_type(type_)
        self.return_type = _normalize_type(return_type)
        self.gb_obj = gb_obj
        self.gb_name = gb_name

    def __repr__(self):
        classname = self.opclass.lower()
        if classname.endswith("op"):
            classname = classname[:-2]
        return f"{classname}.{self.name}[{self.type}]"

    @property
    def _carg(self):
        return self.gb_obj


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


class TypedBuiltinBinaryOp(TypedOpBase):
    __slots__ = ()
    opclass = "BinaryOp"

    def __call__(self, left, right=None, *, require_monoid=None):
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


class TypedBuiltinMonoid(TypedOpBase):
    __slots__ = "_identity"
    opclass = "Monoid"

    def __init__(self, parent, name, type_, return_type, gb_obj, gb_name):
        super().__init__(parent, name, type_, return_type, gb_obj, gb_name)
        self._identity = None

    def __call__(self, left, right=None):
        return _call_op(self, left, right)

    @property
    def identity(self):
        if self._identity is None:
            from .recorder import skip_record
            from .vector import Vector

            with skip_record:
                self._identity = Vector.new(size=1, dtype=self.type, name="").reduce(self).value
        return self._identity

    @property
    def binaryop(self):
        return getattr(binary, self.name)[self.type]


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
        if binop.return_type not in val.types and binary_name in {"land", "lor", "lxor", "lxnor"}:
            # e.g., with `plus_land`, `land` always returns a BOOL, but `plus` doesn't
            # operate on BOOL, so assume the return of `land` is coerced.
            return val[self.type]
        return val[binop.return_type]


class TypedUserUnaryOp(TypedOpBase):
    __slots__ = "orig_func", "numba_func"
    opclass = "UnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, orig_func, numba_func):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")
        self.orig_func = orig_func
        self.numba_func = numba_func

    __call__ = TypedBuiltinUnaryOp.__call__


class TypedUserBinaryOp(TypedOpBase):
    __slots__ = "orig_func", "numba_func", "_monoid"
    opclass = "BinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, orig_func, numba_func):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")
        self.orig_func = orig_func
        self.numba_func = numba_func
        self._monoid = None

    @property
    def monoid(self):
        if self._monoid is None and not self.parent._anonymous:
            monoid = Monoid._find(self.name)
            if monoid is not None and self.type in monoid._typed_ops:  # pragma: no cover
                # This may be used by grblas.binary.numpy objects
                self._monoid = monoid[self.type]
        return self._monoid

    __call__ = TypedBuiltinBinaryOp.__call__


class TypedUserMonoid(TypedOpBase):
    __slots__ = "binaryop", "identity"
    opclass = "Monoid"

    def __init__(self, parent, name, type_, return_type, gb_obj, binaryop, identity):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")
        self.binaryop = binaryop
        self.identity = identity
        binaryop._monoid = self

    __call__ = TypedBuiltinMonoid.__call__


class TypedUserSemiring(TypedOpBase):
    __slots__ = "monoid", "binaryop"
    opclass = "Semiring"

    def __init__(self, parent, name, type_, return_type, gb_obj, monoid, binaryop):
        super().__init__(parent, name, type_, return_type, gb_obj, f"{name}_{type_}")
        self.monoid = monoid
        self.binaryop = binaryop

    __call__ = TypedBuiltinSemiring.__call__


class ParameterizedUdf:
    __slots__ = "name", "__call__", "_anonymous", "__weakref__"

    def __init__(self, name, anonymous):
        self.name = name
        self._anonymous = anonymous
        # lru_cache per instance
        method = self._call.__get__(self, type(self))
        self.__call__ = lru_cache(maxsize=1024)(method)

    def _call(self, *args, **kwargs):
        raise NotImplementedError()


class ParameterizedUnaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__"

    def __init__(self, name, func, *, anonymous=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)

    def _call(self, *args, **kwargs):
        unary = self.func(*args, **kwargs)
        return UnaryOp.register_anonymous(unary, self.name)


class ParameterizedBinaryOp(ParameterizedUdf):
    __slots__ = "func", "__signature__", "_monoid", "_cached_call"

    def __init__(self, name, func, *, anonymous=False):
        self.func = func
        self.__signature__ = inspect.signature(func)
        self._monoid = None
        if name is None:
            name = getattr(func, "__name__", name)
        super().__init__(name, anonymous)
        method = self._call_to_cache.__get__(self, type(self))
        self._cached_call = lru_cache(maxsize=1024)(method)
        self.__call__ = self._call

    def _call_to_cache(self, *args, **kwargs):
        binary = self.func(*args, **kwargs)
        return BinaryOp.register_anonymous(binary, self.name)

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
            assert binop._monoid is not binop
        return binop

    @property
    def monoid(self):
        return self._monoid


class ParameterizedMonoid(ParameterizedUdf):
    __slots__ = "binaryop", "identity", "__signature__"

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


class OpBase:
    __slots__ = "name", "_typed_ops", "types", "coercions", "_anonymous", "__weakref__"
    _parse_config = None
    _initialized = False
    _module = None

    def __init__(self, name, *, anonymous=False):
        self.name = name
        self._typed_ops = {}
        self.types = {}
        self.coercions = {}
        self._anonymous = anonymous

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
            # Can't use `hasattr` here, b/c we use `__getattr__` in numpy namespaces
            if strict and funcname in module.__dict__:
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
            if "complex" in re_str and not _supports_complex:  # pragma: no cover
                continue
            for r in reversed(cls._parse_config[re_str]):
                for varname in varnames:
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
                            obj = cls(name)
                            setattr(cls._module, name, obj)
                            if not hasattr(op, name):
                                setattr(op, name, obj)
                        else:
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
                        builtin_op = cls._typed_class(
                            obj, name, type_, return_type, gb_obj, gb_name
                        )
                        obj._add(builtin_op)
        cls._initialized = True


class UnaryOp(OpBase):
    __slots__ = ()
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
            re.compile("^GxB_(LGAMMA|TGAMMA|ERF|ERFC|FREXPX|FREXPE)_(FP32|FP64)$"),
            re.compile("^GxB_(IDENTITY|AINV|MINV|ONE|CONJ)_(FC32|FC64)$"),
        ],
        "re_exprs_return_bool": [
            re.compile("^GrB_LNOT$"),
            re.compile("^GxB_(ISINF|ISNAN|ISFINITE)_(FP32|FP64|FC32|FC64)$"),
        ],
        "re_exprs_return_float": [re.compile("^GxB_(CREAL|CIMAG|CARG|ABS)_(FC32|FC64)$")],
    }

    @classmethod
    def _build(cls, name, func, *, anonymous=False):
        if type(func) is not FunctionType:
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_unary>")
        success = False
        new_type_obj = cls(name, anonymous=anonymous)
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
                    ("INT" in ret_type.name and "INT" in type_.name)
                    or ("FP" in ret_type.name and "FP" in type_.name)
                    or ("FC" in ret_type.name and "FC" in type_.name)
                    or (
                        type_ == "UINT64"
                        and ret_type == "FP64"
                        and return_types.get("INT64") == "INT64"
                    )
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
                    nt.CPointer(return_type.numba_type),
                    nt.CPointer(input_type.numba_type),
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
                check_status_carg(
                    lib.GrB_UnaryOp_new(
                        new_unary, unary_wrapper.cffi, ret_type.gb_obj, type_.gb_obj
                    ),
                    "UnaryOp",
                    new_unary,
                )
                op = TypedUserUnaryOp(
                    new_type_obj, name, type_.name, ret_type.name, new_unary[0], func, unary_udf
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
            return ParameterizedUnaryOp(name, func, anonymous=True)
        return cls._build(name, func, anonymous=True)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            unary_op = ParameterizedUnaryOp(name, func)
        else:
            unary_op = cls._build(name, func)
        setattr(module, funcname, unary_op)
        # Also save it to `grblas.op` if not yet defined
        module, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not hasattr(module, funcname):
            setattr(module, funcname, unary_op)
        return unary_op

    @classmethod
    def _initialize(cls):
        super()._initialize()
        # Update type information with sane coercion
        for names, *types in (
            # fmt: off
            (
                (
                    "erf", "erfc", "lgamma", "tgamma", "acos", "acosh", "asin", "asinh",
                    "atan", "atanh", "ceil", "cos", "cosh", "exp", "exp2", "expm1", "floor",
                    "log", "log10", "log1p", "log2", "round", "signum", "sin", "sinh", "sqrt",
                    "tan", "tanh", "trunc",
                ),
                (("BOOL", "INT8", "INT16", "UINT8", "UINT16"), "FP32"),
                (("INT32", "INT64", "UINT32", "UINT64"), "FP64"),
            ),
            (
                ("positioni", "positioni1", "positionj", "positionj1"),
                (
                    (
                        "BOOL", "FC32", "FC64", "FP32", "FP64", "INT8", "INT16",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "INT64",
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
                        if dtype not in op.types:
                            op.types[dtype] = output_type
                            op._typed_ops[dtype] = typed_op
                            op.coercions[dtype] = target_type

    __call__ = TypedBuiltinUnaryOp.__call__


class BinaryOp(OpBase):
    __slots__ = "_monoid"
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
        "re_exprs_return_complex": [re.compile("^GxB_(CMPLX)_(FP32|FP64)$")],
    }

    @classmethod
    def _build(cls, name, func, *, anonymous=False):
        if not isinstance(func, FunctionType):
            raise TypeError(f"UDF argument must be a function, not {type(func)}")
        if name is None:
            name = getattr(func, "__name__", "<anonymous_binary>")
        success = False
        new_type_obj = cls(name, anonymous=anonymous)
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
                    ("INT" in ret_type.name and "INT" in type_.name)
                    or ("FP" in ret_type.name and "FP" in type_.name)
                    or ("FC" in ret_type.name and "FC" in type_.name)
                    or (
                        type_ == "UINT64"
                        and ret_type == "FP64"
                        and return_types.get("INT64") == "INT64"
                    )
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
                op = TypedUserBinaryOp(
                    new_type_obj, name, type_.name, ret_type.name, new_binary[0], func, binary_udf
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
            return ParameterizedBinaryOp(name, func, anonymous=True)
        return cls._build(name, func, anonymous=True)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            binary_op = ParameterizedBinaryOp(name, func)
        else:
            binary_op = cls._build(name, func)
        setattr(module, funcname, binary_op)
        # Also save it to `grblas.op` if not yet defined
        module, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not hasattr(module, funcname):
            setattr(module, funcname, binary_op)
        return binary_op

    @classmethod
    def _initialize(cls):
        super()._initialize()
        # Rename div to cdiv
        cdiv = binary.cdiv = BinaryOp("cdiv")
        for dtype, ret_type in binary.div.types.items():
            orig_op = binary.div[dtype]
            op = TypedBuiltinBinaryOp(
                cdiv, "cdiv", dtype, ret_type, orig_op.gb_obj, orig_op.gb_name
            )
            cdiv._add(op)
        del binary.div
        # Add truediv which always points to floating point cdiv
        # We are effectively hacking cdiv to always return floating point values
        # If the inputs are FP32, we use DIV_FP32; use DIV_FP64 for all other input dtypes
        truediv = binary.truediv = BinaryOp("truediv")
        for dtype in binary.cdiv.types:
            float_type = "FP32" if dtype == "FP32" else "FP64"
            orig_op = binary.cdiv[float_type]
            op = TypedBuiltinBinaryOp(
                truediv,
                "truediv",
                dtype,
                binary.cdiv.types[float_type],
                orig_op.gb_obj,
                orig_op.gb_name,
            )
            truediv._add(op)
        # Add floordiv
        # cdiv truncates towards 0, while floordiv truncates towards -inf
        BinaryOp.register_new("floordiv", lambda x, y: x // y)

        # For aggregators
        BinaryOp.register_new("absfirst", lambda x, y: abs(x))

        def isclose(rel_tol=1e-7, abs_tol=0.0):
            def inner(x, y):
                return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)

            return inner

        BinaryOp.register_new("isclose", isclose, parameterized=True)

        # Update type information with sane coercion
        name_types = [
            # fmt: off
            (
                ("atan2", "copysign", "fmod", "hypot", "ldexp", "remainder"),
                (("BOOL", "INT8", "INT16", "UINT8", "UINT16"), "FP32"),
                (("INT32", "INT64", "UINT32", "UINT64"), "FP64"),
            ),
            (
                (
                    "firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1",
                    "secondj", "secondj1",),
                (
                    (
                        "BOOL", "FC32", "FC64", "FP32", "FP64", "INT8", "INT16",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "INT64",
                ),
            ),
            (
                ["lxnor"],
                (
                    (
                        "FP32", "FP64", "INT8", "INT16", "INT32", "INT64",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "BOOL",
                ),
            ),
            # fmt: on
        ]
        if _supports_complex:  # pragma: no branch
            name_types.append(
                (
                    ["cmplx"],
                    (("BOOL", "INT8", "INT16", "UINT8", "UINT16"), "FP32"),
                    (("INT32", "INT64", "UINT32", "UINT64"), "FP64"),
                )
            )
        for names, *types in name_types:
            for name in names:
                op = getattr(binary, name)
                for input_types, target_type in types:
                    typed_op = op._typed_ops[target_type]
                    output_type = op.types[target_type]
                    for dtype in input_types:
                        if dtype not in op.types:  # pragma: no branch
                            op.types[dtype] = output_type
                            op._typed_ops[dtype] = typed_op
                            op.coercions[dtype] = target_type

    def __init__(self, name, *, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._monoid = None

    __call__ = TypedBuiltinBinaryOp.__call__

    @property
    def monoid(self):
        if self._monoid is None and not self._anonymous:
            self._monoid = Monoid._find(self.name)
        return self._monoid


class Monoid(OpBase):
    __slots__ = "_binaryop"
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
        new_type_obj = cls(name, binaryop, anonymous=anonymous)
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
            # Skip complex dtypes for now, because they segfault!
            if type_ != ret_type and not explicit_identities or "FC" in type_.name:
                continue
            new_monoid = ffi_new("GrB_Monoid*")
            func = libget(f"GrB_Monoid_new_{type_.name}")
            zcast = ffi.cast(type_.c_type, identity)
            check_status_carg(
                func(new_monoid, binaryop[type_].gb_obj, zcast), "Monoid", new_monoid[0]
            )
            op = TypedUserMonoid(
                new_type_obj, name, type_.name, ret_type, new_monoid[0], binaryop[type_], identity
            )
            new_type_obj._add(op)
        return new_type_obj

    @classmethod
    def register_anonymous(cls, binaryop, identity, name=None):
        if type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedMonoid(name, binaryop, identity, anonymous=True)
        return cls._build(name, binaryop, identity, anonymous=True)

    @classmethod
    def register_new(cls, name, binaryop, identity):
        module, funcname = cls._remove_nesting(name)
        if type(binaryop) is ParameterizedBinaryOp:
            monoid = ParameterizedMonoid(name, binaryop, identity)
        else:
            monoid = cls._build(name, binaryop, identity)
        setattr(module, funcname, monoid)
        # Also save it to `grblas.op` if not yet defined
        module, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not hasattr(module, funcname):
            setattr(module, funcname, monoid)
        return monoid

    def __init__(self, name, binaryop=None, *, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._binaryop = binaryop
        if binaryop is not None:
            binaryop._monoid = self

    @property
    def binaryop(self):
        if self._binaryop is not None:
            return self._binaryop
        # Must be builtin
        return getattr(binary, self.name)

    @property
    def identities(self):
        return {dtype: val.identity for dtype, val in self._typed_ops.items()}

    @classmethod
    def _initialize(cls):
        super()._initialize()
        lor = monoid.lor._typed_ops["BOOL"]
        land = monoid.land._typed_ops["BOOL"]
        for cur_op, typed_op in [
            (monoid.max, lor),
            (monoid.min, land),
            # (monoid.plus, lor),  # two choices: lor, or plus[int]
            (monoid.times, land),
        ]:
            if "BOOL" not in cur_op.types:  # pragma: no branch
                cur_op.types["BOOL"] = "BOOL"
                cur_op.coercions["BOOL"] = "BOOL"
                cur_op._typed_ops["BOOL"] = typed_op

        for cur_op in (monoid.lor, monoid.land, monoid.lxnor, monoid.lxor):
            bool_op = cur_op._typed_ops["BOOL"]
            for dtype in (
                "FP32",
                "FP64",
                "INT8",
                "INT16",
                "INT32",
                "INT64",
                "UINT8",
                "UINT16",
                "UINT32",
                "UINT64",
            ):
                if dtype in cur_op.types:  # pragma: no cover
                    continue
                cur_op.types[dtype] = "BOOL"
                cur_op.coercions[dtype] = "BOOL"
                cur_op._typed_ops[dtype] = bool_op

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
        for binary_in, binary_func in binaryop._typed_ops.items():
            binary_out = binary_func.return_type
            # Unfortunately, we can't have user-defined monoids over bools yet
            # because numba can't compile correctly.
            if binary_out not in monoid.types:
                continue
            binary_out = lookup_dtype(binary_out)
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

    @classmethod
    def register_anonymous(cls, monoid, binaryop, name=None):
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            return ParameterizedSemiring(name, monoid, binaryop, anonymous=True)
        return cls._build(name, monoid, binaryop, anonymous=True)

    @classmethod
    def register_new(cls, name, monoid, binaryop):
        module, funcname = cls._remove_nesting(name)
        if type(monoid) is ParameterizedMonoid or type(binaryop) is ParameterizedBinaryOp:
            semiring = ParameterizedSemiring(name, monoid, binaryop)
        else:
            semiring = cls._build(name, monoid, binaryop)
        setattr(module, funcname, semiring)
        # Also save it to `grblas.op` if not yet defined
        module, funcname = cls._remove_nesting(name, module=op, modname="op", strict=False)
        if not hasattr(module, funcname):
            setattr(module, funcname, semiring)
        return semiring

    @classmethod
    def _initialize(cls):
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
            delattr(semiring, orig_name)
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
            cls.register_new(f"{orig_name[:-3]}truediv", orig.monoid, binary.truediv)
            cls.register_new(f"{orig_name[:-3]}floordiv", orig.monoid, binary.floordiv)
        # For aggregators
        cls.register_new("plus_pow", monoid.plus, binary.pow)
        cls.register_new("plus_absfirst", monoid.plus, binary.absfirst)
        cls.register_new("max_absfirst", monoid.max, binary.absfirst)

        # Update type information with sane coercion
        for lname in ("any", "eq", "land", "lor", "lxnor", "lxor"):
            target_name = f"{lname}_ne"
            source_name = f"{lname}_lxor"
            if not hasattr(semiring, target_name):
                continue
            target_op = getattr(semiring, target_name)
            if "BOOL" not in target_op.types:  # pragma: no branch
                source_op = getattr(semiring, source_name)
                typed_op = source_op._typed_ops["BOOL"]
                target_op.types["BOOL"] = "BOOL"
                target_op._typed_ops["BOOL"] = typed_op
                target_op.coercions[dtype] = "BOOL"

        for lnames, rnames, *types in (
            # fmt: off
            (
                ("any", "max", "min", "plus", "times"),
                (
                    "firsti", "firsti1", "firstj", "firstj1",
                    "secondi", "secondi1", "secondj", "secondj1",
                ),
                (
                    (
                        "BOOL", "FC32", "FC64", "FP32", "FP64", "INT8", "INT16",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "INT64",
                ),
            ),
            (
                ("eq", "land", "lor", "lxnor", "lxor"),
                ("first", "pair", "second"),
                # TODO: check if FC coercion works here
                (
                    (
                        "FC32", "FC64", "FP32", "FP64", "INT8", "INT16", "INT32", "INT64",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "BOOL",
                ),
            ),
            (
                ("band", "bor", "bxnor", "bxor"),
                ("band", "bor", "bxnor", "bxor"),
                (["INT8"], "UINT16"),
                (["INT16"], "UINT32"),
                (["INT32"], "UINT64"),
                (["INT64"], "UINT64"),
            ),
            (
                ("any", "eq", "land", "lor", "lxnor", "lxor"),
                ("eq", "land", "lor", "lxnor", "lxor", "ne"),
                (
                    (
                        "FP32", "FP64", "INT8", "INT16", "INT32", "INT64",
                        "UINT8", "UINT16", "UINT32", "UINT64",
                    ),
                    "BOOL",
                ),
            ),
            # fmt: on
        ):
            for left, right in itertools.product(lnames, rnames):
                name = f"{left}_{right}"
                if not hasattr(semiring, name):
                    continue
                op = getattr(semiring, name)
                for input_types, target_type in types:
                    typed_op = op._typed_ops[target_type]
                    output_type = op.types[target_type]
                    for dtype in input_types:
                        if dtype not in op.types:
                            op.types[dtype] = output_type
                            op._typed_ops[dtype] = typed_op
                            op.coercions[dtype] = target_type

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
            op = getattr(semiring, opname)
            target = getattr(semiring, targetname)
            if "BOOL" in op.types or "BOOL" not in target.types:  # pragma: no cover
                continue
            op.types["BOOL"] = target.types["BOOL"]
            op._typed_ops["BOOL"] = target._typed_ops["BOOL"]
            op.coercions["BOOL"] = "BOOL"

    def __init__(self, name, monoid=None, binaryop=None, *, anonymous=False):
        super().__init__(name, anonymous=anonymous)
        self._monoid = monoid
        self._binaryop = binaryop

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

    __call__ = TypedBuiltinSemiring.__call__


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
    elif isinstance(op, Aggregator):
        return op[dtype]
    elif isinstance(op, TypedAggregator):
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


def get_semiring(monoid, binaryop):
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
        rv = Semiring.register_anonymous(monoid, binaryop)
    else:
        name = f"{monoid.name}_{binaryop.name}".replace(".", "_")
        rv = getattr(semiring, name, None)
        if rv is None:
            rv = Semiring.register_new(name, monoid, binaryop)
        elif rv.monoid is not monoid or rv.binaryop is not binaryop:  # pragma: no cover
            # It's not the object we expect (can this happen?)
            rv = Semiring.register_anonymous(monoid, binaryop)
    if binary_type is not None:
        return rv[binary_type]
    elif monoid_type is not None:
        return rv[monoid_type]
    else:
        return rv


# Now initialize all the things!
UnaryOp._initialize()
BinaryOp._initialize()
Monoid._initialize()
Semiring._initialize()

from .agg import Aggregator, TypedAggregator  # noqa isort:skip

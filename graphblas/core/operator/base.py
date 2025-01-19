from functools import lru_cache
from operator import getitem
from types import BuiltinFunctionType, ModuleType

from ... import _STANDARD_OPERATOR_NAMES, backend, op
from ...dtypes import BOOL, INT8, UINT64, _supports_complex, lookup_dtype
from .. import _has_numba, _supports_udfs, lib
from ..expr import InfixExprBase
from ..utils import output_type

if _has_numba:
    import numba
    from numba import NumbaError
else:
    NumbaError = TypeError

UNKNOWN_OPCLASS = "UnknownOpClass"

# These now live as e.g. `gb.unary.ss.positioni`
# Deprecations such as `gb.unary.positioni` will be removed in 2023.9.0 or later.
_SS_OPERATORS = {
    # unary
    "erf",  # scipy.special.erf
    "erfc",  # scipy.special.erfc
    "frexpe",  # np.frexp[1]
    "frexpx",  # np.frexp[0]
    "lgamma",  # scipy.special.loggamma
    "tgamma",  # scipy.special.gamma
    # Positional
    # unary
    "positioni",
    "positioni1",
    "positionj",
    "positionj1",
    # binary
    "firsti",
    "firsti1",
    "firstj",
    "firstj1",
    "secondi",
    "secondi1",
    "secondj",
    "secondj1",
    # semiring
    "any_firsti",
    "any_firsti1",
    "any_firstj",
    "any_firstj1",
    "any_secondi",
    "any_secondi1",
    "any_secondj",
    "any_secondj1",
    "max_firsti",
    "max_firsti1",
    "max_firstj",
    "max_firstj1",
    "max_secondi",
    "max_secondi1",
    "max_secondj",
    "max_secondj1",
    "min_firsti",
    "min_firsti1",
    "min_firstj",
    "min_firstj1",
    "min_secondi",
    "min_secondi1",
    "min_secondj",
    "min_secondj1",
    "plus_firsti",
    "plus_firsti1",
    "plus_firstj",
    "plus_firstj1",
    "plus_secondi",
    "plus_secondi1",
    "plus_secondj",
    "plus_secondj1",
    "times_firsti",
    "times_firsti1",
    "times_firstj",
    "times_firstj1",
    "times_secondi",
    "times_secondi1",
    "times_secondj",
    "times_secondj1",
}


def _hasop(module, name):
    return (
        name in module.__dict__
        or name in module._delayed
        or name in getattr(module, "_deprecated", ())
    )


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
            return getattr(left.left, f"_{left.method_name}")(
                left.right, op, is_infix=True, **kwargs
            )
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
    from ..matrix import Matrix, TransposedMatrix
    from ..vector import Vector

    if (left_type := output_type(left)) in {Vector, Matrix, TransposedMatrix}:
        if thunk is not None:
            return left.select(op, thunk=thunk, **kwargs)
        return left.apply(op, right=right, **kwargs)
    if (right_type := output_type(right)) in {Vector, Matrix, TransposedMatrix}:
        return right.apply(op, left=left, **kwargs)

    from ..scalar import Scalar, _as_scalar

    if left_type is Scalar:
        if thunk is not None:
            return left.select(op, thunk=thunk, **kwargs)
        return left.apply(op, right=right, **kwargs)
    if right_type is Scalar:
        return right.apply(op, left=left, **kwargs)
    try:
        left_scalar = _as_scalar(left, is_cscalar=False)
    except Exception:
        pass
    else:
        if thunk is not None:
            return left_scalar.select(op, thunk=thunk, **kwargs)
        return left_scalar.apply(op, right=right, **kwargs)
    raise TypeError(
        f"Bad types when calling {op!r}.  Got types: {type(left)}, {type(right)}.\n"
        "Expected an infix expression or an apply with a Vector or Matrix and a scalar:\n"
        f"    - {op!r}(A & B)\n"
        f"    - {op!r}(A, 1)\n"
        f"    - {op!r}(1, A)"
    )


if _has_numba:

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
        exec(text, d)  # pylint: disable=exec-used
        return d["wrapper"], wrapper_sig


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
        classname = classname.removesuffix("op")
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
        return (getitem, (self.parent, (self.type, self._type2)))


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
        raise NotImplementedError


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
            from .utils import get_typed_op

            dtype1, dtype2 = type_
            dtype1 = lookup_dtype(dtype1)
            dtype2 = lookup_dtype(dtype2)
            return get_typed_op(self, dtype1, dtype2)
        if not self._is_udt:
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

    def _add(self, op, *, is_jit=False):
        if is_jit:
            if hasattr(op, "type2") or hasattr(op, "thunk_type"):
                dtypes = (op.type, op._type2)
            else:
                dtypes = op.type
            self.types[dtypes] = op.return_type  # This is a different use of .types
            self._udt_types[dtypes] = op.return_type
            self._udt_ops[dtypes] = op
        else:
            self._typed_ops[op.type] = op
            self.types[op.type] = op.return_type

    def __delitem__(self, type_):
        type_ = lookup_dtype(type_)
        del self._typed_ops[type_]
        del self.types[type_]

    def __contains__(self, type_):
        try:
            self[type_]
        except (TypeError, KeyError, NumbaError):
            return False
        return True

    @classmethod
    def _remove_nesting(cls, funcname, *, module=None, modname=None, strict=True):
        if module is None:
            module = cls._module
        if modname is None:
            modname = cls._modname
        if "." not in funcname:
            if strict and _hasop(module, funcname):
                raise AttributeError(f"{modname}.{funcname} is already defined")
        else:
            path, funcname = funcname.rsplit(".", 1)
            for folder in path.split("."):
                if not _hasop(module, folder):
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
            if attr in getattr(rv, "_deprecated", ()):
                rv = rv._deprecated[attr]
            else:
                rv = getattr(rv, attr, None)
            if rv is None:
                break
        return rv

    @classmethod
    def _initialize(cls, include_in_ops=True):
        """Initialize operators for this operator type.

        include_in_ops determines whether the operators are included in the
        ``gb.ops`` namespace in addition to the defined module.
        """
        if cls._initialized:  # pragma: no cover (safety)
            return
        # Read in the parse configs
        trim_from_front = cls._parse_config.get("trim_from_front", 0)
        delete_exact = cls._parse_config.get("delete_exact")
        num_underscores = cls._parse_config["num_underscores"]

        for re_str, return_prefix in [
            ("re_exprs", None),
            ("re_exprs_return_bool", "BOOL"),
            ("re_exprs_return_float", "FP"),
            ("re_exprs_return_complex", "FC"),
        ]:
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
                        if not _hasop(cls._module, name):
                            if backend == "suitesparse" and name in _SS_OPERATORS:
                                fullname = f"ss.{name}"
                            else:
                                fullname = name
                            if cls._positional is None:
                                obj = cls(fullname)
                            else:
                                obj = cls(fullname, is_positional=name in cls._positional)
                            if name in _SS_OPERATORS:
                                if backend == "suitesparse":
                                    setattr(cls._module.ss, name, obj)
                                cls._module._deprecated[name] = obj
                                if include_in_ops and not _hasop(op, name):  # pragma: no branch
                                    op._deprecated[name] = obj
                                    if backend == "suitesparse":
                                        setattr(op.ss, name, obj)
                            else:
                                setattr(cls._module, name, obj)
                                if include_in_ops and not _hasop(op, name):
                                    setattr(op, name, obj)
                            _STANDARD_OPERATOR_NAMES.add(f"{cls._modname}.{fullname}")
                        elif name in _SS_OPERATORS:
                            obj = cls._module._deprecated[name]
                        else:
                            obj = getattr(cls._module, name)
                        gb_obj = getattr(lib, varname)
                        # Determine return type
                        if return_prefix == "BOOL":
                            return_type = BOOL
                            if type_ is None:
                                type_ = BOOL
                        else:
                            if type_ is None:  # pragma: no cover (safety)
                                raise TypeError(f"Unable to determine return type for {varname}")
                            if return_prefix is None:
                                return_type = type_
                            else:
                                # Grab the number of bits from type_
                                num_bits = type_[-2:]
                                if num_bits not in {"32", "64"}:  # pragma: no cover (safety)
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
        if (rv := cls._find(name)) is not None:
            return rv  # Should we verify this is what the user expects?
        return cls.register_new(name, *args)

    @classmethod
    def _check_supports_udf(cls, method_name):
        if not _supports_udfs:
            raise RuntimeError(
                f"{cls.__name__}.{method_name}(...) unavailable; install numba for UDF support"
            )


_builtin_to_op = {}  # Populated in .utils


def find_opclass(gb_op):
    if isinstance(gb_op, OpBase):
        opclass = type(gb_op).__name__
    elif isinstance(gb_op, TypedOpBase):
        opclass = gb_op.opclass
    elif isinstance(gb_op, ParameterizedUdf):
        gb_op = gb_op()  # Use default parameters of parameterized UDFs
        gb_op, opclass = find_opclass(gb_op)
    elif isinstance(gb_op, BuiltinFunctionType) and gb_op in _builtin_to_op:
        gb_op, opclass = find_opclass(_builtin_to_op[gb_op])
    else:
        opclass = UNKNOWN_OPCLASS
    return gb_op, opclass

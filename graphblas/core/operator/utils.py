from types import BuiltinFunctionType, FunctionType, ModuleType

from ... import backend, binary, config, indexunary, monoid, op, select, semiring, unary
from ...dtypes import UINT64, lookup_dtype, unify
from ..expr import InfixExprBase
from .base import (
    _SS_OPERATORS,
    OpBase,
    OpPath,
    ParameterizedUdf,
    TypedOpBase,
    _builtin_to_op,
    _hasop,
    find_opclass,
)
from .binary import BinaryOp
from .indexunary import IndexUnaryOp
from .monoid import Monoid
from .select import SelectOp
from .semiring import Semiring
from .unary import UnaryOp

# Now initialize all the things!
try:
    UnaryOp._initialize()
    IndexUnaryOp._initialize()
    SelectOp._initialize()
    BinaryOp._initialize()
    Monoid._initialize()
    Semiring._initialize()
except Exception:  # pragma: no cover (debug)
    # Exceptions here can often get ignored by Python
    import traceback

    traceback.print_exc()
    raise


def get_typed_op(op, dtype, dtype2=None, *, is_left_scalar=False, is_right_scalar=False, kind=None):
    if isinstance(op, OpBase):
        # UDTs always get compiled
        if op._is_udt:
            return op._compile_udt(dtype, dtype2)
        # Single dtype is simple lookup
        if dtype2 is None:
            return op[dtype]
        # Handle special cases such as first and second (may have UDTs)
        if op._custom_dtype is not None and (rv := op._custom_dtype(op, dtype, dtype2)) is not None:
            return rv
        # Generic case: try to unify the two dtypes
        try:
            return op[
                unify(dtype, dtype2, is_left_scalar=is_left_scalar, is_right_scalar=is_right_scalar)
            ]
        except (TypeError, AttributeError):
            # Failure to unify implies a dtype is UDT; some builtin operators can handle UDTs
            if op.is_positional:
                return op[UINT64]
            if op._udt_types is None:
                raise
            return op._compile_udt(dtype, dtype2)
    if isinstance(op, ParameterizedUdf):
        op = op()  # Use default parameters of parameterized UDFs
        return get_typed_op(
            op,
            dtype,
            dtype2,
            is_left_scalar=is_left_scalar,
            is_right_scalar=is_right_scalar,
            kind=kind,
        )
    if isinstance(op, TypedOpBase):
        return op

    from .agg import Aggregator, TypedAggregator

    if isinstance(op, Aggregator):
        # agg._any_dtype basically serves the same purpose as op._custom_dtype
        if op._any_dtype is not None and op._any_dtype is not True:
            return op[op._any_dtype]
        return op[dtype]
    if isinstance(op, TypedAggregator):
        return op
    if isinstance(op, str):
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
    if isinstance(op, FunctionType):
        if kind == "unary":
            op = UnaryOp.register_anonymous(op, is_udt=True)
            return op._compile_udt(dtype, dtype2)
        if kind.startswith("binary"):
            op = BinaryOp.register_anonymous(op, is_udt=True)
            return op._compile_udt(dtype, dtype2)
    if isinstance(op, BuiltinFunctionType) and op in _builtin_to_op:
        return get_typed_op(
            _builtin_to_op[op],
            dtype,
            dtype2,
            is_left_scalar=is_left_scalar,
            is_right_scalar=is_right_scalar,
            kind=kind,
        )
    raise TypeError(f"Unable to get typed operator from object with type {type(op)}")


def _get_typed_op_from_exprs(op, left, right, *, kind=None):
    if isinstance(left, InfixExprBase):
        left_op = _get_typed_op_from_exprs(op, left.left, left.right, kind=kind)
        left_dtype = left_op.type
    else:
        left_op = None
        left_dtype = left.dtype
    if isinstance(right, InfixExprBase):
        right_op = _get_typed_op_from_exprs(op, right.left, right.right, kind=kind)
        if right_op is left_op:
            return right_op
        right_dtype = right_op.type2
    else:
        right_dtype = right.dtype
    return get_typed_op(
        op,
        left_dtype,
        right_dtype,
        is_left_scalar=left._is_scalar,
        is_right_scalar=right._is_scalar,
        kind=kind,
    )


def get_semiring(monoid, binaryop, name=None):
    """Get or create a Semiring object from a monoid and binaryop.

    If either are typed, then the returned semiring will also be typed.

    See Also
    --------
    semiring.register_anonymous
    semiring.register_new
    semiring.from_string

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
            or backend == "suitesparse"
            and binary_name in _SS_OPERATORS
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
            else getattr(module, "_deprecated", {}).get(funcname)
        )
        if rv is None and name != canonical_name:
            module, funcname = Semiring._remove_nesting(name, strict=False)
            rv = (
                getattr(module, funcname)
                if funcname in module.__dict__ or funcname in module._delayed
                else getattr(module, "_deprecated", {}).get(funcname)
            )
        if rv is None:
            rv = Semiring.register_new(canonical_name, monoid, binaryop)
        elif rv.monoid is not monoid or rv.binaryop is not binaryop:  # pragma: no cover
            # It's not the object we expect (can this happen?)
            rv = Semiring.register_anonymous(monoid, binaryop, name=name)
        if name != canonical_name:
            module, funcname = Semiring._remove_nesting(name, strict=False)
            if not _hasop(module, funcname):  # pragma: no branch (safety)
                setattr(module, funcname, rv)

    if binary_type is not None:
        return rv[binary_type]
    if monoid_type is not None:
        return rv[monoid_type]
    return rv


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
semiring.get_semiring = get_semiring

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

_builtin_to_op.update(
    {
        abs: unary.abs,
        max: binary.max,
        min: binary.min,
        # Maybe someday: all, any, pow, sum
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
    if dtype:
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
        if isinstance(op, str):
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


from ... import agg  # noqa: E402 isort:skip

agg.from_string = aggregator_from_string

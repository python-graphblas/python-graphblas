from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.binary import BinaryOp, TypedUserBinaryOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitBinaryOp(TypedOpBase):
    __slots__ = "_monoid", "_jit_c_definition"
    opclass = "BinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._monoid = None
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    monoid = TypedUserBinaryOp.monoid
    commutes_to = TypedUserBinaryOp.commutes_to
    _semiring_commutes_to = TypedUserBinaryOp._semiring_commutes_to
    is_commutative = TypedUserBinaryOp.is_commutative
    type2 = TypedUserBinaryOp.type2
    __call__ = TypedUserBinaryOp.__call__


def register_new(name, jit_c_definition, left_type, right_type, ret_type):
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.binary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    left_type = lookup_dtype(left_type)
    right_type = lookup_dtype(right_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = BinaryOp._remove_nesting(name)

    rv = BinaryOp(name)
    gb_obj = ffi_new("GrB_BinaryOp*")
    check_status_carg(
        lib.GxB_BinaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            left_type._carg,
            right_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "BinaryOp",
        gb_obj[0],
    )
    op = TypedJitBinaryOp(
        rv, funcname, left_type, ret_type, gb_obj[0], jit_c_definition, dtype2=right_type
    )
    rv._add(op)
    setattr(module, funcname, rv)
    return rv

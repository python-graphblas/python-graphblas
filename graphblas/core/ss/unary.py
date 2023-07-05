from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.unary import TypedUserUnaryOp, UnaryOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitUnaryOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "UnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition):
        super().__init__(parent, name, type_, return_type, gb_obj, name)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    __call__ = TypedUserUnaryOp.__call__


def register_new(name, jit_c_definition, input_type, ret_type):
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.unary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    input_type = lookup_dtype(input_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = UnaryOp._remove_nesting(name)

    rv = UnaryOp(name)
    gb_obj = ffi_new("GrB_UnaryOp*")
    check_status_carg(
        lib.GxB_UnaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            input_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "UnaryOp",
        gb_obj[0],
    )
    op = TypedJitUnaryOp(rv, funcname, input_type, ret_type, gb_obj[0], jit_c_definition)
    rv._add(op)
    setattr(module, funcname, rv)
    return rv

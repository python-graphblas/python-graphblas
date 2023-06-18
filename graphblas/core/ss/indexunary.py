from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.indexunary import IndexUnaryOp, TypedUserIndexUnaryOp

ffi_new = ffi.new


class TypedJitIndexUnaryOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "IndexUnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    __call__ = TypedUserIndexUnaryOp.__call__


def register_new(name, jit_c_definition, input_type, thunk_type, ret_type):
    if backend != "suitesparse":
        raise RuntimeError(
            "`gb.indexunary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    input_type = lookup_dtype(input_type)
    thunk_type = lookup_dtype(thunk_type)
    ret_type = lookup_dtype(ret_type)
    module, funcname = IndexUnaryOp._remove_nesting(name)

    rv = IndexUnaryOp(name)
    gb_obj = ffi_new("GrB_IndexUnaryOp*")
    check_status_carg(
        lib.GxB_IndexUnaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            input_type._carg,
            thunk_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "IndexUnaryOp",
        gb_obj,
    )
    op = TypedJitIndexUnaryOp(
        rv, funcname, input_type, ret_type, gb_obj[0], jit_c_definition, dtype2=thunk_type
    )
    rv._add(op)
    return rv

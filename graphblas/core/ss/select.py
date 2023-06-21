from ... import backend, indexunary
from ...dtypes import BOOL, lookup_dtype
from .. import ffi
from ..operator.base import TypedOpBase
from ..operator.select import SelectOp, TypedUserSelectOp

ffi_new = ffi.new


class TypedJitSelectOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "SelectOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    __call__ = TypedUserSelectOp.__call__


def register_new(name, jit_c_definition, input_type, thunk_type):
    if backend != "suitesparse":
        raise RuntimeError(
            "`gb.select.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    input_type = lookup_dtype(input_type)
    thunk_type = lookup_dtype(thunk_type)
    if not name.startswith("ss."):
        name = f"ss.{name}"
    # Register to both `gb.indexunary.ss` and `gb.select.ss.`
    indexunary.ss.register_new(name, jit_c_definition, input_type, thunk_type, BOOL)
    module, funcname = SelectOp._remove_nesting(name, strict=False)
    return getattr(module, funcname)

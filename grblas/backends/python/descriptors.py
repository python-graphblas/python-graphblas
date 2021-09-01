from .base import BasePointer
from .context import handle_panic
from .exceptions import GrB_Info, return_error


class GrB_Desc_Field:
    GrB_OUTP = object()
    GrB_INP0 = object()
    GrB_INP1 = object()
    GrB_MASK = object()


class GrB_Desc_Value:
    GrB_STRUCTURE = object()
    GrB_COMP = object()
    GrB_TRAN = object()
    GrB_REPLACE = object()


class DescriptorPtr(BasePointer):
    def set_descriptor(self, descriptor):
        self.instance = descriptor


class Descriptor:
    trans0 = False
    trans1 = False
    clear_output = False
    mask_comp = False
    mask_struct = False


@handle_panic
def Descriptor_new(desc_ptr: DescriptorPtr):
    desc = Descriptor()
    desc_ptr.set_descriptor(desc)
    return GrB_Info.GrB_SUCCESS


@handle_panic
def Descriptor_set(desc: Descriptor, field, value):
    if field is GrB_Desc_Field.GrB_OUTP:
        if value is GrB_Desc_Value.GrB_REPLACE:
            desc.clear_output = True
        else:
            return_error(GrB_Info.GrB_INVALID_VALUE, "Invalid value for GrB_OUTP")
    elif field is GrB_Desc_Field.GrB_INP0:
        if value is GrB_Desc_Value.GrB_TRAN:
            desc.trans0 = True
        else:
            return_error(GrB_Info.GrB_INVALID_VALUE, "Invalid value for GrB_INP0")
    elif field is GrB_Desc_Field.GrB_INP1:
        if value is GrB_Desc_Value.GrB_TRAN:
            desc.trans1 = True
        else:
            return_error(GrB_Info.GrB_INVALID_VALUE, "Invalid value for GrB_INP1")
    elif field is GrB_Desc_Field.GrB_MASK:
        if value is GrB_Desc_Value.GrB_COMP:
            desc.mask_comp = True
        elif value is GrB_Desc_Value.GrB_STRUCTURE:
            desc.mask_struct = True
        else:
            return_error(GrB_Info.GrB_INVALID_VALUE, "Invalid value for GrB_MASK")
    else:
        return_error(GrB_Info.GrB_INVALID_VALUE, "Invalid field")

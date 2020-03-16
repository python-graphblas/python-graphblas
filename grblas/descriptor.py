from . import lib, ffi
from .exceptions import check_status

_desc_map = {}


# TODO: this will need to update for GraphBLAS 1.3 mask options
def build(*, output_replace=False, mask_complement=False,
          transpose_first=False, transpose_second=False):
    key = (mask_complement, output_replace, transpose_first, transpose_second)
    if key not in _desc_map:
        desc = ffi.new('GrB_Descriptor*')
        if not any(key):
            # Default descriptor stays a NULL pointer
            desc[0] = ffi.NULL
        else:
            check_status(lib.GrB_Descriptor_new(desc))
            for cond, field, val in [(output_replace, lib.GrB_OUTP, lib.GrB_REPLACE),
                                     (mask_complement, lib.GrB_MASK, lib.GrB_SCMP),
                                     (transpose_first, lib.GrB_INP0, lib.GrB_TRAN),
                                     (transpose_second, lib.GrB_INP1, lib.GrB_TRAN)]:
                if cond:
                    check_status(lib.GrB_Descriptor_set(desc[0], field, val))
        _desc_map[key] = desc[0]
    return _desc_map[key]

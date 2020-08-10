from . import ffi, lib
from .exceptions import check_status

NULL = ffi.NULL
_desc_map = {
    # OUTP, MSK_COMP, MSK_STRUCT, TRANS0, TRANS1
    (False, False, False, False, False): NULL,
    (False, False, False, False, True): lib.GrB_DESC_T1,
    (False, False, False, True, False): lib.GrB_DESC_T0,
    (False, False, False, True, True): lib.GrB_DESC_T0T1,
    (False, True, False, False, False): lib.GrB_DESC_C,
    (False, False, True, False, False): lib.GrB_DESC_S,
    (False, True, False, False, True): lib.GrB_DESC_CT1,
    (False, False, True, False, True): lib.GrB_DESC_ST1,
    (False, True, False, True, False): lib.GrB_DESC_CT0,
    (False, False, True, True, False): lib.GrB_DESC_ST0,
    (False, True, False, True, True): lib.GrB_DESC_CT0T1,
    (False, False, True, True, True): lib.GrB_DESC_ST0T1,
    (False, True, True, False, False): lib.GrB_DESC_SC,
    (False, True, True, False, True): lib.GrB_DESC_SCT1,
    (False, True, True, True, False): lib.GrB_DESC_SCT0,
    (False, True, True, True, True): lib.GrB_DESC_SCT0T1,
    (True, False, False, False, False): lib.GrB_DESC_R,
    (True, False, False, False, True): lib.GrB_DESC_RT1,
    (True, False, False, True, False): lib.GrB_DESC_RT0,
    (True, False, False, True, True): lib.GrB_DESC_RT0T1,
    (True, True, False, False, False): lib.GrB_DESC_RC,
    (True, False, True, False, False): lib.GrB_DESC_RS,
    (True, True, False, False, True): lib.GrB_DESC_RCT1,
    (True, False, True, False, True): lib.GrB_DESC_RST1,
    (True, True, False, True, False): lib.GrB_DESC_RCT0,
    (True, False, True, True, False): lib.GrB_DESC_RST0,
    (True, True, False, True, True): lib.GrB_DESC_RCT0T1,
    (True, False, True, True, True): lib.GrB_DESC_RST0T1,
    (True, True, True, False, False): lib.GrB_DESC_RSC,
    (True, True, True, False, True): lib.GrB_DESC_RSCT1,
    (True, True, True, True, False): lib.GrB_DESC_RSCT0,
    (True, True, True, True, True): lib.GrB_DESC_RSCT0T1,
}


def lookup(
    *,
    output_replace=False,
    mask_complement=False,
    mask_structure=False,
    transpose_first=False,
    transpose_second=False
):
    key = (
        output_replace,
        mask_complement,
        mask_structure,
        transpose_first,
        transpose_second,
    )
    if key not in _desc_map:  # pragma: no cover
        # We currently don't need this block of code!
        # All 32 possible descriptors are currently already added to _desc_map.
        # Nevertheless, this code may be useful some day.
        desc = ffi.new("GrB_Descriptor*")
        if not any(key):
            # Default descriptor stays a NULL pointer
            desc[0] = NULL
        else:
            check_status(lib.GrB_Descriptor_new(desc))
            for cond, field, val in [
                (output_replace, lib.GrB_OUTP, lib.GrB_REPLACE),
                (mask_complement, lib.GrB_MASK, lib.GrB_COMP),
                (mask_structure, lib.GrB_MASK, lib.GrB_STRUCTURE),
                (transpose_first, lib.GrB_INP0, lib.GrB_TRAN),
                (transpose_second, lib.GrB_INP1, lib.GrB_TRAN),
            ]:
                if cond:
                    check_status(lib.GrB_Descriptor_set(desc[0], field, val))
        _desc_map[key] = desc[0]
    return _desc_map[key]

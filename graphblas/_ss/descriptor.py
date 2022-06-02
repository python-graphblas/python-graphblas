from graphblas import ffi, lib
from graphblas.descriptor import Descriptor, _desc_map
from graphblas.exceptions import check_status

str_to_compression = {
    "none": lib.GxB_COMPRESSION_NONE,
    "default": lib.GxB_COMPRESSION_DEFAULT,
    "lz4": lib.GxB_COMPRESSION_LZ4,
    "lz4hc": lib.GxB_COMPRESSION_LZ4HC,
}


def get_nthreads_descriptor(nthreads, _cache=True):
    nthreads = max(0, int(nthreads))
    key = ("nthreads", nthreads)
    if _cache and key in _desc_map:
        return _desc_map[key]
    desc_obj = ffi.new("GrB_Descriptor*")
    lib.GrB_Descriptor_new(desc_obj)
    desc = Descriptor(desc_obj[0], f"nthreads{nthreads}")
    check_status(
        lib.GxB_Desc_set(
            desc._carg,
            lib.GxB_NTHREADS,
            ffi.cast("GrB_Desc_Value", nthreads),
        ),
        desc,
    )
    if _cache:
        _desc_map[key] = desc
    return desc


def get_compression_descriptor(compression="default", level=None, nthreads=None):
    if compression is None:
        comp = str_to_compression["none"]
    elif not isinstance(compression, str) or compression.lower() not in str_to_compression:
        valid = ", ".join(sorted(map(repr, str_to_compression)))
        raise ValueError(f"compression argument should be one of {valid}")
    else:
        compression = compression.lower()
        comp = str_to_compression[compression]
    if level is not None:
        if compression != "lz4hc":
            raise TypeError('level argument is only valid when using "lz4hc" compression')
        level = int(level)
        if level < 1 or level > 9:
            raise ValueError(
                f"level argument should be an integer between 1 and 9 (got {level}).  "
                "1 is the fastest, 9 is the most compression (default is 9)."
            )
        comp += level
    if nthreads is not None:
        nthreads = max(0, int(nthreads))
        key = frozenset([("compression", comp), ("nthreads", nthreads)])
    else:
        key = frozenset([("compression", comp)])
    if key in _desc_map:
        return _desc_map[key]
    if nthreads is not None:
        desc = get_nthreads_descriptor(nthreads, _cache=False)
    else:
        desc_obj = ffi.new("GrB_Descriptor*")
        lib.GrB_Descriptor_new(desc_obj)
        desc = Descriptor(desc_obj[0], "")
    name = f"compression_{compression}{level or ''}"
    if nthreads and nthreads > 0:
        name += f"_nthreads{nthreads}"
    desc.name = name
    check_status(
        lib.GxB_Desc_set(
            desc._carg,
            lib.GxB_COMPRESSION,
            ffi.cast("GrB_Desc_Value", comp),
        ),
        desc,
    )
    _desc_map[key] = desc
    return desc

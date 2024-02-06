from ...exceptions import check_status, check_status_carg
from .. import ffi, lib
from ..descriptor import Descriptor
from . import _IS_SSGB7
from .config import BaseConfig

ffi_new = ffi.new

_str_to_compression = {
    "none": lib.GxB_COMPRESSION_NONE,
    "default": lib.GxB_COMPRESSION_DEFAULT,
    "lz4": lib.GxB_COMPRESSION_LZ4,
    "lz4hc": lib.GxB_COMPRESSION_LZ4HC,
    "zstd": lib.GxB_COMPRESSION_ZSTD,
}


# It would be great if we could make Descriptor class more like this
class _DescriptorConfig(BaseConfig):
    _get_function = "GxB_Desc_get"
    _set_function = "GxB_Desc_set"
    if not _IS_SSGB7:
        _context_keys = {"chunk", "gpu_id", "nthreads"}
    _options = {
        # GrB
        "output_replace": (lib.GrB_OUTP, "GrB_Desc_Value"),
        "mask_complement": (lib.GrB_MASK, "GrB_Desc_Value"),
        "mask_structure": (lib.GrB_MASK, "GrB_Desc_Value"),
        "transpose_first": (lib.GrB_INP0, "GrB_Desc_Value"),
        "transpose_second": (lib.GrB_INP1, "GrB_Desc_Value"),
        # GxB
        "axb_method": (lib.GxB_AxB_METHOD, "GrB_Desc_Value"),
        "sort": (lib.GxB_SORT, "int"),
        "secure_import": (lib.GxB_IMPORT, "int"),
    }
    if _IS_SSGB7:
        _options.update(
            {
                "nthreads": (lib.GxB_DESCRIPTOR_NTHREADS, "int"),
                "chunk": (lib.GxB_DESCRIPTOR_CHUNK, "double"),
            }
        )
    else:
        _options.update(
            {
                "chunk": (lib.GxB_CONTEXT_CHUNK, "double"),
                "gpu_id": (lib.GxB_CONTEXT_GPU_ID, "int"),
                "nthreads": (lib.GxB_CONTEXT_NTHREADS, "int"),
            }
        )
    _enumerations = {
        # GrB
        "output_replace": {
            True: lib.GrB_REPLACE,
            False: False,
        },
        "mask_complement": {
            True: lib.GrB_COMP,
            False: False,
        },
        "mask_structure": {
            True: lib.GrB_STRUCTURE,
            False: False,
        },
        "transpose_first": {
            True: lib.GrB_TRAN,
            False: False,
        },
        "transpose_second": {
            True: lib.GrB_TRAN,
            False: False,
        },
        # GxB
        "axb_method": {
            "gustavson": lib.GxB_AxB_GUSTAVSON,
            "dot": lib.GxB_AxB_DOT,
            "hash": lib.GxB_AxB_HASH,
            "saxpy": lib.GxB_AxB_SAXPY,
            "default": lib.GxB_DEFAULT,
        },
        "secure_import": {
            True: lib.GxB_SECURE_IMPORT,
            False: False,
        },
        "sort": {
            False: False,
            True: lib.GxB_SORT,
        },
    }
    _defaults = {
        # GrB
        "output_replace": False,
        "mask_complement": False,
        "mask_structure": False,
        "transpose_first": False,
        "transpose_second": False,
        # GxB
        "nthreads": 0,
        "chunk": 0,
        "axb_method": "default",
        "sort": False,
        "secure_import": False,
    }
    if not _IS_SSGB7:
        _defaults["gpu_id"] = -1

    def __init__(self):
        gb_obj = ffi_new("GrB_Descriptor*")
        check_status_carg(lib.GrB_Descriptor_new(gb_obj), "Descriptor", gb_obj[0])
        parent = Descriptor(gb_obj)
        initialized = self._initialized
        super().__init__(parent)
        if not initialized:
            # These are actually bitwise, but we treat them as boolean, add extra mappings for get
            self._enumerations["mask_complement"][lib.GrB_COMP | lib.GrB_STRUCTURE] = lib.GrB_COMP
            self._enumerations["mask_structure"][
                lib.GrB_COMP | lib.GrB_STRUCTURE
            ] = lib.GrB_STRUCTURE
            self._enumerations["mask_complement"][lib.GrB_STRUCTURE] = False
            self._enumerations["mask_structure"][lib.GrB_COMP] = False


def get_descriptor(**opts):
    """Create descriptor with SuiteSparse:GraphBLAS options.

    See SuiteSparse:GraphBLAS documentation for more details.

    Default descriptor parameters
    -----------------------------
    output_replace : bool, default False
    mask_complement : bool, default False
    mask_structure : bool, default False
    transpose_first : bool, default False
    transpose_second : bool: default False

    Extended descriptor parameters
    ------------------------------
    nthreads : int
        Maximum number of OpenMP threads to use. 0 or negative means no limit.
        If not set, use nthreads from global config ``gb.ss.config["nthreads"]``
    chunk : double
    axb_method : str, {"gustavson", "dot", "hash", "saxpy", "default"}
        A hint for which matrix multiply method to use
    sort : bool, default False
        A hint for whether methods may return a "jumbled" matrix
    secure_import : bool, default False
        Whether to trust the data for ``import`` and ``pack`` functions.
        When True, checks are performed to ensure input data is valid.
    compression : str, {"none", "default", "lz4", "lz4hc", "zstd"}
        Whether and how to compress the data for serialization.
        The default is "zstd" with ``compression_level=1``
    compression_level : int
        1-9 for "lz4hc" compression: (1) is the fastest, (9) is the most compact (default 1).
        1-19 for "zstd" compression: (1) is the fastest, (19) is the most compact (default 1).
        Ignored for other compression methods.

    Returns
    -------
    Descriptor or None

    """
    if not opts or all(val is False or val is None for val in opts.values()):
        return
    config = _DescriptorConfig()
    desc = config._parent
    if "compression" in opts or "compression_level" in opts:
        compression = opts.pop("compression", None)
        level = opts.pop("compression_level", None)
        _set_compression(desc, compression, level)
    if (nthreads := opts.pop("nthreads", None)) is not None:
        config["nthreads"] = max(0, nthreads)
    for key, val in opts.items():
        if val is False or val is None:
            continue
        try:
            config[key] = val
        except KeyError:
            if key in config._enumerations:
                values = {x for x in config._enumerations[key] if isinstance(x, (bool, str))}
                raise ValueError(
                    f"Invalid value for {key!r} descriptor option: {val!r}. "
                    f"Must be one of {', '.join(repr(x) for x in sorted(values))}"
                ) from None
            if key in config:  # pragma: no cover (safety)
                raise ValueError(f"Invalid value for {key!r} descriptor option") from None
            valid = set(config) | {"compression", "compression_level"}
            valid -= {
                "mask_complement",
                "mask_structure",
                "output_replace",
                "transpose_first",
                "transpose_second",
            }
            raise ValueError(
                f"Descriptor option {key!r} not understood with suitesparse backend. "
                f"Valid options: {', '.join(sorted(valid))}"
            ) from None
    return desc


def _set_compression(desc, compression, level):
    if compression is None:
        comp = _str_to_compression["none"]
    elif not isinstance(compression, str) or compression.lower() not in _str_to_compression:
        valid = ", ".join(sorted(map(repr, _str_to_compression)))
        raise ValueError(f"compression argument should be one of {valid}")
    else:
        compression = compression.lower()
        comp = _str_to_compression[compression]
    if level is not None:
        if compression not in {"lz4hc", "zstd"}:
            raise TypeError('level argument is only valid when using "lz4hc" compression')
        level = int(level)
        upper = 9 if compression == "lz4hc" else 19
        default = 9 if compression == "lz4hc" else 1
        if level < 1 or level > upper:
            raise ValueError(
                f"level argument should be an integer between 1 and {upper} (got {level}).  "
                f"1 is the fastest, {upper} is the most compression (default is {default})."
            )
        comp += level
    check_status(
        lib.GxB_Desc_set_INT32(
            desc._carg,
            lib.GxB_COMPRESSION,
            ffi.cast("int32_t", comp),
        ),
        desc,
    )
    return desc

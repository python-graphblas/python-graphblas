import numpy as np

from ... import backend, core, dtypes
from ...exceptions import check_status_carg
from .. import _has_numba, ffi, lib
from . import _IS_SSGB7

ffi_new = ffi.new
if _has_numba:
    import numba
    from cffi import FFI
    from numba.core.typing import cffi_utils

    jit_ffi = FFI()


def register_new(name, jit_c_definition, *, np_type=None):
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.dtypes.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    if not name.isidentifier():
        raise ValueError(f"`name` argument must be a valid Python identifier; got: {name!r}")
    if name in core.dtypes._registry or hasattr(dtypes.ss, name):
        raise ValueError(f"{name!r} name for dtype is unavailable")
    if len(name) > lib.GxB_MAX_NAME_LEN:
        raise ValueError(
            f"`name` argument is too large. Max size is {lib.GxB_MAX_NAME_LEN}; got {len(name)}"
        )
    if name not in jit_c_definition:
        raise ValueError("`name` argument must be same name as the typedef in `jit_c_definition`")
    if "struct" not in jit_c_definition:
        raise ValueError("Only struct typedefs are currently allowed for JIT dtypes")

    gb_obj = ffi.new("GrB_Type*")
    status = lib.GxB_Type_new(
        gb_obj, 0, ffi_new("char[]", name.encode()), ffi_new("char[]", jit_c_definition.encode())
    )
    check_status_carg(status, "Type", gb_obj[0])

    # Let SuiteSparse:GraphBLAS determine the size (we gave 0 as size above)
    size_ptr = ffi_new("size_t*")
    check_status_carg(lib.GxB_Type_size(size_ptr, gb_obj[0]), "Type", gb_obj[0])
    size = size_ptr[0]

    save_np_type = True
    if np_type is None and _has_numba and numba.__version__[:5] > "0.56.":
        jit_ffi.cdef(jit_c_definition)
        numba_type = cffi_utils.map_type(jit_ffi.typeof(name), use_record_dtype=True)
        np_type = numba_type.dtype
        if np_type.itemsize != size:  # pragma: no cover
            raise RuntimeError(
                "Size of compiled user-defined type does not match size of inferred numpy type: "
                f"{size} != {np_type.itemsize} != {size}.\n\n"
                f"UDT C definition: {jit_c_definition}\n"
                f"numpy dtype: {np_type}\n\n"
                "To get around this, you may pass `np_type=` keyword argument."
            )
    else:
        if np_type is not None:
            np_type = np.dtype(np_type)
        else:
            # Not an ideal numpy type, but minimally useful
            np_type = np.dtype((np.uint8, size))
            save_np_type = False
        if _has_numba:
            numba_type = numba.typeof(np_type).dtype
        else:
            numba_type = None

    # For now, let's use "opaque" unsigned bytes for the c type.
    rv = core.dtypes.DataType(name, gb_obj, None, f"uint8_t[{size}]", numba_type, np_type)
    core.dtypes._registry[gb_obj] = rv
    if save_np_type or np_type not in core.dtypes._registry:
        core.dtypes._registry[np_type] = rv
        if numba_type is not None and (save_np_type or numba_type not in core.dtypes._registry):
            core.dtypes._registry[numba_type] = rv
            core.dtypes._registry[numba_type.name] = rv
    setattr(dtypes.ss, name, rv)
    return rv

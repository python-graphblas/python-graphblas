from ... import backend, dtypes
from ...exceptions import check_status_carg
from .. import _has_numba, ffi, lib

ffi_new = ffi.new
if _has_numba:
    from cffi import FFI
    from numba.core.typing import cffi_utils

    jit_ffi = FFI()


def register_new(name, jit_c_definition):
    if backend != "suitesparse":
        raise RuntimeError(
            "`gb.dtypes.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if not name.isidentifier():
        raise ValueError(f"`name` argument must be a valid Python identifier; got: {name!r}")
    if name in dtypes._core._registry or hasattr(dtypes.ss, name):
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

    if _has_numba:
        jit_ffi.cdef(jit_c_definition)
        numba_type = cffi_utils.map_type(jit_ffi.typeof(name), use_record_dtype=True)
        np_type = numba_type.dtype
        if np_type.itemsize != size:  # pragma: no cover
            # TODO: Should we warn or raise?
            numba_type = np_type = None
    else:
        # Instead of None, should we make these e.g. np.dtype((np.uint8, size))`?
        numba_type = np_type = None

    # For now, let's use "opaque" unsigned bytes for the c type.
    rv = dtypes._core.DataType(name, gb_obj, None, f"uint8_t[{size}]", numba_type, np_type)
    dtypes._core._registry[gb_obj] = rv
    if _has_numba:
        dtypes._core._registry[np_type] = rv
        dtypes._core._registry[numba_type] = rv
        dtypes._core._registry[numba_type.name] = rv
    setattr(dtypes.ss, name, rv)
    return rv

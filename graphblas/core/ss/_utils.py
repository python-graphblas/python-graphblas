import itertools

from ...exceptions import _error_code_lookup
from .. import ffi, lib
from ..dtypes import _string_to_dtype


def _resolve_serialized_dtype(data_obj, data_nbytes, obj_kind):
    """Resolve the dtype of a serialized Matrix/Vector buffer.

    ``GxB_deserialize_type_name`` returns the type's short name (a built-in
    like ``'INT64'`` or, for a UDT, its ``GxB_JIT_C_NAME``). For UDTs the
    short name may not match any Python-side registration (e.g., when an
    anonymous UDT was registered by-shape on the producer side). Fall back
    to ``GrB_NAME`` (the numpy repr that ``serialize`` writes into the
    object) when the short name fails to resolve.

    ``obj_kind`` is ``"Matrix"`` or ``"Vector"`` and only shows up in error
    messages.
    """
    cname = ffi.new(f"char[{lib.GxB_MAX_NAME_LEN}]")
    info = lib.GxB_deserialize_type_name(cname, data_obj, data_nbytes)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info](f"{obj_kind} deserialize failed to get the dtype name")
    dtype_name = b"".join(itertools.takewhile(b"\x00".__ne__, cname)).decode()
    orig_error = None
    if dtype_name:
        try:
            return _string_to_dtype(dtype_name)
        except (ValueError, TypeError, SyntaxError) as exc:
            orig_error = exc
    if hasattr(lib, "GxB_Serialized_get_String"):
        dtype_size = ffi.new("size_t*")
        info = lib.GxB_Serialized_get_SIZE(data_obj, dtype_size, lib.GrB_NAME, data_nbytes)
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](f"{obj_kind} deserialize failed to get the size of name")
        dtype_char = ffi.new(f"char[{dtype_size[0]}]")
        info = lib.GxB_Serialized_get_String(data_obj, dtype_char, lib.GrB_NAME, data_nbytes)
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](f"{obj_kind} deserialize failed to get the name")
        return _string_to_dtype(ffi.string(dtype_char).decode())
    # No GrB_NAME fallback available (older SS); surface the original
    # resolution failure with the actual dtype name attached.
    raise ValueError(
        f"{obj_kind} deserialize cannot resolve dtype {dtype_name!r} and "
        f"GxB_Serialized_get_String is unavailable in this SuiteSparse build"
    ) from orig_error

import hashlib
import itertools
import re

from suitesparse_graphblas.utils import unclaim_buffer

from ...exceptions import _error_code_lookup
from .. import _has_numba, ffi, lib
from ..dtypes import _string_to_dtype


def _unclaim_taken(*arrays_and_pointers):
    """Unclaim each array whose pointer GraphBLAS cleared, as it does when it takes the data.

    Call it right after an import or pack, before checking the status. GraphBLAS
    moves the arrays into the object first and only then runs a secure import's
    check; when that fails, it frees the object and the arrays with it, so NumPy
    must stop owning them although the call failed. A call that fails before the
    move leaves every pointer as it was, and the arrays stay the caller's.
    """
    for array, ptr in arrays_and_pointers:
        if ptr[0] == ffi.NULL:
            unclaim_buffer(array)


def _pack_securely(import_, obj, format, opts, **kwargs):
    """Pack ``kwargs`` (data in ``format``, for ``import_``) into ``obj`` with a secure check.

    A secure pack whose check fails frees ``obj``'s GraphBLAS object itself
    (SuiteSparse:GraphBLAS 10.5.2 and earlier), and the Python object keeps the
    freed handle. A secure import that fails frees only the new object it made.
    So the data is imported securely into a new object, and its content moves
    into ``obj`` with an unpack and a fast pack, which take its arrays as they
    are. A failure leaves ``obj`` unchanged.
    """
    if obj.ndim == 1:
        shape = {"size": obj._size}
    else:
        shape = {"nrows": obj._nrows, "ncols": obj._ncols}
    new = import_(
        **kwargs, **shape, dtype=obj.dtype, secure_import=True, method="import", opts=opts
    )
    obj.ss.pack_any(**new.ss.unpack(format, **opts), take_ownership=True, **opts)
    return obj


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


def _jit_c_source(name, funcname, jit_c_definition, types=()):
    """Return the C name and definition to give SuiteSparse for an op registered from C.

    SuiteSparse finds a compiled kernel in memory by its op's name (and the types in the
    kernel's encoding), and checks an op's definition only when it loads a kernel from disk.
    So two ops of one C name with different definitions would run each other's kernel: an op
    defined again under its name, or two of one last name in two namespaces (``ss.f`` and
    ``ss.nested.f``). The C name is ``funcname``, then ``_{len(t)}_{t}`` for each of
    ``types`` (built-in types by name, UDTs by their C name), then a digest of these, the
    full dotted ``name`` and the definition, as ``f_3a1c...``. SuiteSparse keeps a name of at
    most 127 characters, so a longer one is ``funcname``'s start and a longer digest. One
    name and definition give one C name on purpose: registered again, they reuse the kernel
    compiled for them. The function is renamed in the definition wherever it is named, so a
    recursive call follows, but not a struct member of the same name (after ``.`` or ``->``).
    """
    parts = [re.sub(r"\W", "_", t.jit_c_name or t.name) for t in types]
    readable = funcname + "".join(f"_{len(part)}_{part}" for part in parts)
    digest = hashlib.sha256(f"{readable}\0{name}\0{jit_c_definition}".encode()).hexdigest()
    c_name = f"{readable}_{digest[:12]}"
    if len(c_name) >= lib.GxB_MAX_NAME_LEN:
        c_name = f"{funcname[:100]}_{digest[:16]}"

    def rename(match):
        if jit_c_definition[: match.start()].rstrip().endswith((".", "->")):
            return match.group()
        return c_name

    return c_name, re.sub(rf"\b{re.escape(funcname)}\b", rename, jit_c_definition)


# Here rather than in matrix.py, so vector.py can decorate with njit before it imports
# matrix.py (see the end of vector.py)
if _has_numba:
    from numba import njit, prange
else:

    def njit(func=None, **kwargs):
        if func is not None:
            return func
        return njit

    prange = range

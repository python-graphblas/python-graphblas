import warnings
from ast import literal_eval

import numpy as np
from numpy import promote_types, result_type

from .. import backend, dtypes
from ..core import NULL, _has_numba, ffi, lib

if _has_numba:
    import numba

# Default assumption unless FC32/FC64 are found in lib
_supports_complex = hasattr(lib, "GrB_FC64") or hasattr(lib, "GxB_FC64")


class DataType:
    __slots__ = "name", "gb_obj", "gb_name", "c_type", "numba_type", "np_type", "__weakref__"

    def __init__(self, name, gb_obj, gb_name, c_type, numba_type, np_type):
        self.name = name
        self.gb_obj = gb_obj
        self.gb_name = gb_name
        self.c_type = c_type
        self.numba_type = numba_type
        self.np_type = np.dtype(np_type) if np_type is not None else None

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        try:
            return self is lookup_dtype(other)
        except ValueError:
            raise TypeError(f"Invalid or unknown datatype: {other}") from None

    def __hash__(self):
        return hash(self.np_type)

    def __lt__(self, other):
        # Let us sort for prettier error reporting
        try:
            t1 = self.np_type
            t2 = lookup_dtype(other).np_type
        except ValueError:
            raise TypeError(f"Invalid or unknown datatype: {other}") from None
        return (t1.kind, t1.itemsize, t1.name) < (t2.kind, t2.itemsize, t2.name)

    def __reduce__(self):
        if self._is_udt:
            return (self._deserialize, (self.name, self.np_type, self._is_anonymous))
        if self.gb_name == "GrB_Index":
            return "_INDEX"
        return self.name

    @property
    def _carg(self):
        return self.gb_obj[0] if self.gb_name is None else self.gb_obj

    @property
    def _is_anonymous(self):
        return getattr(dtypes, self.name, None) is not self

    @property
    def _is_udt(self):
        return self.gb_name is None

    @staticmethod
    def _deserialize(name, dtype, is_anonymous):
        if is_anonymous:
            return register_anonymous(dtype, name)
        if name in _registry:
            return _registry[name]
        return register_new(name, dtype)


def register_new(name, dtype):
    if not name.isidentifier():
        raise ValueError(f"`name` argument must be a valid Python identifier; got: {name!r}")
    if name in _registry or hasattr(dtypes, name):
        raise ValueError(f"{name!r} name for dtype is unavailable")
    rv = register_anonymous(dtype, name)
    _registry[name] = rv
    setattr(dtypes, name, rv)
    return rv


def register_anonymous(dtype, name=None):
    try:
        dtype = np.dtype(dtype)
    except TypeError:
        if isinstance(dtype, dict):
            # Allow dtypes such as `{'x': int, 'y': float}` for convenience
            dtype = np.dtype(
                [(key, lookup_dtype(val).np_type) for key, val in dtype.items()], align=True
            )
        elif isinstance(dtype, str) and "[" in dtype and dtype.endswith("]"):
            # Allow dtypes such as `"INT64[3, 4]"` for convenience
            base_dtype, shape = dtype.split("[", 1)
            base_dtype = lookup_dtype(base_dtype)
            shape = literal_eval(f"[{shape}")
            dtype = np.dtype((base_dtype.np_type, shape))
        else:
            raise
    if dtype in _registry:
        # Always use the same object, but use the latest name
        rv = _registry[dtype]
        if name is not None:
            if rv.gb_name is not None and name != rv.gb_name:
                raise ValueError("dtype must not be a builtin type")
            rv.name = name  # Rename an existing object (a little weird, but okay)
        return rv
    if dtype.hasobject:
        raise ValueError("dtype must not allow Python objects")

    from ..exceptions import check_status_carg

    gb_obj = ffi.new("GrB_Type*")

    if hasattr(lib, "GrB_Type_set_String"):
        # We name this so that we can serialize and deserialize UDTs
        # We don't yet have C definitions
        np_repr = _dtype_to_string(dtype)
        status = lib.GrB_Type_new(gb_obj, dtype.itemsize)
        check_status_carg(status, "Type", gb_obj[0])
        val_obj = ffi.new("char[]", np_repr.encode())
        status = lib.GrB_Type_set_String(gb_obj[0], val_obj, lib.GrB_NAME)
    elif backend == "suitesparse":
        # For SuiteSparse < 9
        # We name this so that we can serialize and deserialize UDTs
        # We don't yet have C definitions
        np_repr = _dtype_to_string(dtype).encode()
        if len(np_repr) > lib.GxB_MAX_NAME_LEN:
            msg = (
                f"UDT repr is too large to serialize ({len(repr(dtype).encode())} > "
                f"{lib.GxB_MAX_NAME_LEN})."
            )
            if name is not None:
                np_repr = name.encode()[: lib.GxB_MAX_NAME_LEN]
            else:
                np_repr = np_repr[: lib.GxB_MAX_NAME_LEN]
            warnings.warn(
                f"{msg}.  It will use the following name, "
                f"and the dtype may need to be specified when deserializing: {np_repr}",
                stacklevel=2,
            )
        status = lib.GxB_Type_new(gb_obj, dtype.itemsize, np_repr, NULL)
    else:
        status = lib.GrB_Type_new(gb_obj, dtype.itemsize)
    check_status_carg(status, "Type", gb_obj[0])

    # For now, let's use "opaque" unsigned bytes for the c type.
    if name is None:
        name = _default_name(dtype)
    numba_type = numba.typeof(dtype).dtype if _has_numba else None
    rv = DataType(name, gb_obj, None, f"uint8_t[{dtype.itemsize}]", numba_type, dtype)
    _registry[gb_obj] = rv
    _registry[dtype] = rv
    if _has_numba:
        _registry[numba_type] = rv
        _registry[numba_type.name] = rv
    return rv


BOOL = DataType(
    "BOOL",
    lib.GrB_BOOL,
    "GrB_BOOL",
    "_Bool",
    numba.types.bool_ if _has_numba else None,
    np.bool_,
)
INT8 = DataType(
    "INT8", lib.GrB_INT8, "GrB_INT8", "int8_t", numba.types.int8 if _has_numba else None, np.int8
)
UINT8 = DataType(
    "UINT8",
    lib.GrB_UINT8,
    "GrB_UINT8",
    "uint8_t",
    numba.types.uint8 if _has_numba else None,
    np.uint8,
)
INT16 = DataType(
    "INT16",
    lib.GrB_INT16,
    "GrB_INT16",
    "int16_t",
    numba.types.int16 if _has_numba else None,
    np.int16,
)
UINT16 = DataType(
    "UINT16",
    lib.GrB_UINT16,
    "GrB_UINT16",
    "uint16_t",
    numba.types.uint16 if _has_numba else None,
    np.uint16,
)
INT32 = DataType(
    "INT32",
    lib.GrB_INT32,
    "GrB_INT32",
    "int32_t",
    numba.types.int32 if _has_numba else None,
    np.int32,
)
UINT32 = DataType(
    "UINT32",
    lib.GrB_UINT32,
    "GrB_UINT32",
    "uint32_t",
    numba.types.uint32 if _has_numba else None,
    np.uint32,
)
INT64 = DataType(
    "INT64",
    lib.GrB_INT64,
    "GrB_INT64",
    "int64_t",
    numba.types.int64 if _has_numba else None,
    np.int64,
)
# _Index (like UINT64) is for internal use only and shouldn't be exposed to the user
_INDEX = DataType(
    "UINT64",
    lib.GrB_UINT64,
    "GrB_Index",
    "GrB_Index",
    numba.types.uint64 if _has_numba else None,
    np.uint64,
)
UINT64 = DataType(
    "UINT64",
    lib.GrB_UINT64,
    "GrB_UINT64",
    "uint64_t",
    numba.types.uint64 if _has_numba else None,
    np.uint64,
)
FP32 = DataType(
    "FP32",
    lib.GrB_FP32,
    "GrB_FP32",
    "float",
    numba.types.float32 if _has_numba else None,
    np.float32,
)
FP64 = DataType(
    "FP64",
    lib.GrB_FP64,
    "GrB_FP64",
    "double",
    numba.types.float64 if _has_numba else None,
    np.float64,
)

if _supports_complex and hasattr(lib, "GxB_FC32"):
    FC32 = DataType(
        "FC32",
        lib.GxB_FC32,
        "GxB_FC32",
        "float _Complex",
        numba.types.complex64 if _has_numba else None,
        np.complex64,
    )
if _supports_complex and hasattr(lib, "GrB_FC32"):  # pragma: no cover (unused)
    FC32 = DataType(
        "FC32",
        lib.GrB_FC32,
        "GrB_FC32",
        "float _Complex",
        numba.types.complex64 if _has_numba else None,
        np.complex64,
    )
if _supports_complex and hasattr(lib, "GxB_FC64"):
    FC64 = DataType(
        "FC64",
        lib.GxB_FC64,
        "GxB_FC64",
        "double _Complex",
        numba.types.complex128 if _has_numba else None,
        np.complex128,
    )
if _supports_complex and hasattr(lib, "GrB_FC64"):  # pragma: no cover (unused)
    FC64 = DataType(
        "FC64",
        lib.GrB_FC64,
        "GrB_FC64",
        "double _Complex",
        numba.types.complex128 if _has_numba else None,
        np.complex128,
    )

# Used for testing user-defined functions
_sample_values = {
    INT8: np.int8(1),
    UINT8: np.uint8(1),
    INT16: np.int16(1),
    UINT16: np.uint16(1),
    INT32: np.int32(1),
    UINT32: np.uint32(1),
    INT64: np.int64(1),
    UINT64: np.uint64(1),
    FP32: np.float32(0.5),
    FP64: np.float64(0.5),
    BOOL: np.bool_(True),
}
if _supports_complex:
    _sample_values.update(
        {
            FC32: np.complex64(complex(0, 0.5)),
            FC64: np.complex128(complex(0, 0.5)),
        }
    )

# Create register to easily lookup types by name, gb_obj, or c_type
_registry = {}
_dtypes_to_register = [
    BOOL,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FP32,
    FP64,
]
if _supports_complex:
    _dtypes_to_register.extend([FC32, FC64])

for dtype in _dtypes_to_register:
    _registry[dtype.name] = dtype
    _registry[dtype.name.lower()] = dtype
    _registry[dtype.gb_obj] = dtype
    _registry[dtype.gb_name] = dtype
    _registry[dtype.gb_name.lower()] = dtype
    _registry[dtype.c_type] = dtype
    _registry[dtype.c_type.upper()] = dtype
    if _has_numba:
        _registry[dtype.numba_type] = dtype
        _registry[dtype.numba_type.name] = dtype
    val = _sample_values[dtype]
    _registry[val.dtype] = dtype
    _registry[val.dtype.name] = dtype
del dtype, val

# Add some common Python types as lookup keys
_registry[bool] = BOOL
_registry[int] = INT64
_registry[float] = FP64
_registry["bool"] = BOOL
_registry["int"] = INT64
_registry["float"] = FP64  # Choose 'float' to match numpy/Python; c_type 'float' would be FP32
if _supports_complex:
    _registry[complex] = FC64
    _registry["complex"] = FC64


def lookup_dtype(key, value=None):
    # Check for silly lookup where key is already a DataType
    if type(key) is DataType:
        return key
    try:
        return _registry[key]
    except (KeyError, TypeError):
        pass
    if value is not None and hasattr(value, "dtype") and value.dtype in _registry:
        return _registry[value.dtype]
    # np.dtype(x) accepts some weird values; we may want to guard against some
    if key is None:
        raise TypeError("Bad dtype: None.  A valid dtype must be provided.")
    try:
        # Auto-register!
        return register_anonymous(key)
    except Exception:
        pass
    try:
        return lookup_dtype(key.literal_type)  # For numba dtype inference
    except Exception:
        pass
    raise ValueError(f"Unknown dtype: {key} of type {type(key)}")


def unify(type1, type2, *, is_left_scalar=False, is_right_scalar=False):
    """Returns a type that can hold both type1 and type2.

    For example:
    unify(INT32, INT64) -> INT64
    unify(INT8, UINT16) -> INT32
    unify(BOOL, UINT16) -> UINT16
    unify(FP32, INT32) -> FP64
    """
    if type1 is type2:
        return type1
    if is_left_scalar:
        if not is_right_scalar:
            return lookup_dtype(result_type(np.array(0, type1.np_type), type2.np_type))
    elif is_right_scalar:
        return lookup_dtype(result_type(type1.np_type, np.array(0, type2.np_type)))
    return lookup_dtype(promote_types(type1.np_type, type2.np_type))


def _default_name(dtype):
    if dtype in _registry:
        dt = _registry[dtype]
        if not dt._is_udt:
            return dt.name
    if dtype.subdtype is not None:
        subdtype = _default_name(dtype.subdtype[0])
        shape = ", ".join(map(str, dtype.subdtype[1]))
        return f"{subdtype}[{shape}]"
    if dtype.names:
        args = ", ".join(
            f"{name!r}: {_default_name(dtype.fields[name][0])}" for name in dtype.names
        )
        return f"{{{args}}}"
    return repr(dtype)


def _dtype_to_string(dtype):
    """Convert a dtype to a string that can be safely evaluated to recreate the dtype.

    This is useful when serializing UDT.  To recreate the dtype, do:

    >>> s = _dtype_to_string(dtype)
    >>> new_dtype = _string_to_dtype(s)
    >>> dtype == new_dtype
    True
    """
    if isinstance(dtype, np.dtype) and dtype not in _registry:
        np_type = dtype
    else:
        dtype = lookup_dtype(dtype)
        if not dtype._is_udt:
            return dtype.name
        np_type = dtype.np_type
    s = str(np_type)
    try:
        if np.dtype(literal_eval(s)) == np_type:  # pragma: no branch (safety)
            return s
    except Exception:
        pass
    if np.dtype(np_type.str) != np_type:  # pragma: no cover (safety)
        raise ValueError(f"Unable to reliably convert dtype to string and back: {dtype}")
    return repr(np_type.str)


def _string_to_dtype(s):
    """Convert a string back to a dtype.

    >>> _string_to_dtype(_dtype_to_string(dtype)) == dtype
    True
    """
    try:
        return lookup_dtype(s)
    except Exception:
        pass
    np_type = np.dtype(literal_eval(s))
    return lookup_dtype(np_type)

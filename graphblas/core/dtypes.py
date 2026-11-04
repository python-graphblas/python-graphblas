import dataclasses
from ast import literal_eval

import numpy as np
from numpy import promote_types

from ..core import _has_numba, ffi, lib

# graphblas.dtypes imports names from this module, so the functions here that use it
# import it themselves: importing it at the top fails when this module is imported first.

if _has_numba:
    import numba
    from numba.np.numpy_support import as_dtype as _numba_as_dtype

# Default assumption unless FC32/FC64 are found in lib
_supports_complex = hasattr(lib, "GrB_FC64") or hasattr(lib, "GxB_FC64")


class DataType:
    __slots__ = (
        "name",
        "gb_obj",
        "gb_name",
        "c_type",
        "numba_type",
        "np_type",
        # ``(c_name, c_definition)`` given to SuiteSparse at first
        # registration, or ``None``. SuiteSparse takes each only once, so a
        # later rename of the UDT (``register_anonymous``) does not change them.
        "_jit_c_info",
        "__weakref__",
    )

    def __init__(self, name, gb_obj, gb_name, c_type, numba_type, np_type):
        self.name = name
        self.gb_obj = gb_obj
        self.gb_name = gb_name
        self.c_type = c_type
        self.numba_type = numba_type
        self.np_type = np.dtype(np_type) if np_type is not None else None
        self._jit_c_info = None

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
        from .. import dtypes

        return getattr(dtypes, self.name, None) is not self

    @property
    def _is_udt(self):
        return self.gb_name is None

    @property
    def jit_c_name(self):
        """The C type name SuiteSparse uses for this UDT, or ``None``.

        Use it to name the UDT in C source of your own. It is set when the
        numpy dtype is first registered, so it can differ from ``name`` after
        a later registration renames the UDT. It is ``None`` for built-in
        types and for UDTs with no C definition. See :ref:`tn_jit_c_names`.
        """
        return self._jit_c_info[0] if self._jit_c_info is not None else None

    @property
    def jit_c_definition(self):
        """The C struct typedef SuiteSparse uses for this UDT, or ``None``."""
        return self._jit_c_info[1] if self._jit_c_info is not None else None

    @staticmethod
    def _deserialize(name, dtype, is_anonymous):
        if is_anonymous:
            return register_anonymous(dtype, name)
        if name in _registry:
            return _registry[name]
        return register_new(name, dtype)


def _get_udt_c_type_info(datatype):
    """Return ``(c_name, c_typedef)`` for a UDT, or ``None`` if it has no C definition.

    ``udt_codegen._udt_c_typedef`` builds it and lists when there is none.
    """
    from .operator.udt_codegen import _udt_c_typedef

    # Every UDT registered from numpy has its np_type; one from C source never comes here.
    if datatype.np_type is None:  # pragma: no cover (safety)
        return None
    return _udt_c_typedef(datatype.name, datatype.np_type)


def _set_udt_jit_c_definition(datatype):
    """Set JIT C name and typedef on a UDT so SuiteSparse can JIT-compile kernels.

    Without them, SuiteSparse knows a UDT only by its size, and its JIT
    templates for operations on the UDT cannot expand. Both
    ``GxB_JIT_C_NAME`` and ``GxB_JIT_C_DEFINITION`` are needed.

    Setting ``GxB_JIT_C_NAME`` also makes ``GxB_deserialize_type_name``
    return that name rather than the numpy repr stored in ``GrB_NAME``, so
    ``Matrix.ss.deserialize`` and ``Vector.ss.deserialize`` fall back to
    ``GrB_NAME`` when the C name is not a known dtype.
    """
    from .operator.udt_codegen import (
        _has_jit_set,
        _jit_c_type_defs,
        _synthetic_udt_counter,
        _udt_c_typedef,
    )

    if not _has_jit_set or not datatype._is_udt:
        return
    info = _get_udt_c_type_info(datatype)
    if info is None:
        return
    c_name, typedef = info
    # A name already given to another layout (two UDTs registered under one
    # name) gets a fresh one: SuiteSparse would share their JIT kernels.
    while _jit_c_type_defs.setdefault(c_name, typedef) != typedef:
        c_name = f"_gbudt_{next(_synthetic_udt_counter)}"
        info = _udt_c_typedef(c_name, datatype.np_type, is_c_name=True)
        c_name, typedef = info
    lib.GrB_Type_set_String(datatype._carg, ffi.new("char[]", c_name.encode()), lib.GxB_JIT_C_NAME)
    lib.GrB_Type_set_String(
        datatype._carg, ffi.new("char[]", typedef.encode()), lib.GxB_JIT_C_DEFINITION
    )
    datatype._jit_c_info = info


# Elements Numba cannot type in an array (its NestedArray needs a bitwidth they lack), and
# what to store instead
_NUMBA_UNTYPED_ARRAY_ELEMENTS = {
    "b": ("bool", "uint8"),
    "M": ("datetime64", "int64"),
    "m": ("timedelta64", "int64"),
    "U": ("str", "uint32 (one per character)"),
    "S": ("bytes", "uint8 (one per byte)"),
}


def _iter_leaves(np_type):
    """Yield each leaf dtype of ``np_type``: a scalar, or an array of scalars, shape and all.

    Walks records, nested records, arrays of records and arrays of arrays
    (numpy keeps ``np.dtype((np.dtype(("?", (2,))), (3,)))`` as it is given),
    in field order, so it reaches every value an element holds. A leaf inside
    an array of records or of arrays comes once, as its record or inner array
    declares it. ``udt._iter_record_leaves``, which pairs two records' leaves
    for an op, yields an array of records as one leaf.
    """
    if np_type.names is not None:
        for name in np_type.names:
            yield from _iter_leaves(np_type.fields[name][0])
    elif np_type.subdtype is not None and (
        np_type.subdtype[0].names is not None or np_type.subdtype[0].subdtype is not None
    ):
        yield from _iter_leaves(np_type.subdtype[0])
    else:
        yield np_type


def _numba_untyped_array_element(dtype):
    """The kind of the first array element in dtype that Numba cannot type, or None."""
    for leaf in _iter_leaves(dtype):
        if leaf.subdtype is not None and leaf.base.kind in _NUMBA_UNTYPED_ARRAY_ELEMENTS:
            return leaf.base.kind
    return None


def _numba_type(dtype):
    try:
        return numba.typeof(dtype).dtype
    except AttributeError as exc:
        # Numba's own error is "'Boolean' object has no attribute 'bitwidth'" and the like
        kind = _numba_untyped_array_element(dtype)
        if kind is None:  # pragma: no cover (safety)
            raise
        element, instead = _NUMBA_UNTYPED_ARRAY_ELEMENTS[kind]
        raise ValueError(
            f"Numba, which python-graphblas uses for UDTs, cannot type an array of {element}, "
            f"as in {dtype}; use an array of {instead} instead"
        ) from exc
    except numba.core.errors.NumbaError as exc:
        # Numba's own error is "NumbaNotImplementedError: |V4", with terminal colors
        raw = _raw_bytes_element(dtype)
        if raw is None:  # pragma: no cover (safety)
            raise
        raise ValueError(
            f"Numba, which python-graphblas uses for UDTs, cannot type raw bytes ({raw}), "
            f"as in {dtype}; use an array of uint8 instead, np.dtype((np.uint8, "
            f"{raw.itemsize}))"
        ) from exc


def _raw_bytes_element(dtype):
    """The first void dtype with no fields (raw bytes) in dtype, or None."""
    return next((leaf.base for leaf in _iter_leaves(dtype) if leaf.base.kind == "V"), None)


def register_new(name, dtype=None):
    """Register a user-defined type and put it on the ``gb.dtypes`` namespace.

    ``name`` must be a Python identifier not already used there. ``dtype``
    takes the same forms as for :func:`register_anonymous`.
    ``register_new(MyDataclass)``, with a dataclass or an instance of one,
    is short for ``register_new("MyDataclass", MyDataclass)``. See
    :ref:`tn_udt_registration`.
    """
    if dtype is None:
        if dataclasses.is_dataclass(name):
            dtype = name
            name = name.__name__ if isinstance(name, type) else type(name).__name__
        else:
            raise TypeError(
                "register_new() requires both `name` and `dtype`, or a single "
                "@dataclass class/instance whose class name becomes the dtype name"
            )
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"`name` argument must be a valid Python identifier; got: {name!r}")
    from .. import dtypes

    if name in _registry or hasattr(dtypes, name):
        raise ValueError(f"{name!r} name for dtype is unavailable")
    rv = register_anonymous(dtype, name)
    _registry[name] = rv
    setattr(dtypes, name, rv)
    return rv


def register_anonymous(dtype, name=None):
    """Register a user-defined type without adding it to the ``gb.dtypes`` namespace.

    Returns the existing ``DataType`` if one is already registered for this
    ``np.dtype``. The new ``name`` (if given) replaces its name, but not the
    C name SuiteSparse knows it by (``DataType.jit_c_name``).

    Nested subarrays are flattened first, so ``np.dtype((np.dtype((np.float64,
    (3,))), (2,)))``, two ``FP64[3]`` elements, registers as ``FP64[2, 3]`` and
    returns the same ``DataType``. This applies inside records too.

    See :ref:`tn_udt_types` for every rule.

    Parameters
    ----------
    dtype : np.dtype | dict | str | @dataclass
        The dtype to register. Supported forms:

        - A ``np.dtype``: a record, an array (of any number of dimensions), or
          a single value GraphBLAS does not define, such as ``np.dtype("S5")``.
        - A dict like ``{"x": int, "y": float}``: keys become field names,
          and each value is anything ``lookup_dtype`` accepts, including an
          array type such as ``"FP64[3]"`` or a registered UDT.
        - A string like ``"INT64[3, 4]"`` for array UDTs.
        - A ``@dataclass`` class or instance: fields become record fields,
          and the class name is used as the default UDT name. Field type
          annotations may be either real types (``int``) or strings
          (``"int"``), e.g., with ``from __future__ import annotations`` or
          PEP 649. Compound annotations like ``Optional[int]`` raise
          ``ValueError``.

        A dict or dataclass gives a record aligned as a C struct is. Python
        objects and a dtype of size 0 raise ``ValueError``.
        :ref:`tn_udt_field_types` lists what a UDT can hold.
    name : str, optional
        A human-readable name for this UDT. Used in error messages and
        ``repr``. Its C name is ``gbudt_`` and the name, or a generated one
        when that is not a C identifier or is too long (:ref:`tn_jit_c_names`).

    Returns
    -------
    DataType
        The registered UDT.
    """
    # ``lookup_dtype`` accepts a type or a string annotation.
    if dataclasses.is_dataclass(dtype):
        fields = dataclasses.fields(dtype)
        if not fields:
            raise ValueError("dataclass must have at least one field to convert to a UDT")
        if name is None:
            name = dtype.__name__ if isinstance(dtype, type) else type(dtype).__name__
        dtype = np.dtype([(f.name, lookup_dtype(f.type).np_type) for f in fields], align=True)
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        # numpy's own dict form maps each name to a tuple, so a dict whose first value is a
        # string (``{"pos": "FP64[3]", ...}``) raises ValueError there, not TypeError
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
    dtype = _flatten_subarrays(dtype)
    if dtype in _registry:
        # Always use the same object, but use the latest name. Its C name stays
        # the first one (``DataType._jit_c_info``).
        rv = _registry[dtype]
        if name is not None:
            if rv.gb_name is not None and name != rv.gb_name:
                raise ValueError("dtype must not be a builtin type")
            rv.name = name
        return rv
    if dtype.hasobject:
        raise ValueError("dtype must not allow Python objects")
    if dtype.itemsize == 0:
        # GrB_Type_new would raise InvalidValue; a ValueError names the problem
        raise ValueError(f"dtype must not have a size of 0: {dtype}")
    # Before GrB_Type_new, so a dtype Numba cannot type does not leave a GrB_Type behind
    numba_type = _numba_type(dtype) if _has_numba else None

    from ..exceptions import check_status_carg

    gb_obj = ffi.new("GrB_Type*")
    # We name this so that we can serialize and deserialize UDTs
    # We don't yet have C definitions
    np_repr = _dtype_to_string(dtype)
    status = lib.GrB_Type_new(gb_obj, dtype.itemsize)
    check_status_carg(status, "Type", gb_obj[0])
    val_obj = ffi.new("char[]", np_repr.encode())
    status = lib.GrB_Type_set_String(gb_obj[0], val_obj, lib.GrB_NAME)
    check_status_carg(status, "Type", gb_obj[0])

    # For now, let's use "opaque" unsigned bytes for the c type.
    if name is None:
        name = _default_name(dtype)
    rv = DataType(name, gb_obj, None, f"uint8_t[{dtype.itemsize}]", numba_type, dtype)
    _registry[gb_obj] = rv
    _registry[dtype] = rv
    _set_udt_jit_c_definition(rv)
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
    if _has_numba and isinstance(key, (numba.types.Record, numba.types.NestedArray)):
        # A UDT's Numba type is not a registry key. Convert it to its numpy
        # dtype, which is already flat, and look that up, so a layout no UDT
        # has registered yet (e.g., a record's array field returned by a UDF)
        # resolves the same way regardless of what else is registered.
        return lookup_dtype(_numba_as_dtype(key))
    if value is not None and hasattr(value, "dtype") and value.dtype in _registry:
        return _registry[value.dtype]
    # np.dtype(x) accepts some weird values; we may want to guard against some
    if key is None:
        raise TypeError("Bad dtype: None.  A valid dtype must be provided.")
    try:
        # Auto-register!
        return register_anonymous(key)
    except Exception as exc:
        # Why it cannot be registered, such as an array of bool, which Numba cannot type.
        reason = exc
    try:
        return lookup_dtype(key.literal_type)  # For numba dtype inference
    except Exception:
        pass
    raise ValueError(f"Unknown dtype: {key} of type {type(key)}") from reason


def _raise_dtype_or_arraylike(cls_name, dtype, exc):
    """Turn a failed constructor dtype lookup into a helpful error.

    Call from ``Vector``/``Matrix`` after ``lookup_dtype(dtype)`` has already
    raised ``exc``. The constructors take a dtype as the first argument, so a
    common mistake is to pass the data instead (``Vector([1, 2, 3])``). When
    ``dtype`` is array-like data rather than a dtype spec, point at the ``from_*``
    constructors; otherwise re-raise the original ``Unknown dtype`` error. Valid
    list/tuple dtype specs (structured and subarray dtypes) never reach here
    because ``lookup_dtype`` accepts them, but a malformed one does, so the
    lookup's error stays attached as the cause.
    """
    if isinstance(dtype, (list, tuple, np.ndarray)):
        raise TypeError(
            f"{cls_name}() expects a dtype as the first argument, not "
            f"{type(dtype).__name__} data. If you meant to pass values, use a "
            f"constructor such as {cls_name}.from_coo(...) or "
            f"{cls_name}.from_dense(...)."
        ) from exc
    raise exc


def unify(type1, type2, *, is_left_scalar=False, is_right_scalar=False):
    """Returns a type that can hold both type1 and type2.

    For example:
    unify(INT32, INT64) -> INT64
    unify(INT8, UINT16) -> INT32
    unify(BOOL, UINT16) -> UINT16
    unify(FP32, INT32) -> FP64

    ``is_left_scalar`` and ``is_right_scalar`` are accepted and ignored, for code
    that still passes them; python-graphblas itself no longer does. A Python
    number is typed before it gets here (``literals._literal_operand``), and a
    typed scalar promotes as any operand does, as numpy 2 promotes it.
    """
    if type1 is type2:
        return type1
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


def _flatten_subarrays(np_type):
    """Collapse each run of nested subarray dtypes in ``np_type`` into one subarray.

    numpy keeps ``np.dtype((np.dtype((np.float64, (3,))), (2,)))`` layered, as
    two ``FP64[3]`` elements, and compares it unequal to the flat
    ``np.dtype((np.float64, (2, 3)))``. Nothing else tells them apart: the bytes
    are the same, an array of either expands to a ``(..., 2, 3)`` float64 array,
    and Numba types both as ``nestedarray(float64, (2, 3))``. Registering the flat
    form gives each layout one DataType. Records are rebuilt only when a field
    changes, keeping their offsets, titles, itemsize, and alignment flag.
    """
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        flat_base = _flatten_subarrays(base)
        if flat_base.subdtype is not None:
            flat_base, inner_shape = flat_base.subdtype
            shape += inner_shape
        elif flat_base is base:
            return np_type
        return np.dtype((flat_base, shape))
    if np_type.names is not None:
        fields = [np_type.fields[name] for name in np_type.names]
        formats = [_flatten_subarrays(field[0]) for field in fields]
        if all(new is field[0] for new, field in zip(formats, fields, strict=True)):
            return np_type
        spec = {
            "names": list(np_type.names),
            "formats": formats,
            "offsets": [field[1] for field in fields],
            "itemsize": np_type.itemsize,
        }
        if any(len(field) > 2 for field in fields):
            spec["titles"] = [field[2] if len(field) > 2 else None for field in fields]
        return np.dtype(spec, align=np_type.isalignedstruct)
    return np_type


def _view_if_same_layout(values, np_type):
    """Return ``values`` viewed as ``np_type`` if the two differ only in subarray nesting.

    A record whose field nests subarrays has a different dtype from its
    registered, flattened form, and numpy casts between the two wrongly in
    either direction: ``astype``, ``np.array(..., dtype=)``, and item assignment
    all scramble the field's values. The bytes already match, so a view is exact.
    """
    if (
        isinstance(values, (np.ndarray, np.void))
        and values.dtype != np_type
        and values.dtype.itemsize == np_type.itemsize
        and _flatten_subarrays(values.dtype) == np_type
    ):
        return values.view(np_type)
    return values


def _nats_in_leaf_units(values, np_type):
    """Return ``values`` with each NaT duration in it in the unit of the leaf it fills.

    numpy before 2.3 converts a NaT ``timedelta64`` scalar, alone or in a tuple
    or list, into another unit as the integer it is stored as: NaT ms into
    ``timedelta64[s]`` is 9223372036854774 s, and NaT s into ``timedelta64[ms]``
    is 0 ms. Converting an array (``astype``) keeps NaT, as does a date, so each
    NaT scalar is converted that way first. ``values`` holds elements of
    ``np_type``: a list is a sequence of them, and a tuple is a record's fields
    (or, for any other dtype, a sequence too).
    """
    if not _has_duration_leaf(np_type):
        return values
    while np_type.subdtype is not None:
        np_type = np_type.subdtype[0]
    if isinstance(values, list) or (isinstance(values, tuple) and np_type.names is None):
        items = [_nats_in_leaf_units(x, np_type) for x in values]
        return items if isinstance(values, list) else tuple(items)
    if isinstance(values, tuple) and len(values) == len(np_type.names):
        return tuple(
            _nats_in_leaf_units(x, np_type.fields[name][0])
            for x, name in zip(values, np_type.names, strict=True)
        )
    if np_type.kind == "m" and isinstance(values, np.timedelta64) and np.isnat(values):
        return np.asarray(values).astype(np_type)[()]
    return values


def _has_duration_leaf(np_type):
    """Whether ``np_type`` has a timedelta64 leaf: a record field, an array's element, or itself."""
    return any(leaf.base.kind == "m" for leaf in _iter_leaves(np_type))


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
        if np.dtype(literal_eval(s)) == np_type:
            return s
    except Exception:
        pass
    if np.dtype(np_type.str) == np_type:
        return repr(np_type.str)
    # Some nested dtypes (e.g. ``align=True`` outer + ``packed`` inner) don't
    # round-trip via ``str(np_type)`` because numpy's reconstruction enforces
    # the outer's alignment on the inner. Fall back to an explicit per-field
    # dict using literal offsets/itemsize, which encodes the exact layout
    # without invoking ``align``. ``literal_eval`` accepts the result, so
    # ``_string_to_dtype`` doesn't need a special case for it.
    return repr(_dtype_to_explicit_dict(np_type))


def _dtype_to_explicit_dict(np_type):
    """Encode ``np_type`` as a plain dict/tuple/str tree of literals.

    Layout-preserving fallback for :func:`_dtype_to_string`. Records become a
    ``{names, formats, offsets, itemsize}`` dict whose nested ``formats`` may
    be further nested dicts; array dtypes become a ``(base_str, shape)``
    tuple. The result round-trips through ``np.dtype(literal_eval(s))``
    without ever using ``align=True``, so the original byte layout is
    preserved verbatim even for the aligned-outer / packed-inner cases.
    """
    if np_type.names is not None:
        formats = []
        for name in np_type.names:
            sub = np_type.fields[name][0]
            if sub.names is not None or sub.subdtype is not None:
                formats.append(_dtype_to_explicit_dict(sub))
            else:
                formats.append(sub.str)
        return {
            "names": list(np_type.names),
            "formats": formats,
            "offsets": [np_type.fields[name][1] for name in np_type.names],
            "itemsize": np_type.itemsize,
        }
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        if base.names is not None:
            base_repr = _dtype_to_explicit_dict(base)
        else:
            base_repr = base.str
        return (base_repr, shape)
    # Only records and arrays get here: any other dtype round-trips through its ``.str``.
    return np_type.str  # pragma: no cover (safety)


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

from collections.abc import MutableMapping

from ...dtypes import lookup_dtype
from ...exceptions import _error_code_lookup, check_status
from .. import NULL, ffi, lib
from ..utils import maybe_integral, values_to_numpy_buffer


class BaseConfig(MutableMapping):
    _initialized = False
    # Subclasses should redefine these
    _get_function = None
    _set_function = None
    _context_get_function = "GxB_Context_get"
    _context_set_function = "GxB_Context_set"
    _context_keys = set()
    _null_valid = {}
    _options = {}
    _defaults = {}
    # We add reverse lookups for _enumerations and _bitwise in __init__
    _bitwise = {}
    _enumerations = {}
    _read_only = set()
    _int32_ctypes = {
        "bool",
        "int",
        "int32_t",
        "GrB_Desc_Value",
        "GrB_Mode",
        "GxB_Format_Value",
    }

    def __init__(self, parent=None, context=None):
        cls = type(self)
        if not cls._initialized:
            cls._reverse_enumerations = {}
            for key, d in self._enumerations.items():
                cls._reverse_enumerations[key] = rd = {}
                for k, v in list(d.items()):
                    if v not in d:
                        d[v] = v
                    rd[v] = k
                    if k not in rd:
                        rd[k] = k
            cls._reverse_bitwise = {}
            for key, d in self._bitwise.items():
                cls._reverse_bitwise[key] = rd = {}
                for k, v in list(d.items()):
                    if v not in d:  # pragma: no branch (safety)
                        d[v] = v
                    rd[v] = k
                    if k not in rd:  # pragma: no branch (safety)
                        rd[k] = k
            cls._initialized = True
        self._parent = parent
        self._context = context

    def __delitem__(self, key):
        raise TypeError("Configuration options can't be deleted.")

    def __getitem__(self, key):
        key = key.lower()
        if key not in self._options:
            raise KeyError(key)
        key_obj, ctype = self._options[key]
        is_bool = ctype == "bool"
        if is_context := (key in self._context_keys):
            get_function_base = self._context_get_function
        else:
            get_function_base = self._get_function
        if ctype in self._int32_ctypes:
            ctype = "int32_t"
            get_function_name = f"{get_function_base}_INT32"
        elif ctype.startswith("int64_t"):
            get_function_name = f"{get_function_base}_INT64"
        elif ctype.startswith("double"):
            get_function_name = f"{get_function_base}_FP64"
        elif ctype.startswith("char"):
            get_function_name = f"{get_function_base}_CHAR"
        else:  # pragma: no cover (sanity)
            raise ValueError(ctype)
        get_function = getattr(lib, get_function_name)
        is_array = "[" in ctype
        val_ptr = ffi.new(ctype if is_array else f"{ctype}*")
        if is_context:
            info = get_function(self._context._carg, key_obj, val_ptr)
        elif self._parent is None:
            info = get_function(key_obj, val_ptr)
        else:
            info = get_function(self._parent._carg, key_obj, val_ptr)
        if info == lib.GrB_SUCCESS:  # pragma: no branch (safety)
            if is_array:
                return list(val_ptr)
            if key in self._reverse_enumerations:
                return self._reverse_enumerations[key][val_ptr[0]]
            if key in self._reverse_bitwise:
                val = val_ptr[0]
                if val in (reverse_bitwise := self._reverse_bitwise[key]):
                    return {reverse_bitwise[val]}
                rv = set()
                for k, v in self._bitwise[key].items():
                    if isinstance(k, str) and val & v and (v).bit_count() == 1:
                        rv.add(k)
                return rv
            if is_bool:
                return bool(val_ptr[0])
            if ctype.startswith("char"):
                return ffi.string(val_ptr[0]).decode()
            return val_ptr[0]
        raise _error_code_lookup[info](f"Failed to get info for {key!r}")  # pragma: no cover

    def __setitem__(self, key, val):
        key = key.lower()
        if key not in self._options:
            raise KeyError(key)
        if key in self._read_only:
            raise ValueError(f"Config option {key!r} is read-only")
        key_obj, ctype = self._options[key]
        if is_context := (key in self._context_keys):
            set_function_base = self._context_set_function
        else:
            set_function_base = self._set_function
        if ctype in self._int32_ctypes:
            ctype = "int32_t"
            set_function_name = f"{set_function_base}_INT32"
        elif ctype == "double":
            set_function_name = f"{set_function_base}_FP64"
        elif ctype.startswith("int64_t["):
            set_function_name = f"{set_function_base}_INT64_ARRAY"
        elif ctype.startswith("double["):
            set_function_name = f"{set_function_base}_FP64_ARRAY"
        elif ctype.startswith("char"):
            set_function_name = f"{set_function_base}_CHAR"
        else:  # pragma: no cover (sanity)
            raise ValueError(ctype)
        set_function = getattr(lib, set_function_name)
        if val is None:
            pass
        elif key in self._enumerations:
            if isinstance(val, str):
                val = val.lower()
                val = self._enumerations[key][val]
            else:
                val = self._enumerations[key].get(val, val)
        elif key in self._bitwise:
            bitwise = self._bitwise[key]
            if isinstance(val, str):
                val = bitwise[val.lower()]
            elif (x := maybe_integral(val)) is not None:
                val = bitwise.get(x, x)
            else:
                bits = 0
                for x in val:
                    if isinstance(x, str):
                        bits |= bitwise[x.lower()]
                    else:
                        bits |= x
                val = bits
        if val is None:
            if key not in self._defaults:
                raise ValueError(f"Unable to set default value for {key!r}")
            val = self._defaults[key]
        if val is None:
            val_obj = NULL
        elif "[" in ctype:
            dtype, size = ctype.split("[", 1)
            size = size.split("]", 1)[0]
            dtype = lookup_dtype(dtype)
            vals, dtype = values_to_numpy_buffer(val, dtype.np_type)
            if int(size) != vals.size:
                raise ValueError(
                    f"Wrong number of elements when setting {key!r} config.  "
                    f"expected {size}, got {vals.size}: {val}"
                )
            val_obj = ffi.from_buffer(ctype, vals)
        elif ctype.startswith("char"):
            val_obj = ffi.new("char[]", val.encode())
        else:
            val_obj = ffi.cast(ctype, val)
        if is_context:
            if self._context is None:
                from .context import Context

                self._context = Context(engage=False)
                self._context._engage()  # Disengage when context goes out of scope
                self._parent._context = self._context  # Set context to descriptor
            info = set_function(self._context._carg, key_obj, val_obj)
        elif self._parent is None:
            info = set_function(key_obj, val_obj)
        else:
            info = set_function(self._parent._carg, key_obj, val_obj)
        if info != lib.GrB_SUCCESS:
            if self._parent is not None:  # pragma: no branch (safety)
                check_status(info, self._parent)
            raise _error_code_lookup[info](  # pragma: no cover (safety)
                f"Failed to set info for {key!r}"
            )

    def __iter__(self):
        return iter(sorted(self._options))

    def __len__(self):
        return len(self._options)

    def __repr__(self):
        return (
            type(self).__name__
            + "({"
            + ",\n ".join(f"{k!r}: {v!r}" for k, v in self.items())
            + "})"
        )

    def _ipython_key_completions_(self):  # pragma: no cover (ipython)
        return list(self)

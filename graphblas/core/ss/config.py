from collections.abc import MutableMapping
from numbers import Integral

from suitesparse_graphblas import vararg

from ...dtypes import lookup_dtype
from ...exceptions import _error_code_lookup
from .. import NULL, ffi, lib
from ..utils import values_to_numpy_buffer


class BaseConfig(MutableMapping):
    # Subclasses should redefine these
    _get_function = None
    _set_function = None
    _null_valid = {}
    _options = {}
    _defaults = {}
    # We add reverse lookups for _enumerations and _bitwise in __init__
    _bitwise = {}
    _enumerations = {}
    _read_only = set()
    _set_ctypes = {
        "GxB_Format_Value": "int",
        "bool": "int",
    }

    def __init__(self, parent=None):
        for d in self._enumerations.values():
            for k, v in list(d.items()):
                d[v] = k
        for d in self._bitwise.values():
            for k, v in list(d.items()):
                d[v] = k
        self._parent = parent

    def __delitem__(self, key):
        raise TypeError("Configuration options can't be deleted.")

    def __getitem__(self, key):
        key = key.lower()
        if key not in self._options:
            raise KeyError(key)
        key_obj, ctype = self._options[key]
        is_array = "[" in ctype
        val_ptr = ffi.new(ctype if is_array else f"{ctype}*")
        if self._parent is None:
            info = self._get_function(key_obj, vararg(val_ptr))
        else:
            info = self._get_function(self._parent._carg, key_obj, vararg(val_ptr))
        if info == lib.GrB_SUCCESS:  # pragma: no branch
            if is_array:
                return list(val_ptr)
            elif key in self._enumerations:
                return self._enumerations[key][val_ptr[0]]
            elif key in self._bitwise:
                bitwise = self._bitwise[key]
                val = val_ptr[0]
                if val in bitwise:
                    return {bitwise[val]}
                rv = set()
                for k, v in bitwise.items():
                    if isinstance(k, str) and val & v and bin(v).count("1") == 1:
                        rv.add(k)
                return rv
            return val_ptr[0]
        raise _error_code_lookup[info](f"Failed to get info for {key!r}")  # pragma: no cover

    def __setitem__(self, key, val):
        key = key.lower()
        if key not in self._options:
            raise KeyError(key)
        if key in self._read_only:
            raise ValueError(f"Config option {key!r} is read-only")
        key_obj, ctype = self._options[key]
        ctype = self._set_ctypes.get(ctype, ctype)
        if key in self._enumerations and isinstance(val, str):
            val = val.lower()
            val = self._enumerations[key][val]
        elif key in self._bitwise and val is not None and not isinstance(val, Integral):
            bitwise = self._bitwise[key]
            if isinstance(val, str):
                val = bitwise[val.lower()]
            else:
                bits = 0
                for x in val:
                    if isinstance(x, str):
                        bits |= bitwise[x.lower()]
                    else:
                        bits |= x
                val = bits
        if val is None:
            if key in self._defaults:
                val = self._defaults[key]
            else:
                raise ValueError(f"Unable to set default value for {key!r}")
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
        else:
            val_obj = ffi.cast(ctype, val)
        if self._parent is None:
            info = self._set_function(key_obj, vararg(val_obj))
        else:
            info = self._set_function(self._parent._carg, key_obj, vararg(val_obj))
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](f"Failed to set info for {key!r}")

    def __iter__(self):
        return iter(sorted(self._options))

    def __len__(self):
        return len(self._options)

    def __repr__(self):
        return "{" + ",\n ".join(f"{k!r}: {v!r}" for k, v in self.items()) + "}"

    def _ipython_key_completions_(self):  # pragma: no cover
        return list(self)

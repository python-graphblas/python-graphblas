import numpy as np
from numba import types as nt


# Create a singleton class for dtypes
class GrB_Type:
    def __init__(self):
        self._types = []
        self._reverse_types = {}

    def _add_type(self, name, dtype, numba_type):
        setattr(self, f"GrB_{name}", dtype)
        self._types.append(name)
        self._reverse_types[dtype] = name
        self._reverse_types[numba_type] = name

    @property
    def types(self):
        return tuple(self._types)

    def lookup_name(self, dtype):
        return self._reverse_types[dtype]


GrB_Type = GrB_Type()
#                                 name     numpy dtype  numba type
for name, dtype, numba_type in [
    ("BOOL", np.bool, nt.boolean),
    ("INT8", np.int8, nt.int8),
    ("UINT8", np.uint8, nt.uint8),
    ("INT16", np.int16, nt.int16),
    ("UINT16", np.uint16, nt.uint16),
    ("INT32", np.int32, nt.int32),
    ("UINT32", np.uint32, nt.uint32),
    ("INT64", np.int64, nt.int64),
    ("UINT64", np.uint64, nt.uint64),
    ("FP32", np.float32, nt.float32),
    ("FP64", np.float64, nt.float64),
]:
    GrB_Type._add_type(name, dtype, numba_type)

# Add GrB_Index alias manually -- don't need a reverse lookup
GrB_Type.GrB_Index = np.uint64

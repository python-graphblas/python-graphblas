import numpy as np

from .exceptions import GraphBlasException

GrB_ALL = object()
GrB_NULL = object()


class GraphBlasContainer:
    @classmethod
    def get_pointer(cls):
        raise NotImplementedError()


class BasePointer:
    def __init__(self):
        self.gb_obj = self  # hack to make `self.gb_obj[0]` work correctly
        self.instance = None

    def __getitem__(self, key):
        if key != 0:
            raise KeyError("Only [0] is available for pointers")
        return self.instance

    @property
    def is_initialized(self):
        return self.instance is not None


def new_pointer(dtype):
    if issubclass(dtype, GraphBlasContainer):
        return dtype.get_pointer()
    return np.empty((1,), dtype)


def new_array(dtype, length):
    if np.dtype(dtype) == object:
        raise GraphBlasException(f"Cannot create an array of type {dtype}")
    return np.empty((length,), dtype)

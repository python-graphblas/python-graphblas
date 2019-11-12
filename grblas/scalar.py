from .base import (
    lib,
    ffi,
    NULL,
    GbContainer,
)
from . import dtypes


class Scalar(GbContainer):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """
    can_mask = False

    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)

    def __repr__(self):
        return f'<Scalar {self.value}:{self.dtype}>'

    def __eq__(self, other):
        if type(other) == Scalar:
            if self.dtype != other.dtype:
                return False
            return self.value == other.value
        else:
            return self.value == other

    def __bool__(self):
        return bool(self.value)

    def clear(self):
        if self.dtype == bool:
            self.value = False
        else:
            self.value = 0

    @property
    def value(self):
        return self.gb_obj[0]

    @value.setter
    def value(self, val):
        self.gb_obj[0] = val

    @classmethod
    def new_from_type(cls, dtype):
        """
        Create a new empty Scalar from the given type
        """
        dtype = dtypes.lookup(dtype)
        new_scalar_pointer = ffi.new(f'{dtype.c_type}*')
        return cls(new_scalar_pointer, dtype)

    @classmethod
    def new_from_existing(cls, scalar):
        """Create a new Scalar by duplicating an existing one
        """
        if not isinstance(scalar, GbContainer):
            raise TypeError(f'Must pass in a Scalar object, not {type(scalar)}')
        new_scalar = cls.new_from_type(scalar.dtype)
        new_scalar.value = scalar.value
        return new_scalar

    @classmethod
    def new_from_value(cls, value):
        """Create a new Scalar from a Python value
        """
        dtype = dtypes.lookup(type(value))
        new_scalar = cls.new_from_type(dtype)
        new_scalar.value = value
        return new_scalar

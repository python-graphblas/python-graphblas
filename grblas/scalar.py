from .base import ffi, GbContainer
from . import dtypes


class Scalar(GbContainer):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """
    is_scalar = True

    def __init__(self, gb_obj, dtype, empty=False):
        super().__init__(gb_obj, dtype)
        self.is_empty = empty

    def __repr__(self):
        return f'<Scalar {self.value}:{self.dtype}>'

    def __eq__(self, other):
        if type(other) is Scalar:
            if self.dtype != other.dtype:
                return False
            return self.value == other.value
        else:
            return self.value == other

    def __bool__(self):
        if self.is_empty:
            return False
        return bool(self.value)

    def clear(self):
        if self.dtype == bool:
            self.value = False
        else:
            self.value = 0
        self.is_empty = True

    @property
    def value(self):
        if self.is_empty:
            return None
        return self.gb_obj[0]

    @value.setter
    def value(self, val):
        if val is None:
            self.clear()
        else:
            self.gb_obj[0] = val
            self.is_empty = False

    @classmethod
    def new_from_type(cls, dtype):
        """
        Create a new empty Scalar from the given type
        """
        dtype = dtypes.lookup(dtype)
        new_scalar_pointer = ffi.new(f'{dtype.c_type}*')
        return cls(new_scalar_pointer, dtype, empty=True)

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
    def new_from_value(cls, value, dtype=None):
        """Create a new Scalar from a Python value
        """
        if dtype is None:
            dtype = dtypes.lookup(type(value))
        new_scalar = cls.new_from_type(dtype)
        new_scalar.value = value
        return new_scalar

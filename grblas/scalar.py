import grblas
from .base import ffi, GbContainer
from . import dtypes


class Scalar(GbContainer):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """
    _is_scalar = True

    def __init__(self, gb_obj, dtype, empty=False):
        super().__init__(gb_obj, dtype)
        self.is_empty = empty

    def __repr__(self):
        return f'<Scalar {self.value}:{self.dtype}>'

    def __eq__(self, other):
        return self.isequal(other)

    def __bool__(self):
        if self.is_empty:
            return False
        return bool(self.value)

    def isequal(self, other, *, check_dtype=False):
        """
        Check for exact equality

        If `check_dtype` is True, also checks that dtypes match
        For equality of floating point Vectors, consider using `isclose`
        """
        if not isinstance(other, Scalar):
            if other is None:
                return self.is_empty
            try:
                # Check if other is a literal scalar, which we should handle
                dtype = dtypes.lookup(type(other))
            except KeyError:
                raise TypeError(f'Argument of isequal must be a known scalar type, not {type(other)}')
            other = Scalar.from_value(other, dtype=dtype)
            # Don't check dtype if we had to infer dtype of `other`
            check_dtype = False
        if check_dtype and self.dtype != other.dtype:
            return False
        if self.is_empty or other.is_empty:
            return self.is_empty is other.is_empty
        # For now, compare values in Python.  We can get more sophisticated
        # if there is a need by converting both scalars to 1-d Vectors.
        # Hopefully scalar types will be added to the GraphBLAS spec.
        return self.value == other.value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality

        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        if not isinstance(other, Scalar):
            if other is None:
                return self.is_empty
            try:
                # Check if other is a literal scalar, which we should handle
                dtype = dtypes.lookup(type(other))
            except KeyError:
                raise TypeError(f'Argument of isclose must be a known scalar type, not {type(other)}')
            other = Scalar.from_value(other, dtype=dtype)
            # Don't check dtype if we had to infer dtype of `other`
            check_dtype = False
        if check_dtype and self.dtype != other.dtype:
            return False
        if self.is_empty or other.is_empty:
            return self.is_empty is other.is_empty
        # We can't yet call a UDF on a scalar, so lets convert to 1-d vector
        left = grblas.vector.Vector.from_values([0], [self.value], size=1, dtype=self.dtype)
        right = grblas.vector.Vector.from_values([0], [other.value], size=1, dtype=other.dtype)
        matches = left.ewise_mult(right, grblas.binary.isclose(rel_tol, abs_tol)).new(dtype=bool)
        return matches.reduce(grblas.monoid.land).value

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

    def dup(self, *, dtype=None):
        """Create a new Scalar by duplicating this one
        """
        if dtype is None:
            new_scalar = self.__class__.new(self.dtype)
            new_scalar.value = self.value
        else:
            new_scalar = self.__class__.new(dtype)
            new_scalar.value = new_scalar.dtype.numba_type(self.value)
        return new_scalar

    @classmethod
    def new(cls, dtype):
        """
        Create a new empty Scalar from the given type
        """
        dtype = dtypes.lookup(dtype)
        new_scalar_pointer = ffi.new(f'{dtype.c_type}*')
        return cls(new_scalar_pointer, dtype, empty=True)

    @classmethod
    def from_value(cls, value, dtype=None):
        """Create a new Scalar from a Python value
        """
        if dtype is None:
            dtype = dtypes.lookup(type(value))
        new_scalar = cls.new(dtype)
        new_scalar.value = value
        return new_scalar

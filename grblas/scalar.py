import itertools
from . import ffi, backend
from .base import BaseExpression, BaseType
from .binary import isclose
from .dtypes import lookup_dtype
from .ops import get_typed_op

ffi_new = ffi.new


class Scalar(BaseType):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """

    _is_scalar = True
    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, empty=False, name=None):
        if name is None:
            name = f"s_{next(Scalar._name_counter)}"
        super().__init__(gb_obj, dtype, name)
        self._is_empty = empty

    def __repr__(self):
        from .formatting import format_scalar

        return format_scalar(self)

    def _repr_html_(self):
        from .formatting import format_scalar_html

        return format_scalar_html(self)

    def __eq__(self, other):
        return self.isequal(other)

    __hash__ = None

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
        if type(other) is not Scalar:
            if other is None:
                return self.is_empty
            try:
                other = Scalar.from_value(other, name="s_isequal")
            except TypeError:
                self._expect_type(
                    other,
                    Scalar,
                    within="isequal",
                    argname="other",
                    extra_message="Literal scalars also accepted.",
                )
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
        if type(other) is not Scalar:
            if other is None:
                return self.is_empty
            try:
                other = Scalar.from_value(other, name="s_isclose")
            except TypeError:
                self._expect_type(
                    other,
                    Scalar,
                    within="isclose",
                    argname="other",
                    extra_message="Literal scalars also accepted.",
                )
            # Don't check dtype if we had to infer dtype of `other`
            check_dtype = False
        if check_dtype and self.dtype != other.dtype:
            return False
        if self.is_empty or other.is_empty:
            return self.is_empty is other.is_empty
        # We can't yet call a UDF on a scalar as part of the spec, so let's do it ourselves
        isclose_func = isclose(rel_tol, abs_tol)
        isclose_func = get_typed_op(isclose_func, self.dtype, other.dtype)
        return isclose_func.numba_func(self.value, other.value)

    def clear(self):
        if self.dtype == bool:
            self.value = False
        else:
            self.value = 0
        self._is_empty = True

    @property
    def is_empty(self):
        return self._is_empty

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
            self._is_empty = False

    @property
    def nvals(self):
        if self.is_empty:
            return 0
        return 1

    @property
    def _carg(self):
        return self.gb_obj

    def dup(self, *, dtype=None, name=None):
        """Create a new Scalar by duplicating this one
        """
        if dtype is None:
            new_scalar = type(self).new(self.dtype, name=name)
            new_scalar.value = self.value
        else:
            new_scalar = type(self).new(dtype, name=name)
            if not self.is_empty:
                new_scalar.value = new_scalar.dtype.numba_type(self.value)
        return new_scalar

    @classmethod
    def new(cls, dtype, *, name=None):
        """
        Create a new empty Scalar from the given type
        """
        dtype = lookup_dtype(dtype)
        new_scalar_pointer = ffi_new(f"{dtype.c_type}*")
        return cls(new_scalar_pointer, dtype, name=name, empty=True)

    @classmethod
    def from_value(cls, value, dtype=None, *, name=None):
        """Create a new Scalar from a Python value
        """
        if dtype is None:
            try:
                dtype = lookup_dtype(type(value))
            except ValueError:
                raise TypeError(
                    f"Argument of from_value must be a known scalar type, not {type(value)}"
                )
        new_scalar = cls.new(dtype, name=name)
        new_scalar.value = value
        return new_scalar

    if backend == "pygraphblas":  # pragma: no cover

        def to_pygraphblas(self):
            """ Convert to a new `pygraphblas.Scalar`

            This copies data.
            """
            import pygraphblas

            return pygraphblas.Scalar.from_value(self.value)

        @classmethod
        def from_pygraphblas(cls, scalar):
            """ Convert a `pygraphblas.Scalar` to a new `grblas.Scalar`

            This copies data.
            """
            dtype = lookup_dtype(scalar.gb_type)
            return cls.from_value(scalar[0], dtype)


class ScalarExpression(BaseExpression):
    output_type = Scalar

    @property
    def value(self):
        return self.new(name="s_value").value

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Scalar.new(dtype, name=name)

    def new(self, *, dtype=None, name=None):
        return super().new(dtype=dtype, name=name)

    def __repr__(self):
        from .formatting import format_scalar_expression

        return format_scalar_expression(self)

    def _repr_html_(self):
        from .formatting import format_scalar_expression_html

        return format_scalar_expression_html(self)


class _CScalar:
    def __init__(self, scalar):
        self.scalar = scalar
        self.dtype = scalar.dtype

    def __repr__(self):
        return repr(self.scalar.value)

    def _repr_html_(self):
        return self.scalar._repr_html_()

    @property
    def _carg(self):
        return self.scalar.value

    def __eq__(self, other):
        if type(other) is _CScalar:
            return self.scalar == other.scalar
        return self.scalar == other

    __hash__ = None

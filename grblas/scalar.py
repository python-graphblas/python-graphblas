import itertools

import numpy as np

from . import _automethods, backend, ffi, utils
from .base import BaseExpression, BaseType
from .binary import isclose
from .dtypes import _INDEX, lookup_dtype
from .operator import get_typed_op
from .utils import output_type, wrapdoc

ffi_new = ffi.new


class Scalar(BaseType):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """

    __slots__ = "_is_empty"
    shape = ()
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

    def __bool__(self):
        if self.is_empty:
            return False
        return bool(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __complex__(self):
        return complex(self.value)

    def __neg__(self):
        dtype = self.dtype
        if dtype.name[0] == "U" or dtype.name == "BOOL":
            raise TypeError(f"The negative operator, `-`, is not supported for {dtype.name} dtype")
        rv = Scalar.new(dtype, name=f"-{self.name}")
        if self.is_empty:
            return rv
        rv.value = -self.value
        return rv

    __index__ = __int__

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype.np_type
        return np.array(self.value, dtype=dtype)

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
                other = self._expect_type(
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
                other = self._expect_type(
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

    _nvals = nvals

    def dup(self, *, dtype=None, name=None):
        """Create a new Scalar by duplicating this one"""
        if dtype is None:
            new_scalar = Scalar.new(self.dtype, name=name)
            new_scalar.value = self.value
        else:
            new_scalar = Scalar.new(dtype, name=name)
            if not self.is_empty:
                new_scalar.value = new_scalar.dtype.np_type(self.value)
        return new_scalar

    def wait(self):
        pass

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
        """Create a new Scalar from a Python value"""
        if type(value) is Scalar:
            return value.dup(dtype=dtype, name=name)
        elif output_type(value) is Scalar:
            return value.new(dtype=dtype, name=name)
        elif dtype is None:
            try:
                dtype = lookup_dtype(type(value))
            except ValueError:
                raise TypeError(
                    f"Argument of from_value must be a known scalar type, not {type(value)}"
                )
        new_scalar = cls.new(dtype, name=name)
        new_scalar.value = value
        return new_scalar

    def __reduce__(self):
        return Scalar._deserialize, (self.value, self.dtype, self.name)

    @staticmethod
    def _deserialize(value, dtype, name):
        return Scalar.from_value(value, dtype=dtype, name=name)

    def to_pygraphblas(self):  # pragma: no cover
        """Convert to a new `pygraphblas.Scalar`

        This copies data.
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"to_pygraphblas only works with 'suitesparse' backend, not {backend}"
            )
        import pygraphblas as pg

        return pg.Scalar.from_value(self.value)

    @classmethod
    def from_pygraphblas(cls, scalar):  # pragma: no cover
        """Convert a `pygraphblas.Scalar` to a new `grblas.Scalar`

        This copies data.
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"from_pygraphblas only works with 'suitesparse' backend, not {backend!r}"
            )
        import pygraphblas as pg

        if not isinstance(scalar, pg.Scalar):
            raise TypeError(f"Expected pygraphblas.Scalar object.  Got type: {type(scalar)}")
        dtype = lookup_dtype(scalar.gb_type)
        return cls.from_value(scalar[0], dtype)


class ScalarExpression(BaseExpression):
    __slots__ = ()
    output_type = Scalar
    shape = ()
    _is_scalar = True

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Scalar.new(dtype, name=name)

    def new(self, *, dtype=None, name=None):
        return super().new(dtype=dtype, name=name)

    dup = new

    def __repr__(self):
        from .formatting import format_scalar_expression

        return format_scalar_expression(self)

    def _repr_html_(self):
        from .formatting import format_scalar_expression_html

        return format_scalar_expression_html(self)

    # Paste here from _automethods.py
    _get_value = _automethods._get_value
    __array__ = wrapdoc(Scalar.__array__)(property(_automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(_automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(_automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(_automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(_automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(_automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(_automethods.__int__))
    __neg__ = wrapdoc(Scalar.__neg__)(property(_automethods.__neg__))
    _name_html = wrapdoc(Scalar._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Scalar._nvals)(property(_automethods._nvals))
    gb_obj = wrapdoc(Scalar.gb_obj)(property(_automethods.gb_obj))
    is_empty = wrapdoc(Scalar.is_empty)(property(_automethods.is_empty))
    isclose = wrapdoc(Scalar.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Scalar.isequal)(property(_automethods.isequal))
    name = wrapdoc(Scalar.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Scalar.nvals)(property(_automethods.nvals))
    to_pygraphblas = wrapdoc(Scalar.to_pygraphblas)(property(_automethods.to_pygraphblas))
    value = wrapdoc(Scalar.value)(property(_automethods.value))
    wait = wrapdoc(Scalar.wait)(property(_automethods.wait))
    # These raise exceptions
    __and__ = wrapdoc(Scalar.__and__)(Scalar.__and__)
    __iand__ = wrapdoc(Scalar.__iand__)(Scalar.__iand__)
    __imatmul__ = wrapdoc(Scalar.__imatmul__)(Scalar.__imatmul__)
    __ior__ = wrapdoc(Scalar.__ior__)(Scalar.__ior__)
    __matmul__ = wrapdoc(Scalar.__matmul__)(Scalar.__matmul__)
    __or__ = wrapdoc(Scalar.__or__)(Scalar.__or__)
    __rand__ = wrapdoc(Scalar.__rand__)(Scalar.__rand__)
    __rmatmul__ = wrapdoc(Scalar.__rmatmul__)(Scalar.__rmatmul__)
    __ror__ = wrapdoc(Scalar.__ror__)(Scalar.__ror__)


class _CScalar:
    """Wrap scalars for calling into C.

    If a Scalar is not provided, then a datatype of GrB_Index is assumed.
    """

    __slots__ = "scalar", "dtype"

    def __init__(self, scalar, dtype=_INDEX):
        if type(scalar) is not Scalar:
            scalar = Scalar.from_value(scalar, name="", dtype=dtype)
        self.scalar = scalar
        self.dtype = scalar.dtype

    def __repr__(self):
        return repr(self.scalar.value)

    def _repr_html_(self):
        return self.scalar._repr_html_()

    @property
    def _carg(self):
        return self.scalar.value

    @property
    def name(self):
        return self.scalar.name or repr(self.scalar.value)

    def __eq__(self, other):
        if type(other) is _CScalar:
            return self.scalar == other.scalar
        return self.scalar == other


utils._output_types[Scalar] = Scalar
utils._output_types[ScalarExpression] = Scalar

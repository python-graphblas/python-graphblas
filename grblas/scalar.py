import itertools

import numpy as np

from . import _automethods, backend, ffi, lib, utils
from .base import BaseExpression, BaseType, call
from .binary import isclose
from .dtypes import _INDEX, BOOL, lookup_dtype
from .exceptions import check_status
from .operator import get_typed_op
from .utils import _Pointer, output_type, wrapdoc

ffi_new = ffi.new


class Scalar(BaseType):
    """
    GraphBLAS Scalar
    Pseudo-object for GraphBLAS functions which accumlate into a scalar type
    """

    __slots__ = "_empty", "_is_cscalar"
    ndim = 0
    shape = ()
    _is_scalar = True
    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, empty=False, is_cscalar=False, name=None):
        if name is None:
            name = f"s_{next(Scalar._name_counter)}"
        super().__init__(gb_obj, dtype, name)
        self._is_cscalar = is_cscalar
        if is_cscalar:
            self._empty = empty

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and "GB_Scalar_opaque" in str(gb_obj):
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Scalar_free(gb_obj), self)

    @property
    def is_cscalar(self):
        return self._is_cscalar

    @property
    def is_grbscalar(self):
        return not self._is_cscalar

    def __repr__(self):
        from .formatting import format_scalar

        return format_scalar(self)

    def _repr_html_(self, collapse=False):
        from .formatting import format_scalar_html

        return format_scalar_html(self)

    def __eq__(self, other):
        return self.isequal(other)

    def __bool__(self):
        if self._is_empty:
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
        if dtype.name[0] == "U" or dtype == BOOL:
            raise TypeError(f"The negative operator, `-`, is not supported for {dtype.name} dtype")
        rv = Scalar.new(dtype, is_cscalar=self._is_cscalar, name=f"-{self.name}")
        if self._is_empty:
            return rv
        rv.value = -self.value
        return rv

    def __invert__(self):
        if self.dtype != BOOL:
            raise TypeError(
                f"The invert operator, `~`, is not supported for {self.dtype.name} dtype.  "
                "It is only supported for BOOL dtype."
            )
        rv = Scalar.new(BOOL, is_cscalar=self._is_cscalar, name=f"~{self.name}")
        if self._is_empty:
            return rv
        rv.value = not self.value
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
                return self._is_empty
            try:
                other = Scalar.from_value(
                    other, is_cscalar=True, name="s_isequal"  # pragma: to_grb
                )
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
        if self._is_empty or other._is_empty:
            return self._is_empty is other._is_empty
        # For now, compare values in Python.  We can get more sophisticated
        return self.value == other.value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality

        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        if type(other) is not Scalar:
            if other is None:
                return self._is_empty
            try:
                other = Scalar.from_value(
                    other, is_cscalar=True, name="s_isclose"  # pragma: to_grb
                )
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
        if self._is_empty or other._is_empty:
            return self._is_empty is other._is_empty
        # We can't yet call a UDF on a scalar as part of the spec, so let's do it ourselves
        isclose_func = isclose(rel_tol, abs_tol)
        isclose_func = get_typed_op(
            isclose_func,
            self.dtype,
            other.dtype,
            is_left_scalar=True,
            is_right_scalar=True,
            kind="binary",
        )
        return isclose_func.numba_func(self.value, other.value)

    def clear(self):
        if self._is_empty:
            return
        if self._is_cscalar:
            if self.dtype == bool:
                self.value = False
            else:
                self.value = 0
            self._empty = True
        else:
            call("GrB_Scalar_clear", [self])

    @property
    def is_empty(self):
        if self._is_cscalar:
            return self._empty
        return self.nvals == 0

    @property
    def _is_empty(self):
        """Like is_empty, but doesn't record calls"""
        if self._is_cscalar:
            return self._empty
        return self._nvals == 0

    @property
    def value(self):
        if self._is_empty:
            return None
        if self._is_cscalar:
            return self.gb_obj[0]
        else:
            scalar = Scalar.new(self.dtype, is_cscalar=True)
            if not self.name:
                # Empty name plays havoc on the recorder, so give it a placeholder name
                name = self.name
                self.name = "scalar"
                try:
                    call(f"GrB_Scalar_extractElement_{self.dtype.name}", [_Pointer(scalar), self])
                finally:
                    self.name = name
            else:
                call(f"GrB_Scalar_extractElement_{self.dtype.name}", [_Pointer(scalar), self])
            return scalar.gb_obj[0]

    @value.setter
    def value(self, val):
        if val is None:
            self.clear()
        elif self._is_cscalar:
            if type(val) is Scalar and val._is_empty:
                self.clear()
            else:
                self.gb_obj[0] = val
                self._empty = False
        else:
            val = _as_scalar(val, is_cscalar=True)  # XXX: set dtype to self.dtype?
            if val._is_empty:
                self.clear()
            else:
                call(f"GrB_Scalar_setElement_{val.dtype}", [self, val])

    @property
    def nvals(self):
        if self._is_cscalar:
            if self._empty:
                return 0
            return 1
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", is_cscalar=True, empty=True)
        call("GrB_Scalar_nvals", [_Pointer(scalar), self])
        return n[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        if self._is_cscalar:
            if self._empty:
                return 0
            return 1
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Scalar_nvals(n, self.gb_obj[0]), self)
        return n[0]

    @property
    def _carg(self):
        # i.e., return None if we are empty and a C scalar
        if not self._is_cscalar or not self._is_empty:
            return self.gb_obj[0]

    def dup(self, dtype=None, *, is_cscalar=None, name=None):
        """Create a new Scalar by duplicating this one"""
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        if not is_cscalar and not self._is_scalar and (dtype is None or dtype == self.dtype):
            scalar = ffi_new("GrB_Scalar*")
            new_scalar = Scalar(
                scalar, self.dtype, is_cscalar=False, name=name  # pragma: is_grbscalar
            )
            call("GrB_Scalar_dup", [_Pointer(new_scalar), self])
        elif dtype is None:
            new_scalar = Scalar.new(self.dtype, is_cscalar=is_cscalar, name=name)
            new_scalar.value = self.value
        else:
            new_scalar = Scalar.new(dtype, is_cscalar=is_cscalar, name=name)
            if not self._is_empty:
                new_scalar.value = new_scalar.dtype.np_type(self.value)
        return new_scalar

    def wait(self):
        pass

    @classmethod
    def new(cls, dtype, *, is_cscalar=False, name=None):
        """
        Create a new empty Scalar from the given type
        """
        dtype = lookup_dtype(dtype)
        if is_cscalar:
            new_scalar = ffi_new(f"{dtype.c_type}*")
        else:
            new_scalar = ffi_new("GrB_Scalar*")
        rv = cls(new_scalar, dtype, name=name, empty=True, is_cscalar=is_cscalar)
        if not is_cscalar:
            call("GrB_Scalar_new", [_Pointer(rv), dtype])
        return rv

    @classmethod
    def from_value(cls, value, dtype=None, *, is_cscalar=False, name=None):
        """Create a new Scalar from a Python value"""
        if dtype is None:
            if output_type(value) is Scalar:
                dtype = value.dtype
            else:
                try:
                    dtype = lookup_dtype(type(value))
                except ValueError:
                    raise TypeError(
                        f"Argument of from_value must be a known scalar type, not {type(value)}"
                    )
        new_scalar = cls.new(dtype, is_cscalar=is_cscalar, name=name)
        new_scalar.value = value
        return new_scalar

    def __reduce__(self):
        return Scalar._deserialize, (self.value, self.dtype, self._is_cscalar, self.name)

    @staticmethod
    def _deserialize(value, dtype, is_cscalar, name):
        return Scalar.from_value(value, dtype, is_cscalar=is_cscalar, name=name)

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

    def _as_vector(self):
        """Copy this Scalar to a Vector

        In the future, we may _cast_ instead of _copy_ when using SuiteSparse.
        """
        from .vector import Vector

        if self._is_cscalar:
            rv = Vector.new(self.dtype, size=1)
            if not self._is_empty:
                rv[0] = self
        else:
            rv = Vector(
                ffi.cast("GrB_Vector*", self.gb_obj),
                self.dtype,
                parent=self,
            )
            rv._size = 1
        return rv


class ScalarExpression(BaseExpression):
    __slots__ = "_is_cscalar"
    output_type = Scalar
    ndim = 0
    shape = ()
    _is_scalar = True

    def __init__(self, *args, is_cscalar, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_cscalar = is_cscalar

    def construct_output(self, dtype=None, *, is_cscalar=None, name=None):
        if dtype is None:
            dtype = self.dtype
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        return Scalar.new(dtype, is_cscalar=is_cscalar, name=name)

    def new(self, dtype=None, *, is_cscalar=None, name=None):
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        return super()._new(dtype, None, name, is_cscalar=is_cscalar)

    dup = new

    def __repr__(self):
        from .formatting import format_scalar_expression

        return format_scalar_expression(self)

    def _repr_html_(self):
        from .formatting import format_scalar_expression_html

        return format_scalar_expression_html(self)

    is_cscalar = Scalar.is_cscalar
    is_grbscalar = Scalar.is_grbscalar

    # Begin auto-generated code: Scalar
    _get_value = _automethods._get_value
    __array__ = wrapdoc(Scalar.__array__)(property(_automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(_automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(_automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(_automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(_automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(_automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(_automethods.__int__))
    __invert__ = wrapdoc(Scalar.__invert__)(property(_automethods.__invert__))
    __neg__ = wrapdoc(Scalar.__neg__)(property(_automethods.__neg__))
    _is_empty = wrapdoc(Scalar._is_empty)(property(_automethods._is_empty))
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
    __and__ = Scalar.__and__
    __matmul__ = Scalar.__matmul__
    __or__ = Scalar.__or__
    __rand__ = Scalar.__rand__
    __rmatmul__ = Scalar.__rmatmul__
    __ror__ = Scalar.__ror__
    # End auto-generated code: Scalar


def _as_scalar(scalar, dtype=None, *, is_cscalar):
    if type(scalar) is not Scalar:
        return Scalar.from_value(scalar, dtype, is_cscalar=is_cscalar, name="")
    elif scalar._is_cscalar != is_cscalar or dtype is not None and scalar.dtype != dtype:
        return scalar.dup(dtype, is_cscalar=is_cscalar, name=scalar.name)
    else:
        return scalar


utils._output_types[Scalar] = Scalar
utils._output_types[ScalarExpression] = Scalar

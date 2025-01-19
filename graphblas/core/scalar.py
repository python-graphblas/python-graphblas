import itertools

import numpy as np

from .. import backend, binary, config, monoid
from ..dtypes import _INDEX, FP64, _index_dtypes, lookup_dtype, unify
from ..exceptions import EmptyObject, check_status
from . import _has_numba, _supports_udfs, automethods, ffi, lib, utils
from .base import BaseExpression, BaseType, call
from .expr import AmbiguousAssignOrExtract
from .operator import get_typed_op
from .utils import _Pointer, output_type, wrapdoc

if _supports_udfs:
    from ..binary import isclose
else:
    from .operator.binary import _isclose as isclose

ffi_new = ffi.new


def _scalar_index(name):
    """Fast way to create scalars with GrB_Index type; used internally."""
    self = object.__new__(Scalar)
    self.name = name
    self.dtype = _INDEX
    self.gb_obj = ffi_new("GrB_Index*")
    self._is_cscalar = True
    self._empty = True
    return self


def _s_union_s(updater, left, right, left_default, right_default, op):
    opts = updater.opts
    new_left = left.dup(op.type, clear=True)
    new_left(**opts) << binary.second(right, left_default)
    new_left(**opts) << binary.first(left | new_left)
    new_right = right.dup(op.type2, clear=True)
    new_right(**opts) << binary.second(left, right_default)
    new_right(**opts) << binary.first(right | new_right)
    updater << op(new_left & new_right)


class Scalar(BaseType):
    """Create a new GraphBLAS Sparse Scalar.

    Parameters
    ----------
    dtype :
        Data type of the Scalar.
    is_cscalar : bool, default=False
        If True, the empty state is managed on the Python side rather than
        with a proper GrB_Scalar object.
    name : str, optional
        Name to give the Scalar. This will be displayed in the ``__repr__``.

    """

    __slots__ = "_empty", "_is_cscalar"
    ndim = 0
    shape = ()
    _is_scalar = True
    _name_counter = itertools.count()

    def __new__(cls, dtype=FP64, *, is_cscalar=False, name=None):
        self = object.__new__(cls)
        dtype = self.dtype = lookup_dtype(dtype)
        self.name = f"s_{next(Scalar._name_counter)}" if name is None else name
        if is_cscalar is None:
            # Internally, we sometimes use `is_cscalar=None` to either defer to `expr.is_cscalar`
            # or to create a C scalar from a Python scalar.  For example, see Matrix.apply.
            is_cscalar = True  # pragma: to_grb
        self._is_cscalar = is_cscalar
        if is_cscalar:
            self._empty = True
            if dtype._is_udt:
                self.gb_obj = ffi_new(dtype.c_type)
            else:
                self.gb_obj = ffi_new(f"{dtype.c_type}*")
        else:
            self.gb_obj = ffi_new("GrB_Scalar*")
            call("GrB_Scalar_new", [_Pointer(self), dtype])
        return self

    @classmethod
    def _from_obj(cls, gb_obj, dtype, *, is_cscalar=False, name=None):
        self = object.__new__(cls)
        self.name = f"s_{next(Scalar._name_counter)}" if name is None else name
        self.gb_obj = gb_obj
        self.dtype = dtype
        self._is_cscalar = is_cscalar
        return self

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None and "GB_Scalar_opaque" in str(gb_obj):
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Scalar_free(gb_obj), self)

    @property
    def is_cscalar(self):
        """Returns True if the empty state is managed on the Python side."""
        return self._is_cscalar

    @property
    def is_grbscalar(self):
        """Returns True if the empty state is managed by the GraphBLAS backend."""
        return not self._is_cscalar

    @property
    def _expr_name(self):
        """The name used in the text for expressions."""
        # Always using `repr(self.value)` may also be reasonable
        return self.name or repr(self.value)

    @property
    def _expr_name_html(self):
        """The name used in the text for expressions in HTML formatting."""
        return self._name_html or repr(self.value)

    def __repr__(self, expr=None):
        from .formatting import format_scalar

        return format_scalar(self, expr=expr)

    def _repr_html_(self, collapse=False, expr=None):
        from .formatting import format_scalar_html

        return format_scalar_html(self, expr=expr)

    def __eq__(self, other):
        """Check equality.

        Equality comparison uses :meth:`isequal`. Use that directly for finer control of
        what is considered equal.
        """
        return self.isequal(other)

    def __ne__(self, other):
        return not self.isequal(other)

    def __bool__(self):
        """Truthiness check.

        The scalar is considered truthy if it is non-empty and the value inside is truthy.

        To only check if a value is present, use :attr:`is_empty`.
        """
        if self._is_empty:
            return False
        return bool(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __complex__(self):
        return complex(self.value)

    @property
    def __index__(self):
        if self.dtype in _index_dtypes:
            return self.__int__
        raise AttributeError("Scalar object only has `__index__` for integral dtypes")

    def __array__(self, dtype=None, *, copy=None):
        if dtype is None:
            dtype = self.dtype.np_type
        return np.array(self.value, dtype=dtype)

    def __sizeof__(self):
        base = object.__sizeof__(self)
        if self._is_cscalar:
            return base + self.gb_obj.__sizeof__() + ffi.sizeof(self.dtype.c_type)
        if backend == "suitesparse":
            size = ffi_new("size_t*")
            check_status(lib.GxB_Scalar_memoryUsage(size, self.gb_obj[0]), self)
            return base + size[0]
        raise TypeError("Unable to get size of GrB_Scalar with backend: {backend}")

    def isequal(self, other, *, check_dtype=False):
        """Check for exact equality (including whether the value is missing).

        Parameters
        ----------
        other : Scalar
            Scalar to compare against
        check_dtypes : bool, default=False
            If True, also checks that dtypes match

        Returns
        -------
        bool

        See Also
        --------
        :meth:`isclose` : For equality check of floating point dtypes

        """
        if type(other) is not Scalar:
            if other is None:
                return self._is_empty
            try:
                dtype = self.dtype if self.dtype._is_udt else None
                other = Scalar.from_value(other, dtype, is_cscalar=None, name="s_isequal")
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
        if self._is_empty:
            return other._is_empty
        if other._is_empty:
            return False
        # For now, compare values in Python
        rv = self.value == other.value
        try:
            return bool(rv)
        except ValueError:
            return bool(rv.all())

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """Check for approximate equality (including whether the value is missing).

        Equivalent to: ``abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)``.

        Parameters
        ----------
        other : Scalar
            Scalar to compare against
        rel_tol : float
            Relative tolerance
        abs_tol : float
            Absolute tolerance
        check_dtype : bool
            If True, also checks that dtypes match

        Returns
        -------
        bool

        """
        if type(other) is not Scalar:
            if other is None:
                return self._is_empty
            try:
                other = Scalar.from_value(other, is_cscalar=None, name="s_isclose")
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
        if self._is_empty:
            return other._is_empty
        if other._is_empty:
            return False
        # We can't yet call a UDF on a scalar as part of the spec, so let's do it ourselves
        isclose_func = isclose(rel_tol, abs_tol)
        if not _has_numba:
            # Check if types are compatible
            get_typed_op(
                binary.eq,
                self.dtype,
                other.dtype,
                is_left_scalar=True,
                is_right_scalar=True,
                kind="binary",
            )
            return isclose_func(self.value, other.value)
        isclose_func = get_typed_op(
            isclose_func,
            self.dtype,
            other.dtype,
            is_left_scalar=True,
            is_right_scalar=True,
            kind="binary",
        )
        return isclose_func._numba_func(self.value, other.value)

    def clear(self):
        """In-place operation which clears the value in the Scalar.

        After the call, :attr:`nvals` will return 0.
        """
        if self._is_empty:
            return
        if self._is_cscalar:
            self._empty = True
        else:
            call("GrB_Scalar_clear", [self])

    @property
    def is_empty(self):
        """Indicates whether the Scalar is empty or not."""
        if self._is_cscalar:
            return self._empty
        return self.nvals == 0

    @property
    def _is_empty(self):
        """Like is_empty, but doesn't record calls."""
        if self._is_cscalar:
            return self._empty
        return self._nvals == 0

    @property
    def value(self):
        """Returns the value held by the Scalar as a Python object,
        or None if the Scalar is empty.

        Assigning to ``value`` will update the Scalar.

        Example Usage:

            >>> s.value
            15
            >>> s.value = 16
            >>> s.value
            16
        """
        if self._is_empty:
            return
        is_udt = self.dtype._is_udt
        if self._is_cscalar:
            scalar = self
        else:
            scalar = Scalar(self.dtype, is_cscalar=True, name="s_temp")
            dtype_name = "UDT" if is_udt else self.dtype.name
            call(f"GrB_Scalar_extractElement_{dtype_name}", [_Pointer(scalar), self])
        if is_udt:
            np_type = self.dtype.np_type
            rv = np.array(ffi.buffer(scalar.gb_obj[0 : np_type.itemsize]))
            if np_type.subdtype is None:
                return rv.view(np_type)[0]
            return rv.view(np_type.subdtype[0]).reshape(np_type.subdtype[1])
        return scalar.gb_obj[0]

    @value.setter
    def value(self, val):
        if val is None or output_type(val) is Scalar and val._is_empty:
            self.clear()
        elif self._is_cscalar:
            if output_type(val) is Scalar:
                val = val.value  # raise below if wrong type (as determined by cffi)
            if self.dtype._is_udt:
                np_type = self.dtype.np_type
                if np_type.subdtype is None:
                    arr = np.empty(1, dtype=np_type)
                    if isinstance(val, dict) and np_type.names is not None:
                        val = _dict_to_record(np_type, val)
                else:
                    arr = np.empty(np_type.subdtype[1], dtype=np_type.subdtype[0])
                arr[:] = val
                self.gb_obj[0 : self.dtype.np_type.itemsize] = arr.view(np.uint8)
                # self.gb_obj[0:self.dtype.np_type.itemsize] = bytes(val)
            else:
                self.gb_obj[0] = val
            self._empty = False
        else:
            if self.dtype._is_udt:
                val = _Pointer(_as_scalar(val, self.dtype, is_cscalar=True))
                dtype_name = "UDT"
            else:
                val = _as_scalar(val, is_cscalar=True)
                dtype_name = val.dtype.name
            call(f"GrB_Scalar_setElement_{dtype_name}", [self, val])

    @property
    def nvals(self):
        """Number of non-empty values.

        Can only be 0 or 1.
        """
        if self._is_cscalar:
            return 0 if self._empty else 1
        scalar = _scalar_index("s_nvals")
        call("GrB_Scalar_nvals", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls."""
        if self._is_cscalar:
            return 0 if self._empty else 1
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Scalar_nvals(n, self.gb_obj[0]), self)
        return n[0]

    @property
    def _carg(self):
        if not self._is_cscalar or not self._is_empty:
            return self.gb_obj[0]
        raise EmptyObject(
            "Empty C scalar is invalid when when passed as value (not pointer) to C functions.  "
            "Perhaps use GrB_Scalar instead (e.g., `my_scalar.dup(is_cscalar=False)`)"
        )

    def dup(self, dtype=None, *, clear=False, is_cscalar=None, name=None):
        """Create a duplicate of the Scalar.

        This is a full copy, not a view on the original.

        Parameters
        ----------
        dtype :
            Data type of the new Scalar. Normal typecasting rules apply.
        clear : bool, default=False
            If True, the returned Scalar will be empty.
        is_cscalar : bool
            If True, the empty state is managed on the Python side rather
            than with a proper GrB_Scalar object.
        name : str, optional
            Name to give the Scalar.

        Returns
        -------
        Scalar

        """
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        if (
            not is_cscalar
            and not self._is_cscalar
            and not clear
            and (dtype is None or dtype == self.dtype)
        ):
            new_scalar = Scalar._from_obj(
                ffi_new("GrB_Scalar*"),
                self.dtype,
                is_cscalar=False,  # pragma: is_grbscalar
                name=name,
            )
            call("GrB_Scalar_dup", [_Pointer(new_scalar), self])
        elif dtype is None:
            new_scalar = Scalar(self.dtype, is_cscalar=is_cscalar, name=name)
            if not clear:
                new_scalar.value = self
        else:
            new_scalar = Scalar(dtype, is_cscalar=is_cscalar, name=name)
            if not clear and not self._is_empty:
                if new_scalar.is_cscalar and not new_scalar.dtype._is_udt:
                    # Cast value so we don't raise given explicit dup with dtype
                    new_scalar.value = new_scalar.dtype.np_type.type(self.value)
                else:
                    new_scalar.value = self.value
        return new_scalar

    def wait(self, how="materialize"):
        """Wait for a computation to complete or establish a "happens-before" relation.

        Parameters
        ----------
        how : {"materialize", "complete"}
            "materialize" fully computes an object.
            "complete" establishes a "happens-before" relation useful with multi-threading.
            See GraphBLAS documentation for more details.

        In `non-blocking mode <../user_guide/init.html#graphblas-modes>`__,
        the computations may be delayed and not yet safe to use by multiple threads.
        Use wait to force completion of the Scalar.

        Has no effect in `blocking mode <../user_guide/init.html#graphblas-modes>`__.

        """
        how = how.lower()
        if how == "materialize":
            mode = _MATERIALIZE
        elif how == "complete":
            mode = _COMPLETE
        else:
            raise ValueError(f'`how` argument must be "materialize" or "complete"; got {how!r}')
        if not self._is_cscalar:
            call("GrB_Scalar_wait", [self, mode])
        return self

    def get(self, default=None):
        """Get the internal value of the Scalar as a Python scalar.

        Parameters
        ----------
        default :
            Value returned if internal value is missing.

        Returns
        -------
        Python scalar

        """
        return default if self._is_empty else self.value

    @classmethod
    def from_value(cls, value, dtype=None, *, is_cscalar=False, name=None):
        """Create a new Scalar from a value.

        Parameters
        ----------
        value : Python scalar
            Internal value of the Scalar.
        dtype :
            Data type of the Scalar. If not provided, the value will be
            inspected to choose an appropriate dtype.
        is_cscalar : bool, default=False
            If True, the empty state is managed on the Python side
            rather than with a proper GrB_Scalar object.
        name : str, optional
            Name to give the Scalar.

        Returns
        -------
        Scalar

        """
        typ = output_type(value)
        if dtype is None:
            if typ is Scalar:
                dtype = value.dtype
            else:
                try:
                    dtype = lookup_dtype(type(value), value)
                except ValueError:
                    raise TypeError(
                        f"Argument of from_value must be a known scalar type, not {type(value)}"
                    ) from None
        if typ is Scalar and type(value) is not Scalar:
            if config.get("autocompute"):
                return value.new(dtype=dtype, is_cscalar=is_cscalar, name=name)
            cls()._expect_type(
                value,
                Scalar,
                within="from_value",
                argname="value",
                extra_message="Literal scalars expected.",
            )
        new_scalar = cls(dtype, is_cscalar=is_cscalar, name=name)
        new_scalar.value = value
        return new_scalar

    def __reduce__(self):
        return Scalar._deserialize, (self.value, self.dtype, self._is_cscalar, self.name)

    @staticmethod
    def _deserialize(value, dtype, is_cscalar, name):
        return Scalar.from_value(value, dtype, is_cscalar=is_cscalar, name=name)

    def _as_vector(self, *, name=None):
        """Copy or cast this Scalar to a Vector.

        This casts to a Vector when using GrB_Scalar from SuiteSparse.
        """
        from .vector import Vector

        if backend == "suitesparse" and not self._is_cscalar:
            return Vector._from_obj(
                ffi.cast("GrB_Vector*", self.gb_obj),
                self.dtype,
                1,
                parent=self,
                name=f"(GrB_Vector){self.name or 's_temp'}" if name is None else name,
            )
        rv = Vector(self.dtype, size=1, name=name)
        if not self._is_empty:
            rv[0] = self
        return rv

    def _as_matrix(self, *, name=None):
        """Copy or cast this Scalar to a Matrix.

        This casts to a Matrix when using GrB_Scalar from SuiteSparse.
        """
        from .matrix import Matrix

        if backend == "suitesparse" and not self._is_cscalar:
            return Matrix._from_obj(
                ffi.cast("GrB_Matrix*", self.gb_obj),
                self.dtype,
                1,
                1,
                parent=self,
                name=f"(GrB_Matrix){self.name or 's_temp'}" if name is None else name,
            )
        rv = Matrix(self.dtype, ncols=1, nrows=1, name=name)
        if not self._is_empty:
            rv[0, 0] = self
        return rv

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to update to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus):
        """Perform element-wise computation on the union of sparse values, similar to how
        one expects addition to work for sparse data.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_add and :meth:`ewise_union`.

        Parameters
        ----------
        other : Scalar
            The other scalar in the computation; Python scalars also accepted
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use on intersecting values

        Returns
        -------
        ScalarExpression that will be non-empty if any of the inputs is non-empty

        Examples
        --------
        .. code-block:: python

            # Method syntax
            c << a.ewise_add(b, op=monoid.max)

            # Functional syntax
            c << monoid.max(a | b)

        """
        return self._ewise_add(other, op)

    def _ewise_add(self, other, op=monoid.plus, is_infix=False):
        method_name = "ewise_add"
        if is_infix:
            from .infix import ScalarEwiseAddExpr

            # This is a little different than how we handle ewise_add for Vector and
            # Matrix where we are super-careful to handle dtypes well to support UDTs.
            # For Scalar, we're going to let dtypes in expressions resolve themselves.
            # Scalars are more challenging, because they may be literal scalars.
            # Also, we have not yet resolved `op` here, so errors may be different.
            if isinstance(self, ScalarEwiseAddExpr):
                self = op(self).new()
            if isinstance(other, ScalarEwiseAddExpr):
                other = op(other).new()

        if type(other) is not Scalar:
            dtype = self.dtype if self.dtype._is_udt else None
            try:
                other = Scalar.from_value(other, dtype, is_cscalar=False, name="")
            except TypeError:
                other = self._expect_type(
                    other,
                    Scalar,
                    within=method_name,
                    keyword_name="other",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        return ScalarExpression(
            method_name,
            f"GrB_Vector_eWiseAdd_{op.opclass}",
            [self._as_vector(), other._as_vector()],
            op=op,
            is_cscalar=False,
            scalar_as_vector=True,
        )

    def ewise_mult(self, other, op=binary.times):
        """Perform element-wise computation on the intersection of sparse values,
        similar to how one expects multiplication to work for sparse data.

        See the
        `Element-wise Intersection <../user_guide/operations.html#element-wise-intersection>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Scalar
            The other scalar in the computation; Python scalars also accepted
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use on intersecting values

        Returns
        -------
        ScalarExpression that will be empty if any of the inputs is empty

        Examples
        --------
        .. code-block:: python

            # Method syntax
            c << a.ewise_mult(b, op=binary.gt)

            # Functional syntax
            c << binary.gt(a & b)

        """
        return self._ewise_mult(other, op)

    def _ewise_mult(self, other, op=binary.times, is_infix=False):
        method_name = "ewise_mult"
        if is_infix:
            from .infix import ScalarEwiseMultExpr

            # This is a little different than how we handle ewise_mult for Vector and
            # Matrix where we are super-careful to handle dtypes well to support UDTs.
            # For Scalar, we're going to let dtypes in expressions resolve themselves.
            # Scalars are more challenging, because they may be literal scalars.
            # Also, we have not yet resolved `op` here, so errors may be different.
            if isinstance(self, ScalarEwiseMultExpr):
                self = op(self).new()
            if isinstance(other, ScalarEwiseMultExpr):
                other = op(other).new()

        if type(other) is not Scalar:
            dtype = self.dtype if self.dtype._is_udt else None
            try:
                other = Scalar.from_value(other, dtype, is_cscalar=False, name="")
            except TypeError:
                other = self._expect_type(
                    other,
                    Scalar,
                    within=method_name,
                    keyword_name="other",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        return ScalarExpression(
            method_name,
            f"GrB_Vector_eWiseMult_{op.opclass}",
            [self._as_vector(), other._as_vector()],
            op=op,
            is_cscalar=False,
            scalar_as_vector=True,
        )

    def ewise_union(self, other, op, left_default, right_default):
        """Perform element-wise computation on the union of sparse values,
        similar to how one expects subtraction to work for sparse data.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_union and :meth:`ewise_add`.

        Parameters
        ----------
        other : Scalar
            The other scalar in the computation; Python scalars also accepted
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use
        left_default :
            Scalar value to use when the index on the left is missing
        right_default :
            Scalar value to use when the index on the right is missing

        Returns
        -------
        ScalarExpression with a structure formed as the union of the input structures

        Examples
        --------
        .. code-block:: python

            # Method syntax
            c << a.ewise_union(b, op=binary.div, left_default=1, right_default=1)

            # Functional syntax
            c << binary.div(a | b, left_default=1, right_default=1)

        """
        return self._ewise_union(other, op, left_default, right_default)

    def _ewise_union(self, other, op, left_default, right_default, is_infix=False):
        method_name = "ewise_union"
        if is_infix:
            from .infix import ScalarEwiseAddExpr

            # This is a little different than how we handle ewise_union for Vector and
            # Matrix where we are super-careful to handle dtypes well to support UDTs.
            # For Scalar, we're going to let dtypes in expressions resolve themselves.
            # Scalars are more challenging, because they may be literal scalars.
            # Also, we have not yet resolved `op` here, so errors may be different.
            if isinstance(self, ScalarEwiseAddExpr):
                self = op(self, left_default=left_default, right_default=right_default).new()
            if isinstance(other, ScalarEwiseAddExpr):
                other = op(other, left_default=left_default, right_default=right_default).new()

        right_dtype = self.dtype
        dtype = right_dtype if right_dtype._is_udt else None
        if type(other) is not Scalar:
            try:
                other = Scalar.from_value(other, dtype, is_cscalar=False, name="")
            except TypeError:
                other = self._expect_type(
                    other,
                    Scalar,
                    within=method_name,
                    keyword_name="other",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        else:
            other = _as_scalar(other, dtype, is_cscalar=False)  # pragma: is_grbscalar

        temp_op = get_typed_op(op, self.dtype, other.dtype, kind="binary")

        left_dtype = temp_op.type
        dtype = left_dtype if left_dtype._is_udt else None
        if type(left_default) is not Scalar:
            try:
                left = Scalar.from_value(
                    left_default, dtype, is_cscalar=False, name=""  # pragma: is_grbscalar
                )
            except TypeError:
                left = self._expect_type(
                    left_default,
                    Scalar,
                    within=method_name,
                    keyword_name="left_default",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        else:
            left = _as_scalar(left_default, dtype, is_cscalar=False)  # pragma: is_grbscalar
        right_dtype = temp_op.type2
        dtype = right_dtype if right_dtype._is_udt else None
        if type(right_default) is not Scalar:
            try:
                right = Scalar.from_value(
                    right_default, dtype, is_cscalar=False, name=""  # pragma: is_grbscalar
                )
            except TypeError:
                right = self._expect_type(
                    right_default,
                    Scalar,
                    within=method_name,
                    keyword_name="right_default",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        else:
            right = _as_scalar(right_default, dtype, is_cscalar=False)  # pragma: is_grbscalar

        op1 = get_typed_op(op, self.dtype, right.dtype, kind="binary")
        op2 = get_typed_op(op, left.dtype, other.dtype, kind="binary")
        if op1 is not op2:
            left_dtype = unify(op1.type, op2.type, is_right_scalar=True)
            right_dtype = unify(op1.type2, op2.type2, is_left_scalar=True)
            op = get_typed_op(op, left_dtype, right_dtype, kind="binary")
        else:
            op = op1
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        expr_repr = "{0.name}.{method_name}({2.name}, {op}, {1._expr_name}, {3._expr_name})"
        if backend == "suitesparse":
            expr = ScalarExpression(
                method_name,
                "GxB_Vector_eWiseUnion",
                [self._as_vector(), left, other._as_vector(), right],
                op=op,
                expr_repr=expr_repr,
                is_cscalar=False,
                scalar_as_vector=True,
            )
        else:
            expr = ScalarExpression(
                method_name,
                None,
                [self, left, other, right, _s_union_s, (self, other, left, right, op)],
                op=op,
                expr_repr=expr_repr,
                is_cscalar=False,
                scalar_as_vector=True,
            )
        return expr

    def apply(self, op, right=None, *, left=None):
        """Create a new Scalar by applying ``op``.

        See the `Apply <../user_guide/operations.html#apply>`__
        section in the User Guide for more details.

        Common usage is to pass a :class:`~graphblas.core.operator.UnaryOp`,
        in which case ``right`` and ``left`` may not be defined.

        A :class:`~graphblas.core.operator.BinaryOp` can also be used, in
        which case a scalar must be passed as ``left`` or ``right``.

        An :class:`~graphblas.core.operator.IndexUnaryOp` can also be used
        with the thunk passed in as ``right``.

        Parameters
        ----------
        op : UnaryOp or BinaryOp or IndexUnaryOp
            Operator to apply
        right :
            Scalar used with BinaryOp or IndexUnaryOp
        left :
            Scalar used with BinaryOp

        Returns
        -------
        ScalarExpression

        Examples
        --------
        .. code-block:: python

            # Method syntax
            b << a.apply(op.abs)

            # Functional syntax
            b << op.abs(a)

        """
        expr = self._as_vector().apply(op, right, left=left)
        return ScalarExpression(
            expr.method_name,
            expr.cfunc_name,
            expr.args,
            op=expr.op,
            dtype=expr.dtype,
            expr_repr=expr.expr_repr,
            is_cscalar=False,
            scalar_as_vector=True,
        )

    def select(self, op, thunk=None):
        expr = self._as_vector().select(op, thunk)
        return ScalarExpression(
            expr.method_name,
            expr.cfunc_name,
            expr.args,
            op=expr.op,
            dtype=expr.dtype,
            expr_repr=expr.expr_repr,
            is_cscalar=False,
            scalar_as_vector=True,
        )


class ScalarExpression(BaseExpression):
    __slots__ = "_is_cscalar", "_scalar_as_vector"
    output_type = Scalar
    ndim = 0
    shape = ()
    _is_scalar = True

    def __init__(self, *args, is_cscalar, scalar_as_vector=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_cscalar = is_cscalar
        self._scalar_as_vector = scalar_as_vector

    def construct_output(self, dtype=None, *, is_cscalar=None, name=None):
        if dtype is None:
            dtype = self.dtype
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        return Scalar(dtype, is_cscalar=is_cscalar, name=name)

    def new(self, dtype=None, *, is_cscalar=None, name=None, **opts):
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        return super()._new(dtype, None, name, is_cscalar=is_cscalar, **opts)

    @wrapdoc(Scalar.dup)
    def dup(self, dtype=None, *, clear=False, is_cscalar=None, name=None, **opts):
        if dtype is None:
            dtype = self.dtype
        if is_cscalar is None:
            is_cscalar = self._is_cscalar
        if clear:
            return Scalar(dtype, is_cscalar=is_cscalar, name=name)
        return self.new(dtype, is_cscalar=is_cscalar, name=name, **opts)

    def __repr__(self):
        from .formatting import format_scalar_expression

        return format_scalar_expression(self)

    def _repr_html_(self):
        from .formatting import format_scalar_expression_html

        return format_scalar_expression_html(self)

    is_cscalar = Scalar.is_cscalar
    is_grbscalar = Scalar.is_grbscalar

    # Begin auto-generated code: Scalar
    _get_value = automethods._get_value
    __and__ = wrapdoc(Scalar.__and__)(property(automethods.__and__))
    __array__ = wrapdoc(Scalar.__array__)(property(automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(automethods.__int__))
    __ne__ = wrapdoc(Scalar.__ne__)(property(automethods.__ne__))
    __or__ = wrapdoc(Scalar.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Scalar.__rand__)(property(automethods.__rand__))
    __ror__ = wrapdoc(Scalar.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Scalar._as_matrix)(property(automethods._as_matrix))
    _as_vector = wrapdoc(Scalar._as_vector)(property(automethods._as_vector))
    _is_empty = wrapdoc(Scalar._is_empty)(property(automethods._is_empty))
    _name_html = wrapdoc(Scalar._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Scalar._nvals)(property(automethods._nvals))
    apply = wrapdoc(Scalar.apply)(property(automethods.apply))
    ewise_add = wrapdoc(Scalar.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Scalar.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Scalar.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Scalar.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Scalar.get)(property(automethods.get))
    is_empty = wrapdoc(Scalar.is_empty)(property(automethods.is_empty))
    isclose = wrapdoc(Scalar.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Scalar.isequal)(property(automethods.isequal))
    name = wrapdoc(Scalar.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Scalar.nvals)(property(automethods.nvals))
    select = wrapdoc(Scalar.select)(property(automethods.select))
    value = wrapdoc(Scalar.value)(property(automethods.value))
    wait = wrapdoc(Scalar.wait)(property(automethods.wait))
    # These raise exceptions
    __matmul__ = Scalar.__matmul__
    __rmatmul__ = Scalar.__rmatmul__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Scalar


class ScalarIndexExpr(AmbiguousAssignOrExtract):
    output_type = Scalar
    ndim = 0
    shape = ()
    _is_scalar = True
    _is_cscalar = False

    def new(self, dtype=None, *, is_cscalar=None, name=None, **opts):
        if is_cscalar is None:
            is_cscalar = False
        return self.parent._extract_element(
            self.resolved_indexes, dtype, opts, is_cscalar=is_cscalar, name=name
        )

    @wrapdoc(Scalar.dup)
    def dup(self, dtype=None, *, clear=False, is_cscalar=False, name=None, **opts):
        if dtype is None:
            dtype = self.dtype
        if clear:
            return Scalar(dtype, is_cscalar=is_cscalar, name=name)
        return self.new(dtype, is_cscalar=is_cscalar, name=name, **opts)

    is_cscalar = Scalar.is_cscalar
    is_grbscalar = Scalar.is_grbscalar

    # Begin auto-generated code: Scalar
    _get_value = automethods._get_value
    __and__ = wrapdoc(Scalar.__and__)(property(automethods.__and__))
    __array__ = wrapdoc(Scalar.__array__)(property(automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(automethods.__int__))
    __ne__ = wrapdoc(Scalar.__ne__)(property(automethods.__ne__))
    __or__ = wrapdoc(Scalar.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Scalar.__rand__)(property(automethods.__rand__))
    __ror__ = wrapdoc(Scalar.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Scalar._as_matrix)(property(automethods._as_matrix))
    _as_vector = wrapdoc(Scalar._as_vector)(property(automethods._as_vector))
    _is_empty = wrapdoc(Scalar._is_empty)(property(automethods._is_empty))
    _name_html = wrapdoc(Scalar._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Scalar._nvals)(property(automethods._nvals))
    apply = wrapdoc(Scalar.apply)(property(automethods.apply))
    ewise_add = wrapdoc(Scalar.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Scalar.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Scalar.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Scalar.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Scalar.get)(property(automethods.get))
    is_empty = wrapdoc(Scalar.is_empty)(property(automethods.is_empty))
    isclose = wrapdoc(Scalar.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Scalar.isequal)(property(automethods.isequal))
    name = wrapdoc(Scalar.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Scalar.nvals)(property(automethods.nvals))
    select = wrapdoc(Scalar.select)(property(automethods.select))
    value = wrapdoc(Scalar.value)(property(automethods.value))
    wait = wrapdoc(Scalar.wait)(property(automethods.wait))
    # These raise exceptions
    __matmul__ = Scalar.__matmul__
    __rmatmul__ = Scalar.__rmatmul__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Scalar


def _as_scalar(scalar, dtype=None, *, is_cscalar):
    if type(scalar) is not Scalar:
        return Scalar.from_value(scalar, dtype, is_cscalar=is_cscalar, name="")
    if scalar._is_cscalar != is_cscalar or dtype is not None and scalar.dtype != dtype:
        return scalar.dup(dtype, is_cscalar=is_cscalar, name=scalar.name)
    return scalar


def _dict_to_record(np_type, d):
    """Converts e.g. ``{"x": 1, "y": 2.3}`` to ``(1, 2.3)``."""
    rv = []
    for name, (dtype, _) in np_type.fields.items():
        val = d[name]
        if dtype.names is not None and isinstance(val, dict):
            rv.append(_dict_to_record(dtype, val))
        else:
            rv.append(val)
    return tuple(rv)


_MATERIALIZE = Scalar.from_value(lib.GrB_MATERIALIZE, is_cscalar=True, name="GrB_MATERIALIZE")
_COMPLETE = Scalar.from_value(lib.GrB_COMPLETE, is_cscalar=True, name="GrB_COMPLETE")

utils._output_types[Scalar] = Scalar
utils._output_types[ScalarIndexExpr] = Scalar
utils._output_types[ScalarExpression] = Scalar

# Import vector to import matrix to import infix to import infixmethods, which has side effects
from . import vector  # noqa: E402, F401 isort:skip

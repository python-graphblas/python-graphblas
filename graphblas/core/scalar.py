import itertools
import math
from functools import lru_cache

import numpy as np

from .. import backend, binary, config, monoid
from ..dtypes import _INDEX, FP64, _index_dtypes, lookup_dtype, unify
from ..exceptions import EmptyObject, check_status
from . import _has_numba, _supports_udfs, automethods, ffi, lib, utils
from .base import BaseExpression, BaseType, _is_recording, call
from .dtypes import _view_if_same_layout
from .expr import AmbiguousAssignOrExtract
from .operator import IndexUnaryOp, SelectOp, TypedOpBase, binary_from_string, get_typed_op
from .operator.udt_utils import _LITERAL_KINDS, _weak_literal_leaf
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

        Compares with ``binary.eq``, as ``==`` on a Vector or Matrix does, so the
        elements of array UDTs broadcast (an ``INT8[3]`` Scalar of ``[1, 1, 1]``
        equals ``1``), but returns a bool. Two empty Scalars are equal, and
        ``s == None`` checks whether ``s`` is empty. :meth:`isequal` is stricter.
        Without Numba, which ``binary.eq`` needs for UDTs, a UDT's values are
        compared in numpy, which broadcasts them the same way.
        """
        return _scalar_eq(self, other)

    __hash__ = None

    def __ne__(self, other):
        return not _scalar_eq(self, other)

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
        raise TypeError(f"Unable to get size of GrB_Scalar with backend: {backend}")

    def isequal(self, other, *, check_dtype=False):
        """Check for exact equality (including whether the value is missing).

        Parameters
        ----------
        other : Scalar
            Scalar to compare against
        check_dtype : bool, default=False
            If True, also checks that dtypes match

        Returns
        -------
        bool

        Notes
        -----
        Elements of array UDTs must have the same shape, apart from leading axes
        of length 1, as in ``np.array_equal``; ``==`` broadcasts instead.

        See Also
        --------
        :meth:`isclose` : For equality check of floating point dtypes

        """
        if type(other) is not Scalar:
            if other is None:
                return self._is_empty
            dtype = None
            if self.dtype._is_udt:
                # A literal is compared as given, as ``==`` types it, not converted
                # into the UDT (0.5 would become 0 in an int field), and an array
                # literal must have the element's shape, as in np.array_equal.
                if _literal_shape_differs(self.dtype, other):
                    return False
                dtype = _literal_type(self.dtype, other, exact=True) or self.dtype
            try:
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
        if _element_shapes_differ(self.dtype, other.dtype):
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
        if self.dtype._is_udt:
            raise TypeError(
                f"Scalar.isclose is not defined for user-defined types (got {self.dtype.name!r}); "
                "use isequal for exact comparison."
            )
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
            base, shape = np_type.subdtype
            return rv.view(base).reshape(shape)
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
                # Zeros, not empty: when numpy writes only the fields, a
                # record's alignment padding is whatever the allocation
                # happened to contain. ``_udt_bytes`` zeros the padding in the
                # other direction, when numpy copies whole items instead.
                if np_type.subdtype is None:
                    arr = np.zeros(1, dtype=np_type)
                    if isinstance(val, dict) and np_type.names is not None:
                        val = _dict_to_record(np_type, val)
                else:
                    arr = np.zeros(np_type.subdtype[1], dtype=np_type.subdtype[0])
                val = _view_if_same_layout(val, arr.dtype)
                if _has_narrow_float_leaf(np_type) and np.geterr()["over"] == "warn":
                    # A float too large for a float32 leaf is infinity, and numpy
                    # warns; warn at the caller's line instead, as a built-in
                    # FP32 does, so the default filter shows each place it happens.
                    try:
                        with np.errstate(over="raise"):
                            arr[:] = val
                    except FloatingPointError:
                        utils._warn_from_caller(
                            f"overflow encountered in cast: {val!r} as {self.dtype}",
                            RuntimeWarning,
                        )
                        with np.errstate(over="ignore"):
                            arr[:] = val
                else:
                    arr[:] = val
                self.gb_obj[0 : np_type.itemsize] = _udt_bytes(np_type, arr)
            else:
                self.gb_obj[0] = val
            self._empty = False
        else:
            if self.dtype._is_udt:
                val = _Pointer(_as_scalar(val, self.dtype, is_cscalar=True))
                dtype_name = "UDT"
            else:
                val = _as_scalar(val, _literal_store_dtype(self.dtype, val), is_cscalar=True)
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
                if self.dtype._is_udt:
                    # Cast as storing it would (``base._store_cast_op``), not as numpy
                    # does; a UDT does not cast to a built-in dtype at all.
                    new_scalar << self
                elif new_scalar.is_cscalar and not new_scalar.dtype._is_udt:
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
            other, dtype = _literal_operand(self.dtype, other, op)
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
        if _ewise_add_needs_cast(op, self.dtype, other.dtype):
            if not _is_lifted_op(op.parent):
                raise _ewise_add_cast_error(op, self.dtype, other.dtype)
            # GraphBLAS casts an unpaired entry to the result type, which a UDT
            # cannot be. Convert both to the result type first, which changes no
            # value, and apply the op in that one type: an int8 UDT plus 0.5 is
            # float64, and an empty Scalar still adds as 0. A Scalar is one
            # element, so the converted copies cost nothing.
            if self.dtype != op.return_type:
                self = _converted_to(self, op.return_type, name="")
            if other.dtype != op.return_type:
                other = _converted_to(other, op.return_type, name="")
            op = get_typed_op(op.parent, self.dtype, other.dtype, kind="binary")
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
            other, dtype = _literal_operand(self.dtype, other, op)
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
            other, literal_dtype = _literal_operand(right_dtype, other, op)
            try:
                other = Scalar.from_value(other, literal_dtype, is_cscalar=False, name="")
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
            # A typed Scalar keeps its dtype beside a lifted op, as it does for
            # Vector.ewise_union: ``s8 - sf`` is float, not ``sf`` cut to int8.
            if dtype is not None and other.dtype != dtype and _is_lifted_op(op):
                dtype = None
            _check_scalar_fits(dtype, other)
            other = _as_scalar(other, dtype, is_cscalar=False)  # pragma: is_grbscalar

        temp_op = get_typed_op(op, self.dtype, other.dtype, kind="binary")

        left_dtype = temp_op.type
        dtype = left_dtype if left_dtype._is_udt else None
        if type(left_default) is not Scalar:
            if dtype is not None:
                _check_literal_fits(dtype, left_default)
            else:
                left_default, dtype = _literal_operand(left_dtype, left_default, op)
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
            _check_scalar_fits(dtype, left_default)
            left = _as_scalar(left_default, dtype, is_cscalar=False)  # pragma: is_grbscalar
        right_dtype = temp_op.type2
        dtype = right_dtype if right_dtype._is_udt else None
        if type(right_default) is not Scalar:
            if dtype is not None:
                _check_literal_fits(dtype, right_default)
            else:
                right_default, dtype = _literal_operand(right_dtype, right_default, op)
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
            _check_scalar_fits(dtype, right_default)
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
    __hash__ = None

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
        parent = self.parent
        # Fast path for the default `expr.new()`: extract a single element
        # straight into a fresh GrB_Scalar via GrB_*_extractElement_Scalar,
        # skipping the `call` wrapper's per-arg _carg marshalling. Falls back
        # for a dtype cast, cscalar output, opts, UDTs, and an active Recorder
        # (so the call is recorded). Result is a GrB_Scalar (is_cscalar=False),
        # empty exactly when the element is missing, same as _extract_element.
        if (
            dtype is None
            and not is_cscalar
            and not opts
            and not parent.dtype._is_udt
            and not _is_recording()
        ):
            indices = self.resolved_indexes.indices
            result = Scalar(parent.dtype, is_cscalar=False, name=name)
            if len(indices) == 1:
                err_code = lib.GrB_Vector_extractElement_Scalar(
                    result.gb_obj[0], parent.gb_obj[0], indices[0].index._carg
                )
            else:
                rowidx, colidx = indices
                if parent._is_transposed:
                    rowidx, colidx = colidx, rowidx
                err_code = lib.GrB_Matrix_extractElement_Scalar(
                    result.gb_obj[0], parent.gb_obj[0], rowidx.index._carg, colidx.index._carg
                )
            if err_code:
                check_status(err_code, [result])
            return result
        return parent._extract_element(
            self.resolved_indexes, dtype, opts, is_cscalar=is_cscalar, name=name
        )

    @wrapdoc(Scalar.dup)
    def dup(self, dtype=None, *, clear=False, is_cscalar=False, name=None, **opts):
        if dtype is None:
            dtype = self.dtype
        if clear:
            return Scalar(dtype, is_cscalar=is_cscalar, name=name)
        return self.new(dtype, is_cscalar=is_cscalar, name=name, **opts)

    def _extract_fast(self):
        """Resolve a value read (``.value``, ``float(...)``, ...) with one extract.

        Those readers only need the raw element, so extract it straight into a
        cscalar and skip the extra GrB_Scalar round-trip that ``.new()`` followed
        by ``Scalar.value`` would perform. Defer to the full ``.new()`` for UDTs
        (whose values need numpy conversion in ``Scalar.value``) and while a
        Recorder is active (so it observes the same calls as the expression path).
        ``automethods._get_value`` consults this hook for ``_fast_scalar_attrs``.
        """
        parent = self.parent
        if parent.dtype._is_udt or _is_recording():
            return self.new()
        return parent._extract_element(self.resolved_indexes, None, {}, is_cscalar=True)

    is_cscalar = Scalar.is_cscalar
    is_grbscalar = Scalar.is_grbscalar
    __hash__ = None

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


def _literal_operand(dtype, value, op):
    """Return literal ``value`` and the dtype to make it, as ``op``'s operand beside ``dtype``.

    A dtype of ``None`` means the value's own dtype. This is where
    python-graphblas decides how a literal operand is typed; beside a built-in
    dtype, see :func:`_weak_builtin_literal_dtype`.

    Beside a UDT, a literal becomes an element of that UDT, as it must for a
    user's op, when its type allows it (:func:`_check_literal_fits`). With a
    built-in op that lifts to UDTs, a literal is an operand in its own right,
    typed as numpy 2 types it (NEP 50), by :func:`_literal_type`:

    - a numpy scalar or array keeps its dtype (strong); an array's type is the
      array layout of its dtype (``udt_utils._array_literal_udt``);
    - a Python ``bool``, ``int``, ``float`` or ``complex`` is weak: in each
      element or leaf it takes that element's dtype when it fits the kind, so
      ``int8_udt + 1`` stays ``int8_udt`` and ``int8_udt += 1`` can be stored,
      while ``int8_udt * 2.5`` has float64 elements (see
      ``udt_utils._weak_literal_udt``), where converting it to the UDT would
      multiply by 2;
    - a tuple, list or dict of Python numbers, laid out like an element, is
      weak in the same way, leaf by leaf: ``int8_udt + (1, 2, 3)`` stays
      ``int8_udt`` and ``+ (0.5, 1.5, 2.5)`` has float64 elements.

    ``eq`` and ``ne`` type a literal the same way, as numpy compares, so
    ``fp32_udt == 0.1`` compares in float32, as ``fp32_vec == 0.1`` does.

    A comparison keeps no result type, so a Python int beyond an integer
    element's range is compared exactly, as numpy 2 compares it, as infinity
    of its sign: every element compares with that infinity as with the int,
    while a type that held both would round them (in FP64, the INT64
    ``2**63 - 1`` and the int ``2**63`` are equal). So ``int64_vec < 2**63`` is
    all True and ``int8_vec == 300`` all False. Beside a UDT, where only ``eq``
    and ``ne`` compare, a literal with such an int in an integer leaf equals no
    element: ``int8_udt == 300`` is False, not an error.
    """
    if getattr(op, "is_positional", False):
        return value, None
    if not dtype._is_udt:
        literal_dtype = _weak_builtin_literal_dtype(dtype, value, op)
        if (
            literal_dtype is not dtype
            and literal_dtype is not None
            and type(value) is int
            and dtype.np_type.kind in "iu"
        ):
            # Only an int beyond dtype's range takes another type here, typed so
            # by a comparison or by an op that gives another type (truediv).
            if isinstance(op, str):
                op = binary_from_string(op)
            if op.name in _COMPARISONS:
                return (math.inf if value > 0 else -math.inf), FP64
        return value, literal_dtype
    if getattr(op, "_udt_types", True) is None:
        # An op that does not take UDTs at all (``lt``) reports that itself.
        return value, dtype
    from .operator.binary import BinaryOp
    from .operator.monoid import Monoid
    from .operator.udt_utils import BUILTIN_UDT_BINARY_OPS, _int_beyond_leaf

    if isinstance(op, Monoid) and not op._anonymous:
        # Element-wise, a Monoid is its BinaryOp (``get_typed_op`` types it so).
        op = op.binaryop
    if type(op) is BinaryOp and not op._anonymous:
        is_comparison = op.name in ("eq", "ne")
        if is_comparison or op.name in BUILTIN_UDT_BINARY_OPS:
            # truediv gives float leaves whatever the int, so, as a comparison, it
            # types an int out of an integer leaf's range exactly instead of raising.
            exact = is_comparison or op.name == "truediv"
            literal_type = _literal_type(dtype, value, exact=exact)
            if literal_type is not None:
                if is_comparison and not isinstance(value, (np.generic, np.ndarray)):
                    np_type = dtype.np_type
                    parts = _dict_to_record(np_type, value) if isinstance(value, dict) else value
                    beyond = _int_beyond_leaf(np_type, parts)
                    if beyond is not None:
                        # As infinity in every leaf, it still equals no element.
                        value = math.inf if beyond > 0 else -math.inf
                        literal_type = _literal_type(dtype, value)
                return value, literal_type
    _check_literal_fits(dtype, value)
    return value, dtype


# Built-in ops that compare their operands, and so keep no result type.
_COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge"}
_COMPARISONS.update([f"value{name}" for name in list(_COMPARISONS)])


# The types a Python float or complex can overflow.
_SINGLE_PRECISION = {np.dtype(np.float32), np.dtype(np.complex64)}


def _weak_builtin_literal_dtype(dtype, value, op=None):
    """Return the dtype of Python number ``value`` beside built-in ``dtype``, or ``None``.

    A Python ``bool``, ``int``, ``float`` or ``complex`` is weak, as numpy 2
    types it (NEP 50), by the table UDT leaves use
    (``udt_utils._weak_literal_leaf``): ``int8_vec + 1`` is INT8,
    ``fp32_vec * 0.5`` is FP32, ``int8_vec * 0.5`` is FP64, and
    ``int8_vec + 300`` raises ``OverflowError``. python-graphblas applies the
    table itself, so numpy 1 gives the same answers. ``None`` keeps the value's
    own dtype (strong): for anything but a Python number, for a typed op, which
    fixes the operand types itself, and for the thunk of an index op that does
    not compare it with the values, such as a row index.

    A comparison keeps no result type, so an int beyond ``dtype``'s range
    takes a type that holds it rather than raise (:func:`_literal_operand` then
    compares it as infinity). ``op`` of ``None``, for a fill value, does the
    same, so the value is not lost.
    """
    kind = _LITERAL_KINDS.get(type(value))
    if kind is None:
        return None
    if isinstance(op, str):
        try:
            op = binary_from_string(op)
        except Exception:
            return None
    exact = op is None
    if op is not None:
        if isinstance(op, TypedOpBase):
            return None
        exact = getattr(op, "name", None) in _COMPARISONS and not op._anonymous
        if isinstance(op, (IndexUnaryOp, SelectOp)) and not exact:
            return None
    try:
        np_type = _weak_literal_leaf([value], kind, dtype.np_type, exact=exact)
    except OverflowError:
        if exact or not _gives_another_type(op, dtype):
            raise
        # An op whose result is not the operand's type has no narrow result to
        # keep, so an int out of range is typed exactly, as for a comparison:
        # int8_vec / 300 is FP64, as numpy divides integers in float64.
        np_type = _weak_literal_leaf([value], kind, dtype.np_type, exact=True)
    if np_type == dtype.np_type:
        rv = dtype
    else:
        try:
            rv = lookup_dtype(np_type)
        except ValueError:  # A complex literal where the backend has no complex types
            return None
    if kind != "b" and rv.np_type in _SINGLE_PRECISION:
        import struct

        parts = (value.real, value.imag) if kind == "c" else (float(value),)
        try:
            for part in parts:
                struct.pack("<f", part)  # raises when a finite part rounds to infinity
        except OverflowError:
            if op is None:
                # A fill value widens rather than lose its value, as an int out of
                # range does: FP32's to_dense(fill_value=1e300) is float64.
                return lookup_dtype(np.complex128 if kind == "c" else np.float64)
            # As numpy 2 and a UDT warn: fp32_vec + 1e300 (or + 2**200) is inf, and
            # so is the 1e300 of fp32_vec == 1e300, which numpy compares with inf too.
            utils._warn_from_caller(
                f"overflow encountered in cast: {value!r} is infinity as {rv}", RuntimeWarning
            )
    return rv


def _gives_another_type(op, dtype):
    """Whether built-in ``op`` on two operands of built-in ``dtype`` gives another type.

    ``truediv`` on integers gives FP64, and the numpy ufuncs that compute in
    float give a float type, as numpy computes them in a float loop.
    """
    if op is None or getattr(op, "_anonymous", True) or not hasattr(op, "__getitem__"):
        return False
    try:
        return op[dtype].return_type != dtype
    except Exception:  # An op that does not take this dtype reports it itself, later.
        return False


def _literal_type(dtype, value, *, exact=False):
    """Return the type of literal ``value`` beside UDT ``dtype``, or ``None`` when it has none.

    A Python number (a ``Fraction`` or ``Decimal`` counts as the int or float
    it is worth) or a tuple, list or dict of them is weak, leaf by leaf
    (``udt_utils._weak_literal_udt``); with ``exact``, for a comparison, an int
    out of a leaf's range takes a type that holds it. A numpy value is
    strong: its dtype, in the array layout of its shape for an array
    (``udt_utils._array_literal_udt``); a sequence holding numpy values is the
    array numpy makes of it. Anything else has no type here, and
    ``Scalar.from_value`` reports it.
    """
    from .operator.udt_utils import (
        _LITERAL_KINDS,
        _array_literal_udt,
        _as_python_number,
        _weak_literal_udt,
    )

    value = _as_python_number(value)
    np_type = dtype.np_type
    if isinstance(value, dict) and np_type.names is not None:
        value = _dict_to_record(np_type, value)
    if type(value) in _LITERAL_KINDS or isinstance(value, (tuple, list)):
        literal_type = _weak_literal_udt(dtype, value, exact=exact)
        if literal_type is not None or type(value) in _LITERAL_KINDS:
            return literal_type
        try:
            value = np.asarray(value)
        except (TypeError, ValueError):  # ragged
            return None
    if isinstance(value, (np.bool_, np.number)):
        return lookup_dtype(value.dtype)
    if isinstance(value, (np.ndarray, np.void)):
        if value.ndim == 0 and value.dtype.kind in "biufc":
            return lookup_dtype(value.dtype)
        return _array_literal_udt(dtype, value)
    return None


def _check_literal_fits(dtype, value):
    """Raise unless literal ``value`` may become an element of UDT ``dtype``, by its type.

    A literal becomes an element of the UDT when nothing else types it: beside
    a user-defined op, as an IndexUnaryOp thunk, or as an ``ewise_union``
    default, which GraphBLAS casts to the op's input type, and a UDT casts only
    to itself. Its type decides, not its values (:func:`_literal_type`): a
    Python number or a sequence of them is weak, so ``1`` and ``(1, 2, 3)``
    beside ``INT8[3]`` are ``INT8[3]`` and fit, ``0.5`` is ``FP64[3]`` and does
    not, and ``300`` is ``OverflowError``; a numpy value is strong and fits when
    each leaf casts safely into the UDT's (``np.can_cast``), so ``np.int8(1)``
    fits ``INT8[3]`` and ``np.int64(1)`` does not. ``Scalar.from_value`` then
    converts it, which changes no value. A literal that does not fit by type but
    converts exactly still converts, with a DeprecationWarning
    (:func:`_refuse_unless_exact`). A ``dtype`` of ``None``, a built-in type,
    takes any literal.
    """
    if dtype is None:
        return
    from .operator.udt_utils import _casts_safely_into

    literal_type = _literal_type(dtype, value)
    if literal_type is not None and not _casts_safely_into(literal_type, dtype):
        _refuse_unless_exact(
            dtype,
            value,
            f"{value!r} does not fit {dtype}: it is typed as {literal_type}, and a literal "
            "becomes an element of a UDT only when its type is that UDT or casts into it "
            "safely, field by field (a Python number or sequence takes a field's dtype when "
            "the field's kind holds it).",
            "Write it in the element's types, such as False rather than 0 for a bool field "
            "and 2 rather than 2.0 for an integer field.",
        )
        return
    # A weak float fits a float32 field by type, but a finite one too large for
    # it would become infinity; numpy warns as it converts, on numpy 1 and 2.
    import warnings

    np_type = dtype.np_type
    if np_type.names is not None:
        target = np.zeros(1, dtype=np_type)
        if isinstance(value, dict):
            value = _dict_to_record(np_type, value)
    else:
        target = np.zeros(np_type.subdtype[1], dtype=np_type.subdtype[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            target[...] = value
        except RuntimeWarning:
            raise OverflowError(
                f"{value!r} overflows {dtype}: a literal that becomes an element of a UDT "
                "must not turn a finite number into infinity."
            ) from None
        except (TypeError, ValueError):
            pass  # Scalar.from_value reports these.


def _check_scalar_fits(dtype, scalar):
    """Raise unless a typed Scalar casts safely into UDT ``dtype``, as a literal must.

    An ``ewise_union`` default must have the op's input type, so it is
    converted; the same rule as for a literal (:func:`_check_literal_fits`).
    """
    from .operator.udt_utils import _casts_safely_into

    if (
        dtype is not None
        and scalar.dtype != dtype
        and not scalar._is_empty
        and not _casts_safely_into(scalar.dtype, dtype)
    ):
        _refuse_unless_exact(
            dtype,
            scalar.value,
            f"a Scalar of type {scalar.dtype} does not fit {dtype}: it is used where only "
            "that UDT can go, and casts into it only when every field casts safely.",
            "Convert it first with .dup(dtype=...).",
        )


def _refuse_unless_exact(dtype, value, reason, advice):
    """Raise ValueError ``reason``, or only warn if ``value`` converts into ``dtype`` exactly.

    MAINT 2026-10-07: a literal or Scalar that must become an element of a UDT
    must fit it by type since #591. Before, any value was converted and what did
    not fit was lost; one that converts exactly, such as ``0`` for a bool field
    or ``2.0`` for an integer field, still converts, with a DeprecationWarning.
    Eight months after the release that adds the warning (the deprecation policy
    in docs/getting_started/faq.rst), always raise here and delete
    ``_converts_exactly``, ``_object_layout`` and ``utils._warn_from_caller``.
    """
    if not _converts_exactly(dtype, value):
        raise ValueError(reason)
    utils._warn_from_caller(
        f"{reason} Its values convert exactly, so it is converted for now; this is "
        f"deprecated and will raise ValueError in a future version. {advice}",
        DeprecationWarning,
    )


def _converts_exactly(dtype, value):
    """Whether converting ``value`` into UDT ``dtype`` keeps every number in it.

    A float may round to a narrower float, as numpy 2 rounds a Python number
    beside float32, but may not become infinite or lose an imaginary part.
    """
    import warnings

    np_type = dtype.np_type
    try:
        if np_type.names is not None:
            if isinstance(value, dict):
                value = _dict_to_record(np_type, value)
            given = np.zeros(1, dtype=_object_layout(np_type))
            converted = np.zeros(1, dtype=np_type)
        else:
            base, shape = np_type.subdtype
            given = np.empty(shape, dtype=object)
            converted = np.zeros(shape, dtype=base)
        given[:] = value
        with warnings.catch_warnings():
            # numpy warns for some lossy conversions; the comparison below finds each.
            warnings.simplefilter("ignore")
            converted[:] = _view_if_same_layout(value, converted.dtype)
    except (TypeError, ValueError, OverflowError):
        return False
    for given_leaf, converted_leaf in zip(
        _leaf_arrays(given), _leaf_arrays(converted), strict=True
    ):
        kind = converted_leaf.dtype.kind
        if kind not in "biufc":
            continue
        for x, y in zip(given_leaf.ravel().tolist(), converted_leaf.ravel().tolist(), strict=True):
            if kind in "fc":
                z = complex(x)
                if (kind == "f" and z.imag != 0) or (np.isfinite(z) and not np.isfinite(y)):
                    return False
            elif x != y:
                return False
    return True


def _object_layout(np_type):
    """Return ``np_type`` with every leaf an object, keeping field names, nesting and shapes."""
    if np_type.names is not None:
        return np.dtype([(name, _object_layout(np_type.fields[name][0])) for name in np_type.names])
    if np_type.subdtype is not None:
        base, shape = np_type.subdtype
        return np.dtype((_object_layout(base), shape))
    return np.dtype(object)


def _is_lifted_op(op):
    """Whether ``op`` is a built-in BinaryOp that lifts to UDTs, or a Monoid of one."""
    from .operator.binary import BinaryOp
    from .operator.monoid import Monoid
    from .operator.udt_utils import BUILTIN_UDT_BINARY_OPS

    if isinstance(op, Monoid):
        op = op.binaryop
    return type(op) is BinaryOp and not op._anonymous and op.name in BUILTIN_UDT_BINARY_OPS


def _leaf_arrays(arr):
    """Yield the array of each leaf of a structured array, in field order."""
    if arr.dtype.names is None:
        yield arr
    else:
        for name in arr.dtype.names:
            yield from _leaf_arrays(arr[name])


def _ewise_add_needs_cast(op, left_dtype, right_dtype):
    """Whether ``ewise_add`` with ``op`` would have GraphBLAS cast to or from a UDT.

    eWiseAdd copies an entry present in only one input into the result, cast
    to the op's output type, and GraphBLAS checks that cast before it looks at
    the data. A UDT casts only to itself, so such a pair is a bare
    GrB_DOMAIN_MISMATCH even when every entry has a partner.
    """
    out = op.return_type
    if not (out._is_udt or left_dtype._is_udt or right_dtype._is_udt):
        return False
    return left_dtype != out or right_dtype != out


def _ewise_add_cast_error(op, left_dtype, right_dtype):
    """Return the ``DomainMismatch`` for an ``ewise_add`` that would need a UDT cast."""
    from ..exceptions import DomainMismatch

    return DomainMismatch(
        f"ewise_add cannot use {op.parent!r} on {left_dtype} and {right_dtype}: an entry "
        f"present in only one input is copied into the result as {op.return_type}, and "
        "GraphBLAS cannot cast a UDT, so this would need converted copies of the operands. "
        "Use ewise_union, which takes a default for each side (infix + does, with zeros), or "
        "ewise_mult; or convert an operand first with .dup(dtype=...)."
    )


def _scalar_eq(scalar, other):
    """Return ``scalar == other`` as a bool, by ``binary.eq`` (see ``Scalar.__eq__``)."""
    if other is None:
        return scalar._is_empty
    if type(other) is not Scalar and output_type(other) is Scalar:
        other = other.new(name="s_eq_other")  # an expression, such as ``-s`` or ``v[0]``
    if type(other) is Scalar and (scalar._is_empty or other._is_empty):
        return scalar._is_empty and other._is_empty
    if scalar._is_empty:
        return False
    if not scalar.dtype._is_udt:
        # For built-in dtypes, compare the two values in the type ``eq`` is typed
        # for, which is what it computes, without a GraphBLAS call: ``s == 0``
        # takes about 2 us instead of 11.
        if type(other) is Scalar:
            other_dtype, value = other.dtype, other.value
        else:
            value, other_dtype = _literal_operand(scalar.dtype, other, binary.eq)
        if other_dtype is not None and not other_dtype._is_udt:
            if other_dtype is not scalar.dtype:
                other_dtype = get_typed_op(binary.eq, scalar.dtype, other_dtype).type
            convert = other_dtype.np_type.type
            if other_dtype.np_type in _SINGLE_PRECISION:
                # 1e300 beside FP32 is infinity, and _literal_operand has
                # warned at the caller's line, so numpy need not warn again.
                with np.errstate(over="ignore"):
                    return (convert(scalar.value) == convert(value)).item()
            return (convert(scalar.value) == convert(value)).item()
    if not _has_numba and (scalar.dtype._is_udt or (type(other) is Scalar and other.dtype._is_udt)):
        # binary.eq on a UDT needs Numba, so compare the values in numpy, as ==
        # on a Scalar did before it used binary.eq, with the literal typed as
        # binary.eq types it, and leaf by leaf as binary.eq compares.
        from .operator.udt_utils import _check_udt_pair, _get_udt_info

        if type(other) is not Scalar:
            value, dtype = _literal_operand(scalar.dtype, other, binary.eq)
            other = Scalar.from_value(value, dtype, is_cscalar=None, name="s_eq_other")
        # Two UDTs pair as for binary.eq, or raise its KeyError (records with other
        # field names, array shapes that do not broadcast, a record and an array).
        info = _get_udt_info(scalar.dtype), _get_udt_info(other.dtype)
        _check_udt_pair("eq", scalar.dtype, other.dtype, *info)
        return _leaves_equal(scalar.value, scalar.dtype.np_type, other.value, other.dtype.np_type)
    return bool(scalar.ewise_mult(other, binary.eq).new(name="s_eq").value)


def _leaves_equal(x, x_type, y, y_type):
    """Whether values ``x`` and ``y`` are equal in every leaf, as ``binary.eq`` compares them.

    For ``==`` on a UDT Scalar without Numba. Records pair their fields in
    order (the caller checks, by ``_check_udt_pair``, that two records pair as
    for ``binary.eq``), and a value that is not a record stands in every
    field. Each pair of leaves compares in the type their dtypes promote to,
    whatever the values (numpy 1 types a scalar by its value), and broadcasts.
    """
    if x_type.names is not None or y_type.names is not None:
        x_names = x_type.names or [None] * len(y_type.names)
        y_names = y_type.names or [None] * len(x_type.names)
        return all(
            _leaves_equal(
                x if x_name is None else x[x_name],
                x_type if x_name is None else x_type.fields[x_name][0],
                y if y_name is None else y[y_name],
                y_type if y_name is None else y_type.fields[y_name][0],
            )
            for x_name, y_name in zip(x_names, y_names, strict=True)
        )
    x = np.asarray(x, dtype=x_type.base)
    y = np.asarray(y, dtype=y_type.base)
    common = np.result_type(x.dtype, y.dtype)
    return bool(np.all(x.astype(common) == y.astype(common)))


def _element_shapes_differ(dtype1, dtype2):
    """Whether elements of ``dtype1`` and ``dtype2`` differ in shape, for ``isequal``.

    ``==`` on array UDTs broadcasts, as numpy's does, but ``isequal`` asks
    whether two values are the same array, as ``np.array_equal`` does: the
    shapes must match, apart from leading axes of length 1 (the rule a store
    uses), and an element of a built-in dtype has shape ``()``. Records pair
    field by field instead (``udt_utils._check_udt_pair``).
    """
    if dtype1 is dtype2 or not (dtype1._is_udt or dtype2._is_udt):
        return False
    from .operator.udt_utils import _strip_leading_ones

    shapes = []
    for np_type in (dtype1.np_type, dtype2.np_type):
        if np_type.names is not None:
            return False
        shapes.append(_strip_leading_ones(np_type.subdtype[1]) if np_type.subdtype else ())
    return shapes[0] != shapes[1]


def _literal_shape_differs(dtype, value):
    """Whether literal ``value`` differs in shape from an element of array UDT ``dtype``.

    As :func:`_element_shapes_differ`, for ``Scalar.isequal`` with a literal: a
    number has shape ``()``, so it equals an ``FP64[1]`` element but not an
    ``FP64[3]`` one, as in ``np.array_equal``. ``False`` for a record UDT.
    """
    from .operator.udt_utils import _strip_leading_ones

    np_type = dtype.np_type
    if np_type.subdtype is None or isinstance(value, dict):
        return False
    try:
        shape = np.shape(value)
    except ValueError:  # ragged; Scalar.from_value reports it
        return False
    return _strip_leading_ones(shape) != _strip_leading_ones(np_type.subdtype[1])


def _plus_zero(dtype):
    """Return a Scalar of ``dtype`` that ``plus`` adds to any value without changing it.

    Its float and complex leaves are ``-0.0``, because ``-0.0 + x`` is ``x``
    for every float, where ``0.0 + -0.0`` is ``0.0``.
    """
    np_type = dtype.np_type
    if np_type.subdtype is not None:
        zero = np.zeros(np_type.subdtype[1], dtype=np_type.subdtype[0])
    else:
        zero = np.zeros(1, dtype=np_type)
    for leaf in _leaf_arrays(zero):
        if leaf.dtype.kind in "fc":
            leaf[...] = complex(-0.0, -0.0) if leaf.dtype.kind == "c" else -0.0
    return Scalar.from_value(zero if np_type.subdtype is not None else zero[0], dtype, name="")


def _converted_to(operand, dtype, **new_kwargs):
    """Return ``operand`` converted to UDT ``dtype``, which holds its elements, value for value.

    It adds ``dtype``'s zero on the left with the lifted ``plus``. With the
    zero on the left, the result is ``dtype`` itself even when the operand has
    the same elements in another layout (a packed and an aligned record),
    since a left operand that holds the promoted elements is the result type.
    """
    rv = operand.apply(binary.plus, left=_plus_zero(dtype)).new(**new_kwargs)
    if rv.dtype != dtype:  # pragma: no cover (safety)
        from ..exceptions import DomainMismatch

        raise DomainMismatch(f"cannot convert {operand.dtype} to {dtype}")
    return rv


def _literal_store_dtype(dtype, value):
    """Return the dtype to make literal ``value`` a Scalar of, to store it as ``dtype``.

    A UDT types the literal itself. A Python int is INT64 by itself, so one
    beyond INT64 (the top half of UINT64) is made ``dtype``:
    ``uint64_vec[0] = 2**64 - 1`` works, and a type that cannot hold it raises.
    ``None`` lets ``Scalar.from_value`` choose.
    """
    if dtype._is_udt or type(value) is int and not -(2**63) <= value < 2**63:
        return dtype
    return None


def _cast_for_store(value, dtype):
    """Return Scalar, Vector or Matrix ``value``, or its expression, to assign as ``dtype``.

    GraphBLAS's assign and setElement take a UDT value as raw bytes, so one of
    another type would be read as the wrong type; ``base._store_cast_op``
    casts it or raises. Assign cannot cast as it writes, so a value of
    another UDT type needs a converted copy: one element for a Scalar, which
    is made here, but the whole value for a Vector or Matrix, which raises.
    """
    from .base import _store_cast_op

    if (cast_op := _store_cast_op(value.dtype, dtype)) is None:
        return value
    if output_type(value) is not Scalar:
        from ..exceptions import DomainMismatch

        raise DomainMismatch(
            f"cannot assign a value of type {value.dtype} into an object of type {dtype}: "
            "assign cannot cast a UDT as it writes, so this would need a converted copy of "
            "the whole value. Make it explicitly with .dup(dtype=...) first."
        )
    if not isinstance(value, BaseType):
        value = value.new(name="")
    return value.apply(cast_op).new(name=value.name)


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


@lru_cache
def _has_narrow_float_leaf(np_type):
    """Whether ``np_type`` has a float leaf narrower than float64, which a float may overflow."""
    if np_type.subdtype is not None:
        return _has_narrow_float_leaf(np_type.subdtype[0])
    if np_type.names is not None:
        return any(_has_narrow_float_leaf(np_type.fields[name][0]) for name in np_type.names)
    kind, size = np_type.kind, np_type.itemsize
    return (kind == "f" and size < 8) or (kind == "c" and size < 16)


@lru_cache
def _padding_ranges(np_type):
    """The ``(start, stop)`` byte ranges of ``np_type`` that are alignment padding.

    Walks records, nested records, and array dtypes, and returns the gaps
    between the field data.
    """
    data = []
    _collect_data_ranges(np_type, 0, data)
    data.sort()
    ranges = []
    pos = 0
    for start, stop in data:
        if start > pos:
            ranges.append((pos, start))
        pos = max(pos, stop)
    if pos < np_type.itemsize:
        ranges.append((pos, np_type.itemsize))
    return tuple(ranges)


def _collect_data_ranges(np_type, offset, out):
    """Append the ``(start, stop)`` byte ranges of ``np_type`` that hold field data."""
    if np_type.subdtype is not None:
        base = np_type.subdtype[0]
        for start in range(offset, offset + np_type.itemsize, base.itemsize):
            _collect_data_ranges(base, start, out)
    elif np_type.names is not None:
        for field_type, field_offset in (np_type.fields[name][:2] for name in np_type.names):
            _collect_data_ranges(field_type, offset + field_offset, out)
    else:
        out.append((offset, offset + np_type.itemsize))


def _udt_bytes(np_type, arr):
    """The bytes of a UDT buffer, with any alignment padding zeroed.

    ``tobytes()`` flattens any-rank numpy buffers (in particular, multi-dim
    array UDTs like ``FP64[2, 3]``) to a 1-D byte buffer that cffi can copy
    into GrB_Scalar storage.

    numpy promises nothing about the padding bytes it leaves behind, and which
    way it goes has changed across releases: numpy 2.5 made ``arr[:] = record``
    a whole-item copy out of an uninitialized temporary, so even a buffer from
    ``np.zeros`` comes back with junk between the fields. A caller's own array
    can carry junk padding in too. Since the whole record is copied into the
    GrB_Scalar and read back out of ``Scalar.value``, anything that compares or
    hashes those raw bytes would otherwise see an equal-valued scalar as
    unequal.
    """
    raw = arr.tobytes()
    ranges = _padding_ranges(np_type)
    if not ranges:
        return raw
    buf = bytearray(raw)
    for start, stop in ranges:
        buf[start:stop] = bytes(stop - start)
    return bytes(buf)


_MATERIALIZE = Scalar.from_value(lib.GrB_MATERIALIZE, is_cscalar=True, name="GrB_MATERIALIZE")
_COMPLETE = Scalar.from_value(lib.GrB_COMPLETE, is_cscalar=True, name="GrB_COMPLETE")

utils._output_types[Scalar] = Scalar
utils._output_types[ScalarIndexExpr] = Scalar
utils._output_types[ScalarExpression] = Scalar

# Import vector to import matrix to import infix to import infixmethods, which has side effects
from . import vector  # noqa: E402, F401 isort:skip

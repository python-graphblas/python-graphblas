import itertools
import warnings

import numpy as np

from .. import backend, binary, monoid, select, semiring, unary
from ..dtypes import _INDEX, FP64, INT64, lookup_dtype, unify
from ..exceptions import DimensionMismatch, NoValue, check_status
from . import _supports_udfs, automethods, ffi, lib, utils
from .base import BaseExpression, BaseType, _check_mask, call
from .descriptor import lookup as descriptor_lookup
from .expr import _ALL_INDICES, AmbiguousAssignOrExtract, IndexerResolver, Updater
from .mask import Mask, StructuralMask, ValueMask
from .operator import UNKNOWN_OPCLASS, find_opclass, get_semiring, get_typed_op, op_from_string
from .scalar import (
    _COMPLETE,
    _MATERIALIZE,
    Scalar,
    ScalarExpression,
    ScalarIndexExpr,
    _as_scalar,
    _scalar_index,
)
from .utils import (
    _CArray,
    _Pointer,
    class_property,
    ints_to_numpy_buffer,
    normalize_values,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)

if backend == "suitesparse":
    from .ss.vector import ss

ffi_new = ffi.new


# Custom recipes
def _v_add_m(updater, left, right, op):
    full = Vector(left.dtype, right._ncols, name="v_full")
    full(**updater.opts)[:] = 0
    temp = left.outer(full, binary.first).new(
        name="M_temp", mask=updater.kwargs.get("mask"), **updater.opts
    )
    updater << temp.ewise_add(right, op)


def _v_mult_m(updater, left, right, op):
    updater << left.diag(name="M_temp").mxm(right, get_semiring(monoid.any, op))


def _v_union_m(updater, left, right, left_default, right_default, op):
    full = Vector(left.dtype, right._ncols, name="v_full")
    full(**updater.opts)[:] = 0
    temp = left.outer(full, binary.first).new(
        name="M_temp", mask=updater.kwargs.get("mask"), **updater.opts
    )
    updater << temp.ewise_union(right, op, left_default=left_default, right_default=right_default)


def _v_union_v(updater, left, right, left_default, right_default, op, dtype):
    mask = updater.kwargs.get("mask")
    opts = updater.opts
    new_left = left.dup(dtype, clear=True)
    new_left(mask=mask, **opts) << binary.second(right, left_default)
    new_left(mask=mask, **opts) << binary.first(left | new_left)
    new_right = right.dup(dtype, clear=True)
    new_right(mask=mask, **opts) << binary.second(left, right_default)
    new_right(mask=mask, **opts) << binary.first(right | new_right)
    updater << op(new_left & new_right)


def _reposition(updater, indices, chunk):
    updater[indices] = chunk


def _select_mask(updater, obj, mask):
    if updater.kwargs.get("mask") is None:
        orig_kwargs = updater.kwargs
        try:
            if updater.kwargs.get("accum") is None:
                updater.kwargs = dict(orig_kwargs, mask=mask, replace=True)
            else:
                updater.kwargs = dict(orig_kwargs, mask=mask)
            updater << obj
        finally:
            updater.kwargs = orig_kwargs
    else:
        # Can we do any better depending on accum, replace, and type of masks?
        updater << obj.dup(mask=mask)


def _isclose_recipe(self, other, rel_tol, abs_tol, **opts):
    #  x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
    isequal = self.ewise_mult(other, binary.eq).new(bool, name="isclose", **opts)
    if isequal._nvals != self._nvals:
        return False
    if type(isequal) is Vector:
        val = isequal.reduce(monoid.land, allow_empty=False).new(**opts).value
    else:
        val = isequal.reduce_scalar(monoid.land, allow_empty=False).new(**opts).value
    if val:
        return True
    # So we can use structural mask below
    isequal(**opts) << select.value(isequal == True)  # noqa: E712

    # abs(x)
    x = self.apply(unary.abs).new(FP64, mask=~isequal.S, **opts)
    # abs(y)
    y = other.apply(unary.abs).new(FP64, mask=~isequal.S, **opts)
    # max(abs(x), abs(y))
    x(**opts) << x.ewise_mult(y, binary.max)
    max_x_y = x
    # rel_tol * max(abs(x), abs(y))
    max_x_y(**opts) << max_x_y.apply(binary.times, rel_tol)
    # max(rel_tol * max(abs(x), abs(y)), abs_tol)
    max_x_y(**opts) << max_x_y.apply(binary.max, abs_tol)

    # x - y
    y(~isequal.S, replace=True, **opts) << self.ewise_mult(other, binary.minus)
    abs_x_y = y
    # abs(x - y)
    abs_x_y(**opts) << abs_x_y.apply(unary.abs)

    # abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
    isequal(**opts) << abs_x_y.ewise_mult(max_x_y, binary.le)
    if isequal.ndim == 1:
        return isequal.reduce(monoid.land, allow_empty=False).new(**opts).value
    return isequal.reduce_scalar(monoid.land, allow_empty=False).new(**opts).value


class Vector(BaseType):
    """Create a new GraphBLAS Sparse Vector.

    Parameters
    ----------
    dtype :
        Data type for elements in the Vector.
    size : int
        Size of the Vector.
    name : str, optional
        Name to give the Vector. This will be displayed in the ``__repr__``.
    """

    __slots__ = "_size", "_parent", "ss"
    ndim = 1
    _name_counter = itertools.count()

    def __new__(cls, dtype=FP64, size=0, *, name=None):
        self = object.__new__(cls)
        self.dtype = lookup_dtype(dtype)
        size = _as_scalar(size, _INDEX, is_cscalar=True)
        self.name = f"v_{next(Vector._name_counter)}" if name is None else name
        self.gb_obj = ffi_new("GrB_Vector*")
        call("GrB_Vector_new", [_Pointer(self), self.dtype, size])
        self._size = size.value
        self._parent = None
        if backend == "suitesparse":
            self.ss = ss(self)
        return self

    @classmethod
    def _from_obj(cls, gb_obj, dtype, size, *, parent=None, name=None):
        self = object.__new__(cls)
        self.name = f"v_{next(Vector._name_counter)}" if name is None else name
        self.gb_obj = gb_obj
        self.dtype = dtype
        self._size = size
        self._parent = parent
        if backend == "suitesparse":
            self.ss = ss(self)
        return self

    def __del__(self):
        parent = getattr(self, "_parent", None)
        if parent is not None:
            return
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Vector_free(gb_obj), self)

    def _as_matrix(self, *, name=None):
        """Cast this Vector to a Matrix (such as a column vector).

        This is SuiteSparse-specific and may change in the future.
        This does not copy the vector.
        """
        from .matrix import Matrix

        if backend == "suitesparse":
            return Matrix._from_obj(
                ffi.cast("GrB_Matrix*", self.gb_obj),
                self.dtype,
                self._size,
                1,
                parent=self,
                name=f"(GrB_Matrix){self.name}" if name is None else name,
            )
        rv = Matrix(self.dtype, self._size, 1, name=self.name if name is None else name)
        rv[:, 0] = self
        return rv

    def __repr__(self, mask=None, expr=None):
        from .formatting import format_vector
        from .recorder import skip_record

        with skip_record:
            return format_vector(self, mask=mask, expr=expr)

    def _repr_html_(self, mask=None, collapse=False, expr=None):
        if self._parent is not None and mask is None:
            # Scalars repr can't handle mask
            return self._parent._repr_html_(collapse=collapse)
        from .formatting import format_vector_html
        from .recorder import skip_record

        with skip_record:
            return format_vector_html(self, mask=mask, collapse=collapse, expr=expr)

    @property
    def _name_html(self):
        if self._parent is not None:
            return self._parent._name_html
        return super()._name_html

    def __reduce__(self):
        # TODO: we should probably use (or compare to) GraphBLAS serialize methods
        if backend == "suitesparse":
            pieces = self.ss.export(raw=True)
        else:
            indices, values = self.to_coo(sort=False)
            pieces = (indices, values, self.dtype, self._size)
        return self._deserialize, (pieces, self.name)

    @staticmethod
    def _deserialize(pieces, name):
        if backend == "suitesparse":
            return Vector.ss.import_any(name=name, **pieces)
        indices, values, dtype, size = pieces
        return Vector.from_coo(indices, values, dtype, size=size, name=name)

    @property
    def S(self):
        """Create a Mask based on the structure of the Vector."""
        return StructuralMask(self)

    @property
    def V(self):
        """Create a Mask based on the values in the Vector (treating each value as truthy)."""
        return ValueMask(self)

    def __delitem__(self, keys, **opts):
        """Delete a single element or subvector.

        Examples
        --------
            >>> del v[1:-1]
        """
        del Updater(self, opts=opts)[keys]

    def __getitem__(self, keys):
        """Extract a single element or subvector.

        See the `Extract section <../user_guide/operations.html#extract>`__
        in the User Guide for more details.

        Examples
        --------
        .. code-block:: python

            sub_v = v[[1, 3, 5]].new()
        """
        resolved_indexes = IndexerResolver(self, keys)
        shape = resolved_indexes.shape
        if not shape:
            return ScalarIndexExpr(self, resolved_indexes)
        return VectorIndexExpr(self, resolved_indexes, *shape)

    def __setitem__(self, keys, expr, **opts):
        """Assign values to a single element or subvector.

        See the `Assign section <../user_guide/operations.html#assign>`__
        in the User Guide for more details.

        Examples
        --------
        .. code-block:: python

            # This makes a dense iso-value vector
            v[:] = 1
        """
        Updater(self, opts=opts)[keys] = expr

    def __contains__(self, index):
        """Indicates whether a value is present at the index.

        Examples
        --------
        .. code-block:: python

            # Check if v[15] is non-empty
            15 in v
        """
        extractor = self[index]
        if not extractor._is_scalar:
            raise TypeError(
                f"Invalid index to Vector contains: {index!r}.  An integer is expected.  "
                "Doing `index in my_vector` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar._is_empty

    def __iter__(self):
        """Iterate over indices which are present in the vector."""
        indices, _ = self.to_coo(values=False)
        return indices.flat

    def __sizeof__(self):
        if backend == "suitesparse":
            size = ffi_new("size_t*")
            check_status(lib.GxB_Vector_memoryUsage(size, self.gb_obj[0]), self)
            return size[0] + object.__sizeof__(self)
        raise TypeError("Unable to get size of Vector with backend: {backend}")

    def isequal(self, other, *, check_dtype=False, **opts):
        """Check for exact equality (same size, same structure).

        Parameters
        ----------
        other : Vector
            The vector to compare against
        check_dtypes : bool, default=False
            If True, also checks that dtypes match

        Returns
        -------
        bool

        See Also
        --------
        :meth:`isclose` : For equality check of floating point dtypes
        """
        other = self._expect_type(other, Vector, within="isequal", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
            return False
        if self._nvals != other._nvals:
            return False
        if check_dtype:
            # dtypes are equivalent, so not need to unify
            op = binary.eq[self.dtype]
        else:
            op = get_typed_op(binary.eq, self.dtype, other.dtype, kind="binary")

        matches = Vector(bool, self._size, name="v_isequal")
        matches(**opts) << self.ewise_mult(other, op)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce(monoid.land, allow_empty=False).new(**opts).value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False, **opts):
        """Check for approximate equality (including same size and same structure).

        Equivalent to: ``abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)``.

        Parameters
        ----------
        other : Vector
            Vector to compare against
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
        other = self._expect_type(other, Vector, within="isclose", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
            return False
        if self._nvals != other._nvals:
            return False
        if not _supports_udfs:
            return _isclose_recipe(self, other, rel_tol, abs_tol, **opts)

        matches = self.ewise_mult(other, binary.isclose(rel_tol, abs_tol)).new(
            bool, name="M_isclose", **opts
        )
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce(monoid.land, allow_empty=False).new(**opts).value

    @property
    def size(self):
        """Size of the Vector."""
        scalar = _scalar_index("s_size")
        call("GrB_Vector_size", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def shape(self):
        """A tuple of ``(size,)``."""
        return (self._size,)

    @property
    def nvals(self):
        """Number of non-empty values in the Vector."""
        scalar = _scalar_index("s_nvals")
        call("GrB_Vector_nvals", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls."""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]), self)
        return n[0]

    def clear(self):
        """In-place operation which clears all values in the Vector.

        After the call, :attr:`nvals` will return 0. The :attr:`size` will not change.
        """
        call("GrB_Vector_clear", [self])

    def resize(self, size):
        """In-place operation which changes the :attr:`size`.

        | Increasing :attr:`size` will expand with empty values.
        | Decreasing :attr:`size` will drop existing values above the new maximum index.
        """
        size = _as_scalar(size, _INDEX, is_cscalar=True)
        call("GrB_Vector_resize", [self, size])
        self._size = size.value

    def to_values(self, dtype=None, *, indices=True, values=True, sort=True):
        """Extract the indices and values as a 2-tuple of numpy arrays.

        .. deprecated:: 2022.11.0
            `Vector.to_values` will be removed in a future release.
            Use `Vector.to_coo` instead. Will be removed in version 2023.9.0 or later

        Parameters
        ----------
        dtype :
            Requested dtype for the output values array.
        indices :bool, default=True
            Whether to return indices; will return `None` for indices if `False`
        values : bool, default=True
            Whether to return values; will return `None` for values if `False`
        sort : bool, default=True
            Whether to require sorted indices.

        Returns
        -------
        np.ndarray[dtype=uint64] : Indices
        np.ndarray : Values
        """
        warnings.warn(
            "`Vector.to_values(...)` is deprecated; please use `Vector.to_coo(...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_coo(dtype, indices=indices, values=values, sort=sort)

    def to_coo(self, dtype=None, *, indices=True, values=True, sort=True):
        """Extract the indices and values as a 2-tuple of numpy arrays.

        Parameters
        ----------
        dtype :
            Requested dtype for the output values array.
        indices :bool, default=True
            Whether to return indices; will return `None` for indices if `False`
        values : bool, default=True
            Whether to return values; will return `None` for values if `False`
        sort : bool, default=True
            Whether to require sorted indices.

        See Also
        --------
        to_dense
        to_dict
        from_coo

        Returns
        -------
        np.ndarray[dtype=uint64] : Indices
        np.ndarray : Values
        """
        if sort and backend == "suitesparse":
            self.wait()  # sort in SS
        nvals = self._nvals
        if indices or backend != "suitesparse":
            c_indices = _CArray(size=nvals, name="&index_array")
        else:
            c_indices = None
        if values or backend != "suitesparse":
            c_values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        else:
            c_values = None
        scalar = _scalar_index("s_nvals")
        scalar.value = nvals
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(
            f"GrB_Vector_extractTuples_{dtype_name}", [c_indices, c_values, _Pointer(scalar), self]
        )
        if values:
            c_values = normalize_values(self, c_values.array, dtype)
        if sort and backend != "suitesparse":
            c_indices = c_indices.array
            ind = np.argsort(c_indices)
            return (
                c_indices[ind] if indices else None,
                c_values[ind] if values else None,
            )
        return (
            c_indices.array if indices else None,
            c_values if values else None,
        )

    def build(self, indices, values, *, dup_op=None, clear=False, size=None):
        """Rarely used method to insert values into an existing Vector. The typical use case
        is to create a new Vector and insert values at the same time using :meth:`from_coo`.

        All the arguments are used identically in :meth:`from_coo`, except for `clear`, which
        indicates whether to clear the Vector prior to adding the new values.
        """
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        values, dtype = values_to_numpy_buffer(values, self.dtype)
        n = values.shape[0]
        if indices.size != n:
            raise ValueError(
                f"`indices` and `values` lengths must match: {indices.size} != {values.size}"
            )
        if clear:
            self.clear()
        if size is not None:
            self.resize(size)
        if n == 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            if not self.dtype._is_udt:
                dup_op = binary.plus
            elif backend != "suitesparse":
                dup_op = binary.any
            # SS:SuiteSparse-specific: we use NULL for dup_op
        if dup_op is not None:
            dup_op = get_typed_op(dup_op, self.dtype, kind="binary")
            if dup_op.opclass == "Monoid":
                dup_op = dup_op.binaryop
            else:
                self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        indices = _CArray(indices)
        values = _CArray(values, self.dtype)
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(
            f"GrB_Vector_build_{dtype_name}",
            [self, indices, values, _as_scalar(n, _INDEX, is_cscalar=True), dup_op],
        )

        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, dtype=None, *, clear=False, mask=None, name=None, **opts):
        """Create a duplicate of the Vector.

        This is a full copy, not a view on the original.

        Parameters
        ----------
        dtype :
            Data type of the new Vector. Normal typecasting rules apply.
        clear : bool, default=False
            If True, the returned Vector will be empty.
        mask : Mask, optional
            Mask controlling which elements of the original to include in the copy.
        name : str, optional
            Name to give the Vector.

        Returns
        -------
        Vector
        """
        if dtype is not None or mask is not None or clear:
            if dtype is None:
                dtype = self.dtype
            rv = Vector(dtype, size=self._size, name=name)
            if not clear:
                rv(mask=mask, **opts)[...] = self
        else:
            if opts:
                # Ignore opts for now
                descriptor_lookup(**opts)
            rv = Vector._from_obj(ffi_new("GrB_Vector*"), self.dtype, self._size, name=name)
            call("GrB_Vector_dup", [_Pointer(rv), self])
        return rv

    def diag(self, k=0, *, name=None):
        """Return a Matrix with values on the diagonal built from the Vector.

        Parameters
        ----------
        k : int
            Off-diagonal offset in the Matrix.
        dtype :
            Data type of the new Matrix. Normal typecasting rules apply.
        name : str, optional
            Name to give the new Matrix.

        Returns
        -------
        :class:`~graphblas.Matrix`
        """
        from .matrix import Matrix

        k = _as_scalar(k, INT64, is_cscalar=True)
        n = self._size + abs(k.value)
        rv = Matrix._from_obj(ffi_new("GrB_Matrix*"), self.dtype, n, n, name=name)
        call("GrB_Matrix_diag", [_Pointer(rv), self, k])
        return rv

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
        Use wait to force completion of the Vector.

        Has no effect in `blocking mode <../user_guide/init.html#graphblas-modes>`__.
        """
        how = how.lower()
        if how == "materialize":
            mode = _MATERIALIZE
        elif how == "complete":
            mode = _COMPLETE
        else:
            raise ValueError(f'`how` argument must be "materialize" or "complete"; got {how!r}')
        call("GrB_Vector_wait", [self, mode])
        return self

    def get(self, index, default=None):
        """Get an element at ``index`` as a Python scalar.

        Parameters
        ----------
        index : int
            Vector index
        default :
            Value returned if no element exists at index

        Returns
        -------
        Python scalar
        """
        expr = self[index]
        if expr._is_scalar:
            rv = expr.new().value
            return default if rv is None else rv
        raise ValueError(
            "Bad index in Vector.get(...).  "
            "A single index should be given, and the result will be a Python scalar."
        )

    @classmethod
    def from_values(cls, indices, values, dtype=None, *, size=None, dup_op=None, name=None):
        """Create a new Vector from indices and values.

        .. deprecated:: 2022.11.0
            `Vector.from_values` will be removed in a future release.
            Use `Vector.from_coo` instead. Will be removed in version 2023.9.0 or later

        Parameters
        ----------
        indices : list or np.ndarray
            Vector indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Vector. If not provided, the values will be inspected
            to choose an appropriate dtype.
        size : int, optional
            Size of the Vector. If not provided, ``size`` is computed from
            the maximum index found in ``indices``.
        dup_op : BinaryOp, optional
            Function used to combine values if duplicate indices are found.
            Leaving ``dup_op=None`` will raise an error if duplicates are found.
        name : str, optional
            Name to give the Vector.

        Returns
        -------
        Vector
        """
        warnings.warn(
            "`Vector.from_values(...)` is deprecated; please use `Vector.from_coo(...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_coo(indices, values, dtype, size=size, dup_op=dup_op, name=name)

    @classmethod
    def from_coo(cls, indices, values=1.0, dtype=None, *, size=None, dup_op=None, name=None):
        """Create a new Vector from indices and values.

        Parameters
        ----------
        indices : list or np.ndarray
            Vector indices.
        values : list or np.ndarray or scalar, default 1.0
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Vector. If not provided, the values will be inspected
            to choose an appropriate dtype.
        size : int, optional
            Size of the Vector. If not provided, ``size`` is computed from
            the maximum index found in ``indices``.
        dup_op : BinaryOp, optional
            Function used to combine values if duplicate indices are found.
            Leaving ``dup_op=None`` will raise an error if duplicates are found.
        name : str, optional
            Name to give the Vector.

        See Also
        --------
        from_dense
        from_dict
        from_pairs
        to_coo

        Returns
        -------
        Vector
        """
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        values, dtype = values_to_numpy_buffer(values, dtype, subarray_after=1)
        # Compute size if not provided
        if size is None:
            if indices.size == 0:
                raise ValueError("No indices provided. Unable to infer size.")
            size = int(indices.max()) + 1
        # Create the new vector
        w = cls(dtype, size, name=name)
        if values.ndim == 0:
            if dup_op is not None:
                raise ValueError(
                    "dup_op must be None if values is a scalar so that all "
                    "values can be identical.  Duplicate indices will be ignored."
                )
            if backend == "suitesparse":
                w.ss.build_scalar(indices, values.tolist())
            else:
                w.build(indices, np.broadcast_to(values, indices.size), dup_op=binary.any)
        else:
            # This needs to be the original data to get proper error messages
            w.build(indices, values, dup_op=dup_op)
        return w

    @classmethod
    def from_pairs(cls, pairs, dtype=None, *, size=None, dup_op=None, name=None):
        """Create a new Vector from indices and values.

        This transforms the data and calls ``Vector.from_coo``.

        Parameters
        ----------
        pairs : list or iterable
            A sequence of ``(index, value)`` pairs.
        dtype :
            Data type of the Vector. If not provided, the values will be inspected
            to choose an appropriate dtype.
        size : int, optional
            Size of the Vector. If not provided, ``size`` is computed from
            the maximum index found in ``pairs``.
        dup_op : BinaryOp, optional
            Function used to combine values if duplicate indices are found.
            Leaving ``dup_op=None`` will raise an error if duplicates are found.
        name : str, optional
            Name to give the Vector.

        See Also
        --------
        from_coo
        from_dense
        from_dict
        to_coo

        Returns
        -------
        Vector
        """
        if isinstance(pairs, np.ndarray):
            raise TypeError("pairs as NumPy array is not supported; use `Vector.from_coo` instead")
        unzipped = list(zip(*pairs))
        if len(unzipped) == 2:
            indices, values = unzipped
        elif not unzipped:
            # Empty pairs (size should be given)
            indices = values = unzipped
        else:
            raise ValueError(
                "Each item in the pairs must have two elements (for index and value); "
                f"got {len(unzipped)}"
            )
        return cls.from_coo(indices, values, dtype, size=size, dup_op=dup_op, name=name)

    @classmethod
    def from_scalar(cls, value, size, dtype=None, *, name=None, **opts):
        """Create a fully dense Vector filled with a scalar value.

        For SuiteSparse:GraphBLAS backend, this creates an iso-valued full Vector
        that stores a single value regardless of the size of the Vector, so large
        vectors created by ``Vector.from_scalar`` will use very low memory.

        If instead you want to create a new iso-valued Vector with the same structure
        as an existing Vector, you may do: ``w = binary.second(v, value).new()``.

        Parameters
        ----------
        value : scalar
            Scalar value used to fill the Vector.
        nrows : int
            Number of rows.
        ncols : int
            Number of columns.
        dtype : DataType, optional
            Data type of the Vector. If not provided, the scalar value will be
            inspected to choose an appropriate dtype.
        name : str, optional
            Name to give the Vector.

        See Also
        --------
        from_coo
        from_dense
        from_dict
        from_pairs

        Returns
        -------
        Vector
        """
        if type(value) is not Scalar:
            try:
                value = Scalar.from_value(value, dtype, is_cscalar=None, name="")
            except TypeError:
                value = cls()._expect_type(
                    value,
                    Scalar,
                    within="from_scalar",
                    keyword_name="value",
                    extra_message="Literal scalars also accepted.",
                )
            dtype = value.dtype
        elif dtype is None:
            dtype = value.dtype
        else:
            dtype = lookup_dtype(dtype)
        if backend == "suitesparse" and not dtype._is_udt:
            # `Vector.ss.import_full` does not yet handle all cases with UDTs
            return cls.ss.import_full(value, dtype=dtype, size=size, is_iso=True, name=name)
        rv = cls(dtype, size, name=name)
        rv(**opts) << value
        return rv

    @classmethod
    def from_dense(cls, values, missing_value=None, *, dtype=None, name=None, **opts):
        """Create a Vector from a NumPy array or list.

        Parameters
        ----------
        values : list or np.ndarray
            List of values.
        missing_value : scalar, optional
            A scalar value to consider "missing"; elements of this value will be dropped.
            If None, then the resulting Vector will be dense.
        dtype : DataType, optional
            Data type of the Vector. If not provided, the values will be inspected
            to choose an appropriate dtype.
        name : str, optional
            Name to give the Vector.

        See Also
        --------
        from_coo
        from_dict
        from_pairs
        from_scalar
        to_dense

        Returns
        -------
        Vector
        """
        values, dtype = values_to_numpy_buffer(values, dtype, subarray_after=1)
        if values.ndim == 0:
            raise TypeError(
                "values must be an array or list, not a scalar. "
                "To create a dense Vector from a scalar, use `Vector.from_scalar`."
            )
        if values.ndim == 1 and dtype.np_type.subdtype is not None:
            raise ValueError("A >1d array is required to create a dense Vector with subdtype")
        if values.ndim > 1 and dtype.np_type.subdtype is None:
            raise ValueError(f"values array must be 1d to create dense Vector with dtype {dtype}")
        if backend == "suitesparse":
            rv = cls.ss.import_full(values, dtype=dtype, name=name)
        else:
            # TODO: GraphBLAS needs a better way to import or assign dense
            rv = cls.from_coo(
                np.arange(values.shape[0], dtype=np.uint64),
                values,
                dtype,
                size=values.shape[0],
                name=name,
            )
        if missing_value is not None:
            rv(**opts) << select.valuene(rv, missing_value)
        return rv

    def to_dense(self, fill_value=None, dtype=None, **opts):
        """Convert Vector to NumPy array of the same shape with missing values filled.

        .. warning::
            This can create very large arrays that require a lot of memory; please use caution.

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill missing values. This is required if there are missing values.
        dtype : DataType, optional
            Requested dtype for the output values array.

        See Also
        --------
        to_coo
        to_dict
        from_dense

        Returns
        -------
        np.ndarray
        """
        if fill_value is None or self._nvals == self._size:
            if self._nvals != self._size:
                raise TypeError(
                    "fill_value must be given in `to_dense` when there are missing values"
                )
            if backend == "suitesparse":
                info = self.ss.export("full")
                return normalize_values(self, info["values"], dtype, self._size, info["is_iso"])
            return self.to_coo(dtype, indices=False)[1]

        if dtype is None and not self.dtype._is_udt:
            # dtype of fill_value can upcast the dtype
            if type(fill_value) is not Scalar:
                try:
                    fill_value = Scalar.from_value(fill_value, is_cscalar=None, name="")
                except TypeError:
                    fill_value = self._expect_type(
                        fill_value,
                        Scalar,
                        within="to_dense",
                        keyword_name="fill_value",
                        extra_message="Literal scalars also accepted.",
                    )
            dtype = unify(fill_value.dtype, self.dtype, is_left_scalar=True)

        rv = self.dup(dtype, clear=True, name="to_dense", **opts)
        rv(**opts) << fill_value
        rv(self.S, **opts) << self
        return rv.to_dense(**opts)

    @property
    def _carg(self):
        return self.gb_obj[0]

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to update to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus):
        """Perform element-wise computation on the union of sparse values, similar to how
        one expects addition to work for sparse vectors.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_add and :meth:`ewise_union`.

        Parameters
        ----------
        other : Vector
            The other vector in the computation
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use on intersecting values

        Returns
        -------
        VectorExpression with a structure formed as the union of the input structures

        Examples
        --------
        .. code-block:: python

            # Method syntax
            w << u.ewise_add(v, op=monoid.max)

            # Functional syntax
            w << monoid.max(u | v)
        """
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        method_name = "ewise_add"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                # Check this before we compute a possibly large matrix below
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _v_add_m, (self, other, op)],
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        expr = VectorExpression(
            method_name,
            f"GrB_Vector_eWiseAdd_{op.opclass}",
            [self, other],
            op=op,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_mult(self, other, op=binary.times):
        """Perform element-wise computation on the intersection of sparse values,
        similar to how one expects multiplication to work for sparse vectors.

        See the
        `Element-wise Intersection <../user_guide/operations.html#element-wise-intersection>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Vector
            The other vector in the computation
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use on intersecting values

        Returns
        -------
        VectorExpression with a structure formed as the intersection of the input structures

        Examples
        --------
        .. code-block:: python

            # Method syntax
            w << u.ewise_mult(v, op=binary.gt)

            # Functional syntax
            w << binary.gt(u & v)
        """
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        method_name = "ewise_mult"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _v_mult_m, (self, other, op)],
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        expr = VectorExpression(
            method_name,
            f"GrB_Vector_eWiseMult_{op.opclass}",
            [self, other],
            op=op,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_union(self, other, op, left_default, right_default):
        """Perform element-wise computation on the union of sparse values,
        similar to how one expects subtraction to work for sparse vectors.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_union and :meth:`ewise_add`.

        Parameters
        ----------
        other : Vector
            The other vector in the computation
        op : :class:`~graphblas.core.operator.Monoid` or :class:`~graphblas.core.operator.BinaryOp`
            Operator to use
        left_default :
            Scalar value to use when the index on the left is missing
        right_default :
            Scalar value to use when the index on the right is missing

        Returns
        -------
        VectorExpression with a structure formed as the union of the input structures

        Examples
        --------
        .. code-block:: python

            # Method syntax
            w << u.ewise_union(v, op=binary.div, left_default=1, right_default=1)

            # Functional syntax
            w << binary.div(u | v, left_default=1, right_default=1)
        """
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        method_name = "ewise_union"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        dtype = self.dtype if self.dtype._is_udt else None
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
        scalar_dtype = unify(left.dtype, right.dtype)
        nonscalar_dtype = unify(self.dtype, other.dtype)
        op = get_typed_op(op, scalar_dtype, nonscalar_dtype, is_left_scalar=True, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        expr_repr = "{0.name}.{method_name}({2.name}, {op}, {1._expr_name}, {3._expr_name})"
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, left, other, right, _v_union_m, (self, other, left, right, op)],
                expr_repr=expr_repr,
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        if backend == "suitesparse":
            expr = VectorExpression(
                method_name,
                "GxB_Vector_eWiseUnion",
                [self, left, other, right],
                op=op,
                expr_repr=expr_repr,
            )
        else:
            dtype = unify(scalar_dtype, nonscalar_dtype, is_left_scalar=True)
            expr = VectorExpression(
                method_name,
                None,
                [self, left, other, right, _v_union_v, (self, other, left, right, op, dtype)],
                expr_repr=expr_repr,
                size=self._size,
                op=op,
            )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def vxm(self, other, op=semiring.plus_times):
        """Perform vector-matrix multiplication with the Vector being treated as a
        (1xn) row vector on the left side of the computation.

        See the
        `Matrix Multiply <../user_guide/operations.html#matrix-multiply>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other: Matrix
            The matrix on the right side in the computation
        op : :class:`~graphblas.core.operator.Semiring`
            Semiring used in the computation

        Returns
        -------
        VectorExpression

        Examples
        --------
        .. code-block:: python

            # Method syntax
            C << v.vxm(A, op=semiring.min_plus)

            # Functional syntax
            C << semiring.min_plus(v @ A)
        """
        from .matrix import Matrix, TransposedMatrix

        method_name = "vxm"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = VectorExpression(
            method_name,
            "GrB_vxm",
            [self, other],
            op=op,
            size=other._ncols,
            bt=other._is_transposed,
        )
        if self._size != other._nrows:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def apply(self, op, right=None, *, left=None):
        """Create a new Vector by applying ``op`` to each element of the Vector.

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
        VectorExpression

        Examples
        --------
        .. code-block:: python

            # Method syntax
            w << v.apply(op.abs)

            # Functional syntax
            w << op.abs(v)
        """
        method_name = "apply"
        extra_message = (
            "apply only accepts UnaryOp with no scalars or BinaryOp with `left` or `right` scalar"
            "or IndexUnaryOp with `right` thunk."
        )
        if isinstance(op, str):
            op = op_from_string(op)
        op, opclass = find_opclass(op)
        if opclass in {"IndexUnaryOp", "SelectOp"}:
            # Provide default value for index unary
            if right is None:
                right = False  # most basic form of 0 when unifying dtypes
            if left is not None:
                raise TypeError("Do not pass `left` when applying IndexUnaryOp")

        if left is None and right is None:
            op = get_typed_op(op, self.dtype, kind="unary")
            self._expect_op(
                op,
                "UnaryOp",
                within=method_name,
                argname="op",
                extra_message=extra_message,
            )
            cfunc_name = "GrB_Vector_apply"
            args = [self]
            expr_repr = None
        elif right is None:
            if type(left) is not Scalar:
                dtype = self.dtype if self.dtype._is_udt else None
                try:
                    left = Scalar.from_value(left, dtype, is_cscalar=None, name="")
                except TypeError:
                    left = self._expect_type(
                        left,
                        Scalar,
                        within=method_name,
                        keyword_name="left",
                        extra_message="Literal scalars also accepted.",
                        op=op,
                    )
            op = get_typed_op(op, left.dtype, self.dtype, is_left_scalar=True, kind="binary")
            if opclass == "Monoid":
                op = op.binaryop
            else:
                self._expect_op(
                    op,
                    "BinaryOp",
                    within=method_name,
                    argname="op",
                    extra_message=extra_message,
                )
            if left._is_cscalar:
                if left.dtype._is_udt:
                    dtype_name = "UDT"
                    left = _Pointer(left)
                else:
                    dtype_name = left.dtype.name
                cfunc_name = f"GrB_Vector_apply_BinaryOp1st_{dtype_name}"
            else:
                cfunc_name = "GrB_Vector_apply_BinaryOp1st_Scalar"
            args = [left, self]
            expr_repr = "{1.name}.apply({op}, left={0._expr_name})"
        elif left is None:
            if type(right) is not Scalar:
                dtype = self.dtype if (self.dtype._is_udt and not op.is_positional) else None
                try:
                    right = Scalar.from_value(right, dtype, is_cscalar=None, name="")
                except TypeError:
                    right = self._expect_type(
                        right,
                        Scalar,
                        within=method_name,
                        keyword_name="right",
                        extra_message="Literal scalars also accepted.",
                        op=op,
                    )
            if opclass in {"IndexUnaryOp", "SelectOp"}:
                op = get_typed_op(
                    op, self.dtype, right.dtype, is_right_scalar=True, kind="indexunary"
                )
                cfunc_method = "IndexOp"
            else:
                op = get_typed_op(op, self.dtype, right.dtype, is_right_scalar=True, kind="binary")
                cfunc_method = "BinaryOp2nd"
                if opclass == "Monoid":
                    op = op.binaryop
                else:
                    self._expect_op(
                        op,
                        "BinaryOp",
                        within=method_name,
                        argname="op",
                        extra_message=extra_message,
                    )
            if right._is_cscalar:
                if right.dtype._is_udt:
                    dtype_name = "UDT"
                    right = _Pointer(right)
                else:
                    dtype_name = right.dtype.name
                cfunc_name = f"GrB_Vector_apply_{cfunc_method}_{dtype_name}"
            else:
                cfunc_name = f"GrB_Vector_apply_{cfunc_method}_Scalar"
            args = [self, right]
            expr_repr = "{0.name}.apply({op}, right={1._expr_name})"
        else:
            raise TypeError("Cannot provide both `left` and `right` to apply")
        return VectorExpression(
            method_name,
            cfunc_name,
            args,
            op=op,
            expr_repr=expr_repr,
            size=self._size,
        )

    def select(self, op, thunk=None):
        """Create a new Vector by applying ``op`` to each element of the Vector and keeping
        those elements where ``op`` returns True.

        See the `Select <../user_guide/operations.html#select>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.SelectOp`
            Operator to apply
        thunk :
            Scalar passed to operator

        Returns
        -------
        VectorExpression

        Examples
        --------
        .. code-block:: python

            # Method syntax
            w << v.select(">=", 1)

            # Functional syntax
            w << select.value(v >= 1)
        """
        method_name = "select"
        if isinstance(op, str):
            op = select.from_string(op)
        else:
            if isinstance(op, VectorExpression):
                # Try to rewrite e.g. `v.select(v == 7)` to `gb.select.value(v == 7)`
                if thunk is not None:
                    raise TypeError(
                        "thunk argument not None when calling select with mask or boolean object"
                    )
                expr = select._match_expr(self, op)
                if expr is not None:
                    return expr
                opclass = UNKNOWN_OPCLASS
            else:
                op, opclass = find_opclass(op)
            if opclass == UNKNOWN_OPCLASS:
                # e.g., `v.select(w.S)` or `v.select(w < 7)`
                mask = _check_mask(op)
                if thunk is not None:
                    raise TypeError(
                        "thunk argument not None when calling select with mask or boolean object"
                    )
                self._expect_type(mask.parent, (Vector, Mask), within=method_name, argname="op")
                return VectorExpression(
                    "select",
                    None,
                    [self, mask, _select_mask, (self, mask)],  # [*expr_args, func, args]
                    expr_repr="{0.name}.select({1.name})",
                    size=self._size,
                    dtype=self.dtype,
                )

        if thunk is None:
            thunk = False  # most basic form of 0 when unifying dtypes
        if type(thunk) is not Scalar:
            dtype = self.dtype if (self.dtype._is_udt and not op.is_positional) else None
            try:
                thunk = Scalar.from_value(thunk, dtype, is_cscalar=None, name="")
            except TypeError:
                thunk = self._expect_type(
                    thunk,
                    Scalar,
                    within=method_name,
                    keyword_name="thunk",
                    extra_message="Literal scalars also accepted.",
                    op=op,
                )
        op = get_typed_op(op, self.dtype, thunk.dtype, is_right_scalar=True, kind="select")
        self._expect_op(op, ("SelectOp", "IndexUnaryOp"), within=method_name, argname="op")
        if thunk._is_cscalar:
            if thunk.dtype._is_udt:
                dtype_name = "UDT"
                thunk = _Pointer(thunk)
                # NOT COVERED
            else:
                dtype_name = thunk.dtype.name
            cfunc_name = f"GrB_Vector_select_{dtype_name}"
        else:
            cfunc_name = "GrB_Vector_select_Scalar"
        return VectorExpression(
            method_name,
            cfunc_name,
            [self, thunk],
            op=op,
            expr_repr="{0.name}.select({op}, thunk={1._expr_name})",
            size=self._size,
            dtype=self.dtype,
        )

    def reduce(self, op=monoid.plus, *, allow_empty=True):
        """Reduce all values in the Vector into a single value using ``op``.

        See the `Reduce <../user_guide/operations.html#reduce>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.Monoid`
            Reduction operator
        allow_empty : bool, default=True
            If False and the Vector is empty, the Scalar result
            will hold the monoid identity rather than a missing value

        Returns
        -------
        ScalarExpression

        Examples
        --------
        .. code-block:: python

            total << v.reduce(monoid.plus)
        """
        method_name = "reduce"
        op = get_typed_op(op, self.dtype, kind="binary|aggregator")
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        else:
            self._expect_op(op, ("Monoid", "Aggregator"), within=method_name, argname="op")
        if not allow_empty and op.opclass == "Aggregator" and op.parent._monoid is None:
            # But we still kindly allow it if it's a monoid-only aggregator such as sum
            raise ValueError("allow_empty=False not allowed when using Aggregators")
        if allow_empty:
            cfunc_name = "GrB_Vector_reduce_Monoid_Scalar"
        elif self.dtype._is_udt:
            cfunc_name = "GrB_Vector_reduce_UDT"
        else:
            cfunc_name = "GrB_Vector_reduce_{output_dtype}"
        return ScalarExpression(
            method_name,
            cfunc_name,
            [self],
            op=op,  # to be determined later
            is_cscalar=not allow_empty,
        )

    # Unofficial methods
    def inner(self, other, op=semiring.plus_times):
        """Perform vector-vector inner (or dot) product.

        Parameters
        ----------
        other : Vector
            The vector on the right side in the computation
        op : :class:`~graphblas.core.operator.Semiring`
            Semiring used in the computation

        Returns
        -------
        ScalarExpression

        Examples
        --------
        .. code-block:: python

            # Method syntax
            s << v.inner(w, op=semiring.min_plus)

            # Functional syntax
            s << semiring.min_plus(v @ w)

        *Note*: This is not a standard GraphBLAS function, but fits with other functions in the
        `Matrix Multiplication <../user_guide/operations.html#matrix-multiply>`__
        family of functions.
        """
        method_name = "inner"
        other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = ScalarExpression(
            method_name,
            "GrB_vxm",
            [self, other._as_matrix()],
            op=op,
            is_cscalar=False,
            scalar_as_vector=True,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def outer(self, other, op=binary.times):
        """Perform vector-vector outer (or cross) product.

        Parameters
        ----------
        other : Vector
            The vector on the right side in the computation
        op : :class:`~graphblas.core.operator.BinaryOp`
            Operator used in the computation

        Returns
        -------
        MatrixExpression

        Examples
        --------
        .. code-block:: python

            C << v.outer(w, op=binary.times)

        *Note*: This is not a standard GraphBLAS function.
        """
        from .matrix import MatrixExpression

        method_name = "outer"
        other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        op = get_semiring(monoid.any, op)
        expr = MatrixExpression(
            method_name,
            "GrB_mxm",
            [self._as_matrix(), other._as_matrix()],
            op=op,
            nrows=self._size,
            ncols=other._size,
            bt=True,
        )
        return expr

    def reposition(self, offset, *, size=None):
        """Create a new Vector with values identical to the original Vector,
        but repositioned within the total size by adding ``offset`` to the indices.

        Positive offset moves values to the right, negative to the left.
        Values repositioned outside of the new Vector are dropped (i.e. they don't wrap around).

        *Note*: This is not a standard GraphBLAS method.
        It is implemented using extract and assign.

        Parameters
        ----------
        offset : int
            Offset for the indices.
        size : int, optional
            If specified, the new Vector will be resized.
            Default is the same size as the original Vector.

        Returns
        -------
        VectorExpression

        Examples
        --------
        .. code-block:: python

            w = v.reposition(20).new()
        """
        if size is None:
            size = self._size
        else:
            size = int(size)
        offset = int(offset)
        if offset < 0:
            start = -offset
            stop = start + size
        else:
            start = 0
            stop = max(0, size - offset)
        chunk = self[start:stop].new(name="v_repositioning")
        indices = slice(start + offset, start + offset + chunk._size)
        return VectorExpression(
            "reposition",
            None,
            [self, _reposition, (indices, chunk)],  # [*expr_args, func, args]
            expr_repr=f"{{0.name}}.reposition({offset})",
            size=size,
            dtype=self.dtype,
        )

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(
        self, resolved_indexes, dtype, opts, *, is_cscalar, name=None, result=None
    ):
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = lookup_dtype(dtype)
        idx = resolved_indexes.indices[0]
        if result is None:
            result = Scalar(dtype, is_cscalar=is_cscalar, name=name)
        if opts:
            # Ignore opts for now
            descriptor_lookup(**opts)
        if is_cscalar:
            dtype_name = "UDT" if dtype._is_udt else dtype.name
            if (
                call(f"GrB_Vector_extractElement_{dtype_name}", [_Pointer(result), self, idx.index])
                is not NoValue
            ):
                result._empty = False
        else:
            call("GrB_Vector_extractElement_Scalar", [result, self, idx.index])
        return result

    def _prep_for_extract(self, resolved_indexes, mask=None, is_submask=False):
        index = resolved_indexes.indices[0]
        return VectorExpression(
            "__getitem__",
            "GrB_Vector_extract",
            [self, index, index.cscalar],
            expr_repr="{0.name}[{1._expr_name}]",
            size=index.size,
            dtype=self.dtype,
        )

    def _assign_element(self, resolved_indexes, value):
        idx = resolved_indexes.indices[0]
        if type(value) is not Scalar:
            dtype = self.dtype if self.dtype._is_udt else None
            try:
                value = Scalar.from_value(value, dtype, is_cscalar=None, name="")
            except TypeError:
                value = self._expect_type(
                    value,
                    Scalar,
                    within="__setitem__",
                    argname="value",
                    extra_message="Literal scalars also accepted.",
                )
        if value._is_cscalar:
            if value._empty:
                call("GrB_Vector_removeElement", [self, idx.index])
                return
            if value.dtype._is_udt:
                dtype_name = "UDT"
                value = _Pointer(value)
            else:
                dtype_name = value.dtype.name
            cfunc_name = f"GrB_Vector_setElement_{dtype_name}"
        else:
            cfunc_name = "GrB_Vector_setElement_Scalar"
        call(cfunc_name, [self, value, idx.index])

    def _prep_for_assign(self, resolved_indexes, value, mask, is_submask, replace, opts):
        method_name = "__setitem__"
        idx = resolved_indexes.indices[0]
        size = idx.size
        cscalar = idx.cscalar
        index = idx.index

        if output_type(value) is Vector:
            if type(value) is not Vector:
                value = self._expect_type(
                    value,
                    Vector,
                    within=method_name,
                )
            if is_submask:
                if size is None:
                    # v[i](m) << w
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << w
                if backend == "suitesparse":
                    cfunc_name = "GxB_Vector_subassign"
                else:
                    cfunc_name = "GrB_Vector_assign"
                    mask = _vanilla_subassign_mask(self, mask, idx, replace, opts)
                expr_repr = (
                    "[[{2._expr_name} elements]]"
                    f"({mask.name})"  # fmt: skip
                    " = {0.name}"
                )
            else:
                # v(m)[I] << w
                # v[I] << w
                cfunc_name = "GrB_Vector_assign"
                expr_repr = "[[{2._expr_name} elements]] = {0.name}"
        else:
            if type(value) is not Scalar:
                dtype = self.dtype if self.dtype._is_udt else None
                try:
                    value = Scalar.from_value(value, dtype, is_cscalar=None, name="")
                except (TypeError, ValueError):
                    if size is not None:
                        # v[I] << [1, 2, 3]
                        # v(m)[I] << [1, 2, 3]
                        # v[I](m) << [1, 2, 3]
                        try:
                            values, dtype = values_to_numpy_buffer(value, dtype)
                        except Exception:
                            extra_message = "Literal scalars and lists also accepted."
                        else:
                            shape = values.shape
                            try:
                                vals = Vector.from_dense(values, dtype=dtype)
                            except Exception:
                                vals = None
                            else:
                                if dtype.np_type.subdtype is not None:
                                    shape = vals.shape
                            if vals is None or shape != (size,):
                                if dtype.np_type.subdtype is not None:
                                    # NOT COVERED
                                    extra = (
                                        " (this is assigning to a vector with sub-array dtype "
                                        f"({dtype}), so array shape should include dtype shape)"
                                    )
                                else:
                                    extra = ""
                                raise ValueError(
                                    f"shape mismatch: value array of shape {shape} "
                                    f"does not match indexing of shape ({size},)"
                                    f"{extra}"
                                ) from None
                            return self._prep_for_assign(
                                resolved_indexes,
                                vals,
                                mask,
                                is_submask,
                                replace,
                                opts,
                            )
                    else:
                        extra_message = "Literal scalars also accepted."
                    value = self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        argname="value",
                        extra_message=extra_message,
                    )
            if is_submask:
                if size is None:
                    # v[i](m) << c
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << c
                if value._is_cscalar:
                    if value.dtype._is_udt:
                        dtype_name = "UDT"
                        value = _Pointer(value)
                    else:
                        dtype_name = value.dtype.name
                    if backend == "suitesparse":
                        cfunc_name = f"GxB_Vector_subassign_{dtype_name}"
                    else:
                        cfunc_name = f"GrB_Vector_assign_{dtype_name}"
                        mask = _vanilla_subassign_mask(self, mask, idx, replace, opts)
                elif backend == "suitesparse":
                    cfunc_name = "GxB_Vector_subassign_Scalar"
                else:
                    cfunc_name = "GrB_Vector_assign_Scalar"
                    mask = _vanilla_subassign_mask(self, mask, idx, replace, opts)
                expr_repr = (
                    "[[{2._expr_name} elements]]"
                    f"({mask.name})"  # fmt: skip
                    " = {0._expr_name}"
                )
            else:
                # v(m)[I] << c
                # v[I] << c
                if size is None:
                    index = _CArray([index.value])
                    cscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                if value._is_cscalar:
                    if value.dtype._is_udt:
                        dtype_name = "UDT"
                        value = _Pointer(value)
                    else:
                        dtype_name = value.dtype.name
                    cfunc_name = f"GrB_Vector_assign_{dtype_name}"
                else:
                    cfunc_name = "GrB_Vector_assign_Scalar"
                expr_repr = "[[{2._expr_name} elements]] = {0._expr_name}"
        expr = VectorExpression(
            method_name,
            cfunc_name,
            [value, index, cscalar],
            expr_repr=expr_repr,
            size=self._size,
            dtype=self.dtype,
        )
        return expr, mask

    def _delete_element(self, resolved_indexes):
        idx = resolved_indexes.indices[0]
        call("GrB_Vector_removeElement", [self, idx.index])

    @classmethod
    def from_dict(cls, d, dtype=None, *, size=None, name=None):
        """Create a new Vector from a dict with keys as indices and values as values.

        Parameters
        ----------
        d : Mapping
            The dict-like object to convert. The keys will be cast to uint64 for the indices.
        dtype :
            Data type of the Vector. If not provided, the values will be inspected
            to choose an appropriate dtype.
        size : int, optional
            Size of the Vector. If not provided, ``size`` is computed from
            the maximum index found in ``indices``.
        name : str, optional
            Name to give the Vector.

        See Also
        --------
        from_coo
        from_dense
        from_pairs
        to_dict

        Returns
        -------
        Vector
        """
        indices = np.fromiter(d.keys(), np.uint64)
        if dtype is None:
            values, dtype = values_to_numpy_buffer(list(d.values()), subarray_after=1)
        else:
            # If we know the dtype, then using `np.fromiter` is much faster
            dtype = lookup_dtype(dtype)
            if dtype.np_type.subdtype is not None and np.__version__[:5] in {"1.21.", "1.22."}:
                values, dtype = values_to_numpy_buffer(list(d.values()), dtype)  # FLAKY COVERAGE
            else:
                values = np.fromiter(d.values(), dtype.np_type)
        if size is None and indices.size == 0:
            size = 0
        return cls.from_coo(indices, values, dtype, size=size, name=name)

    def to_dict(self):
        """Return Vector as a dict in the form ``{index: val}``.

        See Also
        --------
        to_coo
        to_dense
        from_dict

        Returns
        -------
        dict
        """
        indices, values = self.to_coo(sort=False)
        return dict(zip(indices.tolist(), values.tolist()))


if backend == "suitesparse":
    Vector.ss = class_property(Vector.ss, ss)
else:
    Vector.ss = class_property(
        Vector.ss, 'ss attribute is only available with "suitesparse" backend', exceptional=True
    )


def _vanilla_subassign_mask(self, mask, indices, replace, opts):
    _check_mask(mask, self)
    if not replace and indices.index is _ALL_INDICES:
        return mask
    indices = indices._py_index()
    val = Vector(mask.parent.dtype, size=self._size, name="v_temp")
    val(**opts)[indices] = mask.parent
    mask = type(mask)(val)
    if replace:
        val = self.dup()
        val.__delitem__(indices, **opts)  # del val[indices]
        mask = mask.__or__(val.S, **opts)  # mask |= val.S
    return mask


class VectorExpression(BaseExpression):
    __slots__ = "_size"
    ndim = 1
    output_type = Vector

    def __init__(
        self,
        method_name,
        cfunc_name,
        args,
        *,
        at=False,
        bt=False,
        op=None,
        dtype=None,
        expr_repr=None,
        size=None,
    ):
        super().__init__(
            method_name,
            cfunc_name,
            args,
            at=at,
            bt=bt,
            op=op,
            dtype=dtype,
            expr_repr=expr_repr,
        )
        if size is None:
            size = args[0]._size
        self._size = size

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Vector(dtype, self._size, name=name)

    def __repr__(self):
        from .formatting import format_vector_expression

        return format_vector_expression(self)

    def _repr_html_(self):
        from .formatting import format_vector_expression_html

        return format_vector_expression_html(self)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    @wrapdoc(Vector.dup)
    def dup(self, dtype=None, *, clear=False, mask=None, name=None, **opts):
        if clear:
            return Vector(self.dtype if dtype is None else dtype, self._size, name=name)
        return self._new(dtype, mask, name)

    # Begin auto-generated code: Vector
    _get_value = automethods._get_value
    S = wrapdoc(Vector.S)(property(automethods.S))
    V = wrapdoc(Vector.V)(property(automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(automethods.apply))
    diag = wrapdoc(Vector.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Vector.get)(property(automethods.get))
    inner = wrapdoc(Vector.inner)(property(automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(automethods.isequal))
    name = wrapdoc(Vector.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(automethods.reduce))
    reposition = wrapdoc(Vector.reposition)(property(automethods.reposition))
    select = wrapdoc(Vector.select)(property(automethods.select))
    if backend == "suitesparse":
        ss = wrapdoc(Vector.ss)(property(automethods.ss))
    else:
        ss = Vector.__dict__["ss"]  # raise if used
    to_coo = wrapdoc(Vector.to_coo)(property(automethods.to_coo))
    to_dense = wrapdoc(Vector.to_dense)(property(automethods.to_dense))
    to_dict = wrapdoc(Vector.to_dict)(property(automethods.to_dict))
    to_values = wrapdoc(Vector.to_values)(property(automethods.to_values))
    vxm = wrapdoc(Vector.vxm)(property(automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imatmul__ = automethods.__imatmul__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Vector


class VectorIndexExpr(AmbiguousAssignOrExtract):
    __slots__ = "_size"
    ndim = 1
    output_type = Vector

    def __init__(self, parent, resolved_indexes, size):
        super().__init__(parent, resolved_indexes)
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    @wrapdoc(Vector.dup)
    def dup(self, dtype=None, *, clear=False, mask=None, name=None, **opts):
        if clear:
            if dtype is None:
                dtype = self.dtype
            return self.output_type(dtype, *self.shape, name=name)
        return self.new(dtype, mask=mask, name=name, **opts)

    # Begin auto-generated code: Vector
    _get_value = automethods._get_value
    S = wrapdoc(Vector.S)(property(automethods.S))
    V = wrapdoc(Vector.V)(property(automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(automethods.apply))
    diag = wrapdoc(Vector.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Vector.get)(property(automethods.get))
    inner = wrapdoc(Vector.inner)(property(automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(automethods.isequal))
    name = wrapdoc(Vector.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(automethods.reduce))
    reposition = wrapdoc(Vector.reposition)(property(automethods.reposition))
    select = wrapdoc(Vector.select)(property(automethods.select))
    if backend == "suitesparse":
        ss = wrapdoc(Vector.ss)(property(automethods.ss))
    else:
        ss = Vector.__dict__["ss"]  # raise if used
    to_coo = wrapdoc(Vector.to_coo)(property(automethods.to_coo))
    to_dense = wrapdoc(Vector.to_dense)(property(automethods.to_dense))
    to_dict = wrapdoc(Vector.to_dict)(property(automethods.to_dict))
    to_values = wrapdoc(Vector.to_values)(property(automethods.to_values))
    vxm = wrapdoc(Vector.vxm)(property(automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imatmul__ = automethods.__imatmul__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Vector


utils._output_types[Vector] = Vector
utils._output_types[VectorIndexExpr] = Vector
utils._output_types[VectorExpression] = Vector

# Import matrix to import infix to import infixmethods, which has side effects
from . import matrix  # noqa: E402, F401 isort:skip

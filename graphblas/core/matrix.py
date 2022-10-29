import itertools
import warnings

import numpy as np

from .. import backend, binary, monoid, select, semiring
from ..dtypes import _INDEX, FP64, lookup_dtype, unify
from ..exceptions import DimensionMismatch, InvalidValue, NoValue, check_status
from . import automethods, ffi, lib, utils
from .base import BaseExpression, BaseType, _check_mask, call
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
from .mask import Mask, StructuralMask, ValueMask
from .operator import UNKNOWN_OPCLASS, find_opclass, get_semiring, get_typed_op, op_from_string
from .scalar import (
    _MATERIALIZE,
    Scalar,
    ScalarExpression,
    ScalarIndexExpr,
    _as_scalar,
    _scalar_index,
)
from .ss.matrix import ss
from .utils import (
    _CArray,
    _Pointer,
    class_property,
    ints_to_numpy_buffer,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)
from .vector import Vector, VectorExpression, VectorIndexExpr, _select_mask

ffi_new = ffi.new

_CSR_FORMAT = Scalar.from_value(
    lib.GrB_CSR_FORMAT, dtype=_INDEX, name="GrB_CSR_FORMAT", is_cscalar=True
)
_CSC_FORMAT = Scalar.from_value(
    lib.GrB_CSC_FORMAT, dtype=_INDEX, name="GrB_CSC_FORMAT", is_cscalar=True
)
# COO format is not used yet.
# _COO_FORMAT = Scalar.from_value(
#     lib.GrB_COO_FORMAT, dtype=_INDEX, name="GrB_COO_FORMAT", is_cscalar=True
# )


# Custom recipes
def _m_add_v(updater, left, right, op):
    full = Vector(right.dtype, left._nrows, name="v_full")
    full[:] = 0
    temp = full.outer(right, binary.second).new(name="M_temp", mask=updater.kwargs.get("mask"))
    updater << left.ewise_add(temp, op)


def _m_mult_v(updater, left, right, op):
    updater << left.mxm(right.diag(name="M_temp"), get_semiring(monoid.any, op))


def _m_union_v(updater, left, right, left_default, right_default, op):
    full = Vector(right.dtype, left._nrows, name="v_full")
    full[:] = 0
    temp = full.outer(right, binary.second).new(name="M_temp", mask=updater.kwargs.get("mask"))
    updater << left.ewise_union(temp, op, left_default=left_default, right_default=right_default)


def _reposition(updater, indices, chunk):
    updater[indices] = chunk


class Matrix(BaseType):
    """Create a new GraphBLAS Sparse Matrix.

    Parameters
    ----------
    dtype :
        Data type for elements in the Matrix.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    name : str, optional
        Name to give the Matrix. This will be displayed in the ``__repr__``.
    """

    __slots__ = "_nrows", "_ncols", "_parent", "ss"
    ndim = 2
    _is_transposed = False
    _name_counter = itertools.count()
    __networkx_plugin__ = "graphblas"

    def __new__(cls, dtype=FP64, nrows=0, ncols=0, *, name=None):
        self = object.__new__(cls)
        self.dtype = lookup_dtype(dtype)
        nrows = _as_scalar(nrows, _INDEX, is_cscalar=True)
        ncols = _as_scalar(ncols, _INDEX, is_cscalar=True)
        self.name = f"M_{next(Matrix._name_counter)}" if name is None else name
        self.gb_obj = ffi_new("GrB_Matrix*")
        call("GrB_Matrix_new", [_Pointer(self), self.dtype, nrows, ncols])
        self._nrows = nrows.value
        self._ncols = ncols.value
        self._parent = None
        self.ss = ss(self)
        return self

    @classmethod
    def _from_obj(cls, gb_obj, dtype, nrows, ncols, *, parent=None, name=None):
        self = object.__new__(cls)
        self.gb_obj = gb_obj
        self.dtype = dtype
        self.name = f"M_{next(Matrix._name_counter)}" if name is None else name
        self._nrows = nrows
        self._ncols = ncols
        self._parent = parent
        self.ss = ss(self)
        return self

    def __del__(self):
        parent = getattr(self, "_parent", None)
        if parent is not None:
            return
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Matrix_free(gb_obj), self)

    def _as_vector(self, *, name=None):
        """Cast this Matrix with one column to a Vector.

        This is SuiteSparse-specific and may change in the future.
        This does not copy the matrix.
        """
        from .vector import Vector

        if self._ncols != 1:
            raise ValueError(
                f"Matrix must have a single column (not {self._ncols}) to be cast to a Vector"
            )
        return Vector._from_obj(
            ffi.cast("GrB_Vector*", self.gb_obj),
            self.dtype,
            self._nrows,
            parent=self,
            name=f"(GrB_Vector){self.name}" if name is None else name,
        )

    def __repr__(self, mask=None, expr=None):
        from .formatting import format_matrix
        from .recorder import skip_record

        with skip_record:
            return format_matrix(self, mask=mask, expr=expr)

    def _repr_html_(self, mask=None, collapse=False, expr=None):
        if self._parent is not None:
            return self._parent._repr_html_(mask=mask, collapse=collapse)
        from .formatting import format_matrix_html
        from .recorder import skip_record

        with skip_record:
            return format_matrix_html(self, mask=mask, collapse=collapse, expr=expr)

    @property
    def _name_html(self):
        if self._parent is not None:
            return self._parent._name_html
        return super()._name_html

    def __reduce__(self):
        # SS, SuiteSparse-specific: export
        pieces = self.ss.export(raw=True)
        return self._deserialize, (pieces, self.name)

    @staticmethod
    def _deserialize(pieces, name):
        # SS, SuiteSparse-specific: import
        return Matrix.ss.import_any(name=name, **pieces)

    @property
    def S(self):
        """Create a Mask based on the structure of the Matrix."""
        return StructuralMask(self)

    @property
    def V(self):
        """Create a Mask based on the values in the Matrix (treating each value as truthy)."""
        return ValueMask(self)

    def __delitem__(self, keys):
        """Delete a single element, row/column, or submatrix.

        Examples
        --------

            >>> del M[1, 5]
        """
        del Updater(self)[keys]

    def __getitem__(self, keys):
        """Extract a single element, row/column, or submatrix.

        See the `Extract section <../user_guide/operations.html#extract>`__
        in the User Guide for more details.

        Examples
        --------

        .. code-block:: python

            subM = M[[1, 3, 5], :].new()
        """
        resolved_indexes = IndexerResolver(self, keys)
        shape = resolved_indexes.shape
        if not shape:
            return ScalarIndexExpr(self, resolved_indexes)
        elif len(shape) == 1:
            return VectorIndexExpr(self, resolved_indexes, *shape)
        else:
            return MatrixIndexExpr(self, resolved_indexes, *shape)

    def __setitem__(self, keys, expr):
        """Assign values to a single element, row/column, or submatrix.

        See the `Assign section <../user_guide/operations.html#assign>`__
        in the User Guide for more details.

        Examples
        --------

        .. code-block:: python

            M[0, 0:3] = 17
        """
        Updater(self)[keys] = expr

    def __contains__(self, index):
        """Indicates whether the (row, col) index has a value present.

        Examples
        --------

        .. code-block:: python

            (10, 15) in M
        """
        extractor = self[index]
        if not extractor._is_scalar:
            raise TypeError(
                f"Invalid index to Matrix contains: {index!r}.  A 2-tuple of ints is expected.  "
                "Doing `(i, j) in my_matrix` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar._is_empty

    def __iter__(self):
        """Iterate over (row, col) indices which are present in the matrix."""
        rows, columns, _ = self.to_values(values=False)
        return zip(rows.flat, columns.flat)

    def __sizeof__(self):
        size = ffi_new("size_t*")
        check_status(lib.GxB_Matrix_memoryUsage(size, self.gb_obj[0]), self)
        return size[0] + object.__sizeof__(self)

    def isequal(self, other, *, check_dtype=False):
        """Check for exact equality (same size, same structure).

        Parameters
        ----------
        other : Matrix
            The matrix to compare against
        check_dtypes : bool
            If True, also checks that dtypes match

        Returns
        -------
        bool

        See Also
        --------
        :meth:`isclose` : For equality check of floating point dtypes
        """
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isequal", argname="other"
        )
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._nrows != other._nrows:
            return False
        if self._ncols != other._ncols:
            return False
        if self._nvals != other._nvals:
            return False
        if check_dtype:
            op = binary.eq[self.dtype]
        else:
            op = get_typed_op(binary.eq, self.dtype, other.dtype, kind="binary")

        matches = Matrix(bool, self._nrows, self._ncols, name="M_isequal")
        matches << self.ewise_mult(other, op)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land, allow_empty=False).new().value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """Check for approximate equality (including same size and same structure).

        Equivalent to: ``abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)``.

        Parameters
        ----------
        other : Matrix
            The matrix to compare against.
        rel_tol : float
            Relative tolerance.
        abs_tol : float
            Absolute tolerance.
        check_dtype : bool
            If True, also checks that dtypes match

        Returns
        -------
        bool
            Whether all values of the Matrix are close to the values in `other`.
        """
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isclose", argname="other"
        )
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._nrows != other._nrows:
            return False
        if self._ncols != other._ncols:
            return False
        if self._nvals != other._nvals:
            return False

        matches = self.ewise_mult(other, binary.isclose(rel_tol, abs_tol)).new(
            bool, name="M_isclose"
        )
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land, allow_empty=False).new().value

    @property
    def nrows(self):
        """Number of rows in the Matrix."""
        scalar = _scalar_index("s_nrows")
        call("GrB_Matrix_nrows", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def ncols(self):
        """Number of columns in the Matrix."""
        scalar = _scalar_index("s_ncols")
        call("GrB_Matrix_ncols", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def shape(self):
        """A tuple of ``(nrows, ncols)``."""
        return (self._nrows, self._ncols)

    @property
    def nvals(self):
        """Number of non-empty values in the Matrix."""
        scalar = _scalar_index("s_nvals")
        call("GrB_Matrix_nvals", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Matrix_nvals(n, self.gb_obj[0]), self)
        return n[0]

    @property
    def T(self):
        """Indicates the transpose of the Matrix.

        Can be used in the arguments of most operations. It also can be used standalone
        as the `Transpose operation <../user_guide/operations.html#transpose>`__.
        """
        return TransposedMatrix(self)

    def clear(self):
        """In-place operation which clears all values in the Matrix.

        After the call, :attr:`nvals` will return 0. The :attr:`shape` will not change.
        """
        call("GrB_Matrix_clear", [self])

    def resize(self, nrows, ncols):
        """In-place operation which changes the :attr:`shape`.

        | Increasing :attr:`nrows` or :attr:`ncols` will expand with empty values.
        | Decreasing :attr:`nrows` or :attr:`ncols` will drop existing values above
            the new indices.
        """
        nrows = _as_scalar(nrows, _INDEX, is_cscalar=True)
        ncols = _as_scalar(ncols, _INDEX, is_cscalar=True)
        call("GrB_Matrix_resize", [self, nrows, ncols])
        self._nrows = nrows.value
        self._ncols = ncols.value

    def to_values(self, dtype=None, *, rows=True, columns=True, values=True, sort=True):
        """Extract the indices and values as a 3-tuple of numpy arrays
        corresponding to the COO format of the Matrix.

        Parameters
        ----------
        dtype :
            Requested dtype for the output values array.
        rows : bool, default=True
            Whether to return rows; will return `None` for rows if `False`
        columns  :bool, default=True
            Whether to return columns; will return `None` for columns if `False`
        values : bool, default=True
            Whether to return values; will return `None` for values if `False`
        sort : bool, default=True
            Whether to require sorted indices.
            If internally stored rowwise, the sorting will be first by rows, then by column.
            If internally stored columnwise, the sorting will be first by column, then by row.

        Returns
        -------
        np.ndarray[dtype=uint64] : Rows
        np.ndarray[dtype=uint64] : Columns
        np.ndarray : Values
        """
        if sort:
            if backend != "suitesparse":
                raise NotImplementedError()
            self.wait()  # sort in SS
        nvals = self._nvals
        if rows or backend != "suitesparse":
            c_rows = _CArray(size=nvals, name="&rows_array")
        else:
            c_rows = None
        if columns or backend != "suitesparse":
            c_columns = _CArray(size=nvals, name="&columns_array")
        else:
            c_columns = None
        if values or backend != "suitesparse":
            c_values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        else:
            c_values = None
        scalar = _scalar_index("s_nvals")
        scalar.value = nvals
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(
            f"GrB_Matrix_extractTuples_{dtype_name}",
            [c_rows, c_columns, c_values, _Pointer(scalar), self],
        )
        if values:
            c_values = c_values.array
            if dtype is not None:
                dtype = lookup_dtype(dtype)
                if dtype != self.dtype:
                    c_values = c_values.astype(dtype.np_type)  # copies
        return (
            c_rows.array if rows else None,
            c_columns.array if columns else None,
            c_values if values else None,
        )

    def build(self, rows, columns, values, *, dup_op=None, clear=False, nrows=None, ncols=None):
        """Rarely used method to insert values into an existing Matrix.

        The typical use case is to create a new Matrix and insert values
        at the same time using :meth:`from_values`.

        All the arguments are used identically in :meth:`from_values`, except for `clear`, which
        indicates whether to clear the Matrix prior to adding the new values.
        """
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        rows = ints_to_numpy_buffer(rows, np.uint64, name="row indices")
        columns = ints_to_numpy_buffer(columns, np.uint64, name="column indices")
        values, dtype = values_to_numpy_buffer(values, self.dtype)
        n = values.shape[0]
        if rows.size != n or columns.size != n:
            raise ValueError(
                "`rows` and `columns` and `values` lengths must match: "
                f"{rows.size}, {columns.size}, {values.size}"
            )
        if clear:
            self.clear()
        if nrows is not None or ncols is not None:
            if nrows is None:
                nrows = self._nrows
            if ncols is None:
                ncols = self._ncols
            self.resize(nrows, ncols)
        if n == 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            if not self.dtype._is_udt:
                dup_op = binary.plus
            else:
                dup_op = binary.any
        # SS:SuiteSparse-specific: we could use NULL for dup_op
        dup_op = get_typed_op(dup_op, self.dtype, kind="binary")
        if dup_op.opclass == "Monoid":
            dup_op = dup_op.binaryop
        else:
            self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        rows = _CArray(rows)
        columns = _CArray(columns)
        values = _CArray(values, self.dtype)
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(
            f"GrB_Matrix_build_{dtype_name}",
            [self, rows, columns, values, _as_scalar(n, _INDEX, is_cscalar=True), dup_op],
        )
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, dtype=None, *, clear=False, mask=None, name=None):
        """Create a duplicate of the Matrix.

        This is a full copy, not a view on the original.

        Parameters
        ----------
        dtype :
            Data type of the new Matrix. Normal typecasting rules apply.
        clear : bool, default=False
            If True, the returned Matrix will be empty.
        mask : Mask, optional
            Mask controlling which elements of the original to
            include in the copy.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix
        """
        if dtype is not None or mask is not None or clear:
            if dtype is None:
                dtype = self.dtype
            rv = Matrix(dtype, nrows=self._nrows, ncols=self._ncols, name=name)
            if not clear:
                rv(mask=mask)[...] = self
        else:
            new_mat = ffi_new("GrB_Matrix*")
            rv = Matrix._from_obj(new_mat, self.dtype, self._nrows, self._ncols, name=name)
            call("GrB_Matrix_dup", [_Pointer(rv), self])
        return rv

    def diag(self, k=0, dtype=None, *, name=None):
        """Return a Vector built from the diagonal values of the Matrix.

        Parameters
        ----------
        k : int
            Off-diagonal offset.
        dtype :
            Data type of the new Vector. Normal typecasting rules apply.
        name : str, optional
            Name to give the new Vector.

        Returns
        -------
        :class:`~graphblas.Vector`
        """
        from ..ss._core import diag

        return diag(self, k=k, dtype=dtype, name=name)

    def wait(self):
        """Wait for a computation to complete.

        In `non-blocking mode <../user_guide/init.html#graphblas-modes>`__,
        the computations may be delayed and not yet safe to use by multiple threads.
        Use wait to force completion of the Matrix.

        Has no effect in `blocking mode <../user_guide/init.html#graphblas-modes>`__.
        """
        # TODO: expose COMPLETE or MATERIALIZE options to the user
        call("GrB_Matrix_wait", [self, _MATERIALIZE])

    def get(self, row, col, default=None):
        """Get an element at (``row``, ``col``) indices as a Python scalar.

        Parameters
        ----------
        row : int
            Row index
        col : int
            Column index
        default :
            Value returned if no element exists at (row, col)

        Returns
        -------
        Python scalar
        """
        expr = self[row, col]
        if expr._is_scalar:
            rv = expr.new().value
            return default if rv is None else rv
        raise ValueError(
            "Bad row, col arguments in Matrix.get(...).  "
            "Indices should get a single element, which will be extracted as a Python scalar."
        )

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        warnings.warn(
            "`Matrix.new(...)` is deprecated; please use `Matrix(...)` instead.",
            DeprecationWarning,
        )
        return Matrix(dtype, nrows, ncols, name=name)

    @classmethod
    def from_values(
        cls,
        rows,
        columns,
        values,
        dtype=None,
        *,
        nrows=None,
        ncols=None,
        dup_op=None,
        name=None,
    ):
        """Create a new Matrix from row and column indices and values.

        Parameters
        ----------
        rows : list or np.ndarray
            Row indices.
        columns : list or np.ndarray
            Column indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Matrix. If not provided, the values will be inspected
            to choose an appropriate dtype.
        nrows : int, optional
            Number of rows in the Matrix. If not provided, ``nrows`` is computed
            from the maximum row index found in ``rows``.
        ncols : int, optional
            Number of columns in the Matrix. If not provided, ``ncols`` is computed
            from the maximum column index found in ``columns``.
        dup_op : :class:`~graphblas.core.operator.BinaryOp`, optional
            Function used to combine values if duplicate indices are found.
            Leaving ``dup_op=None`` will raise an error if duplicates are found.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix
        """
        rows = ints_to_numpy_buffer(rows, np.uint64, name="row indices")
        columns = ints_to_numpy_buffer(columns, np.uint64, name="column indices")
        values, new_dtype = values_to_numpy_buffer(values, dtype)
        # Compute nrows and ncols if not provided
        if nrows is None:
            if rows.size == 0:
                raise ValueError("No row indices provided. Unable to infer nrows.")
            nrows = int(rows.max()) + 1
        if ncols is None:
            if columns.size == 0:
                raise ValueError("No column indices provided. Unable to infer ncols.")
            ncols = int(columns.max()) + 1
        if dtype is None and values.ndim > 1:
            # Look for array-subtdype
            new_dtype = lookup_dtype(np.dtype((new_dtype.np_type, values.shape[1:])))
        # Create the new matrix
        C = cls(new_dtype, nrows, ncols, name=name)
        if values.ndim == 0:
            if dup_op is not None:
                raise ValueError(
                    "dup_op must be None if values is a scalar so that all "
                    "values can be identical.  Duplicate indices will be ignored."
                )
            # SS, SuiteSparse-specific: build_Scalar
            C.ss.build_scalar(rows, columns, values.tolist())
        else:
            # Add the data
            # This needs to be the original data to get proper error messages
            C.build(rows, columns, values, dup_op=dup_op)
        return C

    @classmethod
    def _from_csx(cls, fmt, indptr, indices, values, dtype, num, name):
        if fmt is _CSR_FORMAT:
            indices_name = "column indices"
        else:
            indices_name = "row indices"
        indptr = ints_to_numpy_buffer(indptr, np.uint64, name="index pointers")
        indices = ints_to_numpy_buffer(indices, np.uint64, name=indices_name)
        values, new_dtype = values_to_numpy_buffer(values, dtype)
        if num is None:
            if indices.size > 0:
                num = int(indices.max()) + 1
            else:
                num = 0
        if fmt is _CSR_FORMAT:
            nrows = indptr.size - 1
            ncols = num
        else:
            ncols = indptr.size - 1
            nrows = num
        if dtype is None and values.ndim > 1:
            # Look for array-subtdype
            new_dtype = lookup_dtype(np.dtype((new_dtype.np_type, values.shape[1:])))
        if values.ndim == 0:
            values = np.repeat(values, indices.size)
        new_mat = ffi_new("GrB_Matrix*")
        rv = Matrix._from_obj(new_mat, new_dtype, nrows, ncols, name=name)
        if new_dtype._is_udt:
            dtype_name = "UDT"
        else:
            dtype_name = new_dtype.name
        call(
            f"GrB_Matrix_import_{dtype_name}",
            [
                _Pointer(rv),
                new_dtype,
                _as_scalar(nrows, _INDEX, is_cscalar=True),
                _as_scalar(ncols, _INDEX, is_cscalar=True),
                _CArray(indptr),
                _CArray(indices),
                _CArray(values, dtype=new_dtype),
                _as_scalar(indptr.size, _INDEX, is_cscalar=True),
                _as_scalar(indices.size, _INDEX, is_cscalar=True),
                _as_scalar(values.shape[0], _INDEX, is_cscalar=True),
                fmt,
            ],
        )
        return rv

    @classmethod
    def from_csr(cls, indptr, col_indices, values, dtype=None, *, ncols=None, name=None):
        """Create a new Matrix from standard CSR representation of data.

        In CSR, the column indices for row i are stored in ``col_indices[indptr[i]:indptr[i+1]]``
        and the values are stored in ``values[indptr[i]:indptr[i+1]]``. The number of rows is
        inferred as ``nrows = len(indptr) - 1``.

        This always copies data. For zero-copy import with move semantics, see Matrix.ss.import_csr

        Parameters
        ----------
        indptr : list or np.ndarray
            Pointers for each row into col_indices and values; `indptr.size == nrows + 1`.
        col_indices : list or np.ndarray
            Column indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Matrix. If not provided, the values will be inspected
            to choose an appropriate dtype.
        ncols : int, optional
            Number of columns in the Matrix. If not provided, ``ncols`` is computed
            from the maximum column index found in ``col_indices``.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix

        See Also
        --------
        from_csc
        from_dcsr
        from_values
        Matrix.ss.import_csr
        io.from_scipy_sparse
        """
        return cls._from_csx(_CSR_FORMAT, indptr, col_indices, values, dtype, ncols, name)

    @classmethod
    def from_csc(cls, indptr, row_indices, values, dtype=None, *, nrows=None, name=None):
        """Create a new Matrix from standard CSC representation of data.

        In CSC, the row indices for column i are stored in ``row_indices[indptr[i]:indptr[i+1]]``
        and the values are stored in ``values[indptr[i]:indptr[i+1]]``. The number of columns is
        inferred as ``ncols = len(indptr) - 1``.

        This always copies data. For zero-copy import with move semantics, see Matrix.ss.import_csc

        Parameters
        ----------
        indptr : list or np.ndarray
            Pointers for each column into row_indices and values; `indptr.size == ncols + 1`.
        col_indices : list or np.ndarray
            Column indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Matrix. If not provided, the values will be inspected
            to choose an appropriate dtype.
        nrows : int, optional
            Number of rows in the Matrix. If not provided, ``ncols`` is computed
            from the maximum row index found in ``row_indices``.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix

        See Also
        --------
        from_csr
        from_dcsc
        from_values
        Matrix.ss.import_csc
        io.from_scipy_sparse
        """
        return cls._from_csx(_CSC_FORMAT, indptr, row_indices, values, dtype, nrows, name)

    @classmethod
    def from_dcsr(
        cls,
        compressed_rows,
        indptr,
        col_indices,
        values,
        dtype=None,
        *,
        nrows=None,
        ncols=None,
        name=None,
    ):
        """Create a new Matrix from DCSR (a.k.a. HyperCSR) representation of data.

        In DCSR, we store the index of each non-empty row in ``compressed_rows``.
        The column indices for row ``compressed_rows[i]`` are stored in
        ``col_indices[indptr[compressed_row[i]]:indptr[compressed_row[i]+1]]`` and the values
        are stored in ``values[indptr[compressed_row[i]]:indptr[compressed_row[i]+1]]``.

        This always copies data. For zero-copy import with move semantics,
        see Matrix.ss.import_hypercsr.

        Parameters
        ----------
        compressed_rows : list or np.ndarray
            Indices of non-empty rows; unique and sorted.
        indptr : list or np.ndarray
            Pointers for each non-empty row into col_indices and values.
        col_indices : list or np.ndarray
            Column indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Matrix. If not provided, the values will be inspected
            to choose an appropriate dtype.
        nrows : int, optional
            Number of rows in the Matrix. If not provided, ``nrows`` is computed
            from the maximum row index found in ``compressed_rows``.
        ncols : int, optional
            Number of columns in the Matrix. If not provided, ``ncols`` is computed
            from the maximum column index found in ``col_indices``.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix

        See Also
        --------
        from_csr
        from_dcsc
        from_values
        Matrix.ss.import_hypercsr
        io.from_scipy_sparse
        """
        if backend == "suitesparse":
            return cls.ss.import_hypercsr(
                rows=compressed_rows,
                indptr=indptr,
                col_indices=col_indices,
                values=values,
                nrows=nrows,
                ncols=ncols,
                dtype=dtype,
                name=name,
            )
        else:
            indptr = ints_to_numpy_buffer(indptr, np.int64, name="indptr")  # Ensure int, not uint
            if indptr.size == 0:
                raise InvalidValue("indptr array must not be empty")
            elif indptr.size == 1:
                nrows = 0 if nrows is None else nrows
                ncols = 0 if ncols is None else ncols
                rows = np.empty(0, np.uint64)
            else:
                rows = np.repeat(compressed_rows, np.diff(indptr))
            return cls.from_coo(
                rows, col_indices, values, dtype, nrows=nrows, ncols=ncols, name=name
            )

    @classmethod
    def from_dcsc(
        cls,
        compressed_cols,
        indptr,
        row_indices,
        values,
        dtype=None,
        *,
        nrows=None,
        ncols=None,
        name=None,
    ):
        """Create a new Matrix from DCSC (a.k.a. HyperCSC) representation of data.

        In DCSC, we store the index of each non-empty column in ``compressed_cols``.
        The row indices for column ``compressed_cols[i]`` are stored in
        ``col_indices[indptr[compressed_cols[i]]:indptr[compressed_cols[i]+1]]`` and the values
        are stored in ``values[indptr[compressed_cols[i]]:indptr[compressed_cols[i]+1]]``.

        This always copies data. For zero-copy import with move semantics,
        see Matrix.ss.import_hypercsc.

        Parameters
        ----------
        compressed_cols : list or np.ndarray
            Indices of non-empty columns; unique and sorted.
        indptr : list or np.ndarray
            Pointers for each non-empty columns into row_indices and values.
        row_indices : list or np.ndarray
            Row indices.
        values : list or np.ndarray or scalar
            List of values. If a scalar is provided, all values will be set to this single value.
        dtype :
            Data type of the Matrix. If not provided, the values will be inspected
            to choose an appropriate dtype.
        nrows : int, optional
            Number of rows in the Matrix. If not provided, ``nrows`` is computed
            from the maximum row index found in ``row_indices``.
        ncols : int, optional
            Number of columns in the Matrix. If not provided, ``ncols`` is computed
            from the maximum column index found in ``compressed_cols``.
        name : str, optional
            Name to give the Matrix.

        Returns
        -------
        Matrix

        See Also
        --------
        from_csc
        from_dcsr
        from_values
        Matrix.ss.import_hypercsc
        io.from_scipy_sparse
        """
        if backend == "suitesparse":
            return cls.ss.import_hypercsc(
                cols=compressed_cols,
                indptr=indptr,
                row_indices=row_indices,
                values=values,
                nrows=nrows,
                ncols=ncols,
                dtype=dtype,
                name=name,
            )
        else:
            indptr = ints_to_numpy_buffer(indptr, np.int64, name="indptr")  # Ensure int, not uint
            if indptr.size == 0:
                raise InvalidValue("indptr array must not be empty")
            elif indptr.size == 1:
                nrows = 0 if nrows is None else nrows
                ncols = 0 if ncols is None else ncols
                cols = np.empty(0, np.uint64)
            else:
                cols = np.repeat(compressed_cols, np.diff(indptr))
            return cls.from_coo(
                row_indices, cols, values, dtype, nrows=nrows, ncols=ncols, name=name
            )

    def _to_csx(self, fmt, dtype=None):
        Ap_len = _scalar_index("Ap_len")
        Ai_len = _scalar_index("Ai_len")
        Ax_len = _scalar_index("Ax_len")
        call(
            "GrB_Matrix_exportSize",
            [_Pointer(Ap_len), _Pointer(Ai_len), _Pointer(Ax_len), fmt, self],
        )
        Ap = _CArray(size=Ap_len.gb_obj[0], name="&Ap")
        Ai = _CArray(size=Ai_len.gb_obj[0], name="&Ai")
        Ax = _CArray(size=Ax_len.gb_obj[0], name="&Ax", dtype=self.dtype)
        if self.dtype._is_udt:
            dtype_name = "UDT"
        else:
            dtype_name = self.dtype.name
        call(
            f"GrB_Matrix_export_{dtype_name}",
            [Ap, Ai, Ax, _Pointer(Ap_len), _Pointer(Ai_len), _Pointer(Ax_len), fmt, self],
        )

        # Unwrap objects with walrus operator and trim to proper size if necessary
        if (Ap := Ap.array).size > (Ap_len := Ap_len.gb_obj[0]):  # pragma: no cover
            Ap = Ap[:Ap_len]
        if (Ai := Ai.array).size > (Ai_len := Ai_len.gb_obj[0]):  # pragma: no cover
            Ai = Ai[:Ai_len]
        if (Ax := Ax.array).shape[0] > (Ax_len := Ax_len.gb_obj[0]):  # pragma: no cover
            Ax = Ax[:Ax_len]

        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                Ax = Ax.astype(dtype.np_type)
        return Ap, Ai, Ax

    def to_csr(self, dtype=None):
        """Returns three arrays of the standard CSR representation: indptr, col_indices, values

        In CSR, the column indices for row i are stored in ``col_indices[indptr[i]:indptr[i+1]]``
        and the values are stored in ``values[indptr[i]:indptr[i+1]]``.

        This copies data and leaves the Matrix unmodified. For zero-copy move semantics,
        see Matrix.ss.export.

        Returns
        -------
        np.ndarray[dtype=uint64] : indptr
        np.ndarray[dtype=uint64] : col_indices
        np.ndarray : values

        See Also
        --------
        to_csc
        to_dcsr
        to_values
        Matrix.ss.export
        io.to_scipy_sparse
        """
        return self._to_csx(_CSR_FORMAT, dtype)

    def to_csc(self, dtype=None):
        """Returns three arrays of the standard CSC representation: indptr, row_indices, values

        In CSC, the row indices for column i are stored in ``row_indices[indptr[i]:indptr[i+1]]``
        and the values are stored in ``values[indptr[i]:indptr[i+1]]``.

        This copies data and leaves the Matrix unmodified. For zero-copy move semantics,
        see Matrix.ss.export.

        Returns
        -------
        np.ndarray[dtype=uint64] : indptr
        np.ndarray[dtype=uint64] : row_indices
        np.ndarray : values

        See Also
        --------
        to_csr
        to_dcsc
        to_values
        Matrix.ss.export
        io.to_scipy_sparse
        """
        return self._to_csx(_CSC_FORMAT, dtype)

    def to_dcsr(self, dtype=None):
        """Returns four arrays of DCSR representation: compressed_rows, indptr, col_indices, values

        In DCSR, we store the index of each non-empty row in ``compressed_rows``.
        The column indices for row ``compressed_rows[i]`` are stored in
        ``col_indices[indptr[compressed_row[i]]:indptr[compressed_row[i]+1]]`` and the values
        are stored in ``values[indptr[compressed_row[i]]:indptr[compressed_row[i]+1]]``.

        This copies data and leaves the Matrix unmodified. For zero-copy move semantics,
        see Matrix.ss.export.

        Returns
        -------
        np.ndarray[dtype=uint64] : compressed_rows
        np.ndarray[dtype=uint64] : indptr
        np.ndarray[dtype=uint64] : col_indices
        np.ndarray : values

        See Also
        --------
        to_csr
        to_dcsc
        to_values
        Matrix.ss.export
        io.to_scipy_sparse
        """
        if backend == "suitesparse":
            info = self.ss.export("hypercsr", sort=True)
            compressed_rows = info["rows"]
            indptr = info["indptr"]
            cols = info["col_indices"]
            values = info["values"]
        else:
            rows, cols, values = self.to_coo()  # sorted by row then col
            compressed_rows, indices = np.unique(rows, return_index=True)
            indptr = np.empty(indices.size + 1, np.uint64)
            indptr[:-1] = indices
            indptr[-1] = rows.size
        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                values = values.astype(dtype.np_type)
        return compressed_rows, indptr, cols, values

    def to_dcsc(self, dtype=None):
        """Returns four arrays of DCSC representation: compressed_cols, indptr, row_indices, values

        In DCSC, we store the index of each non-empty column in ``compressed_cols``.
        The row indices for column ``compressed_cols[i]`` are stored in
        ``col_indices[indptr[compressed_cols[i]]:indptr[compressed_cols[i]+1]]`` and the values
        are stored in ``values[indptr[compressed_cols[i]]:indptr[compressed_cols[i]+1]]``.

        This copies data and leaves the Matrix unmodified. For zero-copy move semantics,
        see Matrix.ss.export.

        Returns
        -------
        np.ndarray[dtype=uint64] : compressed_cols
        np.ndarray[dtype=uint64] : indptr
        np.ndarray[dtype=uint64] : row_indices
        np.ndarray : values

        See Also
        --------
        to_csc
        to_dcsr
        to_values
        Matrix.ss.export
        io.to_scipy_sparse
        """
        if backend == "suitesparse":
            info = self.ss.export("hypercsc", sort=True)
            compressed_cols = info["cols"]
            indptr = info["indptr"]
            rows = info["row_indices"]
            values = info["values"]
        else:
            rows, cols, values = self.to_coo(sort=False)
            ind = np.lexsort((rows, cols))  # sort by columns, then rows
            rows = rows[ind]
            cols = cols[ind]
            values = values[ind]
            compressed_cols, indices = np.unique(cols, return_index=True)
            indptr = np.empty(indices.size + 1, np.uint64)
            indptr[:-1] = indices
            indptr[-1] = cols.size
        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                values = values.astype(dtype.np_type)
        return compressed_cols, indptr, rows, values

    from_coo = from_values  # Alias
    to_coo = to_values  # Alias

    @property
    def _carg(self):
        return self.gb_obj[0]

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to __setitem__ to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=None):
        """Perform element-wise computation on the union of sparse values,
        similar to how one expects addition to work for sparse matrices.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_add and :meth:`ewise_union`.

        Parameters
        ----------
        other : Matrix
            The other matrix in the computation
        op : Monoid or BinaryOp
            Operator to use on intersecting values
        require_monoid : deprecated

        Returns
        -------
        Matrix with a structure formed as the union of the input structures

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.ewise_add(B, op=monoid.max)

            # Functional syntax
            C << monoid.max(A | B)
        """
        if require_monoid is not None:
            warnings.warn(
                "require_monoid keyword is deprecated; "
                "future behavior will be like `require_monoid=False`",
                DeprecationWarning,
            )
        else:
            require_monoid = False
        method_name = "ewise_add"
        other = self._expect_type(
            other,
            (Matrix, TransposedMatrix, Vector),
            within=method_name,
            argname="other",
            op=op,
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        if require_monoid:  # pragma: no cover
            if op.opclass != "BinaryOp" or op.monoid is None:
                self._expect_op(
                    op,
                    "Monoid",
                    within=method_name,
                    argname="op",
                    extra_message="A BinaryOp may be given if require_monoid keyword is False.",
                )
        else:
            self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if other.ndim == 1:
            # Broadcast rowwise from the right
            if self._ncols != other._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the right "
                    f"to rows of Matrix in {method_name}.  Matrix.ncols (={self._ncols}) "
                    f"must equal Vector.size (={other._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _m_add_v, (self, other, op)],  # [*expr_args, func, args]
                nrows=self._nrows,
                ncols=self._ncols,
                op=op,
            )
        expr = MatrixExpression(
            method_name,
            f"GrB_Matrix_eWiseAdd_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )
        if self.shape != other.shape:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_mult(self, other, op=binary.times):
        """Perform element-wise computation on the intersection of sparse values,
        similar to how one expects multiplication to work for sparse matrices.

        See the
        `Element-wise Intersection <../user_guide/operations.html#element-wise-intersection>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Matrix
            The other matrix in the computation
        op : Monoid or BinaryOp
            Operator to use on intersecting values

        Returns
        -------
        Matrix with a structure formed as the intersection of the input structures

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.ewise_mult(B, op=binary.gt)

            # Functional syntax
            C << binary.gt(A & B)
        """
        method_name = "ewise_mult"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix, Vector), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if other.ndim == 1:
            # Broadcast rowwise from the right
            if self._ncols != other._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the right "
                    f"to rows of Matrix in {method_name}.  Matrix.ncols (={self._ncols}) "
                    f"must equal Vector.size (={other._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _m_mult_v, (self, other, op)],  # [*expr_args, func, args]
                nrows=self._nrows,
                ncols=self._ncols,
                op=op,
            )
        expr = MatrixExpression(
            method_name,
            f"GrB_Matrix_eWiseMult_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )
        if self.shape != other.shape:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_union(self, other, op, left_default, right_default):
        """Perform element-wise computation on the union of sparse values,
        similar to how one expects subtraction to work for sparse matrices.

        See the `Element-wise Union <../user_guide/operations.html#element-wise-union>`__
        section in the User Guide for more details, especially about the difference between
        ewise_union and :meth:`ewise_add`.

        Parameters
        ----------
        other : Matrix
            The other matrix in the computation
        op : Monoid or BinaryOp
            Operator to use
        left_default :
            Scalar value to use when the index on the left is missing
        right_default :
            Scalar value to use when the index on the right is missing

        Returns
        -------
        Matrix with a structure formed as the union of the input structures

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.ewise_union(B, op=binary.div, left_default=1, right_default=1)

            # Functional syntax
            C << binary.div(A | B, left_default=1, right_default=1)
        """
        # SS, SuiteSparse-specific: eWiseUnion
        method_name = "ewise_union"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix, Vector), within=method_name, argname="other", op=op
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
        if other.ndim == 1:
            # Broadcast rowwise from the right
            if self._ncols != other._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the right "
                    f"to rows of Matrix in {method_name}.  Matrix.ncols (={self._ncols}) "
                    f"must equal Vector.size (={other._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, left, other, right, _m_union_v, (self, other, left, right, op)],
                expr_repr=expr_repr,
                nrows=self._nrows,
                ncols=self._ncols,
                op=op,
            )
        expr = MatrixExpression(
            method_name,
            "GxB_Matrix_eWiseUnion",
            [self, left, other, right],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
            expr_repr=expr_repr,
        )
        if self.shape != other.shape:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def mxv(self, other, op=semiring.plus_times):
        """Perform matrix-vector multiplication.

        See the `Matrix Multiply <../user_guide/operations.html#matrix-multiply>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Vector
            The vector, treated as an (nx1) column matrix
        op : :class:`~graphblas.core.operator.Semiring`
            Semiring used in the computation

        Returns
        -------
        :class:`~graphblas.Vector`

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.mxv(v, op=semiring.min_plus)

            # Functional syntax
            C << semiring.min_plus(A @ v)
        """
        method_name = "mxv"
        other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = VectorExpression(
            method_name,
            "GrB_mxv",
            [self, other],
            op=op,
            size=self._nrows,
            at=self._is_transposed,
        )
        if self._ncols != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def mxm(self, other, op=semiring.plus_times):
        """Perform matrix-matrix multiplication.

        See the `Matrix Multiply <../user_guide/operations.html#matrix-multiply>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Matrix
            The matrix on the right side in the computation
        op : :class:`~graphblas.core.operator.Semiring`
            Semiring used in the computation

        Returns
        -------
        Matrix

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.mxm(B, op=semiring.min_plus)

            # Functional syntax
            C << semiring.min_plus(A @ B)
        """
        method_name = "mxm"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = MatrixExpression(
            method_name,
            "GrB_mxm",
            [self, other],
            op=op,
            nrows=self._nrows,
            ncols=other._ncols,
            at=self._is_transposed,
            bt=other._is_transposed,
        )
        if self._ncols != other._nrows:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def kronecker(self, other, op=binary.times):
        """Compute the kronecker product or sum (depending on the op).

        See the `Kronecker <../user_guide/operations.html#kronecker>`__
        section in the User Guide for more details.

        Parameters
        ----------
        other : Matrix
            The matrix on the right side in the computation
        op : :class:`~graphblas.core.operator.BinaryOp`
            Operator used on the combination of elements

        Returns
        -------
        Matrix

        Examples
        --------

        .. code-block:: python

            C << A.kronecker(B, op=binary.times)
        """
        method_name = "kronecker"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        return MatrixExpression(
            method_name,
            f"GrB_Matrix_kronecker_{op.opclass}",
            [self, other],
            op=op,
            nrows=self._nrows * other._nrows,
            ncols=self._ncols * other._ncols,
            at=self._is_transposed,
            bt=other._is_transposed,
        )

    def apply(self, op, right=None, *, left=None):
        """Create a new Matrix by applying ``op`` to each element of the Matrix.

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
        Matrix

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.apply(op.abs)

            # Functional syntax
            C << op.abs(A)
        """
        method_name = "apply"
        extra_message = (
            "apply only accepts UnaryOp with no scalars or BinaryOp with `left` or `right` scalar "
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
            cfunc_name = "GrB_Matrix_apply"
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
                cfunc_name = f"GrB_Matrix_apply_BinaryOp1st_{dtype_name}"
            else:
                cfunc_name = "GrB_Matrix_apply_BinaryOp1st_Scalar"
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
                cfunc_name = f"GrB_Matrix_apply_{cfunc_method}_{dtype_name}"
            else:
                cfunc_name = f"GrB_Matrix_apply_{cfunc_method}_Scalar"
            args = [self, right]
            expr_repr = "{0.name}.apply({op}, right={1._expr_name})"
        else:
            raise TypeError("Cannot provide both `left` and `right` to apply")
        return MatrixExpression(
            method_name,
            cfunc_name,
            args,
            op=op,
            nrows=self._nrows,
            ncols=self._ncols,
            expr_repr=expr_repr,
            at=self._is_transposed,
            bt=self._is_transposed,
        )

    def select(self, op, thunk=None):
        """Create a new Matrix by applying ``op`` to each element of the Matrix
        and keeping those elements where ``op`` returns True.

        See the `Select <../user_guide/operations.html#select>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.SelectOp`or
             :class:`~graphblas.core.operator.IndexUnaryOp`
            Operator to apply
        thunk :
            Scalar passed to operator

        Returns
        -------
        Matrix

        Examples
        --------

        .. code-block:: python

            # Method syntax
            C << A.select(">=", 1)

            # Functional syntax
            C << select.value(A >= 1)
        """
        method_name = "select"
        if isinstance(op, str):
            op = select.from_string(op)
        else:
            if isinstance(op, MatrixExpression):
                # Try to rewrite e.g. `A.select(A == 7)` to `gb.select.value(A == 7)`
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
                # e.g., `A.select(B.S)` or `A.select(B < 7)`
                mask = _check_mask(op)
                if thunk is not None:
                    raise TypeError(
                        "thunk argument not None when calling select with mask or boolean object"
                    )
                self._expect_type(mask.parent, (Matrix, Mask), within=method_name, argname="op")
                return MatrixExpression(
                    "select",
                    None,
                    [self, mask, _select_mask, (self, mask)],  # [*expr_args, func, args]
                    expr_repr="{0.name}.select({1.name})",
                    nrows=self.nrows,
                    ncols=self.ncols,
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
            else:
                dtype_name = thunk.dtype.name
            cfunc_name = f"GrB_Matrix_select_{dtype_name}"
        else:
            cfunc_name = "GrB_Matrix_select_Scalar"
        return MatrixExpression(
            method_name,
            cfunc_name,
            [self, thunk],
            op=op,
            expr_repr="{0.name}.select({op}, thunk={1._expr_name})",
            nrows=self._nrows,
            ncols=self._ncols,
            dtype=self.dtype,
            at=self._is_transposed,
        )

    def reduce_rowwise(self, op=monoid.plus):
        """Create a new Vector by reducing values per-row in the Matrix using ``op``.

        See the `Reduce <../user_guide/operations.html#reduce>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.Monoid`
            Reduction operator

        Returns
        -------
        Vector

        Examples
        --------

        .. code-block:: python

            w << A.reduce_rowwise(monoid.plus)
        """
        method_name = "reduce_rowwise"
        op = get_typed_op(op, self.dtype, kind="binary|aggregator")
        self._expect_op(op, ("BinaryOp", "Monoid", "Aggregator"), within=method_name, argname="op")
        # Using a monoid may be more efficient, so change to one if possible.
        # Also, SuiteSparse doesn't like user-defined binarops here.
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        return VectorExpression(
            method_name,
            f"GrB_Matrix_reduce_{op.opclass}",
            [self],
            op=op,
            size=self._nrows,
            at=self._is_transposed,
        )

    def reduce_columnwise(self, op=monoid.plus):
        """Create a new Vector by reducing values per-column in the Matrix using ``op``.

        See the `Reduce <../user_guide/operations.html#reduce>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.Monoid`
            Reduction operator

        Returns
        -------
        :class:`~graphblas.Vector`

        Examples
        --------

        .. code-block:: python

            w << A.reduce_columnwise(monoid.plus)
        """
        method_name = "reduce_columnwise"
        op = get_typed_op(op, self.dtype, kind="binary|aggregator")
        self._expect_op(op, ("BinaryOp", "Monoid", "Aggregator"), within=method_name, argname="op")
        # Using a monoid may be more efficient, so change to one if possible.
        # Also, SuiteSparse doesn't like user-defined binarops here.
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        return VectorExpression(
            method_name,
            f"GrB_Matrix_reduce_{op.opclass}",
            [self],
            op=op,
            size=self._ncols,
            at=not self._is_transposed,
        )

    def reduce_scalar(self, op=monoid.plus, *, allow_empty=True):
        """
        Reduce all values in the Matrix into a single value using ``op``.

        See the `Reduce <../user_guide/operations.html#reduce>`__
        section in the User Guide for more details.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.Monoid`
            Reduction operator
        allow_empty : bool, default=True
            If False and the Matrix is empty, the Scalar result
            will hold the monoid identity rather than a missing value

        Returns
        -------
        :class:`~graphblas.Scalar`

        Examples
        --------

        .. code-block:: python

            total << A.reduce_scalar(monoid.plus)
        """
        method_name = "reduce_scalar"
        op = get_typed_op(op, self.dtype, kind="binary|aggregator")
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        else:
            self._expect_op(op, ("Monoid", "Aggregator"), within=method_name, argname="op")
        if op.opclass == "Aggregator":
            if op.name in {"argmin", "argmax", "first_index", "last_index"}:
                raise ValueError(f"Aggregator {op.name} may not be used with Matrix.reduce_scalar.")
            if not allow_empty and op.parent._monoid is None:
                # But we still kindly allow it if it's a monoid-only aggregator such as sum
                raise ValueError("allow_empty=False not allowed when using Aggregators")
        if allow_empty:
            cfunc_name = "GrB_Matrix_reduce_Monoid_Scalar"
        elif self.dtype._is_udt:
            cfunc_name = "GrB_Matrix_reduce_UDT"
        else:
            cfunc_name = "GrB_Matrix_reduce_{output_dtype}"
        return ScalarExpression(
            method_name,
            cfunc_name,
            [self],
            op=op,  # to be determined later
            is_cscalar=not allow_empty,
        )

    # Unofficial methods
    def reposition(self, row_offset, column_offset, *, nrows=None, ncols=None):
        """Create a new Matrix with values identical to the original Matrix,
        but repositioned within the (nrows x ncols) space by adding offsets to the indices.

        Positive offset moves values to the right (or down), negative to the left (or up).
        Values repositioned outside of the new Matrix are dropped (i.e. they don't wrap around).

        *Note*: This is not a standard GraphBLAS method.
        It is implemented using extract and assign.

        Parameters
        ----------
        row_offset : int
            Offset for the row indices.
        column_offset : int
            Offset for the column indices.
        nrows : int, optional
            If specified, the new Matrix will be sized with nrows.
            Default is the same number of rows as the original Matrix.
        ncols : int, optional
            If specified, the new Matrix will be sized with ncols.
            Default is the same number of columns as the original Matrix.

        Returns
        -------
        Matrix

        Examples
        --------

        .. code-block:: python

            C = A.reposition(1, 2).new()
        """
        if nrows is None:
            nrows = self._nrows
        else:
            nrows = int(nrows)
        if ncols is None:
            ncols = self._ncols
        else:
            ncols = int(ncols)
        row_offset = int(row_offset)
        if row_offset < 0:
            row_start = -row_offset
            row_stop = row_start + nrows
        else:
            row_start = 0
            row_stop = max(0, nrows - row_offset)
        col_offset = int(column_offset)
        if col_offset < 0:
            col_start = -col_offset
            col_stop = col_start + ncols
        else:
            col_start = 0
            col_stop = max(0, ncols - col_offset)
        if self._is_transposed:
            chunk = (
                self._matrix[col_start:col_stop, row_start:row_stop].new(name="M_repositioning").T
            )
        else:
            chunk = self[row_start:row_stop, col_start:col_stop].new(name="M_repositioning")
        indices = (
            slice(row_start + row_offset, row_start + row_offset + chunk._nrows),
            slice(col_start + col_offset, col_start + col_offset + chunk._ncols),
        )
        return MatrixExpression(
            "reposition",
            None,
            [self, _reposition, (indices, chunk)],  # [*expr_args, func, args]
            expr_repr="{0.name}.reposition(%d, %d)" % (row_offset, column_offset),
            nrows=nrows,
            ncols=ncols,
            dtype=self.dtype,
        )

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(self, resolved_indexes, dtype=None, *, is_cscalar, name=None, result=None):
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = lookup_dtype(dtype)
        rowidx, colidx = resolved_indexes.indices
        if self._is_transposed:
            rowidx, colidx = colidx, rowidx
        if result is None:
            result = Scalar(dtype, is_cscalar=is_cscalar, name=name)
        if is_cscalar:
            dtype_name = "UDT" if dtype._is_udt else dtype.name
            if (
                call(
                    f"GrB_Matrix_extractElement_{dtype_name}",
                    [_Pointer(result), self, rowidx.index, colidx.index],
                )
                is not NoValue
            ):
                result._empty = False
        else:
            call("GrB_Matrix_extractElement_Scalar", [result, self, rowidx.index, colidx.index])
        return result

    def _prep_for_extract(self, resolved_indexes):
        method_name = "__getitem__"
        rowidx, colidx = resolved_indexes.indices

        if rowidx.size is None:
            # Row-only selection; GraphBLAS doesn't have this method, so we hack it using transpose
            return VectorExpression(
                method_name,
                "GrB_Col_extract",
                [self, colidx, colidx.cscalar, rowidx],
                expr_repr="{0.name}[{3._expr_name}, {1._expr_name}]",
                size=colidx.size,
                dtype=self.dtype,
                at=not self._is_transposed,
            )
        elif colidx.size is None:
            # Column-only selection
            return VectorExpression(
                method_name,
                "GrB_Col_extract",
                [self, rowidx, rowidx.cscalar, colidx],
                expr_repr="{0.name}[{1._expr_name}, {3._expr_name}]",
                size=rowidx.size,
                dtype=self.dtype,
                at=self._is_transposed,
            )
        else:
            return MatrixExpression(
                method_name,
                "GrB_Matrix_extract",
                [self, rowidx, rowidx.cscalar, colidx, colidx.cscalar],
                expr_repr="{0.name}[{1._expr_name}, {3._expr_name}]",
                nrows=rowidx.size,
                ncols=colidx.size,
                dtype=self.dtype,
                at=self._is_transposed,
            )

    def _assign_element(self, resolved_indexes, value):
        rowidx, colidx = resolved_indexes.indices
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
                call("GrB_Matrix_removeElement", [self, rowidx.index, colidx.index])
                return
            if value.dtype._is_udt:
                dtype_name = "UDT"
                value = _Pointer(value)
            else:
                dtype_name = value.dtype.name
            cfunc_name = f"GrB_Matrix_setElement_{dtype_name}"
        else:
            cfunc_name = "GrB_Matrix_setElement_Scalar"
        call(cfunc_name, [self, value, rowidx.index, colidx.index])

    def _prep_for_assign(self, resolved_indexes, value, mask=None, is_submask=False):
        method_name = "__setitem__"
        rowidx, colidx = resolved_indexes.indices
        rowsize = rowidx.size
        rows = rowidx.index
        rowscalar = rowidx.cscalar
        colsize = colidx.size
        cols = colidx.index
        colscalar = colidx.cscalar

        if rowsize is not None or colsize is not None:
            extra_message = "Literal scalars and lists also accepted."
        else:
            extra_message = "Literal scalars also accepted."

        value_type = output_type(value)
        if value_type is Vector:
            if type(value) is not Vector:
                value = self._expect_type(
                    value,
                    Vector,
                    within=method_name,
                )
            if rowsize is None and colsize is not None:
                # Row-only selection
                if mask is not None and type(mask.parent) is Matrix:
                    if is_submask:
                        # C[i, J](M) << v
                        raise TypeError(
                            "Indices for subassign imply Vector submask, "
                            "but got Matrix mask instead"
                        )
                    else:
                        # C(M)[i, J] << v
                        # Upcast v to a Matrix and use Matrix_assign
                        rows = _CArray([rows.value])
                        rowscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                        expr = MatrixExpression(
                            method_name,
                            "GrB_Matrix_assign",
                            [value._as_matrix(), rows, rowscalar, cols, colscalar],
                            expr_repr="[[{2._expr_name} rows], [{4._expr_name} cols]] = {0.name}",
                            nrows=self._nrows,
                            ncols=self._ncols,
                            dtype=self.dtype,
                            at=True,
                        )
                else:
                    if is_submask:
                        # C[i, J](m) << v
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Row_subassign"
                        expr_repr = (
                            "[{1._expr_name}, [{3._expr_name} cols]](%s) << {0.name}" % mask.name
                        )
                    else:
                        # C(m)[i, J] << v
                        # C[i, J] << v
                        cfunc_name = "GrB_Row_assign"
                        expr_repr = "[{1._expr_name}, [{3._expr_name} cols]] = {0.name}"
                    expr = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value, rows, cols, colscalar],
                        expr_repr=expr_repr,
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
            elif colsize is None and rowsize is not None:
                # Column-only selection
                if mask is not None and type(mask.parent) is Matrix:
                    if is_submask:
                        # C[I, j](M) << v
                        raise TypeError(
                            "Indices for subassign imply Vector submask, "
                            "but got Matrix mask instead"
                        )
                    else:
                        # C(M)[I, j] << v
                        # Upcast v to a Matrix and use Matrix_assign
                        cols = _CArray([cols.value])
                        colscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                        expr = MatrixExpression(
                            method_name,
                            "GrB_Matrix_assign",
                            [value._as_matrix(), rows, rowscalar, cols, colscalar],
                            expr_repr="[[{2._expr_name} rows], [{4._expr_name} cols]] = {0.name}",
                            nrows=self._nrows,
                            ncols=self._ncols,
                            dtype=self.dtype,
                        )
                else:
                    if is_submask:
                        # C[I, j](m) << v
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Col_subassign"
                        expr_repr = (
                            "[{1._expr_name}, [{3._expr_name} cols]](%s) << {0.name}" % mask.name
                        )
                    else:
                        # C(m)[I, j] << v
                        # C[I, j] << v
                        cfunc_name = "GrB_Col_assign"
                        expr_repr = "[{1._expr_name}, [{3._expr_name} cols]] = {0.name}"
                    expr = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value, rows, rowscalar, cols],
                        expr_repr=expr_repr,
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
            elif colsize is None and rowsize is None:
                # C[i, j] << v  (mask doesn't matter)
                value = self._expect_type(
                    value,
                    Scalar,
                    within=method_name,
                    extra_message=extra_message,
                )
            else:
                # C[I, J] << v  (mask doesn't matter)
                value = self._expect_type(
                    value,
                    (Scalar, Matrix, TransposedMatrix),
                    within=method_name,
                    extra_message=extra_message,
                )
        elif value_type in {Matrix, TransposedMatrix}:
            if type(value) not in {Matrix, TransposedMatrix}:
                value = self._expect_type(
                    value,
                    (Matrix, TransposedMatrix),
                    within=method_name,
                )
            if rowsize is None or colsize is None:
                if rowsize is None and colsize is None:
                    # C[i, j] << A  (mask doesn't matter)
                    value = self._expect_type(
                        value,
                        Scalar,
                        within=method_name,
                        extra_message=extra_message,
                    )
                else:
                    # C[I, j] << A
                    # C[i, J] << A  (mask doesn't matter)
                    value = self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        extra_message=extra_message,
                    )
            if is_submask:
                # C[I, J](M) << A
                # SS, SuiteSparse-specific: subassign
                cfunc_name = "GrB_Matrix_subassign"
                expr_repr = (
                    "[[{2._expr_name} rows], [{4._expr_name} cols]](%s) << {0.name}" % mask.name
                )
            else:
                # C[I, J] << A
                # C(M)[I, J] << A
                cfunc_name = "GrB_Matrix_assign"
                expr_repr = "[[{2._expr_name} rows], [{4._expr_name} cols]] = {0.name}"
            expr = MatrixExpression(
                method_name,
                cfunc_name,
                [value, rows, rowscalar, cols, colscalar],
                expr_repr=expr_repr,
                nrows=self._nrows,
                ncols=self._ncols,
                dtype=self.dtype,
                at=value._is_transposed,
            )
        else:
            if type(value) is not Scalar:
                dtype = self.dtype if self.dtype._is_udt else None
                try:
                    value = Scalar.from_value(value, dtype, is_cscalar=None, name="")
                except (TypeError, ValueError):
                    if rowsize is not None or colsize is not None:
                        try:
                            values, dtype = values_to_numpy_buffer(value, dtype, copy=True)
                        except Exception:
                            pass
                        else:
                            shape = values.shape
                            if rowsize is None or colsize is None:
                                # C[I, j] << [1, 2, 3]
                                # C[i, J] << [1, 2, 3]
                                # C(M)[I, j] << [1, 2, 3]
                                # C(M)[i, J] << [1, 2, 3]
                                # C[I, j](m) << [1, 2, 3]
                                # C[i, J](m) << [1, 2, 3]
                                expected_shape = (rowsize or colsize,)
                                try:
                                    vals = Vector.ss.import_full(
                                        values, dtype=dtype, take_ownership=True
                                    )
                                    if dtype.np_type.subdtype is not None:
                                        shape = vals.shape
                                except Exception:
                                    vals = None
                            else:
                                # C[I, J] << [[1, 2, 3], [4, 5, 6]]
                                # C(M)[I, J] << [[1, 2, 3], [4, 5, 6]]
                                # C[I, J](M) << [[1, 2, 3], [4, 5, 6]]
                                expected_shape = (rowsize, colsize)
                                try:
                                    vals = Matrix.ss.import_fullr(
                                        values, dtype=dtype, take_ownership=True
                                    )
                                    if dtype.np_type.subdtype is not None:
                                        shape = vals.shape
                                except Exception:
                                    vals = None
                            if vals is None or shape != expected_shape:
                                if dtype.np_type.subdtype is not None:
                                    extra = (
                                        " (this is assigning to a matrix with sub-array dtype "
                                        f"({dtype}), so array shape should include dtype shape)"
                                    )
                                else:
                                    extra = ""
                                raise ValueError(
                                    f"shape mismatch: value array of shape {shape} "
                                    f"does not match indexing of shape {expected_shape}"
                                    f"{extra}"
                                ) from None
                            return self._prep_for_assign(
                                resolved_indexes, vals, mask=mask, is_submask=is_submask
                            )
                    if rowsize is None or colsize is None:
                        types = (Scalar, Vector)
                    else:
                        types = (Scalar, Matrix, TransposedMatrix)
                    value = self._expect_type(
                        value,
                        types,
                        within=method_name,
                        argname="value",
                        extra_message=extra_message,
                    )
            if mask is not None and type(mask.parent) is Vector:
                if rowsize is None and colsize is not None:
                    if is_submask:
                        # C[i, J](m) << c
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Row_subassign"
                        value_vector = Vector(value.dtype, size=mask.parent._size, name="v_temp")
                        expr_repr = (
                            "[{1._expr_name}, [{3._expr_name} cols]](%s) << {0.name}" % mask.name
                        )
                    else:
                        # C(m)[i, J] << c
                        # C[i, J] << c
                        cfunc_name = "GrB_Row_assign"
                        value_vector = Vector(value.dtype, size=colsize, name="v_temp")
                        expr_repr = "[{1._expr_name}, [{3._expr_name} cols]] = {0.name}"
                    # SS, SuiteSparse-specific: assume efficient vector with single scalar
                    value_vector << value

                    # Row-only selection
                    expr = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value_vector, rows, cols, colscalar],
                        expr_repr=expr_repr,
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
                elif colsize is None and rowsize is not None:
                    if is_submask:
                        # C[I, j](m) << c
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Col_subassign"
                        value_vector = Vector(value.dtype, size=mask.parent._size, name="v_temp")
                    else:
                        # C(m)[I, j] << c
                        # C[I, j] << c
                        cfunc_name = "GrB_Col_assign"
                        value_vector = Vector(value.dtype, size=rowsize, name="v_temp")
                    # SS, SuiteSparse-specific: assume efficient vector with single scalar
                    value_vector << value

                    # Column-only selection
                    expr = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value_vector, rows, rowscalar, cols],
                        expr_repr="[[{2._expr_name} rows], {3._expr_name}] = {0.name}",
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
                elif colsize is None and rowsize is None:
                    # Matrix object, Vector mask, scalar index
                    # C(m)[i, j] << c
                    # C[i, j](m) << c
                    raise TypeError(
                        "Unable to use Vector mask on single element assignment to a Matrix"
                    )
                else:
                    # Matrix object, Vector mask, Matrix index
                    # C(m)[I, J] << c
                    # C[I, J](m) << c
                    raise TypeError("Unable to use Vector mask on Matrix assignment to a Matrix")
            else:
                if is_submask:
                    if rowsize is None or colsize is None:
                        if rowsize is None and colsize is None:
                            # C[i, j](M) << c
                            raise TypeError("Single element assign does not accept a submask")
                        else:
                            # C[i, J](M) << c
                            # C[I, j](M) << c
                            raise TypeError(
                                "Indices for subassign imply Vector submask, "
                                "but got Matrix mask instead"
                            )
                    # C[I, J](M) << c
                    # SS, SuiteSparse-specific: subassign
                    if value._is_cscalar:
                        if value.dtype._is_udt:
                            dtype_name = "UDT"
                            value = _Pointer(value)
                        else:
                            dtype_name = value.dtype.name
                        cfunc_name = f"GrB_Matrix_subassign_{dtype_name}"
                    else:
                        cfunc_name = "GrB_Matrix_subassign_Scalar"
                    expr_repr = (
                        "[[{2._expr_name} rows], [{4._expr_name} cols]](%s) = {0._expr_name}"
                        % mask.name
                    )
                else:
                    # C(M)[I, J] << c
                    # C(M)[i, J] << c
                    # C(M)[I, j] << c
                    # C(M)[i, j] << c
                    if rowsize is None:
                        rows = _CArray([rows.value])
                        rowscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                    if colsize is None:
                        cols = _CArray([cols.value])
                        colscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                    if value._is_cscalar:
                        if value.dtype._is_udt:
                            dtype_name = "UDT"
                            value = _Pointer(value)
                        else:
                            dtype_name = value.dtype.name
                        cfunc_name = f"GrB_Matrix_assign_{dtype_name}"
                    else:
                        cfunc_name = "GrB_Matrix_assign_Scalar"
                    expr_repr = "[[{2._expr_name} rows], [{4._expr_name} cols]] = {0._expr_name}"
                expr = MatrixExpression(
                    method_name,
                    cfunc_name,
                    [value, rows, rowscalar, cols, colscalar],
                    expr_repr=expr_repr,
                    nrows=self._nrows,
                    ncols=self._ncols,
                    dtype=self.dtype,
                )
        return expr

    def _delete_element(self, resolved_indexes):
        rowidx, colidx = resolved_indexes.indices
        call("GrB_Matrix_removeElement", [self, rowidx.index, colidx.index])

    def to_pygraphblas(self):  # pragma: no cover
        """Convert to a ``pygraphblas.Matrix`` without copying data.

        This gives control of the underlying GraphBLAS memory to pygraphblas,
        meaning further operations on the current :class:`Matrix` object will fail!

        This method requires the
        `pygraphblas <https://graphegon.github.io/pygraphblas/pygraphblas/index.html>`_
        library to be installed.
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"to_pygraphblas only works with 'suitesparse' backend, not {backend}"
            )
        import pygraphblas as pg

        matrix = pg.Matrix(self.gb_obj, pg.types._gb_type_to_type(self.dtype.gb_obj))
        self.gb_obj = ffi_new("GrB_Matrix*")
        return matrix

    @classmethod
    def from_pygraphblas(cls, matrix):  # pragma: no cover
        """Convert a ``pygraphblas.Matrix`` to a new :class:`Matrix` without copying data.

        This gives control of the underlying GraphBLAS memory to python-graphblas,
        meaning further operations on the original ``pygraphblas`` matrix object will fail!

        This method requires the
        `pygraphblas <https://graphegon.github.io/pygraphblas/pygraphblas/index.html>`_
        library to be installed.
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"from_pygraphblas only works with 'suitesparse' backend, not {backend!r}"
            )
        import pygraphblas as pg

        if not isinstance(matrix, pg.Matrix):
            raise TypeError(f"Expected pygraphblas.Matrix object.  Got type: {type(matrix)}")
        dtype = lookup_dtype(matrix.gb_type)
        rv = cls(matrix._matrix, dtype)
        rv._nrows = matrix.nrows
        rv._ncols = matrix.ncols
        matrix._matrix = ffi_new("GrB_Matrix*")
        return rv


Matrix.ss = class_property(Matrix.ss, ss)


class MatrixExpression(BaseExpression):
    __slots__ = "_ncols", "_nrows"
    ndim = 2
    output_type = Matrix
    _is_transposed = False
    __networkx_plugin__ = "graphblas"

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
        ncols=None,
        nrows=None,
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
        if ncols is None:
            ncols = args[0]._ncols
        if nrows is None:
            nrows = args[0]._nrows
        self._ncols = ncols
        self._nrows = nrows

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Matrix(dtype, self._nrows, self._ncols, name=name)

    def __repr__(self):
        from .formatting import format_matrix_expression

        return format_matrix_expression(self)

    def _repr_html_(self):
        from .formatting import format_matrix_expression_html

        return format_matrix_expression_html(self)

    @property
    def ncols(self):
        return self._ncols

    @property
    def nrows(self):
        return self._nrows

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    # Begin auto-generated code: Matrix
    _get_value = automethods._get_value
    S = wrapdoc(Matrix.S)(property(automethods.S))
    T = wrapdoc(Matrix.T)(property(automethods.T))
    V = wrapdoc(Matrix.V)(property(automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(automethods.__ror__))
    _as_vector = wrapdoc(Matrix._as_vector)(property(automethods._as_vector))
    _carg = wrapdoc(Matrix._carg)(property(automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Matrix.get)(property(automethods.get))
    isclose = wrapdoc(Matrix.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(automethods.mxv))
    name = wrapdoc(Matrix.name)(property(automethods.name))
    name = name.setter(automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(automethods.nvals))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(automethods.reduce_scalar))
    reposition = wrapdoc(Matrix.reposition)(property(automethods.reposition))
    select = wrapdoc(Matrix.select)(property(automethods.select))
    ss = wrapdoc(Matrix.ss)(property(automethods.ss))
    to_coo = wrapdoc(Matrix.to_coo)(property(automethods.to_coo))
    to_csc = wrapdoc(Matrix.to_csc)(property(automethods.to_csc))
    to_csr = wrapdoc(Matrix.to_csr)(property(automethods.to_csr))
    to_dcsc = wrapdoc(Matrix.to_dcsc)(property(automethods.to_dcsc))
    to_dcsr = wrapdoc(Matrix.to_dcsr)(property(automethods.to_dcsr))
    to_pygraphblas = wrapdoc(Matrix.to_pygraphblas)(property(automethods.to_pygraphblas))
    to_values = wrapdoc(Matrix.to_values)(property(automethods.to_values))
    wait = wrapdoc(Matrix.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
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
    # End auto-generated code: Matrix


class MatrixIndexExpr(AmbiguousAssignOrExtract):
    __slots__ = "_ncols", "_nrows"
    ndim = 2
    output_type = Matrix
    _is_transposed = False
    __networkx_plugin__ = "graphblas"

    def __init__(self, parent, resolved_indexes, nrows, ncols):
        super().__init__(parent, resolved_indexes)
        self._nrows = nrows
        self._ncols = ncols

    @property
    def ncols(self):
        return self._ncols

    @property
    def nrows(self):
        return self._nrows

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    # Begin auto-generated code: Matrix
    _get_value = automethods._get_value
    S = wrapdoc(Matrix.S)(property(automethods.S))
    T = wrapdoc(Matrix.T)(property(automethods.T))
    V = wrapdoc(Matrix.V)(property(automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(automethods.__ror__))
    _as_vector = wrapdoc(Matrix._as_vector)(property(automethods._as_vector))
    _carg = wrapdoc(Matrix._carg)(property(automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Matrix.get)(property(automethods.get))
    isclose = wrapdoc(Matrix.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(automethods.mxv))
    name = wrapdoc(Matrix.name)(property(automethods.name))
    name = name.setter(automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(automethods.nvals))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(automethods.reduce_scalar))
    reposition = wrapdoc(Matrix.reposition)(property(automethods.reposition))
    select = wrapdoc(Matrix.select)(property(automethods.select))
    ss = wrapdoc(Matrix.ss)(property(automethods.ss))
    to_coo = wrapdoc(Matrix.to_coo)(property(automethods.to_coo))
    to_csc = wrapdoc(Matrix.to_csc)(property(automethods.to_csc))
    to_csr = wrapdoc(Matrix.to_csr)(property(automethods.to_csr))
    to_dcsc = wrapdoc(Matrix.to_dcsc)(property(automethods.to_dcsc))
    to_dcsr = wrapdoc(Matrix.to_dcsr)(property(automethods.to_dcsr))
    to_pygraphblas = wrapdoc(Matrix.to_pygraphblas)(property(automethods.to_pygraphblas))
    to_values = wrapdoc(Matrix.to_values)(property(automethods.to_values))
    wait = wrapdoc(Matrix.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
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
    # End auto-generated code: Matrix


class TransposedMatrix:
    __slots__ = "_matrix", "_ncols", "_nrows", "__weakref__"
    ndim = 2
    _is_scalar = False
    _is_transposed = True
    __networkx_plugin__ = "graphblas"

    def __init__(self, matrix):
        self._matrix = matrix
        self._nrows = matrix._ncols
        self._ncols = matrix._nrows

    def __repr__(self):
        from .formatting import format_matrix

        return format_matrix(self)

    def _repr_html_(self, collapse=False):
        from .formatting import format_matrix_html

        return format_matrix_html(self, collapse=collapse)

    def new(self, dtype=None, *, mask=None, name=None):
        if dtype is None:
            dtype = self.dtype
        output = Matrix(dtype, self._nrows, self._ncols, name=name)
        if mask is None:
            output.update(self)
        else:
            output(mask=mask).update(self)
        return output

    dup = new

    @property
    def T(self):
        return self._matrix

    @property
    def gb_obj(self):
        return self._matrix.gb_obj

    @property
    def dtype(self):
        return self._matrix.dtype

    @wrapdoc(Matrix.to_values)
    def to_values(self, dtype=None, *, rows=True, columns=True, values=True, sort=True):
        rows, cols, vals = self._matrix.to_values(
            dtype, rows=rows, columns=columns, values=values, sort=sort
        )
        return cols, rows, vals

    @wrapdoc(Matrix.diag)
    def diag(self, k=0, dtype=None, *, name=None):
        return self._matrix.diag(-k, dtype, name=name)

    @property
    def _carg(self):
        return self._matrix.gb_obj[0]

    @property
    def name(self):
        return f"{self._matrix.name}.T"

    @property
    def _name_html(self):
        return f"{self._matrix._name_html}.T"

    @wrapdoc(Matrix.to_csr)
    def to_csr(self, dtype=None):
        return self._matrix.to_csc(dtype)

    @wrapdoc(Matrix.to_csc)
    def to_csc(self, dtype=None):
        return self._matrix.to_csr(dtype)

    @wrapdoc(Matrix.to_csr)
    def to_dcsr(self, dtype=None):
        return self._matrix.to_dcsc(dtype)

    @wrapdoc(Matrix.to_csc)
    def to_dcsc(self, dtype=None):
        return self._matrix.to_dcsr(dtype)

    to_coo = to_values  # Alias

    # Properties
    nrows = Matrix.ncols
    ncols = Matrix.nrows
    shape = Matrix.shape
    nvals = Matrix.nvals
    _nvals = Matrix._nvals

    # Delayed methods
    ewise_add = Matrix.ewise_add
    ewise_mult = Matrix.ewise_mult
    ewise_union = Matrix.ewise_union
    mxv = Matrix.mxv
    mxm = Matrix.mxm
    kronecker = Matrix.kronecker
    apply = Matrix.apply
    select = Matrix.select
    reduce_rowwise = Matrix.reduce_rowwise
    reduce_columnwise = Matrix.reduce_columnwise
    reduce_scalar = Matrix.reduce_scalar
    reposition = Matrix.reposition

    # Operator sugar
    __or__ = Matrix.__or__
    __ror__ = Matrix.__ror__
    __and__ = Matrix.__and__
    __rand__ = Matrix.__rand__
    __matmul__ = Matrix.__matmul__
    __rmatmul__ = Matrix.__rmatmul__

    # Bad sugar
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

    # Misc.
    get = Matrix.get
    isequal = Matrix.isequal
    isclose = Matrix.isclose
    wait = Matrix.wait
    _extract_element = Matrix._extract_element
    _prep_for_extract = Matrix._prep_for_extract
    __eq__ = Matrix.__eq__
    __bool__ = Matrix.__bool__
    __getitem__ = Matrix.__getitem__
    __contains__ = Matrix.__contains__
    __iter__ = Matrix.__iter__
    _expect_type = Matrix._expect_type
    _expect_op = Matrix._expect_op
    __array__ = Matrix.__array__


utils._output_types[Matrix] = Matrix
utils._output_types[MatrixIndexExpr] = Matrix
utils._output_types[MatrixExpression] = Matrix
utils._output_types[TransposedMatrix] = TransposedMatrix

# Import infix to import infixmethods, which has side effects
from . import infix  # noqa isort:skip

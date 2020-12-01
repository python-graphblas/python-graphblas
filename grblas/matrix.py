import itertools
import numpy as np
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType, call, _Pointer
from .dtypes import lookup_dtype, unify, UINT64
from .exceptions import check_status, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater, _CArray
from .mask import StructuralMask, ValueMask
from .ops import get_typed_op
from .vector import Vector, VectorExpression
from .scalar import Scalar, ScalarExpression, _CScalar

ffi_new = ffi.new


class Matrix(BaseType):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """

    _is_transposed = False
    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, name=None):
        if name is None:
            name = f"M_{next(Matrix._name_counter)}"
        self._nrows = None
        self._ncols = None
        super().__init__(gb_obj, dtype, name)
        # Add ss extension methods
        self.ss = Matrix.ss(self)

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Matrix_free(gb_obj))

    def __repr__(self, mask=None):
        from .formatting import format_matrix
        from .recorder import skip_record

        with skip_record:
            return format_matrix(self, mask=mask)

    def _repr_html_(self, mask=None):
        from .formatting import format_matrix_html
        from .recorder import skip_record

        with skip_record:
            return format_matrix_html(self, mask=mask)

    @property
    def S(self):
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    def __delitem__(self, keys):
        del Updater(self)[keys]

    def __getitem__(self, keys):
        resolved_indexes = IndexerResolver(self, keys)
        return AmbiguousAssignOrExtract(self, resolved_indexes)

    def __setitem__(self, keys, delayed):
        Updater(self)[keys] = delayed

    def isequal(self, other, *, check_dtype=False):
        """
        Check for exact equality (same size, same empty values)
        If `check_dtype` is True, also checks that dtypes match
        For equality of floating point Vectors, consider using `isclose`
        """
        self._expect_type(other, (Matrix, TransposedMatrix), within="isequal", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._nrows != other._nrows:
            return False
        if self._ncols != other._ncols:
            return False
        if self._nvals != other._nvals:
            return False
        if check_dtype:
            common_dtype = self.dtype
        else:
            common_dtype = unify(self.dtype, other.dtype)

        matches = Matrix.new(bool, self._nrows, self._ncols, name="M_isequal")
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land).value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        self._expect_type(other, (Matrix, TransposedMatrix), within="isclose", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._nrows != other._nrows:
            return False
        if self._ncols != other._ncols:
            return False
        if self._nvals != other._nvals:
            return False

        matches = self.ewise_mult(other, binary.isclose(rel_tol, abs_tol)).new(
            dtype=bool, name="M_isclose"
        )
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land).value

    @property
    def nrows(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nrows", empty=True)  # Actually GrB_Index dtype
        call("GrB_Matrix_nrows", (_Pointer(scalar), self))
        return n[0]

    @property
    def ncols(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_ncols", empty=True)  # Actually GrB_Index dtype
        call("GrB_Matrix_ncols", (_Pointer(scalar), self))
        return n[0]

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nvals", empty=True)  # Actually GrB_Index dtype
        call("GrB_Matrix_nvals", (_Pointer(scalar), self))
        return n[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Matrix_nvals(n, self.gb_obj[0]))
        return n[0]

    @property
    def T(self):
        return TransposedMatrix(self)

    def clear(self):
        call("GrB_Matrix_clear", [self])

    def resize(self, nrows, ncols):
        nrows = _CScalar(nrows)
        ncols = _CScalar(ncols)
        call("GrB_Matrix_resize", (self, nrows, ncols))
        self._nrows = nrows.scalar.value
        self._ncols = ncols.scalar.value

    def to_values(self, *, dtype=None):
        """
        GrB_Matrix_extractTuples
        Extract the rows, columns and values as a 3-tuple of numpy arrays
        """
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = lookup_dtype(dtype)
        nvals = self._nvals
        rows = _CArray(nvals, name="&rows_array")
        columns = _CArray(nvals, name="&columns_array")
        values = _CArray(nvals, ctype=dtype.c_type, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nvals", empty=True)  # Actually GrB_Index dtype
        scalar.value = nvals
        call(
            f"GrB_Matrix_extractTuples_{dtype.name}",
            (rows, columns, values, _Pointer(scalar), self),
        )
        return (
            np.frombuffer(ffi.buffer(rows._carg), dtype=np.uint64),
            np.frombuffer(ffi.buffer(columns._carg), dtype=np.uint64),
            np.frombuffer(ffi.buffer(values._carg), dtype=dtype.np_type),
        )

    def build(self, rows, columns, values, *, dup_op=None, clear=False, nrows=None, ncols=None):
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        np_rows = isinstance(rows, np.ndarray)
        if not np_rows and not isinstance(rows, (tuple, list)):
            rows = tuple(rows)
        np_cols = isinstance(columns, np.ndarray)
        if not np_cols and not isinstance(columns, (tuple, list)):
            columns = tuple(columns)
        np_vals = isinstance(values, np.ndarray)
        if not np_vals and not isinstance(values, (tuple, list)):
            values = tuple(values)
        n = len(values)
        if len(rows) != n or len(columns) != n:
            raise ValueError(
                f"`rows` and `columns` and `values` lengths must match: "
                f"{len(rows)}, {len(columns)}, {len(values)}"
            )
        if clear:
            self.clear()
        if nrows is not None or ncols is not None:
            if nrows is None:
                nrows = self.nrows
            if ncols is None:
                ncols = self.ncols
            self.resize(nrows, ncols)
        if n <= 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        if np_rows:
            urows = rows.astype(np.uint64, copy=False)
            rows = _CArray(urows, from_buffer=True)
        else:
            rows = _CArray(rows)
        if np_cols:
            ucols = columns.astype(np.uint64, copy=False)
            columns = _CArray(ucols, from_buffer=True)
        else:
            columns = _CArray(columns)
        if np_vals:
            verified_vals = values.astype(self.dtype.np_type, copy=False)
            values = _CArray(verified_vals, ctype=self.dtype.c_type, from_buffer=True)
        else:
            values = _CArray(values, ctype=self.dtype.c_type)
        call(
            f"GrB_Matrix_build_{self.dtype.name}",
            (self, rows, columns, values, _CScalar(n), dup_op),
        )
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, *, dtype=None, mask=None, name=None):
        """
        GrB_Matrix_dup
        Create a new Matrix by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            rv = Matrix.new(dtype, nrows=self._nrows, ncols=self._ncols, name=name)
            rv(mask=mask)[:, :] << self
        else:
            new_mat = ffi_new("GrB_Matrix*")
            rv = Matrix(new_mat, self.dtype, name=name)
            call("GrB_Matrix_dup", (_Pointer(rv), self))
        rv._nrows = self._nrows
        rv._ncols = self._ncols
        return rv

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        """
        GrB_Matrix_new
        Create a new empty Matrix from the given type, number of rows, and number of columns
        """
        new_matrix = ffi_new("GrB_Matrix*")
        dtype = lookup_dtype(dtype)
        rv = cls(new_matrix, dtype, name=name)
        if type(nrows) is not _CScalar:
            nrows = _CScalar(nrows)
        if type(ncols) is not _CScalar:
            ncols = _CScalar(ncols)
        call("GrB_Matrix_new", (_Pointer(rv), dtype, nrows, ncols))
        rv._nrows = nrows.scalar.value
        rv._ncols = ncols.scalar.value
        return rv

    @classmethod
    def from_values(
        cls,
        rows,
        columns,
        values,
        *,
        nrows=None,
        ncols=None,
        dup_op=None,
        dtype=None,
        name=None,
    ):
        """Create a new Matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max row and coumn index found.
        """
        rarr = rows
        carr = columns
        varr = values
        if not isinstance(rarr, np.ndarray):
            if not isinstance(rows, (tuple, list)):
                rows = tuple(rows)
            rarr = np.array(rows)
        if not isinstance(carr, np.ndarray):
            if not isinstance(columns, (tuple, list)):
                columns = tuple(columns)
            carr = np.array(columns)
        if not isinstance(varr, np.ndarray):
            if not isinstance(values, (tuple, list)):
                values = tuple(values)
            varr = np.array(values)

        if dtype is None:
            if len(varr) <= 0:
                raise ValueError("No values provided. Unable to determine type.")
            dtype = varr.dtype
            if dtype == object:
                raise ValueError("Unable to convert values to a usable dtype")
        dtype = lookup_dtype(dtype)
        if varr.dtype != dtype.np_type:
            varr = varr.astype(dtype.np_type)
        # Compute nrows and ncols if not provided
        if nrows is None:
            if len(rarr) <= 0:
                raise ValueError("No row indices provided. Unable to infer nrows.")
            nrows = int(rarr.max() + 1)
        if ncols is None:
            if len(carr) <= 0:
                raise ValueError("No column indices provided. Unable to infer ncols.")
            ncols = int(carr.max() + 1)
        if len(rarr) > 0 and "int" not in rarr.dtype.name:
            raise ValueError(f"row indices must be integers, not {rarr.dtype.name}")
        if len(carr) > 0 and "int" not in carr.dtype.name:
            raise ValueError(f"column indices must be integers, not {carr.dtype.name}")
        # Create the new matrix
        C = cls.new(dtype, nrows, ncols, name=name)
        # Add the data
        # This needs to be the original data to get proper error messages
        C.build(rows, columns, values, dup_op=dup_op)
        return C

    @property
    def _carg(self):
        return self.gb_obj[0]

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to __setitem__ to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        """
        GrB_eWiseAdd_Matrix

        Result will contain the union of indices from both Matrices

        Default op is monoid.plus.

        Unless explicitly disabled, this method requires a monoid (directly or from a semiring).
        The reason for this is that binary operators can create very confusing behavior when
        only one of the two elements is present.

        Examples:
            - binary.minus where left=N/A and right=4 yields 4 rather than -4 as might be expected
            - binary.gt where left=N/A and right=4 yields True
            - binary.gt where left=N/A and right=0 yields False

        The behavior is caused by grabbing the non-empty value and using it directly without
        performing any operation. In the case of `gt`, the non-empty value is cast to a boolean.
        For these reasons, users are required to be explicit when choosing this surprising behavior.
        """
        method_name = "ewise_add"
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        if require_monoid:
            self._expect_op(
                op,
                ("Monoid", "Semiring"),
                within=method_name,
                argname="op",
                extra_message="A BinaryOp may be given if require_monoid keyword is False",
            )
        else:
            self._expect_op(
                op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op"
            )
        expr = MatrixExpression(
            method_name,
            f"GrB_eWiseAdd_Matrix_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )
        if self.shape != other.shape:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_mult(self, other, op=binary.times):
        """
        GrB_eWiseMult_Matrix

        Result will contain the intersection of indices from both Matrices
        Default op is binary.times
        """
        method_name = "ewise_mult"
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op")
        expr = MatrixExpression(
            method_name,
            f"GrB_eWiseMult_Matrix_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )
        if self.shape != other.shape:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def mxv(self, other, op=semiring.plus_times):
        """
        GrB_mxv
        Matrix-Vector multiplication. Result is a Vector.
        Default op is semiring.plus_times
        """
        method_name = "mxv"
        self._expect_type(other, Vector, within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
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
        """
        GrB_mxm
        Matrix-Matrix multiplication. Result is a Matrix.
        Default op is semiring.plus_times
        """
        method_name = "mxm"
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
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
        """
        GrB_kronecker
        Kronecker product or sum (depending on op used)
        Default op is binary.times
        """
        method_name = "kronecker"
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op")
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

    def apply(self, op, *, left=None, right=None):
        """
        GrB_Matrix_apply
        Apply UnaryOp to each element of the calling Matrix
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        method_name = "apply"
        extra_message = (
            "apply only accepts UnaryOp with no scalars or BinaryOp with `left` or `right` scalar."
        )
        if left is None and right is None:
            op = get_typed_op(op, self.dtype)
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
            try:
                left = _CScalar(left)
            except TypeError:
                self._expect_type(
                    left,
                    Scalar,
                    within=method_name,
                    keyword_name="left",
                    extra_message="Literal scalars also accepted.",
                )
            op = get_typed_op(op, self.dtype, left.dtype)
            self._expect_op(
                op,
                "BinaryOp",
                within=method_name,
                argname="op",
                extra_message=extra_message,
            )
            cfunc_name = f"GrB_Matrix_apply_BinaryOp1st_{left.dtype}"
            args = [left, self]
            expr_repr = "{1.name}.apply({op}, left={0})"
        elif left is None:
            try:
                right = _CScalar(right)
            except TypeError:
                self._expect_type(
                    right,
                    Scalar,
                    within=method_name,
                    keyword_name="right",
                    extra_message="Literal scalars also accepted.",
                )
            op = get_typed_op(op, self.dtype, right.dtype)
            self._expect_op(
                op,
                "BinaryOp",
                within=method_name,
                argname="op",
                extra_message=extra_message,
            )
            cfunc_name = f"GrB_Matrix_apply_BinaryOp2nd_{right.dtype}"
            args = [self, right]
            expr_repr = "{0.name}.apply({op}, right={1})"
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
        )

    def reduce_rows(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each row, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_rows"
        op = get_typed_op(op, self.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        return VectorExpression(
            method_name,
            f"GrB_Matrix_reduce_{op.opclass}",
            [self],
            op=op,
            size=self._nrows,
            at=self._is_transposed,
        )

    def reduce_columns(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_columns"
        op = get_typed_op(op, self.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        return VectorExpression(
            method_name,
            f"GrB_Matrix_reduce_{op.opclass}",
            [self],
            op=op,
            size=self._ncols,
            at=not self._is_transposed,
        )

    def reduce_scalar(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_scalar"
        op = get_typed_op(op, self.dtype)
        self._expect_op(op, "Monoid", within=method_name, argname="op")
        return ScalarExpression(
            method_name,
            "GrB_Matrix_reduce_{output_dtype}",
            [self],
            op=op,  # to be determined later
        )

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(self, resolved_indexes, dtype=None, name="s_extract"):
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = lookup_dtype(dtype)
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        if self._is_transposed:
            row, col = col, row
        result = Scalar.new(dtype, name=name)
        try:
            call(f"GrB_Matrix_extractElement_{dtype}", (_Pointer(result), self, row, col))
        except NoValue:
            pass
        else:
            result._is_empty = False
        return result

    def _prep_for_extract(self, resolved_indexes):
        method_name = "__getitem__"
        rows, rowsize = resolved_indexes.indices[0]
        cols, colsize = resolved_indexes.indices[1]
        if rowsize is None:
            # Row-only selection; GraphBLAS doesn't have this method, so we hack it using transpose
            row_index = rows
            return VectorExpression(
                method_name,
                "GrB_Col_extract",
                [self, cols, colsize, row_index],
                expr_repr="{0.name}[{3}, [{2} cols]]",
                size=colsize,
                dtype=self.dtype,
                at=not self._is_transposed,
            )
        elif colsize is None:
            # Column-only selection
            col_index = cols
            return VectorExpression(
                method_name,
                "GrB_Col_extract",
                [self, rows, rowsize, col_index],
                expr_repr="{0.name}[[{2} rows], {3}]",
                size=rowsize,
                dtype=self.dtype,
                at=self._is_transposed,
            )
        else:
            return MatrixExpression(
                method_name,
                "GrB_Matrix_extract",
                [self, rows, rowsize, cols, colsize],
                expr_repr="{0.name}[[{2} rows], [{4} cols]]",
                nrows=rowsize,
                ncols=colsize,
                dtype=self.dtype,
                at=self._is_transposed,
            )

    def _assign_element(self, resolved_indexes, value):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        try:
            value = _CScalar(value)
        except TypeError:
            self._expect_type(
                value,
                Scalar,
                within="__setitem__",
                argname="value",
                extra_message="Literal scalars also accepted.",
            )
        # should we cast?
        call(f"GrB_Matrix_setElement_{value.dtype}", (self, value, row, col))

    def _prep_for_assign(self, resolved_indexes, value):
        method_name = "__setitem__"
        rows, rowsize = resolved_indexes.indices[0]
        cols, colsize = resolved_indexes.indices[1]
        extra_message = "Literal scalars also accepted."
        if type(value) is Vector:
            if rowsize is None and colsize is not None:
                # Row-only selection
                row_index = rows
                delayed = MatrixExpression(
                    method_name,
                    "GrB_Row_assign",
                    [value, row_index, cols, colsize],
                    expr_repr="[{1}, [{3} cols]] = {0.name}",
                    nrows=self._nrows,
                    ncols=self._ncols,
                    dtype=self.dtype,
                )
            elif colsize is None and rowsize is not None:
                # Column-only selection
                col_index = cols
                delayed = MatrixExpression(
                    method_name,
                    "GrB_Col_assign",
                    [value, rows, rowsize, col_index],
                    expr_repr="[[{2} rows], {3}] = {0.name}",
                    nrows=self._nrows,
                    ncols=self._ncols,
                    dtype=self.dtype,
                )
            else:
                self._expect_type(
                    value,
                    (Scalar, Matrix, TransposedMatrix),
                    within=method_name,
                    extra_message=extra_message,
                )
        elif type(value) in {Matrix, TransposedMatrix}:
            if rowsize is None or colsize is None:
                self._expect_type(
                    value,
                    (Scalar, Vector),
                    within=method_name,
                    extra_message=extra_message,
                )
            delayed = MatrixExpression(
                method_name,
                "GrB_Matrix_assign",
                [value, rows, rowsize, cols, colsize],
                expr_repr="[[{2} rows], [{4} cols]] = {0.name}",
                nrows=self._nrows,
                ncols=self._ncols,
                dtype=self.dtype,
                at=value._is_transposed,
            )
        else:
            try:
                value = _CScalar(value)
            except TypeError:
                if rowsize is None or colsize is None:
                    types = (Scalar, Vector)
                else:
                    types = (Scalar, Matrix, TransposedMatrix)
                self._expect_type(
                    value,
                    types,
                    within=method_name,
                    argname="value",
                    extra_message=extra_message,
                )
            if rowsize is None:
                rows = _CArray([rows.scalar.value])
                rowsize = _CScalar(1)
            if colsize is None:
                cols = _CArray([cols.scalar.value])
                colsize = _CScalar(1)
            delayed = MatrixExpression(
                method_name,
                f"GrB_Matrix_assign_{value.dtype}",
                [value, rows, rowsize, cols, colsize],
                expr_repr="[[{2} rows], [{4} cols]] = {0}",
                nrows=self._nrows,
                ncols=self._ncols,
                dtype=self.dtype,
            )
        return delayed

    def _delete_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        call("GrB_Matrix_removeElement", (self, row, col))

    if backend == "pygraphblas":

        def to_pygraphblas(self):
            """Convert to a new `pygraphblas.Matrix`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `pygraphblas`.
            This means operations on the current `grblas` object will fail!
            """
            import pygraphblas as pg

            matrix = pg.Matrix(self.gb_obj, pg.types.gb_type_to_type(self.dtype.gb_obj))
            self.gb_obj = ffi.NULL
            return matrix

        @classmethod
        def from_pygraphblas(cls, matrix):
            """Convert a `pygraphblas.Matrix` to a new `grblas.Matrix`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `grblas`.
            This means operations on the original `pygraphblas` object will fail!
            """
            dtype = lookup_dtype(matrix.gb_type)
            rv = cls(matrix.matrix, dtype)
            rv._nrows = matrix.nrows
            rv._ncols = matrix.ncols
            matrix.matrix = ffi.NULL
            return rv

    class ss:
        def __init__(self, parent):
            self._parent = parent

        def fast_export(self, format=None):
            """
            GxB_Matrix_export_xxx

            Returns a dict of the constituent parts:
             - format: str
             - nrows: number of rows (int)
             - ncols: number of columns (int)
             - h: header mapping (ndarray<uinte64>) (only for HyperCSR or HyperCSC)
             - p: pointers (ndarray<uint64>)
             - i or j: indices (ndarray<uint64>) (i for CSC, j for CSR)
             - x: values (ndarray of appropriate dtype)

            To reimport the Matrix:
            ```
            pieces = A.fast_export()
            A2 = Matrix.fast_import(**pieces)
            ```

            The underlying GraphBLAS object transfers ownership to numpy,
            disallowing further access. The caller should delete or stop using
            the Matrix after calling `fast_export`.

            If `format` is not specified, this method exports in the currently stored format
            To control the export format, set `format` to one of:
              - "csr"
              - "csc"
              - "hypercsr"
              - "hypercsc"
            """
            dtype = np.dtype(self._parent.dtype.np_type)
            index_dtype = np.dtype(np.uint64)

            if format is None:
                # Determine current format
                hyper_ptr = ffi_new("bool*")
                format_ptr = ffi_new("int*")
                call("GxB_Matrix_Option_get", (self, lib.GxB_IS_HYPER, hyper_ptr))
                call("GxB_Matrix_Option_get", (self, lib.GxB_FORMAT, format_ptr))
                if hyper_ptr[0]:
                    format = "hypercsc" if format_ptr[0] == lib.GxB_BY_COL else "hypercsr"
                else:
                    format = "csc" if format_ptr[0] == lib.GxB_BY_COL else "csr"
            format = format.lower()

            mhandle = ffi_new("GrB_Matrix*", self._parent._carg)
            type_ = ffi_new("GrB_Type*")
            nrows = ffi_new("GrB_Index*")
            ncols = ffi_new("GrB_Index*")
            nvals = ffi_new("GrB_Index*")
            nonempty = ffi_new("int64_t*")
            Ap = ffi_new("GrB_Index**")
            Ax = ffi_new("void**")
            if format == "csr":
                Aj = ffi_new("GrB_Index**")
                check_status(
                    lib.GxB_Matrix_export_CSR(
                        mhandle, type_, nrows, ncols, nvals, nonempty, Ap, Aj, Ax, ffi.NULL
                    )
                )
                rv = {
                    "indptr": np.frombuffer(
                        ffi.buffer(Ap[0], (nrows[0] + 1) * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "col_indices": np.frombuffer(
                        ffi.buffer(Aj[0], nvals[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "values": np.frombuffer(
                        ffi.buffer(Ax[0], nvals[0] * dtype.itemsize), dtype=dtype
                    ),
                }
            elif format == "csc":
                Ai = ffi_new("GrB_Index**")
                check_status(
                    lib.GxB_Matrix_export_CSC(
                        mhandle, type_, nrows, ncols, nvals, nonempty, Ap, Ai, Ax, ffi.NULL
                    )
                )
                rv = {
                    "indptr": np.frombuffer(
                        ffi.buffer(Ap[0], (ncols[0] + 1) * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "row_indices": np.frombuffer(
                        ffi.buffer(Ai[0], nvals[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "values": np.frombuffer(
                        ffi.buffer(Ax[0], nvals[0] * dtype.itemsize), dtype=dtype
                    ),
                }
            elif format == "hypercsr":
                nvec = ffi_new("GrB_Index*")
                Ah = ffi_new("GrB_Index**")
                Aj = ffi_new("GrB_Index**")
                check_status(
                    lib.GxB_Matrix_export_HyperCSR(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        nvals,
                        nonempty,
                        nvec,
                        Ah,
                        Ap,
                        Aj,
                        Ax,
                        ffi.NULL,
                    )
                )
                rv = {
                    "rows": np.frombuffer(
                        ffi.buffer(Ah[0], nvec[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "indptr": np.frombuffer(
                        ffi.buffer(Ap[0], (nvec[0] + 1) * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "col_indices": np.frombuffer(
                        ffi.buffer(Aj[0], nvals[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "values": np.frombuffer(
                        ffi.buffer(Ax[0], nvals[0] * dtype.itemsize), dtype=dtype
                    ),
                }
            elif format == "hypercsc":
                nvec = ffi_new("GrB_Index*")
                Ah = ffi_new("GrB_Index**")
                Ai = ffi_new("GrB_Index**")
                check_status(
                    lib.GxB_Matrix_export_HyperCSC(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        nvals,
                        nonempty,
                        nvec,
                        Ah,
                        Ap,
                        Ai,
                        Ax,
                        ffi.NULL,
                    )
                )
                rv = {
                    "cols": np.frombuffer(
                        ffi.buffer(Ah[0], nvec[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "indptr": np.frombuffer(
                        ffi.buffer(Ap[0], (nvec[0] + 1) * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "row_indices": np.frombuffer(
                        ffi.buffer(Ai[0], nvals[0] * index_dtype.itemsize), dtype=index_dtype
                    ),
                    "values": np.frombuffer(
                        ffi.buffer(Ax[0], nvals[0] * dtype.itemsize), dtype=dtype
                    ),
                }
            else:
                raise ValueError(f"Invalid format: {format}")

            rv.update(
                {
                    "format": format,
                    "nrows": nrows[0],
                    "ncols": ncols[0],
                }
            )
            self._parent.gb_obj = ffi.NULL
            return rv

        @classmethod
        def fast_import(
            cls,
            *,
            nrows,
            ncols,
            indptr,
            values,
            row_indices=None,
            col_indices=None,
            rows=None,
            cols=None,
            format=None,
            name=None,
        ):
            """
            GxB_Matrix_import_xxx

            The new Matrix uses the underlying buffer of the input arrays.
            The caller should delete or stop using the input arrays after calling `fast_import`.

            Valid formats:
             - csr  (needs indptr, col_indices, values)
             - csc  (needs indptr, row_indices, values)
             - hypercsr  (needs rows, indptr, col_indices, values)
             - hypercsc  (needs cols, indptr, row_indices, values)
            """
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)

            if format is None:
                # Determine format based on provided inputs
                if row_indices is None and col_indices is None:
                    raise ValueError("Must provide either `row_indices` or `col_indices`")
                if row_indices is not None and col_indices is not None:
                    raise ValueError("Cannot provide both `row_indices` and `col_indices`")
                if rows is not None and cols is not None:
                    raise ValueError("Cannot provide both `rows` and `cols`")
                elif rows is None and cols is None:
                    format = "csr" if row_indices is None else "csc"
                elif rows is not None:
                    if col_indices is None:
                        raise ValueError("HyperCSR requires col_indices, not row_indices")
                    format = "hypercsr"
                else:
                    if row_indices is None:
                        raise ValueError("HyperCSC requires row_indices, not col_indices")
                    format = "hypercsc"
            format = format.lower()

            Ap = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indptr)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if format == "csr":
                Aj = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(col_indices)))
                check_status(
                    lib.GxB_Matrix_import_CSR(
                        mhandle, dtype._carg, nrows, ncols, len(values), -1, Ap, Aj, Ax, ffi.NULL
                    )
                )
            elif format == "csc":
                Ai = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(row_indices)))
                check_status(
                    lib.GxB_Matrix_import_CSC(
                        mhandle, dtype._carg, nrows, ncols, len(values), -1, Ap, Ai, Ax, ffi.NULL
                    )
                )
            elif format == "hypercsr":
                Ah = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(rows)))
                Aj = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(col_indices)))
                check_status(
                    lib.GxB_Matrix_import_HyperCSR(
                        mhandle,
                        dtype._carg,
                        nrows,
                        ncols,
                        len(values),
                        -1,
                        len(rows),
                        Ah,
                        Ap,
                        Aj,
                        Ax,
                        ffi.NULL,
                    )
                )
            elif format == "hypercsc":
                Ah = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(cols)))
                Ai = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(row_indices)))
                check_status(
                    lib.GxB_Matrix_import_HyperCSC(
                        mhandle,
                        dtype._carg,
                        nrows,
                        ncols,
                        len(values),
                        -1,
                        len(cols),
                        Ah,
                        Ap,
                        Ai,
                        Ax,
                        ffi.NULL,
                    )
                )
            else:
                raise ValueError(f"Invalid format: {format}")

            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv


class MatrixExpression(BaseExpression):
    output_type = Matrix

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
        self.ncols = self._ncols = ncols
        self.nrows = self._nrows = nrows

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Matrix.new(dtype, self._nrows, self._ncols, name=name)

    def __repr__(self):
        from .formatting import format_matrix_expression

        return format_matrix_expression(self)

    def _repr_html_(self):
        from .formatting import format_matrix_expression_html

        return format_matrix_expression_html(self)


class TransposedMatrix:
    _is_scalar = False
    _is_transposed = True

    def __init__(self, matrix):
        self._matrix = matrix
        self._nrows = matrix._ncols
        self._ncols = matrix._nrows

    def __repr__(self):
        from .formatting import format_matrix

        return format_matrix(self)

    def _repr_html_(self):
        from .formatting import format_matrix_html

        return format_matrix_html(self)

    def new(self, *, dtype=None, mask=None, name=None):
        if dtype is None:
            dtype = self.dtype
        output = Matrix.new(dtype, self._nrows, self._ncols, name=name)
        if mask is None:
            output.update(self)
        else:
            output(mask=mask).update(self)
        return output

    @property
    def T(self):
        return self._matrix

    @property
    def gb_obj(self):
        return self._matrix.gb_obj

    @property
    def dtype(self):
        return self._matrix.dtype

    def to_values(self):
        rows, cols, vals = self._matrix.to_values()
        return cols, rows, vals

    @property
    def _carg(self):
        return self._matrix.gb_obj[0]

    @property
    def name(self):
        return f"{self._matrix.name}.T"

    @property
    def _name_html(self):
        return f"{self._matrix._name_html}.T"

    # Properties
    nrows = Matrix.ncols
    ncols = Matrix.nrows
    shape = Matrix.shape
    nvals = Matrix.nvals
    _nvals = Matrix._nvals

    # Delayed methods
    ewise_add = Matrix.ewise_add
    ewise_mult = Matrix.ewise_mult
    mxv = Matrix.mxv
    mxm = Matrix.mxm
    kronecker = Matrix.kronecker
    apply = Matrix.apply
    reduce_rows = Matrix.reduce_rows
    reduce_columns = Matrix.reduce_columns
    reduce_scalar = Matrix.reduce_scalar

    # Misc.
    isequal = Matrix.isequal
    isclose = Matrix.isclose
    _extract_element = Matrix._extract_element
    _prep_for_extract = Matrix._prep_for_extract
    __eq__ = Matrix.__eq__
    __getitem__ = Matrix.__getitem__
    _expect_type = Matrix._expect_type
    _expect_op = Matrix._expect_op

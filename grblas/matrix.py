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
            check_status(lib.GrB_Matrix_free(gb_obj), self)

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
        check_status(lib.GrB_Matrix_nvals(n, self.gb_obj[0]), self)
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
        nvals = self._nvals
        rows = _CArray(nvals, name="&rows_array")
        columns = _CArray(nvals, name="&columns_array")
        values = _CArray(nvals, ctype=self.dtype.c_type, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nvals", empty=True)  # Actually GrB_Index dtype
        scalar.value = nvals
        call(
            f"GrB_Matrix_extractTuples_{self.dtype.name}",
            (rows, columns, values, _Pointer(scalar), self),
        )
        values = np.frombuffer(ffi.buffer(values._carg), dtype=self.dtype.np_type)
        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                values = values.astype(dtype.np_type)  # copies
        return (
            np.frombuffer(ffi.buffer(rows._carg), dtype=np.uint64),
            np.frombuffer(ffi.buffer(columns._carg), dtype=np.uint64),
            values,
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
        GrB_Matrix_eWiseAdd

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
        """
        GrB_Matrix_eWiseMult

        Result will contain the intersection of indices from both Matrices
        Default op is binary.times
        """
        method_name = "ewise_mult"
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op")
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
        if (
            call(f"GrB_Matrix_extractElement_{dtype}", (_Pointer(result), self, row, col))
            is not NoValue
        ):
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

    def _prep_for_assign(self, resolved_indexes, value, mask=None, is_submask=False):
        method_name = "__setitem__"
        rows, rowsize = resolved_indexes.indices[0]
        cols, colsize = resolved_indexes.indices[1]
        extra_message = "Literal scalars also accepted."
        if type(value) is Vector:
            if rowsize is None and colsize is not None:
                # Row-only selection
                row_index = rows
                if mask is not None and type(mask.mask) is Matrix:
                    if is_submask:
                        # C[i, J](M) << v
                        raise TypeError(
                            "Indices for subassign imply Vector submask, "
                            "but got Matrix mask instead"
                        )
                    else:
                        # C(M)[i, J] << v
                        # Upcast v to a Matrix and use Matrix_assign
                        rows = _CArray([rows.scalar.value])
                        rowsize = _CScalar(1)
                        new_value = Matrix.new(
                            value.dtype, nrows=1, ncols=value.size, name=f"{value.name}_as_matrix"
                        )
                        new_value[0, :] = value
                        delayed = MatrixExpression(
                            method_name,
                            "GrB_Matrix_assign",
                            [new_value, rows, rowsize, cols, colsize],
                            expr_repr="[[{2} rows], [{4} cols]] = {0.name}",
                            nrows=self._nrows,
                            ncols=self._ncols,
                            dtype=self.dtype,
                        )
                else:
                    if is_submask:
                        # C[i, J](m) << v
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Row_subassign"
                        expr_repr = "[{1}, [{3} cols]](%s) << {0.name}" % mask.name
                    else:
                        # C(m)[i, J] << v
                        # C[i, J] << v
                        cfunc_name = "GrB_Row_assign"
                        expr_repr = "[{1}, [{3} cols]] = {0.name}"
                    delayed = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value, row_index, cols, colsize],
                        expr_repr=expr_repr,
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
            elif colsize is None and rowsize is not None:
                # Column-only selection
                col_index = cols
                if mask is not None and type(mask.mask) is Matrix:
                    if is_submask:
                        # C[I, j](M) << v
                        raise TypeError(
                            "Indices for subassign imply Vector submask, "
                            "but got Matrix mask instead"
                        )
                    else:
                        # C(M)[I, j] << v
                        # Upcast v to a Matrix and use Matrix_assign
                        cols = _CArray([cols.scalar.value])
                        colsize = _CScalar(1)
                        new_value = Matrix.new(
                            value.dtype, nrows=value.size, ncols=1, name=f"{value.name}_as_matrix"
                        )
                        new_value[:, 0] = value
                        delayed = MatrixExpression(
                            method_name,
                            "GrB_Matrix_assign",
                            [new_value, rows, rowsize, cols, colsize],
                            expr_repr="[[{2} rows], [{4} cols]] = {0.name}",
                            nrows=self._nrows,
                            ncols=self._ncols,
                            dtype=self.dtype,
                        )
                else:
                    if is_submask:
                        # C[I, j](m) << v
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Col_subassign"
                        expr_repr = "[{1}, [{3} cols]](%s) << {0.name}" % mask.name
                    else:
                        # C(m)[I, j] << v
                        # C[I, j] << v
                        cfunc_name = "GrB_Col_assign"
                        expr_repr = "[{1}, [{3} cols]] = {0.name}"
                    delayed = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value, rows, rowsize, col_index],
                        expr_repr=expr_repr,
                        nrows=self._nrows,
                        ncols=self._ncols,
                        dtype=self.dtype,
                    )
            elif colsize is None and rowsize is None:
                # C[i, j] << v  (mask doesn't matter)
                self._expect_type(
                    value,
                    Scalar,
                    within=method_name,
                    extra_message=extra_message,
                )
            else:
                # C[I, J] << v  (mask doesn't matter)
                self._expect_type(
                    value,
                    (Scalar, Matrix, TransposedMatrix),
                    within=method_name,
                    extra_message=extra_message,
                )
        elif type(value) in {Matrix, TransposedMatrix}:
            if rowsize is None or colsize is None:
                if rowsize is None and colsize is None:
                    # C[i, j] << A  (mask doesn't matter)
                    self._expect_type(
                        value,
                        Scalar,
                        within=method_name,
                        extra_message=extra_message,
                    )
                else:
                    # C[I, j] << A
                    # C[i, J] << A  (mask doesn't matter)
                    self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        extra_message=extra_message,
                    )
            if is_submask:
                # C[I, J](M) << A
                # SS, SuiteSparse-specific: subassign
                cfunc_name = "GrB_Matrix_subassign"
                expr_repr = "[[{2} rows], [{4} cols]](%s) << {0.name}" % mask.name
            else:
                # C[I, J] << A
                # C(M)[I, J] << A
                cfunc_name = "GrB_Matrix_assign"
                expr_repr = "[[{2} rows], [{4} cols]] = {0.name}"
            delayed = MatrixExpression(
                method_name,
                cfunc_name,
                [value, rows, rowsize, cols, colsize],
                expr_repr=expr_repr,
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
            if mask is not None and type(mask.mask) is Vector:
                value = value.scalar
                if rowsize is None and colsize is not None:
                    if is_submask:
                        # C[i, J](m) << c
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Row_subassign"
                        value_vector = Vector.new(value.dtype, size=mask.mask.size, name="v_temp")
                        expr_repr = "[{1}, [{3} cols]](%s) << {0.name}" % mask.name
                    else:
                        # C(m)[i, J] << c
                        # C[i, J] << c
                        cfunc_name = "GrB_Row_assign"
                        value_vector = Vector.new(value.dtype, size=colsize, name="v_temp")
                        expr_repr = "[{1}, [{3} cols]] = {0.name}"
                    # SS, SuiteSparse-specific: assume efficient vector with single scalar
                    value_vector << value

                    # Row-only selection
                    row_index = rows
                    delayed = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value_vector, row_index, cols, colsize],
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
                        value_vector = Vector.new(value.dtype, size=mask.mask.size, name="v_temp")
                    else:
                        # C(m)[I, j] << c
                        # C[I, j] << c
                        cfunc_name = "GrB_Col_assign"
                        value_vector = Vector.new(value.dtype, size=rowsize, name="v_temp")
                    # SS, SuiteSparse-specific: assume efficient vector with single scalar
                    value_vector << value

                    # Column-only selection
                    col_index = cols
                    delayed = MatrixExpression(
                        method_name,
                        cfunc_name,
                        [value_vector, rows, rowsize, col_index],
                        expr_repr="[[{2} rows], {3}] = {0.name}",
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
                    cfunc_name = f"GrB_Matrix_subassign_{value.dtype}"
                    expr_repr = "[[{2} rows], [{4} cols]](%s) = {0}" % mask.name
                else:
                    # C(M)[I, J] << c
                    # C(M)[i, J] << c
                    # C(M)[I, j] << c
                    # C(M)[i, j] << c
                    if rowsize is None:
                        rows = _CArray([rows.scalar.value])
                        rowsize = _CScalar(1)
                    if colsize is None:
                        cols = _CArray([cols.scalar.value])
                        colsize = _CScalar(1)
                    cfunc_name = f"GrB_Matrix_assign_{value.dtype}"
                    expr_repr = "[[{2} rows], [{4} cols]] = {0}"
                delayed = MatrixExpression(
                    method_name,
                    cfunc_name,
                    [value, rows, rowsize, cols, colsize],
                    expr_repr=expr_repr,
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

        def fast_export(self, format=None, *, sort=False):
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
              - "bitmapr"
              - "bitmapc"
              - "fullr"
              - "fullc"
            """
            dtype = np.dtype(self._parent.dtype.np_type)
            index_dtype = np.dtype(np.uint64)

            if format is None:
                # Determine current format
                hyper_ptr = ffi_new("bool*")
                format_ptr = ffi_new("int*")
                call("GxB_Matrix_Option_get", (self._parent, lib.GxB_IS_HYPER, hyper_ptr))
                call("GxB_Matrix_Option_get", (self._parent, lib.GxB_FORMAT, format_ptr))
                if hyper_ptr[0]:
                    format = "hypercsc" if format_ptr[0] == lib.GxB_BY_COL else "hypercsr"
                else:
                    format = "csc" if format_ptr[0] == lib.GxB_BY_COL else "csr"
            format = format.lower()

            mhandle = ffi_new("GrB_Matrix*", self._parent._carg)
            type_ = ffi_new("GrB_Type*")
            nrows = ffi_new("GrB_Index*")
            ncols = ffi_new("GrB_Index*")
            Ap = ffi_new("GrB_Index**")
            Ax = ffi_new("void**")
            Ap_size = ffi_new("GrB_Index*")
            Ax_size = ffi_new("GrB_Index*")
            if sort:
                jumbled = ffi.NULL
            else:
                jumbled = ffi_new("bool*")
            nvals = self._parent._nvals
            if format == "csr":
                Aj = ffi_new("GrB_Index**")
                Aj_size = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Matrix_export_CSR(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ap,
                        Aj,
                        Ax,
                        Ap_size,
                        Aj_size,
                        Ax_size,
                        jumbled,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                indptr = np.frombuffer(
                    ffi.buffer(Ap[0], Ap_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                col_indices = np.frombuffer(
                    ffi.buffer(Aj[0], Aj_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                if len(indptr) > nrows[0] + 1:
                    indptr = indptr[: nrows[0] + 1]
                if len(col_indices) > nvals:
                    col_indices = col_indices[:nvals]
                if len(values) > nvals:
                    values = values[:nvals]
                rv = {
                    "indptr": indptr,
                    "col_indices": col_indices,
                    "sorted_index": True if sort else not jumbled[0],
                }
            elif format == "csc":
                Ai = ffi_new("GrB_Index**")
                Ai_size = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Matrix_export_CSC(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ap,
                        Ai,
                        Ax,
                        Ap_size,
                        Ai_size,
                        Ax_size,
                        jumbled,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                indptr = np.frombuffer(
                    ffi.buffer(Ap[0], Ap_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                row_indices = np.frombuffer(
                    ffi.buffer(Ai[0], Ai_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                if len(indptr) > ncols[0] + 1:
                    indptr = indptr[: ncols[0] + 1]
                if len(row_indices) > nvals:
                    row_indices = row_indices[:nvals]
                if len(values) > nvals:
                    values = values[:nvals]
                rv = {
                    "indptr": indptr,
                    "row_indices": row_indices,
                    "sorted_index": True if sort else not jumbled[0],
                }
            elif format == "hypercsr":
                nvec = ffi_new("GrB_Index*")
                Ah = ffi_new("GrB_Index**")
                Aj = ffi_new("GrB_Index**")
                Ah_size = ffi_new("GrB_Index*")
                Aj_size = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Matrix_export_HyperCSR(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ap,
                        Ah,
                        Aj,
                        Ax,
                        Ap_size,
                        Ah_size,
                        Aj_size,
                        Ax_size,
                        nvec,
                        jumbled,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                indptr = np.frombuffer(
                    ffi.buffer(Ap[0], Ap_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                rows = np.frombuffer(
                    ffi.buffer(Ah[0], Ah_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                col_indices = np.frombuffer(
                    ffi.buffer(Aj[0], Aj_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                nvec = nvec[0]
                if len(indptr) > nvec + 1:
                    indptr = indptr[: nvec + 1]
                if len(rows) > nvec:
                    rows = rows[:nvec]
                if len(col_indices) > nvals:
                    col_indices = col_indices[:nvals]
                if len(values) > nvals:
                    values = values[:nvals]
                rv = {
                    "rows": rows,
                    "indptr": indptr,
                    "col_indices": col_indices,
                    "sorted_index": True if sort else not jumbled[0],
                }
            elif format == "hypercsc":
                nvec = ffi_new("GrB_Index*")
                Ah = ffi_new("GrB_Index**")
                Ai = ffi_new("GrB_Index**")
                Ah_size = ffi_new("GrB_Index*")
                Ai_size = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Matrix_export_HyperCSC(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ap,
                        Ah,
                        Ai,
                        Ax,
                        Ap_size,
                        Ah_size,
                        Ai_size,
                        Ax_size,
                        nvec,
                        jumbled,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                indptr = np.frombuffer(
                    ffi.buffer(Ap[0], Ap_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                cols = np.frombuffer(
                    ffi.buffer(Ah[0], Ah_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                row_indices = np.frombuffer(
                    ffi.buffer(Ai[0], Ai_size[0] * index_dtype.itemsize), dtype=index_dtype
                )
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                nvec = nvec[0]
                if len(indptr) > nvec + 1:
                    indptr = indptr[: nvec + 1]
                if len(cols) > nvec:
                    cols = cols[:nvec]
                if len(row_indices) > nvals:
                    row_indices = row_indices[:nvals]
                if len(values) > nvals:
                    values = values[:nvals]
                rv = {
                    "cols": cols,
                    "indptr": indptr,
                    "row_indices": row_indices,
                    "sorted_index": True if sort else not jumbled[0],
                }
            elif format == "bitmapr" or format == "bitmapc":
                if format == "bitmapr":
                    cfunc = lib.GxB_Matrix_export_BitmapR
                else:
                    cfunc = lib.GxB_Matrix_export_BitmapC
                Ab = ffi_new("int8_t**")
                Ab_size = ffi_new("GrB_Index*")
                nvals_ = ffi_new("GrB_Index*")
                check_status(
                    cfunc(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ab,
                        Ax,
                        Ab_size,
                        Ax_size,
                        nvals_,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                bitmap = np.frombuffer(ffi.buffer(Ab[0], Ab_size[0]), dtype=np.bool8)
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                size = nrows[0] * ncols[0]
                if len(bitmap) > size:
                    bitmap = bitmap[:size]
                if len(values) > size:
                    values = values[:size]
                rv = {"bitmap": bitmap, "nvals": nvals_[0]}
            elif format == "fullr" or format == "fullc":
                if format == "fullr":
                    cfunc = lib.GxB_Matrix_export_FullR
                else:
                    cfunc = lib.GxB_Matrix_export_FullC
                check_status(
                    cfunc(
                        mhandle,
                        type_,
                        nrows,
                        ncols,
                        Ax,
                        Ax_size,
                        ffi.NULL,
                    ),
                    self._parent,
                )
                values = np.frombuffer(ffi.buffer(Ax[0], Ax_size[0] * dtype.itemsize), dtype=dtype)
                size = nrows[0] * ncols[0]
                if len(values) > size:
                    values = values[:size]
                rv = {}
            else:
                raise ValueError(f"Invalid format: {format}")

            rv.update(
                format=format,
                nrows=nrows[0],
                ncols=ncols[0],
                values=values,
            )
            self._parent.gb_obj = ffi.NULL
            return rv

        @classmethod
        def fast_import_csr(
            cls,
            *,
            nrows,
            ncols,
            indptr,
            values,
            col_indices,
            sorted_index=False,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "csr":
                raise ValueError(f"Invalid format: {format!r}.  Must be None or 'csr'.")
            # if not indptr.flags.owndata:  # TODO: fix leak
            #     indptr = indptr.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            # if not col_indices.flags.owndata:  # TODO: fix leak
            #     col_indices = col_indices.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ap = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indptr)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            Aj = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(col_indices)))
            check_status(
                lib.GxB_Matrix_import_CSR(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ap,
                    Aj,
                    Ax,
                    len(indptr),
                    len(col_indices),
                    len(values),
                    not sorted_index,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_csc(
            cls,
            *,
            nrows,
            ncols,
            indptr,
            values,
            row_indices,
            sorted_index=False,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "csc":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'csc'.")
            # if not indptr.flags.owndata:  # TODO: fix leak
            #     indptr = indptr.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            # if not row_indices.flags.owndata:  # TODO: fix leak
            #     row_indices = row_indices.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ap = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indptr)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            Ai = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(row_indices)))
            check_status(
                lib.GxB_Matrix_import_CSC(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ap,
                    Ai,
                    Ax,
                    len(indptr),
                    len(row_indices),
                    len(values),
                    not sorted_index,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_hypercsr(
            cls,
            *,
            nrows,
            ncols,
            rows,
            indptr,
            values,
            col_indices,
            sorted_index=False,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "hypercsr":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'hypercsr'.")
            # if not rows.flags.owndata:  # TODO: fix leak
            #     rows = rows.copy()
            # if not indptr.flags.owndata:  # TODO: fix leak
            #     indptr = indptr.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            # if not col_indices.flags.owndata:  # TODO: fix leak
            #     col_indices = col_indices.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ap = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indptr)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            Ah = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(rows)))
            Aj = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(col_indices)))
            nvec = len(rows)
            check_status(
                lib.GxB_Matrix_import_HyperCSR(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ap,
                    Ah,
                    Aj,
                    Ax,
                    len(indptr),
                    len(rows),
                    len(col_indices),
                    len(values),
                    nvec,
                    not sorted_index,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_hypercsc(
            cls,
            *,
            nrows,
            ncols,
            cols,
            indptr,
            values,
            row_indices,
            sorted_index=False,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "hypercsc":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'hypercsc'.")
            # if not cols.flags.owndata:  # TODO: fix leak
            #     cols = cols.copy()
            # if not indptr.flags.owndata:  # TODO: fix leak
            #     indptr = indptr.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            # if not row_indices.flags.owndata:  # TODO: fix leak
            #     row_indices = row_indices.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ap = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indptr)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            Ah = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(cols)))
            Ai = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(row_indices)))
            nvec = len(cols)
            check_status(
                lib.GxB_Matrix_import_HyperCSC(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ap,
                    Ah,
                    Ai,
                    Ax,
                    len(indptr),
                    len(cols),
                    len(row_indices),
                    len(values),
                    nvec,
                    not sorted_index,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_bitmapr(
            cls,
            *,
            nrows,
            ncols,
            bitmap,
            values,
            nvals=None,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "bitmapr":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapr'.")
            # if not bitmap.flags.owndata:  # TODO: fix leak
            #     bitmap = bitmap.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ab = ffi_new("int8_t**", ffi.cast("int8_t*", ffi.from_buffer(bitmap)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if nvals is None:
                nvals = np.count_nonzero(bitmap)
            check_status(
                lib.GxB_Matrix_import_BitmapR(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ab,
                    Ax,
                    len(bitmap),
                    len(values),
                    nvals,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_bitmapc(
            cls,
            *,
            nrows,
            ncols,
            bitmap,
            values,
            nvals=None,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "bitmapc":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapc'.")
            # if not bitmap.flags.owndata:  # TODO: fix leak
            #     bitmap = bitmap.copy()
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ab = ffi_new("int8_t**", ffi.cast("int8_t*", ffi.from_buffer(bitmap)))
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if nvals is None:
                nvals = np.count_nonzero(bitmap)
            check_status(
                lib.GxB_Matrix_import_BitmapC(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ab,
                    Ax,
                    len(bitmap),
                    len(values),
                    nvals,
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_fullr(
            cls,
            *,
            nrows,
            ncols,
            values,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "fullr":
                raise ValueError(f"Invalid format: {format!r}  Must be None or 'fullr'.")
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            check_status(
                lib.GxB_Matrix_import_FullR(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ax,
                    len(values),
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import_fullc(
            cls,
            *,
            nrows,
            ncols,
            values,
            format=None,
            name=None,
        ):
            if format is not None and format.lower() != "fullc":
                raise ValueError(f"Invalid format: {format!r}.  Must be None or 'fullc'.")
            # if not values.flags.owndata:  # TODO: fix leak
            #     values = values.copy()
            mhandle = ffi_new("GrB_Matrix*")
            dtype = lookup_dtype(values.dtype)
            Ax = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            check_status(
                lib.GxB_Matrix_import_FullC(
                    mhandle,
                    dtype._carg,
                    nrows,
                    ncols,
                    Ax,
                    len(values),
                    ffi.NULL,
                ),
                "Matrix",
                mhandle[0],
            )
            rv = Matrix(mhandle, dtype, name=name)
            rv._nrows = nrows
            rv._ncols = ncols
            return rv

        @classmethod
        def fast_import(
            cls,
            *,
            # All
            nrows,
            ncols,
            values,
            format=None,
            name=None,
            # CSR/CSC/HyperCSR/HyperCSC
            indptr=None,
            sorted_index=False,
            # CSR/HyperCSR
            col_indices=None,
            # HyperCSR
            rows=None,
            # CSC/HyperCSC
            row_indices=None,
            # HyperCSC
            cols=None,
            # BitmapR/BitmapC
            bitmap=None,
            nvals=None,  # optional
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
             - bitmapr  (needs bitmap, nvals (optional), values)
             - bitmapc  (needs bitmap, nvals (optional), values)
             - fullr  (needs values)
             - fullc  (needs values)
            """
            if format is None:
                # Determine format based on provided inputs
                if indptr is not None:
                    if bitmap is not None:
                        raise ValueError("Cannot provide both `indptr` and `bitmap`")
                    if row_indices is None and col_indices is None:
                        raise ValueError("Must provide either `row_indices` or `col_indices`")
                    if row_indices is not None and col_indices is not None:
                        raise ValueError("Cannot provide both `row_indices` and `col_indices`")
                    if rows is not None and cols is not None:
                        raise ValueError("Cannot provide both `rows` and `cols`")
                    elif rows is None and cols is None:
                        if row_indices is None:
                            format = "csr"
                        else:
                            format = "csc"
                    elif rows is not None:
                        if col_indices is None:
                            raise ValueError("HyperCSR requires col_indices, not row_indices")
                        format = "hypercsr"
                    else:
                        if row_indices is None:
                            raise ValueError("HyperCSC requires row_indices, not col_indices")
                        format = "hypercsc"
                elif bitmap is not None:
                    if col_indices is not None:
                        raise ValueError("Cannot provide both `bitmap` and `col_indices`")
                    if row_indices is not None:
                        raise ValueError("Cannot provide both `bitmap` and `row_indices`")
                    if cols is not None:
                        raise ValueError("Cannot provide both `bitmap` and `cols`")
                    if rows is not None:
                        raise ValueError("Cannot provide both `bitmap` and `rows`")
                    # Assume row-oriented
                    format = "bitmapr"
                else:
                    # Assume row-oriented
                    format = "fullr"

            format = format.lower()
            if format == "csr":
                return cls.fast_import_csr(
                    nrows=nrows,
                    ncols=ncols,
                    indptr=indptr,
                    values=values,
                    col_indices=col_indices,
                    sorted_index=sorted_index,
                    format=format,
                    name=name,
                )
            elif format == "csc":
                return cls.fast_import_csc(
                    nrows=nrows,
                    ncols=ncols,
                    indptr=indptr,
                    values=values,
                    row_indices=row_indices,
                    sorted_index=sorted_index,
                    format=format,
                    name=name,
                )
            elif format == "hypercsr":
                return cls.fast_import_hypercsr(
                    nrows=nrows,
                    ncols=ncols,
                    rows=rows,
                    indptr=indptr,
                    values=values,
                    col_indices=col_indices,
                    sorted_index=sorted_index,
                    format=format,
                    name=name,
                )
            elif format == "hypercsc":
                return cls.fast_import_hypercsc(
                    nrows=nrows,
                    ncols=ncols,
                    cols=cols,
                    indptr=indptr,
                    values=values,
                    row_indices=row_indices,
                    sorted_index=sorted_index,
                    format=format,
                    name=name,
                )
            elif format == "bitmapr":
                return cls.fast_import_bitmapr(
                    nrows=nrows,
                    ncols=ncols,
                    values=values,
                    nvals=nvals,
                    bitmap=bitmap,
                    format=format,
                    name=name,
                )
            elif format == "bitmapc":
                return cls.fast_import_bitmapc(
                    nrows=nrows,
                    ncols=ncols,
                    values=values,
                    nvals=nvals,
                    bitmap=bitmap,
                    format=format,
                    name=name,
                )
            elif format == "fullr":
                return cls.fast_import_fullr(
                    nrows=nrows,
                    ncols=ncols,
                    values=values,
                    format=format,
                    name=name,
                )
            elif format == "fullc":
                return cls.fast_import_fullc(
                    nrows=nrows,
                    ncols=ncols,
                    values=values,
                    format=format,
                    name=name,
                )
            else:
                raise ValueError(f"Invalid format: {format}")


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

    def to_values(self, *, dtype=None):
        rows, cols, vals = self._matrix.to_values(dtype=dtype)
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

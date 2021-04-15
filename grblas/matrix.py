import itertools
import numpy as np
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType, call
from .dtypes import lookup_dtype, unify, _INDEX
from .exceptions import check_status, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
from .mask import StructuralMask, ValueMask
from .operator import get_typed_op
from .vector import Vector, VectorExpression
from .scalar import Scalar, ScalarExpression, _CScalar
from .utils import (
    ints_to_numpy_buffer,
    values_to_numpy_buffer,
    wrapdoc,
    _CArray,
    _Pointer,
    class_property,
)
from . import expr
from ._ss.matrix import ss

ffi_new = ffi.new


class Matrix(BaseType):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """

    __slots__ = "_nrows", "_ncols", "ss"
    _is_transposed = False
    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, name=None):
        if name is None:
            name = f"M_{next(Matrix._name_counter)}"
        self._nrows = None
        self._ncols = None
        super().__init__(gb_obj, dtype, name)
        # Add ss extension methods
        self.ss = ss(self)

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

    def __contains__(self, index):
        extractor = self[index]
        if not extractor.resolved_indexes.is_single_element:
            raise TypeError(
                f"Invalid index to Matrix contains: {index!r}.  A 2-tuple of ints is expected.  "
                "Doing `(i, j) in my_matrix` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar.is_empty

    def __iter__(self):
        rows, columns, values = self.to_values()
        return zip(rows.flat, columns.flat)

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
        scalar = Scalar(n, _INDEX, name="s_nrows", empty=True)
        call("GrB_Matrix_nrows", [_Pointer(scalar), self])
        return n[0]

    @property
    def ncols(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_ncols", empty=True)
        call("GrB_Matrix_ncols", [_Pointer(scalar), self])
        return n[0]

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", empty=True)
        call("GrB_Matrix_nvals", [_Pointer(scalar), self])
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
        call("GrB_Matrix_resize", [self, nrows, ncols])
        self._nrows = nrows.scalar.value
        self._ncols = ncols.scalar.value

    def to_values(self, *, dtype=None):
        """
        GrB_Matrix_extractTuples
        Extract the rows, columns and values as a 3-tuple of numpy arrays
        """
        nvals = self._nvals
        rows = _CArray(size=nvals, name="&rows_array")
        columns = _CArray(size=nvals, name="&columns_array")
        values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", empty=True)
        scalar.value = nvals
        call(
            f"GrB_Matrix_extractTuples_{self.dtype.name}",
            [rows, columns, values, _Pointer(scalar), self],
        )
        values = values.array
        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                values = values.astype(dtype.np_type)  # copies
        return (
            rows.array,
            columns.array,
            values,
        )

    def build(self, rows, columns, values, *, dup_op=None, clear=False, nrows=None, ncols=None):
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        rows = ints_to_numpy_buffer(rows, np.uint64, name="row indices")
        columns = ints_to_numpy_buffer(columns, np.uint64, name="column indices")
        values, dtype = values_to_numpy_buffer(values, self.dtype)
        n = values.size
        if rows.size != n or columns.size != n:
            raise ValueError(
                f"`rows` and `columns` and `values` lengths must match: "
                f"{rows.size}, {columns.size}, {values.size}"
            )
        if clear:
            self.clear()
        if nrows is not None or ncols is not None:
            if nrows is None:
                nrows = self.nrows
            if ncols is None:
                ncols = self.ncols
            self.resize(nrows, ncols)
        if n == 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        if dup_op.opclass == "Monoid":
            dup_op = dup_op.binaryop
        else:
            self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        rows = _CArray(rows)
        columns = _CArray(columns)
        values = _CArray(values, dtype=self.dtype)
        call(
            f"GrB_Matrix_build_{self.dtype.name}",
            [self, rows, columns, values, _CScalar(n), dup_op],
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
            call("GrB_Matrix_dup", [_Pointer(rv), self])
        rv._nrows = self._nrows
        rv._ncols = self._ncols
        return rv

    def wait(self):
        """
        GrB_Matrix_wait

        In non-blocking mode, the computations may be delayed and not yet safe
        to use by multiple threads.  Use wait to force completion of a Matrix
        and make it safe to use as input parameters on multiple threads.
        """
        call("GrB_Matrix_wait", [_Pointer(self)])

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
        call("GrB_Matrix_new", [_Pointer(rv), dtype, nrows, ncols])
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
        rows = ints_to_numpy_buffer(rows, np.uint64, name="row indices")
        columns = ints_to_numpy_buffer(columns, np.uint64, name="column indices")
        values, dtype = values_to_numpy_buffer(values, dtype)
        # Compute nrows and ncols if not provided
        if nrows is None:
            if rows.size == 0:
                raise ValueError("No row indices provided. Unable to infer nrows.")
            nrows = int(rows.max()) + 1
        if ncols is None:
            if columns.size == 0:
                raise ValueError("No column indices provided. Unable to infer ncols.")
            ncols = int(columns.max()) + 1
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
        # Per the spec, op may be a semiring, but this is weird, so don't.
        if require_monoid:
            if op.opclass != "BinaryOp" or op.monoid is None:
                self._expect_op(
                    op,
                    "Monoid",
                    within=method_name,
                    argname="op",
                    extra_message="A BinaryOp may be given if require_monoid keyword is False",
                )
        else:
            self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
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
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
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
            if type(left) is not Scalar:
                try:
                    left = Scalar.from_value(left, name="")
                except TypeError:
                    self._expect_type(
                        left,
                        Scalar,
                        within=method_name,
                        keyword_name="left",
                        extra_message="Literal scalars also accepted.",
                    )
            op = get_typed_op(op, self.dtype, left.dtype)
            if op.opclass == "Monoid":
                op = op.binaryop
            else:
                self._expect_op(
                    op,
                    "BinaryOp",
                    within=method_name,
                    argname="op",
                    extra_message=extra_message,
                )
            cfunc_name = f"GrB_Matrix_apply_BinaryOp1st_{left.dtype}"
            args = [_CScalar(left), self]
            expr_repr = "{1.name}.apply({op}, left={0})"
        elif left is None:
            if type(right) is not Scalar:
                try:
                    right = Scalar.from_value(right, name="")
                except TypeError:
                    self._expect_type(
                        right,
                        Scalar,
                        within=method_name,
                        keyword_name="right",
                        extra_message="Literal scalars also accepted.",
                    )
            op = get_typed_op(op, self.dtype, right.dtype)
            if op.opclass == "Monoid":
                op = op.binaryop
            else:
                self._expect_op(
                    op,
                    "BinaryOp",
                    within=method_name,
                    argname="op",
                    extra_message=extra_message,
                )
            cfunc_name = f"GrB_Matrix_apply_BinaryOp2nd_{right.dtype}"
            args = [self, _CScalar(right)]
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

    def reduce_columns(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_columns"
        op = get_typed_op(op, self.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
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

    def reduce_scalar(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_scalar"
        op = get_typed_op(op, self.dtype)
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        else:
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
            call(f"GrB_Matrix_extractElement_{dtype}", [_Pointer(result), self, row, col])
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
        if type(value) is not Scalar:
            try:
                value = Scalar.from_value(value, name="")
            except TypeError:
                self._expect_type(
                    value,
                    Scalar,
                    within="__setitem__",
                    argname="value",
                    extra_message="Literal scalars also accepted.",
                )
        # should we cast?
        call(f"GrB_Matrix_setElement_{value.dtype}", [self, _CScalar(value), row, col])

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
            if type(value) is not Scalar:
                try:
                    value = Scalar.from_value(value, name="")
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
                    [_CScalar(value), rows, rowsize, cols, colsize],
                    expr_repr=expr_repr,
                    nrows=self._nrows,
                    ncols=self._ncols,
                    dtype=self.dtype,
                )
        return delayed

    def _delete_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        call("GrB_Matrix_removeElement", [self, row, col])

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


Matrix.ss = class_property(Matrix.ss, ss)


class MatrixExpression(BaseExpression):
    __slots__ = "_ncols", "_nrows"
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
        self._ncols = ncols
        self._nrows = nrows

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

    @property
    def ncols(self):
        return self._ncols

    @property
    def nrows(self):
        return self._nrows


class TransposedMatrix:
    __slots__ = "_matrix", "_ncols", "_nrows", "__weakref__"
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

    @wrapdoc(Matrix.to_values)
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

    # Operator sugar
    __or__ = Matrix.__or__
    __ror__ = Matrix.__ror__
    __ior__ = Matrix.__ior__
    __and__ = Matrix.__and__
    __rand__ = Matrix.__rand__
    __iand__ = Matrix.__iand__
    __matmul__ = Matrix.__matmul__
    __rmatmul__ = Matrix.__rmatmul__
    __imatmul__ = Matrix.__imatmul__

    # Misc.
    isequal = Matrix.isequal
    isclose = Matrix.isclose
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
    __array_struct__ = Matrix.__array_struct__


expr.MatrixEwiseAddExpr.output_type = MatrixExpression
expr.MatrixEwiseMultExpr.output_type = MatrixExpression
expr.MatrixMatMulExpr.output_type = MatrixExpression

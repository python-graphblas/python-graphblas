import itertools
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType
from .dtypes import libget, lookup_dtype, unify
from .exceptions import check_status, is_error, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
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
        super().__init__(gb_obj, dtype, name)

    def __del__(self):
        check_status(lib.GrB_Matrix_free(self.gb_obj))

    def __repr__(self, mask=None):
        from .formatting import format_matrix

        return format_matrix(self, mask=mask)

    def _repr_html_(self, mask=None):
        from .formatting import format_matrix_html

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
        if self.nrows != other.nrows:
            return False
        if self.ncols != other.ncols:
            return False
        if self.nvals != other.nvals:
            return False
        if check_dtype:
            common_dtype = self.dtype
        else:
            common_dtype = unify(self.dtype, other.dtype)

        matches = Matrix.new(bool, self.nrows, self.ncols, name="M_isequal")
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
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
        if self.nrows != other.nrows:
            return False
        if self.ncols != other.ncols:
            return False
        if self.nvals != other.nvals:
            return False

        matches = self.ewise_mult(other, binary.isclose(rel_tol, abs_tol)).new(
            dtype=bool, name="M_isclose"
        )
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land).value

    @property
    def nrows(self):
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Matrix_nrows(n, self.gb_obj[0]))
        return n[0]

    @property
    def ncols(self):
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Matrix_ncols(n, self.gb_obj[0]))
        return n[0]

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Matrix_nvals(n, self.gb_obj[0]))
        return n[0]

    @property
    def T(self):
        return TransposedMatrix(self)

    def clear(self):
        check_status(lib.GrB_Matrix_clear(self.gb_obj[0]))

    def resize(self, nrows, ncols):
        check_status(lib.GrB_Matrix_resize(self.gb_obj[0], nrows, ncols))

    def to_values(self):
        """
        GrB_Matrix_extractTuples
        Extract the rows, columns and values as 3 generators
        """
        rows = ffi_new("GrB_Index[]", self.nvals)
        columns = ffi_new("GrB_Index[]", self.nvals)
        values = ffi_new(f"{self.dtype.c_type}[]", self.nvals)
        n = ffi_new("GrB_Index*")
        n[0] = self.nvals
        func = libget(f"GrB_Matrix_extractTuples_{self.dtype.name}")
        check_status(func(rows, columns, values, n, self.gb_obj[0]))
        return tuple(rows), tuple(columns), tuple(values)

    def build(self, rows, columns, values, *, dup_op=None, clear=False):
        # TODO: add `size` option once .resize is available
        if not isinstance(rows, (tuple, list)):
            rows = tuple(rows)
        if not isinstance(columns, (tuple, list)):
            columns = tuple(columns)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        n = len(values)
        if len(rows) != n or len(columns) != n:
            raise ValueError(
                f"`rows` and `columns` and `values` lengths must match: "
                f"{len(rows)}, {len(columns)}, {len(values)}"
            )
        if clear:
            self.clear()
        if n <= 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")
        rows = ffi_new("GrB_Index[]", rows)
        columns = ffi_new("GrB_Index[]", columns)
        values = ffi_new(f"{self.dtype.c_type}[]", values)
        # Push values into w
        func = libget(f"GrB_Matrix_build_{self.dtype.name}")
        check_status(func(self.gb_obj[0], rows, columns, values, n, dup_op.gb_obj))
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self.nvals < len(values):
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, *, dtype=None, mask=None, name=None):
        """
        GrB_Matrix_dup
        Create a new Matrix by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            new_mat = type(self).new(dtype, nrows=self.nrows, ncols=self.ncols, name=name)
            new_mat(mask=mask)[:, :] << self
            return new_mat
        new_mat = ffi_new("GrB_Matrix*")
        check_status(lib.GrB_Matrix_dup(new_mat, self.gb_obj[0]))
        return type(self)(new_mat, self.dtype, name=name)

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        """
        GrB_Matrix_new
        Create a new empty Matrix from the given type, number of rows, and number of columns
        """
        new_matrix = ffi_new("GrB_Matrix*")
        dtype = lookup_dtype(dtype)
        check_status(lib.GrB_Matrix_new(new_matrix, dtype.gb_type, nrows, ncols))
        return cls(new_matrix, dtype, name=name)

    @classmethod
    def from_values(
        cls, rows, columns, values, *, nrows=None, ncols=None, dup_op=None, dtype=None, name=None,
    ):
        """Create a new Matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max row and coumn index found.
        """
        if not isinstance(rows, (tuple, list)):
            rows = tuple(rows)
        if not isinstance(columns, (tuple, list)):
            columns = tuple(columns)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if dtype is None:
            if len(values) <= 0:
                raise ValueError("No values provided. Unable to determine type.")
            # Find dtype from any of the values (assumption is they are the same type)
            dtype = type(values[0])
        dtype = lookup_dtype(dtype)
        # Compute nrows and ncols if not provided
        if nrows is None:
            if not rows:
                raise ValueError("No row indices provided. Unable to infer nrows.")
            nrows = max(rows) + 1
        if ncols is None:
            if not columns:
                raise ValueError("No column indices provided. Unable to infer ncols.")
            ncols = max(columns) + 1
        # Create the new matrix
        C = cls.new(dtype, nrows, ncols, name=name)
        # Add the data
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
        return MatrixExpression(
            method_name,
            f"GrB_eWiseAdd_Matrix_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )

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
        return MatrixExpression(
            method_name,
            f"GrB_eWiseMult_Matrix_{op.opclass}",
            [self, other],
            op=op,
            at=self._is_transposed,
            bt=other._is_transposed,
        )

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
        return VectorExpression(
            method_name, "GrB_mxv", [self, other], op=op, size=self.nrows, at=self._is_transposed,
        )

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
        return MatrixExpression(
            method_name,
            "GrB_mxm",
            [self, other],
            op=op,
            nrows=self.nrows,
            ncols=other.ncols,
            at=self._is_transposed,
            bt=other._is_transposed,
        )

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
            nrows=self.nrows * other.nrows,
            ncols=self.ncols * other.ncols,
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
                op, "UnaryOp", within=method_name, argname="op", extra_message=extra_message,
            )
            cfunc_name = "GrB_Matrix_apply"
            args = [self]
            expr_repr = None
        elif right is None:
            if type(left) is not Scalar:
                try:
                    left = Scalar.from_value(left, name="left")
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
                op, "BinaryOp", within=method_name, argname="op", extra_message=extra_message,
            )
            cfunc_name = f"GrB_Matrix_apply_BinaryOp1st_{left.dtype}"
            args = [_CScalar(left), self]
            expr_repr = "{1.name}.apply({op}, left={0})"
        elif left is None:
            if type(right) is not Scalar:
                try:
                    right = Scalar.from_value(right, name="right")
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
                op, "BinaryOp", within=method_name, argname="op", extra_message=extra_message,
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
            nrows=self.nrows,
            ncols=self.ncols,
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
            size=self.nrows,
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
            size=self.ncols,
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
    def _extract_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        func = libget(f"GrB_Matrix_extractElement_{self.dtype}")
        result = ffi_new(f"{self.dtype.c_type}*")
        if self._is_transposed:
            row, col = col, row
        err_code = func(result, self.gb_obj[0], row, col)
        # Don't raise error for no value, simply return `None`
        if is_error(err_code, NoValue):
            return None, self.dtype
        check_status(err_code)
        return result[0], self.dtype

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
                value = Scalar.from_value(value, name="s_assign")
            except TypeError:
                self._expect_type(
                    value,
                    Scalar,
                    within="__setitem__",
                    argname="value",
                    extra_message="Literal scalars also accepted.",
                )
        func = libget(f"GrB_Matrix_setElement_{value.dtype}")
        check_status(func(self.gb_obj[0], value.value, row, col))  # should we cast?

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
                    nrows=self.nrows,
                    ncols=self.ncols,
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
                    nrows=self.nrows,
                    ncols=self.ncols,
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
                    value, (Scalar, Vector), within=method_name, extra_message=extra_message,
                )
            delayed = MatrixExpression(
                method_name,
                "GrB_Matrix_assign",
                [value, rows, rowsize, cols, colsize],
                expr_repr="[[{2} rows], [{4} cols]] = {0.name}",
                nrows=self.nrows,
                ncols=self.ncols,
                dtype=self.dtype,
                at=value._is_transposed,
            )
        else:
            if type(value) is not Scalar:
                try:
                    value = Scalar.from_value(value, name="s_assign")
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
                rows = [rows]
                rowsize = 1
            if colsize is None:
                cols = [cols]
                colsize = 1
            delayed = MatrixExpression(
                method_name,
                f"GrB_Matrix_assign_{value.dtype}",
                [_CScalar(value), rows, rowsize, cols, colsize],
                expr_repr="[[{2} rows], [{4} cols]] = {0}",
                nrows=self.nrows,
                ncols=self.ncols,
                dtype=self.dtype,
            )
        return delayed

    def _delete_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        check_status(lib.GrB_Matrix_removeElement(self.gb_obj[0], row, col))

    if backend == "pygraphblas":  # pragma: no cover

        def to_pygraphblas(self):
            """ Convert to a new `pygraphblas.Matrix`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `pygraphblas`.
            This means operations on the current `grblas` object will fail!
            """
            import pygraphblas

            matrix = pygraphblas.Matrix(self.gb_obj, self.dtype.gb_type)
            self.gb_obj = ffi.NULL
            return matrix

        @classmethod
        def from_pygraphblas(cls, matrix):
            """ Convert a `pygraphblas.Matrix` to a new `grblas.Matrix`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `grblas`.
            This means operations on the original `pygraphblas` object will fail!
            """
            dtype = lookup_dtype(matrix.gb_type)
            rv = cls(matrix.matrix, dtype)
            matrix.matrix = ffi.NULL
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
            method_name, cfunc_name, args, at=at, bt=bt, op=op, dtype=dtype, expr_repr=expr_repr,
        )
        if ncols is None:
            ncols = args[0].ncols
        if nrows is None:
            nrows = args[0].nrows
        self.ncols = ncols
        self.nrows = nrows

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Matrix.new(dtype, self.nrows, self.ncols, name=name)

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

    def __repr__(self):
        from .formatting import format_matrix

        return format_matrix(self)

    def _repr_html_(self):
        from .formatting import format_matrix_html

        return format_matrix_html(self)

    def new(self, *, dtype=None, mask=None, name=None):
        if dtype is None:
            dtype = self.dtype
        output = Matrix.new(dtype, self.nrows, self.ncols, name=name)
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

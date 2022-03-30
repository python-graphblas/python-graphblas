import itertools

import numpy as np

from . import _automethods, backend, binary, ffi, lib, monoid, semiring, utils
from ._ss.matrix import ss
from .base import BaseExpression, BaseType, call
from .dtypes import _INDEX, lookup_dtype, unify
from .exceptions import NoValue, check_status
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
from .mask import StructuralMask, ValueMask
from .operator import get_typed_op
from .scalar import _MATERIALIZE, Scalar, ScalarExpression, _as_scalar
from .utils import (
    _CArray,
    _Pointer,
    class_property,
    ints_to_numpy_buffer,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)
from .vector import Vector, VectorExpression

ffi_new = ffi.new


class Matrix(BaseType):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """

    __slots__ = "_nrows", "_ncols", "_parent", "ss"
    ndim = 2
    _is_transposed = False
    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, parent=None, name=None):
        if name is None:
            name = f"M_{next(Matrix._name_counter)}"
        self._nrows = None
        self._ncols = None
        super().__init__(gb_obj, dtype, name)
        # Add ss extension methods
        self.ss = ss(self)
        self._parent = parent

    def __del__(self):
        parent = getattr(self, "_parent", None)
        if parent is not None:
            return
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Matrix_free(gb_obj), self)

    def __repr__(self, mask=None):
        from .formatting import format_matrix
        from .recorder import skip_record

        with skip_record:
            return format_matrix(self, mask=mask)

    def _repr_html_(self, mask=None, collapse=False):
        if self._parent is not None:
            return self._parent._repr_html_(mask=mask, collapse=collapse)
        from .formatting import format_matrix_html
        from .recorder import skip_record

        with skip_record:
            return format_matrix_html(self, mask=mask, collapse=collapse)

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
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    def __delitem__(self, keys):
        del Updater(self)[keys]

    def __getitem__(self, keys):
        resolved_indexes = IndexerResolver(self, keys)
        shape = resolved_indexes.shape
        if not shape:
            from .scalar import ScalarIndexExpr

            return ScalarIndexExpr(self, resolved_indexes)
        elif len(shape) == 1:
            from .vector import VectorIndexExpr

            return VectorIndexExpr(self, resolved_indexes, *shape)
        else:
            return MatrixIndexExpr(self, resolved_indexes, *shape)

    def __setitem__(self, keys, expr):
        Updater(self)[keys] = expr

    def __contains__(self, index):
        extractor = self[index]
        if not extractor._is_scalar:
            raise TypeError(
                f"Invalid index to Matrix contains: {index!r}.  A 2-tuple of ints is expected.  "
                "Doing `(i, j) in my_matrix` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar._is_empty

    def __iter__(self):
        self.wait()  # sort in SS
        rows, columns, values = self.to_values()
        return zip(rows.flat, columns.flat)

    def __sizeof__(self):
        size = ffi_new("size_t*")
        scalar = Scalar(size, _INDEX, name="s_size", is_cscalar=True, empty=True)
        call("GxB_Matrix_memoryUsage", [_Pointer(scalar), self])
        return size[0] + object.__sizeof__(self)

    def isequal(self, other, *, check_dtype=False):
        """
        Check for exact equality (same size, same empty values)
        If `check_dtype` is True, also checks that dtypes match
        For equality of floating point Vectors, consider using `isclose`
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
            common_dtype = self.dtype
        else:
            common_dtype = unify(self.dtype, other.dtype)

        matches = Matrix.new(bool, self._nrows, self._ncols, name="M_isequal")
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land, allow_empty=False).new().value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
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
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nrows", is_cscalar=True, empty=True)
        call("GrB_Matrix_nrows", [_Pointer(scalar), self])
        return n[0]

    @property
    def ncols(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_ncols", is_cscalar=True, empty=True)
        call("GrB_Matrix_ncols", [_Pointer(scalar), self])
        return n[0]

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", is_cscalar=True, empty=True)
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
        nrows = _as_scalar(nrows, _INDEX, is_cscalar=True)
        ncols = _as_scalar(ncols, _INDEX, is_cscalar=True)
        call("GrB_Matrix_resize", [self, nrows, ncols])
        self._nrows = nrows.value
        self._ncols = ncols.value

    def to_values(self, dtype=None):
        """
        GrB_Matrix_extractTuples
        Extract the rows, columns and values as a 3-tuple of numpy arrays
        """
        nvals = self._nvals
        rows = _CArray(size=nvals, name="&rows_array")
        columns = _CArray(size=nvals, name="&columns_array")
        values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", is_cscalar=True, empty=True)
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
                nrows = self._nrows
            if ncols is None:
                ncols = self._ncols
            self.resize(nrows, ncols)
        if n == 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype, kind="binary")
        if dup_op.opclass == "Monoid":
            dup_op = dup_op.binaryop
        else:
            self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        rows = _CArray(rows)
        columns = _CArray(columns)
        values = _CArray(values, self.dtype)
        call(
            f"GrB_Matrix_build_{self.dtype.name}",
            [self, rows, columns, values, _as_scalar(n, _INDEX, is_cscalar=True), dup_op],
        )
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, dtype=None, *, mask=None, name=None):
        """
        GrB_Matrix_dup
        Create a new Matrix by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            rv = Matrix.new(dtype, nrows=self._nrows, ncols=self._ncols, name=name)
            rv(mask=mask)[...] = self
        else:
            new_mat = ffi_new("GrB_Matrix*")
            rv = Matrix(new_mat, self.dtype, name=name)
            call("GrB_Matrix_dup", [_Pointer(rv), self])
        rv._nrows = self._nrows
        rv._ncols = self._ncols
        return rv

    def diag(self, k=0, dtype=None, *, name=None):
        from .ss._core import diag

        return diag(self, k=k, dtype=dtype, name=name)

    def wait(self):
        """
        GrB_Matrix_wait

        In non-blocking mode, the computations may be delayed and not yet safe
        to use by multiple threads.  Use wait to force completion of a Matrix
        and make it safe to use as input parameters on multiple threads.
        """
        # TODO: expose COMPLETE or MATERIALIZE options to the user
        call("GrB_Matrix_wait", [self, _MATERIALIZE])

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        """
        GrB_Matrix_new
        Create a new empty Matrix from the given type, number of rows, and number of columns
        """
        new_matrix = ffi_new("GrB_Matrix*")
        dtype = lookup_dtype(dtype)
        rv = cls(new_matrix, dtype, name=name)
        nrows = _as_scalar(nrows, _INDEX, is_cscalar=True)
        ncols = _as_scalar(ncols, _INDEX, is_cscalar=True)
        call("GrB_Matrix_new", [_Pointer(rv), dtype, nrows, ncols])
        rv._nrows = nrows.value
        rv._ncols = ncols.value
        return rv

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
        """Create a new Matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max row and column index found.

        values may be a scalar, in which case duplicate indices are ignored.
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
        other = self._expect_type(
            other,
            (Matrix, TransposedMatrix),
            within=method_name,
            argname="other",
            op=op,
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        if require_monoid:
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
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
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

    def ewise_union(self, other, op, left_default, right_default):
        """
        GxB_Matrix_eWiseUnion

        This is similar to `ewise_add` in that result will contain the union of
        indices from both Matrices.  Unlike `ewise_add`, this will use
        ``left_default`` for the left value when there is a value on the right
        but not the left, and ``right_default`` for the right value when there
        is a value on the left but not the right.

        ``op`` should be a BinaryOp or Monoid.
        """
        # SS, SuiteSparse-specific: eWiseUnion
        method_name = "ewise_union"
        other = self._expect_type(other, Matrix, within=method_name, argname="other", op=op)
        if type(left_default) is not Scalar:
            try:
                left = Scalar.from_value(
                    left_default, is_cscalar=False, name=""  # pragma: is_grbscalar
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
            left = _as_scalar(left_default, is_cscalar=False)  # pragma: is_grbscalar
        if type(right_default) is not Scalar:
            try:
                right = Scalar.from_value(
                    right_default, is_cscalar=False, name=""  # pragma: is_grbscalar
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
            right = _as_scalar(right_default, is_cscalar=False)  # pragma: is_grbscalar
        scalar_dtype = unify(left.dtype, right.dtype)
        nonscalar_dtype = unify(self.dtype, other.dtype)
        op = get_typed_op(op, scalar_dtype, nonscalar_dtype, is_left_scalar=True, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        expr = MatrixExpression(
            method_name,
            "GxB_Matrix_eWiseUnion",
            [self, left, other, right],
            op=op,
            expr_repr="{0.name}.{method_name}({2.name}, {op}, {1._expr_name}, {3._expr_name})",
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
        """
        GrB_mxm
        Matrix-Matrix multiplication. Result is a Matrix.
        Default op is semiring.plus_times
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
        """
        GrB_kronecker
        Kronecker product or sum (depending on op used)
        Default op is binary.times
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
                try:
                    left = Scalar.from_value(left, is_cscalar=None, name="")
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
            if left._is_cscalar:
                cfunc_name = f"GrB_Matrix_apply_BinaryOp1st_{left.dtype}"
            else:
                cfunc_name = "GrB_Matrix_apply_BinaryOp1st_Scalar"
            args = [left, self]
            expr_repr = "{1.name}.apply({op}, left={0._expr_name})"
        elif left is None:
            if type(right) is not Scalar:
                try:
                    right = Scalar.from_value(right, is_cscalar=None, name="")
                except TypeError:
                    right = self._expect_type(
                        right,
                        Scalar,
                        within=method_name,
                        keyword_name="right",
                        extra_message="Literal scalars also accepted.",
                        op=op,
                    )
            op = get_typed_op(op, self.dtype, right.dtype, is_right_scalar=True, kind="binary")
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
            if right._is_cscalar:
                cfunc_name = f"GrB_Matrix_apply_BinaryOp2nd_{right.dtype}"
            else:
                cfunc_name = "GrB_Matrix_apply_BinaryOp2nd_Scalar"
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

    def reduce_rowwise(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each row, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
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
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
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
        GrB_Matrix_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise

        For empty Matrix objects, the result will be the monoid identity if
        `allow_empty` is False or empty Scalar if `allow_empty` is True.
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
        else:
            cfunc_name = "GrB_Matrix_reduce_{output_dtype}"
        return ScalarExpression(
            method_name,
            cfunc_name,
            [self],
            op=op,  # to be determined later
            is_cscalar=not allow_empty,
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
            result = Scalar.new(dtype, is_cscalar=is_cscalar, name=name)
        if is_cscalar:
            if (
                call(
                    f"GrB_Matrix_extractElement_{dtype}",
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
            try:
                value = Scalar.from_value(value, is_cscalar=None, name="")
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
            cfunc_name = f"GrB_Matrix_setElement_{value.dtype}"
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
                try:
                    value = Scalar.from_value(value, is_cscalar=None, name="")
                except TypeError:
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
            if mask is not None and type(mask.mask) is Vector:
                if rowsize is None and colsize is not None:
                    if is_submask:
                        # C[i, J](m) << c
                        # SS, SuiteSparse-specific: subassign
                        cfunc_name = "GrB_Row_subassign"
                        value_vector = Vector.new(value.dtype, size=mask.mask._size, name="v_temp")
                        expr_repr = (
                            "[{1._expr_name}, [{3._expr_name} cols]](%s) << {0.name}" % mask.name
                        )
                    else:
                        # C(m)[i, J] << c
                        # C[i, J] << c
                        cfunc_name = "GrB_Row_assign"
                        value_vector = Vector.new(value.dtype, size=colsize, name="v_temp")
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
                        value_vector = Vector.new(value.dtype, size=mask.mask._size, name="v_temp")
                    else:
                        # C(m)[I, j] << c
                        # C[I, j] << c
                        cfunc_name = "GrB_Col_assign"
                        value_vector = Vector.new(value.dtype, size=rowsize, name="v_temp")
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
                        cfunc_name = f"GrB_Matrix_subassign_{value.dtype}"
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
                        cfunc_name = f"GrB_Matrix_assign_{value.dtype}"
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
        """Convert to a new `pygraphblas.Matrix`

        This does not copy data.

        This gives control of the underlying GraphBLAS object to `pygraphblas`.
        This means operations on the current `grblas` object will fail!
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"to_pygraphblas only works with 'suitesparse' backend, not {backend}"
            )
        import pygraphblas as pg

        matrix = pg.Matrix(self.gb_obj, pg.types._gb_type_to_type(self.dtype.gb_obj))
        self.gb_obj = ffi.NULL
        return matrix

    @classmethod
    def from_pygraphblas(cls, matrix):  # pragma: no cover
        """Convert a `pygraphblas.Matrix` to a new `grblas.Matrix`

        This does not copy data.

        This gives control of the underlying GraphBLAS object to `grblas`.
        This means operations on the original `pygraphblas` object will fail!
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
        matrix._matrix = ffi.NULL
        return rv


Matrix.ss = class_property(Matrix.ss, ss)


class MatrixExpression(BaseExpression):
    __slots__ = "_ncols", "_nrows"
    ndim = 2
    output_type = Matrix
    _is_transposed = False

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

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    # Begin auto-generated code: Matrix
    _get_value = _automethods._get_value
    S = wrapdoc(Matrix.S)(property(_automethods.S))
    T = wrapdoc(Matrix.T)(property(_automethods.T))
    V = wrapdoc(Matrix.V)(property(_automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(_automethods.__ror__))
    _carg = wrapdoc(Matrix._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(_automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(_automethods.gb_obj))
    isclose = wrapdoc(Matrix.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(_automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(_automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(_automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(_automethods.mxv))
    name = wrapdoc(Matrix.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(_automethods.nvals))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(_automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(_automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(_automethods.reduce_scalar))
    ss = wrapdoc(Matrix.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Matrix.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Matrix.to_values)(property(_automethods.to_values))
    wait = wrapdoc(Matrix.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
    __iadd__ = _automethods.__iadd__
    __iand__ = _automethods.__iand__
    __ifloordiv__ = _automethods.__ifloordiv__
    __imatmul__ = _automethods.__imatmul__
    __imod__ = _automethods.__imod__
    __imul__ = _automethods.__imul__
    __ior__ = _automethods.__ior__
    __ipow__ = _automethods.__ipow__
    __isub__ = _automethods.__isub__
    __itruediv__ = _automethods.__itruediv__
    __ixor__ = _automethods.__ixor__
    # End auto-generated code: Matrix


class MatrixIndexExpr(AmbiguousAssignOrExtract):
    __slots__ = "_ncols", "_nrows"
    ndim = 2
    output_type = Matrix
    _is_transposed = False

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
    _get_value = _automethods._get_value
    S = wrapdoc(Matrix.S)(property(_automethods.S))
    T = wrapdoc(Matrix.T)(property(_automethods.T))
    V = wrapdoc(Matrix.V)(property(_automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(_automethods.__ror__))
    _carg = wrapdoc(Matrix._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(_automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(_automethods.gb_obj))
    isclose = wrapdoc(Matrix.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(_automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(_automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(_automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(_automethods.mxv))
    name = wrapdoc(Matrix.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(_automethods.nvals))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(_automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(_automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(_automethods.reduce_scalar))
    ss = wrapdoc(Matrix.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Matrix.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Matrix.to_values)(property(_automethods.to_values))
    wait = wrapdoc(Matrix.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
    __iadd__ = _automethods.__iadd__
    __iand__ = _automethods.__iand__
    __ifloordiv__ = _automethods.__ifloordiv__
    __imatmul__ = _automethods.__imatmul__
    __imod__ = _automethods.__imod__
    __imul__ = _automethods.__imul__
    __ior__ = _automethods.__ior__
    __ipow__ = _automethods.__ipow__
    __isub__ = _automethods.__isub__
    __itruediv__ = _automethods.__itruediv__
    __ixor__ = _automethods.__ixor__
    # End auto-generated code: Matrix


class TransposedMatrix:
    __slots__ = "_matrix", "_ncols", "_nrows", "__weakref__"
    ndim = 2
    _is_scalar = False
    _is_transposed = True

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
        output = Matrix.new(dtype, self._nrows, self._ncols, name=name)
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
    def to_values(self, dtype=None):
        rows, cols, vals = self._matrix.to_values(dtype)
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
    reduce_rowwise = Matrix.reduce_rowwise
    reduce_columnwise = Matrix.reduce_columnwise
    reduce_scalar = Matrix.reduce_scalar

    # Operator sugar
    __or__ = Matrix.__or__
    __ror__ = Matrix.__ror__
    __and__ = Matrix.__and__
    __rand__ = Matrix.__rand__
    __matmul__ = Matrix.__matmul__
    __rmatmul__ = Matrix.__rmatmul__

    # Bad sugar
    __iadd__ = _automethods.__iadd__
    __iand__ = _automethods.__iand__
    __ifloordiv__ = _automethods.__ifloordiv__
    __imatmul__ = _automethods.__imatmul__
    __imod__ = _automethods.__imod__
    __imul__ = _automethods.__imul__
    __ior__ = _automethods.__ior__
    __ipow__ = _automethods.__ipow__
    __isub__ = _automethods.__isub__
    __itruediv__ = _automethods.__itruediv__
    __ixor__ = _automethods.__ixor__

    # Misc.
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

# Import infix to import _infixmethods, which has side effects
from . import infix  # noqa isort:skip

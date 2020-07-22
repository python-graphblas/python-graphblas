from functools import partial
from .base import lib, ffi, GbContainer, GbDelayed, IndexerResolver, AmbiguousAssignOrExtract, Updater, libget
from .vector import Vector
from .scalar import Scalar
from .ops import get_typed_op
from . import dtypes, binary, monoid, semiring
from .mask import StructuralMask, ValueMask
from .exceptions import check_status, is_error, NoValue


class Matrix(GbContainer):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """
    _is_transposed = False

    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)

    def __del__(self):
        check_status(lib.GrB_Matrix_free(self.gb_obj))

    def __repr__(self):
        return f'<Matrix {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

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
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Argument of isequal must be of type Matrix, not {type(other)}')
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
            common_dtype = dtypes.unify(self.dtype, other.dtype)

        matches = Matrix.new(bool, self.nrows, self.ncols)
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
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Argument of isclose must be of type Matrix, not {type(other)}')
        if check_dtype and self.dtype != other.dtype:
            return False
        if self.nrows != other.nrows:
            return False
        if self.ncols != other.ncols:
            return False
        if self.nvals != other.nvals:
            return False

        matches = self.ewise_mult(other, binary.isclose(rel_tol, abs_tol)).new(dtype=bool)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
            return False

        # Check if all results are True
        return matches.reduce_scalar(monoid.land).value

    @property
    def nrows(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Matrix_nrows(n, self.gb_obj[0]))
        return n[0]

    @property
    def ncols(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Matrix_ncols(n, self.gb_obj[0]))
        return n[0]

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Matrix_nvals(n, self.gb_obj[0]))
        return n[0]

    @property
    def T(self):
        return TransposedMatrix(self)

    def clear(self):
        check_status(lib.GrB_Matrix_clear(self.gb_obj[0]))

    def resize(self, nrows, ncols):
        check_status(lib.GrB_Matrix_resize(
            self.gb_obj[0],
            nrows,
            ncols)
        )

    def to_values(self):
        """
        GrB_Matrix_extractTuples
        Extract the rows, columns and values as 3 generators
        """
        rows = ffi.new('GrB_Index[]', self.nvals)
        columns = ffi.new('GrB_Index[]', self.nvals)
        values = ffi.new(f'{self.dtype.c_type}[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = libget(f'GrB_Matrix_extractTuples_{self.dtype.name}')
        check_status(func(
            rows,
            columns,
            values,
            n,
            self.gb_obj[0]))
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
            raise ValueError(f'`rows` and `columns` and `values` lengths must match: '
                             f'{len(rows)}, {len(columns)}, {len(values)}')
        if clear:
            self.clear()
        if n <= 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        if dup_op.opclass != 'BinaryOp':
            raise TypeError(f'dup_op must be BinaryOp')
        rows = ffi.new('GrB_Index[]', rows)
        columns = ffi.new('GrB_Index[]', columns)
        values = ffi.new(f'{self.dtype.c_type}[]', values)
        # Push values into w
        func = libget(f'GrB_Matrix_build_{self.dtype.name}')
        check_status(func(
            self.gb_obj[0],
            rows,
            columns,
            values,
            n,
            dup_op.gb_obj))
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self.nvals < len(values):
            raise ValueError('Duplicate indices found, must provide `dup_op` BinaryOp')

    def dup(self, *, dtype=None, mask=None):
        """
        GrB_Matrix_dup
        Create a new Matrix by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            new_mat = self.__class__.new(dtype, nrows=self.nrows, ncols=self.ncols)
            new_mat(mask=mask)[:, :] << self
            return new_mat
        new_mat = ffi.new('GrB_Matrix*')
        check_status(lib.GrB_Matrix_dup(new_mat, self.gb_obj[0]))
        return self.__class__(new_mat, self.dtype)

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0):
        """
        GrB_Matrix_new
        Create a new empty Matrix from the given type, number of rows, and number of columns
        """
        new_matrix = ffi.new('GrB_Matrix*')
        dtype = dtypes.lookup(dtype)
        check_status(lib.GrB_Matrix_new(new_matrix, dtype.gb_type, nrows, ncols))
        return cls(new_matrix, dtype)

    @classmethod
    def from_values(cls, rows, columns, values, *, nrows=None, ncols=None, dup_op=None, dtype=None):
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
                raise ValueError('No values provided. Unable to determine type.')
            # Find dtype from any of the values (assumption is they are the same type)
            dtype = type(values[0])
        dtype = dtypes.lookup(dtype)
        # Compute nrows and ncols if not provided
        if nrows is None:
            if not rows:
                raise ValueError('No row indices provided. Unable to infer nrows.')
            nrows = max(rows) + 1
        if ncols is None:
            if not columns:
                raise ValueError('No column indices provided. Unable to infer ncols.')
            ncols = max(columns) + 1
        # Create the new matrix
        C = cls.new(dtype, nrows, ncols)
        # Add the data
        C.build(rows, columns, values, dup_op=dup_op)
        return C

    #########################################################
    # Delayed methods
    #
    # These return a GbDelayed object which must be passed
    # to __setitem__ to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        """
        GrB_eWiseAdd_Matrix

        Result will contain the union of indices from both Matrices
        Default op is monoid.plus
        Unless explicitly disabled, this method requires a monoid (directly or from a semiring).
            The reason for this is that binary operators can create very confusing behavior when only
            one of the two elements is present.
            Examples: binary.minus where left=Missing and right=4 yields 4 rather than -4 as might be expected
                      binary.gt where left=Missing and right=4 yields True
                      binary.gt where left=Missing and right=0 yields False
            The behavior is caused by grabbing the non-empty value and using it directly without performing
            any operation. In the case of `gt`, the non-empty value is cast to a boolean.
            For these reasons, users are required to be explicit when choosing this surprising behavior.
        """
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        op = get_typed_op(op, self.dtype, other.dtype)
        if op.opclass not in {'BinaryOp', 'Monoid', 'Semiring'}:
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if require_monoid and op.opclass not in {'Monoid', 'Semiring'}:
            raise TypeError(f'op must be Monoid or Semiring unless require_monoid is False')
        func = libget(f'GrB_eWiseAdd_Matrix_{op.opclass}')
        output_constructor = partial(Matrix.new,
                                     dtype=op.return_type,
                                     nrows=self.nrows, ncols=self.ncols)
        return GbDelayed(func,
                         [op.gb_obj, self.gb_obj[0], other.gb_obj[0]],
                         at=self._is_transposed,
                         bt=other._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, other, op))

    def ewise_mult(self, other, op=binary.times):
        """
        GrB_eWiseMult_Matrix

        Result will contain the intersection of indices from both Matrices
        Default op is binary.times
        """
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        op = get_typed_op(op, self.dtype, other.dtype)
        if op.opclass not in {'BinaryOp', 'Monoid', 'Semiring'}:
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = libget(f'GrB_eWiseMult_Matrix_{op.opclass}')
        output_constructor = partial(Matrix.new,
                                     dtype=op.return_type,
                                     nrows=self.nrows, ncols=self.ncols)
        return GbDelayed(func,
                         [op.gb_obj, self.gb_obj[0], other.gb_obj[0]],
                         at=self._is_transposed,
                         bt=other._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, other, op))

    def mxv(self, other, op=semiring.plus_times):
        """
        GrB_mxv
        Matrix-Vector multiplication. Result is a Vector.
        Default op is semiring.plus_times
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        op = get_typed_op(op, self.dtype, other.dtype)
        if op.opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        output_constructor = partial(Vector.new,
                                     dtype=op.return_type,
                                     size=self.nrows)
        return GbDelayed(lib.GrB_mxv,
                         [op.gb_obj, self.gb_obj[0], other.gb_obj[0]],
                         at=self._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, other, op))

    def mxm(self, other, op=semiring.plus_times):
        """
        GrB_mxm
        Matrix-Matrix multiplication. Result is a Matrix.
        Default op is semiring.plus_times
        """
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Expected Matrix or Vector, found {type(other)}')
        if op is None:
            op = semiring.plus_times
        op = get_typed_op(op, self.dtype, other.dtype)
        if op.opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        output_constructor = partial(Matrix.new,
                                     dtype=op.return_type,
                                     nrows=self.nrows, ncols=other.ncols)
        return GbDelayed(lib.GrB_mxm,
                         [op.gb_obj, self.gb_obj[0], other.gb_obj[0]],
                         at=self._is_transposed,
                         bt=other._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, other, op))

    def kronecker(self, other, op=binary.times):
        """
        GrB_kronecker
        Kronecker product or sum (depending on op used)
        Default op is binary.times
        """
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        op = get_typed_op(op, self.dtype, other.dtype)
        if op.opclass not in {'BinaryOp', 'Monoid', 'Semiring'}:
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = libget(f'GrB_Matrix_kronecker_{op.opclass}')
        output_constructor = partial(Matrix.new,
                                     dtype=op.return_type,
                                     nrows=self.nrows*other.nrows, ncols=self.ncols*other.ncols)
        return GbDelayed(func,
                         [op.gb_obj, self.gb_obj[0], other.gb_obj[0]],
                         at=self._is_transposed,
                         bt=other._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, other, op))

    def apply(self, op, left=None, right=None):
        """
        GrB_Matrix_apply
        Apply UnaryOp to each element of the calling Matrix
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        # This doesn't yet take into account the dtype of left or right (if provided)
        op = get_typed_op(op, self.dtype)
        if op.opclass == 'UnaryOp':
            if left is not None or right is not None:
                raise TypeError('Cannot provide `left` or `right` for a UnaryOp')
        elif op.opclass == 'BinaryOp':
            if left is None and right is None:
                raise TypeError('Must provide either `left` or `right` for a BinaryOp')
            elif left is not None and right is not None:
                raise TypeError('Cannot provide both `left` and `right`')
        else:
            raise TypeError('apply only accepts UnaryOp or BinaryOp')
        output_constructor = partial(Matrix.new,
                                     dtype=op.return_type,
                                     nrows=self.nrows, ncols=self.ncols)
        if op.opclass == 'UnaryOp':
            func = lib.GrB_Matrix_apply
            call_args = [op.gb_obj, self.gb_obj[0]]
        else:
            if left is not None:
                if isinstance(left, Scalar):
                    dtype = left.dtype
                    left = left.value
                else:
                    dtype = dtypes.lookup(type(left))
                func = libget(f'GrB_Matrix_apply_BinaryOp1st_{dtype}')
                call_args = [op.gb_obj, ffi.cast(dtype.c_type, left), self.gb_obj[0]]
            elif right is not None:
                if isinstance(right, Scalar):
                    dtype = right.dtype
                    right = right.value
                else:
                    dtype = dtypes.lookup(type(right))
                func = libget(f'GrB_Matrix_apply_BinaryOp2nd_{dtype}')
                call_args = [op.gb_obj, self.gb_obj[0], ffi.cast(dtype.c_type, right)]

        return GbDelayed(func,
                         call_args,
                         at=self._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, op))

    def reduce_rows(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each row, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        op = get_typed_op(op, self.dtype)
        if op.opclass not in {'BinaryOp', 'Monoid'}:
            raise TypeError(f'op must be BinaryOp or Monoid')
        func = libget(f'GrB_Matrix_reduce_{op.opclass}')
        output_constructor = partial(Vector.new,
                                     dtype=op.return_type,
                                     size=self.nrows)
        return GbDelayed(func,
                         [op.gb_obj, self.gb_obj[0]],
                         at=self._is_transposed,
                         output_constructor=output_constructor,
                         objects=(self, op))

    def reduce_columns(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        return self.T.reduce_rows(op)

    def reduce_scalar(self, op=monoid.plus):
        """
        GrB_Matrix_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        op = get_typed_op(op, self.dtype)
        if op.opclass != 'Monoid':
            raise TypeError(f'op must be Monoid')
        func = libget(f'GrB_Matrix_reduce_{op.return_type}')
        output_constructor = partial(Scalar.new,
                                     dtype=op.return_type)
        return GbDelayed(func,
                         [op.gb_obj, self.gb_obj[0]],
                         output_constructor=output_constructor,
                         objects=(self, op))

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        func = libget(f'GrB_Matrix_extractElement_{self.dtype}')
        result = ffi.new(f'{self.dtype.c_type}*')
        if self._is_transposed:
            row, col = col, row

        err_code = func(result,
                        self.gb_obj[0],
                        row,
                        col)
        # Don't raise error for no value, simply return `None`
        if is_error(err_code, NoValue):
            return None, self.dtype
        check_status(err_code)
        return result[0], self.dtype

    def _prep_for_extract(self, resolved_indexes):
        rows, rowsize = resolved_indexes.indices[0]
        cols, colsize = resolved_indexes.indices[1]
        if rowsize is None:
            # Row-only selection; GraphBLAS doesn't have this method, so we hack it using transpose
            row_index = rows
            output_constructor = partial(Vector.new,
                                         dtype=self.dtype,
                                         size=colsize)
            return GbDelayed(lib.GrB_Col_extract,
                             [self.gb_obj[0], cols, colsize, row_index],
                             at=(not self._is_transposed),
                             output_constructor=output_constructor,
                             objects=self)
        elif colsize is None:
            # Column-only selection
            col_index = cols
            output_constructor = partial(Vector.new,
                                         dtype=self.dtype,
                                         size=rowsize)
            return GbDelayed(lib.GrB_Col_extract,
                             [self.gb_obj[0], rows, rowsize, col_index],
                             at=self._is_transposed,
                             output_constructor=output_constructor,
                             objects=self)
        else:
            output_constructor = partial(Matrix.new,
                                         dtype=self.dtype,
                                         nrows=rowsize, ncols=colsize)
            return GbDelayed(lib.GrB_Matrix_extract,
                             [self.gb_obj[0], rows, rowsize, cols, colsize],
                             at=self._is_transposed,
                             output_constructor=output_constructor,
                             objects=self)

    def _assign_element(self, resolved_indexes, value):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        func = libget(f'GrB_Matrix_setElement_{self.dtype}')
        check_status(func(
                     self.gb_obj[0],
                     ffi.cast(self.dtype.c_type, value),
                     row,
                     col))

    def _prep_for_assign(self, resolved_indexes, obj):
        rows, rowsize = resolved_indexes.indices[0]
        cols, colsize = resolved_indexes.indices[1]
        if isinstance(obj, Scalar):
            obj = obj.value
        if isinstance(obj, (int, float, bool, complex)):
            if rowsize is None:
                rows = [rows]
                rowsize = 1
            if colsize is None:
                cols = [cols]
                colsize = 1
            dtype = self.dtype
            scalar = ffi.cast(dtype.c_type, obj)
            func = libget(f'GrB_Matrix_assign_{dtype.name}')
            delayed = GbDelayed(func,
                                [scalar, rows, rowsize, cols, colsize],
                                objects=self)
        else:
            if rowsize is None and colsize is None:
                raise TypeError(f'Expected scalar for assignment value; found {type(obj)}')
            elif rowsize is None:
                if not isinstance(obj, Vector):
                    raise TypeError(f'Expected Vector for assignment value; found {type(obj)}')
                # Row-only selection
                row_index = rows
                delayed = GbDelayed(lib.GrB_Row_assign,
                                    [obj.gb_obj[0], row_index, cols, colsize],
                                    objects=(self, obj))
            elif colsize is None:
                if not isinstance(obj, Vector):
                    raise TypeError(f'Expected Vector for assignment value; found {type(obj)}')
                # Column-only selection
                col_index = cols
                delayed = GbDelayed(lib.GrB_Col_assign,
                                    [obj.gb_obj[0], rows, rowsize, col_index],
                                    objects=(self, obj))
            else:
                if not isinstance(obj, (Matrix, TransposedMatrix)):
                    raise TypeError(f'Expected Matrix for assignment value; found {type(obj)}')
                delayed = GbDelayed(lib.GrB_Matrix_assign,
                                    [obj.gb_obj[0], rows, rowsize, cols, colsize],
                                    at=obj._is_transposed,
                                    objects=(self, obj))
        return delayed

    def _delete_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        check_status(lib.GrB_Matrix_removeElement(
                     self.gb_obj[0],
                     row,
                     col))


class TransposedMatrix:
    _is_scalar = False
    _is_transposed = True

    def __init__(self, matrix):
        self._matrix = matrix

    def __repr__(self):
        return f'<Matrix.T {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

    def new(self, *, dtype=None, mask=None):
        if dtype is None:
            dtype = self.dtype
        output = Matrix.new(dtype, self.nrows, self.ncols)
        if mask is None:
            output.update(self)
        else:
            output(mask).update(self)
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
    show = Matrix.show
    _extract_element = Matrix._extract_element
    _prep_for_extract = Matrix._prep_for_extract
    __getitem__ = Matrix.__getitem__

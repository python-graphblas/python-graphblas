from functools import partial
from .base import lib, ffi, GbContainer, GbDelayed
from .vector import Vector
from .scalar import Scalar
from .ops import BinaryOp, find_opclass, find_return_type, reify_op
from . import dtypes, unary, binary, monoid, semiring
from .exceptions import check_status, is_error, NoValue


class Matrix(GbContainer):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """
    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)

    def __del__(self):
        check_status(lib.GrB_Matrix_free(self.gb_obj))

    def __repr__(self):
        return f'<Matrix {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

    def isequal(self, other, *, check_dtype=False):
        """
        Check for exact equality (same size, same empty values)
        If `check_dtype` is True, also checks that dtypes match
        For equality of floating point Vectors, consider using `isclose`
        """
        if type(other) is not self.__class__:
            return False
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

        matches = Matrix.new_from_type(bool, self.nrows, self.ncols)
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
            return False

        # Check if all results are True
        result = Scalar.new_from_type(bool)
        result << matches.reduce_scalar(monoid.land)
        return result.value

    def isclose(self, other, rtol=1e-7, atol=0.0, *, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rtol * max(abs(a), abs(b)), atol)`
        """
        if type(other) is not self.__class__:
            return False
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

        matches = Matrix.new_from_type(bool, self.nrows, self.ncols)
        tmp1 = self.apply(unary.abs).new(dtype=common_dtype)
        tmp2 = other.apply(unary.abs).new(dtype=common_dtype)
        tmp1 << tmp1.ewise_mult(tmp2, monoid.max)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if tmp1.nvals != self.nvals:
            return False
        tmp1[:, :](mask=tmp1, accum=binary.times) << rtol
        tmp1[:, :](mask=tmp1, accum=binary.max) << atol
        tmp2 << self.ewise_mult(other, binary.minus)
        tmp2 << tmp2.apply(unary.abs)
        matches << tmp2.ewise_mult(tmp1, binary.le[common_dtype])

        # Check if all results are True
        result = Scalar.new_from_type(bool)
        result << matches.reduce_scalar(monoid.land)
        return result.value

    def __len__(self):
        return self.nvals

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

    @property
    def is_transposed(self):
        return False

    def clear(self):
        check_status(lib.GrB_Matrix_clear(self.gb_obj[0]))

    def resize(self, nrows, ncols):
        raise NotImplementedError('Not implemented in GraphBLAS 1.2')
        check_status(lib.GxB_Matrix_resize(
            self.gb_obj[0],
            nrows,
            ncols))

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
        func = getattr(lib, f'GrB_Matrix_extractTuples_{self.dtype.name}')
        check_status(func(
            rows,
            columns,
            values,
            n,
            self.gb_obj[0]))
        return tuple(rows), tuple(columns), tuple(values)

    def rebuild_from_values(self, rows, columns, values, *, dup_op=None):
        # TODO: add `size` option once .resize is available
        self.clear()
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
        if n <= 0:
            return
        dup_orig = dup_op
        if dup_op is None:
            dup_op = binary.plus
        if isinstance(dup_op, BinaryOp):
            dup_op = dup_op[self.dtype]
        rows = ffi.new('GrB_Index[]', rows)
        columns = ffi.new('GrB_Index[]', columns)
        values = ffi.new(f'{self.dtype.c_type}[]', values)
        # Push values into w
        func = getattr(lib, f'GrB_Matrix_build_{self.dtype.name}')
        check_status(func(
            self.gb_obj[0],
            rows,
            columns,
            values,
            n,
            dup_op))
        # Check for duplicates when dup_op was not provided
        if dup_orig is None and self.nvals < len(values):
            raise ValueError('Duplicate indices found, must provide `dup_op` BinaryOp')

    @classmethod
    def new_from_type(cls, dtype, nrows=0, ncols=0):
        """
        GrB_Matrix_new
        Create a new empty Matrix from the given type, number of rows, and number of columns
        """
        new_matrix = ffi.new('GrB_Matrix*')
        dtype = dtypes.lookup(dtype)
        check_status(lib.GrB_Matrix_new(new_matrix, dtype.gb_type, nrows, ncols))
        return cls(new_matrix, dtype)

    @classmethod
    def new_from_existing(cls, matrix):
        """
        GrB_Matrix_dup
        Create a new Matrix by duplicating an existing one
        """
        new_mat = ffi.new('GrB_Matrix*')
        check_status(lib.GrB_Matrix_dup(new_mat, matrix.gb_obj[0]))
        return cls(new_mat, matrix.dtype)

    @classmethod
    def new_from_values(cls, rows, columns, values, *, nrows=None, ncols=None, dup_op=None, dtype=None):
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
        C = cls.new_from_type(dtype, nrows, ncols)
        # Add the data
        C.rebuild_from_values(rows, columns, values, dup_op=dup_op)
        return C

    #########################################################
    # Delayed methods
    #
    # These return a GbDelayed object which must be passed
    # to __setitem__ to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=None, *, require_monoid=True):
        """
        GrB_eWiseAdd_Matrix

        Result will contain the union of indices from both Matrices
        Default op is binary.plus
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
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is None:
            op = binary.plus
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if require_monoid and opclass not in {'Monoid', 'Semiring'}:
            raise TypeError(f'op must be Monoid or Semiring unless require_monoid is False')
        func = getattr(lib, f'GrB_eWiseAdd_Matrix_{opclass}')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Matrix.new_from_type,
                                     dtype=find_return_type(op),
                                     nrows=self.nrows, ncols=self.ncols)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed,
                         output_constructor=output_constructor)

    def ewise_mult(self, other, op=None):
        """
        GrB_eWiseMult_Matrix

        Result will contain the intersection of indices from both Matrices
        Default op is binary.times
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is None:
            op = binary.times
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = getattr(lib, f'GrB_eWiseMult_Matrix_{opclass}')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Matrix.new_from_type,
                                     dtype=find_return_type(op),
                                     nrows=self.nrows, ncols=self.ncols)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed,
                         output_constructor=output_constructor)

    def mxv(self, other, op=None):
        """
        GrB_mxv
        Matrix-Vector multiplication. Result is a Vector.
        Default op is semiring.plus_times
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is None:
            op = semiring.plus_times
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op),
                                     size=self.nrows)
        return GbDelayed(lib.GrB_mxv,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         output_constructor=output_constructor)

    def mxm(self, other, op=None):
        """
        GrB_mxm
        Matrix-Matrix multiplication. Result is a Matrix.
        Default op is semiring.plus_times
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix or Vector, found {type(other)}')
        if op is None:
            op = semiring.plus_times
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Matrix.new_from_type,
                                     dtype=find_return_type(op),
                                     nrows=self.nrows, ncols=other.ncols)
        return GbDelayed(lib.GrB_mxm,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed,
                         output_constructor=output_constructor)

    def kronecker(self, other, op=None):
        """
        GrB_kronecker
        Kronecker product or sum (depending on op used)
        Default op is binary.times
        """
        raise NotImplementedError('Not available in GraphBLAS 1.2')
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is None:
            op = binary.times
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = getattr(lib, f'GrB_kronecker_{opclass}')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Matrix.new_from_type,
                                     dtype=find_return_type(op),
                                     nrows=self.nrows*other.nrows, ncols=self.ncols*other.ncols)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed,
                         output_constructor=output_constructor)

    def apply(self, op, left=None, right=None):
        """
        GrB_Matrix_apply
        Apply UnaryOp to each element of the calling Matrix
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        opclass = find_opclass(op)
        if opclass == 'UnaryOp':
            if left is not None or right is not None:
                raise TypeError('Cannot provide `left` or `right` for a UnaryOp')
        elif opclass == 'BinaryOp':
            if left is None and right is None:
                raise TypeError('Must provide either `left` or `right` for a BinaryOp')
            elif left is not None and right is not None:
                raise TypeError('Cannot provide both `left` and `right`')
        else:
            raise TypeError('apply only accepts UnaryOp or BinaryOp')
        op = reify_op(op, self.dtype)
        output_constructor = partial(Matrix.new_from_type,
                                     dtype=find_return_type(op),
                                     nrows=self.nrows, ncols=self.ncols)
        if opclass == 'UnaryOp':
            return GbDelayed(lib.GrB_Matrix_apply,
                             [op, self.gb_obj[0]],
                             output_constructor=output_constructor)
        else:
            raise NotImplementedError('apply with BinaryOp not available in GraphBLAS 1.2')
            # TODO: fill this in once function is available

    def reduce_rows(self, op=None):
        """
        GrB_Matrix_reduce
        Reduce all values in each row, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        if op is None:
            if self.dtype == bool:
                op = monoid.lor
            else:
                op = monoid.plus
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid'):
            raise TypeError(f'op must be BinaryOp or Monoid')
        func = getattr(lib, f'GrB_Matrix_reduce_{opclass}')
        op = reify_op(op, self.dtype)
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op),
                                     size=self.nrows)
        return GbDelayed(func,
                         [op, self.gb_obj[0]],
                         at=self.is_transposed,
                         output_constructor=output_constructor)

    def reduce_columns(self, op=None):
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        return self.T.reduce_rows(op)

    def reduce_scalar(self, op=None):
        """
        GrB_Matrix_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        if op is None:
            if self.dtype == bool:
                op = monoid.lor
            else:
                op = monoid.plus
        func = getattr(lib, f'GrB_Matrix_reduce_{self.dtype.name}')
        op = reify_op(op, self.dtype)
        output_constructor = partial(Scalar.new_from_type,
                                     dtype=find_return_type(op))
        return GbDelayed(func,
                         [op, self.gb_obj[0]],
                         output_constructor=output_constructor)

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(self, resolved_indexes):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        func = getattr(lib, f'GrB_Matrix_extractElement_{self.dtype}')
        result = ffi.new(f'{self.dtype.c_type}*')

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
            output_constructor = partial(Vector.new_from_type,
                                         dtype=self.dtype,
                                         size=colsize)
            return GbDelayed(lib.GrB_Col_extract,
                             [self.gb_obj[0], cols, colsize, row_index],
                             at=(not self.is_transposed),
                             output_constructor=output_constructor)
        elif colsize is None:
            # Column-only selection
            col_index = cols
            output_constructor = partial(Vector.new_from_type,
                                         dtype=self.dtype,
                                         size=rowsize)
            return GbDelayed(lib.GrB_Col_extract,
                             [self.gb_obj[0], rows, rowsize, col_index],
                             at=self.is_transposed,
                             output_constructor=output_constructor)
        else:
            output_constructor = partial(Matrix.new_from_type,
                                         dtype=self.dtype,
                                         nrows=rowsize, ncols=colsize)
            return GbDelayed(lib.GrB_Matrix_extract,
                             [self.gb_obj[0], rows, rowsize, cols, colsize],
                             at=self.is_transposed,
                             output_constructor=output_constructor)

    def _assign_element(self, resolved_indexes, value):
        row, _ = resolved_indexes.indices[0]
        col, _ = resolved_indexes.indices[1]
        func = getattr(lib, f'GrB_Matrix_setElement_{self.dtype}')
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

        if isinstance(obj, (int, float, bool)):
            if rowsize is None:
                rows = [rows]
                rowsize = 1
            if colsize is None:
                cols = [cols]
                colsize = 1
            dtype = self.dtype
            scalar = ffi.cast(dtype.c_type, obj)
            func = getattr(lib, f'GrB_Matrix_assign_{dtype.name}')
            delayed = GbDelayed(func,
                                [scalar, rows, rowsize, cols, colsize])
        else:
            if rowsize is None and colsize is None:
                raise TypeError(f'Expected scalar for assignment value; found {type(obj)}')
            elif rowsize is None:
                if not isinstance(obj, Vector):
                    raise TypeError(f'Expected Vector for assignment value; found {type(obj)}')
                # Row-only selection
                row_index = rows
                delayed = GbDelayed(lib.GrB_Row_assign,
                                    [obj.gb_obj[0], row_index, cols, colsize])
            elif colsize is None:
                if not isinstance(obj, Vector):
                    raise TypeError(f'Expected Vector for assignment value; found {type(obj)}')
                # Column-only selection
                col_index = cols
                delayed = GbDelayed(lib.GrB_Col_assign,
                                    [obj.gb_obj[0], rows, rowsize, col_index])
            else:
                if not isinstance(obj, Matrix):
                    raise TypeError(f'Expected Matrix for assignment value; found {type(obj)}')
                delayed = GbDelayed(lib.GrB_Matrix_assign,
                                    [obj.gb_obj[0], rows, rowsize, cols, colsize],
                                    at=obj.is_transposed)

        return delayed


class TransposedMatrix(Matrix):
    def __init__(self, matrix):
        super().__init__(matrix.gb_obj, matrix.dtype)
        self._matrix = matrix

    # Override the default behavior. Don't free gb_obj
    # because it's shared with the untransposed matrix
    def __del__(self):
        pass

    def __repr__(self):
        return f'<Matrix.T {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

    def new(self, mask=None):
        output = Matrix.new_from_type(self.dtype, self.nrows, self.ncols)
        if mask is None:
            output.update(self)
        else:
            if type(mask) is not Matrix:
                raise TypeError('Mask must be a Matrix')
            output(mask).update(self)
        return output

    @property
    def nrows(self):
        return super().ncols

    @property
    def ncols(self):
        return super().nrows

    @property
    def T(self):
        return self._matrix

    @property
    def is_transposed(self):
        return True

    def clear(self):
        raise Exception('Modification of a transposed Matrix is not allowed')

    def resize(self, nrows, ncols):
        raise Exception('Modification of a transposed Matrix is not allowed')

    def rebuild_from_values(self, rows, columns, values, *, dup_op=None):
        raise Exception('Modification of a transposed Matrix is not allowed')

    def __setitem__(self, key, val):
        raise Exception('Assignment to a transposed Matrix is not allowed')

    def to_values(self):
        rows, cols, vals = super().to_values()
        return cols, rows, vals

import types
from .base import lib, ffi, NULL, GbContainer, GbDelayed
from .vector import Vector
from .scalar import Scalar
from .ops import OpBase, UnaryOp, BinaryOp, Monoid, Semiring, find_opclass, build_udf, free_udf
from . import dtypes
from .exceptions import check_status, is_error, NoValue


class Matrix(GbContainer):
    """
    GraphBLAS Sparse Matrix
    High-level wrapper around GrB_Matrix type
    """
    can_mask = True

    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)
        self.element = Matrix.ElementManipulator(self)
        self.extract = Matrix.Extractor(self)
        self.assign = Matrix.Assigner(self)

    def __del__(self):
        check_status(lib.GrB_Matrix_free(self.gb_obj))
        free_udf(self)
    
    def __repr__(self):
        return f'<Matrix {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

    def __eq__(self, other):
        # Borrowed this recipe from LAGraph
        if type(other) != self.__class__:
            return False
        if self.dtype != other.dtype:
            return False
        if self.nrows != other.nrows:
            return False
        if self.ncols != other.ncols:
            return False
        if self.nvals != other.nvals:
            return False
        # Use ewise_mult to compare equality via intersection
        matches = Matrix.new_from_type(bool, self.nrows, self.ncols)
        matches[:] = self.ewise_mult(other, BinaryOp.EQ)
        if matches.nvals != self.nvals:
            return False
        # Check if all results are True
        result = Scalar.new_from_type(bool)
        result[:] = matches.reduce_scalar(Monoid.LAND)
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
        return iter(rows), iter(columns), iter(values)

    def rebuild_from_values(self, rows, columns, values, *, dup_op=NULL):
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
        if dup_op is NULL:
            dup_op = BinaryOp.PLUS
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
        if dup_orig is NULL and self.nvals < len(values):
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
    def new_from_values(cls, rows, columns, values, *, nrows=None, ncols=None, dup_op=NULL):
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
        if len(values) <= 0:
            raise ValueError('No values provided. Unable to determine type.')
        # Find dtype from any of the values (assumption is they are the same type)
        dtype = dtypes.lookup(type(values[0]))
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

    def ewise_add(self, other, op=NULL):
        """
        GrB_eWiseAdd_Matrix

        Result will contain the union of indices from both Matrices
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is NULL:
            op = BinaryOp.PLUS
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        func = getattr(lib, f'GrB_eWiseAdd_Matrix_{opclass}')
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed)

    def ewise_mult(self, other, op=NULL):
        """
        GrB_eWiseMult_Matrix

        Result will contain the intersection of indices from both Matrices
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is NULL:
            op = BinaryOp.TIMES
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        func = getattr(lib, f'GrB_eWiseMult_Matrix_{opclass}')
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed)

    def mxv(self, other, op=NULL):
        """
        GrB_mxv
        Matrix-Vector multiplication. Result is a Vector.
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is NULL:
            op = Semiring.PLUS_TIMES
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        return GbDelayed(lib.GrB_mxv,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed)

    def mxm(self, other, op=NULL):
        """
        GrB_mxm
        Matrix-Matrix multiplication. Result is a Matrix.
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix or Vector, found {type(other)}')
        if op is NULL:
            op = Semiring.PLUS_TIMES
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        return GbDelayed(lib.GrB_mxm,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed)

    def kronecker(self, other, op=NULL):
        """
        GrB_kronecker
        Kronecker product or sum (depending on op used)
        """
        raise NotImplementedError('Not available in GraphBLAS 1.2')
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is NULL:
            op = BinaryOp.TIMES
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        func = getattr(lib, f'GrB_kronecker_{opclass}')
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         at=self.is_transposed,
                         bt=other.is_transposed)

    def apply(self, op, left=None, right=None):
        """
        GrB_Matrix_apply
        Apply UnaryOp to each element of the calling Matrix
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        if isinstance(op, types.FunctionType):
            op = build_udf(op, self)
            opclass = 'UnaryOp'
        else:
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

        if isinstance(op, OpBase):
            op = op[self.dtype]
        if opclass == 'UnaryOp':
            return GbDelayed(lib.GrB_Matrix_apply,
                             [op, self.gb_obj[0]])
        else:
            raise NotImplementedError('apply with BinaryOp not available in GraphBLAS 1.2')
            # TODO: fill this in once function is available

    def reduce_rows(self, op=NULL):
        """
        GrB_Matrix_reduce
        Reduce all values in each row, converting the matrix to a vector
        """
        if op is NULL:
            if self.dtype == bool:
                op = Monoid.LOR
            else:
                op = Monoid.PLUS
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid'):
            raise TypeError(f'op must be BinaryOp or Monoid')
        if isinstance(op, OpBase):
            op = op[self.dtype]
        func = getattr(lib, f'GrB_Matrix_reduce_{opclass}')
        return GbDelayed(func,
                         [op, self.gb_obj[0]],
                         at=self.is_transposed)
    
    def reduce_columns(self, op=NULL):
        """
        GrB_Matrix_reduce
        Reduce all values in each column, converting the matrix to a vector
        """
        return self.T.reduce_rows(op)
    
    def reduce_scalar(self, op=NULL):
        """
        GrB_Matrix_reduce
        Reduce all values into a scalar
        """
        if op is NULL:
            if self.dtype == bool:
                op = Monoid.LOR
            else:
                op = Monoid.PLUS
        if isinstance(op, Monoid):
            op = op[self.dtype]
        func = getattr(lib, f'GrB_Matrix_reduce_{self.dtype.name}')
        return GbDelayed(func,
                        [op, self.gb_obj[0]])


    class ElementManipulator:
        def __init__(self, matrix):
            self._matrix = matrix
        
        def __repr__(self):
            return 'MatrixElementManipulator'
        
        def _parse_index(self, index):
            if (
                not isinstance(index, tuple)
                or len(index) != 2
                or not isinstance(index[0], int)
                or not isinstance(index[1], int)
            ):
                raise TypeError('Index must be a 2-tuple of ints')
            row, col = index
            shape = self._matrix.shape
            if row >= shape[0]:
                raise IndexError(f'row_index={row}, nrows={shape[0]}')
            if col >= shape[1]:
                raise IndexError(f'col_index={col}, ncols={shape[1]}')
            return row, col

        def __getitem__(self, index):
            row, col = self._parse_index(index)
            mat = self._matrix
            func = getattr(lib, f'GrB_Matrix_extractElement_{mat.dtype}')
            result = ffi.new(f'{mat.dtype.c_type}*')
            err_code = func(result,
                            mat.gb_obj[0],
                            row,
                            col)
            # If no value, return Python `None`
            if is_error(err_code, NoValue):
                return None
            check_status(err_code)
            return result[0]
        
        def __setitem__(self, index, value):
            row, col = self._parse_index(index)
            mat = self._matrix
            func = getattr(lib, f'GrB_Matrix_setElement_{mat.dtype}')
            check_status(func(
                mat.gb_obj[0],
                ffi.cast(mat.dtype.c_type, value),
                row,
                col))

        def __delitem__(self, index):
            row, col = self._parse_index(index)
            raise NotImplementedError('Not available in GraphBLAS 1.2')

    class _Indexer:
        def _parse_indices(self, indices):
            """
            Returns rows, rowsize, cols, colsize
            For row-only, rowsize=None and type(rows)==int
            For col-only, colsize=None and type(cols)==int
            """
            if type(indices) != tuple or len(indices) != 2:
                raise TypeError('Index must be a 2-tuple')
            rows, cols = indices
            rtyp, ctyp = type(rows), type(cols)
            
            if rtyp == int and ctyp == int:
                # Single index
                return rows, None, cols, None
            if rtyp == tuple or ctyp == tuple:
                raise TypeError(f'{self} cannot accept a tuple as index; use slice or list')

            rows, rowsize = self._parse_index(rows, rtyp, self._matrix.nrows)
            cols, colsize = self._parse_index(cols, ctyp, self._matrix.ncols)
            return rows, rowsize, cols, colsize

        def _parse_index(self, index, typ, size):
            if typ == int:
                return index, None
            if typ == slice:
                if index == slice(None):
                    # [:] means all indices; use special GrB_ALL indicator
                    return lib.GrB_ALL, size
                index = tuple(range(size)[index])
            elif typ != list:
                try:
                    index = tuple(index)
                except Exception:
                    raise TypeError()
            return ffi.new('GrB_Index[]', index), len(index)

    class Extractor(_Indexer):
        def __init__(self, matrix):
            self._matrix = matrix
        
        def __repr__(self):
            return 'MatrixExtractor'
        
        def __getitem__(self, indices):
            rows, rowsize, cols, colsize = self._parse_indices(indices)
            if rowsize is None and colsize is None:
                raise TypeError('Use A.element[i, j] to get a single element')
            if rowsize is None:
                # Row-only selection; GraphBLAS doesn't have this method, so we hack it using transpose
                row_index = rows
                return GbDelayed(lib.GrB_Col_extract,
                                 [self._matrix.gb_obj[0], cols, colsize, row_index],
                                 at=(not self._matrix.is_transposed))
            elif colsize is None:
                # Column-only selection
                col_index = cols
                return GbDelayed(lib.GrB_Col_extract,
                                 [self._matrix.gb_obj[0], rows, rowsize, col_index],
                                 at=self._matrix.is_transposed)
            else:
                return GbDelayed(lib.GrB_Matrix_extract,
                                 [self._matrix.gb_obj[0], rows, rowsize, cols, colsize],
                                 at=self._matrix.is_transposed)
    
    class Assigner(_Indexer):
        def __init__(self, matrix):
            self._matrix = matrix
        
        def __repr__(self):
            return 'MatrixAssigner'
        
        def __setitem__(self, keys, other):
            # Note: keys contains assignment rows, cols, mask, accum, and REPLACE
            #       rows, cols must always come first (if given, otherwise assumes all indexes)
            #       C.assign[:] will be interpreted as all indexes with no mask
            if type(keys) != tuple:
                keys = (keys,)
            if len(keys) >= 2:
                try:
                    # Try to parse the first two items as the row/col indexes
                    rows, rowsize, cols, colsize = self._parse_indices(keys[:2])
                    keys = keys[2:]
                except TypeError:
                    # No index given; use the default
                    rows, rowsize, cols, colsize = self._parse_indices((slice(None), slice(None)))
            else:
                # No index given; use the default
                rows, rowsize, cols, colsize = self._parse_indices((slice(None), slice(None)))
            if not keys or (len(keys) == 1 and keys[0] == slice(None)):
                # No keys given; use the default
                keys = slice(None)
            else:
                for key in keys:
                    if type(key) in (list, slice):
                        raise TypeError('Assignment indexes for rows and columns must come first')

            if isinstance(other, (int, float, bool)):
                if rowsize is None:
                    rows = [rows]
                    rowsize = 1
                if colsize is None:
                    cols = [cols]
                    colsize = 1
                dtype = self._matrix.dtype
                scalar = ffi.cast(dtype.c_type, other)
                func = getattr(lib, f'GrB_Matrix_assign_{dtype.name}')
                dval = GbDelayed(func,
                                 [scalar, rows, rowsize, cols, colsize])
            else:
                if rowsize is None and colsize is None:
                    raise TypeError(f'Expected scalar for assignment value; found {type(other)}')
                elif rowsize is None:
                    if not isinstance(other, Vector):
                        raise TypeError(f'Expected Vector for assignment value; found {type(other)}')
                    # Row-only selection
                    row_index = rows
                    dval = GbDelayed(lib.GrB_Row_assign,
                                    [other.gb_obj[0], row_index, cols, colsize])
                elif colsize is None:
                    if not isinstance(other, Vector):
                        raise TypeError(f'Expected Vector for assignment value; found {type(other)}')
                    # Column-only selection
                    col_index = cols
                    dval = GbDelayed(lib.GrB_Col_assign,
                                    [other.gb_obj[0], rows, rowsize, col_index])
                else:
                    if not isinstance(other, Matrix):
                        raise TypeError(f'Expected Matrix for assignment value; found {type(other)}')
                    dval = GbDelayed(lib.GrB_Matrix_assign,
                                        [other.gb_obj[0], rows, rowsize, cols, colsize],
                                        at=other.is_transposed)
            # Forward the __setitem__ call so it is resolved with mask and accum
            self._matrix[keys] = dval

class TransposedMatrix(Matrix):
    def __init__(self, matrix):
        super().__init__(matrix.gb_obj, matrix.dtype)
        self._matrix = matrix
        # Remove items that aren't allowed to be accessed post-transpose
        del self.element
        del self.assign
    
    # Override the default behavior. Don't free gb_obj 
    # because it's shared with the untransposed matrix
    def __del__(self):
        pass

    def __repr__(self):
        return f'<Matrix.T {self.nvals}/({self.nrows}x{self.ncols}):{self.dtype.name}>'

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

    def rebuild_from_values(self, rows, columns, values, *, dup_op=NULL):
        raise Exception('Modification of a transposed Matrix is not allowed')

    def __setitem__(self, key, val):
        raise Exception('Assignment to a transposed Matrix is not allowed')

    def to_values(self):
        rows, cols, vals = super().to_values()
        return cols, rows, vals

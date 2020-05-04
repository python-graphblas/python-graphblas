from functools import lru_cache, partial
from .base import lib, ffi, GbContainer, GbDelayed
from .scalar import Scalar
from .ops import BinaryOp, find_opclass, find_return_type, reify_op
from . import dtypes, binary, monoid, semiring
from .exceptions import check_status, is_error, NoValue


@lru_cache(maxsize=1024)
def _generate_isclose(rel_tol, abs_tol):
    # numba will inline the current values of `rel_tol` and `abs_tol` below
    def isclose(x, y):
        return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
    return BinaryOp.register_anonymous(isclose, f'isclose(rel_tol={rel_tol:g}, abs_tol={abs_tol:g})')


class Vector(GbContainer):
    """
    GraphBLAS Sparse Vector
    High-level wrapper around GrB_Vector type
    """
    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)

    def __del__(self):
        check_status(lib.GrB_Vector_free(self.gb_obj))

    def __repr__(self):
        return f'<Vector {self.nvals}/{self.size}:{self.dtype.name}>'

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
        if self.size != other.size:
            return False
        if self.nvals != other.nvals:
            return False
        if check_dtype:
            # dtypes are equivalent, so not need to unify
            common_dtype = self.dtype
        else:
            common_dtype = dtypes.unify(self.dtype, other.dtype)

        matches = Vector.new(bool, self.size)
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
            return False

        # Check if all results are True
        result = Scalar.new(bool)
        result << matches.reduce(monoid.land)
        return result.value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        if type(other) is not self.__class__:
            return False
        if check_dtype and self.dtype != other.dtype:
            return False
        if self.size != other.size:
            return False
        if self.nvals != other.nvals:
            return False

        isclose = _generate_isclose(rel_tol, abs_tol)
        matches = self.ewise_mult(other, isclose).new(dtype=bool)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
            return False

        # Check if all results are True
        return matches.reduce(monoid.land).value

    def __len__(self):
        return self.nvals

    @property
    def size(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Vector_size(n, self.gb_obj[0]))
        return n[0]

    @property
    def shape(self):
        return (self.size,)

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]))
        return n[0]

    def clear(self):
        check_status(lib.GrB_Vector_clear(self.gb_obj[0]))

    def resize(self, size):
        raise NotImplementedError('Not implemented in GraphBLAS 1.2')
        check_status(lib.GrB_Vector_resize(self.gb_obj[0], size))

    def to_values(self):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as 2 generators
        """
        indices = ffi.new('GrB_Index[]', self.nvals)
        values = ffi.new(f'{self.dtype.c_type}[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = getattr(lib, f'GrB_Vector_extractTuples_{self.dtype.name}')
        check_status(func(
            indices,
            values,
            n,
            self.gb_obj[0]))
        return tuple(indices), tuple(values)

    def build(self, indices, values, *, dup_op=None, clear=False):
        # TODO: add `size` option once .resize is available
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if len(indices) != len(values):
            raise ValueError(f'`indices` and `values` have different lengths '
                             f'{len(indices)} != {len(values)}')
        if clear:
            self.clear()
        n = len(indices)
        if n <= 0:
            return
        dup_orig = dup_op
        if dup_op is None:
            dup_op = binary.plus
        if isinstance(dup_op, BinaryOp):
            dup_op = dup_op[self.dtype]
        indices = ffi.new('GrB_Index[]', indices)
        values = ffi.new(f'{self.dtype.c_type}[]', values)
        # Push values into w
        func = getattr(lib, f'GrB_Vector_build_{self.dtype.name}')
        check_status(func(
            self.gb_obj[0],
            indices,
            values,
            n,
            dup_op))
        # Check for duplicates when dup_op was not provided
        if dup_orig is None and self.nvals < len(values):
            raise ValueError('Duplicate indices found, must provide `dup_op` BinaryOp')

    def dup(self, *, dtype=None, mask=None):
        """
        GrB_Vector_dup
        Create a new Vector by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            new_vec = self.__class__.new(dtype, size=self.size)
            new_vec(mask=mask)[:] << self
            return new_vec
        new_vec = ffi.new('GrB_Vector*')
        check_status(lib.GrB_Vector_dup(new_vec, self.gb_obj[0]))
        return self.__class__(new_vec, self.dtype)

    @classmethod
    def new(cls, dtype, size=0):
        """
        GrB_Vector_new
        Create a new empty Vector from the given type and size
        """
        new_vector = ffi.new('GrB_Vector*')
        dtype = dtypes.lookup(dtype)
        check_status(lib.GrB_Vector_new(new_vector, dtype.gb_type, size))
        return cls(new_vector, dtype)

    @classmethod
    def from_values(cls, indices, values, *, size=None, dup_op=None, dtype=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.
        """
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if dtype is None:
            if len(values) <= 0:
                raise ValueError('No values provided. Unable to determine type.')
            # Find dtype from any of the values (assumption is they are the same type)
            dtype = type(values[0])
        dtype = dtypes.lookup(dtype)
        # Compute size if not provided
        if size is None:
            if not indices:
                raise ValueError('No indices provided. Unable to infer size.')
            size = max(indices) + 1
        # Create the new vector
        w = cls.new(dtype, size)
        # Add the data
        w.build(indices, values, dup_op=dup_op)
        return w

    #########################################################
    # Delayed methods
    #
    # These return a GbDelayed object which must be passed
    # to update to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=None, *, require_monoid=True):
        """
        GrB_eWiseAdd_Vector

        Result will contain the union of indices from both Vectors
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
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is None:
            op = monoid.plus
        opclass = find_opclass(op)
        if opclass not in {'BinaryOp', 'Monoid', 'Semiring'}:
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        if require_monoid and opclass not in {'Monoid', 'Semiring'}:
            raise TypeError(f'op must be Monoid or Semiring unless require_monoid is False')
        func = getattr(lib, f'GrB_eWiseAdd_Vector_{opclass}')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Vector.new,
                                     dtype=find_return_type(op),
                                     size=self.size)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         output_constructor=output_constructor)

    def ewise_mult(self, other, op=None):
        """
        GrB_eWiseMult_Vector

        Result will contain the intersection of indices from both Vectors
        Default op is binary.times
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is None:
            op = binary.times
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = getattr(lib, f'GrB_eWiseMult_Vector_{opclass}')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Vector.new,
                                     dtype=find_return_type(op),
                                     size=self.size)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         output_constructor=output_constructor)

    def vxm(self, other, op=None):
        """
        GrB_vxm
        Vector-Matrix multiplication. Result is a Vector.
        Default op is semiring.plus_times
        """
        from .matrix import Matrix, TransposedMatrix
        if not isinstance(other, (Matrix, TransposedMatrix)):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is None:
            op = semiring.plus_times
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        op = reify_op(op, self.dtype, other.dtype)
        output_constructor = partial(Vector.new,
                                     dtype=find_return_type(op),
                                     size=other.ncols)
        return GbDelayed(lib.GrB_vxm,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         bt=other._is_transposed,
                         output_constructor=output_constructor)

    def apply(self, op, left=None, right=None):
        """
        GrB_Vector_apply
        Apply UnaryOp to each element of the calling Vector
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
        output_constructor = partial(Vector.new,
                                     dtype=find_return_type(op),
                                     size=self.size)
        if opclass == 'UnaryOp':
            return GbDelayed(lib.GrB_Vector_apply,
                             [op, self.gb_obj[0]],
                             output_constructor=output_constructor)
        else:
            raise NotImplementedError('apply with BinaryOp not available in GraphBLAS 1.2')
            # TODO: fill this in once function is available

    def reduce(self, op=None):
        """
        GrB_Vector_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        if op is None:
            if self.dtype == bool:
                op = monoid.lor
            else:
                op = monoid.plus
        func = getattr(lib, f'GrB_Vector_reduce_{self.dtype.name}')
        op = reify_op(op, self.dtype)
        output_constructor = partial(Scalar.new,
                                     dtype=find_return_type(op))
        return GbDelayed(func,
                         [op, self.gb_obj[0]],
                         output_constructor=output_constructor)

    ##################################
    # Extract and Assign index methods
    ##################################
    def _extract_element(self, resolved_indexes):
        index, _ = resolved_indexes.indices[0]
        func = getattr(lib, f'GrB_Vector_extractElement_{self.dtype}')
        result = ffi.new(f'{self.dtype.c_type}*')

        err_code = func(result,
                        self.gb_obj[0],
                        index)
        # Don't raise error for no value, simply return `None`
        if is_error(err_code, NoValue):
            return None, self.dtype
        check_status(err_code)
        return result[0], self.dtype

    def _prep_for_extract(self, resolved_indexes):
        index, isize = resolved_indexes.indices[0]
        output_constructor = partial(Vector.new,
                                     dtype=self.dtype,
                                     size=isize)
        return GbDelayed(lib.GrB_Vector_extract,
                         [self.gb_obj[0], index, isize],
                         output_constructor=output_constructor)

    def _assign_element(self, resolved_indexes, value):
        index, _ = resolved_indexes.indices[0]
        func = getattr(lib, f'GrB_Vector_setElement_{self.dtype}')
        check_status(func(
                     self.gb_obj[0],
                     ffi.cast(self.dtype.c_type, value),
                     index))

    def _prep_for_assign(self, resolved_indexes, obj):
        index, isize = resolved_indexes.indices[0]

        if isinstance(obj, Scalar):
            obj = obj.value

        if isinstance(obj, (int, float, bool)):
            dtype = self.dtype
            func = getattr(lib, f'GrB_Vector_assign_{dtype.name}')
            scalar = ffi.cast(dtype.c_type, obj)
            delayed = GbDelayed(func,
                                [scalar, index, isize])
        elif isinstance(obj, Vector):
            delayed = GbDelayed(lib.GrB_Vector_assign,
                                [obj.gb_obj[0], index, isize])
        else:
            raise TypeError(f'Unexpected type for assignment value: {type(obj)}')

        return delayed

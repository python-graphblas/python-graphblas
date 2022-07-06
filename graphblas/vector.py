import itertools
import warnings

import numpy as np

from . import _automethods, backend, binary, ffi, lib, monoid, select, semiring, utils
from ._ss.vector import ss
from .base import BaseExpression, BaseType, _check_mask, call
from .dtypes import _INDEX, FP64, INT64, lookup_dtype, unify
from .exceptions import DimensionMismatch, NoValue, check_status
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
from .utils import (
    _CArray,
    _Pointer,
    class_property,
    ints_to_numpy_buffer,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)

ffi_new = ffi.new


# Custom recipes
def _v_add_m(updater, left, right, op):
    full = Vector(left.dtype, right._ncols, name="v_full")
    full[:] = 0
    temp = left.outer(full, binary.first).new(name="M_temp", mask=updater.kwargs.get("mask"))
    updater << temp.ewise_add(right, op)


def _v_mult_m(updater, left, right, op):
    updater << left.diag(name="M_temp").mxm(right, get_semiring(monoid.any, op))


def _v_union_m(updater, left, right, left_default, right_default, op):
    full = Vector(left.dtype, right._ncols, name="v_full")
    full[:] = 0
    temp = left.outer(full, binary.first).new(name="M_temp", mask=updater.kwargs.get("mask"))
    updater << temp.ewise_union(right, op, left_default=left_default, right_default=right_default)


def _reposition(updater, indices, chunk):
    updater[indices] = chunk


def _select_mask(updater, obj, mask):
    if updater.kwargs.get("mask") is None:
        orig_kwargs = updater.kwargs
        try:
            if updater.kwargs.get("accum") is None:
                updater.kwargs = dict(orig_kwargs, mask=mask, replace=True)
            else:
                updater.kwargs = dict(orig_kwargs, mask=mask)
            updater << obj
        finally:
            updater.kwargs = orig_kwargs
    else:
        # Can we do any better depending on accum, replace, and type of masks?
        updater << obj.dup(mask=mask)


class Vector(BaseType):
    """
    GraphBLAS Sparse Vector
    High-level wrapper around GrB_Vector type
    """

    __slots__ = "_size", "_parent", "ss"
    ndim = 1
    _name_counter = itertools.count()

    def __new__(cls, dtype=FP64, size=0, *, name=None):
        self = object.__new__(cls)
        self.dtype = lookup_dtype(dtype)
        size = _as_scalar(size, _INDEX, is_cscalar=True)
        self.name = f"v_{next(Vector._name_counter)}" if name is None else name
        self.gb_obj = ffi_new("GrB_Vector*")
        call("GrB_Vector_new", [_Pointer(self), self.dtype, size])
        self._size = size.value
        self._parent = None
        self.ss = ss(self)
        return self

    @classmethod
    def _from_obj(cls, gb_obj, dtype, size, *, parent=None, name=None):
        self = object.__new__(cls)
        self.name = f"v_{next(Vector._name_counter)}" if name is None else name
        self.gb_obj = gb_obj
        self.dtype = dtype
        self._size = size
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
            check_status(lib.GrB_Vector_free(gb_obj), self)

    def _as_matrix(self):
        """Cast this Vector to a Matrix (such as a column vector).

        This is SuiteSparse-specific and may change in the future.
        This does not copy the vector.
        """
        from .matrix import Matrix

        return Matrix._from_obj(
            ffi.cast("GrB_Matrix*", self.gb_obj),
            self.dtype,
            self._size,
            1,
            parent=self,
            name=f"(GrB_Matrix){self.name}",
        )

    def __repr__(self, mask=None):
        from .formatting import format_vector
        from .recorder import skip_record

        with skip_record:
            return format_vector(self, mask=mask)

    def _repr_html_(self, mask=None, collapse=False):
        if self._parent is not None and mask is None:
            # Scalars repr can't handle mask
            return self._parent._repr_html_(collapse=collapse)
        from .formatting import format_vector_html
        from .recorder import skip_record

        with skip_record:
            return format_vector_html(self, mask=mask, collapse=collapse)

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
        return Vector.ss.import_any(name=name, **pieces)

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
            return ScalarIndexExpr(self, resolved_indexes)
        else:
            return VectorIndexExpr(self, resolved_indexes, *shape)

    def __setitem__(self, keys, expr):
        Updater(self)[keys] = expr

    def __contains__(self, index):
        extractor = self[index]
        if not extractor._is_scalar:
            raise TypeError(
                f"Invalid index to Vector contains: {index!r}.  An integer is expected.  "
                "Doing `index in my_vector` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar._is_empty

    def __iter__(self):
        self.wait()  # sort in SS
        indices, values = self.to_values()
        return indices.flat

    def __sizeof__(self):
        size = ffi_new("size_t*")
        check_status(lib.GxB_Vector_memoryUsage(size, self.gb_obj[0]), self)
        return size[0] + object.__sizeof__(self)

    def isequal(self, other, *, check_dtype=False):
        """
        Check for exact equality (same size, same empty values)
        If `check_dtype` is True, also checks that dtypes match
        For equality of floating point Vectors, consider using `isclose`
        """
        other = self._expect_type(other, Vector, within="isequal", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
            return False
        if self._nvals != other._nvals:
            return False
        if check_dtype:
            # dtypes are equivalent, so not need to unify
            op = binary.eq[self.dtype]
        else:
            op = get_typed_op(binary.eq, self.dtype, other.dtype, kind="binary")

        matches = Vector(bool, self._size, name="v_isequal")
        matches << self.ewise_mult(other, op)
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce(monoid.land, allow_empty=False).new().value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        other = self._expect_type(other, Vector, within="isclose", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
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
        return matches.reduce(monoid.land, allow_empty=False).new().value

    @property
    def size(self):
        scalar = _scalar_index("s_size")
        call("GrB_Vector_size", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def shape(self):
        return (self._size,)

    @property
    def nvals(self):
        scalar = _scalar_index("s_nvals")
        call("GrB_Vector_nvals", [_Pointer(scalar), self])
        return scalar.gb_obj[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]), self)
        return n[0]

    def clear(self):
        call("GrB_Vector_clear", [self])

    def resize(self, size):
        size = _as_scalar(size, _INDEX, is_cscalar=True)
        call("GrB_Vector_resize", [self, size])
        self._size = size.value

    def to_values(self, dtype=None, *, sort=True):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as a 2-tuple of numpy arrays

        If sort is True, the indices is guaranteed to be sorted.
        """
        if sort:
            if backend != "suitesparse":
                raise NotImplementedError()
            self.wait()  # sort in SS
        nvals = self._nvals
        indices = _CArray(size=nvals, name="&index_array")
        values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        scalar = _scalar_index("s_nvals")
        scalar.value = nvals
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(f"GrB_Vector_extractTuples_{dtype_name}", [indices, values, _Pointer(scalar), self])
        values = values.array
        if dtype is not None:
            dtype = lookup_dtype(dtype)
            if dtype != self.dtype:
                values = values.astype(dtype.np_type)  # copies
        return (
            indices.array,
            values,
        )

    def build(self, indices, values, *, dup_op=None, clear=False, size=None):
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        values, dtype = values_to_numpy_buffer(values, self.dtype)
        n = values.shape[0]
        if indices.size != n:
            raise ValueError(
                f"`indices` and `values` lengths must match: {indices.size} != {values.size}"
            )
        if clear:
            self.clear()
        if size is not None:
            self.resize(size)
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

        indices = _CArray(indices)
        values = _CArray(values, self.dtype)
        dtype_name = "UDT" if self.dtype._is_udt else self.dtype.name
        call(
            f"GrB_Vector_build_{dtype_name}",
            [self, indices, values, _as_scalar(n, _INDEX, is_cscalar=True), dup_op],
        )

        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, dtype=None, *, mask=None, name=None):
        """
        GrB_Vector_dup
        Create a new Vector by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            rv = Vector(dtype, size=self._size, name=name)
            rv(mask=mask)[...] = self
        else:
            rv = Vector._from_obj(ffi_new("GrB_Vector*"), self.dtype, self._size, name=name)
            call("GrB_Vector_dup", [_Pointer(rv), self])
        return rv

    def diag(self, k=0, *, name=None):
        """
        GrB_Matrix_diag
        Create a Matrix whose kth diagonal is this Vector
        """
        from .matrix import Matrix

        k = _as_scalar(k, INT64, is_cscalar=True)
        n = self._size + abs(k.value)
        rv = Matrix._from_obj(ffi_new("GrB_Matrix*"), self.dtype, n, n, name=name)
        call("GrB_Matrix_diag", [_Pointer(rv), self, k])
        return rv

    def wait(self):
        """
        GrB_Vector_wait

        In non-blocking mode, the computations may be delayed and not yet safe
        to use by multiple threads.  Use wait to force completion of a Vector
        and make it safe to use as input parameters on multiple threads.
        """
        # TODO: expose COMPLETE or MATERIALIZE options to the user
        call("GrB_Vector_wait", [self, _MATERIALIZE])

    def get(self, index, default=None):
        """Get an element at index as a Python scalar.

        If there is no element at ``vector[index]``, then the default value is returned.
        """
        expr = self[index]
        if expr._is_scalar:
            rv = expr.new().value
            return default if rv is None else rv
        raise ValueError(
            "Bad index in Vector.get(...).  "
            "A single index should be given, and the result will be a Python scalar."
        )

    @classmethod
    def new(cls, dtype, size=0, *, name=None):
        """
        GrB_Vector_new
        Create a new empty Vector from the given type and size
        """
        warnings.warn(
            "`Vector.new(...)` is deprecated; please use `Vector(...)` instead.",
            DeprecationWarning,
        )
        return Vector(dtype, size, name=name)

    @classmethod
    def from_values(cls, indices, values, dtype=None, *, size=None, dup_op=None, name=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.

        values may be a scalar, in which case duplicate indices are ignored.
        """
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        values, new_dtype = values_to_numpy_buffer(values, dtype)
        # Compute size if not provided
        if size is None:
            if indices.size == 0:
                raise ValueError("No indices provided. Unable to infer size.")
            size = int(indices.max()) + 1
        if dtype is None and values.ndim > 1:
            # Look for array-subtdype
            new_dtype = lookup_dtype(np.dtype((new_dtype.np_type, values.shape[1:])))
        # Create the new vector
        w = cls(new_dtype, size, name=name)
        if values.ndim == 0:
            if dup_op is not None:
                raise ValueError(
                    "dup_op must be None if values is a scalar so that all "
                    "values can be identical.  Duplicate indices will be ignored."
                )
            # SS, SuiteSparse-specific: build_Scalar
            w.ss.build_scalar(indices, values.tolist())
        else:
            # This needs to be the original data to get proper error messages
            w.build(indices, values, dup_op=dup_op)
        return w

    @property
    def _carg(self):
        return self.gb_obj[0]

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to update to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=None):
        """
        GrB_Vector_eWiseAdd

        Result will contain the union of indices from both Vectors.

        Default op is monoid.plus.

        *Warning*: using binary operators can create very confusing behavior
        when only one of the two elements is present.

        Examples:
            - binary.minus where left=N/A and right=4 yields 4 rather than -4 as might be expected
            - binary.gt where left=N/A and right=4 yields True
            - binary.gt where left=N/A and right=0 yields False

        The behavior is caused by grabbing the non-empty value and using it directly without
        performing any operation. In the case of `gt`, the non-empty value is cast to a boolean.
        """
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        if require_monoid is not None:  # pragma: no cover
            warnings.warn(
                "require_monoid keyword is deprecated; "
                "future behavior will be like `require_monoid=False`",
                DeprecationWarning,
            )
        else:
            require_monoid = False
        method_name = "ewise_add"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
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
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                # Check this before we compute a possibly large matrix below
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _v_add_m, (self, other, op)],
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        expr = VectorExpression(
            method_name,
            f"GrB_Vector_eWiseAdd_{op.opclass}",
            [self, other],
            op=op,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_mult(self, other, op=binary.times):
        """
        GrB_Vector_eWiseMult

        Result will contain the intersection of indices from both Vectors
        Default op is binary.times
        """
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        method_name = "ewise_mult"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # Per the spec, op may be a semiring, but this is weird, so don't.
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, other, _v_mult_m, (self, other, op)],
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        expr = VectorExpression(
            method_name,
            f"GrB_Vector_eWiseMult_{op.opclass}",
            [self, other],
            op=op,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_union(self, other, op, left_default, right_default):
        """
        GxB_Vector_eWiseUnion

        This is similar to `ewise_add` in that result will contain the union of
        indices from both Vectors.  Unlike `ewise_add`, this will use
        ``left_default`` for the left value when there is a value on the right
        but not the left, and ``right_default`` for the right value when there
        is a value on the left but not the right.

        ``op`` should be a BinaryOp or Monoid.
        """
        # SS, SuiteSparse-specific: eWiseUnion
        from .matrix import Matrix, MatrixExpression, TransposedMatrix

        method_name = "ewise_union"
        other = self._expect_type(
            other, (Vector, Matrix, TransposedMatrix), within=method_name, argname="other", op=op
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
        if other.ndim == 2:
            # Broadcast columnwise from the left
            if other._nrows != self._size:
                raise DimensionMismatch(
                    "Dimensions not compatible for broadcasting Vector from the left "
                    f"to columns of Matrix in {method_name}.  Matrix.nrows (={other._nrows}) "
                    f"must equal Vector.size (={self._size})."
                )
            return MatrixExpression(
                method_name,
                None,
                [self, left, other, right, _v_union_m, (self, other, left, right, op)],
                expr_repr=expr_repr,
                nrows=other._nrows,
                ncols=other._ncols,
                op=op,
            )
        expr = VectorExpression(
            method_name,
            "GxB_Vector_eWiseUnion",
            [self, left, other, right],
            op=op,
            expr_repr=expr_repr,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def vxm(self, other, op=semiring.plus_times):
        """
        GrB_vxm
        Vector-Matrix multiplication. Result is a Vector.
        Default op is semiring.plus_times
        """
        from .matrix import Matrix, TransposedMatrix

        method_name = "vxm"
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within=method_name, argname="other", op=op
        )
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = VectorExpression(
            method_name,
            "GrB_vxm",
            [self, other],
            op=op,
            size=other._ncols,
            bt=other._is_transposed,
        )
        if self._size != other._nrows:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def apply(self, op, right=None, *, left=None):
        """
        GrB_Vector_apply
        Apply UnaryOp to each element of the calling Vector
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        method_name = "apply"
        extra_message = (
            "apply only accepts UnaryOp with no scalars or BinaryOp with `left` or `right` scalar"
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
            cfunc_name = "GrB_Vector_apply"
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
                cfunc_name = f"GrB_Vector_apply_BinaryOp1st_{dtype_name}"
            else:
                cfunc_name = "GrB_Vector_apply_BinaryOp1st_Scalar"
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
                cfunc_name = f"GrB_Vector_apply_{cfunc_method}_{dtype_name}"
            else:
                cfunc_name = f"GrB_Vector_apply_{cfunc_method}_Scalar"
            args = [self, right]
            expr_repr = "{0.name}.apply({op}, right={1._expr_name})"
        else:
            raise TypeError("Cannot provide both `left` and `right` to apply")
        return VectorExpression(
            method_name,
            cfunc_name,
            args,
            op=op,
            expr_repr=expr_repr,
            size=self._size,
        )

    def select(self, op, thunk=None):
        """
        GrB_Vector_select
        Compute SelectOp at each element of the calling Vector, keeping
        elements which return True.
        """
        method_name = "select"
        if isinstance(op, str):
            op = select.from_string(op)
        else:
            if isinstance(op, VectorExpression):
                # Try to rewrite e.g. `v.select(v == 7)` to `gb.select.value(v == 7)`
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
                # e.g., `v.select(w.S)` or `v.select(w < 7)`
                mask = _check_mask(op)
                if thunk is not None:
                    raise TypeError(
                        "thunk argument not None when calling select with mask or boolean object"
                    )
                self._expect_type(mask.parent, (Vector, Mask), within=method_name, argname="op")
                return VectorExpression(
                    "select",
                    None,
                    [self, mask, _select_mask, (self, mask)],  # [*expr_args, func, args]
                    expr_repr="{0.name}.select({1.name})",
                    size=self.size,
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
            cfunc_name = f"GrB_Vector_select_{dtype_name}"
        else:
            cfunc_name = "GrB_Vector_select_Scalar"
        return VectorExpression(
            method_name,
            cfunc_name,
            [self, thunk],
            op=op,
            expr_repr="{0.name}.select({op}, thunk={1._expr_name})",
            size=self._size,
            dtype=self.dtype,
        )

    def reduce(self, op=monoid.plus, *, allow_empty=True):
        """
        GrB_Vector_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise

        For empty Vector objects, the result will be the monoid identity if
        `allow_empty` is False or empty Scalar if `allow_empty` is True.
        """
        method_name = "reduce"
        op = get_typed_op(op, self.dtype, kind="binary|aggregator")
        if op.opclass == "BinaryOp" and op.monoid is not None:
            op = op.monoid
        else:
            self._expect_op(op, ("Monoid", "Aggregator"), within=method_name, argname="op")
        if not allow_empty and op.opclass == "Aggregator" and op.parent._monoid is None:
            # But we still kindly allow it if it's a monoid-only aggregator such as sum
            raise ValueError("allow_empty=False not allowed when using Aggregators")
        if allow_empty:
            cfunc_name = "GrB_Vector_reduce_Monoid_Scalar"
        elif self.dtype._is_udt:
            cfunc_name = "GrB_Vector_reduce_UDT"
        else:
            cfunc_name = "GrB_Vector_reduce_{output_dtype}"
        return ScalarExpression(
            method_name,
            cfunc_name,
            [self],
            op=op,  # to be determined later
            is_cscalar=not allow_empty,
        )

    # Unofficial methods
    def inner(self, other, op=semiring.plus_times):
        """
        Vector-vector inner (or dot) product. Result is a Scalar.

        Default op is semiring.plus_times

        *This is not a standard GraphBLAS function*
        """
        method_name = "inner"
        other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        self._expect_op(op, "Semiring", within=method_name, argname="op")
        expr = ScalarExpression(
            method_name,
            "GrB_vxm",
            [self, other._as_matrix()],
            op=op,
            is_cscalar=False,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def outer(self, other, op=binary.times):
        """
        Vector-vector outer (or cross) product. Result is a Matrix.

        Default op is binary.times

        *This is not a standard GraphBLAS function*
        """
        from .matrix import MatrixExpression

        method_name = "outer"
        other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        if op.opclass == "Monoid":
            op = op.binaryop
        op = get_semiring(monoid.any, op)
        expr = MatrixExpression(
            method_name,
            "GrB_mxm",
            [self._as_matrix(), other._as_matrix()],
            op=op,
            nrows=self._size,
            ncols=other._size,
            bt=True,
        )
        return expr

    def reposition(self, offset, *, size=None):
        """Reposition the values by adding `offset` to the indices.

        Positive offset moves values to the right, negative to the left.
        Values repositioned outside of the new Vector are dropped (i.e., they don't wrap around).

        This is not a standard GraphBLAS method.  This is implemented with an extract and assign.

        Parameters
        ----------
        offset : int
        size : int, optional
            The size of the new Vector.  If not specified, same size as input Vector.

        """
        if size is None:
            size = self._size
        else:
            size = int(size)
        offset = int(offset)
        if offset < 0:
            start = -offset
            stop = start + size
        else:
            start = 0
            stop = max(0, size - offset)
        chunk = self[start:stop].new(name="v_repositioning")
        indices = slice(start + offset, start + offset + chunk._size)
        return VectorExpression(
            "reposition",
            None,
            [self, _reposition, (indices, chunk)],  # [*expr_args, func, args]
            expr_repr="{0.name}.reposition(%d)" % offset,
            size=size,
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
        idx = resolved_indexes.indices[0]
        if result is None:
            result = Scalar(dtype, is_cscalar=is_cscalar, name=name)
        if is_cscalar:
            dtype_name = "UDT" if dtype._is_udt else dtype.name
            if (
                call(f"GrB_Vector_extractElement_{dtype_name}", [_Pointer(result), self, idx.index])
                is not NoValue
            ):
                result._empty = False
        else:
            call("GrB_Vector_extractElement_Scalar", [result, self, idx.index])
        return result

    def _prep_for_extract(self, resolved_indexes, mask=None, is_submask=False):
        index = resolved_indexes.indices[0]
        return VectorExpression(
            "__getitem__",
            "GrB_Vector_extract",
            [self, index, index.cscalar],
            expr_repr="{0.name}[{1._expr_name}]",
            size=index.size,
            dtype=self.dtype,
        )

    def _assign_element(self, resolved_indexes, value):
        idx = resolved_indexes.indices[0]
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
                call("GrB_Vector_removeElement", [self, idx.index])
                return
            if value.dtype._is_udt:
                dtype_name = "UDT"
                value = _Pointer(value)
            else:
                dtype_name = value.dtype.name
            cfunc_name = f"GrB_Vector_setElement_{dtype_name}"
        else:
            cfunc_name = "GrB_Vector_setElement_Scalar"
        call(cfunc_name, [self, value, idx.index])

    def _prep_for_assign(self, resolved_indexes, value, mask=None, is_submask=False):
        method_name = "__setitem__"
        idx = resolved_indexes.indices[0]
        size = idx.size
        cscalar = idx.cscalar
        index = idx.index

        if output_type(value) is Vector:
            if type(value) is not Vector:
                value = self._expect_type(
                    value,
                    Vector,
                    within=method_name,
                )
            if is_submask:
                if size is None:
                    # v[i](m) << w
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << w
                cfunc_name = "GrB_Vector_subassign"
                expr_repr = "[[{2._expr_name} elements]](%s) = {0.name}" % mask.name
            else:
                # v(m)[I] << w
                # v[I] << w
                cfunc_name = "GrB_Vector_assign"
                expr_repr = "[[{2._expr_name} elements]] = {0.name}"
        else:
            if type(value) is not Scalar:
                dtype = self.dtype if self.dtype._is_udt else None
                try:
                    value = Scalar.from_value(value, dtype, is_cscalar=None, name="")
                except (TypeError, ValueError):
                    if size is not None:
                        # v[I] << [1, 2, 3]
                        # v(m)[I] << [1, 2, 3]
                        # v[I](m) << [1, 2, 3]
                        try:
                            values, dtype = values_to_numpy_buffer(value, dtype, copy=True)
                        except Exception:
                            extra_message = "Literal scalars and lists also accepted."
                        else:
                            shape = values.shape
                            try:
                                vals = Vector.ss.import_full(
                                    values, dtype=dtype, take_ownership=True
                                )
                                if dtype.np_type.subdtype is not None:
                                    shape = vals.shape
                            except Exception:
                                vals = None
                            if vals is None or shape != (size,):
                                if dtype.np_type.subdtype is not None:
                                    extra = (
                                        " (this is assigning to a vector with sub-array dtype "
                                        f"({dtype}), so array shape should include dtype shape)"
                                    )
                                else:
                                    extra = ""
                                raise ValueError(
                                    f"shape mismatch: value array of shape {shape} "
                                    f"does not match indexing of shape ({size},)"
                                    f"{extra}"
                                ) from None
                            return self._prep_for_assign(
                                resolved_indexes, vals, mask=mask, is_submask=is_submask
                            )
                    else:
                        extra_message = "Literal scalars also accepted."
                    value = self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        argname="value",
                        extra_message=extra_message,
                    )
            if is_submask:
                if size is None:
                    # v[i](m) << c
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << c
                if value._is_cscalar:
                    if value.dtype._is_udt:
                        dtype_name = "UDT"
                        value = _Pointer(value)
                    else:
                        dtype_name = value.dtype.name
                    cfunc_name = f"GrB_Vector_subassign_{dtype_name}"
                else:
                    cfunc_name = "GrB_Vector_subassign_Scalar"
                expr_repr = "[[{2._expr_name} elements]](%s) = {0._expr_name}" % mask.name
            else:
                # v(m)[I] << c
                # v[I] << c
                if size is None:
                    index = _CArray([index.value])
                    cscalar = _as_scalar(1, _INDEX, is_cscalar=True)
                if value._is_cscalar:
                    if value.dtype._is_udt:
                        dtype_name = "UDT"
                        value = _Pointer(value)
                    else:
                        dtype_name = value.dtype.name
                    cfunc_name = f"GrB_Vector_assign_{dtype_name}"
                else:
                    cfunc_name = "GrB_Vector_assign_Scalar"
                expr_repr = "[[{2._expr_name} elements]] = {0._expr_name}"
        return VectorExpression(
            method_name,
            cfunc_name,
            [value, index, cscalar],
            expr_repr=expr_repr,
            size=self._size,
            dtype=self.dtype,
        )

    def _delete_element(self, resolved_indexes):
        idx = resolved_indexes.indices[0]
        call("GrB_Vector_removeElement", [self, idx.index])

    def to_pygraphblas(self):  # pragma: no cover
        """Convert to a new `pygraphblas.Vector`

        This does not copy data.

        This gives control of the underlying GraphBLAS object to `pygraphblas`.
        This means operations on the current `graphblas` object will fail!
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"to_pygraphblas only works with 'suitesparse' backend, not {backend}"
            )
        import pygraphblas as pg

        vector = pg.Vector(self.gb_obj, pg.types._gb_type_to_type(self.dtype.gb_obj))
        self.gb_obj = ffi.NULL
        return vector

    @classmethod
    def from_pygraphblas(cls, vector):  # pragma: no cover
        """Convert a `pygraphblas.Vector` to a new `graphblas.Vector`

        This does not copy data.

        This gives control of the underlying GraphBLAS object to `graphblas`.
        This means operations on the original `pygraphblas` object will fail!
        """
        if backend != "suitesparse":
            raise RuntimeError(
                f"from_pygraphblas only works with 'suitesparse' backend, not {backend!r}"
            )
        import pygraphblas as pg

        if not isinstance(vector, pg.Vector):
            raise TypeError(f"Expected pygraphblas.Vector object.  Got type: {type(vector)}")
        dtype = lookup_dtype(vector.gb_type)
        rv = cls(vector._vector, dtype)
        rv._size = vector.size
        vector._vector = ffi.NULL
        return rv


Vector.ss = class_property(Vector.ss, ss)


class VectorExpression(BaseExpression):
    __slots__ = "_size"
    ndim = 1
    output_type = Vector

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
        size=None,
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
        if size is None:
            size = args[0]._size
        self._size = size

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Vector(dtype, self._size, name=name)

    def __repr__(self):
        from .formatting import format_vector_expression

        return format_vector_expression(self)

    def _repr_html_(self):
        from .formatting import format_vector_expression_html

        return format_vector_expression_html(self)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    # Begin auto-generated code: Vector
    _get_value = _automethods._get_value
    S = wrapdoc(Vector.S)(property(_automethods.S))
    V = wrapdoc(Vector.V)(property(_automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(_automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(_automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(_automethods.apply))
    diag = wrapdoc(Vector.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(_automethods.gb_obj))
    get = wrapdoc(Vector.get)(property(_automethods.get))
    inner = wrapdoc(Vector.inner)(property(_automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(_automethods.isequal))
    name = wrapdoc(Vector.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(_automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(_automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(_automethods.reduce))
    reposition = wrapdoc(Vector.reposition)(property(_automethods.reposition))
    select = wrapdoc(Vector.select)(property(_automethods.select))
    ss = wrapdoc(Vector.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Vector.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Vector.to_values)(property(_automethods.to_values))
    vxm = wrapdoc(Vector.vxm)(property(_automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
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
    # End auto-generated code: Vector


class VectorIndexExpr(AmbiguousAssignOrExtract):
    __slots__ = "_size"
    ndim = 1
    output_type = Vector

    def __init__(self, parent, resolved_indexes, size):
        super().__init__(parent, resolved_indexes)
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    # Begin auto-generated code: Vector
    _get_value = _automethods._get_value
    S = wrapdoc(Vector.S)(property(_automethods.S))
    V = wrapdoc(Vector.V)(property(_automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(_automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(_automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(_automethods.apply))
    diag = wrapdoc(Vector.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(_automethods.gb_obj))
    get = wrapdoc(Vector.get)(property(_automethods.get))
    inner = wrapdoc(Vector.inner)(property(_automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(_automethods.isequal))
    name = wrapdoc(Vector.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(_automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(_automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(_automethods.reduce))
    reposition = wrapdoc(Vector.reposition)(property(_automethods.reposition))
    select = wrapdoc(Vector.select)(property(_automethods.select))
    ss = wrapdoc(Vector.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Vector.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Vector.to_values)(property(_automethods.to_values))
    vxm = wrapdoc(Vector.vxm)(property(_automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
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
    # End auto-generated code: Vector


utils._output_types[Vector] = Vector
utils._output_types[VectorIndexExpr] = Vector
utils._output_types[VectorExpression] = Vector

# Import matrix to import infix to import _infixmethods, which has side effects
from . import matrix  # noqa isort:skip

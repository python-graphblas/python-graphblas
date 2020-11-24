import itertools
import numpy as np
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType, call, _Pointer
from .dtypes import lookup_dtype, unify, UINT64
from .exceptions import check_status, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater, _CArray
from .mask import StructuralMask, ValueMask
from .ops import get_typed_op
from .scalar import Scalar, ScalarExpression, _CScalar

ffi_new = ffi.new


class Vector(BaseType):
    """
    GraphBLAS Sparse Vector
    High-level wrapper around GrB_Vector type
    """

    _name_counter = itertools.count()

    def __init__(self, gb_obj, dtype, *, name=None):
        if name is None:
            name = f"v_{next(Vector._name_counter)}"
        self._size = None
        super().__init__(gb_obj, dtype, name)

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Vector_free(gb_obj))

    def __repr__(self, mask=None):
        from .formatting import format_vector
        from .recorder import skip_record

        with skip_record:
            return format_vector(self, mask=mask)

    def _repr_html_(self, mask=None):
        from .formatting import format_vector_html
        from .recorder import skip_record

        with skip_record:
            return format_vector_html(self, mask=mask)

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
        self._expect_type(other, Vector, within="isequal", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
            return False
        if self._nvals != other._nvals:
            return False
        if check_dtype:
            # dtypes are equivalent, so not need to unify
            common_dtype = self.dtype
        else:
            common_dtype = unify(self.dtype, other.dtype)

        matches = Vector.new(bool, self._size, name="v_isequal")
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches._nvals != self._nvals:
            return False

        # Check if all results are True
        return matches.reduce(monoid.land).value

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        """
        Check for approximate equality (including same size and empty values)
        If `check_dtype` is True, also checks that dtypes match
        Closeness check is equivalent to `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`
        """
        self._expect_type(other, Vector, within="isclose", argname="other")
        if check_dtype and self.dtype != other.dtype:
            return False
        if self._size != other._size:
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
        return matches.reduce(monoid.land).value

    @property
    def size(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_size", empty=True)  # Actually GrB_Index dtype
        call("GrB_Vector_size", (_Pointer(scalar), self))
        return n[0]

    @property
    def shape(self):
        return (self._size,)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nvals", empty=True)  # Actually GrB_Index dtype
        call("GrB_Vector_nvals", (_Pointer(scalar), self))
        return n[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]))
        return n[0]

    def clear(self):
        call("GrB_Vector_clear", [self])

    def resize(self, size):
        size = _CScalar(size)
        call("GrB_Vector_resize", (self, size))
        self._size = size.scalar.value

    def to_values(self, *, dtype=None):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as a 2-tuple of numpy arrays
        """
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = lookup_dtype(dtype)
        nvals = self._nvals
        indices = _CArray(nvals, name="&index_array")
        values = _CArray(nvals, ctype=self.dtype.c_type, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, UINT64, name="s_nvals", empty=True)  # Actually GrB_Index dtype
        scalar.value = nvals
        call(f"GrB_Vector_extractTuples_{dtype.name}", (indices, values, _Pointer(scalar), self))
        return (
            np.frombuffer(ffi.buffer(indices._carg), dtype=np.uint64),
            np.frombuffer(ffi.buffer(values._carg), dtype=dtype.np_type),
        )

    def fast_export(self):
        """
        GxB_Vector_export

        Returns a dict of the constituent parts:
         - n: size (int)
         - i: indices (ndarray<uint64>)
         - x: values (ndarray of appropriate dtype)

        To reimport the Vector:
        ```
        pieces = v.fast_export()
        v2 = Vector.fast_import(**pieces)
        ```

        The underlying GraphBLAS object transfers ownership to numpy, disallowing further access.
        The caller should delete or stop using the Vector after calling `fast_export`.
        """
        dtype = np.dtype(self.dtype.np_type)
        index_dtype = np.dtype(np.uint64)

        vhandle = ffi_new("GrB_Vector*", self._carg)
        type_ = ffi_new("GrB_Type*")
        n = ffi_new("GrB_Index*")
        nvals = ffi_new("GrB_Index*")
        vi = ffi_new("GrB_Index**")
        vx = ffi_new("void**")
        check_status(lib.GxB_Vector_export(vhandle, type_, n, nvals, vi, vx, ffi.NULL))
        return {
            "size": n[0],
            "indices": np.frombuffer(
                ffi.buffer(vi[0], nvals[0] * index_dtype.itemsize), dtype=index_dtype
            ),
            "values": np.frombuffer(ffi.buffer(vx[0], nvals[0] * dtype.itemsize), dtype=dtype),
        }

    def build(self, indices, values, *, dup_op=None, clear=False, size=None):
        # TODO: accept `dtype` keyword to match the dtype of `values`?
        np_index = isinstance(indices, np.ndarray)
        if not np_index and not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        np_value = isinstance(values, np.ndarray)
        if not np_value and not isinstance(values, (tuple, list)):
            values = tuple(values)
        n = len(values)
        if len(indices) != n:
            raise ValueError(
                f"`indices` and `values` lengths must match: {len(indices)} != {len(values)}"
            )
        if clear:
            self.clear()
        if size is not None:
            self.resize(size)
        if n <= 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        if np_index:
            uindices = indices.astype(np.uint64, copy=False)
            indices = _CArray(uindices, from_buffer=True)
        else:
            indices = _CArray(indices)
        if np_value:
            verified_values = values.astype(self.dtype.np_type, copy=False)
            values = _CArray(verified_values, ctype=self.dtype.c_type, from_buffer=True)
        else:
            values = _CArray(values, ctype=self.dtype.c_type)
        call(f"GrB_Vector_build_{self.dtype.name}", (self, indices, values, _CScalar(n), dup_op))

        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self._nvals < n:
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, *, dtype=None, mask=None, name=None):
        """
        GrB_Vector_dup
        Create a new Vector by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            rv = Vector.new(dtype, size=self._size, name=name)
            rv(mask=mask)[:] << self
        else:
            new_vec = ffi_new("GrB_Vector*")
            rv = Vector(new_vec, self.dtype, name=name)
            call("GrB_Vector_dup", (_Pointer(rv), self))
        rv._size = self._size
        return rv

    @classmethod
    def new(cls, dtype, size=0, *, name=None):
        """
        GrB_Vector_new
        Create a new empty Vector from the given type and size
        """
        new_vector = ffi_new("GrB_Vector*")
        dtype = lookup_dtype(dtype)
        rv = cls(new_vector, dtype, name=name)
        if type(size) is not _CScalar:
            size = _CScalar(size)
        call("GrB_Vector_new", (_Pointer(rv), dtype, size))
        rv._size = size.scalar.value
        return rv

    @classmethod
    def from_values(cls, indices, values, *, size=None, dup_op=None, dtype=None, name=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.
        """
        iarr = indices
        varr = values
        if not isinstance(iarr, np.ndarray):
            if not isinstance(indices, (list, tuple)):
                indices = tuple(indices)
            iarr = np.array(indices)
        if not isinstance(varr, np.ndarray):
            if not isinstance(values, (list, tuple)):
                values = tuple(values)
            varr = np.array(values)

        if dtype is None:
            if len(varr) <= 0:
                raise ValueError("No values provided. Unable to determine type.")
            dtype = varr.dtype
            if dtype == object:
                raise ValueError("Unable to convert values to a usable dtype")
        dtype = lookup_dtype(dtype)
        # Compute size if not provided
        if size is None:
            if len(iarr) <= 0:
                raise ValueError("No indices provided. Unable to infer size.")
            size = int(iarr.max() + 1)
        if len(iarr) > 0 and "int" not in iarr.dtype.name:
            raise ValueError(f"indices must be integers, not {iarr.dtype.name}")
        # Create the new vector
        w = cls.new(dtype, size, name=name)
        # Add the data
        # This needs to be the original data to get proper error messages
        w.build(indices, values, dup_op=dup_op)
        return w

    @classmethod
    def fast_import(cls, *, size, indices, values, name=None):
        """
        GxB_Vector_import

        Returns a new Vector created from the pieces.

        The new Vector uses the underlying buffer of the input arrays.
        The caller should delete or stop using the input arrays after calling `fast_import`.
        """
        vhandle = ffi_new("GrB_Vector*")
        dtype = lookup_dtype(values.dtype)
        vi = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indices)))
        vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
        check_status(
            lib.GxB_Vector_import(vhandle, dtype._carg, size, len(values), vi, vx, ffi.NULL)
        )
        rv = cls(vhandle, dtype, name=name)
        rv._size = size
        return rv

    @property
    def _carg(self):
        return self.gb_obj[0]

    #########################################################
    # Delayed methods
    #
    # These return a delayed expression object which must be passed
    # to update to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        """
        GrB_eWiseAdd_Vector

        Result will contain the union of indices from both Vectors.

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
        self._expect_type(other, Vector, within=method_name, argname="other")
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
        expr = VectorExpression(
            method_name, f"GrB_eWiseAdd_Vector_{op.opclass}", [self, other], op=op,
        )
        if self._size != other._size:
            expr.new(name="")  # incompatible shape; raise now
        return expr

    def ewise_mult(self, other, op=binary.times):
        """
        GrB_eWiseMult_Vector

        Result will contain the intersection of indices from both Vectors
        Default op is binary.times
        """
        method_name = "ewise_add"
        self._expect_type(other, Vector, within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op")
        expr = VectorExpression(
            method_name, f"GrB_eWiseMult_Vector_{op.opclass}", [self, other], op=op,
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
        self._expect_type(other, (Matrix, TransposedMatrix), within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
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

    def apply(self, op, *, left=None, right=None):
        """
        GrB_Vector_apply
        Apply UnaryOp to each element of the calling Vector
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
            cfunc_name = "GrB_Vector_apply"
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
                op, "BinaryOp", within=method_name, argname="op", extra_message=extra_message,
            )
            cfunc_name = f"GrB_Vector_apply_BinaryOp1st_{left.dtype}"
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
                op, "BinaryOp", within=method_name, argname="op", extra_message=extra_message,
            )
            cfunc_name = f"GrB_Vector_apply_BinaryOp2nd_{right.dtype}"
            args = [self, right]
            expr_repr = "{0.name}.apply({op}, right={1})"
        else:
            raise TypeError("Cannot provide both `left` and `right` to apply")
        return VectorExpression(
            method_name, cfunc_name, args, op=op, expr_repr=expr_repr, size=self._size,
        )

    def reduce(self, op=monoid.plus):
        """
        GrB_Vector_reduce
        Reduce all values into a scalar
        Default op is monoid.lor for boolean and monoid.plus otherwise
        """
        method_name = "reduce_scalar"
        op = get_typed_op(op, self.dtype)
        self._expect_op(op, "Monoid", within=method_name, argname="op")
        return ScalarExpression(
            method_name,
            "GrB_Vector_reduce_{output_dtype}",
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
        index, _ = resolved_indexes.indices[0]
        result = Scalar.new(dtype, name=name)
        try:
            call(f"GrB_Vector_extractElement_{dtype}", (_Pointer(result), self, index))
        except NoValue:
            pass
        else:
            result._is_empty = False
        return result

    def _prep_for_extract(self, resolved_indexes):
        index, isize = resolved_indexes.indices[0]
        return VectorExpression(
            "__getitem__",
            "GrB_Vector_extract",
            [self, index, isize],
            expr_repr="{0.name}[[{2} elements]]",
            size=isize,
            dtype=self.dtype,
        )

    def _assign_element(self, resolved_indexes, value):
        index, _ = resolved_indexes.indices[0]
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
        call(f"GrB_Vector_setElement_{value.dtype}", (self, value, index))

    def _prep_for_assign(self, resolved_indexes, value):
        method_name = "__setitem__"
        index, isize = resolved_indexes.indices[0]
        if type(value) is Vector:
            cfunc_name = "GrB_Vector_assign"
            expr_repr = "[[{2} elements]] = {0.name}"
        else:
            try:
                value = _CScalar(value)
            except TypeError:
                self._expect_type(
                    value,
                    (Scalar, Vector),
                    within=method_name,
                    argname="value",
                    extra_message="Literal scalars also accepted.",
                )
            cfunc_name = f"GrB_Vector_assign_{value.dtype}"
            expr_repr = "[[{2} elements]] = {0}"
        return VectorExpression(
            method_name,
            cfunc_name,
            [value, index, isize],
            expr_repr=expr_repr,
            size=self._size,
            dtype=self.dtype,
        )

    def _delete_element(self, resolved_indexes):
        index, _ = resolved_indexes.indices[0]
        call("GrB_Vector_removeElement", (self, index))

    if backend == "pygraphblas":

        def to_pygraphblas(self):
            """Convert to a new `pygraphblas.Vector`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `pygraphblas`.
            This means operations on the current `grblas` object will fail!
            """
            import pygraphblas as pg

            vector = pg.Vector(self.gb_obj, pg.types.gb_type_to_type(self.dtype.gb_obj))
            self.gb_obj = ffi.NULL
            return vector

        @classmethod
        def from_pygraphblas(cls, vector):
            """Convert a `pygraphblas.Vector` to a new `grblas.Vector`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `grblas`.
            This means operations on the original `pygraphblas` object will fail!
            """
            dtype = lookup_dtype(vector.gb_type)
            rv = cls(vector.vector, dtype)
            rv._size = vector.size
            vector.vector = ffi.NULL
            return rv


class VectorExpression(BaseExpression):
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
            method_name, cfunc_name, args, at=at, bt=bt, op=op, dtype=dtype, expr_repr=expr_repr,
        )
        if size is None:
            size = args[0]._size
        self.size = self._size = size

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Vector.new(dtype, self._size, name=name)

    def __repr__(self):
        from .formatting import format_vector_expression

        return format_vector_expression(self)

    def _repr_html_(self):
        from .formatting import format_vector_expression_html

        return format_vector_expression_html(self)

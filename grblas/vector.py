import itertools
import numpy as np
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType, call
from .dtypes import lookup_dtype, unify, _INDEX
from .exceptions import check_status, check_status_carg, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
from .mask import StructuralMask, ValueMask
from .ops import get_typed_op
from .scalar import Scalar, ScalarExpression, _CScalar
from .utils import ints_to_numpy_buffer, values_to_numpy_buffer, _CArray, _Pointer
from . import _ss

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
        # Add ss extension methods
        self.ss = Vector.ss(self)

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None:
            # it's difficult/dangerous to record the call, b/c `self.name` may not exist
            check_status(lib.GrB_Vector_free(gb_obj), self)

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

    def __contains__(self, index):
        extractor = self[index]
        if not extractor.resolved_indexes.is_single_element:
            raise TypeError(
                f"Invalid index to Vector contains: {index!r}.  An integer is expected.  "
                "Doing `index in my_vector` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar.is_empty

    def __iter__(self):
        indices, values = self.to_values()
        return indices.flat

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
        scalar = Scalar(n, _INDEX, name="s_size", empty=True)
        call("GrB_Vector_size", [_Pointer(scalar), self])
        return n[0]

    @property
    def shape(self):
        return (self._size,)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", empty=True)
        call("GrB_Vector_nvals", [_Pointer(scalar), self])
        return n[0]

    @property
    def _nvals(self):
        """Like nvals, but doesn't record calls"""
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]), self)
        return n[0]

    def clear(self):
        call("GrB_Vector_clear", [self])

    def resize(self, size):
        size = _CScalar(size)
        call("GrB_Vector_resize", [self, size])
        self._size = size.scalar.value

    def to_values(self, *, dtype=None):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as a 2-tuple of numpy arrays
        """
        nvals = self._nvals
        indices = _CArray(size=nvals, name="&index_array")
        values = _CArray(size=nvals, dtype=self.dtype, name="&values_array")
        n = ffi_new("GrB_Index*")
        scalar = Scalar(n, _INDEX, name="s_nvals", empty=True)
        scalar.value = nvals
        call(
            f"GrB_Vector_extractTuples_{self.dtype.name}", [indices, values, _Pointer(scalar), self]
        )
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
        n = values.size
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
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        indices = _CArray(indices)
        values = _CArray(values, dtype=self.dtype)
        call(f"GrB_Vector_build_{self.dtype.name}", [self, indices, values, _CScalar(n), dup_op])

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
            call("GrB_Vector_dup", [_Pointer(rv), self])
        rv._size = self._size
        return rv

    def wait(self):
        """
        GrB_Vector_wait

        In non-blocking mode, the computations may be delayed and not yet safe
        to use by multiple threads.  Use wait to force completion of a Vector
        and make it safe to use as input parameters on multiple threads.
        """
        call("GrB_Vector_wait", [_Pointer(self)])

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
        call("GrB_Vector_new", [_Pointer(rv), dtype, size])
        rv._size = size.scalar.value
        return rv

    @classmethod
    def from_values(cls, indices, values, *, size=None, dup_op=None, dtype=None, name=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.
        """
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        values, dtype = values_to_numpy_buffer(values, dtype)
        # Compute size if not provided
        if size is None:
            if indices.size == 0:
                raise ValueError("No indices provided. Unable to infer size.")
            size = int(indices.max()) + 1
        # Create the new vector
        w = cls.new(dtype, size, name=name)
        # Add the data
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

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        """
        GrB_Vector_eWiseAdd

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
        method_name = "ewise_add"
        self._expect_type(other, Vector, within=method_name, argname="other")
        op = get_typed_op(op, self.dtype, other.dtype)
        self._expect_op(op, ("BinaryOp", "Monoid", "Semiring"), within=method_name, argname="op")
        expr = VectorExpression(
            method_name,
            f"GrB_Vector_eWiseMult_{op.opclass}",
            [self, other],
            op=op,
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
                try:
                    left = Scalar.from_value(left)
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
            cfunc_name = f"GrB_Vector_apply_BinaryOp1st_{left.dtype}"
            args = [_CScalar(left), self]
            expr_repr = "{1.name}.apply({op}, left={0})"
        elif left is None:
            if type(right) is not Scalar:
                try:
                    right = Scalar.from_value(right)
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
            cfunc_name = f"GrB_Vector_apply_BinaryOp2nd_{right.dtype}"
            args = [self, _CScalar(right)]
            expr_repr = "{0.name}.apply({op}, right={1})"
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
        if (
            call(f"GrB_Vector_extractElement_{dtype}", [_Pointer(result), self, index])
            is not NoValue
        ):
            result._is_empty = False
        return result

    def _prep_for_extract(self, resolved_indexes, mask=None, is_submask=False):
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
        if type(value) is not Scalar:
            try:
                value = Scalar.from_value(value)
            except TypeError:
                self._expect_type(
                    value,
                    Scalar,
                    within="__setitem__",
                    argname="value",
                    extra_message="Literal scalars also accepted.",
                )
        # should we cast?
        call(f"GrB_Vector_setElement_{value.dtype}", [self, _CScalar(value), index])

    def _prep_for_assign(self, resolved_indexes, value, mask=None, is_submask=False):
        method_name = "__setitem__"
        index, isize = resolved_indexes.indices[0]
        if type(value) is Vector:
            if is_submask:
                if isize is None:
                    # v[i](m) << w
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << w
                cfunc_name = "GrB_Vector_subassign"
                expr_repr = "[[{2} elements]](%s) = {0.name}" % mask.name
            else:
                # v(m)[I] << w
                # v[I] << w
                cfunc_name = "GrB_Vector_assign"
                expr_repr = "[[{2} elements]] = {0.name}"
        else:
            if type(value) is not Scalar:
                try:
                    value = Scalar.from_value(value)
                except TypeError:
                    self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        argname="value",
                        extra_message="Literal scalars also accepted.",
                    )
            value = _CScalar(value)
            if is_submask:
                if isize is None:
                    # v[i](m) << c
                    raise TypeError("Single element assign does not accept a submask")
                # v[I](m) << c
                cfunc_name = f"GrB_Vector_subassign_{value.dtype}"
                expr_repr = "[[{2} elements]](%s) = {0}" % mask.name
            else:
                # v(m)[I] << c
                # v[I] << c
                if isize is None:
                    index = _CArray([index.scalar.value])
                    isize = _CScalar(1)
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
        call("GrB_Vector_removeElement", [self, index])

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

    class ss:
        def __init__(self, parent):
            self._parent = parent

        @property
        def format(self):
            parent = self._parent
            sparsity_ptr = ffi_new("GxB_Option_Field*")
            check_status(
                lib.GxB_Vector_Option_get(parent._carg, lib.GxB_SPARSITY_STATUS, sparsity_ptr),
                parent,
            )
            sparsity_status = sparsity_ptr[0]
            if sparsity_status == lib.GxB_SPARSE:
                format = "sparse"
            elif sparsity_status == lib.GxB_BITMAP:
                format = "bitmap"
            elif sparsity_status == lib.GxB_FULL:
                format = "full"
            else:  # pragma: no cover
                raise NotImplementedError(f"Unknown sparsity status: {sparsity_status}")
            return format

        def export(self, format=None, sort=False, give_ownership=False, raw=False):
            """
            GxB_Vextor_export_xxx

            Parameters
            ----------
            format : str or None, default None
                If `format` is not specified, this method exports in the currently stored format.
                To control the export format, set `format` to one of:
                    - "sparse"
                    - "bitmap"
                    - "full"
            sort : bool, default False
                Whether to sort indices if the format is "sparse"
            give_ownership : bool, default False
                Perform a zero-copy data transfer to Python if possible.  This gives ownership of
                the underlying memory buffers to Numpy.
                ** If True, this nullifies the current object, which should no longer be used! **
            raw : bool, default False
                If True, always return array the same size as returned by SuiteSparse.
                If False, arrays may be trimmed to be the expected size.
                It may make sense to choose ``raw=True`` if one wants to use the data to perform
                a zero-copy import back to SuiteSparse.

            Returns
            -------
            dict; keys depend on `format` and `raw` arguments (see below).

            See Also
            --------
            Vector.to_values
            Vector.ss.import_any

            Return values
                - Note: for `raw=True`, arrays may be larger than specified.
                - "sparse" format
                    - indices : ndarray(dtype=uint64, size=nvals)
                    - values : ndarray(size=nvals)
                    - sorted_index : bool
                        - True if the values in "indices" are sorted
                    - size : int
                    - nvals : int, only present if raw == True
                - "bitmap" format
                    - bitmap : ndarray(dtype=bool8, size=size)
                    - values : ndarray(size=size)
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                    - size : int, only present if raw == True
                - "bitmap" format
                    - bitmap : ndarray(dtype=bool8, size=size)
                    - values : ndarray(size=size)
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                    - size : int, only present if raw == True
                - "full" format
                    - values : ndarray(size=size)
                    - size : int, only present if raw == True

            Examples
            --------
            Simple usage:

            >>> pieces = v.ss.export()
            >>> v2 = Vector.ss.import_any(**pieces)

            """
            if give_ownership:
                parent = self._parent
            else:
                parent = self._parent.dup(name="v_export")
            dtype = np.dtype(parent.dtype.np_type)
            index_dtype = np.dtype(np.uint64)

            if format is None:
                format = self.format
            else:
                format = format.lower()

            vhandle = ffi_new("GrB_Vector*", parent._carg)
            type_ = ffi_new("GrB_Type*")
            size = ffi_new("GrB_Index*")
            vx = ffi_new("void**")
            vx_size = ffi_new("GrB_Index*")
            if sort:
                jumbled = ffi.NULL
            else:
                jumbled = ffi_new("bool*")

            if format == "sparse":
                vi = ffi_new("GrB_Index**")
                vi_size = ffi_new("GrB_Index*")
                nvals = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Vector_export_CSC(
                        vhandle, type_, size, vi, vx, vi_size, vx_size, nvals, jumbled, ffi.NULL
                    ),
                    parent,
                )
                nvals = nvals[0]
                indices = _ss.claim_buffer(ffi, vi[0], vi_size[0], index_dtype)
                values = _ss.claim_buffer(ffi, vx[0], vx_size[0], dtype)
                if not raw:
                    if indices.size > nvals:  # pragma: no cover
                        indices = indices[:nvals]
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
                rv = {
                    "size": size[0],
                    "indices": indices,
                    "sorted_index": True if sort else not jumbled[0],
                }
                if raw:
                    rv["nvals"] = nvals
            elif format == "bitmap":
                vb = ffi_new("int8_t**")
                vb_size = ffi_new("GrB_Index*")
                nvals = ffi_new("GrB_Index*")
                check_status(
                    lib.GxB_Vector_export_Bitmap(
                        vhandle, type_, size, vb, vx, vb_size, vx_size, nvals, ffi.NULL
                    ),
                    parent,
                )
                bitmap = _ss.claim_buffer(ffi, vb[0], vb_size[0], np.dtype(np.bool8))
                values = _ss.claim_buffer(ffi, vx[0], vx_size[0], dtype)
                size = size[0]
                if not raw:
                    if bitmap.size > size:  # pragma: no cover
                        bitmap = bitmap[:size]
                    if values.size > size:  # pragma: no cover
                        values = values[:size]
                rv = {
                    "bitmap": bitmap,
                    "nvals": nvals[0],
                }
                if raw:
                    rv["size"] = size
            elif format == "full":
                check_status(
                    lib.GxB_Vector_export_Full(vhandle, type_, size, vx, vx_size, ffi.NULL),
                    parent,
                )
                values = _ss.claim_buffer(ffi, vx[0], vx_size[0], dtype)
                size = size[0]
                if not raw:
                    if values.size > size:  # pragma: no cover
                        values = values[:size]
                rv = {}
                if raw:
                    rv["size"] = size
            else:
                raise ValueError(f"Invalid format: {format}")

            rv.update(
                format=format,
                values=values,
            )
            parent.gb_obj = ffi.NULL
            return rv

        @classmethod
        def import_any(
            cls,
            *,
            # All
            values,
            size=None,
            take_ownership=False,
            format=None,
            name=None,
            # Sparse
            indices=None,
            sorted_index=False,
            # Bitmap
            bitmap=None,
            # Bitmap/Sparse
            nvals=None,  # optional
        ):
            """
            GxB_Vector_import_xxx

            Dispatch to appropriate import method inferred from inputs.
            See the other import functions and `Vector.ss.export`` for details.

            Returns
            -------
            Vector

            See Also
            --------
            Vector.from_values
            Vector.ss.export
            Vector.ss.import_sparse
            Vector.ss.import_bitmap
            Vector.ss.import_full

            Examples
            --------
            Simple usage:

            >>> pieces = v.ss.export()
            >>> v2 = Vector.ss.import_any(**pieces)

            """
            if format is None:
                if indices is not None:
                    if bitmap is not None:
                        raise TypeError("Cannot provide both `indptr` and `bitmap`")
                    format = "sparse"
                elif bitmap is not None:
                    format = "bitmap"
                else:
                    format = "full"
            else:
                format = format.lower()
            if format == "sparse":
                return cls.import_sparse(
                    size=size,
                    indices=indices,
                    values=values,
                    nvals=nvals,
                    sorted_index=sorted_index,
                    take_ownership=take_ownership,
                    name=name,
                )
            elif format == "bitmap":
                return cls.import_bitmap(
                    nvals=nvals,
                    bitmap=bitmap,
                    values=values,
                    size=size,
                    take_ownership=take_ownership,
                    name=name,
                )
            elif format == "full":
                return cls.import_full(
                    values=values,
                    size=size,
                    take_ownership=take_ownership,
                    name=name,
                )
            else:
                raise ValueError(f"Invalid format: {format}")

        @classmethod
        def import_sparse(
            cls,
            *,
            size,
            indices,
            values,
            nvals=None,
            sorted_index=False,
            take_ownership=False,
            dtype=None,
            format=None,
            name=None,
        ):
            """
            GxB_Vector_import_CSC

            Create a new Vector from sparse input.

            Parameters
            ----------
            size : int
            indices : array-like
            values : array-like
            nvals : int, optional
                The number of elements in "values" to use.
                If not specified, will be set to ``len(values)``.
            sorted_index : bool, default False
                Indicate whether the values in "col_indices" are sorted.
            take_ownership : bool, default False
                If True, perform a zero-copy data transfer from input numpy arrays
                to GraphBLAS if possible.  To give ownership of the underlying
                memory buffers to GraphBLAS, the arrays must:
                    - be C contiguous
                    - have the correct dtype (uint64 for indices)
                    - own its own data
                    - be writeable
                If all of these conditions are not met, then the data will be
                copied and the original array will be unmodified.  If zero copy
                to GraphBLAS is successful, then the array will be mofied to be
                read-only and will no longer own the data.
            dtype : dtype, optional
                dtype of the new Vector.
                If not specified, this will be inferred from `values`.
            format : str, optional
                Must be "sparse" or None.  This is included to be compatible with
                the dict returned from exporting.
            name : str, optional
                Name of the new Vector.

            Returns
            -------
            Vector
            """
            if format is not None and format.lower() != "sparse":
                raise ValueError(f"Invalid format: {format!r}.  Must be None or 'sparse'.")
            copy = not take_ownership
            indices = ints_to_numpy_buffer(
                indices, np.uint64, copy=copy, ownable=True, name="indices"
            )
            values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
            if indices is values:
                values = np.copy(values)
            vhandle = ffi_new("GrB_Vector*")
            vi = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indices)))
            vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if nvals is None:
                nvals = values.size
            check_status_carg(
                lib.GxB_Vector_import_CSC(
                    vhandle,
                    dtype._carg,
                    size,
                    vi,
                    vx,
                    indices.size,
                    values.size,
                    nvals,
                    not sorted_index,
                    ffi.NULL,
                ),
                "Vector",
                vhandle[0],
            )
            rv = Vector(vhandle, dtype, name=name)
            rv._size = size
            _ss.unclaim_buffer(indices)
            _ss.unclaim_buffer(values)
            return rv

        @classmethod
        def import_bitmap(
            cls,
            *,
            bitmap,
            values,
            nvals=None,
            size=None,
            take_ownership=False,
            dtype=None,
            format=None,
            name=None,
        ):
            """
            GxB_Vector_import_Bitmap

            Create a new Vector from values and bitmap (as mask) arrays.

            Parameters
            ----------
            bitmap : array-like
                True elements indicate where there are values in "values".
            values : array-like
            nvals : int, optional
                The number of True elements in the bitmap for this Vector.
            size : int, optional
                The size of the new Vector.
                If not specified, it will be set to the size of values.
            take_ownership : bool, default False
                If True, perform a zero-copy data transfer from input numpy arrays
                to GraphBLAS if possible.  To give ownership of the underlying
                memory buffers to GraphBLAS, the arrays must:
                    - be C contiguous
                    - have the correct dtype (bool8 for bitmap)
                    - own its own data
                    - be writeable
                If all of these conditions are not met, then the data will be
                copied and the original array will be unmodified.  If zero copy
                to GraphBLAS is successful, then the array will be mofied to be
                read-only and will no longer own the data.
            dtype : dtype, optional
                dtype of the new Vector.
                If not specified, this will be inferred from `values`.
            format : str, optional
                Must be "bitmap" or None.  This is included to be compatible with
                the dict returned from exporting.
            name : str, optional
                Name of the new Vector.

            Returns
            -------
            Vector
            """
            if format is not None and format.lower() != "bitmap":
                raise ValueError(f"Invalid format: {format!r}.  Must be None or 'bitmap'.")
            copy = not take_ownership
            bitmap = ints_to_numpy_buffer(bitmap, np.bool8, copy=copy, ownable=True, name="bitmap")
            values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
            if bitmap is values:
                values = np.copy(values)
            vhandle = ffi_new("GrB_Vector*")
            vb = ffi_new("int8_t**", ffi.cast("int8_t*", ffi.from_buffer(bitmap)))
            vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if size is None:
                size = values.size
            if nvals is None:
                if bitmap.size == size:
                    nvals = np.count_nonzero(bitmap)
                else:
                    nvals = np.count_nonzero(bitmap.ravel()[:size])
            check_status_carg(
                lib.GxB_Vector_import_Bitmap(
                    vhandle,
                    dtype._carg,
                    size,
                    vb,
                    vx,
                    bitmap.size,
                    values.size,
                    nvals,
                    ffi.NULL,
                ),
                "Vector",
                vhandle[0],
            )
            rv = Vector(vhandle, dtype, name=name)
            rv._size = size
            _ss.unclaim_buffer(bitmap)
            _ss.unclaim_buffer(values)
            return rv

        @classmethod
        def import_full(
            cls, *, values, size=None, take_ownership=False, dtype=None, format=None, name=None
        ):
            """
            GxB_Vector_import_Full

            Create a new Vector from values.

            Parameters
            ----------
            values : array-like
            size : int, optional
                The size of the new Vector.
                If not specified, it will be set to the size of values.
            take_ownership : bool, default False
                If True, perform a zero-copy data transfer from input numpy arrays
                to GraphBLAS if possible.  To give ownership of the underlying
                memory buffers to GraphBLAS, the arrays must:
                    - be C contiguous
                    - have the correct dtype (bool8 for bitmap)
                    - own its own data
                    - be writeable
                If all of these conditions are not met, then the data will be
                copied and the original array will be unmodified.  If zero copy
                to GraphBLAS is successful, then the array will be mofied to be
                read-only and will no longer own the data.
            dtype : dtype, optional
                dtype of the new Vector.
                If not specified, this will be inferred from `values`.
            format : str, optional
                Must be "full" or None.  This is included to be compatible with
                the dict returned from exporting.
            name : str, optional
                Name of the new Vector.

            Returns
            -------
            Vector
            """
            if format is not None and format.lower() != "full":
                raise ValueError(f"Invalid format: {format!r}.  Must be None or 'full'.")
            copy = not take_ownership
            values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
            vhandle = ffi_new("GrB_Vector*")
            vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
            if size is None:
                size = values.size
            check_status_carg(
                lib.GxB_Vector_import_Full(
                    vhandle,
                    dtype._carg,
                    size,
                    vx,
                    values.size,
                    ffi.NULL,
                ),
                "Vector",
                vhandle[0],
            )
            rv = Vector(vhandle, dtype, name=name)
            rv._size = size
            _ss.unclaim_buffer(values)
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

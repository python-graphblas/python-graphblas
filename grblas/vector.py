import itertools
from . import ffi, lib, backend, binary, monoid, semiring
from .base import BaseExpression, BaseType
from .dtypes import libget, lookup_dtype, unify
from .exceptions import check_status, is_error, NoValue
from .expr import AmbiguousAssignOrExtract, IndexerResolver, Updater
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
        super().__init__(gb_obj, dtype, name)

    def __del__(self):
        check_status(lib.GrB_Vector_free(self.gb_obj))

    def __repr__(self, mask=None):
        from .formatting import format_vector

        return format_vector(self, mask=mask)

    def _repr_html_(self, mask=None):
        from .formatting import format_vector_html

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
        if self.size != other.size:
            return False
        if self.nvals != other.nvals:
            return False
        if check_dtype:
            # dtypes are equivalent, so not need to unify
            common_dtype = self.dtype
        else:
            common_dtype = unify(self.dtype, other.dtype)

        matches = Vector.new(bool, self.size, name="v_isequal")
        matches << self.ewise_mult(other, binary.eq[common_dtype])
        # ewise_mult performs intersection, so nvals will indicate mismatched empty values
        if matches.nvals != self.nvals:
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
        if self.size != other.size:
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
        return matches.reduce(monoid.land).value

    @property
    def size(self):
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_size(n, self.gb_obj[0]))
        return n[0]

    @property
    def shape(self):
        return (self.size,)

    @property
    def nvals(self):
        n = ffi_new("GrB_Index*")
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]))
        return n[0]

    def clear(self):
        check_status(lib.GrB_Vector_clear(self.gb_obj[0]))

    def resize(self, size):
        check_status(lib.GrB_Vector_resize(self.gb_obj[0], size))

    def to_values(self):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as 2 generators
        """
        indices = ffi_new("GrB_Index[]", self.nvals)
        values = ffi_new(f"{self.dtype.c_type}[]", self.nvals)
        n = ffi_new("GrB_Index*")
        n[0] = self.nvals
        func = libget(f"GrB_Vector_extractTuples_{self.dtype.name}")
        check_status(func(indices, values, n, self.gb_obj[0]))
        return tuple(indices), tuple(values)

    def build(self, indices, values, *, dup_op=None, clear=False):
        # TODO: add `size` option once .resize is available
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if len(indices) != len(values):
            raise ValueError(
                f"`indices` and `values` have different lengths " f"{len(indices)} != {len(values)}"
            )
        if clear:
            self.clear()
        n = len(indices)
        if n <= 0:
            return

        dup_op_given = dup_op is not None
        if not dup_op_given:
            dup_op = binary.plus
        dup_op = get_typed_op(dup_op, self.dtype)
        self._expect_op(dup_op, "BinaryOp", within="build", argname="dup_op")

        indices = ffi_new("GrB_Index[]", indices)
        values = ffi_new(f"{self.dtype.c_type}[]", values)
        # Push values into w
        func = libget(f"GrB_Vector_build_{self.dtype.name}")
        check_status(func(self.gb_obj[0], indices, values, n, dup_op.gb_obj))
        # Check for duplicates when dup_op was not provided
        if not dup_op_given and self.nvals < len(values):
            raise ValueError("Duplicate indices found, must provide `dup_op` BinaryOp")

    def dup(self, *, dtype=None, mask=None, name=None):
        """
        GrB_Vector_dup
        Create a new Vector by duplicating this one
        """
        if dtype is not None or mask is not None:
            if dtype is None:
                dtype = self.dtype
            new_vec = type(self).new(dtype, size=self.size, name=name)
            new_vec(mask=mask)[:] << self
            return new_vec
        new_vec = ffi_new("GrB_Vector*")
        check_status(lib.GrB_Vector_dup(new_vec, self.gb_obj[0]))
        return type(self)(new_vec, self.dtype, name=name)

    @classmethod
    def new(cls, dtype, size=0, *, name=None):
        """
        GrB_Vector_new
        Create a new empty Vector from the given type and size
        """
        new_vector = ffi_new("GrB_Vector*")
        dtype = lookup_dtype(dtype)
        check_status(lib.GrB_Vector_new(new_vector, dtype.gb_type, size))
        return cls(new_vector, dtype, name=name)

    @classmethod
    def from_values(cls, indices, values, *, size=None, dup_op=None, dtype=None, name=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.
        """
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if dtype is None:
            if len(values) <= 0:
                raise ValueError("No values provided. Unable to determine type.")
            # Find dtype from any of the values (assumption is they are the same type)
            dtype = type(values[0])
        dtype = lookup_dtype(dtype)
        # Compute size if not provided
        if size is None:
            if not indices:
                raise ValueError("No indices provided. Unable to infer size.")
            size = max(indices) + 1
        # Create the new vector
        w = cls.new(dtype, size, name=name)
        # Add the data
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
        return VectorExpression(
            method_name, f"GrB_eWiseAdd_Vector_{op.opclass}", [self, other], op=op,
        )

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
        return VectorExpression(
            method_name, f"GrB_eWiseMult_Vector_{op.opclass}", [self, other], op=op,
        )

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
        return VectorExpression(
            method_name, "GrB_vxm", [self, other], op=op, size=other.ncols, bt=other._is_transposed,
        )

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
            cfunc_name = f"GrB_Vector_apply_BinaryOp1st_{left.dtype}"
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
            cfunc_name = f"GrB_Vector_apply_BinaryOp2nd_{right.dtype}"
            args = [self, _CScalar(right)]
            expr_repr = "{0.name}.apply({op}, right={1})"
        else:
            raise TypeError("Cannot provide both `left` and `right` to apply")
        return VectorExpression(
            method_name, cfunc_name, args, op=op, expr_repr=expr_repr, size=self.size,
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
    def _extract_element(self, resolved_indexes):
        index, _ = resolved_indexes.indices[0]
        func = libget(f"GrB_Vector_extractElement_{self.dtype}")
        result = ffi_new(f"{self.dtype.c_type}*")
        err_code = func(result, self.gb_obj[0], index)
        # Don't raise error for no value, simply return `None`
        if is_error(err_code, NoValue):
            return None, self.dtype
        check_status(err_code)
        return result[0], self.dtype

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
        func = libget(f"GrB_Vector_setElement_{value.dtype}")
        check_status(func(self.gb_obj[0], value.value, index))  # should we cast?

    def _prep_for_assign(self, resolved_indexes, value):
        method_name = "__setitem__"
        index, isize = resolved_indexes.indices[0]
        if type(value) is Vector:
            cfunc_name = "GrB_Vector_assign"
            expr_repr = "[[{2} elements]] = {0.name}"
        else:
            if type(value) is not Scalar:
                try:
                    value = Scalar.from_value(value, name="s_assign")
                except TypeError:
                    self._expect_type(
                        value,
                        (Scalar, Vector),
                        within=method_name,
                        argname="value",
                        extra_message="Literal scalars also accepted.",
                    )
            cfunc_name = f"GrB_Vector_assign_{value.dtype}"
            value = _CScalar(value)
            expr_repr = "[[{2} elements]] = {0}"
        return VectorExpression(
            method_name,
            cfunc_name,
            [value, index, isize],
            expr_repr=expr_repr,
            size=self.size,
            dtype=self.dtype,
        )

    def _delete_element(self, resolved_indexes):
        index, _ = resolved_indexes.indices[0]
        check_status(lib.GrB_Vector_removeElement(self.gb_obj[0], index))

    if backend == "pygraphblas":  # pragma: no cover

        def to_pygraphblas(self):
            """ Convert to a new `pygraphblas.Vector`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `pygraphblas`.
            This means operations on the current `grblas` object will fail!
            """
            import pygraphblas

            vector = pygraphblas.Vector(self.gb_obj, self.dtype.gb_type)
            self.gb_obj = ffi.NULL
            return vector

        @classmethod
        def from_pygraphblas(cls, vector):
            """ Convert a `pygraphblas.Vector` to a new `grblas.Vector`

            This does not copy data.

            This gives control of the underlying GraphBLAS object to `grblas`.
            This means operations on the original `pygraphblas` object will fail!
            """
            dtype = lookup_dtype(vector.gb_type)
            rv = cls(vector.vector, dtype)
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
            size = args[0].size
        self.size = size

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return Vector.new(dtype, self.size, name=name)

    def __repr__(self):
        from .formatting import format_vector_expression

        return format_vector_expression(self)

    def _repr_html_(self):
        from .formatting import format_vector_expression_html

        return format_vector_expression_html(self)

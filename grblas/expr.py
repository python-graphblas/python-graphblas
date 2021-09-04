import numpy as np

from . import lib, utils
from .utils import _CArray, output_type


class _AllIndices:
    __slots__ = "_carg", "name"

    def __init__(self):
        self._carg = lib.GrB_ALL
        self.name = "GrB_ALL"


_ALL_INDICES = _AllIndices()


class IndexerResolver:
    __slots__ = "obj", "indices", "__weakref__"

    def __init__(self, obj, indices):
        self.obj = obj
        self.indices = self.parse_indices(indices, obj.shape)

    @property
    def is_single_element(self):
        for idx, size in self.indices:
            if size is not None:
                return False
        return True

    def parse_indices(self, indices, shape):
        """
        Returns
            [(rows, rowsize), (cols, colsize)] for Matrix
            [(idx, idx_size)] for Vector

        Within each tuple, if the index is of type int, the size will be None
        """
        if len(shape) == 1:
            if type(indices) is tuple:
                raise TypeError(f"Index for {type(self.obj).__name__} cannot be a tuple")
            # Convert to tuple for consistent processing
            indices = (indices,)
        else:  # len(shape) == 2
            if type(indices) is not tuple or len(indices) != 2:
                raise TypeError(f"Index for {type(self.obj).__name__} must be a 2-tuple")

        out = []
        for i, idx in enumerate(indices):
            typ = output_type(idx)
            if typ is tuple:
                raise TypeError(
                    f"Index in position {i} cannot be a tuple; must use slice or list or int"
                )
            out.append(self.parse_index(idx, typ, shape[i]))
        return out

    def parse_index(self, index, typ, size):
        from .scalar import _CScalar

        if np.issubdtype(typ, np.integer):
            if index >= size:
                raise IndexError(f"Index out of range: index={index}, size={size}")
            return _CScalar(int(index)), None
        if typ is list:
            pass
        elif typ is slice:
            if index == slice(None) or index == slice(0, None):
                # [:] means all indices; use special GrB_ALL indicator
                return _ALL_INDICES, _CScalar(size)
            index = list(range(size)[index])
        elif typ is np.ndarray:
            if len(index.shape) != 1:
                raise TypeError(f"Invalid number of dimensions for index: {len(index.shape)}")
            if not np.issubdtype(index.dtype, np.integer):
                raise TypeError(f"Invalid dtype for index: {index.dtype}")
            return _CArray(index), _CScalar(len(index))
        else:
            from .scalar import Scalar

            if typ is Scalar:
                if index.dtype.name.startswith("F"):
                    raise TypeError(f"An integer is required for indexing.  Got: {index.dtype}")
                return _CScalar(index), None

            from .matrix import Matrix, TransposedMatrix
            from .vector import Vector

            if typ is Vector or typ is Matrix:
                raise TypeError(
                    f"Invalid type for index: {typ.__name__}.\n"
                    f"If you want to apply a mask, perhaps do something like "
                    f"`x.dup(mask={index.name}.S)`.\n"
                    f"If you want to assign with a mask, perhaps do something like "
                    f"`x(mask={index.name}.S) << value`."
                )
            elif typ is TransposedMatrix:
                raise TypeError(f"Invalid type for index: {typ.__name__}.")
            try:
                index = list(index)
            except Exception:
                from .mask import Mask

                if isinstance(index, Mask):
                    raise TypeError(
                        f"Invalid type for index: {typ.__name__}.\n"
                        f"If you want to apply a mask, perhaps do something like "
                        f"`x.dup(mask={index.name})`.\n"
                        f"If you want to assign with a mask, perhaps do something like "
                        f"`x(mask={index.name}) << value`."
                    )
                raise TypeError(f"Invalid type for index: {typ}; unable to convert to list")
        return _CArray(index), _CScalar(len(index))

    def get_index(self, dim):
        """Return a new IndexerResolver with index for the selected dimension"""
        rv = object.__new__(IndexerResolver)
        rv.obj = self.obj
        rv.indices = (self.indices[dim],)
        return rv


class Assigner:
    __slots__ = "updater", "resolved_indexes", "is_submask", "__weakref__"

    def __init__(self, updater, resolved_indexes, *, is_submask):
        # We could check here whether mask dimensions match index dimensions.
        # We could also check for valid `updater.kwargs` if `resolved_indexes.is_single_element`.
        self.updater = updater
        self.resolved_indexes = resolved_indexes
        self.is_submask = is_submask
        if updater.kwargs.get("input_mask") is not None:
            raise TypeError("`input_mask` argument may only be used for extract")

    def update(self, obj):
        # Occurs when user calls `C[index](...).update(obj)` or `C(...)[index].update(obj)`
        self.updater._setitem(self.resolved_indexes, obj, is_submask=self.is_submask)

    def __lshift__(self, obj):
        # Occurs when user calls `C[index](...) << obj` or `C(...)[index] << obj`
        self.updater._setitem(self.resolved_indexes, obj, is_submask=self.is_submask)

    def __eq__(self, other):
        raise TypeError(f"__eq__ not defined for objects of type {type(self)}.")

    def __bool__(self):
        raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")


class AmbiguousAssignOrExtract:
    __slots__ = "parent", "resolved_indexes", "__weakref__"

    def __init__(self, parent, resolved_indexes):
        self.parent = parent
        self.resolved_indexes = resolved_indexes

    def __call__(self, *args, **kwargs):
        # Occurs when user calls `C[index](params)`
        # Reverse the call order so we can parse the call args and kwargs
        updater = self.parent(*args, **kwargs)
        is_submask = updater.kwargs.get("mask") is not None
        return Assigner(updater, self.resolved_indexes, is_submask=is_submask)

    def __lshift__(self, obj):
        # Occurs when user calls `C[index] << obj`
        self.update(obj)

    def update(self, obj):
        # Occurs when user calls `C[index].update(obj)`
        if getattr(self.parent, "_is_transposed", False):
            raise TypeError("'TransposedMatrix' object does not support item assignment")
        Updater(self.parent)._setitem(self.resolved_indexes, obj, is_submask=False)

    @property
    def value(self):
        if not self.resolved_indexes.is_single_element:
            raise AttributeError("Only Scalars have `.value` attribute")
        scalar = self.parent._extract_element(self.resolved_indexes, name="s_extract")
        return scalar.value

    def new(self, *, dtype=None, mask=None, input_mask=None, name=None):
        """
        Force extraction of the indexes into a new object
        dtype and mask are the only controllable parameters.
        """
        if self.resolved_indexes.is_single_element:
            if mask is not None or input_mask is not None:
                raise TypeError("mask is not allowed for single element extraction")
            return self.parent._extract_element(self.resolved_indexes, dtype=dtype, name=name)
        else:
            if input_mask is not None:
                if mask is not None:
                    raise TypeError("mask and input_mask arguments cannot both be given")
                from .base import _check_mask

                _check_mask(input_mask, output=self.parent)
                mask = self._input_mask_to_mask(input_mask)
            delayed_extractor = self.parent._prep_for_extract(self.resolved_indexes)
            return delayed_extractor.new(dtype=dtype, mask=mask, name=name)

    def __eq__(self, other):
        if not self.resolved_indexes.is_single_element:
            raise TypeError(
                f"__eq__ not defined for objects of type {type(self)}.  "
                f"Use `.new()` to create a new object, then use `.isequal` method."
            )
        return self.value == other

    def __bool__(self):
        if not self.resolved_indexes.is_single_element:
            raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")
        return bool(self.value)

    def __float__(self):
        if not self.resolved_indexes.is_single_element:
            raise TypeError(f"__float__ not defined for objects of type {type(self)}.")
        return float(self.value)

    def __int__(self):
        if not self.resolved_indexes.is_single_element:
            raise TypeError(f"__int__ not defined for objects of type {type(self)}.")
        return int(self.value)

    __index__ = __int__

    def _extract_delayed(self):
        """Return an Expression object, treating this as an extract call"""
        return self.parent._prep_for_extract(self.resolved_indexes)

    def _input_mask_to_mask(self, input_mask):
        from .vector import Vector

        if type(input_mask.mask) is Vector and type(self.parent) is not Vector:
            (_, rowsize), (_, colsize) = self.resolved_indexes.indices
            if rowsize is None:
                if self.parent._ncols != input_mask.mask._size:
                    raise ValueError(
                        "Size of `input_mask` Vector does not match ncols of Matrix:\n"
                        f"{self.parent.name}.ncols != {input_mask.mask.name}.size  -->  "
                        f"{self.parent._ncols} != {input_mask.mask._size}"
                    )
                mask_expr = input_mask.mask._prep_for_extract(self.resolved_indexes.get_index(1))
            elif colsize is None:
                if self.parent._nrows != input_mask.mask._size:
                    raise ValueError(
                        "Size of `input_mask` Vector does not match nrows of Matrix:\n"
                        f"{self.parent.name}.nrows != {input_mask.mask.name}.size  -->  "
                        f"{self.parent._nrows} != {input_mask.mask._size}"
                    )
                mask_expr = input_mask.mask._prep_for_extract(self.resolved_indexes.get_index(0))
            else:
                raise TypeError(
                    "Got Vector `input_mask` when extracting a submatrix from a Matrix.  "
                    "Vector `input_mask` with a Matrix (or TransposedMatrix) input is "
                    "only valid when extracting from a single row or column."
                )
        elif input_mask.mask.shape != self.parent.shape:
            if type(self.parent) is Vector:
                attr = "size"
                shape1 = self.parent._size
                shape2 = input_mask.mask._size
            else:
                attr = "shape"
                shape1 = self.parent.shape
                shape2 = input_mask.mask.shape
            raise ValueError(
                f"{attr.capitalize()} of `input_mask` does not match {attr} of input:\n"
                f"{self.parent.name}.{attr} != {input_mask.mask.name}.{attr}  -->  "
                f"{shape1} != {shape2}"
            )
        else:
            mask_expr = input_mask.mask._prep_for_extract(self.resolved_indexes)
        mask_value = mask_expr.new(name="mask_temp")
        return type(input_mask)(mask_value)


class Updater:
    __slots__ = "parent", "kwargs", "__weakref__"

    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.kwargs = kwargs

    def __getitem__(self, keys):
        # Occurs when user calls `C(params)[index]`
        # Need something prepared to receive `<<` or `.update()`
        if self.parent._is_scalar:
            raise TypeError("Indexing not supported for Scalars")
        resolved_indexes = IndexerResolver(self.parent, keys)
        return Assigner(self, resolved_indexes, is_submask=False)

    def _setitem(self, resolved_indexes, obj, *, is_submask):
        # Occurs when user calls C(params)[index] = delayed
        if resolved_indexes.is_single_element and not self.kwargs:
            # Fast path using assignElement
            self.parent._assign_element(resolved_indexes, obj)
        else:
            mask = self.kwargs.get("mask")
            delayed = self.parent._prep_for_assign(
                resolved_indexes, obj, mask=mask, is_submask=is_submask
            )
            self.update(delayed)

    def __setitem__(self, keys, obj):
        if self.parent._is_scalar:
            raise TypeError("Indexing not supported for Scalars")
        resolved_indexes = IndexerResolver(self.parent, keys)
        self._setitem(resolved_indexes, obj, is_submask=False)

    def __delitem__(self, keys):
        # Occurs when user calls `del C(params)[index]`
        if self.parent._is_scalar:
            raise TypeError("Indexing not supported for Scalars")
        resolved_indexes = IndexerResolver(self.parent, keys)
        if resolved_indexes.is_single_element:
            self.parent._delete_element(resolved_indexes)
        else:
            raise TypeError("Remove Element only supports a single index")

    def __lshift__(self, delayed):
        # Occurs when user calls `C(params) << delayed`
        self.parent._update(delayed, **self.kwargs)

    def update(self, delayed):
        # Occurs when user calls `C(params).update(delayed)`
        self.parent._update(delayed, **self.kwargs)

    def __eq__(self, other):
        raise TypeError(f"__eq__ not defined for objects of type {type(self)}.")

    def __bool__(self):
        raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")


class InfixExprBase:
    __slots__ = "left", "right", "_value", "__weakref__"
    _is_scalar = False

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self._value = None

    def new(self, *, dtype=None, mask=None, name=None):
        if (
            mask is None
            and self._value is not None
            and (dtype is None or self._value.dtype == dtype)
        ):
            rv = self._value
            if name is not None:
                rv.name = name
            self._value = None
            return rv
        # Rely on the default operator for the method
        expr = getattr(self.left, self.method_name)(self.right)
        return expr.new(dtype=dtype, mask=mask, name=name)

    dup = new

    def _to_expr(self):
        # Rely on the default operator for the method
        return getattr(self.left, self.method_name)(self.right)

    def _format_expr(self):
        return f"{self.left.name} {self._infix} {self.right.name}"

    def _format_expr_html(self):
        return f"{self.left._name_html} {self._infix} {self.right._name_html}"

    def _repr_html_(self):
        from . import formatting

        if self.output_type.__name__ == "VectorExpression":
            return formatting.format_vector_infix_expression_html(self)
        elif self.output_type.__name__ == "MatrixExpression":
            return formatting.format_matrix_infix_expression_html(self)
        return formatting.format_scalar_infix_expression_html(self)

    def __repr__(self):
        from . import formatting

        if self.output_type.__name__ == "VectorExpression":
            return formatting.format_vector_infix_expression(self)
        elif self.output_type.__name__ == "MatrixExpression":
            return formatting.format_matrix_infix_expression(self)
        return formatting.format_scalar_infix_expression(self)

    @property
    def dtype(self):
        if self._value is not None:
            return self._value.dtype
        return self._to_expr().dtype


# Mistakes
utils._output_types[AmbiguousAssignOrExtract] = AmbiguousAssignOrExtract
utils._output_types[Assigner] = Assigner
utils._output_types[Updater] = Updater

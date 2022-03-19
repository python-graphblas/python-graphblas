import numpy as np

from . import lib, utils
from .dtypes import _INDEX
from .utils import _CArray, output_type


class _AllIndices:
    __slots__ = "_carg", "name", "_expr_name"

    def __init__(self):
        self._carg = lib.GrB_ALL
        self.name = "GrB_ALL"
        self._expr_name = ":"

    def __reduce__(self):
        return "_ALL_INDICES"


_ALL_INDICES = _AllIndices()


class AxisIndex:
    __slots__ = "size", "index", "cscalar", "dimsize"

    def __init__(self, size, index, cscalar, dimsize):
        self.size = size
        self.index = index
        self.cscalar = cscalar
        self.dimsize = dimsize

    @property
    def _carg(self):
        return self.index._carg

    @property
    def name(self):
        return self.index.name

    @property
    def _expr_name(self):
        if self.size is None:
            return f"{self.index.value}"
        idx = self._py_index()
        if type(idx) is slice:
            rv = f"{'' if idx.start is None else idx.start}:{'' if idx.stop is None else idx.stop}"
            if idx.step is not None:
                return f"{rv}:{idx.step}"
            return rv
        if idx.size < 6:
            return f"[{', '.join(map(str, idx))}]"
        else:
            return f"[{', '.join(map(str, idx[:3]))}, ...]"

    def _py_index(self):
        """Convert resolved index back into a valid Python index"""
        if self.size is None:
            return self.index.value
        if self.index is _ALL_INDICES:
            return slice(None)
        from ._slice import gxb_backwards, gxb_range, gxb_stride

        if self.cscalar is gxb_backwards:
            start, stop, step = self.index.array.tolist()
            size = self.dimsize
            stop -= size + 1
            step = -step
        elif self.cscalar is gxb_range:
            start, stop = self.index.array.tolist()
            step = None
            stop += 1
        elif self.cscalar is gxb_stride:
            start, stop, step = self.index.array.tolist()
            stop += 1
        else:
            return self.index.array
        if (
            start == 0
            and (step is None or step > 0)
            or start == self.dimsize - 1
            and step is not None
            and step < 0
        ):
            start = None
        if (
            stop == self.dimsize
            and (step is None or step > 0)
            or stop == -self.dimsize - 1
            and step is not None
            and step < 0
        ):
            stop = None
        return slice(start, stop, step)


class IndexerResolver:
    __slots__ = "obj", "indices", "shape", "__weakref__"

    def __init__(self, obj, indices):
        self.obj = obj
        if indices is Ellipsis:
            from .scalar import _as_scalar
            from .vector import Vector

            if type(obj) is Vector:
                self.indices = [
                    AxisIndex(
                        obj._size,
                        _ALL_INDICES,
                        _as_scalar(obj._size, _INDEX, is_cscalar=True),
                        obj._size,
                    )
                ]
                self.shape = (obj._size,)
            else:
                self.indices = [
                    AxisIndex(
                        obj._nrows,
                        _ALL_INDICES,
                        _as_scalar(obj._nrows, _INDEX, is_cscalar=True),
                        obj._nrows,
                    ),
                    AxisIndex(
                        obj._ncols,
                        _ALL_INDICES,
                        _as_scalar(obj._ncols, _INDEX, is_cscalar=True),
                        obj._ncols,
                    ),
                ]
                self.shape = (obj._nrows, obj._ncols)
        else:
            self.indices = self.parse_indices(indices, obj.shape)
            self.shape = tuple(index.size for index in self.indices if index.size is not None)

    @property
    def is_single_element(self):
        return not self.shape

    @property
    def py_indices(self):
        if self.obj.ndim > 1:
            return tuple(index._py_index() for index in self.indices)
        return self.indices[0]._py_index()

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
        from .scalar import _as_scalar

        if np.issubdtype(typ, np.integer):
            if index >= size:
                raise IndexError(f"Index out of range: index={index}, size={size}")
            if np.issubdtype(typ, np.signedinteger) and index < 0:
                index = index + size
                if index < 0:
                    raise IndexError(f"Index out of range: index={index - size}, size={size}")
            return AxisIndex(None, _as_scalar(int(index), _INDEX, is_cscalar=True), None, size)
        if typ is list:
            pass
        elif typ is slice:
            from ._slice import slice_to_index

            return slice_to_index(index, size)

        elif typ is np.ndarray:
            if len(index.shape) != 1:
                raise TypeError(f"Invalid number of dimensions for index: {len(index.shape)}")
            if not np.issubdtype(index.dtype, np.integer):
                raise TypeError(f"Invalid dtype for index: {index.dtype}")
            if np.issubdtype(index.dtype, np.signedinteger):
                is_negative = index < 0
                if is_negative.any():
                    index = np.where(is_negative, index + size, index)
                    if (index < 0).any():
                        bad_index = index[index < 0][0] - size
                        raise IndexError(f"Index out of range: index={bad_index}, size={size}")
            return AxisIndex(
                len(index), _CArray(index), _as_scalar(len(index), _INDEX, is_cscalar=True), size
            )
        else:
            from .scalar import Scalar

            if typ is Scalar:
                if not np.issubdtype(index.dtype.np_type, np.integer):
                    raise TypeError(f"An integer is required for indexing.  Got: {index.dtype}")
                value = int(index)
                if np.issubdtype(index.dtype.np_type, np.signedinteger) and value < 0:
                    value = value + size
                    if value < 0:
                        raise IndexError(f"Index out of range: index={value - size}, size={size}")
                return AxisIndex(None, _as_scalar(value, _INDEX, is_cscalar=True), None, size)

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
                    ) from None
                raise TypeError(
                    f"Invalid type for index: {typ}; unable to convert to list"
                ) from None
        return self.parse_index(np.array(index), np.ndarray, size)

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
    __slots__ = "parent", "resolved_indexes", "_value", "__weakref__"
    _is_scalar = False

    def __init__(self, parent, resolved_indexes):
        self.parent = parent
        self.resolved_indexes = resolved_indexes
        self._value = None

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

    def new(self, dtype=None, *, mask=None, input_mask=None, name=None):
        """
        Force extraction of the indexes into a new object
        dtype and mask are the only controllable parameters.
        """
        if input_mask is not None:
            if mask is not None:
                raise TypeError("mask and input_mask arguments cannot both be given")
            from .base import _check_mask

            _check_mask(input_mask, output=self.parent)
            mask = self._input_mask_to_mask(input_mask)
        delayed_extractor = self.parent._prep_for_extract(self.resolved_indexes)
        return delayed_extractor.new(dtype, mask=mask, name=name)

    dup = new

    def _extract_delayed(self):
        """Return an Expression object, treating this as an extract call"""
        return self.parent._prep_for_extract(self.resolved_indexes)

    def _input_mask_to_mask(self, input_mask):
        from .vector import Vector

        if type(input_mask.mask) is Vector and type(self.parent) is not Vector:
            rowidx, colidx = self.resolved_indexes.indices
            if rowidx.size is None:
                if self.parent._ncols != input_mask.mask._size:
                    raise ValueError(
                        "Size of `input_mask` Vector does not match ncols of Matrix:\n"
                        f"{self.parent.name}.ncols != {input_mask.mask.name}.size  -->  "
                        f"{self.parent._ncols} != {input_mask.mask._size}"
                    )
                mask_expr = input_mask.mask._prep_for_extract(self.resolved_indexes.get_index(1))
            elif colidx.size is None:
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

    def _repr_html_(self):
        from . import formatting

        return formatting.format_index_expression_html(self)

    def __repr__(self):
        from . import formatting

        return formatting.format_index_expression(self)

    @property
    def dtype(self):
        return self.parent.dtype


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
        # Occurs when user calls C(params)[index] = expr
        if resolved_indexes.is_single_element and not self.kwargs:
            # Fast path using assignElement
            self.parent._assign_element(resolved_indexes, obj)
        else:
            mask = self.kwargs.get("mask")
            expr = self.parent._prep_for_assign(
                resolved_indexes, obj, mask=mask, is_submask=is_submask
            )
            self.update(expr)

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
            # Delete selection by assigning an empty scalar
            from .scalar import Scalar

            scalar = Scalar.new(
                self.parent.dtype, is_cscalar=False, name="s_empty"  # pragma: is_grbscalar
            )
            self._setitem(resolved_indexes, scalar, is_submask=False)

    def __lshift__(self, expr):
        # Occurs when user calls `C(params) << expr`
        self.parent._update(expr, **self.kwargs)

    def update(self, expr):
        # Occurs when user calls `C(params).update(expr)`
        self.parent._update(expr, **self.kwargs)

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

    def new(self, dtype=None, *, mask=None, name=None):
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
        expr = self._to_expr()
        return expr.new(dtype, mask=mask, name=name)

    dup = new

    def _to_expr(self):
        if self._value is None:
            # Rely on the default operator for `x @ y`
            self._value = getattr(self.left, self.method_name)(self.right)
        return self._value

    def _get_value(self, attr=None, default=None):
        expr = self._to_expr()
        return expr._get_value(attr=attr, default=default)

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

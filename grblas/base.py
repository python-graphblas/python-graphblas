from . import lib, ffi
from . import dtypes, ops, descriptor
from .exceptions import check_status
from .ops import OpBase

NULL = ffi.NULL


class Updater:
    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.kwargs = kwargs

    def __getitem__(self, keys):
        # Occurs when user calls C(params)[index]; need something prepared to receive `<<` or `.update()`
        if self.parent.is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        if type(keys) == IndexerResolver:
            resolved_indexes = keys
        else:
            resolved_indexes = IndexerResolver(self.parent, keys)
        return AmbiguousAssignOrExtract(self, resolved_indexes)

    def __setitem__(self, keys, obj):
        # Occurs when user calls C(params)[index] = delayed
        if self.parent.is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        if type(keys) == IndexerResolver:
            resolved_indexes = keys
        else:
            resolved_indexes = IndexerResolver(self.parent, keys)

        if resolved_indexes.is_single_element() and not self.kwargs:
            # Fast path using assignElement
            self.parent._assign_element(resolved_indexes, obj)
        else:
            delayed = self.parent._prep_for_assign(resolved_indexes, obj)
            self.update(delayed)

    def __lshift__(self, delayed):
        # Occurs when user calls C(params) << delayed
        self.parent._update(delayed, **self.kwargs)

    def update(self, delayed):
        # Occurs when user calls C(params).update(delayed)
        self.parent._update(delayed, **self.kwargs)


class GbContainer:
    # Flag for operations which depend on scalar vs vector/matrix
    is_scalar = False

    def __init__(self, gb_obj, dtype):
        if not isinstance(gb_obj, ffi.CData):
            raise TypeError('Object passed to __init__ must be CData type')
        if not isinstance(dtype, dtypes.DataType):
            dtype = dtypes.lookup(dtype)
        
        self.gb_obj = gb_obj
        self.dtype = dtype

    def __invert__(self):
        return ComplementedMask(self)

    def __delitem__(self, keys):
        if self.is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        raise NotImplementedError('Not available until GraphBLAS v1.3')

    def __getitem__(self, keys):
        if self.is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        resolved_indexes = IndexerResolver(self, keys)
        return AmbiguousAssignOrExtract(self, resolved_indexes)

    def __setitem__(self, keys, delayed):
        Updater(self)[keys] = delayed

    def __call__(self, *optional_mask_and_accum, mask=None, accum=None, replace=False):
        # Pick out mask and accum from positional arguments
        mask_arg, accum_arg = None, None
        for key in optional_mask_and_accum:
            if isinstance(key, GbContainer):
                mask_arg = key
            elif type(key) == ComplementedMask:
                mask_arg = key
            elif isinstance(key, ops.BinaryOp):
                accum_arg = key
            elif type(key) == ffi.CData and ops.find_opclass(key) != ops.UNKNOWN_OPCLASS:
                accum_arg = key
            else:
                raise TypeError(f'Invalid item found in output params: {type(key)}')
        # Merge positional and keyword arguments
        if mask_arg is not None and mask is not None:
            raise TypeError("got multiple values for argument 'mask'")
        if mask_arg is not None:
            mask = mask_arg
        if mask is None:
            mask = NULL
        if accum_arg is not None and accum is not None:
            raise TypeError("got multiple values for argument 'accum")
        if accum_arg is not None:
            accum = accum_arg
        if accum is None:
            accum = NULL
        return Updater(self, mask=mask, accum=accum, replace=replace)

    def __lshift__(self, delayed):
        return self._update(delayed)

    def update(self, delayed):
        """
        Convenience function when no output arguments (mask, accum, replace) are used
        """
        return self._update(delayed)

    def _update(self, delayed, mask=NULL, accum=NULL, replace=False):
        # TODO: check expected output type (need to include in GbDelayed object)
        if not isinstance(delayed, GbDelayed):
            from .ops import UnaryOp
            from .matrix import Matrix, TransposedMatrix
            if type(delayed) == AmbiguousAssignOrExtract:
                # Extract (C << A[rows, cols])
                delayed = delayed._extract_delayed()
            elif type(delayed) == self.__class__:
                # Simple assignment (w << v)
                delayed = delayed.apply(UnaryOp.IDENTITY)
            elif type(delayed) == TransposedMatrix and type(self) == Matrix:
                # Transpose (C << A.T)
                delayed = GbDelayed(lib.GrB_transpose, [delayed.gb_obj[0]])
            else:
                raise TypeError(f'assignment value must be GbDelayed object, not {type(delayed)}')

        # Normalize mask and separate out complement flag
        complement = False
        if mask is NULL:
            pass
        elif isinstance(mask, GbContainer):
            mask = mask.gb_obj[0]
        elif type(mask) == ComplementedMask:
            mask = mask.mask.gb_obj[0]
            complement = True
        else:
            raise TypeError(f"Invalid mask: {type(mask)}")

        # Normalize accumulator
        if accum is NULL:
            pass
        elif isinstance(accum, ops.BinaryOp):
            accum = accum[self.dtype]
        elif type(accum) == ffi.CData and ops.find_opclass(accum) != ops.UNKNOWN_OPCLASS:
            pass
        else:
            raise TypeError(f"Invalid accum: {type(accum)}")

        # Build descriptor based on flags
        desc = descriptor.build(transpose_first=delayed.at,
                                transpose_second=delayed.bt,
                                mask_complement=complement,
                                output_replace=replace)

        # Resolve any ops in tail_args
        tail_args = [x[self.dtype] if isinstance(x, OpBase) else x
                     for x in delayed.tail_args]

        # Build args and call GraphBLAS function
        if self.is_scalar:
            if mask is not NULL:
                raise TypeError('Mask not allowed for Scalars')
            call_args = [self.gb_obj, accum] + tail_args + [desc]
            # Ensure the scalar isn't flagged as empty after the update
            self.empty = False
        else:
            call_args = [self.gb_obj[0], mask, accum] + tail_args + [desc]

        # Make the GraphBLAS call
        check_status(delayed.func(*call_args))

    def _extract_element(self, resolved_indexes):
        raise TypeError(f'Cannot extract from {self.__class__.__name__}')

    def _prep_for_extract(self, resolved_indexes):
        raise TypeError(f'Cannot extract from {self.__class__.__name__}')

    def _assign_element(self, resolved_indexes, value):
        raise TypeError(f'Cannot assign to {self.__class__.__name__}')

    def _prep_for_assign(self, resolved_indexes, obj):
        raise TypeError(f'Cannot assign to {self.__class__.__name__}')

    @classmethod
    def new_from_existing(cls, obj):
        raise NotImplementedError()

    def dup(self):
        return type(self).new_from_existing(self)

    def show(self):
        from . import io
        return io.show(self)


class GbDelayed:
    def __init__(self, func, tail_args, at=False, bt=False, output_constructor=None):
        """
        func: the GraphBLAS function to call
        tail_args: arguments specific to func other than the standard INOUT, mask, accum, and desc
        at: (bool) whether first input argument (A) is transposed
        bt: (bool) whether second input argument (B) is transposed
        output_constructor: functools.partial, must be callable with no additional arguments to
                            create an output object from GbDelayed; used when delayed.new() is called
        """
        self.func = func
        self.tail_args = tail_args
        self.at = at
        self.bt = bt
        self.output_constructor = output_constructor

    def __repr__(self):
        return f'GbDelayed<{self.func.__name__}>'

    def new(self, *, dtype=None, mask=None):
        """
        Force computation of the GbDelayed object.
        dtype and mask are the only controllable parameters.
        """
        if self.output_constructor is None:
            raise Exception('output_constructor was not defined. Unable to use `new` method.')
        if dtype is not None:
            if 'dtype' not in self.output_constructor.keywords:
                raise Exception('output_constructor does not use `dtype`; invalid to specify for this usage')
            output = self.output_constructor(dtype=dtype)
        else:
            output = self.output_constructor()
        if mask is None:
            output.update(self)
        else:
            if not isinstance(mask, output.__class__):
                raise TypeError(f'Mask must be type {output.__class__}')
            output(mask).update(self)
        return output


class AmbiguousAssignOrExtract:
    def __init__(self, parent, resolved_indexes):
        self.parent = parent
        self.resolved_indexes = resolved_indexes

    def __call__(self, *args, **kwargs):
        # Occurs when user calls C[index](params)
        # Reverse the call order so we can parse the call args and kwargs
        updater = self.parent(*args, **kwargs)
        return updater[self.resolved_indexes]

    def __lshift__(self, obj):
        # Occurs when user calls C(params)[index] << obj or C[index] << obj
        # Delegate back to parent's __setitem__ method
        self.parent[self.resolved_indexes] = obj

    def update(self, obj):
        # Occurs when user calls C(params)[index].update(obj) or C[index].update(obj)
        # Delegate back to parent's __setitem__ method
        self.parent[self.resolved_indexes] = obj

    @property
    def value(self):
        if isinstance(self.parent, Updater):
            raise TypeError('Cannot extract from an Updater')
        if not self.resolved_indexes.is_single_element():
            raise AttributeError("Only Scalars have `.value` attribute")
        val, _ = self.parent._extract_element(self.resolved_indexes)
        return val

    def new(self, *, dtype=None, mask=None):
        """
        Force extraction of the indexes into a new object
        dtype and mask are the only controllable parameters.
        """
        if isinstance(self.parent, Updater):
            raise TypeError('Cannot extract from an Updater')
        if self.resolved_indexes.is_single_element():
            if mask is not None:
                raise TypeError('mask is not allowed for single element extraction')
            val, cur_dtype = self.parent._extract_element(self.resolved_indexes)
            if dtype is None:
                dtype = cur_dtype
            from .scalar import Scalar
            return Scalar.new_from_value(val, dtype=dtype)
        else:
            delayed_extractor = self.parent._prep_for_extract(self.resolved_indexes)
            return delayed_extractor.new(dtype=dtype, mask=mask)

    def _extract_delayed(self):
        """Return a GbDelayed object, treating this as an extract call"""
        if isinstance(self.parent, Updater):
            raise TypeError('Cannot extract from an Updater')
        if self.resolved_indexes.is_single_element():
            raise TypeError('extract and update is not allowed for single element')
        return self.parent._prep_for_extract(self.resolved_indexes)


class IndexerResolver:
    def __init__(self, obj, indices):
        if obj.is_scalar:
            raise TypeError("Cannot index into Scalars")
        self.obj = obj
        self.indices = self.parse_indices(indices, obj.shape)

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
            if type(indices) == tuple:
                raise TypeError(f'Index for {self.obj.__class__.__name__} cannot be a tuple')
            # Convert to tuple for consistent processing
            indices = (indices,)
        elif len(shape) == 2:
            if type(indices) != tuple or len(indices) != 2:
                raise TypeError(f'Index for {self.obj.__class__.__name__} must be a 2-tuple')

        out = []
        for i, idx in enumerate(indices):
            typ = type(idx)
            if type == tuple:
                raise TypeError(f'Index in position {i} cannot be a tuple; must use slice or list or int')
            out.append(self.parse_index(idx, typ, shape[i]))
        return out

    def parse_index(self, index, typ, size):
        if typ == int:
            if index >= size:
                raise IndexError(f'index={index}, size={size}')
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
                raise TypeError('Unable to convert to tuple')
        return ffi.new('GrB_Index[]', index), len(index)


class ComplementedMask:
    def __init__(self, mask):
        self.mask = mask

    def __invert__(self):
        return self.mask

    def __repr__(self):
        return f'MaskComplement of {self.mask}'

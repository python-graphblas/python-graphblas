from . import lib, ffi
from . import dtypes, ops, descriptor, unary
from .exceptions import check_status
from .mask import Mask

NULL = ffi.NULL


class Updater:
    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.kwargs = kwargs

    def __getitem__(self, keys):
        # Occurs when user calls C(params)[index]; need something prepared to receive `<<` or `.update()`
        if self.parent._is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        if type(keys) is IndexerResolver:
            resolved_indexes = keys
        else:
            resolved_indexes = IndexerResolver(self.parent, keys)
        return AmbiguousAssignOrExtract(self, resolved_indexes)

    def __setitem__(self, keys, obj):
        # Occurs when user calls C(params)[index] = delayed
        if self.parent._is_scalar:
            raise TypeError('Indexing not supported for Scalars')
        if type(keys) is IndexerResolver:
            resolved_indexes = keys
        else:
            resolved_indexes = IndexerResolver(self.parent, keys)

        if resolved_indexes.is_single_element and not self.kwargs:
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
    _is_scalar = False

    def __init__(self, gb_obj, dtype):
        if not isinstance(gb_obj, ffi.CData):
            raise TypeError('Object passed to __init__ must be CData type')
        if not isinstance(dtype, dtypes.DataType):
            dtype = dtypes.lookup(dtype)
        self.gb_obj = gb_obj
        self.dtype = dtype

    def __call__(self, *optional_mask_and_accum, mask=None, accum=None, replace=False):
        # Pick out mask and accum from positional arguments
        mask_arg, accum_arg = None, None
        for arg in optional_mask_and_accum:
            if isinstance(arg, GbContainer):
                raise TypeError('Mask must indicate values (M.V) or structure (M.S)')
            elif isinstance(arg, Mask):
                mask_arg = arg
            else:
                accum_arg, opclass = ops.find_opclass(arg)
                if opclass == ops.UNKNOWN_OPCLASS:
                    raise TypeError(f'Invalid item found in output params: {type(arg)}')
                if opclass != 'BinaryOp':
                    raise TypeError(f'accum must be a BinaryOp, not {opclass}')
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

    def _update(self, delayed, mask=None, accum=None, replace=False):
        if mask is None:
            mask = NULL
        if accum is None:
            accum = NULL
        # TODO: check expected output type (need to include in GbDelayed object)
        if self._is_scalar and mask is not NULL:
            raise TypeError('Mask not allowed for Scalars')
        if not isinstance(delayed, GbDelayed):
            from .matrix import Matrix, TransposedMatrix
            if type(delayed) is AmbiguousAssignOrExtract:
                if delayed.resolved_indexes.is_single_element and self._is_scalar:
                    # Extract element (s << v[1])
                    if accum is not NULL:
                        raise TypeError(
                            'Scalar accumulation with extract element'
                            '--such as `s(accum=accum) << v[0]`--is not supported'
                        )
                    self.value = delayed.new(dtype=self.dtype).value
                    return

                # Extract (C << A[rows, cols])
                delayed = delayed._extract_delayed()
            elif type(delayed) is self.__class__:
                # Simple assignment (w << v)
                if self._is_scalar:
                    if accum is not NULL:
                        raise TypeError(
                            'Scalar update with accumulation--such as `s(accum=accum) << t`--is not supported'
                        )
                    self.value = delayed.value
                    return

                delayed = delayed.apply(unary.identity)
            elif type(delayed) is TransposedMatrix and type(self) is Matrix:
                # Transpose (C << A.T)
                delayed = GbDelayed(lib.GrB_transpose, [delayed.gb_obj[0]])
            elif self._is_scalar:
                if accum is not NULL:
                    raise TypeError('Scalar update with accumulation--such as `s(accum=accum) << t`--is not supported')
                self.value = delayed
                return

            else:
                raise TypeError(f'assignment value must be GbDelayed object, not {type(delayed)}')

        # Normalize mask and separate out complement and structural flags
        complement = False
        structure = False
        if mask is NULL:
            pass
        elif isinstance(mask, GbContainer):
            raise TypeError('Mask must indicate values (M.V) or structure (M.S)')
        elif isinstance(mask, Mask):
            if not mask.value and not mask.structure:
                raise TypeError('Mask must indicate values (M.V) or structure (M.S)')
            complement = mask.complement
            structure = mask.structure
            mask = mask.mask.gb_obj[0]
        else:
            raise TypeError(f"Invalid mask: {type(mask)}")

        # Normalize accumulator
        if accum is not NULL:
            orig_accum = accum
            accum = ops.get_typed_op(accum, self.dtype)
            if accum.opclass != 'BinaryOp':
                raise TypeError(f'accum must be a BinaryOp, not {accum.opclass}')
            accum = accum.gb_obj

        # Get descriptor based on flags
        desc = descriptor.lookup(transpose_first=delayed.at,
                                 transpose_second=delayed.bt,
                                 mask_complement=complement,
                                 mask_structure=structure,
                                 output_replace=replace)

        # Build args and call GraphBLAS function
        if self._is_scalar:
            temp_result = delayed.output_constructor()
            if self.dtype != temp_result.dtype:
                if accum is not NULL:
                    temp_result = self.dup(dtype=temp_result.dtype)
                    temp_result(accum=orig_accum).update(delayed)
                else:
                    temp_result.update(delayed)
                self.value = temp_result.value
                return

            call_args = [self.gb_obj, accum] + delayed.tail_args + [desc]
            # Ensure the scalar isn't flagged as empty after the update
            self._is_empty = False
        else:
            call_args = [self.gb_obj[0], mask, accum] + delayed.tail_args + [desc]

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

    @property
    def value(self):
        from .scalar import Scalar
        output = self.new()
        if type(output) is not Scalar:
            raise ValueError(f"`.value` is only valid for Scalars, not {type(output)}")
        return output.value

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
            if isinstance(mask, Mask) and not isinstance(mask.mask, output.__class__):
                raise TypeError(f'Mask object must be type {output.__class__}')
            output(mask).update(self)
        return output


class AmbiguousAssignOrExtract:
    def __init__(self, parent, resolved_indexes):
        self.parent = parent
        self.resolved_indexes = resolved_indexes

    def __call__(self, *args, **kwargs):
        if type(self.parent) is Updater:
            parent_kwargs = []
            if self.parent.kwargs['accum'] is not NULL:
                parent_kwargs.append(f"accum={self.parent.kwargs['accum']}")
            if self.parent.kwargs['mask'] is not NULL:
                # It would sure be nice if we knew the mask type.
                # Passing around C objects directly is sometimes inconvenient.
                parent_kwargs.append("mask=<Mask>")
                parent_kwargs.append(f"replace={self.parent.kwargs['replace']}")
            if not parent_kwargs:
                raise ValueError(f'GraphBLAS object already called (with no keywords)')
            parent_kwargs = ', '.join(parent_kwargs)
            raise ValueError(f'GraphBLAS object already called with keywords: {parent_kwargs}')
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
        if not self.resolved_indexes.is_single_element:
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
        if self.resolved_indexes.is_single_element:
            if mask is not None:
                raise TypeError('mask is not allowed for single element extraction')
            val, cur_dtype = self.parent._extract_element(self.resolved_indexes)
            if dtype is None:
                dtype = cur_dtype
            from .scalar import Scalar
            return Scalar.from_value(val, dtype=dtype)
        else:
            delayed_extractor = self.parent._prep_for_extract(self.resolved_indexes)
            return delayed_extractor.new(dtype=dtype, mask=mask)

    def _extract_delayed(self):
        """Return a GbDelayed object, treating this as an extract call"""
        if isinstance(self.parent, Updater):
            raise TypeError('Cannot extract from an Updater')
        return self.parent._prep_for_extract(self.resolved_indexes)


class IndexerResolver:
    def __init__(self, obj, indices):
        if obj._is_scalar:
            raise TypeError("Cannot index into Scalars")
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
                raise TypeError(f'Index for {self.obj.__class__.__name__} cannot be a tuple')
            # Convert to tuple for consistent processing
            indices = (indices,)
        elif len(shape) == 2:
            if type(indices) is not tuple or len(indices) != 2:
                raise TypeError(f'Index for {self.obj.__class__.__name__} must be a 2-tuple')

        out = []
        for i, idx in enumerate(indices):
            typ = type(idx)
            if typ is tuple:
                raise TypeError(f'Index in position {i} cannot be a tuple; must use slice or list or int')
            out.append(self.parse_index(idx, typ, shape[i]))
        return out

    def parse_index(self, index, typ, size):
        if typ is int:
            if index >= size:
                raise IndexError(f'index={index}, size={size}')
            return index, None
        if typ is slice:
            if index == slice(None):
                # [:] means all indices; use special GrB_ALL indicator
                return lib.GrB_ALL, size
            index = tuple(range(size)[index])
        elif typ is not list:
            try:
                index = tuple(index)
            except Exception:
                raise TypeError('Unable to convert to tuple')
        return ffi.new('GrB_Index[]', index), len(index)

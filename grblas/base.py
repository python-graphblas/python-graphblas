from . import lib, ffi
from . import dtypes, ops, descriptor
from .exceptions import check_status
from .ops import OpBase

NULL = ffi.NULL

# Sentinel object to indicate GrB_OUTP = GrB_REPLACE in the descriptor
REPLACE = object()


class GbContainer:
    def __init__(self, gb_obj, dtype):
        if not isinstance(gb_obj, ffi.CData):
            raise TypeError('Object passed to __init__ must be CData type')
        if not isinstance(dtype, dtypes.DataType):
            dtype = dtypes.lookup(dtype)
        
        self.gb_obj = gb_obj
        self.dtype = dtype

    def __invert__(self):
        return ComplementedMask(self)

    def __setitem__(self, keys, delayed):
        # TODO: check expected output type (need to include in GbDelayed object)
        if not isinstance(delayed, GbDelayed):
            from .ops import UnaryOp
            from .matrix import Matrix, TransposedMatrix
            if type(delayed) == self.__class__:
                # Simple assignment (w[:] = v)
                delayed = delayed.apply(UnaryOp.IDENTITY)
            elif type(delayed) == TransposedMatrix and type(self) == Matrix:
                # Transpose (C[:] = A.T)
                delayed = GbDelayed(lib.GrB_transpose, [delayed.gb_obj[0]])
            else:
                raise TypeError(f'assignment value must be GbDelayed object, not {type(delayed)}')
        if type(keys) != tuple:
            keys = (keys,)
        mask, accum, complement, replace = self._parse_keys(keys)
        desc = descriptor.build(transpose_first=delayed.at,
                                transpose_second=delayed.bt,
                                mask_complement=complement,
                                output_replace=replace)
        # Resolve any ops in tail_args
        tail_args = [x[self.dtype] if isinstance(x, OpBase) else x
                     for x in delayed.tail_args]

        # Build args and call GraphBLAS function
        if self.can_mask:
            call_args = [self.gb_obj[0], mask, accum] + tail_args + [desc]
        else:
            if mask is not NULL:
                raise TypeError('Mask not allowed')
            call_args = [self.gb_obj, accum] + tail_args + [desc]
        # Make the GraphBLAS call
        check_status(delayed.func(*call_args))

    def _parse_keys(self, keys):
        """
        Returns mask, accum, complement, replace based on the keys
        """
        mask = NULL
        accum = NULL
        complement = False
        replace = False

        for key in keys:
            if key is REPLACE:
                replace = True
            elif isinstance(key, GbContainer):
                mask = key.gb_obj[0]
            elif type(key) == ComplementedMask:
                mask = key.mask.gb_obj[0]
                complement = True
            elif isinstance(key, slice):
                if key == slice(None, None, None):
                    continue
                else:
                    raise TypeError('Only the "ALL" slice [:] is allowed, to indicate no mask')
            elif isinstance(key, ops.BinaryOp):
                accum = key[self.dtype]
            elif type(key) == ffi.CData and ops.find_opclass(key) != ops.UNKNOWN_OPCLASS:
                accum = key
            else:
                raise TypeError(f'Invalid item found in output params: {type(key)}')

        return mask, accum, complement, replace


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

    def new(self, mask=None):
        if self.output_constructor is None:
            raise Exception('output_constructor was not defined. Unable to use `new` method.')
        output = self.output_constructor()
        if mask is None:
            mask = slice(None)  # [:] indicates no mask
        elif not isinstance(mask, output.__class__):
            raise TypeError(f'Mask must be type {output.__class__}')
        output[mask] = self
        return output


class ComplementedMask:
    def __init__(self, mask):
        self.mask = mask

    def __invert__(self):
        return self.mask

    def __repr__(self):
        return f'MaskComplement of {self.mask}'

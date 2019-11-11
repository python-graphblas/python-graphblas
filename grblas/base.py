from _grblas import lib, ffi
from . import dtypes, ops, descriptor
from .exceptions import check_status

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
        # TODO: We should allow C[:] = A (simple assignment) or C[:] = A.T (transpose)
        # TODO: check expected output type (need to include in GbDelayed object)
        if not isinstance(delayed, GbDelayed):
            raise TypeError('assignment value must be GbDelayed object')
        if type(keys) != tuple:
            keys = (keys,)
        mask, accum, complement, replace = self._parse_keys(keys)
        if accum is not NULL:
            accum = accum[self.dtype]
        desc = descriptor.build(transpose_first=delayed.at,
                                transpose_second=delayed.bt,
                                mask_complement=complement,
                                output_replace=replace)

        # Build args and call GraphBLAS function
        if self.can_mask:
            call_args = [self.gb_obj[0], mask, accum] + delayed.tail_args + [desc]
        else:
            if mask is not NULL:
                raise TypeError('Mask not allowed')
            call_args = [self.gb_obj, accum] + delayed.tail_args + [desc]
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
            elif isinstance(key, ops.BinaryOp):
                accum = key
            elif isinstance(key, slice):
                if key == slice(None, None, None):
                    continue
                else:
                    raise TypeError('Only the "ALL" slice [:] is allowed, to indicate no mask')
            else:
                raise TypeError(f'Invalid item found in output params: {type(key)}')

        return mask, accum, complement, replace


class GbDelayed:
    def __init__(self, func, tail_args, at=False, bt=False):
        self.func = func
        self.tail_args = tail_args
        self.at = at
        self.bt = bt

    def __repr__(self):
        return f'GbDelayed<{self.func.__name__}>'


class ComplementedMask:
    def __init__(self, mask):
        self.mask = mask

    def __invert__(self):
        return self.mask

    def __repr__(self):
        return f'MaskComplement of {self.mask}'

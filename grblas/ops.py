import re
import numba
from types import FunctionType
from . import lib, ffi, dtypes, unary, binary, monoid, semiring
from .exceptions import GrblasException


UNKNOWN_OPCLASS = 'UnknownOpClass'


class UdfParseError(GrblasException):
    pass


class OpBase:
    _parse_config = None
    _initialized = False
    _module = None

    def __init__(self, name):
        self.name = name
        self._specific_types = {}

    def __repr__(self):
        return f'{type(self).__name__}.{self.name}'

    def __getitem__(self, type_):
        type_ = self._normalize_type(type_)
        if type_ not in self._specific_types:
            raise KeyError(f'{self.name} does not work with {type_}')
        return self._specific_types[type_]

    def __setitem__(self, type_, obj):
        type_ = self._normalize_type(type_)
        self._specific_types[type_] = obj

    def __delitem__(self, type_):
        type_ = self._normalize_type(type_)
        del self._specific_types[type_]

    def __contains__(self, type_):
        type_ = self._normalize_type(type_)
        return type_ in self._specific_types

    def _normalize_type(self, type_):
        return type_.name if isinstance(type_, dtypes.DataType) else type_

    @property
    def types(self):
        return set(self._specific_types)

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        # Read in the parse configs
        trim_from_front = cls._parse_config.get('trim_from_front', 0)
        trim_from_back = cls._parse_config.get('trim_from_back', None)
        if trim_from_back is not None:
            trim_from_back = -trim_from_back
        num_underscores = cls._parse_config['num_underscores']

        for re_str, returns_bool in (('re_exprs', False),
                                     ('re_exprs_return_bool', True)):
            if re_str not in cls._parse_config:
                continue
            for r in cls._parse_config[re_str]:
                for varname in dir(lib):
                    m = r.match(varname)
                    if m:
                        # Parse function into name and datatype
                        splitname = m.string[trim_from_front:trim_from_back].split('_')
                        if len(splitname) == num_underscores + 1:
                            *splitname, type_ = splitname
                        else:
                            type_ = 'BOOL'
                        name = '_'.join(splitname).lower()
                        # Create object for name unless it already exists
                        if not hasattr(cls._module, name):
                            setattr(cls._module, name, cls(name))
                        obj = getattr(cls._module, name)
                        gb_obj = getattr(lib, varname)
                        obj[type_] = gb_obj
                        # Add to map of return types
                        _return_type[gb_obj] = 'BOOL' if returns_bool else type_
        cls._initialized = True


class UnaryOp(OpBase):
    _module = unary
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GrB_(IDENTITY|AINV|MINV)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(ABS|LNOT|ONE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GrB_LNOT$'),
        ],
    }

    @classmethod
    def register_new(cls, name, func):
        if type(func) is not FunctionType:
            raise TypeError(f'udf must be a function, not {type(func)}')
        if hasattr(cls, name):
            raise AttributeError(f'unary.{name} is already defined')
        success = False
        new_type_obj = cls(name)
        for type_, sample_val in dtypes._sample_values.items():
            # Check if func can handle this data type
            try:
                ret = func(sample_val)
                if type(ret) is bool:
                    ret_type = dtypes.BOOL
                elif type_ == 'BOOL':
                    # type_ == bool, but return type != bool; invalid
                    continue
                else:
                    ret_type = type_

                nt = numba.types
                # JIT the func so it can be used from a cfunc
                unary_udf = numba.njit(func)
                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(nt.CPointer(ret_type.numba_type),
                                      nt.CPointer(type_.numba_type))

                @numba.cfunc(wrapper_sig, nopython=True)
                def unary_wrapper(z, x):
                    result = unary_udf(x[0])
                    z[0] = result

                new_unary = ffi.new('GrB_UnaryOp*')
                lib.GrB_UnaryOp_new(new_unary, unary_wrapper.cffi,
                                    ret_type.gb_type, type_.gb_type)
                new_type_obj[type_.name] = new_unary[0]
                _return_type[new_unary[0]] = ret_type.name
                success = True
            except Exception:
                continue
        if success:
            setattr(cls._module, name, new_type_obj)
        else:
            raise UdfParseError('Unable to parse function using Numba')


class BinaryOp(OpBase):
    _module = binary
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GrB_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|TIMES|DIV)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GrB_(LOR|LAND|LXOR)$'),
            re.compile('^GxB_(RMINUS|RDIV|PAIR|ANY|ISEQ|ISNE|ISGT|ISLT|ISLE|ISGE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
        're_exprs_return_bool': [
            re.compile('^GrB_(EQ|NE|GT|LT|GE|LE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(LOR|LAND|LXOR)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
    }

    @classmethod
    def register_new(cls, name, func):
        if type(func) is not FunctionType:
            raise TypeError(f'udf must be a function, not {type(func)}')
        if hasattr(cls, name):
            raise AttributeError(f'unary.{name} is already defined')
        success = False
        new_type_obj = cls(name)
        for type_, sample_val in dtypes._sample_values.items():
            # Check if func can handle this data type
            try:
                ret = func(sample_val, sample_val)
                if type(ret) is bool:
                    ret_type = dtypes.BOOL
                elif type_ == 'BOOL':
                    # type_ == bool, but return type != bool; invalid
                    continue
                else:
                    ret_type = type_

                nt = numba.types
                # JIT the func so it can be used from a cfunc
                binary_udf = numba.njit(func)
                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(nt.CPointer(ret_type.numba_type),
                                      nt.CPointer(type_.numba_type),
                                      nt.CPointer(type_.numba_type))

                @numba.cfunc(wrapper_sig, nopython=True)
                def binary_wrapper(z, x, y):
                    result = binary_udf(x[0], y[0])
                    z[0] = result

                new_binary = ffi.new('GrB_BinaryOp*')
                lib.GrB_BinaryOp_new(new_binary, binary_wrapper.cffi,
                                     ret_type.gb_type, type_.gb_type, type_.gb_type)
                new_type_obj[type_.name] = new_binary[0]
                _return_type[new_binary[0]] = ret_type.name
                success = True
            except Exception:
                continue
        if success:
            setattr(cls._module, name, new_type_obj)
        else:
            raise UdfParseError('Unable to parse function using Numba')

    @classmethod
    def _initialize(cls):
        super()._initialize()
        # Rename div to cdiv
        binary.cdiv = BinaryOp('cdiv')
        for dtype in binary.div.types:
            binary.cdiv[dtype] = binary.div[dtype]
        del binary.div
        # Add truediv which always points to floating point cdiv
        # We are effectively hacking cdiv to always return floating point values
        # If the inputs are FP32, we use DIV_FP32; use DIV_FP64 for all other input dtypes
        binary.truediv = BinaryOp('truediv')
        for dtype in binary.cdiv.types:
            float_type = 'FP32' if dtype == 'FP32' else 'FP64'
            binary.truediv[dtype] = binary.cdiv[float_type]
        # Add floordiv
        # cdiv truncates towards 0, while floordiv truncates towards -inf
        BinaryOp.register_new('floordiv', lambda x, y: x // y)


class Monoid(OpBase):
    _module = monoid
    _parse_config = {
        'trim_from_front': 4,
        'trim_from_back': 7,
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GxB_(MAX|MIN|PLUS|TIMES|ANY)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)_MONOID$'),
            re.compile('^GxB_(EQ|LAND|LOR|LXOR|ANY)_BOOL_MONOID$'),
        ],
    }

    @classmethod
    def register_new(cls, name, binaryop, zero):
        if type(binaryop) is not BinaryOp:
            raise TypeError(f'binaryop must be a BinaryOp, not {type(binaryop)}')
        new_type_obj = cls(name)
        for type_ in binaryop.types:
            type_ = dtypes.lookup(type_)
            new_monoid = ffi.new('GrB_Monoid*')
            func = getattr(lib, f'GrB_Monoid_new_{type_.name}')
            zcast = ffi.cast(type_.c_type, zero)
            func(new_monoid, binaryop[type_], zcast)
            new_type_obj[type_.name] = new_monoid[0]
            ret_type = find_return_type(binaryop[type_])
            _return_type[new_monoid[0]] = ret_type
        setattr(cls._module, name, new_type_obj)


class Semiring(OpBase):
    _module = semiring
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 2,
        're_exprs': [
            re.compile('^GxB_(MIN|MAX|PLUS|TIMES|ANY)_(FIRST|SECOND|PAIR|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(LOR|LAND|LXOR|EQ|ANY)_(FIRST|SECOND|PAIR|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_BOOL$'),
        ],
        're_exprs_return_bool': [
            re.compile('^GxB_(LOR|LAND|LXOR|EQ|ANY)_(EQ|NE|GT|LT|GE|LE)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
    }

    @classmethod
    def register_new(cls, name, monoid, binaryop):
        if type(monoid) is not Monoid:
            raise TypeError(f'monoid must be a Monoid, not {type(monoid)}')
        if type(binaryop) != BinaryOp:
            raise TypeError(f'binaryop must be a BinaryOp, not {type(binaryop)}')
        new_type_obj = cls(name)
        for type_ in binaryop.types & monoid.types:
            type_ = dtypes.lookup(type_)
            new_semiring = ffi.new('GrB_Semiring*')
            lib.GrB_Semiring_new(new_semiring, monoid[type_], binaryop[type_])
            new_type_obj[type_.name] = new_semiring[0]
            ret_type = find_return_type(monoid[type_])
            _return_type[new_semiring[0]] = ret_type
        setattr(cls._module, name, new_type_obj)


def find_opclass(gb_op):
    if isinstance(gb_op, OpBase):
        return gb_op.__class__.__name__
    elif isinstance(gb_op, ffi.CData):
        cname = ffi.typeof(gb_op).cname
        for tc in ('UnaryOp', 'BinaryOp', 'Monoid', 'Semiring'):
            if tc in cname:
                return tc
    return UNKNOWN_OPCLASS


def reify_op(gb_op, dtype, dtype2=None):
    if dtype2 is not None:
        dtype = dtypes.unify(dtype, dtype2)
    if isinstance(gb_op, OpBase):
        gb_op = gb_op[dtype]
    return gb_op


_return_type = {}


def find_return_type(gb_op):
    if isinstance(gb_op, OpBase):
        raise ValueError('Requires concrete operator. Call `reify_op` first.')
    if gb_op not in _return_type:
        raise KeyError('Unknown operator. You must register function prior to use.')
    return _return_type[gb_op]

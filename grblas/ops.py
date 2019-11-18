import re
from _grblas import lib, ffi
from .dtypes import DataType
from .exceptions import check_status


UNKNOWN_OPCLASS = 'UnknownOpClass'


class OpBase:
    _parse_config = None
    _initialized = False

    def __init__(self, name):
        self.name = name
        self._specific_types = {}

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
        return type_.name if isinstance(type_, DataType) else type_
    
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
                        name = '_'.join(splitname)
                        # Create object for name unless it already exists
                        if not hasattr(cls, name):
                            setattr(cls, name, cls(name))
                        obj = getattr(cls, name)
                        gb_obj = getattr(lib, varname)
                        obj[type_] = gb_obj
                        # Add to map of return types
                        _return_type[gb_obj] = 'BOOL' if returns_bool else type_
        cls._initialized = True


class UnaryOp(OpBase):
    _parse_config = {
        'trim_from_front': 4, 
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GrB_(IDENTITY|AINV|MINV)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GrB_LNOT$'),
        ],
    }


class BinaryOp(OpBase):
    _parse_config = {
        'trim_from_front': 4, 
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GrB_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|TIMES|DIV)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GrB_(LOR|LAND|LXOR)$'),
            re.compile('^GxB_(RMINUS|RDIV|ISEQ|ISNE|ISGT|ISLT|ISLE|ISGE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
        're_exprs_return_bool': [
            re.compile('^GrB_(EQ|NE|GT|LT|GE|LE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
    }


class Monoid(OpBase):
    _parse_config = {
        'trim_from_front': 4,
        'trim_from_back': 7, 
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GxB_(MAX|MIN|PLUS|TIMES)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)_MONOID$'),
            re.compile('^GxB_(EQ|LAND|LOR|LXOR)_BOOL_MONOID$'),
        ],
    }


class Semiring(OpBase):
    _parse_config = {
        'trim_from_front': 4, 
        'num_underscores': 2,
        're_exprs': [
            re.compile('^GxB_(MIN|MAX|PLUS|TIMES)_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(LOR|LAND|LXOR|EQ)_(FIRST|SECOND|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_BOOL$'),
        ],
        're_exprs_return_bool': [
            re.compile('^GxB_(LOR|LAND|LXOR|EQ)_(EQ|NE|GT|LT|GE|LE)_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
    }


def find_opclass(gb_op):
    if isinstance(gb_op, OpBase):
        return gb_op.__class__.__name__
    elif isinstance(gb_op, ffi.CData):
        cname = ffi.typeof(gb_op).cname
        for tc in ('UnaryOp', 'BinaryOp', 'Monoid', 'Semiring'):
            if tc in cname:
                return tc
    return UNKNOWN_OPCLASS


_return_type = {}

def find_return_type(gb_op, dtype):
    if isinstance(gb_op, OpBase):
        gb_op = gb_op[dtype]
    return _return_type[gb_op]


_udf_holder = {}

def build_udf(func, container):
    # TODO: Involve numba here somehow
    dtype = container.dtype
    udf = ffi.new('GrB_UnaryOp*')
    def op_func(z, x):
        c_type = dtype.c_type
        z = ffi.cast(f'{c_type}*', z)
        x = ffi.cast(f'{c_type}*', x)
        z[0] = func(x[0])
    new_func = ffi.callback('void(void*, const void*)', op_func)
    check_status(lib.GrB_UnaryOp_new(
        udf,
        new_func,
        dtype.gb_type,
        dtype.gb_type))
    _udf_holder[id(container)] = (func, udf, new_func)
    return udf[0]

def free_udf(container):
    # Remove any functions associated with container
    _udf_holder.pop(id(container), None)

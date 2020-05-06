import inspect
import re
import types
import numpy as np
import numba
from collections.abc import Mapping
from functools import lru_cache
from types import FunctionType
from . import lib, ffi, dtypes, unary, binary, monoid, semiring
from .exceptions import GrblasException, check_status


UNKNOWN_OPCLASS = 'UnknownOpClass'


class UdfParseError(GrblasException):
    pass


class OpPath:
    def __init__(self, parent, name):
        self._parent = parent
        self._name = name


class ParameterizedUdf:
    def __init__(self, name):
        self.name = name
        # lru_cache per instance
        method = self.__call__.__get__(self, self.__class__)
        self.__call__ = lru_cache(maxsize=1024)(method)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ParameterizedUnaryOp(ParameterizedUdf):
    def __init__(self, name, func):
        if not callable(func):
            return TypeError('func must be callable')
        self.func = func
        self.__signature__ = inspect.signature(func)
        if name is None:
            name = getattr(func, '__name__', name)
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        unary = self.func(*args, **kwargs)
        return UnaryOp.register_anonymous(unary, self.name)


class ParameterizedBinaryOp(ParameterizedUdf):
    def __init__(self, name, func):
        if not callable(func):
            return TypeError('func must be callable')
        self.func = func
        self.__signature__ = inspect.signature(func)
        if name is None:
            name = getattr(func, '__name__', name)
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        binary = self.func(*args, **kwargs)
        return BinaryOp.register_anonymous(binary, self.name)


class ParameterizedMonoid(ParameterizedUdf):
    def __init__(self, name, binaryop, identity):
        if not isinstance(binaryop, ParameterizedBinaryOp):
            raise TypeError('binaropy must be parameterized')
        self.binaryop = binaryop
        self.__signature__ = binaryop.__signature__
        if callable(identity):
            # assume it must be parameterized as well, so signature must match
            sig = inspect.signature(identity)
            if sig != self.__signature__:
                raise ValueError(
                    f'Signatures of binarop and identity passed to {type(self).__name__} must be the same.  Got:\n'
                    f'    binaryop{self.__signature__}\n'
                    f'    !=\n'
                    f'    identity{sig}'
                )
        self.identity = identity
        if name is None:
            name = binaryop.name
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        binary = self.binaryop
        binary = binary(*args, **kwargs)
        identity = self.identity
        if callable(identity):
            identity = identity(*args, **kwargs)
        return Monoid.register_anonymous(binary, identity, self.name)


class ParameterizedSemiring(ParameterizedUdf):
    def __init__(self, name, monoid, binaryop):
        if not isinstance(monoid, (ParameterizedMonoid, Monoid)):
            raise TypeError('monoid must be of type Monoid or ParameterizedMonoid')
        if isinstance(binaryop, ParameterizedBinaryOp):
            self.__signature__ = binaryop.__signature__
            if isinstance(monoid, ParameterizedMonoid) and monoid.__signature__ != self.__signature__:
                raise ValueError(
                    f'Signatures of monoid and binarop passed to {type(self).__name__} must be the same.  Got:\n'
                    f'    monoid{monoid.__signature__}\n'
                    f'    !=\n'
                    f'    binaryop{self.__signature__}\n'
                    '\nPerhaps call monoid or binaryop with parameters before creating the semiring.'
                )
        elif isinstance(binaryop, BinaryOp):
            if isinstance(monoid, Monoid):
                raise TypeError('At least one of monoid or binaryop must be parameterized')
            self.__signature__ = monoid.__signature__
        else:
            raise TypeError('binaryop must be of type BinaryOp or ParameterizedBinaryOp')
        self.monoid = monoid
        self.binaryop = binaryop
        if name is None:
            name = f'{monoid.name}_{binaryop.name}'
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        monoid = self.monoid
        if isinstance(monoid, ParameterizedMonoid):
            monoid = monoid(*args, **kwargs)
        binary = self.binaryop
        if isinstance(binary, ParameterizedBinaryOp):
            binary = binary(*args, **kwargs)
        return Semiring.register_anonymous(monoid, binary, self.name)


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

    @classmethod
    def _remove_nesting(cls, funcname):
        module = cls._module
        modname = cls._modname
        if '.' not in funcname:
            if hasattr(module, funcname):
                raise AttributeError(f'{modname}.{funcname} is already defined')
        else:
            path, funcname = funcname.rsplit('.', 1)
            for folder in path.split('.'):
                if not hasattr(module, folder):
                    setattr(module, folder, OpPath(module, folder))
                module = getattr(module, folder)
                modname = f'{modname}.{folder}'
                if not isinstance(module, (OpPath, types.ModuleType)):
                    raise AttributeError(f'{modname} is already defined. Cannot use as a nested path.')
        return module, funcname

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
                        # Add to set of all known instances (for checking function type by object)
                        cls.all_known_instances.add(gb_obj)
        cls._initialized = True


class UnaryOp(OpBase):
    _module = unary
    _modname = 'unary'
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 1,
        're_exprs': [
            re.compile('^GrB_(IDENTITY|AINV|MINV)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(ABS|LNOT|ONE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GrB_LNOT$'),
        ],
    }
    all_known_instances = set()

    @classmethod
    def _build(cls, name, func):
        if type(func) is not FunctionType:
            raise TypeError(f'udf must be a function, not {type(func)}')
        if name is None:
            name = getattr(func, '__name__', '<anonymous_unary>')
        success = False
        new_type_obj = cls(name)
        return_types = {}
        nt = numba.types
        for type_, sample_val in dtypes._sample_values.items():
            type_ = dtypes.lookup(type_)
            # Check if func can handle this data type
            try:
                with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                    ret = func(sample_val)
                ret_type = dtypes.lookup(type(ret))
                if ret_type != type_ and (
                    'INT' in ret_type.name and 'INT' in type_.name
                    or 'FP' in ret_type.name and 'FP' in type_.name
                    or type_ == 'UINT64' and ret_type == 'FP64' and return_types.get('INT64') == 'INT64'
                ):
                    # Downcast `ret_type` to `type_`.  This is probably what users want most of the time,
                    # but we can't make a perfect rule.  There should be a way for users to be explicit.
                    ret_type = type_
                elif type_ == 'BOOL' and ret_type == 'INT64' and return_types.get('INT8') == 'INT8':
                    ret_type = dtypes.INT8

                # Numba is unable to handle BOOL correctly right now, but we have a workaround
                # See: https://github.com/numba/numba/issues/5395
                # We're relying on coercion behaving correctly here
                input_type = dtypes.INT8 if type_ == 'BOOL' else type_
                return_type = dtypes.INT8 if ret_type == 'BOOL' else ret_type

                # JIT the func so it can be used from a cfunc
                unary_udf = numba.njit(func)
                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(nt.CPointer(return_type.numba_type),
                                      nt.CPointer(input_type.numba_type))

                if type_ == 'BOOL':
                    if ret_type == 'BOOL':
                        def unary_wrapper(z, x):
                            z[0] = bool(unary_udf(bool(x[0])))
                    else:
                        def unary_wrapper(z, x):
                            z[0] = unary_udf(bool(x[0]))
                elif ret_type == 'BOOL':
                    def unary_wrapper(z, x):
                        z[0] = bool(unary_udf(x[0]))
                else:
                    def unary_wrapper(z, x):
                        z[0] = unary_udf(x[0])

                unary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(unary_wrapper)
                new_unary = ffi.new('GrB_UnaryOp*')
                check_status(lib.GrB_UnaryOp_new(new_unary, unary_wrapper.cffi,
                                                 ret_type.gb_type, type_.gb_type))
                new_type_obj[type_.name] = new_unary[0]
                _return_type[new_unary[0]] = ret_type.name
                cls.all_known_instances.add(new_unary[0])
                success = True
                return_types[type_.name] = ret_type.name
            except Exception:
                continue
        if success:
            return new_type_obj
        else:
            raise UdfParseError('Unable to parse function using Numba')

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False):
        if parameterized:
            return ParameterizedUnaryOp(name, func)
        return cls._build(name, func)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            unary_op = ParameterizedUnaryOp(name, func)
        else:
            unary_op = cls._build(name, func)
        setattr(module, funcname, unary_op)


class BinaryOp(OpBase):
    _module = binary
    _modname = 'binary'
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 1,
        're_exprs': [
            re.compile(
                '^GrB_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|TIMES|DIV)'
                '_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'
            ),
            re.compile('^GrB_(LOR|LAND|LXOR)$'),
            re.compile(
                '^GxB_(RMINUS|RDIV|PAIR|ANY|ISEQ|ISNE|ISGT|ISLT|ISLE|ISGE)'
                '_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'
            ),
        ],
        're_exprs_return_bool': [
            re.compile('^GrB_(EQ|NE|GT|LT|GE|LE)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
            re.compile('^GxB_(LOR|LAND|LXOR)_(BOOL|INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'),
        ],
    }
    all_known_instances = set()

    @classmethod
    def _build(cls, name, func):
        if type(func) is not FunctionType:
            raise TypeError(f'udf must be a function, not {type(func)}')
        if name is None:
            name = getattr(func, '__name__', '<anonymous_binary>')
        success = False
        new_type_obj = cls(name)
        return_types = {}
        nt = numba.types
        for type_, sample_val in dtypes._sample_values.items():
            type_ = dtypes.lookup(type_)
            # Check if func can handle this data type
            try:
                with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                    ret = func(sample_val, sample_val)
                ret_type = dtypes.lookup(type(ret))
                if ret_type != type_ and (
                    'INT' in ret_type.name and 'INT' in type_.name
                    or 'FP' in ret_type.name and 'FP' in type_.name
                    or type_ == 'UINT64' and ret_type == 'FP64' and return_types.get('INT64') == 'INT64'
                ):
                    # Downcast `ret_type` to `type_`.  This is probably what users want most of the time,
                    # but we can't make a perfect rule.  There should be a way for users to be explicit.
                    ret_type = type_
                elif type_ == 'BOOL' and ret_type == 'INT64' and return_types.get('INT8') == 'INT8':
                    ret_type = dtypes.INT8

                # Numba is unable to handle BOOL correctly right now, but we have a workaround
                # See: https://github.com/numba/numba/issues/5395
                # We're relying on coercion behaving correctly here
                input_type = dtypes.INT8 if type_ == 'BOOL' else type_
                return_type = dtypes.INT8 if ret_type == 'BOOL' else ret_type

                # JIT the func so it can be used from a cfunc
                binary_udf = numba.njit(func)

                # Build wrapper because GraphBLAS wants pointers and void return
                wrapper_sig = nt.void(nt.CPointer(return_type.numba_type),
                                      nt.CPointer(input_type.numba_type),
                                      nt.CPointer(input_type.numba_type))

                if type_ == 'BOOL':
                    if ret_type == 'BOOL':
                        def binary_wrapper(z, x, y):
                            z[0] = bool(binary_udf(bool(x[0]), bool(y[0])))
                    else:
                        def binary_wrapper(z, x, y):
                            z[0] = binary_udf(bool(x[0]), bool(y[0]))
                elif ret_type == 'BOOL':
                    def binary_wrapper(z, x, y):
                        z[0] = bool(binary_udf(x[0], y[0]))
                else:
                    def binary_wrapper(z, x, y):
                        z[0] = binary_udf(x[0], y[0])

                binary_wrapper = numba.cfunc(wrapper_sig, nopython=True)(binary_wrapper)
                new_binary = ffi.new('GrB_BinaryOp*')
                check_status(
                    lib.GrB_BinaryOp_new(new_binary, binary_wrapper.cffi,
                                         ret_type.gb_type, type_.gb_type, type_.gb_type)
                )
                new_type_obj[type_.name] = new_binary[0]
                _return_type[new_binary[0]] = ret_type.name
                cls.all_known_instances.add(new_binary[0])
                success = True
                return_types[type_.name] = ret_type.name
            except Exception:
                continue
        if success:
            return new_type_obj
        else:
            raise UdfParseError('Unable to parse function using Numba')

    @classmethod
    def register_anonymous(cls, func, name=None, *, parameterized=False):
        if parameterized:
            return ParameterizedBinaryOp(name, func)
        return cls._build(name, func)

    @classmethod
    def register_new(cls, name, func, *, parameterized=False):
        module, funcname = cls._remove_nesting(name)
        if parameterized:
            binary_op = ParameterizedBinaryOp(name, func)
        else:
            binary_op = cls._build(name, func)
        setattr(module, funcname, binary_op)

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

        def isclose(rel_tol=1e-7, abs_tol=0.0):
            def inner(x, y):
                return x == y or abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
            return inner

        BinaryOp.register_new('isclose', isclose, parameterized=True)


class Monoid(OpBase):
    _module = monoid
    _modname = 'monoid'
    _parse_config = {
        'trim_from_front': 4,
        'trim_from_back': 7,
        'num_underscores': 1,
        're_exprs': [
            re.compile(
                '^GxB_(MAX|MIN|PLUS|TIMES|ANY)'
                '_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)_MONOID$'
            ),
            re.compile('^GxB_(EQ|LAND|LOR|LXOR|ANY)_BOOL_MONOID$'),
        ],
    }
    all_known_instances = set()

    @classmethod
    def _build(cls, name, binaryop, identity):
        if type(binaryop) is not BinaryOp:
            raise TypeError(f'binaryop must be a BinaryOp, not {type(binaryop)}')
        if name is None:
            name = binaryop.name
        new_type_obj = cls(name)
        if not isinstance(identity, Mapping):
            identities = dict.fromkeys(binaryop.types, identity)
            explicit_identities = False
        else:
            identities = identity
            explicit_identities = True
        for type_, identity in identities.items():
            type_ = dtypes.lookup(type_)
            ret_type = find_return_type(binaryop[type_])
            # If there is a domain mismatch, then DomainMismatch will be raised
            # below if identities were explicitly given.
            if type_ != ret_type and not explicit_identities:
                continue
            new_monoid = ffi.new('GrB_Monoid*')
            func = getattr(lib, f'GrB_Monoid_new_{type_.name}')
            zcast = ffi.cast(type_.c_type, identity)
            check_status(func(new_monoid, binaryop[type_], zcast))
            new_type_obj[type_.name] = new_monoid[0]
            _return_type[new_monoid[0]] = ret_type
            cls.all_known_instances.add(new_monoid[0])
        return new_type_obj

    @classmethod
    def register_anonymous(cls, binaryop, identity, name=None):
        if isinstance(binaryop, ParameterizedBinaryOp):
            return ParameterizedMonoid(name, binaryop, identity)
        return cls._build(name, binaryop, identity)

    @classmethod
    def register_new(cls, name, binaryop, identity):
        module, funcname = cls._remove_nesting(name)
        if isinstance(binaryop, ParameterizedBinaryOp):
            monoid = ParameterizedMonoid(name, binaryop, identity)
        else:
            monoid = cls._build(name, binaryop, identity)
        setattr(module, funcname, monoid)


class Semiring(OpBase):
    _module = semiring
    _modname = 'semiring'
    _parse_config = {
        'trim_from_front': 4,
        'num_underscores': 2,
        're_exprs': [
            re.compile(
                '^GxB_(MIN|MAX|PLUS|TIMES|ANY)'
                '_(FIRST|SECOND|PAIR|MIN|MAX|PLUS|MINUS|RMINUS|TIMES'
                '|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)'
                '_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'
            ),
            re.compile('^GxB_(LOR|LAND|LXOR|EQ|ANY)_(FIRST|SECOND|PAIR|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_BOOL$'),
        ],
        're_exprs_return_bool': [
            re.compile(
                '^GxB_(LOR|LAND|LXOR|EQ|ANY)_(EQ|NE|GT|LT|GE|LE)'
                '_(INT8|UINT8|INT16|UINT16|INT32|UINT32|INT64|UINT64|FP32|FP64)$'
            ),
        ],
    }
    all_known_instances = set()

    @classmethod
    def _build(cls, name, monoid, binaryop):
        if type(monoid) is not Monoid:
            raise TypeError(f'monoid must be a Monoid, not {type(monoid)}')
        if type(binaryop) != BinaryOp:
            raise TypeError(f'binaryop must be a BinaryOp, not {type(binaryop)}')
        if name is None:
            name = f'{monoid.name}_{binaryop.name}'
        new_type_obj = cls(name)
        for binary_in, binary_func in binaryop._specific_types.items():
            binary_out = find_return_type(binary_func)
            # Unfortunately, we can't have user-defined monoids over bools yet
            # because numba can't compile correctly.
            if binary_out not in monoid.types:
                continue
            binary_out = dtypes.lookup(binary_out)
            new_semiring = ffi.new('GrB_Semiring*')
            check_status(lib.GrB_Semiring_new(new_semiring, monoid[binary_out], binary_func))
            new_type_obj[binary_in] = new_semiring[0]
            ret_type = find_return_type(monoid[binary_out])
            _return_type[new_semiring[0]] = ret_type
            cls.all_known_instances.add(new_semiring[0])
        return new_type_obj

    @classmethod
    def register_anonymous(cls, monoid, binaryop, name=None):
        if isinstance(monoid, ParameterizedMonoid) or isinstance(binaryop, ParameterizedBinaryOp):
            return ParameterizedSemiring(name, monoid, binaryop)
        return cls._build(name, monoid, binaryop)

    @classmethod
    def register_new(cls, name, monoid, binaryop):
        module, funcname = cls._remove_nesting(name)
        if isinstance(monoid, ParameterizedMonoid) or isinstance(binaryop, ParameterizedBinaryOp):
            semiring = ParameterizedSemiring(name, monoid, binaryop)
        else:
            semiring = cls._build(name, monoid, binaryop)
        setattr(module, funcname, semiring)


def find_opclass(gb_op):
    if isinstance(gb_op, (OpBase, ParameterizedUdf)):
        opclass = gb_op.__class__.__name__
    else:
        for opclass in (UnaryOp, BinaryOp, Monoid, Semiring):
            if gb_op in opclass.all_known_instances:
                opclass = opclass.__name__
                break
        else:
            opclass = UNKNOWN_OPCLASS
    if opclass.startswith('Parameterized'):
        gb_op = gb_op()  # Use default parameters of parameterized UDFs
        return find_opclass(gb_op)
    return gb_op, opclass


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


# Now initialize all the things!
UnaryOp._initialize()
BinaryOp._initialize()
Monoid._initialize()
Semiring._initialize()

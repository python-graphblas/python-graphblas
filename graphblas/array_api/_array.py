"""The Array: a Tensor of stored entries plus a fill value for every other position."""

import math

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._utils import np_dtype

CPU_DEVICE = "cpu"
API_VERSIONS = ("2021.12", "2022.12", "2023.12", "2024.12", "2025.12")


def check_device(device):
    if device is not None and device != CPU_DEVICE:
        raise ValueError(f"graphblas.array_api has only the device {CPU_DEVICE!r}, not {device!r}")


class Array:
    """An n-dimensional array whose value at a position is its stored entry there, or
    ``fill_value`` where it stores none (pydata/sparse's model).

    Create Arrays with the namespace's functions (``asarray``, ``zeros``, ``from_coo``, ...).
    The backing ``graphblas.Tensor`` holds the stored entries; stored entries may equal the
    fill value.
    """

    # The shape and dtype are kept here, not read from the Tensor, so that an Array whose
    # Tensor is not computed yet (``_pending``, a deferred contraction) can answer them
    __slots__ = "_tensor", "_pending", "_fill", "_facts", "_shape", "_dtype", "__weakref__"
    # numpy hands its operators over to ours (``ndarray + Array``), as it does for other
    # array types, rather than converting the Array to a dense ndarray
    __array_ufunc__ = None

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "Array is not created directly; use graphblas.array_api.asarray (or zeros, full, "
            "from_coo, ...)"
        )

    @classmethod
    def _new(cls, tensor, fill):
        """An Array of ``tensor``'s entries and ``fill`` (cast to its dtype) elsewhere.

        A 0-d Array stores nothing: its value is its fill.
        """
        if tensor.ndim == 0 and tensor.nvals:
            fill = tensor.to_scalar().value
            tensor = Tensor(tensor.dtype, ())
        self = object.__new__(cls)
        self._tensor = tensor
        self._pending = None
        self._shape = tensor.shape
        self._dtype = np_dtype(tensor.dtype)
        self._fill = _utils.scalar(fill, self._dtype)
        self._facts = None
        return self

    @classmethod
    def _deferred(cls, pending, shape, dtype, fill):
        """An Array whose Tensor ``pending.compute()`` gives on first use (see ``_lazy``)."""
        self = object.__new__(cls)
        self._tensor = None
        self._pending = pending
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._fill = _utils.scalar(fill, self._dtype)
        self._facts = None
        return self

    def _T(self):
        """The Tensor of stored entries. Every read of it goes through here: a pending Array
        computes it here, once.
        """
        if (T := self._tensor) is None:
            T = self._tensor = self._pending.compute()
            self._pending = None  # (releases the operands' snapshots)
        return T

    def _replace(self, other):
        """Make this Array hold ``other``'s values (for in-place operators and assignment)."""
        self._tensor = other._T()
        self._shape = other._shape
        self._dtype = other._dtype
        self._fill = other._fill
        self._facts = None

    def _fact(self, name, compute):
        """A cached property of the stored values. Code that changes the stored entries in
        place must clear the cache (``_facts = None``), as ``_replace`` does.
        """
        facts = self._facts
        if facts is None:
            facts = self._facts = {}
        if name not in facts:
            facts[name] = compute(self)
        return facts[name]

    def _copy(self):
        return Array._new(self._T().dup(), self._fill)

    # Attributes
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return CPU_DEVICE

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return math.prod(self._shape)

    @property
    def T(self):
        """The axes reversed (numpy's ``.T``; the Array API defines it for 2-d arrays)."""
        from ._manipulation import permute_dims

        return permute_dims(self, tuple(range(self.ndim - 1, -1, -1)))

    @property
    def mT(self):
        from ._manipulation import permute_dims

        if self.ndim < 2:
            raise ValueError("mT needs at least 2 dimensions")
        axes = list(range(self.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return permute_dims(self, tuple(axes))

    # Extensions
    @property
    def fill_value(self):
        """The value at every position the Array does not store (a numpy scalar)."""
        return self._fill

    @property
    def nnz(self):
        """The number of stored entries (some may equal the fill value)."""
        return self._T().nvals

    @property
    def tensor(self):
        """The stored entries as a ``graphblas.Tensor`` (positions holding the fill value are
        missing); a copy that shares the data until either side changes.
        """
        return self._T().dup()

    def todense(self):
        """A numpy array of every position (raises past ``array_api_densify_limit``)."""
        _utils.check_dense(self.size, f"densifying an array of shape {self.shape}")
        dt = self.dtype
        T = self._T()
        if T.ndim == 0 or T.nvals == 0:
            return np.full(self.shape, self._fill, dtype=dt)
        if T.nvals == self.size:
            return T.to_dense(dtype=T.dtype).astype(dt, copy=False)
        coords, values = T.to_coo()
        out = np.full(self.shape, self._fill, dtype=dt)
        out[coords] = values
        return out

    def to_coo(self):
        """``(coords, values)`` of the stored entries: a tuple of index arrays (one per axis)
        and the values, in C order.
        """
        return self._T().to_coo()

    def __repr__(self):
        fill = self._fill.item()
        if self.ndim == 0:
            return f"Array({fill!r}, dtype={self.dtype})"
        if self._tensor is None:  # (its nnz would compute it)
            return f"Array(shape={self.shape}, dtype={self.dtype}, fill_value={fill!r}, pending)"
        return f"Array(shape={self.shape}, dtype={self.dtype}, nnz={self.nnz}, fill_value={fill!r})"

    # Protocols
    def __array_namespace__(self, *, api_version=None):
        if api_version is not None and api_version not in API_VERSIONS:
            raise ValueError(f"unsupported Array API version {api_version!r}")
        from .. import array_api

        return array_api

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("converting an Array to a numpy array always copies")
        out = self.todense()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if copy is False:
            raise BufferError("exporting an Array through DLPack always copies")
        kwargs = {} if max_version is None else {"max_version": max_version}
        if dl_device is not None:
            kwargs["dl_device"] = dl_device
        return self.todense().__dlpack__(stream=stream, **kwargs)

    def __dlpack_device__(self):
        return np.empty(0).__dlpack_device__()

    def to_device(self, device, /, *, stream=None):
        check_device(device)
        return self

    def _value(self, what):
        if self.ndim != 0:
            raise TypeError(f"{what} needs a 0-d array; this one has shape {self.shape}")
        return self._fill

    def __bool__(self):
        return bool(self._value("bool()"))

    def __int__(self):
        value = self._value("int()")
        if self.dtype.kind == "c":
            raise TypeError("int() of a complex array")
        return int(value)

    def __float__(self):
        value = self._value("float()")
        if self.dtype.kind == "c":
            raise TypeError("float() of a complex array")
        return float(value)

    def __complex__(self):
        return complex(self._value("complex()"))

    def __index__(self):
        value = self._value("an index")
        if self.dtype.kind not in "iu":
            raise TypeError(f"only integer arrays are indices, not {self.dtype}")
        return int(value)

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d array")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        # (computed first: each row of a pending product would be a product of its own)
        self._T()
        return (self[i] for i in range(self.shape[0]))

    __hash__ = None

    # Indexing (graphblas.array_api._indexing)
    def __getitem__(self, key):
        from ._indexing import getitem

        return getitem(self, key)

    def __setitem__(self, key, value):
        from ._indexing import setitem

        setitem(self, key, value)

    # Operators
    def __matmul__(self, other):
        if not _is_operand(other):
            return NotImplemented
        from ._linalg import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        if not _is_operand(other):
            return NotImplemented
        from ._linalg import matmul

        return matmul(other, self)

    def __imatmul__(self, other):
        from ._linalg import matmul

        return self._inplace(matmul(self, other))

    def _inplace(self, result):
        if result.shape != self.shape:
            raise ValueError(
                f"an in-place operation would change the shape {self.shape} to {result.shape}"
            )
        if result.dtype != self.dtype:
            raise TypeError(
                f"an in-place operation would change the data type {self.dtype} to "
                f"{result.dtype}"
            )
        self._replace(result)
        return self

    def __neg__(self):
        from ._elementwise import negative

        return negative(self)

    def __pos__(self):
        from ._elementwise import positive

        return positive(self)

    def __abs__(self):
        from ._elementwise import abs as abs_

        return abs_(self)

    def __invert__(self):
        from ._elementwise import bitwise_invert

        return bitwise_invert(self)


def _is_operand(x):
    return isinstance(x, (Array, bool, int, float, complex, np.generic, np.ndarray))


def _binary_methods(name, func_name):
    def forward(self, other):
        if not _is_operand(other):
            return NotImplemented
        from . import _elementwise

        return getattr(_elementwise, func_name)(self, other)

    def reflected(self, other):
        if not _is_operand(other):
            return NotImplemented
        from . import _elementwise

        return getattr(_elementwise, func_name)(other, self)

    def inplace(self, other):
        from . import _elementwise

        return self._inplace(getattr(_elementwise, func_name)(self, other))

    forward.__name__ = f"__{name}__"
    reflected.__name__ = f"__r{name}__"
    inplace.__name__ = f"__i{name}__"
    return forward, reflected, inplace


for _name, _func in [
    ("add", "add"),
    ("sub", "subtract"),
    ("mul", "multiply"),
    ("truediv", "divide"),
    ("floordiv", "floor_divide"),
    ("mod", "remainder"),
    ("pow", "pow"),
    ("and", "bitwise_and"),
    ("or", "bitwise_or"),
    ("xor", "bitwise_xor"),
    ("lshift", "bitwise_left_shift"),
    ("rshift", "bitwise_right_shift"),
]:
    _f, _r, _i = _binary_methods(_name, _func)
    setattr(Array, f"__{_name}__", _f)
    setattr(Array, f"__r{_name}__", _r)
    setattr(Array, f"__i{_name}__", _i)

for _name, _func in [
    ("eq", "equal"),
    ("ne", "not_equal"),
    ("lt", "less"),
    ("le", "less_equal"),
    ("gt", "greater"),
    ("ge", "greater_equal"),
]:
    _f, _r, _i = _binary_methods(_name, _func)
    setattr(Array, f"__{_name}__", _f)
del _name, _func, _f, _r, _i

"""Creation functions.

``zeros``, ``ones``, ``full``, ``empty`` and their ``_like`` forms store nothing: their value is
the fill, so they cost O(1). Conversions from dense data store every position whose value is
not identical to the fill (0 of the dtype unless ``fill_value=`` is given): ``-0.0`` is stored
when the fill is ``0.0``, and NaN is stored unless the fill is NaN.
"""

import builtins
import math

import numpy as np

from ..core.tensor import Tensor
from . import _utils
from ._array import Array, check_device
from ._dtypes import as_dtype, check_supported
from ._utils import gb_dtype

__all__ = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_coo",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

_DEFAULT_FLOAT = np.dtype("float64")


def _shape(shape):
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    shape = tuple(int(n) for n in shape)
    if any(n < 0 for n in shape):
        raise ValueError(f"negative dimensions are not allowed: {shape}")
    return shape


def _default_fill(dtype):
    return _utils.scalar(0, dtype)


def _from_numpy(a, fill_value=None):
    """An Array of the numpy array ``a`` storing what differs from the fill."""
    dt = check_supported(a.dtype, "an array")
    fill = _default_fill(dt) if fill_value is None else _utils.scalar(fill_value, dt)
    if a.ndim == 0:
        return Array._new(Tensor(gb_dtype(dt), ()), a[()])
    if a.size == 0:
        return Array._new(Tensor(gb_dtype(dt), a.shape), fill)
    keep = _utils.differs(a, fill)
    if keep.all():
        T = Tensor.from_dense(a, dtype=gb_dtype(dt))
    else:
        coords = np.nonzero(keep)
        T = Tensor.from_coo(coords, a[coords], gb_dtype(dt), shape=a.shape)
    return Array._new(T, fill)


def _from_graphblas(obj, dtype, fill_value):
    """A Tensor, Matrix, Vector or Scalar as an Array whose missing entries hold the fill."""
    from .. import Matrix, Scalar, Vector
    from ..core.matrix import TransposedMatrix

    if isinstance(obj, Scalar):
        dt = _utils.np_dtype(obj.dtype)
        value = obj.value if obj.nvals else fill_value if fill_value is not None else 0
        out = Array._new(Tensor(gb_dtype(dt), ()), value)
    else:
        if isinstance(obj, Tensor):
            T = obj.dup()
        elif isinstance(obj, Vector):
            T = Tensor.from_vector(obj)
        elif isinstance(obj, (Matrix, TransposedMatrix)):
            T = Tensor.from_matrix(obj)
        else:  # pragma: no cover (safety)
            raise TypeError(type(obj))
        dt = check_supported(_utils.np_dtype(T.dtype), "an array")
        fill = _default_fill(dt) if fill_value is None else fill_value
        out = Array._new(T, fill)
    if dtype is not None and as_dtype(dtype) != out.dtype:
        from ._elementwise import cast

        out = cast(out, as_dtype(dtype))
    return out


def _from_sparse(obj, dtype, fill_value):
    """A scipy.sparse array or matrix, or a pydata/sparse array."""
    if hasattr(obj, "tocoo") and hasattr(obj, "format") and not hasattr(obj, "fill_value"):
        coo = obj.tocoo()
        coords = tuple(np.asarray(c) for c in coo.coords)
        data, shape, fill = np.asarray(coo.data), coo.shape, None
        if fill_value is not None:
            # scipy.sparse's missing entries are zeros
            raise ValueError("a scipy.sparse array has fill value 0; convert it first")
        if not getattr(coo, "has_canonical_format", True):
            coo.sum_duplicates()
            coords, data = tuple(np.asarray(c) for c in coo.coords), np.asarray(coo.data)
    else:
        if obj.format != "coo":
            obj = obj.asformat("coo")
        coords = tuple(np.asarray(c) for c in obj.coords)
        data, shape, fill = np.asarray(obj.data), obj.shape, obj.fill_value
        if fill_value is not None and not _utils.same_value(fill_value, fill):
            raise ValueError(f"fill_value={fill_value!r} differs from the sparse array's {fill!r}")
    return from_coo(coords, data, shape, fill_value=fill, dtype=dtype)


def _is_sparse(obj):
    module = type(obj).__module__
    return module.startswith(("scipy.sparse", "sparse"))


def asarray(obj, /, *, dtype=None, device=None, copy=None, fill_value=None):
    """Convert ``obj`` to an Array.

    ``obj`` is an Array, a Python scalar or (nested) sequence, an object numpy converts (a
    buffer, an ndarray), or (extensions) a ``graphblas`` Tensor, Matrix, Vector or Scalar, a
    scipy.sparse array or a pydata/sparse array, whose missing entries hold the fill.
    ``fill_value`` (an extension) is the value of unstored positions: 0 of the dtype by
    default, or the sparse array's own fill value.
    """
    check_device(device)
    if isinstance(obj, Array):
        if fill_value is not None and not _utils.same_value(fill_value, obj._fill):
            raise ValueError(
                "asarray of an Array keeps its fill value; xp.refill(x, fill_value) gives one "
                "with another (the same values)"
            )
        if dtype is not None and as_dtype(dtype) != obj.dtype:
            if copy is False:
                raise ValueError("changing the data type needs a copy, and copy=False")
            from ._elementwise import cast

            return cast(obj, as_dtype(dtype))
        return obj._copy() if copy else obj
    if copy is False:
        raise ValueError("asarray stores its own copy of anything but an Array; copy=False")
    from .. import Matrix, Scalar, Vector
    from ..core.matrix import TransposedMatrix

    if isinstance(obj, (Tensor, Matrix, Vector, Scalar, TransposedMatrix)):
        return _from_graphblas(obj, dtype, fill_value)
    if _is_sparse(obj) and (hasattr(obj, "tocoo") or hasattr(obj, "coords")):
        return _from_sparse(obj, dtype, fill_value)
    dt = None if dtype is None else as_dtype(dtype)
    if isinstance(obj, (list, tuple)):
        stacked = _stack_arrays(obj)
        if stacked is not None:
            return asarray(stacked, dtype=dtype, fill_value=fill_value)
        obj = _unwrap_nested(obj)
    if dt is None and isinstance(obj, builtins.bool):
        dt = np.dtype(bool)
    elif dt is None and isinstance(obj, int) and not isinstance(obj, np.generic):
        dt = np.dtype("int64")
    a = np.asarray(obj, dtype=dt)
    if not a.dtype.isnative:
        a = a.astype(a.dtype.newbyteorder("="))
    if dt is None and a.dtype.kind == "f" and a.dtype.itemsize < 4:
        a = a.astype(np.float32)
    return _from_numpy(a, fill_value)


def _stack_arrays(obj):
    """A sequence of n-d Arrays of one shape (or such sequences), stacked sparsely; None for
    anything else.
    """
    from ._manipulation import stack

    if not obj:
        return None
    parts = []
    for o in obj:
        if isinstance(o, (list, tuple)):
            o = _stack_arrays(o)
        if not isinstance(o, Array) or o.ndim == 0:
            return None
        parts.append(o)
    if len({p.shape for p in parts}) != 1:
        return None
    return stack(parts)


def _unwrap_nested(obj):
    """A (nested) sequence with its 0-d Arrays replaced by their values (numpy would densify
    every Array through ``__array__``, which works too, but slower).
    """
    if isinstance(obj, Array):
        return obj._fill if obj.ndim == 0 else obj.todense()
    if isinstance(obj, (list, tuple)):
        return [_unwrap_nested(o) for o in obj]
    return obj


def from_coo(coords, data, shape=None, *, fill_value=None, dtype=None):
    """An Array storing ``data`` at ``coords`` (one index array per axis, or an (ndim, nnz)
    array), and ``fill_value`` (0 by default) elsewhere. ``data`` may be a scalar.
    """
    data = np.asarray(data)
    dt = as_dtype(dtype) if dtype is not None else check_supported(data.dtype, "from_coo data")
    if dtype is not None:
        data = data.astype(dt, copy=False)
    if isinstance(coords, np.ndarray) and coords.ndim == 2:
        coords = tuple(coords)
    coords = tuple(np.asarray(c) for c in coords)
    if shape is None:
        shape = tuple(int(c.max()) + 1 if c.size else 0 for c in coords)
    shape = _shape(shape)
    fill = _default_fill(dt) if fill_value is None else _utils.scalar(fill_value, dt)
    if len(shape) == 0:
        value = data.reshape(-1)[0] if data.size else fill
        return Array._new(Tensor(gb_dtype(dt), ()), value)
    if coords and coords[0].size:
        T = Tensor.from_coo(coords, data, gb_dtype(dt), shape=shape)
    else:
        T = Tensor(gb_dtype(dt), shape)
    return Array._new(T, fill)


def _full(shape, value, dtype):
    shape = _shape(shape)
    return Array._new(Tensor(gb_dtype(dtype), shape), value)


def _fill_dtype(fill_value):
    if isinstance(fill_value, Array):
        return fill_value.dtype
    if isinstance(fill_value, np.generic):
        return check_supported(fill_value.dtype, "full")
    if isinstance(fill_value, builtins.bool):
        return np.dtype(bool)
    if isinstance(fill_value, int):
        return np.dtype("int64")
    if isinstance(fill_value, float):
        return _DEFAULT_FLOAT
    if isinstance(fill_value, complex):
        return np.dtype("complex128")
    raise TypeError(f"fill_value must be a Python scalar, not {type(fill_value).__name__}")


def full(shape, fill_value, *, dtype=None, device=None):
    check_device(device)
    dt = _fill_dtype(fill_value) if dtype is None else as_dtype(dtype)
    if isinstance(fill_value, Array):
        fill_value = fill_value._value("full's fill_value")
    if isinstance(fill_value, int) and dt.kind in "iu":
        info = np.iinfo(dt)
        if not info.min <= fill_value <= info.max:
            raise OverflowError(f"fill_value {fill_value} is out of range for {dt}")
    return _full(shape, fill_value, dt)


def zeros(shape, *, dtype=None, device=None):
    check_device(device)
    dt = _DEFAULT_FLOAT if dtype is None else as_dtype(dtype)
    return _full(shape, 0, dt)


def ones(shape, *, dtype=None, device=None):
    check_device(device)
    dt = _DEFAULT_FLOAT if dtype is None else as_dtype(dtype)
    return _full(shape, 1, dt)


def empty(shape, *, dtype=None, device=None):
    """Zeros: an empty array stores nothing, and its fill is 0 of the dtype."""
    return zeros(shape, dtype=dtype, device=device)


def _like(x, dtype, device):
    if not isinstance(x, Array):
        raise TypeError(f"needs an Array, not {type(x).__name__}")
    check_device(device)
    return x.shape, x.dtype if dtype is None else as_dtype(dtype)


def full_like(x, /, fill_value, *, dtype=None, device=None):
    shape, dt = _like(x, dtype, device)
    if isinstance(fill_value, Array):
        fill_value = fill_value._value("full_like's fill_value")
    return _full(shape, fill_value, dt)


def zeros_like(x, /, *, dtype=None, device=None):
    shape, dt = _like(x, dtype, device)
    return _full(shape, 0, dt)


def ones_like(x, /, *, dtype=None, device=None):
    shape, dt = _like(x, dtype, device)
    return _full(shape, 1, dt)


def empty_like(x, /, *, dtype=None, device=None):
    return zeros_like(x, dtype=dtype, device=device)


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
    check_device(device)
    dt = _DEFAULT_FLOAT if dtype is None else as_dtype(dtype)
    n_cols = n_rows if n_cols is None else n_cols
    shape = _shape((n_rows, n_cols))
    i = np.arange(builtins.max(0, -k), builtins.max(0, builtins.min(n_rows, n_cols - k)))
    if i.size == 0:
        return _full(shape, 0, dt)
    T = Tensor.from_coo((i, i + k), _utils.scalar(1, dt), gb_dtype(dt), shape=shape)
    return Array._new(T, 0)


def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
    check_device(device)
    if stop is None:
        start, stop = 0, start
    dt = None if dtype is None else as_dtype(dtype)
    if step == 0:
        raise ValueError("arange's step must not be 0")
    n = builtins.max(0, math.ceil((stop - start) / step))
    _utils.check_dense(n, "arange")
    if dt is None:
        if builtins.all(isinstance(v, int) for v in (start, stop, step)):
            dt = np.dtype("int64")
        else:
            dt = _DEFAULT_FLOAT
    a = np.arange(start, stop, step, dtype=dt)
    return _from_numpy(a[:n] if a.size > n else a)


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    check_device(device)
    _utils.check_dense(num, "linspace")
    if dtype is None:
        is_complex = builtins.any(isinstance(v, complex) for v in (start, stop))
        dt = np.dtype("complex128") if is_complex else _DEFAULT_FLOAT
    else:
        dt = as_dtype(dtype)
    a = np.linspace(start, stop, num, endpoint=endpoint, dtype=dt)
    return _from_numpy(a)


def from_dlpack(x, /, *, device=None, copy=None):
    check_device(device)
    if isinstance(x, Array):
        return x._copy() if copy else x
    if copy is False:
        raise ValueError("from_dlpack stores its own copy of anything but an Array; copy=False")
    return _from_numpy(np.from_dlpack(x))


def meshgrid(*arrays, indexing="xy"):
    from ._manipulation import _broadcast_to, reshape

    if indexing not in {"xy", "ij"}:
        raise ValueError(f"indexing must be 'xy' or 'ij', not {indexing!r}")
    for x in arrays:
        if not isinstance(x, Array) or x.ndim != 1:
            raise ValueError("meshgrid needs 1-d arrays")
    if len({x.dtype for x in arrays}) > 1:
        raise ValueError("meshgrid needs arrays of one data type")
    n = len(arrays)
    shape = [x.shape[0] for x in arrays]
    if indexing == "xy" and n >= 2:
        shape[0], shape[1] = shape[1], shape[0]
    out = []
    for i, x in enumerate(arrays):
        axis = i
        if indexing == "xy" and n >= 2 and i < 2:
            axis = 1 - i
        s = [1] * n
        s[axis] = x.shape[0]
        out.append(_broadcast_to(reshape(x, tuple(s)), tuple(shape)))
    return tuple(out)


def _tri(x, k, lower):
    if not isinstance(x, Array):
        raise TypeError(f"needs an Array, not {type(x).__name__}")
    if x.ndim < 2:
        raise ValueError("tril and triu need at least 2 dimensions")
    zero = _default_fill(x.dtype)
    if not _utils.same_value(x._fill, zero):
        # The kept triangle holds the fill where nothing is stored: store it there first
        _utils.check_dense(x.size, "tril or triu of an array whose fill is not 0")
        x = _from_numpy(x.todense())
    T = x._T()
    if T.nvals == 0:
        return _full(x.shape, 0, x.dtype)
    coords, values = T.to_coo()
    i, j = coords[-2], coords[-1]
    keep = (j - i <= k) if lower else (j - i >= k)
    if not keep.any():
        return _full(x.shape, 0, x.dtype)
    coords = tuple(c[keep] for c in coords)
    return Array._new(Tensor.from_coo(coords, values[keep], T.dtype, shape=T.shape), zero)


def tril(x, /, *, k=0):
    return _tri(x, k, lower=True)


def triu(x, /, *, k=0):
    return _tri(x, k, lower=False)

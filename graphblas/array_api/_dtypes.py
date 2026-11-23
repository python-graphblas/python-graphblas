"""Data types, type promotion and the data type functions.

The namespace's dtypes are numpy dtype instances (``xp.float64 is np.dtype("float64")``), and
an Array's ``dtype`` is one of them, so they compare and hash alike. Type promotion is numpy's,
with Python scalars weak (NEP 50).
"""

import builtins
from typing import NamedTuple

import numpy as np

__all__ = ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]

_NAMES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
DTYPES = {name: np.dtype(name) for name in _NAMES}
_SUPPORTED = frozenset(DTYPES.values())

_KINDS = {
    "bool": frozenset({DTYPES["bool"]}),
    "signed integer": frozenset(DTYPES[n] for n in ("int8", "int16", "int32", "int64")),
    "unsigned integer": frozenset(DTYPES[n] for n in ("uint8", "uint16", "uint32", "uint64")),
    "real floating": frozenset({DTYPES["float32"], DTYPES["float64"]}),
    "complex floating": frozenset({DTYPES["complex64"], DTYPES["complex128"]}),
}
_KINDS["integral"] = _KINDS["signed integer"] | _KINDS["unsigned integer"]
_KINDS["numeric"] = _KINDS["integral"] | _KINDS["real floating"] | _KINDS["complex floating"]


def as_dtype(dtype):
    """Return ``dtype`` as a supported numpy dtype, or raise TypeError.

    ``dtype`` is a dtype of this namespace, or anything ``np.dtype`` accepts.
    """
    if dtype is None:
        raise TypeError("dtype must not be None here")
    try:
        dt = np.dtype(dtype)
    except TypeError:
        raise TypeError(f"{dtype!r} is not a data type") from None
    if dt not in _SUPPORTED:
        raise TypeError(f"data type {dt} is not supported by graphblas.array_api")
    return dt


def check_supported(dt, what):
    if dt not in _SUPPORTED:
        raise TypeError(f"{what} would have data type {dt}, which graphblas.array_api lacks")
    return dt


def _dtype_of(x):
    from ._array import Array

    if isinstance(x, Array):
        return x.dtype
    return as_dtype(x)


def isdtype(dtype, kind):
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)
    if not isinstance(dtype, np.dtype) or dtype not in _SUPPORTED:
        raise TypeError(f"isdtype needs a data type of this namespace, not {dtype!r}")
    if isinstance(kind, tuple):
        return builtins.any(isdtype(dtype, k) for k in kind)
    if isinstance(kind, str):
        try:
            return dtype in _KINDS[kind]
        except KeyError:
            raise ValueError(f"unknown data type kind {kind!r}") from None
    return dtype == as_dtype(kind)


def result_type(*arrays_and_dtypes):
    from ._array import Array

    if not arrays_and_dtypes:
        raise ValueError("result_type needs at least one array or data type")
    args = []
    has_typed = False
    for x in arrays_and_dtypes:
        if isinstance(x, Array):
            args.append(x.dtype)
            has_typed = True
        elif isinstance(x, (builtins.bool, int, float, complex)) and not isinstance(x, np.generic):
            args.append(x)  # weak, as numpy treats a Python scalar
        else:
            args.append(as_dtype(x))
            has_typed = True
    if not has_typed:
        raise ValueError("result_type needs at least one array or data type, not only scalars")
    # numpy promotes all the arguments at once. For dtypes of one kind this follows the Array
    # API's promotion table. For mixed kinds, which the standard leaves unspecified, it does not
    # depend on the order, as promoting pairwise would: (int8, uint16, float32) is float32 here
    # and in np.concatenate, where pairwise it would be float64.
    return check_supported(np.result_type(*args), "result_type")


def can_cast(from_, to, /):
    return builtins.bool(np.can_cast(_dtype_of(from_), as_dtype(to), casting="safe"))


class finfo_object(NamedTuple):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: np.dtype


class iinfo_object(NamedTuple):
    bits: int
    max: int
    min: int
    dtype: np.dtype


def finfo(type, /):  # noqa: A002 (the Array API's name)
    dt = _dtype_of(type)
    if dt.kind not in "fc":
        raise TypeError(f"finfo needs a floating-point data type, not {dt}")
    info = np.finfo(dt)
    return finfo_object(
        builtins.int(info.bits),
        float(info.eps),
        float(info.max),
        float(info.min),
        float(info.smallest_normal),
        np.dtype(info.dtype),
    )


def iinfo(type, /):  # noqa: A002 (the Array API's name)
    dt = _dtype_of(type)
    if dt.kind not in "iu":
        raise TypeError(f"iinfo needs an integer data type, not {dt}")
    info = np.iinfo(dt)
    return iinfo_object(builtins.int(info.bits), builtins.int(info.max), builtins.int(info.min), dt)


def astype(x, dtype, /, *, copy=True, device=None):
    from ._array import Array, check_device
    from ._elementwise import cast

    check_device(device)
    if not isinstance(x, Array):
        raise TypeError(f"astype needs an Array, not {type(x)}")
    dt = as_dtype(dtype)
    if dt == x.dtype:
        return x._copy() if copy else x
    return cast(x, dt)

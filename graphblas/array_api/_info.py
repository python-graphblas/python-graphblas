"""``__array_namespace_info__``: what the namespace supports."""

from ._array import CPU_DEVICE
from ._dtypes import _KINDS, DTYPES

__all__ = ["__array_namespace_info__"]


class Info:
    __module__ = "graphblas.array_api"

    def capabilities(self):
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max dimensions": None,
        }

    def default_device(self):
        return CPU_DEVICE

    def devices(self):
        return (CPU_DEVICE,)

    def default_dtypes(self, *, device=None):
        _check(device)
        return {
            "real floating": DTYPES["float64"],
            "complex floating": DTYPES["complex128"],
            "integral": DTYPES["int64"],
            "indexing": DTYPES["int64"],
        }

    def dtypes(self, *, device=None, kind=None):
        _check(device)
        if kind is None:
            return dict(DTYPES)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        selected = set()
        for k in kinds:
            if k not in _KINDS:
                raise ValueError(f"unknown data type kind {k!r}")
            selected |= _KINDS[k]
        return {name: dt for name, dt in DTYPES.items() if dt in selected}


def _check(device):
    if device is not None and device != CPU_DEVICE:
        raise ValueError(f"unknown device {device!r}")


def __array_namespace_info__():
    return Info()

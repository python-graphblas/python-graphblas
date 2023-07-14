from ..core.dtypes import (
    _INDEX,
    BOOL,
    FP32,
    FP64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    DataType,
    _supports_complex,
    lookup_dtype,
    register_anonymous,
    register_new,
    unify,
)

if _supports_complex:
    from ..core.dtypes import FC32, FC64


def __dir__():
    return globals().keys() | {"ss"}


def __getattr__(key):
    if key == "ss":
        from .. import backend

        if backend != "suitesparse":
            raise AttributeError(
                f'module {__name__!r} only has attribute "ss" when backend is "suitesparse"'
            )
        from importlib import import_module

        ss = import_module(".ss", __name__)
        globals()["ss"] = ss
        return ss
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


_index_dtypes = {BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, _INDEX}

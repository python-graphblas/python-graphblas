try:
    import numba
except ImportError:
    _has_numba = _supports_udfs = False
else:
    _has_numba = _supports_udfs = True
    del numba


def __getattr__(name):
    if name in {"ffi", "lib", "NULL"}:
        from .. import _autoinit

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

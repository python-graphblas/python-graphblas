def __getattr__(name):
    if name in {"ffi", "lib", "NULL"}:
        from .. import _autoinit  # noqa

        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

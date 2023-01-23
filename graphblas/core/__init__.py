def __getattr__(name):
    if name in {"ffi", "lib", "NULL"}:
        from .. import _autoinit

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

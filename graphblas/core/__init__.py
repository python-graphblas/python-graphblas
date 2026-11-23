import threading

try:
    import numba
except ImportError:
    _has_numba = _supports_udfs = False
else:
    _has_numba = _supports_udfs = True
    del numba


# A lazily registered op (``binary.floordiv``) is built under this lock, one for every op
# namespace, and re-entrant, since building a semiring gets its binary op and monoid
_lazy_lock = threading.RLock()


def _build_delayed(namespace, delayed, key, build=None):
    """Return ``namespace[key]``, built once from ``delayed[key]``, or ``None`` if in neither.

    ``build(key, delayed.pop(key))`` builds it, by default ``func(**kwargs)`` of the pair.
    An op namespace's ``__getattr__`` calls this for every name it does not have, so a thread
    that asks for a name another thread is building waits for it, rather than finding it in
    neither place.
    """
    with _lazy_lock:
        if key in namespace:
            return namespace[key]
        if key not in delayed:
            return None
        # (out of ``delayed`` before the build, which refuses a name already registered)
        item = delayed.pop(key)
        try:
            if build is None:
                func, kwargs = item
                rv = func(**kwargs)
            else:
                rv = build(key, item)
        except BaseException:
            delayed[key] = item  # (so the next lookup builds it again, and says why it fails)
            raise
        namespace[key] = rv
        return rv


def __getattr__(name):
    if name in {"ffi", "lib", "NULL"}:
        from .. import _autoinit

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

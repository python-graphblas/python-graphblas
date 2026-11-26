import threading

try:
    import numba
except ImportError:
    _has_numba = _supports_udfs = False
else:
    _has_numba = _supports_udfs = True
    del numba


# One lock, shared by every op namespace. A lazily registered op such as
# ``binary.floordiv`` is built under it (see _build_delayed), and the numpy namespaces
# also hold it while they build an op on its first use. It is re-entrant, since a build
# can get other lazy names (a semiring gets its binary op and monoid), and since the
# numpy namespaces call _build_delayed while they hold it.
_lazy_lock = threading.RLock()


def _build_delayed(namespace, delayed, key, build=None):
    """Return ``namespace[key]``, built once from ``delayed[key]``, or ``None`` if in neither.

    The build is ``build(key, item)`` for the ``item`` popped from ``delayed``, or, with no
    ``build``, ``func(**kwargs)`` for an item that is the pair ``(func, kwargs)``. An op
    namespace's ``__getattr__`` calls this for every name it does not have, so a thread
    that asks for a name another thread is building waits for it, rather than finding it in
    neither place.

    A build that raises an exception drops the name, so later lookups raise AttributeError, as
    for a SelectOp registered lazily from an op that does not return BOOL. A build interrupted
    by ``KeyboardInterrupt`` or another ``BaseException`` stays delayed for the next lookup,
    since the interrupt says nothing about the op: a Ctrl-C during the first compile of
    ``binary.floordiv`` would otherwise lose it for the session.
    """
    with _lazy_lock:
        if key in namespace:
            return namespace[key]
        if key not in delayed:
            return None
        # Pop first: registering the name, as most builds do, raises while it is in ``delayed``
        item = delayed.pop(key)
        try:
            if build is None:
                func, kwargs = item
                rv = func(**kwargs)
            else:
                rv = build(key, item)
        except Exception:
            raise
        except BaseException:
            delayed[key] = item
            raise
        namespace[key] = rv
        return rv


def __getattr__(name):
    if name in {"ffi", "lib", "NULL"}:
        from .. import _autoinit

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from ..core import _build_delayed, operator

_delayed = {}

del operator


# register_new is imported on first use. core.ss.binary imports the operators, whose
# setup imports this module, so importing it here would fail if core.ss.binary came first.
def __getattr__(name):
    if name == "register_new":
        from ..core.ss.binary import register_new

        globals()[name] = register_new
        return register_new
    from . import _build

    if (rv := _build_delayed(globals(), _delayed, name, _build)) is not None:
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_delayed, "register_new"})

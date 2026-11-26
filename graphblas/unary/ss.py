from ..core import _build_delayed, operator

_delayed = {}

del operator


# register_new is imported on first use. core.ss.unary imports the operators, whose
# setup imports this module, so importing it here would fail if core.ss.unary came first.
def __getattr__(name):
    if name == "register_new":
        from ..core.ss.unary import register_new

        globals()[name] = register_new
        return register_new
    if (rv := _build_delayed(globals(), _delayed, name)) is not None:
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_delayed, "register_new"})

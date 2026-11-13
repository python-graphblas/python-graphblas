from ..core import operator

_delayed = {}

del operator


# core.ss.binary imports the operators, whose setup imports this module, so
# register_new is imported on first use: that lets core.ss.binary be imported first.
def __getattr__(name):
    if name == "register_new":
        from ..core.ss.binary import register_new

        globals()[name] = register_new
        return register_new
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), "register_new"})

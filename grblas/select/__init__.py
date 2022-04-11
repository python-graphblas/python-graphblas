# All items are dynamically added by classes in operator.py
# This module acts as a container of IndexUnaryOp instances
_delayed = {}
from grblas import operator  # noqa isort:skip

del operator


def __dir__():
    return globals().keys() | _delayed.keys()


def __getattr__(key):
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
        rv = func(**kwargs)
        globals()[key] = rv
        return rv
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


def _resolve_expr(expr, callname, opname):
    from grblas.base import BaseExpression

    if not isinstance(expr, BaseExpression) and hasattr(expr, "select"):
        raise TypeError(
            f"Expected VectorExpression or MatrixExpression; found {type(expr)}\n"
            f"Typical usage: select.{callname}(x <= 5)"
        )
    tensor = expr.args[0]
    thunk = expr.args[1]
    method = f"{opname}{expr.op.name}"
    if method not in globals():
        # Attempt to convert < into <= (rowlt is not part of official spec, but rowle is)
        if expr.op.name == "lt":
            method = f"{opname}le"
            thunk.value -= 1
        # Attempt to convert >= into > (rowge is not part of official spec, but rowgt is)
        elif expr.op.name == "ge":
            method = f"{opname}gt"
            thunk.value -= 1
        if method not in globals():
            raise ValueError(f"Unknown or unregistered select method: {method}")
    return globals()[method](tensor, thunk)


def value(expr):
    return _resolve_expr(expr, "value", "value")


def row(expr):
    return _resolve_expr(expr, "row", "row")


def column(expr):
    return _resolve_expr(expr, "column", "col")

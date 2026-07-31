# All items are dynamically added by classes in operator.py
# This module acts as a container of IndexUnaryOp instances
_delayed = {}


def __dir__():
    return globals().keys() | _delayed.keys() | {"ss"}


def __getattr__(key):
    if key in _delayed:
        func, kwargs = _delayed.pop(key)
        rv = func(**kwargs)
        globals()[key] = rv
        return rv
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
    from ..core.utils import _module_attr_error

    raise _module_attr_error(__name__, key, __dir__())


def _resolve_expr(expr, callname, opname):
    from ..core.operator.utils import _resolve_index_expr

    return _resolve_index_expr(globals(), "indexunary", expr, callname, opname)


def value(expr):
    """An advanced indexunary method for easily expressing value comparison logic.

    Example usage:
    >>> gb.indexunary.value(A > 0)

    The example will dispatch to ``gb.indexunary.valuegt(A, 0)``
    while being nicer to read.
    """
    return _resolve_expr(expr, "value", "value")


def row(expr):
    """An advanced indexunary method for easily expressing Matrix row index comparison logic.

    Example usage:
    >>> gb.indexunary.row(A <= 5)

    The example will dispatch to ``gb.indexunary.rowle(A, 5)``
    while being potentially nicer to read.
    """
    return _resolve_expr(expr, "row", "row")


def column(expr):
    """An advanced indexunary method for easily expressing Matrix column index comparison logic.

    Example usage:
    >>> gb.indexunary.column(A <= 5)

    The example will dispatch to ``gb.indexunary.colle(A, 5)``
    while being potentially nicer to read.
    """
    return _resolve_expr(expr, "column", "col")


# Note: an ``index`` helper (the Vector analogue of ``select.index``) is *not*
# provided here because ``indexunary.index`` already exists as an alias for the
# positional ``rowindex`` op (INT64). It is relied on as an operator, e.g.
# ``v.apply(indexunary.index)``, which a helper function would break. For a
# Vector index comparison use ``indexunary.row(v < k)`` (resolves to ``rowle``,
# the same op ``select.index`` uses) or the explicit ``indexunary.indexle`` /
# ``indexunary.indexgt``.


from ..core import operator  # noqa: E402 isort:skip

del operator

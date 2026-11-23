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
    """Return the apply of the IndexUnaryOp that a comparison of values names.

    ``gb.indexunary.value(A > 0)`` is ``gb.indexunary.valuegt(A, 0)``, which is True
    where an entry of ``A`` is greater than 0. The comparison is ``<``, ``<=``, ``>``,
    ``>=``, ``==`` or ``!=``, with a scalar on its right.
    """
    return _resolve_expr(expr, "value", "value")


def row(expr):
    """Return the apply of the IndexUnaryOp that a comparison of row indices names.

    ``gb.indexunary.row(A <= 5)`` is ``gb.indexunary.rowle(A, 5)``, which is True for the
    entries in rows 0 through 5. The comparison is ``<``, ``<=``, ``>`` or ``>=``, with a
    number or Scalar on its right. For a Vector, ``gb.indexunary.row(v < 4)`` compares
    indices.

    ``gb.indexunary.index`` is not a helper like ``gb.select.index`` but the ``rowindex``
    op, so ``gb.indexunary.index(v < 4)`` gives the index of each entry of ``v < 4``.
    """
    return _resolve_expr(expr, "row", "row")


def column(expr):
    """Return the apply of the IndexUnaryOp that a comparison of column indices names.

    ``gb.indexunary.column(A <= 5)`` is ``gb.indexunary.colle(A, 5)``, which is True for
    the entries in columns 0 through 5. The comparison is ``<``, ``<=``, ``>`` or ``>=``,
    with a number or Scalar on its right.
    """
    return _resolve_expr(expr, "column", "col")


# No ``index`` helper here, unlike in ``select``: ``indexunary.index`` is the
# ``rowindex`` op, and code such as ``v.apply(indexunary.index)`` uses it as one.


from ..core import operator  # noqa: E402 isort:skip

del operator

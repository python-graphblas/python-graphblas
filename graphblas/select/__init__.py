# All items are dynamically added by classes in operator.py
# This module acts as a container of SelectOp instances (which are
# identical to IndexUnaryOp under the hood, except they must return
# boolean and they default to `select` when called, rather than
# `apply`).
import math
import numbers

_delayed = {}
_binary_to_select = {}


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
    from ..core.base import BaseExpression

    if not isinstance(expr, BaseExpression):
        raise TypeError(
            f"Expected ScalarExpression, VectorExpression, or MatrixExpression; "
            f"found {type(expr)}\nTypical usage: select.{callname}(x <= 5)"
        )
    tensor = expr.args[0]
    thunk = expr.args[1]
    if expr._is_scalar:
        # Handle ScalarExpressions that change their arguments to Vector
        if tensor._parent is not None:  # e.g., suitesparse
            tensor = tensor._parent
            thunk = thunk._parent
        else:  # e.g., suitesparse-vanilla
            tensor = tensor[0].new()
            thunk = thunk[0].new()
    name = expr.op.name
    if opname != "value":
        if name in {"lt", "ge"} and f"{opname}{name}" in globals():
            # An op registered for it (a user's rowlt): the number as given
            if expr._right_literal is not None:
                thunk = expr._right_literal
        else:
            thunk, name = _index_bound(expr._right_literal, thunk, name)
    method = f"{opname}{name}"
    if method not in globals():  # pragma: no cover (sanity)
        raise ValueError(f"Unknown or unregistered select method: {method}")
    return globals()[method](tensor, thunk)


def _index_bound(literal, thunk, name):
    """Return the thunk and comparison that select an index ``name`` (lt, le, gt, ge) a bound.

    The bound is the number the caller gave (``literal``), else the thunk Scalar's value.
    The comparison typed its thunk beside the values, which is wrong for an index: 300
    beside INT8 values is infinity, and 2**24 + 1 beside FP32 rounds. Only rowle and
    rowgt exist (and colle, colgt, indexle, indexgt), and indices are whole numbers, so
    ``i < t`` is ``i <= ceil(t) - 1``, ``i >= t`` is ``i > ceil(t) - 1``, and ``i <= t``
    and ``i > t`` take ``floor(t)``, where GraphBLAS would truncate a float toward zero
    (``i > -0.5`` dropped index 0). A whole number beyond INT64, which holds every index,
    is clamped to it. NaN compares False with every index, so it keeps none. An empty
    Scalar goes to the op as it is, which raises EmptyObject, and so does a thunk that is
    no Scalar (from ``A.apply(binary.le, left=3)``), which raises the op's TypeError.
    """
    shift = name in {"lt", "ge"}
    if shift:
        name = "le" if name == "lt" else "gt"
    # The caller's number, if it is one; a Scalar expression (``w[0]``), a 0-d array or
    # ``np.True_`` made the thunk keep its own type, so the thunk's value is the bound.
    if isinstance(literal, numbers.Real):
        bound = literal
    elif getattr(thunk, "_is_scalar", False):
        bound = thunk.value
    else:
        return thunk, name
    if not isinstance(bound, numbers.Real):
        return thunk, name
    if math.isnan(bound):
        return (-1 if name == "le" else 2**63 - 1), name  # no index is <= -1 or > INT64_MAX
    if math.isfinite(bound):
        bound = math.ceil(bound) - 1 if shift else math.floor(bound)
        bound = min(max(bound, -(2**63)), 2**63 - 1)
    return bound, name


def _match_expr(parent, expr):
    """Match expressions to rewrite ``A.select(A < 5)`` into select expression.

    The argument must match the parent, so this _won't_ be rewritten: ``A.select(B < 5)``
    """
    args = expr.args
    op = expr.op
    if (
        expr.method_name == "apply"
        and len(args) == 2
        and op.parent in _binary_to_select
        and args[1]._is_scalar
        and (args[0] is parent or getattr(args[0], "T", None) is parent)
        and op.type in _binary_to_select[op.parent]
    ):
        selectop = _binary_to_select[op.parent][op.type]
        return selectop(parent, args[1])


def value(expr):
    """An advanced select method for easily expressing value comparison logic.

    Example usage:
    >>> gb.select.value(A > 0)

    The example will dispatch to ``gb.select.valuegt(A, 0)``
    while being nicer to read.
    """
    return _resolve_expr(expr, "value", "value")


def row(expr):
    """An advanced select method for easily expressing Matrix row index comparison logic.

    Example usage:
    >>> gb.select.row(A <= 5)

    The example will dispatch to ``gb.select.rowle(A, 5)``
    while being potentially nicer to read.
    """
    return _resolve_expr(expr, "row", "row")


def column(expr):
    """An advanced select method for easily expressing Matrix column index comparison logic.

    Example usage:
    >>> gb.select.column(A <= 5)

    The example will dispatch to ``gb.select.colle(A, 5)``
    while being potentially nicer to read.
    """
    return _resolve_expr(expr, "column", "col")


def index(expr):
    """An advanced select method which allows for easily expressing
    Vector index comparison logic.

    Example usage:
    >>> gb.select.index(v <= 5)

    The example will dispatch to ``gb.select.indexle(v, 5)``
    while being potentially nicer to read.
    """
    return _resolve_expr(expr, "index", "index")


from ..core import operator  # noqa: E402 isort:skip

del operator

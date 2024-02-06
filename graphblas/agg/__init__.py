"""``graphblas.agg`` is an experimental module for exploring Aggregators.

Aggregators may be used in reduce methods:
    - Matrix.reduce_rowwise
    - Matrix.reduce_columnwise
    - Matrix.reduce_scalar
    - Vector.reduce

Aggregators are implemented as recipes that often use monoids or semirings.

Monoid-only aggregators:
    - sum
    - prod
    - all
    - any
    - min
    - max
    - any_value
    - bitwise_all
    - bitwise_any

Semiring aggregators with O(1) dense vector:
    - count
    - count_nonzero
    - count_zero
    - sum_of_squares
    - sum_of_inverses
    - exists, 1 if any values

Semiring aggregators with UnaryOp applied to the final result:
    - hypot
    - logaddexp
    - logaddexp2

Vector norms:
    - L0norm (=count_nonzero), sum(x != 0).  Note: not a proper norm.
    - L1norm, sum(abs(x))
    - L2norm (=hypot), sum(x**2)**0.5
    - Linfnorm, max(abs(x))

Composite aggregators (require multiple aggregation steps):
    - mean
    - peak_to_peak, max - min
    - varp, population variance
    - vars, sample variance
    - stdp, population standard deviation
    - stds, sample standard deviation
    - geometric_mean
    - harmonic_mean
    - root_mean_square

Custom recipes (specific to SuiteSparse:GraphBLAS):
    - ss.first
    - ss.last
    # These don't work with Matrix.reduce_scalar
    - ss.first_index
    - ss.last_index
    - ss.argmin
    - ss.argmax

.. deprecated:: 2023.1.0
    Aggregators ``first``, ``last``, ``first_index``, ``last_index``, ``argmin``, and ``argmax``
    are deprecated in the ``agg`` namespace such as ``agg.first``. Use them from ``agg.ss``
    namespace instead such as ``agg.ss.first``. Will be removed in version 2023.9.0 or later.

# Possible aggregators:
#   - absolute_deviation, sum(abs(x - mean(x))),  sum_absminus(x, mean(x))
#   - mean_absolute_deviation, absolute_deviation / count
#   - argmini, argminj, argmaxi, argmaxj
#   - firsti, firstj, lasti, lastj
#   - lxnor monoid: even number of true
#   - lxor monoid: odd number of true
#   - bxnor monoid: even bits
#   - bnor monoid: odd bits
"""

# All items are dynamically added by classes in core/operator/agg.py
# This module acts as a container of Aggregator instances
_deprecated = {}


def __dir__():
    return globals().keys() | _deprecated.keys() | {"ss"}


def __getattr__(key):
    if key in _deprecated:
        import warnings

        warnings.warn(
            f"`gb.agg.{key}` is deprecated; please use `gb.agg.ss.{key}` instead. "
            f"`{key}` is specific to SuiteSparse:GraphBLAS. "
            f"`gb.agg.{key}` will be removed in version 2023.9.0 or later.",
            DeprecationWarning,
            stacklevel=2,
        )
        rv = _deprecated[key]
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
    raise AttributeError(f"module {__name__!r} has no attribute {key!r}")


from ..core import operator  # noqa: E402 isort:skip

del operator

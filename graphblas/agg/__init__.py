"""``graphblas.agg`` is an experimental module for exploring Aggregators.

Aggregators may be used in reduce methods:
    - Matrix.reduce_rowwise
    - Matrix.reduce_columnwise
    - Matrix.reduce_scalar
    - Vector.reduce

Aggregators are implemented as recipes that often use monoids or semirings.

``count``, ``count_nonzero``, ``count_zero`` and ``L0norm`` return INT64 for values of any
type. Indexed with a type, as in ``agg.count_nonzero[FP32]``, they return that type.

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

Position aggregators:
    - first
    - last
    # These don't work with Matrix.reduce_scalar
    - first_index
    - last_index
    - argmin
    - argmax

With the "suitesparse" backend, the position aggregators are also in ``agg.ss``, such as
``agg.ss.argmin``.

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


def __dir__():
    return globals().keys() | {"ss"}


def __getattr__(key):
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


from ..core import operator  # noqa: E402 isort:skip

del operator

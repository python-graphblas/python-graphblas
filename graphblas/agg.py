"""`graphblas.agg` is an experimental module for exploring Aggregators.

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

Custom recipes:
    - first
    - last
    # These don't work with Matrix.reduce_scalar
    - first_index
    - last_index
    - argmin
    - argmax

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
# All items are dynamically added by classes in _agg.py
# This module acts as a container of Aggregator instances
from . import operator

del operator
from . import _agg  # noqa isort:skip

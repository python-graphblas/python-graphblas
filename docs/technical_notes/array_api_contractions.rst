.. _tn_array_api_contractions:

Contractions in the Array API Namespace
=======================================

``graphblas.array_api`` has four contractions: ``matmul`` (also ``x1 @ x2``), ``tensordot``,
``vecdot``, and the extension ``einsum``. Each sums products of the operands' elements.
An Array has a value at every position: its stored entries, and its fill value everywhere
else. This page says how a contraction treats the fills, which is where its results can
differ from numpy's. :ref:`array_api` introduces the namespace.

The examples on this page use these arrays:

.. code-block:: python

    >>> import numpy as np
    >>> import graphblas as gb
    >>> import graphblas.array_api as xp
    >>> A = xp.asarray([[1.0, 0.0], [0.0, 2.0]])  # stores 2 entries; fill 0.0
    >>> B = xp.asarray([[np.inf, 1.0], [3.0, np.inf]])  # stores 4 entries

.. _tn_array_api_zero_fills:

Zero fills
----------

A contraction treats a fill of zero as a structural zero, as scipy.sparse and pydata/sparse
do. The terms at a position an operand does not store are skipped, so a ``0 * inf`` there
is not NaN:

.. code-block:: python

    >>> (A @ B).todense()
    array([[inf,  1.],
           [ 6., inf]])
    >>> A.todense() @ B.todense()  # numpy: 0 * inf is NaN, with a RuntimeWarning
    array([[inf, nan],
           [nan, inf]])

The result has the fill 0 of its dtype, and its dtype is numpy's for the same operands.
Booleans contract with logical or and logical and, as in numpy.

If an operand stores an infinity or NaN, the contraction first drops each stored entry equal
to its array's fill, so such an entry contributes no term either. So whether a zero is
stored does not change the result, except perhaps the sign of a zero: a stored ``0.0``
times ``-1.0`` is ``-0.0``. ``-0.0`` is not identical to the fill ``0.0``, so a stored
``-0.0`` times ``inf`` is NaN.

Sums may round differently from numpy's, because the terms are added in another order.
``einsum`` may also sum over an index that only one operand has before it multiplies, so
with an infinity stored, a result can be an infinity where numpy's term-by-term sum is NaN:
``xp.einsum("i,j->", xp.asarray([1.0, -2.0]), xp.asarray([-2.0, np.inf]))`` is ``-inf``,
and ``np.einsum`` of the same values is NaN.

.. _tn_array_api_other_fills:

Other fills
-----------

An operand whose fill is not zero becomes one whose fill is zero, if it can:

- An operand that stores every position never shows its fill, so its fill is taken as 0, at
  no cost.
- An operand with another finite fill first stores its fill at each position it does not
  store, at a cost of O(its size). Then its fill is 0. ``xp.ones((2, 2)) @ A`` stores four
  ones first, and returns ``[[1.0, 2.0], [1.0, 2.0]]``.

Any other nonzero fill, such as ``inf`` or NaN, makes the contraction compute with numpy on
dense copies of the operands: ``matmul`` with ``np.matmul``, ``tensordot`` with
``np.tensordot``, and so on. ``xp.full((2, 2), np.inf) @ A`` is NaN at every position, as in
numpy.

Storing the fill and computing with numpy both need memory for every position: storing the
fill for that operand, and numpy for every operand and for the result. Past
``gb.config["array_api_densify_limit"]`` positions (10**8 by default), both raise
``xp.DensifyError``:

.. code-block:: python

    >>> with gb.config.set(array_api_densify_limit=3):
    ...     xp.ones((2, 2)) @ A
    Traceback (most recent call last):
    ...
    graphblas.array_api.DensifyError: a contraction with a nonzero fill that does not store
    every position would hold 4 elements densely, more than the limit of 3. ...

A zero stored by an operand whose fill is not 0 is a value, not a fill. When an operand
stores an infinity or NaN, the contraction computes with numpy too, so that zero's terms
count: ``xp.asarray([[np.inf]]) @ xp.asarray([[0.0]], fill_value=1.0)`` is NaN.

A contraction computed with numpy gives the answer of the numpy function named above, and
those functions do not always agree on ``0 * inf``. With numpy 2.3 and 2.4 on OpenBLAS
0.3.33 (macOS), ``np.tensordot([[np.inf], [2.0]], [[0.0]], axes=1)`` is ``[[0.], [0.]]``: its BLAS product
skips the zero. ``np.matmul`` and ``np.einsum`` of the same arrays give ``[[nan], [0.]]``.

In ``einsum``, a 0-d operand with a finite nonzero value multiplies every term, so it is
factored out of the sum. A 0-d operand whose value is infinite or NaN makes ``einsum``
compute with numpy on dense copies.

.. _tn_array_api_semirings:

Semirings
---------

As an extension, each contraction takes ``semiring=``: a ``graphblas`` Semiring, such as
``gb.semiring.min_plus``, or its name, such as ``"min_plus"`` or ``"max_min"``. Its monoid
takes the place of the sum, and its binary operator the place of the product.

The contraction skips the unstored positions, which is right when every operand's fill is
the identity ``e`` of the monoid, and ``e`` annihilates the binary operator from both sides:
``mult(e, y) == mult(y, e) == e`` for every ``y``. The result has the fill ``e``.

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Semiring
     - Fill ``e`` for floats
   * - ``plus_times`` (the default)
     - ``0``
   * - ``min_plus``
     - ``inf``
   * - ``max_plus``
     - ``-inf``
   * - ``max_min``
     - ``-inf``
   * - ``lor_land`` (the default for bool)
     - ``False``

For an integer dtype, the identity of ``min`` is the dtype's largest value and that of
``max`` its smallest, so ``max_min`` on int64 needs the fill ``np.iinfo(np.int64).min``.

For example, with an array of path lengths whose fill ``inf`` means no edge:

.. code-block:: python

    >>> D = xp.asarray([[0.0, 3.0], [np.inf, 0.0]], fill_value=np.inf)  # stores 3 entries
    >>> R = xp.matmul(D, D, semiring="min_plus")  # shortest paths of up to two edges
    >>> R.todense(), R.fill_value
    (array([[ 0.,  3.],
           [inf,  0.]]), np.float64(inf))

The skipping is structural, as with zero fills: an infinity stored in one operand makes no
NaN at a position the other operand does not store, such as ``-inf + inf`` in ``min_plus``.

An operand with another fill makes the call raise ``ValueError``, unless the operand stores
every position and its dtype can represent ``e``: then its fill is taken as ``e``. An
integer array cannot represent ``inf``. The error names the fill to use, and
``xp.refill(x, xp.inf)`` returns a copy of ``x`` with the fill ``inf`` and the same values:

.. code-block:: python

    >>> xp.matmul(A, A, semiring="min_plus")
    Traceback (most recent call last):
    ...
    ValueError: matmul with the semiring min_plus needs operands whose fill is inf, ...

A semiring whose identity does not annihilate its binary operator raises ``ValueError``
whatever the fills are. ``min_plus`` on int64 is one. Its identity is the largest int64, and
adding a positive number to it wraps. Positional semirings raise too. For these, use the
Tensors (``x.tensor``), which skip missing entries and have no fills.

Whether ``e`` annihilates the binary operator is checked once for each typed semiring, on a
few finite values of each operand type: both signs, zeros, and the extremes.

With a float ``min`` or ``max`` monoid, a stored NaN makes NaN at each result position that
has a term with it, as numpy's minimum and maximum propagate NaN. SuiteSparse's ``min`` and
``max`` skip NaN, so such a contraction computes at once (it is not deferred, see below).

The default semiring follows the rules for zero fills above, whether it is given or not. On
bool, ``plus_times`` means ``lor_land``, as numpy contracts booleans.

.. _tn_array_api_deferred:

Deferred results
----------------

A contraction that skips the unstored positions, and whose result has a dimension, returns
an Array whose value is not computed yet. Three cases compute at once instead: NaN under a
float ``min`` or ``max`` monoid (above), and two kinds of ``einsum``, one with a finite
nonzero 0-d operand, and one of three operands whose third masks the product of the other
two. A deferred result is computed the first time its data is read, and only in part when
a consumer needs only part of it:

- ``xp.multiply(S, P)`` or ``xp.multiply(P, S)``, where ``P`` is a deferred result and ``S``
  an array of the same shape, both with fill 0, computes ``P`` only at the positions ``S``
  stores (a sampled dense-dense product). For a float result, this needs every operand's
  values to be finite and no sum to overflow; otherwise ``P`` is computed whole first.
- A basic index (integers and slices) of a deferred result of ``matmul``, ``tensordot`` or
  ``einsum`` slices the operands along the indexed axes, and the result is deferred again.
  (Not for a ``matmul`` whose batch axes broadcast.)

``xp.compute(x)`` computes a deferred ``x`` now, for timing or before handing it on, and
returns it. Values do not depend on whether a result was deferred, and changing an operand
afterwards does not change the result.

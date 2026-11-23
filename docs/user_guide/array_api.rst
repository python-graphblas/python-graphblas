.. _array_api:

Array API Namespace (experimental)
==================================

.. warning::

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.** ``graphblas.array_api`` needs the
    ``suitesparse`` backend with SuiteSparse:GraphBLAS 10 or later, as ``Tensor`` does. On the
    vanilla backend it imports, but using it raises ``NotImplementedError``.

``graphblas.array_api`` implements the `Python array API standard
<https://data-apis.org/array-api/2025.12/>`_ (version 2025.12) over :class:`~graphblas.Tensor`,
with every function of the standard except the ``fft`` extension. Code written against the
standard can run on these arrays unchanged, and the arrays store only the positions they
need to. The results are numpy's on the same values, with a few exceptions. A contraction
skips unstored positions, so ``0 * inf`` there is not NaN. The sign of a zero that the
result does not store is unspecified, and sums and transcendental functions may round
differently. An operation past the densify limit (below) raises instead of allocating. And
for ``x[key] = value``, the standard's `in-place semantics
<https://data-apis.org/array-api/2025.12/API_specification/broadcasting.html#in-place-semantics>`_
ask that the value broadcast to the selection's shape. numpy also accepts extra leading axes
of length 1, such as a value of shape ``(1, 2, 4)`` for ``x[0:2]`` of a 3 x 4 ``x``, and this
namespace raises ``ValueError``.

.. code-block:: python

    >>> import numpy as np
    >>> import graphblas.array_api as xp
    >>> x = xp.asarray([[0.0, 1.5], [2.0, 0.0]])
    >>> x
    Array(shape=(2, 2), dtype=float64, nnz=2, fill_value=0.0)
    >>> x.__array_namespace__() is xp
    True

Fill values
-----------

An Array has a value at every position, as a numpy array does, but it stores only some of
them. Every other position has the Array's ``fill_value``. ``nnz`` counts the stored entries.
``zeros``, ``ones``, ``full`` and ``empty`` store nothing, so they cost the same at any
size:

.. code-block:: python

    >>> z = xp.zeros((1000, 1000))
    >>> z[0, 1] = 3.0
    >>> z
    Array(shape=(1000, 1000), dtype=float64, nnz=1, fill_value=0.0)
    >>> xp.sum(z, axis=0)
    Array(shape=(1000,), dtype=float64, nnz=1, fill_value=0.0)

An operation computes the result's fill from the operands' fills. So ``exp`` of an array
whose fill is 0 has the fill 1, and stores no more entries than its input:

.. code-block:: python

    >>> y = xp.exp(x)
    >>> y
    Array(shape=(2, 2), dtype=float64, nnz=2, fill_value=1.0)
    >>> y.todense()
    array([[1.        , 4.48168907],
           [7.3890561 , 1.        ]])

``xp.refill(x, fill_value)`` returns a copy with another fill and the same values, and
``xp.prune(x)`` drops the stored entries equal to the fill. Both are extensions to the
standard.

An operation that would store more positions densely than
``graphblas.config["array_api_densify_limit"]`` (10**8 by default) raises
``xp.DensifyError`` instead of allocating them. Raise the limit with
``graphblas.config.set(array_api_densify_limit=...)``.

Converting
----------

``xp.asarray`` accepts Python scalars and sequences and numpy arrays, and also a
``graphblas`` Tensor, Matrix, Vector or Scalar, a scipy.sparse array, and a pydata/sparse
array. The positions these do not store get the fill. Going back:

- ``x.todense()`` returns a numpy array (under the densify limit);
- ``x.to_coo()`` returns the stored entries' coordinates and values;
- ``x.tensor`` returns the stored entries as a :class:`~graphblas.Tensor`;
- ``x.to_scipy_sparse(format)`` returns a scipy.sparse array, for a fill of 0.

Contractions
------------

``matmul`` (``x1 @ x2``), ``tensordot`` and ``vecdot`` take a ``semiring=`` keyword, an
extension to the standard. So does ``einsum``, itself an extension, which takes numpy's
subscripts. For example, ``min_plus`` with fill ``inf`` relaxes shortest paths:

.. code-block:: python

    >>> D = xp.asarray([[0.0, 3.0], [np.inf, 0.0]], fill_value=np.inf)
    >>> xp.matmul(D, D, semiring="min_plus").todense()
    array([[ 0.,  3.],
           [inf,  0.]])

A contraction skips the positions an operand with fill 0 does not store, so ``0 * inf``
there is not NaN, unlike numpy. Its result may be computed only when it is first used.
:ref:`tn_array_api_contractions` gives the rules for other fills, for semirings, and for
deferred results.

See :ref:`api_array_api` for the extensions' reference.

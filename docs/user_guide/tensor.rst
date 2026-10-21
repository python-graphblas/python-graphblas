.. _tensor:

Tensors (experimental)
======================

.. warning::

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.** ``Tensor`` and ``einsum`` are new, and their
    names, defaults and behavior may change in any release. They need the ``suitesparse``
    backend with SuiteSparse:GraphBLAS 10 or later; on the vanilla backend they import, and
    using them raises ``NotImplementedError``.

A **Tensor** is an n-dimensional sparse array: a Matrix and a Vector generalized to any
number of axes. Like the other collections it has one data type and a shape, and positions
may be empty (missing). It follows GraphBLAS for missing values (there is no fill value) and
numpy and the Array API for shapes, axes and broadcasting.

The examples below assume:

.. code-block:: python

    >>> import numpy as np
    >>> import graphblas as gb
    >>> from graphblas import Tensor, binary, einsum, monoid, semiring, unary

The model: logical shape and stored layout
------------------------------------------

A Tensor stores its entries in one SuiteSparse Matrix, held by row, plus a **layout**: which
logical axes are raveled (in C order, in the order given) into the Matrix's rows, and which
into its columns. The shape is always the logical one; the layout is how it is stored.

.. code-block:: python

    >>> X = Tensor.from_coo(
    ...     [[0, 0, 1], [1, 2, 0], [3, 0, 2]], [1.0, 2.0, 3.0], shape=(2, 3, 4), name="X"
    ... )
    >>> X
    gb.Tensor  "X"  nvals 3  shape (2, 3, 4)  dtype FP64  layout (0|1,2)  format bitmapr
    entries:
      (0, 1, 3): 1.0
      (0, 2, 0): 2.0
      (1, 0, 2): 3.0
    >>> X.layout
    Layout(rows=(0,), cols=(1, 2))
    >>> X.ss.matrix.shape  # 2 rows (axis 0) by 3 * 4 columns (axes 1 and 2)
    (2, 12)
    >>> X.shape, X.ndim, X.size, X.nvals
    ((2, 3, 4), 3, 24, 3)

``(0|1,2)`` is short for that layout. Shape operations change only the layout, in O(1), and
share the data. Operations move the data to another layout when they need one (a planned
"relayout"), so the layout is mostly something to know about, not to manage. Tensors that
share stored data copy it before one of them changes (copy-on-write), so a Tensor never sees
another's writes.

Construction
------------

.. code-block:: python

    >>> E = Tensor(gb.dtypes.INT64, (2, 3, 4))  # empty
    >>> E.nvals
    0
    >>> D = Tensor.from_dense(np.arange(6.0).reshape(1, 2, 3), missing_value=0.0)
    >>> D.to_coo()
    ((array([0, 0, 0, 0, 0]), array([0, 0, 1, 1, 1]), array([1, 2, 0, 1, 2])), array([1., 2., 3., 4., 5.]))
    >>> A = gb.Matrix.from_coo([0, 1], [1, 0], [5, 6], nrows=2, ncols=2)
    >>> Tensor.from_matrix(A).shape
    (2, 2)
    >>> Tensor.from_matrices([A, A, A], axis=0).shape  # stacked along a new axis
    (3, 2, 2)

``from_coo`` takes one index array per axis (the ``np.nonzero`` form) or an ``(ndim, nvals)``
array, and ``layout=`` to choose the stored layout (kept by later updates). Conversions out
are ``to_coo``, ``to_dense``, ``to_matrix``, ``to_vector`` and ``to_scalar``.

Shape operations
----------------

.. code-block:: python

    >>> X.permute_dims((2, 0, 1)).shape
    (4, 2, 3)
    >>> X.T.shape, X.mT.shape  # numpy's .T reverses all axes; .mT swaps the last two
    ((4, 3, 2), (2, 4, 3))
    >>> X.expand_dims(0).shape, X.expand_dims(0).squeeze().shape
    ((1, 2, 3, 4), (2, 3, 4))
    >>> X.reshape(2, 3, 2, 2).shape  # splits axis 2: a view, O(1)
    (2, 3, 2, 2)
    >>> X.reshape(6, -1).shape  # fuses axes 0 and 1 across the layout's groups: data moves
    (6, 4)
    >>> X.reshape(6, 4, copy=False)
    Traceback (most recent call last):
    ...
    ValueError: reshape from (2, 3, 4) to (6, 4) with copy=False: the layout (0|1,2) does not hold the result, so the data would move

``reshape`` (C order, one size may be -1) is a view, in O(1), when each run of axes it splits
or fuses lies within one group of the layout, in order; otherwise the data moves once.
``copy=False`` raises where it would move, as in the Array API. ``ravel()`` and ``flatten()``
are ``reshape(-1)`` and ``reshape(-1, copy=True)``. ``relayout(layout)`` stores a Tensor in a
layout of your choice.

Element-wise operations and broadcasting
----------------------------------------

Element-wise operations follow the Matrix and Vector ones, with numpy broadcasting:
``ewise_mult`` works on the intersection of entries, ``ewise_add`` on the union, and
``ewise_union`` on the union with a default for a missing side. Infix operators are
available as for Matrix (``*`` and ``&`` intersect, ``+`` and ``|`` unite), and operations
are lazy: ``.new()`` computes them.

.. code-block:: python

    >>> v = gb.Vector.from_coo([0, 3], [10.0, 20.0], size=4)
    >>> (X * v).new(name="Xv")  # v broadcast along the last axis, as in numpy
    gb.Tensor  "Xv"  nvals 2  shape (2, 3, 4)  dtype FP64  layout (0|1,2)  format bitmapr
    entries:
      (0, 1, 3): 20.0
      (0, 2, 0): 20.0
    >>> X.ewise_add(X, binary.plus).new().to_coo()[1]
    array([2., 4., 6.])
    >>> X.apply(unary.ainv).new().to_coo()[1]
    array([-1., -2., -3.])
    >>> X.select(gb.select.valuelt, 3.0).new().nvals  # value selectors only
    2

A union with a broadcast operand replicates it over the broadcast axes. That is refused past
``gb.config["tensor_max_expansion"]`` entries unless ``allow_expansion=True`` is given.

Reductions
----------

.. code-block:: python

    >>> X.reduce(monoid.plus, axis=2).new().to_dense(fill_value=0)
    array([[0., 1., 2.],
           [3., 0., 0.]])
    >>> X.sum(axis=(1, 2)).new().to_dense()
    array([3., 3.])
    >>> X.max().new().to_scalar().value  # over every axis: a 0-d Tensor
    3.0

A fiber with no entries stays missing (no initial value), as in GraphBLAS. ``sum``, ``prod``,
``min``, ``max``, ``any`` and ``all`` are shorthands for ``reduce`` with the matching monoid.

Contractions
------------

``tensordot`` contracts axes with a semiring (``plus_times`` by default); ``@`` (``matmul``)
contracts the last axis of the left operand with the second-to-last of the right one, with
the leading axes as batch axes; ``einsum`` takes numpy's subscripts and any semiring.

.. code-block:: python

    >>> U = np.ones((4, 2))
    >>> X.tensordot(U, ([2], [0])).new().shape  # a TTM (tensor times matrix)
    (2, 3, 2)
    >>> B = Tensor.from_dense(np.ones((2, 4, 5)))
    >>> (X @ B).new().shape  # batch axis 0
    (2, 3, 5)
    >>> einsum("ijk,k->ij", X, np.ones(4)).new().to_dense(fill_value=0)
    array([[0., 1., 2.],
           [3., 0., 0.]])
    >>> einsum("ijk,ijk->", X, X, op=semiring.max_times).new().to_scalar().value
    9.0
    >>> F = [np.ones((n, 2)) for n in X.shape]
    >>> X.mttkrp(F, 0).new().to_dense()  # MTTKRP, never forming the Khatri-Rao product
    array([[3., 3.],
           [3., 3.]])

``einsum`` with three or more operands picks a pairwise order (``opt_einsum`` when it is
installed), and recognizes the MTTKRP pattern (``"ijk,jr,kr->ir"``).

Masks and updates
-----------------

Updates work as for Matrix: ``C(mask, accum, replace) << expression``. A mask is a Tensor's
``.S`` (structure) or ``.V`` (values), or a Matrix's or Vector's of the same shape.

.. code-block:: python

    >>> C = Tensor(gb.dtypes.FP64, (2, 3, 4))
    >>> C(X.S) << X.apply(binary.times, 10.0)
    >>> C.to_coo()[1]
    array([10., 20., 30.])
    >>> C(accum=binary.plus) << X
    >>> C.to_coo()[1]
    array([11., 22., 33.])

Indexing
--------

``T[...]`` extracts with outer (orthogonal) indexing, as GraphBLAS extract does: ints drop
their axis, while slices, lists and integer arrays keep it (``T[[0, 1], [0, 1]]`` is 2 x 2,
not numpy's two elements).

.. code-block:: python

    >>> X[0].new().shape, X[:, [0, 2], 1:3].new().shape
    ((3, 4), (2, 2, 2))
    >>> X[0, 1, 3].new().to_scalar().value
    1.0

Assignment by index (``T[...] = value``) is not implemented yet.

Input and output
----------------

.. code-block:: python

    >>> import scipy.sparse
    >>> S = X.to_scipy_sparse()  # an n-d coo_array
    >>> Tensor.from_scipy_sparse(S).isequal(X)
    True

``from_pydata_sparse`` and ``to_pydata_sparse`` convert to and from pydata/sparse arrays
(COO, GCXS, DOK). A GCXS array is the Tensor's own storage (its compressed axes as rows),
so it converts with one copy of each array.

Engines
-------

Some operations have more than one implementation, chosen per call:

- **MTTKRP** runs a Numba kernel over the stored arrays when Numba is installed (float64
  values, ``plus_times``), else GraphBLAS products.
- **TTM** (``tensordot`` of one axis with a full float64 factor and ``plus_times``) runs
  one SuiteSparse product with array-valued entries when the SuiteSparse JIT is on, a Numba
  kernel, or GraphBLAS products; the choice follows measurements.

The Numba kernels are compiled on first use and cached on disk, under
``$GRAPHBLAS_KERNEL_CACHE`` (by default ``python-graphblas/tensor-kernels`` under
``$XDG_CACHE_HOME`` or ``~/.cache``; an empty value turns the cache off). The JIT engine
raises SuiteSparse's JIT control from its default ``"run"`` to ``"on"``, as user-defined
types do (see :doc:`udt`); if a compile then fails, it puts the control back, warns once,
and another engine runs. Every engine gives the same results; only the speed differs.

Limits
------

- Experimental: names and behavior may change.
- No assignment by index yet; no ``...`` and no repeated index (diagonal) in ``einsum``.
- No fill value: a missing entry has no value (as in all of python-graphblas).
- Each group of the layout holds at most 2\ :sup:`60` positions (a Matrix dimension).
- The ``suitesparse`` backend with SuiteSparse:GraphBLAS 10 or later only.

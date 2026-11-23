.. _tensor:

Tensors (experimental)
======================

.. warning::

    **THIS API IS EXPERIMENTAL AND MAY CHANGE.** ``Tensor`` and ``einsum`` are new, and their
    names, defaults and behavior may change in any release. They need the ``suitesparse``
    backend with SuiteSparse:GraphBLAS 10 or later. On the vanilla backend they import, but
    using them raises ``NotImplementedError``.

A **Tensor** is an n-dimensional sparse array: a Matrix and a Vector generalized to any
number of axes. Like the other collections it has one data type and a shape, and positions
may be empty (missing). It follows GraphBLAS for missing values (there is no fill value) and
numpy and the Array API for shapes, axes and broadcasting.

Two notebooks in the repository's ``notebooks`` directory run the Tensor end to end:
``Tensor Quickstart.ipynb`` works through this page on one small example, and
``Tensor Multiplex PageRank.ipynb`` ranks European airports over 37 airline networks at once.

The examples below assume:

.. code-block:: python

    >>> import numpy as np
    >>> import graphblas as gb
    >>> from graphblas import Tensor, binary, einsum, monoid, semiring, unary

The model: logical shape and stored layout
------------------------------------------

A Tensor stores its entries in one SuiteSparse Matrix, stored by row, plus a **layout**:
which logical axes are raveled (in C order, in the order given) into the Matrix's rows, and
which into its columns. The shape is always the logical one; the layout is how it is stored.

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

``(0|1,2)`` is short for that layout. Most shape operations change only the layout, in O(1),
and share the data. Other operations move the data to another layout when they need one, so
you rarely need to manage the layout yourself. Tensors that share stored data copy it before
one of them changes (copy-on-write), so a Tensor never sees another's writes.

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
    >>> Tensor.from_matrix(A).isequal(A)  # isequal and isclose accept a Matrix, Vector or Scalar
    True
    >>> A.isequal(Tensor.from_matrix(A))  # and a Matrix or Vector compares with a Tensor
    True

``from_coo`` accepts one index array per axis (the form ``np.nonzero`` returns) or one
``(ndim, nvals)`` array. Its ``layout=`` argument chooses the stored layout, which later
updates keep. Conversions out are ``to_coo``, ``to_dense``, ``to_matrix``, ``to_vector`` and
``to_scalar``.

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

``reshape`` works in C order, and one size may be -1. It is a view, in O(1), when each run of
axes it splits or fuses lies within one group of the layout, in order; otherwise the data
moves once. ``copy=False`` raises where the data would move, as in the Array API. ``ravel()``
and ``flatten()`` are ``reshape(-1)`` and ``reshape(-1, copy=True)``. ``relayout(layout)``
returns the Tensor stored in a layout of your choice.

``pad(pad_width)`` grows axes as ``numpy.pad`` does, with ``(before, after)`` positions per
axis. The entries keep their values and the new positions are missing:

.. code-block:: python

    >>> P = X.pad(((0, 0), (1, 0), (0, 2)))
    >>> P.shape
    (2, 4, 6)
    >>> P.to_coo()[0]  # each entry moved by the leading pads: axis 1 by one
    (array([0, 0, 1]), array([2, 3, 1]), array([3, 0, 2]))

Element-wise operations and broadcasting
----------------------------------------

Element-wise operations follow the Matrix and Vector ones, with numpy broadcasting:
``ewise_mult`` works on the intersection of entries, ``ewise_add`` on the union, and
``ewise_union`` on the union with a default for a missing side. Infix operators are
available as for Matrix: ``*`` and ``&`` intersect, ``+``, ``|`` and ``^`` unite. On their
own, ``X | Y``, ``X & Y`` and ``X ^ Y`` are lor, land and lxor of BOOL Tensors, and other
dtypes raise; give ``|`` and ``&`` an operator instead, as in ``binary.plus(X | Y)``.
Operations are lazy: ``.new()`` computes them.

.. code-block:: python

    >>> v = gb.Vector.from_coo([0, 3], [10.0, 20.0], size=4)
    >>> (X * v).new(name="Xv")  # v broadcasts along the last axis, as in numpy
    gb.Tensor  "Xv"  nvals 2  shape (2, 3, 4)  dtype FP64  layout (0|1,2)  format bitmapr
    entries:
      (0, 1, 3): 20.0
      (0, 2, 0): 20.0
    >>> X.ewise_add(X, binary.plus).new().to_coo()[1]
    array([2., 4., 6.])
    >>> X.apply(unary.ainv).new().to_coo()[1]
    array([-1., -2., -3.])
    >>> X.select(gb.select.valuelt, 3.0).new().nvals  # only value selectors work
    2

A union with a broadcast operand replicates it over the broadcast axes. When that would make
more than ``gb.config["tensor_max_expansion"]`` entries, it raises ``ValueError`` unless you
pass ``allow_expansion=True``.

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

``reduce`` also accepts aggregators, including those made of several reductions, such as
``agg.mean`` and ``agg.varp``, with the result types they have for Matrix. Integer and
boolean values are summed in that floating-point type, as numpy does for a mean, so the sums
do not wrap:

.. code-block:: python

    >>> X.reduce(gb.agg.count, axis=(1, 2)).new().to_dense()
    array([2, 1])
    >>> X.reduce(gb.agg.hypot).new().to_scalar().value  # sqrt(1 + 4 + 9)
    3.7416573867739413
    >>> X.reduce(gb.agg.mean, axis=(1, 2)).new().to_dense()  # (1 + 2) / 2 and 3 / 1
    array([1.5, 3. ])

Aggregators of positions work along one axis, as along the rows of a Matrix; ``agg.first``
and ``agg.last`` also take several axes, in C order over them:

.. code-block:: python

    >>> X.reduce(gb.agg.argmax, axis=2).new().to_dense(fill_value=-1)
    array([[-1,  3,  0],
           [ 2, -1, -1]])
    >>> X.reduce(gb.agg.first, axis=(1, 2)).new().to_dense()  # X[0, 1, 3] and X[1, 0, 2]
    array([1., 3.])

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

As in numpy, ``...`` stands for the axes the letters leave, a size-1 axis broadcasts against
the same index's axis in another operand, and an index repeated within one operand takes its
diagonal, which costs one pass over that operand's entries:

.. code-block:: python

    >>> einsum("...k,k->...", X, np.ones(4)).new().shape  # the leading axes, whatever they are
    (2, 3)
    >>> D = Tensor.from_dense(np.arange(9.0).reshape(3, 3))
    >>> einsum("ii", D).new().to_scalar().value  # a trace: 0 + 4 + 8
    12.0

With three or more operands, ``einsum`` contracts them in pairs when the semiring's multiply
distributes over its monoid, as with ``plus_times``, ``min_plus`` and ``max_plus``. The result
is then exact up to floating-point rounding and integer overflow. When the multiply also
commutes, ``einsum`` chooses the order of the pairs (with ``opt_einsum`` when it is
installed) and computes the MTTKRP pattern (``"ijk,jr,kr->ir"``) with ``mttkrp``. With other
semirings (``plus_pair``, ``plus_first``, ``max_times``, ...) it multiplies the operands of
each term left to right and forms every term before summing, which takes more memory.

A contraction moves an operand whose layout does not suit it, and does not keep the moved
copy. In a loop that contracts the same Tensor every step (a PageRank or a path query), store
it once in the layout the contraction uses: ``E = E.relayout(((0, 1), (2, 3)))`` for
``einsum("qu,qupv->pv", F, E)``, which contracts axes 0 and 1.

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

A repeated index repeats its slice, as with ``np.ix_``: ``X[:, [1, 1]]`` is 2 x 2 x 4. A
negative index counts from the end. An index out of range raises the exception Matrix raises
for the same key (``IndexError``, ``IndexOutOfBound`` or ``InvalidIndex``).

Assignment by index
-------------------

``T[...] = value`` and ``T[...] << value`` assign to the region the outer index selects, with
GraphBLAS assign semantics, as for Matrix. The value is a Tensor, Matrix, Vector, Scalar or
Python scalar and broadcasts against the region as in numpy (a Scalar fills it). Without
``accum``, the region takes the value's structure: where the value has no entry, the region's
entry is deleted. With ``accum``, the value's entries combine with the region's.

.. code-block:: python

    >>> Y = Tensor(gb.dtypes.FP64, (2, 3, 4))
    >>> Y[0, :, 1:3] = 1.0  # fill a 3 x 2 region
    >>> Y.nvals
    6
    >>> Y[1] = gb.Vector.from_coo([0, 3], [5.0, 6.0], size=4)  # broadcast over the 3 rows
    >>> Y[1].new().to_dense(fill_value=0)
    array([[5., 0., 0., 6.],
           [5., 0., 0., 6.],
           [5., 0., 0., 6.]])
    >>> Y(accum=binary.plus)[0, 0, :] << 10.0
    >>> Y[0, 0].new().to_dense(fill_value=0)
    array([10., 11., 11., 10.])
    >>> Y[0, 0] = gb.Vector.from_coo([2], [7.0], size=4)  # no accum: the rest is deleted
    >>> Y[0, 0].new().to_dense(fill_value=0)
    array([0., 0., 7., 0.])
    >>> del Y[1, :, 0]  # delete a region
    >>> Y[1].new().nvals
    3

A mask goes in one of two places, as for Matrix. ``T(mask)[...] << value`` is an *assign*:
the mask has the shape of ``T`` and applies to all of it, so with ``replace=True`` the
entries outside the region that the mask does not cover are deleted too.
``T[...](mask) << value`` is a *subassign*: the mask has the shape of the region and applies
only there.

.. code-block:: python

    >>> R = Tensor.from_dense(np.eye(3, 4, dtype=bool), missing_value=False)  # a 3 x 4 mask
    >>> Y[1](R.S) << 9.0  # only where the mask has entries
    >>> Y[1].new().to_dense(fill_value=0)
    array([[9., 0., 0., 6.],
           [0., 9., 0., 6.],
           [0., 0., 9., 6.]])

A repeated index assigns the last value given for it, and with ``accum`` the region's entry
combines with that one value, as in Matrix assignment.

Gathering elements
------------------

``take``, ``take_along_axis`` and ``vindex`` gather elements with numpy's rules for integer
(advanced) indexing; ``T[...]`` stays outer. The indices are ints, Scalars, lists or integer
arrays, and may repeat. A position that lands on a missing entry is missing in the result.
They compute at once and return a Tensor.

.. code-block:: python

    >>> X.take([3, 0, 3], axis=2).shape  # the indices' shape in place of axis 2
    (2, 3, 3)
    >>> X.take([23, 14]).to_coo()  # axis=None: of the flattened Tensor (C order)
    ((array([1]),), array([3.]))
    >>> X.vindex[[0, 1, 0], [1, 0, 0], [3, 2, 0]].to_coo()  # three points; the last is missing
    ((array([0, 1]),), array([1., 3.]))
    >>> X.take_along_axis(np.array([[[2], [3], [0]]]), axis=2).to_dense(fill_value=0)[..., 0]
    array([[0., 1., 2.],
           [3., 0., 0.]])

``vindex`` broadcasts its index arrays together, as numpy does. A negative index counts
from the end; with ``wraparound=False`` (``T.vindex(wraparound=False)[...]``) it raises
``IndexError`` instead. An index out of range raises ``IndexError``, except that ``take`` has
numpy's ``mode="wrap"`` and ``mode="clip"``.

Input and output
----------------

.. code-block:: python

    >>> import scipy.sparse
    >>> S = X.to_scipy_sparse()  # an n-d coo_array
    >>> Tensor.from_scipy_sparse(S).isequal(X)
    True

``to_scipy_sparse`` of a Tensor with more than 2 dimensions needs scipy 1.15 or later, which
added n-d COO arrays; with an older scipy it raises ``NotImplementedError``. 1-d and 2-d
Tensors convert with any scipy that python-graphblas supports.

``from_pydata_sparse`` and ``to_pydata_sparse`` convert to and from pydata/sparse arrays
(COO, GCXS, DOK). A GCXS array stores its data the way a Tensor does (its compressed axes as
rows), so it converts with one copy of each array. Its row pointer has one slot per position
of the compressed axes, so a GCXS export takes memory in proportion to the product of their
lengths, not to the number of entries: two compressed axes of length 2^20 make 2^40 slots.
Choose ``compressed_axes`` to keep that product small. A COO export takes memory in
proportion to the entries.

Engines
-------

Some operations have more than one implementation, chosen per call:

- **MTTKRP** runs a Numba kernel over the stored arrays when Numba is installed, else
  GraphBLAS products.
- **TTM** (``tensordot`` of one axis with a factor) runs one SuiteSparse product with
  array-valued entries when the SuiteSparse JIT is on, a Numba kernel, or GraphBLAS
  products, whichever measured faster for the case.
- **Gathers** run one SuiteSparse extract when the result is a product of index lists
  (``take``), else a Numba binary search over the stored arrays, else (without Numba)
  a SuiteSparse intersection with the points as a pattern Matrix.

The Numba and JIT engines for MTTKRP and TTM need full float64 factors and ``plus_times``
with a float64 result. The Tensor's own values may be bool, integer, float32 or float64
(count data often loads as int64 or bool); they are converted to float64 as GraphBLAS
typecasts them, so there is no need to cast first. Other cases use the GraphBLAS products:
integer factors, which give an integer result that wraps on overflow; complex values, which
give a complex result; and other semirings.

The Numba kernels are compiled on first use and cached on disk, under
``$GRAPHBLAS_KERNEL_CACHE`` (by default ``python-graphblas/tensor-kernels`` under
``$XDG_CACHE_HOME`` or ``~/.cache``; an empty value turns the cache off). When the JIT
engine runs, it raises SuiteSparse's JIT control from its default ``"run"`` to ``"on"``, as
user-defined types do (see :doc:`udt`). A case it leaves to another engine leaves the control
as it was. If a compile fails, the engine warns once and another engine runs. The control
then stays at ``"load"``, as after any failed compile, so later operations do not try to
compile again. Every engine computes the same results, up to the order of floating-point
sums; only the speed differs.

Limits
------

- No Tensor or Vector as indices.
- For complex values, the variance and standard deviation aggregators are Matrix's, from the
  complex square, not numpy's real variance.
- No fill value: a missing entry has no value (as in all of python-graphblas).
- Each group of the layout has at most 2\ :sup:`60` positions (a Matrix dimension).
- The ``suitesparse`` backend with SuiteSparse:GraphBLAS 10 or later only.
- scipy.sparse conversion of more than 2 dimensions needs scipy 1.15 or later.

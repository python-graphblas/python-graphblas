
Collections
===========

1. Introduce Matrix, Vector, Scalar

  - show repr for each
  - show from_values syntax for each
  - show attributes for each (size, nrows, ncols, nvals, ndim, shape)

2. Show new, dup, clear
3. For scalar, show .value to extract value, .get to extract with default
4. Discuss delayed objects and .new()

Creating new Vectors / Matrices
-------------------------------

.. code-block:: python

    A = Matrix.new(dtype, num_rows, num_cols)   # new_type
    B = A.dup()                                 # dup
    A = Matrix.from_values([row_indices], [col_indices], [values])  # build

New from delayed
----------------

Delayed objects can be used to create a new object using ``.new()`` method.

.. code-block:: python

    C = A.mxm(B, semiring).new()

Properties
----------

.. code-block:: python

    size = v.size                               # size
    nrows = M.nrows                             # nrows
    ncols = M.ncols                             # ncols
    nvals = M.nvals                             # nvals
    rindices, cindices, vals = M.to_values()    # extractTuples

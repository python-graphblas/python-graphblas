
.. module:: python-graphblas

python-graphblas documentation
==============================

:mod:`python-graphblas` is a Pythonic interface to the highly performant
`SuiteSparse:GraphBLAS <https://github.com/DrTimothyAldenDavis/GraphBLAS>`_ library
for performing graph analytics in the language of linear algebra.

Licensed under the `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   api_reference/index
   contributor_guide/index

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :fa:`play-circle,fa-3x`

    Getting Started
    ^^^^^^^^^^^^^^^

    .. link-button:: getting_started
        :text:
        :classes: stretched-link

    Exactly what you need to get up and running with
    python-graphblas. Contains a short introduction
    to the main ideas behind GraphBLAS.

    ---
    :fa:`code,fa-3x`

    User Guide
    ^^^^^^^^^^

    .. link-button:: user_guide
        :text:
        :classes: stretched-link

    The user guide covers the fundamental ideas and usage
    patterns of python-graphblas. It also contains an example
    notebook showing Single-Source Shortest Path.

    ---
    :fa:`cogs,fa-3x`

    API Reference
    ^^^^^^^^^^^^^

    .. link-button:: api_reference
        :text:
        :classes: stretched-link

    The API reference contains details about each class and method.
    It assumes familiarity with the core ideas and usage patterns.

    ---
    :fa:`plus-circle,fa-3x`

    Contributor Guide
    ^^^^^^^^^^^^^^^^^

    .. link-button:: contributor_guide
        :text:
        :classes: stretched-link

.. code-block:: python

    M.update(A.mxm(B, semiring.min_plus))

Operations
----------

.. code-block:: python

    M(mask, accum) << A.mxm(B, semiring)        # mxm
    w(mask, accum) << A.mxv(v, semiring)        # mxv
    w(mask, accum) << v.vxm(B, semiring)        # vxm
    M(mask, accum) << A.ewise_add(B, binaryop)  # eWiseAdd
    M(mask, accum) << A.ewise_mult(B, binaryop) # eWiseMult
    M(mask, accum) << A.kronecker(B, binaryop)  # kronecker
    M(mask, accum) << A.T                       # transpose

Extract
-------

.. code-block:: python

    M(mask, accum) << A[rows, cols]             # rows and cols are a list or a slice
    w(mask, accum) << A[rows, col_index]        # extract column
    w(mask, accum) << A[row_index, cols]        # extract row
    s = A[row_index, col_index].value           # extract single element

Assign
------

.. code-block:: python

    M(mask, accum)[rows, cols] << A             # rows and cols are a list or a slice
    M(mask, accum)[rows, col_index] << v        # assign column
    M(mask, accum)[row_index, cols] << v        # assign row
    M(mask, accum)[rows, cols] << s             # assign scalar to many elements
    M[row_index, col_index] << s                # assign scalar to single element
                                                # (mask and accum not allowed)
    del M[row_index, col_index]                 # remove single element

Apply
-----

.. code-block:: python

    M(mask, accum) << A.apply(unaryop)
    M(mask, accum) << A.apply(binaryop, left=s)   # bind-first
    M(mask, accum) << A.apply(binaryop, right=s)  # bind-second

Reduce
------

.. code-block:: python

    v(mask, accum) << A.reduce_rowwise(op)      # reduce row-wise
    v(mask, accum) << A.reduce_columnwise(op)   # reduce column-wise
    s(accum) << A.reduce_scalar(op)
    s(accum) << v.reduce(op)

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

Initialization
--------------

There is a mechanism to initialize ``graphblas`` with a context prior to use. This allows for setting the backend to
use as well as the blocking/non-blocking mode. If the context is not initialized, a default initialization will
be performed automatically.

.. code-block:: python

    import graphblas as gb
    # Context initialization must happen before any other imports
    gb.init('suitesparse', blocking=True)

    # Now we can import other items from graphblas
    from graphblas import binary, semiring
    from graphblas import Matrix, Vector, Scalar

Performant User Defined Functions
---------------------------------

Python-graphblas requires ``numba`` which enables compiling user-defined Python functions to native machine code for use in GraphBLAS.

Example customized UnaryOp:

.. code-block:: python

    from graphblas import unary
    from graphblas.operator import UnaryOp

    def force_odd_func(x):
        if x % 2 == 0:
            return x + 1
        return x

    UnaryOp.register_new('force_odd', force_odd_func)

    v = Vector.from_values([0, 1, 3], [1, 2, 3])
    w = v.apply(unary.force_odd).new()
    w  # indexes=[0, 1, 3], values=[1, 3, 3]

Similar methods exist for BinaryOp, Monoid, and Semiring.

Import/Export connectors to the Python ecosystem
------------------------------------------------

``graphblas.io`` contains functions for converting to and from:

.. code-block:: python

    import graphblas as gb

    # numpy arrays
    # 1-D array becomes Vector, 2-D array becomes Matrix
    A = gb.io.from_numpy(m)
    m = gb.io.to_numpy(A)

    # scipy.sparse matrices
    A = gb.io.from_scipy_sparse(m)
    m = gb.io.to_scipy_sparse(m, format='csr')

    # networkx graphs
    A = gb.io.from_networkx(g)
    g = gb.io.to_networkx(A)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

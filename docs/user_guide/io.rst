
Input/Output
============

There are several ways to get data into and out of python-graphblas.

.. _from-to-values:

From/To Values
--------------

Scalar has ``.from_value()`` and ``.value`` to get a Python scalar into and out of python-graphblas.

Matrix and Vector, instead, have a ``.from_coo()`` and a ``.to_coo()`` method.

``.from_coo()`` takes index(es) and values as either:

  - Python lists
  - NumPy arrays

If no dtype is provided, the data type is inferred from the values.

If no size (Vector) or nrows/ncols (Matrix) are provided, the shape is inferred based on the largest
index found for each dimension.

``.to_coo()`` returns numpy arrays. Indexes are always returned as ``uint64``, while the values
array will match the collection dtype.

.. code-block:: python

    v = gb.Vector.from_coo([1, 3, 6], [2, 3, 4], float, size=10)

.. csv-table::
    :class: matrix
    :header: 0,1,2,3,4,5,6,7,8,9,10

    ,2.0,,3.0,,,4.0,,,

.. code-block:: python

    >>> idx, vals = v.to_coo()
    >>> idx
    array([1, 3, 6], dtype=uint64)
    >>> vals
    array([2., 3., 4.])


NetworkX
--------

A python-graphblas Matrix can be created from a NetworkX graph using ``gb.io.from_networkx()``.

The dtype and weight label can be specified. A list of nodes can also be provided to only pull out a subset
of the NetworkX graph.

``gb.io.to_networkx()`` goes the other direction. It always returns a DiGraph with weights labelled as "weight".


Scipy.Sparse
------------

A python-graphblas Matrix can be created from a 2-D scipy.sparse array or matrix using
``gb.io.from_scipy_sparse()``.

``gb.io.to_scipy_sparse()`` will output a 2-D scipy.sparse array given a python-graphblas Matrix.
The scipy.sparse format can be specified. It defaults to "csr".

*Note about zero-weighted edges in scipy.sparse:* scipy.sparse makes the assumption that missing values
and zero-weighted edges can be treated identically. There are conversions within the scipy.sparse codebase
that drop zero-weighted edges. The conversion from python-graphblas to scipy.sparse attempts to preserve
zero-weighted edges, but the user should be aware of the potential for errors occurring when zero-weighted
edges are handled by scipy.sparse.

PyData.Sparse
-------------

A python-graphblas Matrix can be created from a 2-D (PyData) sparse array or matrix using
``gb.io.from_pydata_sparse()``.

``gb.io.to_pydata_sparse()`` will output a 2-D (PyData) sparse array given a python-graphblas Matrix.
The sparse format can be specified. It defaults to "coo".

NumPy (Dense)
-------------

While not useful for very large graphs, converting to and from small dense numpy arrays can be useful.

``Vector.from_dense()`` converts a 1-D array into a Vector and
``Matrix.from_dense()`` a 2-D array into a Matrix. When converting from numpy, a value may be
chosen to become a missing value, such as ``Matrix.from_dense(a, missing_value=0)``.

``.to_dense()`` converts a Vector or Matrix into a numpy array. If there are missing values, a fill
value should be given such as ``.to_dense(fill_value=0)``.

SuiteSparse Export/Import
-------------------------

For the fastest possible access to the underlying data structures in python-graphblas, SuiteSparse
extensions exist in the ``.ss`` namespace on Matrix and Vector collections.

``ss.export()`` gives options to export in a specific format or to leave the underlying data structures
as-is for the fastest access. This can be very useful when serializing a graph and sending it across the
wire.

The result is a Python dict with numpy arrays and other metadata inside. See the :ref:`api_reference`
for more details.

.. code-block:: python

    >>> A.ss.export('csr')
    {'indptr': array([0, 2, 4, 6], dtype=uint64),
     'col_indices': array([0, 2, 1, 2, 0, 2], dtype=uint64),
     'sorted_cols': True,
     'nrows': 3,
     'ncols': 3,
     'is_iso': False,
     'format': 'csr',
     'values': array([10, 20, 30, 40, 50, 60])}

Importing requires choosing the collection and calling the ``<collection>.ss.import_*`` methods.
A generic ``import_any()`` method will inspect the format and call the appropriate importer.
Otherwise, the format-specific import method name may be directly called.

.. code-block:: python

    >>> d = A.ss.export('csr')
    >>> d["values"][0] = -100  # modify the serialized data
    >>> M = gb.Matrix.ss.import_csr(**d)

Note that A is unchanged in the above example.

The SuiteSparse export has a ``give_ownership`` option. This performs a zero-copy
move operation and invalidates the original python-graphblas object. When extreme speed is needed or memory is
too limited to make a copy, this option may be needed.

Matrix Market files
-------------------

The `Matrix Market file format <https://math.nist.gov/MatrixMarket/formats.html>`_ is a common
file format for storing sparse arrays in human-readable ASCII.
Matrix Market files--also called MM files--often use ".mtx" file extension.
For example, many datasets in MM format can be found in `the SuiteSparse Matrix Collection <https://sparse.tamu.edu/>`_.

Use ``gb.io.mmread()`` to read a Matrix Market file to a python-graphblas Matrix,
and ``gb.io.mmwrite()`` to write a Matrix to a Matrix Market file.
These names match the equivalent functions in `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html>`_.

``scipy`` is required to be installed to read Matrix Market files.
If ``fast_matrix_market`` is installed, it will be used by default for
`much better performance <https://github.com/alugowski/fast_matrix_market>`_.

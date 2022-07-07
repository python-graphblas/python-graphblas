
Input/Output
============

There are several ways to get data into and out of python-graphblas.

From/To Values
--------------

Scalar has ``.from_value()`` and ``.value`` to get a Python scalar into and out of python-graphblas.

Matrix and Vector, instead, have a ``.from_values()`` and a ``.to_values()`` method.

``.from_values()`` takes index(es) and values as either:

  - Python lists
  - Numpy arrays

If no dtype is provided, the data type is inferred from the values.

If no size (Vector) or nrows/ncols (Matrix) are provided, the shape is inferred based on the largest
index found for each dimension.

``.to_values()`` returns numpy arrays. Indexes are always returned as ``uint64``, while the values
array will match the collection dtype.

.. code-block:: python

    v = gb.Vector.from_values([1, 3, 6], [2, 3, 4], float, size=10)

.. csv-table::
    :header: 0,1,2,3,4,5,6,7,8,9,10

    ,2.0,,3.0,,,4.0,,,

.. code-block:: python

    >>> idx, vals = v.to_values()
    >>> idx
    array([0, 1, 3], dtype=uint64)
    >>> vals
    array([4., 5., 6.])


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


Numpy (Dense)
-------------

While not useful for very large graphs, converting to and from small dense numpy arrays can be useful.

``gb.io.from_numpy()`` will convert a 1-D array into a Vector and a 2-D array into a Matrix. When converting
from numpy, zeros are treated as missing values.

``gb.io.to_numpy()`` will convert a Vector or Matrix into the dense equivalent in numpy, filling missing
values with zero.


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

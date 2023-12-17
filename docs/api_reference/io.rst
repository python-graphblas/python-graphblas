Input/Output
------------

NetworkX
~~~~~~~~

These methods require `networkx <https://networkx.org/>`_ to be installed.

.. autofunction:: graphblas.io.from_networkx

.. autofunction:: graphblas.io.to_networkx

NumPy
~~~~~

These methods convert to and from dense arrays. For more, see :ref:`IO in the user guide <from-to-values>`.

.. automethod:: graphblas.core.matrix.Matrix.from_dense

.. automethod:: graphblas.core.matrix.Matrix.to_dense

.. automethod:: graphblas.core.vector.Vector.from_dense

.. automethod:: graphblas.core.vector.Vector.to_dense

Scipy Sparse
~~~~~~~~~~~~

These methods require `scipy <https://scipy.org/>`_ to be installed.

.. autofunction:: graphblas.io.from_scipy_sparse

.. autofunction:: graphblas.io.to_scipy_sparse

PyData Sparse
~~~~~~~~~~~~~

These methods require `sparse <https://sparse.pydata.org/>`_ to be installed.

.. autofunction:: graphblas.io.from_pydata_sparse

.. autofunction:: graphblas.io.to_pydata_sparse

Matrix Market
~~~~~~~~~~~~~

Matrix Market is a `plain-text format <https://math.nist.gov/MatrixMarket/formats.html>`_ for storing graphs.

These methods require `scipy <https://scipy.org/>`_ to be installed.

.. autofunction:: graphblas.io.mmread

.. autofunction:: graphblas.io.mmwrite

Awkward Array
~~~~~~~~~~~~~

`Awkward Array <https://awkward-array.org/doc/main/>`_ is a library for nested,
variable-sized data, including arbitrary-length lists, records, mixed types,
and missing data, using NumPy-like idioms. Note that the intended use of the
``awkward-array``-related ``io`` functions is to convert ``graphblas`` objects to awkward,
perform necessary computations/transformations and, if required, convert the
awkward array back to ``graphblas`` format. To facilitate this conversion process,
``graphblas.io.to_awkward`` adds top-level attribute ``format``, describing the
format of the ``graphblas`` object (this attributed is used by the
``graphblas.io.from_awkward`` function to reconstruct the ``graphblas`` object).

.. autofunction:: graphblas.io.to_awkward

.. autofunction:: graphblas.io.from_awkward

Visualization
~~~~~~~~~~~~~

.. autofunction:: graphblas.viz.draw

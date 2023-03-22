Input/Output
------------

NetworkX
~~~~~~~~

These methods require `networkx <https://networkx.org/>`_ to be installed.

.. autofunction:: graphblas.io.from_networkx

.. autofunction:: graphblas.io.to_networkx

Numpy
~~~~~

These methods require `scipy <https://scipy.org/>`_ to be installed, as some
of the scipy.sparse machinery is used during the conversion process.

.. autofunction:: graphblas.io.from_numpy

.. autofunction:: graphblas.io.to_numpy

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

Visualization
~~~~~~~~~~~~~

.. autofunction:: graphblas.io.draw

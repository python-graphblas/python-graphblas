
.. _api_reference:

API Reference
=============

Collections
-----------

Matrix
~~~~~~

.. autoclass:: graphblas.Matrix
    :members:
    :special-members: __getitem__, __setitem__, __delitem__, __contains__, __iter__

Vector
~~~~~~

.. autoclass:: graphblas.Vector
    :members:
    :special-members: __getitem__, __setitem__, __delitem__, __contains__, __iter__

Scalar
~~~~~~

.. autoclass:: graphblas.Scalar
    :members:
    :special-members: __eq__, __bool__

Operators
---------

UnaryOp
~~~~~~~

.. autoclass:: graphblas.core.operator.UnaryOp()
    :members:

BinaryOp
~~~~~~~~

.. autoclass:: graphblas.core.operator.BinaryOp()
    :members:

Monoid
~~~~~~

.. autoclass:: graphblas.core.operator.Monoid()
    :members:

Semiring
~~~~~~~~

.. autoclass:: graphblas.core.operator.Semiring()
    :members:

IndexUnaryOp
~~~~~~~~~~~~

.. autoclass:: graphblas.core.operator.IndexUnaryOp()
    :members:

SelectOp
~~~~~~~~

.. autoclass:: graphblas.core.operator.SelectOp()
    :members:


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


Exceptions
----------

.. automodule:: graphblas.exceptions
    :members: InvalidObject, InvalidIndex, DomainMismatch, DimensionMismatch,
              OutputNotEmpty, OutOfMemory, IndexOutOfBound, Panic, EmptyObject,
              NotImplementedException, UdfParseError

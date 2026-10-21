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

Tensor (experimental)
~~~~~~~~~~~~~~~~~~~~~

**THIS API IS EXPERIMENTAL AND MAY CHANGE.** See :ref:`tensor`.

.. autoclass:: graphblas.Tensor
    :members:
    :special-members: __getitem__

.. autofunction:: graphblas.einsum

.. autoclass:: graphblas.core.tensor.Layout
    :members:

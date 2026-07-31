Types
-----

DataType
~~~~~~~~

The object returned by :func:`~graphblas.dtypes.register_new` and
:func:`~graphblas.dtypes.register_anonymous`, and carried by every collection's
``.dtype``. The JIT introspection properties below report what SuiteSparse
actually registered for a user-defined type; see :doc:`../user_guide/udt` for
the operator-side counterparts on a typed operator (``op.jit_c_name``,
``op.jit_c_source``).

.. autoclass:: graphblas.dtypes.DataType()
    :members: jit_c_name, jit_c_definition

Registering user-defined types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphblas.dtypes.register_new

.. autofunction:: graphblas.dtypes.register_anonymous

Types
-----

DataType
~~~~~~~~

The type of every collection's ``.dtype``, and what
:func:`~graphblas.dtypes.register_new` and :func:`~graphblas.dtypes.register_anonymous`
return. For a user-defined type, ``jit_c_name`` and ``jit_c_definition`` show the C name
and ``typedef`` SuiteSparse has for it. A typed operator has ``jit_c_name`` and
``jit_c_source``; see :ref:`tn_jit_introspection`.

.. autoclass:: graphblas.dtypes.DataType()
    :members: jit_c_name, jit_c_definition

Registering user-defined types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphblas.dtypes.register_new

.. autofunction:: graphblas.dtypes.register_anonymous


GraphBLAS Initialization
========================

GraphBLAS must be initialized before it can be used. This is done with the
``graphblas.init`` function. The backend and the mode must be specified.

.. code-block:: python

    import graphblas as gb

    # Context initialization must happen before any other imports
    gb.init("suitesparse", blocking=False)

    # Now we can import other items from graphblas
    from graphblas import binary, semiring
    from graphblas import Matrix, Vector, Scalar


Supported Backends
------------------

The only supported backend is current ``suitesparse``, although the plan is to support
additional backends in the future.

GraphBLAS Modes
---------------

The GraphBLAS spec contains the idea of **BLOCKING** vs. **NONBLOCKING** modes of operation.

Blocking mode requires immediate execution, while nonblocking mode allows for some aspects of
a function call to be delayed. Error checking on the inputs, compatibility with the output, etc
must be checked immediately, but the actual computation is allowed to be delayed. The hope is
that some GraphBLAS implementations will find optimizations if they see a string of computations
and realize that the output of an intermediate computation is not used and there is a way to
fuse operations together.

Thus far, this mechanism hasn't been fully realized. The only known usages are with adding or
removing a single element at a time. In nonblocking mode, this is much more efficient as the
element additions are queued up and the object container is fully computed at the last possible
moment.

Default Initialization
----------------------

If submodules are imported before calling ``graphblas.init``, a default initialization occurs.
The default backend is ``suitesparse`` and the default mode is ``nonblocking``.

If ``graphblas.init`` is called after the default initialization with different parameters,
an error will be raised.

Duplicate Initialization
------------------------

If ``graphblas.init`` is called multiple times with the same arguments, no error will be
raised. However, if the arguments do not match, an error will be raised.

There is no mechanism to change the initialization arguments after graphblas has been
initialized the first time.

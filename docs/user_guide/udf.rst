
User-defined Functions
======================

1. Show example of register_new
2. Discuss commonality for other operators
3. Discuss register_anonymous

Python-graphblas requires ``numba`` which enables compiling user-defined Python functions
to native machine code for use by the GraphBLAS backend. This provides functions which are
very performant.

Example user-defined UnaryOp:

.. code-block:: python

    from graphblas import unary

    def force_odd_func(x):
        if x % 2 == 0:
            return x + 1
        return x

    unary.register_new("force_odd", force_odd_func)

    v = Vector.from_coo([0, 1, 3, 4, 5], [1, 2, 3, 8, 14])
    w = v.apply(unary.force_odd).new()

.. csv-table:: w
    :class: matrix
    :header: 0,1,2,3,4,5

    1,3,,3,9,15


Similar methods exist for BinaryOp and IndexUnaryOp. User-defined Monoids and Semirings are
constructed out of previously defined and built-in UnaryOps and BinaryOps.

Auto-registration of Lambdas
----------------------------

As a convenience, any lambda expression used in place of a UnaryOp will be automatically
compiled as registered anonymously.

Example lambda usage:

.. code-block:: python

    v.apply(lambda x: x % 5 - 2).new()

.. csv-table::
    :class: matrix
    :header: 0,1,2,3,4,5

    -1,0,,1,1,2

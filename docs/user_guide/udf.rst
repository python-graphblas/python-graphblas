
User-defined Functions
======================

1. Show example of register_new
2. Discuss commonality for other operators
3. Discuss register_anonymous

Python-graphblas requires ``numba`` which enables compiling user-defined Python functions to native machine code for use in GraphBLAS.

Example customized UnaryOp:

.. code-block:: python

    from graphblas import unary
    from graphblas.operator import UnaryOp

    def force_odd_func(x):
        if x % 2 == 0:
            return x + 1
        return x

    UnaryOp.register_new('force_odd', force_odd_func)

    v = Vector.from_values([0, 1, 3], [1, 2, 3])
    w = v.apply(unary.force_odd).new()
    w  # indexes=[0, 1, 3], values=[1, 3, 3]

Similar methods exist for BinaryOp, Monoid, and Semiring.

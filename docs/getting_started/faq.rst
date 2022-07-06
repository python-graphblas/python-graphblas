
.. _faq:

Frequently Asked Questions
==========================

FAQs
----

What is the relationship between python-graphblas and pygraphblas?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

They are different libraries with similar goals. Both wrap SuiteSparse:GraphBLAS and use the
same underlying ``python-suitesparse-graphblas`` CFFI wrapper, making their underlying objects
compatible with each other.

The biggest difference is in how the C calls are mapped to equivalent Python functions.

For simple expressions, the two libraries are very similar.

.. code-block:: python

    # pygraphblas
    A += 1

.. code-block:: python

    # python-graphblas
    A += 1

For more complex expressions, however, the two libraries diverge significantly.
`pygraphblas` tends towards the `numpy` style of immediate execution and using
keyword arguments to affect the output.

.. code-block:: python

    # pygraphblas
    A.mxm(B.transpose(), mask=A, out=C, accum=FP64.PLUS, semiring=FP64.MIN_PLUS)

`python-graphblas` uses delayed expressions and keeps the output-affecting arguments
together with the output.

.. code-block:: python

    # python-graphblas
    C(A, accum=binary.plus) << semiring.min_plus(A @ B.T)

`python-graphblas` also contains additional features, such as the `Recorder` and advanced aggregators.

What is the performance penalty of writing algorithms with python-graphblas vs writing them directly in C?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For large graphs, the performance penalty is very small as the bulk of the work
of operating on Matrices and Vectors is done purely in C. The overhead is only in the calls made
to C.

For small graphs, the call overhead may become more significant, but smaller graphs usually don't
take much time to compute, so the extra overhead may not be noticeable at a human scale.
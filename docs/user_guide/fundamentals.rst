
Fundamentals
============

1. Pull from powerpoint to show mapping from C to python
2. Discuss delayed objects
3. Discuss << and .update()
4. Discuss .new()
5. Discuss functional vs method calling syntax
6. Discuss comparison using .isequal and .isclose


The approach taken with this library is to follow the C-API specification as closely as possible while making improvements
allowed with the Python syntax. Because the spec always passes in the output object to be written to, we follow the same,
which is very different from the way Python normally operates. In fact, many who are familiar with other Python data
libraries (numpy, pandas, etc) will find it strange to not create new objects for every call.

At the highest level, the goal is to separate output, mask, and accumulator on the left side of the assignment
operator ``=`` and put the computation on the right side. Unfortunately, that approach doesn't always work very well
with how Python handles assignment, so instead we (ab)use the left-shift ``<<`` notation to give the same flavor of
assignment. This opens up all kinds of nice possibilities.

This is an example of how the mapping works:

.. code-block:: c

    // C call
    GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, NULL)

.. code-block:: python

    # Python call
    M(mask.V, accum=binary.plus) << A.mxm(B, semiring.min_plus)

The expression on the right ``A.mxm(B)`` creates a delayed object which does no computation. Once it is used in the
``<<`` expression with ``M``, the whole thing is translated into the equivalent GraphBLAS call.

Delayed objects also have a ``.new()`` method which can be used to force computation and return a new
object. This is convenient and often appropriate, but will create many unnecessary objects if used in a loop. It
also loses the ability to perform accumulation with existing results. For best performance, following the standard
GraphBLAS approach of (1) creating the object outside the loop and (2) using the object repeatedly within each loop
is a much better approach, even if it doesn't feel very Pythonic.

Descriptor flags are set on the appropriate elements to keep logic close to what it affects. Here is the same call
with descriptor bits set. ``ttcsr`` indicates transpose the first and second matrices, complement the structure of the mask,
and do a replacement on the output.

.. code-block:: c

    // C call
    GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, desc.ttcsr)

.. code-block:: python

    # Python call
    M(~mask.S, accum=binary.plus, replace=True) << A.T.mxm(B.T, semiring.min_plus)

The objects receiving the flag operations (``A.T``, ``~mask``, etc) are also delayed objects. They hold on to the state but
do no computation, allowing the correct descriptor bits to be set in a single GraphBLAS call.

If no mask or accumulator is used, the call looks like this:

.. code-block:: python

    M << A.mxm(B, semiring.min_plus)

The use of ``<<`` to indicate updating is actually just syntactic sugar for a real ``.update()`` method. The above
expression could be written as:

.. code-block:: python

    M.update(A.mxm(B, semiring.min_plus))

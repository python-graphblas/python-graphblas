
Operations
==========

1. Walk thru operations, describing what the operation does (show examples)
2. Show functional and method syntax for both (where appropriate)
3. Show single element get/set/del operations as well as the .get(index, default=...) method

Operations
----------

.. code-block:: python

    M(mask, accum) << A.mxm(B, semiring)        # mxm
    w(mask, accum) << A.mxv(v, semiring)        # mxv
    w(mask, accum) << v.vxm(B, semiring)        # vxm
    M(mask, accum) << A.ewise_add(B, binaryop)  # eWiseAdd
    M(mask, accum) << A.ewise_mult(B, binaryop) # eWiseMult
    M(mask, accum) << A.kronecker(B, binaryop)  # kronecker
    M(mask, accum) << A.T                       # transpose

Extract
-------

.. code-block:: python

    M(mask, accum) << A[rows, cols]             # rows and cols are a list or a slice
    w(mask, accum) << A[rows, col_index]        # extract column
    w(mask, accum) << A[row_index, cols]        # extract row
    s = A[row_index, col_index].value           # extract single element

Assign
------

.. code-block:: python

    M(mask, accum)[rows, cols] << A             # rows and cols are a list or a slice
    M(mask, accum)[rows, col_index] << v        # assign column
    M(mask, accum)[row_index, cols] << v        # assign row
    M(mask, accum)[rows, cols] << s             # assign scalar to many elements
    M[row_index, col_index] << s                # assign scalar to single element
                                                # (mask and accum not allowed)
    del M[row_index, col_index]                 # remove single element

Apply
-----

.. code-block:: python

    M(mask, accum) << A.apply(unaryop)
    M(mask, accum) << A.apply(binaryop, left=s)   # bind-first
    M(mask, accum) << A.apply(binaryop, right=s)  # bind-second

Reduce
------

.. code-block:: python

    v(mask, accum) << A.reduce_rowwise(op)      # reduce row-wise
    v(mask, accum) << A.reduce_columnwise(op)   # reduce column-wise
    s(accum) << A.reduce_scalar(op)
    s(accum) << v.reduce(op)
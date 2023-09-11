
Operations
==========

Matrix Multiply
---------------

The GraphBLAS spec contains three methods for matrix multiplication, depending on whether
the inputs are Matrix or Vector.

  - **mxm** -- Matrix-Matrix multplications
  - **mxv** -- Matrix-Vector multiplication
  - **vxm** -- Vector-Matrix multiplication

These three methods exist on python-graphblas collections, but the preferred approach is using
the ``@`` symbol, which indicates matrix multiplication in Python.

The default semiring for matrix multiplication is ``plus_times``, but any semiring may be used
instead.

Vectors do not have a natural row or column orientation. However, when multiplied on the left side
of a Matrix, a Vector is treated as a 1xn row matrix. When multiplied on the right side of a Matrix,
a Vector is treated as an nx1 column matrix.

**Matrix-Matrix** Multiply Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2],
        [1, 2, 2, 3, 3],
        [2., 5., 1.5, 4.25, 0.5],
        nrows=4,
        ncols=4
    )
    B = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 2, 0, 1, 1, 2, 0, 1],
        [3., 2., 9., 6., 3., 1., 0., 5.]
    )
    C = gb.Matrix(float, A.nrows, B.ncols)

    # These are equivalent
    C << A.mxm(B, op="min_plus")  # method style
    C << gb.semiring.min_plus(A @ B)  # functional style

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2,3
    :stub-columns: 1

    **0**,,2.0,5.0,
    **1**,,,1.5,4.25
    **2**,,,,0.5
    **3**,,,,

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,3.0,2.0
    **1**,9.0,6.0,
    **2**,,3.0,1.0
    **3**,0.0,5.0,

.. csv-table:: C << min_plus(A @ B)
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,11.0,8.0,6.0
    **1**,4.25,4.5,2.5
    **2**,0.5,5.0,
    **3**,,,

**Matrix-Vector** Multiply Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2],
        [1, 2, 2, 3, 3],
        [2., 5., 1.5, 4.25, 0.5],
        nrows=4,
        ncols=4
    )
    v = gb.Vector.from_coo([0, 1, 3], [10., 20., 40.])
    w = gb.Vector(float, A.nrows)

    # These are equivalent
    w << A.mxv(v, op="plus_times")  # method style
    w << gb.semiring.plus_times(A @ v)  # functional style

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2,3
    :stub-columns: 1

    **0**,,2.0,5.0,
    **1**,,,1.5,4.25
    **2**,,,,0.5
    **3**,,,,

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3

    10.0,20.0,,40.0

.. csv-table:: w << plus_times(A @ v)
    :class: inline matrix
    :header: 0,1,2,3

    40.0,170.0,20.0,

**Vector-Matrix** Multiply Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3], [10., 20., 40.])
    B = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 2, 0, 1, 1, 2, 0, 1],
        [3., 2., 9., 6., 3., 1., 0., 5.]
    )
    u = gb.Vector(float, B.ncols)

    # These are equivalent
    u << v.vxm(B, op="plus_plus")  # method style
    u << gb.semiring.plus_plus(v @ B)  # functional style

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3

    10.0,20.0,,40.0

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,3.0,2.0
    **1**,9.0,6.0,
    **2**,,3.0,1.0
    **3**,0.0,5.0,

.. csv-table:: u << plus_plus(v @ B)
    :class: inline matrix
    :header: 0,1,2

    69.0,84.0,12.0

Element-wise Intersection
-------------------------

Two identically shaped collections can be intersected element-wise. Locations where only one of the
two collections contains a value will be missing in the output.

The GraphBLAS spec calls this operation ``eWiseMult`` because it has the same behavior as sparse
multiplication when missing values are treated as zero. As a result, ``binary.times`` is the
default operator for element-wise intersection.

The method is named ``ewise_mult``, following the spec. The functional syntax uses the Python
symbol for intersection ``&``.

Example usage:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2],
        [1, 2, 0, 2, 1],
        [2., 5., 1.5, 4., 0.5]
    )
    B = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 1, 1, 2],
        [3., -2., 0., 6., 3., 1.]
    )
    C = gb.Matrix(float, A.nrows, A.ncols)

    # These are equivalent
    C << A.ewise_mult(B, op="min")  # method style
    C << gb.binary.min(A & B)  # functional style

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,,4.0
    **2**,,0.5,

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,3.0,-2.0
    **1**,0.0,6.0,
    **2**,,3.0,1.0

.. csv-table:: C << min(A & B)
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,-2.0
    **1**,0.0,,
    **2**,,0.5,

Element-wise Union
------------------

Two identically shaped collections can perform a union element-wise. Locations where only one of the
two collections contains a value will contain that value in the output. Where they overlap, the operator
will compute the result.

The GraphBLAS spec calls this operation ``eWiseAdd`` because it has the same behavior as sparse
addition when missing values are treated as zero. As a result, ``binary.plus`` is the
default operator for element-wise union.

There are two methods in python-graphblas for element-wise union.

  - ``ewise_add``

    This is the official method based on the spec. It follows the spec by using a single value as-is when
    there is no overlap. For binary operations which are associative (plus, and, min, etc),
    ewise_add always gives the right answer. However, when the operation is not associative (minus, div, etc),
    ewise_add can have surprising results.

      - ``7 - 3 = 4``
      - ``7 - missing = 7``
      - ``missing - 7 = 7``  <-- *This might seem unexpected, but it is correct*

  - ``ewise_union``

    This is an extension provided by SuiteSparse:GraphBLAS. It adds a ``left_default`` and ``right_default``
    parameter that specify what the missing value should be when there is only a single value.

    If we set ``left_default=0`` and ``right_default=0``, then

      - ``7 - 3 = 4``
      - ``7 - missing = 7 - 0 = 7``
      - ``missing - 7 = 0 - 7 = -7`` <-- *This gives the expected answer*


The functional syntax uses the Python symbol for union ``|`` for both methods. To specify that ``ewise_union``
should be used with the functional syntax, ``left_default`` and ``right_default`` keywords are used.

**eWiseAdd** Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 2],
        [9., 2., 5., 1.5, 4.],
        nrows=3
    )
    B = gb.Matrix.from_coo(
        [0, 0, 0, 2, 2, 2],
        [0, 1, 2, 0, 1, 2],
        [4., 0., -2., 6., 3., 1.]
    )
    C = gb.Matrix(float, A.nrows, A.ncols)

    # These are equivalent
    C << A.ewise_add(B, op="minus")  # method style
    C << gb.binary.minus(A | B)  # functional style

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,9.0,2.0,5.0
    **1**,1.5,,4.0
    **2**,,,

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,4.0,0.0,-2.0
    **1**,,,
    **2**,6.0,3.0,1.0

.. csv-table:: C << A.ewise_add(B, 'minus')
    :class: inline matrix
    :header: ,0,1,2,
    :stub-columns: 1

    **0**,5.0,2.0,7.0
    **1**,1.5,,4.0
    **2**,6.0,3.0,1.0

**eWiseUnion** Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 2],
        [9., 2., 5., 1.5, 4.],
        nrows=3
    )
    B = gb.Matrix.from_coo(
        [0, 0, 0, 2, 2, 2],
        [0, 1, 2, 0, 1, 2],
        [4., 0., -2., 6., 3., 1.]
    )
    C = gb.Matrix(float, A.nrows, A.ncols)

    # These are equivalent
    C << A.ewise_union(B, op="minus", left_default=0, right_default=0)  # method style
    C << gb.binary.minus(A | B, left_default=0, right_default=0)  # functional style

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,9.0,2.0,5.0
    **1**,1.5,,4.0
    **2**,,,

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,4.0,0.0,-2.0
    **1**,,,
    **2**,6.0,3.0,1.0

.. csv-table:: C << A.ewise_union(B, 'minus', 0, 0)
    :class: inline matrix
    :header: ,0,1,2,
    :stub-columns: 1

    **0**,5.0,2.0,7.0
    **1**,1.5,,4.0
    **2**,-6.0,-3.0,-1.0

Extract
-------

Extraction is GraphBLAS takes a subset of a Vector or Matrix based on a set of indices.
Extraction uses normal Python slice syntax.

Extraction is not a shape-preserving operation, so indexes may be remapped during the process. For
example, extracting indices [1, 3, 5] will yield an object with indices [0, 1, 2].

If the index is a list of indices or a slice, that dimension will be preserved. If the index
is an integer, the dimension will be collapsed.

  - **Matrix[**\ *list/slice*, *list/slice*\ **] -> Matrix**
  - **Matrix[**\ *list/slice*, *int*\ **] -> Vector**
  - **Matrix[**\ *int*, *list/slice*\ **] -> Vector**
  - **Matrix[**\ *int*, *int*\ **] -> Scalar**
  - **Vector[**\ *list/slice*\ **] -> Vector**
  - **Vector[**\ *int*\ **] -> Scalar**

Vector Slice Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3, 4, 6], [10., 2., 40., -5., 24.])
    w = gb.Vector(float, 4)

    w << v[:4]

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    10.0,2.0,,40.0,-5.0,,24.0

.. csv-table:: w << v[:4]
    :class: inline matrix
    :header: 0,1,2,3

    10.0,2.0,,40.0

Matrix List Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 1, 0, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    C = gb.Matrix(float, 2, A.ncols)

    C << A[[0, 2], :]

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,4.0,
    **2**,0.5,,-7.0

.. csv-table:: C << A[[0, 2], :]
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,0.5,,-7.0

Assign
------

Assignment in GraphBLAS takes a smaller collection and expands it into a larger collection based
on defined indices. It can be thought of as the inverse of Extract. The number of indices must match the
shape of the input collection being assigned. However, the actual index positions refer to the location
within the output object.

Assignment uses normal Python slice syntax.

Smaller rank objects can be assigned if the index is an integer rather than a list or slice.
For example, assigning a Vector into a Matrix is possible if either the row or column index is
an integer.

Assigning a Scalar is also possible for any combination of integer, list or slice index.

Matrix-Matrix Assignment Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 1, 0, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    B = gb.Matrix.from_coo(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [-99., -98., -97., -96.]
    )
    A[::2, ::2] << B

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,4.0,
    **2**,0.5,,-7.0

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1
    :stub-columns: 1

    **0**,-99.0,-98.0
    **1**,-97.0,-96.0

.. csv-table:: A[::2, ::2] << B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,-99.0,2.0,-98.0
    **1**,1.5,4.0,
    **2**,-97.0,,-96.0

Matrix-Vector Assignment Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 1, 0, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    v = gb.Vector.from_coo([2], [-99.])

    A[1, :] << v

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,4.0,
    **2**,0.5,,-7.0

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2

    ,,-99.0

.. csv-table:: A[1, :] << v
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,,,-99.0
    **2**,0.5,,-7.0

Vector-Scalar Assignment Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3, 4, 6], [10, 2, 40, -5, 24])

    v[:4] << 99

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    10,2,,40,-5,,24

.. csv-table:: v[:4] << 99
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    99,99,99,99,-5,,24

Apply
-----

Apply takes an operator and applies it to every non-empty element in a collection.
The operator can be unary, index unary, or binary.

For the case of binary, an additional scalar argument must be provided as either the
left or right argument, with the other argument being provided by the collection elements.

The method name is ``apply`` and the functional form simply uses the operator as a calling
function with the collection as the argument.

**Unary** Apply Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3], [10., 20., 40.])
    w = gb.Vector(float, v.size)

    # These are equivalent
    w << v.apply(gb.unary.minv)
    w << gb.unary.minv(v)

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3

    10.0,20.0,,40.0

.. csv-table:: w << minv(v)
    :class: inline matrix
    :header: 0,1,2,3

    0.1,0.05,,0.025

**IndexUnary** Apply Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3], [10., 20., 40.])
    w = gb.Vector(int, v.size)

    # These are equivalent
    w << v.apply(gb.indexunary.index)
    w << gb.indexunary.index(v)

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3

    10.0,20.0,,40.0

.. csv-table:: w << index(v)
    :class: inline matrix
    :header: 0,1,2,3

    0,1,,3

**Binary** Apply Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3], [10., 20., 40.])
    w = gb.Vector(float, v.size)

    # These are all equivalent
    w << v.apply("minus", right=15)
    w << gb.binary.minus(v, right=15)
    w << v - 15

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3

    10.0,20.0,,40.0

.. csv-table:: w << v.apply('minus', right=15)
    :class: inline matrix
    :header: 0,1,2,3,

    -5.0,5.0,,25.0

Select
------

Select takes an index unary operator and applies it to every non-missing element of a collection.
If the result is True, the element remains in the output. If the result is False, the element
becomes missing in the result. Thus the output is a filtered version of the original collection.

Upper Triangle Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 1, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    C = gb.Matrix(float, A.nrows, A.ncols)

    # These are equivalent
    C << A.select("triu")
    C << gb.select.triu(A)

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,,4.0
    **2**,,0.5,-7.0

.. csv-table:: C << select.triu(A)
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,,,4.0
    **2**,,,-7.0

Select by Value Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3, 4, 6], [10., 2., 40., -5., 24.])
    w = gb.Vector(float, v.size)

    # These are equivalent
    w << v.select(">=", 5)
    w << gb.select.value(v >= 5)

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    10.0,2.0,,40.0,-5.0,,24.0

.. csv-table:: w << select.value(v >= 5)
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    10.0,,,40.0,,,24.0

Reduce
------

Reduction reduces the number of dimensions of a collection. A Matrix can become a Vector, and a Matrix or
Vector can be reduced to a Scalar.

When reducing a Matrix to a Vector, the reduction can be done rowwise or columnwise.

A monoid or aggregator is used to perform the reduction.

**Matrix-to-Vector** Columnwise Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 3, 0, 1, 0, 1],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    w = gb.Vector(float, A.ncols)

    w << A.reduce_columnwise("times")

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2,3
    :stub-columns: 1

    **0**,,2.0,,5.0
    **1**,1.5,4.0,,
    **2**,0.5,-7.0,,

.. csv-table:: w << A.reduce_columnwise('times')
    :class: inline matrix
    :header: ,0,1,2,3

    ,0.75,-56.0,,5.0

**Matrix-to-Scalar** Example:

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 3, 0, 1, 0, 1],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    s = gb.Scalar(float)

    s << A.reduce_scalar("max")

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2,3
    :stub-columns: 1

    **0**,,2.0,,5.0
    **1**,1.5,4.0,,
    **2**,0.5,-7.0,,

.. csv-table:: s << A.reduce_scalar('max')
    :class: inline matrix
    :header: ,,,,

    5.0

**Vector-to-Scalar** Aggregator Example:

.. code-block:: python

    v = gb.Vector.from_coo([0, 1, 3, 4, 6], [10., 2., 40., -5., 24.])
    s = gb.Scalar(int)

    # These are equivalent
    s << v.reduce("argmin")
    s << gb.agg.argmin(v)

.. csv-table:: v
    :class: inline matrix
    :header: 0,1,2,3,4,5,6

    10.0,2.0,,40.0,-5.0,,24.0

.. csv-table:: s << argmin(v)
    :class: inline matrix
    :header: ,,,

    4

Transpose
---------

The transpose can either a descriptor flag set on the input of a computation or the final computation
itself.

To force the transpose to be computed by itself, use it by itself as the right-hand side of a computation.

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 3, 0, 1, 0, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    C = gb.Matrix(float, A.ncols, A.nrows)

    C << A.T

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1,2,3
    :stub-columns: 1

    **0**,,2.0,,5.0
    **1**,1.5,4.0,,
    **2**,0.5,,-7.0,

.. csv-table:: C << A.T
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,1.5,0.5
    **1**,2.0,4.0,
    **2**,,,-7.0
    **3**,5.0,,

Kronecker
---------

The `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product>`_ of two matrices multiplies
every element of A (m×n) by every element of B (p×q) to create a pm×qn block matrix.

The Kronecker product uses a binary operator.

.. code-block:: python

    A = gb.Matrix.from_coo(
        [0, 0, 1],
        [0, 1, 0],
        [1., -2., 3.]
    )
    B = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 1, 0, 2],
        [2., 5., 1.5, 4., 0.5, -7.]
    )
    C = gb.Matrix(float, A.nrows * B.nrows, A.ncols * B.ncols)

    C << A.kronecker(B, "times")

.. csv-table:: A
    :class: inline matrix
    :header: ,0,1
    :stub-columns: 1

    **0**,1.0,-2.0
    **1**,3.0,

.. csv-table:: B
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,2.0,5.0
    **1**,1.5,4.0,
    **2**,0.5,,-7.0

.. csv-table:: C << A.kronecker(B, 'times')
    :class: inline matrix
    :header: ,0,1,2,3,4,5
    :stub-columns: 1

    **0**,,2.0,5.0,,-4.0,-10.0
    **1**,1.5,4.0,,-3.0,-8.0,
    **2**,0.5,,-7.0,-1.0,,14.0
    **3**,,6.0,15.0,,,
    **4**,4.5,12.0,,,,
    **5**,1.5,,-21.0,,,

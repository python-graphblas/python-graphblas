
User-defined Functions (UDFs)
=============================

Python-graphblas requires ``numba`` which enables compiling user-defined Python functions
to native machine code for use by the GraphBLAS backend. This provides functions which are
very performant.

Example user-defined UnaryOp:

.. code-block:: python

    from graphblas import Vector, unary


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

``register_new`` adds the operator to the ``graphblas.unary`` namespace, as
``unary.force_odd``, and compiles the function for each built-in type Numba can compile
it for. ``register_anonymous`` returns the operator without adding it to a namespace,
for an operator used in one place or made inside a function.

Kinds of operators
------------------

Similar methods exist for BinaryOp, IndexUnaryOp, SelectOp and IndexBinaryOp. The
function's signature depends on the kind of operator:

==================  ======================================  ====================================
Operator            Function signature                      Returns
==================  ======================================  ====================================
``UnaryOp``         ``f(x)``                                a value
``BinaryOp``        ``f(x, y)``                             a value
``IndexUnaryOp``    ``f(x, ix, jx, thunk)``                 a value
``SelectOp``        ``f(x, ix, jx, thunk)``                 a bool: whether to keep ``x``
``IndexBinaryOp``   ``f(x, ix, jx, y, iy, jy, theta)``      a value
==================  ======================================  ====================================

``ix`` and ``jx`` are the row and column of ``x``, and ``iy`` and ``jy`` those of ``y``;
in a Vector, ``ix`` is the index and ``jx`` is 0. ``thunk`` and ``theta`` are a scalar
given when the operator is used. IndexBinaryOps need SuiteSparse:GraphBLAS 9.4 or later.

A user-defined Monoid is built from a user-defined BinaryOp and its identity, and a
Semiring from a Monoid and a BinaryOp, built-in or user-defined:

.. code-block:: python

    from graphblas import Matrix, Vector, binary, monoid, select, semiring


    def absmax(x, y):
        return max(abs(x), abs(y))


    binary.register_new("absmax", absmax)
    monoid.register_new("absmax", binary.absmax, 0)
    semiring.register_new("absmax_times", monoid.absmax, binary.times)

    v = Vector.from_coo([0, 1, 2], [3, -7, 5])
    largest = v.reduce(monoid.absmax).new()  # 7


    def upper_and_big(x, i, j, thunk):
        return j > i and x > thunk


    select.register_new("upper_and_big", upper_and_big)

    A = Matrix.from_coo([0, 0, 1, 2], [0, 1, 2, 1], [5, 1, 7, 9])
    B = select.upper_and_big(A, 2).new()  # keeps only A[1, 2], which is 7

Auto-registration of Lambdas
----------------------------

As a convenience, any function, such as a lambda expression, used in place of a UnaryOp
or BinaryOp is compiled for the operand types and registered anonymously.

Example lambda usage:

.. code-block:: python

    v = Vector.from_coo([0, 1, 3, 4, 5], [1, 2, 3, 8, 14])
    v.apply(lambda x: x % 5 - 2).new()

.. csv-table::
    :class: matrix
    :header: 0,1,2,3,4,5

    -1,0,,1,1,2

Parameterized operators
-----------------------

To give an operator a parameter when it is used, such as a scale factor, write a function
that takes the parameter and returns the function to compile, and register it with
``parameterized=True``:

.. code-block:: python

    def make_scaled_add(scale):
        def inner(x, y):
            return scale * (x + y)

        return inner


    binary.register_new("scaled_add", make_scaled_add, parameterized=True)

    x = Vector.from_coo([0, 1], [1.0, 2.0])
    y = Vector.from_coo([0, 1], [3.0, 4.0])
    x.ewise_mult(y, binary.scaled_add(2.0)).new()  # [8.0, 12.0]

.. _udf_indexbinary:

IndexBinaryOps and theta
------------------------

An IndexBinaryOp receives the row and column of both operands, and a scalar ``theta``.
Binding ``theta`` makes a BinaryOp, which works in ``ewise_mult``, ``ewise_add``,
``ewise_union`` and ``kronecker``. SuiteSparse does not accept it in ``apply`` or as an
accumulator.

.. code-block:: python

    from graphblas import Matrix, indexbinary


    def discounted_sum(x, ix, jx, y, iy, jy, theta):
        return (x + y) * theta


    indexbinary.register_new("discounted_sum", discounted_sum)

    A = Matrix.from_coo([0, 1], [0, 1], [1, 2])
    B = Matrix.from_coo([0, 1], [0, 1], [3, 4])
    op = indexbinary.discounted_sum(0.5)  # bind theta to 0.5
    C = A.ewise_mult(B, op).new()  # FP64: C[0, 0] is 2.0, C[1, 1] is 3.0

Like any BinaryOp, the bound op takes its type when it runs: the one type that holds both
inputs and ``theta``. Here the INT64 matrices and the float ``theta`` run in FP64. To
choose the type, name it: ``indexbinary.discounted_sum[float](0.5)``. The full typing
rules, including for UDTs, are in :ref:`tn_indexbinary_typing`.

To use a bound IndexBinaryOp in ``mxm``, ``mxv`` or ``vxm``, make it the multiplier of a
Semiring:

.. code-block:: python

    from graphblas import monoid, semiring

    sr = semiring.register_anonymous(monoid.plus, op)
    D = A.mxm(B, sr).new()

A Monoid cannot be made from an IndexBinaryOp.

Operators on user-defined types
-------------------------------

Pass ``is_udt=True`` to use an operator on user-defined types (UDTs). It then compiles
for each type when first used, instead of for every built-in type at registration. The
function reads a record's fields by name (``x["weight"]``) and receives an array UDT's
element as a numpy array. See :doc:`udt`.

What Numba compiles
-------------------

Numba compiles the function in its nopython mode, where numbers, tuples, numpy arrays,
many numpy functions and the ``math`` module work (see Numba's
`supported Python features <https://numba.readthedocs.io/en/stable/reference/pysupported.html>`_
and `supported numpy features <https://numba.readthedocs.io/en/stable/reference/numpysupported.html>`_).
The function can use numbers and numpy arrays from outside it, but not lists or dicts.
Dividing by zero gives numpy's answer (``inf``, ``nan``, or 0 for integers) instead of
raising.

If the function compiles for no built-in type, ``register_new`` raises
``UdfParseError``. An operator registered with ``is_udt=True`` raises it when it is first
used on a type it does not compile for, with Numba's message.

When operators compile
----------------------

Compiling for every built-in type takes time at registration. To defer it, pass
``lazy=True``, which registers the operator at the first access of its name:

.. code-block:: python

    def heavy_func(x):
        return x * 2


    unary.register_new("heavy_op", heavy_func, lazy=True)
    # heavy_func compiles at the first access of unary.heavy_op

A compiled operator lasts for the rest of the process; a new process compiles it again.
:ref:`tn_udfs` has the details, including the result types of an operator on built-in
types.

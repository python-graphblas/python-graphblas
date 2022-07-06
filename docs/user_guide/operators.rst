
Operators
=========

Operators operate on the individual elements of collections. There are various operator classes in
GraphBLAS, each with defined mathematical properties.

Unary Operators
---------------

Unary operators map one input value to another in a possibly different domain.

Example usage:

.. code-block:: python

    # Compute the absolute value of each element in M
    M_abs = gb.unary.abs(M).new()

Common unary operators are:

  - **identity** -- returns the input unchanged
  - **abs** -- absolute value
  - **ainv** -- additive inverse (f(x) = -x)
  - **minv** -- multiplicative inverse (f(x) = 1/x)
  - **lnot** -- logical not (True -> False)
  - **bnot** -- binary not (110010 -> 001101)
  - **one** -- result is always 1.0 (or equivalent for the dtype)
  - **round** -- floating-point round function
  - **floor** -- floating-point floor function
  - **ceil** -- floating-point ceiling function
  - **sin** -- trigonometric sine function
  - **cos** -- trigonometric cosine function
  - **tan** -- trigonometric tangent function
  - **exp** -- exponential
  - **log** -- logarithm

Unary operators are located in the ``graphblas.unary`` namespace. Additional unary operators
registered from numpy are located in ``graphblas.unary.numpy``.

Binary Operators
----------------

Binary operators take two inputs and return a single value. The input domains and output domain
do not need to match.

Output accumulators are binary operators.

Example usage:

.. code-block:: python

    # Element-wise union, taking the max for intersecting element
    # Accumulate the result into M via addition
    M(accum=gb.binary.plus) << gb.binary.max(A | B)

Common binary operators are:

  - **pair** -- result is always 1.0 (or equivalent for the dtype)
  - **first** -- f(a, b) = a
  - **second** -- f(a, b) = b
  - **min** -- min(a, b)
  - **max** -- max(a, b)
  - **eq** -- a == b
  - **ne** -- a != b
  - **gt** -- a > b
  - **lt** -- a < b
  - **ge** -- a >= b
  - **le** -- a <= b
  - **plus** -- a + b
  - **minus** -- a - b
  - **times** -- a * b
  - **truediv** -- a / b
  - **fmod** -- a % b
  - **pow** -- a ** b
  - **atan2** -- math.atan2(a, b)
  - **lor** -- logical or (a | b)
  - **land** -- logical and (a & b)
  - **lxor** -- logical xor (a ^ b)
  - **lxnor** -- logical xnor ~(a ^ b)
  - **bor** -- binary or
  - **band** -- binary and
  - **bxor** -- binary xor
  - **bxnor** -- binary xnor

Binary operators are located in the ``graphblas.binary`` namespace. Additional binary operators
registered from numpy are located in ``graphblas.binary.numpy``.

Monoids
-------

Monoids extend the concept of a binary operator to require a single domain for all inputs and
the output. Monoids are also associative, so the order of the inputs does not matter. And finally,
monoids have a default identity such that ``A op identity == A``.

Monoids are commonly for reductions, collapsing all elements down to a single value.

Example usage:

.. code-block:: python

    # Sum up all non-empty elements in M
    total = M.reduce_scalar(gb.monoid.plus).value

Common monoids are:

  - **any** -- return either input
  - **min** -- min(a, b)
  - **max** -- max(a, b)
  - **plus** -- a + b
  - **times** -- a * b
  - **land** -- a & b
  - **lor** -- a | b
  - **lxor** -- a ^ b
  - **lxnor** -- ~(a ^ b)

Monoids are located in the ``graphblas.monoid`` namespace. Additional monoids registered from
numpy are located in ``graphblas.monoid.numpy``.

Semirings
---------

Semirings are a combination of a monoid and a binary operator. The binary operator is used for the
"multiplication" part of a dot product, while the monoid is used for the reduction.

Standard matrix multiplication uses the "plus_times" semiring.

Semirings are primarily used during matrix multiplication.

Example usage:

.. code-block:: python

    C << gb.semiring.min_plus(A @ B)

Common semirings are:

  - **plus_times** (standard matrix multiplication)
  - **min_plus** (used for shortest path computations)
  - **max_plus**
  - **min_times**
  - **max_times**
  - **min_max**
  - **max_min**
  - **min_first**
  - **min_second**
  - **max_first**
  - **max_second**
  - **plus_min**
  - **lor_land**
  - **land_lor**

Semirings are located in the ``graphblas.semiring`` namespace. Additional semirings registered
from numpy are located in ``graphblas.semiring.numpy``.

IndexUnary Operators
--------------------

A variant of unary operators are indexunary operators. They behave exactly like unary operators,
but the inputs are the value, the index position(s) of that value, and an thunk parameter.

For example, an IndexUnary operator applied to a Matrix would be given the value, row, and column
of each element (plus the thunk). The operator can use all of those pieces to determine an appropriate output.

IndexUnary operators are used primarily in ``select`` to filter based on the index positions.

Example usage:

.. code-block:: python

    # Select the upper triangle
    A_upper = gb.select.triu(A).new(name="A_upper")

.. image:: ../_static/img/Matrix-A-upper.png

Example usage with a thunk parameter:

.. code-block:: python

    # Select the upper triangle, excluding the diagonal
    A_upper = gb.select.triu(A, 1).new(name="A_strictly_upper")

.. image:: ../_static/img/Matrix-A-strictly-upper.png

Defined IndexUnary operators are:

  - **rowindex** -- return the row index
  - **colindex** -- return the column index
  - **tril** -- lower triangle matrix (True if column >= row)
  - **triu** -- upper triangle matrix (True if column <= row)
  - **diag** -- matrix diagonal (True if row == column)
  - **offdiag** -- matrix off-diagonal (True if row != column)
  - **colle** -- column index <= thunk
  - **colgt** -- column index > thunk
  - **rowle** -- row index <= thunk
  - **rowgt** -- row index > thunk
  - **valueeq** -- value == thunk
  - **valuene** -- value != thunk
  - **valuelt** -- value < thunk
  - **valuele** -- value <= thunk
  - **valuegt** -- value > thunk
  - **valuege** -- value >= thunk

IndexUnary operators are located in two places.

  - ``graphblas.indexunary``

    All IndexUnary operators are contained here.
    Calling these with a collection will perform an ``apply`` operation.

  - ``graphblas.select``

    Only the IndexUnary operators which return a boolean are contained here (i.e. all except rowindex and colindex).
    Calling these with a collection will perform a ``select`` operation.

Aggregators
-----------



Operator Type Specialization
----------------------------



The gb.op Namespace
-------------------


Infix Notation
--------------

Discuss details of +, -, *, /, %, **, >, ==, etc

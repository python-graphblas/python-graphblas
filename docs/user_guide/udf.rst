
User-defined Functions (UDFs)
=============================

python-graphblas lets you write custom operators in Python. ``numba``
JIT-compiles them to native machine code so they run inside
SuiteSparse:GraphBLAS at C speed. This guide covers every operator type
that accepts a user function: ``UnaryOp``, ``BinaryOp``, ``IndexUnaryOp``,
``SelectOp``, and ``IndexBinaryOp``.

A first example
---------------

.. code-block:: python

    from graphblas import unary, Vector

    def force_odd(x):
        if x % 2 == 0:
            return x + 1
        return x

    unary.register_new("force_odd", force_odd)

    v = Vector.from_coo([0, 1, 3, 4, 5], [1, 2, 3, 8, 14])
    w = v.apply(unary.force_odd).new()
    # w = [1, 3, _, 3, 9, 15]

``register_new`` vs ``register_anonymous``
------------------------------------------

- ``register_new(name, func)`` puts the operator on ``gb.unary.{name}`` (or
  ``gb.binary.{name}``, etc.). Use this for operators you'll reference by
  name across files.
- ``register_anonymous(func)`` returns the operator without adding it to a
  namespace. Use this for one-off operators or operators created inside
  another function.

Lambdas are auto-registered as anonymous wherever a UnaryOp is expected:

.. code-block:: python

    v.apply(lambda x: x % 5 - 2).new()

Operator signatures
-------------------

The function signature depends on the operator type:

==================  ====================================================  =====================================
Operator            Function signature                                    Returns
==================  ====================================================  =====================================
``UnaryOp``         ``f(x) -> z``                                         a value
``BinaryOp``        ``f(x, y) -> z``                                      a value
``IndexUnaryOp``    ``f(x, ix, jx, theta) -> z``                          a value (often bool, for select)
``SelectOp``        same as IndexUnaryOp                                  ``bool``
``IndexBinaryOp``   ``f(x, ix, jx, y, iy, jy, theta) -> z``               a value (requires SS >= 9.4)
==================  ====================================================  =====================================

``ix, jx`` are the row and column indices of ``x``; same for ``iy, jy`` and ``y``.
``theta`` is a scalar parameter bound when the operator is used (see
:ref:`udf_indexbinary` below).

Parameterized UDFs
------------------

To parameterize a UDF (e.g., a scale factor known only at call time), write a
factory function that returns the actual operator, and register it with
``parameterized=True``:

.. code-block:: python

    from graphblas import binary

    def make_scaled_add(scale):
        def inner(x, y):
            return scale * (x + y)
        return inner

    binary.register_new("scaled_add", make_scaled_add, parameterized=True)

    # Bind the parameter, then use the resulting op:
    op = binary.scaled_add(2.0)
    result = v.ewise_mult(w, op).new()

User-defined types (UDTs)
-------------------------

Pass ``is_udt=True`` when your UDF operates on user-defined record or array
types. See :doc:`udt` for the full story. In short:

.. code-block:: python

    from graphblas import binary, dtypes
    import numpy as np

    edge_dtype = dtypes.register_anonymous(
        np.dtype([("weight", np.float64), ("hops", np.int32)], align=True),
        "Edge",
    )

    def combine_edges(x, y):
        return (x["weight"] + y["weight"], x["hops"] + y["hops"])

    binary.register_new("combine_edges", combine_edges, is_udt=True)

The function can return a tuple matching the record's fields, an existing
record value, or a numpy array for array UDTs.

.. _udf_indexbinary:

IndexBinaryOps and the theta parameter
--------------------------------------

An IndexBinaryOp receives row and column indices for both operands plus a
scalar ``theta``. Binding ``theta`` produces a regular ``BinaryOp`` that
works in ``ewise_mult``, ``ewise_add``, and the other elementwise paths:

.. code-block:: python

    from graphblas import indexbinary, Matrix

    def discounted_distance(x, ix, jx, y, iy, jy, theta):
        return (x + y) * theta

    indexbinary.register_new("discounted_dist", discounted_distance)

    bound = indexbinary.discounted_dist[float](0.5)   # theta = 0.5
    A = Matrix.from_coo([0, 1], [0, 1], [1.0, 2.0])
    B = Matrix.from_coo([0, 1], [0, 1], [3.0, 4.0])
    C = A.ewise_mult(B, bound).new()
    # C[0,0] = (1+3)*0.5 = 2;  C[1,1] = (2+4)*0.5 = 3

To use a bound IndexBinaryOp as the multiplier in ``mxm`` / ``mxv`` /
``vxm``, wrap it in a Semiring. ``Semiring.register_anonymous`` accepts a
bound IBO directly:

.. code-block:: python

    from graphblas import monoid, semiring

    sr = semiring.register_anonymous(monoid.plus, bound)
    C = A.mxm(B, sr).new()

The resulting Semiring is monomorphic in the bound IBO's input/output types
(SuiteSparse builds exactly one ``GrB_Semiring`` for that type pair, rather
than the type-polymorphic table the standard semirings carry). To reuse the
same IBO at a different type, bind theta again under that type and build a
new Semiring. Per SuiteSparse, monoids themselves cannot be built from an
IndexBinaryOp; only the multiplier slot accepts one.

IndexBinaryOps require SuiteSparse:GraphBLAS >= 9.4.

What numba accepts
------------------

The function body is compiled by ``numba.njit`` and must be pure
numerical Python, with these constraints:

- No closures over Python objects (capture scalars, not lists or dicts).
- ``numpy`` array operations and standard library calls like ``math.sin``
  work; complex Python types (sets, dicts) do not.
- Records (UDT fields) are accessed by name: ``x["weight"]``, not ``x.weight``.
- Tuple returns work for UDTs (one tuple element per field).

When compilation fails, you get a ``UdfParseError`` with the actionable
diagnostic line pulled out of Numba's typing pass, rather than the full
multi-hundred-line traceback. The most common causes:

- Referencing a field that doesn't exist on the record UDT.
- Returning a tuple whose length doesn't match the record's field count
  (the error names the expected arity).
- Calling a function Numba doesn't support in nopython mode.

See :doc:`udt` for UDT-specific guidance.

Lazy registration
-----------------

Pass ``lazy=True`` to defer Numba compilation until the operator is first
used. This is useful for libraries that register many operators at import
time:

.. code-block:: python

    unary.register_new("heavy_op", heavy_func, lazy=True)
    # heavy_func isn't compiled yet; the operator object exists, but no
    # numba.njit has run.

The compile happens on first lookup (``unary.heavy_op[int]``).

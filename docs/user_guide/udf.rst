
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
types. See :doc:`udt` for the full story.

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

    def discounted_sum(x, ix, jx, y, iy, jy, theta):
        return (x + y) * theta

    indexbinary.register_new("discounted_sum", discounted_sum)

    bound = indexbinary.discounted_sum(0.5)   # theta = 0.5
    A = Matrix.from_coo([0, 1], [0, 1], [1, 2])
    B = Matrix.from_coo([0, 1], [0, 1], [3, 4])
    C = A.ewise_mult(B, bound).new()
    # C is FP64: C[0,0] = (1+3)*0.5 = 2.0;  C[1,1] = (2+4)*0.5 = 3.0

Like any other BinaryOp, the bound op takes its type each time it runs: the
type that holds both inputs and ``theta``, as ``apply`` types an IndexUnaryOp
from its input and thunk. One type serves x, y and ``theta``. ``theta`` has
the type of its value: a Python int is INT64 and a float FP64, and a numpy
scalar or a ``Scalar`` keeps its own type. With no ``theta``, ``theta`` is 0
(``False``), which changes no type. Above, the INT64 matrices and the float
``theta`` run in FP64. To choose the type yourself, name it:
``indexbinary.discounted_sum[float](0.5)``, or
``indexbinary.discounted_sum(0.5, dtype=float)``. The inputs and ``theta`` are
then cast to that type.

An op registered with ``is_udt=True`` is compiled for the types of x, y and
``theta`` as they are, so none of them is cast: ``op(0.5)`` on INT64 inputs
gets the float 0.5. A numpy array ``theta`` has the array UDT of its dtype
and shape. A tuple has no type of its own, so give a record or array
``theta`` as a ``Scalar`` of its UDT (``Scalar.from_value((1, 2.5), udt)``).
``op[X, Y, T](theta)`` names all three types, and ``op[X, Y](theta)`` gives
``theta`` the type of y.

This differs from an IndexUnaryOp in ``apply``, which makes a number thunk an
element of the UDT data's type. Beside UDT data, a number ``theta`` stays a
number: with ``x + theta`` on INT32[3] data, ``op(5)`` adds an INT64 to each
element, which gives an int64 array, not an INT32[3] element, so the op
raises ``UdfParseError`` when it is used. To make ``theta`` the data's type,
pass a ``Scalar`` of that type, or a numpy value of the element type
(``op(np.int32(5))``), or name the type: ``op(5, dtype=udt)`` binds
``op[udt](5)``, which makes 5 an element of ``udt``.

A ``Scalar`` ``theta`` is copied when it is bound, so changing the ``Scalar``
afterwards does not change the bound op. An ``apply`` expression, by
contrast, reads its ``Scalar`` thunk when it runs.

To use a bound IndexBinaryOp as the multiplier in ``mxm`` / ``mxv`` /
``vxm``, wrap it in a Semiring. ``Semiring.register_anonymous`` accepts a
bound IBO directly:

.. code-block:: python

    from graphblas import monoid, semiring

    sr = semiring.register_anonymous(monoid.plus, bound)
    C = A.mxm(B, sr).new()

The Semiring is typed each time it runs, as the bound op is. One built from
an op bound at a type (``discounted_sum[float](0.5)``) is monomorphic in that
op's input/output types instead (SuiteSparse builds exactly one
``GrB_Semiring`` for that type pair, rather than the type-polymorphic table
the standard semirings carry). Per SuiteSparse, monoids themselves cannot be
built from an IndexBinaryOp; only the multiplier slot accepts one.

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

Compilation caching
-------------------

A handful of built-in operators are themselves UDFs (``binary.floordiv``,
``rfloordiv``, ``absfirst``, ``abssecond``, and ``rpow``). Because these are
plain module-level functions, python-graphblas compiles them with Numba's
``cache=True``, so the Numba half of the compile is paid once per machine
instead of once per Python process. Numba writes the cache (``.nbi`` index and
``.nbc`` data files) into the ``__pycache__`` directory next to the operator
module. A warm cache shaves roughly 20% off a single op's first-touch build and
about 45% when several are used; the rest is Numba/LLVM startup and per-process
object linking, which caching cannot remove.

If that ``__pycache__`` directory is not writable (a read-only install, for
example), Numba silently falls back to its per-user cache directory
(``~/.cache/numba`` on Linux, ``~/Library/Caches/numba`` on macOS; set
``NUMBA_CACHE_DIR`` to override). No warning is emitted for the fallback.

Operators you create with ``register_new`` or ``register_anonymous`` are never
passed ``cache=True``: Numba can only cache a function with a stable on-disk
source, which a lambda or an interactively defined function does not have. A
registered operator's compile is paid once per process, and ``lazy=True`` is
the tool for deferring it.

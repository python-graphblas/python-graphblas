.. _tn_udfs:

User-defined Functions
======================

How python-graphblas compiles a Python function into an operator, and how it types the
operator. The user guide's introduction is :doc:`../user_guide/udf`. The rules for
operators on UDTs are in :ref:`tn_udt_udf_returns`.

.. _tn_udf_compile:

When an operator compiles
-------------------------

Numba compiles the function in nopython mode, with Numba's numpy error model, so a
division by zero gives numpy's answer instead of raising: ``x / 0`` is ``inf`` or ``nan``,
and an integer ``x // 0`` is 0.

- ``register_new`` and ``register_anonymous`` compile the function at once, for each
  built-in type Numba can compile it for, and the operator works on those types. If it
  compiles for none, they raise ``UdfParseError("Unable to parse function using
  Numba")``.
- With ``is_udt=True``, nothing compiles at registration. The operator compiles for each
  combination of operand types, built-in or UDT, the first time it is used with them, or
  indexed with them (``op[Edge]``). A compile error then raises ``UdfParseError`` with
  Numba's message.
- With ``lazy=True`` (``register_new`` only), the operator is registered at the first
  access of its name, such as ``unary.heavy_op``, and compiles then as it would without
  ``lazy``.
- A function passed where an operator is expected, such as ``v.apply(lambda x: x + 1)``,
  is registered anonymously with ``is_udt=True``, and compiles for the operand types.

A compiled operator stays compiled for the life of the process. Compiling for every
built-in type at registration takes time; ``is_udt=True`` and ``lazy=True`` defer it.

.. _tn_udf_return_type:

Result types on built-in types
------------------------------

An operator registered without ``is_udt`` has, for each built-in type, the result type
Numba infers, with one adjustment: a result of a wider type of the same kind as the
input takes the input's type. So ``lambda x: x + 1`` gives INT8 on INT8 and FP32 on FP32,
where Numba computes int64 and float64. On BOOL, an integer result is INT8, and on
UINT64 a float result is UINT64 when the result on INT64 is INT64 (Numba adds a uint64
and an int64 in float64). A result of another kind keeps Numba's type: ``lambda x: x / 2``
gives FP64 on INT8.

An operator registered with ``is_udt=True`` has Numba's type with no adjustment:
``lambda x: x + 1`` gives INT64 on INT8.

.. _tn_indexbinary_typing:

IndexBinaryOp types and theta
-----------------------------

Binding ``theta`` to an IndexBinaryOp (``op(theta)``) gives a BinaryOp. Like any
BinaryOp, it is typed each time it runs, from its inputs and ``theta``: x, y and
``theta`` all take the one type that holds all three, as ``apply`` types an
IndexUnaryOp from its input and thunk. ``theta`` has the type of its value: a Python
``int`` is INT64 (not weak) and a ``float`` FP64, and a numpy scalar or a ``Scalar``
keeps its own type. With no ``theta``, ``theta`` is 0 (``False``), which changes no type.
So INT64 inputs with ``op(0.5)`` run in FP64, and INT8 inputs with ``op(2)`` in INT64.

To choose the type, name it: ``op[float](0.5)`` or ``op(0.5, dtype=float)``. The inputs
and ``theta`` are then cast to that type, so ``op[int](0.5)`` gets a ``theta`` of 0.

An op registered with ``is_udt=True`` is compiled for the types of x, y and ``theta`` as
they are, and casts none of them: ``op(0.5)`` on INT64 inputs gets the float 0.5. A
numpy array ``theta`` has the array UDT of its dtype and shape. A tuple has no type of
its own, so give a record or array ``theta`` as a ``Scalar`` of its UDT
(``Scalar.from_value((1, 2.5), udt)``); a tuple raises ``TypeError``. ``op[X, Y, T](theta)``
names all three types, and ``op[X, Y](theta)`` gives ``theta`` the type of y.

With UDT data, a number ``theta`` stays a number. This differs from an IndexUnaryOp in
``apply``, which makes a number thunk an element of the data's UDT. With ``x + theta``
on ``INT32[3]`` data, ``op(5)`` adds an INT64 to each element, which gives an int64
array that no operand's type matches, so the op raises ``UdfParseError`` when it is
used. To give ``theta`` the data's type, pass a numpy value of the element type
(``op(np.int32(5))``), a ``Scalar`` of the UDT, or name the type: ``op(5, dtype=udt)``
binds ``op[udt](5)``, which makes 5 an element of ``udt``.

A ``Scalar`` ``theta`` is copied when it is bound, so changing the ``Scalar`` afterwards
does not change the bound op. An ``apply`` expression, by contrast, reads its ``Scalar``
thunk when it runs.

A Semiring built from a bound op (``semiring.register_anonymous(monoid.plus, op(0.5))``)
is typed each time it runs, as the bound op is. One built from an op bound at a type
(``op[float](0.5)``) works on that type only, and raises ``KeyError`` on others. A Monoid
cannot be built from an IndexBinaryOp.

.. _tn_udf_cache:

The Numba disk cache
--------------------

A few built-in operators are UDFs: ``binary.floordiv``, ``rfloordiv``, ``absfirst``,
``abssecond`` and ``rpow``. They are module-level functions, so python-graphblas compiles
them with Numba's ``cache=True``: Numba saves the compiled code on disk, and later
processes load it instead of compiling. Numba writes the cache (``.nbi`` and ``.nbc``
files) into the ``__pycache__`` directory of ``graphblas/core/operator``. If that is not
writable, as in a read-only install, Numba uses its per-user cache directory
(``~/.cache/numba`` on Linux, ``~/Library/Caches/numba`` on macOS, or ``NUMBA_CACHE_DIR``)
without a warning. If neither is writable, these operators compile once per process.

An operator you register is never compiled with ``cache=True``. Numba can cache only a
function with stable source in a file, which a lambda or an interactively defined
function does not have. Its compile is paid once per process; ``lazy=True`` defers it to
first use.

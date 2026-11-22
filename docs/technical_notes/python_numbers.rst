.. _tn_python_numbers:

Python Numbers in Operations
============================

A Python ``bool``, ``int``, ``float`` or ``complex`` used as an operand gets the type numpy 2
gives it (NEP 50): it is *weak*, and takes the type of the other operand where it can.
The user guide's summary is :ref:`python_numbers`.

In the examples on this page, ``v`` is an INT8 Vector, ``w`` an INT64 Vector, and ``f`` an
FP32 Vector.

.. _tn_python_numbers_builtin:

With a built-in type
--------------------

A Python number takes the other operand's type when that type is of a kind the table
lists for it. Otherwise it takes the type in the last column.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Python number
     - Takes the other operand's type if it is
     - Otherwise
   * - ``bool``
     - any type
     -
   * - ``int``
     - an integer, float or complex type
     - INT64
   * - ``float``
     - a float or complex type
     - FP64
   * - ``complex``
     - a complex type
     - FC32 if the other type is FP32, else FC64

So ``v + 1`` is INT8, ``v * 0.5`` is FP64, ``f * 0.5`` is FP32, and ``f + 1j`` is FC32.

- An ``int`` that is out of range of the integer type it takes raises ``OverflowError``
  instead of wrapping: ``v + 300`` raises. With a BOOL operand, an ``int`` is INT64 and
  must be in its range.
- The arithmetic itself wraps, as in numpy: ``v * 2`` is INT8, and an element of 100
  becomes -56.
- A ``float`` too large for FP32 becomes infinity, with a ``RuntimeWarning``, as in numpy:
  ``f + 1e300``.

A numpy scalar, a 0-d numpy array and a ``Scalar`` are *strong*: they keep their own type.
``v + np.int64(1)`` is INT64, and so is ``np.int64(1) + v``, because numpy's operators
leave an operation with a python-graphblas object to python-graphblas.

.. _tn_python_numbers_comparisons:

Comparisons and operators that change the type
----------------------------------------------

A comparison has no result type to keep, so an ``int`` out of the operand type's range
does not raise. It compares exactly, as in numpy 2: ``v < 300`` and ``w < 2**63`` are True
for every element, and ``v == 300`` is False for every element. (In FP64, a type that holds
both, the INT64 value ``2**63 - 1`` and the ``int`` ``2**63`` would be equal.) This applies
to ``eq``, ``ne``, ``lt``, ``le``, ``gt`` and ``ge``, and to the ``value*`` operators of
``select`` and ``indexunary`` (``select.valuegt``).

A ``float`` in a comparison is typed as in arithmetic: ``f == 0.1`` compares in FP32, so an
element stored from ``0.1`` equals it, and ``f == 1e300`` compares with infinity, with a
``RuntimeWarning``.

A built-in operator whose result has another type than its operands does not raise
either: ``v / 300`` is FP64 whatever the ``int``, and ``v.apply(binary.isclose,
right=300)`` is False for every element.

A user-defined operator whose result has another type gets the ``int`` exactly when the
operand's type and the ``int`` fit a common integer type (INT16, for ``v`` and ``300``). It
raises ``OverflowError`` when they fit only a common float type, which could round the
``int``: for ``w`` and ``2**63`` that is FP64.

.. _tn_python_numbers_other:

Typed operators, thunks and fill values
---------------------------------------

- A typed operator, such as ``binary.plus["INT8"]``, takes each operand in its own input
  type, and GraphBLAS casts a literal to it: ``v.apply(binary.plus["INT8"], right=300)``
  adds 300 cast to INT8, which is 44. ``v.apply(binary.plus["INT64"], right=1)`` is INT64.
- The defaults of ``ewise_union`` follow the rules above:
  ``v.ewise_union(u, binary.plus, 0.5, 0)`` is FP64, and a default of ``300`` raises.
- The ``fill_value`` of ``to_dense`` follows them too, except that a value out of range
  widens the result instead of raising: ``v.to_dense(fill_value=300)`` is an int64 array.
- A ``select`` or ``indexunary`` thunk that is not compared with the values, such as the
  row of ``select.rowle``, keeps its own type.

.. _tn_python_numbers_udt:

With a user-defined type
------------------------

With a lifted operator (see :ref:`tn_udt_lifted_ops`), a Python number is weak field by
field: in each field of a record, or each element of an array, it takes that value's
built-in type by the table above. A tuple, list or dict of Python numbers laid out like an
element is weak the same way, each number in its own field or array element. With ``a`` a
Vector of an ``INT8[3]`` UDT:

- ``a + 1``, ``a + (1, 2, 3)`` and ``a + [1, 2, 3]`` are ``INT8[3]``.
- ``a * 2.5`` and ``a + (0.5, 1, 2)`` are ``FP64[3]``.
- ``a + 300`` raises ``OverflowError``, but ``a / 300`` is ``FP64[3]``, since ``truediv``
  gives floats whatever the ``int``.

A numpy scalar is strong here too, also inside a tuple for a record:
``a + np.int64(1)`` is ``INT64[3]``. A numpy array is strong, and pairs as an array UDT of
its own dtype and shape: ``a + np.array([[1, 2, 3]])`` is ``INT64[1, 3]``. A literal for an
array field of a record must broadcast to that field's shape, or ``ValueError`` names
both: ``np.array([[1, 2, 3]])`` does not fit an ``INT8[3]`` field.

``eq`` and ``ne`` type a literal the same way, as numpy's ``==`` does, except that an
``int`` out of range of the type it takes makes the comparison False instead of raising:
``a == 300`` is False for every element. An ``FP32[3]`` element ``== 0.1`` compares in
FP32.

.. _tn_literal_element:

A literal that becomes an element
---------------------------------

Where no lifted operator types it, a literal becomes an element of the UDT: as an operand
of a user-defined operator, as an IndexUnaryOp thunk, and as a default of ``ewise_union``.
It must *fit* the UDT: its type, by the rules above, must be the UDT, or cast into it
safely, field by field (``numpy.can_cast`` with ``"safe"``). For an ``INT8[3]`` UDT:

- ``1`` and ``(1, 2, 3)`` are typed ``INT8[3]``, and fit.
- ``0.5`` is typed ``FP64[3]``, and does not fit: ``a.ewise_union(b, binary.plus, 0.5, 0)``
  raises ``ValueError``.
- ``np.int8(1)`` fits; ``np.int64(1)`` does not.

A Python ``int`` does not fit a BOOL field; write ``True`` or ``False``. A value that does
not fit but converts exactly, such as ``0`` for a BOOL field, ``2.0`` for an integer field
or ``np.int64(1)`` above, still converts, with a ``DeprecationWarning``; it will raise
``ValueError`` in version 2027.7.0 or later. Any other value that does not fit raises
``ValueError``. A ``float`` too large for a float32 field raises ``OverflowError`` instead
of becoming infinity.

A user-defined operator may not work with the literal as an element: it does not compile
for the element, or it raises on the sample values it is checked with
(:ref:`tn_udt_udf_returns`). A Python or numpy number is then typed as itself, as a
``Scalar`` of it would be. For example, take an IndexUnaryOp that returns ``x[:2] * t`` and
is registered with a 2-element ``ret_dtype``. Applied to a Vector of a 6-element array UDT,
it cannot take a 6-element ``t``, so the thunk ``3.0`` is typed FP64.

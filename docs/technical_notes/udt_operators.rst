.. _tn_udt_operators:

Operators on User-defined Types
===============================

The rules for computing on record and array UDTs, and for storing the results. The user
guide's introduction is :doc:`../user_guide/udt`. Everything on this page needs Numba.

.. _tn_udt_lifted_ops:

Lifted operators
----------------

A built-in operator *lifts* to a UDT when it applies to each field of a record, at every
level of nesting, or each element of an array, as it applies to that value's built-in type.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Kind
     - Operators that lift
   * - BinaryOp
     - ``plus``, ``minus``, ``times``, ``truediv``, ``floordiv``, ``min``, ``max``;
       ``eq`` and ``ne``, which compare whole elements and give BOOL; ``first``,
       ``second``, ``any`` and ``pair``, which do not compute on the fields
   * - UnaryOp
     - ``ainv``, ``abs``
   * - Monoid
     - ``plus``, ``times``, ``min``, ``max``, ``any``
   * - Semiring
     - one of those monoids with one of those BinaryOps, such as ``plus_times``,
       ``min_plus``, ``max_times`` or ``any_first``
   * - Aggregator
     - ``sum``, ``prod``, ``min``, ``max``, ``any_value``, ``count``; ``first`` and
       ``last`` work on any type

Any other built-in operator or aggregator raises ``KeyError`` on a UDT, among them
``pow``, ``lt``, ``fmod``, ``lor``, ``minv`` and the aggregators ``mean``, ``hypot``,
``L1norm``, ``Linfnorm``, ``sum_of_squares`` and ``sum_of_inverses``. ``min``, ``max``
and ``floordiv`` also raise ``KeyError`` on a UDT with a complex field or element, as
complex numbers have no order and no floor.

Used element-wise (``apply``, ``ewise_mult``, ``ewise_union``), a Monoid is its BinaryOp,
so it takes a literal and two different UDTs as the BinaryOp does. Indexed with two
types, a lifted Monoid is the Monoid of their result type: ``monoid.plus[I8x3, F32x3]``
is ``monoid.plus[F32x3]``, for an ``INT8[3]`` UDT ``I8x3`` and an ``FP32[3]`` UDT
``F32x3``.

.. _tn_udt_pairing:

Which operand types pair
------------------------

A lifted BinaryOp accepts these pairs of operand types, in either order:

- **Two records** with the same field names in the same order, nested the same way, and
  the same shape for each array field. The field types may differ.
- **Two array UDTs** whose shapes broadcast, as numpy's do: ``INT8[3]`` with
  ``INT64[2, 3]``, or ``(3, 1)`` with ``(1, 4)``.
- **A UDT and a built-in type**: the built-in value applies to every field or element.
- **A UDT and a numpy array**, which pairs as an array UDT of its own dtype and shape.

Any other pair, such as a record and an array, raises ``KeyError``. A UDT that is
neither a record nor an array pairs only with itself; for ``eq`` and ``ne``, text also
pairs with text of the same kind and another width, and a date or duration with one in
another unit.

.. _tn_udt_result_types:

Result types
------------

Each field of a record, or each element of an array, is computed as the built-in operator
computes it on the two built-in types there, so the result never narrows. With ``a`` a
Vector of an ``INT8[3]`` UDT, ``a / a`` is ``FP64[3]``, and ``a * f`` for an ``FP32[3]``
Vector ``f`` is ``FP32[3]``. Two array UDTs give the broadcast shape: ``INT8[3]`` with
``INT64[2, 3]`` is ``INT64[2, 3]``.

The result type is:

1. an operand's type, when it is exactly the computed layout and field types, so an
   operator on one UDT usually keeps it;
2. otherwise, the UDT registered with that numpy dtype, if there is one;
3. otherwise, a new anonymous UDT. A new record result is aligned when either operand is.

So ``{x: INT64, y: FP64}`` with ``{x: FP64, y: INT64}`` is ``{x: FP64, y: FP64}``.

A Python number, or a tuple, list or dict of them, is typed field by field by the NEP 50
rules: see :ref:`tn_python_numbers_udt`.

.. _tn_udt_comparisons:

Comparisons
-----------

``eq`` and ``ne`` compare whole elements and give BOOL. Each pair of fields compares in
the type the two promote to, and array elements broadcast, as numpy's ``==`` does.
``==`` on a ``Scalar`` of a UDT gives a Python ``bool``.

``isequal`` compares as ``numpy.array_equal`` does, except that leading axes of length 1
are ignored: the elements must have the same shape apart from those, and equal values.
So a ``Scalar`` ``s`` of an ``INT8[3]`` UDT holding ``[1, 1, 1]`` has ``s == 1`` true but
``s.isequal(1)`` false, while an ``INT8[1]`` Scalar holding ``[1]`` is ``isequal`` to ``1``.

A literal with an ``int`` out of a field's range equals no element, instead of raising
(:ref:`tn_python_numbers_udt`).

.. _tn_udt_numerics:

How each value is computed
--------------------------

Each field or element is computed as the built-in operator computes it on that type,
apart from the edge cases below, and the JIT kernel and the Numba function give the same
bits:

- ``min`` and ``max`` ignore NaN, as C's ``fmin`` and ``fmax`` do. A tie between ``-0.0``
  and ``0.0`` is ``-0.0`` for ``min`` and ``0.0`` for ``max``, on every platform.
- ``floordiv`` is numpy's, with 0 for an integer divided by zero.
- ``truediv`` is numpy's, with numpy's infinities and NaN for a zero divisor.
- A complex quotient uses Smith's method, as numpy's does, but divides where numpy
  multiplies by a reciprocal, so it can differ from numpy's in the last bit.

In these edge cases, the UDT gives numpy's answer, and a built-in Vector of the same type
does not:

- ``INT64_MIN // -1`` is ``INT64_MIN``, where an INT64 Vector gives 0;
- a complex product or quotient with an infinite or NaN part;
- a complex quotient by ``-0.0``;
- the signed-zero tie of ``min`` and ``max``, which SuiteSparse's built-in ``min`` and
  ``max`` leave to the C library.

.. _tn_udt_storing:

Storing in another type
-----------------------

GraphBLAS casts a built-in type to another when it stores a result, but it cannot cast a
UDT. python-graphblas casts between UDTs instead, and never makes a temporary copy of the
data to do it. (Data GraphBLAS works on is often too large to hold twice.)

A value can be stored in an object of another UDT when the two layouts correspond:
records that pair as operands do (:ref:`tn_udt_pairing`), or arrays of the same shape
apart from leading axes of length 1. Each field or element is cast as GraphBLAS casts that
pair of built-in types:

- an integer out of range of the integer type it is stored as wraps;
- a float stored as an integer truncates toward zero, saturates at the integer's bounds,
  and is 0 if NaN;
- a complex number stored as a real loses its imaginary part;
- anything nonzero stored as BOOL is ``True``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Store
     - What happens
   * - An existing object: ``w << v``, ``v.dup(dtype=...)``, with or without a mask, an
       accumulator or ``replace``
     - Cast in one pass.
   * - An element-wise lifted operation: ``apply``, ``ewise_mult``, ``ewise_union``, and
       the infix operators that use them
     - The operator writes the object's type as it computes. ``a += 0.5`` on an
       ``INT8[3]`` Vector computes in FP64 and truncates, as it does on an INT8 Vector.
   * - Anything else: ``mxm``, ``mxv``, a reduction into a Vector, ``ewise_add``, a
       user-defined operator, or a whole Vector or Matrix assigned into part of an object
     - ``DomainMismatch``, since this would need a converted copy of the whole result.
       Compute the result, then store it (``w << expr.new()``); to assign, cast the
       value first with ``.dup(dtype=...)``.
   * - A ``Scalar`` or one element
     - Computed in its own type, then cast. With an accumulator, the value is cast before
       it accumulates, as GraphBLAS casts a built-in value: an ``INT8[3]`` Scalar's
       ``s += 0.5`` adds 0, as an INT8 Scalar's does.
   * - Between a UDT and a built-in type, or layouts that do not correspond
     - ``DomainMismatch``.

.. _tn_udt_ewise_add:

``ewise_add`` with two types
----------------------------

``ewise_add`` on Vectors and Matrices raises ``DomainMismatch`` when an operand's type
differs from the result's, as for two different UDTs, or ``truediv`` on an integer UDT.
GraphBLAS would copy an element with no partner into the result without casting it.
``ewise_union`` and ``ewise_mult`` work.

The infix ``+`` on two such operands uses ``ewise_union`` instead, with each operand's
zero for a missing element, as ``-`` uses 0. That zero is ``-0.0`` in a float field, so an
element of one operand alone keeps its value, as ``ewise_add`` would copy it. The one
exception is a ``-0.0`` in a field that is an integer in the other operand: it becomes
``0.0``.

A Scalar holds one element, so ``ewise_add`` on two Scalars casts its operands to the
result type instead of raising.

.. _tn_udt_udf_returns:

Operators you write
-------------------

An operator registered with ``is_udt=True`` is compiled for each combination of operand
types the first time it is used with them, or when it is indexed with them
(``op[Edge]``). Its function receives a record operand as a Numba record, whose fields it
reads as ``x["name"]`` or ``x.name``, and an array operand as a numpy array view of the
element, in the UDT's shape: an ``FP64[2, 4]`` operand is indexed ``x[i, j]``.

The result type is ``ret_dtype`` when the operator was registered with one. Otherwise it
depends on what the function returns:

- **A number**: the built-in type Numba infers for it.
- **A record**, such as one of the operands: that record's type.
- **A tuple**: the type of a record operand with one field for each value, preferring one
  whose field types match the values' types. Each value is cast to its field's type. For a
  nested record, the tuple is flat over the innermost fields: for
  ``[("id", int32), ("pt", [("x", float64), ("y", float64)])]`` it is ``(id, x, y)``, not
  ``(id, (x, y))``.
- **An array operand**: that operand's type.
- **An array the function builds**: the type of the array operand with the same element
  type and number of dimensions. On an ``INT8[3]`` UDT, ``x * 2`` is an int64 array, which
  matches no operand and raises; return ``(x * 2).astype(np.int8)`` instead. The array may
  have the element's shape or a shape that broadcasts to it: a ``(1,)`` array fills every
  value of a ``(6,)`` element.

An array field of a record takes an array the same way. It also takes a tuple or list,
but only when the field is one-dimensional and the tuple has exactly its length; a tuple
does not broadcast.

With ``ret_dtype``, the function must return something that can be written as that type:

- **A record type**: a tuple with one value per field, flat over nested records as above,
  or a record of that type.
- **An array UDT**: an array that fills the element, or a number, which fills every value.
- **A date, duration or text UDT**: a value of that type.
- **A built-in type**: a number. A complex number cannot be written as a real type, but it
  can be written as BOOL.

These raise ``UdfParseError`` when the operator is compiled for the types, not while it
runs:

- a function Numba cannot compile, such as one reading a field the record does not have;
- a tuple whose length matches no record operand;
- an array whose element type or number of dimensions matches no array operand;
- a field that can be ``None`` on some path (``v if cond else None``), since a record
  cannot hold ``None``;
- an array, tuple or list for a field that holds one number;
- an array that cannot fill the element, such as ``x[:2]`` from a 3-element UDT.

Numba knows the length of a tuple but not the shape of an array the function builds, so
for an array result the check calls the function on sample values. A function that raises
on those values is not checked, since it may raise only for some values.

An exception the function raises while the operator runs is printed, and the operation
goes on, since Numba cannot pass the exception to GraphBLAS. The element it raised for is
0 (``False`` for a select, which then drops the entry) when the result type is built-in.
An element of a UDT result is kept but not written, so its value is undefined.

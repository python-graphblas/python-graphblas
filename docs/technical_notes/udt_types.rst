.. _tn_udt_types:

User-defined Types: Registration and Values
===========================================

The user guide's introduction is :doc:`../user_guide/udt`.

.. _tn_udt_registration:

Ways to declare a UDT
---------------------

``dtypes.register_anonymous(dtype, name=None)`` and ``dtypes.register_new(name, dtype)``
accept the same forms of ``dtype``:

- A ``numpy.dtype``: a record (structured) dtype, an array dtype such as
  ``np.dtype((np.float64, (2, 4)))``, or a single value numpy stores but GraphBLAS does
  not define, such as ``np.dtype("S5")``. A record dtype keeps numpy's layout, which is
  packed unless it was made with ``align=True``.
- A dict, such as ``{"dist": float, "parent": int}``. Each key is a field name, and each
  value is anything ``dtypes.lookup_dtype`` accepts: a Python or numpy type, a type name
  (``"INT8"``, ``"f4"``), an array type written as ``"FP64[3]"``, or a registered UDT. The
  fields are aligned as a C compiler aligns a struct.
- A string naming an array type, such as ``"INT64[3, 4]"``.
- A dataclass, or an instance of one. Each field becomes a record field, aligned, and the
  class name is the default UDT name. An annotation may be a type or a string (``"int"``,
  ``"FP64[3]"``), as under ``from __future__ import annotations``. A compound annotation
  such as ``Optional[int]`` raises ``ValueError``.

``register_new`` also puts the UDT on ``gb.dtypes``. Its name must be a Python identifier
not already used there. ``register_new(SomeDataclass)`` uses the class name.

An array of arrays has the layout of one array, so it registers as that array UDT:
``np.dtype((np.dtype((np.float64, (3,))), (2,)))`` is ``FP64[2, 3]``. Array fields of
records are flattened the same way.

.. _tn_udt_same_dtype:

One UDT per numpy dtype
-----------------------

A UDT is identified by its numpy dtype. Registering a dtype that is already registered
returns the existing ``DataType``. A ``name`` given then replaces its Python name, but not
the C name SuiteSparse knows it by, which is fixed at the first registration (see
:ref:`tn_jit_c_names`). A built-in dtype, such as ``np.dtype(np.int64)``, returns the
built-in type, and giving it another name raises ``ValueError``.

.. _tn_udt_field_types:

What a UDT can hold
-------------------

- **Numbers.** Every built-in type (``bool``, the signed and unsigned integers, the floats
  and the complex types) as a field, and every one but ``bool`` as the element of an
  array.
- **Fixed-shape arrays and nested records**, as fields. Lifted operators apply to each of
  their values.
- **Text, dates and durations** (``str``, ``bytes``, ``datetime64``, ``timedelta64``), as
  fields or as a whole UDT that is neither a record nor an array. Such a UDT can be stored
  and extracted. With Numba, it can be compared (``==``, ``!=`` and ``isequal``), and
  ``first``, ``second``, ``any`` and ``pair`` work on it. Arithmetic on a whole such UDT
  raises ``KeyError``. In a record, a lifted operator computes such a field with Numba's
  operator for its type where Numba has one (``plus`` joins text, cut to the field's width;
  ``min`` compares), and raises ``UdfParseError`` where it has none.
- **float16** can be registered and stored. Numba cannot compute on it, so an operator on
  a UDT with a float16 field or element raises ``KeyError``, and such a UDT does not cast.

Not supported:

- Python objects (``np.dtype(object)``, or a field of it): ``ValueError``.
- A dtype of size 0, such as ``np.dtype("S0")``: ``ValueError``.
- With Numba installed, an *array* of ``bool``, ``datetime64``, ``timedelta64``, ``str``
  or ``bytes``, as a UDT or a field, raises ``ValueError``, because Numba cannot type it.
  Use an array of ``uint8`` for ``bool`` and ``bytes`` (one per byte), of ``uint32`` for
  ``str`` (one per character), and of ``int64`` for dates and durations.

.. _tn_udt_values:

Values
------

An element of a record UDT is given as a tuple of field values in field order, or a dict
by field name, and read as a numpy record (``numpy.void``). An element of an array UDT is
given as a nested sequence or a numpy array of the element's shape, and read as a numpy
array of that shape. ``from_coo`` takes a list of elements, or for an array UDT a numpy
array with one more leading axis. ``to_coo`` returns a numpy structured array for a
record UDT, and an array with one more leading axis for an array UDT.

.. _tn_udt_without_numba:

Without Numba
-------------

Without Numba, a UDT can be registered, and its values stored, extracted, pickled and
serialized. Of the operations on it, only ``binary.pair`` and a ``Scalar``'s ``==``,
``!=`` and ``isequal`` work. Other built-in operators raise ``KeyError``, user-defined
operators cannot be registered, and storing a value in an object of another type raises
``DomainMismatch``.

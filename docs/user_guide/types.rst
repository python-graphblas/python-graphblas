
Data Types
==========

Each collections must have a single data type, indicated by the ``.dtype`` attribute.

Defined data types are:

  - BOOL
  - UINT8
  - UINT16
  - UINT32
  - UINT64
  - INT8
  - INT16
  - INT32
  - INT64
  - FP32 (float32)
  - FP64 (float64)
  - FC32 (complex float32) (*not supported on Windows*)
  - FC64 (complex float64)  (*not supported on Windows*)

The ``graphblas.dtypes`` namespace contains objects for each of these data types.

Each of these defined types has a string representation (accessed by ``.name``),
a corresponding numpy dtype (accessed by ``.np_type``), and a corresponding
numba dtype (accessed by ``.numba_type``).

When a data type is needed in an API call, a string or numpy or numba dtype may be used
instead of the actual data type object. Additionally, the Python builtin
``bool``, ``int``, and ``float`` may be used. ``int`` indicates INT64 and ``float`` indicates FP64.

.. _python_numbers:

Python numbers in operations
----------------------------

A Python ``bool``, ``int``, ``float`` or ``complex`` used as an operand, such as the ``1``
in ``v + 1``, is typed as in numpy 2 (NEP 50). It takes the data type of the other
operand when that type can hold its kind of number: a ``bool`` takes any type, an ``int``
an integer, float or complex type, and a ``float`` a float or complex type. Otherwise it is
INT64, FP64 or FC64. With an INT8 Vector ``v``:

.. code-block:: python

    (v + 1).new()  # INT8
    (v * 0.5).new()  # FP64
    (v * 2).new()  # INT8, so an element of 100 wraps to -56, as in numpy
    # (v + 300).new()  raises OverflowError: 300 is out of range for INT8

A numpy scalar or a ``Scalar`` keeps its own type: ``v + np.int64(1)`` is INT64. Use
one to compute in a wider type. python-graphblas applies these rules itself, so they do
not depend on the installed version of numpy. The full rules, for comparisons, typed
operators, thunks and user-defined types too, are in :ref:`tn_python_numbers`.

User-defined Types
------------------

python-graphblas supports user-defined types.

First create a custom numpy dtype. Then register it in the ``graphblas.dtypes`` namespace.

.. code-block:: python

    NP_Point = np.dtype([("x", np.int64), ("y", np.int64)], align=True)
    Point = gb.dtypes.register_new("Point", NP_Point)
    # Create a 10-element sparse vector holding Points
    v = gb.Vector(Point, size=10)

See :doc:`udt` for more.

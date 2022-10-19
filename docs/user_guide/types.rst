
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

User-defined Types
------------------

python-graphblas supports user-defined types.

First create a custom numpy dtype. Then register it in the ``graphblas.dtypes`` namespace.

.. code-block:: python

    NP_Point = np.dtype([("x", np.int64), ("y", np.int64)], align=True)
    Point = gb.dtypes.register_new("Point", NP_Point)
    # Create a 10-element sparse vector holding Points
    v = gb.Vector(Point, size=10)

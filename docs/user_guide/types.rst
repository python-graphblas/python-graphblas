
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

Python numbers in operations
----------------------------

A Python ``bool``, ``int``, ``float`` or ``complex`` used as an operand, such as
the ``1`` in ``v + 1`` or ``v.apply(binary.times, right=0.5)``, is *weak*, as in
numpy 2 (NEP 50). It takes the other operand's data type when that type's kind
can hold it, and its own default type (BOOL, INT64, FP64 or FC64) when not. So
with an INT8 vector ``v``, ``v + 1`` is INT8 and ``v * 0.5`` is FP64, while
``v + 300`` raises ``OverflowError`` rather than wrapping, and ``v * 2`` wraps
in INT8 as it does in numpy. An FP32 vector times ``0.5`` is FP32, plus ``1j``
is FC32, and plus ``1e300`` is infinity, with a ``RuntimeWarning`` as in numpy. numpy scalars, 0-d arrays and ``Scalar`` objects keep their own
type: ``v + np.int64(1)`` is INT64, and so is ``np.int64(1) + v``, since numpy's
operators leave an operation with a python-graphblas object to python-graphblas.
python-graphblas applies this rule itself,
so it does not depend on the installed version of numpy. User-defined types
follow the same rule, field by field (see :doc:`udt`).

A comparison has no result type to keep, so an integer outside the operand's
range compares exactly, as in numpy: ``v < 300`` is True for every element, as
is ``w < 2**63`` for an INT64 vector ``w``.
Nor does an operator whose result has another type raise: ``v / 300`` is FP64
whatever the integer, and ``v.apply(binary.isclose, right=300)`` is False for
every element. A user-defined operator whose result has another type gets the
integer exactly where the operand's type and the integer's share an integer
type (INT16 for ``300`` here), and raises ``OverflowError`` where they would
share only a floating-point type, which can round the integer: beside an INT64
vector, ``2**63`` would meet the elements in FP64, where ``2**63 - 1`` and
``2**63`` are equal. An
FP32 vector ``== 0.1`` compares in FP32, and ``== 1e300`` compares with
infinity, with a ``RuntimeWarning`` as in numpy. The defaults of ``ewise_union``
follow the rule, and so does the ``fill_value`` of ``to_dense``, except that a
value that does not fit widens the result instead of raising. A literal given
to a typed operator, such as ``binary.plus["INT64"]``, takes that operator's
type, and a ``select`` or ``IndexUnaryOp`` thunk that is not compared with the
values, such as a row index, keeps its own type.

User-defined Types
------------------

python-graphblas supports user-defined types.

First create a custom numpy dtype. Then register it in the ``graphblas.dtypes`` namespace.

.. code-block:: python

    NP_Point = np.dtype([("x", np.int64), ("y", np.int64)], align=True)
    Point = gb.dtypes.register_new("Point", NP_Point)
    # Create a 10-element sparse vector holding Points
    v = gb.Vector(Point, size=10)

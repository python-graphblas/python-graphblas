
User-defined Types (UDTs)
=========================

A user-defined type (UDT) lets each element of a Vector, Matrix or Scalar hold more than
one number. A *record* UDT has named fields, like a C struct, and an *array* UDT holds a
fixed-shape array. Use one when each node or edge carries values that belong together: a
distance and the node it was reached from, a weight and a hop count, or a feature vector.

The built-in arithmetic operators work on UDTs field by field, and so do the monoids,
semirings and aggregators built from them. Your own operators on UDTs are Python
functions, compiled by Numba (see :doc:`udf`). Computing on UDTs needs Numba, which
python-graphblas installs on the Python versions Numba supports.

A first example
---------------

.. code-block:: python

    from dataclasses import dataclass

    from graphblas import Matrix, agg, dtypes


    @dataclass
    class Trip:
        miles: float
        minutes: float
        tolls: int


    trip = dtypes.register_new("Trip", Trip)

    # Three trips between three places
    A = Matrix.from_coo(
        [0, 0, 1], [1, 2, 2], [(3.5, 9, 0), (12.0, 15, 2), (8.0, 11, 1)], dtype=trip
    )
    totals = A.reduce_rowwise(agg.sum).new()
    cost = A.apply(lambda t: t["minutes"] + 2 * t["tolls"]).new()

.. csv-table:: totals
    :class: inline matrix
    :header: 0,1

    "(15.5, 24.0, 2)","(8.0, 11.0, 1)"

.. csv-table:: cost
    :class: inline matrix
    :header: ,0,1,2
    :stub-columns: 1

    **0**,,9.0,19.0
    **1**,,,13.0

``agg.sum`` adds each field, so ``totals`` holds the miles, minutes and tolls of each
row's trips. The lambda reads fields by name and returns a number, so ``cost`` is an
ordinary FP64 Matrix.

Registering a UDT
-----------------

Declare a record with a dict, a dataclass (as above) or a numpy dtype, and an array UDT
with a numpy dtype or a string such as ``"FP64[3]"``:

.. code-block:: python

    import numpy as np

    from graphblas import dtypes

    # A record, from a dict
    Count = dtypes.register_new("Count", {"total": float, "n": int})

    # A record with an array field
    Body = dtypes.register_new("Body", {"mass": float, "pos": "FP64[3]"})

    # A record, from a numpy dtype; pass align=True
    Edge = dtypes.register_new(
        "Edge", np.dtype([("weight", np.float64), ("hops", np.int32)], align=True)
    )

    # An array UDT, such as a color
    Color = dtypes.register_new("Color", "UINT8[3]")  # or np.dtype((np.uint8, 3))

``dtypes.register_new`` also puts the UDT on ``gb.dtypes``, as ``gb.dtypes.Count``.
``dtypes.register_anonymous(dtype, name)`` only returns it; the name is optional, and
shows in reprs and error messages.

A field can be a number type, a fixed-shape array of one (``"FP64[3]"`` in a dict), or
another record. A dict or a dataclass lays out its fields as a C compiler does. A numpy
dtype packs them unless you pass ``align=True``. A packed record works, but if a C
compiler would pad it, its operators cannot use the JIT (see `Performance and the JIT`_).
Registering the same numpy dtype again returns the same UDT. Text, dates and float16
work only in part, and Python objects not at all: see :ref:`tn_udt_field_types`.

Creating and reading values
---------------------------

Give a record element as a tuple of its fields in order, or as a dict by field name. Give
an array element as a sequence or a numpy array. ``from_coo`` takes a list of elements,
or a numpy array with one row per element. Reading an element gives a numpy record
(``numpy.void``) or a numpy array:

.. code-block:: python

    from graphblas import Vector

    v = Vector(Edge, size=3)
    v[0] = (1.5, 4)  # fields in order
    v[1] = {"weight": 2.5, "hops": 7}  # or by name
    x = v.get(0)  # a numpy record: (1.5, 4)
    w = x["weight"]  # 1.5
    indices, values = v.to_coo()  # values is a numpy structured array

    P = Vector.from_coo([0, 1], [[255, 0, 0], [0, 128, 255]], dtype=Color)
    p = P.get(1)  # a numpy array: [0, 128, 255]

Built-in operators on UDTs
--------------------------

These built-in operators *lift* to any record or array UDT: they apply to each field, or
each element of an array, as they would to that field's built-in type.

- BinaryOps ``plus``, ``minus``, ``times``, ``truediv``, ``floordiv``, ``min`` and
  ``max``; ``eq`` and ``ne``, which compare whole elements and give BOOL; and ``first``,
  ``second``, ``any`` and ``pair``.
- UnaryOps ``ainv`` and ``abs``.
- Monoids ``plus``, ``times``, ``min``, ``max`` and ``any``, and the semirings made from
  them and the BinaryOps above, such as ``plus_times`` and ``min_plus``.
- Aggregators ``sum``, ``prod``, ``min``, ``max``, ``any_value`` and ``count``, and
  ``first`` and ``last``.

Other built-in operators and aggregators, such as ``pow``, ``lt`` or ``mean``, raise
``KeyError`` on a UDT; write your own operator for them. ``min``, ``max`` and
``floordiv`` also raise on a UDT with complex values. The complete list is in
:ref:`tn_udt_lifted_ops`.

An operand can be of a built-in type. Here an FP64 Matrix multiplies a Vector of
4-element arrays, so each weight multiplies every element of an array:

.. code-block:: python

    import numpy as np

    from graphblas import Matrix, Vector, binary, dtypes, semiring

    Feat = dtypes.register_new("Feat", np.dtype((np.float64, 4)))

    # A weighted graph, and one-hot features on its 3 nodes
    A = Matrix.from_coo([0, 0, 1, 2], [1, 2, 2, 0], [0.5, -0.5, 1.0, 1.0])
    X = Vector.from_coo([0, 1, 2], np.eye(3, 4), dtype=Feat)

    H = A.mxv(X, semiring.plus_times).new()  # H[i] is the sum of A[i, j] * X[j]
    H = binary.max(H, 0).new()  # ReLU: the larger of each element and 0

.. csv-table:: H
    :class: matrix
    :header: 0,1,2

    "[0.0, 0.5, 0.0, 0.0]","[0.0, 0.0, 1.0, 0.0]","[1.0, 0.0, 0.0, 0.0]"

How results are typed
~~~~~~~~~~~~~~~~~~~~~

Each field of a result has the type the built-in operator gives for that field's types,
so a result never narrows. Two records pair as operands when they have the same field
names in the same order; two array UDTs pair when their shapes broadcast, as in numpy. A
Python number is typed in each field by the same rule as with a built-in type
(:ref:`python_numbers`): it takes the field's type when that type can hold its kind of
number.

.. code-block:: python

    from graphblas import Vector, dtypes

    I8x2 = dtypes.register_new("I8x2", "INT8[2]")
    F32x2 = dtypes.register_new("F32x2", "FP32[2]")
    v = Vector.from_coo([0, 1], [[1, 2], [3, 4]], dtype=I8x2)
    f = Vector.from_coo([0, 1], [[0.5, 0.5], [1.5, 1.5]], dtype=F32x2)

    (v + 1).new()  # I8x2: 1 takes each element's type, INT8
    (v * 2.5).new()  # FP64[2], as INT8 times 2.5 is FP64
    (v / v).new()  # FP64[2], as INT8 divided by INT8 is FP64
    (v * f).new()  # F32x2, as INT8 times FP32 is FP32
    # (v + 300).new()  raises OverflowError: 300 is out of range for INT8

When the result's fields have the same types as an operand's, the result has that
operand's type. Otherwise it is a UDT for the promoted fields: one registered with that
numpy dtype, or a new anonymous one, such as ``FP64[2]`` above. The full rules are in
:ref:`tn_udt_operators`.

Storing results in another type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GraphBLAS cannot cast a UDT, so python-graphblas does: storing in an object of another UDT
with the same field names, or array shape, casts each field as GraphBLAS casts built-in
types. It never makes a temporary copy of the data to do this, so only some stores can
cast. Storing a whole Vector or Matrix casts it (``w << u``, ``u.dup(dtype=...)``), as
does storing a Scalar or one element. So do ``apply``, ``ewise_mult`` and ``ewise_union``
with a built-in operator, which write the object's type as they compute. Anything else,
such as ``mxm``, a reduction into a Vector, your own operator, or assigning a Vector or
Matrix into part of another, raises ``DomainMismatch``. For those, compute the result with
``.new()`` first, and store that. Continuing the example above:

.. code-block:: python

    from graphblas import Matrix, semiring

    A = Matrix.from_coo([0, 1], [1, 0], [2.0, 2.0])

    v << v * 2.5  # computed as FP64[2], stored as INT8: [[2, 5], [7, 10]]
    g = v.dup(dtype=F32x2)  # a copy cast to FP32[2]

    # v << A.mxv(f, semiring.plus_times)  raises DomainMismatch
    v << A.mxv(f, semiring.plus_times).new()  # works: compute, then store

``ewise_add`` raises ``DomainMismatch`` when an operand's type differs from the
result's, as with ``v`` and ``f``: it would copy an element with no partner into the
result without casting it. Use ``ewise_union``, or the infix ``+``, which on such operands
uses ``ewise_union`` with zero defaults. The rules for storing are in
:ref:`tn_udt_storing`.

Writing operators for UDTs
--------------------------

Register a function with ``is_udt=True`` to use it on UDTs. The function receives a record
operand as a record, whose fields it reads by name (``x["dist"]``), and an array operand
as a numpy array in the element's shape. It returns a number for a result of a built-in
type, or one of these for a UDT result:

- a tuple with a value for each field of a record operand, in field order;
- a record, such as one of its operands;
- an array with the element type of an array operand, in the element's shape. On an
  ``INT8[3]`` UDT, ``x * 2`` is an int64 array, which raises: return
  ``(x * 2).astype(np.int8)``.

.. code-block:: python

    import numpy as np

    from graphblas import Vector, dtypes, unary

    Point = dtypes.register_new("Point", np.dtype((np.float64, 3)))


    def unit(x):  # x is a numpy array of shape (3,)
        return x / np.sqrt(np.sum(x * x))


    unary.register_new("unit", unit, is_udt=True)

    P = Vector.from_coo([0, 2], [[3.0, 0.0, 4.0], [0.0, 2.0, 0.0]], dtype=Point, size=4)
    U = P.apply(unary.unit).new()  # [0.6, 0.0, 0.8] and [0.0, 1.0, 0.0]

A function passed directly in place of an operator, such as the lambda in the first
example, is compiled the same way.

A user-defined BinaryOp works in a Semiring with a lifted Monoid. Here each element is a
dense 2x2 block, so ``mxm`` multiplies block-sparse matrices (``@`` in a Numba function
needs scipy):

.. code-block:: python

    import numpy as np

    from graphblas import Matrix, binary, dtypes, monoid, semiring

    # Each element is a dense 2x2 block
    Block = dtypes.register_new("Block", np.dtype((np.float64, (2, 2))))

    # x and y arrive as 2x2 numpy arrays
    binary.register_new("matmul", lambda x, y: x @ y, is_udt=True)
    semiring.register_new("plus_matmul", monoid.plus, binary.matmul)

    blocks = [[[1, 2], [3, 4]], [[0, 1], [1, 0]], [[2, 0], [0, 2]]]
    A = Matrix.from_coo([0, 0, 1], [0, 1, 1], blocks, dtype=Block)
    C = A.mxm(A, semiring.plus_matmul).new()
    C.get(0, 1)  # array([[2., 3.], [6., 3.]])

The function is compiled for a UDT the first time the operator is used on it, or when you
index the operator with it (``unary.unit[Point]``). If it does not compile, or returns
something that cannot fill the result's element, that raises ``UdfParseError``. The full
rules are in :ref:`tn_udt_udf_returns`.

Naming the output type
~~~~~~~~~~~~~~~~~~~~~~

The result type is worked out from what the function returns, matched against the operand
types, so it can only be an operand's type or a built-in type. A function that returns a
smaller array, such as ``x[:3]`` on a 9-element UDT, fills no operand's element and raises
``UdfParseError``. Pass ``ret_dtype`` to name the result type. It takes anything
``dtypes.lookup_dtype`` accepts and requires ``is_udt=True``:

.. code-block:: python

    import numpy as np

    from graphblas import dtypes, unary

    Nine = dtypes.register_anonymous(np.dtype((np.float64, (9,))), "Nine")
    Three = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "Three")

    head = unary.register_anonymous(lambda x: x[:3], "head", is_udt=True, ret_dtype=Three)
    head[Nine].return_type  # Three

``register_anonymous`` and ``register_new`` of ``unary``, ``binary``, ``indexunary`` and
``indexbinary`` take ``ret_dtype``; ``select`` operators always return BOOL and do not.
It is the result type for every operand type, so an operator whose result type should
depend on its operands needs one registration for each result type. The function must
return something that can be written as ``ret_dtype`` (:ref:`tn_udt_udf_returns`), or
compiling the operator for a type, with ``op[dtype]`` or on first use, raises
``UdfParseError``.

Example: shortest paths with parents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A record, a monoid of your own and an IndexBinaryOp (:ref:`udf_indexbinary`) together
find the shortest paths from a node, with the node before each one on its path. Each
element of ``d`` holds a distance and that parent. ``relax`` is an IndexBinaryOp, so it
receives the index of the node the path comes from. IndexBinaryOps need
SuiteSparse:GraphBLAS 9.4 or later.

.. code-block:: python

    import numpy as np

    from graphblas import Matrix, Vector, binary, dtypes, indexbinary, monoid, semiring

    # Each element holds a distance and the node before it on the path
    Hop = dtypes.register_new("Hop", {"dist": float, "parent": int})


    def closer(x, y):
        return x if x["dist"] <= y["dist"] else y


    def relax(x, ix, jx, w, iy, jy, theta):  # x is d[u], w is A[u, v]
        return (x["dist"] + w, iy)  # iy is u, the parent of v on this path


    binary.register_new("closer", closer, is_udt=True)
    monoid.register_new("closer", binary.closer, (np.inf, -1))
    indexbinary.register_new("relax", relax, is_udt=True)
    semiring.register_new("closer_relax", monoid.closer, indexbinary.relax())

    # Edges 0->1 (4), 0->2 (1), 1->2 (2), 1->3 (10) and 2->3 (5)
    A = Matrix.from_coo(
        [0, 0, 1, 1, 2], [1, 2, 2, 3, 3], [4.0, 1, 2, 10, 5], nrows=4, ncols=4
    )

    d = Vector.from_coo([0], [(0.0, 0)], dtype=Hop, size=4)  # start at node 0
    for _ in range(A.nrows - 1):
        d(binary.closer) << d.vxm(A, semiring.closer_relax)

.. csv-table:: d
    :class: matrix
    :header: 0,1,2,3

    "(0.0, 0)","(4.0, 0)","(1.0, 0)","(6.0, 2)"

``monoid.closer`` keeps the closer of two paths, with ``(inf, -1)`` as its identity.
``indexbinary.relax()`` binds its ``theta``, unused here, and gives a BinaryOp. The
accumulator ``binary.closer`` keeps a node's old path when the new one is not closer.
Node 3 is reached through node 2, at distance 6.

Performance and the JIT
-----------------------

For a lifted operator, python-graphblas gives SuiteSparse:GraphBLAS a C definition of the
UDT and of the operator. SuiteSparse's JIT then compiles one kernel for the whole
operation, which is often several times faster than calling the operator's Numba
function for each element. The first use of each kernel compiles it, typically in under a
second, and kernels are cached on disk for later runs.

By default, the JIT compiles nothing (``jit_c_control`` is ``"run"``). The first lifted
operator that needs a kernel turns it on, for the rest of the process and for every
operation. :ref:`suitesparse_jit` explains the setting, how to keep the JIT off, and why
processes running at the same time need separate caches.

Operators you write run as Numba functions, called for each element; the JIT does not
compile them.

A lifted operator also runs as a Numba function when its UDT has no C definition. Then a
``NoJITWarning`` gives the possible causes, the first time. The common ones:

- a packed record (from a numpy dtype without ``align=True``) that a C compiler would
  pad;
- a field named like a C keyword or macro, such as ``class`` or ``INT_MAX``;
- a record with an array field.

The full list is in :ref:`tn_jit_skipped`. To see what SuiteSparse compiles:

.. code-block:: python

    from graphblas import binary

    # Edge is from "Registering a UDT" above
    Edge.jit_c_name  # 'gbudt_Edge'
    Edge.jit_c_definition  # 'typedef struct { double weight ; int32_t hops ; } gbudt_Edge ;'
    binary.plus[Edge].jit_c_source  # 'void plus_gbudt_Edge (gbudt_Edge *z, ...'

Saving and loading
------------------

A Vector or Matrix of a UDT pickles. The pickle carries the UDT, so it loads in another
process too, and a UDT registered with ``register_new`` is back on ``gb.dtypes``:

.. code-block:: python

    import pickle

    data = pickle.dumps(v)
    w = pickle.loads(data)  # in this process or another

To load the output of ``v.ss.serialize()`` with ``Vector.ss.deserialize`` in another
process, register the UDT there first, under the same name.

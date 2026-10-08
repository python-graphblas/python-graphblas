
User-defined Types (UDTs)
=========================

python-graphblas supports user-defined types (record-style structs and
fixed-shape arrays) as the value type of any ``Scalar``, ``Vector``, or
``Matrix``. Built-in arithmetic operators automatically lift to UDTs
field-by-field, and SuiteSparse:GraphBLAS JIT-compiles dedicated C kernels
for them when possible.

What is a UDT
-------------

A UDT is any ``numpy.dtype`` you register with python-graphblas. There are three shapes.

**Record UDTs** have heterogeneous fields, like a C struct:

.. code-block:: python

    import numpy as np
    from graphblas import dtypes

    edge_dtype = dtypes.register_anonymous(
        np.dtype([("weight", np.float64), ("hops", np.int32)], align=True),
        "Edge",
    )

**Array UDTs** are fixed-shape, homogeneous values, like an inline C array:

.. code-block:: python

    point3 = dtypes.register_anonymous(np.dtype((np.float64, (3,))), "Point3")

Multi-dimensional shapes work too (``np.dtype((np.float64, (2, 4)))``); the
layout is flattened row-major in C. An array of arrays has the same layout, so
it registers as the same UDT: ``np.dtype((point3.np_type, (2,)))`` is
``FP64[2, 3]``. Record fields that nest arrays are flattened the same way.

**Dataclass UDTs** are record UDTs derived from a ``@dataclass``:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class Edge:
        weight: float
        hops: int

    edge_dtype = dtypes.register_anonymous(Edge)

Field annotations may be real types (``int``, ``float``) or string forms
(``"int"``, e.g. under ``from __future__ import annotations``). Compound
annotations like ``Optional[int]`` raise from ``lookup_dtype``.

Registration
------------

Two forms:

.. code-block:: python

    # register_anonymous returns the DataType but does not add it to gb.dtypes.
    udt = dtypes.register_anonymous(numpy_dtype_or_dataclass, "MyUdt")

    # register_new is the same, but also assigns to gb.dtypes.MyUdt.
    udt = dtypes.register_new("MyUdt", numpy_dtype_or_dataclass)

Field type rules:

- Numeric scalar types (``int``, ``float``, ``bool``, ``complex``, the
  corresponding numpy scalar types) are supported.
- Fixed-shape arrays (``("pos", np.float64, (3,))``) and records (a nested
  struct) are supported as fields, and built-in operators lift through them.
  In the dict and dataclass forms, write an array field as ``"FP64[3]"``; the
  dict form also takes an existing UDT, such as ``{"id": int, "pos": point3}``.
- Strings and Python objects are not supported.
- Dataclass annotation strings (e.g., ``"int"``) resolve through
  ``lookup_dtype``.

Anonymous UDTs share one ``DataType`` per ``numpy.dtype``. Re-registering the
same dtype under a different name updates the Python-side ``name`` but does
*not* change the SuiteSparse-side ``GxB_JIT_C_NAME``, which is frozen at first
registration. See :ref:`udt_jit_introspection` below.

Working with UDT values
-----------------------

Construct, get, set, and iterate as usual; values are numpy structured scalars
or arrays, depending on the UDT shape:

.. code-block:: python

    from graphblas import Vector

    v = Vector(edge_dtype, size=3)
    v[0] = (1.5, 4)              # tuple matches the record fields
    v[1] = (2.5, 7)
    print(v[0].new().value)      # numpy.void; access fields by name (e.g., ['weight'])

For array UDTs, pass a sequence or numpy array:

.. code-block:: python

    p = Vector(point3, size=2)
    p[0] = (1.0, 2.0, 3.0)
    p[1] = np.array([4.0, 5.0, 6.0])

Built-in operators on UDTs
--------------------------

The following operators auto-lift to any UDT on first use:

- BinaryOps: ``plus``, ``minus``, ``times``, ``truediv``, ``floordiv``,
  ``min``, ``max``, ``eq``, ``ne``. The positional selectors ``first``,
  ``second``, ``any``, and ``pair`` work too, since they don't touch field
  values.
- UnaryOps: ``ainv``, ``abs``.
- Monoids: ``plus``, ``times``, ``min``, ``max``, ``any``.
- Semirings combining a UDT-lifting monoid with a UDT-lifting BinaryOp
  (e.g., ``plus_times``, ``min_plus``, ``max_times``, ``any_first``).
- Aggregators built on those monoids: ``sum``, ``prod``, ``min``, ``max``,
  ``any_value``, ``count``. The positional ``agg.ss.first`` and
  ``agg.ss.last`` work on any dtype, including UDTs.

For example:

.. code-block:: python

    from graphblas import binary, monoid, semiring, agg

    plus_edge   = binary.plus[edge_dtype]            # field-wise add
    sum_edges   = monoid.plus[edge_dtype]            # field-wise additive monoid
    plus_times  = semiring.plus_times[edge_dtype]    # field-wise multiply-add semiring
    total       = v.reduce(agg.sum[edge_dtype]).new()  # field-wise reduce

How results are typed
~~~~~~~~~~~~~~~~~~~~~

A lifted op types each field or element as it types a Vector of that built-in
dtype, so it never narrows. Below, ``v`` is a Vector of an ``INT8[3]`` UDT.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Operands
     - Rule
     - Example
   * - One UDT
     - Each field or element as on its built-in dtype.
     - ``v / v`` is ``FP64[3]``
   * - Two UDTs
     - Records pair when they have the same field names, in the same order, at
       every level, and the same shape for each field. Array UDTs pair when
       their shapes broadcast as numpy's do. Any other pair raises
       ``KeyError``.
     - ``INT8[3]`` with ``INT64[2, 3]`` gives ``INT64[2, 3]``
   * - Result type
     - An operand's type when it holds the result; else the promoted layout,
       which is the UDT registered with that layout if there is one, else an
       anonymous UDT.
     - ``{x: INT64, y: FP64}`` with ``{x: FP64, y: INT64}`` gives
       ``{x: FP64, y: FP64}``
   * - A built-in Vector, ``Scalar``, numpy scalar or numpy array
     - Strong: keeps its dtype. A scalar combines with every field or element;
       an array pairs as an array UDT of its own shape.
     - ``v + np.int64(1)`` is ``INT64[3]``
   * - A Python number, or a tuple, list or dict of them laid out like an
       element
     - Weak, field by field: a field's dtype when its kind holds the number,
       else INT64, FP64 or FC64 (numpy 2's NEP 50 rule, on numpy 1 too, and
       extended to sequences). An int out of an integer field's range raises
       ``OverflowError``, except for ``truediv``, whose fields are floats
       whatever the int (``v / 300`` is ``FP64[3]``).
     - ``v + 1`` is ``INT8[3]``, ``v * 2.5`` and ``v + (0.5, 1, 2)`` are
       ``FP64[3]``, ``v + 300`` raises
   * - ``eq`` and ``ne``
     - Type a literal as above, as built-in comparisons do, except that an int
       out of a field's range takes a type that holds it, so the comparison is
       exact. Return ``BOOL``. Array elements broadcast, as numpy's ``==``
       does, and so does ``==`` on a Scalar, which returns a bool. ``isequal``
       compares as ``np.array_equal``: elements must have the same shape,
       apart from leading axes of length 1.
     - ``v == 300`` is ``False``; an ``FP32[3]`` UDT ``== 0.1`` compares in
       FP32; a Scalar ``s`` of ``[1, 1, 1]`` has ``s == 1`` but not
       ``s.isequal(1)``
   * - A literal that must become an element: beside a user-defined op, an
       ``IndexUnaryOp`` thunk, an ``ewise_union`` default
     - Its type by the rules above must fit the UDT: the UDT itself, or a
       numpy value or ``Scalar`` that casts to it safely. A weak int does not
       fit a bool field: write ``True`` or ``False``. A value that does not fit
       but converts exactly, such as ``0`` for a bool field or ``2.0`` for an
       int field, still converts with a ``DeprecationWarning`` and will raise
       in a future version; any other raises ``ValueError``, and a float too
       large for a float field raises ``OverflowError``.
     - ``v.ewise_union(w, binary.plus, 0.5, 0)`` raises

A Monoid used element-wise (``apply``, ``ewise_mult``, ``ewise_union``) uses
its BinaryOp, for literals and for two different UDTs. ``min``, ``max`` and ``floordiv`` reject UDTs
with a complex field; use a custom op for those.

Each element, including each element of an array field, is computed as the
built-in op computes it, and the Numba cfunc and a C JIT kernel give the same
bits: ``min`` and ``max`` ignore NaN (C's ``fmin`` and ``fmax``) and resolve a
tie between ``-0.0`` and ``0.0`` to ``-0.0`` for ``min`` and ``0.0`` for
``max``, ``floordiv`` is numpy's, with 0 for an integer divided by zero, and
``truediv`` is numpy's (Smith's method for complex numbers), with numpy's
infinities for a zero divisor. A few edge cases can differ from built-in
vectors, where the UDT gives numpy's answer: ``INT64_MIN // -1`` is
``INT64_MIN`` (an ``INT64`` vector gives 0), a complex product or quotient with
an infinite or NaN part, a complex quotient with a ``-0.0`` divisor, and the
signed-zero tie, which SuiteSparse's built-in ``min`` leaves to the C library.

Storing results
~~~~~~~~~~~~~~~

GraphBLAS casts built-in dtypes when it stores a result, but it cannot cast a
UDT, so python-graphblas does it, without ever making a temporary copy of the
data.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Store
     - What happens
   * - Into an object of another UDT type of the same layout: records that
       pair as operands do, or arrays of the same shape apart from leading
       axes of length 1
     - Each field or element is cast as GraphBLAS casts that pair of built-in
       dtypes: integers wrap; a float stored as an integer truncates toward
       zero, saturates at the bounds, and is 0 if NaN; a complex number loses
       its imaginary part; anything nonzero is ``True``.
   * - ...from an existing object: ``w << v``, ``v.dup(dtype=...)``, with a
       mask, an accumulator or ``replace``
     - One pass through a cast op.
   * - ...from an element-wise lifted op: ``apply``, ``ewise_mult``,
       ``ewise_union``, and infix operators that use them
     - The op writes the object's type itself, in the same call:
       ``int_udt_vec += 0.5`` truncates, as it does for an ``INT8`` vector.
   * - ...from anything else: ``mxm``, ``reduce`` into a Vector, ``ewise_add``,
       a user-defined op, or a whole Vector or Matrix assigned into part of an
       object
     - ``DomainMismatch``: this would need a converted copy of the whole
       result. Make it explicitly: compute the result, then store it
       (``w << expr.new()``); to assign, convert the value with
       ``.dup(dtype=...)`` first.
   * - ...from a ``Scalar`` or one element
     - Computed in its own type, then cast.
   * - Between a UDT and a built-in dtype, or layouts that do not correspond
     - ``DomainMismatch``.

For the same reason, ``ewise_add`` on Vectors and Matrices raises
``DomainMismatch`` when an operand's type differs from the result's, as for two
different UDTs or ``truediv`` on an integer UDT: it would copy a value that has
no partner into the result type. ``ewise_union`` and ``ewise_mult`` work. Infix
``+`` on two such operands takes the union instead, with each operand's zero for
a missing value, as ``-`` takes 0. That zero is ``-0.0`` in a float field, so an
entry of one operand alone keeps its value, as ``ewise_add`` would copy it;
only a ``-0.0`` beside an integer field of the other operand becomes ``0.0``. A
Scalar is one element, so ``ewise_add`` converts its operands to the result type
instead.

.. code-block:: python

    v = Vector(int8x3, size=2)          # an INT8[3] UDT
    f = Vector(fp32x3, size=2)          # an FP32[3] UDT
    (v * 2.5).new()                     # FP64[3]: the literal is weak, 2.5 needs FP64
    (v * f).new()                       # FP32[3], as INT8 times FP32
    v += 1                              # stays INT8[3]
    v << v * 2.5                        # computes in FP64, stores as INT8 (truncates)
    (v + f).new()                       # FP32[3], by ewise_union; v.ewise_add(f) raises

Composite aggregators (``agg.hypot``, ``agg.L1norm``, ``agg.Linfnorm``,
``agg.sum_of_squares``, ``agg.sum_of_inverses``) do *not* auto-lift to UDTs;
they reference scalar-only binary ops that don't generalize trivially.
Use a custom monoid plus reduction if you need this on UDT-valued data.

Custom UDFs over UDTs
---------------------

Write a function with explicit field access, register it with ``is_udt=True``:

.. code-block:: python

    from graphblas import binary

    def merge_edges(x, y):
        # Returns a new Edge with the lower weight and summed hops.
        return (min(x["weight"], y["weight"]), x["hops"] + y["hops"])

    op = binary.register_new("merge_edges", merge_edges, is_udt=True)

    a = Vector(edge_dtype, size=2)
    a[0] = (1.5, 2)
    a[1] = (3.0, 4)
    b = Vector(edge_dtype, size=2)
    b[0] = (1.0, 3)
    b[1] = (2.5, 1)

    c = a.ewise_mult(b, op[edge_dtype]).new()
    # c[0] = (1.0, 5);  c[1] = (2.5, 5)

UDT UDFs can return a tuple matching the field layout, an existing record
value, or a numpy array (for array UDTs).

For nested record UDTs the tuple is *flat over the leaves*. Given
``[("id", int32), ("pt", [("x", float64), ("y", float64)])]``, the UDF should
return ``(id, x, y)``, not ``(id, (x, y))``. Returning an existing record value
(e.g., one of the inputs) is also fine and preserves the nested shape.

Every field needs a value on every path through the UDF. A record cannot hold
``None``, so a UDF that can return ``None`` for a field
(``v if cond else None``) is rejected when the op is typed. So is an array,
tuple, or list returned for a scalar field.

For array UDTs each operand arrives as a numpy view of that element's values, in
the UDT's declared shape: a ``np.dtype((np.float64, (2, 4)))`` UDT hands the UDF
a 2-by-4 array, indexable as ``x[i, j]``. Array expressions work as written, and
the UDF may return one of its operands or build a new array that fills the
element, either at the element's own shape or at one that broadcasts to it (a
``(1,)`` return fills every slot of a ``(6,)`` element):

.. code-block:: python

    def midpoint(x, y):
        return (x + y) / 2

    op = binary.register_new("midpoint", midpoint, is_udt=True)

    a = Vector(point3, size=1)
    a[0] = [0.0, 2.0, 4.0]
    b = Vector(point3, size=1)
    b[0] = [10.0, 20.0, 30.0]

    c = a.ewise_mult(b, op[point3]).new()
    # c[0] = [5.0, 11.0, 17.0]

A return that cannot fill the element, such as ``x[:2]`` from a 3-element UDT,
is rejected with a ``UdfParseError`` when the op is typed for the UDT
(``op[point3]``, or the first operation that uses it). The same goes for an
array returned into an array-typed field of a record UDT. A tuple or list also
fills such a field, but only a one-dimensional one and only at its exact
length; unlike an array, it does not broadcast. Numba does not know the shape
of an array the UDF builds, or the length of a list, so the check runs the UDF
once on sample values; a UDF that raises on those values is not checked.

If your UDF references a field that doesn't exist, or returns the wrong arity,
you'll get a ``UdfParseError`` with the actionable diagnostic line surfaced
from Numba's typing pass instead of a 200-line traceback.

.. _udt_jit_introspection:

JIT and introspection
---------------------

When the UDT name and all field names are valid C identifiers (not C reserved
words), and the field types map to C primitives, SuiteSparse JIT-compiles a
dedicated C kernel for each ``op[udt]`` lookup. The kernel is cached on disk
and reused across processes (keyed by content hash, so renaming a UDT doesn't
invalidate). JIT lets SuiteSparse inline the kernel into its eWise and reduce
templates, eliminating the per-element function-call overhead the Numba
function-pointer fallback incurs. Elementwise operations on UDTs are
typically **2-3x** faster.

Inspect what SuiteSparse is JIT-ing from Python:

.. code-block:: python

    udt.jit_c_name              # 'Edge', or None if not JIT-able
    udt.jit_c_definition        # 'typedef struct { ... } Edge ;'

    op = binary.plus[udt]
    op.jit_c_name               # 'plus_Edge'
    op.jit_c_source             # full C source SS will compile

    monoid.plus[udt].jit_c_source                # same kernel
    semiring.plus_times[udt].jit_c_source        # the multiplier
    semiring.plus_times[udt].monoid.jit_c_source # the additive monoid
    agg.sum[udt].jit_c_source                    # walks to monoid.plus
    agg.count[udt].jit_c_source                  # None (composite agg)

JIT is skipped when:

- A field name (or the UDT name, if you supplied one) is a C reserved word
  (``class``, ``return``, ...) or a stdlib macro/typedef pulled in by
  ``GraphBLAS.h`` (``NULL``, ``FILE``, ``M_PI``, ``complex``, ...). The op
  falls through to the Numba cfunc path.
- A field type isn't in the numpy-to-C map (rare; the standard numeric
  scalar types all map).
- A record has an array-valued field. Built-in operators still lift to it
  through the Numba cfunc.
- The numpy layout doesn't match what a C compiler would produce. The most
  common case is a packed record with mixed-width fields (e.g.,
  ``np.dtype([("a", int32), ("b", float64)])`` without ``align=True``).
  Pass ``align=True`` to ``np.dtype``, or use the dict / dataclass form,
  which auto-aligns.
- The JIT compiler isn't usable after auto-fix (see below).

Anonymous UDTs (no ``name=`` argument) still take the JIT path: a synthetic
``_gbudt_NNN`` C name is minted so SuiteSparse always has a registerable
identifier. Pass ``name=`` only for readable JIT cache filenames and
introspection.

When the cause is the UDT itself (the first four cases above), the first time
auto-lift produces a non-JIT'd op for a given ``(op, dtype)`` pair in a
process, a ``graphblas.exceptions.NoJITWarning`` (a subclass of
``UserWarning``) names the cause. An unusable compiler does not warn;
``gb.ss.fix_jit_config()`` returns ``False`` when it cannot compile, and
``gb.ss.jit_compiler_is_usable()`` is the cheap check. Silence the warning by
category or by message::

    import warnings
    from graphblas.exceptions import NoJITWarning
    warnings.filterwarnings("ignore", category=NoJITWarning)
    # or, equivalently, match the message text:
    warnings.filterwarnings("ignore", message="UDT operator running without JIT")

In every skipped case the operation still works correctly via the Numba
function-pointer path; you only lose the JIT speedup.

JIT compiler auto-fix
~~~~~~~~~~~~~~~~~~~~~

A conda-installed ``python-suitesparse-graphblas`` bakes in the build host's
compiler path (e.g., ``/Users/runner/...``), which doesn't exist on user
machines. With the bogus default, SuiteSparse emits the JIT ``.c`` source
but never compiles a ``.dylib`` or ``.so``, and silently falls back to the
cfunc path. The 2-3x JIT speedup is silently lost.

python-graphblas repairs the compiler path when ``graphblas.ss`` is
imported, which the first access to ``gb.ss`` does. If
``jit_c_compiler_name`` doesn't exist on disk, it is replaced with one from
``$CONDA_PREFIX/bin/`` (or from ``sysconfig`` for pure-pip installs). The
import changes nothing else, so reading ``gb.ss.about`` does not change what
later operations compute.

``jit_c_control`` is raised from the SS default ``'run'`` (run cached kernels
only; no compile, no load from disk) to ``'on'`` (compile, load, and run) by
each of these, since each asks SuiteSparse to compile C source:

- a UDT operator that gets C source, such as the first ``binary.plus[udt]``;
- ``gb.dtypes.ss.register_new``, which registers a type from a C typedef;
- ``gb.binary.ss.register_new`` and its ``unary``, ``indexunary``, ``select``,
  and ``indexbinary`` siblings, which register an operator from a C definition.

A setting other than ``'run'`` is kept, including ``'off'``, ``'pause'``, and
``'load'`` (load and run cached kernels, never compile). SuiteSparse itself
sets ``'load'`` after a compile fails, so keeping it means a failing compile
is tried once, not again at every later request.

Call the helper manually to re-fix or verify::

    gb.ss.fix_jit_config()           # repair compiler path, set 'on', probe
    gb.ss.jit_compiler_is_usable()   # cheap check: True iff path exists

Pickle and serialize
--------------------

UDT-typed Matrices and Vectors pickle round-trip in the same process. For
cross-process work (``multiprocessing.spawn``, ``dask``, etc.) the receiving
process must also have the UDT registered under the same name. UDTs registered
via ``register_new`` re-register automatically on unpickle.

``ss.serialize`` and ``ss.deserialize`` work too: the dtype name is stored
in the blob and re-resolved on load, falling back to ``GrB_NAME`` for UDTs
whose C identifier differs from the Python name.

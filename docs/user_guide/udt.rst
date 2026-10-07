
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

The lift is field-by-field for record UDTs and element-by-element for array
UDTs. Each field or element of the result has the dtype the op gives on the
built-in dtypes, so a lifted op does not narrow: ``truediv`` on an ``INT64[3]``
UDT gives ``FP64[3]``, as ``truediv`` on two INT64 vectors gives FP64. Field
types that don't support the operation raise ``KeyError`` on the first lookup.
``binary.min``, ``binary.max``, and ``binary.floordiv`` reject UDTs with any
complex leaf (no ordering, no integer modulus); use ``plus``, ``minus``,
``times``, or ``truediv`` for complex arithmetic, or register a custom binary
op.

A scalar operand combines with every field or element. A vector, ``Scalar``,
numpy scalar or 0-d array of a built-in dtype keeps its dtype, so with ``v`` of
an ``INT8[3]`` UDT, ``v + np.int64(1)`` is ``INT64[3]``. A Python number is
weak, as numpy 2 treats it (NEP 50): in each field or element it takes that
field's dtype when the field's kind can hold it, and its own default dtype
(INT64, FP64 or FC64) when not. So ``v + 1`` stays ``INT8[3]``, ``v * 2.5``
is ``FP64[3]``, and ``v + 300`` raises ``OverflowError``. A record with an ``INT8`` field and an ``FP32``
field, plus ``0.5``, has an ``FP64`` field and an ``FP32`` one. python-graphblas
applies this rule itself, so it does not depend on the installed numpy (vectors
of built-in dtypes still type a Python number through numpy, which differs
between numpy 1 and 2). ``eq`` and ``ne`` compare with the number as it is.

A literal is converted to the UDT instead when it is a whole element (a tuple,
a list, or a numpy array that is not 0-d), when the op is user-defined or a
Monoid, and for the defaults of ``ewise_union``, which must have the op's input
types (a default given as a Scalar of another type is converted too).
``apply`` with a Monoid applies its BinaryOp, so there a literal is typed as
that BinaryOp types it. The conversion must keep every value:
``v + (1, 2, 3)`` works, but ``v + (0.5, 1.5, 2.5)`` raises ``ValueError``
rather than adding ``(0, 1, 2)``, and ``v.ewise_union(w, binary.plus, 0.5, 0)``
raises rather than using ``0`` for a missing ``v`` entry. A number may still
round to a narrower float field, as it does beside an ``FP32`` field above.

The lifted binary ops also take two different UDTs whose shapes match. Two
record UDTs match when they nest the same way, with the same field names in the
same order at every level, and each array field has the same shape in both;
array fields do not broadcast against each other. Two array UDTs match when
their shapes broadcast together as numpy arrays do, and the result has the
broadcast shape: ``FP64[3]`` pairs with ``INT64[2, 3]`` to give ``FP64[2, 3]``,
and ``FP64[3, 1]`` with ``FP64[1, 4]`` gives ``FP64[3, 4]``. ``eq`` is True
when every pair of elements numpy would compare is equal. Any other pair, such
as ``FP64[2, 3]`` with ``FP64[3, 2]``, raises ``KeyError`` on the first lookup.
Field and element dtypes may differ, and each pair of them follows the
built-in dtype rules; a dtype that is not a built-in one, such as bytes, pairs
only with itself. A Monoid takes one type, so ``ewise_add``, ``ewise_mult`` and
``ewise_union`` with a Monoid on two different UDTs use its BinaryOp.

The result is an operand's UDT when that type holds it, on either side.
Otherwise it is the promoted type: for array UDTs, the structural type such as
``FP64[3]``, and for records, the record with the same names and nesting and
the promoted field dtypes, so ``{"x": INT64, "y": FP64}`` plus
``{"x": FP64, "y": INT64}`` gives ``{"x": FP64, "y": FP64}``. That is the UDT
registered with that layout if there is one, else an anonymous UDT. ``eq`` and
``ne`` compare the values and return ``BOOL``.

A result stored in an object of another UDT type is cast to that type, as
GraphBLAS casts built-in dtypes when it stores a result: the object's type is
the request. GraphBLAS cannot cast a UDT itself, so python-graphblas casts each
element or record leaf the way GraphBLAS casts that pair of built-in dtypes.
Integers wrap; a float stored as an integer is truncated toward zero, saturates
at the integer type's bounds, and is 0 if NaN; a complex number loses its
imaginary part; and anything nonzero is ``True``. So
``int_udt_vec << int_udt_vec + 0.5``, ``int_udt_vec += 0.5`` and
``(int_udt_vec + 0.5).new(dtype=int_udt)`` truncate, as they do for an ``INT8``
vector, and so does assigning, accumulating or masking into an object of
another UDT type. The two types must correspond as the operands of a lifted op
must, above (same field names and nesting, same shapes). numpy also casts
records by position whatever their names, and arrays to another length, but
those raise ``DomainMismatch`` here, as does storing a UDT result in an object
of a built-in dtype or the reverse. The cast is an extra pass over the result,
which is first computed in its own type. ``ewise_add`` has a related limit:
GraphBLAS casts a value that has no partner to the op's output type, so it
takes UDT operands only when both have the result's type. With a lifted
arithmetic op, it converts each operand to the result type first, which changes
no value, and applies the op there. So ``v + w`` and
``v.ewise_add(w, binary.max)`` work for two UDTs of different types, and
``truediv`` works on an integer UDT. With any other op it raises
``DomainMismatch``; use ``ewise_union`` with a default for each side instead.

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

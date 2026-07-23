
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
layout is flattened row-major in C.

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
- Strings, Python objects, and nested UDTs are not.
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
UDTs. Field types that don't support the operation raise ``KeyError`` on the
first lookup. ``binary.min``, ``binary.max``, and ``binary.floordiv`` reject
UDTs with any complex leaf (no ordering, no integer modulus); use ``plus``,
``minus``, ``times``, or ``truediv`` for complex arithmetic, or register a
custom binary op.

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

The first time auto-lift produces a non-JIT'd op for a given ``(op, dtype)``
pair in a process, a ``graphblas.exceptions.NoJITWarning`` (a subclass of
``UserWarning``) is emitted with the cause and the remediation. The warning
fires once per ``(op, dtype)`` rather than once per process, so distinct
fallback causes each surface a warning. Silence it by category or by
message::

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

python-graphblas auto-fixes this at import. If ``jit_c_compiler_name``
doesn't exist on disk, it is replaced with one from ``$CONDA_PREFIX/bin/``
(or from ``sysconfig`` for pure-pip installs), and ``jit_c_control`` is
bumped from the SS default ``'run'`` (run cached kernels only; no compile,
no load from disk) to ``'on'`` (compile, load, and run). When the default
config is already valid, only the mode bump applies.

Call the helper manually to re-fix or verify::

    gb.ss.fix_jit_config()           # repair compiler path (full probe)
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

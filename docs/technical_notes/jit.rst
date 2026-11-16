.. _tn_jit:

The SuiteSparse:GraphBLAS JIT
=============================

How python-graphblas sets the JIT control, repairs the C compiler setting, and gives
SuiteSparse:GraphBLAS C definitions for user-defined types and lifted operators. The user
guide covers the JIT in :ref:`suitesparse_jit` and :doc:`../user_guide/udt`.

.. _tn_jit_control:

The JIT control
---------------

``gb.ss.config["jit_c_control"]`` takes these values:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - SuiteSparse
   * - ``"on"``
     - compiles, loads and runs JIT kernels
   * - ``"load"``
     - loads kernels from its cache on disk and runs them, but compiles none
   * - ``"run"``
     - runs kernels already loaded in the process, but loads and compiles none
   * - ``"pause"``
     - runs no JIT kernels, and keeps those loaded
   * - ``"off"``
     - runs no JIT kernels, and unloads them

**At initialization.** The ``suitesparse`` backend sets the control from
``graphblas.config["jit_c_control"]``, read once. Set it with the
``GRAPHBLAS_JIT_C_CONTROL`` environment variable, with ``gb.config.set`` before graphblas
initializes, or in a YAML file in ``~/.config/graphblas/``, where an unquoted ``on`` or
``off`` also works. Any other value raises ``ValueError`` before anything initializes.
Unset, the control is ``"run"``. The ``suitesparse-vanilla`` backend leaves the control
as the SuiteSparse build sets it, and SuiteSparse:GraphBLAS 7 has no JIT.

A configured ``"on"`` is set only where it works. python-graphblas repairs the compiler
setting if its path does not exist (:ref:`tn_jit_compiler`), checks that the cache
directory is writable, and compiles a tiny type to check the compiler. The check runs once
for each cache directory and compiler setting; after that it loads the type from the
cache. Where any step fails, the control is ``"run"``, with no error or warning:

- SuiteSparse built without the JIT, as in the PyPI wheels of
  python-suitesparse-graphblas;
- no C compiler, or one that fails the check (in a conda environment,
  ``conda install c-compiler`` provides one);
- a cache directory it cannot write to.

**Later.** ``gb.ss.config["jit_c_control"] = value`` sets the control for the rest of the
process, without the check. With a C compiler that does not work, the first operation
that needs a new kernel raises ``JitError`` (from SuiteSparse 9.4 on), and SuiteSparse
lowers the control to ``"load"``.

**Raised for C source.** The default ``"run"`` is raised to ``"on"`` the first time one
of these needs a kernel compiled:

- a lifted operator on a UDT that gets a C definition, such as the first
  ``binary.plus[udt]`` or ``binary.plus[udt, "INT64"]``;
- ``gb.dtypes.ss.register_new``, which registers a type from a C ``typedef``;
- ``gb.binary.ss.register_new``, and the ``register_new`` of ``gb.unary.ss``,
  ``gb.indexunary.ss``, ``gb.select.ss`` and ``gb.indexbinary.ss``, which register an
  operator from C source.

Before it raises the control the first time, python-graphblas runs the same compiler check
as for a configured ``"on"``, once per process. If the check fails, the control stays
``"run"``. Otherwise it stays ``"on"`` for the rest of the process, for every operation,
built-in ones included.

**Kept.** A control you set is kept: ``"off"``, ``"pause"`` and ``"load"`` always, and
``"run"`` when it was configured before initialization, or written to ``gb.ss.config``
while another value was configured. With nothing configured, a ``"run"`` (or ``None``)
written to ``gb.ss.config`` is the default again, and is raised. Under a kept control,
lifted operators on UDTs call their Numba functions, and the registrations above raise an
error that says why and how to allow compiling. SuiteSparse itself sets ``"load"`` after a
compile fails; keeping it means a failing compile is tried once, not at every later
request.

.. _tn_jit_compiler:

The compiler setting
--------------------

A conda-installed python-suitesparse-graphblas has the build machine's compiler path in
``gb.ss.config["jit_c_compiler_name"]``, which does not exist on other machines. With that
path, SuiteSparse writes a kernel's C source but cannot compile it.

Importing ``graphblas.ss``, which the first access of ``gb.ss`` does, repairs the setting
when the path does not exist. It takes a compiler from ``$CONDA_PREFIX/bin`` (on Windows,
``$CONDA_PREFIX/Library/bin``), and removes compiler flags that refer to the build machine
(``-isysroot`` with a missing SDK, ``-arch`` for another architecture,
``-fdebug-prefix-map``). Outside conda, it takes the compiler and flags ``sysconfig``
names. The import changes nothing else; in
particular it leaves ``jit_c_control`` alone.

- ``gb.ss.fix_jit_config()`` repairs the setting, sets the control to ``"on"``, and
  compiles a tiny type to check it. It returns ``True`` when that works, ``False`` when the
  check fails, and ``None`` when there is no compiler to repair from.
- ``gb.ss.jit_compiler_is_usable()`` returns whether the configured compiler path exists,
  without compiling anything.

.. _tn_jit_c_names:

C definitions and names
-----------------------

When a record or array UDT is registered, python-graphblas gives SuiteSparse a C
``typedef`` for it and a C name. A record becomes a struct with the same field names, and
a nested record a struct of its own, named ``_gbnest_`` and a number. An array UDT
becomes a struct with one flat array, ``v``, in row-major order:
``typedef struct { double v [4] ; } gbudt_Feat ;``.

A UDT's C name is ``gbudt_`` followed by its name, so it cannot collide with an
identifier in a kernel's C code, whatever the name. A name that starts with ``_gb``,
which python-graphblas keeps for the types it makes itself, is the C name as it is. A UDT
gets a C name of ``_gbudt_`` and a number instead when:

- it has no name;
- its name does not make a C identifier;
- the C name would be longer than 118 characters, which leaves room for the longest
  operator prefix, ``floordiv_``, within SuiteSparse's limit on names;
- or another layout already has that C name.

The C name is fixed at the first registration of the numpy dtype
(:ref:`tn_udt_same_dtype`).

A lifted operator on one UDT is named after the operator and the UDT's C name:
``plus_gbudt_Edge``, ``eq_gbudt_Edge``. One on two types also has the length of the first
operand's C name and both C names: ``plus_10_gbudt_Edge_int64_t``.

To name a UDT in C source of your own, as for ``gb.binary.ss.register_new``, use its
``jit_c_name``, not its Python name, which SuiteSparse does not know. A type registered
from C source (``gb.dtypes.ss.register_new``) keeps the name you give it as its C name.
Keep that name to 118 characters or fewer: the lifted operators on it are named with the
name after them, and a longer one gives them names SuiteSparse refuses, so they run
without the JIT.

.. _tn_jit_skipped:

When a lifted operator runs without the JIT
-------------------------------------------

A lifted operator always has a Numba function, which SuiteSparse calls for each element
when it has no JIT kernel. These lifted operators have no C definition, and so always run
that way:

- operators on a UDT that has no C ``typedef``, because:

  - a field name is a C keyword (``class``, ``return``), a macro or typedef of the C
    headers the kernels include (``NULL``, ``FILE``, ``M_PI``, ``complex``, ``INT_MAX``,
    ``SIZE_MAX``, ``BUFSIZ``, ``SIGINT``, ...), or starts with ``GB_``, ``GrB_`` or
    ``GxB_``, which GraphBLAS keeps for its own names;
  - a field has a type with no C type: text, a date or a duration;
  - a record has an array field;
  - a record's numpy layout differs from what a C compiler gives the struct, as for a
    packed record of mixed widths such as ``np.dtype([("a", np.int32), ("b",
    np.float64)])``. Pass ``align=True``, or declare the record with a dict or a
    dataclass, which align it;

- an operator on one UDT whose result has another type: ``a / a`` on ``INT8[3]`` is
  ``FP64[3]``;
- an operator on two types where an array UDT broadcasts to another shape, or a record
  has an array field;
- ``eq`` and ``ne`` on two different types.

For the first group, the first time a lifted operator is typed for such a UDT in a
process, a ``graphblas.exceptions.NoJITWarning`` (a ``UserWarning``) names the possible
causes. It fires once for each operator and UDT. The other cases do not warn. Silence the
warning by category or by message:

.. code-block:: python

    import warnings

    from graphblas.exceptions import NoJITWarning

    warnings.filterwarnings("ignore", category=NoJITWarning)
    # or, equally, by the start of the message:
    warnings.filterwarnings("ignore", message="UDT operator running without JIT")

A field named like a macro that only some platform's C headers define is not caught. Its
kernel fails to compile, which raises ``JitError`` and sets the control to ``"load"``;
rename the field.

Operators you write with Numba (:doc:`../user_guide/udf`) have no C definition either,
and always run as Numba functions. An unusable compiler, or a control that does not
compile, also leaves lifted operators on their Numba functions, without a warning.

With SuiteSparse:GraphBLAS 7 or 8, or with the ``suitesparse-vanilla`` backend, no UDT or
lifted operator gets a C definition, and nothing warns.

.. _tn_jit_introspection:

Introspection
-------------

These properties show what python-graphblas gave SuiteSparse:

- ``DataType.jit_c_name`` and ``DataType.jit_c_definition``: the UDT's C name and
  ``typedef``, as SuiteSparse has them, or ``None`` for a built-in type or a UDT with no C
  definition.
- ``jit_c_name`` and ``jit_c_source`` on a typed operator, such as ``binary.plus[udt]``:
  the operator's C name and function, or ``None``. ``monoid.plus[udt]``,
  ``agg.sum[udt]`` and the monoid of ``semiring.plus_times[udt]`` show their
  ``binary.plus``; ``semiring.plus_times[udt]`` shows its multiplier, ``binary.times``.
  ``agg.count[udt]`` has none.

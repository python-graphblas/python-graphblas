
GraphBLAS Initialization
========================

GraphBLAS must be initialized before it can be used. This is done with the
``graphblas.init`` function. The backend and the mode must be specified.

.. code-block:: python

    import graphblas as gb

    # Context initialization must happen before any other imports
    gb.init("suitesparse", blocking=False)

    # Now we can import other items from graphblas
    from graphblas import binary, semiring
    from graphblas import Matrix, Vector, Scalar


Supported Backends
------------------

The only supported backend is current ``suitesparse``, although the plan is to support
additional backends in the future.

GraphBLAS Modes
---------------

The GraphBLAS spec contains the idea of **BLOCKING** vs. **NONBLOCKING** modes of operation.

Blocking mode requires immediate execution, while nonblocking mode allows for some aspects of
a function call to be delayed. Error checking on the inputs, compatibility with the output, etc
must be checked immediately, but the actual computation is allowed to be delayed. The hope is
that some GraphBLAS implementations will find optimizations if they see a string of computations
and realize that the output of an intermediate computation is not used and there is a way to
fuse operations together.

Thus far, this mechanism hasn't been fully realized. The only known usages are with adding or
removing a single element at a time. In nonblocking mode, this is much more efficient as the
element additions are queued up and the object container is fully computed at the last possible
moment.

Default Initialization
----------------------

If submodules are imported before calling ``graphblas.init``, a default initialization occurs.
The default backend is ``suitesparse`` and the default mode is ``nonblocking``.

If ``graphblas.init`` is called after the default initialization with different parameters,
an error will be raised.

Duplicate Initialization
------------------------

If ``graphblas.init`` is called multiple times with the same arguments, no error will be
raised. However, if the arguments do not match, an error will be raised.

There is no mechanism to change the initialization arguments after graphblas has been
initialized the first time.

.. _suitesparse_jit:

The SuiteSparse:GraphBLAS JIT
-----------------------------

SuiteSparse:GraphBLAS ships precompiled kernels for common operations. For the rest it
can compile a kernel with a C compiler (its JIT), or else run a generic kernel that is
often several times slower (``Matrix.reduce_rowwise`` with ``lor`` on ``BOOL`` or
``plus`` on ``UINT64`` ran 2.7 to 8 times slower on 10 million entries). The JIT control,
``gb.ss.config["jit_c_control"]``, is ``"run"`` by default: run only kernels already
loaded in the process, and compile nothing. UDT operators, and types and operators
registered from C source, raise this default to ``"on"`` when they need a kernel (see
:doc:`udt`), but leave a control you set as it is. To opt in for everything, set the
control to ``"on"`` (compile, load and run):

.. code-block:: bash

    export GRAPHBLAS_JIT_C_CONTROL=on

.. code-block:: python

    import graphblas as gb

    gb.config.set(jit_c_control="on")     # before graphblas initializes
    gb.ss.config["jit_c_control"] = "on"  # or later, for the rest of the process

``graphblas.config["jit_c_control"]`` is read once, when graphblas initializes. It can be
``"on"``, ``"load"`` (load and run cached kernels, never compile), ``"run"``, ``"pause"``
(run no JIT kernels) or ``"off"`` (run none, and unload them). Set it with the
``GRAPHBLAS_JIT_C_CONTROL`` environment variable, with ``gb.config.set`` before graphblas
initializes, or in a YAML file in ``~/.config/graphblas/``. Left unset, it is the
default ``"run"``, which UDT work raises. With nothing configured, a ``"run"`` written
later to ``gb.ss.config`` is that same default, so to keep UDT work from compiling, set
``"run"`` in one of those three ways, or write ``"load"`` or ``"off"`` later. With
``"on"``, python-graphblas checks the C compiler by compiling a tiny type (once per cache
directory and compiler), and falls back to ``"run"``, with no error, where the JIT cannot
work: SuiteSparse built without the JIT (as in the PyPI wheels), SuiteSparse 7, no working
C compiler (in an activated conda environment, ``conda install c-compiler``), or a cache
directory it cannot write to. On Windows, SuiteSparse built with MSVC (as conda-forge
builds it) compiles JIT kernels only through cmake, which finds the Visual Studio C
compiler itself. There ``"on"`` needs cmake (``conda install cmake``) and Visual Studio or
its Build Tools rather than the configured compiler, and UDT work does not raise the
default ``"run"``.

With the JIT on:

- **Do not share one cache directory between processes running at the same time** on
  SuiteSparse 10.1.1 or later, which no longer locks the cache. They race when they
  compile the same new kernel, and calls fail with ``JitError``, leave their outputs
  invalid or cleared, return wrong values with no error, or hang
  (`GraphBLAS issue #449 <https://github.com/DrTimothyAldenDavis/GraphBLAS/issues/449>`_).
  With multiprocessing, dask, pytest-xdist and the like, set
  ``GRAPHBLAS_JIT_C_CONTROL=run`` in the environment the workers start in, so that none
  of them compiles: UDT work raises only the default ``"run"``, not one you set. That
  costs UDTs their compiled kernels: UDT operators call their Numba functions instead
  (typically 2 to 3 times slower for elementwise operations), and registering types or
  operators from C source fails. To keep the kernels, give each process its own cache
  instead, by setting the ``GRAPHBLAS_CACHE_PATH`` environment variable before graphblas
  initializes. The default will change once SuiteSparse publishes compiled kernels
  atomically.
- The first use of each new kernel compiles it, typically in 0.2 to 1 second, or about 10
  seconds through cmake on Windows. Kernels are cached on disk, in
  ``gb.ss.config["jit_cache_path"]`` (by default under ``~/.SuiteSparse``), so later runs
  load them instead.
- The cache is keyed only on the SuiteSparse:GraphBLAS version, so two builds of one
  version (a conda build and a source build, say) can load each other's kernels, and get
  wrong results or errors if their internals differ. Give each build its own cache path.

If you hit a bug, freeze or crash that may involve the JIT, set
``GRAPHBLAS_JIT_C_CONTROL=run`` (which UDT work leaves alone) and delete the cache
directory, then try again. Please
`report it <https://github.com/python-graphblas/python-graphblas/issues>`_ either way. The
``suitesparse-vanilla`` backend leaves the control as the SuiteSparse build sets it.

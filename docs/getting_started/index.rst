
Getting Started
===============

Installation
------------

Using conda:

::

    conda install -c conda-forge python-graphblas

Using pip:

::

    pip install python-graphblas[default]

Whether installing with conda or pip, the underlying package that is imported in Python
is named ``graphblas``. The convention is to import as:

.. code-block:: python

    >>> import graphblas as gb

In the browser (Pyodide)
++++++++++++++++++++++++

python-graphblas also runs in `Pyodide <https://pyodide.org>`__ 314 (Python 3.14) and 0.29.4
or later (Python 3.13), where PyPI has WebAssembly wheels of python-suitesparse-graphblas.
So, in Pyodide or JupyterLite (jupyterlite-pyodide-kernel 0.8 uses Pyodide 314):

.. code-block:: python

    import micropip
    await micropip.install("python-graphblas")

Older versions of Pyodide don't recognize these wheels: micropip says it can't find a pure
Python 3 wheel. What differs in Pyodide:

  - NumPy's default integer is int32, so a NumPy array of integers made without a dtype, such
    as ``np.arange(5)``, makes INT32 Matrix and Vector objects, which wrap past 2**31 - 1.
    Python ints make INT64, as on other platforms. Give ``dtype=int`` (INT64) where values or
    results can be larger.
  - GraphBLAS runs single-threaded, since Pyodide has no threads.
  - A computation that asks for 4 GiB of memory or more at once crashes Pyodide instead of
    raising ``OutOfMemory``, because of a bug in SuiteSparse:GraphBLAS on 32-bit platforms.
  - Numba is not available, so neither are :doc:`../user_guide/udf` nor computing on
    user-defined types.
  - scipy's Matrix Market reader and writer start threads, so ``gb.io.mmread`` and
    ``gb.io.mmwrite`` fail unless they run inside ``threadpoolctl.threadpool_limits(1)``
    (``await micropip.install("threadpoolctl")``). Import ``scipy.io`` before entering it:
    the limit does not reach libraries imported after it starts.

Optional Dependencies
+++++++++++++++++++++

The following are not required by ``python-graphblas``, but may be needed for certain functionality
to work.

  - `pandas <https://pandas.pydata.org/>`__: required for nicer ``__repr__``
  - `matplotlib <https://matplotlib.org>`__: required for basic plotting of graphs
  - `scipy <https://scipy.org/>`__: used in ``io`` module to read/write ``scipy.sparse`` format
  - `networkx <https://networkx.org>`__: used in ``io`` module to interface with networkx graphs

GraphBLAS Fundamentals
----------------------

For a short introduction to the concepts of graph analytics using linear algebra,
read the :ref:`primer`.

For more details, the best resource for learning about GraphBLAS is `graphblas.org <https://graphblas.org>`_.
It contains information about the history, core ideas, and the full specification. It also contains links to
many videos and papers about GraphBLAS, as well as a list of implementations and language wrappers.

.. toctree::
    :maxdepth: 1
    :hidden:

    primer
    faq


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

Optional Dependencies
+++++++++++++++++++++

The following are not required by ``python-graphblas``, but may be needed for certain functionality
to work.

  - `pandas <https://pandas.pydata.org/>`__ -- required for nicer ``__repr__``
  - `matplotlib <https://matplotlib.org>`__ -- required for basic plotting of graphs
  - `scipy <https://scipy.org/>`__ -- used in ``io`` module to read/write ``scipy.sparse`` format
  - `networkx <https://networkx.org>`__ -- used in ``io`` module to interface with networkx graphs
  - `fast-matrix-market <https://github.com/alugowski/fast_matrix_market>`__ -- for faster read/write of Matrix Market files with ``gb.io.mmread`` and ``gb.io.mmwrite``

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

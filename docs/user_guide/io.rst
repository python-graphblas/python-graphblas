
Input/Output
============

1. Discuss from_values/to_values and numpy connection
2. Discuss graphblas.io methods
3. Discuss ss import/export

``graphblas.io`` contains functions for converting to and from:

.. code-block:: python

    import graphblas as gb

    # numpy arrays
    # 1-D array becomes Vector, 2-D array becomes Matrix
    A = gb.io.from_numpy(m)
    m = gb.io.to_numpy(A)

    # scipy.sparse matrices
    A = gb.io.from_scipy_sparse_matrix(m)
    m = gb.io.to_scipy_sparse_matrix(m, format='csr')

    # networkx graphs
    A = gb.io.from_networkx(g)
    g = gb.io.to_networkx(A)

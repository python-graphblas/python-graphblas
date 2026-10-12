import numpy as np

from ..dtypes import lookup_dtype


def from_networkx(G, nodelist=None, dtype=None, weight="weight", name=None):
    """Create a square adjacency Matrix from a networkx Graph.

    Parameters
    ----------
    G : nx.Graph
        Graph to convert
    nodelist : list, optional
        List of nodes in the nx.Graph. If not provided, all nodes will be used.
    dtype :
        Data type
    weight : str, default="weight"
        Weight attribute
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`

    Raises
    ------
    networkx.NetworkXError
        If the graph has no nodes, or ``nodelist`` is empty, repeats a node, or
        names one the graph does not have (as ``nx.to_scipy_sparse_array`` does).
    ValueError
        If a weight is not a bool, int (at most 64 bits), float or complex scalar.

    Notes
    -----
    When ``dtype`` is given, numpy converts the weights to it first, as
    ``np.array(weights, dtype=dtype)`` does. A weight it cannot convert raises
    numpy's TypeError, ValueError or OverflowError, and some weights convert without
    an error: None becomes nan in a float dtype, and the string ``"3"`` becomes 3.0.

    """
    import networkx as nx

    if dtype is not None:
        dtype = lookup_dtype(dtype).np_type

    # The checks of the graph and nodelist below are nx.to_scipy_sparse_array's, so
    # the errors and the node order are the same. Building the coo arrays here, not
    # through scipy, skips scipy's coo -> csr conversion and the scipy.sparse import
    # (about 100 ms on its own).
    from .. import backend
    from ..binary import plus
    from ..core.matrix import Matrix
    from ..core.udt import _builtin_dtype

    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if nlen < len(G):
            G = G.subgraph(nodelist)

    index = dict(zip(nodelist, range(nlen), strict=True))
    if weight is None and not G.is_multigraph() and (dtype is None or dtype.kind in "biufc"):
        # Every entry is 1, so read only the coordinates: row by row in nodelist
        # order, straight from the adjacency, which for an undirected graph already
        # holds both directions and each self-loop once. No tuple and no value per
        # edge makes this two to three times faster than the general path below.
        nbrs = [G._adj[u] for u in nodelist]
        rows = np.repeat(np.arange(nlen, dtype=np.uint64), [len(nbr) for nbr in nbrs])
        if rows.size:  # with no edge, the general path gives the dtype (scipy's float64)
            cols = np.fromiter((index[v] for nbr in nbrs for v in nbr), np.uint64, rows.size)
            dtype = np.int64 if dtype is None else dtype  # as the general path infers the 1s
            return Matrix.from_coo(rows, cols, 1, dtype=dtype, nrows=nlen, ncols=nlen, name=name)
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1)),
        strict=True,
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the (sub)graph
        row, col, data = (), (), ()

    values = np.array(data, dtype=dtype)
    if values.ndim != 1 or values.dtype.kind not in "biufc":
        # Weights a Matrix cannot hold: objects (ints beyond 64 bits give an object
        # array too), strings (a <U array), and sequences that all have the same
        # length, which give a 2-D array that from_coo would read as a UDT.
        # nx.to_scipy_sparse_array rejects them too, but its error depends on the scipy
        # version, and without scipy it raises ImportError.
        got = f"dtype {values.dtype}" if values.ndim == 1 else f"arrays of shape {values.shape[1:]}"
        raise ValueError(
            "edge weights must be bool, int (at most 64 bits), float or complex scalars; "
            f"attribute {weight!r} gives {got}"
        )
    if values.size == 0:
        # With no edges there is no weight to infer a dtype from, and the values are
        # float64 when dtype is not given, as nx.to_scipy_sparse_array gives them.
        return Matrix(lookup_dtype(values.dtype), nrows=nlen, ncols=nlen, name=name)
    if _builtin_dtype(values.dtype) is None:
        # A numpy type that GraphBLAS has no type for, such as float16, would make the
        # Matrix a UDT. lookup_dtype would register one, so check without it.
        raise ValueError(
            f"edge weights must have a GraphBLAS type; attribute {weight!r} gives dtype "
            f"{values.dtype}"
        )

    rows = np.array(row, dtype=np.uint64)
    cols = np.array(col, dtype=np.uint64)
    if not G.is_directed():
        # Mirror each edge into the other triangle. A self-loop is its own
        # mirror, so it is stored once with its own weight. (nx.to_scipy_sparse_array
        # mirrors every entry and sums ``wt + wt - wt`` on the diagonal instead,
        # which turns an infinite self-loop into nan, overflows a large finite one,
        # and cannot be written in an unsigned dtype.)
        offdiag = rows != cols
        rows, cols = np.concatenate([rows, cols[offdiag]]), np.concatenate([cols, rows[offdiag]])
        values = np.concatenate([values, values[offdiag]])
    if G.is_multigraph():
        # Parallel edges share a coordinate; summing them with ``plus`` matches
        # scipy's coo -> csr accumulation.
        A = Matrix.from_coo(rows, cols, values, nrows=nlen, ncols=nlen, dup_op=plus, name=name)
        if backend == "suitesparse":
            # The sums can all be the same, as when no edge is parallel to another. Then
            # store that value once (iso), as ``from_scipy_sparse`` does.
            sums = A.to_coo(rows=False, columns=False, sort=False)[2]
            if (sums[[0]] == sums).all():
                A(A.S) << sums[0]
        return A
    if backend == "suitesparse" and (values[[0]] == values).all():
        # Every edge has the same weight, as in an unweighted graph, so store the value
        # once (iso), as ``from_scipy_sparse`` does. SuiteSparse 10.5.0 and later do not
        # find that out after a build.
        A = Matrix(lookup_dtype(values.dtype), nrows=nlen, ncols=nlen, name=name)
        A.ss.build_scalar(rows, cols, values[0])
        return A
    # A simple graph has no repeated coordinate, so it needs no accumulator
    return Matrix.from_coo(rows, cols, values, nrows=nlen, ncols=nlen, name=name)


# A DiGraph spends about 350 bytes on each node, so a graph with this many nodes would
# need hundreds of terabytes. Refusing up front beats a long climb to out-of-memory,
# and it never refuses a graph that could have been built.
_MAX_ISOLATES = 2**40


# TODO: add parameters to allow different networkx classes and attribute names
def to_networkx(m, edge_attribute="weight", create_isolates=True):
    """Create a networkx DiGraph from a square adjacency Matrix.

    The nodes are the indices of the Matrix, in order.

    Parameters
    ----------
    m : Matrix
        Square adjacency Matrix
    edge_attribute : str, optional
        Name of edge attribute from values of Matrix. If None, values will be skipped.
        Default is "weight".
    create_isolates : bool, default=True
        Include the nodes with no edges (index i, where row i and column i are both empty).
        Every index then becomes a node, which costs memory for each one, so a Matrix
        with huge dimensions (often hypersparse) needs False. More than 2**40 rows raise
        ValueError unless this is False.

    Returns
    -------
    nx.DiGraph

    """
    import networkx as nx

    if m.nrows != m.ncols:
        # Row i and column i are both node i, so only a square Matrix is an adjacency matrix
        raise ValueError(f"to_networkx needs a square Matrix, not shape {m.shape}")
    if create_isolates and m.nrows > _MAX_ISOLATES:
        raise ValueError(
            f"Too many nodes to create: the Matrix has {m.nrows} rows. "
            "Use create_isolates=False to create only the nodes that have edges."
        )
    rows, cols, vals = m.to_coo()
    G = nx.DiGraph()
    # Added first and in index order, so the nodes iterate in that order and not in
    # the order the edges name them: from_networkx then gives back the same matrix
    # (without its empty rows and columns if create_isolates is False).
    if create_isolates:
        G.add_nodes_from(range(m.nrows))
    else:
        G.add_nodes_from(np.union1d(rows, cols).tolist())
    rows = rows.tolist()
    cols = cols.tolist()
    if edge_attribute is None:
        G.add_edges_from(zip(rows, cols, strict=True))
    else:
        G.add_weighted_edges_from(
            zip(rows, cols, vals.tolist(), strict=True), weight=edge_attribute
        )
    return G

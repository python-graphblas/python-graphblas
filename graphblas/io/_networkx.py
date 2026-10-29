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

    """
    import networkx as nx

    if dtype is not None:
        dtype = lookup_dtype(dtype).np_type

    # The node selection below mirrors nx.to_scipy_sparse_array so that empty
    # graphs, nodelist subsets, missing nodes, and duplicate nodes raise the
    # same errors and produce the same ordering. Building the coo arrays here
    # (instead of via a scipy sparse round-trip) skips the extra coo -> csr
    # materialization and the scipy.sparse import, which alone costs over 100ms.
    import numpy as np

    from .. import backend
    from ..binary import plus
    from ..core.matrix import Matrix

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
    if dtype is None and values.dtype == np.int32:
        # ``data`` is a Python sequence, and from_coo widens an int32 dtype
        # INFERRED from sequence input to int64 (values_to_numpy_buffer): python
        # ints infer int32 on win64 with numpy < 2, and np.int32 scalar weights
        # infer it on every platform. Widen the same way so from_networkx and
        # Matrix.from_coo given the same weights agree on the result dtype.
        # An explicit dtype= is applied above and never reaches this branch.
        values = values.astype(np.int64)
    if values.ndim != 1 or values.dtype.kind not in "biufc":
        # Two kinds of weight land here: non-numeric attributes (object arrays,
        # which ints beyond 64 bits also give, and e.g. all-string weights, which
        # infer a <U dtype rather than object), and sequence-valued attributes of
        # uniform length, which infer a 2-D numeric array that from_coo would
        # otherwise accept as a UDT. nx.to_scipy_sparse_array rejects the same
        # weights, but asking it made the error depend on scipy: ValueError since
        # 1.15, TypeError or a NetworkXError blaming the sparse format before, and
        # ImportError without scipy installed.
        got = f"dtype {values.dtype}" if values.ndim == 1 else f"arrays of shape {values.shape[1:]}"
        raise ValueError(
            "edge weights must be bool, int (at most 64 bits), float or complex scalars; "
            f"attribute {weight!r} gives {got}"
        )
    if values.size == 0:
        # An empty graph has no data to infer a dtype from; scipy defaults an
        # empty coo array to float64, so match that when dtype is unset.
        return Matrix(lookup_dtype(values.dtype), nrows=nlen, ncols=nlen, name=name)

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
    dup_op = None
    if G.is_multigraph():
        # Parallel edges share a coordinate; summing them with ``plus`` matches
        # scipy's coo -> csr accumulation. A simple graph has no repeated
        # coordinate, so it needs no accumulator.
        dup_op = plus
    elif backend == "suitesparse" and (values[[0]] == values).all():
        # Every edge carries the same weight, which is the whole story for an
        # unweighted graph. Store one value rather than one per entry, as
        # ``from_scipy_sparse`` does for the same data. SuiteSparse used to
        # notice this after a build on its own; 10.5.0 stopped ("no longer do
        # a post-iso check"), so ask for it outright. A multigraph is excluded
        # because ``plus`` can make uniform inputs non-uniform.
        A = Matrix(lookup_dtype(values.dtype), nrows=nlen, ncols=nlen, name=name)
        A.ss.build_scalar(rows, cols, values[0])
        return A
    return Matrix.from_coo(rows, cols, values, nrows=nlen, ncols=nlen, dup_op=dup_op, name=name)


# TODO: add parameters to allow different networkx classes and attribute names
def to_networkx(m, edge_attribute="weight"):
    """Create a networkx DiGraph from a square adjacency Matrix.

    Every row is a node, numbered by its index, including rows with no entries.

    Parameters
    ----------
    m : Matrix
        Square adjacency Matrix
    edge_attribute : str, optional
        Name of edge attribute from values of Matrix. If None, values will be skipped.
        Default is "weight".

    Returns
    -------
    nx.DiGraph

    """
    import networkx as nx

    rows, cols, vals = m.to_coo()
    rows = rows.tolist()
    cols = cols.tolist()
    G = nx.DiGraph()
    # Added first and in order, so a node with no edges is kept and the nodes
    # iterate in index order: from_networkx then gives back the same matrix.
    G.add_nodes_from(range(m.nrows))
    if edge_attribute is None:
        G.add_edges_from(zip(rows, cols, strict=True))
    else:
        G.add_weighted_edges_from(
            zip(rows, cols, vals.tolist(), strict=True), weight=edge_attribute
        )
    return G

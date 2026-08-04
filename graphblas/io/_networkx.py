from ..dtypes import lookup_dtype
from ._scipy import from_scipy_sparse


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
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1)),
        strict=True,
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the (sub)graph
        row, col, data = (), (), ()

    if G.is_directed():
        rows, cols, vals = row, col, data
        # A multigraph can have parallel edges (duplicate ``(u, v)``); summing
        # them with ``plus`` matches scipy's coo -> csr accumulation. A simple
        # graph has no duplicate coordinates, so ``dup_op=None`` is exact and
        # skips the accumulator.
        dup_op = plus if G.is_multigraph() else None
    else:
        # Symmetrize: mirror off-diagonal entries. Self-loops would be double
        # counted, so subtract the diagonal contribution once, matching
        # nx.to_scipy_sparse_array. dup_op=plus then sums the diagonal triple
        # (wt + wt - wt) back to wt. For a multigraph, plus also sums parallel
        # edges (duplicate coordinates) the same way scipy's coo -> csr does;
        # for a simple graph off-diagonal entries are unique so plus is a no-op
        # there.
        d = data + data
        r = row + col
        c = col + row
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops), strict=True)
            d += diag_data
            r += diag_index
            c += diag_index
        rows, cols, vals = r, c, d
        dup_op = plus

    values = np.array(vals, dtype=dtype)
    if dtype is None and values.dtype == np.int32:  # pragma: no cover (win64 numpy < 2)
        # numpy < 2 infers the platform C long for a sequence of Python ints, which
        # is 32-bit on Windows. values_to_numpy_buffer widens the same way for
        # non-numpy input, so this keeps from_networkx agreeing with
        # Matrix.from_coo on INT64 for an unweighted graph on every platform.
        values = values.astype(np.int64)
    if values.ndim != 1 or values.dtype.kind not in "biufc":
        # Defer to scipy so the error matches the previous behavior exactly.
        # Two kinds of weight land here: non-numeric attributes (object arrays,
        # but also e.g. all-string weights, which infer a <U dtype rather than
        # object), and sequence-valued attributes of uniform length, which infer
        # a 2-D numeric array that from_coo would otherwise accept as a UDT.
        return _from_networkx_via_scipy(G, nodelist, dtype, weight, name)
    if values.size == 0:
        # An empty graph has no data to infer a dtype from; scipy defaults an
        # empty coo array to float64, so match that when dtype is unset.
        return Matrix(lookup_dtype(values.dtype), nrows=nlen, ncols=nlen, name=name)

    rows = np.array(rows, dtype=np.uint64)
    cols = np.array(cols, dtype=np.uint64)
    return Matrix.from_coo(rows, cols, values, nrows=nlen, ncols=nlen, dup_op=dup_op, name=name)


def _from_networkx_via_scipy(G, nodelist, dtype, weight, name):
    """Fallback path: convert through a scipy sparse array.

    ``dtype`` is already normalized to a numpy type (or None) by the caller.
    """
    import networkx as nx

    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
    return from_scipy_sparse(A, name=name)


# TODO: add parameters to allow different networkx classes and attribute names
def to_networkx(m, edge_attribute="weight"):
    """Create a networkx DiGraph from a square adjacency Matrix.

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
    if edge_attribute is None:
        G.add_edges_from(zip(rows, cols, strict=True))
    else:
        G.add_weighted_edges_from(
            zip(rows, cols, vals.tolist(), strict=True), weight=edge_attribute
        )
    return G

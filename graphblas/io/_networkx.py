import numpy as np

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
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
    return from_scipy_sparse(A, name=name)


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

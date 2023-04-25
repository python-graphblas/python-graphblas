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
        G.add_edges_from(zip(rows, cols))
    else:
        G.add_weighted_edges_from(zip(rows, cols, vals.tolist()), weight=edge_attribute)
    return G

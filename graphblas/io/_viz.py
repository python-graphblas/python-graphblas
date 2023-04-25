from warnings import warn


def draw(m):  # pragma: no cover (deprecated)
    """Draw a square adjacency Matrix as a graph.

    Requires `networkx <https://networkx.org/>`_ and
    `matplotlib <https://matplotlib.org/>`_ to be installed.

    Example output:

    .. image:: /_static/img/draw-example.png
    """
    from .. import viz

    warn(
        "`graphblas.io.draw` is deprecated; it has been moved to `graphblas.viz.draw`",
        DeprecationWarning,
        stacklevel=2,
    )
    viz.draw(m)

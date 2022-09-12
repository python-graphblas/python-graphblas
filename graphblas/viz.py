import numpy as np

from .io import to_networkx, to_scipy_sparse
from .matrix import Matrix, TransposedMatrix

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception:
    mpl = None
    plt = None


def draw(m):  # pragma: no cover
    """Draw a square adjacency Matrix as a graph.

    Requires `networkx <https://networkx.org/>`_ and
    `matplotlib <https://matplotlib.org/>`_ to be installed.

    Example output:

    .. image:: /_static/img/draw-example.png
    """
    try:
        import networkx as nx
    except ImportError:
        print("`draw` requires networkx to be installed")
        return
    if plt is None:
        print("`draw` requires matplotlib to be installed")
        return

    if not isinstance(m, (Matrix, TransposedMatrix)):
        print(f"Can only draw a Matrix, not {type(m)}")
        return

    g = to_networkx(m)
    pos = nx.spring_layout(g)
    edge_labels = {(i, j): d["weight"] for i, j, d in g.edges(data=True)}
    nx.draw_networkx(g, pos, node_color="red", node_size=500)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()


def spy(M, xxx=True, *, show=True, figure=None, axes=None, figsize=None, **kwargs):
    if mpl is None:
        raise ImportError("`spy` requires matplotlib to be installed")
    A = to_scipy_sparse(M, "coo")
    if show:
        plt.ion()
        plt.show()
    if axes is None:
        if figure is None:
            fig = mpl.figure.Figure(figsize=figsize)
        axes = fig.subplots()
    if kwargs.get("markersize") is None:
        # Make the square markers "fill" their space
        markersize = min(axes.bbox.width / A.shape[1], axes.bbox.height / A.shape[0]) * (
            72 / fig.dpi
        )
        kwargs["markersize"] = max(0.002, markersize)
    axes.spy(A, **kwargs)
    # Fix offsets
    if xxx:  # TODO: need name for option
        axes.figure.draw_without_rendering()  # Generates tick labels
        axes.set_xticks(axes.get_xticks()[1:] - 0.5, axes.get_xticklabels()[1:])
        axes.set_yticks(axes.get_yticks()[1:] - 0.5, axes.get_yticklabels()[1:])
    return axes.figure


def datashade(M, aggs="count", *, width=None, height=None, opts_kwargs=None, **kwargs):
    try:
        import hvplot.pandas  # noqa
    except ImportError:
        raise ImportError("`datashade` requires hvplot to be installed")
    try:
        import datashader  # noqa
    except ImportError:
        raise ImportError("`datashade` requires datashader to be installed")
    import bokeh as bk
    import pandas as pd

    if "df" not in kwargs:
        rows, cols, vals = M.to_values()
        max_int = np.iinfo(np.int64).max
        if M.nrows > max_int and rows.max() > max_int:
            rows = rows.astype(np.float64)
        else:
            rows = rows.astype(np.int64)
        if M.ncols > max_int and cols.max() > max_int:
            cols = cols.astype(np.float64)
        else:
            cols = cols.astype(np.int64)
        df = pd.DataFrame({"row": rows, "col": cols, "val": vals})
    else:
        df = kwargs.pop("df")

    aggregator = aggs
    if width is None and height is None:
        width = 500

    kwds = dict(  # noqa
        x="col",
        y="row",
        c="val",
        aggregator=aggregator,
        frame_width=width,
        frame_height=height,
        cmap="fire",
        cnorm="eq_hist",
        xlim=(0, M.ncols),
        ylim=(0, M.nrows),
        rasterize=True,
        flip_yaxis=True,
        hover=True,
        xlabel="",
        ylabel="",
        data_aspect=1,
        x_sampling=1,
        y_sampling=1,
        xaxis="top",
        xformatter="%d",
        yformatter="%d",
        rot=60,
    )
    kwds.update(kwargs)
    im = df.hvplot.scatter(**kwds)
    if opts_kwargs is None:
        opts_kwargs = {}
    if "bgcolor" not in opts_kwargs:
        opts_kwargs["bgcolor"] = "black"
    if "tools" not in opts_kwargs:
        # Format rows and columns as integers
        hover = bk.models.HoverTool(
            tooltips=[("col", "$x{i}"), ("row", "$y{i}"), (aggregator, "@image")],
            formatters={"@col": "printf", "@row": "printf"},
        )
        opts_kwargs["tools"] = [hover]
    return im.opts(**opts_kwargs)

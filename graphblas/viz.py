from importlib import import_module as _import_module

from .core.matrix import Matrix as _Matrix
from .core.matrix import TransposedMatrix as _TransposedMatrix
from .core.utils import output_type as _output_type
from .io import to_networkx, to_scipy_sparse

_LAZY_IMPORTS = {
    "bk": "bokeh",
    "ds": "datashader",
    "hv": "holoviews",
    "hp": "hvplot.pandas",
    "mpl": "matplotlib",
    "plt": "matplotlib.pyplot",
    "np": "numpy",
    "nx": "networkx",
    "pd": "pandas",
    "ss": "scipy.sparse",
}


def _get_imports(names, within):
    is_string = isinstance(names, str)
    if is_string:
        names = [names]
    rv = []
    for name in names:
        if name not in _LAZY_IMPORTS:  # pragma: no cover (safety)
            raise KeyError(f"Unknown library to import: {name}")
        if name in globals():
            val = globals()[name]
        else:
            try:
                val = _import_module(_LAZY_IMPORTS[name])
            except ImportError:
                modname = _LAZY_IMPORTS[name].split(".")[0]
                raise ImportError(f"`{within}` requires {modname} to be installed") from None
            globals()[name] = val
        rv.append(val)
    if is_string:
        return rv[0]
    return rv


def draw(m):
    """Draw a square adjacency Matrix as a graph.

    Requires `networkx <https://networkx.org/>`_ and
    `matplotlib <https://matplotlib.org/>`_ to be installed.

    Reciprocal directed edges (``u -> v`` and ``v -> u``) are drawn as curves so
    both arrows and both edge weights stay visible; all other edges are straight.
    Curving them needs networkx 3.3 or newer; with older versions every edge is
    drawn straight.

    Example output:

    .. image:: /_static/img/draw-example.png
    """
    nx, plt = _get_imports(["nx", "plt"], "draw")
    typ = _output_type(m)
    if typ is not _Matrix and typ is not _TransposedMatrix:
        raise TypeError(f"Can only draw a Matrix, not {type(m)}")

    g = to_networkx(m)
    pos = nx.spring_layout(g)
    node_size = 500
    nx.draw_networkx_nodes(g, pos, node_color="red", node_size=node_size)
    nx.draw_networkx_labels(g, pos)

    # A reciprocal pair (u -> v and v -> u) drawn as two straight lines coincides,
    # hiding one edge's weight (python-graphblas #474).  Curving both edges makes
    # each bend toward its own side, so both arrows and both labels stay visible
    # and attributable.  Self-loops (u == v) are not reciprocal; leave them straight.
    #
    # networkx only learned to place edge labels along a curve in 3.3, and we
    # support >=2.8, so fall back to the previous straight rendering without it.
    # Curving the edges but not the labels would be worse than not curving at all:
    # the labels would sit back on the shared chord midpoint, which is the overlap
    # #474 is about, and they would no longer track their arrows.  Check the
    # parameter rather than pin a version.
    import inspect

    if "connectionstyle" in inspect.signature(nx.draw_networkx_edge_labels).parameters:
        curved = {(u, v) for u, v in g.edges if u != v and g.has_edge(v, u)}
    else:
        curved = set()
    straight = [e for e in g.edges if e not in curved]
    connectionstyle = "arc3,rad=0.1"

    def _edge_labels(edges):
        return {(u, v): g[u][v]["weight"] for u, v in edges}

    if straight:
        nx.draw_networkx_edges(g, pos, edgelist=straight, node_size=node_size)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=_edge_labels(straight))
    if curved:
        curved = list(curved)
        nx.draw_networkx_edges(
            g, pos, edgelist=curved, node_size=node_size, connectionstyle=connectionstyle
        )
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=_edge_labels(curved), connectionstyle=connectionstyle
        )
    plt.show()


def spy(M, *, centered=False, show=True, figure=None, axes=None, figsize=None, **kwargs):
    """Plot the sparsity pattern of a Matrix using ``matplotlib.spy``.

    See:
    - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.spy.html
    - https://matplotlib.org/stable/gallery/images_contours_and_fields/spy_demos.html

    By default, this function automatically calculates markersize to properly tile
    the sparsity pattern.  That is, the square plotted for a visible element abuts
    adjacent element.

    See Also
    --------
    datashade

    """
    mpl, plt, _ss = _get_imports(["mpl", "plt", "ss"], "spy")
    A = to_scipy_sparse(M, "coo")
    if show:
        plt.ion()
        plt.show()
    if axes is None:
        if figure is None:
            figure = mpl.figure.Figure(figsize=figsize)
        axes = figure.subplots()
    if kwargs.get("markersize") is None:
        # Make the square markers "fill" their space
        markersize = min(axes.bbox.width / A.shape[1], axes.bbox.height / A.shape[0])
        kwargs["markersize"] = max(0.002, markersize * 72 / axes.figure.dpi)
    axes.spy(A, **kwargs)
    # Fix offsets
    if not centered:
        axes.figure.draw_without_rendering()  # Generates tick labels
        axes.set_xticks(axes.get_xticks()[1:-1] - 0.5, axes.get_xticklabels()[1:-1])
        axes.set_yticks(axes.get_yticks()[1:-1] - 0.5, axes.get_yticklabels()[1:-1])
    return axes.figure


def _matrix_to_dataframe(M):
    """Build the ``(row, col, val)`` DataFrame that ``datashade`` rasterizes.

    Factored out of ``datashade`` so the coordinate convention can be checked
    without rendering an interactive plot (see ``_cell_centered_limits``).
    """
    np, pd = _get_imports(["np", "pd"], "datashade")
    rows, cols, vals = M.to_coo()
    max_int = np.iinfo(np.int64).max
    if M.nrows > max_int and rows.max() > max_int:
        rows = rows.astype(np.float64)
    else:
        rows = rows.astype(np.int64)
    if M.ncols > max_int and cols.max() > max_int:
        cols = cols.astype(np.float64)
    else:
        cols = cols.astype(np.int64)
    return pd.DataFrame({"row": rows, "col": cols, "val": vals})


def _cell_centered_limits(M):
    """Axis limits that center each element on its integer index, like ``spy``.

    datashader bins points into pixels by ``x_range``/``y_range``.  With limits
    ``(0, N)`` the pixel for index ``k`` spans ``[k, k+1)``, so an element lands
    half a cell to the lower-right of the tick labeled ``k``.  Offsetting the
    limits by half a cell makes the pixel for index ``k`` span ``[k-0.5, k+0.5)``,
    centered on tick ``k`` and matching what ``spy`` draws (python-graphblas #473).
    """
    return (-0.5, M.ncols - 0.5), (-0.5, M.nrows - 0.5)


def datashade(M, agg="count", *, width=None, height=None, opts_kwargs=None, **kwargs):
    """Interactive plot of the sparsity pattern of a Matrix using hvplot and datashader.

    The ``datashader`` library rasterizes large data into a 2d grid of pixels.  Each pixel
    may contain multiple data points, which are combined by an aggregator (``agg="count"``).
    Common aggregators are "count", "sum", "mean", "min", and "max".  See full list here:
    - https://datashader.org/api.html#reductions

    Multiple aggregators may be given to create a grid of linked plots.  For example,

    >>> datashade(A, agg=[["count", "sum"], ["min", "max"]])

    creates a 2x2 grid of plots.  They share axes, so when you pan or zoom on one plot,
    the other plots pan and zoom as well.

    You can combine multiple datashade plots together:

    >>> datashade(A) + datashade(B)

    will show two plots side by side.

    Learn more about customization here:
    - https://hvplot.holoviz.org/user_guide/Customization.html

    See Also
    --------
    spy

    """
    bk, hv, _hp, _ds = _get_imports(["bk", "hv", "hp", "ds"], "datashade")
    if "df" not in kwargs:
        df = _matrix_to_dataframe(M)
    else:
        df = kwargs.pop("df")

    if width is None and height is None:
        width = 500

    if isinstance(agg, list):
        if not agg:
            return
        kwargs["M"] = M
        kwargs["df"] = df
        kwargs["height"] = height
        kwargs["opts_kwargs"] = opts_kwargs
        if any(isinstance(x, list) for x in agg):
            ncols = max(len(x) for x in agg if isinstance(x, list))
            agg = [x if isinstance(x, list) else [x] for x in agg]
        else:
            ncols = len(agg)
            agg = [agg]
        if width is not None:
            width //= ncols
        kwargs["width"] = width
        images = []
        for i, row in enumerate(agg):
            kwargs["_row"] = i
            image_row = []
            for j, aggregator in enumerate(row):
                if aggregator is None:
                    image_row.append(hv.Empty())
                    continue
                kwargs["_col"] = j
                kwargs["agg"] = aggregator
                image_row.append(datashade(**kwargs))
            while len(image_row) < ncols:
                image_row.append(hv.Empty())
            images.extend(image_row)
        return hv.Layout(images).cols(ncols)

    xlim, ylim = _cell_centered_limits(M)
    kwds = {
        "x": "col",
        "y": "row",
        "c": "val",
        "aggregator": agg,
        "frame_width": width,
        "frame_height": height,
        "cmap": "fire",
        "cnorm": "eq_hist",
        "xlim": xlim,
        "ylim": ylim,
        "rasterize": True,
        "flip_yaxis": True,
        "hover": True,
        "xlabel": "",
        "ylabel": "",
        "data_aspect": 1,
        "x_sampling": 1,
        "y_sampling": 1,
        "xaxis": "top",
        "xformatter": "%d",
        "yformatter": "%d",
        "rot": 60,
    }
    # Only show axes on outer-most plots
    if kwargs.pop("_col", 0) != 0:
        kwds["yaxis"] = None
    if kwargs.pop("_row", 0) != 0:
        kwds["xaxis"] = None

    kwds.update(kwargs)
    im = df.hvplot.scatter(**kwds)
    if opts_kwargs is None:
        opts_kwargs = {}
    if "bgcolor" not in opts_kwargs:
        opts_kwargs["bgcolor"] = "black"
    if "tools" not in opts_kwargs:
        # Format rows and columns as integers
        hover = bk.models.HoverTool(
            tooltips=[("row", "$y{i}"), ("col", "$x{i}"), (agg, "@image")],
            formatters={"@col": "printf", "@row": "printf"},
        )
        opts_kwargs["tools"] = [hover]
    return im.opts(**opts_kwargs)

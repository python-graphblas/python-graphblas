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
            finally:
                globals()[name] = val
        rv.append(val)
    if is_string:
        return rv[0]
    return rv


def draw(m):  # pragma: no cover
    """Draw a square adjacency Matrix as a graph.

    Requires `networkx <https://networkx.org/>`_ and
    `matplotlib <https://matplotlib.org/>`_ to be installed.

    Example output:

    .. image:: /_static/img/draw-example.png
    """
    nx, plt = _get_imports(["nx", "plt"], "draw")
    typ = _output_type(m)
    if typ is not _Matrix and typ is not _TransposedMatrix:
        raise TypeError(f"Can only draw a Matrix, not {type(m)}")

    g = to_networkx(m)
    pos = nx.spring_layout(g)
    edge_labels = {(i, j): d["weight"] for i, j, d in g.edges(data=True)}
    nx.draw_networkx(g, pos, node_color="red", node_size=500)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()


def spy(M, *, centered=False, show=True, figure=None, axes=None, figsize=None, **kwargs):
    """Plot the sparsity pattern of a Matrix using `matplotlib.spy`.

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
    mpl, plt, ss = _get_imports(["mpl", "plt", "ss"], "spy")
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
        markersize = min(axes.bbox.width / A.shape[1], axes.bbox.height / A.shape[0])
        kwargs["markersize"] = max(0.002, markersize * 72 / fig.dpi)
    axes.spy(A, **kwargs)
    # Fix offsets
    if not centered:
        axes.figure.draw_without_rendering()  # Generates tick labels
        axes.set_xticks(axes.get_xticks()[1:-1] - 0.5, axes.get_xticklabels()[1:-1])
        axes.set_yticks(axes.get_yticks()[1:-1] - 0.5, axes.get_yticklabels()[1:-1])
    return axes.figure


def datashade(M, agg="count", *, width=None, height=None, opts_kwargs=None, **kwargs):
    """Interactive plot of the sparsity pattern of a Matrix using hvplot and datashader.

    The `datashader` library rasterizes large data into a 2d grid of pixels.  Each pixel
    may contain multiple data points, which are combined by an aggregator (`agg="count"`).
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
    np, pd, bk, hv, hp, ds = _get_imports(["np", "pd", "bk", "hv", "hp", "ds"], "datashade")
    if "df" not in kwargs:
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
        df = pd.DataFrame({"row": rows, "col": cols, "val": vals})
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

    kwds = {
        "x": "col",
        "y": "row",
        "c": "val",
        "aggregator": agg,
        "frame_width": width,
        "frame_height": height,
        "cmap": "fire",
        "cnorm": "eq_hist",
        "xlim": (0, M.ncols),
        "ylim": (0, M.nrows),
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

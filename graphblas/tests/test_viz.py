"""Smoke tests for graphblas.viz.

The viz module is optional-dependency heavy (matplotlib, networkx, scipy for
``spy``/``draw``; datashader + holoviews + hvplot + bokeh + pandas for
``datashade``). These tests only check that each public function runs end to end
under the headless Agg backend and populates a figure/returns an object; they do
not assert on pixel output. Anything missing is skipped, not failed, so a
minimal-dependency CI run sees clean skips.
"""

import math

import pytest

from graphblas import Matrix, Vector, viz

# Skip the whole module if matplotlib is absent (draw and spy both need it).
# Set the backend to Agg before pyplot is imported so no display is required.
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _close_figures():
    # Close every figure after each test to avoid matplotlib's
    # "More than 20 figures have been opened" warning (which the project's
    # ``filterwarnings = error`` config would turn into a failure).
    yield
    plt.close("all")


def square_matrix():
    # Small square adjacency matrix with distinct weights.
    return Matrix.from_coo([0, 0, 1, 2], [1, 2, 2, 0], [1.0, 2.0, 3.0, 4.0], nrows=3, ncols=3)


def test_spy_default():
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    fig = viz.spy(A, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig.axes, "spy should populate at least one Axes"
    # matplotlib's Axes.spy draws the pattern as a single markered Line2D.
    assert fig.axes[0].lines, "spy should plot the sparsity markers"


def test_spy_centered():
    # centered=True skips the tick-offset fixup branch.
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    fig = viz.spy(A, show=False, centered=True)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig.axes[0].lines


def test_spy_with_axes():
    # Passing an explicit Axes exercises the ``axes is not None`` branch,
    # including the auto-markersize path (which once raised NameError here).
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    fig = mpl.figure.Figure()
    axes = fig.subplots()
    result = viz.spy(A, show=False, axes=axes)
    assert result is fig
    assert axes.lines


def test_spy_with_figure():
    # Passing an explicit Figure (no Axes) once raised NameError; spy should
    # create the Axes on the given figure and return that same figure.
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    fig = mpl.figure.Figure()
    result = viz.spy(A, show=False, figure=fig)
    assert result is fig
    assert fig.axes
    assert fig.axes[0].lines


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_draw():
    # draw() renders onto the current pyplot Axes via networkx and calls
    # plt.show(); on Agg that show() emits the non-interactive UserWarning,
    # which we ignore here.
    pytest.importorskip("networkx")
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    viz.draw(A)
    axes = plt.gcf().get_axes()
    assert axes, "draw should populate the current figure"
    ax = axes[0]
    # Nodes render as patches/collections and labels as texts.
    assert ax.collections or ax.patches
    assert ax.texts, "draw should render node/edge labels"


def test_draw_rejects_non_matrix():
    pytest.importorskip("networkx")
    v = Vector.from_coo([0, 1, 2], [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="Can only draw a Matrix"):
        viz.draw(v)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_draw_reciprocal_edges_both_labels_visible():
    # Regression for gh-474: reciprocal directed edges (0->1 and 1->0) used to be
    # drawn as coincident straight lines, so one weight hid the other. draw() now
    # curves reciprocal pairs; both weights must appear at distinct positions.
    pytest.importorskip("networkx")
    pytest.importorskip("scipy.sparse")
    M = Matrix.from_coo([0, 1], [1, 0], [10, 20], nrows=2, ncols=2)
    viz.draw(M)
    ax = plt.gcf().get_axes()[0]

    weight_labels = [t for t in ax.texts if t.get_text() in {"10", "20"}]
    assert {t.get_text() for t in weight_labels} == {"10", "20"}, "both weights must be drawn"
    assert len(weight_labels) == 2

    # Old behavior placed both labels on the shared straight-line midpoint.
    # networkx returns two anchors there that differ only by floating-point
    # noise (order 1e-6), so an exact ``!=`` comparison passes even when the
    # labels sit on top of each other.  Require a separation that is a real
    # fraction of the distance between the two nodes instead.
    node_positions = [t.get_position() for t in ax.texts if t.get_text() in {"0", "1"}]
    assert len(node_positions) == 2
    edge_length = math.dist(*node_positions)
    separation = math.dist(*(t.get_position() for t in weight_labels))
    assert separation > 0.001 * edge_length, "reciprocal edge labels still overlap"


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_draw_without_networkx_curved_label_support(monkeypatch):
    # draw_networkx_edge_labels gained connectionstyle in networkx 3.3, and the
    # project supports >=2.8.  Standing in a pre-3.3 signature must not raise; the
    # gh-474 curving is skipped and every edge renders straight, as it did before.
    nx = pytest.importorskip("networkx")
    pytest.importorskip("scipy.sparse")
    real = nx.draw_networkx_edge_labels

    def pre_33_draw_networkx_edge_labels(g, pos, edge_labels=None, **kwargs):
        if "connectionstyle" in kwargs:
            raise TypeError(
                "draw_networkx_edge_labels() got an unexpected keyword argument "
                "'connectionstyle'"
            )
        return real(g, pos, edge_labels=edge_labels, **kwargs)

    monkeypatch.setattr(nx, "draw_networkx_edge_labels", pre_33_draw_networkx_edge_labels)
    M = Matrix.from_coo([0, 1], [1, 0], [10, 20], nrows=2, ncols=2)
    viz.draw(M)
    ax = plt.gcf().get_axes()[0]
    assert {t.get_text() for t in ax.texts if t.get_text() in {"10", "20"}} == {"10", "20"}


def _import_datashade_deps():
    for name in ("numpy", "pandas", "datashader", "holoviews", "hvplot", "bokeh"):
        pytest.importorskip(name)


def test_datashade_single():
    _import_datashade_deps()
    import holoviews as hv

    A = square_matrix()
    obj = viz.datashade(A)
    assert obj is not None
    assert isinstance(obj, hv.core.dimension.Dimensioned)


def test_datashade_agg_list():
    # A flat list of aggregators produces one row of linked plots.
    _import_datashade_deps()
    import holoviews as hv

    A = square_matrix()
    layout = viz.datashade(A, agg=["count", "sum"])
    assert isinstance(layout, hv.Layout)


def test_datashade_agg_grid():
    # A list-of-lists produces a 2d grid of linked plots.
    _import_datashade_deps()
    import holoviews as hv

    A = square_matrix()
    layout = viz.datashade(A, agg=[["count", "sum"], ["min", "max"]])
    assert isinstance(layout, hv.Layout)


def test_datashade_empty_agg():
    # An empty aggregator list is a no-op that returns None.
    _import_datashade_deps()
    A = square_matrix()
    assert viz.datashade(A, agg=[]) is None


def test_datashade_positions_match_spy():
    # Regression for gh-473: element (row=r, col=c) must render centered on the
    # integer tick pair (col, row), the same convention ``spy`` uses.  We check
    # the datashader aggregation directly (no display) over the limits the
    # interactive path uses, at one pixel per matrix cell.
    _import_datashade_deps()
    import datashader as ds
    import numpy as np

    # Non-square (3x4) with distinct row/col so a row<->col swap would show.
    M = Matrix.from_coo([0, 0, 2], [1, 3, 3], [1.0, 1.0, 1.0], nrows=3, ncols=4)
    df = viz._matrix_to_dataframe(M)
    xlim, ylim = viz._cell_centered_limits(M)
    assert xlim == (-0.5, M.ncols - 0.5)
    assert ylim == (-0.5, M.nrows - 0.5)

    canvas = ds.Canvas(plot_width=M.ncols, plot_height=M.nrows, x_range=xlim, y_range=ylim)
    agg = canvas.points(df, "col", "row", ds.count())

    # Pixel centers land on integers, so ticks label the cells they sit on.
    assert agg.coords["col"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert agg.coords["row"].values.tolist() == [0.0, 1.0, 2.0]

    # Counts are nonzero exactly at the (row, col) indices of the elements.
    xs = agg.coords["col"].values
    ys = agg.coords["row"].values
    nonzero = {(round(float(ys[i])), round(float(xs[j]))) for i, j in np.argwhere(agg.values > 0)}
    assert nonzero == {(0, 1), (0, 3), (2, 3)}

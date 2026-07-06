"""Smoke tests for graphblas.viz.

The viz module is optional-dependency heavy (matplotlib, networkx, scipy for
``spy``/``draw``; datashader + holoviews + hvplot + bokeh + pandas for
``datashade``). These tests only check that each public function runs end to end
under the headless Agg backend and populates a figure/returns an object; they do
not assert on pixel output. Anything missing is skipped, not failed, so a
minimal-dependency CI run sees clean skips.
"""

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
    # Passing an explicit Axes exercises the ``axes is not None`` branch.
    # markersize must be supplied here: the auto-markersize path references a
    # ``fig`` local that only exists when spy creates the figure itself.
    pytest.importorskip("scipy.sparse")
    A = square_matrix()
    fig = mpl.figure.Figure()
    axes = fig.subplots()
    result = viz.spy(A, show=False, axes=axes, markersize=5)
    assert result is fig
    assert axes.lines


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

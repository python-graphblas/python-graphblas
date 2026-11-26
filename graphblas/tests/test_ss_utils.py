import ctypes
import gc
import os
import re
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import Matrix, Vector, backend
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.utils import _distinct_buffers, values_to_numpy_buffer
from graphblas.exceptions import InvalidValue

if backend != "suitesparse":
    pytest.skip("gb.ss and A.ss only available with suitesparse backend", allow_module_level=True)

try:
    from suitesparse_graphblas.utils import can_unclaim_buffer, empty
except ImportError:  # pragma: no cover (import)
    # suitesparse-graphblas < 10.5.1.1
    can_unclaim_buffer = empty = None


# SuiteSparse:GraphBLAS 10.5.0 and later do not check after a build whether
# every value is the same (its changelog: "GrB_Matrix_build and GrB_Vector_build:
# no longer do a post-iso check"), so ``from_coo`` with equal values is not iso
# there, as it is on earlier versions. ``ss.build_scalar`` builds iso on every
# version, in the same storage formats that the tests below select.
def _iso_vector(indices, value, size):
    v = Vector(int, size)
    v.ss.build_scalar(indices, value)
    return v


def _iso_matrix(rows, cols, value, nrows, ncols):
    A = Matrix(int, nrows, ncols)
    A.ss.build_scalar(rows, cols, value)
    return A


@pytest.mark.parametrize("do_iso", [False, True])
def test_vector_head(do_iso):
    v0 = Vector(int, 5)
    if do_iso:
        values1 = values2 = values3 = [1, 1, 1]
    else:
        values1 = [10, 20, 30]
        values2 = [2, 4, 6]
        values3 = [1, 2, 3]
    if do_iso:
        v1 = _iso_vector([0, 1, 2], values1[0], 3)  # full
        v2 = _iso_vector([1, 3, 5], values2[0], 6)  # bitmap
        v3 = _iso_vector([100, 200, 300], values3[0], 301)  # sparse
    else:
        v1 = Vector.from_coo([0, 1, 2], values1)  # full
        v2 = Vector.from_coo([1, 3, 5], values2)  # bitmap
        v3 = Vector.from_coo([100, 200, 300], values3)  # sparse
    assert v1.ss.export()["format"] == "full"
    assert v2.ss.export()["format"] == "bitmap"
    assert v3.ss.export()["format"] == "sparse"
    assert v1.ss.is_iso is do_iso
    assert v2.ss.is_iso is do_iso
    assert v3.ss.is_iso is do_iso
    for dtype in [None, np.float64]:
        expected_dtype = np.int64 if dtype is None else dtype
        for _ in range(2):
            indices, vals = v0.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [])
            assert_array_equal(vals, [])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v1.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [0, 1])
            assert_array_equal(vals, values1[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [1, 3])
            assert_array_equal(vals, values2[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v3.ss.head(2, sort=False, dtype=dtype)
            assert indices.size == vals.size == 2
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype

            indices, vals = v3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(indices, [100, 200])
            assert_array_equal(vals, values3[:2])
            assert indices.dtype == np.uint64
            assert vals.dtype == expected_dtype


@pytest.mark.parametrize("do_iso", [False, True])
def test_matrix_head(do_iso):
    A0 = Matrix(int, 5, 5)
    if do_iso:
        values1 = [1, 1, 1, 1]
        values2 = values3 = values4 = [1, 1, 1]
    else:
        values1 = [1, 2, 3, 4]
        values2 = [1, 2, 4]
        values3 = values4 = [1, 2, 3]

    if do_iso:
        A1 = _iso_matrix([0, 0, 1, 1], [0, 1, 0, 1], values1[0], 2, 2)  # fullr
        A2 = _iso_matrix([0, 0, 1], [0, 1, 1], values2[0], 2, 2)  # Bitmap
        A3 = _iso_matrix([5, 5, 10], [4, 5, 10], values3[0], 11, 11)  # CSR
        A4 = _iso_matrix([500, 500, 1000], [400, 500, 1000], values4[0], 1001, 1001)  # HyperCSR
    else:
        A1 = Matrix.from_coo([0, 0, 1, 1], [0, 1, 0, 1], values1)  # fullr
        A2 = Matrix.from_coo([0, 0, 1], [0, 1, 1], values2)  # Bitmap
        A3 = Matrix.from_coo([5, 5, 10], [4, 5, 10], values3)  # CSR
        A4 = Matrix.from_coo([500, 500, 1000], [400, 500, 1000], values4)  # HyperCSR
    d = A1.ss.export(raw=True)
    assert d["format"] == "fullr"
    d["format"] = "fullc"
    A5 = Matrix.ss.import_any(**d)  # fullc

    d = A2.ss.export(raw=True)
    assert d["format"] == "bitmapr"
    d["format"] = "bitmapc"
    A6 = Matrix.ss.import_any(**d)  # bitmapc

    d = A3.ss.export(raw=True)
    assert d["format"] == "csr"
    d["format"] = "csc"
    d["row_indices"] = d["col_indices"]
    del d["col_indices"]
    A7 = Matrix.ss.import_any(**d)  # csc

    d = A4.ss.export(raw=True)
    assert d["format"] == "hypercsr"
    d["format"] = "hypercsc"
    d["row_indices"] = d["col_indices"]
    del d["col_indices"]
    d["cols"] = d["rows"]
    del d["rows"]
    A8 = Matrix.ss.import_any(**d)  # hypercsc

    assert A1.ss.is_iso is do_iso
    assert A2.ss.is_iso is do_iso
    assert A3.ss.is_iso is do_iso
    assert A4.ss.is_iso is do_iso
    assert A5.ss.is_iso is do_iso
    assert A6.ss.is_iso is do_iso
    assert A7.ss.is_iso is do_iso
    assert A8.ss.is_iso is do_iso

    for dtype in [None, np.float64]:
        expected_dtype = np.int64 if dtype is None else dtype
        for _ in range(2):
            rows, cols, vals = A0.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [])
            assert_array_equal(cols, [])
            assert_array_equal(vals, [])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A1.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 0])
            assert_array_equal(cols, [0, 1])
            assert_array_equal(vals, values1[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A2.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 0])
            assert_array_equal(cols, [0, 1])
            assert_array_equal(vals, values2[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A3.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [5, 5])
            assert_array_equal(cols, [4, 5])
            assert_array_equal(vals, values3[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A4.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [500, 500])
            assert_array_equal(cols, [400, 500])
            assert_array_equal(vals, values4[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A5.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, values1[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A6.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [0, 1])
            assert_array_equal(cols, [0, 0])
            assert_array_equal(vals, values2[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A7.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [4, 5])
            assert_array_equal(cols, [5, 5])
            assert_array_equal(vals, values3[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A8.ss.head(2, sort=False, dtype=dtype)
            assert rows.size == cols.size == vals.size == 2
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype

            rows, cols, vals = A8.ss.head(2, sort=True, dtype=dtype)
            assert_array_equal(rows, [400, 500])
            assert_array_equal(cols, [500, 500])
            assert_array_equal(vals, values4[:2])
            assert rows.dtype == cols.dtype == np.uint64
            assert vals.dtype == expected_dtype


def test_about():
    d = {}
    about = gb.ss.about
    for k in about:
        d[k] = about[k]
    assert "openmp" in about
    assert d == about
    assert len(d) == len(about)
    with pytest.raises(KeyError):
        about["badkey"]
    assert "SuiteSparse" in about["library_name"]
    # Not a GPU mode, which SuiteSparse:GraphBLAS 10.5.2 makes an alias of these
    # (see ``About._modes``)
    assert about["mode"] == ("blocking" if gb._init_params["blocking"] else "nonblocking")
    with pytest.raises(TypeError):
        del about["library_name"]  # pylint: disable=unsupported-delete-operation
    assert "library_name" in repr(about)


def test_openmp_enabled():
    # A SuiteSparse:GraphBLAS built without OpenMP runs single-threaded, and
    # nothing else reports it. Keep this a hard failure: it caught the conda-forge
    # osx-arm64 build of graphblas 10.5.0 that lacked OpenMP.
    assert gb.ss.about["openmp"], (
        "libgraphblas was built without OpenMP and will run single-threaded. "
        "This is a defect in the installed build, not in python-graphblas. "
        "Reinstall or rebuild graphblas with OpenMP enabled "
        "(SuiteSparse only warns when its CMake cannot find OpenMP, so a "
        "serial library can ship looking normal)."
    )


def test_global_config(monkeypatch):
    d = {}
    from graphblas.core.ss import jit_config

    config = gb.ss.config
    # Writing jit_c_control also sets ``_user_set_run``; restore it afterwards
    monkeypatch.setattr(jit_config, "_user_set_run", jit_config._user_set_run)
    for k in config:
        d[k] = config[k]
    assert d == config
    assert len(d) == len(config)
    for k, v in d.items():
        config[k] = v
    assert d == config
    with pytest.raises(KeyError):
        config["badkey"]
    with pytest.raises(KeyError):
        config["badkey"] = None
    config["format"] = "by_col"
    assert config["format"] == "by_col"
    config["format"] = "by_row"
    assert config["format"] == "by_row"
    with pytest.raises(TypeError):
        del config["format"]
    with pytest.raises(KeyError):
        config["format"] = "bad_format"
    for k in config:
        if k in config._defaults:
            config[k] = None
        elif k == "jit_c_control":
            # None sets the default "run", whatever graphblas.config has, so compiling a
            # C definition may still raise it to "on"
            config[k] = None
            assert config[k] == "run"
            assert jit_config._user_set_run is False
            config[k] = d[k]
        else:
            with pytest.raises(ValueError, match="Unable to set default value for"):
                config[k] = None
    # with pytest.raises(ValueError, match="Wrong number"):
    #     config["memory_pool"] = [1, 2]  # No longer used
    assert "format" in repr(config)


def test_global_config_key_completions():
    # IPython offers these for `config[<tab>]`, so they must be the keys that
    # resolve, not the attributes of the mapping object.
    config = gb.ss.config
    completions = config._ipython_key_completions_()
    assert set(completions) == set(config._options)
    for key in completions:
        config[key]
    # About uses the same hook, for gb.ss.about[<tab>]
    about = gb.ss.about
    completions = about._ipython_key_completions_()
    assert set(completions) == set(about)
    for key in completions:
        about[key]


def test_global_config_attribute_assignment_raises():
    # Attribute assignment raises and points to item assignment, rather than
    # creating an instance attribute that leaves the option alone.
    config = gb.ss.config
    before = config["nthreads"]
    with pytest.raises(AttributeError, match="item assignment"):
        config.nthreads = before + 1
    assert config["nthreads"] == before
    assert "nthreads" not in config.__dict__
    with pytest.raises(AttributeError, match="Unknown config option 'nthread'"):
        config.nthread = before + 1
    # Item assignment works
    config["nthreads"] = before + 1
    assert config["nthreads"] == before + 1
    config["nthreads"] = before
    assert config["nthreads"] == before
    # Reads, iteration, and key completions are unaffected by the guard
    assert dict(config) == {k: config[k] for k in config}
    assert set(config._ipython_key_completions_()) == set(config._options)


def test_object_config_attribute_assignment_raises():
    v = Vector(int, 3)
    with pytest.raises(AttributeError, match="item assignment"):
        v.ss.config.sparsity_control = "bitmap"
    assert v.ss.config["sparsity_control"] == {"auto"}
    with pytest.raises(AttributeError, match="read-only"):
        v.ss.config.sparsity_status = "bitmap"
    A = Matrix(int, 2, 2)
    with pytest.raises(AttributeError, match="item assignment"):
        A.ss.config.format = "by_col"
    assert A.ss.config["format"] == "by_row"
    # Item assignment works
    v.ss.config["sparsity_control"] = "bitmap"
    assert v.ss.config["sparsity_control"] == {"bitmap"}
    v.ss.config["sparsity_control"] = "auto"
    assert v.ss.config["sparsity_control"] == {"auto"}


def test_about_attribute_assignment_raises():
    about = gb.ss.about
    mode = about["mode"]
    with pytest.raises(AttributeError, match="read-only"):
        about.mode = "junk"
    assert about["mode"] == mode
    assert "mode" not in about.__dict__
    with pytest.raises(AttributeError, match="Unknown About option"):
        about.junkattr = 1
    assert dict(about) == {k: about[k] for k in about}


def test_context_attribute_assignment_raises():
    context = gb.ss.Context(engage=False)
    with pytest.raises(AttributeError, match="item assignment"):
        context.nthreads = 4


def test_context():
    context = gb.ss.Context()
    prev = dict(context)
    context["chunk"] += 1
    context["nthreads"] += 1
    assert context["chunk"] == prev["chunk"] + 1
    assert context["nthreads"] == prev["nthreads"] + 1
    context2 = gb.ss.Context(stack=True)
    assert context2 == context
    context3 = gb.ss.Context(stack=False)
    assert context3 == prev
    context4 = gb.ss.Context(
        chunk=context["chunk"] + 1, nthreads=context["nthreads"] + 1, stack=False
    )
    assert context4["chunk"] == context["chunk"] + 1
    assert context4["nthreads"] == context["nthreads"] + 1
    assert context == context.dup()
    assert context4 == context.dup(chunk=context["chunk"] + 1, nthreads=context["nthreads"] + 1)
    if "gpu_id" in gb.ss.Context._options:
        assert context.dup(gpu_id=-1)["gpu_id"] == -1
    context.engage()
    assert gb.core.ss.context.threadlocal.context is context
    with gb.ss.Context(nthreads=1) as ctx:
        assert gb.core.ss.context.threadlocal.context is ctx
        v = Vector(int, 5)
        v(nthreads=2) << v + v
        assert gb.core.ss.context.threadlocal.context is ctx
    assert gb.core.ss.context.threadlocal.context is context
    with pytest.raises(InvalidValue):
        # Wait, why does this raise?!
        ctx.disengage()
    assert gb.core.ss.context.threadlocal.context is context
    context.disengage()
    assert gb.core.ss.context.threadlocal.context is gb.core.ss.context.global_context
    assert context._prev_context is None

    # hackery
    gb.core.ss.context.threadlocal.context = context
    context.disengage()
    context.disengage()
    context.disengage()
    assert gb.core.ss.context.threadlocal.context is gb.core.ss.context.global_context

    context._context = context  # This is allowed to work with config
    with pytest.raises(AttributeError, match="_context"):
        context._context = ctx  # This is not


def _some_context_engaged():
    """Whether GraphBLAS has a Context engaged on this thread.

    Disengaging a Context that was never engaged returns GrB_INVALID_VALUE, and does
    nothing, if another Context is engaged; otherwise it returns GrB_SUCCESS.
    """
    from graphblas.core import lib

    probe = gb.ss.Context(engage=False, stack=False)
    info = lib.GxB_Context_disengage(probe._carg)
    assert info in {lib.GrB_SUCCESS, lib.GrB_INVALID_VALUE}
    return info == lib.GrB_INVALID_VALUE


def _is_engaged(context):
    """Whether ``context`` is the Context GraphBLAS has engaged on this thread."""
    from graphblas.core import lib

    if context is gb.ss.global_context:
        return not _some_context_engaged()
    if not _some_context_engaged():
        return False
    # Disengaging succeeds only for the engaged Context, so engage it again after
    if lib.GxB_Context_disengage(context._carg) != lib.GrB_SUCCESS:
        return False
    assert lib.GxB_Context_engage(context._carg) == lib.GrB_SUCCESS
    return True


def _descriptor_context(exc):
    """The Context of the descriptor that the traceback of ``exc`` keeps alive."""
    tb = exc.__traceback__
    while tb is not None:
        desc = tb.tb_frame.f_locals.get("desc")
        if getattr(desc, "_context", None) is not None:
            return desc._context
        tb = tb.tb_next
    raise AssertionError("no descriptor with a Context in the traceback")


def _mxm_nthreads(capfd, C, A, **opts):
    """Compute ``C(**opts) << A @ A`` and return the thread counts its burble reports."""
    capfd.readouterr()
    with gb.ss.burble:
        C(**opts) << A @ A
    counts = [int(x) for x in re.findall(r"\(nthreads (\d+)", capfd.readouterr().out)]
    if not counts:  # pragma: no cover (capfd may not see GraphBLAS's output everywhere)
        pytest.skip("No burble output from GraphBLAS was captured")
    return counts


@pytest.fixture
def big_square():
    # Enough work that GrB_mxm's saxpy uses every thread it is allowed, up to 37
    rng = np.random.default_rng(0)
    n = 1000
    rows = rng.integers(0, n, 50_000)
    cols = rng.integers(0, n, 50_000)
    return Matrix.from_coo(rows, cols, np.ones(rows.size), nrows=n, ncols=n, dup_op="plus")


def test_descriptor_context_not_engaged_after_error(capfd, big_square):
    """A failed call leaves no descriptor Context engaged, though its traceback lives.

    The traceback keeps the descriptor, and so its Context, alive (as IPython's
    ``sys.last_traceback`` does). A Context engaged until it is freed would make
    later calls on this thread run with ``nthreads=1``.
    """
    A = Matrix.from_coo([0, 1], [1, 0], [1, 2], nrows=2, ncols=2)
    B = Matrix(int, 3, 3)
    C = Matrix(float, *big_square.shape)
    before = _mxm_nthreads(capfd, C, big_square)
    # GrB_mxm, given the descriptor, raises
    with pytest.raises(gb.exceptions.DimensionMismatch) as excinfo:
        B(nthreads=1) << A @ A
    assert _descriptor_context(excinfo.value)["nthreads"] == 1
    assert not _some_context_engaged()
    assert gb.core.ss.context.threadlocal.context is gb.ss.global_context
    assert _mxm_nthreads(capfd, C, big_square) == before
    # An unknown option raises after the descriptor's Context is made
    with pytest.raises(ValueError, match="axb_methd") as excinfo2:
        B(nthreads=1, axb_methd="dot") << A @ A
    assert _descriptor_context(excinfo2.value)["nthreads"] == 1
    assert not _some_context_engaged()


def test_descriptor_context_nested_in_user_context(capfd, big_square):
    """A call's descriptor Context applies to it, then the user's Context is engaged again."""
    A = Matrix.from_coo([0, 1], [1, 0], [1, 2], nrows=2, ncols=2)
    B = Matrix(int, 3, 3)
    C = Matrix(float, *big_square.shape)
    with gb.ss.Context(nthreads=3) as ctx:
        assert _mxm_nthreads(capfd, C, big_square) == [3]
        assert _mxm_nthreads(capfd, C, big_square, nthreads=2) == [2]
        assert gb.core.ss.context.threadlocal.context is ctx
        assert _is_engaged(ctx)
        assert _mxm_nthreads(capfd, C, big_square) == [3]
        with pytest.raises(gb.exceptions.DimensionMismatch) as excinfo:
            B(nthreads=1) << A @ A
        assert _descriptor_context(excinfo.value)["nthreads"] == 1
        assert gb.core.ss.context.threadlocal.context is ctx
        assert _is_engaged(ctx)
        assert _mxm_nthreads(capfd, C, big_square) == [3]
        # A descriptor's Context starts from the current Context's settings
        assert _descriptor_context(excinfo.value)["chunk"] == ctx["chunk"]
    assert not _some_context_engaged()
    assert gb.core.ss.context.threadlocal.context is gb.ss.global_context


def test_descriptor_context_freed_on_another_thread():
    """A descriptor's Context freed on another thread changes neither thread's Context.

    A Context engaged on this thread and freed on another would leave this thread a
    dangling pointer (undefined behavior), and the other thread would disengage its
    own Context.
    """
    import threading
    import weakref

    from graphblas.core import lib

    A = Matrix.from_coo([0, 1], [1, 0], [1, 2], nrows=2, ncols=2)
    B = Matrix(int, 3, 3)
    box = []
    try:
        B(nthreads=1) << A @ A
    except gb.exceptions.DimensionMismatch as exc:
        box.append(exc)
    context_ref = weakref.ref(_descriptor_context(box[0]))
    results = {}

    def drop():
        with gb.ss.Context(nthreads=3) as ctx:
            box.clear()  # The last reference to the exception, and so to the Context
            gc.collect()
            results["freed"] = context_ref() is None
            results["current"] = gb.core.ss.context.threadlocal.context is ctx
            results["engaged"] = _is_engaged(ctx)

    thread = threading.Thread(target=drop)
    thread.start()
    thread.join()
    try:
        assert results == {"freed": True, "current": True, "engaged": True}
        assert not _some_context_engaged()
    finally:
        lib.GxB_Context_engage(lib.GxB_CONTEXT_WORLD)  # Don't leave a freed one engaged


@pytest.mark.parametrize(
    ("prefix", "run"),
    [
        # Calls that take no descriptor
        ("GrB_Matrix_dup", lambda A, v, n: A.dup(nthreads=n)),
        ("GrB_Vector_dup", lambda A, v, n: v.dup(nthreads=n)),
        ("GrB_Matrix_extractElement", lambda A, v, n: A[0, 1].new(nthreads=n)),
        ("GrB_Vector_extractElement", lambda A, v, n: v[1].new(nthreads=n)),
        ("GrB_Matrix_setElement", lambda A, v, n: A.__setitem__((0, 0), 5, nthreads=n)),
        ("GrB_Matrix_extractTuples", lambda A, v, n: A.ss.export("coo", nthreads=n)),
        # Calls given the descriptor
        ("GxB_Matrix_diag", lambda A, v, n: gb.ss.diag(v, nthreads=n)),
        ("GrB_mxm", lambda A, v, n: (A @ A).new(nthreads=n)),
        ("GxB_Matrix_export_CSR", lambda A, v, n: A.ss.export("csr", nthreads=n)),
        ("GrB_Matrix_dup", lambda A, v, n: A.ss.export("csr", nthreads=n)),
        ("GxB_Vector_serialize", lambda A, v, n: v.ss.serialize(nthreads=n)),
    ],
)
def test_descriptor_context_applies_to_the_call(monkeypatch, prefix, run):
    """Options such as nthreads apply to each GraphBLAS function the call runs for them.

    That includes functions that take no descriptor, such as the dup of an export.
    A stand-in for ``lib`` checks the Context as each function whose name starts
    with ``prefix`` runs.
    """
    from graphblas.core import lib
    from graphblas.core.ss import matrix as ss_matrix
    from graphblas.core.ss import vector as ss_vector

    seen = []

    def watch(name, func):
        if not name.startswith(prefix):
            return func

        def watched(*args):
            context = gb.core.ss.context.threadlocal.context
            seen.append((context["nthreads"], _is_engaged(context)))
            return func(*args)

        return watched

    class WatchedLib:
        def __getattr__(self, name):
            return watch(name, getattr(lib, name))

    real_libget = gb.core.base.libget
    monkeypatch.setattr(gb.core.base, "libget", lambda name: watch(name, real_libget(name)))
    monkeypatch.setattr(ss_matrix, "lib", WatchedLib())
    monkeypatch.setattr(ss_vector, "lib", WatchedLib())
    A = Matrix.from_coo([0, 1, 1], [1, 0, 1], [1, 2, 3], nrows=2, ncols=2)
    v = Vector.from_coo([0, 1], [1, 2], size=2)
    n = gb.ss.config["nthreads"] + 1  # Not the global setting
    assert not seen
    run(A, v, n)
    assert seen == [(n, True)]
    assert not _some_context_engaged()


def test_context_that_failed_to_build_leaves_engaged_context(monkeypatch):
    """Freeing a Context whose GxB_Context_new failed leaves the engaged Context alone.

    Its handle is NULL, and disengaging NULL would disengage any Context.
    """
    from graphblas.core import lib
    from graphblas.core.ss import context as context_module

    class FailingNew:
        def __getattr__(self, name):
            return getattr(lib, name)

        def GxB_Context_new(self, handle):
            return lib.GrB_OUT_OF_MEMORY

    with gb.ss.Context(nthreads=3) as ctx:
        with monkeypatch.context() as m:
            m.setattr(context_module, "lib", FailingNew())
            with pytest.raises(gb.exceptions.OutOfMemory):
                gb.ss.Context(engage=False)
        gc.collect()
        assert _is_engaged(ctx)


def test_descriptor_context_restored_after_interrupted_finally(monkeypatch):
    """Context._call restores the current Context if its inner ``finally`` is cut short.

    PyPy can raise KeyboardInterrupt at the first bytecode of a ``finally`` block. A
    stand-in for ``threadlocal`` raises there, at the first restore (an exception of
    its own, since a KeyboardInterrupt that escapes would stop pytest).
    """
    from graphblas.core.ss import context as context_module

    class Interrupt(BaseException):
        pass

    real = context_module.threadlocal

    class InterruptFirstRestore:
        sets = 0

        def __getattr__(self, name):
            return getattr(real, name)

        def __setattr__(self, name, value):
            # The first set engages the descriptor's Context; the second restores
            type(self).sets += 1
            if type(self).sets == 2:
                raise Interrupt
            setattr(real, name, value)

    with gb.ss.Context(nthreads=3) as ctx:
        desc = gb.core.descriptor.lookup(nthreads=2)
        with monkeypatch.context() as m:
            m.setattr(context_module, "threadlocal", InterruptFirstRestore())
            with pytest.raises(Interrupt):
                desc._context._call(lambda: None)
        assert context_module.threadlocal.context is ctx
        assert _is_engaged(ctx)
    assert not _some_context_engaged()


def _libc_arena():
    """Give the last GraphBLAS arena libc's allocator, and return its number.

    Arrays from ``empty(..., arena=k)`` then carry a NumPy memory handler of
    their own, as arrays from any custom handler (NEP 49) do.
    """
    from graphblas.core import ffi, lib

    k = lib.GxB_NARENAS - 1
    initialized = ffi.new("int*")
    assert lib.GxB_arena_initialized(initialized, k) == lib.GrB_SUCCESS
    # GxB_arena_init refuses an arena that is already initialized, which it is
    # only if the test runs again in the same process
    if not initialized[0]:  # pragma: no branch (safety)
        libc = ctypes.CDLL(None)
        funcs = [
            ffi.cast(sig, ctypes.cast(getattr(libc, name), ctypes.c_void_p).value)
            for name, sig in [
                ("malloc", "void *(*)(size_t)"),
                ("calloc", "void *(*)(size_t, size_t)"),
                ("realloc", "void *(*)(void *, size_t)"),
                ("free", "void (*)(void *)"),
            ]
        ]
        assert lib.GxB_arena_init(k, *funcs) == lib.GrB_SUCCESS
    return k


@pytest.mark.skipif("empty is None")
@pytest.mark.skipif("sys.platform == 'win32'")  # no ``ctypes.CDLL(None)``
def test_take_ownership_of_foreign_memory():
    # These arrays own their data, but GraphBLAS would free it with NumPy's
    # allocator, so it must copy them. Regression: GraphBLAS took them, and the
    # process aborted when it freed one (NumPy >= 2.5, PYTHONMALLOC=debug).
    k = _libc_arena()

    def foreign(values, dtype):
        array = empty(len(values), dtype, arena=k)
        array[:] = values
        assert array.flags.owndata
        assert not can_unclaim_buffer(array)
        return array

    indptr = foreign([0, 2, 3], np.uint64)
    col_indices = foreign([0, 2, 1], np.uint64)
    values = foreign([1.0, 2.0, 3.0], np.float64)
    A = Matrix.ss.import_csr(
        nrows=2,
        ncols=3,
        indptr=indptr,
        col_indices=col_indices,
        values=values,
        take_ownership=True,
    )
    expected = Matrix.from_coo([0, 0, 1], [0, 2, 1], [1.0, 2.0, 3.0], nrows=2, ncols=3)
    assert A.isequal(expected)
    for array in [indptr, col_indices, values]:
        assert array.flags.owndata
        assert array.flags.writeable
    values[:] = 0
    assert A.isequal(expected)

    values = foreign([1, 2, 3], np.int64)
    v = Vector.ss.import_full(values=values, take_ownership=True)
    assert v.isequal(Vector.from_coo([0, 1, 2], [1, 2, 3]))
    assert values.flags.owndata
    del A, v


def test_take_ownership_asks_suitesparse_graphblas(monkeypatch):
    # Whether GraphBLAS may take an array is up to ``can_unclaim_buffer``, since
    # ``owndata`` alone cannot tell
    monkeypatch.setattr(gb.core.utils, "can_unclaim_buffer", lambda array: False)
    v = Vector.from_coo([1, 3], [10, 20], size=5)
    d = v.ss.export("sparse")
    w = Vector.ss.import_sparse(take_ownership=True, **d)
    assert w.isequal(v)
    assert d["indices"].flags.owndata
    assert d["values"].flags.owndata


def test_take_ownership_without_can_unclaim_buffer(monkeypatch):
    # suitesparse-graphblas < 10.5.1.1 has neither function. GraphBLAS then takes an
    # array that owns its data and is writeable, and gets a NumPy copy of any other.
    monkeypatch.setattr(gb.core.utils, "can_unclaim_buffer", None)
    monkeypatch.setattr(gb.core.utils, "empty", None)
    v = Vector.from_coo([1, 3], [10, 20], size=5)
    d = v.ss.export("sparse")
    d["indices"].flags.writeable = False
    values = np.array([10, 20, 0, 0])
    w = Vector.ss.import_sparse(take_ownership=True, **dict(d, values=values[:2]))
    assert w.isequal(v)
    assert d["indices"].flags.owndata
    values[:] = 0  # w has a copy of the view
    assert w.isequal(v)

    d = v.ss.export("sparse")
    w = Vector.ss.import_sparse(take_ownership=True, **d)
    assert w.isequal(v)
    assert not d["indices"].flags.owndata
    assert not d["values"].flags.owndata


def test_distinct_buffers():
    # test_take_ownership_of_one_array_twice imports an array given twice
    a = np.arange(3, dtype=np.uint64)
    b = np.asfortranarray(np.ones((2, 3), np.bool_))
    rv = _distinct_buffers(a, b, a, b)
    assert rv[0] is a
    assert rv[1] is b
    for orig, copy in [(a, rv[2]), (b, rv[3])]:
        assert copy is not orig
        assert copy.flags.owndata
        assert copy.flags.f_contiguous == orig.flags.f_contiguous
        assert_array_equal(copy, orig)


def test_ownable_copy_of_nested_subarray_values_is_made_once():
    # Values whose record dtype nests subarrays are viewed with the flattened
    # dtype before the copy (see ``values_to_numpy_buffer``). Regression: the
    # view was taken after, and did not own its data, so it was copied again.
    point = np.dtype((np.float64, (3,)))
    layered = np.dtype([("oc_k", np.int64), ("oc_pts", point, (2,))], align=True)
    vals = np.zeros(2**16, layered)
    vals["oc_pts"] = np.arange(vals.size * 6.0).reshape(-1, 2, 3)
    started = not tracemalloc.is_tracing()
    if started:  # pragma: no branch (safety)
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        arr, dtype = values_to_numpy_buffer(vals, copy=True, ownable=True)
        numpy_bytes = tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:  # pragma: no branch (safety)
            tracemalloc.stop()
    assert numpy_bytes < 1.5 * vals.nbytes
    assert dtype.np_type != layered
    assert arr.dtype == dtype.np_type
    assert arr.flags.owndata
    assert_array_equal(arr["oc_pts"], vals["oc_pts"])


# Run in a subprocess: GraphBLAS is initialized once per process, and this needs
# it initialized with libc's allocator before python-graphblas starts.
_LIBC_ALLOCATOR_PROBE = """
import tracemalloc

import numpy as np
import suitesparse_graphblas as ssgb

ssgb.initialize(memory_manager="c")
import graphblas as gb
from suitesparse_graphblas.utils import empty

print("package: " + gb.__file__)
for take_ownership in [False, True]:
    values = np.arange(1000, dtype=np.float64)
    v = gb.Vector.ss.import_full(values=values, take_ownership=take_ownership)
    assert v.reduce().new().value == values.sum()
    del v

# GraphBLAS can't own memory NumPy allocates here, so each array is copied, or
# converted, once, straight into libc's memory, which tracemalloc doesn't see.
# A copy NumPy made first would show up as an array's worth of NumPy memory.
n = 2**20
indptr = np.arange(n + 1)  # int64, converted to uint64
indices = np.arange(n)
values = np.arange(n, dtype=np.float64)
tracemalloc.start()
for dtype in [None, np.float32]:
    for kwargs in [{}, {"take_ownership": True}, {"take_ownership": True, "secure_import": True}]:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        A = gb.Matrix.ss.import_csr(
            nrows=n, ncols=n, indptr=indptr, col_indices=indices, values=values, dtype=dtype,
            **kwargs,
        )
        numpy_bytes = tracemalloc.get_traced_memory()[1] - before
        assert numpy_bytes < values.nbytes // 4, (dtype, kwargs, numpy_bytes)
        assert A.nvals == n, (dtype, kwargs)
        assert A[n - 1, n - 1].new().value == n - 1, (dtype, kwargs)
        assert values.flags.owndata, (dtype, kwargs)
        del A
tracemalloc.stop()

# Memory from GraphBLAS's allocator is still taken, not copied
values = empty(n)
values[:] = 1
v = gb.Vector.ss.import_full(values=values, take_ownership=True)
assert not values.flags.owndata
assert v.reduce().new().value == n
del v
print("done")
"""


def _run_probe(source, **env):
    """Run ``source`` in a new process that imports this tree's graphblas, and check it ends."""
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1", **env)
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report
    assert result.stdout.splitlines()[-1] == "done", report


@pytest.mark.slow
@pytest.mark.skipif("empty is None")
def test_import_with_libc_allocator():
    # With ``memory_manager="c"`` GraphBLAS frees with libc's ``free``, so not
    # even the copy that ``take_ownership=False`` makes may come from NumPy
    _run_probe(_LIBC_ALLOCATOR_PROBE, PYTHONMALLOC="debug")


# Run in a subprocess: a regression frees an array twice, which aborts the
# process instead of failing the test.
_ONE_ARRAY_TWICE_PROBE = """
import numpy as np

import graphblas as gb
from graphblas import Matrix

print("package: " + gb.__file__)
a = np.array([0, 2], np.uint64)
A = Matrix.ss.import_csr(
    nrows=1, ncols=3, indptr=a, col_indices=a, values=[1.0, 2.0], take_ownership=True
)
assert A.isequal(Matrix.from_coo([0, 0], [0, 2], [1.0, 2.0], nrows=1, ncols=3))
assert not a.flags.owndata  # taken once, copied once

a = np.array([0, 2], np.uint64)
A = Matrix.ss.import_hypercsr(
    nrows=3,
    ncols=3,
    indptr=[0, 1, 2],
    rows=a,
    col_indices=a,
    values=[1.0, 2.0],
    take_ownership=True,
)
assert A.isequal(Matrix.from_coo([0, 2], [0, 2], [1.0, 2.0], nrows=3, ncols=3))

a = np.array([0, 2], np.uint64)
A = Matrix.ss.import_hypercsc(
    nrows=3,
    ncols=3,
    indptr=[0, 1, 2],
    cols=a,
    row_indices=a,
    values=[1.0, 2.0],
    take_ownership=True,
)
assert A.isequal(Matrix.from_coo([0, 2], [0, 2], [1.0, 2.0], nrows=3, ncols=3))
del A
print("done")
"""


def test_take_ownership_of_one_array_twice():
    # GraphBLAS frees every buffer it takes, so an array given as two arguments
    # has to be copied for one of them. Regression: an index array given twice was
    # freed twice, crashing the process.
    _run_probe(_ONE_ARRAY_TWICE_PROBE)


# Run in a subprocess: a regression frees memory twice, which aborts the process
# at the second free or later. The arrays are over 1 KiB because NumPy caches
# smaller blocks instead of freeing them, so a double free of those is silent.
_FAILED_SECURE_IMPORT_PROBE = """
import numpy as np

import graphblas as gb
from graphblas import Matrix, Vector
from graphblas.exceptions import GraphblasException

print("package: " + gb.__file__)
n = 200


def indices():
    # An index out of range: only the secure check reads the indices.
    rv = np.zeros(n, np.uint64)
    rv[-1] = n + 5
    return rv


def bitmap(order="C"):
    # nvals does not match the bitmap: only the secure check counts it.
    return {"bitmap": np.ones((n, n), np.bool_, order=order),
            "values": np.ones((n, n), order=order), "nvals": 5}


indptr = lambda: np.arange(n + 1, dtype=np.uint64)  # noqa: E731
formats = {
    "csr": lambda: {"indptr": indptr(), "col_indices": indices(), "values": np.ones(n)},
    "csc": lambda: {"indptr": indptr(), "row_indices": indices(), "values": np.ones(n)},
    "hypercsr": lambda: {"indptr": indptr(), "rows": np.arange(n, dtype=np.uint64),
                         "col_indices": indices(), "values": np.ones(n)},
    "hypercsc": lambda: {"indptr": indptr(), "cols": np.arange(n, dtype=np.uint64),
                         "row_indices": indices(), "values": np.ones(n)},
    "bitmapr": bitmap,
    "bitmapc": lambda: bitmap("F"),
    "sparse": lambda: {"indices": indices(), "values": np.ones(n)},
    "bitmap": lambda: {"bitmap": np.ones(n * n, np.bool_), "values": np.ones(n * n),
                       "nvals": 5},
}
A = Matrix.from_coo([0, 1], [1, 0], [1.0, 2.0], nrows=n, ncols=n)
v = Vector.from_coo([1], [1.0], size=n * n)
for take_ownership in [False, True]:
    for format, make in formats.items():
        for method in ["import", "pack"]:
            data = make()
            if format in {"sparse", "bitmap"}:
                obj = v.dup()
                func = getattr(obj.ss, f"{method}_{format}")
                if method == "import":
                    data["size"] = n if format == "sparse" else n * n
            else:
                obj = A.dup()
                func = getattr(obj.ss, f"{method}_{format}")
                if method == "import":
                    data.update(nrows=n, ncols=n)
            arrays = [val for val in data.values() if isinstance(val, np.ndarray)]
            try:
                func(**data, secure_import=True, take_ownership=take_ownership)
            except GraphblasException:
                pass
            else:
                raise AssertionError(f"the secure check passed: {method}_{format}")
            # A secure import copies the arrays, even with take_ownership=True, so
            # GraphBLAS freed only the copies: the caller's arrays still own their
            # data and still hold it.
            owndata = [a.flags.owndata for a in arrays]
            assert all(owndata), (method, format, owndata)
            fresh = [val for val in make().values() if isinstance(val, np.ndarray)]
            assert all(np.array_equal(a, b) for a, b in zip(arrays, fresh)), (method, format)
            # A failed pack leaves the object as it was.
            assert obj.isequal(v if format in {"sparse", "bitmap"} else A), (method, format)
            del data, arrays, obj, func
for _ in range(100):  # use the heap, so a corrupted one aborts here
    gb.Vector.from_dense(np.ones(3000))
print("done")
"""


def test_failed_secure_import_frees_once():
    # GraphBLAS moves the arrays into the object before a secure check, and
    # frees them with the object when the check fails, so they must be unclaimed
    # after a failed call too. A secure pack must leave the object alone (see
    # ``_pack_securely``). Regression: NumPy freed the arrays a second time, and
    # the process aborted.
    _run_probe(_FAILED_SECURE_IMPORT_PROBE)


@pytest.mark.parametrize("take_ownership", [False, True])
def test_secure_import(take_ownership):
    # Valid data passes a secure import's or pack's check, in every format.
    sparse = Matrix.from_coo([0, 1, 1], [1, 0, 2], [1.0, 2.0, 3.0], nrows=2, ncols=3)
    dense = Matrix.from_dense(np.arange(6.0).reshape(2, 3))
    for format in ["csr", "csc", "hypercsr", "hypercsc", "bitmapr", "bitmapc", "fullr", "fullc"]:
        A = dense if format.startswith("full") else sparse
        info = A.ss.export(format)
        arrays = [val for val in info.values() if isinstance(val, np.ndarray)]
        owndata = [array.flags.owndata for array in arrays]
        B = Matrix.ss.import_any(**info, secure_import=True, take_ownership=take_ownership)
        assert B.isequal(A, check_dtype=True), format
        # A secure import copies, so the arrays keep their data either way.
        assert [array.flags.owndata for array in arrays] == owndata, format
        B.ss.pack_any(**B.ss.export(format), secure_import=True, take_ownership=take_ownership)
        assert B.isequal(A, check_dtype=True), format
    sparse = Vector.from_coo([1, 3], [1.0, 2.0], size=5)
    dense = Vector.from_dense(np.arange(5.0))
    for format in ["sparse", "bitmap", "full"]:
        v = dense if format == "full" else sparse
        info = v.ss.export(format)
        arrays = [val for val in info.values() if isinstance(val, np.ndarray)]
        owndata = [array.flags.owndata for array in arrays]
        w = Vector.ss.import_any(**info, secure_import=True, take_ownership=take_ownership)
        assert w.isequal(v, check_dtype=True), format
        # A secure import copies, so the arrays keep their data either way.
        assert [array.flags.owndata for array in arrays] == owndata, format
        w.ss.pack_any(**w.ss.export(format), secure_import=True, take_ownership=take_ownership)
        assert w.isequal(v, check_dtype=True), format


def test_indexbinaryop_needs_suitesparse_9_4(monkeypatch):
    # python-suitesparse-graphblas 9.3.1.0, our minimum, bundles SuiteSparse 9.3, which
    # has no IndexBinaryOp. Pretend the library lacks it, so this runs on every version.
    from graphblas.core.operator import indexbinary as core_indexbinary
    from graphblas.core.ss import indexbinary as ss_indexbinary

    monkeypatch.setattr(core_indexbinary, "_has_idxbinop", False)
    monkeypatch.setattr(ss_indexbinary, "_has_idxbinop", False)
    match = r"python-suitesparse-graphblas 9\.4\.5\.0\+"

    def plus_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (never called)
        return x + y + theta

    if supports_udfs:
        with pytest.raises(RuntimeError, match=match):
            gb.indexbinary.register_anonymous(plus_theta)
    else:
        with pytest.raises(RuntimeError, match="install numba for UDF support"):
            gb.indexbinary.register_anonymous(plus_theta)
    cdef = (
        "void plus_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, double *y, "
        "GrB_Index iy, GrB_Index jy, double *theta) { (*z) = (*x) + (*y) + (*theta) ; }"
    )
    with pytest.raises(RuntimeError, match=match):
        gb.indexbinary.ss.register_new("plus_theta", cdef, "FP64", "FP64", "FP64", "FP64")

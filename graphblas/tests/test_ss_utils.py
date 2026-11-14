import ctypes
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import Matrix, Vector, backend
from graphblas.core.utils import _distinct_buffers, values_to_numpy_buffer
from graphblas.exceptions import InvalidValue

if backend != "suitesparse":
    pytest.skip("gb.ss and A.ss only available with suitesparse backend", allow_module_level=True)

try:
    from suitesparse_graphblas.utils import can_unclaim_buffer, empty
except ImportError:  # pragma: no cover (import)
    # suitesparse-graphblas < 10.5.1.1
    can_unclaim_buffer = empty = None


# ``from_coo`` with all-equal values used to come back iso, because SuiteSparse
# checked after a build whether every value was the same. 10.5.0 dropped that
# check ("GrB_Matrix_build and GrB_Vector_build: no longer do a post-iso
# check"), so the iso arm of these tests has to ask for iso outright.
# ``ss.build_scalar`` does, and produces the same storage formats on every
# version, which is what the tests below are really selecting for.
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
    # SS:GraphBLAS 10.5.2 made the GPU modes aliases of these; neither may read as a GPU mode
    assert about["mode"] == ("blocking" if gb._init_params["blocking"] else "nonblocking")
    with pytest.raises(TypeError):
        del about["library_name"]  # pylint: disable=unsupported-delete-operation
    assert "library_name" in repr(about)


def test_openmp_enabled():
    # A SuiteSparse:GraphBLAS built without OpenMP is single-threaded, which is
    # a large and entirely silent performance loss: nothing else about the
    # library reports it, and the conda package still depends on an OpenMP
    # runtime whether or not the build actually found one. Keep this assertion
    # hard; it is how the conda-forge osx-arm64 regression in graphblas 10.5.0
    # was noticed.
    assert gb.ss.about["openmp"], (
        "libgraphblas was built without OpenMP and will run single-threaded. "
        "This is a defect in the installed build, not in python-graphblas. "
        "Reinstall or rebuild graphblas with OpenMP enabled "
        "(SuiteSparse only warns when its CMake cannot find OpenMP, so a "
        "serial library can ship looking normal)."
    )


def test_global_config(monkeypatch):
    d = {}
    config = gb.ss.config
    if "jit_c_control" in config:  # SuiteSparse:GraphBLAS 8+
        from graphblas.core.ss import jit_config

        # Writing jit_c_control also sets whether UDT work keeps a "run"; undo that after
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
            # None is the default "run" whatever graphblas.config has, and not a "run"
            # the user chose, so UDT work may still raise it to "on"
            config[k] = None
            assert config[k] == "run"
            assert jit_config._user_set_run is False
            config[k] = d[k]
        else:
            with pytest.raises(ValueError, match="Unable to set default value for"):
                config[k] = None
    bitmap_switch = config["bitmap_switch"]
    with pytest.raises(ValueError, match="Wrong number of elements when setting 'bitmap_switch'"):
        config["bitmap_switch"] = [0.5, 0.5]
    assert config["bitmap_switch"] == bitmap_switch
    assert "format" in repr(config)


def test_global_config_key_completions():
    # IPython offers these for `config[<tab>]`, so they must be the keys that
    # actually resolve, not the attributes of the mapping object.
    config = gb.ss.config
    completions = config._ipython_key_completions_()
    assert set(completions) == set(config._options)
    for key in completions:
        config[key]
    # About aliases the same hook onto its own __iter__ (gb.ss.about[<tab>])
    about = gb.ss.about
    completions = about._ipython_key_completions_()
    assert set(completions) == set(about)
    for key in completions:
        about[key]


def test_global_config_attribute_assignment_raises():
    # Attribute assignment used to create a dead instance attribute and leave
    # the real config unchanged; now it raises and points at item assignment.
    config = gb.ss.config
    before = config["nthreads"]
    with pytest.raises(AttributeError, match="item assignment"):
        config.nthreads = before + 1
    assert config["nthreads"] == before
    assert "nthreads" not in config.__dict__
    with pytest.raises(AttributeError, match="Unknown config option 'nthread'"):
        config.nthread = before + 1
    # Item assignment is the supported write and still works
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
    # Item assignment is the supported write and still works
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


@pytest.mark.skipif("gb.core.ss._IS_SSGB7")
def test_context_attribute_assignment_raises():
    context = gb.ss.Context(engage=False)
    with pytest.raises(AttributeError, match="item assignment"):
        context.nthreads = 4


@pytest.mark.skipif("gb.core.ss._IS_SSGB7")
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

    # Actually engaged, but not set in threadlocal
    context._engage()
    assert gb.core.ss.context.threadlocal.context is gb.core.ss.context.global_context
    context.disengage()

    context.engage()
    context._engage()
    assert gb.core.ss.context.threadlocal.context is context
    context.disengage()

    context._context = context  # This is allowed to work with config
    with pytest.raises(AttributeError, match="_context"):
        context._context = ctx  # This is not


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
    # allocator, so it must copy them. It used to take them, and then the
    # process aborted when GraphBLAS freed one (NumPy >=2.5, PYTHONMALLOC=debug).
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
    # Whether GraphBLAS may own an array is up to ``can_unclaim_buffer``
    # (``owndata`` alone can't tell), even on a suitesparse-graphblas without it
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
    # (Importing an array given twice is tested in a subprocess, below.)
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
    # dtype. A view of the copy would not own its data, so it was copied again.
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
    # has to be copied for one of them. Only an index array given as the values
    # was, and an index array given twice was freed twice, crashing the process.
    _run_probe(_ONE_ARRAY_TWICE_PROBE)


# Run in a subprocess: a regression frees memory twice, which aborts the process
# at the second free or later. Arrays over 1 KiB: NumPy caches smaller blocks
# instead of freeing them, so a double free of those is silent.
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
            # A pack leaves the object as it was. GraphBLAS freed it.
            assert obj.isequal(v if format in {"sparse", "bitmap"} else A), (method, format)
            del data, arrays, obj, func
for _ in range(100):  # use the heap, so a corrupted one aborts here
    gb.Vector.from_dense(np.ones(3000))
print("done")
"""


def test_failed_secure_import_frees_once():
    # GraphBLAS moves the arrays into the object before a secure check, and
    # frees them with the object when the check fails. The arrays, or the copies
    # made of them, were unclaimed only after a successful call, so NumPy freed
    # them a second time, and the process aborted. A secure pack also freed the
    # packed object itself (a SuiteSparse bug), whose handle Python then used and
    # freed again; it now checks the data with a secure import into a new object.
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

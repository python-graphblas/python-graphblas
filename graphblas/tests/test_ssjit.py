import contextlib
import os
import sysconfig

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import (
    agg,
    backend,
    binary,
    dtypes,
    indexbinary,
    indexunary,
    monoid,
    select,
    semiring,
    unary,
)
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.operator.indexbinary import _has_idxbinop
from graphblas.core.ss import _IS_SSGB7

from .conftest import autocompute, burble

from graphblas import Vector  # isort:skip (for dask-graphblas)

try:
    import numba
except ImportError:
    numba = None

if backend != "suitesparse":
    pytest.skip("not suitesparse backend", allow_module_level=True)


# The fix-conda-build-paths logic lives in ``gb.ss.fix_jit_config`` so users
# can call it themselves. Tests use it through the public surface.
_fix_jit_config = gb.ss.fix_jit_config if not _IS_SSGB7 else (lambda: None)

# Capture the post-import JIT state before the autouse ``_setup_jit`` fixture
# runs. ``_auto_fix_jit_at_import`` probes once; ``jit_c_control`` is ``'on'``
# iff the env can actually JIT-compile. Tests that assert on the import-time
# state read this constant; the fixture may transiently mutate the live config.
_JIT_WORKS_AT_IMPORT = (
    not _IS_SSGB7 and backend == "suitesparse" and gb.ss.config["jit_c_control"] == "on"
)


@pytest.fixture(scope="module", autouse=True)
def _setup_jit():
    """Set up the SuiteSparse:GraphBLAS JIT.

    Strategy:
    1. _fix_jit_config(): fix conda-baked compiler paths and probe.
       - Returns True: JIT works, proceed.
       - Returns False: probe failed. SuiteSparse will have left
         ``jit_c_control = 'load'`` (compile disabled, cache loading
         still allowed); we leave that state untouched.
       - Returns None: no conda env, try sysconfig instead.
    2. Sysconfig fallback: for non-conda installs (pure pip).
    """
    if _IS_SSGB7:
        # SuiteSparse JIT was added in SSGB 8
        yield
        return

    prev = gb.ss.config["jit_c_control"]

    result = _fix_jit_config()
    # ``True``: conda JIT configured and verified. ``False``: probe failed
    # (``_probe_jit`` leaves ``jit_c_control`` at ``'load'``; don't try
    # sysconfig, since if the conda compiler can't build GraphBLAS JIT
    # kernels, Python's sysconfig compiler won't either). Both done.
    if result is None:
        # No conda env. Try sysconfig for non-conda installs.
        cc = sysconfig.get_config_var("CC")
        cflags = sysconfig.get_config_var("CFLAGS")
        include = sysconfig.get_path("include")
        libs = sysconfig.get_config_var("LIBS")
        if cc and cflags and include:
            gb.ss.config["jit_c_control"] = "on"
            gb.ss.config["jit_c_compiler_name"] = cc
            gb.ss.config["jit_c_compiler_flags"] = f"{cflags} -I{include}"
            if libs:
                gb.ss.config["jit_c_libraries"] = libs

    try:
        yield
    finally:
        gb.ss.config["jit_c_control"] = prev


def _require_jit_on():
    """Skip the test if the SuiteSparse JIT can't compile in this environment."""
    if gb.ss.config["jit_c_control"] != "on":
        pytest.skip("JIT compilation not available (probe failed or compiler missing)")


@contextlib.contextmanager
def _jit_mode(mode):
    """Temporarily set ``jit_c_control`` to ``mode``; restore on exit."""
    prev = gb.ss.config["jit_c_control"]
    gb.ss.config["jit_c_control"] = mode
    try:
        yield
    finally:
        gb.ss.config["jit_c_control"] = prev


@pytest.mark.skipif("_IS_SSGB7")
def test_auto_fix_jit_at_import_left_compiler_usable():
    """After ``import graphblas.ss``, a probe-confirmed compiler implies
    ``jit_c_control == 'on'``.
    """
    if not gb.ss.jit_compiler_is_usable():
        pytest.skip("sandboxed env without a usable compiler; auto-fix had nothing to repair")
    if not _JIT_WORKS_AT_IMPORT:
        # Compiler file exists but the import-time probe failed (e.g., the
        # baked-in flags target a different arch than the host). After a
        # compile failure SuiteSparse drops ``jit_c_control`` to a
        # non-compiling mode so downstream ops punt to generic cleanly; the
        # probe absorbs that first failure. Which mode it lands in (``'load'``
        # or ``'run'``) varies by SuiteSparse version.
        assert gb.ss.config["jit_c_control"] in {"load", "run"}
        pytest.skip("compiler present but JIT probe failed in this env")
    assert _JIT_WORKS_AT_IMPORT
    assert gb.ss.config["jit_c_control"] == "on"


@pytest.mark.skipif("_IS_SSGB7")
def test_public_fix_jit_config_repairs_a_broken_compiler():
    """``gb.ss.fix_jit_config()`` repairs a clobbered compiler path and returns ``True``."""
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("env JIT doesn't actually compile; nothing to repair to")
    prev = {
        "control": gb.ss.config["jit_c_control"],
        "cc": gb.ss.config["jit_c_compiler_name"],
        "flags": gb.ss.config["jit_c_compiler_flags"],
    }
    if not os.environ.get("CONDA_PREFIX"):
        pytest.skip("test requires a CONDA_PREFIX to repair from")
    try:
        # Force a broken path; the auto-fix must put back something usable.
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        assert not gb.ss.jit_compiler_is_usable()
        result = gb.ss.fix_jit_config()
        assert result is True
        assert gb.ss.jit_compiler_is_usable()
        # Calling fix_jit_config a second time must not disable JIT. The probe
        # tries to register a one-shot ``_jit_probe`` UDT and previously
        # interpreted the duplicate-name error as a probe failure.
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        result2 = gb.ss.fix_jit_config()
        assert result2 is True
        assert gb.ss.config["jit_c_control"] == "on"
    finally:
        gb.ss.config["jit_c_control"] = prev["control"]
        gb.ss.config["jit_c_compiler_name"] = prev["cc"]
        gb.ss.config["jit_c_compiler_flags"] = prev["flags"]


@pytest.mark.skipif("_IS_SSGB7")
def test_public_fix_jit_config_returns_none_without_env():
    """With no ``CONDA_PREFIX`` and ``use_sysconfig=False``, the helper signals no-environment."""
    if os.environ.get("CONDA_PREFIX"):
        # Drop the env for this call only.
        prev_env = os.environ.pop("CONDA_PREFIX")
    else:
        prev_env = None
    prev_cc = gb.ss.config["jit_c_compiler_name"]
    try:
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        result = gb.ss.fix_jit_config(use_sysconfig=False)
        assert result is None
    finally:
        if prev_env is not None:
            os.environ["CONDA_PREFIX"] = prev_env
        gb.ss.config["jit_c_compiler_name"] = prev_cc


@pytest.fixture
def v():
    return Vector.from_coo([1, 3, 4, 6], [1, 1, 2, 0])


@autocompute
def test_jit_udt():
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            dtypes.ss.register_new(
                "myquaternion", "typedef struct { float x [4][4] ; int color ; } myquaternion ;"
            )
        return
    _require_jit_on()
    with burble():
        dtype = dtypes.ss.register_new(
            "myquaternion", "typedef struct { float x [4][4] ; int color ; } myquaternion ;"
        )
    assert not hasattr(dtypes, "myquaternion")
    assert dtypes.ss.myquaternion is dtype
    assert dtype.name == "myquaternion"
    assert str(dtype) == "myquaternion"
    assert dtype.gb_name is None
    v = Vector(dtype, 2)
    np_type = np.dtype([("x", "<f4", (4, 4)), ("color", "<i4")], align=True)
    if numba is None or numba.__version__[:5] < "0.57.":
        assert dtype.np_type == np.dtype((np.uint8, np_type.itemsize))
        with pytest.raises(TypeError):
            v[0] = {"x": np.arange(16).reshape(4, 4), "color": 100}
        # We can provide dtype directly to make things work more nicely
        dtype = dtypes.ss.register_new(
            "myquaternion2",
            "typedef struct { float x [4][4] ; int color ; } myquaternion2 ;",
            np_type=np_type,
        )
        v = Vector(dtype, 2)
    assert dtype.np_type == np_type
    v[0] = {"x": np.arange(16).reshape(4, 4), "color": 100}
    assert_array_equal(v[0].value["x"], np.arange(16).reshape(4, 4))
    assert v[0].value["color"] == 100
    v[1] = (2, 3)
    if supports_udfs:
        expected = Vector.from_dense([100, 3])
        assert expected.isequal(v.apply(lambda x: x["color"]))  # pragma: no cover (numba)

    np_type = np.dtype([("x", "<f4", (3, 3)), ("color", "<i4")], align=True)
    dtype = dtypes.ss.register_new(
        "notquaternion",
        "typedef struct { float x [3][3] ; int color ; } notquaternion ;",
        np_type=np_type,
    )
    assert dtype.np_type == np_type


def test_jit_unary(v):
    cdef = "void square (float *z, float *x) { (*z) = (*x) * (*x) ; } ;"
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            unary.ss.register_new("square", cdef, "FP32", "FP32")
        return
    _require_jit_on()
    with burble():
        square = unary.ss.register_new("square", cdef, "FP32", "FP32")
    assert not hasattr(unary, "square")
    assert unary.ss.square is square
    assert square.name == "ss.square"
    assert square.types == {dtypes.FP32: dtypes.FP32}
    # JIT ops don't coerce: wrong dtype raises KeyError, not a silent cast.
    with pytest.raises(KeyError, match="square does not work with INT64"):
        v << square(v)
    v = v.dup("FP32")
    v << square(v)
    expected = Vector.from_coo([1, 3, 4, 6], [1, 1, 4, 0], dtype="FP32")
    assert expected.isequal(v)
    assert square["FP32"].jit_c_definition == cdef
    assert "FP64" not in square
    with burble():
        square_fp64 = unary.ss.register_new(
            "square", cdef.replace("float", "double"), "FP64", "FP64"
        )
    assert square_fp64 is square
    assert "FP64" in square
    with pytest.raises(
        TypeError, match="UnaryOp gb.unary.ss.square already defined for FP32 input type"
    ):
        unary.ss.register_new("square", cdef, "FP32", "FP32")
    unary.ss.register_new("nested.square", cdef, "FP32", "FP32")
    with pytest.raises(AttributeError, match="nested is already defined"):
        unary.ss.register_new("nested", cdef, "FP32", "FP32")


@pytest.mark.slow
def test_jit_binary(v):
    cdef = "void absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }"
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            binary.ss.register_new("absdiff", cdef, "FP64", "FP64", "FP64")
        return
    _require_jit_on()
    with burble():
        absdiff = binary.ss.register_new(
            "absdiff",
            cdef,
            "FP64",
            "FP64",
            "FP64",
        )
    assert not hasattr(binary, "absdiff")
    assert binary.ss.absdiff is absdiff
    assert absdiff.name == "ss.absdiff"
    assert absdiff.types == {(dtypes.FP64, dtypes.FP64): dtypes.FP64}  # different than normal
    assert "FP64" in absdiff
    assert absdiff["FP64"].return_type == dtypes.FP64
    # JIT ops don't coerce: wrong dtype raises KeyError, not a silent cast.
    with pytest.raises(KeyError, match="absdiff does not work with .INT64, INT64. types"):
        v << absdiff(v & v)
    w = (v - 1).new("FP64")
    v = v.dup("FP64")
    res = absdiff(v & w).new()
    expected = Vector.from_coo([1, 3, 4, 6], [1, 1, 1, 1], dtype="FP64")
    assert expected.isequal(res)
    res = absdiff(w & v).new()
    assert expected.isequal(res)
    assert absdiff["FP64"].jit_c_definition == cdef
    assert "FP32" not in absdiff
    with burble():
        absdiff_fp32 = binary.ss.register_new(
            "absdiff",
            cdef.replace("FP64", "FP32").replace("fabs", "fabsf"),
            "FP32",
            "FP32",
            "FP32",
        )
    assert absdiff_fp32 is absdiff
    assert "FP32" in absdiff
    with pytest.raises(
        TypeError,
        match="BinaryOp gb.binary.ss.absdiff already defined for .FP64, FP64. input types",
    ):
        binary.ss.register_new("absdiff", cdef, "FP64", "FP64", "FP64")
    binary.ss.register_new("nested.absdiff", cdef, "FP64", "FP64", "FP64")
    with pytest.raises(AttributeError, match="nested is already defined"):
        binary.ss.register_new("nested", cdef, "FP64", "FP64", "FP64")
    # Make sure we can be specific with left/right dtypes
    absdiff_mixed = binary.ss.register_new(
        "absdiff",
        "void absdiff (double *z, double *x, float *y) { (*z) = fabs ((*x) - (double)(*y)) ; }",
        "FP64",
        "FP32",
        "FP64",
    )
    assert absdiff_mixed is absdiff
    assert ("FP64", "FP32") in absdiff
    assert ("FP32", "FP64") not in absdiff


@pytest.mark.slow
def test_jit_indexunary(v):
    cdef = (
        "void diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }"
    )
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            indexunary.ss.register_new("diffy", cdef, "FP64", "FP64", "FP64")
        return
    _require_jit_on()
    with burble():
        diffy = indexunary.ss.register_new("diffy", cdef, "FP64", "FP64", "FP64")
    assert not hasattr(indexunary, "diffy")
    assert indexunary.ss.diffy is diffy
    assert not hasattr(select, "diffy")
    assert not hasattr(select.ss, "diffy")
    assert diffy.name == "ss.diffy"
    assert diffy.types == {(dtypes.FP64, dtypes.FP64): dtypes.FP64}
    assert "FP64" in diffy
    assert diffy["FP64"].return_type == dtypes.FP64
    # JIT ops don't coerce: wrong dtype raises KeyError, not a silent cast.
    with pytest.raises(KeyError, match="diffy does not work with .INT64, INT64. types"):
        v << diffy(v, 1)
    v = v.dup("FP64")
    with pytest.raises(KeyError, match="diffy does not work with .FP64, INT64. types"):
        v << diffy(v, -1)
    res = diffy(v, -1.0).new()
    expected = Vector.from_coo([1, 3, 4, 6], [2, 6, 12, 6], dtype="FP64")
    assert expected.isequal(res)
    assert diffy["FP64"].jit_c_definition == cdef
    assert "FP32" not in diffy
    with burble():
        diffy_fp32 = indexunary.ss.register_new(
            "diffy",
            cdef.replace("double", "float").replace("fabs", "fabsf"),
            "FP32",
            "FP32",
            "FP32",
        )
    assert diffy_fp32 is diffy
    assert "FP32" in diffy
    with pytest.raises(
        TypeError,
        match="IndexUnaryOp gb.indexunary.ss.diffy already defined for .FP64, FP64. input types",
    ):
        indexunary.ss.register_new("diffy", cdef, "FP64", "FP64", "FP64")
    indexunary.ss.register_new("nested.diffy", cdef, "FP64", "FP64", "FP64")
    with pytest.raises(AttributeError, match="nested is already defined"):
        indexunary.ss.register_new("nested", cdef, "FP64", "FP64", "FP64")
    # Make sure we can be specific with left/right dtypes
    diffy_mixed = indexunary.ss.register_new(
        "diffy",
        "void diffy (double *z, double *x, GrB_Index i, GrB_Index j, float *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (double)(*y)) ; }",
        "FP64",
        "FP32",
        "FP64",
    )
    assert diffy_mixed is diffy
    assert ("FP64", "FP32") in diffy
    assert ("FP32", "FP64") not in diffy


@pytest.mark.slow
@pytest.mark.skipif(not _has_idxbinop, reason="requires SuiteSparse:GraphBLAS 9.4+")
def test_jit_indexbinary(v):
    cdef = (
        "void add_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, "
        "double *y, GrB_Index iy, GrB_Index jy, double *theta) "
        "{ (*z) = (*x) + (*y) + (*theta) ; }"
    )
    _require_jit_on()
    with burble():
        add_theta = indexbinary.ss.register_new("add_theta", cdef, "FP64", "FP64", "FP64", "FP64")
    assert not hasattr(indexbinary, "add_theta")
    assert indexbinary.ss.add_theta is add_theta
    assert add_theta.name == "ss.add_theta"
    assert add_theta.types == {(dtypes.FP64, dtypes.FP64): dtypes.FP64}
    assert "FP64" in add_theta
    assert add_theta["FP64"].return_type == dtypes.FP64
    assert add_theta["FP64"].jit_c_definition == cdef
    # Bind theta and use as BinaryOp
    v64 = v.dup("FP64")
    binop = add_theta["FP64"](10.0)
    assert binop.opclass == "BinaryOp"
    res = v64.ewise_mult(v64, binop).new()
    # v has values at [1, 3, 4, 6] with vals [1, 1, 2, 0]
    # ewise_mult: x+y+theta = 2*val + 10
    expected = Vector.from_coo([1, 3, 4, 6], [12.0, 12.0, 14.0, 10.0], dtype="FP64")
    assert expected.isequal(res)
    # Test duplicate registration fails
    assert "FP32" not in add_theta
    with burble():
        add_theta_fp32 = indexbinary.ss.register_new(
            "add_theta",
            cdef.replace("double", "float"),
            "FP32",
            "FP32",
            "FP32",
            "FP32",
        )
    assert add_theta_fp32 is add_theta
    assert "FP32" in add_theta
    with pytest.raises(
        TypeError,
        match="IndexBinaryOp gb.indexbinary.ss.add_theta already defined for .FP64, FP64. input",
    ):
        indexbinary.ss.register_new("add_theta", cdef, "FP64", "FP64", "FP64", "FP64")
    # Test nested names
    indexbinary.ss.register_new("nested.add_theta", cdef, "FP64", "FP64", "FP64", "FP64")
    with pytest.raises(AttributeError, match="nested is already defined"):
        indexbinary.ss.register_new("nested", cdef, "FP64", "FP64", "FP64", "FP64")
    # Test mixed types (x=FP64, y=FP64, theta=FP32)
    mixed_cdef = (
        "void add_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, "
        "double *y, GrB_Index iy, GrB_Index jy, float *theta) "
        "{ (*z) = (*x) + (*y) + (double)(*theta) ; }"
    )
    add_theta_mixed = indexbinary.ss.register_new(
        "add_theta", mixed_cdef, "FP64", "FP64", "FP32", "FP64"
    )
    assert add_theta_mixed is add_theta
    assert ("FP64", "FP32") in add_theta
    assert ("FP32", "FP64") not in add_theta


@pytest.mark.slow
def test_jit_select(v):
    cdef = (
        # SelectOps don't write to their input array, so SuiteSparse requires
        # the x argument to be ``const``.
        "void woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }"
    )
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            select.ss.register_new("woot", cdef, "INT32", "INT32")
        return
    _require_jit_on()
    with burble():
        woot = select.ss.register_new("woot", cdef, "INT32", "INT32")
    assert not hasattr(select, "woot")
    assert select.ss.woot is woot
    assert not hasattr(indexunary, "woot")
    assert hasattr(indexunary.ss, "woot")
    assert woot.name == "ss.woot"
    assert woot.types == {(dtypes.INT32, dtypes.INT32): dtypes.BOOL}
    assert "INT32" in woot
    assert woot["INT32"].return_type == dtypes.BOOL
    # JIT ops don't coerce: wrong dtype raises KeyError, not a silent cast.
    with pytest.raises(KeyError, match="woot does not work with .INT64, INT64. types"):
        v << woot(v, 1)
    v = v.dup("INT32")
    with pytest.raises(KeyError, match="woot does not work with .INT32, INT64. types"):
        v << woot(v, 6)
    res = woot(v, gb.Scalar.from_value(6, "INT32")).new()
    expected = Vector.from_coo([4, 6], [2, 0])
    assert expected.isequal(res)

    res = indexunary.ss.woot(v, gb.Scalar.from_value(6, "INT32")).new()
    expected = Vector.from_coo([1, 3, 4, 6], [False, False, True, True])
    assert expected.isequal(res)
    assert woot["INT32"].jit_c_definition == cdef

    assert "INT64" not in woot
    with burble():
        woot_int64 = select.ss.register_new(
            "woot", cdef.replace("int32", "int64"), "INT64", "INT64"
        )
    assert woot_int64 is woot
    assert "INT64" in woot
    with pytest.raises(TypeError, match="ss.woot already defined for .INT32, INT32. input types"):
        select.ss.register_new("woot", cdef, "INT32", "INT32")
    del indexunary.ss.woot
    with pytest.raises(TypeError, match="ss.woot already defined for .INT32, INT32. input types"):
        select.ss.register_new("woot", cdef, "INT32", "INT32")
    select.ss.register_new("nested.woot", cdef, "INT32", "INT32")
    with pytest.raises(AttributeError, match="nested is already defined"):
        select.ss.register_new("nested", cdef, "INT32", "INT32")
    del indexunary.ss.nested
    with pytest.raises(AttributeError, match="nested is already defined"):
        select.ss.register_new("nested", cdef.replace("woot", "nested"), "INT32", "INT32")
    select.ss.haha = "haha"
    with pytest.raises(AttributeError, match="haha is already defined"):
        select.ss.register_new("haha", cdef.replace("woot", "haha"), "INT32", "INT32")
    # Make sure we can be specific with left/right dtypes
    woot_mixed = select.ss.register_new(
        "woot",
        "void woot (bool *z, const int64_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }",
        "INT64",
        "INT32",
    )
    assert woot_mixed is woot
    assert ("INT64", "INT32") in woot
    assert ("INT32", "INT64") not in woot


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.skipif("_IS_SSGB7")
def test_udt_jit_c_source_introspection():
    """``jit_c_source`` and ``jit_c_name`` should expose what SS sees.

    Covers record UDTs, array UDTs, and ops where no JIT definition was set
    (built-in scalar ops, mixed UDT+scalar binary ops). Also verifies the
    matching ``jit_c_definition`` / ``jit_c_name`` properties on the dtype.

    Uses field names unique to this test to avoid colliding with other
    test UDTs. ``register_anonymous`` shares one DataType per ``np.dtype``,
    so two tests with the same dtype share state (including cached JIT C
    info) and would race when run in the same session.
    """
    record_dtype = np.dtype([("introsp_a", np.int64), ("introsp_b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_IntrospectUDT")

    # dtype-level introspection
    assert udt.jit_c_name == "_IntrospectUDT"
    assert "typedef struct" in udt.jit_c_definition
    assert "int64_t introsp_a" in udt.jit_c_definition
    assert "double introsp_b" in udt.jit_c_definition

    # Builtin scalar dtype: no JIT C definition
    assert dtypes.INT64.jit_c_name is None
    assert dtypes.INT64.jit_c_definition is None

    # Auto-lifted record UDT binary op
    plus_udt = binary.plus[udt]
    assert plus_udt.jit_c_name == "plus__IntrospectUDT"
    src = plus_udt.jit_c_source
    assert src is not None
    assert "z->introsp_a = (x->introsp_a) + (y->introsp_a)" in src
    assert "z->introsp_b = (x->introsp_b) + (y->introsp_b)" in src

    # Auto-lifted record UDT unary op
    ainv_udt = unary.ainv[udt]
    assert ainv_udt.jit_c_name == "ainv__IntrospectUDT"
    assert "z->introsp_a = -(x->introsp_a)" in ainv_udt.jit_c_source

    # Array UDT auto-lifted op. Use a shape no other test uses: ``np.dtype``
    # identity is shared across the session, so once SS sets
    # ``GxB_JIT_C_NAME`` we can't rename. A distinctive shape keeps the
    # C-side name stable for this test.
    arr_udt = dtypes.register_anonymous(np.dtype("(11,)float64"), "_IntrospectArr")
    times_arr = binary.times[arr_udt]
    src_arr = times_arr.jit_c_source
    assert src_arr is not None
    assert "z->v[0] = (x->v[0]) * (y->v[0])" in src_arr
    assert "z->v[10] = (x->v[10]) * (y->v[10])" in src_arr

    # Builtin scalar op: no JIT source
    assert binary.plus[int].jit_c_source is None
    assert binary.plus[int].jit_c_name is None
    assert unary.abs[float].jit_c_source is None

    # Monoid / Semiring / Aggregator walk-down: introspection should follow the
    # underlying binary op so users can chase the JIT'd source from any layer.
    plus_binop_src = binary.plus[udt].jit_c_source
    assert gb.monoid.plus[udt].jit_c_source == plus_binop_src
    # Semiring delegates to the multiplier; .monoid still works on its own.
    semi = gb.semiring.plus_times[udt]
    assert semi.jit_c_source == binary.times[udt].jit_c_source
    assert semi.monoid.jit_c_source == plus_binop_src
    # Monoid-based aggregator walks to its monoid; composite agg has no kernel.
    assert gb.agg.sum[udt].jit_c_source == plus_binop_src
    assert gb.agg.count[udt].jit_c_source is None


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.skipif("_IS_SSGB7")
def test_op_jit_signature_uses_pinned_type_name():
    """After a UDT is renamed, auto-lifted ops must still reference the
    pinned (first-registration) C name in their signature. SS's
    ``GxB_JIT_C_NAME`` on a ``GrB_Type`` is one-shot, so a mismatched op
    signature would reference an undefined struct and SS would silently
    fall back to the Numba cfunc.
    """
    record_dtype = np.dtype([("pinned_a", np.int64), ("pinned_b", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_PinnedOrig")
    udt2 = dtypes.register_anonymous(record_dtype, "_PinnedRenamed")
    assert udt2 is udt
    assert udt.name == "_PinnedRenamed"
    assert udt.jit_c_name == "_PinnedOrig"

    op = binary.plus[udt]
    # Op signature must reference the pinned struct name, not the renamed one.
    assert op.jit_c_name == "plus__PinnedOrig"
    assert "_PinnedOrig *" in op.jit_c_source
    assert "_PinnedRenamed" not in op.jit_c_source


@pytest.mark.skipif("not supports_udfs")
def test_jit_compiles_auto_udt_ops():
    """JIT must actually compile a kernel for an auto-generated UDT op.

    Regression for the bug where ``GxB_JIT_C_NAME`` wasn't being set on
    UDT types, so SuiteSparse's eWise JIT template (which depends on the
    type having a JIT name) failed to expand with errors like
    ``use of undeclared identifier 'Bx'``, disabling JIT for the rest of
    the session. Verifies both that (a) the op runs without JIT erroring
    out, and (b) ``jit_c_control`` stays ``"on"`` after the op.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    record_dtype = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_JitAutoUdt")

    v = gb.Vector(udt, 3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)
    w = v.dup()

    # Force JIT on and probe. If our typedef + JIT_C_NAME wiring is correct,
    # SuiteSparse should compile and load the kernel without errors.
    with _jit_mode("on"):
        with burble():
            result = binary.plus(v & w).new()
        assert result[0].new() == (2, 4.0)
        assert result[2].new() == (10, 12.0)
        # JIT must remain in compile-on mode; a failed compile flips it to 'load'.
        assert (
            gb.ss.config["jit_c_control"] == "on"
        ), "JIT compilation got disabled (flipped from 'on' to 'load') after a built-in UDT op"


@pytest.mark.skipif("not supports_udfs")
def test_floordiv_udt_jit_matches_python_semantics():
    """``binary.floordiv`` on a UDT must round toward minus infinity on the JIT path.

    Regression: the JIT codegen used to lower ``//`` to plain C ``/``, which
    is trunc-toward-zero for ints and true division for floats. For positive
    integer operands the two agreed; for negative operands or any float they
    silently disagreed with the Numba cfunc path (Python ``//`` is floor).

    Use N=50 non-iso vectors with mixed-sign operands to bypass any
    iso/short-vector shortcuts and exercise the actual JIT kernel.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    N = 50

    # float64 record: was returning 3.5 / -3.5 (true division) before the fix.
    # Use field names unique to this test so the anonymous-UDT cache (keyed
    # on np.dtype) doesn't share a DataType with another test, which would
    # also share the SS-side ``jit_c_name``.
    udt = dtypes.register_anonymous(
        np.dtype([("fd_a", np.float64), ("fd_b", np.float64)]), "_FdJitF64"
    )
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (-7.0 - i, 7.0 + i)
        u[i] = (2.0, 2.0)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7.0 // 2.0 == -4.0, 7.0 // 2.0 == 3.0
    assert w[0].new() == (-4.0, 3.0)
    # -8.0 // 2.0 == -4.0, 8.0 // 2.0 == 4.0
    assert w[1].new() == (-4.0, 4.0)

    # int64 record: was returning -3 (trunc-to-zero) for -7//2 before the fix.
    udt_i = dtypes.register_anonymous(np.dtype([("a", np.int64), ("b", np.int64)]), "_FdJitI64")
    v = gb.Vector(udt_i, N)
    u = gb.Vector(udt_i, N)
    for i in range(N):
        v[i] = (-7 - 2 * i, 7 + 2 * i)
        u[i] = (2, 3)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7 // 2 == -4, 7 // 3 == 2
    assert w[0].new() == (-4, 2)
    # -9 // 2 == -5, 9 // 3 == 3
    assert w[1].new() == (-5, 3)
    # -11 // 2 == -6, 11 // 3 == 3
    assert w[2].new() == (-6, 3)

    # Array UDT: same hazard on the flattened ``v[i]`` path.
    udt_arr = dtypes.register_anonymous(np.dtype((np.float32, (4,))), "_FdJitArrF32")
    v = gb.Vector(udt_arr, N)
    u = gb.Vector(udt_arr, N)
    for i in range(N):
        v[i] = np.array([-7.0 - i, 7.0 + i, -8.0 - i, 8.0 + i], dtype=np.float32)
        u[i] = np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7//2=-4, 7//2=3, -8//3=-3, 8//3=2
    assert_array_equal(w[0].new().value, np.array([-4.0, 3.0, -3.0, 2.0], dtype=np.float32))
    # -8//2=-4, 8//2=4, -9//3=-3, 9//3=3
    assert_array_equal(w[1].new().value, np.array([-4.0, 4.0, -3.0, 3.0], dtype=np.float32))


@pytest.mark.skipif("not supports_udfs")
def test_min_max_udt_jit_propagates_nan():
    """``binary.min``/``max`` on a float UDT must propagate NaN like Python/numba.

    Regression: the JIT codegen used to emit ``(a < b ? a : b)``, which
    silently swallows NaN to the right-hand side. Python ``min(a, b)`` (and
    numba's ``min``) returns ``a`` when neither comparison is true (NaN
    involved), so ``min(NaN, 1.0) == NaN`` and ``min(1.0, NaN) == 1.0``.
    The fix swaps the ternary to ``(b < a ? b : a)``.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    # Field names unique to this test; see floordiv test for the cache rationale.
    udt = dtypes.register_anonymous(
        np.dtype([("nan_a", np.float64), ("nan_b", np.float64)]), "_NanJitMM"
    )
    N = 100
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    nan = float("nan")
    for i in range(N):
        # field nan_a: NaN on the left; field nan_b: NaN on the right at odd indices.
        v[i] = (nan, 2.0 + i)
        u[i] = (1.0 + i, nan if i % 2 else 3.0 + i)

    w = v.ewise_mult(u, binary.min).new()
    assert np.isnan(w[0].new().value[0])  # min(NaN, 1.0) -> NaN
    assert w[1].new().value[1] == 3.0  # min(3.0, NaN) -> 3.0 (NaN swallowed)
    assert w[2].new().value[1] == 4.0  # min(4.0, 5.0) -> 4.0 (normal case)

    w = v.ewise_mult(u, binary.max).new()
    assert np.isnan(w[0].new().value[0])  # max(NaN, 1.0) -> NaN
    assert w[1].new().value[1] == 3.0  # max(3.0, NaN) -> 3.0 (NaN swallowed)
    assert w[2].new().value[1] == 5.0  # max(4.0, 5.0) -> 5.0 (normal case)


@pytest.mark.skipif("not supports_udfs")
def test_abs_udt_jit_matches_python_negative_zero():
    """``unary.abs`` on a float UDT must clear the sign bit of ``-0.0``.

    Regression: the JIT codegen used to emit ``(x < 0 ? -x : x)``. For
    ``x == -0.0`` the comparison ``-0.0 < 0`` is false (negative zero is
    not less than zero in IEEE-754), so the ternary returned ``-0.0``
    with the sign bit intact. Python ``abs(-0.0) == 0.0`` (sign bit
    cleared), and the cfunc path uses Python's ``abs``, so the two paths
    silently disagreed on the sign of zero. The fix routes float fields
    through ``fabs`` / ``fabsf``.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    import struct

    def sign_bit(f):
        return struct.pack("<d", f)[7] & 0x80

    # Field names unique to this test; see floordiv test for the cache rationale.
    udt = dtypes.register_anonymous(
        np.dtype([("abs_a", np.float64), ("abs_b", np.float64)]), "_AbsNegZeroJit"
    )
    N = 100
    v = gb.Vector(udt, N)
    for i in range(N):
        # field abs_a is always -0.0; field abs_b varies so the vector isn't iso.
        v[i] = (-0.0, float(i) - 50)

    w = unary.abs(v).new()
    val = w[0].new().value
    assert val[0] == 0.0
    # Sign-bit must be cleared (the bug returned -0.0, sign bit 0x80).
    assert (
        sign_bit(val[0]) == 0
    ), f"abs(-0.0) preserved the sign bit: bits={val[0].view('<u8'):016x}"


# JIT-vs-cfunc parity inputs, keyed by udt_kind.
# Each entry is (np.dtype-spec, "_TypeName", fill(i) -> (a, b)).
#
# Per-variant unique field names: anonymous UDTs are keyed on ``np.dtype``, so
# same-shape dtypes across variants would share a single ``DataType`` (and its
# pinned JIT C name from the first registration), causing test-order coupling.


def _parity_record_i64(i):
    # cross-sign division operands
    return (-7 - 2 * i, 7 + 2 * i), (2, -3)


def _parity_record_f64(i):
    nan = float("nan")
    inf = float("inf")
    if i == 0:
        return (nan, 1.0), (2.0, 3.0)
    if i == 1:
        return (1.0, nan), (inf, -inf)
    if i == 2:
        return (-7.5, 7.5), (2.0, 2.0)
    return (-7.0 - i, 7.0 + i), (2.0 + 0.1 * i, 3.0)


def _parity_record_u32(i):
    return (10 + i, 20 + i), (2, 3)


def _parity_record_mixed(i):
    return (5 + i, 1.5 + i), (2, 0.5)


def _parity_array_f32(i):
    return (
        np.array([-7.0 - i, 7.0 + i, -8.0 - i, 8.0 + i], np.float32),
        np.array([2.0, 2.0, 3.0, 3.0], np.float32),
    )


def _parity_array_i16(i):
    return (
        np.array([1 + i, 2 + i, 3 + i, 4 + i], np.int16),
        np.array([2, 2, 3, 3], np.int16),
    )


_PARITY_VARIANTS = {
    "record_i64": (
        np.dtype([("p_i_x", np.int64), ("p_i_y", np.int64)]),
        "_ParityI64",
        _parity_record_i64,
    ),
    "record_f64": (
        np.dtype([("p_f_x", np.float64), ("p_f_y", np.float64)]),
        "_ParityF64",
        _parity_record_f64,
    ),
    "record_u32": (
        np.dtype([("p_u_x", np.uint32), ("p_u_y", np.uint32)]),
        "_ParityU32",
        _parity_record_u32,
    ),
    "record_mixed": (
        np.dtype([("p_m_i", np.int32), ("p_m_f", np.float64)]),
        "_ParityMixed",
        _parity_record_mixed,
    ),
    "array_f32": (
        np.dtype((np.float32, (4,))),
        "_ParityArrF32",
        _parity_array_f32,
    ),
    "array_i16": (
        np.dtype((np.int16, (4,))),
        "_ParityArrI16",
        _parity_array_i16,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("udt_kind", list(_PARITY_VARIANTS))
def test_udt_op_jit_cfunc_parity(udt_kind):
    """JIT and cfunc paths produce the same result on every auto-lifted op.

    The two paths are independent code generators (string-templated C vs Numba
    njit). N=64 sidesteps SS's iso/short-vector shortcuts so the kernel actually
    runs. The variants cover signed-int floor formula, float floor/NaN/inf,
    unsigned trunc-div, mixed-width with C alignment, and both array shapes.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()
    if numba is None:
        pytest.skip("numba required for the cfunc baseline")

    np_dtype, type_name, fill = _PARITY_VARIANTS[udt_kind]
    udt = dtypes.register_anonymous(np_dtype, type_name)
    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        a, b = fill(i)
        v[i] = a
        u[i] = b

    binary_ops = ["plus", "minus", "times", "truediv", "floordiv", "min", "max"]
    unary_ops = ["ainv", "abs"]

    with _jit_mode("on"):
        jit_binary = {op: v.ewise_mult(u, getattr(binary, op)).new() for op in binary_ops}
        jit_unary = {op: getattr(unary, op)(v).new() for op in unary_ops}
    # cfunc path: JIT off so SS uses the registered function pointer instead of
    # compiling a new kernel.
    with _jit_mode("off"):
        cf_binary = {op: v.ewise_mult(u, getattr(binary, op)).new() for op in binary_ops}
        cf_unary = {op: getattr(unary, op)(v).new() for op in unary_ops}

    def values_equal(j, c):
        # Compare raw bytes via ``to_dense`` rather than ``isequal``. With the
        # IEEE-aware ``binary.eq[udt]`` fix, two NaN-bearing records compare
        # unequal under ``isequal``, so ``isequal`` can't distinguish "JIT and
        # cfunc agree on a NaN bit-pattern" from "they disagree". Byte-equality
        # of the dense numpy view captures the parity question correctly.
        if j.dtype != c.dtype or j.nvals != c.nvals:
            return False
        return j.to_dense().tobytes() == c.to_dense().tobytes()

    for op in binary_ops:
        assert values_equal(
            jit_binary[op], cf_binary[op]
        ), f"binary.{op} on {udt_kind}: JIT and cfunc disagree"
    for op in unary_ops:
        assert values_equal(
            jit_unary[op], cf_unary[op]
        ), f"unary.{op} on {udt_kind}: JIT and cfunc disagree"


@pytest.mark.skipif("not supports_udfs")
def test_anonymous_udt_with_no_name_still_jits():
    """A UDT registered without ``name=`` should still take the JIT path.

    ``register_anonymous(np.dtype)`` gives the DataType a Python-side name
    like ``"{'a': FP64, 'b': INT64}"`` that isn't a valid C identifier, so
    SS can't use it as the JIT type name. ``_pick_c_type_name`` synthesizes
    a ``_gbudt_NNN`` name for SuiteSparse instead, leaving ``udt.name``
    alone for Python-side display. If that fallback regresses, the op runs
    through the slower Numba cfunc path and ``jit_c_source`` is ``None``.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    spec = np.dtype([("anon_a", np.float64), ("anon_b", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec)  # no name=

    # Python-side default name is the np.dtype repr; not a valid C identifier.
    assert udt.name != udt.jit_c_name
    assert udt.jit_c_name is not None
    assert udt.jit_c_name.startswith("_gbudt_")
    assert udt.jit_c_definition is not None

    # The synthetic name flows through to op codegen so the JIT path actually
    # fires and matches the cfunc path.
    plus_op = binary.plus[udt]
    assert plus_op.jit_c_source is not None
    assert plus_op.jit_c_name.startswith("plus__gbudt_")

    # Run the kernel to confirm the JIT path produces correct results.
    v = gb.Vector(udt, 2)
    v[0] = (1.0, 10)
    v[1] = (2.0, 20)
    w = gb.Vector(udt, 2)
    w[0] = (3.0, 30)
    w[1] = (4.0, 40)
    result = v.ewise_mult(w, binary.plus).new()
    assert result[0].new().value.tolist() == (4.0, 40)
    assert result[1].new().value.tolist() == (6.0, 60)


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_complex_field_udt_jits_arithmetic():
    """Complex fields take the JIT path for ``plus``/``minus``/``times``/
    ``truediv``/``abs``/``ainv``. ``min``/``max``/``floordiv`` are rejected
    early with a clear error rather than a numba LLVM crash.

    The C99 ``_Complex`` types arrive via SS's GraphBLAS.h (typedef'd to
    ``GxB_FC32_t`` / ``GxB_FC64_t``), so the generated kernel compiles
    without extra includes. ``abs`` uses ``cabs`` / ``cabsf`` which match
    Numba's ``abs(complex)`` behavior (real magnitude in the real part).
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    spec = np.dtype([("cval", np.complex128), ("scale", np.float64)])
    udt = dtypes.register_anonymous(spec, "_JitCplx1")
    assert "GxB_FC64_t" in udt.jit_c_definition

    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (complex(1.0 + i, 2.0), 3.0 + i)
        u[i] = (complex(0.5, 0.25 + i), 0.1)

    # JIT must agree with cfunc bit-exactly on ops that lower to single C
    # operators. ``truediv`` is intentionally excluded: C99 ``_Complex``
    # division and Numba's complex division differ in their last-bit
    # rounding / overflow handling, so they're equivalent within ulps but
    # not bit-identical.
    for op_name in ("plus", "minus", "times"):
        op = getattr(binary, op_name)
        with _jit_mode("on"):
            wj = v.ewise_mult(u, op).new()
        with _jit_mode("off"):
            wc = v.ewise_mult(u, op).new()
        assert wj.isequal(wc, check_dtype=True), f"binary.{op_name} on complex UDT diverged"
    for op_name in ("abs", "ainv"):
        op = getattr(unary, op_name)
        with _jit_mode("on"):
            wj = op(v).new()
        with _jit_mode("off"):
            wc = op(v).new()
        assert wj.isequal(wc, check_dtype=True), f"unary.{op_name} on complex UDT diverged"

    # abs source must use cabs (not the ternary, which doesn't compile on _Complex).
    assert "cabs(" in unary.abs[udt].jit_c_source

    # min/max/floordiv reject early with a clear KeyError citing complex.
    for op_name in ("min", "max", "floordiv"):
        with pytest.raises(KeyError, match="complex fields"):
            getattr(binary, op_name)[udt]


@pytest.mark.skipif("not supports_udfs")
def test_packed_record_layout_skips_jit():
    """A packed numpy dtype (no inter-field padding) must skip the JIT path.

    Otherwise the JIT-compiled kernel reads fields at the C-aligned offsets
    while the numpy buffer holds them at packed offsets, producing silent
    garbage. The cfunc path still works because Numba mirrors numpy's
    layout. Users wanting JIT can re-register with ``align=True`` or use
    the dict / dataclass form (which auto-aligns).
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")

    # int32 + float64 packed: numpy offsets 0, 4 (itemsize 12);
    # C struct would put float64 at offset 8 (itemsize 16).
    packed = np.dtype([("p_flag", np.int32), ("p_val", np.float64)])  # no align=True
    udt = dtypes.register_anonymous(packed, "_PackedUdt1")
    assert udt.jit_c_name is None  # JIT skipped
    assert udt.jit_c_definition is None

    # cfunc still works (the existing fallback path).
    v = gb.Vector(udt, 4)
    v[0] = (1, 1.5)
    v[1] = (2, 2.5)
    w = v.ewise_mult(v, binary.plus).new()
    assert w[0].new() == (2, 3.0)


@pytest.mark.skipif("not supports_udfs")
def test_packed_nested_record_layout_skips_jit():
    """Packed-inside-packed must skip the JIT path too.

    Regression: ``_is_c_compatible_layout`` originally rebuilt the
    comparison dtype with ``np.dtype([(name, inner_dtype), ...], align=True)``,
    but numpy's ``align=True`` respects the *inner* dtype's own alignment.
    When the inner is itself packed (the default), it stays packed with
    alignment 1, and the outer aligned reconstruction happens to match
    the original packed layout. The JIT then produced garbage because a
    C compiler aligns the inner to its natural 8-byte boundary.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")

    spec = np.dtype(
        [
            ("flag", np.int32),
            ("coord", [("x", np.float64), ("y", np.float64)]),
        ],
    )  # no align=True at either level
    # numpy itemsize 20 (flag@0, coord@4); C would pick 24 (flag@0, pad, coord@8).
    assert spec.itemsize == 20
    udt = dtypes.register_anonymous(spec, "_PackedNestedUdt")
    assert udt.jit_c_name is None  # JIT must be skipped
    assert udt.jit_c_definition is None


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_nested_record_udt_jits():
    """Record-of-record UDTs take the JIT path.

    The codegen emits inner typedefs first in ``GxB_JIT_C_DEFINITION``, with
    synthesized ``_gbnest_NNN`` names to avoid collisions with user types.
    Op bodies use C nested-struct access (``z->outer.inner_a``), and the
    Numba cfunc wrapper writes the flat result tuple leaf-by-leaf back into
    the nested record fields (Numba can't ``setitem`` a tuple to a record
    field, but leaf scalar writes work).
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()

    # ``align=True`` is required for mixed-width fields so numpy's offsets
    # match what a C compiler would produce. Without it, the JIT kernel
    # would read fields at the wrong offsets and produce garbage; the
    # codegen now refuses JIT in that case but the test wants the JIT path
    # exercised.
    spec = np.dtype(
        [
            ("nest_flag", np.int32),
            ("nest_coord", [("nest_x", np.float64), ("nest_y", np.float64)]),
        ],
        align=True,
    )
    udt = dtypes.register_anonymous(spec, "_JitNested1")

    # Outer + inner typedefs both appear in the definition; the outer
    # references the inner by its synthesized C name.
    defn = udt.jit_c_definition
    assert defn.count("typedef struct") == 2
    assert "_gbnest_" in defn
    assert "_JitNested1" in defn

    plus_src = binary.plus[udt].jit_c_source
    # Nested struct access uses ``.`` after the leading ``->``.
    assert "z->nest_coord.nest_x = (x->nest_coord.nest_x) + (y->nest_coord.nest_x)" in plus_src

    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (i, (1.0 + i, 2.0 + i))
        u[i] = (1, (0.5, -0.5))

    for op_name in ("plus", "minus", "times"):
        op = getattr(binary, op_name)
        with _jit_mode("on"):
            wj = v.ewise_mult(u, op).new()
        with _jit_mode("off"):
            wc = v.ewise_mult(u, op).new()
        assert wj.isequal(wc, check_dtype=True), f"binary.{op_name} on nested UDT diverged"
    for op_name in ("abs", "ainv"):
        op = getattr(unary, op_name)
        with _jit_mode("on"):
            wj = op(v).new()
        with _jit_mode("off"):
            wc = op(v).new()
        assert wj.isequal(wc, check_dtype=True), f"unary.{op_name} on nested UDT diverged"


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_nested_record_monoid_semiring_agg():
    r"""Built-in monoid/semiring/agg auto-lift on nested record UDTs.

    Regression: ``_udt_identity`` flattened nested-record fields by calling
    ``_scalar_identity`` directly on each field, but for a record-valued
    field that returned ``numpy.void(b'\x00')`` (uninitialized bytes), and
    ``Scalar.value =`` then choked. Recurse into nested records so the
    identity tuple matches the dtype's nested shape.
    """
    spec = np.dtype(
        [
            ("agg_id", np.int32),
            ("agg_pt", [("agg_x", np.float64), ("agg_y", np.float64)]),
        ],
        align=True,
    )
    udt = dtypes.register_anonymous(spec, "_NestedMonoid1")

    plus_id = monoid.plus[udt].identity.value
    assert plus_id["agg_id"] == 0
    assert plus_id["agg_pt"]["agg_x"] == 0.0
    assert plus_id["agg_pt"]["agg_y"] == 0.0
    min_id = monoid.min[udt].identity.value
    assert min_id["agg_id"] == np.iinfo(np.int32).max
    assert min_id["agg_pt"]["agg_x"] == np.inf
    assert min_id["agg_pt"]["agg_y"] == np.inf
    max_id = monoid.max[udt].identity.value
    assert max_id["agg_id"] == np.iinfo(np.int32).min
    assert max_id["agg_pt"]["agg_x"] == -np.inf
    assert max_id["agg_pt"]["agg_y"] == -np.inf

    v = gb.Vector(udt, size=3)
    v[0] = (1, (1.0, 2.0))
    v[1] = (2, (3.0, 4.0))
    v[2] = (3, (5.0, 6.0))
    s = v.reduce(agg.sum[udt]).new().value
    assert s["agg_id"] == 6
    assert s["agg_pt"]["agg_x"] == 9.0
    assert s["agg_pt"]["agg_y"] == 12.0

    A = gb.Matrix(udt, nrows=2, ncols=2)
    B = gb.Matrix(udt, nrows=2, ncols=2)
    A[0, 0] = (1, (1.0, 0.0))
    A[0, 1] = (2, (0.0, 1.0))
    A[1, 0] = (3, (1.0, 1.0))
    A[1, 1] = (4, (1.0, 1.0))
    B[0, 0] = (10, (1.0, 0.0))
    B[0, 1] = (20, (0.0, 1.0))
    B[1, 0] = (30, (1.0, 1.0))
    B[1, 1] = (40, (1.0, 1.0))
    C = A.mxm(B, semiring.plus_times[udt]).new()
    c00 = C[0, 0].new().value
    c11 = C[1, 1].new().value
    assert (c00["agg_id"], c00["agg_pt"]["agg_x"], c00["agg_pt"]["agg_y"]) == (70, 1.0, 1.0)
    assert (c11["agg_id"], c11["agg_pt"]["agg_x"], c11["agg_pt"]["agg_y"]) == (220, 1.0, 2.0)


# Per-shape fill functions for the eq/ne JIT-vs-cfunc parity test below.
# Each returns ``(v1_record, v2_record)`` for index ``i``. Variants that put
# a NaN at index 0 also exercise IEEE comparison semantics (eq(NaN,NaN) is
# False), pinned by the ``nan_at_zero`` flag in ``_EQ_NE_VARIANTS``.


def _eq_ne_fill_record_float(i):
    nan = float("nan")
    if i == 0:
        return (nan, 1.0), (nan, 1.0)
    return (1.0 + i, 2.0), (1.0 + i, 2.0 if i % 2 == 0 else 3.0)


def _eq_ne_fill_record_int(i):
    return (i, 2 * i), (i, 2 * i if i % 3 == 0 else 2 * i + 1)


def _eq_ne_fill_record_nested(i):
    return (i, (1.0, 2.0)), (i, (1.0, 2.0) if i % 3 != 0 else (1.5, 2.0))


def _eq_ne_fill_record_complex(i):
    return (
        (complex(1.0 + i, 2.0), 3.0),
        (complex(1.0 + i, 2.0 if i % 2 == 0 else 3.0), 3.0),
    )


def _eq_ne_fill_array_f64(i):
    return (
        np.array([1.0 + i, 2.0, 3.0]),
        np.array([1.0 + i, 2.0 if i % 2 == 0 else -2.0, 3.0]),
    )


_EQ_NE_VARIANTS = {
    "record_float": (
        np.dtype([("eq_f_a", np.float64), ("eq_f_b", np.float64)], align=True),
        "_EqJitF",
        _eq_ne_fill_record_float,
        True,
    ),
    "record_int": (
        np.dtype([("eq_i_a", np.int64), ("eq_i_b", np.int32)], align=True),
        "_EqJitI",
        _eq_ne_fill_record_int,
        False,
    ),
    "record_nested": (
        np.dtype(
            [("eq_n_id", np.int32), ("eq_n_pt", [("nx", np.float64), ("ny", np.float64)])],
            align=True,
        ),
        "_EqJitN",
        _eq_ne_fill_record_nested,
        False,
    ),
    "record_complex": (
        np.dtype([("eq_c_z", np.complex128), ("eq_c_s", np.float64)]),
        "_EqJitC",
        _eq_ne_fill_record_complex,
        False,
    ),
    "array_f64": (
        np.dtype((np.float64, (3,))),
        "_EqJitArr",
        _eq_ne_fill_array_f64,
        False,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("shape", list(_EQ_NE_VARIANTS))
def test_eq_ne_udt_jit_matches_cfunc(shape):
    """``binary.eq[udt]`` / ``binary.ne[udt]`` JIT-compile a leaf-wise
    comparison kernel and produce the same result as the cfunc fallback.

    Verifies the JIT kernel for several shapes (flat record, nested
    record, complex, array UDT), and that NaN propagation matches IEEE
    (``eq(NaN, NaN) == False``) on variants flagged ``nan_at_zero``.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    _require_jit_on()
    if numba is None:
        pytest.skip("numba required for the cfunc baseline")

    np_dtype, type_name, fill, nan_at_zero = _EQ_NE_VARIANTS[shape]
    udt = dtypes.register_anonymous(np_dtype, type_name)
    N = 64
    v1 = gb.Vector(udt, N)
    v2 = gb.Vector(udt, N)
    for i in range(N):
        a, b = fill(i)
        v1[i] = a
        v2[i] = b

    # The C kernel must actually be set, not None (i.e. JIT path is engaged).
    assert binary.eq[udt].jit_c_source is not None
    assert binary.ne[udt].jit_c_source is not None

    with _jit_mode("on"):
        eq_jit = v1.ewise_mult(v2, binary.eq).new()
        ne_jit = v1.ewise_mult(v2, binary.ne).new()
    with _jit_mode("off"):
        eq_cf = v1.ewise_mult(v2, binary.eq).new()
        ne_cf = v1.ewise_mult(v2, binary.ne).new()

    # Byte-equal across paths (the result is BOOL, so this is exact).
    assert (
        eq_jit.to_dense().tobytes() == eq_cf.to_dense().tobytes()
    ), f"eq on {shape}: JIT and cfunc disagree"
    assert (
        ne_jit.to_dense().tobytes() == ne_cf.to_dense().tobytes()
    ), f"ne on {shape}: JIT and cfunc disagree"

    if nan_at_zero:
        # Both records carry NaN at index 0; IEEE ``a == a`` is False.
        assert eq_jit[0].new().value is False
        assert ne_jit[0].new().value is True


@pytest.mark.skipif("not supports_udfs")
def test_eq_ne_udt_cfunc_works_for_unrepresentable_dtypes():
    """``binary.eq`` cfunc path produces correct results on UDTs that can't take JIT.

    The udt-level skip is pinned by ``test_packed_record_layout_skips_jit``
    and ``test_udt_with_c_reserved_field_name_falls_back_to_cfunc``; this
    test pins that the eq codegen (separate from plus codegen) still
    handles those UDTs end-to-end.
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    # Packed mixed-width: numpy offset 4 for float, C would use 8.
    packed = np.dtype([("eq_pk_a", np.int32), ("eq_pk_b", np.float64)])
    udt = dtypes.register_anonymous(packed, "_EqJitPacked")
    assert binary.eq[udt].jit_c_source is None
    v1 = gb.Vector(udt, 2)
    v2 = gb.Vector(udt, 2)
    v1[0] = (1, 2.5)
    v2[0] = (1, 2.5)
    v1[1] = (1, 2.5)
    v2[1] = (1, 3.5)
    eq = v1.ewise_mult(v2, binary.eq).new()
    assert eq[0].new().value is True
    assert eq[1].new().value is False


@pytest.mark.skipif("not supports_udfs")
def test_udt_scalar_broadcast_skips_jit():
    """Mixed UDT+scalar arithmetic and eq/ne stay on the cfunc path.

    The JIT codegen helpers (``set_jit_c_on_op`` and
    ``set_jit_c_comparison_on_op``) only run when ``dtype == dtype2``. A
    mixed pair would need a per-side type in the C signature plus one C
    name per (op, left_type, right_type) triple; for now we keep the
    cfunc fallback for broadcast and skip JIT. This test pins that
    routing so a future change doesn't accidentally regress it (either
    direction is fine, but it should be a conscious decision).
    """
    if _IS_SSGB7:
        pytest.skip("JIT requires SuiteSparse:GraphBLAS >= 8")
    record = dtypes.register_anonymous(
        np.dtype([("bcast_u", np.float64), ("bcast_v", np.float64)], align=True),
        name="_BcastJitUV",
    )
    # Same-type pairs do JIT.
    assert binary.plus[record, record].jit_c_source is not None
    assert binary.eq[record, record].jit_c_source is not None
    # Mixed pairs do not JIT (cfunc handles broadcast correctly; this is
    # the documented limit, not an oversight).
    assert binary.plus[record, dtypes.INT64].jit_c_source is None
    assert binary.plus[dtypes.INT64, record].jit_c_source is None
    assert binary.eq[record, dtypes.INT64].jit_c_source is None
    assert binary.ne[record, dtypes.FP64].jit_c_source is None


def test_context_importable():
    if _IS_SSGB7:
        with pytest.raises(ImportError, match="Context was added"):
            from graphblas.core.ss.context import global_context as _  # noqa: F401
        assert not hasattr("gb.ss", "global_context")
        return
    from graphblas.core.ss.context import global_context

    assert gb.ss.global_context is global_context

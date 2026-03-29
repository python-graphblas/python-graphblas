import os
import pathlib
import re
import sysconfig

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import backend, binary, dtypes, indexunary, select, unary
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.ss import _IS_SSGB7

from .conftest import autocompute, burble

from graphblas import Vector  # isort:skip (for dask-graphblas)

try:
    import numba
except ImportError:
    numba = None

if backend != "suitesparse":
    pytest.skip("not suitesparse backend", allow_module_level=True)


def _fix_jit_config():
    """Fix the GraphBLAS JIT configuration for the current conda environment.

    The graphblas C library bakes in build-time compiler paths from conda-build,
    which don't exist in the user's environment. This function:
    1. Replaces the compiler path with the equivalent from $CONDA_PREFIX/bin/
    2. Strips -isysroot flags pointing to non-existent build SDKs (macOS)
    3. Strips -fdebug-prefix-map flags referencing build paths

    Returns True if a working compiler was found, False otherwise.
    Only modifies jit_c_compiler_name, jit_c_compiler_flags, and jit_c_control.
    Linker flags and libraries are left at their defaults (already have correct
    $CONDA_PREFIX paths substituted by the graphblas C library).
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return False

    # Fix compiler name: replace build-time path with local conda equivalent
    jit_cc = gb.ss.config["jit_c_compiler_name"]
    cc_basename = pathlib.Path(jit_cc).name
    bin_dir = pathlib.Path(conda_prefix) / "bin"
    # Try exact name first, then 'cc' fallback
    for candidate in [cc_basename, "cc"]:
        local_cc = bin_dir / candidate
        if local_cc.exists():
            break
    else:
        return False
    gb.ss.config["jit_c_compiler_name"] = str(local_cc)

    # Fix compiler flags: remove build-time-only flags that reference paths
    # that don't exist in the user's environment
    flags = gb.ss.config["jit_c_compiler_flags"]
    # -isysroot <path>: macOS SDK path from conda-build (e.g., /opt/conda-sdks/MacOSX10.13.sdk)
    flags = re.sub(r"-isysroot\s+\S+", "", flags)
    # -fdebug-prefix-map=<build_path>=<src>: debug path remapping from conda-build
    flags = re.sub(r"-fdebug-prefix-map=\S+", "", flags)
    gb.ss.config["jit_c_compiler_flags"] = flags

    gb.ss.config["jit_c_control"] = "on"
    return True


@pytest.fixture(scope="module", autouse=True)
def _setup_jit():
    """Set up the SuiteSparse:GraphBLAS JIT.

    Strategy: try _fix_jit_config() first (works when psg is from conda-forge).
    If that fails (no conda env, or psg from wheel/source), fall through to sysconfig.
    If neither works, turn JIT off.
    """
    if _IS_SSGB7:
        # SuiteSparse JIT was added in SSGB 8
        yield
        return

    prev = gb.ss.config["jit_c_control"]

    if _fix_jit_config():
        try:
            yield
        finally:
            gb.ss.config["jit_c_control"] = prev
        return

    # Fallback: try sysconfig (for non-conda environments)
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
    else:
        gb.ss.config["jit_c_control"] = "off"
    try:
        yield
    finally:
        gb.ss.config["jit_c_control"] = prev


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
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
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
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
    with burble():
        square = unary.ss.register_new("square", cdef, "FP32", "FP32")
    assert not hasattr(unary, "square")
    assert unary.ss.square is square
    assert square.name == "ss.square"
    assert square.types == {dtypes.FP32: dtypes.FP32}
    # The JIT is unforgiving and does not coerce--use the correct types!
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


def test_jit_binary(v):
    cdef = "void absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }"
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            binary.ss.register_new("absdiff", cdef, "FP64", "FP64", "FP64")
        return
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
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
    # The JIT is unforgiving and does not coerce--use the correct types!
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


def test_jit_indexunary(v):
    cdef = (
        "void diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }"
    )
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            indexunary.ss.register_new("diffy", cdef, "FP64", "FP64", "FP64")
        return
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
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
    # The JIT is unforgiving and does not coerce--use the correct types!
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


def test_jit_select(v):
    cdef = (
        # Why does this one insist on `const` for `x` argument?
        "void woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }"
    )
    if _IS_SSGB7:
        with pytest.raises(RuntimeError, match="JIT was added"):
            select.ss.register_new("woot", cdef, "INT32", "INT32")
        return
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
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
    # The JIT is unforgiving and does not coerce--use the correct types!
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


def test_context_importable():
    if _IS_SSGB7:
        with pytest.raises(ImportError, match="Context was added"):
            from graphblas.core.ss.context import global_context as _  # noqa: F401
        assert not hasattr("gb.ss", "global_context")
        return
    from graphblas.core.ss.context import global_context

    assert gb.ss.global_context is global_context

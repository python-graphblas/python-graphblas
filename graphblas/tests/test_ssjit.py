import os
import pathlib
import re
import subprocess
import sysconfig

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import backend, binary, dtypes, indexbinary, indexunary, select, unary
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


def _fix_jit_config():
    """Fix the GraphBLAS JIT configuration for the current conda environment.

    The graphblas C library bakes in build-time compiler paths from conda-build,
    which don't exist in the user's environment. This function:
    1. Replaces the compiler path with the equivalent from $CONDA_PREFIX/bin/
    2. Replaces -isysroot with the local macOS SDK (via xcrun), or strips it on Linux
    3. Strips -fdebug-prefix-map flags referencing build paths

    Returns
    -------
        True:  JIT configured and verified working
        False: JIT configuration attempted but compilation failed (don't retry)
        None:  No conda environment; caller should try a different approach

    Only modifies jit_c_compiler_name, jit_c_compiler_flags, and jit_c_control.
    Linker flags and libraries are left at their defaults (already have correct
    $CONDA_PREFIX paths substituted by the graphblas C library).
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return None  # No conda env; caller should try sysconfig instead

    # Check if the default compiler already works (e.g., /usr/bin/cc on macOS).
    # Only replace it if the baked-in path doesn't exist.
    jit_cc = gb.ss.config["jit_c_compiler_name"]
    if pathlib.Path(jit_cc).exists():
        # Default compiler exists — don't replace it, just fix flags
        pass
    else:
        # Replace build-time path with local conda equivalent.
        cc_basename = pathlib.Path(jit_cc).name
        bin_dir = pathlib.Path(conda_prefix) / "bin"
        for candidate in [cc_basename, "cc", "clang", "gcc"]:
            local_cc = bin_dir / candidate
            if local_cc.exists():
                break
        else:
            return False
        gb.ss.config["jit_c_compiler_name"] = str(local_cc)

    # Fix compiler flags: fix build-time-only paths that don't exist in the
    # user's environment
    flags = gb.ss.config["jit_c_compiler_flags"]
    # -isysroot <path>: macOS SDK path from conda-build (e.g., /opt/conda-sdks/MacOSX10.13.sdk).
    # The conda cross-compiler needs an explicit sysroot. Replace with the local SDK if available.
    isysroot_match = re.search(r"-isysroot\s+(\S+)", flags)
    if isysroot_match and not pathlib.Path(isysroot_match.group(1)).exists():
        try:
            sdk_path = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            flags = re.sub(r"-isysroot\s+\S+", f"-isysroot {sdk_path}", flags)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No Xcode SDK available (linux, or macOS without Xcode CLT)
            flags = re.sub(r"-isysroot\s+\S+", "", flags)
    # -fdebug-prefix-map=<build_path>=<src>: debug path remapping from conda-build
    flags = re.sub(r"-fdebug-prefix-map=\S+", "", flags)
    gb.ss.config["jit_c_compiler_flags"] = flags

    gb.ss.config["jit_c_control"] = "on"

    # Verify the JIT actually works by attempting a trivial compilation.
    # If it fails (e.g., missing libraries, wrong flags), turn JIT off.
    try:
        from graphblas import dtypes as _dtypes

        _dtypes.ss.register_new("_jit_probe", "typedef struct { int _probe ; } _jit_probe ;")
    except Exception:
        gb.ss.config["jit_c_control"] = "off"
        return False
    else:
        return True


@pytest.fixture(scope="module", autouse=True)
def _setup_jit():
    """Set up the SuiteSparse:GraphBLAS JIT.

    Strategy:
    1. _fix_jit_config(): fix conda-baked compiler paths and probe.
       - Returns True: JIT works, proceed.
       - Returns False: probe failed, JIT is broken, turn off.
       - Returns None: no conda env, try sysconfig instead.
    2. Sysconfig fallback: for non-conda installs (pure pip).
    """
    if _IS_SSGB7:
        # SuiteSparse JIT was added in SSGB 8
        yield
        return

    prev = gb.ss.config["jit_c_control"]

    result = _fix_jit_config()
    if result is True:
        pass  # Conda JIT configured and verified
    elif result is False:
        # Probe failed — JIT doesn't work with this psg build.
        # Don't try sysconfig; if the conda compiler can't compile
        # GraphBLAS JIT kernels, Python's sysconfig compiler won't either.
        gb.ss.config["jit_c_control"] = "off"
    else:
        # No conda env (result is None). Try sysconfig for non-conda installs.
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


@pytest.mark.skipif(not _has_idxbinop, reason="requires SuiteSparse:GraphBLAS 9.4+")
def test_jit_indexbinary(v):
    cdef = (
        "void add_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, "
        "double *y, GrB_Index iy, GrB_Index jy, double *theta) "
        "{ (*z) = (*x) + (*y) + (*theta) ; }"
    )
    if gb.ss.config["jit_c_control"] == "off":
        pytest.skip("JIT not available (no C compiler configured)")
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

import os
import pathlib
import sys

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


@pytest.fixture(scope="module", autouse=True)
def _setup_jit():
    # Configuration values below were obtained from the output of the JIT config
    # in CI, but with paths changed to use `{conda_prefix}` where appropriate.
    if "CONDA_PREFIX" not in os.environ or _IS_SSGB7:
        yield
        return
    conda_prefix = os.environ["CONDA_PREFIX"]
    prev = gb.ss.config["jit_c_control"]
    gb.ss.config["jit_c_control"] = "on"
    if sys.platform == "linux":
        gb.ss.config["jit_c_compiler_name"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-cc"
        gb.ss.config["jit_c_compiler_flags"] = (
            "-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong "
            f"-fno-plt -O2 -ffunction-sections -pipe -isystem {conda_prefix}/include -Wundef "
            "-std=c11 -lm -Wno-pragmas -fexcess-precision=fast -fcx-limited-range "
            "-fno-math-errno -fwrapv -O3 -DNDEBUG -fopenmp -fPIC"
        )
        gb.ss.config["jit_c_linker_flags"] = (
            "-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now "
            "-Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined "
            f"-Wl,-rpath,{conda_prefix}/lib -Wl,-rpath-link,{conda_prefix}/lib "
            f"-L{conda_prefix}/lib -shared"
        )
        gb.ss.config["jit_c_libraries"] = (
            f"-lm -ldl {conda_prefix}/lib/libgomp.so "
            f"{conda_prefix}/x86_64-conda-linux-gnu/sysroot/usr/lib/libpthread.so"
        )
        gb.ss.config["jit_c_cmake_libs"] = (
            f"m;dl;{conda_prefix}/lib/libgomp.so;"
            f"{conda_prefix}/x86_64-conda-linux-gnu/sysroot/usr/lib/libpthread.so"
        )
    elif sys.platform == "darwin":
        gb.ss.config["jit_c_compiler_name"] = f"{conda_prefix}/bin/clang"
        gb.ss.config["jit_c_compiler_flags"] = (
            "-march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE "
            f"-fstack-protector-strong -O2 -pipe -isystem {conda_prefix}/include -DGBNCPUFEAT "
            "-Wno-pointer-sign -O3 -DNDEBUG -fopenmp=libomp -fPIC -arch x86_64"
        )
        gb.ss.config["jit_c_linker_flags"] = (
            "-Wl,-pie -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs "
            f"-Wl,-rpath,{conda_prefix}/lib -L{conda_prefix}/lib -dynamiclib"
        )
        gb.ss.config["jit_c_libraries"] = f"-lm -ldl {conda_prefix}/lib/libomp.dylib"
        gb.ss.config["jit_c_cmake_libs"] = f"m;dl;{conda_prefix}/lib/libomp.dylib"
    elif sys.platform == "win32":  # pragma: no branch (sanity)
        if "mingw" in gb.ss.config["jit_c_libraries"]:
            # This probably means we're testing a `python-suitesparse-graphblas` wheel
            # in a conda environment. This is not yet working.
            gb.ss.config["jit_c_control"] = "off"
            yield
            return

        gb.ss.config["jit_c_compiler_name"] = f"{conda_prefix}/bin/cc"
        gb.ss.config["jit_c_compiler_flags"] = (
            '/DWIN32 /D_WINDOWS -DGBNCPUFEAT /O2 -wd"4244" -wd"4146" -wd"4018" '
            '-wd"4996" -wd"4047" -wd"4554" /O2 /Ob2 /DNDEBUG -openmp'
        )
        gb.ss.config["jit_c_linker_flags"] = "/machine:x64"
        gb.ss.config["jit_c_libraries"] = ""
        gb.ss.config["jit_c_cmake_libs"] = ""

    if not pathlib.Path(gb.ss.config["jit_c_compiler_name"]).exists():
        # Can't use the JIT if we don't have a compiler!
        gb.ss.config["jit_c_control"] = "off"
        yield
        return
    yield
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
        return
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
        return
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
        return
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
        return
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
        return
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

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from graphblas import backend, binary, dtypes, indexunary, select, unary
from graphblas.core import _supports_udfs as supports_udfs

from .conftest import autocompute

from graphblas import Vector  # isort:skip (for dask-graphblas)

if backend != "suitesparse":
    pytest.skip("not suitesparse", allow_module_level=True)


@pytest.fixture
def v():
    return Vector.from_coo([1, 3, 4, 6], [1, 1, 2, 0])


@autocompute
def test_jit_udt():
    dtype = dtypes.ss.register_new(
        "myquaternion", "typedef struct { float x [4][4] ; int color ; } myquaternion ;"
    )
    assert not hasattr(dtypes, "myquaternion")
    assert dtypes.ss.myquaternion is dtype
    assert dtype.name == "myquaternion"
    assert str(dtype) == "myquaternion"
    assert dtype.gb_name is None
    assert dtype.np_type == np.dtype([("x", "<f4", (4, 4)), ("color", "<i4")], align=True)
    v = Vector(dtype, 2)
    v[0] = {"x": np.arange(16).reshape(4, 4), "color": 100}
    assert_array_equal(v[0].value["x"], np.arange(16).reshape(4, 4))
    assert v[0].value["color"] == 100
    v[1] = (2, 3)
    if supports_udfs:
        expected = Vector.from_dense([100, 3])
        assert expected.isequal(v.apply(lambda x: x["color"]))


def test_jit_unary(v):
    cdef = "void square (float *z, float *x) { (*z) = (*x) * (*x) ; } ;"
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


def test_jit_binary(v):
    cdef = "void absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }"
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
    assert absdiff.types == {dtypes.FP64: dtypes.FP64}
    # The JIT is unforgiving and does not coerce--use the correct types!
    with pytest.raises(KeyError, match="absdiff does not work with INT64"):
        v << absdiff(v & v)
    w = (v - 1).new("FP64")
    v = v.dup("FP64")
    res = absdiff(v & w).new()
    expected = Vector.from_coo([1, 3, 4, 6], [1, 1, 1, 1], dtype="FP64")
    assert expected.isequal(res)
    res = absdiff(w & v).new()
    assert expected.isequal(res)
    assert absdiff["FP64"].jit_c_definition == cdef


def test_jit_indexunary(v):
    cdef = (
        "void diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }"
    )
    diffy = indexunary.ss.register_new("diffy", cdef, "FP64", "FP64", "FP64")
    assert not hasattr(indexunary, "diffy")
    assert indexunary.ss.diffy is diffy
    assert not hasattr(select, "diffy")
    assert not hasattr(select.ss, "diffy")
    assert diffy.name == "ss.diffy"
    assert diffy.types == {dtypes.FP64: dtypes.FP64}
    # The JIT is unforgiving and does not coerce--use the correct types!
    with pytest.raises(KeyError, match="diffy does not work with INT64"):
        v << diffy(v, 1)
    v = v.dup("FP64")
    res = diffy(v, -1).new()
    expected = Vector.from_coo([1, 3, 4, 6], [2, 6, 12, 6], dtype="FP64")
    assert expected.isequal(res)
    assert diffy["FP64"].jit_c_definition == cdef


def test_jit_select(v):
    cdef = (
        # Why does this one insist on `const` for `x` argument?
        "void woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }"
    )
    woot = select.ss.register_new("woot", cdef, "INT32", "INT32")
    assert not hasattr(select, "woot")
    assert select.ss.woot is woot
    assert not hasattr(indexunary, "woot")
    assert hasattr(indexunary.ss, "woot")
    assert woot.name == "ss.woot"
    assert woot.types == {dtypes.INT32: dtypes.BOOL}
    # The JIT is unforgiving and does not coerce--use the correct types!
    with pytest.raises(KeyError, match="woot does not work with INT64"):
        v << woot(v, 1)
    v = v.dup("INT32")
    res = woot(v, 6).new()
    expected = Vector.from_coo([4, 6], [2, 0])
    assert expected.isequal(res)

    res = indexunary.ss.woot(v, 6).new()
    expected = Vector.from_coo([1, 3, 4, 6], [False, False, True, True])
    assert expected.isequal(res)
    assert woot["INT32"].jit_c_definition == cdef

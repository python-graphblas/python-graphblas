import itertools
import pickle

import numpy as np
import pytest

from graphblas import dtypes, lib
from graphblas.dtypes import lookup_dtype

all_dtypes = [
    dtypes.BOOL,
    dtypes.INT8,
    dtypes.INT16,
    dtypes.INT32,
    dtypes.INT64,
    dtypes.UINT8,
    dtypes.UINT16,
    dtypes.UINT32,
    dtypes.UINT64,
    dtypes.FP32,
    dtypes.FP64,
]
if dtypes._supports_complex:
    all_dtypes.append(dtypes.FC32)
    all_dtypes.append(dtypes.FC64)


def test_names():
    assert dtypes.BOOL.name == "BOOL"
    assert dtypes.INT8.name == "INT8"
    assert dtypes.INT16.name == "INT16"
    assert dtypes.INT32.name == "INT32"
    assert dtypes.INT64.name == "INT64"
    assert dtypes.UINT8.name == "UINT8"
    assert dtypes.UINT16.name == "UINT16"
    assert dtypes.UINT32.name == "UINT32"
    assert dtypes.UINT64.name == "UINT64"
    assert dtypes.FP32.name == "FP32"
    assert dtypes.FP64.name == "FP64"


def test_ctype():
    assert dtypes.BOOL.c_type == "_Bool"
    assert dtypes.INT8.c_type == "int8_t"
    assert dtypes.INT16.c_type == "int16_t"
    assert dtypes.INT32.c_type == "int32_t"
    assert dtypes.INT64.c_type == "int64_t"
    assert dtypes.UINT8.c_type == "uint8_t"
    assert dtypes.UINT16.c_type == "uint16_t"
    assert dtypes.UINT32.c_type == "uint32_t"
    assert dtypes.UINT64.c_type == "uint64_t"
    assert dtypes.FP32.c_type == "float"
    assert dtypes.FP64.c_type == "double"


def test_gbtype():
    assert dtypes.BOOL.gb_obj == lib.GrB_BOOL
    assert dtypes.INT8.gb_obj == lib.GrB_INT8
    assert dtypes.INT16.gb_obj == lib.GrB_INT16
    assert dtypes.INT32.gb_obj == lib.GrB_INT32
    assert dtypes.INT64.gb_obj == lib.GrB_INT64
    assert dtypes.UINT8.gb_obj == lib.GrB_UINT8
    assert dtypes.UINT16.gb_obj == lib.GrB_UINT16
    assert dtypes.UINT32.gb_obj == lib.GrB_UINT32
    assert dtypes.UINT64.gb_obj == lib.GrB_UINT64
    assert dtypes.FP32.gb_obj == lib.GrB_FP32
    assert dtypes.FP64.gb_obj == lib.GrB_FP64


def test_lookup_by_name():
    for dt in all_dtypes:
        assert lookup_dtype(dt.name) is dt


def test_lookup_by_ctype():
    for dt in all_dtypes:
        if dt.c_type == "float":
            # Choose 'float' to match numpy/Python, not C (where 'float' means FP32)
            assert lookup_dtype(dt.c_type) is dtypes.FP64
        else:
            assert lookup_dtype(dt.c_type) is dt


def test_lookup_by_gbtype():
    for dt in all_dtypes:
        assert lookup_dtype(dt.gb_obj) is dt


def test_lookup_by_dtype():
    assert lookup_dtype(bool) == dtypes.BOOL
    assert lookup_dtype(int) == dtypes.INT64
    assert lookup_dtype(float) == dtypes.FP64
    with pytest.raises(TypeError, match="Bad dtype"):
        lookup_dtype(None)


def test_unify_dtypes():
    assert dtypes.unify(dtypes.BOOL, dtypes.BOOL) == dtypes.BOOL
    assert dtypes.unify(dtypes.BOOL, dtypes.INT16) == dtypes.INT16
    assert dtypes.unify(dtypes.INT16, dtypes.BOOL) == dtypes.INT16
    assert dtypes.unify(dtypes.INT16, dtypes.INT8) == dtypes.INT16
    assert dtypes.unify(dtypes.UINT32, dtypes.UINT8) == dtypes.UINT32
    assert dtypes.unify(dtypes.UINT32, dtypes.FP32) == dtypes.FP64
    assert dtypes.unify(dtypes.INT32, dtypes.FP32) == dtypes.FP64
    assert dtypes.unify(dtypes.FP64, dtypes.UINT8) == dtypes.FP64
    assert dtypes.unify(dtypes.FP64, dtypes.FP32) == dtypes.FP64
    assert dtypes.unify(dtypes.INT16, dtypes.UINT16) == dtypes.INT32
    assert dtypes.unify(dtypes.UINT64, dtypes.INT8) == dtypes.FP64


def test_dtype_bad_comparison():
    with pytest.raises(TypeError):
        assert dtypes.BOOL == object()
    with pytest.raises(TypeError):
        assert object() != dtypes.BOOL


def test_dtypes_match_numpy():
    for key, val in dtypes._registry.items():
        try:
            if key is int or (isinstance(key, str) and key == "int"):
                # For win64, numpy treats int as int32, not int64
                # graphblas won't allow this craziness
                npval = np.int64
            else:
                npval = np.dtype(key)
        except Exception:
            continue
        assert dtypes.lookup_dtype(npval) == val, f"{key} of type {type(key)}"


def test_pickle():
    for val in dtypes._registry.values():
        s = pickle.dumps(val)
        val2 = pickle.loads(s)
        if val._is_udt:  # pragma: no cover
            assert val.np_type == val2.np_type
            assert val.name == val2.name
        else:
            assert val == val2
    s = pickle.dumps(dtypes._INDEX)
    val2 = pickle.loads(s)
    assert dtypes._INDEX == val2


def test_unify_matches_numpy():
    for type1, type2 in itertools.product(all_dtypes, all_dtypes):
        gb_type = dtypes.unify(type1, type2)
        np_type = type(type1.np_type.type(0) + type2.np_type.type(0))
        assert gb_type is lookup_dtype(np_type), f"({type1}, {type2}) -> {gb_type}"


def test_lt_dtypes():
    expected = [
        dtypes.BOOL,
        dtypes.FP32,
        dtypes.FP64,
        dtypes.INT8,
        dtypes.INT16,
        dtypes.INT32,
        dtypes.INT64,
        dtypes.UINT8,
        dtypes.UINT16,
        dtypes.UINT32,
        dtypes.UINT64,
    ]
    if dtypes._supports_complex:
        expected.insert(1, dtypes.FC32)
        expected.insert(2, dtypes.FC64)
    assert sorted(all_dtypes) == expected
    assert dtypes.BOOL < "FP32"
    with pytest.raises(TypeError):
        assert dtypes.BOOL < 5


def test_bad_register():
    record_dtype = np.dtype([("x", np.object_), ("y", np.float64)], align=True)
    with pytest.raises(ValueError, match="Python object"):
        dtypes.register_new("has_object", record_dtype)
    record_dtype = np.dtype([("x", np.bool_), ("y", np.float64)], align=True)
    with pytest.raises(ValueError, match="identifier"):
        dtypes.register_new("$", record_dtype)
    with pytest.raises(ValueError, match="builtin"):
        dtypes.register_new("is_builtin", np.int8)
    udt = dtypes.register_anonymous(record_dtype)
    assert udt.name is not None
    with pytest.raises(ValueError, match="name"):
        dtypes.register_new("register_new", record_dtype)
    with pytest.raises(ValueError, match="name"):
        dtypes.register_new("UINT8", record_dtype)


def test_auto_register():
    n = np.random.randint(10, 64)
    np_type = np.dtype(f"({n},)int16")
    assert lookup_dtype(np_type).np_type == np_type


def test_default_names():
    from graphblas.dtypes import _default_name

    assert _default_name(np.dtype([("x", np.int32), ("y", np.float64)], align=True)) == (
        "{'x': INT32, 'y': FP64}"
    )
    assert _default_name(np.dtype("(29,)uint8")) == "UINT8[29]"
    assert _default_name(np.dtype("(3,4)bool")) == "BOOL[3, 4]"
    assert _default_name(np.dtype((np.dtype("(5,)float64"), (6,)))) == "FP64[5][6]"
    assert _default_name(np.dtype("S5")) == "dtype('S5')"


def test_record_dtype_from_dict():
    dtype = dtypes.lookup_dtype({"x": int, "y": float})
    assert dtype.name == "{'x': INT64, 'y': FP64}"

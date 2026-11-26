import pytest

import graphblas as gb
from graphblas import exceptions
from graphblas.core import ffi, lib
from graphblas.exceptions import GraphblasException, _error_code_lookup, check_status_carg

_error_names = sorted(
    name
    for name, info in ffi.typeof("GrB_Info").relements.items()
    if info not in {lib.GrB_SUCCESS, lib.GrB_NO_VALUE}
)


@pytest.mark.parametrize("name", _error_names)
def test_every_error_code_has_a_class(name):
    """Every GrB_Info error code raises an exception class of its own.

    This fails when a new GraphBLAS adds a code, as a reminder to add a class for it.
    """
    info = getattr(lib, name)
    assert info in _error_code_lookup
    expected = _error_code_lookup[info]
    if name == "GxB_EXHAUSTED":
        assert expected is StopIteration
    else:
        assert issubclass(expected, GraphblasException)
        assert expected is not GraphblasException
        assert getattr(exceptions, expected.__name__) is expected
    A = gb.Matrix(bool, 2, 2)
    with pytest.raises(expected):
        check_status_carg(info, "Matrix", A._carg)


def test_unknown_error_code():
    """A code with no class of its own raises GraphblasException, not KeyError."""
    unknown = -12345
    assert unknown not in _error_code_lookup
    assert _error_code_lookup[unknown] is GraphblasException
    assert unknown not in _error_code_lookup  # the lookup does not add the code
    A = gb.Matrix(bool, 2, 2)
    with pytest.raises(GraphblasException, match="GrB_Info -12345") as excinfo:
        check_status_carg(unknown, "Matrix", A._carg)
    assert type(excinfo.value) is GraphblasException


@pytest.mark.skipif(
    not hasattr(lib, "GrB_ALREADY_SET") or not hasattr(lib, "GrB_Type_set_String"),
    reason="Needs SuiteSparse 9.0",
)
def test_already_set():
    """A second GrB_NAME of a type raises AlreadySet and keeps the first name."""
    T = ffi.new("GrB_Type*")
    check_status_carg(lib.GrB_Type_new(T, 8), "Type", T[0])
    try:
        first = ffi.new("char[]", b"first_name")
        check_status_carg(lib.GrB_Type_set_String(T[0], first, lib.GrB_NAME), "Type", T[0])
        second = ffi.new("char[]", b"second_name")
        with pytest.raises(exceptions.AlreadySet):
            check_status_carg(lib.GrB_Type_set_String(T[0], second, lib.GrB_NAME), "Type", T[0])
        name = ffi.new("char[]", 32)
        check_status_carg(lib.GrB_Type_get_String(T[0], name, lib.GrB_NAME), "Type", T[0])
        assert ffi.string(name) == b"first_name"
    finally:
        lib.GrB_Type_free(T)


# The suitesparse-vanilla backend has no GxB functions, and SuiteSparse 10.0 added GxB_Vector_load
@pytest.mark.skipif(not hasattr(lib, "GxB_Vector_load"), reason="Needs GxB_Vector_load")
def test_output_is_readonly():
    """Writing to a Vector loaded with GxB_IS_READONLY raises OutputIsReadonly."""
    buf = ffi.new("int64_t[4]", [1, 2, 3, 4])
    v = gb.Vector(int, 4)
    info = lib.GxB_Vector_load(
        v._carg,
        ffi.new("void**", buf),
        lib.GrB_INT64,
        4,
        ffi.sizeof(buf),
        lib.GxB_IS_READONLY,
        ffi.NULL,
    )
    try:
        assert info == lib.GrB_SUCCESS
        assert v.isequal(gb.Vector.from_dense([1, 2, 3, 4]))
        with pytest.raises(exceptions.OutputIsReadonly):
            v[0] << 9
        assert list(buf) == [1, 2, 3, 4]
    finally:
        del v  # before buf, whose data v borrows

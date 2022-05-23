import pytest

import graphblas


def test_import_special_attrs():
    not_hidden = {x for x in dir(graphblas) if not x.startswith("__")}
    # Is everything imported?
    assert len(not_hidden & graphblas._SPECIAL_ATTRS) == len(graphblas._SPECIAL_ATTRS)
    # Is everything special that needs to be?
    not_special = {x for x in dir(graphblas) if not x.startswith("_")} - graphblas._SPECIAL_ATTRS
    assert not_special == {"backend", "config", "init", "replace"}
    # Make sure these "not special" objects don't have objects that look special within them
    for attr in not_special:
        assert not set(dir(getattr(graphblas, attr))) & graphblas._SPECIAL_ATTRS


def test_bad_init():
    # same params is okay
    params = dict(graphblas._init_params)
    del params["automatic"]
    graphblas.init(**params)
    # different params is bad
    params["blocking"] = not params["blocking"]
    with pytest.raises(graphblas.exceptions.GraphblasException, match="different init parameters"):
        graphblas.init(**params)


def test_bad_libget():
    with pytest.raises(AttributeError, match="GrB_bad_name"):
        graphblas.base.libget("GrB_bad_name")


def test_lib_attrs():
    for attr in dir(graphblas.lib):
        getattr(graphblas.lib, attr)


def test_bad_call():
    class bad:
        name = "bad"
        _carg = 1

    with pytest.raises(TypeError, match="Error calling GrB_Matrix_apply"):
        graphblas.base.call("GrB_Matrix_apply", [bad, bad, bad, bad, bad])
    with pytest.raises(
        TypeError, match=r"Call objects: GrB_Matrix_apply\(bad, bad, bad, bad, bad, bad\)"
    ):
        graphblas.base.call("GrB_Matrix_apply", [bad, bad, bad, bad, bad, bad])

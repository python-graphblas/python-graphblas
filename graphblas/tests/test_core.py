import pathlib

import pytest

import graphblas as gb

try:
    import setuptools
except ImportError:  # pragma: no cover (import)
    setuptools = None
try:
    import tomli
except ImportError:  # pragma: no cover (import)
    tomli = None


def test_import_special_attrs():
    not_hidden = {x for x in dir(gb) if not x.startswith("__")}
    # Is everything imported?
    exclude = {"ss"} if gb.backend != "suitesparse" else set()
    assert len(not_hidden & gb._SPECIAL_ATTRS) == len(gb._SPECIAL_ATTRS - exclude)
    # Is everything special that needs to be?
    not_special = {x for x in dir(gb) if not x.startswith("_")} - gb._SPECIAL_ATTRS
    assert not_special == {"backend", "config", "init", "replace", "tests"}
    # Make sure these "not special" objects don't have objects that look special within them
    for attr in not_special:
        assert not set(dir(getattr(gb, attr))) & gb._SPECIAL_ATTRS
    if gb.backend != "suitesparse":
        with pytest.raises(AttributeError, match="suitesparse"):
            gb.ss


def test_bad_init():
    # same params is okay
    params = dict(gb._init_params)
    del params["automatic"]
    gb.init(**params)
    # different params is bad
    params["blocking"] = not params["blocking"]
    with pytest.raises(gb.exceptions.GraphblasException, match="different init parameters"):
        gb.init(**params)


def test_bad_libget():
    with pytest.raises(AttributeError, match="GrB_bad_name"):
        gb.core.base.libget("GrB_bad_name")


def test_lib_attrs():
    for attr in dir(gb.core.lib):
        getattr(gb.core.lib, attr)


def test_bad_call():
    class bad:
        name = "bad"
        _carg = 1

    with pytest.raises(TypeError, match="Error calling GrB_Matrix_apply"):
        gb.core.base.call("GrB_Matrix_apply", [bad, bad, bad, bad, bad])
    with pytest.raises(
        TypeError, match=r"Call objects: GrB_Matrix_apply\(bad, bad, bad, bad, bad, bad\)"
    ):
        gb.core.base.call("GrB_Matrix_apply", [bad, bad, bad, bad, bad, bad])


def test_version():
    from packaging.version import parse

    assert parse(gb.__version__) > parse("2022.11.0")


@pytest.mark.skipif("not setuptools or not tomli or not gb.__file__")
def test_packages():
    """Ensure all packages are declared in pyproject.toml."""
    # Currently assume s`pyproject.toml` is at the same level as `graphblas` folder.
    # This probably isn't always True, and we can probably do a better job of finding it.
    path = pathlib.Path(gb.__file__).parent
    pkgs = [f"graphblas.{x}" for x in setuptools.find_packages(str(path))]
    pkgs.append("graphblas")
    pkgs.sort()
    pyproject = path.parent / "pyproject.toml"
    if not pyproject.exists():  # pragma: no cover (safety)
        pytest.skip("Did not find pyproject.toml")
    with pyproject.open("rb") as f:
        pkgs2 = sorted(tomli.load(f)["tool"]["setuptools"]["packages"])
    assert (
        pkgs == pkgs2
    ), "If there are extra items on the left, add them to pyproject.toml:tool.setuptools.packages"

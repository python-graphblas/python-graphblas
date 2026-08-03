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

    assert parse(gb.__version__) > parse("2024.2.0")


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
        cfg = tomli.load(f)
    if cfg.get("project", {}).get("name") != "python-graphblas":  # pragma: no cover (safety)
        pytest.skip("Did not find correct pyproject.toml")
    pkgs2 = sorted(cfg["tool"]["setuptools"]["packages"])
    assert (
        pkgs == pkgs2
    ), "If there are extra items on the left, add them to pyproject.toml:tool.setuptools.packages"


def test_index_max():
    assert gb.MAX_SIZE == 2**60  # True for all current backends


def test_config_attribute_assignment_raises():
    # Setting an option by plain attribute assignment used to silently no-op:
    # donfig has no __setattr__ for options, so it created a dead instance
    # attribute and left the real option alone. It should raise and name the
    # canonical API instead.
    orig = gb.config["autocompute"]
    try:
        with pytest.raises(AttributeError, match=r"config\.set\(autocompute"):
            gb.config.autocompute = not orig
        assert gb.config["autocompute"] == orig  # the failed write changed nothing
        # An unknown name is a mistaken write too, not a new attribute
        with pytest.raises(AttributeError, match="not_a_real_option"):
            gb.config.not_a_real_option = 5
        assert "not_a_real_option" not in vars(gb.config)
        # Canonical idioms still work: read by item, write by set()
        gb.config.set(autocompute=not orig)
        assert gb.config["autocompute"] == (not orig)
        with gb.config.set(autocompute=orig):
            assert gb.config["autocompute"] == orig
        assert gb.config["autocompute"] == (not orig)  # restored on exit
    finally:
        gb.config.set(autocompute=orig)


def test_config_donfig_attrs_are_derived(monkeypatch):
    # The allowlist of writable attributes must come from what donfig's
    # __init__ actually set, not a hardcoded list. Simulate a future donfig
    # that grows a field: the derived allowlist absorbs it, where a
    # hardcoded list would start rejecting donfig's own writes.
    import donfig

    real_init = donfig.Config.__init__

    def init_with_extra_field(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.hypothetical_future_field = 1

    monkeypatch.setattr(donfig.Config, "__init__", init_with_extra_field)
    probe = type(gb.config)("test_derived_allowlist")
    assert "hypothetical_future_field" in probe._donfig_attrs
    probe.hypothetical_future_field = 2  # writable, not rejected
    with pytest.raises(AttributeError, match="autocompute"):
        probe.autocompute = False

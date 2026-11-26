import os
import pathlib
import subprocess
import sys
import tomllib

import pytest

import graphblas as gb

try:
    import setuptools
except ImportError:  # pragma: no cover (import)
    setuptools = None


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


def test_jit_c_control_config():
    """``_jit_c_control_config`` checks and lowercases ``config["jit_c_control"]`` for init."""
    import yaml

    path = pathlib.Path(gb.__file__).parent / "graphblas.yaml"
    # Unset by default: init leaves the JIT control at "run" (see "The JIT control"
    # in the Technical Notes)
    assert yaml.safe_load(path.read_text())["jit_c_control"] is None
    # YAML 1.1 reads an unquoted on/off in a user's config file as a bool
    cases = [("on", "on"), ("LOAD", "load"), ("run", "run"), ("pause", "pause"), ("off", "off")]
    cases += [(True, "on"), (False, "off"), (None, None)]
    for val, expected in cases:
        with gb.config.set(jit_c_control=val):
            assert gb._jit_c_control_config() == expected
    for val in ["of", "", 1]:
        with (
            gb.config.set(jit_c_control=val),
            pytest.raises(ValueError, match="GRAPHBLAS_JIT_C_CONTROL"),
        ):
            gb._jit_c_control_config()


@pytest.mark.parametrize("key", ["autocompute", "mapnumpy"])
def test_config_bool(key):
    """A boolean option accepts the strings an environment variable gives, and rejects the rest."""
    for val in [True, 1, "true", "True", "YES", "on", "1"]:
        with gb.config.set({key: val}):
            assert gb._config_bool(key) is True
    for val in [False, 0, "false", "FALSE", "no", "off", "0", "none", "null", None, ""]:
        with gb.config.set({key: val}):
            assert gb._config_bool(key) is False
    for val in ["maybe", 2, "2"]:
        with (
            gb.config.set({key: val}),
            pytest.raises(ValueError, match=f"GRAPHBLAS_{key.upper()}"),
        ):
            gb._config_bool(key)


def test_autocompute_false_string():
    """``autocompute="false"``, which ``GRAPHBLAS_AUTOCOMPUTE=false`` sets, turns it off."""
    v = gb.Vector.from_coo([0], [1])
    with gb.config.set(autocompute="false"), pytest.raises(TypeError, match="autocompute=True"):
        (v + v).nvals
    with gb.config.set(autocompute="true"):
        assert (v + v).nvals == 1


_BOOL_CONFIG_SCRIPT = """
import graphblas as gb

print("package: " + gb.__file__)
print(f"parsed: {gb._config_bool('autocompute')} {gb._config_bool('mapnumpy')}")
"""


def test_bool_config_from_environment():
    """``GRAPHBLAS_AUTOCOMPUTE=false`` and ``GRAPHBLAS_MAPNUMPY=off`` turn the options off."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")]))
    env = dict(
        os.environ, PYTHONPATH=pythonpath, GRAPHBLAS_AUTOCOMPUTE="false", GRAPHBLAS_MAPNUMPY="off"
    )
    result = subprocess.run(
        [sys.executable, "-c", _BOOL_CONFIG_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report
    assert lines.get("parsed") == "False False", report


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


@pytest.mark.skipif("not setuptools or not gb.__file__")
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
        cfg = tomllib.load(f)
    if cfg.get("project", {}).get("name") != "python-graphblas":  # pragma: no cover (safety)
        pytest.skip("Did not find correct pyproject.toml")
    pkgs2 = sorted(cfg["tool"]["setuptools"]["packages"])
    assert (
        pkgs == pkgs2
    ), "If there are extra items on the left, add them to pyproject.toml:tool.setuptools.packages"


def test_index_max():
    assert gb.MAX_SIZE == 2**60  # True for all current backends


def test_config_attribute_assignment_raises():
    # Assigning to an option as an attribute raises and names ``config.set``,
    # rather than creating an instance attribute that leaves the option alone
    # (see the ``Config`` class in graphblas/__init__.py).
    orig = gb.config["autocompute"]
    try:
        with pytest.raises(AttributeError, match=r"config\.set\(autocompute"):
            gb.config.autocompute = not orig
        assert gb.config["autocompute"] == orig  # the failed write changed nothing
        # An unknown name is a mistaken write too, not a new attribute
        with pytest.raises(AttributeError, match="not_a_real_option"):
            gb.config.not_a_real_option = 5
        assert "not_a_real_option" not in vars(gb.config)
        # Reading by item and writing with set() work
        gb.config.set(autocompute=not orig)
        assert gb.config["autocompute"] == (not orig)
        with gb.config.set(autocompute=orig):
            assert gb.config["autocompute"] == orig
        assert gb.config["autocompute"] == (not orig)  # restored on exit
    finally:
        gb.config.set(autocompute=orig)


def test_config_donfig_attrs_are_derived(monkeypatch):
    # The writable attributes are the ones donfig's __init__ set, not a fixed
    # list, so a donfig release that adds a field can still write it. Simulate
    # such a release.
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

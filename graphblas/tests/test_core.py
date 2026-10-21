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


_TENSOR_IMPORT_SCRIPT = """
import sys
import graphblas as gb
print("package:", gb.__file__)
print("after import:", sorted(m for m in ("numba", "graphblas.core.tensor") if m in sys.modules))
if sys.argv[1] == "vanilla":
    gb.init("suitesparse-vanilla")
T = gb.Tensor
from graphblas.core import tensor
print("same:", T is tensor.Tensor and gb.einsum is tensor.einsum)
loaded = ("tensor_numba", "tensor_udt", "tensor_io")
print("engines:", sorted(m for m in loaded if f"graphblas.core.{m}" in sys.modules))
import numpy as np
for call in (lambda: gb.Tensor(gb.dtypes.FP64, (2, 3)), lambda: gb.einsum("ij->ji", np.eye(2))):
    try:
        call()
        print("use: ok")
    except NotImplementedError as exc:
        print("use:", exc)
"""


@pytest.mark.parametrize("backend", ["suitesparse", "vanilla"])
def test_tensor_exports(backend):
    """``gb.Tensor`` and ``gb.einsum`` (experimental) load lazily, like Matrix: ``import
    graphblas`` imports neither them nor Numba, and the Tensor's engines load on use. On the
    vanilla backend they import, and using them raises.
    """
    import subprocess
    import sys

    repo_root = pathlib.Path(gb.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _TENSOR_IMPORT_SCRIPT, backend],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root)},
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = [line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line]
    lines = {key: [v for k, v in lines if k == key] for key, _ in lines}
    assert lines["package"] == [str(repo_root / "graphblas" / "__init__.py")], report
    assert lines["after import"] == ["[]"], report
    assert lines["same"] == ["True"], report
    assert lines["engines"] == ["[]"], report
    if backend == "vanilla":
        assert lines["use"] == ["Tensor requires the suitesparse backend"] * 2, report
    else:
        assert lines["use"] == ["ok", "ok"], report


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

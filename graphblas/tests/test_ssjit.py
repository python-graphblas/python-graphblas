import contextlib
import os
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import graphblas as gb
from graphblas import (
    agg,
    backend,
    binary,
    dtypes,
    indexbinary,
    indexunary,
    monoid,
    select,
    semiring,
    unary,
)
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core.operator.indexbinary import _has_idxbinop
from graphblas.core.ss.jit_config import _enable_jit_for_udt
from graphblas.exceptions import EmptyObject

from .conftest import autocompute, burble

from graphblas import Vector  # isort:skip (for dask-graphblas)

try:
    import numba
except ImportError:
    numba = None

if backend != "suitesparse":
    pytest.skip("not suitesparse backend", allow_module_level=True)


def _jit_works_here():
    """Whether this environment can compile JIT kernels at all, whatever the session's control.

    ``_enable_jit_for_udt`` probes the compiler, but answers no without probing
    under ``'off'``, ``'pause'``, ``'load'`` or a ``'run'`` the user set, so ask
    it under ``'on'`` and put the control back. It caches the answer for the
    process. This runs once, at import, before the autouse ``_setup_jit``
    fixture changes the config.
    """
    if backend != "suitesparse":
        return False
    with _jit_mode("on"):
        return _enable_jit_for_udt()


@pytest.fixture(scope="module", autouse=True)
def _setup_jit():
    """Set up the SuiteSparse:GraphBLAS JIT.

    ``fix_jit_config`` points the JIT at the conda compiler, or else at the one
    in ``sysconfig``, and probes it. It returns True if the JIT works, False if
    the probe failed (``jit_c_control`` is left where SuiteSparse put it), and
    None if it found no compiler.
    """
    prev = gb.ss.config["jit_c_control"]

    result = gb.ss.fix_jit_config()
    # On False, do not try sysconfig: if the conda compiler cannot build GraphBLAS
    # JIT kernels, Python's sysconfig compiler will not either.
    if result is None:
        # No conda env, and fix_jit_config found no sysconfig compiler either.
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

    try:
        yield
    finally:
        gb.ss.config["jit_c_control"] = prev


def _require_jit_on():
    """Skip the test if the SuiteSparse JIT can't compile in this environment."""
    if gb.ss.config["jit_c_control"] != "on":
        pytest.skip("JIT compilation not available (probe failed or compiler missing)")


@contextlib.contextmanager
def _jit_mode(mode):
    """Temporarily set ``jit_c_control`` to ``mode``; restore on exit."""
    prev = gb.ss.config["jit_c_control"]
    gb.ss.config["jit_c_control"] = mode
    try:
        yield
    finally:
        gb.ss.config["jit_c_control"] = prev


_JIT_WORKS_AT_IMPORT = _jit_works_here()


def _run_in_fresh_process(script, jit_c_control=None, **extra_env):
    """Run ``script`` in a new interpreter on this tree; return its ``key: value`` lines and output.

    The script must print ``package: <gb.__file__>`` so the child can be shown
    to have imported this tree before anything it reports is trusted. The
    child gets ``GRAPHBLAS_JIT_C_CONTROL=jit_c_control``, or none at all, and
    ``extra_env``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root), **extra_env)
    env.pop("GRAPHBLAS_JIT_C_CONTROL", None)
    if jit_c_control is not None:
        env["GRAPHBLAS_JIT_C_CONTROL"] = jit_c_control
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, report
    lines = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    assert lines.get("package") == str(repo_root / "graphblas" / "__init__.py"), report
    return lines, report


# Run in a subprocess: the import under test happens once per process, and by
# the time any test executes it is long past. ``jit_c_control`` has to be read
# through the C API rather than ``gb.ss.config``, because reaching for
# ``gb.ss`` is itself the thing under test.
_SS_IMPORT_PROBE = """
import graphblas as gb

gb.Vector                           # initialize the backend, but not gb.ss
from graphblas.core import ffi, lib

def control():
    val_ptr = ffi.new("int32_t*")
    assert lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_C_CONTROL, val_ptr) == lib.GrB_SUCCESS
    return val_ptr[0]

before = control()
gb.ss.about["library_version"]      # the line a user writes for a bug report
print("package: " + gb.__file__)
print("before: " + str(before))
print("after: " + str(control()))
"""


@pytest.mark.parametrize("jit_c_control", [None, "run", "on", "pause"])
def test_ss_import_leaves_jit_c_control_alone(jit_c_control):
    """Reading from ``gb.ss`` does not change ``jit_c_control``.

    ``gb.ss`` is imported on attribute access, so the import has to be inert.
    ``jit_c_control`` decides whether SuiteSparse loads kernels from its on-disk
    cache, and a line as harmless as ``print(gb.ss.about["library_version"])``
    must not change how later operations compute. The control is set by the
    backend's init, from ``gb.config`` (here ``GRAPHBLAS_JIT_C_CONTROL``),
    and ``'on'`` becomes ``'run'`` where the JIT cannot compile. Not set, it
    is ``'run'``. Regression: importing ``gb.ss`` raised the control to ``'on'``.
    """
    lines, report = _run_in_fresh_process(_SS_IMPORT_PROBE, jit_c_control)
    assert lines["after"] == lines["before"], report
    if jit_c_control != "on":
        expected = getattr(gb.core.lib, f"GxB_JIT_{(jit_c_control or 'run').upper()}")
        assert lines["before"] == str(expected), report
    elif _JIT_WORKS_AT_IMPORT:
        assert lines["before"] == str(gb.core.lib.GxB_JIT_ON), report
    else:
        # python-suitesparse-graphblas wheels (built without the JIT), or no working C compiler
        assert lines["before"] == str(gb.core.lib.GxB_JIT_RUN), report


# A pytest-xdist worker sets PYTEST_XDIST_WORKER before it imports conftest.py. The
# ``GxB_IMPLEMENTATION_*`` constants of suitesparse_graphblas.lib are set one patch release
# behind the library it loads, as when it was built against older headers.
_XDIST_CACHE_PATH_PROBE = """
import os

os.environ.pop("GRAPHBLAS_CACHE_PATH", None)
import graphblas as gb
import suitesparse_graphblas


class OlderHeaders:
    def __init__(self, lib):
        self._lib = lib
        self.GxB_IMPLEMENTATION_SUB = lib.GxB_IMPLEMENTATION_SUB - 1

    def __getattr__(self, name):
        return getattr(self._lib, name)


suitesparse_graphblas.lib = OlderHeaders(suitesparse_graphblas.lib)
import graphblas.tests.conftest

print("package: " + gb.__file__)
print("cache: " + os.environ["GRAPHBLAS_CACHE_PATH"])
"""


def test_xdist_jit_cache_path_names_the_library_version(tmp_path):
    """Under ``pytest -n``, each worker's JIT cache path names the version of the library loaded.

    That is, not the version of the headers python-suitesparse-graphblas was built against
    (see conftest.py). Reading the version leaves SuiteSparse's default cache alone.
    """
    lines, report = _run_in_fresh_process(
        _XDIST_CACHE_PATH_PROBE, PYTEST_XDIST_WORKER="gw7", HOME=str(tmp_path)
    )
    version = ".".join(map(str, gb.ss.about["library_version"]))
    expected = tmp_path / ".SuiteSparse_xdist" / f"GrB{version}" / "gw7"
    assert lines["cache"] == str(expected), report
    assert not (tmp_path / ".SuiteSparse").exists(), report


_BAD_JIT_C_CONTROL_PROBE = """
import graphblas as gb
import suitesparse_graphblas as ssgb

print("package: " + gb.__file__)
try:
    gb.Vector
except ValueError as exc:
    print("error: " + str(exc))
print("initialized: " + str(ssgb.is_initialized()))
"""


def test_bad_jit_c_control_config_fails_init_before_it_starts():
    """A bad ``GRAPHBLAS_JIT_C_CONTROL`` says so, and leaves SuiteSparse uninitialized."""
    lines, report = _run_in_fresh_process(_BAD_JIT_C_CONTROL_PROBE, "of")
    assert "GRAPHBLAS_JIT_C_CONTROL" in lines.get("error", ""), report
    assert "'of'" in lines["error"], report
    assert lines["initialized"] == "False", report


def test_set_jit_c_control_at_init(monkeypatch):
    """Init sets each configured value as given, and ``'on'`` only where the JIT compiles.

    Where it cannot, it sets ``'run'`` with no error: a library built without
    the JIT, no compiler on disk, a cache it cannot write to, or a compiler
    that fails the probe. Nothing configured (None) sets ``'run'`` too, and
    only that ``'run'`` is left for UDT work to raise. Called directly here,
    as init calls it.
    """
    from graphblas.core.ss import jit_config
    from graphblas.core.ss.jit_config import _set_jit_c_control_at_init

    cfg = gb.ss.config
    prev_control = cfg["jit_c_control"]
    prev_cc = cfg["jit_c_compiler_name"]
    for name in ["_jit_enabled_for_udt", "_configured_jit_c_control", "_user_set_run"]:
        monkeypatch.setattr(jit_config, name, getattr(jit_config, name))
    try:
        for control in ["load", "run", "pause", "off"]:
            _set_jit_c_control_at_init(control)
            # A library built without the JIT clamps 'load' down to 'run'.
            expected = {control, "run"} if control == "load" else {control}
            assert cfg["jit_c_control"] in expected
            assert jit_config._configured_jit_c_control == control
            assert jit_config._user_set_run is (control == "run")
        _set_jit_c_control_at_init(None)
        assert cfg["jit_c_control"] == "run"
        assert jit_config._configured_jit_c_control is None
        assert jit_config._user_set_run is False
        _set_jit_c_control_at_init("on")
        assert jit_config._configured_jit_c_control == "on"
        assert cfg["jit_c_control"] == ("on" if jit_config._jit_enabled_for_udt else "run")
        if _JIT_WORKS_AT_IMPORT:
            assert cfg["jit_c_control"] == "on"
        # A cache it cannot write to: the cached probe would load, and the first
        # new kernel would fail to compile.
        with monkeypatch.context() as m:
            m.setattr(jit_config, "_cache_is_writable", lambda cfg: False)
            _set_jit_c_control_at_init("on")
            assert cfg["jit_c_control"] == "run"
        # A "compiler" that exists but cannot compile anything. The probe's type
        # is named after the toolchain, so the probe kernel this process has
        # already loaded does not answer for this one. SuiteSparse built with
        # MSVC always compiles through cmake, whose Visual Studio generator
        # chooses the compiler itself, so this setting cannot fail the probe.
        if not cfg["jit_use_cmake"]:
            cfg["jit_c_compiler_name"] = str(Path(__file__).parent)
            _set_jit_c_control_at_init("on")
            assert cfg["jit_c_control"] == "run"
            assert jit_config._jit_enabled_for_udt is False
        # No compiler on disk and nothing to repair it from: no probe at all.
        cfg["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        probes = []
        monkeypatch.setattr(jit_config, "_repair_jit_compiler", lambda cfg: None)
        monkeypatch.setattr(jit_config, "_probe_jit", probes.append)
        _set_jit_c_control_at_init("on")
        assert cfg["jit_c_control"] == "run"
        assert jit_config._jit_enabled_for_udt is False
        assert probes == []
    finally:
        cfg["jit_c_compiler_name"] = prev_cc
        cfg["jit_c_control"] = prev_control


# Run in a subprocess: this is about the very first call in a process, and by
# the time any test executes, plenty of JIT setup has already happened.
_FIX_JIT_TWICE_PROBE = """
import graphblas as gb

print("package: " + gb.__file__)
print("first: " + str(gb.ss.fix_jit_config()))
print("second: " + str(gb.ss.fix_jit_config()))
"""


def test_fix_jit_config_first_call_agrees_with_the_second():
    """``fix_jit_config()`` gives the same answer on its first call as on later ones.

    The probe goes through the C API and registers nothing. Regression: the
    probe started a second probe inside itself, so a working environment
    answered ``False`` on the first call.
    """
    lines, report = _run_in_fresh_process(_FIX_JIT_TWICE_PROBE)
    assert lines["first"] == lines["second"], report


# Run in a subprocess: this process has already probed the JIT and cached the
# answer, and a fresh one has to probe at its first registration. Each
# registration starts from the default ``'run'`` (None), so one registrar
# raising the control cannot cover for another that does not.
_SS_REGISTER_NEW_PROBE = """
import graphblas as gb
from graphblas import binary, indexbinary, indexunary, select, unary
from graphblas.core.operator.indexbinary import _has_idxbinop

print("package: " + gb.__file__)
registrars = {
    "unary": lambda: unary.ss.register_new(
        "rn_neg", "void rn_neg (double *z, double *x) { (*z) = -(*x) ; }", "FP64", "FP64"
    ),
    "binary": lambda: binary.ss.register_new(
        "rn_absdiff",
        "void rn_absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }",
        "FP64", "FP64", "FP64",
    ),
    "indexunary": lambda: indexunary.ss.register_new(
        "rn_diffy",
        "void rn_diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }",
        "FP64", "FP64", "FP64",
    ),
    "select": lambda: select.ss.register_new(
        "rn_woot",
        "void rn_woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }",
        "INT32", "INT32",
    ),
}
if _has_idxbinop:
    registrars["indexbinary"] = lambda: indexbinary.ss.register_new(
        "rn_add_theta",
        "void rn_add_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, "
        "double *y, GrB_Index iy, GrB_Index jy, double *theta) "
        "{ (*z) = (*x) + (*y) + (*theta) ; }",
        "FP64", "FP64", "FP64", "FP64",
    )
for kind, register in registrars.items():
    gb.ss.config["jit_c_control"] = None
    try:
        register()
    except Exception as exc:
        print(kind + ": " + type(exc).__name__)
    else:
        print(kind + ": ok")
"""


def test_ss_register_new_enables_the_jit():
    """Registering an op from C source raises the default ``'run'``.

    SuiteSparse compiles the C definition, or loads it from its cache, when the
    op is created. ``'run'`` does neither, so without the raise the
    registration fails. Each registrar has to ask for the JIT itself, since
    nothing else raises the control first (importing ``gb.ss`` does not).
    """
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("JIT not compiled in or not working here; nothing can compile")
    lines, report = _run_in_fresh_process(_SS_REGISTER_NEW_PROBE)
    kinds = ["unary", "binary", "indexunary", "select"]
    kinds += ["indexbinary"] if _has_idxbinop else []
    assert {kind: lines.get(kind) for kind in kinds} == dict.fromkeys(kinds, "ok"), report


def test_ss_register_new_says_why_the_jit_cannot_build_it(monkeypatch):
    """A C-source registration the JIT cannot build names the reason.

    For each of these SuiteSparse reports a bare ``NullPointer`` (op) or
    ``InvalidValue`` (type) with no message. None of them compiles anything, so
    this runs with or without a working compiler; a library built without the
    JIT skips ``'load'`` and the failed probe.
    """
    from graphblas.core.ss import jit_config
    from graphblas.exceptions import InvalidValue, NullPointer

    # New names every run, so no cached kernel can satisfy ``'load'``.
    suffix = uuid.uuid4().hex[:8]
    op_name = f"why_neg_{suffix}"
    cdef = f"void {op_name} (double *z, double *x) {{ (*z) = -(*x) ; }}"
    type_name = f"why_t_{suffix}"
    tdef = f"typedef struct {{ int64_t why_a ; }} {type_name} ;"
    prev = gb.ss.config["jit_c_control"]
    try:
        for control, match in [
            ("off", "jit_c_control is 'off', so SuiteSparse will not compile it"),
            ("load", "jit_c_control is 'load', so SuiteSparse only loads kernels"),
        ]:
            gb.ss.config["jit_c_control"] = control
            if gb.ss.config["jit_c_control"] != control:
                # A library built without the JIT clamps 'load' down to 'run'.
                continue
            with pytest.raises(NullPointer, match=match):
                unary.ss.register_new(op_name, cdef, "FP64", "FP64")
            with pytest.raises(InvalidValue, match=match):
                dtypes.ss.register_new(type_name, tdef)
        # A process that cannot compile leaves the default 'run' in place.
        monkeypatch.setattr(jit_config, "_jit_enabled_for_udt", False)
        gb.ss.config["jit_c_control"] = None
        with pytest.raises(NullPointer, match="jit_c_control is still 'run'"):
            unary.ss.register_new(op_name, cdef, "FP64", "FP64")
        # A probe that fails in this very call leaves 'load' before the op is created.
        gb.ss.config["jit_c_control"] = "on"
        if gb.ss.config["jit_c_control"] == "on":

            def failed_probe(cfg):
                cfg["jit_c_control"] = "load"
                return False

            monkeypatch.setattr(jit_config, "_jit_enabled_for_udt", None)
            monkeypatch.setattr(jit_config, "jit_compiler_is_usable", lambda: True)
            monkeypatch.setattr(jit_config, "_probe_jit", failed_probe)
            gb.ss.config["jit_c_control"] = None
            with pytest.raises(NullPointer, match="the C compiler does not work here"):
                unary.ss.register_new(op_name, cdef, "FP64", "FP64")
    finally:
        gb.ss.config["jit_c_control"] = prev


def test_ss_register_new_under_a_chosen_run_says_how_to_opt_in(monkeypatch):
    """C source cannot be built under a ``'run'`` the user set, and the error says so.

    UDT work raises only the default ``'run'``. One that was set, often to keep
    processes from compiling into one cache at the same time, is kept, so the
    registration fails, and the message has to say why and what to set instead.
    Here the user configured ``'on'`` before init and writes ``'run'`` later.
    """
    from graphblas.core.ss import jit_config
    from graphblas.exceptions import InvalidValue, NullPointer

    monkeypatch.setattr(jit_config, "_configured_jit_c_control", "on")
    monkeypatch.setattr(jit_config, "_user_set_run", jit_config._user_set_run)
    suffix = uuid.uuid4().hex[:8]
    op_name = f"chosen_neg_{suffix}"
    type_name = f"chosen_t_{suffix}"
    prev = gb.ss.config["jit_c_control"]
    try:
        gb.ss.config["jit_c_control"] = "run"
        assert jit_config._user_set_run is True
        match = "jit_c_control was set to 'run' .*GRAPHBLAS_JIT_C_CONTROL.* = 'on', or None"
        with pytest.raises(NullPointer, match=match):
            unary.ss.register_new(
                op_name,
                f"void {op_name} (double *z, double *x) {{ (*z) = -(*x) ; }}",
                "FP64",
                "FP64",
            )
        with pytest.raises(InvalidValue, match=match):
            dtypes.ss.register_new(type_name, f"typedef struct {{ int64_t c_a ; }} {type_name} ;")
        assert gb.ss.config["jit_c_control"] == "run"
    finally:
        gb.ss.config["jit_c_control"] = prev


# Run in a subprocess: the forms that set the control are read once, at init.
# ``form`` is defined by a line put in front of this.
_UDT_WORK_PROBE = """
import uuid

import numpy as np

import graphblas as gb

if form == "gb.config.set":
    gb.config.set(jit_c_control="run")
elif form == "gb.ss.config":
    gb.ss.config["jit_c_control"] = "run"
print("package: " + gb.__file__)
print("before: " + gb.ss.config["jit_c_control"])
name = "uw_" + uuid.uuid4().hex[:8]
try:
    gb.dtypes.ss.register_new(name, "typedef struct { int64_t uw_a ; } " + name + " ;")
except Exception as exc:
    print("register_new: " + type(exc).__name__ + ": " + str(exc))
else:
    print("register_new: ok")
print("after register_new: " + gb.ss.config["jit_c_control"])
try:
    import numba
except ImportError:
    numba = None
if numba is not None:
    udt = gb.dtypes.register_anonymous(
        np.dtype([("uw_x", np.int64), ("uw_y", np.int64)], align=True), "UdtWorkProbe"
    )
    v = gb.Vector.from_coo([0, 1], np.array([(1, 2), (3, 4)], dtype=udt.np_type), dtype=udt)
    w = v.ewise_add(v, gb.binary.plus).new()
    print("udt op: " + str(w[1].new().value.tolist()))
    print("after udt op: " + gb.ss.config["jit_c_control"])
"""


@pytest.mark.parametrize(
    ("form", "env_control", "kept"),
    [
        pytest.param("default", None, False, id="default"),
        pytest.param("GRAPHBLAS_JIT_C_CONTROL", "run", True, id="env-run"),
        pytest.param("gb.config.set", None, True, id="config-set-run"),
        pytest.param("YAML file", None, True, id="yaml-run"),
        # With nothing configured, a 'run' written later is the default again.
        pytest.param("gb.ss.config", None, False, id="ss-config-run"),
        # Written over a configured 'on', it changes the control, so it counts.
        pytest.param("gb.ss.config", "on", True, id="env-on-then-ss-config-run"),
        pytest.param("GRAPHBLAS_JIT_C_CONTROL", "off", True, id="env-off"),
        pytest.param("GRAPHBLAS_JIT_C_CONTROL", "pause", True, id="env-pause"),
        pytest.param("GRAPHBLAS_JIT_C_CONTROL", "load", True, id="env-load"),
    ],
)
def test_udt_work_raises_only_the_default_run(form, env_control, kept, tmp_path):
    """UDT work raises the default ``'run'`` to ``'on'``, and keeps a control the user set.

    A ``'run'`` set before init (``GRAPHBLAS_JIT_C_CONTROL``, ``gb.config.set``
    or a YAML config file) keeps processes from compiling, at the cost of UDT
    ops running their Numba functions and C source failing to register. One
    written to ``gb.ss.config`` later counts only when it changes the configured
    control. ``'off'``, ``'pause'`` and ``'load'`` are always kept.
    """
    if not kept and not _JIT_WORKS_AT_IMPORT:
        pytest.skip("JIT not compiled in or not working here; nothing raises to 'on'")
    extra_env = {}
    if form == "YAML file":
        (tmp_path / "jit.yaml").write_text("jit_c_control: run\n")
        extra_env["GRAPHBLAS_CONFIG"] = str(tmp_path)
    script = f"form = {form!r}\n" + _UDT_WORK_PROBE
    lines, report = _run_in_fresh_process(script, env_control, **extra_env)
    control = env_control if form == "GRAPHBLAS_JIT_C_CONTROL" else "run"
    if not kept:
        assert lines["before"] == "run", report
        assert lines["register_new"] == "ok", report
        assert lines["after register_new"] == "on", report
        expected = "on"
    else:
        # A library built without the JIT clamps 'load' down to 'run'.
        expected = lines["before"]
        assert expected in {control, "run"}, report
        assert lines["register_new"].startswith("InvalidValue: Cannot create"), report
        if control == "run":
            assert "was set to 'run'" in lines["register_new"], report
        assert lines["after register_new"] == expected, report
    if numba is not None:
        assert lines["udt op"] == "(6, 8)", report
        assert lines["after udt op"] == expected, report


def test_ss_register_new_says_the_compile_failed():
    """C source that does not compile says so, and that ``jit_c_control`` is now ``'load'``."""
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("JIT not compiled in or not working here; nothing compiles to fail")
    from graphblas.exceptions import JitError, NullPointer

    cdef = "void why_bad (double *z, double *x) { (*z) = why_undeclared ; }"
    prev = gb.ss.config["jit_c_control"]
    try:
        gb.ss.config["jit_c_control"] = "on"
        # SuiteSparse reports a failed compile as JitError from 9.4 on.
        with pytest.raises((NullPointer, JitError), match="SuiteSparse failed to compile it"):
            unary.ss.register_new("why_bad", cdef, "FP64", "FP64")
        assert gb.ss.config["jit_c_control"] == "load"
    finally:
        gb.ss.config["jit_c_control"] = prev


def test_auto_fix_jit_at_import_left_compiler_usable():
    """The import-time repair leaves a compiler the UDT path can actually use."""
    if not gb.ss.jit_compiler_is_usable():
        pytest.skip("sandboxed env without a usable compiler; auto-fix had nothing to repair")
    if not _JIT_WORKS_AT_IMPORT:
        # The compiler exists but the probe failed (the baked-in flags may
        # target another architecture). The probe takes the failed compile,
        # after which SuiteSparse leaves ``jit_c_control`` at a mode that does
        # not compile, so later ops use the generic kernels; which mode depends
        # on the SuiteSparse version.
        assert gb.ss.config["jit_c_control"] in {"load", "run"}
        pytest.skip("compiler present but JIT probe failed in this env")
    assert _JIT_WORKS_AT_IMPORT
    assert gb.ss.config["jit_c_control"] == "on"


def test_enable_jit_for_udt_re_raises_a_lowered_control(monkeypatch):
    """A cached "yes" must still raise a lowered ``'run'`` before arming a kernel.

    The cache answers "can this process compile?", which is settled once. The
    control is separate state that anything may have moved since, so returning
    the cached answer without re-raising it would leave an op armed with C
    source that SuiteSparse never builds, and nothing would warn: codegen
    succeeded.

    Driven through a stand-in config, so the assertion holds on every host: a
    library built without the JIT clamps writes down to ``'run'``, and would
    never read back the ``'on'`` this is about. The live round trip is in
    ``test_enable_jit_for_udt_re_raises_the_live_control``.
    """
    from graphblas.core.ss import jit_config

    fake = {}
    monkeypatch.setattr(jit_config, "_ss_config", lambda: fake)
    monkeypatch.setattr(jit_config, "_jit_enabled_for_udt", True)
    monkeypatch.setattr(jit_config, "_user_set_run", False)
    # An explicit opt-out is a choice, not the default to raise, and ``'load'`` is
    # also where SuiteSparse parks the control after a compile fails.
    for start, expected in (("run", "on"), ("load", "load"), ("off", "off"), ("pause", "pause")):
        fake["jit_c_control"] = start
        assert _enable_jit_for_udt() is (expected == "on")
        assert fake["jit_c_control"] == expected
    # A 'run' the user set is an opt-out too.
    jit_config._user_set_run = True
    fake["jit_c_control"] = "run"
    assert _enable_jit_for_udt() is False
    assert fake["jit_c_control"] == "run"


def test_enable_jit_for_udt_does_not_cache_an_opt_out(monkeypatch):
    """An ``'off'`` at the first UDT op must not decide the rest of the process.

    It says nothing about whether this process can compile. Caching a "no"
    for it would leave the default ``'run'``, restored afterwards, unraised for
    good, so every later UDT op would run without the JIT and nothing would
    say so. The same goes for ``'pause'``, ``'load'``, and a ``'run'`` the user
    chose.
    """
    from graphblas.core.ss import jit_config

    fake = {}
    monkeypatch.setattr(jit_config, "_ss_config", lambda: fake)
    monkeypatch.setattr(jit_config, "_jit_enabled_for_udt", None)
    monkeypatch.setattr(jit_config, "jit_compiler_is_usable", lambda: True)
    monkeypatch.setattr(jit_config, "_probe_jit", lambda cfg: True)
    monkeypatch.setattr(jit_config, "_cache_is_writable", lambda cfg: True)
    for start in ("off", "pause", "load", "run"):
        # The last is a 'run' the user set.
        monkeypatch.setattr(jit_config, "_user_set_run", start == "run")
        fake["jit_c_control"] = start
        assert _enable_jit_for_udt() is False
        assert fake["jit_c_control"] == start
        assert jit_config._jit_enabled_for_udt is None
    monkeypatch.setattr(jit_config, "_user_set_run", False)
    # A cache it cannot write to: the default 'run' stays, and the answer is cached.
    fake["jit_c_control"] = "run"
    with monkeypatch.context() as m:
        m.setattr(jit_config, "_cache_is_writable", lambda cfg: False)
        assert _enable_jit_for_udt() is False
        assert fake["jit_c_control"] == "run"
        assert jit_config._jit_enabled_for_udt is False
    monkeypatch.setattr(jit_config, "_jit_enabled_for_udt", None)
    fake["jit_c_control"] = "run"
    assert _enable_jit_for_udt() is True
    assert fake["jit_c_control"] == "on"


def test_enable_jit_for_udt_re_raises_the_live_control(monkeypatch):
    """The same raise, through SuiteSparse's own config rather than a stand-in.

    A ``'run'`` written to ``gb.ss.config`` is the default, and raised, while
    nothing is configured; written over a configured ``'on'``, it is the user's,
    and kept. None is the default either way.
    """
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("JIT not compiled in or not working here; the control need not hold 'on'")

    from graphblas.core.ss import jit_config

    prev_control = gb.ss.config["jit_c_control"]
    prev_enabled = jit_config._jit_enabled_for_udt
    try:
        jit_config._jit_enabled_for_udt = True
        monkeypatch.setattr(jit_config, "_configured_jit_c_control", None)
        monkeypatch.setattr(jit_config, "_user_set_run", jit_config._user_set_run)
        gb.ss.config["jit_c_control"] = "run"
        assert _enable_jit_for_udt() is True
        assert gb.ss.config["jit_c_control"] == "on"
        monkeypatch.setattr(jit_config, "_configured_jit_c_control", "on")
        gb.ss.config["jit_c_control"] = "RUN"
        assert jit_config._user_set_run is True
        assert _enable_jit_for_udt() is False
        assert gb.ss.config["jit_c_control"] == "run"
        gb.ss.config["jit_c_control"] = None
        assert gb.ss.config["jit_c_control"] == "run"
        assert jit_config._user_set_run is False
        assert _enable_jit_for_udt() is True
        assert gb.ss.config["jit_c_control"] == "on"
        gb.ss.config["jit_c_control"] = gb.core.lib.GxB_JIT_RUN
        assert jit_config._user_set_run is True
    finally:
        gb.ss.config["jit_c_control"] = prev_control
        jit_config._jit_enabled_for_udt = prev_enabled


@pytest.mark.skipif("not supports_udfs")
def test_failed_compile_is_not_retried_at_the_next_arming():
    """A compile that fails is tried once, not again at every later UDT op.

    SuiteSparse lowers ``jit_c_control`` to ``'load'`` after a failed compile
    so the failure is not repeated. Arming another op must leave it there:
    raising it back to ``'on'`` would run the failing compiler again at the
    next kernel, and on SuiteSparse 9.4 and later raise ``JitError`` again,
    with ``'off'`` the only way out.
    """
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("JIT not compiled in or not working here; nothing compiles to fail")
    from graphblas.core.ss import jit_config
    from graphblas.exceptions import JitError

    # Field names unique to this test: ``register_anonymous`` caches one
    # DataType per ``np.dtype``.
    udt = dtypes.register_anonymous(
        np.dtype([("nrt_a", np.int64), ("nrt_b", np.int64)], align=True), "_NoRetryUdt"
    )
    v = Vector.from_coo([0, 1], np.array([(1, 2), (3, 4)], dtype=udt.np_type), dtype=udt)
    # Before 9.4 SuiteSparse does not report a failed compile: it returns
    # ``GrB_NO_VALUE`` ("use the generic kernel"), though it still lowers the
    # control.
    reported = gb.ss.about["library_version"] >= (9, 4)
    prev_cc = gb.ss.config["jit_c_compiler_name"]
    prev_control = gb.ss.config["jit_c_control"]
    prev_enabled = jit_config._jit_enabled_for_udt
    try:
        # Start from the default, so the first arming shows the raise.
        gb.ss.config["jit_c_control"] = None
        # A compiler that does not exist fails under every toolchain, where a
        # flag it does not know is only a warning to MSVC. The cache is pinned
        # to yes (true here) so the arming does not repair the path first.
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        jit_config._jit_enabled_for_udt = True
        plus = binary.plus[udt]
        assert plus.jit_c_source is not None
        assert gb.ss.config["jit_c_control"] == "on"
        with pytest.raises(JitError) if reported else contextlib.nullcontext():
            v.ewise_add(v, plus).new()
        assert gb.ss.config["jit_c_control"] == "load"
        assert binary.minus[udt].jit_c_source is not None
        assert gb.ss.config["jit_c_control"] == "load"
        result = v.ewise_add(v, plus).new()  # the generic kernel, without raising
        assert tuple(result[1].new().value) == (6, 8)
    finally:
        gb.ss.config["jit_c_compiler_name"] = prev_cc
        gb.ss.config["jit_c_control"] = prev_control
        jit_config._jit_enabled_for_udt = prev_enabled


def test_public_fix_jit_config_repairs_a_broken_compiler():
    """``gb.ss.fix_jit_config()`` repairs a clobbered compiler path and returns ``True``."""
    if not _JIT_WORKS_AT_IMPORT:
        pytest.skip("env JIT doesn't actually compile; nothing to repair to")
    prev = {
        "control": gb.ss.config["jit_c_control"],
        "cc": gb.ss.config["jit_c_compiler_name"],
        "flags": gb.ss.config["jit_c_compiler_flags"],
    }
    if not os.environ.get("CONDA_PREFIX"):
        pytest.skip("test requires a CONDA_PREFIX to repair from")
    try:
        # Force a broken path; fix_jit_config must put back something usable.
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        assert not gb.ss.jit_compiler_is_usable()
        result = gb.ss.fix_jit_config()
        assert result is True
        assert gb.ss.jit_compiler_is_usable()
        # A second call must not disable the JIT; the probe registers nothing
        # whose name could collide the second time.
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        result2 = gb.ss.fix_jit_config()
        assert result2 is True
        assert gb.ss.config["jit_c_control"] == "on"
    finally:
        gb.ss.config["jit_c_control"] = prev["control"]
        gb.ss.config["jit_c_compiler_name"] = prev["cc"]
        gb.ss.config["jit_c_compiler_flags"] = prev["flags"]


def test_public_fix_jit_config_returns_none_without_env():
    """With no ``CONDA_PREFIX`` and ``use_sysconfig=False``, ``fix_jit_config`` returns None."""
    if os.environ.get("CONDA_PREFIX"):
        # Drop the env for this call only.
        prev_env = os.environ.pop("CONDA_PREFIX")
    else:
        prev_env = None
    prev_cc = gb.ss.config["jit_c_compiler_name"]
    try:
        gb.ss.config["jit_c_compiler_name"] = "/nonexistent/path/to/cc"
        result = gb.ss.fix_jit_config(use_sysconfig=False)
        assert result is None
    finally:
        if prev_env is not None:
            os.environ["CONDA_PREFIX"] = prev_env
        gb.ss.config["jit_c_compiler_name"] = prev_cc


@pytest.fixture
def v():
    return Vector.from_coo([1, 3, 4, 6], [1, 1, 2, 0])


@autocompute
def test_jit_udt():
    _require_jit_on()
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
    if numba is None:
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


@pytest.mark.skipif("numba is None")
def test_jit_udt_np_type_numba_cannot_type():
    # The np_type gets the error register_anonymous gives for an array Numba cannot type
    _require_jit_on()
    np_type = np.dtype([("x", np.bool_, (2,))])
    with pytest.raises(ValueError, match="cannot type an array of bool") as exc_info:
        dtypes.ss.register_new(
            "_jit_bool_pair", "typedef struct { uint8_t x [2] ; } _jit_bool_pair ;", np_type=np_type
        )
    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert not hasattr(dtypes.ss, "_jit_bool_pair")
    # The failure leaves the name free, so it registers with an np_type Numba can type.
    dtype = dtypes.ss.register_new(
        "_jit_bool_pair",
        "typedef struct { uint8_t x [2] ; } _jit_bool_pair ;",
        np_type=np.dtype([("x", np.uint8, (2,))]),
    )
    assert dtypes.ss._jit_bool_pair is dtype


def test_jit_udt_name_leaves_room_for_the_nul():
    """A type name as long as ``GxB_MAX_NAME_LEN`` is rejected before SuiteSparse sees it.

    SuiteSparse keeps one character fewer, so that name would no longer match
    its typedef, and the registration would fail as a compile error.
    """
    from graphblas.core import lib

    limit = lib.GxB_MAX_NAME_LEN
    name = "n" * limit
    with pytest.raises(ValueError, match=f"Max size is {limit - 1}; got {limit}"):
        dtypes.ss.register_new(name, f"typedef struct {{ int64_t a ; }} {name} ;")


def test_jit_unary(v):
    cdef = "void square (float *z, float *x) { (*z) = (*x) * (*x) ; } ;"
    _require_jit_on()
    with burble():
        square = unary.ss.register_new("square", cdef, "FP32", "FP32")
    assert not hasattr(unary, "square")
    assert unary.ss.square is square
    assert square.name == "ss.square"
    assert square.types == {dtypes.FP32: dtypes.FP32}
    # An op from C source does not cast its inputs: another type raises KeyError.
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


@pytest.mark.slow
def test_jit_binary(v):
    cdef = "void absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }"
    _require_jit_on()
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


@pytest.mark.slow
def test_jit_indexunary(v):
    cdef = (
        "void diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "
        "{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }"
    )
    _require_jit_on()
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


@pytest.mark.slow
@pytest.mark.skipif(not _has_idxbinop, reason="requires SuiteSparse:GraphBLAS 9.4+")
def test_jit_indexbinary(v):
    cdef = (
        "void add_theta (double *z, double *x, GrB_Index ix, GrB_Index jx, "
        "double *y, GrB_Index iy, GrB_Index jy, double *theta) "
        "{ (*z) = (*x) + (*y) + (*theta) ; }"
    )
    _require_jit_on()
    with burble():
        add_theta = indexbinary.ss.register_new("add_theta", cdef, "FP64", "FP64", "FP64", "FP64")
    assert not hasattr(indexbinary, "add_theta")
    assert indexbinary.ss.add_theta is add_theta
    assert add_theta.name == "ss.add_theta"
    assert add_theta.types == {(dtypes.FP64, dtypes.FP64, dtypes.FP64): dtypes.FP64}
    assert "FP64" in add_theta
    assert add_theta["FP64"].return_type == dtypes.FP64
    assert add_theta["FP64"].jit_c_definition == cdef
    # SuiteSparse compiles it under a C name of its types, name and definition
    c_name = add_theta["FP64"].jit_c_name
    assert c_name.startswith("add_theta_4_FP64_4_FP64_4_FP64_")
    assert len(c_name) == len("add_theta_4_FP64_4_FP64_4_FP64_") + 12
    assert add_theta["FP64"].jit_c_source == cdef.replace("add_theta", c_name)
    # Bind theta and use as BinaryOp
    v64 = v.dup("FP64")
    binop = add_theta["FP64"](10.0)
    assert binop.opclass == "BinaryOp"
    res = v64.ewise_mult(v64, binop).new()
    # v has values at [1, 3, 4, 6] with vals [1, 1, 2, 0]
    # ewise_mult: x+y+theta = 2*val + 10
    expected = Vector.from_coo([1, 3, 4, 6], [12.0, 12.0, 14.0, 10.0], dtype="FP64")
    assert expected.isequal(res)
    # Bound with no type, it runs at the inputs' types and theta's, which a JIT op must
    # have: like a JIT IndexUnaryOp's thunk, theta is not cast to the op's theta type.
    assert expected.isequal(v64.ewise_mult(v64, add_theta(10.0)).new(), check_dtype=True)
    with pytest.raises(KeyError, match="add_theta does not work with .FP64, FP64, INT64. types"):
        v64.ewise_mult(v64, add_theta(10))
    with pytest.raises(KeyError, match="add_theta does not work with .INT64, INT64, FP64. types"):
        v.ewise_mult(v, add_theta(10.0))
    # An empty theta is refused, as for any IndexBinaryOp (SuiteSparse would read its value)
    for bind in [add_theta, add_theta["FP64"]]:
        with pytest.raises(EmptyObject, match="theta is an empty Scalar"):
            bind(gb.Scalar("FP64"))
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
        match="gb.indexbinary.ss.add_theta already defined for .FP64, FP64, FP64. input",
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
        "{ (*z) = (*x) + (*y) + 2 * (double)(*theta) ; }"
    )
    add_theta_mixed = indexbinary.ss.register_new(
        "add_theta", mixed_cdef, "FP64", "FP64", "FP32", "FP64"
    )
    assert add_theta_mixed is add_theta
    assert ("FP64", "FP64", "FP32") in add_theta
    assert ("FP64", "FP32") not in add_theta  # theta of y's type, FP32
    assert repr(add_theta["FP64", "FP64", "FP32"]) == "indexbinary.add_theta[FP64, FP64, FP32]"
    # An FP32 theta picks it, and its own kernel runs. SuiteSparse finds a compiled kernel by
    # its op's name, and its encoding has no theta type, so each theta type needs its own C
    # name; with one shared C name, the FP64 theta's kernel would read the float as a double:
    # [2, 2, 4, 0].
    assert add_theta["FP64", "FP64", "FP32"].jit_c_name.startswith(
        "add_theta_4_FP64_4_FP64_4_FP32_"
    )
    expected_mixed = Vector.from_coo([1, 3, 4, 6], [22.0, 22.0, 24.0, 20.0])
    res = v64.ewise_mult(v64, add_theta(np.float32(10))).new()
    assert expected_mixed.isequal(res, check_dtype=True)
    assert expected.isequal(v64.ewise_mult(v64, add_theta(10.0)).new(), check_dtype=True)
    # A name too long for SuiteSparse with its types is cut, with a digest of the whole
    long_name = "add_theta_" + "long" * 25
    long_op = indexbinary.ss.register_new(
        long_name, cdef.replace("add_theta", long_name), "FP64", "FP64", "FP64", "FP64"
    )
    c_name = long_op["FP64"].jit_c_name
    assert len(c_name) == 117
    assert c_name.startswith(long_name[:100])
    assert expected.isequal(v64.ewise_mult(v64, long_op(10.0)).new(), check_dtype=True)


@pytest.mark.slow
def test_jit_c_name_per_name_and_definition(v):
    """An op registered from C is compiled under a C name of its full name and definition.

    SuiteSparse finds a compiled kernel in memory by its op's name, and checks a definition
    only when it loads a kernel from disk. Regression: an op defined again after ``delattr``,
    or one with the same last name in another namespace, ran the kernel compiled first.
    """
    _require_jit_on()
    v64 = v.dup("FP64")
    # One last name in two namespaces
    plus = "void jcn_f (double *z, double *x, double *y) { (*z) = (*x) + (*y) ; }"
    f = binary.ss.register_new("jcn_f", plus, "FP64", "FP64", "FP64")
    g = binary.ss.register_new("jcn_ns.jcn_f", plus.replace("+", "-"), "FP64", "FP64", "FP64")
    try:
        expected = Vector.from_coo([1, 3, 4, 6], [2.0, 2.0, 4.0, 0.0])
        assert v64.ewise_mult(v64, f).new().isequal(expected, check_dtype=True)
        expected = Vector.from_coo([1, 3, 4, 6], [0.0, 0.0, 0.0, 0.0])
        assert v64.ewise_mult(v64, g).new().isequal(expected, check_dtype=True)
        assert f["FP64"].jit_c_name != g["FP64"].jit_c_name
    finally:
        delattr(binary.ss, "jcn_f")
        delattr(binary.ss, "jcn_ns")
    # One name defined again, after delattr, with another definition and then the first
    square = "void jcn_u (double *z, double *x) { (*z) = (*x) * (*x) ; }"
    add10 = square.replace("* (*x)", "+ 10")
    names = []
    for cdef, values in [(square, [1, 1, 4, 0]), (add10, [11, 11, 12, 10]), (square, [1, 1, 4, 0])]:
        op = unary.ss.register_new("jcn_u", cdef, "FP64", "FP64")
        try:
            expected = Vector.from_coo([1, 3, 4, 6], values, dtype="FP64")
            assert v64.apply(op).new().isequal(expected, check_dtype=True)
            names.append(op["FP64"].jit_c_name)
        finally:
            delattr(unary.ss, "jcn_u")
    assert names[0] != names[1]
    assert names[0] == names[2]  # one definition, one kernel
    # A struct member of the op's name is not renamed
    rec = dtypes.ss.register_new(
        "JcnRec",
        "typedef struct { int64_t jcn_m ; double b ; } JcnRec ;",
        np_type=np.dtype([("jcn_m", "<i8"), ("b", "<f8")], align=True),
    )
    cdef = "void jcn_m (double *z, const JcnRec *x) { (*z) = x->jcn_m + (*x) . b ; }"
    op = unary.ss.register_new("jcn_m", cdef, rec, "FP64")
    try:
        c_name = op[rec].jit_c_name
        assert op[rec].jit_c_source == cdef.replace("void jcn_m", f"void {c_name}")
        w = Vector(rec, 2)
        w[0] = (3, 0.5)
        w[1] = (4, 0.25)
        expected = Vector.from_coo([0, 1], [3.5, 4.25])
        assert w.apply(op).new().isequal(expected, check_dtype=True)
    finally:
        delattr(unary.ss, "jcn_m")


@pytest.mark.slow
def test_jit_select(v):
    cdef = (
        # SelectOps don't write to their input array, so SuiteSparse requires
        # the x argument to be ``const``.
        "void woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "
        "{ (*z) = ((*x) + i + j == (*y)) ; }"
    )
    _require_jit_on()
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


@pytest.mark.skipif("not supports_udfs")
def test_udt_jit_c_source_introspection():
    """``jit_c_source`` and ``jit_c_name`` show the C that SuiteSparse compiles.

    Covers a record UDT and its ``jit_c_definition``, lifted ops on record and
    array UDTs, built-in types and ops (no C definition), and Monoid, Semiring
    and Aggregator. Field names and shapes are unique to this test:
    ``register_anonymous`` shares one DataType per ``np.dtype``, C name
    included, so a test sharing a dtype would depend on test order.
    """
    record_dtype = np.dtype([("introsp_a", np.int64), ("introsp_b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_IntrospectUDT")

    # dtype-level introspection
    assert udt.jit_c_name == "gbudt__IntrospectUDT"
    assert "typedef struct" in udt.jit_c_definition
    assert "int64_t introsp_a" in udt.jit_c_definition
    assert "double introsp_b" in udt.jit_c_definition

    # Built-in type: no C definition
    assert dtypes.INT64.jit_c_name is None
    assert dtypes.INT64.jit_c_definition is None

    # Lifted BinaryOp on a record UDT
    plus_udt = binary.plus[udt]
    assert plus_udt.jit_c_name == "plus_gbudt__IntrospectUDT"
    src = plus_udt.jit_c_source
    assert src is not None
    assert "z->introsp_a = (x->introsp_a) + (y->introsp_a)" in src
    assert "z->introsp_b = (x->introsp_b) + (y->introsp_b)" in src

    # Lifted UnaryOp on a record UDT
    ainv_udt = unary.ainv[udt]
    assert ainv_udt.jit_c_name == "ainv_gbudt__IntrospectUDT"
    assert "z->introsp_a = -(x->introsp_a)" in ainv_udt.jit_c_source

    # Lifted op on an array UDT
    arr_udt = dtypes.register_anonymous(np.dtype("(11,)float64"), "_IntrospectArr")
    times_arr = binary.times[arr_udt]
    src_arr = times_arr.jit_c_source
    assert src_arr is not None
    assert "for (int64_t i = 0 ; i < 11 ; i++) { z->v[i] = (x->v[i]) * (y->v[i]) ; }" in src_arr

    # Built-in op on a built-in type: no C source
    assert binary.plus[int].jit_c_source is None
    assert binary.plus[int].jit_c_name is None
    assert unary.abs[float].jit_c_source is None

    # A Monoid, Semiring or Aggregator reports the source of the BinaryOp it uses.
    plus_binop_src = binary.plus[udt].jit_c_source
    assert gb.monoid.plus[udt].jit_c_source == plus_binop_src
    # A Semiring reports its multiplier's source, and its Monoid its BinaryOp's.
    semi = gb.semiring.plus_times[udt]
    assert semi.jit_c_source == binary.times[udt].jit_c_source
    assert semi.monoid.jit_c_source == plus_binop_src
    # agg.sum reports its Monoid's; agg.count is not built on one and has none.
    assert gb.agg.sum[udt].jit_c_source == plus_binop_src
    assert gb.agg.count[udt].jit_c_source is None


@pytest.mark.skipif("not supports_udfs")
def test_op_jit_signature_uses_pinned_type_name():
    """After a UDT is renamed, lifted ops use the C name from its first registration.

    The type's C name cannot change (see ``test_udt_jit_c_info_pinned_at_first_register``
    in ``test_dtype.py``), so an op naming the new one would refer to an undefined struct,
    and SuiteSparse would fall back to the Numba cfunc without saying so.
    """
    record_dtype = np.dtype([("pinned_a", np.int64), ("pinned_b", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_PinnedOrig")
    udt2 = dtypes.register_anonymous(record_dtype, "_PinnedRenamed")
    assert udt2 is udt
    assert udt.name == "_PinnedRenamed"
    assert udt.jit_c_name == "gbudt__PinnedOrig"

    op = binary.plus[udt]
    assert op.jit_c_name == "plus_gbudt__PinnedOrig"
    assert "gbudt__PinnedOrig *" in op.jit_c_source
    assert "_PinnedRenamed" not in op.jit_c_source


@pytest.mark.skipif("not supports_udfs")
def test_jit_compiles_auto_udt_ops():
    """The JIT compiles a kernel for a lifted op on a UDT, and ``jit_c_control`` stays ``'on'``.

    Regression: the kernel failed to compile (``use of undeclared identifier
    'Bx'``), which lowered the control to ``'load'``.
    """
    _require_jit_on()

    record_dtype = np.dtype([("a", np.int64), ("b", np.float64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "_JitAutoUdt")

    v = gb.Vector(udt, 3)
    v[0] = (1, 2.0)
    v[1] = (3, 4.0)
    v[2] = (5, 6.0)
    w = v.dup()

    with _jit_mode("on"):
        with burble():
            result = binary.plus(v & w).new()
        assert result[0].new() == (2, 4.0)
        assert result[2].new() == (10, 12.0)
        # A failed compile would lower the control to 'load'.
        assert (
            gb.ss.config["jit_c_control"] == "on"
        ), "JIT compilation got disabled (flipped from 'on' to 'load') after a built-in UDT op"


@pytest.mark.skipif("not supports_udfs")
def test_floordiv_udt_jit_matches_python_semantics():
    """``binary.floordiv`` on a UDT rounds toward minus infinity in the JIT kernel.

    Python's ``//``, and so the Numba function, is floor division. Regression:
    the C kernel used C's ``/``, so negative operands and floats disagreed with
    the Numba function. The vectors have 50 elements that are not iso, so the
    kernel runs.
    """
    _require_jit_on()

    N = 50

    # float64 record; C's ``/`` would give -3.5 and 3.5. Field names unique to this
    # test, so no other test shares its DataType and C name (``register_anonymous``
    # caches one DataType per ``np.dtype``).
    udt = dtypes.register_anonymous(
        np.dtype([("fd_a", np.float64), ("fd_b", np.float64)]), "_FdJitF64"
    )
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (-7.0 - i, 7.0 + i)
        u[i] = (2.0, 2.0)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7.0 // 2.0 == -4.0, 7.0 // 2.0 == 3.0
    assert w[0].new() == (-4.0, 3.0)
    # -8.0 // 2.0 == -4.0, 8.0 // 2.0 == 4.0
    assert w[1].new() == (-4.0, 4.0)

    # int64 record; C's ``/`` would give -3 for -7 / 2.
    udt_i = dtypes.register_anonymous(np.dtype([("a", np.int64), ("b", np.int64)]), "_FdJitI64")
    v = gb.Vector(udt_i, N)
    u = gb.Vector(udt_i, N)
    for i in range(N):
        v[i] = (-7 - 2 * i, 7 + 2 * i)
        u[i] = (2, 3)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7 // 2 == -4, 7 // 3 == 2
    assert w[0].new() == (-4, 2)
    # -9 // 2 == -5, 9 // 3 == 3
    assert w[1].new() == (-5, 3)
    # -11 // 2 == -6, 11 // 3 == 3
    assert w[2].new() == (-6, 3)

    # Array UDT, whose C codegen works on ``v[i]``
    udt_arr = dtypes.register_anonymous(np.dtype((np.float32, (4,))), "_FdJitArrF32")
    v = gb.Vector(udt_arr, N)
    u = gb.Vector(udt_arr, N)
    for i in range(N):
        v[i] = np.array([-7.0 - i, 7.0 + i, -8.0 - i, 8.0 + i], dtype=np.float32)
        u[i] = np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    w = v.ewise_mult(u, binary.floordiv).new()
    # -7//2=-4, 7//2=3, -8//3=-3, 8//3=2
    assert_array_equal(w[0].new().value, np.array([-4.0, 3.0, -3.0, 2.0], dtype=np.float32))
    # -8//2=-4, 8//2=4, -9//3=-3, 9//3=3
    assert_array_equal(w[1].new().value, np.array([-4.0, 4.0, -3.0, 3.0], dtype=np.float32))


@pytest.mark.skipif("not supports_udfs")
def test_min_max_udt_jit_ignores_nan_and_breaks_zero_ties():
    """The JIT kernel for ``binary.min`` and ``max`` on a float UDT answers as the cfunc does.

    Both ignore a NaN operand on either side, as C99 ``fmin`` (``GrB_MIN_FP64``)
    does, and on a tie between ``-0.0`` and ``0.0`` give ``-0.0`` for ``min``
    and ``0.0`` for ``max`` on every platform; see ``_minmax_expr``. The test
    puts NaN on each side, since a plain ``(a < b ? a : b)`` drops a NaN on one
    side only. Integer fields keep the plain comparison.
    """
    _require_jit_on()

    # Field names unique to this test; see test_floordiv_udt_jit_matches_python_semantics.
    udt = dtypes.register_anonymous(
        np.dtype([("nan_a", np.float64), ("nan_b", np.float32), ("nan_c", np.int32)]), "_NanJitMM"
    )
    csrc = binary.min[udt].jit_c_source
    assert "fmin" not in csrc, csrc
    assert "signbit(x->nan_a)" in csrc, csrc
    assert "signbit(x->nan_b)" in csrc, csrc
    assert "((x->nan_c) < (y->nan_c) ? (x->nan_c) : (y->nan_c))" in csrc, csrc
    assert "!signbit(x->nan_a)" in binary.max[udt].jit_c_source

    N = 100
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    nan = float("nan")
    for i in range(N):
        # field nan_a: NaN on the left; field nan_b: NaN on the right at odd indices.
        v[i] = (nan, 2.0 + i, i)
        u[i] = (1.0 + i, nan if i % 2 else 3.0 + i, 2 * i)

    w = v.ewise_mult(u, binary.min).new()
    assert w[0].new().value[0] == 1.0  # min(NaN, 1.0) -> 1.0
    assert w[1].new().value[1] == 3.0  # min(3.0, NaN) -> 3.0
    assert w[2].new().value[1] == 4.0  # min(4.0, 5.0) -> 4.0 (normal case)
    assert w[3].new().value[2] == 3  # integer field is unaffected

    w = v.ewise_mult(u, binary.max).new()
    assert w[0].new().value[0] == 1.0  # max(NaN, 1.0) -> 1.0
    assert w[1].new().value[1] == 3.0  # max(3.0, NaN) -> 3.0
    assert w[2].new().value[1] == 5.0  # max(4.0, 5.0) -> 5.0 (normal case)
    assert w[3].new().value[2] == 6  # integer field is unaffected

    # Both operands NaN is the one case where a NaN survives, for min and max
    # alike, and it is the only case ``fmin`` has no non-NaN answer for.
    nan_only = gb.Vector(udt, N)
    for i in range(N):
        nan_only[i] = (nan, np.float32(i), i)
    w = nan_only.ewise_mult(nan_only, binary.min).new()
    assert np.isnan(w[0].new().value[0])

    # Signed-zero ties, in both orders and both fields, on the C kernel and the cfunc.
    left = gb.Vector(udt, N)
    right = gb.Vector(udt, N)
    for i in range(N):
        a, b = (-0.0, 0.0) if i % 2 else (0.0, -0.0)
        left[i] = (a, a, i)
        right[i] = (b, b, i)
    for mode in ["on", "off"]:
        with _jit_mode(mode):
            lo = left.ewise_mult(right, binary.min).new()
            hi = left.ewise_mult(right, binary.max).new()
        for i in range(2):
            assert np.signbit(lo[i].new().value[0]), mode
            assert np.signbit(lo[i].new().value[1]), mode
            assert not np.signbit(hi[i].new().value[0]), mode
            assert not np.signbit(hi[i].new().value[1]), mode


@pytest.mark.skipif("not supports_udfs")
def test_complex_truediv_udt_jit_matches_cfunc_at_extremes():
    """Complex division in the JIT kernel is the cfunc's, bit for bit, at any magnitude.

    SuiteSparse compiles JIT kernels with ``-fcx-limited-range`` under GCC,
    which makes C's ``_Complex`` division the naive formula: complex64
    ``1e20 / 1e20`` gives ``nan`` and ``(1+1j) / (1e160+1e160j)`` gives ``0``.
    Both paths spell out the division in real arithmetic instead, so no flag
    or Python version changes it; see ``_complex_truediv``.
    """
    _require_jit_on()

    udt = dtypes.register_anonymous(
        np.dtype([("cxd_s", np.complex64), ("cxd_d", np.complex128)], align=True), "_CxDivJit"
    )
    csrc = binary.truediv[udt].jit_c_source
    assert "crealf" in csrc, csrc
    assert "(x->cxd_d) / (y->cxd_d)" not in csrc, csrc
    # (complex64 pair, complex128 pair); the first four are finite, and the C
    # library's division gets them wrong under the limited-range flag.
    pairs = [
        ((1e20, 1e20), (1e200, 1e200)),
        ((1e-25 + 1e-25j, 1e-25), (1e200 + 1e200j, 1e200 + 1e200j)),
        ((1 + 1j, 1e30 + 1e30j), (1 + 1j, 1e160 + 1e160j)),
        ((3e30 + 4e30j, 1e30 - 2e30j), (1e-170, 1e-170 + 1e-170j)),
        ((0j, -2 + 0.5j), (0j, -2 + 0.5j)),
        ((-2 + 0.5j, -2 + 0.5j), (-2 + 0.5j, -2 + 0.5j)),
        ((3 + 4j, complex(0.0, -0.0)), (3 + 4j, complex(0.0, -0.0))),
        ((complex(np.inf, np.nan), 1 + 1j), (complex(np.inf, np.nan), 1 + 1j)),
    ]
    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        (a_s, b_s), (a_d, b_d) = pairs[i % len(pairs)]
        v[i] = (a_s, a_d)
        u[i] = (b_s, b_d)
    with _jit_mode("on"):
        jit = v.ewise_mult(u, binary.truediv).new()
    with _jit_mode("off"):
        cfunc = v.ewise_mult(u, binary.truediv).new()
    assert jit.to_dense().tobytes() == cfunc.to_dense().tobytes()
    result = jit.to_dense()
    for i, ((a_s, b_s), (a_d, b_d)) in enumerate(pairs[:4]):
        assert np.isclose(result[i]["cxd_s"], np.complex64(a_s) / np.complex64(b_s), rtol=1e-6)
        assert np.isclose(result[i]["cxd_d"], np.complex128(a_d) / np.complex128(b_d), rtol=1e-15)


@pytest.mark.skipif("not supports_udfs")
def test_complex_times_udt_jit_matches_cfunc_with_infinities():
    """Complex multiplication in the JIT kernel is the cfunc's, bit for bit.

    numpy multiplies by the textbook formula. clang's ``_Complex`` multiply adds
    C99 Annex G's recovery of infinities, which would make ``1j * (inf+infj)``
    ``-inf+infj`` on macOS, where numpy gives ``nan+nanj``. So both paths spell
    out the formula in real arithmetic, for record fields and array elements
    alike; see ``_complex_times``.
    """
    _require_jit_on()

    rec = dtypes.register_anonymous(
        np.dtype([("cxt_s", np.complex64), ("cxt_d", np.complex128)], align=True), "_CxTimesJit"
    )
    arr = dtypes.register_anonymous(np.dtype((np.complex128, (2,))), "_CxTimesJitArr")
    assert "(x->cxt_d) * (y->cxt_d)" not in binary.times[rec].jit_c_source
    inf, nan = np.inf, np.nan
    pairs = [
        (1j, complex(inf, inf)),
        (-2.5j, complex(inf, -inf)),
        (complex(inf, 0.0), complex(0.0, 1.0)),
        (complex(nan, inf), 2 + 1j),
        (complex(1e30, 1e30), complex(1e30, -1e30)),
        (3 + 4j, -1.5 + 0.25j),
        (complex(-0.0, 0.0), complex(0.0, -0.0)),
    ]
    N = 3 * len(pairs)
    for udt, element in [(rec, lambda z: (z, z)), (arr, lambda z: [z, -z])]:
        v = gb.Vector(udt, N)
        u = gb.Vector(udt, N)
        for i in range(N):
            a, b = pairs[i % len(pairs)]
            v[i] = element(a)
            u[i] = element(b)
        with _jit_mode("on"):
            jit = v.ewise_mult(u, binary.times).new()
        with _jit_mode("off"):
            cfunc = v.ewise_mult(u, binary.times).new()
        assert jit.to_dense().tobytes() == cfunc.to_dense().tobytes()
    result = jit[0].new().value  # 1j * (inf+infj), numpy's nan+nanj
    assert np.isnan(result[0].real)
    assert np.isnan(result[0].imag)


@pytest.mark.skipif("not supports_udfs")
def test_abs_udt_jit_matches_python_negative_zero():
    """``unary.abs`` on a float UDT clears the sign bit of ``-0.0`` in the JIT kernel.

    Python's ``abs``, and so the cfunc, gives ``0.0``. Regression: the C codegen
    emitted ``(x < 0 ? -x : x)``, which returns ``-0.0`` unchanged because
    ``-0.0 < 0`` is false.
    """
    _require_jit_on()

    import struct

    def sign_bit(f):
        return struct.pack("<d", f)[7] & 0x80

    # Field names unique to this test; see test_floordiv_udt_jit_matches_python_semantics.
    udt = dtypes.register_anonymous(
        np.dtype([("abs_a", np.float64), ("abs_b", np.float64)]), "_AbsNegZeroJit"
    )
    N = 100
    v = gb.Vector(udt, N)
    for i in range(N):
        # field abs_a is always -0.0; field abs_b varies so the vector isn't iso.
        v[i] = (-0.0, float(i) - 50)

    w = unary.abs(v).new()
    val = w[0].new().value
    assert val[0] == 0.0
    assert (
        sign_bit(val[0]) == 0
    ), f"abs(-0.0) preserved the sign bit: bits={val[0].view('<u8'):016x}"


# JIT-vs-cfunc parity inputs, keyed by udt_kind.
# Each entry is (np.dtype-spec, "_TypeName", fill(i) -> (a, b)).
#
# Per-variant unique field names: anonymous UDTs are keyed on ``np.dtype``, so
# same-shape dtypes across variants would share a single ``DataType`` (and its
# C name from the first registration), making the tests depend on their order.


def _parity_record_i64(i):
    # cross-sign division operands
    return (-7 - 2 * i, 7 + 2 * i), (2, -3)


def _parity_record_f64(i):
    nan = float("nan")
    inf = float("inf")
    if i == 0:
        return (nan, 1.0), (2.0, 3.0)
    if i == 1:
        return (1.0, nan), (inf, -inf)
    if i == 2:
        return (-7.5, 7.5), (2.0, 2.0)
    return (-7.0 - i, 7.0 + i), (2.0 + 0.1 * i, 3.0)


def _parity_record_u32(i):
    return (10 + i, 20 + i), (2, 3)


def _parity_record_mixed(i):
    return (5 + i, 1.5 + i), (2, 0.5)


def _parity_array_f32(i):
    return (
        np.array([-7.0 - i, 7.0 + i, -8.0 - i, 8.0 + i], np.float32),
        np.array([2.0, 2.0, 3.0, 3.0], np.float32),
    )


def _parity_array_i16(i):
    return (
        np.array([1 + i, 2 + i, 3 + i, 4 + i], np.int16),
        np.array([2, 2, 3, 3], np.int16),
    )


_PARITY_VARIANTS = {
    "record_i64": (
        np.dtype([("p_i_x", np.int64), ("p_i_y", np.int64)]),
        "_ParityI64",
        _parity_record_i64,
    ),
    "record_f64": (
        np.dtype([("p_f_x", np.float64), ("p_f_y", np.float64)]),
        "_ParityF64",
        _parity_record_f64,
    ),
    "record_u32": (
        np.dtype([("p_u_x", np.uint32), ("p_u_y", np.uint32)]),
        "_ParityU32",
        _parity_record_u32,
    ),
    "record_mixed": (
        np.dtype([("p_m_i", np.int32), ("p_m_f", np.float64)]),
        "_ParityMixed",
        _parity_record_mixed,
    ),
    "array_f32": (
        np.dtype((np.float32, (4,))),
        "_ParityArrF32",
        _parity_array_f32,
    ),
    "array_i16": (
        np.dtype((np.int16, (4,))),
        "_ParityArrI16",
        _parity_array_i16,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("udt_kind", list(_PARITY_VARIANTS))
def test_udt_op_jit_cfunc_parity(udt_kind):
    """The JIT kernel and the cfunc give the same bytes for each lifted op listed below.

    The two come from separate code generators (C templates and Numba). N=64
    non-iso elements make SuiteSparse run the kernel. The variants cover
    signed-integer floor division, float floor with NaN and inf, unsigned
    division, mixed widths with C alignment, and two array shapes.
    """
    _require_jit_on()
    if numba is None:
        pytest.skip("numba required for the cfunc baseline")

    np_dtype, type_name, fill = _PARITY_VARIANTS[udt_kind]
    udt = dtypes.register_anonymous(np_dtype, type_name)
    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        a, b = fill(i)
        v[i] = a
        u[i] = b

    binary_ops = ["plus", "minus", "times", "truediv", "floordiv", "min", "max"]
    unary_ops = ["ainv", "abs"]

    with _jit_mode("on"):
        jit_binary = {op: v.ewise_mult(u, getattr(binary, op)).new() for op in binary_ops}
        jit_unary = {op: getattr(unary, op)(v).new() for op in unary_ops}
    # cfunc path: JIT off so SS uses the registered function pointer instead of
    # compiling a new kernel.
    with _jit_mode("off"):
        cf_binary = {op: v.ewise_mult(u, getattr(binary, op)).new() for op in binary_ops}
        cf_unary = {op: getattr(unary, op)(v).new() for op in unary_ops}

    def values_equal(j, c):
        # Compare bytes, not ``isequal``: ``binary.eq`` on a UDT follows IEEE, so
        # records with a NaN never compare equal, whether or not the paths agree.
        if j.dtype != c.dtype or j.nvals != c.nvals:
            return False
        return j.to_dense().tobytes() == c.to_dense().tobytes()

    for op in binary_ops:
        assert values_equal(
            jit_binary[op], cf_binary[op]
        ), f"binary.{op} on {udt_kind}: JIT and cfunc disagree"
    for op in unary_ops:
        assert values_equal(
            jit_unary[op], cf_unary[op]
        ), f"unary.{op} on {udt_kind}: JIT and cfunc disagree"


@pytest.mark.skipif("not supports_udfs")
def test_anonymous_udt_with_no_name_still_jits():
    """A UDT registered without ``name=`` gets a C name, so its lifted ops have JIT kernels.

    Its Python name, such as ``"{'a': FP64, 'b': INT64}"``, is not a C
    identifier, so ``_pick_c_type_name`` makes a ``_gbudt_NNN`` name for
    SuiteSparse and leaves ``udt.name`` alone. Without a C name, the ops run
    their Numba cfuncs and ``jit_c_source`` is None.
    """
    _require_jit_on()

    spec = np.dtype([("anon_a", np.float64), ("anon_b", np.int64)], align=True)
    udt = dtypes.register_anonymous(spec)  # no name=

    # Python-side default name is the np.dtype repr; not a valid C identifier.
    assert udt.name != udt.jit_c_name
    assert udt.jit_c_name is not None
    assert udt.jit_c_name.startswith("_gbudt_")
    assert udt.jit_c_definition is not None

    # Lifted ops use that C name.
    plus_op = binary.plus[udt]
    assert plus_op.jit_c_source is not None
    assert plus_op.jit_c_name.startswith("plus__gbudt_")

    v = gb.Vector(udt, 2)
    v[0] = (1.0, 10)
    v[1] = (2.0, 20)
    w = gb.Vector(udt, 2)
    w[0] = (3.0, 30)
    w[1] = (4.0, 40)
    result = v.ewise_mult(w, binary.plus).new()
    assert result[0].new().value.tolist() == (4.0, 40)
    assert result[1].new().value.tolist() == (6.0, 60)


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_complex_field_udt_jits_arithmetic():
    """A record with a complex field gets JIT kernels that match the cfunc.

    ``min``, ``max`` and ``floordiv`` raise KeyError when typed, rather than
    crashing in Numba. The C99 ``_Complex`` types come from SuiteSparse's
    GraphBLAS.h (as ``GxB_FC32_t`` and ``GxB_FC64_t``), so the kernel needs no
    extra includes. ``abs`` uses ``cabs`` or ``cabsf``, which match Numba's
    ``abs`` of a complex (the magnitude, in the real part).
    """
    _require_jit_on()

    spec = np.dtype([("cval", np.complex128), ("scale", np.float64)])
    udt = dtypes.register_anonymous(spec, "_JitCplx1")
    assert "GxB_FC64_t" in udt.jit_c_definition

    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (complex(1.0 + i, 2.0), 3.0 + i)
        u[i] = (complex(0.5, 0.25 + i), 0.1)

    # truediv has its own test, test_complex_truediv_udt_jit_matches_cfunc_at_extremes.
    for op_name in ("plus", "minus", "times"):
        op = getattr(binary, op_name)
        with _jit_mode("on"):
            wj = v.ewise_mult(u, op).new()
        with _jit_mode("off"):
            wc = v.ewise_mult(u, op).new()
        assert wj.isequal(wc, check_dtype=True), f"binary.{op_name} on complex UDT diverged"
    for op_name in ("abs", "ainv"):
        op = getattr(unary, op_name)
        with _jit_mode("on"):
            wj = op(v).new()
        with _jit_mode("off"):
            wc = op(v).new()
        assert wj.isequal(wc, check_dtype=True), f"unary.{op_name} on complex UDT diverged"

    # abs source must use cabs (not the ternary, which doesn't compile on _Complex).
    assert "cabs(" in unary.abs[udt].jit_c_source

    for op_name in ("min", "max", "floordiv"):
        with pytest.raises(KeyError, match="complex fields"):
            getattr(binary, op_name)[udt]


@pytest.mark.skipif("not supports_udfs")
def test_packed_record_layout_skips_jit():
    """A packed numpy dtype (no inter-field padding) must skip the JIT path.

    Otherwise the JIT-compiled kernel reads fields at the C-aligned offsets
    while the numpy buffer holds them at packed offsets, producing silent
    garbage. The cfunc path still works because Numba mirrors numpy's
    layout. Users wanting JIT can re-register with ``align=True`` or use
    the dict / dataclass form (which auto-aligns).
    """
    # int32 + float64 packed: numpy offsets 0, 4 (itemsize 12);
    # C struct would put float64 at offset 8 (itemsize 16).
    packed = np.dtype([("p_flag", np.int32), ("p_val", np.float64)])  # no align=True
    udt = dtypes.register_anonymous(packed, "_PackedUdt1")
    assert udt.jit_c_name is None
    assert udt.jit_c_definition is None

    v = gb.Vector(udt, 4)
    v[0] = (1, 1.5)
    v[1] = (2, 2.5)
    w = v.ewise_mult(v, binary.plus).new()
    assert w[0].new() == (2, 3.0)


@pytest.mark.skipif("not supports_udfs")
def test_packed_nested_record_layout_skips_jit():
    """A packed record nested in a packed record skips the JIT path too.

    numpy's ``align=True`` keeps a packed inner record packed (alignment 1),
    where a C compiler aligns it to 8 bytes. Regression: the kernel read the
    wrong offsets.
    """
    spec = np.dtype(
        [
            ("flag", np.int32),
            ("coord", [("x", np.float64), ("y", np.float64)]),
        ],
    )  # no align=True at either level
    # numpy itemsize 20 (flag@0, coord@4); C would pick 24 (flag@0, pad, coord@8).
    assert spec.itemsize == 20
    udt = dtypes.register_anonymous(spec, "_PackedNestedUdt")
    assert udt.jit_c_name is None
    assert udt.jit_c_definition is None


@pytest.mark.skipif("not supports_udfs")
def test_record_layouts_c_cannot_declare_skip_jit():
    """A record whose fields a C struct cannot lay out the same way skips the JIT.

    Explicit offsets can give a record a C struct's size with a field
    somewhere else, as can such a record nested in one whose own offsets
    match. A nested record with an array field has no C struct here, so the
    record holding it has none either. Each computes with the cfunc.
    """
    # Same size as the C struct (8 bytes), but the int32 at offset 1, not 4.
    offset = np.dtype(
        {
            "names": ["ol_a", "ol_b"],
            "formats": [np.int8, np.int32],
            "offsets": [0, 1],
            "itemsize": 8,
        }
    )
    outer = np.dtype([("ol_in", offset), ("ol_z", np.float64)])
    with_array = np.dtype(
        [("ol_x", np.float64), ("ol_rec", [("ol_v", np.float64, (2,)), ("ol_w", np.float64)])],
    )
    for np_type, value, doubled in [
        (offset, (1, 400), (2, 800)),
        (outer, ((1, 400), 2.5), ((2, 800), 5.0)),
        (with_array, (1.5, ([2, 3], 4)), (3.0, ([4, 6], 8))),
    ]:
        udt = dtypes.register_anonymous(np_type)
        assert udt.jit_c_name is None
        assert udt.jit_c_definition is None
        v = gb.Vector(udt, 1)
        v[0] = value
        assert ((v + v).new().to_coo()[1] == np.array([doubled], dtype=np_type)).all()


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_nested_record_udt_jits():
    """Record-of-record UDTs take the JIT path.

    The codegen emits inner typedefs first in ``GxB_JIT_C_DEFINITION``, with
    synthesized ``_gbnest_NNN`` names to avoid collisions with user types.
    Op bodies use C nested-struct access (``z->outer.inner_a``), and the
    Numba cfunc wrapper writes the flat result tuple leaf-by-leaf back into
    the nested record fields (Numba can't ``setitem`` a tuple to a record
    field, but leaf scalar writes work).
    """
    _require_jit_on()

    # ``align=True``, so numpy's offsets match a C compiler's: a packed record
    # gets no C definition (test_packed_nested_record_layout_skips_jit).
    spec = np.dtype(
        [
            ("nest_flag", np.int32),
            ("nest_coord", [("nest_x", np.float64), ("nest_y", np.float64)]),
        ],
        align=True,
    )
    udt = dtypes.register_anonymous(spec, "_JitNested1")

    # Outer + inner typedefs both appear in the definition; the outer
    # references the inner by its synthesized C name.
    defn = udt.jit_c_definition
    assert defn.count("typedef struct") == 2
    assert "_gbnest_" in defn
    assert "_JitNested1" in defn

    plus_src = binary.plus[udt].jit_c_source
    # Nested struct access uses ``.`` after the leading ``->``.
    assert "z->nest_coord.nest_x = (x->nest_coord.nest_x) + (y->nest_coord.nest_x)" in plus_src

    N = 64
    v = gb.Vector(udt, N)
    u = gb.Vector(udt, N)
    for i in range(N):
        v[i] = (i, (1.0 + i, 2.0 + i))
        u[i] = (1, (0.5, -0.5))

    for op_name in ("plus", "minus", "times"):
        op = getattr(binary, op_name)
        with _jit_mode("on"):
            wj = v.ewise_mult(u, op).new()
        with _jit_mode("off"):
            wc = v.ewise_mult(u, op).new()
        assert wj.isequal(wc, check_dtype=True), f"binary.{op_name} on nested UDT diverged"
    for op_name in ("abs", "ainv"):
        op = getattr(unary, op_name)
        with _jit_mode("on"):
            wj = op(v).new()
        with _jit_mode("off"):
            wc = op(v).new()
        assert wj.isequal(wc, check_dtype=True), f"unary.{op_name} on nested UDT diverged"


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_nested_record_monoid_semiring_agg():
    r"""Built-in Monoids, Semirings and Aggregators lift to a nested record UDT.

    The identity has the record's nested shape. Regression: ``_udt_identity``
    gave a record field the identity ``numpy.void(b'\x00')``, which
    ``Scalar.value`` rejected.
    """
    spec = np.dtype(
        [
            ("agg_id", np.int32),
            ("agg_pt", [("agg_x", np.float64), ("agg_y", np.float64)]),
        ],
        align=True,
    )
    udt = dtypes.register_anonymous(spec, "_NestedMonoid1")

    plus_id = monoid.plus[udt].identity.value
    assert plus_id["agg_id"] == 0
    assert plus_id["agg_pt"]["agg_x"] == 0.0
    assert plus_id["agg_pt"]["agg_y"] == 0.0
    min_id = monoid.min[udt].identity.value
    assert min_id["agg_id"] == np.iinfo(np.int32).max
    assert min_id["agg_pt"]["agg_x"] == np.inf
    assert min_id["agg_pt"]["agg_y"] == np.inf
    max_id = monoid.max[udt].identity.value
    assert max_id["agg_id"] == np.iinfo(np.int32).min
    assert max_id["agg_pt"]["agg_x"] == -np.inf
    assert max_id["agg_pt"]["agg_y"] == -np.inf

    v = gb.Vector(udt, size=3)
    v[0] = (1, (1.0, 2.0))
    v[1] = (2, (3.0, 4.0))
    v[2] = (3, (5.0, 6.0))
    s = v.reduce(agg.sum[udt]).new().value
    assert s["agg_id"] == 6
    assert s["agg_pt"]["agg_x"] == 9.0
    assert s["agg_pt"]["agg_y"] == 12.0

    A = gb.Matrix(udt, nrows=2, ncols=2)
    B = gb.Matrix(udt, nrows=2, ncols=2)
    A[0, 0] = (1, (1.0, 0.0))
    A[0, 1] = (2, (0.0, 1.0))
    A[1, 0] = (3, (1.0, 1.0))
    A[1, 1] = (4, (1.0, 1.0))
    B[0, 0] = (10, (1.0, 0.0))
    B[0, 1] = (20, (0.0, 1.0))
    B[1, 0] = (30, (1.0, 1.0))
    B[1, 1] = (40, (1.0, 1.0))
    C = A.mxm(B, semiring.plus_times[udt]).new()
    c00 = C[0, 0].new().value
    c11 = C[1, 1].new().value
    assert (c00["agg_id"], c00["agg_pt"]["agg_x"], c00["agg_pt"]["agg_y"]) == (70, 1.0, 1.0)
    assert (c11["agg_id"], c11["agg_pt"]["agg_x"], c11["agg_pt"]["agg_y"]) == (220, 1.0, 2.0)


# Per-shape fill functions for the eq/ne JIT-vs-cfunc parity test below.
# Each returns ``(v1_record, v2_record)`` for index ``i``. Variants that put
# a NaN at index 0 also exercise IEEE comparison semantics (eq(NaN,NaN) is
# False), pinned by the ``nan_at_zero`` flag in ``_EQ_NE_VARIANTS``.


def _eq_ne_fill_record_float(i):
    nan = float("nan")
    if i == 0:
        return (nan, 1.0), (nan, 1.0)
    return (1.0 + i, 2.0), (1.0 + i, 2.0 if i % 2 == 0 else 3.0)


def _eq_ne_fill_record_int(i):
    return (i, 2 * i), (i, 2 * i if i % 3 == 0 else 2 * i + 1)


def _eq_ne_fill_record_nested(i):
    return (i, (1.0, 2.0)), (i, (1.0, 2.0) if i % 3 != 0 else (1.5, 2.0))


def _eq_ne_fill_record_complex(i):
    return (
        (complex(1.0 + i, 2.0), 3.0),
        (complex(1.0 + i, 2.0 if i % 2 == 0 else 3.0), 3.0),
    )


def _eq_ne_fill_array_f64(i):
    return (
        np.array([1.0 + i, 2.0, 3.0]),
        np.array([1.0 + i, 2.0 if i % 2 == 0 else -2.0, 3.0]),
    )


_EQ_NE_VARIANTS = {
    "record_float": (
        np.dtype([("eq_f_a", np.float64), ("eq_f_b", np.float64)], align=True),
        "_EqJitF",
        _eq_ne_fill_record_float,
        True,
    ),
    "record_int": (
        np.dtype([("eq_i_a", np.int64), ("eq_i_b", np.int32)], align=True),
        "_EqJitI",
        _eq_ne_fill_record_int,
        False,
    ),
    "record_nested": (
        np.dtype(
            [("eq_n_id", np.int32), ("eq_n_pt", [("nx", np.float64), ("ny", np.float64)])],
            align=True,
        ),
        "_EqJitN",
        _eq_ne_fill_record_nested,
        False,
    ),
    "record_complex": (
        np.dtype([("eq_c_z", np.complex128), ("eq_c_s", np.float64)]),
        "_EqJitC",
        _eq_ne_fill_record_complex,
        False,
    ),
    "array_f64": (
        np.dtype((np.float64, (3,))),
        "_EqJitArr",
        _eq_ne_fill_array_f64,
        False,
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("shape", list(_EQ_NE_VARIANTS))
def test_eq_ne_udt_jit_matches_cfunc(shape):
    """``binary.eq[udt]`` / ``binary.ne[udt]`` JIT-compile a leaf-wise
    comparison kernel and produce the same result as the cfunc fallback.

    Verifies the JIT kernel for several shapes (flat record, nested
    record, complex, array UDT), and that NaN propagation matches IEEE
    (``eq(NaN, NaN) == False``) on variants flagged ``nan_at_zero``.
    """
    _require_jit_on()
    if numba is None:
        pytest.skip("numba required for the cfunc baseline")

    np_dtype, type_name, fill, nan_at_zero = _EQ_NE_VARIANTS[shape]
    udt = dtypes.register_anonymous(np_dtype, type_name)
    N = 64
    v1 = gb.Vector(udt, N)
    v2 = gb.Vector(udt, N)
    for i in range(N):
        a, b = fill(i)
        v1[i] = a
        v2[i] = b

    assert binary.eq[udt].jit_c_source is not None
    assert binary.ne[udt].jit_c_source is not None

    with _jit_mode("on"):
        eq_jit = v1.ewise_mult(v2, binary.eq).new()
        ne_jit = v1.ewise_mult(v2, binary.ne).new()
    with _jit_mode("off"):
        eq_cf = v1.ewise_mult(v2, binary.eq).new()
        ne_cf = v1.ewise_mult(v2, binary.ne).new()

    # Byte-equal across paths (the result is BOOL, so this is exact).
    assert (
        eq_jit.to_dense().tobytes() == eq_cf.to_dense().tobytes()
    ), f"eq on {shape}: JIT and cfunc disagree"
    assert (
        ne_jit.to_dense().tobytes() == ne_cf.to_dense().tobytes()
    ), f"ne on {shape}: JIT and cfunc disagree"

    if nan_at_zero:
        # Both records carry NaN at index 0; IEEE ``a == a`` is False.
        assert eq_jit[0].new().value is False
        assert ne_jit[0].new().value is True


@pytest.mark.skipif("not supports_udfs")
def test_eq_ne_udt_cfunc_works_for_unrepresentable_dtypes():
    """``binary.eq`` cfunc path produces correct results on UDTs that can't take JIT.

    ``test_packed_record_layout_skips_jit`` and, in ``test_dtype.py``,
    ``test_udt_with_c_reserved_field_name_falls_back_to_cfunc`` check that such
    UDTs get no C definition; this checks that ``eq``, whose codegen is separate
    from that of ``plus``, still computes on them.
    """
    # Packed mixed-width: numpy offset 4 for float, C would use 8.
    packed = np.dtype([("eq_pk_a", np.int32), ("eq_pk_b", np.float64)])
    udt = dtypes.register_anonymous(packed, "_EqJitPacked")
    assert binary.eq[udt].jit_c_source is None
    v1 = gb.Vector(udt, 2)
    v2 = gb.Vector(udt, 2)
    v1[0] = (1, 2.5)
    v2[0] = (1, 2.5)
    v1[1] = (1, 2.5)
    v2[1] = (1, 3.5)
    eq = v1.ewise_mult(v2, binary.eq).new()
    assert eq[0].new().value is True
    assert eq[1].new().value is False


@pytest.mark.skipif("not supports_udfs")
def test_udt_scalar_broadcast_jit_routing():
    """Mixed UDT and scalar arithmetic gets a C definition with a type for each of z, x
    and y; eq/ne on a mixed pair, and ops that have no C form, keep the cfunc.

    Changing this routing, in either direction, should be a deliberate decision.
    """
    record = dtypes.register_anonymous(
        np.dtype([("bcast_u", np.float64), ("bcast_v", np.float64)], align=True),
        name="_BcastJitUV",
    )
    # Same-type pairs do JIT.
    assert binary.plus[record, record].jit_c_source is not None
    assert binary.eq[record, record].jit_c_source is not None
    # Mixed arithmetic does, with the scalar read once for every leaf
    assert binary.plus[record, dtypes.INT64].jit_c_source == (
        "void plus_17_gbudt__BcastJitUV_int64_t (gbudt__BcastJitUV *z, "
        "const gbudt__BcastJitUV *x, const int64_t *y) { "
        "z->bcast_u = (x->bcast_u) + (((double) ((*y)))) ; "
        "z->bcast_v = (x->bcast_v) + (((double) ((*y)))) ; }"
    )
    assert binary.plus[dtypes.INT64, record].jit_c_source is not None
    # Comparisons on a mixed pair keep the cfunc
    assert binary.eq[record, dtypes.INT64].jit_c_source is None
    assert binary.ne[record, dtypes.FP64].jit_c_source is None


@pytest.mark.skipif("not supports_udfs")
def test_udt_mixed_jit_c_source():
    """The C definitions of lifted ops on a scalar and an array UDT, and the cases with no C
    form (an array operand that broadcasts, integer truediv, an array leaf in a record).
    """
    arr = dtypes.register_anonymous(np.dtype((np.float64, (13,))), "_MixedArr13")
    # (the C name of whichever test registered this numpy type first: ``test_op`` has it too)
    c = arr.jit_c_name
    assert binary.times[dtypes.FP64, arr].jit_c_source == (
        f"void times_6_double_{c} ({c} *z, const double *x, "
        f"const {c} *y) {{ for (int64_t i = 0 ; i < 13 ; i++) "
        "{ z->v[i] = ((*x)) * (y->v[i]) ; } }"
    )
    assert binary.times[dtypes.FP64, arr].jit_c_name == f"times_6_double_{c}"
    assert binary.times[arr, dtypes.FP64].jit_c_name == f"times_{len(c)}_{c}_double"
    # plus_times on these types: its monoid is the same-type plus
    semi = gb.semiring.plus_times[dtypes.FP64, arr]
    assert semi.jit_c_source == binary.times[dtypes.FP64, arr].jit_c_source
    assert semi.monoid.jit_c_source is not None
    # Operands converted to the type the op computes in, as the cfunc converts them
    assert "(((double) ((*x)))) - (y->v[i])" in binary.minus[dtypes.INT32, arr].jit_c_source
    arr_i8 = dtypes.register_anonymous(np.dtype((np.int8, (13,))), "_MixedArrI8")
    assert "(((double) (x->v[i]))) + (y->v[i])" in binary.plus[arr_i8, arr].jit_c_source
    # No C form: these keep the cfunc
    arr1 = dtypes.register_anonymous(np.dtype((np.float64, (1,))), "_MixedArr1")
    assert binary.plus[arr1, arr].jit_c_source is None  # (broadcasts)
    assert binary.truediv[dtypes.INT64, arr_i8].jit_c_source is None
    nested = dtypes.register_anonymous(
        np.dtype([("mx_w", np.float64, (2,)), ("mx_c", np.int64)]), "_MixedArrLeaf"
    )
    assert binary.plus[nested, dtypes.INT64].jit_c_source is None


@pytest.mark.skipif("not supports_udfs")
def test_udt_mixed_jit_c_name_is_its_own():
    """A mixed op's C name is not a one-type op's on a UDT named like the pair, nor another
    pair's. SuiteSparse finds a compiled kernel by its op's name. Regression: the second op
    ran the first's kernel, giving wrong values or writing past the end of a smaller element.
    """
    a = dtypes.register_anonymous(np.dtype((np.float64, (19,))), "_CnA")
    b = dtypes.register_anonymous(np.dtype((np.int64, (19,))), "_CnB")
    ab = dtypes.register_anonymous(np.dtype((np.int8, (19,))), f"{a.jit_c_name}_{b.jit_c_name}")
    assert binary.plus[ab, ab].jit_c_name != binary.plus[a, b].jit_c_name
    n = 50
    # Values that vary, or SuiteSparse before 10.5 builds iso vectors and computes an iso
    # result without the kernel
    k = np.broadcast_to(np.arange(n)[:, None] % 7, (n, 19))
    vab = Vector.from_coo(np.arange(n), k.astype(np.int8), dtype=ab)
    va = Vector.from_coo(np.arange(n), k + 1.5, dtype=a)
    vb = Vector.from_coo(np.arange(n), 10 * k, dtype=b)
    with _jit_mode("on"):
        same = vab.ewise_mult(vab, binary.plus).new()
        mixed = va.ewise_mult(vb, binary.plus).new()
    assert (same.to_dense() == 2 * k).all()
    assert (mixed.to_dense() == 11 * k + 1.5).all()


def _mixed_parity_cases():
    rng = np.random.default_rng(17)
    arr_f = dtypes.register_anonymous(np.dtype((np.float64, (5,))), "_MixedParF")
    arr_i = dtypes.register_anonymous(np.dtype((np.int64, (5,))), "_MixedParI")
    arr_c = dtypes.register_anonymous(np.dtype((np.complex128, (5,))), "_MixedParC")
    rec = dtypes.register_anonymous(
        np.dtype([("mp_a", np.float64), ("mp_b", np.int32)], align=True), "_MixedParRec"
    )
    n = 64
    vf = rng.standard_normal((n, 5))
    vf[0, :2] = [np.nan, np.inf]
    vi = rng.integers(-9, 10, (n, 5))
    vc = rng.standard_normal((n, 5)) + 1j * rng.standard_normal((n, 5))
    vr = np.zeros(n, rec.np_type)
    vr["mp_a"] = rng.standard_normal(n)
    vr["mp_b"] = rng.integers(-9, 10, n)
    sf = rng.standard_normal(n)
    sf[1] = 0.0
    si = rng.integers(-3, 4, n)
    su = rng.integers(0, 2**63, n, dtype=np.uint64)
    idx = np.arange(n)
    u = {
        "FP64[5]": Vector.from_coo(idx, vf, dtype=arr_f),
        "INT64[5]": Vector.from_coo(idx, vi, dtype=arr_i),
        "FC64[5]": Vector.from_coo(idx, vc, dtype=arr_c),
        "record": Vector.from_coo(idx, vr, dtype=rec),
    }
    s = {
        "FP64": Vector.from_coo(idx, sf),
        "INT64": Vector.from_coo(idx, si),
        "UINT64": Vector.from_coo(idx, su),
    }
    real_ops = ["plus", "minus", "times", "truediv", "floordiv", "min", "max"]
    return (
        [
            ("FP64[5]", "FP64", real_ops),
            ("FP64[5]", "INT64", real_ops),
            ("INT64[5]", "UINT64", real_ops),  # (computed in FP64, as on built-in vectors)
            ("INT64[5]", "INT64", ["plus", "minus", "times", "floordiv", "min", "max"]),
            ("FC64[5]", "FP64", ["plus", "minus", "times", "truediv"]),
            ("record", "FP64", real_ops),
            ("record", "INT64", ["plus", "minus", "times", "min", "max"]),
        ],
        u,
        s,
    )


@pytest.mark.slow
@pytest.mark.skipif("not supports_udfs")
def test_udt_mixed_jit_cfunc_parity():
    """A mixed op's C kernel and its Numba cfunc give the same bytes, the UDT on either side."""
    _require_jit_on()
    cases, u, s = _mixed_parity_cases()
    checked = 0
    for ukey, skey, ops in cases:
        for op in ops:
            for left, right in [(u[ukey], s[skey]), (s[skey], u[ukey])]:
                typed = getattr(binary, op)[left.dtype, right.dtype]
                if typed.jit_c_source is None:
                    continue
                with _jit_mode("on"):
                    jit = left.ewise_mult(right, getattr(binary, op)).new()
                with _jit_mode("off"):
                    cf = left.ewise_mult(right, getattr(binary, op)).new()
                assert jit.dtype == cf.dtype
                assert jit.nvals == cf.nvals
                assert (
                    jit.to_dense().tobytes() == cf.to_dense().tobytes()
                ), f"binary.{op}[{left.dtype}, {right.dtype}]: JIT and cfunc disagree"
                checked += 1
    assert checked > 60


@pytest.mark.skipif("not supports_udfs")
def test_udt_mixed_semiring_runs_jit(capfd):
    """The scalar x array semiring compiles as one JIT kernel (burble), with the result of the
    generic kernel.
    """
    _require_jit_on()
    arr = dtypes.register_anonymous(np.dtype((np.float64, (6,))), "_MixedSemi6")
    rng = np.random.default_rng(23)
    A = gb.Matrix.from_coo(
        rng.integers(0, 30, 200),
        rng.integers(0, 20, 200),
        rng.random(200),
        nrows=30,
        ncols=20,
        dup_op=binary.plus,
    )
    U = rng.random((20, 6))
    u = Vector.from_coo(np.arange(20), U, dtype=arr)
    semi = gb.semiring.plus_times[dtypes.FP64, arr]
    capfd.readouterr()
    with burble():
        w = A.mxv(u, semi).new()
        w.wait()
    out = capfd.readouterr().out
    assert "GrB_mxv" in out
    assert "generic" not in out
    # (and burble names the generic kernel when it runs)
    with _jit_mode("off"), burble():
        w_generic = A.mxv(u, semi).new()
        w_generic.wait()
    assert "generic" in capfd.readouterr().out
    idx, vals = w.to_coo()
    gidx, gvals = w_generic.to_coo()
    assert_array_equal(idx, gidx)
    np.testing.assert_allclose(vals, gvals, rtol=1e-13)
    np.testing.assert_allclose(vals, (A.to_dense(0) @ U)[idx], rtol=1e-13)


def test_context_importable():
    from graphblas.core.ss.context import global_context

    assert gb.ss.global_context is global_context


def test_fix_from_conda_finds_windows_compilers(tmp_path):
    """The conda repair looks in ``bin`` and, as Windows conda installs them, ``Library/bin``
    with ``.exe`` names; a baked-in Windows path's basename is found too. Nothing there: None.
    """
    from graphblas.core.ss.jit_config import _fix_from_conda

    def cfg_for(name):
        return {"jit_c_compiler_name": name, "jit_c_compiler_flags": "-O3"}

    win = tmp_path / "win"
    (win / "Library" / "bin").mkdir(parents=True)
    (win / "Library" / "bin" / "clang.exe").write_text("")
    cfg = cfg_for(r"C:\bld\_h_env\Library\bin\cl.exe")
    assert _fix_from_conda(cfg, str(win)) is True
    assert cfg["jit_c_compiler_name"] == str(win / "Library" / "bin" / "clang.exe")
    (win / "Library" / "bin" / "cl.exe").write_text("")
    cfg = cfg_for(r"C:\bld\_h_env\Library\bin\cl.exe")
    assert _fix_from_conda(cfg, str(win)) is True
    assert cfg["jit_c_compiler_name"] == str(win / "Library" / "bin" / "cl.exe")
    posix = tmp_path / "posix"
    (posix / "bin").mkdir(parents=True)
    (posix / "bin" / "gcc").write_text("")
    cfg = cfg_for("/home/conda/feedstock_root/build_artifacts/bin/x86_64-conda-linux-gnu-cc")
    assert _fix_from_conda(cfg, str(posix)) is True
    assert cfg["jit_c_compiler_name"] == str(posix / "bin" / "gcc")
    assert _fix_from_conda(cfg_for("/nowhere/cc"), str(tmp_path / "empty")) is None


@pytest.mark.skipif("not supports_udfs")
def test_udt_mixed_jit_c_name_fits_suitesparse():
    """A mixed op's C name from two long UDT names still fits SuiteSparse's 127 characters.

    SuiteSparse refuses a longer name and takes the definition under an empty one; the kernel
    it compiles then calls no function and leaves its output uninitialized.
    """
    _require_jit_on()
    rec = [("ln_x", np.float64), ("ln_y", np.float64)]
    rec32 = [("ln_x", np.float32), ("ln_y", np.float32)]
    a = dtypes.register_new("LongNameA" + "a" * 60, np.dtype(rec, align=True))
    b = dtypes.register_new("LongNameB" + "b" * 60, np.dtype(rec32, align=True))
    op = binary.plus[a, b]
    assert op.jit_c_source is not None
    assert len(op.jit_c_name) < gb.core.lib.GxB_MAX_NAME_LEN
    n = 40
    va = np.zeros(n, a.np_type)
    va["ln_x"] = np.arange(n)
    va["ln_y"] = 2 * np.arange(n)
    vb = np.zeros(n, b.np_type)
    vb["ln_x"] = 0.5
    vb["ln_y"] = np.arange(n) % 3
    u = Vector.from_coo(np.arange(n), va, dtype=a)
    w = Vector.from_coo(np.arange(n), vb, dtype=b)
    with _jit_mode("on"):
        r = u.ewise_mult(w, binary.plus).new()
    _, vals = r.to_coo()
    np.testing.assert_array_equal(vals["ln_x"], np.arange(n) + 0.5)
    np.testing.assert_array_equal(vals["ln_y"], 2 * np.arange(n) + np.arange(n) % 3)


@pytest.mark.skipif("not supports_udfs")
def test_udt_jit_c_name_too_long_keeps_the_cfunc():
    """An op whose C name SuiteSparse refuses (128 characters or more) runs its cfunc.

    Setting the definition after a refused name would compile a kernel that calls no
    function, so the op gets no C form at all.
    """
    _require_jit_on()
    rec = np.dtype([("tl_x", np.float64), ("tl_y", np.int64)], align=True)
    # A numpy UDT's C name leaves room for its ops' names, but one registered from C
    # source keeps the name given: 125 characters fits, plus_ or eq_ and that does not.
    name = "TooLongName" + "t" * 114
    t = dtypes.ss.register_new(
        name, f"typedef struct {{ double tl_x ; int64_t tl_y ; }} {name} ;", np_type=rec
    )
    assert len(t.jit_c_name) == 125
    n = 40
    vals = np.zeros(n, t.np_type)
    vals["tl_x"] = np.arange(n) / 4
    vals["tl_y"] = np.arange(n)
    v = Vector.from_coo(np.arange(n), vals, dtype=t)
    plus_op = binary.plus[t]
    eq_op = binary.eq[t]
    assert plus_op.jit_c_source is None
    assert eq_op.jit_c_source is None
    with _jit_mode("on"):
        r = v.ewise_mult(v, binary.plus).new()
        same = v.ewise_mult(v, binary.eq).new()
    _, got = r.to_coo()
    np.testing.assert_array_equal(got["tl_x"], np.arange(n) / 2)
    np.testing.assert_array_equal(got["tl_y"], 2 * np.arange(n))
    assert same.reduce(monoid.land).new()


@pytest.mark.skipif("not supports_udfs")
def test_udt_field_named_like_a_standard_macro_skips_the_jit():
    """A field named like a macro of the C standard headers the JIT includes skips the JIT.

    The preprocessor would expand it inside the struct. Regression: the kernel failed to
    compile, raising an empty ``JitError`` and leaving 'load'. The test checks the C name
    and the warning, so nothing compiles here.
    """
    from graphblas.exceptions import NoJITWarning

    fields = ["INT_MAX", "UINT8_MAX", "SIZE_MAX", "BUFSIZ", "EXIT_FAILURE", "PRId64", "EDOM"]
    # Macros from <signal.h>, and GraphBLAS's own, such as GrB_INDEX_MAX
    fields += ["SIGINT", "SIG_DFL", "GrB_INDEX_MAX", "GxB_IMPLEMENTATION_MAJOR", "GB_ANY"]
    for i, field in enumerate(fields):
        record = np.dtype([(field, np.float64), (f"macro_other_{i}", np.float64)], align=True)
        udt = dtypes.register_anonymous(record)
        assert udt.jit_c_name is None, field
        v = Vector.from_coo([0], np.array([(1.0, 2.0)], dtype=record), dtype=udt, size=2)
        with pytest.warns(NoJITWarning, match="without JIT"):
            result = v.ewise_mult(v, binary.plus).new()
        assert result[0].new().value.tolist() == (2.0, 4.0), field


@pytest.mark.skipif("not supports_udfs")
def test_udt_named_like_an_identifier_in_the_kernel_takes_the_jit():
    """A UDT named like an identifier in a JIT kernel compiles, under its ``gbudt_`` C name.

    SuiteSparse's kernels use the type's name in the same scope as their own identifiers.
    Regression: a UDT named like one of them, such as ``z``, ``A`` or ``div``, failed to
    compile, raising an empty ``JitError`` and leaving 'load'.
    """
    _require_jit_on()
    for name in ["z", "A", "p", "i", "div", "result"]:
        record = np.dtype([(f"kn_{name}_a", np.float64), (f"kn_{name}_b", np.int64)], align=True)
        udt = dtypes.register_anonymous(record, name)
        assert udt.jit_c_name == f"gbudt_{name}"
        vals = np.array([(1.0, 2), (3.0, 4)], dtype=record)
        v = Vector.from_coo([0, 1], vals, dtype=udt)
        A = gb.Matrix.from_coo([0, 1], [1, 0], vals, dtype=udt)
        with _jit_mode("on"):
            plus = v.ewise_mult(v, binary.plus).new()
            biggest = v.reduce(monoid.max).new()
            product = A.mxv(v, semiring.plus_times).new()
            assert gb.ss.config["jit_c_control"] == "on", name
        assert plus.to_coo()[1].tolist() == [(2.0, 4), (6.0, 8)], name
        assert biggest.value.tolist() == (3.0, 4), name
        assert product.to_coo()[1].tolist() == [(3.0, 8), (3.0, 8)], name


@pytest.mark.skipif("not supports_udfs")
def test_udt_long_name_gets_a_c_name_that_fits():
    """A UDT whose ``gbudt_`` C name would not leave room for its ops' names gets a synthetic one.

    SuiteSparse refuses a name of 128 characters or more, and an op's C name adds a prefix
    such as ``floordiv_`` to the type's, so a type's C name can have at most 118 characters.
    Regression: a longer name was kept, so a kernel raised ``JitError``, or an op lost its
    JIT with a NoJITWarning that blamed the fields.
    """
    from graphblas.core.operator.udt_codegen import _max_type_c_name_len

    _require_jit_on()
    assert _max_type_c_name_len() == 118  # 127 less floordiv_
    for length in [112, 113, 119, 200]:
        record = np.dtype([(f"fit{length}_a", np.float64), (f"fit{length}_b", np.int64)])
        udt = dtypes.register_anonymous(record, "n" * length)
        if length == 112:
            assert udt.jit_c_name == "gbudt_" + "n" * 112  # 118 characters
        else:
            assert udt.jit_c_name.startswith("_gbudt_"), length
        for op in [binary.floordiv[udt], binary.plus[udt], binary.eq[udt], unary.ainv[udt]]:
            assert op.jit_c_source is not None, (length, op)
        vals = np.array([(1.0, 2), (3.0, 4)], dtype=record)
        A = gb.Matrix.from_coo([0, 1], [1, 0], vals, dtype=udt)
        with _jit_mode("on"):
            B = A.T.new()
            C = A.ewise_mult(A, binary.plus).new()
        assert B.to_coo()[2].tolist() == [(3.0, 4), (1.0, 2)], length
        assert C.to_coo()[2].tolist() == [(2.0, 4), (6.0, 8)], length


def test_udt_c_name_keeps_python_graphblas_own_names():
    # A type python-graphblas makes itself has a name starting with _gb, which no kernel
    # identifier starts with either, so it keeps that name as its C name; a user's name gets
    # the gbudt_ prefix, even one that starts with it.
    from graphblas.core.operator.udt_codegen import _pick_c_type_name

    assert _pick_c_type_name("Edge") == "gbudt_Edge"
    assert _pick_c_type_name("gbudt_Edge") == "gbudt_gbudt_Edge"
    for name in ["_gbudt_7", "_gbnest_3", "_gbinternal_fp64x16"]:
        assert _pick_c_type_name(name) == name


@pytest.mark.skipif("not supports_udfs")
def test_udt_from_c_source_lifts_ops_under_its_own_c_name():
    """An op lifted to a type registered from C source names the type as its C source does.

    Regression: ``binary.plus[t]`` named the type with the ``gbudt_`` prefix, which
    SuiteSparse did not know, and raised ``JitError``.
    """
    _require_jit_on()
    np_type = np.dtype([("cs_x", np.float64), ("cs_y", np.float64)], align=True)
    t = dtypes.ss.register_new(
        "cs_lift_t", "typedef struct { double cs_x ; double cs_y ; } cs_lift_t ;", np_type=np_type
    )
    assert t.jit_c_name == "cs_lift_t"
    assert binary.plus[t].jit_c_name == "plus_cs_lift_t"
    assert binary.eq[t].jit_c_name == "eq_cs_lift_t"
    v = Vector.from_coo([0, 1], np.array([(1.0, 2.0), (3.0, 4.0)], dtype=np_type), dtype=t)
    with _jit_mode("on"):
        r = v.ewise_mult(v, binary.plus).new()
        same = (v == v).new()
        assert gb.ss.config["jit_c_control"] == "on"
    assert r.to_coo()[1].tolist() == [(2.0, 4.0), (6.0, 8.0)]
    assert same.to_coo()[1].tolist() == [True, True]
    # A numpy UDT's own name always gets the prefix, even one that starts with it
    udt = dtypes.register_anonymous(np.dtype([("cs_pre", np.float64)]), "gbudt_CsPre")
    assert udt.jit_c_name == "gbudt_gbudt_CsPre"

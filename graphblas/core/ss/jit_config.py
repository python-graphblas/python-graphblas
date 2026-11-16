"""Set the SuiteSparse:GraphBLAS JIT control, and repair the JIT's C compiler setting.

A conda-built SuiteSparse:GraphBLAS has the build machine's compiler path and
flags in ``GxB_JIT_C_COMPILER_NAME`` and ``GxB_JIT_C_COMPILER_FLAGS``, and those
paths (such as ``/Users/runner/...``) do not exist on other machines. With them,
SuiteSparse writes a kernel's C source but cannot compile it. It then lowers
``jit_c_control`` to ``'load'`` and runs a generic kernel instead, which for a
lifted UDT operator calls its Numba function. Before SuiteSparse 9.4 that is
silent; from 9.4 on, the failed compile also raises ``JitError``, once.

``fix_jit_config`` and ``_auto_fix_jit_at_import`` replace the compiler with
one from ``$CONDA_PREFIX/bin/`` (outside conda, from ``sysconfig``) and fix the
flags that refer to the build machine. ``jit_compiler_is_usable`` checks that
the configured compiler exists, without compiling anything.

The control is set when the backend initializes (``_set_jit_c_control_at_init``)
and raised to ``'on'`` when C source needs compiling (``_enable_jit_for_udt``).
The rules are in the Technical Notes, "The JIT control"
(docs/technical_notes/jit.rst), and the trade-offs in the user guide's
initialization page.
"""

import contextlib
import hashlib
import os
import pathlib
import platform
import re
import subprocess

# ``ffi`` and ``lib`` are imported in the functions that use them: importing them
# here initializes the backend when this module is imported first, and init
# imports this module for ``_set_jit_c_control_at_init``.

# For each host architecture, as ``platform.machine().lower()`` names it, the
# ``-arch`` values that build for it.
_ARCH_ALIASES = {
    "x86_64": ("x86_64", "amd64"),
    "arm64": ("arm64", "aarch64"),
    "aarch64": ("arm64", "aarch64"),
    "i386": ("i386", "x86"),
    "ppc": ("ppc", "powerpc"),
    "ppc64": ("ppc64", "powerpc64"),
}


def _ss_config():
    # Imported lazily to avoid pulling all of ``gb.ss`` into module-load time.
    from ... import ss

    return ss.config


def jit_compiler_is_usable():
    """Return whether the configured JIT C compiler exists on disk.

    This compiles nothing, so it does not show that the compiler works;
    ``fix_jit_config()`` checks that.
    """
    return _compiler_exists(_ss_config())


def _compiler_exists(cfg):
    cc = cfg.get("jit_c_compiler_name", "")
    return bool(cc) and pathlib.Path(cc).exists()


def fix_jit_config(*, use_sysconfig=True, probe=True):
    """Repair the SuiteSparse:GraphBLAS JIT compiler configuration.

    If the configured compiler does not exist, as in a conda-built package on
    another machine, take one from ``$CONDA_PREFIX/bin/`` (on Windows,
    ``$CONDA_PREFIX/Library/bin/``). Fix the compiler flags that refer to the
    build machine: point a ``-isysroot`` whose SDK is missing at this machine's
    SDK (or drop it), and drop ``-arch`` for another architecture and
    ``-fdebug-prefix-map``. Then set ``jit_c_control`` to ``'on'``.

    Parameters
    ----------
    use_sysconfig : bool, default True
        When ``$CONDA_PREFIX`` is not set (a pip install), take the compiler
        and flags from ``sysconfig``. Set to ``False`` to repair only in a
        conda environment.
    probe : bool, default True
        After the repair, JIT-compile a tiny type to check that the compiler
        works. A failed compile makes SuiteSparse lower ``jit_c_control`` to
        ``'load'``, and from SuiteSparse 9.4 on it also raises ``JitError``
        that one time. The probe gets that error, so later operations run
        the generic kernels without raising.

    Returns
    -------
    True
        The repair is done and, with ``probe``, the compiler works.
    False
        The probe failed. ``jit_c_control`` is where SuiteSparse left it.
    None
        There was no compiler to repair from: none in ``$CONDA_PREFIX``, or,
        outside conda, none from ``sysconfig`` (or ``use_sysconfig=False``).
    """
    cfg = _ss_config()
    rv = _repair_jit_compiler(cfg, use_sysconfig=use_sysconfig)
    if rv is None:
        return None
    # Enabling compilation is what the caller asked for, whatever
    # ``graphblas.config["jit_c_control"]`` set at init.
    cfg["jit_c_control"] = "on"
    # Start over after a repair: ``NoJITWarning`` may warn again, and a probe
    # answer from before the repair is stale, so the next request to compile C
    # source probes again.
    _warned_no_jit_for.clear()
    global _jit_enabled_for_udt
    _jit_enabled_for_udt = None
    if not probe:
        return True
    return _probe_jit(cfg)


def _repair_jit_compiler(cfg, *, use_sysconfig=True):
    """Point the JIT compiler settings at a compiler that exists on this host.

    Changes only the compiler settings, not ``jit_c_control``; callers that
    want compilation enabled set the control themselves. Returns True, or
    None when there is no compiler to repair from.
    """
    if conda_prefix := os.environ.get("CONDA_PREFIX", ""):
        return _fix_from_conda(cfg, conda_prefix)
    if use_sysconfig:
        return _fix_from_sysconfig(cfg)
    return None


def _fix_from_conda(cfg, conda_prefix):
    """Fix the compiler flags, and the compiler too if it does not exist.

    The compiler comes from ``$CONDA_PREFIX/bin/``, or from
    ``$CONDA_PREFIX/Library/bin/`` with ``.exe`` names (Windows).
    """
    jit_cc = cfg["jit_c_compiler_name"]
    if not pathlib.Path(jit_cc).exists():
        # Split on both separators, so a Windows path gives its basename on any OS
        cc_basename = re.split(r"[\\/]", jit_cc)[-1]
        prefix = pathlib.Path(conda_prefix)
        names = [cc_basename, "cc", "clang", "gcc"]
        names += [n + ".exe" for n in names if not n.lower().endswith(".exe")]
        for bin_dir in [prefix / "bin", prefix / "Library" / "bin"]:
            local_cc = next((bin_dir / n for n in names if (bin_dir / n).exists()), None)
            if local_cc is not None:
                cfg["jit_c_compiler_name"] = str(local_cc)
                break
        else:
            return None  # nothing usable
    _fix_compiler_flags(cfg)
    return True


def _fix_from_sysconfig(cfg):
    """Pure-pip fallback; pull compiler info from ``sysconfig``."""
    import sysconfig

    cc = sysconfig.get_config_var("CC")
    cflags = sysconfig.get_config_var("CFLAGS")
    include = sysconfig.get_path("include")
    if not (cc and cflags and include):
        return None
    cfg["jit_c_compiler_name"] = cc
    cfg["jit_c_compiler_flags"] = f"{cflags} -I{include}"
    if libs := sysconfig.get_config_var("LIBS"):
        cfg["jit_c_libraries"] = libs
    return True


def _fix_compiler_flags(cfg):
    """Replace build-time-only paths in ``jit_c_compiler_flags``."""
    flags = cfg["jit_c_compiler_flags"]
    isysroot_match = re.search(r"-isysroot\s+(\S+)", flags)
    if isysroot_match and not pathlib.Path(isysroot_match.group(1)).exists():
        try:
            sdk_path = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            flags = re.sub(r"-isysroot\s+\S+", f"-isysroot {sdk_path}", flags)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No Xcode SDK (Linux, or macOS without Xcode CLT).
            flags = re.sub(r"-isysroot\s+\S+", "", flags)
    flags = _strip_mismatched_arch(flags)
    # Build-time debug path remapping is irrelevant in a user environment.
    flags = re.sub(r"-fdebug-prefix-map=\S+", "", flags)
    cfg["jit_c_compiler_flags"] = flags


def _strip_mismatched_arch(flags):
    """Strip any ``-arch FOO`` that doesn't match the host architecture.

    conda-forge's ``python-suitesparse-graphblas`` bakes the build host's
    ``-arch`` into ``jit_c_compiler_flags``. On a different host (e.g., an
    arm64 Mac running an x86_64-built package) the JIT compile produces
    objects for the wrong arch and the link fails. Leaving the flag out
    lets the compiler default to the host arch.
    """
    host = platform.machine().lower()
    return re.sub(
        r"-arch\s+(\S+)",
        lambda m: "" if m.group(1).lower() not in _ARCH_ALIASES.get(host, (host,)) else m.group(0),
        flags,
    )


# The settings a compile depends on. The probe's type is named after them; see
# ``_probe_jit``.
_TOOLCHAIN_KEYS = (
    "jit_c_compiler_name",
    "jit_c_compiler_flags",
    "jit_c_linker_flags",
    "jit_c_libraries",
)


def _probe_jit(cfg):
    """Compile a tiny type with the JIT, or load it from the cache, and return whether that worked.

    ``jit_c_control`` must be ``'on'``. Afterwards it is wherever SuiteSparse
    left it: a library built without the JIT clamps every write to ``'run'``
    (``GB_jitifyer_set_control``), a failed compile lowers it to ``'load'``, a
    failed load to ``'run'``, and a failed hash insert to ``'pause'``.

    The callers probe before raising the control to ``'on'`` for good. From
    SuiteSparse 9.4 on, a failed compile raises ``JitError`` once, and without
    the probe the first user operation that needs a new kernel would get it
    (a slice, for example, which has no precompiled kernel).

    The type is created through the C API, so nothing is registered in Python,
    and this can run while the backend initializes. Its name is a hash of the
    toolchain settings, so a compiler or flags this cache has not seen compile
    a probe of their own, rather than load one that another toolchain left in
    a shared cache. A cached probe loads in about 2 ms; a compile takes about
    0.3 s.
    """
    from .. import ffi, lib

    toolchain = "\n".join(cfg.get(key, "") for key in _TOOLCHAIN_KEYS)
    name = "_jit_probe_" + hashlib.sha256(toolchain.encode()).hexdigest()[:16]
    defn = f"typedef struct {{ int _probe ; }} {name} ;"
    type_ptr = ffi.new("GrB_Type*")
    # Size 0 asks SuiteSparse to get the size from a compiled kernel.
    if lib.GxB_Type_new(type_ptr, 0, name.encode(), defn.encode()) != lib.GrB_SUCCESS:
        return False
    lib.GrB_Type_free(type_ptr)
    return cfg["jit_c_control"] == "on"


class _CConfig:
    """The ``gb.ss.config`` keys :func:`_set_jit_c_control_at_init` needs, through the C API.

    That runs while the backend initializes, possibly in the middle of
    importing a core module, and importing ``graphblas.ss`` (for its
    ``config``) would import the core modules.
    """

    _strings = {
        "jit_c_compiler_name": "GxB_JIT_C_COMPILER_NAME",
        "jit_c_compiler_flags": "GxB_JIT_C_COMPILER_FLAGS",
        "jit_c_linker_flags": "GxB_JIT_C_LINKER_FLAGS",
        "jit_c_libraries": "GxB_JIT_C_LIBRARIES",
        "jit_cache_path": "GxB_JIT_CACHE_PATH",
    }

    def __init__(self):
        from .. import lib

        self._controls = {
            "off": lib.GxB_JIT_OFF,
            "pause": lib.GxB_JIT_PAUSE,
            "run": lib.GxB_JIT_RUN,
            "load": lib.GxB_JIT_LOAD,
            "on": lib.GxB_JIT_ON,
        }

    def __getitem__(self, key):
        from .. import ffi, lib

        if key == "jit_c_control":
            val = ffi.new("int32_t*")
            info = lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_C_CONTROL, val)
        else:
            val = ffi.new("char**")
            info = lib.GxB_Global_Option_get_CHAR(getattr(lib, self._strings[key]), val)
        if info != lib.GrB_SUCCESS:  # pragma: no cover (safety)
            raise RuntimeError(f"Could not get SuiteSparse:GraphBLAS option {key!r}")
        if key == "jit_c_control":
            return next(k for k, v in self._controls.items() if v == val[0])
        return ffi.string(val[0]).decode()

    def __setitem__(self, key, value):
        from .. import lib

        if key == "jit_c_control":
            info = lib.GxB_Global_Option_set_INT32(lib.GxB_JIT_C_CONTROL, self._controls[value])
        else:
            option = getattr(lib, self._strings[key])
            info = lib.GxB_Global_Option_set_CHAR(option, value.encode())
        if info != lib.GrB_SUCCESS:  # pragma: no cover (safety)
            raise RuntimeError(f"Could not set SuiteSparse:GraphBLAS option {key!r}")

    def get(self, key, default=None):
        return self[key] if key == "jit_c_control" or key in self._strings else default


def _cache_is_writable(cfg):
    """Whether SuiteSparse can write new kernels to its cache.

    A probe kernel already in the cache loads from a read-only one, and the
    first compile after it would fail (with ``JitError`` from SuiteSparse 9.4
    on), so the probe alone does not answer this.
    """
    path = cfg.get("jit_cache_path")
    return bool(path) and all(
        os.access(pathlib.Path(path, sub), os.W_OK | os.X_OK) for sub in ("c", "lib", "tmp")
    )


def _set_jit_c_control_at_init(control):
    """Set ``jit_c_control`` as the backend initializes, from ``graphblas.config``.

    ``control`` is ``graphblas.config["jit_c_control"]``, already checked, or
    None when it is not set, which means the default ``'run'``. That default
    is raised later for C source (:func:`_enable_jit_for_udt`); a ``'run'``
    that is set is kept (``_user_set_run``).

    Every value but ``'on'`` is set as given. ``'on'`` is set only where it
    works: there is a compiler on disk (after :func:`_repair_jit_compiler`),
    the cache directory is writable, and :func:`_probe_jit` passes. Elsewhere
    the control is ``'run'``, where the conda-forge and PyPI builds of
    SuiteSparse start it, with no error or warning. The rules are in the
    Technical Notes, "The JIT control".
    """
    from . import _IS_SSGB7

    if _IS_SSGB7:
        return  # no JIT before SuiteSparse:GraphBLAS 8
    global _jit_enabled_for_udt, _configured_jit_c_control, _user_set_run
    cfg = _CConfig()
    _configured_jit_c_control = control
    _user_set_run = control == "run"
    if control != "on":
        cfg["jit_c_control"] = "run" if control is None else control
        return
    if not _compiler_exists(cfg):
        _repair_jit_compiler(cfg)
    _jit_enabled_for_udt = False
    if _compiler_exists(cfg) and _cache_is_writable(cfg):
        cfg["jit_c_control"] = "on"
        _jit_enabled_for_udt = cfg["jit_c_control"] == "on" and _probe_jit(cfg)
    if not _jit_enabled_for_udt:
        # Not the 'load' a failed probe leaves: 'load' still loads kernels from
        # the on-disk cache, which other builds of this version may have written.
        cfg["jit_c_control"] = "run"


def _auto_fix_jit_at_import():
    """Repair the JIT compiler setting when ``graphblas.ss`` is imported.

    ``graphblas/ss/__init__.py`` calls this unguarded, so an exception here
    breaks ``import graphblas.ss``; it sticks to the config mapping and
    :func:`_repair_jit_compiler`, which are written not to raise.

    It leaves ``jit_c_control`` alone. An attribute access such as
    ``gb.ss.about["library_version"]`` imports ``graphblas.ss``, and that must
    not change what a later operation computes or which cached kernels
    SuiteSparse loads. The control is set at init
    (:func:`_set_jit_c_control_at_init`) and raised for C source
    (:func:`_enable_jit_for_udt`).
    """
    cfg = _ss_config()
    if "jit_c_control" not in cfg:
        return
    if not jit_compiler_is_usable():
        _repair_jit_compiler(cfg)


# Whether this process can JIT-compile: ``None`` until the compiler is probed,
# at init when that sets ``'on'``, or at the first request to compile C source
# that the control allows. ``fix_jit_config`` resets it to ``None``.
_jit_enabled_for_udt = None

# ``graphblas.config["jit_c_control"]`` as init found it, None when it was not set.
_configured_jit_c_control = None
# Whether a ``'run'`` in effect is one the user set, which ``_enable_jit_for_udt``
# keeps: a ``'run'`` configured before init, or one written to ``gb.ss.config``
# (whose setter keeps this current) while something else is configured. With
# nothing configured, a ``'run'`` written there counts as the default.
_user_set_run = False


def _enable_jit_for_udt():
    """Raise ``jit_c_control`` to ``'on'`` when C source needs compiling.

    Returns whether C source given to SuiteSparse now will be compiled.

    The default control, ``'run'``, neither compiles nor loads kernels. Without
    ``'on'``, a lifted UDT operator calls its Numba function for each element,
    often several times slower, and a type or operator registered from C
    source cannot be created at all. The control is raised here, when C source
    is given to SuiteSparse, which plainly involves compiling; not at import,
    so that reading ``gb.ss.about`` changes nothing.

    Only the default ``'run'`` is raised. A ``'run'`` the user set (see
    ``_user_set_run``) is kept, often to keep processes that run at the same
    time from compiling into one cache (GraphBLAS #449). So are ``'off'``,
    ``'pause'`` and ``'load'``. SuiteSparse sets ``'load'`` itself after a
    compile fails, and raising it would retry that compile at every later UDT
    operator, raising ``JitError`` each time from SuiteSparse 9.4 on. Whether
    this process can compile at all is probed once (``_jit_enabled_for_udt``).
    """
    global _jit_enabled_for_udt
    cfg = _ss_config()
    control = cfg.get("jit_c_control")
    if control in (None, "off", "pause", "load") or (control == "run" and _user_set_run):
        # ``None`` is a library with no JIT at all. The others are honored and
        # not cached, so they decide nothing once the control is back at the
        # default.
        return False
    if _jit_enabled_for_udt is None:
        if not jit_compiler_is_usable():
            _repair_jit_compiler(cfg)
        # A probe kernel already in a read-only cache loads, and the first new
        # kernel would then fail to compile; see ``_cache_is_writable``.
        if _jit_enabled_for_udt := jit_compiler_is_usable() and _cache_is_writable(cfg):
            cfg["jit_c_control"] = "on"
            # A library built without the JIT clamps that write back to ``'run'``.
            # Otherwise probe first; see ``_probe_jit``.
            _jit_enabled_for_udt = cfg["jit_c_control"] == "on" and _probe_jit(cfg)
    elif _jit_enabled_for_udt and control == "run":
        # Whether this process can compile is settled once, but the control is
        # separate state that anything may have lowered since, and C source
        # given now still needs it raised.
        cfg["jit_c_control"] = "on"
    return _jit_enabled_for_udt


@contextlib.contextmanager
def _compiling_c_source(name):
    """Enable the JIT to create ``name`` from C source, and explain a failure.

    SuiteSparse compiles the C source, or loads the kernel from its cache, as
    part of creating the type or op. When it cannot, it reports a bare
    ``NullPointer`` (op) or ``InvalidValue`` (type), or from 9.4 a
    ``JitError`` for a failed compile, none with a message. ``jit_c_control``
    before and after says which case it was, so the same exception type is
    raised again with the reason, chained to the original.
    """
    from ...exceptions import InvalidValue, JitError, NullPointer

    cfg = _ss_config()
    initial = cfg.get("jit_c_control")
    _enable_jit_for_udt()
    before = cfg.get("jit_c_control")
    try:
        yield
    except (NullPointer, InvalidValue, JitError) as exc:
        after = cfg.get("jit_c_control")
        if after in {"off", "pause"} and after == before:
            reason = (
                f"jit_c_control is {after!r}, so SuiteSparse will not compile it. "
                "Set gb.ss.config['jit_c_control'] = 'on' to allow that."
            )
        elif after == "load" and before == "on":
            reason = (
                "SuiteSparse failed to compile it and set jit_c_control to 'load', so "
                "nothing else compiles in this process until jit_c_control is set back "
                "to 'on'. SuiteSparse prints the compiler's output when "
                "gb.ss.config['burble'] is True."
            )
        elif after == before == "load" and initial != "load":
            # ``_enable_jit_for_udt`` only ever writes 'on', so 'load' here means
            # the probe it just ran failed to compile.
            reason = (
                "the C compiler does not work here: SuiteSparse could not compile a "
                "small test type, and set jit_c_control to 'load'. To see the "
                "compiler's output, set gb.ss.config['burble'] = True and "
                "jit_c_control back to 'on', then try again."
            )
        elif after == before == "load":
            reason = (
                "jit_c_control is 'load', so SuiteSparse only loads kernels already in "
                "its cache, and this one is not there. SuiteSparse also sets 'load' "
                "itself after a compile fails. Set gb.ss.config['jit_c_control'] = "
                "'on' to compile it."
            )
        elif after == before == "run" and _user_set_run:
            reason = (
                "jit_c_control was set to 'run' (by graphblas.config['jit_c_control'], "
                "the GRAPHBLAS_JIT_C_CONTROL environment variable, or gb.ss.config), "
                "which neither compiles nor loads, and python-graphblas keeps a 'run' "
                "that was set. To compile it, set gb.ss.config['jit_c_control'] = 'on', "
                "or None for the default 'run', which python-graphblas raises to 'on' "
                "for C source."
            )
        elif after == before == "run":
            reason = (
                "jit_c_control is still 'run', which neither compiles nor loads: "
                "python-graphblas did not find the configured C compiler, or "
                "SuiteSparse was built without the JIT. gb.ss.fix_jit_config() looks "
                "for a compiler and turns the JIT on."
            )
        else:
            raise
        raise type(exc)(f"Cannot create {name} from C source: {reason}") from exc


# The ``(op_name, dtype_name)`` pairs ``_maybe_warn_no_jit`` has warned about.
_warned_no_jit_for = set()


def _maybe_warn_no_jit(*, op_name="", dtype_name=""):
    """Warn, once per operator and type, that a lifted UDT operator has no C definition.

    The caller is ``udt_codegen._maybe_warn_jit_skipped``, after C code
    generation gave nothing. Code generation reads only the dtype, not the
    compiler or ``jit_c_control``, so the warning names the dtype as the cause
    even on a host that could not compile anyway. Do not derive the cause from
    the live config, which is not what code generation read.
    """
    key = (op_name, dtype_name)
    if key in _warned_no_jit_for:
        return
    _warned_no_jit_for.add(key)
    import warnings as _warnings

    from ...exceptions import NoJITWarning

    loc = f" (op={op_name!r}, dtype={dtype_name!r})" if op_name else ""
    _warnings.warn(
        f"UDT operator running without JIT compilation: this UDT is not "
        f"expressible as a C struct{loc}. A field name may be a C keyword or a "
        f"name the C headers define, a field may have a type with no C "
        f"equivalent (such as text or a date), or the record may have an array "
        f"field or a packed layout. The operator still works, by calling its "
        f"Numba function for each element, which is often several times slower. "
        f"This warning appears once per operator and type; silence it with "
        f"warnings.filterwarnings('ignore', "
        f"category=graphblas.exceptions.NoJITWarning).",
        NoJITWarning,
        stacklevel=3,
    )

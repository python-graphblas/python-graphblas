"""Helpers for repairing the SuiteSparse:GraphBLAS JIT compiler configuration.

The C library is built by conda-build with the build host's compiler paths
baked into ``GxB_JIT_C_COMPILER_NAME`` and ``GxB_JIT_C_COMPILER_FLAGS``.
Those paths (e.g., ``/Users/runner/...``) don't exist in the user's
environment. When SuiteSparse JIT tries to compile a UDT kernel the compile
step fails silently: the ``.c`` source is written but no ``.dylib`` or
``.so`` is produced, and SS falls back to the Numba function-pointer path.
The caller sees a working op, but the JIT speedup is lost, and there is
no error or warning to make the regression visible.

``fix_jit_config()`` replaces the baked-in compiler with one from
``$CONDA_PREFIX/bin/`` and strips build-time-only flags (``-isysroot``,
``-fdebug-prefix-map``). For non-conda installs (pure pip), ``sysconfig``
may expose a usable compiler. ``fix_jit_config(use_sysconfig=True)`` will
try that.

``jit_compiler_is_usable()`` is a non-invasive check that returns True iff
the configured compiler exists on disk. Useful for emitting a one-time
warning at import or at first JIT use.

``jit_c_control`` itself is set when the ``suitesparse`` backend initializes,
from ``graphblas.config["jit_c_control"]`` (``'run'`` unless configured
otherwise; see :func:`_set_jit_c_control_at_init`). UDT work raises that
default ``'run'`` to ``'on'`` when it needs C compiled, but keeps a ``'run'``
the user set (:func:`_enable_jit_for_udt`).
The trade-offs of the JIT, and what to do when it misbehaves, are in the user
guide's initialization page (docs/user_guide/init.rst).
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

# Mapping from the values that show up in baked-in ``-arch`` flags to what
# ``platform.machine()`` returns on the running host.
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
    """True iff the configured JIT compiler path exists on disk.

    Cheap (no compile attempt); use before suggesting ``fix_jit_config()``.
    """
    return _compiler_exists(_ss_config())


def _compiler_exists(cfg):
    cc = cfg.get("jit_c_compiler_name", "")
    return bool(cc) and pathlib.Path(cc).exists()


def fix_jit_config(*, use_sysconfig=True, probe=True):
    """Repair the SuiteSparse:GraphBLAS JIT compiler configuration.

    Replaces the baked-in compiler path (which often points at a conda-build
    host that doesn't exist in user environments) with one from
    ``$CONDA_PREFIX/bin/``, strips build-time-only flags (``-isysroot``,
    ``-fdebug-prefix-map``), and sets ``jit_c_control`` to ``'on'``.

    Parameters
    ----------
    use_sysconfig : bool, default True
        When ``$CONDA_PREFIX`` isn't set (pure pip install), try the
        compiler from ``sysconfig.get_config_var("CC")``. Set to ``False``
        to restrict the repair to a conda environment only.
    probe : bool, default True
        After fixing the config, JIT-compile a trivial type to verify the
        compiler actually works. A failed compile makes SuiteSparse lower
        ``jit_c_control`` to ``'load'``, and from SuiteSparse 9.4 on it also
        raises ``JitError`` that one time; the probe takes that failure, so
        later operations use the generic kernels without raising.

    Returns
    -------
    True
        Fix applied and (if ``probe``) verified working.
    False
        Fix attempted but the probe failed. ``jit_c_control`` is now
        whatever SuiteSparse left it at.
    None
        No environment available to fix from. There's no ``$CONDA_PREFIX``,
        and either ``use_sysconfig=False`` or no sysconfig compiler is set.
    """
    cfg = _ss_config()
    rv = _repair_jit_compiler(cfg, use_sysconfig=use_sysconfig)
    if rv is None:
        return None
    # Enabling compilation is what the caller asked for, whatever
    # ``graphblas.config["jit_c_control"]`` set at init.
    cfg["jit_c_control"] = "on"
    # An explicit user-driven fix is a clean opportunity to re-arm
    # ``NoJITWarning``: if the repair worked, the next UDT auto-lift that
    # *still* falls back to cfunc (different cause: UDT layout, etc.)
    # deserves a fresh notification rather than silent suppression.
    _warned_no_jit_for.clear()
    # Re-arm the cached probe answer for the same reason: a pre-repair False
    # is stale now, so let the next UDT op re-derive it against the fixed
    # toolchain instead of trusting the old verdict.
    global _jit_enabled_for_udt
    _jit_enabled_for_udt = None
    if not probe:
        return True
    return _probe_jit(cfg)


def _repair_jit_compiler(cfg, *, use_sysconfig=True):
    """Point the JIT compiler settings at something that exists on this host.

    Only rewrites the compiler name and flags; ``jit_c_control`` is left
    alone, so on its own this changes nothing about what SuiteSparse will
    do. Callers that want compilation enabled set the control themselves.
    """
    if conda_prefix := os.environ.get("CONDA_PREFIX", ""):
        return _fix_from_conda(cfg, conda_prefix)
    if use_sysconfig:
        return _fix_from_sysconfig(cfg)
    return None


def _fix_from_conda(cfg, conda_prefix):
    """Conda-aware fix; swap the compiler path with one from the environment's binaries:
    ``$CONDA_PREFIX/bin/``, or ``$CONDA_PREFIX/Library/bin/`` with ``.exe`` names (Windows).
    """
    jit_cc = cfg["jit_c_compiler_name"]
    if not pathlib.Path(jit_cc).exists():
        # (a Windows path's basename, also when read on another system)
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
    """Compile a trivial type with the JIT, or load it from the cache, to check the config works.

    ``jit_c_control`` must be ``'on'``, and is left wherever SuiteSparse puts
    it, which is not predictable from here. A library built without the JIT
    clamps every write down to ``'run'`` (``GB_jitifyer_set_control``), so
    ``'on'`` never takes on such a build no matter how good the compiler is;
    separately, a compile failure drops it to ``'load'``, a load failure to
    ``'run'``, and a failed hash insert to ``'pause'``. The contract here is
    only "did this work?", so the caller gets a bool.

    It goes through the C API, so nothing is registered on the Python side,
    and it can run while the backend initializes. Once a kernel is in the
    cache, a later probe loads it (about 2 ms) instead of compiling (about
    0.3 s). The type is named after the toolchain settings, so a compiler or
    flags this cache has not seen compile a probe of their own, rather than
    load one that another toolchain left in a shared cache.
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
    None when it is not set: then the control is the default ``'run'``, which
    UDT work raises (:func:`_enable_jit_for_udt`). A ``'run'`` that is set is
    the user's choice, and UDT work keeps it (``_user_set_run``).
    Every value but ``'on'`` is set as given. ``'on'`` is set only where it
    works, which :func:`_probe_jit` checks by compiling a trivial type (or
    loading it from the cache, after the first time). Where it does not, the
    control is set to ``'run'`` (where the conda-forge and PyPI builds of
    SuiteSparse start it), with no error or warning: a library built without
    the JIT, which clamps every write to ``'run'``; no compiler on disk, even
    after :func:`_repair_jit_compiler`; a cache directory it cannot write to;
    or a compiler that fails the probe.
    The probe is load-bearing: from SuiteSparse 9.4 on a failed compile raises
    ``JitError`` (once; SuiteSparse then drops the control to ``'load'``), and
    without the probe that would be the first user operation that needs a new
    kernel, such as a slice, which SuiteSparse precompiles no kernel for.
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
    """Repair the JIT compiler path at ``gb.ss`` import; designed not to raise.

    Called unguarded from ``graphblas/ss/__init__.py``, so any exception here
    breaks ``import graphblas.ss``. The body sticks to dict ops and the
    string rewriting in :func:`_repair_jit_compiler`.

    Deliberately leaves ``jit_c_control`` alone. An attribute access such as
    ``gb.ss.about["library_version"]`` imports this submodule, and that must
    not change what any later operation computes or which kernels SuiteSparse
    is willing to load from its on-disk cache. The control is set when the
    backend initializes (:func:`_set_jit_c_control_at_init`), which also
    repairs the compiler when it sets ``'on'``. Otherwise compilation is
    enabled later, by :func:`_enable_jit_for_udt`, when a UDT actually needs a
    kernel built, or a type or op is registered from C source.
    """
    cfg = _ss_config()
    if "jit_c_control" not in cfg:
        return
    if not jit_compiler_is_usable():
        _repair_jit_compiler(cfg)


# Tri-state: ``None`` until the compiler is probed (at init when that sets
# ``'on'``, or at the first request to compile C source while the control
# allows compiling), then the answer to "can this process JIT-compile?" until
# ``fix_jit_config`` re-arms it.
_jit_enabled_for_udt = None

# ``graphblas.config["jit_c_control"]`` as init found it, None when it was not set.
_configured_jit_c_control = None
# Whether a ``'run'`` in effect is one the user set, which ``_enable_jit_for_udt``
# keeps: a ``'run'`` configured before init, or one written to ``gb.ss.config``
# (whose setter keeps this current) while something else is configured. With
# nothing configured, a ``'run'`` written there is the default, as before it.
_user_set_run = False


def _enable_jit_for_udt():
    """Enable JIT compilation the first time C source needs compiling.

    ``jit_c_control`` defaults to ``'run'`` (``graphblas.config``, set at
    init), which runs kernels already loaded but neither compiles nor loads
    any. UDT auto-lift wants ``'on'``; without it every UDT op falls back to
    the Numba function-pointer path (typically 2-3x slower for elementwise
    ops). An op or type registered from C source has no fallback: under
    ``'run'`` its creation fails outright.

    This is where that bump belongs, rather than at import: registering a
    type or an op from C source, or arming an op with C source, is an act
    that plainly involves compiling C, so enabling the compiler is not a
    surprise. Reading ``gb.ss.about`` is not, so it leaves the setting alone.

    Only the default ``'run'`` is raised. A ``'run'`` the user set (through
    ``graphblas.config["jit_c_control"]``, or ``gb.ss.config``; see
    ``_user_set_run``) is honored, often to keep processes that run at the
    same time from compiling into one cache (GraphBLAS #449). So are
    ``'off'``, ``'pause'``, and ``'load'``, and ``'load'`` is also where
    SuiteSparse leaves the control after a compile fails, so raising it would
    retry that compile at every later UDT op (raising ``JitError`` each time on
    SuiteSparse 9.4 and later). Returns True iff a kernel armed now will be
    compiled. Whether this process can compile at all is probed once and
    cached.
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
            # A library built without the JIT clamps that write back down to
            # ``'run'``. Otherwise the probe is load-bearing: a failed compile
            # drops the control to ``'load'`` and, from SuiteSparse 9.4 on,
            # also raises ``JitError``, so without the probe that one error
            # would reach the first user operation that needs a kernel.
            _jit_enabled_for_udt = cfg["jit_c_control"] == "on" and _probe_jit(cfg)
    elif _jit_enabled_for_udt and control == "run":
        # The cache answers "can this process compile?", which is settled once.
        # The control is separate state that anything may have lowered since,
        # and a kernel armed now still needs it raised.
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


# Keyed by ``(op_name, dtype_name)`` so each distinct pair warns once.
# A user who registers several UDTs gets one warning per (op, dtype) pair
# rather than a single global swallow.
_warned_no_jit_for = set()


def _maybe_warn_no_jit(*, op_name="", dtype_name=""):
    """Emit a ``NoJITWarning`` (once per ``(op_name, dtype_name)``) when UDT auto-lift falls back.

    The only caller is ``udt_codegen._maybe_warn_jit_skipped``, which fires
    after C codegen returned nothing. Codegen reads the dtype and neither the
    compiler path nor ``jit_c_control``, so the dtype is the cause here even
    on a host that could not have compiled a kernel anyway. Do not re-derive
    the cause from the live config; that is not the state codegen read.

    The introspection properties (``DataType.jit_c_definition`` and
    ``TypedUserBinaryOp.jit_c_source``) show what was generated, or ``None``
    when codegen was skipped.
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
        f"expressible as a C struct{loc} (a field name is a C reserved word or "
        f"stdlib macro, a field type isn't in the numpy-to-C map, a field is "
        f"array-typed, or the record has a packed layout). The op still works "
        f"through the Numba function-pointer fallback; only the JIT speedup is "
        f"lost (typically 2-3x slower for elementwise ops, since SuiteSparse "
        f"can't inline the kernel into its eWise and reduce templates). This "
        f"warning fires once per (op, dtype) per process; silence with "
        f"``warnings.filterwarnings('ignore', "
        f"category=gb.exceptions.NoJITWarning)`` or by message match.",
        NoJITWarning,
        stacklevel=3,
    )

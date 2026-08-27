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
"""

import os
import pathlib
import platform
import re
import subprocess

from .. import lib  # noqa: F401  (sets up cffi)

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
    cfg = _ss_config()
    cc = cfg.get("jit_c_compiler_name", "")
    return bool(cc) and pathlib.Path(cc).exists()


def fix_jit_config(*, use_sysconfig=True, probe=True):
    """Repair the SuiteSparse:GraphBLAS JIT compiler configuration.

    Replaces the baked-in compiler path (which often points at a conda-build
    host that doesn't exist in user environments) with one from
    ``$CONDA_PREFIX/bin/``, and strips build-time-only flags (``-isysroot``,
    ``-fdebug-prefix-map``).

    Parameters
    ----------
    use_sysconfig : bool, default True
        When ``$CONDA_PREFIX`` isn't set (pure pip install), try the
        compiler from ``sysconfig.get_config_var("CC")``. Set to ``False``
        to restrict the repair to a conda environment only.
    probe : bool, default True
        After fixing the config, try to JIT-register a trivial UDT to verify
        the compiler actually works. SuiteSparse auto-flips ``jit_c_control``
        from ``'on'`` to ``'load'`` on a failed compile; the probe absorbs
        that first failure so user-visible ops afterwards see a stable
        ``'load'`` (cache-only) state and punt to generic cleanly.

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
    # Enabling compilation is what the caller asked for. The import-time
    # repair deliberately stops short of this; see ``_auto_fix_jit_at_import``.
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
    """Conda-aware fix; swap the compiler path with one from $CONDA_PREFIX/bin/."""
    jit_cc = cfg["jit_c_compiler_name"]
    if not pathlib.Path(jit_cc).exists():
        cc_basename = pathlib.Path(jit_cc).name
        bin_dir = pathlib.Path(conda_prefix) / "bin"
        for candidate in [cc_basename, "cc", "clang", "gcc"]:
            local_cc = bin_dir / candidate
            if local_cc.exists():
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


def _probe_jit(cfg):
    """Probe a trivial JIT compile to verify the config works.

    ``jit_c_control`` is left wherever SuiteSparse puts it, and that is not
    predictable from here. A library built without the JIT clamps every write
    down to ``'run'`` (``GB_jitifyer_set_control``), so ``'on'`` never takes on
    such a build no matter how good the compiler is; separately, a compile
    failure drops it to ``'load'``, a load failure to ``'run'``, and a failed
    hash insert to ``'pause'``. The contract here is only "did this work?", so
    the caller gets a bool.
    """
    from ... import dtypes as _dtypes

    # The probe dtype is installed at ``dtypes.ss._jit_probe`` on first
    # success; reuse it on a second probe so a repeat call doesn't appear to
    # fail. Without the hasattr short-circuit, ``register_new`` would raise
    # ``ValueError("name unavailable")`` on the second call.
    probe_name = "_jit_probe"
    if hasattr(_dtypes.ss, probe_name):
        return True
    global _probing_jit
    _probing_jit = True
    try:
        _dtypes.ss.register_new(probe_name, "typedef struct { int _probe ; } _jit_probe ;")
    except Exception:
        # ``register_new`` can raise ``JitError`` (bad path / flags / arch /
        # SDK), ``RuntimeError`` (SS<8 has no JIT), or one of its input
        # validation ``ValueError``s. The probe's contract is "did this
        # work?", so absorb every failure mode here.
        return False
    finally:
        _probing_jit = False
    return True


def _auto_fix_jit_at_import():
    """Repair the JIT compiler path at ``gb.ss`` import; designed not to raise.

    Called unguarded from ``graphblas/ss/__init__.py``, so any exception here
    breaks ``import graphblas.ss``. The body sticks to dict ops and the
    string rewriting in :func:`_repair_jit_compiler`.

    Deliberately leaves ``jit_c_control`` at whatever SuiteSparse set. An
    attribute access such as ``gb.ss.about["library_version"]`` imports this
    submodule, and that must not change what any later operation computes or
    which kernels SuiteSparse is willing to load from its on-disk cache.
    Compilation is enabled later, by :func:`_enable_jit_for_udt`, when a UDT
    actually needs a kernel built.
    """
    cfg = _ss_config()
    if "jit_c_control" not in cfg:
        return
    if not jit_compiler_is_usable():
        _repair_jit_compiler(cfg)


# Tri-state: ``None`` until the first UDT asks for a JIT kernel, then the
# answer to "can this process JIT-compile?" until something re-arms it.
_jit_enabled_for_udt = None

# True only while ``_probe_jit`` is registering its own UDT. That registration
# goes through ``dtypes.ss.register_new``, which asks to enable the JIT, which
# would probe again; the second probe would register the same name a second
# time and the first one would then die on cffi's "multiple declarations".
_probing_jit = False


def _enable_jit_for_udt():
    """Enable JIT compilation the first time a UDT needs a kernel built.

    SuiteSparse defaults ``jit_c_control`` to ``'run'``, which runs kernels
    already loaded but neither compiles nor loads any. UDT auto-lift wants
    ``'on'``; without it every UDT op falls back to the Numba
    function-pointer path (typically 2-3x slower for elementwise ops).

    This is where that bump belongs, rather than at import: registering a
    UDT or arming an op with C source is an act that plainly involves
    compiling C, so enabling the compiler is not a surprise. Reading
    ``gb.ss.about`` is not, so it leaves the setting alone.

    An explicit ``'off'`` or ``'pause'`` is honored: only SuiteSparse's own
    non-compiling defaults are raised. Returns True iff compilation is
    available, and answers from cache after the first call.
    """
    global _jit_enabled_for_udt
    if _probing_jit:
        # Re-entered from the probe's own registration. The probe is the thing
        # deciding this answer, so say yes and let it finish rather than
        # starting a second one inside it.
        return True
    cfg = _ss_config()
    if _jit_enabled_for_udt is not None:
        # The cache answers "can this process compile?", which is settled once.
        # It does not pin the control: anything may have moved it since, and a
        # kernel armed now still needs it raised. Restoring a saved ``'run'``
        # after an earlier op enabled the JIT used to disable compilation for
        # the rest of the process, silently, because this returned here first.
        if _jit_enabled_for_udt and cfg.get("jit_c_control") in ("run", "load"):
            cfg["jit_c_control"] = "on"
        return _jit_enabled_for_udt
    if "jit_c_control" not in cfg:
        _jit_enabled_for_udt = False
        return False
    if not jit_compiler_is_usable():
        _repair_jit_compiler(cfg)
        if not jit_compiler_is_usable():
            _jit_enabled_for_udt = False
            return False
    if cfg["jit_c_control"] in ("run", "load"):
        cfg["jit_c_control"] = "on"
    _jit_enabled_for_udt = cfg["jit_c_control"] == "on"
    if _jit_enabled_for_udt:
        # The probe is load-bearing. Without it SuiteSparse surfaces
        # ``JitError`` on the first user-triggered compile; a failed compile
        # is only converted to a silent non-compiling fallback afterwards.
        _jit_enabled_for_udt = _probe_jit(cfg)
    return _jit_enabled_for_udt


# Keyed by ``(op_name, dtype_name)`` so each distinct pair warns once.
# A user who registers several UDTs gets one warning per (op, dtype) pair
# rather than a single global swallow.
_warned_no_jit_for = set()


def _maybe_warn_no_jit(*, op_name="", dtype_name=""):
    """Emit a ``NoJITWarning`` (once per ``(op_name, dtype_name)``) when UDT auto-lift falls back.

    The only caller is ``udt_utils._maybe_warn_jit_skipped``, which fires
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

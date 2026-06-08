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
import re
import subprocess

from .. import lib  # noqa: F401  (sets up cffi)


def _ss_config():
    # Imported lazily to avoid pulling all of ``gb.ss`` into module-load time.
    from ... import ss

    return ss.config


def jit_compiler_is_usable():
    """Return True iff the configured JIT compiler exists on disk.

    A cheap probe: it checks the file but doesn't try to compile anything.
    Use this to decide whether to warn the user or suggest ``fix_jit_config()``.
    """
    cfg = _ss_config()
    cc = cfg.get("jit_c_compiler_name", "")
    return bool(cc) and pathlib.Path(cc).exists()


def fix_jit_config(*, use_sysconfig=False, probe=True):
    """Repair the SuiteSparse:GraphBLAS JIT compiler configuration.

    Replaces the baked-in compiler path (which often points at a conda-build
    host that doesn't exist in user environments) with one from
    ``$CONDA_PREFIX/bin/``, and strips build-time-only flags (``-isysroot``,
    ``-fdebug-prefix-map``).

    Parameters
    ----------
    use_sysconfig : bool, default False
        When ``$CONDA_PREFIX`` isn't set (pure pip install), try the
        compiler from ``sysconfig.get_config_var("CC")`` instead.
    probe : bool, default True
        After fixing the config, try to JIT-register a trivial UDT to verify
        the compiler actually works. If the probe fails, set
        ``jit_c_control = 'off'`` and return False so SS doesn't keep
        attempting failing compiles.

    Returns
    -------
    True
        Fix applied and (if ``probe``) verified working.
    False
        Fix attempted but the probe failed, so JIT is now disabled.
    None
        No environment available to fix from. There's no ``$CONDA_PREFIX``,
        and either ``use_sysconfig=False`` or no sysconfig compiler is set.
    """
    cfg = _ss_config()
    if conda_prefix := os.environ.get("CONDA_PREFIX", ""):
        rv = _fix_from_conda(cfg, conda_prefix)
    elif use_sysconfig:
        rv = _fix_from_sysconfig(cfg)
    else:
        return None
    if rv is None:
        return None
    # An explicit user-driven fix is a clean opportunity to re-arm the
    # one-time ``NoJITWarning``: if the repair worked, the next UDT auto-lift
    # that *still* falls back to cfunc (different cause: UDT layout, etc.)
    # deserves a fresh notification rather than silent suppression.
    global _warned_no_jit
    _warned_no_jit = False
    if not probe:
        return True
    return _probe_jit(cfg)


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
    cfg["jit_c_control"] = "on"
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
    cfg["jit_c_control"] = "on"
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
    # Build-time debug path remapping is irrelevant in a user environment.
    flags = re.sub(r"-fdebug-prefix-map=\S+", "", flags)
    cfg["jit_c_compiler_flags"] = flags


def _probe_jit(cfg):
    """Probe a trivial JIT compile to verify the config works."""
    from ... import dtypes as _dtypes

    # ``register_new`` rejects a duplicate name with ``ValueError``. The probe
    # dtype is installed at ``dtypes.ss._jit_probe`` on first success; reuse it
    # on a second probe so a repeat call doesn't flip ``jit_c_control`` to
    # ``off``.
    probe_name = "_jit_probe"
    if hasattr(_dtypes.ss, probe_name):
        return True
    try:
        _dtypes.ss.register_new(probe_name, "typedef struct { int _probe ; } _jit_probe ;")
    except Exception:
        cfg["jit_c_control"] = "off"
        return False
    return True


def _auto_fix_jit_at_import():
    """Auto-fix the JIT config once at ``gb.ss`` import time.

    When the baked-in compiler path is missing, swap it for one from
    ``$CONDA_PREFIX/bin/`` or ``sysconfig``, and bump ``jit_c_control`` from
    the SS default ``'run'`` (load only) to ``'on'`` (compile and load).
    When the compiler is already usable, only the mode bump applies. Never
    raises and does not probe; if the resulting state is still broken,
    ``_maybe_warn_no_jit`` surfaces the issue at first UDT auto-lift.
    """
    cfg = _ss_config()
    if "jit_c_control" not in cfg:
        return
    if jit_compiler_is_usable():
        if cfg["jit_c_control"] in ("run", "load"):
            cfg["jit_c_control"] = "on"
        return
    try:
        fix_jit_config(use_sysconfig=True, probe=False)
    except Exception:  # pragma: no cover (defensive)
        return
    if jit_compiler_is_usable() and cfg["jit_c_control"] in ("run", "load"):
        cfg["jit_c_control"] = "on"


_warned_no_jit = False


def _maybe_warn_no_jit(*, op_name="", dtype_name=""):
    """Emit a one-time ``NoJITWarning`` when UDT auto-lift falls back to the cfunc path.

    The most likely cause (bogus compiler path, ``jit_c_control`` off, or
    UDT not C-expressible) is named in the message along with the remediation.
    """
    global _warned_no_jit
    if _warned_no_jit:
        return
    _warned_no_jit = True
    import warnings as _warnings

    cfg = _ss_config()
    if not jit_compiler_is_usable():
        cause = (
            "the JIT compiler path is not usable "
            f"({cfg.get('jit_c_compiler_name', '<unset>')!r}); "
            "call ``gb.ss.fix_jit_config()`` to repair it"
        )
    elif cfg.get("jit_c_control") in ("off", "pause"):
        cause = (
            f"jit_c_control is {cfg.get('jit_c_control')!r}; "
            "set ``gb.ss.config['jit_c_control'] = 'on'`` to enable compilation"
        )
    else:
        # Compiler is usable and mode is OK; the UDT itself isn't C-expressible.
        # The introspection properties (``DataType.jit_c_definition`` and
        # ``TypedUserBinaryOp.jit_c_source``) show what was generated, or
        # ``None`` when codegen was skipped.
        loc = f" (op={op_name!r}, dtype={dtype_name!r})" if op_name else ""
        cause = (
            "this UDT is not expressible as a C struct"
            f"{loc} (a field name is a C reserved word or stdlib macro, a "
            "field type isn't in the numpy-to-C map, a field is array-typed, "
            "or the record has a packed layout). The op still works via the "
            "Numba cfunc path; only the JIT speedup is lost"
        )
    from ...exceptions import NoJITWarning

    _warnings.warn(
        f"UDT operator running without JIT compilation: {cause}. "
        f"Operations will use the Numba function-pointer fallback "
        f"(typically 2-3x slower for elementwise ops, since SuiteSparse "
        f"can't inline the kernel into its eWise and reduce templates). "
        f"This warning fires once per process; silence with "
        f"``warnings.filterwarnings('ignore', category=gb.exceptions.NoJITWarning)`` "
        f"or by message match.",
        NoJITWarning,
        stacklevel=3,
    )

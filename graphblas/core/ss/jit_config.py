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
        whatever SuiteSparse left it at (typically ``'load'``).
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
    # An explicit user-driven fix is a clean opportunity to re-arm
    # ``NoJITWarning``: if the repair worked, the next UDT auto-lift that
    # *still* falls back to cfunc (different cause: UDT layout, etc.)
    # deserves a fresh notification rather than silent suppression.
    _warned_no_jit_for.clear()
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

    On failure, SuiteSparse will have flipped ``jit_c_control`` from
    ``'on'`` to ``'load'`` (its built-in response to a failed compile);
    we leave that state alone. Both ``'load'`` and ``'off'`` cause
    downstream ops to punt to the generic kernel, but ``'load'`` preserves
    any pre-compiled kernels in the cache.
    """
    from ... import dtypes as _dtypes

    # The probe dtype is installed at ``dtypes.ss._jit_probe`` on first
    # success; reuse it on a second probe so a repeat call doesn't appear to
    # fail. Without the hasattr short-circuit, ``register_new`` would raise
    # ``ValueError("name unavailable")`` on the second call.
    probe_name = "_jit_probe"
    if hasattr(_dtypes.ss, probe_name):
        return True
    try:
        _dtypes.ss.register_new(probe_name, "typedef struct { int _probe ; } _jit_probe ;")
    except Exception:
        # ``register_new`` can raise ``JitError`` (bad path / flags / arch /
        # SDK), ``RuntimeError`` (SS<8 has no JIT), or one of its input
        # validation ``ValueError``s. The probe's contract is "did this
        # work?", so absorb every failure mode here.
        return False
    return True


def _auto_fix_jit_at_import():
    """Run :func:`fix_jit_config` at ``gb.ss`` import; designed not to raise.

    Called unguarded from ``graphblas/ss/__init__.py``, so any exception
    here breaks ``import graphblas.ss``. The body sticks to dict ops and
    delegates the failure-prone work to ``_probe_jit``, which catches
    everything internally.

    The probe is the load-bearing piece: without it, SS would surface
    ``JitError`` on the first user-triggered JIT compile (a failed
    compile is only converted to a silent ``'load'`` fallback on
    subsequent calls).
    """
    cfg = _ss_config()
    if "jit_c_control" not in cfg:
        return
    if jit_compiler_is_usable():
        if cfg["jit_c_control"] in ("run", "load"):
            cfg["jit_c_control"] = "on"
    else:
        fix_jit_config(use_sysconfig=True, probe=False)
        if jit_compiler_is_usable() and cfg["jit_c_control"] in ("run", "load"):
            cfg["jit_c_control"] = "on"
    if cfg.get("jit_c_control") == "on":
        _probe_jit(cfg)


# Keyed by ``(op_name, dtype_name)`` so each distinct pair warns once.
# A user who registers several UDTs gets one warning per (op, dtype) pair
# rather than a single global swallow.
_warned_no_jit_for = set()


def _maybe_warn_no_jit(*, op_name="", dtype_name=""):
    """Emit a ``NoJITWarning`` (once per ``(op_name, dtype_name)``) when UDT auto-lift falls back.

    The most likely cause (bogus compiler path, ``jit_c_control`` off, or
    UDT not C-expressible) is named in the message along with the remediation.
    """
    key = (op_name, dtype_name)
    if key in _warned_no_jit_for:
        return
    _warned_no_jit_for.add(key)
    import warnings as _warnings

    cfg = _ss_config()
    if not jit_compiler_is_usable():
        cause = (
            "the JIT compiler path is not usable "
            f"({cfg.get('jit_c_compiler_name', '<unset>')!r}); "
            "call ``gb.ss.fix_jit_config()`` to repair it"
        )
    elif cfg.get("jit_c_control") != "on":
        cause = (
            f"jit_c_control is {cfg.get('jit_c_control')!r} (must be 'on' to compile); "
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
        f"This warning fires once per (op, dtype) per process; silence with "
        f"``warnings.filterwarnings('ignore', category=gb.exceptions.NoJITWarning)`` "
        f"or by message match.",
        NoJITWarning,
        stacklevel=3,
    )

#!/usr/bin/env python
"""Print JIT diagnostic information for debugging compiler configuration.

Run with: python scripts/jit_diagnostics.py
Or during CI: python -m scripts.jit_diagnostics  (from repo root)

This prints the GraphBLAS JIT configuration, available compilers, sysconfig
values, and attempts a JIT compilation to verify everything works.
"""

import os
import pathlib
import shutil
import sys
import sysconfig


def main():
    print("=" * 60)
    print("JIT Diagnostics")
    print("=" * 60)

    print("\n--- Platform ---")
    print(f"sys.platform: {sys.platform}")
    print(f"Python: {sys.version}")

    print("\n--- Environment ---")
    conda = os.environ.get("CONDA_PREFIX", "")
    print(f"CONDA_PREFIX: {conda or 'NOT SET'}")
    print(f"GITHUB_ACTIONS: {os.environ.get('GITHUB_ACTIONS', 'NOT SET')}")

    print("\n--- sysconfig ---")
    for key in ["CC", "CXX", "CFLAGS", "LDFLAGS", "LIBS"]:
        val = sysconfig.get_config_var(key)
        if val and len(str(val)) > 100:
            val = str(val)[:100] + "..."
        print(f"  {key}: {val}")
    print(f"  include: {sysconfig.get_path('include')}")

    print("\n--- Compiler search ---")
    candidates = ["cc", "gcc", "clang"]
    if sys.platform == "linux":
        candidates.append("x86_64-conda-linux-gnu-cc")
    elif sys.platform == "darwin":
        candidates.extend(
            [
                "x86_64-apple-darwin13.4.0-clang",
                "arm64-apple-darwin20.0.0-clang",
            ]
        )
    for name in candidates:
        which = shutil.which(name)
        conda_path = pathlib.Path(conda) / "bin" / name if conda else None
        conda_exists = conda_path.exists() if conda_path else False
        print(f"  {name}: which={which}, conda={'yes' if conda_exists else 'no'}")

    # Import graphblas
    try:
        import graphblas as gb

        gb.init("suitesparse")
    except Exception as e:
        print(f"\nERROR: Could not initialize graphblas: {e}")
        return 1

    from graphblas.core.ss import _IS_SSGB7

    if _IS_SSGB7:
        print("\nSuiteSparse:GraphBLAS 7.x — JIT not available")
        return 0

    print("\n--- GraphBLAS JIT defaults (from compiled C library) ---")
    for key in [
        "jit_c_control",
        "jit_c_compiler_name",
        "jit_c_compiler_flags",
        "jit_c_linker_flags",
        "jit_c_libraries",
        "jit_c_cmake_libs",
        "jit_cache_path",
    ]:
        val = gb.ss.config[key]
        if isinstance(val, str) and len(val) > 120:
            val = val[:120] + "..."
        print(f"  {key}: {val}")

    # Check if default compiler exists
    jit_cc = gb.ss.config["jit_c_compiler_name"]
    print("\n--- Compiler path analysis ---")
    print(f"  Default compiler: {jit_cc}")
    print(f"  Exists? {pathlib.Path(jit_cc).exists()}")
    cc_basename = pathlib.Path(jit_cc).name
    if conda:
        local_cc = pathlib.Path(conda) / "bin" / cc_basename
        print(f"  Conda equivalent: {local_cc}")
        print(f"  Exists? {local_cc.exists()}")
        fallback_cc = pathlib.Path(conda) / "bin" / "cc"
        print(f"  Fallback (cc): {fallback_cc}")
        print(f"  Exists? {fallback_cc.exists()}")

    # Check for problematic flags
    flags = gb.ss.config["jit_c_compiler_flags"]
    import re

    if isysroot := re.search(r"-isysroot\s+(\S+)", flags):
        path = isysroot.group(1)
        print(f"\n  -isysroot: {path}")
        print(f"  Exists? {pathlib.Path(path).exists()}")
    fdebug = re.findall(r"-fdebug-prefix-map=(\S+)", flags)
    for d in fdebug:
        print(f"  -fdebug-prefix-map: {d}")

    # Try the fix
    print("\n--- Attempting _fix_jit_config ---")
    from graphblas.tests.test_ssjit import _fix_jit_config

    result = _fix_jit_config()
    # True=JIT working, False=probe failed, None=no conda env
    desc = "working" if result is True else "probe failed" if result is False else "no conda"
    print(f"  Result: {result} ({desc})")
    if result is True:
        print(f"  Compiler: {gb.ss.config['jit_c_compiler_name']}")
        flags_after = gb.ss.config["jit_c_compiler_flags"]
        if len(flags_after) > 120:
            flags_after = flags_after[:120] + "..."
        print(f"  Flags: {flags_after}")

    # Try JIT compilation
    print("\n--- JIT compilation test ---")
    if gb.ss.config["jit_c_control"] == "off":
        print("  SKIPPED (JIT is off)")
    else:
        try:
            from graphblas import dtypes

            prev_burble = gb.ss.config["burble"]
            gb.ss.config["burble"] = True
            dtype = dtypes.ss.register_new(
                "jit_diag_test",
                "typedef struct { int val ; } jit_diag_test ;",
            )
            gb.ss.config["burble"] = prev_burble
            print(f"  SUCCESS: registered type '{dtype.name}'")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            try:
                err_log = gb.ss.config["jit_error_log"]
                if err_log:
                    print(f"  JIT error log: {err_log[:500]}")
            except Exception:
                pass

    # Print final JIT state
    print("\n--- Final JIT state ---")
    print(f"  jit_c_control: {gb.ss.config['jit_c_control']}")
    print(f"  jit_c_compiler_name: {gb.ss.config['jit_c_compiler_name']}")

    print(f"\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

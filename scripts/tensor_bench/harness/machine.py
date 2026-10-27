"""What a result needs to be comparable: the machine, the OS, the versions, the JIT.

``machine_info()`` is recorded once per run (``run.py``). Everything is best effort: a field
that cannot be read on this platform is left out rather than guessed.
"""

import os
import platform
import subprocess
import sys
import time
from importlib import metadata


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name):
    return _run(["sysctl", "-n", name])


def cpu_info():
    info = {"logical_cores": os.cpu_count(), "machine": platform.machine()}
    system = platform.system()
    if system == "Darwin":
        info["model"] = _sysctl("machdep.cpu.brand_string")
        info["physical_cores"] = int(_sysctl("hw.physicalcpu") or 0) or None
        # Apple silicon: perflevel0 is the performance cluster, perflevel1 the efficiency one
        levels = {}
        for lvl in range(4):
            name = _sysctl(f"hw.perflevel{lvl}.name")
            if not name:
                break
            levels[name] = int(_sysctl(f"hw.perflevel{lvl}.physicalcpu") or 0)
        if levels:
            info["core_types"] = levels
        info["memory_gb"] = round(int(_sysctl("hw.memsize") or 0) / 2**30, 1)
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        # Intel hybrid CPUs list their performance and efficiency cores separately
        levels = {}
        for kind, label in (("cpu_core", "Performance"), ("cpu_atom", "Efficiency")):
            try:
                with open(f"/sys/devices/{kind}/cpus") as f:
                    levels[label] = _count_cpulist(f.read().strip())
            except OSError:
                pass
        if levels:
            info["core_types"] = levels
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["memory_gb"] = round(kb / 2**20, 1)
        except (OSError, ValueError, IndexError):
            pass
    else:
        info["model"] = platform.processor()
    try:
        import psutil

        info["physical_cores"] = info.get("physical_cores") or psutil.cpu_count(logical=False)
        info.setdefault("memory_gb", round(psutil.virtual_memory().total / 2**30, 1))
    except ImportError:
        pass
    return info


def _count_cpulist(text):
    n = 0
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            n += int(b) - int(a) + 1
        elif part:
            n += 1
    return n


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def versions():
    out = {"python": platform.python_version()}
    for dist in (
        "python-graphblas",
        "suitesparse-graphblas",
        "numpy",
        "scipy",
        "numba",
        "sparse",
        "torch",
        "pandas",
        "tensorly",
        "opt_einsum",
    ):
        v = _version(dist)
        if v is not None:
            out[dist] = v
    return out


def graphblas_info(jit="on"):
    """The SuiteSparse library, the JIT after ``fix_jit_config()`` and run.py's ``--jit``, and
    where graphblas is.
    """
    try:
        import graphblas as gb
        from graphblas.core import _has_numba
        from graphblas.core.ss import jit_config
    except ImportError as e:
        return {"error": f"graphblas does not import: {e}"}
    fixed = jit_config.fix_jit_config()
    cfg = gb.ss.config
    if jit != "on":
        cfg["jit_c_control"] = jit
    return {
        "path": os.path.dirname(gb.__file__),
        "suitesparse": ".".join(map(str, gb.ss.about["library_version"])),
        "fix_jit_config": fixed,
        "jit_c_control": cfg["jit_c_control"],
        "jit_c_compiler_name": cfg["jit_c_compiler_name"],
        "nthreads_default": cfg["nthreads"],
        "has_numba": bool(_has_numba),
    }


def git_commit(path):
    sha = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])
    return (sha + ("+dirty" if dirty else "")) if sha else None


def machine_info(repo=None, jit="on"):
    info = {
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": cpu_info(),
        "versions": versions(),
        "graphblas": graphblas_info(jit),
        "env": {
            k: os.environ[k]
            for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "CONDA_PREFIX")
            if k in os.environ
        },
        "python_executable": sys.executable,
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if hasattr(os, "getloadavg"):
        info["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    if repo is not None:
        info["git_commit"] = git_commit(repo)
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(machine_info(), indent=2))

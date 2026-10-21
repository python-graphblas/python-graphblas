"""Run the SPLATT, TACO and Finch drivers for one (library, dataset, group), as run.py's
Python libraries run: rows in the same JSON-lines file, with the same fingerprints.

The drivers read raw binary copies of the data (``export_bin.py``, made on first use). The
binaries are where ``build_splatt.sh`` and ``build_taco.sh`` put them (``<build>/bin``). Finch
runs with ``julia`` (or ``TENSOR_BENCH_JULIA``) in a Julia project that has Finch
(``JULIA_PROJECT``); see README.md.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

NMAX = "9"
# driver groups for each harness group, and whether the driver runs multithreaded there
TACO_GROUPS = {
    "mttkrp": (["mttkrp16", "mttkrp64"], True),
    "ttm": (["ttm16"], True),  # only the output's zeroing loop is parallel
    # TACO's default schedule emits no parallel loop for sparse outputs
    "ttv": (["ttv"], False),
    "reduce": (["sum"], False),
    "ewise": (["ewise"], False),
}
FINCH_GROUPS = {
    "mttkrp": ["mttkrp16", "mttkrp64"],
    "ttm": ["ttm16"],
    "ttv": ["ttv"],
    "reduce": ["sum"],
    "ewise": ["ewise"],
}


def _load1():
    return round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None


def _append(out, rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _drive(cmd, out, base, env, timeout):
    """Run one driver command; tag and append its JSON lines. Returns a status string."""
    l0 = _load1()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        _append(out, dict(base, case="*", status="timeout", note=f"driver timeout {timeout:.0f} s"))
        return "TIMEOUT"
    n = 0
    for line in p.stdout.splitlines():
        if not line.startswith("{"):
            continue
        rec = dict(base)
        rec.update(json.loads(line))
        rec["lib"] = base["lib"]
        rec.update(
            status="ok", load1_start=l0, load1=_load1(), time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        _append(out, rec)
        n += 1
    if p.returncode:
        tail = " / ".join(p.stderr.strip().splitlines()[-3:])
        _append(out, dict(base, case="*", status="crash", note=tail[:500], cmd=" ".join(cmd)))
        return f"exit {p.returncode} | {tail}"
    return f"{n} rows"


def _export(ds, env):
    prefix = paths.data_dir() / "bin" / ds
    if not Path(f"{prefix}_v.f64").exists():
        subprocess.run([sys.executable, str(HERE / "export_bin.py"), ds], check=True, env=env)
    return str(prefix)


def run(lib, ds, group, out, threads, env, timeout):
    import datasets as D

    base = {"dataset": ds, "group": group, "lib": lib}
    bindir = paths.build_dir() / "bin"
    tlist = threads.split(",")
    if lib == "splatt":
        exe = bindir / "splatt_drv"
        if not exe.exists():
            return f"skipped: {exe} not built (compiled/build_splatt.sh)"
        prefix = _export(ds, env)
        status = []
        for R in (16, 64):
            for alloc in ("allmode", "twomode", "onemode"):
                b = dict(base, lib=f"splatt-{alloc}")
                cmd = [str(exe), f"{prefix}.tb", f"{prefix}_f{R}.f64", str(R), alloc, NMAX, *tlist]
                status.append(_drive(cmd, out, b, env, timeout))
        return "; ".join(status)
    if lib == "taco":
        exe = bindir / "taco_drv"
        if not exe.exists():
            return f"skipped: {exe} not built (compiled/build_taco.sh)"
        if group not in TACO_GROUPS:
            return "n/a"
        prefix = _export(ds, env)
        settings = bindir / "taco_env.json"
        if settings.exists():  # written by build_taco.sh; the caller's environment wins
            env = dict(json.loads(settings.read_text()), **env)
        groups, parallel = TACO_GROUPS[group]
        status = []
        for g in groups:
            cmd = [str(exe), prefix, g, NMAX, *(tlist if parallel else ["1"])]
            status.append(_drive(cmd, out, base, env, timeout))
        return "; ".join(status)
    if lib == "finch":
        julia = os.environ.get("TENSOR_BENCH_JULIA", "julia")
        if group not in FINCH_GROUPS:
            return "n/a"
        if len(D.SPECS[ds][1] or ()) not in (0, 3) or ds in {"uber", "nips", "enron", "chicago"}:
            _append(out, dict(base, case="*", status="n/a", note="the Finch kernels here are 3-d"))
            return "n/a (3-d only)"
        prefix = _export(ds, env)
        status = []
        for g in FINCH_GROUPS[group]:
            cmd = [julia, str(HERE / "finch_drv.jl"), prefix, g, NMAX]
            status.append(_drive(cmd, out, base, env, timeout))
        return "; ".join(status)
    raise ValueError(lib)

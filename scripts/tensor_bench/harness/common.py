"""Timing, correctness fingerprints and the kernel case list shared by all adapters."""

import gc
import itertools
import json
import os
import resource
import time

import numpy as np


class NA(Exception):
    """The library cannot do this kernel (or it would not fit in memory); the message says why."""


# Guard for dense outputs and intermediates that the COO libraries allocate (bytes)
DENSE_LIMIT = 4 * 2**30


def dense_guard(shape, what="dense output", itemsize=8):
    nbytes = int(np.prod(shape, dtype=object)) * itemsize
    if nbytes > DENSE_LIMIT:
        raise NA(f"{what} {tuple(shape)} would need {nbytes / 2**30:.1f} GiB (limit 4 GiB)")


# Guard for TTM outputs: fibers x R entries (every library stores or fingerprints them)
TTM_ENTRY_LIMIT = 4 * 10**8


def ttm_guard(ds, coords, shape, m, R):
    import datasets as D

    if (n := D.fiber_counts(ds, coords, shape)[m] * R) > TTM_ENTRY_LIMIT:
        raise NA(
            f"TTM output has {n / 1e6:.0f}M entries (fibers x R); "
            f"limit {TTM_ENTRY_LIMIT / 1e6:.0f}M"
        )


def perms_for(ndim):
    ident = tuple(range(ndim))
    if ndim == 3:
        return [p for p in itertools.permutations(ident) if p != ident]
    return [(1, 0, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 2, 1, 0)]


def cases(group, shape, dataset):
    """The case names of ``group`` for a tensor of ``shape``."""
    nd = len(shape)
    if group == "construct":
        return ["from_coo_sorted", "from_coo_shuffled", "to_coo"]
    if group == "ewise":
        return ["add_same", "mul_same", "add_perm", "mul_perm"]
    if group == "bcast":
        return [f"mul_vec{a}" for a in range(nd)] + (["mul_mat12"] if nd == 3 else [])
    if group == "reduce":
        out = [f"sum{a}" for a in range(nd)]
        out += ["sum" + "".join(map(str, p)) for p in itertools.combinations(range(nd), 2)]
        return out + ["sumall"]
    if group == "permute":
        return ["perm_" + "".join(map(str, p)) for p in perms_for(nd)]
    if group == "ttv":
        return [f"ttv{m}" for m in range(nd)]
    if group == "ttm":
        return [f"ttm{m}_r16" for m in range(nd)] + [f"ttm{m}_r64" for m in range(nd)]
    if group == "mttkrp":
        return [f"mttkrp{m}_r16" for m in range(nd)]
    if group == "contract":
        return ["tdot1", "tdot2"]
    if group == "bmm":
        return ["bmm"]
    if group == "slice":
        return [f"slice{a}" for a in range(nd)] + [f"range{a}" for a in range(nd)]
    raise ValueError(group)


def parse_perm(case):
    return tuple(int(ch) for ch in case.split("_")[1])


def parse_axes(case):
    """'sum02' -> (0, 2); 'sumall' -> None."""
    tail = case[3:]
    return None if tail == "all" else tuple(int(ch) for ch in tail)


def timeit(fn, max_total=60.0):
    """Run ``fn`` once (warm-up), then N more times; return (best, median, n, first, result).

    N is 9 below 0.1 s per call, 5 below 1 s, 3 below 30 s; a warm-up over 30 s is reported
    as a single run (n = 1). Times in seconds. GC is disabled while timing.
    """
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        result = fn()
        first = time.perf_counter() - t0
        if first > 30:
            return first, first, 1, first, result
        n = 9 if first < 0.1 else 5 if first < 1 else 3
        times = []
        for _ in range(n):
            result = None
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
            if sum(times) > max_total and len(times) >= 3:
                break
        return min(times), float(np.median(times)), len(times), first, result
    finally:
        gc.enable()


_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)


def _mix(lin):
    h = lin.astype(np.uint64) * _M1
    h ^= h >> np.uint64(29)
    h *= _M2
    h ^= h >> np.uint64(32)
    return h


CHUNK = 1 << 25


class _FP:
    """Accumulates the fingerprint over chunks (all parts are order-independent sums)."""

    def __init__(self, shape):
        self.shape = [int(s) for s in shape]
        self.count = 0
        self.h = 0
        self.v = 0.0
        self.w = 0.0
        self.sorted = True
        self.last = None

    def add(self, lin, vals):
        lin = np.asarray(lin).astype(np.uint64, copy=False)
        vals = np.asarray(vals, dtype=np.float64)
        if lin.size:
            if self.last is not None and lin[0] <= self.last:
                self.sorted = False
            if lin.size > 1 and not np.all(lin[1:] > lin[:-1]):
                self.sorted = False
            self.last = lin[-1]
        nz = vals != 0
        if not nz.all():
            lin, vals = lin[nz], vals[nz]
        h = _mix(lin)
        w = (h >> np.uint64(11)).astype(np.float64) * 2.0**-53
        self.count += int(lin.size)
        self.h = (self.h + int(h.sum(dtype=np.uint64))) % 2**64
        self.v += float(vals.sum())
        self.w += float((vals * w).sum())

    def result(self, with_sorted=True):
        out = {
            "shape": self.shape,
            "count": self.count,
            "hsum": hex(self.h),
            "vsum": self.v,
            "wsum": self.w,
        }
        if with_sorted:
            out["sorted"] = self.sorted
        return out


def fingerprint(canon):
    """Order-independent fingerprint of a result with presence = nonzero.

    ``canon`` is ("coo", coords, vals, shape) with no duplicate coordinates, ("dense", array),
    ("scalar", value) or ("fibers", fa, fb, out2d, (sa, sb)) for a TTM result held as dense
    rows over fibers. "sorted" says whether the entries came in C order (coo only).
    """
    kind = canon[0]
    if kind == "scalar":
        fp = _FP(())
        fp.add(np.zeros(1, np.uint64), np.array([canon[1]], float))
        return fp.result(False)
    if kind == "dense":
        arr = np.asarray(canon[1])
        fp = _FP(arr.shape)
        flat = arr.reshape(-1)
        for i in range(0, max(flat.size, 1), CHUNK):
            part = flat[i : i + CHUNK]
            nzi = np.flatnonzero(part)
            fp.add(nzi.astype(np.uint64) + np.uint64(i), part[nzi])
        return fp.result(False)
    if kind == "fibers":
        _, fa, fb, out, (sa, sb) = canon
        R = out.shape[1]
        fp = _FP((sa, sb, R))
        step = max(1, CHUNK // R)
        r = np.arange(R, dtype=np.uint64)
        for i in range(0, fa.size, step):
            base = (
                fa[i : i + step].astype(np.uint64) * np.uint64(sb)
                + fb[i : i + step].astype(np.uint64)
            ) * np.uint64(R)
            fp.add((base[:, None] + r[None, :]).ravel(), out[i : i + step].ravel())
        return fp.result()
    _, coords, vals, shape = canon
    vals = np.asarray(vals)
    if vals.ndim == 0:
        vals = vals.reshape(1)
    fp = _FP(shape)
    n = vals.size
    for i in range(0, max(n, 1), CHUNK):
        lin = np.zeros(min(CHUNK, n - i) if n else 0, np.uint64)
        for c, s in zip(coords, shape, strict=True):
            lin *= np.uint64(s)
            lin += np.asarray(c[i : i + CHUNK]).astype(np.uint64)
        fp.add(lin, vals[i : i + CHUNK])
    return fp.result()


def fp_match(a, b, rtol=1e-8):
    if a is None or b is None:
        return None
    if a["shape"] != b["shape"] or a["count"] != b["count"] or a["hsum"] != b["hsum"]:
        return False
    for k in ("vsum", "wsum"):
        x, y = a[k], b[k]
        if abs(x - y) > rtol * max(abs(x), abs(y), 1e-300):
            return False
    return True


def peak_rss_mb():
    # ru_maxrss is bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def load1():
    return os.getloadavg()[0]


class Recorder:
    def __init__(self, path, base):
        self.path = path
        self.base = base

    def write(self, **fields):
        rec = dict(self.base)
        rec.update(fields)
        rec["load1"] = round(load1(), 2)
        rec["peak_rss_mb"] = round(peak_rss_mb(), 1)
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

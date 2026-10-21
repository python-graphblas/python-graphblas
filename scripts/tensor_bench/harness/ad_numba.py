"""Hand-written Numba kernels for TTV, TTM and MTTKRP (3-d): the "speed of light from Python".

Each mode gets its own sorted copy of the COO data, built once outside the timed region (its
cost is recorded as a ``prep_*`` case):

- TTV / TTM mode m: entries sorted by (other modes ascending, then m), so each output fiber
  (one per distinct pair of other-mode coordinates) is a contiguous run. The output is sparse
  in the fibers (like GraphBLAS: an empty fiber has no output), dense in R for TTM.
- MTTKRP mode m: a CSF tree rooted at m (root m, then the other mode with the smaller size,
  then the last mode at the leaves). out[i] += B[j] * sum_k X[i,j,k] C[k] per fiber.

Parallelism: ``prange`` over 64 chunks of fibers holding about equal numbers of entries, so
heavy slices are split. TTV/TTM fibers belong to one chunk, so no write conflicts. For
MTTKRP, output rows strictly inside a chunk's row range are owned by that chunk; the first
and last row of each chunk accumulate into a per-chunk side buffer that is added serially
afterwards. No atomics, no per-thread copies of the output.
"""

import time

import datasets as D
import numba
import numpy as np
from common import NA, ttm_guard
from numba import njit, prange

NCHUNKS = 64


@njit(parallel=True, fastmath=False)
def _ttv(fptr, leaf, vals, v, chunks):
    nfib = fptr.size - 1
    out = np.empty(nfib)
    for c in prange(chunks.size - 1):
        for f in range(chunks[c], chunks[c + 1]):
            s = 0.0
            for p in range(fptr[f], fptr[f + 1]):
                s += vals[p] * v[leaf[p]]
            out[f] = s
    return out


@njit(parallel=True, fastmath=False)
def _ttm(fptr, leaf, vals, U, chunks):
    nfib = fptr.size - 1
    R = U.shape[1]
    out = np.zeros((nfib, R))
    for c in prange(chunks.size - 1):
        for f in range(chunks[c], chunks[c + 1]):
            for p in range(fptr[f], fptr[f + 1]):
                x = vals[p]
                k = leaf[p]
                for r in range(R):
                    out[f, r] += x * U[k, r]
    return out


@njit(parallel=True, fastmath=False)
def _mttkrp(nrows, fslice, fmid, fptr, leaf, vals, B, C, chunks):
    R = B.shape[1]
    nch = chunks.size - 1
    out = np.zeros((nrows, R))
    side = np.zeros((nch, 2, R))
    for c in prange(nch):
        f0 = chunks[c]
        f1 = chunks[c + 1]
        if f0 == f1:
            continue
        first = fslice[f0]
        last = fslice[f1 - 1]
        tmp = np.empty(R)
        for f in range(f0, f1):
            for r in range(R):
                tmp[r] = 0.0
            for p in range(fptr[f], fptr[f + 1]):
                x = vals[p]
                k = leaf[p]
                for r in range(R):
                    tmp[r] += x * C[k, r]
            i = fslice[f]
            j = fmid[f]
            if i == first:
                for r in range(R):
                    side[c, 0, r] += tmp[r] * B[j, r]
            elif i == last:
                for r in range(R):
                    side[c, 1, r] += tmp[r] * B[j, r]
            else:
                for r in range(R):
                    out[i, r] += tmp[r] * B[j, r]
    for c in range(nch):
        f0 = chunks[c]
        f1 = chunks[c + 1]
        if f0 == f1:
            continue
        first = fslice[f0]
        last = fslice[f1 - 1]
        for r in range(R):
            out[first, r] += side[c, 0, r]
        if last != first:
            for r in range(R):
                out[last, r] += side[c, 1, r]
    return out


def _balanced_chunks(fptr, nchunks=NCHUNKS):
    nnz = fptr[-1]
    targets = np.linspace(0, nnz, nchunks + 1)
    chunks = np.searchsorted(fptr, targets, side="left").astype(np.int64)
    chunks[0], chunks[-1] = 0, fptr.size - 1
    return np.maximum.accumulate(chunks)


def _sorted_by(coords, vals, shape, order_axes):
    """Entries sorted lexicographically by ``order_axes``; returns (sorted coords, vals, keys)."""
    lin = np.zeros(vals.size, np.int64)
    for a in order_axes:
        lin *= shape[a]
        lin += coords[a]
    if np.all(lin[1:] > lin[:-1]):
        perm = None
    else:
        perm = np.argsort(lin, kind="stable")
    sc = [c if perm is None else c[perm] for c in coords]
    sv = vals if perm is None else vals[perm]
    return sc, sv


def _fiber_ptr(*keys):
    """Start offsets of runs where any of ``keys`` changes, plus the end."""
    n = keys[0].size
    change = np.zeros(n, bool)
    change[0] = True
    for k in keys:
        change[1:] |= k[1:] != k[:-1]
    starts = np.flatnonzero(change)
    return np.append(starts, n).astype(np.int64)


class Adapter:
    name = "numba"
    threads = (8, 1)

    def __init__(self, ds, coords, vals, shape):
        if len(shape) != 3:
            raise NA("hand-written Numba kernels are 3-d only")
        self.ds, self.coords, self.vals, self.shape = ds, coords, vals, shape
        self.prep = {}
        self.prep_s = {}

    def info(self):
        try:
            layer = numba.threading_layer()
        except ValueError:
            layer = "not yet initialized"
        return {"version": f"numba {numba.__version__}", "threading_layer": layer}

    def set_threads(self, nt):
        numba.set_num_threads(nt)

    def extra_cases(self, group):
        return []

    def setup(self, group):
        if group not in {"ttv", "ttm", "mttkrp"}:
            raise NA("hand-written Numba kernels exist for TTV, TTM and MTTKRP only")
        shape = self.shape
        if group == "ttv":
            self.vecs = D.vectors(self.ds, shape)
        else:
            self.facs = {R: D.factors(self.ds, shape, R) for R in (16, 64)}

    def _fibers(self, m):
        key = ("fib", m)
        if key not in self.prep:
            t0 = time.perf_counter()
            a, b = (x for x in range(3) if x != m)
            sc, sv = _sorted_by(self.coords, self.vals, self.shape, (a, b, m))
            fptr = _fiber_ptr(sc[a], sc[b])
            starts = fptr[:-1]
            self.prep[key] = (
                fptr,
                sc[m],
                sv,
                _balanced_chunks(fptr),
                sc[a][starts],
                sc[b][starts],
                (a, b),
            )
            self.prep_s[key] = time.perf_counter() - t0
        return self.prep[key]

    def _csf(self, m):
        key = ("csf", m)
        if key not in self.prep:
            t0 = time.perf_counter()
            o = [x for x in range(3) if x != m]
            mid, lf = (o[0], o[1]) if self.shape[o[0]] <= self.shape[o[1]] else (o[1], o[0])
            sc, sv = _sorted_by(self.coords, self.vals, self.shape, (m, mid, lf))
            fptr = _fiber_ptr(sc[m], sc[mid])
            starts = fptr[:-1]
            self.prep[key] = (
                sc[m][starts],
                sc[mid][starts],
                fptr,
                sc[lf],
                sv,
                _balanced_chunks(fptr),
                mid,
                lf,
            )
            self.prep_s[key] = time.perf_counter() - t0
        return self.prep[key]

    def prep_seconds(self, group, case):
        m = int(case[3] if group != "mttkrp" else case[6])
        return self.prep_s.get(("csf" if group == "mttkrp" else "fib", m))

    def case(self, group, case):
        if group == "ttv":
            m = int(case[3:])
            fptr, leaf, sv, chunks, fa, fb, ab = self._fibers(m)
            v = self.vecs[m]
            self._last = ("ttv", fa, fb, ab)
            return lambda: _ttv(fptr, leaf, sv, v, chunks)
        if group == "ttm":
            m, R = int(case[3]), int(case.split("_r")[1])
            ttm_guard(self.ds, self.coords, self.shape, m, R)
            fptr, leaf, sv, chunks, fa, fb, ab = self._fibers(m)
            U = self.facs[R][m]
            self._last = ("ttm", fa, fb, ab)
            return lambda: _ttm(fptr, leaf, sv, U, chunks)
        if group == "mttkrp":
            m = int(case[6])
            fslice, fmid, fptr, leaf, sv, chunks, mid, lf = self._csf(m)
            F = self.facs[16]
            B, C = F[mid], F[lf]
            n = self.shape[m]
            self._last = ("mttkrp",)
            return lambda: _mttkrp(n, fslice, fmid, fptr, leaf, sv, B, C, chunks)
        raise NA(f"unknown case {group}/{case}")

    def canon(self, r):
        kind = self._last[0]
        if kind == "mttkrp":
            return ("dense", r)
        _, fa, fb, (a, b) = self._last
        shp = (self.shape[a], self.shape[b])
        if kind == "ttv":
            return ("coo", [fa, fb], r, shp)
        return ("fibers", fa, fb, r, shp)

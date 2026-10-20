"""Numba kernels for the n-d ``Tensor``, over the stored Matrix's arrays (no copies).

Imported only when Numba is installed; every caller has a GraphBLAS fallback.

MTTKRP of a 3-d tensor reads the Container's arrays in place, in the layout the tensor is
stored in, with no (fibers x R) intermediate:

- rows ``(r0, r1)``, columns ``(c,)`` (CSF-like: the rows are the (r0, r1) fibers in order):
  for mode r0, ``t = sum_c X[r0, r1, c] F_c[c]`` per row, then ``out[r0] += t * F_r1[r1]``;
  for mode r1 the same t goes to ``out[r1] += t * F_r0[r0]``; for mode c each entry adds
  ``x * (F_r0[r0] * F_r1[r1])`` to ``out[c]``.
- rows ``(r,)``, columns ``(c0, c1)``: for mode r, ``out[r] += x * F_c0[c0] * F_c1[c1]``
  over the row; for mode c0 (or c1) each entry adds ``x * F_r[r] * F_c1[c1]`` to ``out[c0]``.

When the mode is the first row axis, chunks of rows start where it changes, so no two chunks
write the same output row. Otherwise each chunk (one per thread) has its own output buffer,
summed at the end; ``mttkrp3`` returns None when those buffers would be too large.
"""

import numba
import numpy as np
from numba import njit, prange

_NCHUNKS = 64
# Kernels are compiled per R: with R a constant the inner loops unroll and vectorize, 1.7-1.9x
# faster than with R read at run time (u7 mode 0 4.3 vs 6.4 ms at 8 threads, 22 vs 39 at 1;
# NELL-2 30 vs 49, 161 vs 301). About 0.3 s to compile each, once per process.
_compiled = {}


def _kernel(name, R):
    if (k := _compiled.get((name, R))) is None:
        k = _compiled[name, R] = njit(parallel=True)(globals()[f"_make{name}"](R))
    return k


def _make_rows2_first(R):
    def kernel(n1, p, h, ci, x, iso, F1, Fc, out, touched, starts):
        # rows (r0, r1), mode r0: chunks own their r0
        for c in prange(starts.size - 1):
            tmp = np.empty(R)
            last = -1
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                for r in range(R):
                    tmp[r] = 0.0
                for q in range(q0, q1):
                    xv = x[0] if iso else x[q]
                    k = ci[q]
                    for r in range(R):
                        tmp[r] += xv * Fc[k, r]
                row = h[s] if h.size > 0 else s
                i0 = row // n1
                i1 = row - i0 * n1
                for r in range(R):
                    out[i0, r] += tmp[r] * F1[i1, r]
                if i0 != last:
                    # Once per output row: a write per fiber would share cache lines across threads
                    touched[i0] = True
                    last = i0

    return kernel


def _make_rows2_second(R):
    def kernel(n1, p, h, ci, x, iso, F0, Fc, outb, touchedb, starts):
        # rows (r0, r1), mode r1: one buffer per chunk
        for c in prange(starts.size - 1):
            tmp = np.empty(R)
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                for r in range(R):
                    tmp[r] = 0.0
                for q in range(q0, q1):
                    xv = x[0] if iso else x[q]
                    k = ci[q]
                    for r in range(R):
                        tmp[r] += xv * Fc[k, r]
                row = h[s] if h.size > 0 else s
                i0 = row // n1
                i1 = row - i0 * n1
                for r in range(R):
                    outb[c, i1, r] += tmp[r] * F0[i0, r]
                touchedb[c, i1] = True

    return kernel


def _make_rows2_col(R):
    def kernel(n1, p, h, ci, x, iso, F0, F1, outb, touchedb, starts):
        # rows (r0, r1), mode c: one buffer per chunk
        for c in prange(starts.size - 1):
            w = np.empty(R)
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                row = h[s] if h.size > 0 else s
                i0 = row // n1
                i1 = row - i0 * n1
                for r in range(R):
                    w[r] = F0[i0, r] * F1[i1, r]
                for q in range(q0, q1):
                    xv = x[0] if iso else x[q]
                    k = ci[q]
                    for r in range(R):
                        outb[c, k, r] += xv * w[r]
                    touchedb[c, k] = True

    return kernel


def _make_rows1_row(R):
    def kernel(nc1, p, h, ci, x, iso, Fc0, Fc1, out, touched, starts):
        # rows (r,), columns (c0, c1), mode r: each row slot is its own output row
        for c in prange(starts.size - 1):
            tmp = np.empty(R)
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                for r in range(R):
                    tmp[r] = 0.0
                for q in range(q0, q1):
                    xv = x[0] if iso else x[q]
                    col = ci[q]
                    k0 = col // nc1
                    k1 = col - k0 * nc1
                    for r in range(R):
                        tmp[r] += xv * Fc0[k0, r] * Fc1[k1, r]
                row = h[s] if h.size > 0 else s
                for r in range(R):
                    out[row, r] += tmp[r]
                touched[row] = True

    return kernel


def _make_rows1_col(R):
    def kernel(nc1, first, p, h, ci, x, iso, Fr, Fother, outb, touchedb, starts):
        # rows (r,), columns (c0, c1), mode c0 (first) or c1: one buffer per chunk
        for c in prange(starts.size - 1):
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                row = h[s] if h.size > 0 else s
                for q in range(q0, q1):
                    xv = x[0] if iso else x[q]
                    col = ci[q]
                    k0 = col // nc1
                    k1 = col - k0 * nc1
                    t = k0 if first else k1
                    o = k1 if first else k0
                    for r in range(R):
                        outb[c, t, r] += xv * Fr[row, r] * Fother[o, r]
                    touchedb[c, t] = True

    return kernel


def _starts(p, nchunks, change=None):
    """Row slots where chunks start, about equal entries each; with ``change`` (the slots
    where the output row changes), each start moves forward to the next of those.
    """
    nslots = p.size - 1
    cand = np.searchsorted(p, np.linspace(0, p[-1], nchunks + 1)[1:-1], side="left")
    if change is not None:
        idx = np.searchsorted(change, cand, side="left")
        cand = change[idx[idx < change.size]]
    return np.unique(np.concatenate([[0], cand, [nslots]])).astype(np.int64)


def mttkrp3(arrays, row_sizes, col_sizes, mode_pos, F, n_out, nthreads, max_buffer):
    """``(out, touched)`` for a 3-d MTTKRP over the stored arrays, or None when it needs
    per-chunk buffers larger than ``max_buffer`` entries.

    ``arrays`` is ``(p, h, ci, x, iso)`` (h empty when not hypersparse); ``row_sizes`` and
    ``col_sizes`` are the sizes of the row and column group axes (two and one, or one and two);
    ``mode_pos`` is the mode's position in rows + columns; ``F`` holds the dense factors in that
    same order (the mode's entry is ignored); ``n_out`` is the mode's size.
    """
    p, h, ci, x, iso = arrays
    R = next(f.shape[1] for k, f in enumerate(F) if k != mode_pos)
    nthreads = max(
        1, min(nthreads or numba.config.NUMBA_NUM_THREADS, numba.config.NUMBA_NUM_THREADS)
    )
    owner = mode_pos == 0
    if not owner and nthreads * n_out * R > max_buffer:
        return None
    prev = numba.get_num_threads()
    numba.set_num_threads(nthreads)
    try:
        if owner:
            out = np.zeros((n_out, R))
            touched = np.zeros(n_out, np.bool_)
            if len(row_sizes) == 2:
                n1 = row_sizes[1]
                if h.size > 0:
                    change = np.flatnonzero(np.diff(h.astype(np.int64) // n1)) + 1
                else:
                    change = np.arange(n1, p.size - 1, n1, dtype=np.int64)
                starts = _starts(p, _NCHUNKS, change)
                _kernel("_rows2_first", R)(n1, p, h, ci, x, iso, F[1], F[2], out, touched, starts)
            else:
                starts = _starts(p, _NCHUNKS)
                _kernel("_rows1_row", R)(
                    col_sizes[1], p, h, ci, x, iso, F[1], F[2], out, touched, starts
                )
            return out, touched
        starts = _starts(p, nthreads)
        nch = starts.size - 1
        outb = np.zeros((nch, n_out, R))
        touchedb = np.zeros((nch, n_out), np.bool_)
        if len(row_sizes) == 2:
            n1 = row_sizes[1]
            if mode_pos == 1:
                _kernel("_rows2_second", R)(
                    n1, p, h, ci, x, iso, F[0], F[2], outb, touchedb, starts
                )
            else:
                _kernel("_rows2_col", R)(n1, p, h, ci, x, iso, F[0], F[1], outb, touchedb, starts)
        else:
            first = mode_pos == 1
            Fother = F[2] if first else F[1]
            _kernel("_rows1_col", R)(
                col_sizes[1], first, p, h, ci, x, iso, F[0], Fother, outb, touchedb, starts
            )
        return outb.sum(axis=0), touchedb.any(axis=0)
    finally:
        numba.set_num_threads(prev)

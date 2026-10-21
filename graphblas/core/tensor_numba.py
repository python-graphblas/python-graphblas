"""Numba kernels for the n-d ``Tensor``, over the stored Matrix's arrays (no copies).

Imported only when Numba is installed; every caller has a GraphBLAS fallback.

MTTKRP of a tensor of any number of axes reads the Container's arrays in place, in the layout
the tensor is stored in (both groups holding at least one axis), with no (fibers x R)
intermediate. Each row of the stored Matrix is a fiber of the row axes; its entries are sorted
by column. Per row the factors of the row axes multiply once; per entry, those of the column
axes (or, for runs of entries sharing the leading column axes, once per run, as a CSF tree
does). The mode's output row is then the row's own (mode in the row group: one add per row)
or the entry's (mode in the column group: one add per entry or per run).

When the mode is the first row axis, chunks of rows start where it changes, so no two chunks
write the same output row. Otherwise each thread has its own output buffer, summed at the end;
``mttkrp_nd`` returns None when those buffers would be too large.
"""

import numba
import numpy as np
from numba import njit, prange

_NCHUNKS = 64
# Kernels are compiled per R and per layout: with R a constant the inner loops unroll and
# vectorize, 1.7-1.9x faster than with R read at run time (u7 mode 0 4.3 vs 6.4 ms at 8
# threads, 22 vs 39 at 1; NELL-2 30 vs 49, 161 vs 301). About 0.2-0.4 s to compile each, once
# per process.
_compiled = {}


def _bins(p, starts, nbins):
    """Chunks dealt into ``nbins`` bins of about equal entries, largest chunk first to the
    lightest bin (prange schedules statically, one bin per thread, so the bins must balance
    themselves; an owner chunk cannot split a heavy output row, and Enron's largest axis-0 slice
    holds 9.7% of the entries). Returns ``(bin_ptr, bins)``, the chunk ids of bin b being
    ``bins[bin_ptr[b]:bin_ptr[b + 1]]``.
    """
    sizes = p[starts[1:]].astype(np.int64) - p[starts[:-1]].astype(np.int64)
    nbins = max(1, min(nbins, sizes.size))
    load = np.zeros(nbins, np.int64)
    which = np.empty(sizes.size, np.int64)
    for c in np.argsort(-sizes, kind="stable"):
        b = int(np.argmin(load))
        which[c] = b
        load[b] += sizes[c]
    bins = np.argsort(which, kind="stable")
    bin_ptr = np.searchsorted(which[bins], np.arange(nbins + 1))
    return bin_ptr.astype(np.int64), bins.astype(np.int64)


def _starts(p, nchunks, change=None):
    """Row slots where chunks start, about equal entries each; with ``change`` (the slots
    where the output row changes), each start moves forward to the next of those.
    """
    nslots = p.size - 1
    # Targets in p's own dtype: float targets would convert all of p (2.5 ms on Uber's 5e6 rows)
    targets = np.linspace(0, p[-1], nchunks + 1)[1:-1].astype(p.dtype)
    cand = np.searchsorted(p, targets, side="left")
    if change is not None:
        idx = np.searchsorted(change, cand, side="left")
        cand = change[idx[idx < change.size]]
    return np.unique(np.concatenate([[0], cand, [nslots]])).astype(np.int64)


_ROWS, _PREFIX, _LAST = 0, 1, 2
_FORCE_RUNS = None  # tests set True or False to cover both ways of reading column runs


def _kernel_source(R, P, Q, kind, mpos, owner, runs):
    """Python source of the kernel for P row axes and Q column axes, the mode being the row
    axis ``mpos`` (``_ROWS``), the column axis ``mpos`` other than the last (``_PREFIX``), or
    the last column axis (``_LAST``). ``owner``: the mode is the first row axis and chunks own
    their output rows; otherwise each thread adds into its own buffer.

    The source is written out per axis, with R a literal, so Numba sees no loop over axes and
    no tuple of factors (a generic loop over both ran 2x slower than the 3-d kernels). Per
    row, the factors of the row axes multiply once (``w``). With ``runs`` (two or more column
    axes), entries come sorted by column, so runs of equal leading column axes (the prefix) sum
    ``x * F_last`` first and multiply by the prefix factors once per run, as a CSF tree does;
    without it each entry multiplies its column factors itself, cheaper when runs are about
    one entry long (NIPS and Enron: 1.0 and 1.7).
    """
    N = P + Q
    L = N - 1  # the last column axis
    m = mpos if kind == _ROWS else P + mpos  # the mode's position in rows + columns
    row_w = [a for a in range(P) if a != m]  # row axes in w
    pre_f = [a for a in range(P, L) if a != m]  # prefix axes multiplied per run
    o = "out[" if owner else "out[b, "  # then the output row, r
    tt = "touched[" if owner else "touched[b, "
    lines = []
    add = lines.append
    args = ", ".join(f"F{a}" for a in range(N))
    add(f"def kernel(p, h, ci, x, iso, S, {args}, out, touched, starts, bin_ptr, bins):")
    for a in range(N):
        add(f"    S{a} = S[{a}]")
    add("    for bb in prange(bin_ptr.size - 1):")
    add("        b = numba.get_thread_id()" if not owner else "        b = 0")
    add(f"        w = np.empty({R})")
    add(f"        t = np.empty({R})")
    add(f"        u = np.empty({R})")
    add(f"        e = np.empty({R})")
    add("        last_i = -1")
    add("        for cc in range(bin_ptr[bb], bin_ptr[bb + 1]):")
    add("          c = bins[cc]")
    add("          for s in range(starts[c], starts[c + 1]):")
    add("            q0 = p[s]")
    add("            q1 = p[s + 1]")
    add("            if q0 == q1:")
    add("                continue")
    add("            rem = h[s] if h.size > 0 else s")
    for a in range(P - 1, 0, -1):
        add(f"            i{a} = rem % S{a}")
        add(f"            rem = rem // S{a}")
    add("            i0 = rem")
    # w: the product of the row factors in row_w (none: 1)
    if row_w:
        add(f"            for r in range({R}):")
        add("                w[r] = " + " * ".join(f"F{a}[i{a}, r]" for a in row_w))
    wr = " * w[r]" if row_w else ""
    if kind == _ROWS:
        add(f"            for r in range({R}):")
        add("                t[r] = 0.0")
    if Q >= 2 and not runs:
        col_f = [a for a in range(P, N) if a != m]
        add("            for q in range(q0, q1):")
        add("                xv = x[0] if iso else x[q]")
        add("                rem = ci[q]")
        for a in range(L, P, -1):
            add(f"                j{a} = rem % S{a}")
            add(f"                rem = rem // S{a}")
        add(f"                j{P} = rem")
        prod = "".join(f" * F{a}[j{a}, r]" for a in col_f)
        add(f"                for r in range({R}):")
        if kind == _ROWS:
            add(f"                    t[r] += xv{prod}")
        else:
            add(f"                    {o}j{m}, r] += xv{prod}{wr}")
            add(f"                {tt}j{m}] = True")
    elif Q == 1:
        if kind == _ROWS:
            add("            for q in range(q0, q1):")
            add("                xv = x[0] if iso else x[q]")
            add("                k = ci[q]")
            add(f"                for r in range({R}):")
            add(f"                    t[r] += xv * F{L}[k, r]")
        else:  # the mode is the only column axis
            add("            for q in range(q0, q1):")
            add("                xv = x[0] if iso else x[q]")
            add("                k = ci[q]")
            add(f"                for r in range({R}):")
            add(f"                    {o}k, r] += xv" + (" * w[r]" if row_w else ""))
            add(f"                {tt}k] = True")
    else:
        add("            q = q0")
        add("            while q < q1:")
        add(f"                pre = ci[q] // S{L}")
        add(f"                base = pre * S{L}")
        add(f"                lim = base + S{L}")
        # decode the prefix into j{P}..j{L-1}
        for a in range(L - 1, P, -1):
            add(f"                j{a} = pre % S{a}")
            add(f"                pre = pre // S{a}")
        add(f"                j{P} = pre")
        if kind == _LAST:
            add(f"                for r in range({R}):")
            add("                    e[r] = 1.0" + wr + "".join(f" * F{a}[j{a}, r]" for a in pre_f))
        else:
            add(f"                for r in range({R}):")
            add("                    u[r] = 0.0")
        # the run of entries with this prefix
        # Columns are sorted within a row: the run ends at the first one past the prefix
        add("                while q < q1:")
        add("                    col = ci[q]")
        add("                    if col >= lim:")
        add("                        break")
        add("                    k = col - base")
        add("                    xv = x[0] if iso else x[q]")
        add(f"                    for r in range({R}):")
        if kind == _LAST:
            add(f"                        {o}k, r] += xv * e[r]")
            add(f"                    {tt}k] = True")
        else:
            add(f"                        u[r] += xv * F{L}[k, r]")
        add("                    q += 1")
        pf = "".join(f" * F{a}[j{a}, r]" for a in pre_f)
        if kind == _ROWS:
            add(f"                for r in range({R}):")
            add(f"                    t[r] += u[r]{pf}")
        elif kind == _PREFIX:
            add(f"                for r in range({R}):")
            add(f"                    {o}j{m}, r] += u[r]{pf}{wr}")
            add(f"                {tt}j{m}] = True")
    if kind == _ROWS:
        add(f"            for r in range({R}):")
        add(f"                {o}i{m}, r] += t[r]{wr}")
        if owner:
            add(f"            if i{m} != last_i:")
            # Once per output row: a write per fiber would share cache lines across threads
            add(f"                touched[i{m}] = True")
            add(f"                last_i = i{m}")
        else:
            add(f"            touched[b, i{m}] = True")
    return "\n".join(lines) + "\n"


def _kernel_for(R, P, Q, kind, mpos, owner, runs):
    key = ("mttkrp", R, P, Q, kind, mpos, owner, runs)
    if (k := _compiled.get(key)) is None:
        ns = {"np": np, "numba": numba, "prange": prange}
        exec(_kernel_source(R, P, Q, kind, mpos, owner, runs), ns)  # noqa: S102 (our own source)
        k = _compiled[key] = njit(parallel=True)(ns["kernel"])
    return k


def mttkrp_nd(arrays, row_sizes, col_sizes, mode_pos, F, n_out, nthreads, max_buffer):
    """``(out, touched)``: the MTTKRP rows (dense, ``n_out x R``) and which of them any entry
    reached; or None when it needs per-thread buffers larger than ``max_buffer`` entries.

    ``arrays`` is ``(p, h, ci, x, iso)`` from the Container (h empty when not hypersparse);
    ``row_sizes`` and ``col_sizes`` are the sizes of the row and column group axes (one or more
    each); ``mode_pos`` is the mode's position in rows + columns; ``F`` holds the dense factors
    in that same order (the mode's entry is ignored); ``n_out`` is the mode's size.
    """
    p, h, ci, x, iso = arrays
    P, Q = len(row_sizes), len(col_sizes)
    R = next(f.shape[1] for k, f in enumerate(F) if k != mode_pos)
    nthreads = max(
        1, min(nthreads or numba.config.NUMBA_NUM_THREADS, numba.config.NUMBA_NUM_THREADS)
    )
    if mode_pos < P:
        kind, mpos = _ROWS, mode_pos
    elif mode_pos < P + Q - 1:
        kind, mpos = _PREFIX, mode_pos - P
    else:
        kind, mpos = _LAST, mode_pos - P
    owner = kind == _ROWS and mpos == 0
    nbuf = 1 if owner else nthreads
    if not owner and nbuf * n_out * R > max_buffer:
        return None
    # Runs of equal leading column axes pay off when they hold several entries; expected
    # length under uniform occupancy, from the entries per row and the prefix positions
    runs = False
    if Q >= 2:
        nrows = max(1, h.size if h.size > 0 else p.size - 1)
        per_row = (p[-1] - p[0]) / nrows
        positions = float(np.prod(np.array(col_sizes[:-1], np.float64)))
        distinct = positions * -np.expm1(-per_row / positions)
        runs = per_row >= 4 * max(distinct, 1.0)
    if _FORCE_RUNS is not None:
        runs = _FORCE_RUNS
    k = _kernel_for(R, P, Q, kind, mpos, owner, runs)
    rsz = np.array(row_sizes, np.int64)
    csz = np.array(col_sizes, np.int64)
    if owner:
        inner = int(np.prod(rsz[1:]))
        if h.size > 0:
            change = np.flatnonzero(np.diff(h.astype(np.int64) // inner)) + 1
        else:
            change = np.arange(inner, p.size - 1, inner, dtype=np.int64)
        starts = _starts(p, _NCHUNKS, change)
    else:
        starts = _starts(p, 8 * nthreads)
    bin_ptr, bins = _bins(p, starts, nthreads)
    out = np.zeros((n_out, R) if owner else (nbuf, n_out, R))
    touched = np.zeros(n_out if owner else (nbuf, n_out), np.bool_)
    prev = numba.get_num_threads()
    numba.set_num_threads(nthreads)
    try:
        k(p, h, ci, x, iso, np.concatenate([rsz, csz]), *F, out, touched, starts, bin_ptr, bins)
    finally:
        numba.set_num_threads(prev)
    if owner:
        return out, touched
    total = np.empty((n_out, R))
    any_touched = np.empty(n_out, np.bool_)
    _sum_buffers(out, touched, total, any_touched)
    return total, any_touched


@njit(parallel=True, cache=True)
def _sum_buffers(out, touched, total, any_touched):
    # In parallel over output rows: numpy's sum of Enron's 8 x 244268 x 64 buffers took 27 ms
    # on one thread, this 8
    for i in prange(total.shape[0]):
        for r in range(total.shape[1]):
            total[i, r] = out[0, i, r]
        t = touched[0, i]
        for b in range(1, out.shape[0]):
            for r in range(total.shape[1]):
                total[i, r] += out[b, i, r]
            t = t or touched[b, i]
        any_touched[i] = t

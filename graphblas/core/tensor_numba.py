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

import hashlib
import os
import sys
import types
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

from . import tensor_census as census

_NCHUNKS = 64
# Kernels are compiled per R and per layout: with R a constant the inner loops unroll and
# vectorize, 1.7-1.9x faster than with R read at run time (u7 mode 0 4.3 vs 6.4 ms at 8
# threads, 22 vs 39 at 1; NELL-2 30 vs 49, 161 vs 301). About 0.2-0.4 s to compile each, once
# per process.
_compiled = {}
# Generated kernels are written to files here, so Numba can cache their compiled code on disk:
# a compile costs 0.2-0.6 s each, per process otherwise. Set the variable to "" to turn it off.
_CACHE_ENV = "GRAPHBLAS_KERNEL_CACHE"


def _cache_dir():
    path = os.environ.get(_CACHE_ENV)
    if path == "":
        return None
    if path is None:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        path = Path(base) / "python-graphblas" / "tensor-kernels"
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path if os.access(path, os.W_OK) else None


def _load_cached(path, name, text):
    """The kernel of ``text`` run as the module ``name`` from the file ``path``, which is
    written first if missing; None if the file holds anything else (corrupt, tampered with, or
    another kernel whose digest shares the prefix), which is then left as it is. The verified
    ``text`` is what runs, not a second read of the file.
    """
    data = text.encode()
    if path.exists():
        if path.read_bytes() != data:
            return None
    else:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            # Another process may have written the file meanwhile: on Windows the rename then
            # fails while that process has the file open. Its file is as good as ours.
            if not (path.exists() and path.read_bytes() == data):
                raise
    module = types.ModuleType(name)
    module.__file__ = str(path)
    # (registered: Numba imports the module by name when it loads cached code)
    sys.modules[name] = module
    try:
        exec(compile(text, str(path), "exec"), module.__dict__)  # noqa: S102 (our own source)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module.kernel


def _compile(source):
    """Compile generated kernel source (a function ``kernel``) with ``njit(parallel=True)``:
    from a file named by its hash in the cache directory, with Numba's disk cache, else (no
    writable directory, a file that is not this source, or any failure there) in memory. The
    file is written once, through a temporary file and a rename, so concurrent processes see
    either nothing or all of it; Numba's cache index checks its own version and the file's
    stamp. Bounds checks (tests) are part of the source, so they get their own file.
    """
    check = ", boundscheck=True" if numba.config.BOUNDSCHECK else ""
    text = "import numba\nimport numpy as np\nfrom numba import njit, prange\n\n" + source.replace(
        "def kernel(", f"@njit(parallel=True, cache=True{check})\ndef kernel(", 1
    )
    digest = hashlib.sha256(text.encode()).hexdigest()[:32]
    if (cache := _cache_dir()) is not None:
        existed = census.on and (cache / f"k_{digest}.py").exists()
        try:
            kernel = _load_cached(cache / f"k_{digest}.py", f"_graphblas_kernel_{digest}", text)
        except Exception:
            # Any trouble with the file (a read-only or full disk): compile in memory instead
            kernel = None
        if kernel is not None:
            if census.on:
                census.note("kernel.cache", "disk: file found" if existed else "disk: written")
            return kernel
    if census.on:
        census.note("kernel.cache", "in memory" + (" (no cache directory)" if cache is None
                    else " (file refused)"))  # fmt: skip
    ns = {}
    exec(text.replace("cache=True", "cache=False"), ns)  # noqa: S102 (our own source)
    return ns["kernel"]


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
_FORCE_COLBLOCK = None  # likewise for column blocks against per-thread buffers
_COLBLOCK_BYTES = 2**22  # a thread's buffer past this (4 MiB) leaves its cache


def _col_blocks(p, ci, csz, nblocks):
    """Column-index bounds of ``nblocks`` blocks of about equal entries, on boundaries of the
    first column axis (quantiles of a sample of the column indices), in ci's dtype.
    """
    stride = int(np.prod(np.array(csz[1:], np.int64)))
    nnz = int(p[-1])
    sample = np.sort(ci[np.random.default_rng(0).integers(0, max(nnz, 1), 2**16)] // stride)
    cuts = sample[(np.arange(1, nblocks) * sample.size) // nblocks] if nnz else []
    k = np.unique(np.concatenate([[0], cuts, [int(csz[0])]]).astype(np.int64))
    return (k * stride).astype(ci.dtype)


def _kernel_source(R, P, Q, kind, mpos, owner, runs, colblock=False):
    """Python source of the kernel for P row axes and Q column axes, the mode being the row
    axis ``mpos`` (``_ROWS``), the column axis ``mpos`` other than the last (``_PREFIX``), or
    the last column axis (``_LAST``). ``owner``: the mode is the first row axis and chunks own
    their output rows. ``colblock``: the mode is the first column axis, and each thread owns a
    block of its values, so of the column indices: it reads every row, but only the entries
    in its block (found by binary search). Otherwise each thread adds into its own buffer.

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
    own = owner or colblock
    o = "out[" if own else "out[b, "  # then the output row, r
    tt = "touched[" if own else "touched[b, "
    lines = []
    add = lines.append
    args = ", ".join(f"F{a}" for a in range(N))
    add(f"def kernel(p, h, ci, x, iso, S, {args}, out, touched, starts, bin_ptr, bins):")
    for a in range(N):
        add(f"    S{a} = S[{a}]")
    add("    for bb in prange(bin_ptr.size - 1):")
    add("        b = numba.get_thread_id()" if not own else "        b = 0")
    add(f"        w = np.empty({R})")
    add(f"        t = np.empty({R})")
    add(f"        u = np.empty({R})")
    add(f"        e = np.empty({R})")
    add("        last_i = -1")
    if colblock:
        # bin_ptr holds the column-index bounds of the blocks, one per iteration
        add("        lo = bin_ptr[bb]")
        add("        hi = bin_ptr[bb + 1]")
        add("        for cc in range(1):")
        add("          for s in range(p.size - 1):")
        add("            q0 = p[s]")
        add("            q1 = p[s + 1]")
        add("            if q0 == q1:")
        add("                continue")
        add("            q1 = q0 + np.searchsorted(ci[q0:q1], hi)")
        add("            q0 = q0 + np.searchsorted(ci[q0:q1], lo)")
        add("            if q0 == q1:")
        add("                continue")
    else:
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


def _kernel_for(R, P, Q, kind, mpos, owner, runs, colblock):
    key = ("mttkrp", R, P, Q, kind, mpos, owner, runs, colblock)
    if census.on:
        census.note("kernel.process", "hit" if key in _compiled else "compile", kernel="mttkrp")
    if (k := _compiled.get(key)) is None:
        k = _compiled[key] = _compile(_kernel_source(R, P, Q, kind, mpos, owner, runs, colblock))
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
    # The first column axis: its values partition the column indices, so threads can own
    # blocks of them instead of keeping buffers (each thread then binary-searches every row).
    # Worth it when a thread's buffer leaves its cache and takes many updates per row:
    # NELL-2 mode 2 (28818 rows, 2670 entries each) at 8 threads, R = 32 / 64: 114 / 150 ms
    # against 140 / 409 with buffers; R = 16 (3.5 MiB buffers) 78 against 42. Enron mode 2
    # (244268 rows, 222 each) gains nothing (R = 64: 133 against 148; R = 16: 65 against 46),
    # u7 mode 2 loses (1e6 rows to search, 27 against 4.8 ms). And when the buffers would be
    # too large altogether, instead of a relayout.
    first_col = kind == _LAST and Q == 1 or kind == _PREFIX and mpos == 0
    nnz = int(p[-1]) - int(p[0])
    colblock = (
        first_col
        and nthreads > 1
        and (
            n_out * R * 8 > _COLBLOCK_BYTES
            and nnz >= 1000 * n_out
            or nthreads * n_out * R > max_buffer
        )
    )
    if _FORCE_COLBLOCK is not None:
        colblock = first_col and _FORCE_COLBLOCK
    own = owner or colblock
    if not own and nthreads * n_out * R > max_buffer:
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
    k = _kernel_for(R, P, Q, kind, mpos, owner, runs, colblock)
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
    if colblock:
        bin_ptr, bins = _col_blocks(p, ci, csz, nthreads), starts[:0]
    else:
        bin_ptr, bins = _bins(p, starts, nthreads)
    out = np.zeros((n_out, R) if own else (nthreads, n_out, R))
    touched = np.zeros(n_out if own else (nthreads, n_out), np.bool_)
    prev = numba.get_num_threads()
    numba.set_num_threads(nthreads)
    try:
        k(p, h, ci, x, iso, np.concatenate([rsz, csz]), *F, out, touched, starts, bin_ptr, bins)
    finally:
        numba.set_num_threads(prev)
    if own:
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


# ---------------------------------------------------------------------------------------------
# TTM: the tensor times a matrix along the last column axis, read in place


def _ttm_source(R, multi, dense):
    """The TTM kernel for R output columns. Each output fiber is a row of the stored Matrix
    (one column axis), or with ``multi`` a run of the row's entries sharing the leading
    column axes (entries are sorted by column); its R values are the sum of ``x * U[k, :]``
    over its entries, k being the last column axis. With ``dense``, fiber values go to row
    ``g`` (the fiber's output row index) of ``V`` and ``B[g]`` is set (a bitmap); otherwise
    to ``V[f]``, f numbering the fibers from ``offs[s]`` for row slot s, with ``hout[f] = g``
    and the column indices ``cols[f * R + r] = r`` (hypersparse rows).
    """
    return f"R = {R}\nmulti = {multi}\ndense = {dense}\n\n" + _TTM_KERNEL


_TTM_KERNEL = """\
def kernel(p, h, ci, x, iso, nlast, npre, U, offs, hout, cols, V, B, starts, bin_ptr, bins):
    for bb in prange(bin_ptr.size - 1):
        t = np.empty(R)
        for cc in range(bin_ptr[bb], bin_ptr[bb + 1]):
            c = bins[cc]
            for s in range(starts[c], starts[c + 1]):
                q0 = p[s]
                q1 = p[s + 1]
                if q0 == q1:
                    continue
                row = np.int64(h[s]) if h.size > 0 else np.int64(s)
                f = offs[s]
                q = q0
                while q < q1:
                    base = 0
                    lim = nlast
                    pre = 0
                    if multi:
                        pre = np.int64(ci[q]) // nlast
                        base = pre * nlast
                        lim = base + nlast
                    for r in range(R):
                        t[r] = 0.0
                    while q < q1:
                        col = np.int64(ci[q])
                        if multi and col >= lim:
                            break
                        xv = x[0] if iso else x[q]
                        k = col - base
                        for r in range(R):
                            t[r] += xv * U[k, r]
                        q += 1
                    g = row * npre + pre
                    if dense:
                        for r in range(R):
                            V[g, r] = t[r]
                        if B.shape[0] > 0:  # (no bitmap when every row exists)
                            for r in range(R):
                                B[g, r] = 1
                    else:
                        for r in range(R):
                            V[f, r] = t[r]
                            cols[f * R + r] = r
                        hout[f] = g
                    f += 1
"""


@njit(parallel=True, cache=True)
def _count_runs(p, ci, nlast, counts):
    # The runs of equal leading column axes in each row: the output fibers of a TTM
    for s in prange(p.size - 1):
        n = 0
        last = -1
        for q in range(p[s], p[s + 1]):
            pre = np.int64(ci[q]) // nlast
            if pre != last:
                n += 1
                last = pre
        counts[s] = n


def ttm(arrays, col_sizes, U, nrows_out, nthreads):
    """The tensor times ``U`` (dense, ``K x R``) along the last column axis of the stored
    arrays, as the parts of a Matrix with ``nrows_out`` rows (the stored row, then the leading
    column axes, raveled) and R columns: ``("full", V)``, ``("bitmap", V, B, nvals)`` when at
    least half the rows exist, else ``("hyper", hout, indptr, cols, V)``. Each part is a 1-d
    array that owns its data, ready to hand to GraphBLAS. ``arrays`` and ``col_sizes`` as for
    ``mttkrp_nd``.
    """
    p, h, ci, x, iso = arrays
    R = U.shape[1]
    nlast = int(col_sizes[-1])
    npre = int(np.prod(np.array(col_sizes[:-1], np.int64)))
    multi = len(col_sizes) > 1
    nthreads = max(
        1, min(nthreads or numba.config.NUMBA_NUM_THREADS, numba.config.NUMBA_NUM_THREADS)
    )
    prev = numba.get_num_threads()
    numba.set_num_threads(nthreads)
    try:
        if multi:
            counts = np.empty(p.size - 1, np.int64)
            _count_runs(p, ci, np.int64(nlast), counts)
        else:
            counts = (np.diff(p) > 0).astype(np.int64)
        offs = np.zeros(p.size, np.int64)
        np.cumsum(counts, out=offs[1:])
        nfib = int(offs[-1])
        # SuiteSparse would hold rows this full as a bitmap (u7: 999,956 of 1e6 fibers)
        dense = 2 * nfib >= nrows_out
        key = ("ttm", R, multi, dense)
        if census.on:
            census.note("kernel.process", "hit" if key in _compiled else "compile", kernel="ttm")
        if (k := _compiled.get(key)) is None:
            k = _compiled[key] = _compile(_ttm_source(R, multi, dense))
        # 1-d owners (a reshaped view would be copied when handed over: 114 ms for 265e6)
        if dense:
            Vflat = np.empty(nrows_out * R)
            Bflat = np.zeros(nrows_out * R if nfib < nrows_out else 0, np.int8)
            hout = np.empty(0, np.int64)
            cols = np.empty(0, np.uint32)
            V = Vflat.reshape(nrows_out, R)
            B = Bflat.reshape(-1, R)
        else:
            Vflat = np.empty(nfib * R)
            Bflat = np.empty(0, np.int8)
            hout = np.empty(nfib, np.int64)
            cols = np.empty(nfib * R, np.uint32)  # (32-bit column indices: R < 2^31)
            V = Vflat.reshape(nfib, R)
            B = Bflat.reshape(0, R)
        starts = _starts(p, 8 * nthreads)
        bin_ptr, bins = _bins(p, starts, nthreads)
        k(p, h, ci, x, iso, np.int64(nlast), np.int64(npre), U, offs, hout, cols, V, B, starts,
          bin_ptr, bins)  # fmt: skip
    finally:
        numba.set_num_threads(prev)
    if dense and nfib == nrows_out:
        return ("full", Vflat)
    if dense:
        return ("bitmap", Vflat, Bflat, nfib * R)
    return ("hyper", hout, np.arange(0, (nfib + 1) * R, R, dtype=np.uint64), cols, Vflat)


@njit(parallel=True, cache=True)
def lookup(p, h, ci, x, iso, rows, cols, vals, found):
    """For each point k, the entry at (rows[k], cols[k]) of a sparse or hypersparse by-row
    Matrix's arrays (columns sorted within rows): ``found[k]`` and ``vals[k]``. Binary
    searches: the row in ``h`` (hypersparse), then the column in the row's segment.
    """
    for k in prange(rows.size):
        r = np.int64(rows[k])
        s = r
        if h.size > 0:
            lo = np.int64(0)
            hi = np.int64(h.size)
            while lo < hi:
                mid = (lo + hi) // 2
                if np.int64(h[mid]) < r:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == h.size or np.int64(h[lo]) != r:
                found[k] = False
                continue
            s = lo
        c = np.int64(cols[k])
        lo = np.int64(p[s])
        end = np.int64(p[s + 1])
        hi = end
        while lo < hi:
            mid = (lo + hi) // 2
            if np.int64(ci[mid]) < c:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and np.int64(ci[lo]) == c:
            found[k] = True
            vals[k] = x[0] if iso else x[lo]
        else:
            found[k] = False

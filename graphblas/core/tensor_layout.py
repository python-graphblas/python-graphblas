"""Physical layouts of a Tensor and the planner that moves data between them (experimental).

A Tensor keeps its entries in one Matrix held by row. Its ``Layout`` says which logical axes
are raveled (C order) into the Matrix rows and which into the columns. The planner works on
physical states ``(seq, split)``: the axis sequence ``rows + cols`` and ``len(rows)``.

A relayout is a sort by the new major key, so the planner builds it like an LSD radix sort
from cheap SuiteSparse primitives:

- RS: ``ss.reshape`` in storage order (moves the split, keeps the sequence)
- T: a materialized transpose (swaps the groups); cheap only into few output rows
- PERM_R: any permutation of the row axes as one ``P @ M``
- REBUILD: ``to_coo``, new indices, ``from_coo`` (any state to any state)

and runs Dijkstra over the states with the measured per-entry costs. Plans and the shape-only
helper matrices (permutations and selectors) are cached.
"""

import heapq
import itertools
import math
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from .. import semiring as _semiring
from ..dtypes import BOOL, UINT32, UINT64, lookup_dtype
from ..exceptions import _error_code_lookup
from . import NULL, ffi, lib
from . import tensor_census as census
from .matrix import Matrix
from .vector import Vector

# GraphBLAS dimensions go up to GrB_INDEX_MAX + 1 = 2^60.
MAX_GROUP = 2**60
# The planner searches states up to this many axes; above it a relayout is one REBUILD.
MAX_PLANNED_NDIM = 5


class Layout(NamedTuple):
    """Which logical axes are raveled into the rows and the columns of the stored Matrix.

    Each group is raveled in C order in the order given, so ``Layout((1, 0), (2,))`` stores
    ``T[i, j, k]`` at row ``j * shape[0] + i`` and column ``k``.
    """

    rows: tuple
    cols: tuple

    @property
    def seq(self):
        return self.rows + self.cols

    @property
    def state(self):
        return (self.rows + self.cols, len(self.rows))

    @classmethod
    def from_state(cls, state):
        seq, p = state
        return cls(seq[:p], seq[p:])

    def _short(self):
        rows = ",".join(map(str, self.rows))
        cols = ",".join(map(str, self.cols))
        return f"({rows}|{cols})"


def prod(shape, axes):
    return math.prod(shape[a] for a in axes)


def as_layout(layout, shape):
    """Validate ``layout`` (a Layout or a pair of axis sequences) for ``shape``."""
    try:
        rows, cols = layout
        rows = tuple(int(a) for a in rows)
        cols = tuple(int(a) for a in cols)
    except (TypeError, ValueError):
        raise TypeError(
            f"layout must be a Layout or a pair of axis sequences (rows, cols); got {layout!r}"
        ) from None
    ndim = len(shape)
    if sorted(rows + cols) != list(range(ndim)):
        raise ValueError(
            f"layout {layout!r} must partition the axes {tuple(range(ndim))} "
            "into rows and cols, each axis once"
        )
    for axes in (rows, cols):
        if prod(shape, axes) > MAX_GROUP:
            raise ValueError(
                f"axes {axes} of shape {shape} have {prod(shape, axes)} positions, more than "
                f"one Matrix dimension can hold (2**60); choose another layout"
            )
    return Layout(rows, cols)


def _valid_split(shape, seq, p):
    return prod(shape, seq[:p]) <= MAX_GROUP and prod(shape, seq[p:]) <= MAX_GROUP


# Rows of at least this many entries each pay for a row group whatever their count (thinner
# sparse rows cost about 2x in ewise: u6 at 1.6 per row, X + Y 3.9 ms against 1.5). The default
# split, the split chosen from the data and _healthy's all use it.
FULL_ROW_ENTRIES = 4


def full_split(shape, seq, nnz):
    """The largest p in ``1..len(seq)-1`` whose rows (``seq[:p]``) hold FULL_ROW_ENTRIES
    entries each even if all are nonempty; else 1.
    """
    for q in range(len(seq) - 1, 0, -1):
        if prod(shape, seq[:q]) * FULL_ROW_ENTRIES <= nnz:
            return q
    return 1


def default_layout(shape, nnz=None):
    """The layout for new data: identity sequence, split by expected entries per row.

    ``p`` is ``full_split``: the largest split in ``1..ndim-1`` with ``prod(shape[:p]) <= nnz /
    FULL_ROW_ENTRIES``, else 1; with no ``nnz`` (an empty Tensor) ``p = ndim - 1``.
    A 1-d Tensor is stored as one row (the bytes of a Vector); a 2-d one as the plain Matrix.
    """
    ndim = len(shape)
    seq = tuple(range(ndim))
    if ndim == 0:
        return Layout((), ())
    if ndim == 1:
        return Layout((), (0,))
    if ndim == 2:
        return Layout((0,), (1,))
    p = ndim - 1 if nnz is None else full_split(shape, seq, nnz)
    if _valid_split(shape, seq, p):
        return Layout(seq[:p], seq[p:])
    for q in range(1, ndim):
        if _valid_split(shape, seq, q):
            return Layout(seq[:q], seq[q:])
    raise ValueError(f"shape {shape} has no split whose groups each fit in 2**60")


# ---------------------------------------------------------------------------------------------
# Raveling


def ravel(coords, shape, axes, n=None):
    """C-order linear index of ``axes`` (int64; every group is < 2^60)."""
    if not axes:
        if n is None:
            n = coords[0].size if coords else 0
        return np.zeros(n, np.int64)
    idx = coords[axes[0]].astype(np.int64, copy=True)
    for a in axes[1:]:
        idx *= shape[a]
        idx += coords[a]
    return idx


def unravel(idx, shape, axes, out):
    """Fill ``out[a]`` for each axis in ``axes`` from the linear index ``idx``."""
    if not axes:
        return
    if len(axes) == 1:
        out[axes[0]] = idx
        return
    for a in reversed(axes[1:]):
        idx, out[a] = np.divmod(idx, shape[a])
    out[axes[0]] = idx


def coords_of(M, shape, state, *, values=True, sort=False):
    """Logical coordinates (a list of int64 arrays) and values of the entries of ``M``."""
    seq, p = state
    r, c, v = M.to_coo(values=values, sort=sort)
    coords = [None] * len(shape)
    unravel(r.view(np.int64), shape, seq[:p], coords)
    unravel(c.view(np.int64), shape, seq[p:], coords)
    return coords, v


# ---------------------------------------------------------------------------------------------
# Matrices held by row, and the Container calls


def _check(rc, what):
    # Direct `lib` calls skip python-graphblas's `call`, so check their status here.
    if rc != 0:  # pragma: no cover (safety)
        exc = _error_code_lookup.get(rc, RuntimeError)
        raise exc(f"{what} failed with GrB_Info {rc}")


def _global_by_col():
    val = ffi.new("int32_t*")
    _check(lib.GxB_Global_Option_get_INT32(lib.GxB_FORMAT, val), "GxB_Global_Option_get")
    return val[0] == lib.GxB_BY_COL


def new_matrix(dtype, nrows, ncols, *, name=None):
    """An empty Matrix held by row whatever the global format is.

    SuiteSparse creates every n x 1 Matrix by column (as a Vector), whatever the global
    format; a per-Matrix setting on the empty Matrix sticks through later updates.
    """
    M = Matrix(dtype, nrows, ncols, name=name)
    if ncols == 1 or _global_by_col():
        M.ss.config["format"] = "by_row"
    return M


def from_coo(rows, cols, values, dtype, nrows, ncols, *, dup_op=None, name=None):
    """``Matrix.from_coo`` held by row. Under a by-column global format, build the transpose
    by column and flip it, which costs nothing extra.
    """
    if _global_by_col():
        M = Matrix.from_coo(
            cols, rows, values, dtype, nrows=ncols, ncols=nrows, dup_op=dup_op, name=name
        )
        flip_orientation(M)
    else:
        M = Matrix.from_coo(
            rows, cols, values, dtype, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    if not is_by_row(M):
        # An n x 1 result (see new_matrix); converting it costs O(nvals).
        M.ss.config["format"] = "by_row"
    return M


def is_by_row(M):
    return M.ss.orientation == "rowwise"


def stored_format(M):
    """``M.ss.format`` once ``M`` is finished. Every format test that decides how to read or
    change the Container goes through here: ``ss.format`` does not finish pending work, and
    finishing it (``_Unloaded`` does) can change the format, hypersparse to sparse or bitmap
    after pending tuples, sparse to hypersparse after zombies. ``nvals`` finishes pending
    tuples but not zombies.
    """
    M.wait()
    return M.ss.format


class _Unloaded:
    """Context manager: ``M`` unloaded into a GxB_Container, loaded back on exit."""

    def __init__(self, M):
        self.M = M

    def __enter__(self):
        # Pending work (zombies, pending tuples, jumbled rows) must not be unloaded.
        self.M.wait()
        self._Cp = ffi.new("GxB_Container*")
        _check(lib.GxB_Container_new(self._Cp), "GxB_Container_new")
        rc = lib.GxB_unload_Matrix_into_Container(self.M.gb_obj[0], self._Cp[0], NULL)
        if rc != 0:  # pragma: no cover (safety)
            lib.GxB_Container_free(self._Cp)
            _check(rc, "GxB_unload_Matrix_into_Container")
        return self._Cp[0]

    def __exit__(self, *exc):
        rc = lib.GxB_load_Matrix_from_Container(self.M.gb_obj[0], self._Cp[0], NULL)
        lib.GxB_Container_free(self._Cp)
        _check(rc, "GxB_load_Matrix_from_Container")


def matrix_from_arrays(dtype, nrows, ncols, fmt, *, x, h=None, p=None, i=None, b=None, nvals):
    """A by-row Matrix made of these arrays through the Container, which takes them over in
    O(1) (no copy, no check: an import scans the values for iso, 210 ms for 265e6). ``fmt``
    is "hypercsr" (h, p, i, x), "bitmapr" (b, x) or "fullr" (x); each array must come from an
    allocator GraphBLAS can free (``numpy.empty`` by default) and is a read-only view after.
    """
    from .utils import _give_buffer

    M = new_matrix(dtype, nrows, ncols)
    with _Unloaded(M) as cont:
        for vec, arr in ((cont.p, p), (cont.h, h), (cont.i, i), (cont.b, b), (cont.x, x)):
            if arr is None:
                continue
            arr = np.ascontiguousarray(arr)
            gtype = dtype if vec == cont.x else lookup_dtype(arr.dtype)
            with _give_buffer(arr) as (ptr, nbytes, arena):
                _check(
                    lib.GxB_Vector_load(
                        vec, ptr, gtype.gb_obj, arr.size, nbytes, lib.GrB_DEFAULT + arena, NULL
                    ),
                    "GxB_Vector_load",
                )
        cont.format = {
            "hypercsr": lib.GxB_HYPERSPARSE, "bitmapr": lib.GxB_BITMAP, "fullr": lib.GxB_FULL
        }[fmt]  # fmt: skip
        cont.nvals = nvals
        cont.nrows_nonempty = h.size if h is not None else -1
        cont.ncols_nonempty = -1
        cont.iso = False
        cont.jumbled = False
    return M


def new_vector_handle(like):
    """A new empty GrB_Vector handle (of the type of ``like``), to put in a Container slot
    whose vector is taken out.
    """
    t = ffi.new("GrB_Type*")
    _check(lib.GxB_Vector_type(t, like), "GxB_Vector_type")
    v = ffi.new("GrB_Vector*")
    _check(lib.GrB_Vector_new(v, t[0], 0), "GrB_Vector_new")
    return v[0]


def give_rows(M, h, nrows):
    """Make the ``F``-row Matrix ``M`` hypersparse with ``nrows`` rows, its row f becoming
    row ``h[f]`` (``h`` a GrB_Vector of F sorted row ids, taken over), in place.
    """
    if stored_format(M) != "csr":
        M.ss.config["sparsity_control"] = "sparse"  # (from bitmap or full: O(entries))
    with _Unloaded(M) as cont:
        old = ffi.new("GrB_Vector*", cont.h)
        _check(lib.GrB_Vector_free(old), "GrB_Vector_free")
        cont.h = h
        cont.nrows = nrows
        cont.format = lib.GxB_HYPERSPARSE
    M._nrows = nrows
    M.ss.config["sparsity_control"] = "auto"


def flip_orientation(M):
    """In place, O(1): an (n x m) Matrix held by column becomes the (m x n) Matrix of its
    transpose held by row (or the reverse), sharing the same arrays.
    """
    with _Unloaded(M) as cont:
        cont.nrows, cont.ncols = cont.ncols, cont.nrows
        cont.nrows_nonempty, cont.ncols_nonempty = cont.ncols_nonempty, cont.nrows_nonempty
        if cont.orientation == lib.GrB_COLMAJOR:
            cont.orientation = lib.GrB_ROWMAJOR
        else:
            cont.orientation = lib.GrB_COLMAJOR
    M._nrows, M._ncols = M._ncols, M._nrows
    return M


def nonempty_rows(M):
    """The number of nonempty rows of a hypersparse Matrix (O(1)); ``nrows`` otherwise."""
    if stored_format(M) != "hypercsr":
        return M._nrows
    n = ffi.new("GrB_Index*")
    with _Unloaded(M) as cont:
        rc = lib.GrB_Vector_size(n, cont.h)
    _check(rc, "GrB_Vector_size")
    return n[0]


def nonempty_row_ids(M):
    """The ids of the nonempty rows of ``M``, sorted, as int64: from the Container's ``h``
    (hypersparse) or ``p`` (sparse) in O(rows), without looking at the entries.
    """
    fmt = stored_format(M)
    if fmt in {"hypercsr", "csr"}:
        with _Unloaded(M) as cont, _VectorArray(cont.h if fmt == "hypercsr" else cont.p) as v:
            arr = np.empty(0, np.int64) if v.array is None else v.array.astype(np.int64)
        if fmt == "hypercsr":
            return arr
        return np.flatnonzero(np.diff(arr))
    r = M.to_coo(columns=False, values=False)[0].view(np.int64)
    return r[np.r_[True, r[1:] != r[:-1]]] if r.size else r


def count_nonempty_rows(M):
    """The exact number of nonempty rows: O(1) for hypersparse, O(nrows) for sparse."""
    fmt = stored_format(M)
    if fmt == "hypercsr":
        return nonempty_rows(M)
    if fmt == "csr":
        with _Unloaded(M) as cont, _VectorArray(cont.p) as v:
            return 0 if v.array is None else int(np.count_nonzero(np.diff(v.array)))
    return M._nrows


def rows_selector(M, shape, group, kept_pos, weights=None):
    """``S`` with ``S @ M`` reducing the axes of the row ``group`` not in ``kept_pos``, built
    over the nonempty rows of ``M`` only, so its size follows the data rather than the
    group's positions (a hypersparse row group). ``weights`` (a dense array and a presence
    mask over one axis of the group, as ``(values, present, position)``) gives each row its
    weight instead of 1 (TTV).
    """
    h = nonempty_row_ids(M)
    gshape = tuple(shape[a] for a in group)
    co = {}
    unravel(h, gshape, tuple(range(len(gshape))), co)
    values = True
    if weights is not None:
        w, present, pos = weights
        keep = present[co[pos]]
        if not keep.all():
            h = h[keep]
            co = {k: c[keep] for k, c in co.items()}
        values = w[co[pos]]
    dst = ravel(co, gshape, kept_pos, n=h.size)
    dtype = BOOL if weights is None else lookup_dtype(values.dtype)
    return from_coo(dst, h, values, dtype, prod(gshape, kept_pos), prod(shape, group))


def steal_vector_as_row(v):
    """Turn a Vector (n x 1 by column) into a (1 x n) Matrix held by row, O(1). ``v`` is
    left empty and must not be used again.
    """
    M = Matrix._from_obj(
        ffi.new("GrB_Matrix*", ffi.cast("GrB_Matrix", v.gb_obj[0])), v.dtype, v._size, 1
    )
    v.gb_obj[0] = NULL
    return flip_orientation(M)


# ---------------------------------------------------------------------------------------------
# Small LRU caches for plans and shape-only matrices


class _LRU:
    """Least recently used entries are dropped past ``maxsize`` items or ``maxweight``."""

    def __init__(self, maxsize, maxweight=math.inf):
        self.maxsize = maxsize
        self.maxweight = maxweight
        self.weight = 0
        self.data = OrderedDict()

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            return None
        self.data.move_to_end(key)
        return item[0]

    def put(self, key, val, weight=0):
        if weight > self.maxweight:
            return
        if (old := self.data.pop(key, None)) is not None:
            self.weight -= old[1]
        self.data[key] = (val, weight)
        self.weight += weight
        while len(self.data) > self.maxsize or self.weight > self.maxweight:
            _, (_, w) = self.data.popitem(last=False)
            self.weight -= w

    def clear(self):
        self.data.clear()
        self.weight = 0


_plan_cache = _LRU(4096)
# Permutation and selector matrices depend only on the shape, so they are reused across calls.
# They hold one entry per position of a group (about 12 bytes each), so the cache is bounded
# by its total entries: 2**25 is about 400 MB.
_matrix_cache = _LRU(64, maxweight=2**25)


def clear_caches():
    _plan_cache.clear()
    _matrix_cache.clear()


# ---------------------------------------------------------------------------------------------
# Cost model: ns per stored entry, (8 threads, 1 thread). The constants
# rank plans; they are not predictions (they run 1.3-2x low).

C_RS = (0.5, 2.1)
# Plus per nonempty output row of a reshape: 1000 x 1e6 -> 1e6 x 1000 at 1e7 entries took 8.7
# / 43 ms at 8 / 1 threads, and (3000, 2e7) -> (6e6, 1e4) at 7.7e6 took 21 / 106 ms
# (2026-10-20); without this term a radix chain looked cheaper than one transpose at 1 thread.
C_RS_ROW = (4.0, 21.0)
# Into mostly empty rows (more than 4 row positions per expected nonempty row) a reshape is
# cheap: 7.1 / 25 ms for 1e7 entries into 9.5e6 of 1e8 rows (skew), 8.6 / 30 into 2.3e6 of
# 1e10 (Zipf). The output is hypersparse, and its hash of the rows, 5-10x the reshape, is built
# later and only when something needs it: a wait, or row lookups in P @ M (C_PERM_ROW); a
# transpose or a reshape does not. (C_RS_ROW was measured with a wait; for rows that fill, the
# next step pays about that much.)
C_RS_HYPER_ROW = (0.25, 0.5)
# A transpose into hypersparse rows: 7.8 / 57 ns per entry for Zipf into 1e10 rows, 10.6 / 57
# for hyper into 2.7e11, far above the T_TABLE values for 1e7 rows (measured 2026-10-20).
C_T_HYPER = (9.0, 55.0)
# REBUILD measured 11-17 / 75-85 ns per entry (NELL-2, skew, Zipf; 2026-10-20), most of it the
# builder's sort; when the new keys arrive sorted (the same axis order, a split move) the
# builder skips it: 5.9 / 9.9 ns
C_REBUILD = (14.0, 75.0)
C_REBUILD_SORTED = (6.0, 10.0)
C_PERM_ENTRY = (0.6, 2.0)
# Per nonempty row of P @ M: rows stored sparse (from a build or a transpose) cost 3-7 / 5-12 ns
# at 8 / 1 threads; rows straight from a reshape are hypersparse with no hash of the rows yet,
# which P @ M builds and looks up: 11 / 80 ns (u7 and u6, 1e6 rows, measured 2026-10-20). The
# planner tracks which states come from a reshape. Rows that are mostly empty
# (hypersparse storage, about one entry each) pay a hash lookup per row: 31 ns to 61 ns
# (skew (1e5, 1e3, 10) rows (j, i)).
C_PERM_ROW = {"sparse": (6.0, 12.0), "reshaped": (11.0, 80.0), "hyper": (50.0, 185.0)}
# Building P (numpy, one thread): per row of a full P (cached when it fits the cache), and
# per nonempty row of a P built from the nonempty rows (unravel, ravel and an unsorted build).
C_PBUILD_FULL = 13.0
# PERM_C: the new column indices through a cached map (GraphBLAS gather) or computed (numpy),
# per entry, then each row's sort, per entry and log2 of the entries per row (1e7 entries,
# measured 2026-10-20: 10 per row 7.2 / 35 ms + 12 / 95; 1e4 per row 38 / 47 ms + 46 / 346)
C_PERM_C_MAP = (0.7, 3.5)
C_PERM_C_ARITH = (3.5, 4.5)
C_PERM_C_SORT = (0.35, 2.8)
C_PBUILD_ROW = (25.0, 70.0)
T_TABLE = [
    (1e3, 0.52, 1.82),
    (1e4, 0.73, 2.65),
    (1e5, 2.2, 6.2),
    (3e5, 16.3, 10.0),
    (1e6, 12.3, 10.5),
    (1e7, 8.2, 14.5),
]
# Python and call overhead per step, so tiny tensors take the plan with the fewest steps.
C_STEP = 10_000.0
C_REBUILD_FIXED = 40_000.0


# A 2-thread transpose into many rows vs 1 thread: uniform 1e3 x 1e6 58 vs 97 ms, FROSTT Uber
# 9.2 vs 11.5 ms (measured 2026-10-20)
C_T_TWO_THREADS = 0.6


def t_cost(out_rows, ti):
    """Transpose cost per entry, log-linear in the number of output rows."""
    if out_rows <= T_TABLE[0][0]:
        return T_TABLE[0][1 + ti]
    for (r0, *a), (r1, *b) in itertools.pairwise(T_TABLE):
        if out_rows <= r1:
            f = (math.log(out_rows) - math.log(r0)) / (math.log(r1) - math.log(r0))
            return a[ti] + f * (b[ti] - a[ti])
    return T_TABLE[-1][1 + ti]


def rs_cost(shape, seq, q, nnz, ti):
    """A reshape moving the split of ``seq`` to ``q``: per entry and per output row."""
    nrows = prod(shape, seq[:q])
    R = expected_rows(nrows, nnz)
    per_row = C_RS_HYPER_ROW[ti] if nrows > 4 * R else C_RS_ROW[ti]
    return C_RS[ti] * nnz + per_row * R + C_STEP


def rebuild_cost(nnz, ti, *, sorted_keys=False):
    return (C_REBUILD_SORTED if sorted_keys else C_REBUILD)[ti] * nnz + C_REBUILD_FIXED


def _neighbors(state, shape, nnz, ti, hyper1, allow_perm, total_ok, reshaped=False,
               src_cols=None):  # fmt: skip
    """``(edge, next_state, cost, next_reshaped)`` for each edge out of ``state``;
    ``reshaped`` says the state's rows come straight from a reshape (see C_PERM_ROW).
    ``src_cols`` is ``(state, nonempty columns)`` measured on the data for its source state.
    """
    seq, p = state
    n = len(seq)
    for q in range(n + 1):
        if q != p and _valid_split(shape, seq, q):
            if total_ok:
                yield ("RS", q), (seq, q), rs_cost(shape, seq, q, nnz, ti), True
            else:
                # Reshape refuses totals past 2^60; a REBUILD keeps the order (no sort)
                cost = rebuild_cost(nnz, ti, sorted_keys=True)
                yield ("REBUILD", (seq, q)), (seq, q), cost, False
    # A transpose with an empty group is a reshape (RS covers it).
    if 0 < p < n:
        out_rows = prod(shape, seq[p:])
        if src_cols is not None and src_cols[0] == state and not reshaped:
            # From the data itself: clustered data has far fewer nonempty columns than uniform
            # (Enron's (2,3) group: 3.0e6 of 2.9e8 against 4.9e7 uniform), so its transpose
            # goes into hypersparse rows at C_T_HYPER, 4.7x T_TABLE at 1 thread
            hyper_out = out_rows > 16 * src_cols[1]
        else:
            hyper_out = out_rows > 16 * expected_rows(out_rows, nnz)
        cost = (C_T_HYPER[ti] if hyper_out else t_cost(out_rows, ti)) * nnz
        threads = None
        # A transpose into many rows is slower at 4 or 8 threads than at 2, which runs
        # SuiteSparse's non-atomic method and is 1.2-1.8x faster than 1 thread
        # (measured on an Apple M5 Pro); price 2 threads too.
        if ti == 0 and out_rows >= 1e5 and not hyper_out:
            cost2 = C_T_TWO_THREADS * t_cost(out_rows, 1) * nnz
            if cost2 < cost:
                cost = cost2
                threads = 2
        yield ("T", threads), (seq[p:] + seq[:p], n - p), cost + C_STEP, False
    if allow_perm in {True, "rows"} and p >= 2:
        nrows = prod(shape, seq[:p])
        R = expected_rows(nrows, nnz)
        kind = "hyper" if nrows > 4 * R else "reshaped" if reshaped else "sparse"
        cost = C_PERM_ENTRY[ti] * nnz + C_PERM_ROW[kind][ti] * R
        # P itself: shape-only and cached when its rows are few; else built per call
        if nrows <= max(4 * nnz, 4096):
            if nrows > _matrix_cache.maxweight:
                cost += C_PBUILD_FULL * nrows
        else:
            cost += C_PBUILD_ROW[ti] * R
        if hyper1:
            # About one entry per row: P @ M pays a hash lookup per row.
            cost = max(cost, C_REBUILD[ti] * nnz)
        cost += C_STEP
        for perm in itertools.permutations(seq[:p]):
            if perm != seq[:p]:
                yield ("PERM_R", perm), (perm + seq[p:], p), cost, False
    if allow_perm in {True, "cols"} and n - p >= 2:
        # Values untouched, so any dtype; the sort of each row is priced in, although a
        # following transpose would skip it
        ncols = prod(shape, seq[p:])
        per_row = nnz / max(expected_rows(prod(shape, seq[:p]), nnz), 1.0)
        gather = C_PERM_C_MAP if _col_map_ok(ncols, nnz) else C_PERM_C_ARITH
        cost = (gather[ti] + C_PERM_C_SORT[ti] * math.log2(max(2.0, per_row))) * nnz + C_STEP
        for perm in itertools.permutations(seq[p:]):
            if perm != seq[p:]:
                yield ("PERM_C", perm), (seq[:p] + perm, p), cost, reshaped


def expected_rows(nrows, nnz):
    """Expected nonempty rows when ``nnz`` entries fall uniformly into ``nrows`` rows."""
    if nrows <= 0 or nnz <= 0:
        return 0.0
    return nrows * -math.expm1(-nnz / nrows)


def plan(shape, src, targets, nnz, nthreads, *, hyper1=False, allow_perm=True, src_cols=None):
    """Cheapest edge sequence from state ``src`` to any state in ``targets`` (a frozenset).

    Returns ``(cost_ns, path, final_state)``. A path is a list of edges: ``("RS", q)``,
    ``("T", nthreads or None)``, ``("PERM_R", new_row_seq)``, ``("PERM_C", new_col_seq)`` or
    ``("REBUILD", state)``. ``allow_perm`` is which permutation edges may be used: True (both),
    ``"rows"`` (PERM_R), ``"cols"`` (PERM_C) or False. ``src_cols`` is the source's number of
    nonempty columns when it was measured (``sampled_nonempty_cols``); else the uniform
    expectation stands in.
    """
    if src in targets:
        return 0.0, [], src
    ti = 0 if nthreads > 1 else 1
    hint = None if src_cols is None else int(src_cols).bit_length()
    key = (shape, src, targets, ti, hyper1, allow_perm, int(nnz).bit_length(), hint)
    if (rv := _plan_cache.get(key)) is not None:
        if census.on:
            census.note("plan.cache", "hit")
        return rv
    if census.on:
        census.note("plan.cache", "miss", ndim=len(shape))
    # Past 2^60 positions in total, prefer a target in the source's axis order: its REBUILD
    # needs no sort. Below, such a target is reached by reshapes, which cost less still in a
    # chain (the sorted REBUILD's price would undercut RS's, measured with a wait: u6 tdot1 at
    # 1 thread took a REBUILD, 33 ms against 24)
    total_ok = math.prod(shape) < MAX_GROUP
    fallback = min(targets, key=lambda t: (t[0] != src[0], t))
    rebuild = rebuild_cost(nnz, ti, sorted_keys=fallback[0] == src[0] and not total_ok)
    rv = (rebuild, [("REBUILD", fallback)], fallback)
    if len(shape) <= MAX_PLANNED_NDIM and nnz > 0:
        # Nodes are (state, whether its rows come straight from a reshape)
        node = (src, False)
        best = {node: 0.0}
        counter = itertools.count()
        heap = [(0.0, next(counter), node, ())]
        while heap:
            cost, _, node, path = heapq.heappop(heap)
            state, reshaped = node
            if state in targets:
                if cost <= rebuild:
                    rv = (cost, list(path), state)
                else:
                    rv = (rebuild, [("REBUILD", state)], state)
                break
            if cost > best.get(node, math.inf) or cost > rebuild:
                continue
            for edge, nxt, w, flag in _neighbors(
                state, shape, nnz, ti, hyper1, allow_perm, total_ok, reshaped,
                None if src_cols is None else (src, src_cols),
            ):  # fmt: skip
                nc = cost + w
                if nc < best.get((nxt, flag), math.inf):
                    best[nxt, flag] = nc
                    heapq.heappush(heap, (nc, next(counter), (nxt, flag), (*path, edge)))
    _plan_cache.put(key, rv)
    return rv


# ---------------------------------------------------------------------------------------------
# Executing a plan


def _perm_semiring(M):
    # For P @ M (or S @ v) where P (a permutation or a selector) has one True entry per row:
    # the monoid never combines two values, so pick one with a factory kernel (the JIT at
    # "off" or "run" runs a generic kernel otherwise, about 2x slower). M may be a Vector.
    # min_second is one for the 32- and 64-bit integer and floating types, and
    # it keeps an iso M iso; SuiteSparse 10 compiled out any_second, and plus_second is generic
    # for UINT32 / UINT64 and drops iso (as SuiteSparse's burble shows).
    # BOOL has no factory SECOND kernel, but lor_land has one, and land(True, m) is m.
    # The 8- and 16-bit integers have no factory kernels at all; complex has no min.
    if M.dtype._is_udt:
        return _semiring.plus_second  # (a UDT never moves by P @ M; this is for S @ v)
    if M.dtype.name in {"FC32", "FC64"}:
        return _semiring.any_second if M.ss.is_iso else _semiring.plus_second
    if M.dtype == BOOL:
        return _semiring.lor_land
    return _semiring.min_second


def _perm_matrix(shape, old_rows, new_rows, M):
    """P with ``P[new_row, old_row] = True``, rows sorted, so ``P @ M`` permutes the rows."""
    nrows = M._nrows
    nvals = M._nvals
    if nrows <= max(4 * nvals, 4096):
        key = ("P", tuple(shape[a] for a in old_rows), old_rows, new_rows)
        if (P := _matrix_cache.get(key)) is not None:
            return P
        # Iterate the new row ids in order so P's rows arrive sorted.
        newr = np.arange(nrows, dtype=np.int64)
        co = {}
        unravel(newr, shape, new_rows, co)
        oldr = ravel(co, shape, old_rows)
        P = from_coo(newr, oldr, True, BOOL, nrows, nrows)
        _matrix_cache.put(key, P, nrows)
        return P
    # Many more rows than entries: build P from the nonempty rows only (not cacheable)
    oldr = nonempty_row_ids(M)
    co = {}
    unravel(oldr, shape, old_rows, co)
    newr = ravel(co, shape, new_rows)
    return from_coo(newr, oldr, True, BOOL, nrows, nrows)


def perm_rows(M, shape, old_rows, new_rows):
    P = _perm_matrix(shape, old_rows, new_rows, M)
    out = new_matrix(M.dtype, M._nrows, M._ncols)
    out << P.mxm(M, _perm_semiring(M))
    return out


def _col_map(shape, old_cols, new_cols, ncols):
    """``g`` with ``g[old_col] = new_col`` for a reordering of the column group's axes: a full
    Vector of 32- or 64-bit indices, shape-only and cached.
    """
    key = ("Q", tuple(shape[a] for a in old_cols), old_cols, new_cols)
    if (g := _matrix_cache.get(key)) is not None:
        return g
    co = {}
    unravel(np.arange(ncols, dtype=np.int64), shape, old_cols, co)
    dtype = UINT32 if ncols <= 2**32 else UINT64
    g = Vector.from_dense(ravel(co, shape, new_cols).astype(dtype.np_type), dtype=dtype)
    _matrix_cache.put(key, g, ncols)
    return g


def _col_map_ok(ncols, nnz):
    # Past about 2^17 columns the map no longer sits in cache and computing the new indices
    # costs the same (3.4-3.7 ns per entry at 8 threads, measured 2026-10-20)
    return ncols <= min(2**17, max(4 * nnz, 4096))


def perm_cols(M, shape, state, new_cols):
    """Reorder the axes of ``M``'s column group (``state``'s) to ``new_cols``, in place: the
    column indices go through the map inside the Container and the rows are marked jumbled, so
    SuiteSparse sorts each row only when something needs it (a transpose does not). The map is
    one-to-one, so the new indices within a row stay distinct. Values are not touched.
    """
    seq, p = state
    old_cols = seq[p:]
    if M._nvals == 0:
        return M
    if stored_format(M) not in {"csr", "hypercsr"}:
        # Bitmap or full storage has no column index array
        return rebuild(M, shape, state, (seq[:p] + new_cols, p))
    ncols = M._ncols
    use_map = _col_map_ok(ncols, M._nvals)  # (M reads as empty while unloaded)
    with _Unloaded(M) as cont:
        if use_map:
            # Gather through the cached map in GraphBLAS (parallel)
            g = _col_map(shape, old_cols, new_cols, ncols)
            n = ffi.new("GrB_Index*")
            _check(lib.GrB_Vector_size(n, cont.i), "GrB_Vector_size")
            w = Vector(g.dtype, n[0])
            _check(
                lib.GxB_Vector_extract_Vector(w.gb_obj[0], NULL, NULL, g.gb_obj[0], cont.i, NULL),
                "GxB_Vector_extract_Vector",
            )
            _check(lib.GrB_Vector_free(ffi.new("GrB_Vector*", cont.i)), "GrB_Vector_free")
            cont.i = w.gb_obj[0]
            w.gb_obj[0] = NULL
        else:
            # A column group far wider than the data: compute the new indices (numpy)
            with _VectorArray(cont.i) as iv:
                co = {}
                unravel(iv.array.astype(np.int64), shape, old_cols, co)
                iv.array[:] = ravel(co, shape, new_cols)
        cont.jumbled = True
    return M


def rebuild(M, shape, src, dst):
    seq, p = dst
    iso = M.ss.is_iso
    coords, v = coords_of(M, shape, src, values=not iso)
    if iso:
        v = M.ss.iso_value.value if M._nvals > 0 else 0
    rows = ravel(coords, shape, seq[:p])
    cols = ravel(coords, shape, seq[p:], n=rows.size)
    return from_coo(rows, cols, v, M.dtype, prod(shape, seq[:p]), prod(shape, seq[p:]))


def transpose(M, *, nthreads=None):
    out = new_matrix(M.dtype, M._ncols, M._nrows)
    if nthreads:
        out(nthreads=nthreads) << M.T
    else:
        out << M.T
    return out


def reshape_owned(M, nrows, ncols):
    """``M``, which the caller may change, reshaped to ``nrows`` x ``ncols``: in place, which
    skips a copy of the values, except past 2^31 rows or columns, where SuiteSparse 10.5.1's
    in-place reshape keeps 32-bit indices that cannot hold the new ones (Issue 462), and the
    copying reshape runs.

    TODO(SuiteSparse 10.5.2): always in place once 10.5.2, which fixes Issue 462, is the
    oldest supported version; then this helper can go.
    """
    if nrows <= 2**31 and ncols <= 2**31:
        M.ss.reshape(nrows, ncols, inplace=True)
        return M
    return M.ss.reshape(nrows, ncols)


def execute(M, shape, state, path):
    """Run ``path`` on ``M`` (held by row, in ``state``); return the new Matrix and state.

    ``M`` itself is never changed; intermediates are reshaped in place, which skips a copy
    of the values (7.1 ms vs 9.6 for 1e7 entries at 8 threads, and less peak memory), except
    where ``reshape_owned`` must copy (SuiteSparse 10.5.1).
    """
    owned = False
    for edge in path:
        seq, p = state
        kind = edge[0]
        if kind == "RS":
            q = edge[1]
            nrows, ncols = prod(shape, seq[:q]), prod(shape, seq[q:])
            if M._nvals == 0:
                M = new_matrix(M.dtype, nrows, ncols)
                how = "empty: new"
            elif owned:
                # Storage order (rowwise on a by-row Matrix), so the other 10.5.1 reshape bug
                # (order != orientation) cannot apply either.
                before = M
                M = reshape_owned(M, nrows, ncols)
                how = "in place" if M is before else "copy (past 2^31)"
            else:
                M = M.ss.reshape(nrows, ncols)
                how = "copy"
            if census.on:
                census.note("reshape", how)
            state = (seq, q)
            owned = True
        elif kind == "T":
            M = transpose(M, nthreads=edge[1])
            state = (seq[p:] + seq[:p], len(seq) - p)
        elif kind == "PERM_R":
            M = perm_rows(M, shape, seq[:p], edge[1])
            state = (edge[1] + seq[p:], p)
        elif kind == "PERM_C":
            M = perm_cols(M if owned else M.dup(), shape, state, edge[1])
            state = (seq[:p] + edge[1], p)
        else:
            M = rebuild(M, shape, state, edge[1])
            state = edge[1]
        owned = True
    if path and path[-1][0] == "RS" and M._nvals > 0:
        _restore_hyper_switch(M)
    return M, state


def _restore_hyper_switch(M):
    """A reshape's output has hyper_switch 1.0, so it stays hypersparse even with every row
    nonempty (SuiteSparse 10.5.1). When at least half the rows are nonempty, give it the
    global switch, which converts it to sparse: a hypersparse A sends mxm to a Gustavson saxpy
    (u7 TTM modes 0 and 1 at 1 thread: 187 -> 128 and 165 -> 105 ms with this). With fewer,
    SuiteSparse could still convert (skew's 9.5e6 of 1e8 rows is past its 1/16), and the row
    pointers would outweigh the data, so it stays. The count is the uniform estimate:
    counting through the Container would wait, and a wait builds the rows' hash (C_RS_HYPER_ROW).
    """
    switch = _global_hyper_switch()
    if M.ss.config["hyper_switch"] != switch and 2 * expected_rows(M._nrows, M._nvals) >= M._nrows:
        M.ss.config["hyper_switch"] = switch


def _global_hyper_switch():
    from .. import ss

    return ss.config["hyper_switch"]


def _nthreads():
    val = ffi.new("int32_t*")
    _check(lib.GxB_Global_Option_get_INT32(lib.GxB_NTHREADS, val), "GxB_Global_Option_get")
    return val[0]


def density(M):
    """True when the rows hold fewer than about 2 entries each (the hypersparse branch)."""
    nvals = M._nvals
    if nvals < 65536:
        # Small: the choice does not matter, and the Container round trip costs time.
        return False
    return nvals < 2 * nonempty_rows(M)


def relayout(M, shape, src, targets, *, nthreads=None, allow_perm=None):
    """Move ``M`` from Layout ``src`` to a layout in ``targets`` (a Layout or a frozenset of
    states). Returns ``(M2, layout2)``; ``M2`` is ``M`` itself when no move is needed.
    """
    if len(src.rows) + len(src.cols) != len(shape):
        # A layout of another rank would leave axes without coordinates in a REBUILD
        raise ValueError(f"layout {src} does not match a shape of {len(shape)} axes: {shape}")
    src_state = src.state
    if isinstance(targets, Layout):
        if targets == src:
            return M, src
        targets = frozenset([targets.state])
    elif src_state in targets:
        return M, src
    if nthreads is None:
        nthreads = _nthreads()
    nvals = M._nvals
    if allow_perm is None:
        # PERM_R is a product (P @ M) and needs a semiring for the values; PERM_C moves column
        # indices only, so it takes any type
        allow_perm = "cols" if M.dtype._is_udt else True
    if census.on and (want := census.force.get("relayout")) in _CENSUS_PERMS:
        allow_perm = _perms_value(_perms_set(allow_perm) - _CENSUS_PERMS[want])
    hyper1 = density(M)
    # A transpose out of the source is the costliest step a plan can start with: price it on
    # the data's own nonempty columns, sampled, when the data is large enough to matter
    src_cols = None
    if nvals >= _SAMPLE_COLS_PAST and len(src.rows) > 0 and len(src.cols) > 1:
        src_cols = sampled_nonempty_cols(M)
    cost, path, _ = plan(shape, src_state, targets, nvals, nthreads, hyper1=hyper1,
                         allow_perm=allow_perm, src_cols=src_cols)  # fmt: skip
    if census.on:
        if census.force.get("relayout") == "REBUILD" and path:
            fallback = min(targets, key=lambda t: (t[0] != src_state[0], t))
            path = [("REBUILD", fallback)]
        _census_relayout(shape, src_state, targets, nvals, nthreads, hyper1, allow_perm, cost,
                         path)  # fmt: skip
    if nvals == 0:
        seq, p = _final_state(src_state, path)
        return new_matrix(M.dtype, prod(shape, seq[:p]), prod(shape, seq[p:])), Layout(
            seq[:p], seq[p:]
        )
    M, state = execute(M, shape, src_state, path)
    return M, Layout.from_state(state)


_CENSUS_PERMS = {"no PERM": {"rows", "cols"}, "no PERM_R": {"rows"}, "no PERM_C": {"cols"}}


def _perms_set(allow_perm):
    return {True: {"rows", "cols"}, False: set()}.get(allow_perm, {allow_perm})


def _perms_value(kinds):
    return True if len(kinds) == 2 else kinds.pop() if kinds else False


def _census_relayout(shape, src, targets, nvals, nthreads, hyper1, allow_perm, cost, path):
    """Note an executed plan: its edges, and what the cheapest plan without each kind of edge
    it uses would cost (the planner's margin for that kind; REBUILD is always there).
    """
    kinds = sorted({e[0] for e in path})
    without = {}
    census.on = False  # (the counterfactual plans are not decisions)
    try:
        for kind in kinds:
            if kind == "REBUILD":
                continue
            if kind in {"PERM_R", "PERM_C"}:
                rest = _perms_value(
                    _perms_set(allow_perm) - {"rows" if kind == "PERM_R" else "cols"}
                )
                without[kind] = plan(shape, src, targets, nvals, nthreads, hyper1=hyper1,
                                     allow_perm=rest)[0]  # fmt: skip
            else:
                # Without RS or T, only REBUILD and permutations remain priced alike: bound
                # the margin by REBUILD's cost
                without[kind] = rebuild_cost(nvals, 0 if nthreads > 1 else 1)
    finally:
        census.on = True
    census.note(
        "relayout",
        " ".join(e[0] for e in path) or "none",
        {"plan": cost, **without},
        length=len(path),
        nvals=nvals,
        ndim=len(shape),
    )


# Below this many entries the uniform estimate does (a sample costs about 0.3 ms)
_SAMPLE_COLS_PAST = 2**22
_SAMPLE = 2**14


def sampled_nonempty_cols(M):
    """A low estimate of the nonempty columns of ``M`` (held by row, sparse or hypersparse),
    from a sample of its column indices: the distinct count d in s of them, solving
    D * (1 - exp(-s / D)) = d (equal columns; skewed ones give less). None for bitmap or full.
    """
    if stored_format(M) not in {"csr", "hypercsr"}:
        return None
    n = M._nvals
    with _Unloaded(M) as cont, _VectorArray(cont.i) as i_arr:
        if i_arr.array is None:
            return None
        idx = np.random.default_rng(0).integers(0, n, min(n, _SAMPLE))
        keys = np.sort(i_arr.array[idx])
    s = keys.size
    d = int(np.count_nonzero(keys[1:] != keys[:-1])) + 1
    if d >= s:
        return n
    lo, hi = float(d), float(n)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid * -math.expm1(-s / mid) < d:
            lo = mid
        else:
            hi = mid
    return lo


def _final_state(state, path):
    for edge in path:
        seq, p = state
        if edge[0] == "RS":
            state = (seq, edge[1])
        elif edge[0] == "T":
            state = (seq[p:] + seq[:p], len(seq) - p)
        elif edge[0] == "PERM_R":
            state = (edge[1] + seq[p:], p)
        else:
            state = edge[1]
    return state


def relayout_cost(shape, src, dst, nnz, nthreads):
    """Estimated ns to move ``nnz`` entries from Layout ``src`` to Layout ``dst``."""
    if src == dst:
        return 0.0
    return plan(shape, src.state, frozenset([dst.state]), nnz, nthreads)[0]


# ---------------------------------------------------------------------------------------------
# Selectors: kron-structured 0/1 matrices that reduce a subset of a group's axes in one mxm


def selector(group_shape, kept, *, transposed=False):
    """S with ``S[ravel(x[kept]), ravel(x)] = True`` over the positions ``x`` of a group, as a
    BOOL iso Matrix (``prod(kept) x prod(group)``); its transpose if ``transposed``.

    ``kept`` are positions within the group, in group order. ``S @ M`` reduces the other row
    axes and ``M @ S.T`` the other column axes, with semiring ``(monoid, second/first)``.
    """
    key = ("S", group_shape, kept, transposed)
    if (S := _matrix_cache.get(key)) is not None:
        return S
    n = math.prod(group_shape)
    src = np.arange(n, dtype=np.int64)
    co = {}
    axes = tuple(range(len(group_shape)))
    unravel(src, group_shape, axes, co)
    dst = ravel(co, group_shape, kept, n=n)
    nkept = prod(group_shape, kept)
    if transposed:
        S = from_coo(src, dst, True, BOOL, n, nkept)
    else:
        S = from_coo(dst, src, True, BOOL, nkept, n)
    _matrix_cache.put(key, S, n)
    return S


# ---------------------------------------------------------------------------------------------
# Block-diagonal embedding for batched contractions

_INDEX_NP = None


def _index_dtypes():
    global _INDEX_NP
    if _INDEX_NP is None:
        _INDEX_NP = [
            (lib.GrB_UINT32, np.uint32),
            (lib.GrB_UINT64, np.uint64),
            (lib.GrB_INT32, np.int32),
            (lib.GrB_INT64, np.int64),
        ]
    return _INDEX_NP


class _VectorArray:
    """The array inside one Container vector as a numpy view; loaded back on exit. Index
    arrays are recognized by their GraphBLAS type; for values, pass the numpy dtype.
    """

    def __init__(self, V, dtype=None):
        self.V = V
        self._np_dtype = dtype

    def __enter__(self):
        self._X = ffi.new("void**")
        self._type = ffi.new("GrB_Type*")
        self._n = ffi.new("uint64_t*")
        self._size = ffi.new("uint64_t*")
        self._handling = ffi.new("int*")
        _check(
            lib.GxB_Vector_unload(
                self.V, self._X, self._type, self._n, self._size, self._handling, NULL
            ),
            "GxB_Vector_unload",
        )
        if self._np_dtype is not None:
            dtype = self._np_dtype
        else:
            dtype = next((d for t, d in _index_dtypes() if t == self._type[0]), None)
        self.handling = self._handling[0]
        if dtype is None or self._n[0] == 0:
            self.array = None
        else:
            buf = ffi.buffer(self._X[0], self._n[0] * np.dtype(dtype).itemsize)
            self.array = np.frombuffer(buf, dtype)
        return self

    def __exit__(self, *exc):
        self.array = None
        _check(
            lib.GxB_Vector_load(
                self.V, self._X, self._type[0], self._n[0], self._size[0], self._handling[0], NULL
            ),
            "GxB_Vector_load",
        )


def gather_operand(M, shape, layout, axes, values, dtype):
    """A Matrix with ``M``'s pattern whose entry at logical coordinates x holds
    ``values[ravel(x[axes])]`` (``values`` is a dense numpy array over ``axes``, raveled in
    that order), or None when that is not possible (iso, bitmap or full storage).

    The pattern is copied in GraphBLAS (parallel); the values are gathered with numpy into the
    copy's own value array through the Container, O(nnz), with no data movement. A
    broadcast multiply along axes inside a wide group then takes one same-pattern ewise_mult.
    """
    if stored_format(M) not in {"csr", "hypercsr"} or M.ss.is_iso or M._nvals == 0:
        return None
    W = new_matrix(dtype, M._nrows, M._ncols)
    W << M
    if stored_format(W) not in {"csr", "hypercsr"} or W.ss.is_iso:
        return None
    rows, cols = layout
    need_rows = any(a in rows for a in axes)
    with _Unloaded(W) as cont:
        coords = {}
        if need_rows:
            with _VectorArray(cont.p) as pv:
                counts = np.diff(pv.array.astype(np.int64))
            if cont.format == lib.GxB_HYPERSPARSE:
                with _VectorArray(cont.h) as hv:
                    ids = hv.array.astype(np.int64)
            else:
                ids = np.arange(counts.size, dtype=np.int64)
            r = np.repeat(ids, counts)
            _coords_along(r, shape, rows, axes, coords)
        with _VectorArray(cont.i) as iv:
            _coords_along(iv.array, shape, cols, axes, coords)
        idx = ravel(coords, shape, axes, n=M._nvals)
        with _VectorArray(cont.x, dtype.np_type) as xv:
            xv.array[:] = values[idx]
    return W


def _coords_along(linear, shape, group, axes, out):
    """Fill ``out[a]`` for the ``axes`` in ``group`` from linear indices over the group."""
    stride = 1
    for a in reversed(group):
        if a in axes:
            out[a] = (linear // stride) % shape[a] if stride > 1 else linear % shape[a]
        stride *= shape[a]


def embed_block_diagonal(M, rows_per_block, nblocks):
    """In place: ``M`` (``nblocks * rows_per_block`` x K, held by row, owned by the caller)
    becomes the block-diagonal (``nblocks * rows_per_block`` x ``nblocks * K``) Matrix whose
    block b holds rows ``b * rows_per_block ...``: entry (r, c) moves to column
    ``(r // rows_per_block) * K + c``.

    Each row gains one constant, so sorted rows stay sorted, and the column indices are
    changed in the Container's own array, keeping their integer width (``ss.unpack`` would
    widen them to 64 bits for good). Returns False when that is not possible (the new width
    does not fit, a read-only array, bitmap or full storage); then nothing changed.
    """
    K = M._ncols
    new_ncols = nblocks * K
    if stored_format(M) not in {"csr", "hypercsr"}:
        return False
    with _Unloaded(M) as cont:
        hyper = cont.format == lib.GxB_HYPERSPARSE
        with _VectorArray(cont.i) as i_arr:
            if i_arr.array is None or i_arr.handling != 0:
                return _abort_embed(i_arr.array is None and cont.nvals == 0, cont, new_ncols, M)
            # SuiteSparse stores 32-bit indices only for dimensions below 2^31
            if new_ncols > (2**31 if i_arr.array.dtype.itemsize == 4 else 2**60):
                return False
            with _VectorArray(cont.p) as p_arr:
                counts = np.diff(p_arr.array.astype(np.int64))
            if hyper:
                with _VectorArray(cont.h) as h_arr:
                    rows = h_arr.array.astype(np.int64)
            else:
                rows = np.arange(counts.size, dtype=np.int64)
            offsets = np.repeat((rows // rows_per_block) * K, counts)
            i_arr.array += offsets.astype(i_arr.array.dtype, copy=False)
        cont.ncols = new_ncols
        cont.ncols_nonempty = -1
    M._ncols = new_ncols
    return True


def _abort_embed(empty, cont, new_ncols, M):
    if empty:
        cont.ncols = new_ncols
        cont.ncols_nonempty = -1
        M._ncols = new_ncols
        return True
    return False

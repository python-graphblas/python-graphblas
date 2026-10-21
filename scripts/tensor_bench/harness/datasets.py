"""Seeded, reproducible datasets and operands for the benchmark kit.

Every dataset is an n-d sparse tensor given as ``(coords, vals, shape)``: a list of int64
coordinate arrays (0-based, sorted in C order, no duplicates) and float64 values (synthetic:
in [0.001, 1.001); FROSTT: the file's positive values). Positive values make "present" and
"nonzero" the same thing, so results from libraries with and without missing-value semantics
compare.

Synthetic data is generated on first use and cached in ``<data dir>/synth/``. FROSTT tensors
come from ``fetch_data.py`` (``<data dir>/frostt/``) and are converted once to
``<data dir>/frostt/cache/<name>.npz``. The data directory is ``TENSOR_BENCH_DATA`` (see
``paths.py``).
"""

import os
import sys
from pathlib import Path

import numpy as np

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

# isort: on


SYNTH = str(paths.data_dir() / "synth")
FROSTT = str(paths.data_dir() / "frostt" / "cache")
FROSTT_SRC = str(paths.data_dir() / "frostt")

# name: (kind, shape, nnz, seed, extra)
SPECS = {
    "small": ("uniform", (20, 30, 40), 2000, 11, None),
    "small_skew": ("uniform", (60, 7, 90), 3000, 12, None),
    "u6": ("uniform", (1000, 1000, 1000), 10**6, 1, None),
    "u7": ("uniform", (1000, 1000, 1000), 10**7, 2, None),
    "skew": ("uniform", (100_000, 1000, 10), 10**7, 3, None),
    "hyper": ("uniform", (2**19, 2**19, 2**19), 10**6, 4, None),
    "zipf": ("zipf", (100_000, 100_000, 100_000), 10**7, 5, 1.5),
    "nell2": ("frostt", None, None, None, "nell-2"),
    "uber": ("frostt", None, None, None, "uber"),
    "nips": ("frostt", None, None, None, "nips"),
    "enron": ("frostt", None, None, None, "enron"),
    "chicago": ("frostt", None, None, None, "chicago-crime-comm"),
    # batched (b, 1000, 1000), 5 entries per row on average (density 5e-3)
    "bmm_small": ("batched", (4, 20, 30), 4 * 20 * 3, 21, (4, 20, 30, 25)),
    "bmm10": ("batched", (10, 1000, 1000), 10 * 1000 * 5, 22, (10, 1000, 1000, 1000)),
    "bmm100": ("batched", (100, 1000, 1000), 100 * 1000 * 5, 23, (100, 1000, 1000, 1000)),
    "bmm1000": ("batched", (1000, 1000, 1000), 1000 * 1000 * 5, 24, (1000, 1000, 1000, 1000)),
}

SMALL = {"small", "small_skew", "bmm_small"}


def _ravel(coords, shape):
    lin = np.zeros(coords[0].size, np.int64)
    for c, s in zip(coords, shape, strict=True):
        lin *= s
        lin += c
    return lin


def _unravel(lin, shape):
    coords = []
    for s in shape[::-1]:
        coords.append(lin % s)
        lin = lin // s
    return [c.astype(np.int64) for c in coords[::-1]]


def _uniform(shape, nnz, seed):
    rng = np.random.default_rng(seed)
    total = int(np.prod(shape, dtype=object))
    assert total < 2**62
    lin = np.unique(rng.integers(0, total, int(nnz * 1.05) + 10, dtype=np.int64))
    while lin.size < nnz:
        lin = np.union1d(lin, rng.integers(0, total, nnz, dtype=np.int64))
    lin = np.sort(lin[rng.permutation(lin.size)[:nnz]])
    return _unravel(lin, shape), rng.random(nnz) + 1e-3


def _zipf(shape, nnz, seed, a):
    """Each mode's index drawn from a bounded Zipf(a) over its size, then the index labels
    of each mode permuted at random (heavy slices land at arbitrary positions).
    """
    rng = np.random.default_rng(seed)
    perms, cdfs = [], []
    for n in shape:
        k = np.arange(1, n + 1, dtype=np.float64)
        cdf = np.cumsum(k**-a)
        cdfs.append(cdf / cdf[-1])
        perms.append(rng.permutation(n))
    chunks, have = [], 0
    while have < nnz * 1.02:
        c = [perms[m][np.searchsorted(cdfs[m], rng.random(20_000_000))] for m in range(3)]
        chunks.append(np.unique(_ravel(c, shape)))
        lin = np.unique(np.concatenate(chunks))
        chunks, have = [lin], lin.size
    lin = np.sort(lin[rng.permutation(lin.size)[:nnz]])
    return _unravel(lin, shape), rng.random(nnz) + 1e-3


def _batched(b, n, m, per_row, seed):
    """(b, n, m) with ``per_row`` entries per row on average (uniform columns)."""
    rng = np.random.default_rng(seed)
    nnz = b * n * per_row
    lin = np.unique(rng.integers(0, b * n * m, int(nnz * 1.05) + 10, dtype=np.int64))
    lin = np.sort(lin[rng.permutation(lin.size)[:nnz]])
    return _unravel(lin, (b, n, m)), rng.random(lin.size) + 1e-3


def _save(path, coords, vals, shape):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, shape=np.array(shape), vals=vals, **{f"i{a}": c for a, c in enumerate(coords)})


def _load_npz(path):
    d = np.load(path)
    shape = tuple(int(s) for s in d["shape"])
    coords = [d[f"i{a}"] for a in range(len(shape))]
    return coords, d["vals"], shape


def convert_frostt(stem):
    """Convert ``<data>/frostt/<stem>.tns.gz`` (1-based text) once into the .npz cache:
    ``i0 .. i{N-1}`` (int64, 0-based), ``vals`` (float64), ``shape`` (max index + 1 per mode).
    """
    import gzip

    src = os.path.join(FROSTT_SRC, f"{stem}.tns.gz")
    dst = os.path.join(FROSTT, f"{stem}.npz")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} is missing: run `python fetch_data.py frostt-{stem}` first")
    with gzip.open(src, "rb") as f:
        order = len(f.readline().split()) - 1
    try:
        import pandas as pd

        dtypes = dict.fromkeys(range(order), np.int64)
        dtypes[order] = np.float64
        parts = [
            [chunk[c].to_numpy(copy=True) for c in range(order + 1)]
            for chunk in pd.read_csv(src, sep=" ", header=None, dtype=dtypes, chunksize=10_000_000)
        ]
        cols = [np.concatenate([p[c] for p in parts]) for c in range(order + 1)]
    except ImportError:  # slower, without pandas
        table = np.loadtxt(src, dtype=np.float64)
        cols = [table[:, c].astype(np.int64) for c in range(order)] + [table[:, order]]
    coords = [c - 1 for c in cols[:order]]
    shape = np.array([int(c.max()) + 1 for c in coords], dtype=np.int64)
    os.makedirs(FROSTT, exist_ok=True)
    np.savez(dst, shape=shape, vals=cols[order], **{f"i{a}": coords[a] for a in range(order)})
    return dst


def load(name, which=0):
    """The dataset ``name``; ``which=1`` gives the second operand for batched data."""
    kind, shape, nnz, seed, extra = SPECS[name]
    if kind == "frostt":
        path = os.path.join(FROSTT, f"{extra}.npz")
        if not os.path.exists(path):
            convert_frostt(extra)
        coords, vals, shape = _load_npz(path)
        lin = _ravel(coords, shape)
        if not np.all(lin[1:] > lin[:-1]):
            order = np.argsort(lin, kind="stable")
            coords = [c[order] for c in coords]
            vals = vals[order]
        return coords, vals, shape
    tag = f"{name}" if which == 0 else f"{name}_b"
    path = os.path.join(SYNTH, f"{tag}.npz")
    if name not in SMALL and os.path.exists(path):
        return _load_npz(path)
    if kind == "uniform":
        coords, vals = _uniform(shape, nnz, seed)
    elif kind == "zipf":
        coords, vals = _zipf(shape, nnz, seed, extra)
    elif kind == "batched":
        b, n, k, m = extra
        if which == 0:
            coords, vals = _batched(b, n, k, nnz // (b * n), seed)
            shape = (b, n, k)
        else:
            coords, vals = _batched(b, k, m, nnz // (b * n), seed + 1000)
            shape = (b, k, m)
    else:
        raise ValueError(kind)
    if name not in SMALL:
        _save(path, coords, vals, shape)
    return coords, vals, tuple(shape)


def second_operand(name, coords, vals, shape):
    """The second operand Y for element-wise ops: an independent sample from the same
    generator for synthetic data; for FROSTT a seeded random half of X's entries with new
    values.
    """
    kind, _, nnz, seed, extra = SPECS[name]
    rng = np.random.default_rng(1000 + (seed or 0))
    if kind == "frostt":
        keep = np.sort(rng.permutation(vals.size)[: vals.size // 2])
        return [c[keep] for c in coords], rng.random(keep.size) + 1e-3
    tag = f"{name}_y"
    path = os.path.join(SYNTH, f"{tag}.npz")
    if name not in SMALL and os.path.exists(path):
        c, v, _ = _load_npz(path)
        return c, v
    if kind == "uniform":
        c, v = _uniform(shape, nnz, seed + 100)
    elif kind == "zipf":
        c, v = _zipf(shape, nnz, seed + 100, extra)
    else:
        raise ValueError(kind)
    if name not in SMALL:
        _save(path, c, v, shape)
    return c, v


def sparse_matrix(nrows, ncols, per_row, seed):
    """A (nrows, ncols) sparse matrix with ``per_row`` uniform entries per row, as COO."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(nrows, dtype=np.int64), per_row)
    cols = rng.integers(0, ncols, rows.size, dtype=np.int64)
    lin = np.unique(rows * ncols + cols)
    return lin // ncols, lin % ncols, rng.random(lin.size) + 1e-3


def vectors(name, shape):
    """Dense vectors (one per axis) for broadcast and TTV."""
    rng = np.random.default_rng(2000 + (SPECS[name][3] or 7))
    return [rng.random(s) + 1e-3 for s in shape]


def dense_matrix12(name, shape, limit=3 * 10**8):
    """Dense (s1, s2) matrix for the broadcast over modes (1, 2), or None if too big."""
    if shape[1] * shape[2] > limit:
        return None
    rng = np.random.default_rng(3000 + (SPECS[name][3] or 7))
    return rng.random((shape[1], shape[2])) + 1e-3


def factors(name, shape, R):
    seed = SPECS[name][3] or 7
    rng = np.random.default_rng(4000 + 10 * seed + R)
    return [rng.random((s, R)) + 1e-3 for s in shape]


def tdot_operands(name, shape):
    """Operands for the contractions: a sparse (s2, 1000) matrix with 2 entries per row for
    'ijk,kl->ijl', and a sparse (s1, s2, 1000) tensor at density 1e-3 (at most nnz(X)
    entries) for 'ijk,jkl->il'.
    """
    _, _, nnz, seed, _ = SPECS[name]
    L = 1000 if name not in SMALL else 25
    m = sparse_matrix(shape[2], L, 2, 5000 + seed)
    shape2 = (shape[1], shape[2], L)
    tag = f"{name}_y2"
    path = os.path.join(SYNTH, f"{tag}.npz")
    if name not in SMALL and os.path.exists(path):
        c2, v2, _ = _load_npz(path)
    else:
        nnz2 = min(nnz, int(np.prod(shape2, dtype=object)) // (1000 if name not in SMALL else 20))
        c2, v2 = _uniform(shape2, nnz2, 6000 + seed)
        if name not in SMALL:
            _save(path, c2, v2, shape2)
    return m + (L,), (c2, v2, shape2)


def slices(shape):
    """Slicing keys: the middle index of each axis, then a range of 10% of each axis."""
    keys = {}
    for a, s in enumerate(shape):
        key = [slice(None)] * len(shape)
        key[a] = s // 2
        keys[f"slice{a}"] = tuple(key)
    for a, s in enumerate(shape):
        key = [slice(None)] * len(shape)
        lo = s // 10
        key[a] = slice(lo, lo + max(1, s // 10))
        keys[f"range{a}"] = tuple(key)
    return keys


def fiber_counts(name, coords, shape):
    """Number of distinct mode-m fibers (distinct coordinates of the other modes) per mode;
    cached in ``<data dir>/synth/<name>_fibers.json`` for non-small datasets.
    """
    import json

    path = os.path.join(SYNTH, f"{name}_fibers.json")
    if name not in SMALL and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = []
    for m in range(len(shape)):
        others = [a for a in range(len(shape)) if a != m]
        out.append(
            int(np.unique(_ravel([coords[a] for a in others], [shape[a] for a in others])).size)
        )
    if name not in SMALL:
        os.makedirs(SYNTH, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
    return out

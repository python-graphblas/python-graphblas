"""Shared data builders for the python-graphblas asv benchmark suite.

Each benchmark module imports these helpers and calls them from ``setup`` so the
input data is constructed once per benchmark (outside the timed region). All
builders take a seed and use ``numpy.random.default_rng`` so runs are
deterministic and comparable across commits.

Size constants are chosen so the "large" kernels land near ~1e6 nonzeros (enough
to exercise the C library) while still finishing in well under a second each, and
so no dense intermediate blows up memory (a 1e6 x 1e6 dense matrix is never
materialized; dense conversions use a separate, small, fully dense matrix).
"""

import numpy as np

import graphblas as gb
from graphblas import Matrix, Vector

# Sizes

# "Large" sparse operands: square matrix and vector near 1e6 nonzeros. Average
# degree ~1 keeps mxm output bounded (A @ A stays near 1e6 nnz) so the kernel
# benchmarks do not accidentally measure quadratic fill-in.
LARGE_N = 1_000_000
LARGE_NNZ = 1_000_000

# "Small" operands: overhead-dominated. At this size the wall time is almost all
# Python-side expression machinery, which is exactly what we want to track.
SMALL_SIZE = 100

# Dense conversions use a fully dense square matrix small enough to materialize.
DENSE_DIM = 1000  # 1e6 dense elements

# scipy interop matrix: ~1e6 nnz at 1% density.
SCIPY_DIM = 10_000
SCIPY_DENSITY = 0.01


# Builders


def make_coo(n, nnz, seed=0, dtype="FP64"):
    """Return (rows, cols, vals) numpy arrays for an n x n matrix with ~nnz entries.

    Duplicates are possible; callers that build a Matrix should pass a ``dup_op``.
    """
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, nnz, dtype=np.uint64)
    cols = rng.integers(0, n, nnz, dtype=np.uint64)
    if dtype == "BOOL":
        vals = rng.integers(0, 2, nnz, dtype=np.bool_)
    elif dtype in ("INT64", "INT32"):
        vals = rng.integers(1, 100, nnz, dtype=np.int64)
    else:
        vals = rng.random(nnz)
    return rows, cols, vals


def make_matrix(n=LARGE_N, nnz=LARGE_NNZ, seed=0, dtype="FP64"):
    """Square Matrix with ~nnz entries (duplicates summed)."""
    rows, cols, vals = make_coo(n, nnz, seed=seed, dtype=dtype)
    dup = gb.binary.lor if dtype == "BOOL" else gb.binary.plus
    return Matrix.from_coo(rows, cols, vals, nrows=n, ncols=n, dtype=dtype, dup_op=dup)


def make_vector(size=LARGE_N, nnz=LARGE_NNZ, seed=1, dtype="FP64"):
    """Vector of length ``size`` with ~nnz entries (duplicates summed)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, size, nnz, dtype=np.uint64)
    vals = rng.random(nnz) if dtype not in ("INT64", "INT32") else rng.integers(1, 100, nnz)
    dup = gb.binary.plus
    return Vector.from_coo(idx, vals, size=size, dtype=dtype, dup_op=dup)


def make_dense_vector(size, seed=2, dtype="FP64"):
    """Fully dense Vector (no missing entries), e.g. an mxv operand."""
    rng = np.random.default_rng(seed)
    return Vector.from_dense(rng.random(size))


def make_dense_matrix(dim=DENSE_DIM, seed=3):
    """Fully dense square Matrix built from a numpy array."""
    rng = np.random.default_rng(seed)
    return Matrix.from_dense(rng.random((dim, dim)))


def make_scipy(dim=SCIPY_DIM, density=SCIPY_DENSITY, seed=4, fmt="csr"):
    """A scipy.sparse matrix for interop benchmarks."""
    import scipy.sparse as sp

    return sp.random(dim, dim, density=density, format=fmt, random_state=seed)

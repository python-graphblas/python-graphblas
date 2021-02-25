import numpy as np
import grblas as gb
from numba import njit
from .dtypes import lookup_dtype


@njit
def _head_indices_vector_bitmap(bitmap, values, size, dtype, n):
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=dtype)
    j = 0
    for i in range(size):
        if bitmap[i]:
            indices[j] = i
            vals[j] = values[i]
            j += 1
            if j == n:
                break
    return indices, vals


def vector_head(vector, n=10, *, sort=False, dtype=None):
    """Like ``vector.to_values()``, but only returns the first n elements.

    If sort is True, then the results will be sorted by index, otherwise the order of the
    result is not guaranteed.  Formats full and bitmap should always return in sorted order.

    This changes ``vector.gb_obj``, so care should be taken when using multiple threads.
    """
    if dtype is None:
        dtype = vector.dtype
    else:
        dtype = lookup_dtype(dtype)
    n = min(n, vector._nvals)
    if n == 0:
        return (np.empty(0, dtype=np.uint64), np.empty(0, dtype=dtype.np_type))
    d = vector.ss.export(raw=True, give_ownership=True, sort=sort)
    fmt = d["format"]
    try:
        if fmt == "full":
            indices = np.arange(n, dtype=np.uint64)
            vals = d["values"][:n].astype(dtype.np_type)
        elif fmt == "bitmap":
            indices, vals = _head_indices_vector_bitmap(
                d["bitmap"], d["values"], d["size"], dtype.np_type, n
            )
        elif fmt == "sparse":
            indices = d["indices"][:n].copy()
            vals = d["values"][:n].astype(dtype.np_type)
        else:  # pragma: no cover
            raise RuntimeError(f"Invalid format: {fmt}")
    finally:
        rebuilt = gb.Vector.ss.import_any(take_ownership=True, name="", **d)
        # We need to set rebuilt.gb_obj to NULL so it doesn't get deleted early, so might
        # as well do a swap, b/c vector.gb_obj is already "destroyed" from the export.
        vector.gb_obj, rebuilt.gb_obj = rebuilt.gb_obj, vector.gb_obj
    return indices, vals


@njit
def _head_matrix_full(values, nrows, ncols, dtype, n):
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=dtype)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            rows[k] = i
            cols[k] = j
            vals[k] = values[i * ncols + j]
            k += 1
            if k == n:
                return rows, cols, vals
    return rows, cols, vals


@njit
def _head_matrix_bitmap(bitmap, values, nrows, ncols, dtype, n):
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=dtype)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if bitmap[i * ncols + j]:
                rows[k] = i
                cols[k] = j
                vals[k] = values[i * ncols + j]
                k += 1
                if k == n:
                    return rows, cols, vals
    return rows, cols, vals


@njit
def _head_csr_rows(indptr, n):
    rows = np.empty(n, dtype=np.uint64)
    idx = 0
    index = 0
    for row in range(indptr.size - 1):
        next_idx = indptr[row + 1]
        while next_idx != idx:
            rows[index] = row
            index += 1
            if index == n:
                return rows
            idx += 1
    return rows


@njit
def _head_hypercsr_rows(indptr, rows, n):
    rv = np.empty(n, dtype=np.uint64)
    idx = 0
    index = 0
    for ptr in range(indptr.size - 1):
        next_idx = indptr[ptr + 1]
        row = rows[ptr]
        while next_idx != idx:
            rv[index] = row
            index += 1
            if index == n:
                return rv
            idx += 1
    return rv


def matrix_head(matrix, n=10, *, sort=False, dtype=None):
    """Like ``matrix.to_values()``, but only returns the first n elements.

    If sort is True, then the results will be sorted as appropriate for the internal format,
    otherwise the order of the result is not guaranteed.  Specifically, row-oriented formats
    (fullr, bitmapr, csr, hypercsr) will sort by row index first, then by column index.
    Column-oriented formats, naturally, will sort by column index first, then by row index.
    Formats fullr, fullc, bitmapr, and bitmapc should always return in sorted order.

    This changes ``matrix.gb_obj``, so care should be taken when using multiple threads.
    """
    if dtype is None:
        dtype = matrix.dtype
    else:
        dtype = lookup_dtype(dtype)
    n = min(n, matrix._nvals)
    if n == 0:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=dtype.np_type),
        )
    d = matrix.ss.export(raw=True, give_ownership=True, sort=sort)
    try:
        fmt = d["format"]
        if fmt == "fullr":
            rows, cols, vals = _head_matrix_full(
                d["values"], d["nrows"], d["ncols"], dtype.np_type, n
            )
        elif fmt == "fullc":
            cols, rows, vals = _head_matrix_full(
                d["values"], d["ncols"], d["nrows"], dtype.np_type, n
            )
        elif fmt == "bitmapr":
            rows, cols, vals = _head_matrix_bitmap(
                d["bitmap"], d["values"], d["nrows"], d["ncols"], dtype.np_type, n
            )
        elif fmt == "bitmapc":
            cols, rows, vals = _head_matrix_bitmap(
                d["bitmap"], d["values"], d["ncols"], d["nrows"], dtype.np_type, n
            )
        elif fmt == "csr":
            vals = d["values"][:n].astype(dtype.np_type)
            cols = d["col_indices"][:n].copy()
            rows = _head_csr_rows(d["indptr"], n)
        elif fmt == "csc":
            vals = d["values"][:n].astype(dtype.np_type)
            rows = d["row_indices"][:n].copy()
            cols = _head_csr_rows(d["indptr"], n)
        elif fmt == "hypercsr":
            vals = d["values"][:n].astype(dtype.np_type)
            cols = d["col_indices"][:n].copy()
            rows = _head_hypercsr_rows(d["indptr"], d["rows"], n)
        elif fmt == "hypercsc":
            vals = d["values"][:n].astype(dtype.np_type)
            rows = d["row_indices"][:n].copy()
            cols = _head_hypercsr_rows(d["indptr"], d["cols"], n)
        else:  # pragma: no cover
            raise RuntimeError(f"Invalid format: {fmt}")
    finally:
        rebuilt = gb.Matrix.ss.import_any(take_ownership=True, name="", **d)
        # We need to set rebuilt.gb_obj to NULL so it doesn't get deleted early, so might
        # as well do a swap, b/c matrix.gb_obj is already "destroyed" from the export.
        matrix.gb_obj, rebuilt.gb_obj = rebuilt.gb_obj, matrix.gb_obj
    return rows, cols, vals

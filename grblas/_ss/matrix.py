import warnings
from numbers import Integral, Number

import numba
import numpy as np
from numba import njit
from suitesparse_graphblas.utils import claim_buffer, claim_buffer_2d, unclaim_buffer

import grblas as gb

from .. import ffi, lib, monoid
from ..base import call, record_raw
from ..dtypes import _INDEX, INT64, lookup_dtype
from ..exceptions import check_status, check_status_carg
from ..scalar import Scalar, _as_scalar
from ..utils import (
    _CArray,
    _Pointer,
    get_shape,
    ints_to_numpy_buffer,
    libget,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)
from .utils import get_order

ffi_new = ffi.new


@njit
def _head_matrix_full(values, nrows, ncols, dtype, n, is_iso):  # pragma: no cover
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    if is_iso:
        vals = np.empty(1, dtype=dtype)
        vals[0] = values[0]
    else:
        vals = np.empty(n, dtype=dtype)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            rows[k] = i
            cols[k] = j
            if not is_iso:
                vals[k] = values[i * ncols + j]
            k += 1
            if k == n:
                return rows, cols, vals
    return rows, cols, vals


@njit
def _head_matrix_bitmap(bitmap, values, nrows, ncols, dtype, n, is_iso):  # pragma: no cover
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    if is_iso:
        vals = np.empty(1, dtype=dtype)
        vals[0] = values[0]
    else:
        vals = np.empty(n, dtype=dtype)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if bitmap[i * ncols + j]:
                rows[k] = i
                cols[k] = j
                if not is_iso:
                    vals[k] = values[i * ncols + j]
                k += 1
                if k == n:
                    return rows, cols, vals
    return rows, cols, vals


@njit
def _head_csr_rows(indptr, n):  # pragma: no cover
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
def _head_hypercsr_rows(indptr, rows, n):  # pragma: no cover
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


def head(matrix, n=10, dtype=None, *, sort=False):
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
    is_iso = matrix.ss.is_iso
    d = matrix.ss.unpack(raw=True, sort=sort)
    try:
        fmt = d["format"]
        if fmt == "fullr":
            rows, cols, vals = _head_matrix_full(
                d["values"], d["nrows"], d["ncols"], dtype.np_type, n, is_iso
            )
        elif fmt == "fullc":
            cols, rows, vals = _head_matrix_full(
                d["values"], d["ncols"], d["nrows"], dtype.np_type, n, is_iso
            )
        elif fmt == "bitmapr":
            rows, cols, vals = _head_matrix_bitmap(
                d["bitmap"], d["values"], d["nrows"], d["ncols"], dtype.np_type, n, is_iso
            )
        elif fmt == "bitmapc":
            cols, rows, vals = _head_matrix_bitmap(
                d["bitmap"], d["values"], d["ncols"], d["nrows"], dtype.np_type, n, is_iso
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
        matrix.ss.pack_any(take_ownership=True, **d)
    if is_iso:
        vals = np.broadcast_to(vals[:1], (n,))
    return rows, cols, vals


def normalize_chunks(chunks, shape):
    """Normalize chunks argument for use by `Matrix.ss.split`.

    Examples
    --------
    >>> shape = (10, 20)
    >>> normalize_chunks(10, shape)
    [(10,), (10, 10)]
    >>> normalize_chunks((10, 10), shape)
    [(10,), (10, 10)]
    >>> normalize_chunks([None, (5, 15)], shape)
    [(10,), (5, 15)]
    >>> normalize_chunks((5, (5, None)), shape)
    [(5, 5), (5, 15)]
    """
    if isinstance(chunks, (list, tuple)):
        pass
    elif isinstance(chunks, Number):
        chunks = (chunks,) * len(shape)
    elif isinstance(chunks, np.ndarray):
        chunks = chunks.tolist()
    else:
        raise TypeError(
            f"chunks argument must be a list, tuple, or numpy array; got: {type(chunks)}"
        )
    if len(chunks) != len(shape):
        typ = "Vector" if len(shape) == 1 else "Matrix"
        raise ValueError(
            f"chunks argument must be of length {len(shape)} (one for each dimension of a {typ})"
        )
    chunksizes = []
    for size, chunk in zip(shape, chunks):
        if chunk is None:
            cur_chunks = [size]
        elif isinstance(chunk, Integral) or isinstance(chunk, float) and chunk.is_integer():
            chunk = int(chunk)
            if chunk < 0:
                raise ValueError(f"Chunksize must be greater than 0; got: {chunk}")
            div, mod = divmod(size, chunk)
            cur_chunks = [chunk] * div
            if mod:
                cur_chunks.append(mod)
        elif isinstance(chunk, (list, tuple)):
            cur_chunks = []
            none_index = None
            for c in chunk:
                if isinstance(c, Integral) or isinstance(c, float) and c.is_integer():
                    c = int(c)
                    if c < 0:
                        raise ValueError(f"Chunksize must be greater than 0; got: {c}")
                elif c is None:
                    if none_index is not None:
                        raise TypeError(
                            'None value in chunks for "the rest" can only appear once per dimension'
                        )
                    none_index = len(cur_chunks)
                    c = 0
                else:
                    raise TypeError(
                        "Bad type for element in chunks; expected int or None, but got: "
                        f"{type(chunks)}"
                    )
                cur_chunks.append(c)
            if none_index is not None:
                fill = size - sum(cur_chunks)
                if fill < 0:
                    raise ValueError(
                        "Chunks are too large; None value in chunks would need to be negative "
                        "to match size of input"
                    )
                cur_chunks[none_index] = fill
        elif isinstance(chunk, np.ndarray):
            if not np.issubdtype(chunk.dtype, np.integer):
                raise TypeError(f"numpy array for chunks must be integer dtype; got {chunk.dtype}")
            if chunk.ndim != 1:
                raise TypeError(
                    f"numpy array for chunks must be 1-dimension; got ndim={chunk.ndim}"
                )
            if (chunk < 0).any():
                raise ValueError(f"Chunksize must be greater than 0; got: {chunk[chunk < 0]}")
            cur_chunks = chunk.tolist()
        else:
            raise TypeError(
                "Chunks for a dimension must be an integer, a list or tuple of integers, or None."
                f"  Got: {type(chunk)}"
            )
        chunksizes.append(cur_chunks)
    return chunksizes


def _concat_mn(tiles, *, is_matrix=None):
    """Argument checking for `Matrix.ss.concat` and returns number of tiles in each dimension"""
    from ..matrix import Matrix, TransposedMatrix
    from ..scalar import Scalar
    from ..vector import Vector

    valid_types = (Matrix, TransposedMatrix, Vector, Scalar)
    if not isinstance(tiles, (list, tuple)):
        raise TypeError(f"tiles argument must be list or tuple; got: {type(tiles)}")
    if not tiles:
        raise ValueError("tiles argument must not be empty")
    dummy = Matrix.__new__(Matrix)
    m = len(tiles)
    n = None
    new_tiles = []
    for i, row_tiles in enumerate(tiles):
        if not isinstance(row_tiles, (list, tuple)):
            if not is_matrix and output_type(row_tiles) in {Vector, Scalar}:
                new_tiles.append(
                    dummy._expect_type(
                        row_tiles, (Vector, Scalar), within="ss.concat", argname="tiles"
                    )._as_matrix()
                )
                is_matrix = False
                n = 1
                continue
            raise TypeError(f"tiles must be lists or tuples; got: {type(row_tiles)}")
        if is_matrix is False:
            raise TypeError(f"Bad tile type for concat at position {i}")
        if n is None:
            n = len(row_tiles)
            if n == 0:
                raise ValueError("tiles must not be empty")
        elif len(row_tiles) != n:
            raise ValueError(
                f"tiles must all be the same length; got tiles of length {n} and "
                f"{len(row_tiles)}"
            )
        new_tiles.append(
            [
                _as_matrix(
                    dummy._expect_type(
                        tile,
                        valid_types,
                        within="ss.concat",
                        argname="tiles",
                        extra_message=f"Bad tile type for concat at position [{i}, {j}]",
                    )
                )
                for j, tile in enumerate(row_tiles)
            ]
        )
        is_matrix = True
    return new_tiles, m, n, is_matrix


def _as_matrix(x):
    return x._as_matrix() if hasattr(x, "_as_matrix") else x


class MatrixArray:
    __slots__ = "_carg", "_exc_arg", "name"

    def __init__(self, matrices, exc_arg=None, *, name):
        self._carg = matrices
        self._exc_arg = exc_arg
        self.name = name
        record_raw(f"GrB_Matrix {name}[{len(matrices)}];")


class ss:
    __slots__ = "_parent"

    def __init__(self, parent):
        self._parent = parent

    @property
    def nbytes(self):
        size = ffi_new("size_t*")
        check_status(lib.GxB_Matrix_memoryUsage(size, self._parent._carg), self._parent)
        return size[0]

    @property
    def is_iso(self):
        is_iso = ffi_new("bool*")
        check_status(lib.GxB_Matrix_iso(is_iso, self._parent._carg), self._parent)
        return is_iso[0]

    @property
    def iso_value(self):
        if self.is_iso:
            return self._parent.reduce_scalar(monoid.any).new(name="")
        raise ValueError("Matrix is not iso-valued")

    @property
    def format(self):
        # Determine current format
        parent = self._parent
        format_ptr = ffi_new("GxB_Option_Field*")
        sparsity_ptr = ffi_new("GxB_Option_Field*")
        check_status(
            lib.GxB_Matrix_Option_get(parent._carg, lib.GxB_FORMAT, format_ptr),
            parent,
        )
        check_status(
            lib.GxB_Matrix_Option_get(parent._carg, lib.GxB_SPARSITY_STATUS, sparsity_ptr),
            parent,
        )
        sparsity_status = sparsity_ptr[0]
        if sparsity_status == lib.GxB_HYPERSPARSE:
            format = "hypercs"
        elif sparsity_status == lib.GxB_SPARSE:
            format = "cs"
        elif sparsity_status == lib.GxB_BITMAP:
            format = "bitmap"
        elif sparsity_status == lib.GxB_FULL:
            format = "full"
        else:  # pragma: no cover
            raise NotImplementedError(f"Unknown sparsity status: {sparsity_status}")
        if format_ptr[0] == lib.GxB_BY_COL:
            format = f"{format}c"
        else:
            format = f"{format}r"
        return format

    @property
    def orientation(self):
        parent = self._parent
        format_ptr = ffi_new("GxB_Option_Field*")
        check_status(
            lib.GxB_Matrix_Option_get(parent._carg, lib.GxB_FORMAT, format_ptr),
            parent,
        )
        if format_ptr[0] == lib.GxB_BY_COL:
            return "columnwise"
        else:
            return "rowwise"

    def diag(self, vector, k=0):
        """
        GxB_Matrix_diag

        **This function is deprecated.  Use ``Vector.diag`` or ``Matrix.ss.build_diag`` instead.**

        """
        warnings.warn(
            "`Matrix.ss.diag` is deprecated; "
            "please use `Vector.diag` or `Matrix.ss.build_diag` instead",
            DeprecationWarning,
        )
        self.build_diag(vector, k)

    def build_diag(self, vector, k=0):
        """
        GxB_Matrix_diag

        Construct a diagonal Matrix from the given vector.
        Existing entries in the Matrix are discarded.

        Parameters
        ----------
        vector : Vector
            Create a diagonal from this Vector.
        k : int, default 0
            Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
            and `k<0` for diagonals below the main diagonal.

        See Also
        --------
        Matrix.diag
        Vector.diag

        """
        vector = self._parent._expect_type(
            vector, gb.Vector, within="ss.build_diag", argname="vector"
        )
        call("GxB_Matrix_diag", [self._parent, vector, _as_scalar(k, INT64, is_cscalar=True), None])

    def split(self, chunks, *, name=None):
        """
        GxB_Matrix_split

        Split a Matrix into a 2D array of sub-matrices according to `chunks`.

        This performs the opposite operation as ``concat``.

        `chunks` is short for "chunksizes" and indicates the chunk sizes for each dimension.
        `chunks` may be a single integer, or a length 2 tuple or list.  Example chunks:

        - ``chunks=10``
            - Split each dimension into chunks of size 10 (the last chunk may be smaller).
        - ``chunks=(10, 20)``
            - Split rows into chunks of size 10 and columns into chunks of size 20.
        - ``chunks=(None, [5, 10])``
            - Don't split rows into chunks, and split columns into two chunks of size 5 and 10.
        ` ``chunks=(10, [20, None])``
            - Split columns into two chunks of size 20 and ``ncols - 20``

        See Also
        --------
        Matrix.ss.concat
        grblas.ss.concat
        """
        from ..matrix import Matrix

        tile_nrows, tile_ncols = normalize_chunks(chunks, self._parent.shape)
        m = len(tile_nrows)
        n = len(tile_ncols)
        tiles = ffi.new("GrB_Matrix[]", m * n)
        call(
            "GxB_Matrix_split",
            [
                MatrixArray(tiles, self._parent, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(n, _INDEX, is_cscalar=True),
                _CArray(tile_nrows),
                _CArray(tile_ncols),
                self._parent,
                None,
            ],
        )
        rv = []
        dtype = self._parent.dtype
        if name is None:
            name = self._parent.name
        index = 0
        for i, nrows in enumerate(tile_nrows):
            cur = []
            for j, ncols in enumerate(tile_ncols):
                # Copy to a new handle so we can free `tiles`
                new_matrix = ffi.new("GrB_Matrix*")
                new_matrix[0] = tiles[index]
                tile = Matrix(new_matrix, dtype, name=f"{name}_{i}x{j}")
                tile._nrows = nrows
                tile._ncols = ncols
                cur.append(tile)
                index += 1
            rv.append(cur)
        return rv

    def _concat(self, tiles, m, n):
        from ..matrix import TransposedMatrix

        ctiles = ffi.new("GrB_Matrix[]", m * n)
        index = 0
        for row_tiles in tiles:
            for j, tile in enumerate(row_tiles):
                if type(tile) is TransposedMatrix:
                    tile = row_tiles[j] = tile.new()
                ctiles[index] = tile.gb_obj[0]
                index += 1
        call(
            "GxB_Matrix_concat",
            [
                self._parent,
                MatrixArray(ctiles, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(n, _INDEX, is_cscalar=True),
                None,
            ],
        )

    def concat(self, tiles):
        """
        GxB_Matrix_concat

        Concatenate a 2D list of Matrix objects into the current Matrix.
        Any existing values in the current Matrix will be discarded.
        To concatenate into a new Matrix, use `grblas.ss.concat`.

        Vectors may be used as `Nx1` Matrix objects.

        This performs the opposite operation as ``split``.

        See Also
        --------
        Matrix.ss.split
        grblas.ss.concat
        """
        tiles, m, n, is_matrix = _concat_mn(tiles, is_matrix=True)
        self._concat(tiles, m, n)

    def build_scalar(self, rows, columns, value):
        """
        GxB_Matrix_build_Scalar

        Like ``build``, but uses a scalar for all the values.

        See Also
        --------
        Matrix.build
        Matrix.from_values
        """
        rows = ints_to_numpy_buffer(rows, np.uint64, name="row indices")
        columns = ints_to_numpy_buffer(columns, np.uint64, name="column indices")
        if rows.size != columns.size:
            raise ValueError(
                f"`rows` and `columns` lengths must match: {rows.size}, {columns.size}"
            )
        scalar = _as_scalar(value, self._parent.dtype, is_cscalar=False)  # pragma: is_grbscalar
        call(
            "GxB_Matrix_build_Scalar",
            [
                self._parent,
                _CArray(rows),
                _CArray(columns),
                scalar,
                _as_scalar(rows.size, _INDEX, is_cscalar=True),
            ],
        )

    def export(self, format=None, *, sort=False, give_ownership=False, raw=False):
        """
        GxB_Matrix_export_xxx

        Parameters
        ----------
        format : str, optional
            If `format` is not specified, this method exports in the currently stored format.
            To control the export format, set `format` to one of:
                - "csr"
                - "csc"
                - "hypercsr"
                - "hypercsc"
                - "bitmapr"
                - "bitmapc"
                - "fullr"
                - "fullc"
                - "coor"
                - "cooc"
                - "coo"
                - "rowwise"
                - "columnwise"
            The last five ("coor", "cooc", "coo", "rowwise", "columnwise") are not native
            SuiteSparse formats.  "coor", "cooc", and "coo" export in coordinate formats.
            "coor" and "cooc" will always sort the row and column indices respectively.
            "rowwise" will export to "csr", "hypercsr", "bitmapr", or "fullr".
            "columnwise" will export to "csc", "hypercsc", "bitmapc", or "fullc".
        sort : bool, default False
            Whether to sort indices if the format is "csr", "csc", "hypercsr", "hypercsc",
            "coo", "coor", or "cooc".
        give_ownership : bool, default False
            Perform a zero-copy data transfer to Python if possible.  This gives ownership of
            the underlying memory buffers to Numpy.
            ** If True, this nullifies the current object, which should no longer be used! **
        raw : bool, default False
            If True, always return 1d arrays the same size as returned by SuiteSparse.
            If False, arrays may be trimmed to be the expected size, and 2d arrays are
            returned when format is "bitmapr", "bitmapc", "fullr", or "fullc".
            It may make sense to choose ``raw=True`` if one wants to use the data to perform
            a zero-copy import back to SuiteSparse.

        Returns
        -------
        dict; keys depend on `format` and `raw` arguments (see below).

        See Also
        --------
        Matrix.to_values
        Matrix.ss.import_any

        Return values
            - Note: for ``raw=True``, arrays may be larger than specified.
            - "csr" format
                - indptr : ndarray(dtype=uint64, ndim=1, size=nrows + 1)
                - col_indices : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - sorted_cols : bool
                    - True if the values in "col_indices" are sorted
                - nrows : int
                - ncols : int
            - "csc" format
                - indptr : ndarray(dtype=uint64, ndim=1, size=ncols + 1)
                - row_indices : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - sorted_rows : bool
                    - True if the values in "row_indices" are sorted
                - nrows : int
                - ncols : int
            - "hypercsr" format
                - indptr : ndarray(dtype=uint64, ndim=1, size=nvec + 1)
                - rows : ndarray(dtype=uint64, ndim=1, size=nvec)
                - col_indices : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - sorted_cols : bool
                    - True if the values in "col_indices" are sorted
                - nrows : int
                - ncols : int
                - nvec : int, only present if raw == True
                    - The number of rows present in the data structure
            - "hypercsc" format
                - indptr : ndarray(dtype=uint64, ndim=1, size=nvec + 1)
                - cols : ndarray(dtype=uint64, ndim=1, size=nvec)
                - row_indices : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - sorted_rows : bool
                    - True if the values in "row_indices" are sorted
                - nrows : int
                - ncols : int
                - nvec : int, only present if raw == True
                    - The number of cols present in the data structure
            - "bitmapr" format
                - ``raw=False``
                    - bitmap : ndarray(dtype=bool8, ndim=2, shape=(nrows, ncols), order="C")
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="C")
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                - ``raw=True``
                    - bitmap : ndarray(dtype=bool8, ndim=1, size=nrows * ncols)
                        - Stored row-oriented
                    - values : ndarray(ndim=1, size=nrows * ncols)
                        - Elements where bitmap is False are undefined
                        - Stored row-oriented
                    - nvals : int
                        - The number of True elements in the bitmap
                    - nrows : int
                    - ncols : int
            - "bitmapc" format
                - ``raw=False``
                    - bitmap : ndarray(dtype=bool8, ndim=2, shape=(nrows, ncols), order="F")
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="F")
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                - ``raw=True``
                    - bitmap : ndarray(dtype=bool8, ndim=1, size=nrows * ncols)
                        - Stored column-oriented
                    - values : ndarray(ndim=1, size=nrows * ncols)
                        - Elements where bitmap is False are undefined
                        - Stored column-oriented
                    - nvals : int
                        - The number of True elements in the bitmap
                    - nrows : int
                    - ncols : int
            - "fullr" format
                - ``raw=False``
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="C")
                - ``raw=False, is_iso=True``
                    - values : ndarray(ndim=1, size=1)
                    - nrows : int
                    - ncols : int
                - ``raw=True``
                    - values : ndarray(ndim=1, size=nrows * ncols)
                        - Stored row-oriented
                    - nrows : int
                    - ncols : int
            - "fullc" format
                - ``raw=False``
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="F")
                - ``raw=True``
                    - values : ndarray(ndim=1, size=nrows * ncols)
                        - Stored row-oriented
                    - nrows : int
                    - ncols : int
            - "coor" format
                - rows : ndarray(dtype=uint64, ndim=1, size=nvals)
                    - Always sorted
                - cols : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - nrows : int
                - ncols : int
                - sorted_rows : True
                - sorted_cols : bool
                    - True if the values in "cols" are sorted
            - "cooc" format
                - rows : ndarray(dtype=uint64, ndim=1, size=nvals)
                - cols : ndarray(dtype=uint64, ndim=1, size=nvals)
                    - Always sorted
                - values : ndarray(ndim=1, size=nvals)
                - nrows : int
                - ncols : int
                - sorted_rows : bool
                    - True if the values in "rows" are sorted
                - sorted_cols : True
            - "coo" format
                - Note: exporting "coo" will return "coor" or "cooc" format if ``sort=True``
                - rows : ndarray(dtype=uint64, ndim=1, size=nvals)
                - cols : ndarray(dtype=uint64, ndim=1, size=nvals)
                - values : ndarray(ndim=1, size=nvals)
                - nrows : int
                - ncols : int
                - sorted_rows : bool
                - sorted_cols : bool
                    - Both sorted_rows and sorted_cols may be True, which means the
                      arrays are sorted in lexicographic order, but we don't know if
                      it's by rows then columns ("coor"), or columns then rows ("cooc").

        Examples
        --------
        Simple usage:

        >>> pieces = A.ss.export()
        >>> A2 = Matrix.ss.import_any(**pieces)
        """
        return self._export(
            format, sort=sort, give_ownership=give_ownership, raw=raw, method="export"
        )

    def unpack(self, format=None, *, sort=False, raw=False):
        """
        GxB_Matrix_unpack_xxx

        `unpack` is like `export`, except that the Matrix remains valid but empty.
        `pack_*` methods are the opposite of `unpack`.

        See `Matrix.ss.export` documentation for more details.
        """
        return self._export(format, sort=sort, raw=raw, give_ownership=True, method="unpack")

    def _export(self, format=None, *, sort=False, give_ownership=False, raw=False, method):
        if format is None:
            format = self.format
        else:
            format = format.lower()
            if format == "rowwise":
                format = f"{self.format[:-1]}r"
            elif format == "columnwise":
                format = f"{self.format[:-1]}c"
        if give_ownership or format == "coo":
            parent = self._parent
        else:
            parent = self._parent.dup(name=f"M_{method}")
        dtype = np.dtype(parent.dtype.np_type)
        index_dtype = np.dtype(np.uint64)

        nrows = parent._nrows
        ncols = parent._ncols
        if format.startswith("coo"):
            if format == "coo":
                if sort:
                    # It's weird, but waiting makes values sorted (according to the
                    # storage orientation) If we don't wait, we don't know whether
                    # the values are sorted or not.
                    parent.wait()
                if self.is_iso:
                    # Should we expose a way to do `to_values` without values?
                    # Passing NULL for values is SuiteSparse-specific.
                    nvals = parent._nvals
                    rows = _CArray(size=nvals, name="&rows_array")
                    columns = _CArray(size=nvals, name="&columns_array")
                    n = ffi_new("GrB_Index*")
                    scalar = Scalar(n, _INDEX, name="s_nvals", is_cscalar=True, empty=True)
                    scalar.value = nvals
                    call(
                        f"GrB_Matrix_extractTuples_{parent.dtype.name}",
                        [rows, columns, None, _Pointer(scalar), parent],
                    )
                    value = parent.reduce_scalar(gb.monoid.any, allow_empty=False).new().value
                    rv = {
                        "format": "coo",
                        "nrows": nrows,
                        "ncols": ncols,
                        "rows": rows.array,
                        "cols": columns.array,
                        "values": np.array([value], dtype=dtype),
                        "is_iso": True,
                    }
                else:
                    rows, columns, values = parent.to_values()
                    rv = {
                        "format": "coo",
                        "nrows": nrows,
                        "ncols": ncols,
                        "rows": rows,
                        "cols": columns,
                        "values": values,
                        "is_iso": False,
                    }

                # If rowwise, rows is probably always sorted (but I'm not 100% certain)
                rv["sorted_rows"] = sort
                rv["sorted_cols"] = sort
                if sort:
                    if self.orientation == "rowwise":
                        rv["format"] += "r"
                    else:
                        rv["format"] += "c"
                if give_ownership:
                    if method == "export":
                        parent.__del__()
                        parent.gb_obj = ffi.NULL
                    else:
                        parent.clear()
                return rv
            elif format == "coor":
                info = self._export(
                    "csr", sort=sort, give_ownership=give_ownership, raw=False, method=method
                )
                info["rows"] = indptr_to_indices(info.pop("indptr"))
                info["cols"] = info.pop("col_indices")
                info["sorted_rows"] = True
                info["format"] = "coor"
                return info
            elif format == "cooc":
                info = self._export(
                    "csc", sort=sort, give_ownership=give_ownership, raw=False, method=method
                )
                info["cols"] = indptr_to_indices(info.pop("indptr"))
                info["rows"] = info.pop("row_indices")
                info["sorted_cols"] = True
                info["format"] = "cooc"
                return info

        if method == "export":
            mhandle = ffi_new("GrB_Matrix*", parent._carg)
            type_ = ffi_new("GrB_Type*")
            nrows_ = ffi_new("GrB_Index*")
            ncols_ = ffi_new("GrB_Index*")
            args = (type_, nrows_, ncols_)
        else:
            mhandle = parent._carg
            args = ()
        Ap = ffi_new("GrB_Index**")
        Ax = ffi_new("void**")
        Ap_size = ffi_new("GrB_Index*")
        Ax_size = ffi_new("GrB_Index*")
        if sort:
            jumbled = ffi.NULL
        else:
            jumbled = ffi_new("bool*")
        is_iso = ffi_new("bool*")
        nvals = parent._nvals
        if format == "csr":
            Aj = ffi_new("GrB_Index**")
            Aj_size = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Matrix_{method}_CSR")(
                    mhandle,
                    *args,
                    Ap,
                    Aj,
                    Ax,
                    Ap_size,
                    Aj_size,
                    Ax_size,
                    is_iso,
                    jumbled,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            col_indices = claim_buffer(ffi, Aj[0], Aj_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indptr.size > nrows + 1:  # pragma: no cover
                    indptr = indptr[: nrows + 1]
                if col_indices.size > nvals:  # pragma: no cover
                    col_indices = col_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
            # Note: nvals is also at `indptr[nrows]`
            rv = {
                "indptr": indptr,
                "col_indices": col_indices,
                "sorted_cols": True if sort else not jumbled[0],
                "nrows": nrows,
                "ncols": ncols,
            }
        elif format == "csc":
            Ai = ffi_new("GrB_Index**")
            Ai_size = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Matrix_{method}_CSC")(
                    mhandle,
                    *args,
                    Ap,
                    Ai,
                    Ax,
                    Ap_size,
                    Ai_size,
                    Ax_size,
                    is_iso,
                    jumbled,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            row_indices = claim_buffer(ffi, Ai[0], Ai_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indptr.size > ncols + 1:  # pragma: no cover
                    indptr = indptr[: ncols + 1]
                if row_indices.size > nvals:  # pragma: no cover
                    row_indices = row_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
            # Note: nvals is also at `indptr[ncols]`
            rv = {
                "indptr": indptr,
                "row_indices": row_indices,
                "sorted_rows": True if sort else not jumbled[0],
                "nrows": nrows,
                "ncols": ncols,
            }
        elif format == "hypercsr":
            nvec = ffi_new("GrB_Index*")
            Ah = ffi_new("GrB_Index**")
            Aj = ffi_new("GrB_Index**")
            Ah_size = ffi_new("GrB_Index*")
            Aj_size = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Matrix_{method}_HyperCSR")(
                    mhandle,
                    *args,
                    Ap,
                    Ah,
                    Aj,
                    Ax,
                    Ap_size,
                    Ah_size,
                    Aj_size,
                    Ax_size,
                    is_iso,
                    nvec,
                    jumbled,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            rows = claim_buffer(ffi, Ah[0], Ah_size[0] // index_dtype.itemsize, index_dtype)
            col_indices = claim_buffer(ffi, Aj[0], Aj_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            nvec = nvec[0]
            if not raw:
                if indptr.size > nvec + 1:  # pragma: no cover
                    indptr = indptr[: nvec + 1]
                if rows.size > nvec:  # pragma: no cover
                    rows = rows[:nvec]
                if col_indices.size > nvals:  # pragma: no cover
                    col_indices = col_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
            # Note: nvals is also at `indptr[nvec]`
            rv = {
                "indptr": indptr,
                "rows": rows,
                "col_indices": col_indices,
                "sorted_cols": True if sort else not jumbled[0],
                "nrows": nrows,
                "ncols": ncols,
            }
            if raw:
                rv["nvec"] = nvec
        elif format == "hypercsc":
            nvec = ffi_new("GrB_Index*")
            Ah = ffi_new("GrB_Index**")
            Ai = ffi_new("GrB_Index**")
            Ah_size = ffi_new("GrB_Index*")
            Ai_size = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Matrix_{method}_HyperCSC")(
                    mhandle,
                    *args,
                    Ap,
                    Ah,
                    Ai,
                    Ax,
                    Ap_size,
                    Ah_size,
                    Ai_size,
                    Ax_size,
                    is_iso,
                    nvec,
                    jumbled,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            cols = claim_buffer(ffi, Ah[0], Ah_size[0] // index_dtype.itemsize, index_dtype)
            row_indices = claim_buffer(ffi, Ai[0], Ai_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            nvec = nvec[0]
            if not raw:
                if indptr.size > nvec + 1:  # pragma: no cover
                    indptr = indptr[: nvec + 1]
                if cols.size > nvec:  # pragma: no cover
                    cols = cols[:nvec]
                if row_indices.size > nvals:  # pragma: no cover
                    row_indices = row_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
            # Note: nvals is also at `indptr[nvec]`
            rv = {
                "indptr": indptr,
                "cols": cols,
                "row_indices": row_indices,
                "sorted_rows": True if sort else not jumbled[0],
                "nrows": nrows,
                "ncols": ncols,
            }
            if raw:
                rv["nvec"] = nvec
        elif format == "bitmapr" or format == "bitmapc":
            if format == "bitmapr":
                cfunc = libget(f"GxB_Matrix_{method}_BitmapR")
            else:
                cfunc = libget(f"GxB_Matrix_{method}_BitmapC")
            Ab = ffi_new("int8_t**")
            Ab_size = ffi_new("GrB_Index*")
            nvals_ = ffi_new("GrB_Index*")
            check_status(
                cfunc(
                    mhandle,
                    *args,
                    Ab,
                    Ax,
                    Ab_size,
                    Ax_size,
                    is_iso,
                    nvals_,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            bool_dtype = np.dtype(np.bool8)
            if raw:
                bitmap = claim_buffer(ffi, Ab[0], Ab_size[0] // bool_dtype.itemsize, bool_dtype)
                values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            else:
                is_c_order = format == "bitmapr"
                bitmap = claim_buffer_2d(
                    ffi,
                    Ab[0],
                    Ab_size[0] // bool_dtype.itemsize,
                    nrows,
                    ncols,
                    bool_dtype,
                    is_c_order,
                )
                if is_iso:
                    values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    values = claim_buffer_2d(
                        ffi,
                        Ax[0],
                        Ax_size[0] // dtype.itemsize,
                        nrows,
                        ncols,
                        dtype,
                        is_c_order,
                    )
            rv = {"bitmap": bitmap, "nvals": nvals_[0]}
            if raw:
                rv["nrows"] = nrows
                rv["ncols"] = ncols
        elif format == "fullr" or format == "fullc":
            if format == "fullr":
                cfunc = libget(f"GxB_Matrix_{method}_FullR")
            else:
                cfunc = libget(f"GxB_Matrix_{method}_FullC")
            check_status(
                cfunc(
                    mhandle,
                    *args,
                    Ax,
                    Ax_size,
                    is_iso,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            if raw:
                values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
                rv = {"nrows": nrows, "ncols": ncols}
            elif is_iso:
                values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
                if values.size > 1:  # pragma: no cover
                    values = values[:1]
                rv = {"nrows": nrows, "ncols": ncols}
            else:
                is_c_order = format == "fullr"
                values = claim_buffer_2d(
                    ffi, Ax[0], Ax_size[0] // dtype.itemsize, nrows, ncols, dtype, is_c_order
                )
                rv = {}
        else:
            raise ValueError(f"Invalid format: {format}")

        rv["is_iso"] = is_iso
        rv["format"] = format
        rv["values"] = values
        if method == "export":
            parent.gb_obj = ffi.NULL
        return rv

    @classmethod
    def import_csr(
        cls,
        *,
        nrows,
        ncols,
        indptr,
        values,
        col_indices,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_CSR

        Create a new Matrix from standard CSR format.

        Parameters
        ----------
        nrows : int
        ncols : int
        indptr : array-like
        values : array-like
        col_indices : array-like
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_cols : bool, default False
            Indicate whether the values in "col_indices" are sorted.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indptr and col_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "csr" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_csr(
            nrows=nrows,
            ncols=ncols,
            indptr=indptr,
            values=values,
            col_indices=col_indices,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_csr(
        self,
        *,
        indptr,
        values,
        col_indices,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        format=None,
        **ignored_kwargs,
    ):
        """
        GxB_Matrix_pack_CSR

        `pack_csr` is like `import_csr` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("csr")``

        See `Matrix.ss.import_csr` documentation for more details.
        """
        return self._import_csr(
            indptr=indptr,
            values=values,
            col_indices=col_indices,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_csr(
        cls,
        *,
        nrows=None,
        ncols=None,
        indptr,
        values,
        col_indices,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "csr":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'csr'.")
        copy = not take_ownership
        indptr = ints_to_numpy_buffer(
            indptr, np.uint64, copy=copy, ownable=True, name="index pointers"
        )
        col_indices = ints_to_numpy_buffer(
            col_indices, np.uint64, copy=copy, ownable=True, name="column indices"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if col_indices is values:
            values = np.copy(values)
        Ap = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indptr))
        Aj = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", col_indices))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_CSR")(
            mhandle,
            *args,
            Ap,
            Aj,
            Ax,
            indptr.nbytes,
            col_indices.nbytes,
            values.nbytes,
            is_iso,
            not sorted_cols,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(indptr)
        unclaim_buffer(col_indices)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_csc(
        cls,
        *,
        nrows,
        ncols,
        indptr,
        values,
        row_indices,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_CSC

        Create a new Matrix from standard CSC format.

        Parameters
        ----------
        nrows : int
        ncols : int
        indptr : array-like
        values : array-like
        row_indices : array-like
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_rows : bool, default False
            Indicate whether the values in "row_indices" are sorted.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indptr and row_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "csc" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_csc(
            nrows=nrows,
            ncols=ncols,
            indptr=indptr,
            values=values,
            row_indices=row_indices,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_csc(
        self,
        *,
        indptr,
        values,
        row_indices,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        format=None,
        **ignored_kwargs,
    ):
        """
        GxB_Matrix_pack_CSC

        `pack_csc` is like `import_csc` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("csc")``

        See `Matrix.ss.import_csc` documentation for more details.
        """
        return self._import_csc(
            indptr=indptr,
            values=values,
            row_indices=row_indices,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_csc(
        cls,
        *,
        nrows=None,
        ncols=None,
        indptr,
        values,
        row_indices,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "csc":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'csc'.")
        copy = not take_ownership
        indptr = ints_to_numpy_buffer(
            indptr, np.uint64, copy=copy, ownable=True, name="index pointers"
        )
        row_indices = ints_to_numpy_buffer(
            row_indices, np.uint64, copy=copy, ownable=True, name="row indices"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if row_indices is values:
            values = np.copy(values)
        Ap = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indptr))
        Ai = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", row_indices))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_CSC")(
            mhandle,
            *args,
            Ap,
            Ai,
            Ax,
            indptr.nbytes,
            row_indices.nbytes,
            values.nbytes,
            is_iso,
            not sorted_rows,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(indptr)
        unclaim_buffer(row_indices)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_hypercsr(
        cls,
        *,
        nrows,
        ncols,
        rows,
        indptr,
        values,
        col_indices,
        nvec=None,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_HyperCSR

        Create a new Matrix from standard HyperCSR format.

        Parameters
        ----------
        nrows : int
        ncols : int
        rows : array-like
        indptr : array-like
        values : array-like
        col_indices : array-like
        nvec : int, optional
            The number of elements in "rows" to use.
            If not specified, will be set to ``len(rows)``.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_cols : bool, default False
            Indicate whether the values in "col_indices" are sorted.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for rows, indptr, col_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "hypercsr" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_hypercsr(
            nrows=nrows,
            ncols=ncols,
            rows=rows,
            indptr=indptr,
            values=values,
            col_indices=col_indices,
            nvec=nvec,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_hypercsr(
        self,
        *,
        rows,
        indptr,
        values,
        col_indices,
        nvec=None,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        format=None,
        **ignored_kwargs,
    ):
        """
        GxB_Matrix_pack_HyperCSR

        `pack_hypercsr` is like `import_hypercsr` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("hypercsr")``

        See `Matrix.ss.import_hypercsr` documentation for more details.
        """
        return self._import_hypercsr(
            rows=rows,
            indptr=indptr,
            values=values,
            col_indices=col_indices,
            nvec=nvec,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_hypercsr(
        cls,
        *,
        nrows=None,
        ncols=None,
        rows,
        indptr,
        values,
        col_indices,
        nvec=None,
        is_iso=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "hypercsr":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'hypercsr'.")
        copy = not take_ownership
        indptr = ints_to_numpy_buffer(
            indptr, np.uint64, copy=copy, ownable=True, name="index pointers"
        )
        rows = ints_to_numpy_buffer(rows, np.uint64, copy=copy, ownable=True, name="rows")
        col_indices = ints_to_numpy_buffer(
            col_indices, np.uint64, copy=copy, ownable=True, name="column indices"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if col_indices is values:
            values = np.copy(values)
        Ap = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indptr))
        Ah = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", rows))
        Aj = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", col_indices))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvec is None:
            nvec = rows.size
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_HyperCSR")(
            mhandle,
            *args,
            Ap,
            Ah,
            Aj,
            Ax,
            indptr.nbytes,
            rows.nbytes,
            col_indices.nbytes,
            values.nbytes,
            is_iso,
            nvec,
            not sorted_cols,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(indptr)
        unclaim_buffer(rows)
        unclaim_buffer(col_indices)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_hypercsc(
        cls,
        *,
        nrows,
        ncols,
        cols,
        indptr,
        values,
        row_indices,
        nvec=None,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_HyperCSC

        Create a new Matrix from standard HyperCSC format.

        Parameters
        ----------
        nrows : int
        ncols : int
        indptr : array-like
        values : array-like
        row_indices : array-like
        nvec : int, optional
            The number of elements in "cols" to use.
            If not specified, will be set to ``len(cols)``.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_rows : bool, default False
            Indicate whether the values in "row_indices" are sorted.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indptr and row_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "hypercsc" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_hypercsc(
            nrows=nrows,
            ncols=ncols,
            cols=cols,
            indptr=indptr,
            values=values,
            row_indices=row_indices,
            nvec=nvec,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_hypercsc(
        self,
        *,
        cols,
        indptr,
        values,
        row_indices,
        nvec=None,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        format=None,
        **ignored_kwargs,
    ):
        """
        GxB_Matrix_pack_HyperCSC

        `pack_hypercsc` is like `import_hypercsc` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("hypercsc")``

        See `Matrix.ss.import_hypercsc` documentation for more details.
        """
        return self._import_hypercsc(
            cols=cols,
            indptr=indptr,
            values=values,
            row_indices=row_indices,
            nvec=nvec,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_hypercsc(
        cls,
        *,
        nrows=None,
        ncols=None,
        cols,
        indptr,
        values,
        row_indices,
        nvec=None,
        is_iso=False,
        sorted_rows=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "hypercsc":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'hypercsc'.")
        copy = not take_ownership
        indptr = ints_to_numpy_buffer(
            indptr, np.uint64, copy=copy, ownable=True, name="index pointers"
        )
        cols = ints_to_numpy_buffer(cols, np.uint64, copy=copy, ownable=True, name="columns")
        row_indices = ints_to_numpy_buffer(
            row_indices, np.uint64, copy=copy, ownable=True, name="row indices"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if row_indices is values:
            values = np.copy(values)
        Ap = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indptr))
        Ah = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", cols))
        Ai = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", row_indices))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvec is None:
            nvec = cols.size
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_HyperCSC")(
            mhandle,
            *args,
            Ap,
            Ah,
            Ai,
            Ax,
            indptr.nbytes,
            cols.nbytes,
            row_indices.nbytes,
            values.nbytes,
            is_iso,
            nvec,
            not sorted_rows,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(indptr)
        unclaim_buffer(cols)
        unclaim_buffer(row_indices)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_bitmapr(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_BitmapR

        Create a new Matrix from values and bitmap (as mask) arrays.

        Parameters
        ----------
        bitmap : array-like
            True elements indicate where there are values in "values".
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        values : array-like
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        nvals : int, optional
            The number of True elements in the bitmap for this Matrix.
        nrows : int, optional
            The number of rows for the Matrix.
            If not provided, will be inferred from values or bitmap if either is 2d.
        ncols : int
            The number of columns for the Matrix.
            If not provided, will be inferred from values or bitmap if either is 2d.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (bool8 for bitmap)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "bitmapr" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_bitmapr(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_bitmapr(
        self,
        *,
        bitmap,
        values,
        nvals=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_BitmapR

        `pack_bitmapr` is like `import_bitmapr` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("bitmapr")``

        See `Matrix.ss.import_bitmapr` documentation for more details.
        """
        return self._import_bitmapr(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_bitmapr(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "bitmapr":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapr'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(
            bitmap, np.bool8, copy=copy, ownable=True, order="C", name="bitmap"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True, order="C")
        if bitmap is values:
            values = np.copy(values)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, values=values, bitmap=bitmap)
        else:
            nrows, ncols = matrix.shape
        Ab = ffi_new("int8_t**", ffi.from_buffer("int8_t*", bitmap))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvals is None:
            if bitmap.size == nrows * ncols:
                nvals = np.count_nonzero(bitmap)
            else:
                nvals = np.count_nonzero(bitmap.ravel()[: nrows * ncols])
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_BitmapR")(
            mhandle,
            *args,
            Ab,
            Ax,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(bitmap)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_bitmapc(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_BitmapC

        Create a new Matrix from values and bitmap (as mask) arrays.

        Parameters
        ----------
        bitmap : array-like
            True elements indicate where there are values in "values".
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        values : array-like
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        nvals : int, optional
            The number of True elements in the bitmap for this Matrix.
        nrows : int, optional
            The number of rows for the Matrix.
            If not provided, will be inferred from values or bitmap if either is 2d.
        ncols : int
            The number of columns for the Matrix.
            If not provided, will be inferred from values or bitmap if either is 2d.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be FORTRAN contiguous
                - have the correct dtype (bool8 for bitmap)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "bitmapc" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_bitmapc(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_bitmapc(
        self,
        *,
        bitmap,
        values,
        nvals=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_BitmapC

        `pack_bitmapc` is like `import_bitmapc` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("bitmapc")``

        See `Matrix.ss.import_bitmapc` documentation for more details.
        """
        return self._import_bitmapc(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_bitmapc(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "bitmapc":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapc'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(
            bitmap, np.bool8, copy=copy, ownable=True, order="F", name="bitmap"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True, order="F")
        if bitmap is values:
            values = np.copy(values)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, values=values, bitmap=bitmap)
        else:
            nrows, ncols = matrix.shape
        Ab = ffi_new("int8_t**", ffi.from_buffer("int8_t*", bitmap.T))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values.T))
        if nvals is None:
            if bitmap.size == nrows * ncols:
                nvals = np.count_nonzero(bitmap)
            else:
                nvals = np.count_nonzero(bitmap.ravel("F")[: nrows * ncols])
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_BitmapC")(
            mhandle,
            *args,
            Ab,
            Ax,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(bitmap)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_fullr(
        cls,
        values,
        *,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_FullR

        Create a new Matrix from values.

        Parameters
        ----------
        values : array-like
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        nrows : int, optional
            The number of rows for the Matrix.
            If not provided, will be inferred from values if it is 2d.
        ncols : int
            The number of columns for the Matrix.
            If not provided, will be inferred from values if it is 2d.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "fullr" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_fullr(
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_fullr(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_FullR

        `pack_fullr` is like `import_fullr` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("fullr")``

        See `Matrix.ss.import_fullr` documentation for more details.
        """
        return self._import_fullr(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_fullr(
        cls,
        *,
        values,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "fullr":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'fullr'.")
        copy = not take_ownership
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, order="C", ownable=True)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, values=values)
        else:
            nrows, ncols = matrix.shape

        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_FullR")(
            mhandle,
            *args,
            Ax,
            values.nbytes,
            is_iso,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_fullc(
        cls,
        values,
        *,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_FullC

        Create a new Matrix from values.

        Parameters
        ----------
        values : array-like
            May be 1d or 2d, but there need to have at least ``nrows*ncols`` elements.
        nrows : int, optional
            The number of rows for the Matrix.
            If not provided, will be inferred from values if it is 2d.
        ncols : int
            The number of columns for the Matrix.
            If not provided, will be inferred from values if it is 2d.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be FORTRAN contiguous
                - have the correct dtype
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "fullc" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_fullc(
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_fullc(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_FullC

        `pack_fullc` is like `import_fullc` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("fullc")``

        See `Matrix.ss.import_fullc` documentation for more details.
        """
        return self._import_fullc(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_fullc(
        cls,
        *,
        values,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "fullc":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'fullc'.")
        copy = not take_ownership
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, order="F", ownable=True)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, values=values)
        else:
            nrows, ncols = matrix.shape
        Ax = ffi_new("void**", ffi.from_buffer("void*", values.T))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        status = libget(f"GxB_Matrix_{method}_FullC")(
            mhandle,
            *args,
            Ax,
            values.nbytes,
            is_iso,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix(mhandle, dtype, name=name)
            matrix._nrows = nrows
            matrix._ncols = ncols
        else:
            check_status(status, matrix)
        unclaim_buffer(values)
        return matrix

    @classmethod
    def import_coo(
        cls,
        rows,
        cols,
        values,
        *,
        nrows,
        ncols,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GrB_Matrix_build_XXX and GxB_Matrix_build_Scalar

        Create a new Matrix from indices and values in coordinate format.

        Parameters
        ----------
        rows : array-like
        cols : array-like
        values : array-like
        nrows : int
            The number of rows for the Matrix.
        ncols : int
            The number of columns for the Matrix.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_rows : bool, default False
            True if rows are sorted or when (cols, rows) are sorted lexicographically
        sorted_cols : bool, default False
            True if cols are sorted or when (rows, cols) are sorted lexicographically
        take_ownership : bool, default False
            Ignored.  Zero-copy is not possible for "coo" format.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "coo" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_coo(
            rows=rows,
            cols=cols,
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_coo(
        self,
        rows,
        cols,
        values,
        *,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GrB_Matrix_build_XXX and GxB_Matrix_build_Scalar

        `pack_coo` is like `import_coo` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("coo")``

        See `Matrix.ss.import_coo` documentation for more details.
        """
        return self._import_coo(
            nrows=self._parent._nrows,
            ncols=self._parent._ncols,
            rows=rows,
            cols=cols,
            values=values,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_coo(
        cls,
        rows,
        cols,
        values,
        *,
        nrows=None,
        ncols=None,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "coo":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'coo'.")
        if sorted_rows and (not sorted_cols or issorted(rows)):
            return cls._import_coor(
                rows=rows,
                cols=cols,
                values=values,
                nrows=nrows,
                ncols=ncols,
                is_iso=is_iso,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
                method=method,
                matrix=matrix,
            )
        elif sorted_cols and (not sorted_rows or issorted(cols)):
            return cls._import_cooc(
                rows=rows,
                cols=cols,
                values=values,
                nrows=nrows,
                ncols=ncols,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
                method=method,
                matrix=matrix,
            )

        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype)
        if method == "import":
            matrix = gb.Matrix.new(dtype, nrows=nrows, ncols=ncols, name=name)
        if is_iso:
            matrix.ss.build_scalar(rows, cols, values[0])
        else:
            matrix.build(rows, cols, values)
        return matrix

    @classmethod
    def import_coor(
        cls,
        rows,
        cols,
        values,
        *,
        nrows,
        ncols,
        is_iso=False,
        sorted_rows=True,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_CSR

        Create a new Matrix from indices and values in coordinate format.
        Rows must be sorted.

        Parameters
        ----------
        rows : array-like
        cols : array-like
        values : array-like
        nrows : int
            The number of rows for the Matrix.
        ncols : int
            The number of columns for the Matrix.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_cols : bool, default False
            True indicates indices are sorted by column, then row.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indptr and row_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
            For "coor", ownership of "rows" will never change.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "coor" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_coor(
            rows=rows,
            cols=cols,
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_coor(
        self,
        rows,
        cols,
        values,
        *,
        is_iso=False,
        sorted_rows=True,
        sorted_cols=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_CSR

        `pack_coor` is like `import_coor` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("coor")``

        See `Matrix.ss.import_coor` documentation for more details.
        """
        return self._import_coor(
            rows=rows,
            cols=cols,
            nrows=self._parent._nrows,
            values=values,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_coor(
        cls,
        rows,
        cols,
        values,
        *,
        nrows,
        ncols=None,
        is_iso=False,
        sorted_rows=True,
        sorted_cols=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "coor":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'coor'.")
        if not sorted_rows:
            raise ValueError("sorted_rows must be True when importing 'coor' format")
        indptr = indices_to_indptr(rows, nrows + 1)
        return cls._import_csr(
            nrows=nrows,
            ncols=ncols,
            indptr=indptr,
            values=values,
            col_indices=cols,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            name=name,
            method=method,
            matrix=matrix,
        )

    @classmethod
    def import_cooc(
        cls,
        rows,
        cols,
        values,
        *,
        nrows,
        ncols,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=True,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Matrix_import_CSC

        Create a new Matrix from indices and values in coordinate format.
        Rows must be sorted.

        Parameters
        ----------
        rows : array-like
        cols : array-like
        values : array-like
        nrows : int
            The number of rows for the Matrix.
        ncols : int
            The number of columns for the Matrix.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_rows : bool, default False
            True indicates indices are sorted by column, then row.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indptr and row_indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
            For "cooc", ownership of "cols" will never change.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "cooc" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix
        """
        return cls._import_cooc(
            rows=rows,
            cols=cols,
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_cooc(
        self,
        rows,
        cols,
        values,
        *,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=True,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Matrix_pack_CSC

        `pack_cooc` is like `import_cooc` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("cooc")``

        See `Matrix.ss.import_cooc` documentation for more details.
        """
        return self._import_cooc(
            ncols=self._parent._ncols,
            rows=rows,
            cols=cols,
            values=values,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_cooc(
        cls,
        rows,
        cols,
        values,
        *,
        ncols,
        nrows=None,
        is_iso=False,
        sorted_rows=False,
        sorted_cols=True,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
    ):
        if format is not None and format.lower() != "cooc":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'cooc'.")
        if not sorted_cols:
            raise ValueError("sorted_cols must be True when importing 'cooc' format")
        indptr = indices_to_indptr(cols, ncols + 1)
        return cls._import_csc(
            nrows=nrows,
            ncols=ncols,
            indptr=indptr,
            values=values,
            row_indices=rows,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            dtype=dtype,
            name=name,
            method=method,
            matrix=matrix,
        )

    @classmethod
    def import_any(
        cls,
        *,
        # All
        values,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        dtype=None,
        name=None,
        # CSR/CSC/HyperCSR/HyperCSC
        indptr=None,
        # CSR/HyperCSR
        col_indices=None,
        # CSR/HyperCSR/COO
        sorted_cols=False,
        # HyperCSR/COO
        rows=None,
        # CSC/HyperCSC
        row_indices=None,
        # CSC/HyperCSC/COO
        sorted_rows=False,
        # HyperCSC/COO
        cols=None,
        # HyperCSR/HyperCSC
        nvec=None,  # optional
        # BitmapR/BitmapC
        bitmap=None,
        nvals=None,  # optional
    ):
        """
        GxB_Matrix_import_xxx

        Dispatch to appropriate import method inferred from inputs.
        See the other import functions and `Matrix.ss.export`` for details.

        Returns
        -------
        Matrix

        See Also
        --------
        Matrix.from_values
        Matrix.ss.export
        Matrix.ss.import_csr
        Matrix.ss.import_csc
        Matrix.ss.import_hypercsr
        Matrix.ss.import_hypercsc
        Matrix.ss.import_bitmapr
        Matrix.ss.import_bitmapc
        Matrix.ss.import_fullr
        Matrix.ss.import_fullc

        Examples
        --------
        Simple usage:

        >>> pieces = A.ss.export()
        >>> A2 = Matrix.ss.import_any(**pieces)
        """
        return cls._import_any(
            values=values,
            nrows=nrows,
            ncols=ncols,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            dtype=dtype,
            name=name,
            # CSR/CSC/HyperCSR/HyperCSC
            indptr=indptr,
            # CSR/HyperCSR
            col_indices=col_indices,
            # CSR/HyperCSR/COO
            sorted_cols=sorted_cols,
            # HyperCSR/COO
            rows=rows,
            # CSC/HyperCSC
            row_indices=row_indices,
            # CSC/HyperCSC/COO
            sorted_rows=sorted_rows,
            # HyperCSC/COO
            cols=cols,
            # HyperCSR/HyperCSC
            nvec=nvec,
            # BitmapR/BitmapC
            bitmap=bitmap,
            nvals=nvals,
            method="import",
        )

    def pack_any(
        self,
        *,
        # All
        values,
        is_iso=False,
        take_ownership=False,
        format=None,
        # CSR/CSC/HyperCSR/HyperCSC
        indptr=None,
        # CSR/HyperCSR
        col_indices=None,
        # CSR/HyperCSR/COO
        sorted_cols=False,
        # HyperCSR/COO
        rows=None,
        # CSC/HyperCSC
        row_indices=None,
        # CSC/HyperCSC/COO
        sorted_rows=False,
        # HyperCSC/COO
        cols=None,
        # HyperCSR/HyperCSC
        nvec=None,  # optional
        # BitmapR/BitmapC
        bitmap=None,
        nvals=None,  # optional
        # Unused for pack
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
    ):
        """
        GxB_Matrix_pack_xxx

        `pack_any` is like `import_any` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack()``

        See `Matrix.ss.import_any` documentation for more details.
        """
        return self._import_any(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            # CSR/CSC/HyperCSR/HyperCSC
            indptr=indptr,
            # CSR/HyperCSR
            col_indices=col_indices,
            # CSR/HyperCSR/COO
            sorted_cols=sorted_cols,
            # HyperCSR/COO
            rows=rows,
            # CSC/HyperCSC
            row_indices=row_indices,
            # CSC/HyperCSC/COO
            sorted_rows=sorted_rows,
            # HyperCSC/COO
            cols=cols,
            # HyperCSR/HyperCSC
            nvec=nvec,
            # BitmapR/BitmapC
            bitmap=bitmap,
            nvals=nvals,
            method="pack",
            matrix=self._parent,
        )

    @classmethod
    def _import_any(
        cls,
        *,
        # All
        values,
        nrows=None,
        ncols=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        dtype=None,
        name=None,
        # CSR/CSC/HyperCSR/HyperCSC
        indptr=None,
        # CSR/HyperCSR
        col_indices=None,
        # CSR/HyperCSR/COO
        sorted_cols=False,
        # HyperCSR/COO
        rows=None,
        # CSC/HyperCSC
        row_indices=None,
        # CSC/HyperCSC/COO
        sorted_rows=False,
        # HyperCSC/COO
        cols=None,
        # HyperCSR/HyperCSC
        nvec=None,  # optional
        # BitmapR/BitmapC
        bitmap=None,
        nvals=None,  # optional
        method,
        matrix=None,
    ):
        if format is None:
            # Determine format based on provided inputs
            if indptr is not None:
                if bitmap is not None:
                    raise TypeError("Cannot provide both `indptr` and `bitmap`")
                if row_indices is None and col_indices is None:
                    raise TypeError("Must provide either `row_indices` or `col_indices`")
                if row_indices is not None and col_indices is not None:
                    raise TypeError("Cannot provide both `row_indices` and `col_indices`")
                if rows is not None and cols is not None:
                    raise TypeError("Cannot provide both `rows` and `cols`")
                elif rows is None and cols is None:
                    if row_indices is None:
                        format = "csr"
                    else:
                        format = "csc"
                elif rows is not None:
                    if col_indices is None:
                        raise TypeError("HyperCSR requires col_indices, not row_indices")
                    format = "hypercsr"
                else:
                    if row_indices is None:
                        raise TypeError("HyperCSC requires row_indices, not col_indices")
                    format = "hypercsc"
            elif bitmap is not None:
                if col_indices is not None:
                    raise TypeError("Cannot provide both `bitmap` and `col_indices`")
                if row_indices is not None:
                    raise TypeError("Cannot provide both `bitmap` and `row_indices`")
                if cols is not None:
                    raise TypeError("Cannot provide both `bitmap` and `cols`")
                if rows is not None:
                    raise TypeError("Cannot provide both `bitmap` and `rows`")

                # Choose format based on contiguousness of values fist
                if isinstance(values, np.ndarray) and values.ndim == 2:
                    if values.flags.f_contiguous:
                        format = "bitmapc"
                    elif values.flags.c_contiguous:
                        format = "bitmapr"
                # Then consider bitmap contiguousness if necessary
                if format is None and isinstance(bitmap, np.ndarray) and bitmap.ndim == 2:
                    if bitmap.flags.f_contiguous:
                        format = "bitmapc"
                    elif bitmap.flags.c_contiguous:
                        format = "bitmapr"
                # Then default to row-oriented
                if format is None:
                    format = "bitmapr"
            elif rows is not None or cols is not None:
                if rows is None or cols is None:
                    raise ValueError("coo requires both `rows` and `cols`")
                if sorted_rows:
                    if sorted_cols:
                        format = "coo"  # can't tell yet
                    else:
                        format = "coor"
                elif sorted_cols:
                    format = "cooc"
                else:
                    format = "coo"
            else:
                if (
                    isinstance(values, np.ndarray)
                    and values.ndim == 2
                    and values.flags.f_contiguous
                ):
                    format = "fullc"
                else:
                    format = "fullr"
        else:
            format = format.lower()
        if method == "pack":
            obj = matrix.ss
        else:
            obj = cls
        if format == "csr":
            return getattr(obj, f"{method}_csr")(
                nrows=nrows,
                ncols=ncols,
                indptr=indptr,
                values=values,
                col_indices=col_indices,
                is_iso=is_iso,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "csc":
            return getattr(obj, f"{method}_csc")(
                nrows=nrows,
                ncols=ncols,
                indptr=indptr,
                values=values,
                row_indices=row_indices,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "hypercsr":
            return getattr(obj, f"{method}_hypercsr")(
                nrows=nrows,
                ncols=ncols,
                nvec=nvec,
                rows=rows,
                indptr=indptr,
                values=values,
                col_indices=col_indices,
                is_iso=is_iso,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "hypercsc":
            return getattr(obj, f"{method}_hypercsc")(
                nrows=nrows,
                ncols=ncols,
                nvec=nvec,
                cols=cols,
                indptr=indptr,
                values=values,
                row_indices=row_indices,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "bitmapr":
            return getattr(obj, f"{method}_bitmapr")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                nvals=nvals,
                bitmap=bitmap,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "bitmapc":
            return getattr(obj, f"{method}_bitmapc")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                nvals=nvals,
                bitmap=bitmap,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "fullr":
            return getattr(obj, f"{method}_fullr")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "fullc":
            return getattr(obj, f"{method}_fullc")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "coo":
            return getattr(obj, f"{method}_coo")(
                nrows=nrows,
                ncols=ncols,
                rows=rows,
                cols=cols,
                values=values,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "coor":
            return getattr(obj, f"{method}_coor")(
                nrows=nrows,
                ncols=ncols,
                rows=rows,
                cols=cols,
                values=values,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "cooc":
            return getattr(obj, f"{method}_cooc")(
                nrows=nrows,
                ncols=ncols,
                rows=rows,
                cols=cols,
                values=values,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                sorted_cols=sorted_cols,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        else:
            raise ValueError(f"Invalid format: {format}")

    @wrapdoc(head)
    def head(self, n=10, dtype=None, *, sort=False):
        return head(self._parent, n, dtype, sort=sort)

    def scan_columnwise(self, op=monoid.plus, *, name=None):
        """Perform a prefix scan across columns with the given monoid.

        For example, use `monoid.plus` (the default) to perform a cumulative sum,
        and `monoid.times` for cumulative product.  Works with any monoid.

        Returns
        -------
        Vector
        """
        from .prefix_scan import prefix_scan

        return prefix_scan(self._parent.T, op, name=name, within="scan_columnwise")

    def scan_rowwise(self, op=monoid.plus, *, name=None):
        """Perform a prefix scan across rows with the given monoid.

        For example, use `monoid.plus` (the default) to perform a cumulative sum,
        and `monoid.times` for cumulative product.  Works with any monoid.

        Returns
        -------
        Vector
        """
        from .prefix_scan import prefix_scan

        return prefix_scan(self._parent, op, name=name, within="scan_rowwise")

    def flatten(self, order="rowwise", *, name=None):
        """Return a copy of the Matrix collapsed into a Vector.

        Parameters
        ----------
        order : {"rowwise", "columnwise"}, optional
            "rowwise" means to flatten in row-major (C-style) order.
            Aliases of "rowwise" also accepted: "row", "rows", "C".
            "columnwise" means to flatten in column-major (F-style) order.
            Aliases of "rowwise" also accepted: "col", "cols", "column", "columns", "F".
            The default is "rowwise".
        name : str, optional
            Name of the new Vector.

        Returns
        -------
        Vector

        See Also
        --------
        Vector.ss.reshape : copy a Vector to a Matrix.
        """
        order = get_order(order)
        info = self.export(order, raw=True)
        fmt = info["format"]
        if fmt == "csr":
            indptr = info["indptr"]
            nrows = info["nrows"]
            ncols = info["ncols"]
            indices = flatten_csr(indptr, info["col_indices"], nrows, ncols)
            return gb.Vector.ss.import_sparse(
                size=nrows * ncols,  # Should we check if this is less than GxB_INDEX_MAX?
                indices=indices,
                values=info["values"],
                nvals=indptr[nrows],
                is_iso=info["is_iso"],
                sorted_index=info["sorted_cols"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "hypercsr":
            rows = info["rows"]
            indptr = info["indptr"]
            nrows = info["nrows"]
            ncols = info["ncols"]
            nvec = info["nvec"]
            indices = flatten_hypercsr(rows, indptr, info["col_indices"], nrows, ncols, nvec)
            return gb.Vector.ss.import_sparse(
                size=nrows * ncols,
                indices=indices,
                values=info["values"],
                nvals=indptr[nvec],
                is_iso=info["is_iso"],
                sorted_index=info["sorted_cols"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "bitmapr":
            return gb.Vector.ss.import_bitmap(
                bitmap=info["bitmap"],
                values=info["values"],
                nvals=info["nvals"],
                size=info["nrows"] * info["ncols"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "fullr":
            return gb.Vector.ss.import_full(
                values=info["values"],
                size=info["nrows"] * info["ncols"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "csc":
            indptr = info["indptr"]
            nrows = info["nrows"]
            ncols = info["ncols"]
            indices = flatten_csr(indptr, info["row_indices"], ncols, nrows)
            return gb.Vector.ss.import_sparse(
                size=nrows * ncols,
                indices=indices,
                values=info["values"],
                nvals=indptr[ncols],
                is_iso=info["is_iso"],
                sorted_index=info["sorted_rows"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "hypercsc":
            cols = info["cols"]
            indptr = info["indptr"]
            nrows = info["nrows"]
            ncols = info["ncols"]
            nvec = info["nvec"]
            indices = flatten_hypercsr(cols, indptr, info["row_indices"], ncols, nrows, nvec)
            return gb.Vector.ss.import_sparse(
                size=nrows * ncols,
                indices=indices,
                values=info["values"],
                nvals=indptr[nvec],
                is_iso=info["is_iso"],
                sorted_index=info["sorted_rows"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "bitmapc":
            return gb.Vector.ss.import_bitmap(
                bitmap=info["bitmap"],
                values=info["values"],
                nvals=info["nvals"],
                size=info["nrows"] * info["ncols"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "fullc":
            return gb.Vector.ss.import_full(
                values=info["values"],
                size=info["nrows"] * info["ncols"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        else:
            raise NotImplementedError(fmt)

    def selectk_rowwise(self, how, k, *, name=None):
        """Select (up to) k elements from each row.

        Parameters
        ----------
        how : str
            "random": choose k elements with equal probability
            "first": choose the first k elements
            "last": choose the last k elements
        k : int
            The number of elements to choose from each row

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**
        """
        # TODO: largest, smallest, random_weighted
        how = how.lower()
        fmt = "hypercsr"
        indices = "col_indices"
        sort_axis = "sorted_cols"
        if how == "random":
            choose_func = choose_random
            is_random = True
            do_sort = False
        elif how == "first":
            choose_func = choose_first
            is_random = False
            do_sort = True
        elif how == "last":
            choose_func = choose_last
            is_random = False
            do_sort = True
        else:
            raise ValueError('`how` argument must be one of: "random"')
        return self._select_random(
            k, fmt, indices, sort_axis, choose_func, is_random, do_sort, name
        )

    def selectk_columnwise(self, how, k, *, name=None):
        """Select (up to) k elements from each column.

        Parameters
        ----------
        how : str
            - "random": choose elements with equal probability
            - "first": choose the first k elements
            - "last": choose the last k elements
        k : int
            The number of elements to choose from each column

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**
        """
        how = how.lower()
        fmt = "hypercsc"
        indices = "row_indices"
        sort_axis = "sorted_rows"
        if how == "random":
            choose_func = choose_random
            is_random = True
            do_sort = False
        elif how == "first":
            choose_func = choose_first
            is_random = False
            do_sort = True
        elif how == "last":
            choose_func = choose_last
            is_random = False
            do_sort = True
        else:
            raise ValueError('`how` argument must be one of: "random", "first", "last"')
        return self._select_random(
            k, fmt, indices, sort_axis, choose_func, is_random, do_sort, name
        )

    def _select_random(self, k, fmt, indices, sort_axis, choose_func, is_random, do_sort, name):
        if k < 0:
            raise ValueError("negative k is not allowed")
        info = self._parent.ss.export(fmt, sort=do_sort)
        choices, indptr = choose_func(info["indptr"], k)
        newinfo = dict(info, indptr=indptr)
        newinfo[indices] = info[indices][choices]
        if not info["is_iso"]:
            newinfo["values"] = info["values"][choices]
        if k == 1:
            newinfo[sort_axis] = True
        elif is_random:
            newinfo[sort_axis] = False
        return self.import_any(
            **newinfo,
            take_ownership=True,
            name=name,
        )

    def compactify_rowwise(
        self, how="first", ncols=None, *, reverse=False, asindex=False, name=None
    ):
        """Shift all values to the left so all values in a row are contiguous.

        This returns a new Matrix.

        Parameters
        ----------
        how : {"first", "last", "smallest", "largest", "random"}, optional
            How to compress the values:
            - first : take the values furthest to the left
            - last : take the values furthest to the right
            - smallest : take the smallest values (if tied, may take any)
            - largest : take the largest values (if tied, may take any)
            - random : take values randomly with equal probability and without replacement
        reverse : bool, default False
            Reverse the values in each row when True
        asindex : bool, default False
            Return the column index of the value when True.  If there are ties for
            "smallest" and "largest", then any valid index may be returned.
        ncols : int, optional
            The number of columns of the returned Matrix.  If not specified, then
            the Matrix will be "compacted" to the smallest ncols that doesn't lose
            values.

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**

        """
        return self._compactify(
            how, reverse, asindex, "ncols", ncols, "hypercsr", "col_indices", name
        )

    def compactify_columnwise(
        self, how="first", nrows=None, *, reverse=False, asindex=False, name=None
    ):
        """Shift all values to the top so all values in a column are contiguous.

        This returns a new Matrix.

        Parameters
        ----------
        how : {"first", "last", "smallest", "largest", "random"}, optional
            How to compress the values:
            - first : take the values furthest to the top
            - last : take the values furthest to the bottom
            - smallest : take the smallest values (if tied, may take any)
            - largest : take the largest values (if tied, may take any)
            - random : take values randomly with equal probability and without replacement
        reverse : bool, default False
            Reverse the values in each column when True
        asindex : bool, default False
            Return the row index of the value when True.  If there are ties for
            "smallest" and "largest", then any valid index may be returned.
        nrows : int, optional
            The number of rows of the returned Matrix.  If not specified, then
            the Matrix will be "compacted" to the smallest nrows that doesn't lose
            values.

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**

        """
        return self._compactify(
            how, reverse, asindex, "nrows", nrows, "hypercsc", "row_indices", name
        )

    def _compactify(self, how, reverse, asindex, nkey, nval, fmt, indices_name, name):
        how = how.lower()
        if how not in {"first", "last", "smallest", "largest", "random"}:
            raise ValueError(
                '`how` argument must be one of: "first", "last", "smallest", "largest", "random"'
            )
        info = self.export(fmt, sort=True)
        values = info["values"]
        orig_indptr = info["indptr"]
        new_indptr, new_indices, N = compact_indices(orig_indptr, nval)
        values_need_trimmed = nval is not None and new_indices.size < info[indices_name].size
        if nval is None:
            nval = N
        if info["is_iso"]:
            if how in {"smallest", "largest"} or how == "random" and not asindex:
                # order of smallest/largest doesn't matter
                how = "first"
                reverse = False
            if not asindex:
                how = "finished"
                values_need_trimmed = False
                reverse = False
            else:
                info["is_iso"] = False

        if how == "random":
            # Random without replacement
            reverse = False
            # Should we shuffle the values if values_need_trimmed is True?
            values_need_trimmed = False
            # This recalculates new_indptr unnecessarily
            choices, new_indptr = choose_random(orig_indptr, nval)
            if asindex:
                values = info[indices_name][choices]
            else:
                values = values[choices]
        elif how in {"first", "last"}:
            if asindex:
                values = info[indices_name]
            if how == "last":
                if values_need_trimmed:
                    # Optimization: don't call `reverse_values` twice when reverse is True
                    values = reverse_values(orig_indptr, values)
                else:
                    reverse = not reverse
        elif how in {"smallest", "largest"}:
            if asindex:
                values = argsort_values(orig_indptr, info[indices_name], values)
            else:
                values = sort_values(orig_indptr, values)
            if how == "largest":
                if values_need_trimmed:
                    values = reverse_values(orig_indptr, values)
                else:
                    reverse = not reverse
        if values_need_trimmed:
            values = compact_values(orig_indptr, new_indptr, values)
        if reverse:
            values = reverse_values(new_indptr, values)
        newinfo = dict(info, indptr=new_indptr, values=values)
        newinfo[indices_name] = new_indices
        newinfo[nkey] = nval
        return self.import_any(
            **newinfo,
            take_ownership=True,
            name=name,
        )


@numba.njit(parallel=True)
def argsort_values(indptr, indices, values):  # pragma: no cover
    rv = np.empty(indptr[-1], dtype=np.uint64)
    for i in numba.prange(indptr.size - 1):
        rv[indptr[i] : indptr[i + 1]] = indices[
            np.int64(indptr[i]) + np.argsort(values[indptr[i] : indptr[i + 1]])
        ]
    return rv


@numba.njit(parallel=True)
def sort_values(indptr, values):  # pragma: no cover
    rv = np.empty(indptr[-1], dtype=values.dtype)
    for i in numba.prange(indptr.size - 1):
        rv[indptr[i] : indptr[i + 1]] = np.sort(values[indptr[i] : indptr[i + 1]])
    return rv


@numba.njit(parallel=True)
def compact_values(old_indptr, new_indptr, values):  # pragma: no cover
    rv = np.empty(new_indptr[-1], dtype=values.dtype)
    for i in numba.prange(new_indptr.size - 1):
        start = np.int64(new_indptr[i])
        offset = np.int64(old_indptr[i]) - start
        for j in range(start, new_indptr[i + 1]):
            rv[j] = values[j + offset]
    return rv


@numba.njit(parallel=True)
def reverse_values(indptr, values):  # pragma: no cover
    rv = np.empty(indptr[-1], dtype=values.dtype)
    for i in numba.prange(indptr.size - 1):
        offset = np.int64(indptr[i]) + np.int64(indptr[i + 1]) - 1
        for j in range(indptr[i], indptr[i + 1]):
            rv[j] = values[offset - j]
    return rv


@numba.njit(parallel=True)
def compact_indices(indptr, k):  # pragma: no cover
    """Given indptr from hypercsr, create a new col_indices array that is compact.

    That is, for each row with degree N, the column indices will be 0..N-1.
    """
    if k is not None:
        indptr = create_indptr(indptr, k)
    col_indices = np.empty(indptr[-1], dtype=np.uint64)
    N = np.int64(0)
    for i in numba.prange(indptr.size - 1):
        start = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - start
        N = max(N, deg)
        for j in range(deg):
            col_indices[start + j] = j
    return indptr, col_indices, N


@njit(parallel=True)
def choose_random1(indptr):  # pragma: no cover
    choices = np.empty(indptr.size - 1, dtype=indptr.dtype)
    new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
    for i in numba.prange(indptr.size - 1):
        idx = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - idx
        if deg == 1:
            choices[i] = idx
        else:
            choices[i] = np.random.randint(idx, idx + deg)
    return choices, new_indptr


@njit
def create_indptr(indptr, k):  # pragma: no cover
    new_indptr = np.empty(indptr.size, dtype=indptr.dtype)
    new_indptr[0] = 0
    prev = np.int64(indptr[0])
    count = 0
    for i in range(1, indptr.size):
        idx = np.int64(indptr[i])
        deg = idx - prev
        prev = idx
        if k < deg:
            deg = k
        count += deg
        new_indptr[i] = count
    return new_indptr


# Assume we are HyperCSR or HyperCSC
@njit(parallel=True)
def choose_random(indptr, k):  # pragma: no cover
    if k == 1:
        return choose_random1(indptr)

    # The results in choices don't need to be random.  In fact, it may
    # be nice to have them sorted if convenient to do so.
    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in numba.prange(indptr.size - 1):
        idx = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - idx
        if k < deg:
            curk = k
        else:
            curk = deg
        index = np.int64(new_indptr[i])
        # We call np.random.randint `min(curk, deg - curk)` times
        if 2 * curk <= deg:
            if curk == 1:
                # Select a single edge
                choices[index] = np.random.randint(idx, idx + deg)
            elif curk == 2:
                # Select two edges
                choices[index] = np.random.randint(idx, deg + idx)
                choices[index + 1] = np.random.randint(idx, deg + idx - 1)
                if choices[index] <= choices[index + 1]:
                    choices[index + 1] += 1
            else:
                # Move the ones we want to keep to the front of `a`
                a = np.arange(idx, idx + deg)
                for j in range(curk):
                    jj = np.random.randint(j, deg)
                    a[j], a[jj] = a[jj], a[j]
                    choices[index + j] = a[j]
        elif curk == deg:
            # Select all edges
            j = index
            for jj in range(idx, idx + deg):
                choices[j] = jj
                j += 1
        elif curk == deg - 1:
            # Select all but one edge
            curk = np.random.randint(idx, idx + deg)
            j = index
            for jj in range(idx, curk):
                choices[j] = jj
                j += 1
            for jj in range(curk + 1, idx + deg):
                choices[j] = jj
                j += 1
        elif curk == deg - 2:
            # Select all but two edges
            curk = np.random.randint(idx, idx + deg)
            count = np.random.randint(idx, idx + deg - 1)
            if curk <= count:
                count += 1
                curk, count = count, curk
            j = index
            for jj in range(idx, count):
                choices[j] = jj
                j += 1
            for jj in range(count + 1, curk):
                choices[j] = jj
                j += 1
            for jj in range(curk + 1, idx + deg):
                choices[j] = jj
                j += 1
        else:
            # Move the ones we don't want to keep to the front of `a`
            a = np.arange(idx, idx + deg)
            for j in range(deg - curk):
                jj = np.random.randint(j, deg)
                a[j], a[jj] = a[jj], a[j]
            deg -= curk
            for j in range(curk):
                choices[index + j] = a[deg + j]
    return choices, new_indptr


# Assume we are HyperCSR or HyperCSC
@njit(parallel=True)
def choose_first(indptr, k):  # pragma: no cover
    if k == 1:
        choices = indptr[:-1]
        new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
        return choices, new_indptr

    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in numba.prange(indptr.size - 1):
        idx = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - idx
        if k < deg:
            curk = k
        else:
            curk = deg
        j = np.int64(new_indptr[i])
        for jj in range(idx, idx + curk):
            choices[j] = jj
            j += 1
    return choices, new_indptr


# Assume we are HyperCSR or HyperCSC
@njit(parallel=True)
def choose_last(indptr, k):  # pragma: no cover
    if k == 1:
        choices = (indptr[1:].astype(np.int64) - 1).astype(indptr.dtype)
        new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
        return choices, new_indptr

    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in numba.prange(indptr.size - 1):
        idx = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - idx
        if k < deg:
            curk = k
        else:
            curk = deg
        j = np.int64(new_indptr[i])
        for jj in range(idx + deg - curk, idx + deg):
            choices[j] = jj
            j += 1
    return choices, new_indptr


@njit(parallel=True)
def flatten_csr(indptr, indices, nrows, ncols):  # pragma: no cover
    rv = np.empty(indices.size, indices.dtype)
    for i in numba.prange(nrows):
        offset = i * ncols
        for j in range(indptr[i], indptr[i + 1]):
            rv[j] = indices[j] + offset
    return rv


@njit
def issorted(arr):  # pragma: no cover
    if arr.size > 1:
        prev = arr[0]
        for i in range(1, arr.size):
            cur = arr[i]
            if cur == prev:
                continue
            elif cur < prev:
                return False
            else:
                prev = cur
    return True


@njit(parallel=True)
def flatten_hypercsr(rows, indptr, indices, nrows, ncols, nvec):  # pragma: no cover
    rv = np.empty(indices.size, indices.dtype)
    for i in numba.prange(nvec):
        row = rows[i]
        offset = row * ncols
        for j in range(indptr[i], indptr[i + 1]):
            rv[j] = indices[j] + offset
    return rv


@njit
def indices_to_indptr(indices, size):  # pragma: no cover
    """Calculate the indptr for e.g. CSR from sorted COO rows."""
    indptr = np.zeros(size, dtype=indices.dtype)
    index = np.uint64(0)
    for i in range(indices.size):
        row = indices[i]
        if row != index:
            indptr[index + 1] = i
            index = row
    indptr[index + 1] = indices.size
    return indptr


@njit(parallel=True)
def indptr_to_indices(indptr):  # pragma: no cover
    indices = np.empty(indptr[-1], dtype=indptr.dtype)
    for i in numba.prange(indptr.size - 1):
        for j in range(indptr[i], indptr[i + 1]):
            indices[j] = i
    return indices

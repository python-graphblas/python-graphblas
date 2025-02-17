import itertools

import numpy as np
from suitesparse_graphblas.utils import claim_buffer, claim_buffer_2d, unclaim_buffer

import graphblas as gb

from ... import binary, monoid
from ...dtypes import _INDEX, BOOL, INT64, UINT64, lookup_dtype
from ...exceptions import _error_code_lookup, check_status, check_status_carg
from .. import NULL, _has_numba, ffi, lib
from ..base import call
from ..dtypes import _string_to_dtype
from ..operator import get_typed_op
from ..scalar import Scalar, _as_scalar, _scalar_index
from ..utils import (
    _CArray,
    _MatrixArray,
    _Pointer,
    get_order,
    get_shape,
    ints_to_numpy_buffer,
    normalize_chunks,
    output_type,
    values_to_numpy_buffer,
    wrapdoc,
)
from .config import BaseConfig
from .descriptor import get_descriptor

if _has_numba:
    from numba import njit, prange
else:

    def njit(func=None, **kwargs):
        if func is not None:
            return func
        return njit

    prange = range
ffi_new = ffi.new


def head(matrix, n=10, dtype=None, *, sort=False):
    """Like ``matrix.to_coo()``, but only returns the first n elements.

    If sort is True, then the results will be sorted as appropriate for the internal format,
    otherwise the order of the result is not guaranteed.  Specifically, row-oriented formats
    (fullr, bitmapr, csr, hypercsr) will sort by row index first, then by column index.
    Column-oriented formats, naturally, will sort by column index first, then by row index.
    Formats fullr, fullc, bitmapr, and bitmapc should always return in sorted order.
    """
    if matrix._nvals <= n:
        return matrix.to_coo(dtype, sort=sort)
    if sort:
        matrix.wait()
    if dtype is None:
        dtype = matrix.dtype
    else:
        dtype = lookup_dtype(dtype)
    rows, cols, vals = zip(*itertools.islice(matrix.ss.iteritems(), n), strict=True)
    return np.array(rows, np.uint64), np.array(cols, np.uint64), np.array(vals, dtype.np_type)


def _concat_mn(tiles, *, is_matrix=None):
    """Argument checking for ``Matrix.ss.concat`` and returns number of tiles in each dimension."""
    from ..matrix import Matrix, TransposedMatrix
    from ..vector import Vector

    valid_types = (Matrix, TransposedMatrix, Vector, Scalar)
    if not isinstance(tiles, (list, tuple)):
        raise TypeError(f"tiles argument must be list or tuple; got: {type(tiles)}")
    if not tiles:
        raise ValueError("tiles argument must not be empty")
    dummy = object.__new__(Matrix)
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


class MatrixConfig(BaseConfig):
    """Get and set configuration options for this Matrix.

    See SuiteSparse:GraphBLAS documentation for more details.

    Config parameters
    -----------------
    format : str, {"by_row", "by_col"}
        Rowwise or columnwise orientation
    hyper_switch : double
        Threshold that determines when to switch to hypersparse format
    bitmap_switch : double
        Threshold that determines when to switch to bitmap format
    sparsity_control : Set[str] from {"hypersparse", "sparse", "bitmap", "full", "auto"}
        Allowed sparsity formats.  May be set with a single string or a set of strings.
    sparsity_status : str, {"hypersparse", "sparse", "bitmap", "full"}
        Current sparsity format
    """

    _get_function = "GxB_Matrix_Option_get"
    _set_function = "GxB_Matrix_Option_set"
    _options = {
        "format": (lib.GxB_FORMAT, "GxB_Format_Value"),
        "hyper_switch": (lib.GxB_HYPER_SWITCH, "double"),
        "bitmap_switch": (lib.GxB_BITMAP_SWITCH, "double"),
        "sparsity_control": (lib.GxB_SPARSITY_CONTROL, "int"),
        # read-only
        "sparsity_status": (lib.GxB_SPARSITY_STATUS, "int"),
    }
    _bitwise = {
        "sparsity_control": {
            "hypersparse": lib.GxB_HYPERSPARSE,
            "sparse": lib.GxB_SPARSE,
            "bitmap": lib.GxB_BITMAP,
            "full": lib.GxB_FULL,
            "auto": lib.GxB_AUTO_SPARSITY,
        },
    }
    _enumerations = {
        "format": {
            "by_row": lib.GxB_BY_ROW,
            "by_col": lib.GxB_BY_COL,
            # "no_format": lib.GxB_NO_FORMAT,  # Used by iterators; not valid here
        },
        "sparsity_status": {
            "hypersparse": lib.GxB_HYPERSPARSE,
            "sparse": lib.GxB_SPARSE,
            "bitmap": lib.GxB_BITMAP,
            "full": lib.GxB_FULL,
        },
    }
    _defaults = {
        "hyper_switch": lib.GxB_HYPER_DEFAULT,
        "format": lib.GxB_FORMAT_DEFAULT,
        "sparsity_control": "auto",
    }
    _read_only = {"sparsity_status"}


class ss:
    __slots__ = "_parent", "config"

    def __init__(self, parent):
        self._parent = parent
        self.config = MatrixConfig(parent)

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
            # This may not be thread-safe if the parent is being modified in another thread
            return Scalar.from_value(next(self.itervalues()), dtype=self._parent.dtype, name="")
        raise ValueError("Matrix is not iso-valued")

    @property
    def format(self):
        # Determine current format
        parent = self._parent
        format_ptr = ffi_new("int32_t*")
        sparsity_ptr = ffi_new("int32_t*")
        check_status(
            lib.GxB_Matrix_Option_get_INT32(parent._carg, lib.GxB_FORMAT, format_ptr),
            parent,
        )
        check_status(
            lib.GxB_Matrix_Option_get_INT32(parent._carg, lib.GxB_SPARSITY_STATUS, sparsity_ptr),
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
        else:  # pragma: no cover (sanity)
            raise NotImplementedError(f"Unknown sparsity status: {sparsity_status}")
        if format_ptr[0] == lib.GxB_BY_COL:
            format = f"{format}c"
        else:
            format = f"{format}r"
        return format

    @property
    def orientation(self):
        parent = self._parent
        format_ptr = ffi_new("int32_t*")
        check_status(
            lib.GxB_Matrix_Option_get_INT32(parent._carg, lib.GxB_FORMAT, format_ptr),
            parent,
        )
        if format_ptr[0] == lib.GxB_BY_COL:
            return "columnwise"
        return "rowwise"

    def build_diag(self, vector, k=0, **opts):
        """GxB_Matrix_diag.

        Construct a diagonal Matrix from the given vector.
        Existing entries in the Matrix are discarded.

        Parameters
        ----------
        vector : Vector
            Create a diagonal from this Vector.
        k : int, default 0
            Diagonal in question.  Use ``k>0`` for diagonals above the main diagonal,
            and ``k<0`` for diagonals below the main diagonal.

        See Also
        --------
        Matrix.diag
        Vector.diag

        """
        vector = self._parent._expect_type(
            vector, gb.Vector, within="ss.build_diag", argname="vector"
        )
        call(
            "GxB_Matrix_diag",
            [self._parent, vector, _as_scalar(k, INT64, is_cscalar=True), get_descriptor(**opts)],
        )

    def split(self, chunks, *, name=None, **opts):
        """GxB_Matrix_split.

        Split a Matrix into a 2D array of sub-matrices according to ``chunks``.

        This performs the opposite operation as ``concat``.

        ``chunks`` is short for "chunksizes" and indicates the chunk sizes for each dimension.
        ``chunks`` may be a single integer, or a length 2 tuple or list.  Example chunks:

        - ``chunks=10``
            - Split each dimension into chunks of size 10 (the last chunk may be smaller).
        - ``chunks=(10, 20)``
            - Split rows into chunks of size 10 and columns into chunks of size 20.
        - ``chunks=(None, [5, 10])``
            - Don't split rows into chunks, and split columns into two chunks of size 5 and 10.
        - ``chunks=(10, [20, None])``
            - Split columns into two chunks of size 20 and ``ncols - 20``

        See Also
        --------
        Matrix.ss.concat
        graphblas.ss.concat

        """
        from ..matrix import Matrix

        tile_nrows, tile_ncols = normalize_chunks(chunks, self._parent.shape)
        m = len(tile_nrows)
        n = len(tile_ncols)
        tiles = ffi.new("GrB_Matrix[]", m * n)
        call(
            "GxB_Matrix_split",
            [
                _MatrixArray(tiles, self._parent, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(n, _INDEX, is_cscalar=True),
                _CArray(tile_nrows),
                _CArray(tile_ncols),
                self._parent,
                get_descriptor(**opts),
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
                tile = Matrix._from_obj(new_matrix, dtype, nrows, ncols, name=f"{name}_{i}x{j}")
                cur.append(tile)
                index += 1
            rv.append(cur)
        return rv

    def _concat(self, tiles, m, n, opts):
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
                _MatrixArray(ctiles, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(n, _INDEX, is_cscalar=True),
                get_descriptor(**opts),
            ],
        )

    def concat(self, tiles, **opts):
        """GxB_Matrix_concat.

        Concatenate a 2D list of Matrix objects into the current Matrix.
        Any existing values in the current Matrix will be discarded.
        To concatenate into a new Matrix, use ``graphblas.ss.concat``.

        Vectors may be used as ``Nx1`` Matrix objects.

        This performs the opposite operation as ``split``.

        See Also
        --------
        Matrix.ss.split
        graphblas.ss.concat

        """
        tiles, m, n, is_matrix = _concat_mn(tiles, is_matrix=True)
        self._concat(tiles, m, n, opts)

    def build_scalar(self, rows, columns, value):
        """GxB_Matrix_build_Scalar.

        Like ``build``, but uses a scalar for all the values.

        See Also
        --------
        Matrix.build
        Matrix.from_coo

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

    def _begin_iter(self, seek):
        it_ptr = ffi.new("GxB_Iterator*")
        info = lib.GxB_Iterator_new(it_ptr)
        it = it_ptr[0]
        success = lib.GrB_SUCCESS
        info = lib.GxB_Matrix_Iterator_attach(it, self._parent._carg, NULL)
        if info != success:  # pragma: no cover (safety)
            lib.GxB_Iterator_free(it_ptr)
            raise _error_code_lookup[info]("Matrix iterator failed to attach")
        if seek < 0:
            seek = max(0, seek + lib.GxB_Matrix_Iterator_getpmax(it))
        info = lib.GxB_Matrix_Iterator_seek(it, seek)
        if info != success:
            lib.GxB_Iterator_free(it_ptr)
            raise _error_code_lookup[info]("Matrix iterator failed to seek")
        return it_ptr

    def iterkeys(self, seek=0):
        """Iterate over all the row and column indices of a Matrix.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Matrix objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Matrix should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        key_func = lib.GxB_Matrix_Iterator_getIndex
        next_func = lib.GxB_Matrix_Iterator_next
        row_ptr = ffi_new("GrB_Index*")
        col_ptr = ffi_new("GrB_Index*")
        try:
            while info == success:
                key_func(it, row_ptr, col_ptr)
                yield (row_ptr[0], col_ptr[0])
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Matrix iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def itervalues(self, seek=0):
        """Iterate over all the values of a Matrix.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Matrix objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Matrix should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        val_func = getattr(lib, f"GxB_Iterator_get_{self._parent.dtype.name}")
        next_func = lib.GxB_Matrix_Iterator_next
        try:
            while info == success:
                yield val_func(it)
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Matrix iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def iteritems(self, seek=0):
        """Iterate over all the row, column, and value triples of a Matrix.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Matrix objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Matrix should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        key_func = lib.GxB_Matrix_Iterator_getIndex
        val_func = getattr(lib, f"GxB_Iterator_get_{self._parent.dtype.name}")
        next_func = lib.GxB_Matrix_Iterator_next
        row_ptr = ffi_new("GrB_Index*")
        col_ptr = ffi_new("GrB_Index*")
        try:
            while info == success:
                key_func(it, row_ptr, col_ptr)
                yield (row_ptr[0], col_ptr[0], val_func(it))
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Matrix iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def export(self, format=None, *, sort=False, give_ownership=False, raw=False, **opts):
        """GxB_Matrix_export_xxx.

        Parameters
        ----------
        format : str, optional
            If ``format`` is not specified, this method exports in the currently stored format.
            To control the export format, set ``format`` to one of:
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
            the underlying memory buffers to NumPy.
            ** If True, this nullifies the current object, which should no longer be used! **
        raw : bool, default False
            If True, always return 1d arrays the same size as returned by SuiteSparse.
            If False, arrays may be trimmed to be the expected size, and 2d arrays are
            returned when format is "bitmapr", "bitmapc", "fullr", or "fullc".
            It may make sense to choose ``raw=True`` if one wants to use the data to perform
            a zero-copy import back to SuiteSparse.

        Returns
        -------
        dict; keys depend on ``format`` and ``raw`` arguments (see below).

        See Also
        --------
        Matrix.to_coo
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
                    - bitmap : ndarray(dtype=bool, ndim=2, shape=(nrows, ncols), order="C")
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="C")
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                - ``raw=True``
                    - bitmap : ndarray(dtype=bool, ndim=1, size=nrows * ncols)
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
                    - bitmap : ndarray(dtype=bool, ndim=2, shape=(nrows, ncols), order="F")
                    - values : ndarray(ndim=2, shape=(nrows, ncols), order="F")
                        - Elements where bitmap is False are undefined
                    - nvals : int
                        - The number of True elements in the bitmap
                - ``raw=True``
                    - bitmap : ndarray(dtype=bool, ndim=1, size=nrows * ncols)
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
                        - Stored column-oriented
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
            format,
            sort=sort,
            give_ownership=give_ownership,
            raw=raw,
            method="export",
            opts=opts,
        )

    def unpack(self, format=None, *, sort=False, raw=False, **opts):
        """GxB_Matrix_unpack_xxx.

        ``unpack`` is like ``export``, except that the Matrix remains valid but empty.
        ``pack_*`` methods are the opposite of ``unpack``.

        See ``Matrix.ss.export`` documentation for more details.
        """
        return self._export(
            format, sort=sort, raw=raw, give_ownership=True, method="unpack", opts=opts
        )

    def _export(self, format=None, *, sort=False, give_ownership=False, raw=False, method, opts):
        if format is None:
            format = self.format
        else:
            format = format.lower()
            try:
                order = get_order(format)
            except ValueError:
                pass
            else:
                if order == "rowwise":
                    format = f"{self.format[:-1]}r"
                else:  # columnwise
                    format = f"{self.format[:-1]}c"
        if give_ownership or format == "coo":
            parent = self._parent
        else:
            parent = self._parent.dup(name=f"M_{method}")
        dtype = parent.dtype.np_type
        index_dtype = np.dtype(np.uint64)
        desc = get_descriptor(**opts)
        desc_obj = NULL if desc is None else desc._carg

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
                    # Should we expose a way to do `to_coo` without values?
                    # Passing NULL for values is SuiteSparse-specific.
                    nvals = parent._nvals
                    rows = _CArray(size=nvals, name="&rows_array")
                    columns = _CArray(size=nvals, name="&columns_array")
                    scalar = _scalar_index("s_nvals")
                    scalar.value = nvals
                    call(
                        f"GrB_Matrix_extractTuples_{parent.dtype.name}",
                        [rows, columns, None, _Pointer(scalar), parent],
                    )
                    value = parent.reduce_scalar(monoid.any, allow_empty=False).new().value
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
                    rows, columns, values = parent.to_coo()
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
                        parent.gb_obj[0] = NULL
                    else:
                        parent.clear()
            elif format == "coor":
                rv = self._export(
                    "csr",
                    sort=sort,
                    give_ownership=give_ownership,
                    raw=False,
                    method=method,
                    opts=opts,
                )
                rv["rows"] = indptr_to_indices(rv.pop("indptr"))
                rv["cols"] = rv.pop("col_indices")
                rv["sorted_rows"] = True
                rv["format"] = "coor"
            elif format == "cooc":
                rv = self._export(
                    "csc",
                    sort=sort,
                    give_ownership=give_ownership,
                    raw=False,
                    method=method,
                    opts=opts,
                )
                rv["cols"] = indptr_to_indices(rv.pop("indptr"))
                rv["rows"] = rv.pop("row_indices")
                rv["sorted_cols"] = True
                rv["format"] = "cooc"
            else:
                raise ValueError(f"Invalid format: {format}")
            if parent.dtype._is_udt:
                rv["dtype"] = parent.dtype
            return rv

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
            jumbled = NULL
        else:
            jumbled = ffi_new("bool*")
        is_iso = ffi_new("bool*")
        nvals = parent._nvals
        if format == "csr":
            Aj = ffi_new("GrB_Index**")
            Aj_size = ffi_new("GrB_Index*")
            check_status(
                getattr(lib, f"GxB_Matrix_{method}_CSR")(
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
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            col_indices = claim_buffer(ffi, Aj[0], Aj_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indptr.size > nrows + 1:  # pragma: no cover (suitesparse)
                    indptr = indptr[: nrows + 1]
                if col_indices.size > nvals:
                    col_indices = col_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no branch (suitesparse)
                        values = values[:1]
                elif values.size > nvals:  # pragma: no branch (suitesparse)
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
                getattr(lib, f"GxB_Matrix_{method}_CSC")(
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
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            indptr = claim_buffer(ffi, Ap[0], Ap_size[0] // index_dtype.itemsize, index_dtype)
            row_indices = claim_buffer(ffi, Ai[0], Ai_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indptr.size > ncols + 1:  # pragma: no cover (suitesparse)
                    indptr = indptr[: ncols + 1]
                if row_indices.size > nvals:
                    row_indices = row_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover (suitesparse)
                        values = values[:1]
                elif values.size > nvals:
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
                getattr(lib, f"GxB_Matrix_{method}_HyperCSR")(
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
                    desc_obj,
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
                if indptr.size > nvec + 1:
                    indptr = indptr[: nvec + 1]
                if rows.size > nvec:
                    rows = rows[:nvec]
                if col_indices.size > nvals:
                    col_indices = col_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover (suitesparse)
                        values = values[:1]
                elif values.size > nvals:
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
                getattr(lib, f"GxB_Matrix_{method}_HyperCSC")(
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
                    desc_obj,
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
                if indptr.size > nvec + 1:
                    indptr = indptr[: nvec + 1]
                if cols.size > nvec:
                    cols = cols[:nvec]
                if row_indices.size > nvals:
                    row_indices = row_indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover (suitesparse)
                        values = values[:1]
                elif values.size > nvals:
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
        elif format in {"bitmapr", "bitmapc"}:
            if format == "bitmapr":
                cfunc = getattr(lib, f"GxB_Matrix_{method}_BitmapR")
            else:
                cfunc = getattr(lib, f"GxB_Matrix_{method}_BitmapC")
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
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            bool_dtype = np.dtype(np.bool_)
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
                    if values.size > 1:  # pragma: no cover (suitesparse)
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
        elif format in {"fullr", "fullc"}:
            if format == "fullr":
                cfunc = getattr(lib, f"GxB_Matrix_{method}_FullR")
            else:
                cfunc = getattr(lib, f"GxB_Matrix_{method}_FullC")
            check_status(
                cfunc(
                    mhandle,
                    *args,
                    Ax,
                    Ax_size,
                    is_iso,
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            if raw:
                values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
                rv = {"nrows": nrows, "ncols": ncols}
            elif is_iso:
                values = claim_buffer(ffi, Ax[0], Ax_size[0] // dtype.itemsize, dtype)
                if values.size > 1:
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
            parent.gb_obj[0] = NULL
        if parent.dtype._is_udt:
            rv["dtype"] = parent.dtype
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_CSR.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_CSR.

        ``pack_csr`` is like ``import_csr`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("csr")``

        See ``Matrix.ss.import_csr`` documentation for more details.
        """
        return self._import_csr(
            indptr=indptr,
            values=values,
            col_indices=col_indices,
            is_iso=is_iso,
            sorted_cols=sorted_cols,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
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
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_CSR")(
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
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_CSC.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_CSC.

        ``pack_csc`` is like ``import_csc`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("csc")``

        See ``Matrix.ss.import_csc`` documentation for more details.
        """
        return self._import_csc(
            indptr=indptr,
            values=values,
            row_indices=row_indices,
            is_iso=is_iso,
            sorted_rows=sorted_rows,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
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
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_CSC")(
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
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_HyperCSR.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_HyperCSR.

        ``pack_hypercsr`` is like ``import_hypercsr`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("hypercsr")``

        See ``Matrix.ss.import_hypercsr`` documentation for more details.
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
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
        if not is_iso and values.ndim == 0:
            is_iso = True
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
            if nrows is None:
                if rows.size == 0:
                    nrows = 0
                else:
                    nrows = rows[-1] + np.uint64(1)
            if ncols is None:
                if col_indices.size == 0:
                    ncols = 0
                else:
                    ncols = col_indices.max() + np.uint64(1)
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_HyperCSR")(
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
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_HyperCSC.

        Create a new Matrix from standard HyperCSC format.

        Parameters
        ----------
        nrows : int
        ncols : int
        cols : array-like
        indptr : array-like
        values : array-like
        row_indices : array-like
        nvec : int, optional
            The number of elements in "cols" to use.
            If not specified, will be set to ``len(cols)``.
        is_iso : bool, default False
            Is the Matrix iso-valued (meaning all the same value)?
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_HyperCSC.

        ``pack_hypercsc`` is like ``import_hypercsc`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("hypercsc")``

        See ``Matrix.ss.import_hypercsc`` documentation for more details.
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
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
        if row_indices is values:
            values = np.copy(values)
        if not is_iso and values.ndim == 0:
            is_iso = True
        Ap = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indptr))
        Ah = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", cols))
        Ai = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", row_indices))
        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvec is None:
            nvec = cols.size
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            if nrows is None:
                if row_indices.size == 0:
                    nrows = 0
                else:
                    nrows = row_indices.max() + np.uint64(1)
            if ncols is None:
                if cols.size == 0:
                    ncols = 0
                else:
                    ncols = cols[-1] + np.uint64(1)
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_HyperCSC")(
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
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_BitmapR.

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
            If true, then ``values`` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (bool for bitmap)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_bitmapr(
        self,
        *,
        bitmap,
        values,
        nvals=None,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_BitmapR.

        ``pack_bitmapr`` is like ``import_bitmapr`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("bitmapr")``

        See ``Matrix.ss.import_bitmapr`` documentation for more details.
        """
        return self._import_bitmapr(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
    ):
        if format is not None and format.lower() != "bitmapr":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapr'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(
            bitmap, np.bool_, copy=copy, ownable=True, order="C", name="bitmap"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, order="C", subarray_after=2
        )
        if bitmap is values:
            values = np.copy(values)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, dtype, bitmap=bitmap, values=values)
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
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_BitmapR")(
            mhandle,
            *args,
            Ab,
            Ax,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_BitmapC.

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
            If true, then ``values`` should be a length 1 array.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be FORTRAN contiguous
                - have the correct dtype (bool for bitmap)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_bitmapc(
        self,
        *,
        bitmap,
        values,
        nvals=None,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_BitmapC.

        ``pack_bitmapc`` is like ``import_bitmapc`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("bitmapc")``

        See ``Matrix.ss.import_bitmapc`` documentation for more details.
        """
        return self._import_bitmapc(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
    ):
        if format is not None and format.lower() != "bitmapc":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'bitmapc'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(
            bitmap, np.bool_, copy=copy, ownable=True, order="F", name="bitmap"
        )
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, order="F", subarray_after=2
        )
        if bitmap is values:
            values = np.copy(values)
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, dtype, bitmap=bitmap, values=values)
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
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_BitmapC")(
            mhandle,
            *args,
            Ab,
            Ax,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_FullR.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_fullr(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_FullR.

        ``pack_fullr`` is like ``import_fullr`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("fullr")``

        See ``Matrix.ss.import_fullr`` documentation for more details.
        """
        return self._import_fullr(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
    ):
        if format is not None and format.lower() != "fullr":
            raise ValueError(f"Invalid format: {format!r}  Must be None or 'fullr'.")
        copy = not take_ownership
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, order="C", ownable=True, subarray_after=2
        )
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, dtype, values=values)
        else:
            nrows, ncols = matrix.shape

        Ax = ffi_new("void**", ffi.from_buffer("void*", values))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_FullR")(
            mhandle, *args, Ax, values.nbytes, is_iso, NULL if desc is None else desc._carg
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_FullC.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_fullc(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_FullC.

        ``pack_fullc`` is like ``import_fullc`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("fullc")``

        See ``Matrix.ss.import_fullc`` documentation for more details.
        """
        return self._import_fullc(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
    ):
        if format is not None and format.lower() != "fullc":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'fullc'.")
        copy = not take_ownership
        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, order="F", ownable=True, subarray_after=2
        )
        if method == "import":
            nrows, ncols = get_shape(nrows, ncols, dtype, values=values)
        else:
            nrows, ncols = matrix.shape
        Ax = ffi_new("void**", ffi.from_buffer("void*", values.T))
        if method == "import":
            mhandle = ffi_new("GrB_Matrix*")
            args = (dtype._carg, nrows, ncols)
        else:
            mhandle = matrix._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Matrix_{method}_FullC")(
            mhandle, *args, Ax, values.nbytes, is_iso, NULL if desc is None else desc._carg
        )
        if method == "import":
            check_status_carg(
                status,
                "Matrix",
                mhandle[0],
            )
            matrix = gb.Matrix._from_obj(mhandle, dtype, nrows, ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GrB_Matrix_build_XXX and GxB_Matrix_build_Scalar.

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
            If true, then ``values`` should be a length 1 array.
        sorted_rows : bool, default False
            True if rows are sorted or when (cols, rows) are sorted lexicographically
        sorted_cols : bool, default False
            True if cols are sorted or when (rows, cols) are sorted lexicographically
        take_ownership : bool, default False
            Ignored.  Zero-copy is not possible for "coo" format.
        dtype : dtype, optional
            dtype of the new Matrix.
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GrB_Matrix_build_XXX and GxB_Matrix_build_Scalar.

        ``pack_coo`` is like ``import_coo`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("coo")``

        See ``Matrix.ss.import_coo`` documentation for more details.
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
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                method=method,
                matrix=matrix,
                opts=opts,
            )
        if sorted_cols and (not sorted_rows or issorted(cols)):
            return cls._import_cooc(
                rows=rows,
                cols=cols,
                values=values,
                nrows=nrows,
                ncols=ncols,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                method=method,
                matrix=matrix,
                opts=opts,
            )

        if method == "pack":
            dtype = matrix.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, subarray_after=1)
        if method == "import":
            matrix = gb.Matrix(dtype, nrows=nrows, ncols=ncols, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_CSR.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_CSR.

        ``pack_coor`` is like ``import_coor`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("coor")``

        See ``Matrix.ss.import_coor`` documentation for more details.
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
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
            secure_import=secure_import,
            dtype=dtype,
            name=name,
            method=method,
            matrix=matrix,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_import_CSC.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_CSC.

        ``pack_cooc`` is like ``import_cooc`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack("cooc")``

        See ``Matrix.ss.import_cooc`` documentation for more details.
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
            secure_import=secure_import,
            format=format,
            method="pack",
            matrix=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        matrix=None,
        opts,
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
            secure_import=secure_import,
            dtype=dtype,
            name=name,
            method=method,
            matrix=matrix,
            opts=opts,
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
        secure_import=False,
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
        **opts,
    ):
        """GxB_Matrix_import_xxx.

        Dispatch to appropriate import method inferred from inputs.
        See the other import functions and ``Matrix.ss.export`` for details.

        Returns
        -------
        Matrix

        See Also
        --------
        Matrix.from_coo
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
            secure_import=secure_import,
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
            opts=opts,
        )

    def pack_any(
        self,
        *,
        # All
        values,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
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
        # Unused for pack, ignored
        nrows=None,
        ncols=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Matrix_pack_xxx.

        ``pack_any`` is like ``import_any`` except it "packs" data into an
        existing Matrix.  This is the opposite of ``unpack()``

        See ``Matrix.ss.import_any`` documentation for more details.
        """
        return self._import_any(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
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
            opts=opts,
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
        secure_import=False,
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
        opts,
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
                if rows is None and cols is None:
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
            elif isinstance(values, np.ndarray) and values.ndim == 2 and values.flags.f_contiguous:
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "csc":
            return getattr(obj, f"{method}_csc")(
                nrows=nrows,
                ncols=ncols,
                indptr=indptr,
                values=values,
                row_indices=row_indices,
                is_iso=is_iso,
                sorted_rows=sorted_rows,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "hypercsr":
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "hypercsc":
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "bitmapr":
            return getattr(obj, f"{method}_bitmapr")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                nvals=nvals,
                bitmap=bitmap,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "bitmapc":
            return getattr(obj, f"{method}_bitmapc")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                nvals=nvals,
                bitmap=bitmap,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "fullr":
            return getattr(obj, f"{method}_fullr")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "fullc":
            return getattr(obj, f"{method}_fullc")(
                nrows=nrows,
                ncols=ncols,
                values=values,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "coo":
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "coor":
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "cooc":
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        raise ValueError(f"Invalid format: {format}")

    def unpack_hyperhash(self, *, compute=False, name=None, **opts):
        """Unpacks the hyper_hash of a hypersparse matrix if possible.

        Will return None if the matrix is not hypersparse, if the hash is not computed,
        or if the hash is not needed. Use ``compute=True`` to try to compute the hyper_hash
        if the input is hypersparse. The hyper_hash is optional in SuiteSparse:GraphBLAS,
        so it may not be computed even with ``compute=True``.

        Use ``pack_hyperhash`` to move a hyper_hash matrix that was previously unpacked
        back into a matrix.

        This may be used before unpacking a HyperCSR or HyperCSC matrix to preserve the
        full underlying data structure that can be packed back into an empty matrix.
        """
        from ..matrix import Matrix

        if compute and self.format.startswith("hypercs"):
            self._parent.wait()
        rv = Matrix._from_obj(ffi_new("GrB_Matrix*"), INT64, 0, 0, name=name)
        call("GxB_unpack_HyperHash", [self._parent, _Pointer(rv), get_descriptor(**opts)])
        if rv.gb_obj[0] == NULL:
            return
        rv._nrows = rv.nrows
        rv._ncols = rv.ncols
        return rv

    def pack_hyperhash(self, Y, **opts):
        """Pack a hyper_hash matrix Y into the current hypersparse matrix.

        The hyper_hash matrix Y should be from ``unpack_hyperhash`` and unmodified.

        This uses move semantics. Y will become an invalid matrix.
        """
        call("GxB_pack_HyperHash", [self._parent, _Pointer(Y), get_descriptor(**opts)])

    @wrapdoc(head)
    def head(self, n=10, dtype=None, *, sort=False):
        return head(self._parent, n, dtype, sort=sort)

    def scan(self, op=monoid.plus, order="rowwise", *, name=None, **opts):
        """Perform a prefix scan across rows (default) or columns with the given monoid.

        For example, use ``monoid.plus`` (the default) to perform a cumulative sum,
        and ``monoid.times`` for cumulative product.  Works with any monoid.

        Returns
        -------
        Matrix

        """
        order = get_order(order)
        parent = self._parent
        if order == "columnwise":
            parent = parent.T
        return prefix_scan(parent, op, name=name, within="scan", **opts)

    def flatten(self, order="rowwise", *, name=None, **opts):
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
        rv = self.reshape(-1, 1, order=order, name=name, **opts)
        return rv._as_vector()

    def reshape(self, nrows, ncols=None, order="rowwise", *, inplace=False, name=None, **opts):
        """Return a copy of Matrix with a new shape without changing its data.

        The shape of the Matrix must be compatible with the original shape.
        That is, the number of elements must be equal:
        ``nrows * ncols == old_nrows * old_ncols``.
        One of the dimensions may be -1, which will infer the correct size.

        Parameters
        ----------
        nrows : int or tuple of ints
        ncols : int or None
        order : {"rowwise", "columnwise"}, optional
            "rowwise" means to traverse and fill the Matrix in row-major (C-style) order.
            Aliases of "rowwise" also accepted: "row", "rows", "C".
            "columnwise" means to traverse and fill the Matrix in column-major (F-style) order.
            Aliases of "columnwise" also accepted: "col", "cols", "column", "columns", "F".
            The default is "rowwise".
        inplace : bool, default=False
            If True, perform operation in-place.
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix or None
            Reshaped Matrix or None if ``inplace=True``.

        See Also
        --------
        Matrix.ss.flatten : flatten a Matrix into a Vector.
        Vector.ss.reshape : copy a Vector to a Matrix.

        """
        from ..matrix import Matrix

        order = get_order(order)
        parent = self._parent
        array = np.broadcast_to(False, parent.shape)
        if ncols is None:
            array = array.reshape(nrows)
        else:
            array = array.reshape(nrows, ncols)
        if array.ndim != 2:
            raise ValueError(f"Shape tuple must be of length 2, not {array.ndim}")
        nrows, ncols = array.shape
        if inplace:
            call(
                "GxB_Matrix_reshape",
                [
                    parent,
                    _as_scalar(order == "columnwise", BOOL, is_cscalar=True),
                    _as_scalar(nrows, _INDEX, is_cscalar=True),
                    _as_scalar(ncols, _INDEX, is_cscalar=True),
                    get_descriptor(**opts),
                ],
            )
            parent._nrows = nrows
            parent._ncols = ncols
            return
        rv = Matrix._from_obj(ffi_new("GrB_Matrix*"), parent.dtype, nrows, ncols, name=name)
        call(
            "GxB_Matrix_reshapeDup",
            [
                _Pointer(rv),
                parent,
                _as_scalar(order == "columnwise", BOOL, is_cscalar=True),
                _as_scalar(nrows, _INDEX, is_cscalar=True),
                _as_scalar(ncols, _INDEX, is_cscalar=True),
                get_descriptor(**opts),
            ],
        )
        return rv

    def selectk(self, how, k, order="rowwise", *, name=None):
        """Select (up to) k elements from each row (default) or column.

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
        order = get_order(order)
        how = how.lower()
        if order == "rowwise":
            fmt = "hypercsr"
            indices = "col_indices"
            sort_axis = "sorted_cols"
        else:
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

    def compactify(
        self, how="first", k=None, order="rowwise", *, reverse=False, asindex=False, name=None
    ):
        """Shift all values to the left (or up) so all values in a row (or column) are contiguous.

        This returns a new Matrix.

        Parameters
        ----------
        how : {"first", "last", "smallest", "largest", "random"}, optional
            How to compress the values:
            - first : take the values furthest to the left (or top)
            - last : take the values furthest to the right (or bottom)
            - smallest : take the smallest values (if tied, may take any)
            - largest : take the largest values (if tied, may take any)
            - random : take values randomly with equal probability and without replacement
              Chosen values may not be ordered randomly
        k : int, optional
            The number of columns (or rows) of the returned Matrix.  If not specified,
            then the Matrix will be "compacted" to the smallest ncols (or nrows) that
            doesn't lose values.
        order : {"rowwise", "columnwise"}, optional
            Whether to compactify rowwise or columnwise. Rowwise shifts all values
            to the left, and columnwise shifts all values to the top.
            The default is "rowwise".
        reverse : bool, default False
            Reverse the values in each row (or column) when True
        asindex : bool, default False
            Return the column (or row) index of the value when True.  If there are ties
            for "smallest" and "largest", then any valid index may be returned.

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**

        """
        order = get_order(order)
        if order == "rowwise":
            fmt = "hypercsr"
            dimname = "ncols"
            indices = "col_indices"
        else:
            fmt = "hypercsc"
            dimname = "nrows"
            indices = "row_indices"
        return self._compactify(how, reverse, asindex, dimname, k, fmt, indices, name)

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
        newinfo = dict(info, indptr=new_indptr)
        newinfo["values"] = values
        newinfo[indices_name] = new_indices
        newinfo[nkey] = nval
        return self.import_any(
            **newinfo,
            take_ownership=True,
            name=name,
        )

    def sort(self, op=binary.lt, order="rowwise", *, values=True, permutation=True, **opts):
        """GxB_Matrix_sort to sort values along the rows (default) or columns of the Matrix.

        Sorting moves all the elements to the left (if rowwise) or top (if columnwise) just
        like ``compactify``. The returned matrices will be the same shape as the input Matrix.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.BinaryOp`, optional
            Binary operator with a bool return type used to sort the values.
            For example, ``binary.lt`` (the default) sorts the smallest elements first.
            Ties are broken according to indices (smaller first).
        order : {"rowwise", "columnwise"}, optional
            Whether to sort rowwise or columnwise. Rowwise shifts all values to the left,
            and columnwise shifts all values to the top. The default is "rowwise".
        values : bool, default=True
            Whether to return values; will return ``None`` for values if ``False``.
        permutation : bool, default=True
            Whether to compute the permutation Matrix that has the original column
            indices (if rowwise) or row indices (if columnwise) of the sorted values.
            Will return None if ``False``.
        nthreads : int, optional
            The maximum number of threads to use for this operation.
            None, 0 or negative nthreads means to use the default number of threads.

        Returns
        -------
        Matrix : Values
        Matrix[dtype=UINT64] : Permutation

        See Also
        --------
        Matrix.ss.compactify

        """
        from ..matrix import Matrix

        order = get_order(order)
        parent = self._parent
        op = get_typed_op(op, parent.dtype, kind="binary")
        if op.opclass == "Monoid":
            op = op.binaryop
        else:
            parent._expect_op(op, "BinaryOp", within="sort", argname="op")
        if values:
            C = Matrix(parent.dtype, parent._nrows, parent._ncols, name="Values")
        elif not permutation:
            return None, None
        else:
            C = None
        if permutation:
            P = Matrix(UINT64, parent._nrows, parent._ncols, name="Permutation")
        else:
            P = None
        desc = get_descriptor(transpose_first=order == "columnwise", **opts)
        check_status(
            lib.GxB_Matrix_sort(
                NULL if C is None else C._carg,
                NULL if P is None else P._carg,
                op._carg,
                parent._carg,
                NULL if desc is None else desc._carg,
            ),
            parent,
        )
        return C, P

    def serialize(self, compression="default", level=None, **opts):
        """Serialize a Matrix to bytes (as numpy array) using SuiteSparse GxB_Matrix_serialize.

        Parameters
        ----------
        compression : {"default", "lz4", "lz4hc", "zstd", "none", None}, optional
            Whether and how to compress the data.
            - "default": the default in SuiteSparse:GraphBLAS, which is currently ZSTD
            - "lz4": the default LZ4 compression
            - "lz4hc": LZ4 compression that allows the compression level (1-9) to be set.
              Low compression level (1) is faster, high (9) is more compact.  Default is 9.
            - "zstd": ZSTD compression, which allows compression level (1-19) to be set.
              Low compression level (1) is faster, high (19) is more compact.  Default is 19.
            - "none" or None: no compression
        level : int [1-9], optional
            The compression level, between 1 to 9, to use with "lz4hc" and "zstd" compression.
            Level 1 is the fastest and largest, and is the default for "zstd" compression.
            Level 9 is the default when using "lz4hc" compression.
        nthreads : int, optional
            The maximum number of threads to use when serializing the Matrix.
            None, 0 or negative nthreads means to use the default number of threads.

        For best performance, this function returns a numpy array with uint8 dtype.
        Use ``Matrix.ss.deserialize(blob)`` to create a Matrix from the result of serialization

        This method is intended to support all serialization options from SuiteSparse:GraphBLAS.

        *Warning*: Behavior of serializing UDTs is experimental and may change in a future release.

        """
        desc = get_descriptor(compression=compression, compression_level=level, **opts)
        blob_handle = ffi_new("void**")
        blob_size_handle = ffi_new("GrB_Index*")
        parent = self._parent
        if parent.dtype._is_udt and hasattr(lib, "GrB_Type_get_String"):
            # Get the name from the dtype and set it to the name of the matrix so we can
            # recreate the UDT. This is a bit hacky and we should restore the original name.
            # First get the size of name.
            dtype_size = ffi_new("size_t*")
            status = lib.GrB_Type_get_SIZE(parent.dtype.gb_obj[0], dtype_size, lib.GrB_NAME)
            check_status_carg(status, "Type", parent.dtype.gb_obj[0])
            # Then get the name
            dtype_char = ffi_new(f"char[{dtype_size[0]}]")
            status = lib.GrB_Type_get_String(parent.dtype.gb_obj[0], dtype_char, lib.GrB_NAME)
            check_status_carg(status, "Type", parent.dtype.gb_obj[0])
            # Then set the name
            status = lib.GrB_Matrix_set_String(parent._carg, dtype_char, lib.GrB_NAME)
            check_status_carg(status, "Matrix", parent._carg)

        check_status(
            lib.GxB_Matrix_serialize(
                blob_handle,
                blob_size_handle,
                parent._carg,
                NULL if desc is None else desc._carg,
            ),
            parent,
        )
        return claim_buffer(ffi, blob_handle[0], blob_size_handle[0], np.dtype(np.uint8))

    @classmethod
    def deserialize(cls, data, dtype=None, *, name=None, **opts):
        """Deserialize a Matrix from bytes, buffer, or numpy array using GxB_Matrix_deserialize.

        The data should have been previously serialized with a compatible version of
        SuiteSparse:GraphBLAS.  For example, from the result of ``data = matrix.ss.serialize()``.

        Examples
        --------
        >>> data = matrix.serialize()
        >>> new_matrix = Matrix.ss.deserialize(data)
        >>> new_matrix.isequal(matrix)
        True

        Parameters
        ----------
        dtype : DataType, optional
            If given, this should specify the dtype of the object.  This is usually unnecessary.
            If the dtype doesn't match what is in the serialized metadata, deserialize will fail.
            You may need to specify the dtype to load user-defined types.
        nthreads : int, optional
            The maximum number of threads to use when deserializing.
            None, 0 or negative nthreads means to use the default number of threads.

        """
        if isinstance(data, np.ndarray):
            data = ints_to_numpy_buffer(data, np.uint8)
        else:
            data = np.frombuffer(data, np.uint8)
        data_obj = ffi.from_buffer("void*", data)
        if dtype is None:
            # Get the dtype name first (for non-UDTs)
            cname = ffi_new(f"char[{lib.GxB_MAX_NAME_LEN}]")
            info = lib.GxB_deserialize_type_name(
                cname,
                data_obj,
                data.nbytes,
            )
            if info != lib.GrB_SUCCESS:
                raise _error_code_lookup[info]("Matrix deserialize failed to get the dtype name")
            dtype_name = b"".join(itertools.takewhile(b"\x00".__ne__, cname)).decode()
            if not dtype_name and hasattr(lib, "GxB_Serialized_get_String"):
                # Handle UDTs. First get the size of name
                dtype_size = ffi_new("size_t*")
                info = lib.GxB_Serialized_get_SIZE(data_obj, dtype_size, lib.GrB_NAME, data.nbytes)
                if info != lib.GrB_SUCCESS:
                    raise _error_code_lookup[info](
                        "Matrix deserialize failed to get the size of name"
                    )
                # Then get the name
                dtype_char = ffi_new(f"char[{dtype_size[0]}]")
                info = lib.GxB_Serialized_get_String(
                    data_obj, dtype_char, lib.GrB_NAME, data.nbytes
                )
                if info != lib.GrB_SUCCESS:
                    raise _error_code_lookup[info]("Matrix deserialize failed to get the name")
                dtype_name = ffi.string(dtype_char).decode()
            dtype = _string_to_dtype(dtype_name)
        else:
            dtype = lookup_dtype(dtype)
        desc = get_descriptor(**opts)
        gb_obj = ffi_new("GrB_Matrix*")
        check_status_carg(
            lib.GxB_Matrix_deserialize(
                gb_obj, dtype._carg, data_obj, data.nbytes, NULL if desc is None else desc._carg
            ),
            "Matrix",
            gb_obj[0],
        )
        rv = gb.Matrix._from_obj(gb_obj, dtype, -1, -1, name=name)
        rv._nrows = rv.nrows
        rv._ncols = rv.ncols
        return rv


@njit(parallel=True)
def argsort_values(indptr, indices, values):  # pragma: no cover (numba)
    rv = np.empty(indptr[-1], dtype=np.uint64)
    for i in prange(indptr.size - 1):
        rv[indptr[i] : indptr[i + 1]] = indices[
            np.int64(indptr[i]) + np.argsort(values[indptr[i] : indptr[i + 1]])
        ]
    return rv


@njit(parallel=True)
def sort_values(indptr, values):  # pragma: no cover (numba)
    rv = np.empty(indptr[-1], dtype=values.dtype)
    for i in prange(indptr.size - 1):
        rv[indptr[i] : indptr[i + 1]] = np.sort(values[indptr[i] : indptr[i + 1]])
    return rv


@njit(parallel=True)
def compact_values(old_indptr, new_indptr, values):  # pragma: no cover (numba)
    rv = np.empty(new_indptr[-1], dtype=values.dtype)
    for i in prange(new_indptr.size - 1):
        start = np.int64(new_indptr[i])
        offset = np.int64(old_indptr[i]) - start
        for j in range(start, new_indptr[i + 1]):
            rv[j] = values[j + offset]
    return rv


@njit(parallel=True)
def reverse_values(indptr, values):  # pragma: no cover (numba)
    rv = np.empty(indptr[-1], dtype=values.dtype)
    for i in prange(indptr.size - 1):
        offset = np.int64(indptr[i]) + np.int64(indptr[i + 1]) - 1
        for j in range(indptr[i], indptr[i + 1]):
            rv[j] = values[offset - j]
    return rv


@njit(parallel=True)
def compact_indices(indptr, k):  # pragma: no cover (numba)
    """Given indptr from hypercsr, create a new col_indices array that is compact.

    That is, for each row with degree N, the column indices will be 0..N-1.
    """
    if k is not None:
        indptr = create_indptr(indptr, k)
    col_indices = np.empty(indptr[-1], dtype=np.uint64)
    N = np.int64(0)
    for i in prange(indptr.size - 1):
        start = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - start
        N = max(N, deg)
        for j in range(deg):
            col_indices[start + j] = j
    return indptr, col_indices, N


@njit(parallel=True)
def choose_random1(indptr):  # pragma: no cover (numba)
    choices = np.empty(indptr.size - 1, dtype=indptr.dtype)
    new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
    for i in prange(indptr.size - 1):
        idx = np.int64(indptr[i])
        deg = np.int64(indptr[i + 1]) - idx
        if deg == 1:
            choices[i] = idx
        else:
            choices[i] = np.random.randint(idx, idx + deg)
    return choices, new_indptr


@njit
def create_indptr(indptr, k):  # pragma: no cover (numba)
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
def choose_random(indptr, k):  # pragma: no cover (numba)
    if k == 1:
        return choose_random1(indptr)

    # The results in choices don't need to be random.  In fact, it may
    # be nice to have them sorted if convenient to do so.
    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in prange(indptr.size - 1):
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
def choose_first(indptr, k):  # pragma: no cover (numba)
    if k == 1:
        choices = indptr[:-1]
        new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
        return choices, new_indptr

    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in prange(indptr.size - 1):
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
def choose_last(indptr, k):  # pragma: no cover (numba)
    if k == 1:
        choices = (indptr[1:].astype(np.int64) - 1).astype(indptr.dtype)
        new_indptr = np.arange(indptr.size, dtype=indptr.dtype)
        return choices, new_indptr

    new_indptr = create_indptr(indptr, k)
    choices = np.empty(new_indptr[-1], dtype=indptr.dtype)
    for i in prange(indptr.size - 1):
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


@njit
def issorted(arr):  # pragma: no cover (numba)
    if arr.size > 1:
        prev = arr[0]
        for i in range(1, arr.size):
            cur = arr[i]
            if cur == prev:
                continue
            if cur < prev:
                return False
            prev = cur
    return True


@njit
def indices_to_indptr(indices, size):  # pragma: no cover (numba)
    """Calculate the indptr for e.g. CSR from sorted COO rows."""
    indptr = np.zeros(size, dtype=indices.dtype)
    index = np.uint64(0)
    one = np.uint64(1)
    for i in range(indices.size):
        row = indices[i]
        if row != index:
            indptr[index + one] = i
            index = row
    indptr[index + one] = indices.size
    return indptr


@njit(parallel=True)
def indptr_to_indices(indptr):  # pragma: no cover (numba)
    indices = np.empty(indptr[-1], dtype=indptr.dtype)
    for i in prange(indptr.size - 1):
        for j in range(indptr[i], indptr[i + 1]):
            indices[j] = i
    return indices


from .prefix_scan import prefix_scan  # noqa: E402 isort:skip

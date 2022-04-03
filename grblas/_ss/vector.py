import warnings

import numpy as np
from numba import njit
from suitesparse_graphblas.utils import claim_buffer, unclaim_buffer

import grblas as gb

from .. import ffi, lib, monoid
from ..base import call
from ..dtypes import _INDEX, INT64, UINT64, lookup_dtype
from ..exceptions import check_status, check_status_carg
from ..scalar import _as_scalar
from ..utils import _CArray, ints_to_numpy_buffer, libget, values_to_numpy_buffer, wrapdoc
from .matrix import MatrixArray, _concat_mn, normalize_chunks
from .prefix_scan import prefix_scan
from .utils import get_order

ffi_new = ffi.new


@njit
def _head_indices_vector_bitmap(bitmap, values, size, dtype, n, is_iso):  # pragma: no cover
    indices = np.empty(n, dtype=np.uint64)
    if is_iso:
        vals = np.empty(1, dtype=dtype)
        vals[0] = values[0]
    else:
        vals = np.empty(n, dtype=dtype)
    j = 0
    for i in range(size):
        if bitmap[i]:
            indices[j] = i
            if not is_iso:
                vals[j] = values[i]
            j += 1
            if j == n:
                break
    return indices, vals


def head(vector, n=10, dtype=None, *, sort=False):
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
    is_iso = vector.ss.is_iso
    d = vector.ss.unpack(raw=True, sort=sort)
    fmt = d["format"]
    try:
        if fmt == "full":
            indices = np.arange(n, dtype=np.uint64)
            vals = d["values"][:n].astype(dtype.np_type)
        elif fmt == "bitmap":
            indices, vals = _head_indices_vector_bitmap(
                d["bitmap"], d["values"], d["size"], dtype.np_type, n, is_iso
            )
        elif fmt == "sparse":
            indices = d["indices"][:n].copy()
            vals = d["values"][:n].astype(dtype.np_type)
        else:  # pragma: no cover
            raise RuntimeError(f"Invalid format: {fmt}")
    finally:
        vector.ss.pack_any(take_ownership=True, **d)
    if is_iso:
        vals = np.broadcast_to(vals[:1], (n,))
    return indices, vals


class ss:
    __slots__ = "_parent"

    def __init__(self, parent):
        self._parent = parent

    @property
    def nbytes(self):
        size = ffi_new("size_t*")
        check_status(lib.GxB_Vector_memoryUsage(size, self._parent._carg), self._parent)
        return size[0]

    @property
    def is_iso(self):
        is_iso = ffi_new("bool*")
        check_status(lib.GxB_Vector_iso(is_iso, self._parent._carg), self._parent)
        return is_iso[0]

    @property
    def iso_value(self):
        if self.is_iso:
            return self._parent.reduce(monoid.any).new(name="")
        raise ValueError("Vector is not iso-valued")

    @property
    def format(self):
        parent = self._parent
        sparsity_ptr = ffi_new("GxB_Option_Field*")
        check_status(
            lib.GxB_Vector_Option_get(parent._carg, lib.GxB_SPARSITY_STATUS, sparsity_ptr),
            parent,
        )
        sparsity_status = sparsity_ptr[0]
        if sparsity_status == lib.GxB_SPARSE:
            format = "sparse"
        elif sparsity_status == lib.GxB_BITMAP:
            format = "bitmap"
        elif sparsity_status == lib.GxB_FULL:
            format = "full"
        else:  # pragma: no cover
            raise NotImplementedError(f"Unknown sparsity status: {sparsity_status}")
        return format

    def diag(self, matrix, k=0):
        """
        GxB_Vector_diag

        **This function is deprecated.  Use ``Matrix.diag`` or ``Vector.ss.build_diag`` instead.**

        """
        warnings.warn(
            "`Matrix.ss.diag` is deprecated; "
            "please use `Matrix.diag` or `Vector.ss.build_diag` instead",
            DeprecationWarning,
        )
        self.build_diag(matrix, k)

    def build_diag(self, matrix, k=0):
        """
        GxB_Vector_diag

        Extract a diagonal from a Matrix or TransposedMatrix into a Vector.
        Existing entries in the Vector are discarded.

        Parameters
        ----------
        matrix : Matrix or TransposedMatrix
            Extract a diagonal from this matrix.
        k : int, default 0
            Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
            and `k<0` for diagonals below the main diagonal.

        See Also
        --------
        Matrix.diag
        Vector.diag

        """
        matrix = self._parent._expect_type(
            matrix,
            (gb.Matrix, gb.matrix.TransposedMatrix),
            within="ss.build_diag",
            argname="matrix",
        )
        if type(matrix) is gb.matrix.TransposedMatrix:
            # Transpose descriptor doesn't do anything, so use the parent
            k = -k
            matrix = matrix._matrix
        call("GxB_Vector_diag", [self._parent, matrix, _as_scalar(k, INT64, is_cscalar=True), None])

    def split(self, chunks, *, name=None):
        """
        GxB_Matrix_split

        Split a Vector into a 1D array of sub-vectors according to `chunks`.

        This performs the opposite operation as ``concat``.

        `chunks` is short for "chunksizes" and indicates the chunk sizes.
        `chunks` may be a single integer, or a tuple or list.  Example chunks:

        - ``chunks=10``
            - Split vector into chunks of size 10 (the last chunk may be smaller).
        - ``chunks=[5, 10]``
            - Split vector into two chunks of size 5 and 10.

        See Also
        --------
        Vector.ss.concat
        grblas.ss.concat
        """
        from ..vector import Vector

        tile_nrows, _ = normalize_chunks([chunks, None], (self._parent._size, 1))
        m = len(tile_nrows)
        tiles = ffi.new("GrB_Matrix[]", m)
        parent = self._parent._as_matrix()
        call(
            "GxB_Matrix_split",
            [
                MatrixArray(tiles, parent, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(1, _INDEX, is_cscalar=True),
                _CArray(tile_nrows),
                _CArray([1]),
                parent,
                None,
            ],
        )
        rv = []
        dtype = self._parent.dtype
        if name is None:
            name = self._parent.name
        for i, size in enumerate(tile_nrows):
            # Copy to a new handle so we can free `tiles`
            new_vector = ffi.new("GrB_Vector*")
            new_vector[0] = ffi.cast("GrB_Vector", tiles[i])
            tile = Vector(new_vector, dtype, name=f"{name}_{i}")
            tile._size = size
            rv.append(tile)
        return rv

    def _concat(self, tiles, m):
        ctiles = ffi.new("GrB_Matrix[]", m)
        for i, tile in enumerate(tiles):
            ctiles[i] = tile.gb_obj[0]
        call(
            "GxB_Matrix_concat",
            [
                self._parent._as_matrix(),
                MatrixArray(ctiles, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(1, _INDEX, is_cscalar=True),
                None,
            ],
        )

    def concat(self, tiles):
        """
        GxB_Matrix_concat

        Concatenate a 1D list of Vector objects into the current Vector.
        Any existing values in the current Vector will be discarded.
        To concatenate into a new Vector, use `grblas.ss.concat`.

        This performs the opposite operation as ``split``.

        See Also
        --------
        Vector.ss.split
        grblas.ss.concat
        """
        tiles, m, n, is_matrix = _concat_mn(tiles, is_matrix=False)
        self._concat(tiles, m)

    def build_scalar(self, indices, value):
        """
        GxB_Vector_build_Scalar

        Like ``build``, but uses a scalar for all the values.

        See Also
        --------
        Vector.build
        Vector.from_values
        """
        indices = ints_to_numpy_buffer(indices, np.uint64, name="indices")
        scalar = _as_scalar(value, self._parent.dtype, is_cscalar=False)  # pragma: is_grbscalar
        call(
            "GxB_Vector_build_Scalar",
            [
                self._parent,
                _CArray(indices),
                scalar,
                _as_scalar(indices.size, _INDEX, is_cscalar=True),
            ],
        )

    def export(self, format=None, *, sort=False, give_ownership=False, raw=False):
        """
        GxB_Vextor_export_xxx

        Parameters
        ----------
        format : str or None, default None
            If `format` is not specified, this method exports in the currently stored format.
            To control the export format, set `format` to one of:
                - "sparse"
                - "bitmap"
                - "full"
        sort : bool, default False
            Whether to sort indices if the format is "sparse"
        give_ownership : bool, default False
            Perform a zero-copy data transfer to Python if possible.  This gives ownership of
            the underlying memory buffers to Numpy.
            ** If True, this nullifies the current object, which should no longer be used! **
        raw : bool, default False
            If True, always return array the same size as returned by SuiteSparse.
            If False, arrays may be trimmed to be the expected size.
            It may make sense to choose ``raw=True`` if one wants to use the data to perform
            a zero-copy import back to SuiteSparse.

        Returns
        -------
        dict; keys depend on `format` and `raw` arguments (see below).

        See Also
        --------
        Vector.to_values
        Vector.ss.import_any

        Return values
            - Note: for `raw=True`, arrays may be larger than specified.
            - "sparse" format
                - indices : ndarray(dtype=uint64, size=nvals)
                - values : ndarray(size=nvals)
                - sorted_index : bool
                    - True if the values in "indices" are sorted
                - size : int
                - nvals : int, only present if raw == True
            - "bitmap" format
                - bitmap : ndarray(dtype=bool8, size=size)
                - values : ndarray(size=size)
                    - Elements where bitmap is False are undefined
                - nvals : int
                    - The number of True elements in the bitmap
                - size : int, only present if raw == True
            - "bitmap" format
                - bitmap : ndarray(dtype=bool8, size=size)
                - values : ndarray(size=size)
                    - Elements where bitmap is False are undefined
                - nvals : int
                    - The number of True elements in the bitmap
                - size : int, only present if raw == True
            - "full" format
                - values : ndarray(size=size)
                - size : int, only present if raw == True or is_iso == True

        Examples
        --------
        Simple usage:

        >>> pieces = v.ss.export()
        >>> v2 = Vector.ss.import_any(**pieces)
        """
        return self._export(
            format=format, sort=sort, give_ownership=give_ownership, raw=raw, method="export"
        )

    def unpack(self, format=None, *, sort=False, raw=False):
        """
        GxB_Vector_unpack_xxx

        `unpack` is like `export`, except that the Vector remains valid but empty.
        `pack_*` methods are the opposite of `unpack`.

        See `Vector.ss.export` documentation for more details.
        """
        return self._export(format=format, sort=sort, give_ownership=True, raw=raw, method="unpack")

    def _export(self, format=None, *, sort=False, give_ownership=False, raw=False, method):
        if give_ownership:
            parent = self._parent
        else:
            parent = self._parent.dup(name=f"v_{method}")
        dtype = np.dtype(parent.dtype.np_type)
        index_dtype = np.dtype(np.uint64)

        if format is None:
            format = self.format
        else:
            format = format.lower()

        size = parent._size
        if method == "export":
            vhandle = ffi_new("GrB_Vector*", parent._carg)
            type_ = ffi_new("GrB_Type*")
            size_ = ffi_new("GrB_Index*")
            args = (type_, size_)
        else:
            vhandle = parent._carg
            args = ()
        vx = ffi_new("void**")
        vx_size = ffi_new("GrB_Index*")
        if sort:
            jumbled = ffi.NULL
        else:
            jumbled = ffi_new("bool*")
        is_iso = ffi_new("bool*")
        if format == "sparse":
            vi = ffi_new("GrB_Index**")
            vi_size = ffi_new("GrB_Index*")
            nvals = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Vector_{method}_CSC")(
                    vhandle,
                    *args,
                    vi,
                    vx,
                    vi_size,
                    vx_size,
                    is_iso,
                    nvals,
                    jumbled,
                    ffi.NULL,
                ),
                parent,
            )
            is_iso = is_iso[0]
            nvals = nvals[0]
            indices = claim_buffer(ffi, vi[0], vi_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indices.size > nvals:  # pragma: no cover
                    indices = indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > nvals:  # pragma: no cover
                        values = values[:nvals]
            rv = {
                "size": size,
                "indices": indices,
                "sorted_index": True if sort else not jumbled[0],
            }
            if raw:
                rv["nvals"] = nvals
        elif format == "bitmap":
            vb = ffi_new("int8_t**")
            vb_size = ffi_new("GrB_Index*")
            nvals = ffi_new("GrB_Index*")
            check_status(
                libget(f"GxB_Vector_{method}_Bitmap")(
                    vhandle, *args, vb, vx, vb_size, vx_size, is_iso, nvals, ffi.NULL
                ),
                parent,
            )
            is_iso = is_iso[0]
            bool_dtype = np.dtype(np.bool8)
            bitmap = claim_buffer(ffi, vb[0], vb_size[0] // bool_dtype.itemsize, bool_dtype)
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if bitmap.size > size:  # pragma: no cover
                    bitmap = bitmap[:size]
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > size:  # pragma: no cover
                        values = values[:size]
            rv = {
                "bitmap": bitmap,
                "nvals": nvals[0],
            }
            if raw:
                rv["size"] = size
        elif format == "full":
            check_status(
                libget(f"GxB_Vector_{method}_Full")(vhandle, *args, vx, vx_size, is_iso, ffi.NULL),
                parent,
            )
            is_iso = is_iso[0]
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if is_iso:
                    if values.size > 1:  # pragma: no cover
                        values = values[:1]
                else:
                    if values.size > size:  # pragma: no cover
                        values = values[:size]
            rv = {}
            if raw or is_iso:
                rv["size"] = size
        else:
            raise ValueError(f"Invalid format: {format}")

        rv["is_iso"] = is_iso
        rv.update(
            format=format,
            values=values,
        )
        if method == "export":
            parent.gb_obj = ffi.NULL
        return rv

    @classmethod
    def import_any(
        cls,
        *,
        # All
        values,
        size=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        dtype=None,
        name=None,
        # Sparse
        indices=None,
        sorted_index=False,
        # Bitmap
        bitmap=None,
        # Bitmap/Sparse
        nvals=None,  # optional
    ):
        """
        GxB_Vector_import_xxx

        Dispatch to appropriate import method inferred from inputs.
        See the other import functions and `Vector.ss.export`` for details.

        Returns
        -------
        Vector

        See Also
        --------
        Vector.from_values
        Vector.ss.export
        Vector.ss.import_sparse
        Vector.ss.import_bitmap
        Vector.ss.import_full

        Examples
        --------
        Simple usage:

        >>> pieces = v.ss.export()
        >>> v2 = Vector.ss.import_any(**pieces)
        """
        return cls._import_any(
            values=values,
            size=size,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            dtype=dtype,
            name=name,
            # Sparse
            indices=indices,
            sorted_index=sorted_index,
            # Bitmap
            bitmap=bitmap,
            # Bitmap/Sparse
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
        # Sparse
        indices=None,
        sorted_index=False,
        # Bitmap
        bitmap=None,
        # Bitmap/Sparse
        nvals=None,  # optional
        # Unused for pack
        size=None,
        dtype=None,
        name=None,
    ):
        """
        GxB_Vector_pack_xxx

        `pack_any` is like `import_any` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack()``

        See `Vector.ss.import_any` documentation for more details.
        """
        return self._import_any(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            # Sparse
            indices=indices,
            sorted_index=sorted_index,
            # Bitmap
            bitmap=bitmap,
            # Bitmap/Sparse
            nvals=nvals,
            method="pack",
            vector=self._parent,
        )

    @classmethod
    def _import_any(
        cls,
        *,
        # All
        values,
        size=None,
        is_iso=False,
        take_ownership=False,
        format=None,
        dtype=None,
        name=None,
        # Sparse
        indices=None,
        sorted_index=False,
        # Bitmap
        bitmap=None,
        # Bitmap/Sparse
        nvals=None,  # optional
        method,
        vector=None,
    ):
        if format is None:
            if indices is not None:
                if bitmap is not None:
                    raise TypeError("Cannot provide both `indptr` and `bitmap`")
                format = "sparse"
            elif bitmap is not None:
                format = "bitmap"
            else:
                format = "full"
        else:
            format = format.lower()
        if method == "pack":
            obj = vector.ss
        else:
            obj = cls
        if format == "sparse":
            return getattr(obj, f"{method}_sparse")(
                size=size,
                indices=indices,
                values=values,
                nvals=nvals,
                is_iso=is_iso,
                sorted_index=sorted_index,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "bitmap":
            return getattr(obj, f"{method}_bitmap")(
                nvals=nvals,
                bitmap=bitmap,
                values=values,
                size=size,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        elif format == "full":
            return getattr(obj, f"{method}_full")(
                values=values,
                size=size,
                is_iso=is_iso,
                take_ownership=take_ownership,
                dtype=dtype,
                name=name,
            )
        else:
            raise ValueError(f"Invalid format: {format}")

    @classmethod
    def import_sparse(
        cls,
        *,
        size,
        indices,
        values,
        nvals=None,
        is_iso=False,
        sorted_index=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Vector_import_CSC

        Create a new Vector from sparse input.

        Parameters
        ----------
        size : int
        indices : array-like
        values : array-like
        nvals : int, optional
            The number of elements in "values" to use.
            If not specified, will be set to ``len(values)``.
        is_iso : bool, default False
            Is the Vector iso-valued (meaning all the same value)?
            If true, then `values` should be a length 1 array.
        sorted_index : bool, default False
            Indicate whether the values in "col_indices" are sorted.
        take_ownership : bool, default False
            If True, perform a zero-copy data transfer from input numpy arrays
            to GraphBLAS if possible.  To give ownership of the underlying
            memory buffers to GraphBLAS, the arrays must:
                - be C contiguous
                - have the correct dtype (uint64 for indices)
                - own its own data
                - be writeable
            If all of these conditions are not met, then the data will be
            copied and the original array will be unmodified.  If zero copy
            to GraphBLAS is successful, then the array will be modified to be
            read-only and will no longer own the data.
        dtype : dtype, optional
            dtype of the new Vector.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "sparse" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Vector.

        Returns
        -------
        Vector
        """
        return cls._import_sparse(
            size=size,
            indices=indices,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            sorted_index=sorted_index,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_sparse(
        self,
        *,
        indices,
        values,
        nvals=None,
        is_iso=False,
        sorted_index=False,
        take_ownership=False,
        format=None,
        **ignored_kwargs,
    ):
        """
        GxB_Vector_pack_CSC

        `pack_sparse` is like `import_sparse` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("sparse")``

        See `Vector.ss.import_sparse` documentation for more details.
        """
        return self._import_sparse(
            indices=indices,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            sorted_index=sorted_index,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            vector=self._parent,
        )

    @classmethod
    def _import_sparse(
        cls,
        *,
        size=None,
        indices,
        values,
        nvals=None,
        is_iso=False,
        sorted_index=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
    ):
        if format is not None and format.lower() != "sparse":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'sparse'.")
        copy = not take_ownership
        indices = ints_to_numpy_buffer(indices, np.uint64, copy=copy, ownable=True, name="indices")
        if method == "pack":
            dtype = vector.dtype
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if indices is values:
            values = np.copy(values)
        vi = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indices))
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvals is None:
            if is_iso:
                nvals = indices.size
            else:
                nvals = values.size
        if method == "import":
            vhandle = ffi_new("GrB_Vector*")
            args = (dtype._carg, size)
        else:
            vhandle = vector._carg
            args = ()
        status = libget(f"GxB_Vector_{method}_CSC")(
            vhandle,
            *args,
            vi,
            vx,
            indices.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            not sorted_index,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector(vhandle, dtype, name=name)
            vector._size = size
        else:
            check_status(status, vector)
        unclaim_buffer(indices)
        unclaim_buffer(values)
        return vector

    @classmethod
    def import_bitmap(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        size=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Vector_import_Bitmap

        Create a new Vector from values and bitmap (as mask) arrays.

        Parameters
        ----------
        bitmap : array-like
            True elements indicate where there are values in "values".
        values : array-like
        nvals : int, optional
            The number of True elements in the bitmap for this Vector.
        size : int, optional
            The size of the new Vector.
            If not specified, it will be set to the size of values.
        is_iso : bool, default False
            Is the Vector iso-valued (meaning all the same value)?
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
            dtype of the new Vector.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "bitmap" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Vector.

        Returns
        -------
        Vector
        """
        return cls._import_bitmap(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            size=size,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_bitmap(
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
        GxB_Vector_pack_Bitmap

        `pack_bitmap` is like `import_bitmap` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("bitmap")``

        See `Vector.ss.import_bitmap` documentation for more details.
        """
        return self._import_bitmap(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            vector=self._parent,
        )

    @classmethod
    def _import_bitmap(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        size=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
    ):
        if format is not None and format.lower() != "bitmap":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'bitmap'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(bitmap, np.bool8, copy=copy, ownable=True, name="bitmap")
        if method == "pack":
            dtype = vector.dtype
            size = vector._size
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if bitmap is values:
            values = np.copy(values)
        vhandle = ffi_new("GrB_Vector*")
        vb = ffi_new("int8_t**", ffi.from_buffer("int8_t*", bitmap))
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if size is None:
            if is_iso:
                size = bitmap.size
            else:
                size = values.size
        if nvals is None:
            if bitmap.size == size:
                nvals = np.count_nonzero(bitmap)
            else:
                nvals = np.count_nonzero(bitmap.ravel()[:size])
        if method == "import":
            vhandle = ffi_new("GrB_Vector*")
            args = (dtype._carg, size)
        else:
            vhandle = vector._carg
            args = ()
        status = libget(f"GxB_Vector_{method}_Bitmap")(
            vhandle,
            *args,
            vb,
            vx,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector(vhandle, dtype, name=name)
            vector._size = size
        else:
            check_status(status, vector)
        unclaim_buffer(bitmap)
        unclaim_buffer(values)
        return vector

    @classmethod
    def import_full(
        cls,
        values,
        *,
        size=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
    ):
        """
        GxB_Vector_import_Full

        Create a new Vector from values.

        Parameters
        ----------
        values : array-like
        size : int, optional
            The size of the new Vector.
            If not specified, it will be set to the size of values.
        is_iso : bool, default False
            Is the Vector iso-valued (meaning all the same value)?
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
            dtype of the new Vector.
            If not specified, this will be inferred from `values`.
        format : str, optional
            Must be "full" or None.  This is included to be compatible with
            the dict returned from exporting.
        name : str, optional
            Name of the new Vector.

        Returns
        -------
        Vector
        """
        return cls._import_full(
            values=values,
            size=size,
            is_iso=is_iso,
            take_ownership=take_ownership,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
        )

    def pack_full(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        format=None,
        **unused_kwargs,
    ):
        """
        GxB_Vector_pack_Full

        `pack_full` is like `import_full` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("full")``

        See `Vector.ss.import_full` documentation for more details.
        """
        return self._import_full(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            format=format,
            method="pack",
            vector=self._parent,
        )

    @classmethod
    def _import_full(
        cls,
        *,
        values,
        size=None,
        is_iso=False,
        take_ownership=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
    ):
        if format is not None and format.lower() != "full":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'full'.")
        copy = not take_ownership
        if method == "pack":
            dtype = vector.dtype
            size = vector._size
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        vhandle = ffi_new("GrB_Vector*")
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if size is None:
            size = values.size
        if method == "import":
            vhandle = ffi_new("GrB_Vector*")
            args = (dtype._carg, size)
        else:
            vhandle = vector._carg
            args = ()
        status = libget(f"GxB_Vector_{method}_Full")(
            vhandle,
            *args,
            vx,
            values.nbytes,
            is_iso,
            ffi.NULL,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector(vhandle, dtype, name=name)
            vector._size = size
        else:
            check_status(status, vector)
        unclaim_buffer(values)
        return vector

    @wrapdoc(head)
    def head(self, n=10, dtype=None, *, sort=False):
        return head(self._parent, n, dtype, sort=sort)

    def scan(self, op=monoid.plus, *, name=None):
        """Perform a prefix scan with the given monoid.

        For example, use `monoid.plus` (the default) to perform a cumulative sum,
        and `monoid.times` for cumulative product.  Works with any monoid.

        Returns
        -------
        Scalar
        """
        return prefix_scan(self._parent, op, name=name, within="scan")

    def reshape(self, nrows, ncols=None, order="rowwise", *, name=None):
        """Return a copy of the Vector as a Matrix of the given shape.

        The shape of the Matrix must be compatible with the original shape.
        That is, the number of elements must be equal: ``nrows * ncols == size``.
        One of the dimensions may be -1, which will infer the correct size.

        Parameters
        ----------
        nrows : int or tuple of ints
        ncols : int or None
        order : {"rowwise", "columnwise"}, optional
            "rowwise" means to fill the Matrix in row-major (C-style) order.
            Aliases of "rowwise" also accepted: "row", "rows", "C".
            "columnwise" means to fill the Matrix in column-major (F-style) order.
            Aliases of "rowwise" also accepted: "col", "cols", "column", "columns", "F".
            The default is "rowwise".
        name : str, optional
            Name of the new Matrix.

        Returns
        -------
        Matrix

        See Also
        --------
        Matrix.ss.flatten : flatten a Matrix into a Vector.
        """
        order = get_order(order)
        array = np.broadcast_to(False, self._parent._size)
        if ncols is None:
            array = array.reshape(nrows)
        else:
            array = array.reshape(nrows, ncols)
        if array.ndim != 2:
            raise ValueError(f"Shape tuple must be of length 2, not {array.ndim}")
        nrows, ncols = array.shape
        fmt = self.format
        if fmt == "sparse":
            info = self.export(sort=True)
            indices = info["indices"]
            if order == "rowwise":
                return gb.Matrix.ss.import_coor(
                    nrows=nrows,
                    ncols=ncols,
                    rows=indices // ncols,
                    cols=indices % ncols,
                    values=info["values"],
                    is_iso=info["is_iso"],
                    sorted_cols=True,
                    take_ownership=True,
                    name=name,
                )
            else:
                return gb.Matrix.ss.import_cooc(
                    nrows=nrows,
                    ncols=ncols,
                    cols=indices // nrows,
                    rows=indices % nrows,
                    values=info["values"],
                    is_iso=info["is_iso"],
                    sorted_rows=True,
                    take_ownership=True,
                    name=name,
                )
        elif fmt == "bitmap":
            info = self.export(raw=True)
            if order == "rowwise":
                method = gb.Matrix.ss.import_bitmapr
            else:
                method = gb.Matrix.ss.import_bitmapc
            return method(
                nrows=nrows,
                ncols=ncols,
                bitmap=info["bitmap"],
                values=info["values"],
                nvals=info["nvals"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        elif fmt == "full":
            info = self.export(raw=True)
            if order == "rowwise":
                method = gb.Matrix.ss.import_fullr
            else:
                method = gb.Matrix.ss.import_fullc
            return method(
                nrows=nrows,
                ncols=ncols,
                values=info["values"],
                is_iso=info["is_iso"],
                take_ownership=True,
                name=name,
            )
        else:
            raise NotImplementedError(fmt)

    def selectk(self, how, k, *, name=None):
        """Select (up to) k elements.

        Parameters
        ----------
        how : str
            - "random": choose k elements with equal probability
            - "first": choose the first k elements
            - "last": choose the last k elements
            - "largest": choose the k largest elements.  If tied, any may be chosen.
            - "smallest": choose the k smallest elements.  If tied, any may be chosen.
        k : int
            The number of elements to choose

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**
        """
        how = how.lower()
        if k < 0:
            raise ValueError("negative k is not allowed")
        do_sort = how in {"first", "last"}
        info = self._parent.ss.export("sparse", sort=do_sort)
        if how == "random":
            choices = random_choice(self._parent._nvals, k)
        elif how == "first" or info["is_iso"] and how in {"largest", "smallest"}:
            choices = slice(None, k)
        elif how == "last":
            choices = slice(-k, None)
        elif how == "largest":
            choices = np.argpartition(info["values"], -k)[-k:]  # not sorted
        elif how == "smallest":
            choices = np.argpartition(info["values"], k)[:k]  # not sorted
        else:
            raise ValueError(
                '`how` argument must be one of: "random", "first", "last", "largest", "smallest"'
            )
        newinfo = dict(info, indices=info["indices"][choices])
        if not info["is_iso"]:
            newinfo["values"] = info["values"][choices]
        if k == 1:
            newinfo["sorted_index"] = True
        elif not do_sort:
            newinfo["sorted_index"] = False
        return gb.Vector.ss.import_sparse(
            **newinfo,
            take_ownership=True,
            name=name,
        )

    def compactify(self, how="first", size=None, *, reverse=False, asindex=False, name=None):
        """Shift all values to the beginning so all values are contiguous.

        This returns a new Vector.

        Parameters
        ----------
        how : {"first", "last", "smallest", "largest", "random"}, optional
            How to compress the values:
            - first : take the values nearest the beginning
            - last : take the values nearest the end
            - smallest : take the smallest values (if tied, may take any)
            - largest : take the largest values (if tied, may take any)
            - random : take values randomly with equal probability and without replacement
        reverse : bool, default False
            Reverse the values when True
        asindex : bool, default False
            Return the index of the value when True.  If there are ties for
            "smallest" and "largest", then any valid index may be returned.
        size : int, optional
            The size of the returned Vector.  If not specified, then the Vector
            will be "compacted" to the smallest size that doesn't lose values.

        **THIS API IS EXPERIMENTAL AND MAY CHANGE**

        """
        how = how.lower()
        if how not in {"first", "last", "smallest", "largest", "random"}:
            raise ValueError(
                '`how` argument must be one of: "first", "last", "smallest", "largest", "random"'
            )
        if size is None and self._parent._nvals == 0 or size == 0:
            if asindex:
                return gb.Vector.new(UINT64, size=0, name=name)
            else:
                return gb.Vector.new(self._parent.dtype, size=0, name=name)
        do_sort = how in {"first", "last"}
        info = self._parent.ss.export("sparse", sort=do_sort)
        if size is None:
            size = info["indices"].size
        if info["is_iso"]:
            if how in {"smallest", "largest"} or how == "random" and not asindex:
                # order of smallest/largest/random doesn't matter
                how = "first"
                reverse = False
            if not asindex:
                how = "finished"
                reverse = False
            else:
                info["is_iso"] = False

        if how == "random":
            choices = random_choice(self._parent._nvals, size)
        elif how == "first":
            if reverse:
                choices = slice(size - 1, None, -1)
                reverse = False
            else:
                choices = slice(None, size)
        elif how == "last":
            if reverse:
                choices = slice(-size, None)
                reverse = False
            else:
                choices = slice(None, -size - 1, -1)
        elif how in {"largest", "smallest"}:
            values = info["values"]
            if how == "largest":
                slc = slice(-size, None)
                stop = -size
                reverse = not reverse
            else:
                slc = slice(size)
                stop = size
            if asindex:
                if size < values.size:
                    idx = np.argpartition(values, stop)[slc]
                    choices = idx[np.argsort(values[idx])]
                else:
                    choices = np.argsort(values)
                values = info["indices"][choices]
            else:
                if size < values.size:
                    values = np.partition(values, stop)[slc]
                values.sort()
        else:
            choices = slice(None)
        if how not in {"largest", "smallest"}:
            if asindex:
                values = info["indices"][choices]
            else:
                values = info["values"][choices]
        if reverse:
            values = values[::-1]
        newinfo = dict(
            info,
            values=values,
            indices=np.arange(size, dtype=np.uint64),
            sorted_index=True,
            size=size,
        )
        return gb.Vector.ss.import_sparse(
            **newinfo,
            take_ownership=True,
            name=name,
        )


@njit
def random_choice(n, k):  # pragma: no cover
    if k >= n:
        return np.arange(n, dtype=np.uint64)
    choices = np.empty(k, dtype=np.uint64)
    if 2 * k <= n:
        if k == 1:
            # Select a single edge
            choices[0] = np.random.randint(n)
        elif k == 2:
            # Select two edges
            choices[0] = np.random.randint(n)
            choices[1] = np.random.randint(n - 1)
            if choices[0] <= choices[1]:
                choices[1] += 1
        else:
            # Move the ones we want to keep to the front of `a`
            a = np.arange(n)
            for i in range(k):
                j = np.random.randint(i, n)
                a[i], a[j] = a[j], a[i]
                choices[i] = a[i]
    elif k == n - 1:
        # Select all but one edge
        j = np.random.randint(n)
        for i in range(j):
            choices[i] = i
        for i in range(j + 1, n):
            choices[i - 1] = i
    elif k == n - 2:
        # Select all but two edges
        j = np.random.randint(n)
        k = np.random.randint(n - 1)
        if j <= k:
            k += 1
            j, k = k, j
        for i in range(k):
            choices[i] = i
        for i in range(k + 1, j):
            choices[i - 1] = i
        for i in range(j + 1, n):
            choices[i - 2] = i
    else:
        # Move the ones we don't want to keep to the front of `a`
        a = np.arange(n)
        for i in range(n - k):
            j = np.random.randint(i, n)
            a[i], a[j] = a[j], a[i]
        n -= k
        for i in range(k):
            choices[i] = a[n + i]
    return choices

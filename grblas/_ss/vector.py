import numpy as np
from numba import njit
from suitesparse_graphblas.utils import claim_buffer, unclaim_buffer

import grblas as gb

from .. import ffi, lib, monoid
from ..base import call
from ..dtypes import INT64, lookup_dtype
from ..exceptions import check_status, check_status_carg
from ..scalar import _CScalar
from ..utils import (
    _CArray,
    ints_to_numpy_buffer,
    libget,
    values_to_numpy_buffer,
    wrapdoc,
)
from .prefix_scan import prefix_scan
from .scalar import gxb_scalar

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


def head(vector, n=10, *, sort=False, dtype=None):
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
        grblas.ss.diag
        Matrix.ss.diag

        """
        matrix = self._parent._expect_type(
            matrix, (gb.Matrix, gb.matrix.TransposedMatrix), within="ss.diag", argname="matrix"
        )
        if type(matrix) is gb.matrix.TransposedMatrix:
            # Transpose descriptor doesn't do anything, so use the parent
            k = -k
            matrix = matrix._matrix
        call("GxB_Vector_diag", [self._parent, matrix, _CScalar(k, dtype=INT64), None])

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
        scalar = gxb_scalar(self._parent.dtype, value)
        status = lib.GxB_Vector_build_Scalar(
            self._parent._carg, _CArray(indices)._carg, scalar[0], indices.size
        )
        check_status(status, self._parent)

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

        if is_iso:
            rv["is_iso"] = True
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
    def head(self, n=10, *, sort=False, dtype=None):
        return head(self._parent, n, sort=sort, dtype=dtype)

    def scan(self, op=monoid.plus, *, name=None):
        """Perform a prefix scan with the given monoid.

        For example, use `monoid.plus` (the default) to perform a cumulative sum,
        and `monoid.times` for cumulative product.  Works with any monoid.

        Returns
        -------
        Scalar
        """
        return prefix_scan(self._parent, op, name=name, within="scan")

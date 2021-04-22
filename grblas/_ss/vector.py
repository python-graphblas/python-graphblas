import numpy as np
import grblas as gb
from numba import njit
from suitesparse_graphblas.utils import claim_buffer, unclaim_buffer
from .. import ffi, lib
from ..dtypes import lookup_dtype
from ..exceptions import check_status, check_status_carg
from ..utils import ints_to_numpy_buffer, values_to_numpy_buffer, wrapdoc

ffi_new = ffi.new


@njit
def _head_indices_vector_bitmap(bitmap, values, size, dtype, n):  # pragma: no cover
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
        rebuilt = ss.import_any(take_ownership=True, name="", **d)
        # We need to set rebuilt.gb_obj to NULL so it doesn't get deleted early, so might
        # as well do a swap, b/c vector.gb_obj is already "destroyed" from the export.
        vector.gb_obj, rebuilt.gb_obj = rebuilt.gb_obj, vector.gb_obj  # pragma: no branch
    return indices, vals


class ss:
    __slots__ = "_parent"

    def __init__(self, parent):
        self._parent = parent

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

    def export(self, format=None, sort=False, give_ownership=False, raw=False):
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
                - size : int, only present if raw == True

        Examples
        --------
        Simple usage:

        >>> pieces = v.ss.export()
        >>> v2 = Vector.ss.import_any(**pieces)

        """
        if give_ownership:
            parent = self._parent
        else:
            parent = self._parent.dup(name="v_export")
        dtype = np.dtype(parent.dtype.np_type)
        index_dtype = np.dtype(np.uint64)

        if format is None:
            format = self.format
        else:
            format = format.lower()

        vhandle = ffi_new("GrB_Vector*", parent._carg)
        type_ = ffi_new("GrB_Type*")
        size = ffi_new("GrB_Index*")
        vx = ffi_new("void**")
        vx_size = ffi_new("GrB_Index*")
        if sort:
            jumbled = ffi.NULL
        else:
            jumbled = ffi_new("bool*")

        if format == "sparse":
            vi = ffi_new("GrB_Index**")
            vi_size = ffi_new("GrB_Index*")
            nvals = ffi_new("GrB_Index*")
            check_status(
                lib.GxB_Vector_export_CSC(
                    vhandle, type_, size, vi, vx, vi_size, vx_size, nvals, jumbled, ffi.NULL
                ),
                parent,
            )
            nvals = nvals[0]
            indices = claim_buffer(ffi, vi[0], vi_size[0], index_dtype)
            values = claim_buffer(ffi, vx[0], vx_size[0], dtype)
            if not raw:
                if indices.size > nvals:  # pragma: no cover
                    indices = indices[:nvals]
                if values.size > nvals:  # pragma: no cover
                    values = values[:nvals]
            rv = {
                "size": size[0],
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
                lib.GxB_Vector_export_Bitmap(
                    vhandle, type_, size, vb, vx, vb_size, vx_size, nvals, ffi.NULL
                ),
                parent,
            )
            bitmap = claim_buffer(ffi, vb[0], vb_size[0], np.dtype(np.bool8))
            values = claim_buffer(ffi, vx[0], vx_size[0], dtype)
            size = size[0]
            if not raw:
                if bitmap.size > size:  # pragma: no cover
                    bitmap = bitmap[:size]
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
                lib.GxB_Vector_export_Full(vhandle, type_, size, vx, vx_size, ffi.NULL),
                parent,
            )
            values = claim_buffer(ffi, vx[0], vx_size[0], dtype)
            size = size[0]
            if not raw:
                if values.size > size:  # pragma: no cover
                    values = values[:size]
            rv = {}
            if raw:
                rv["size"] = size
        else:
            raise ValueError(f"Invalid format: {format}")

        rv.update(
            format=format,
            values=values,
        )
        parent.gb_obj = ffi.NULL
        return rv

    @classmethod
    def import_any(
        cls,
        *,
        # All
        values,
        size=None,
        take_ownership=False,
        format=None,
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
        if format == "sparse":
            return cls.import_sparse(
                size=size,
                indices=indices,
                values=values,
                nvals=nvals,
                sorted_index=sorted_index,
                take_ownership=take_ownership,
                name=name,
            )
        elif format == "bitmap":
            return cls.import_bitmap(
                nvals=nvals,
                bitmap=bitmap,
                values=values,
                size=size,
                take_ownership=take_ownership,
                name=name,
            )
        elif format == "full":
            return cls.import_full(
                values=values,
                size=size,
                take_ownership=take_ownership,
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
            to GraphBLAS is successful, then the array will be mofied to be
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
        if format is not None and format.lower() != "sparse":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'sparse'.")
        copy = not take_ownership
        indices = ints_to_numpy_buffer(indices, np.uint64, copy=copy, ownable=True, name="indices")
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if indices is values:
            values = np.copy(values)
        vhandle = ffi_new("GrB_Vector*")
        vi = ffi_new("GrB_Index**", ffi.cast("GrB_Index*", ffi.from_buffer(indices)))
        vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
        if nvals is None:
            nvals = values.size
        check_status_carg(
            lib.GxB_Vector_import_CSC(
                vhandle,
                dtype._carg,
                size,
                vi,
                vx,
                indices.size,
                values.size,
                nvals,
                not sorted_index,
                ffi.NULL,
            ),
            "Vector",
            vhandle[0],
        )
        rv = gb.Vector(vhandle, dtype, name=name)
        rv._size = size
        unclaim_buffer(indices)
        unclaim_buffer(values)
        return rv

    @classmethod
    def import_bitmap(
        cls,
        *,
        bitmap,
        values,
        nvals=None,
        size=None,
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
            to GraphBLAS is successful, then the array will be mofied to be
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
        if format is not None and format.lower() != "bitmap":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'bitmap'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(bitmap, np.bool8, copy=copy, ownable=True, name="bitmap")
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        if bitmap is values:
            values = np.copy(values)
        vhandle = ffi_new("GrB_Vector*")
        vb = ffi_new("int8_t**", ffi.cast("int8_t*", ffi.from_buffer(bitmap)))
        vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
        if size is None:
            size = values.size
        if nvals is None:
            if bitmap.size == size:
                nvals = np.count_nonzero(bitmap)
            else:
                nvals = np.count_nonzero(bitmap.ravel()[:size])
        check_status_carg(
            lib.GxB_Vector_import_Bitmap(
                vhandle,
                dtype._carg,
                size,
                vb,
                vx,
                bitmap.size,
                values.size,
                nvals,
                ffi.NULL,
            ),
            "Vector",
            vhandle[0],
        )
        rv = gb.Vector(vhandle, dtype, name=name)
        rv._size = size
        unclaim_buffer(bitmap)
        unclaim_buffer(values)
        return rv

    @classmethod
    def import_full(
        cls, *, values, size=None, take_ownership=False, dtype=None, format=None, name=None
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
            to GraphBLAS is successful, then the array will be mofied to be
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
        if format is not None and format.lower() != "full":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'full'.")
        copy = not take_ownership
        values, dtype = values_to_numpy_buffer(values, dtype, copy=copy, ownable=True)
        vhandle = ffi_new("GrB_Vector*")
        vx = ffi_new("void**", ffi.cast("void**", ffi.from_buffer(values)))
        if size is None:
            size = values.size
        check_status_carg(
            lib.GxB_Vector_import_Full(
                vhandle,
                dtype._carg,
                size,
                vx,
                values.size,
                ffi.NULL,
            ),
            "Vector",
            vhandle[0],
        )
        rv = gb.Vector(vhandle, dtype, name=name)
        rv._size = size
        unclaim_buffer(values)
        return rv

    @wrapdoc(head)
    def head(self, n=10, *, sort=False, dtype=None):
        return head(self._parent, n, sort=sort, dtype=dtype)

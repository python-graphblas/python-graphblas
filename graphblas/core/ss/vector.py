import itertools

import numpy as np
from suitesparse_graphblas.utils import claim_buffer, unclaim_buffer

import graphblas as gb

from ... import binary, monoid
from ...dtypes import _INDEX, INT64, UINT64, lookup_dtype
from ...exceptions import _error_code_lookup, check_status, check_status_carg
from .. import NULL, ffi, lib
from ..base import call
from ..dtypes import _string_to_dtype
from ..operator import get_typed_op
from ..scalar import Scalar, _as_scalar
from ..utils import (
    _CArray,
    _MatrixArray,
    ints_to_numpy_buffer,
    normalize_chunks,
    values_to_numpy_buffer,
    wrapdoc,
)
from .config import BaseConfig
from .descriptor import get_descriptor
from .matrix import _concat_mn, njit
from .prefix_scan import prefix_scan

ffi_new = ffi.new


def head(vector, n=10, dtype=None, *, sort=False):
    """Like ``vector.to_coo()``, but only returns the first n elements.

    If sort is True, then the results will be sorted by index, otherwise the order of the
    result is not guaranteed.  Formats full and bitmap should always return in sorted order.
    """
    if vector._nvals <= n:
        return vector.to_coo(dtype, sort=sort)
    if sort:
        vector.wait()
    if dtype is None:
        dtype = vector.dtype
    else:
        dtype = lookup_dtype(dtype)
    indices, vals = zip(*itertools.islice(vector.ss.iteritems(), n), strict=True)
    return np.array(indices, np.uint64), np.array(vals, dtype.np_type)


class VectorConfig(BaseConfig):
    """Get and set configuration options for this Vector.

    See SuiteSparse:GraphBLAS documentation for more details.

    Config parameters
    -----------------
    bitmap_switch : double
        Threshold that determines when to switch to bitmap format
    sparsity_control : Set[str] from {"sparse", "bitmap", "full", "auto"}
        Allowed sparsity formats.  May be set with a single string or a set of strings.
    sparsity_status : str, {"sparse", "bitmap", "full"}
        Current sparsity format
    """

    _get_function = "GxB_Vector_Option_get"
    _set_function = "GxB_Vector_Option_set"
    _options = {
        "bitmap_switch": (lib.GxB_BITMAP_SWITCH, "double"),
        "sparsity_control": (lib.GxB_SPARSITY_CONTROL, "int"),
        # read-only
        "sparsity_status": (lib.GxB_SPARSITY_STATUS, "int"),
        # "format": (lib.GxB_FORMAT, "GxB_Format_Value"),  # Not useful to show
    }
    _bitwise = {
        "sparsity_control": {
            # "hypersparse": lib.GxB_HYPERSPARSE,  # For matrices, not vectors
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
        "sparsity_control": "auto",
    }
    _read_only = {"sparsity_status", "format"}


class ss:
    __slots__ = "_parent", "config"

    def __init__(self, parent):
        self._parent = parent
        self.config = VectorConfig(parent)

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
            # This may not be thread-safe if the parent is being modified in another thread
            return Scalar.from_value(next(self.itervalues()), dtype=self._parent.dtype, name="")
        raise ValueError("Vector is not iso-valued")

    @property
    def format(self):
        parent = self._parent
        sparsity_ptr = ffi_new("int32_t*")
        check_status(
            lib.GxB_Vector_Option_get_INT32(parent._carg, lib.GxB_SPARSITY_STATUS, sparsity_ptr),
            parent,
        )
        sparsity_status = sparsity_ptr[0]
        if sparsity_status == lib.GxB_SPARSE:
            format = "sparse"
        elif sparsity_status == lib.GxB_BITMAP:
            format = "bitmap"
        elif sparsity_status == lib.GxB_FULL:
            format = "full"
        else:  # pragma: no cover (sanity)
            raise NotImplementedError(f"Unknown sparsity status: {sparsity_status}")
        return format

    def build_diag(self, matrix, k=0, **opts):
        """GxB_Vector_diag.

        Extract a diagonal from a Matrix or TransposedMatrix into a Vector.
        Existing entries in the Vector are discarded.

        Parameters
        ----------
        matrix : Matrix or TransposedMatrix
            Extract a diagonal from this matrix.
        k : int, default 0
            Diagonal in question.  Use ``k>0`` for diagonals above the main diagonal,
            and ``k<0`` for diagonals below the main diagonal.

        See Also
        --------
        Matrix.diag
        Vector.diag

        """
        from ..matrix import Matrix, TransposedMatrix

        matrix = self._parent._expect_type(
            matrix,
            (Matrix, TransposedMatrix),
            within="ss.build_diag",
            argname="matrix",
        )
        if type(matrix) is TransposedMatrix:
            # Transpose descriptor doesn't do anything, so use the parent
            k = -k
            matrix = matrix._matrix
        call(
            "GxB_Vector_diag",
            [self._parent, matrix, _as_scalar(k, INT64, is_cscalar=True), get_descriptor(**opts)],
        )

    def split(self, chunks, *, name=None, **opts):
        """GxB_Matrix_split.

        Split a Vector into a 1D array of sub-vectors according to ``chunks``.

        This performs the opposite operation as ``concat``.

        ``chunks`` is short for "chunksizes" and indicates the chunk sizes.
        ``chunks`` may be a single integer, or a tuple or list.  Example chunks:

        - ``chunks=10``
            - Split vector into chunks of size 10 (the last chunk may be smaller).
        - ``chunks=[5, 10]``
            - Split vector into two chunks of size 5 and 10.

        See Also
        --------
        Vector.ss.concat
        graphblas.ss.concat

        """
        from ..vector import Vector

        tile_nrows, _ = normalize_chunks([chunks, None], (self._parent._size, 1))
        m = len(tile_nrows)
        tiles = ffi_new("GrB_Matrix[]", m)
        parent = self._parent._as_matrix()
        call(
            "GxB_Matrix_split",
            [
                _MatrixArray(tiles, parent, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(1, _INDEX, is_cscalar=True),
                _CArray(tile_nrows),
                _CArray([1]),
                parent,
                get_descriptor(**opts),
            ],
        )
        rv = []
        dtype = self._parent.dtype
        if name is None:
            name = self._parent.name
        for i, size in enumerate(tile_nrows):
            # Copy to a new handle so we can free `tiles`
            new_vector = ffi_new("GrB_Vector*")
            new_vector[0] = ffi.cast("GrB_Vector", tiles[i])
            tile = Vector._from_obj(new_vector, dtype, size, name=f"{name}_{i}")
            rv.append(tile)
        return rv

    def _concat(self, tiles, m, opts):
        ctiles = ffi_new("GrB_Matrix[]", m)
        for i, tile in enumerate(tiles):
            ctiles[i] = tile.gb_obj[0]
        call(
            "GxB_Matrix_concat",
            [
                self._parent._as_matrix(),
                _MatrixArray(ctiles, name="tiles"),
                _as_scalar(m, _INDEX, is_cscalar=True),
                _as_scalar(1, _INDEX, is_cscalar=True),
                get_descriptor(**opts),
            ],
        )

    def concat(self, tiles, **opts):
        """GxB_Matrix_concat.

        Concatenate a 1D list of Vector objects into the current Vector.
        Any existing values in the current Vector will be discarded.
        To concatenate into a new Vector, use ``graphblas.ss.concat``.

        This performs the opposite operation as ``split``.

        See Also
        --------
        Vector.ss.split
        graphblas.ss.concat

        """
        tiles, m, n, is_matrix = _concat_mn(tiles, is_matrix=False)
        self._concat(tiles, m, opts)

    def build_scalar(self, indices, value):
        """GxB_Vector_build_Scalar.

        Like ``build``, but uses a scalar for all the values.

        See Also
        --------
        Vector.build
        Vector.from_coo

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

    def _begin_iter(self, seek):
        it_ptr = ffi_new("GxB_Iterator*")
        info = lib.GxB_Iterator_new(it_ptr)
        it = it_ptr[0]
        success = lib.GrB_SUCCESS
        info = lib.GxB_Vector_Iterator_attach(it, self._parent._carg, NULL)
        if info != success:  # pragma: no cover (safety)
            lib.GxB_Iterator_free(it_ptr)
            raise _error_code_lookup[info]("Vector iterator failed to attach")
        if seek < 0:
            seek = max(0, seek + lib.GxB_Vector_Iterator_getpmax(it))
        info = lib.GxB_Vector_Iterator_seek(it, seek)
        if info != success:
            lib.GxB_Iterator_free(it_ptr)
            raise _error_code_lookup[info]("Vector iterator failed to seek")
        return it_ptr

    def iterkeys(self, seek=0):
        """Iterate over all the indices of a Vector.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Vector objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Vector should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        key_func = lib.GxB_Vector_Iterator_getIndex
        next_func = lib.GxB_Vector_Iterator_next
        try:
            while info == success:
                yield key_func(it)
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Vector iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def itervalues(self, seek=0):
        """Iterate over all the values of a Vector.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Vector objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Vector should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        val_func = getattr(lib, f"GxB_Iterator_get_{self._parent.dtype.name}")
        next_func = lib.GxB_Vector_Iterator_next
        try:
            while info == success:
                yield val_func(it)
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Vector iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def iteritems(self, seek=0):
        """Iterate over all the indices and values of a Vector.

        Parameters
        ----------
        seek : int, default 0
            Index of entry to seek to.  May be negative to seek backwards from the end.
            Vector objects in bitmap format seek as if it's full format (i.e., it
            ignores the bitmap mask).

        The Vector should not be modified during iteration; doing so will
        result in undefined behavior.

        """
        try:
            it_ptr = self._begin_iter(seek)
        except StopIteration:
            return
        it = it_ptr[0]
        info = success = lib.GrB_SUCCESS
        key_func = lib.GxB_Vector_Iterator_getIndex
        val_func = getattr(lib, f"GxB_Iterator_get_{self._parent.dtype.name}")
        next_func = lib.GxB_Vector_Iterator_next
        try:
            while info == success:
                yield (key_func(it), val_func(it))
                info = next_func(it)
        except GeneratorExit:
            pass
        else:
            if info != lib.GxB_EXHAUSTED:  # pragma: no cover (safety)
                raise _error_code_lookup[info]("Vector iterator failed")
        finally:
            lib.GxB_Iterator_free(it_ptr)

    def export(self, format=None, *, sort=False, give_ownership=False, raw=False, **opts):
        """GxB_Vextor_export_xxx.

        Parameters
        ----------
        format : str or None, default None
            If ``format`` is not specified, this method exports in the currently stored format.
            To control the export format, set ``format`` to one of:
                - "sparse"
                - "bitmap"
                - "full"
        sort : bool, default False
            Whether to sort indices if the format is "sparse"
        give_ownership : bool, default False
            Perform a zero-copy data transfer to Python if possible.  This gives ownership of
            the underlying memory buffers to NumPy.
            ** If True, this nullifies the current object, which should no longer be used! **
        raw : bool, default False
            If True, always return array the same size as returned by SuiteSparse.
            If False, arrays may be trimmed to be the expected size.
            It may make sense to choose ``raw=True`` if one wants to use the data to perform
            a zero-copy import back to SuiteSparse.

        Returns
        -------
        dict; keys depend on ``format`` and ``raw`` arguments (see below).

        See Also
        --------
        Vector.to_coo
        Vector.ss.import_any

        Return values
            - Note: for ``raw=True``, arrays may be larger than specified.
            - "sparse" format
                - indices : ndarray(dtype=uint64, size=nvals)
                - values : ndarray(size=nvals)
                - sorted_index : bool
                    - True if the values in "indices" are sorted
                - size : int
                - nvals : int, only present if raw == True
            - "bitmap" format
                - bitmap : ndarray(dtype=bool, size=size)
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
            format=format,
            sort=sort,
            give_ownership=give_ownership,
            raw=raw,
            method="export",
            opts=opts,
        )

    def unpack(self, format=None, *, sort=False, raw=False, **opts):
        """GxB_Vector_unpack_xxx.

        ``unpack`` is like ``export``, except that the Vector remains valid but empty.
        ``pack_*`` methods are the opposite of ``unpack``.

        See ``Vector.ss.export`` documentation for more details.
        """
        return self._export(
            format=format, sort=sort, give_ownership=True, raw=raw, method="unpack", opts=opts
        )

    def _export(self, format=None, *, sort=False, give_ownership=False, raw=False, method, opts):
        if give_ownership:
            parent = self._parent
        else:
            parent = self._parent.dup(name=f"v_{method}")
        dtype = parent.dtype.np_type
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
            jumbled = NULL
        else:
            jumbled = ffi_new("bool*")
        is_iso = ffi_new("bool*")
        desc = get_descriptor(**opts)
        desc_obj = NULL if desc is None else desc._carg
        if format == "sparse":
            vi = ffi_new("GrB_Index**")
            vi_size = ffi_new("GrB_Index*")
            nvals = ffi_new("GrB_Index*")
            check_status(
                getattr(lib, f"GxB_Vector_{method}_CSC")(
                    vhandle,
                    *args,
                    vi,
                    vx,
                    vi_size,
                    vx_size,
                    is_iso,
                    nvals,
                    jumbled,
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            nvals = nvals[0]
            indices = claim_buffer(ffi, vi[0], vi_size[0] // index_dtype.itemsize, index_dtype)
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if indices.size > nvals:
                    indices = indices[:nvals]
                if is_iso:
                    if values.size > 1:  # pragma: no cover (suitesparse)
                        values = values[:1]
                elif values.size > nvals:
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
                getattr(lib, f"GxB_Vector_{method}_Bitmap")(
                    vhandle,
                    *args,
                    vb,
                    vx,
                    vb_size,
                    vx_size,
                    is_iso,
                    nvals,
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            bool_dtype = np.dtype(np.bool_)
            bitmap = claim_buffer(ffi, vb[0], vb_size[0] // bool_dtype.itemsize, bool_dtype)
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if bitmap.size > size:  # pragma: no branch (suitesparse)
                    bitmap = bitmap[:size]
                if is_iso:
                    if values.size > 1:  # pragma: no cover (suitesparse)
                        values = values[:1]
                elif values.size > size:  # pragma: no cover (suitesparse)
                    values = values[:size]
            rv = {
                "bitmap": bitmap,
                "nvals": nvals[0],
            }
            if raw:
                rv["size"] = size
        elif format == "full":
            check_status(
                getattr(lib, f"GxB_Vector_{method}_Full")(
                    vhandle,
                    *args,
                    vx,
                    vx_size,
                    is_iso,
                    desc_obj,
                ),
                parent,
            )
            is_iso = is_iso[0]
            values = claim_buffer(ffi, vx[0], vx_size[0] // dtype.itemsize, dtype)
            if not raw:
                if is_iso:
                    if values.size > 1:
                        values = values[:1]
                elif values.size > size:  # pragma: no branch (suitesparse)
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
            parent.gb_obj[0] = NULL
        if parent.dtype._is_udt:
            rv["dtype"] = parent.dtype
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
        secure_import=False,
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
        **opts,
    ):
        """GxB_Vector_import_xxx.

        Dispatch to appropriate import method inferred from inputs.
        See the other import functions and ``Vector.ss.export`` for details.

        Returns
        -------
        Vector

        See Also
        --------
        Vector.from_coo
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
            secure_import=secure_import,
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
        # Sparse
        indices=None,
        sorted_index=False,
        # Bitmap
        bitmap=None,
        # Bitmap/Sparse
        nvals=None,  # optional
        # Unused for pack, ignored
        size=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_pack_xxx.

        ``pack_any`` is like ``import_any`` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack()``

        See ``Vector.ss.import_any`` documentation for more details.
        """
        return self._import_any(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
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
            opts=opts,
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
        secure_import=False,
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
        opts,
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
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "bitmap":
            return getattr(obj, f"{method}_bitmap")(
                nvals=nvals,
                bitmap=bitmap,
                values=values,
                size=size,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
        if format == "full":
            return getattr(obj, f"{method}_full")(
                values=values,
                size=size,
                is_iso=is_iso,
                take_ownership=take_ownership,
                secure_import=secure_import,
                dtype=dtype,
                name=name,
                **opts,
            )
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_import_CSC.

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
            If true, then ``values`` should be a length 1 array.
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
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
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
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        size=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_pack_CSC.

        ``pack_sparse`` is like ``import_sparse`` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("sparse")``

        See ``Vector.ss.import_sparse`` documentation for more details.
        """
        return self._import_sparse(
            indices=indices,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            sorted_index=sorted_index,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            vector=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
        opts,
    ):
        if format is not None and format.lower() != "sparse":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'sparse'.")
        copy = not take_ownership
        indices = ints_to_numpy_buffer(indices, np.uint64, copy=copy, ownable=True, name="indices")
        if method == "pack":
            dtype = vector.dtype
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
        if indices is values:
            values = np.copy(values)
        vi = ffi_new("GrB_Index**", ffi.from_buffer("GrB_Index*", indices))
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if nvals is None:
            if is_iso:
                nvals = indices.size
            elif dtype.np_type.subdtype is not None:
                nvals = values.shape[0]
            else:
                nvals = values.size
        if method == "import":
            vhandle = ffi_new("GrB_Vector*")
            args = (dtype._carg, size)
        else:
            vhandle = vector._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Vector_{method}_CSC")(
            vhandle,
            *args,
            vi,
            vx,
            indices.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            not sorted_index,
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector._from_obj(vhandle, dtype, size, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_import_Bitmap.

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
            dtype of the new Vector.
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_bitmap(
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
        size=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_pack_Bitmap.

        ``pack_bitmap`` is like ``import_bitmap`` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("bitmap")``

        See ``Vector.ss.import_bitmap`` documentation for more details.
        """
        return self._import_bitmap(
            bitmap=bitmap,
            values=values,
            nvals=nvals,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            vector=self._parent,
            opts=opts,
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
        opts,
    ):
        if format is not None and format.lower() != "bitmap":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'bitmap'.")
        copy = not take_ownership
        bitmap = ints_to_numpy_buffer(bitmap, np.bool_, copy=copy, ownable=True, name="bitmap")
        if method == "pack":
            dtype = vector.dtype
            size = vector._size
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
        if bitmap is values:
            values = np.copy(values)
        vhandle = ffi_new("GrB_Vector*")
        vb = ffi_new("int8_t**", ffi.from_buffer("int8_t*", bitmap))
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if size is None:
            if is_iso:
                size = bitmap.size
            elif dtype.np_type.subdtype is not None:
                size = values.shape[0]
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
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Vector_{method}_Bitmap")(
            vhandle,
            *args,
            vb,
            vx,
            bitmap.nbytes,
            values.nbytes,
            is_iso,
            nvals,
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector._from_obj(vhandle, dtype, size, name=name)
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
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_import_Full.

        Create a new Vector from values.

        Parameters
        ----------
        values : array-like
        size : int, optional
            The size of the new Vector.
            If not specified, it will be set to the size of values.
        is_iso : bool, default False
            Is the Vector iso-valued (meaning all the same value)?
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
            dtype of the new Vector.
            If not specified, this will be inferred from ``values``.
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
            secure_import=secure_import,
            dtype=dtype,
            format=format,
            name=name,
            method="import",
            opts=opts,
        )

    def pack_full(
        self,
        values,
        *,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        format=None,
        # Unused for pack, ignored
        size=None,
        dtype=None,
        name=None,
        **opts,
    ):
        """GxB_Vector_pack_Full.

        ``pack_full`` is like ``import_full`` except it "packs" data into an
        existing Vector.  This is the opposite of ``unpack("full")``

        See ``Vector.ss.import_full`` documentation for more details.
        """
        return self._import_full(
            values=values,
            is_iso=is_iso,
            take_ownership=take_ownership,
            secure_import=secure_import,
            format=format,
            method="pack",
            vector=self._parent,
            opts=opts,
        )

    @classmethod
    def _import_full(
        cls,
        *,
        values,
        size=None,
        is_iso=False,
        take_ownership=False,
        secure_import=False,
        dtype=None,
        format=None,
        name=None,
        method,
        vector=None,
        opts,
    ):
        if format is not None and format.lower() != "full":
            raise ValueError(f"Invalid format: {format!r}.  Must be None or 'full'.")
        copy = not take_ownership
        if method == "pack":
            dtype = vector.dtype
            size = vector._size
        values, dtype = values_to_numpy_buffer(
            values, dtype, copy=copy, ownable=True, subarray_after=1
        )
        vhandle = ffi_new("GrB_Vector*")
        vx = ffi_new("void**", ffi.from_buffer("void*", values))
        if size is None:
            if dtype.np_type.subdtype is not None:
                size = values.shape[0]
            else:
                size = values.size
        if method == "import":
            vhandle = ffi_new("GrB_Vector*")
            args = (dtype._carg, size)
        else:
            vhandle = vector._carg
            args = ()
        desc = get_descriptor(secure_import=secure_import, **opts)
        status = getattr(lib, f"GxB_Vector_{method}_Full")(
            vhandle,
            *args,
            vx,
            values.nbytes,
            is_iso,
            NULL if desc is None else desc._carg,
        )
        if method == "import":
            check_status_carg(
                status,
                "Vector",
                vhandle[0],
            )
            vector = gb.Vector._from_obj(vhandle, dtype, size, name=name)
        else:
            check_status(status, vector)
        unclaim_buffer(values)
        return vector

    @wrapdoc(head)
    def head(self, n=10, dtype=None, *, sort=False):
        return head(self._parent, n, dtype, sort=sort)

    def scan(self, op=monoid.plus, *, name=None, **opts):
        """Perform a prefix scan with the given monoid.

        For example, use ``monoid.plus`` (the default) to perform a cumulative sum,
        and ``monoid.times`` for cumulative product.  Works with any monoid.

        Returns
        -------
        Scalar

        """
        return prefix_scan(self._parent, op, name=name, within="scan", **opts)

    def reshape(self, nrows, ncols=None, order="rowwise", *, name=None, **opts):
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
            Aliases of "columnwise" also accepted: "col", "cols", "column", "columns", "F".
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
        return self._parent._as_matrix().ss.reshape(nrows, ncols, order, name=name, **opts)

    def selectk(self, how, k, *, name=None):
        """Select (up to) k elements.

        Parameters
        ----------
        how : str
            - "random": choose k elements with equal probability
               Chosen values may not be ordered randomly
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
        return self.import_sparse(
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
                return gb.Vector(UINT64, size=0, name=name)
            return gb.Vector(self._parent.dtype, size=0, name=name)
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
        return self.import_sparse(
            **newinfo,
            take_ownership=True,
            name=name,
        )

    def sort(self, op=binary.lt, *, values=True, permutation=True, **opts):
        """GxB_Vector_sort to sort values of the Vector.

        Sorting moves all the elements to the left just like ``compactify``.
        The returned vectors will be the same size as the input Vector.

        Parameters
        ----------
        op : :class:`~graphblas.core.operator.BinaryOp`, optional
            Binary operator with a bool return type used to sort the values.
            For example, ``binary.lt`` (the default) sorts the smallest elements first.
            Ties are broken according to indices (smaller first).
        values : bool, default=True
            Whether to return values; will return ``None`` for values if ``False``.
        permutation : bool, default=True
            Whether to compute the permutation Vector that has the original indices of the
            sorted values. Will return None if ``False``.
        nthreads : int, optional
            The maximum number of threads to use for this operation.
            None, 0 or negative nthreads means to use the default number of threads.

        Returns
        -------
        Vector : values
        Vector[dtype=UINT64] : permutation

        See Also
        --------
        Vector.ss.compactify

        """
        from ..vector import Vector

        parent = self._parent
        op = get_typed_op(op, parent.dtype, kind="binary")
        if op.opclass == "Monoid":
            op = op.binaryop
        else:
            parent._expect_op(op, "BinaryOp", within="sort", argname="op")
        if values:
            w = Vector(parent.dtype, parent._size, name="values")
        elif not permutation:
            return None, None
        else:
            w = None
        if permutation:
            p = Vector(UINT64, parent._size, name="permutation")
        else:
            p = None
        desc = get_descriptor(**opts)
        check_status(
            lib.GxB_Vector_sort(
                NULL if w is None else w._carg,
                NULL if p is None else p._carg,
                op._carg,
                parent._carg,
                NULL if desc is None else desc._carg,
            ),
            parent,
        )
        return w, p

    def serialize(self, compression="default", level=None, **opts):
        """Serialize a Vector to bytes (as numpy array) using SuiteSparse GxB_Vector_serialize.

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
            The maximum number of threads to use when serializing the Vector.
            None, 0 or negative nthreads means to use the default number of threads.

        For best performance, this function returns a numpy array with uint8 dtype.
        Use ``Vector.ss.deserialize(blob)`` to create a Vector from the result of serialization·

        This method is intended to support all serialization options from SuiteSparse:GraphBLAS.

        *Warning*: Behavior of serializing UDTs is experimental and may change in a future release.

        """
        desc = get_descriptor(compression=compression, compression_level=level, **opts)
        blob_handle = ffi_new("void**")
        blob_size_handle = ffi_new("GrB_Index*")
        parent = self._parent
        if parent.dtype._is_udt and hasattr(lib, "GrB_Type_get_String"):
            # Get the name from the dtype and set it to the name of the vector so we can
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
            status = lib.GrB_Vector_set_String(parent._carg, dtype_char, lib.GrB_NAME)
            check_status_carg(status, "Vector", parent._carg)

        check_status(
            lib.GxB_Vector_serialize(
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
        """Deserialize a Vector from bytes, buffer, or numpy array using GxB_Vector_deserialize.

        The data should have been previously serialized with a compatible version of
        SuiteSparse:GraphBLAS.  For example, from the result of ``data = vector.ss.serialize()``.

        Examples
        --------
        >>> data = vector.serialize()
        >>> new_vector = Vector.ss.deserialize(data)
        >>> new_vector.isequal(vector)
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
                raise _error_code_lookup[info]("Vector deserialize failed to get the dtype name")
            dtype_name = b"".join(itertools.takewhile(b"\x00".__ne__, cname)).decode()
            if not dtype_name and hasattr(lib, "GxB_Serialized_get_String"):
                # Handle UDTs. First get the size of name
                dtype_size = ffi_new("size_t*")
                info = lib.GxB_Serialized_get_SIZE(data_obj, dtype_size, lib.GrB_NAME, data.nbytes)
                if info != lib.GrB_SUCCESS:
                    raise _error_code_lookup[info](
                        "Vector deserialize failed to get the size of name"
                    )
                # Then get the name
                dtype_char = ffi_new(f"char[{dtype_size[0]}]")
                info = lib.GxB_Serialized_get_String(
                    data_obj, dtype_char, lib.GrB_NAME, data.nbytes
                )
                if info != lib.GrB_SUCCESS:
                    raise _error_code_lookup[info]("Vector deserialize failed to get the name")
                dtype_name = ffi.string(dtype_char).decode()
            dtype = _string_to_dtype(dtype_name)
        else:
            dtype = lookup_dtype(dtype)
        desc = get_descriptor(**opts)
        gb_obj = ffi_new("GrB_Vector*")
        check_status_carg(
            lib.GxB_Vector_deserialize(
                gb_obj, dtype._carg, data_obj, data.nbytes, NULL if desc is None else desc._carg
            ),
            "Vector",
            gb_obj[0],
        )
        rv = gb.Vector._from_obj(gb_obj, dtype, -1, name=name)
        rv._size = rv.size
        return rv


@njit
def random_choice(n, k):  # pragma: no cover (numba)
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

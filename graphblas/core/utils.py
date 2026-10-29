from operator import index

import numpy as np

from ..dtypes import _INDEX, lookup_dtype
from . import ffi, lib
from .dtypes import _nats_in_leaf_units, _view_if_same_layout

try:
    from suitesparse_graphblas.utils import can_unclaim_buffer, empty
except ImportError:  # pragma: no cover (import)
    # MAINT 2026-10-18: suitesparse-graphblas < 10.5.1.1 has neither
    can_unclaim_buffer = empty = None

_NP2 = np.__version__.startswith("2.")


def libget(name):
    """Helper to get items from GraphBLAS which might be GrB or GxB."""
    try:
        return getattr(lib, name)
    except AttributeError:
        if name[-4:] not in {"FC32", "FC64", "rror"}:
            raise
        ext_name = f"GxB_{name[4:]}"
        try:
            return getattr(lib, ext_name)
        except AttributeError:  # pragma: no cover (sanity)
            pass
        raise  # pragma: no cover (sanity)


def wrapdoc(func_with_doc):
    """Decorator to copy ``__doc__`` from a function onto the wrapped function."""

    def inner(func_wo_doc):
        func_wo_doc.__doc__ = func_with_doc.__doc__
        return func_wo_doc

    return inner


def _module_attr_error(module_name, key, names):
    """Build the AttributeError raised by an operator namespace's ``__getattr__``.

    ``names`` should be the module's ``__dir__()`` so lazily-registered operators
    are offered as "did you mean" suggestions without forcing them to build.
    """
    import difflib

    msg = f"module {module_name!r} has no attribute {key!r}"
    candidates = [name for name in names if not name.startswith("_")]
    matches = difflib.get_close_matches(key, candidates, n=3)
    if matches:
        hint = " or ".join(repr(match) for match in matches)
        msg = f"{msg}. Did you mean {hint}?"
    return AttributeError(msg)


# Include most common types (even mistakes)
_output_types = {
    int: int,
    float: float,
    list: list,
    slice: slice,
    tuple: tuple,
    np.ndarray: np.ndarray,
    # Mistakes
    object: object,
    type: type,
}
_output_types.update((k, k) for k in set(np.sctypeDict.values()))


def _warn_from_caller(message, category):
    """Warn at the first caller outside python-graphblas, so default filters can show it.

    Python shows a DeprecationWarning by default only when it is attributed to
    ``__main__``, and a check that warns may sit at any depth below the public
    method. python-graphblas's own tests count as callers.
    """
    import sys
    import warnings
    from pathlib import Path

    package = Path(__file__).parent.parent
    tests = package / "tests"
    frame = sys._getframe(1)
    stacklevel = 2
    while frame is not None:  # pragma: no branch (safety)
        filename = Path(frame.f_code.co_filename)
        if not filename.is_relative_to(package) or filename.is_relative_to(tests):
            break
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)


def output_type(val):
    try:
        return _output_types[type(val)]
    except KeyError:
        return type(val)


def ints_to_numpy_buffer(array, dtype, *, name="array", copy=False, ownable=False, order="C"):
    if (
        isinstance(array, np.ndarray)
        and not np.issubdtype(array.dtype, np.integer)
        and not np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be integers, not {array.dtype.name}")
    # https://numpy.org/doc/stable/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors
    array = np.array(array, dtype, copy=copy or _NP2 and None, order=order)
    if ownable:
        array = _ownable(array, order)
    return array


def _ownable(array, order):
    """Return ``array``, or a copy of it if GraphBLAS can't take ownership of its data.

    GraphBLAS frees the data it owns with its own allocator, and ``owndata`` can't
    say which allocator made an array: one from a custom NumPy memory handler
    (NEP 49) owns its data too, and the process aborts when GraphBLAS frees it.
    """
    if can_unclaim_buffer is None:
        # Older suitesparse-graphblas can't tell, so assume NumPy's default allocator
        ownable = array.flags.owndata and array.flags.writeable
    else:
        ownable = can_unclaim_buffer(array)
    return array if ownable else _ownable_copy(array, order)


def _ownable_copy(array, order):
    """Copy ``array`` into memory GraphBLAS can take ownership of."""
    if empty is None:
        return array.copy(order)
    # A NumPy copy may be no better: a custom memory handler may be in effect, or
    # GraphBLAS may use libc's allocator (``memory_manager="c"``)
    buf = empty(array.shape, array.dtype, order)
    np.copyto(buf, array)
    return buf


def _distinct_buffers(*arrays):
    """Copy each array given more than once, since GraphBLAS frees every buffer it takes."""
    rv = []
    for array in arrays:
        if any(array is prev for prev in rv):
            order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
            array = _ownable_copy(array, order)
        rv.append(array)
    return rv


def _repeated_value(values, dtype):
    """Return the one value a stride-0 1-d array repeats, as a 0-d array, or None.

    ``np.broadcast_to(x, n)`` spells n copies of ``x`` without making them.
    Values of a UDT (from ``values`` or ``dtype``) give None.
    """
    if (
        type(values) is np.ndarray
        and values.ndim == 1
        and values.size > 0
        and values.strides == (0,)
        and values.dtype.kind in "biufc"
        and (dtype is None or not lookup_dtype(dtype)._is_udt)
    ):
        return values[0, ...]
    return None


def _get_subdtype(dtype):
    while dtype.subdtype is not None:
        dtype = dtype.subdtype[0]
    return dtype


def values_to_numpy_buffer(
    array, dtype=None, *, copy=False, ownable=False, order="C", subarray_after=None
):
    """Convert an array-like object to a numpy array and infer the dtype if necessary.

    Parameters
    ----------
    subarray_after : int, optional
        If dtype is not provided, infer "sub-array" dtype if the array has extra dimensions.

    Returns
    -------
    np.ndarray
    dtype

    """
    if dtype is not None:
        dtype = lookup_dtype(dtype)
        np_type = _get_subdtype(dtype.np_type)
        # https://numpy.org/doc/stable/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors
        array = np.array(
            _nats_in_leaf_units(_view_if_same_layout(array, np_type), np_type),
            np_type,
            copy=copy or _NP2 and None,
            order=order,
        )
    else:
        is_input_np = isinstance(array, np.ndarray)
        # https://numpy.org/doc/stable/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors
        array = np.array(array, copy=copy or _NP2 and None, order=order)
        if array.dtype.hasobject:
            raise ValueError("object dtype for values is not allowed")
        if not is_input_np and array.dtype == np.int32:  # pragma: no cover
            # fix for win64 numpy handling of ints
            array = array.astype(np.int64)
        dtype = lookup_dtype(array.dtype)
        # Registration flattens nested subarrays, so values declared with them
        # carry a dtype the DataType we just looked up no longer matches. The
        # bytes are identical, so view them and return a pair that agrees.
        array = _view_if_same_layout(array, dtype.np_type)
        if subarray_after is not None and array.ndim > subarray_after:
            dtype = lookup_dtype(np.dtype((dtype.np_type, array.shape[subarray_after:])))
    if ownable:
        array = _ownable(array, order)
    return array, dtype


def check_element_shape(values, dtype, ndim):
    """Raise unless ``values`` is an ``ndim``-d array of whole ``dtype`` elements.

    ``values_to_numpy_buffer`` converts to the scalar type of a sub-array dtype, so each
    element's shape must trail the ``ndim`` leading dimensions. GraphBLAS reads whole
    elements, so a wrong shape reads the wrong numbers or reads past the end of the array.
    """
    if values.shape[ndim:] != dtype.np_type.shape:
        raise ValueError(
            f"values must have shape {values.shape[:ndim] + dtype.np_type.shape} "
            f"for dtype {dtype}, not {values.shape}"
        )


def check_values_size(values, dtype, count, is_iso, *, indptr=None):
    """Raise unless ``values`` holds ``count`` elements of ``dtype``, or one if ``is_iso``.

    With ``indptr``, the count is ``indptr[count]``, as GraphBLAS reads it. GraphBLAS
    imports before 10.5.2 compare the byte size of the values with the number of entries,
    not with the bytes those entries need, so a short array gets through and is read past
    its end. 10.5.2 raises InvalidValue with no message instead, so this check stays.
    """
    if is_iso:
        count = 1
    elif indptr is not None:
        # GraphBLAS rejects an indptr this short itself
        count = int(indptr[count]) if count < indptr.size else 0
    if (have := values.nbytes // dtype.np_type.itemsize) < count:
        raise ValueError(f"values must have {count} or more elements of dtype {dtype}, not {have}")


def normalize_values(self, values, dtype, shape=None, is_iso=False):
    """Expand and/or update dtype of values array."""
    if dtype is not None:
        dtype = lookup_dtype(dtype)
        if dtype != self.dtype:
            values = values.astype(dtype.np_type)  # copies
    else:
        dtype = self.dtype
    if is_iso:
        # The iso value of an array UDT is exported as shape (1, *subarray shape)
        values = np.broadcast_to(values, shape + values.shape[1:])
    return values


def get_shape(nrows, ncols, dtype=None, **arrays):
    if nrows is None or ncols is None:
        # Get nrows and ncols from the first 2d array
        is_subarray = dtype.np_type.subdtype is not None
        for name, arr in arrays.items():
            if name == "values" and is_subarray:
                # We could be smarter and determine the shape of the dtype sub-arrays
                if arr.ndim >= 3:
                    break
                # BRANCH NOT COVERED
            elif arr.ndim == 2:
                break
        else:
            raise ValueError(
                "Either nrows and ncols must be provided, or one of the following arrays "
                f'must be 2d (from which to get nrows and ncols): {", ".join(arrays)}'
            )
        if nrows is None:
            nrows = arr.shape[0]
        if ncols is None:
            ncols = arr.shape[1]
    return nrows, ncols


def get_order(order):
    val = order.lower()
    if val in {"c", "row", "rows", "rowwise"}:
        return "rowwise"
    if val in {"f", "col", "cols", "column", "columns", "colwise", "columnwise"}:
        return "columnwise"
    raise ValueError(
        f"Bad value for order: {order!r}.  "
        'Expected "rowwise", "columnwise", "rows", "columns", "C", or "F"'
    )


def maybe_integral(val):
    """Ensure ``val`` is an integer or return None if it's not."""
    try:
        return index(val)
    except TypeError:
        pass
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return None


def normalize_chunks(chunks, shape):
    """Normalize chunks argument for use by ``Matrix.ss.split``.

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
    elif (chunk := maybe_integral(chunks)) is not None:
        chunks = (chunk,) * len(shape)
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
    for size, chunk in zip(shape, chunks, strict=True):
        if chunk is None:
            cur_chunks = [size]
        elif (c := maybe_integral(chunk)) is not None:
            if c < 0:
                raise ValueError(f"Chunksize must be greater than 0; got: {c}")
            div, mod = divmod(size, c)
            cur_chunks = [c] * div
            if mod:
                cur_chunks.append(mod)
        elif isinstance(chunk, (list, tuple)):
            cur_chunks = []
            none_index = None
            for c in chunk:
                if (val := maybe_integral(c)) is not None:
                    if val < 0:
                        raise ValueError(f"Chunksize must be greater than 0; got: {val}")
                    c = val
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


def ensure_type(x, types):
    """Try to ensure ``x`` is one of the given types, computing if necessary.

    ``types`` must be a type or a tuple of types as used in ``isinstance``.

    For example, if ``types`` is a Vector, then a Vector input will be returned,
    and a ``VectorExpression`` input will be computed and returned as a Vector.

    TypeError will be raised if the input is not or can't be converted to types.

    This function ignores ``graphblas.config["autocompute"]``; it always computes
    if the return type will match ``types``.
    """
    if isinstance(x, types):
        return x
    if isinstance(types, tuple):
        if output_type(x) in types:
            return x.new()
    elif output_type(x) is types:
        return x.new()
    raise TypeError(f"{type(x).__name__!r} object is not of type {types}")


class class_property:
    __slots__ = "classval", "member_property", "exceptional"

    def __init__(self, member_property, classval, exceptional=False):
        self.classval = classval
        self.member_property = member_property
        self.exceptional = exceptional

    def __get__(self, instance, owner=None):
        if self.exceptional:
            raise AttributeError(self.classval)
        if instance is None:
            return self.classval
        return self.member_property.__get__(instance, owner)

    @property
    def __set__(self):  # pylint: disable=unexpected-special-method-signature
        return self.member_property.__set__


# A similar object may eventually make it to the GraphBLAS spec.
# Hide this from the user for now.
class _CArray:
    __slots__ = "array", "_carg", "dtype", "_name"

    def __init__(self, array=None, dtype=_INDEX, *, size=None, name=None):
        if size is not None:
            self.array = np.empty(size, dtype=dtype.np_type)
        else:
            # https://numpy.org/doc/stable/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors
            self.array = np.array(
                array, dtype=_get_subdtype(dtype.np_type), copy=_NP2 and None, order="C"
            )
        c_type = dtype.c_type if dtype._is_udt else f"{dtype.c_type}*"
        self._carg = ffi.cast(c_type, ffi.from_buffer(self.array))
        self.dtype = dtype
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        if len(self.array) < 20:
            values = ", ".join(map(str, self.array))
        else:
            values = (
                f"{', '.join(map(str, self.array[:5]))}, "
                "..., "
                f"{', '.join(map(str, self.array[-5:]))}"
            )
        return f"({self.dtype.c_type}[]){{{values}}}"


class _Pointer:
    __slots__ = "val", "_exc_arg"

    def __init__(self, val, exc_arg=None):
        self.val = val
        if exc_arg is not None:
            # check_status asks this object for the error text instead of ``val``, since a
            # C-scalar ``val`` has no GraphBLAS object to ask
            self._exc_arg = exc_arg

    @property
    def _carg(self):
        return self.val.gb_obj

    @property
    def name(self):
        name = self.val.name
        if not name:
            c = type(self.val).__name__[0]
            name = f"{'M' if c == 'M' else c.lower()}_temp"
        return f"&{name}"

    @property
    def _expr_name(self):
        return getattr(self.val, "_expr_name", self.name)

    @property
    def _expr_name_html(self):
        return getattr(self.val, "_expr_name_html", self.name)


class _MatrixArray:
    __slots__ = "_carg", "_exc_arg", "name"

    def __init__(self, matrices, exc_arg=None, *, name):
        from .base import record_raw

        self._carg = matrices
        self._exc_arg = exc_arg
        self.name = name
        record_raw(f"GrB_Matrix {name}[{len(matrices)}];")


def _autogenerate_code(
    filepath,
    text,
    specializer=None,
    begin="# Begin auto-generated code",
    end="# End auto-generated code",
    callblack=True,
):
    """Super low-tech auto-code generation used by automethods.py and infixmethods.py."""
    with filepath.open() as f:  # pragma: no branch (flaky)
        orig_text = f.read()
    if specializer:
        begin = f"{begin}: {specializer}"
        end = f"{end}: {specializer}"
    begin += "\n"
    end += "\n"

    boundaries = []
    stop = 0
    while True:
        try:
            start = orig_text.index(begin, stop)
            stop = orig_text.index(end, start)
        except ValueError:
            break
        boundaries.append((start, stop))
    new_text = orig_text
    for start, stop in reversed(boundaries):
        new_text = f"{new_text[:start]}{begin}{text}{new_text[stop:]}"
    with filepath.open("w") as f:  # pragma: no branch (flaky)
        f.write(new_text)
    import subprocess

    if callblack:
        try:
            subprocess.check_call(["black", filepath])
        except FileNotFoundError:  # pragma: no cover (safety)
            pass  # It's okay if `black` isn't installed; pre-commit hooks will do linting

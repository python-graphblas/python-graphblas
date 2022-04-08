import numpy as np

from . import ffi, lib, mask
from .dtypes import _INDEX, lookup_dtype


def libget(name):
    """Helper to get items from GraphBLAS which might be GrB or GxB"""
    try:
        return getattr(lib, name)
    except AttributeError:
        ext_name = f"GxB_{name[4:]}"
        try:
            return getattr(lib, ext_name)
        except AttributeError:
            pass
        raise


def wrapdoc(func_with_doc):
    """Decorator to copy `__doc__` from a function onto the wrapped function"""

    def inner(func_wo_doc):
        func_wo_doc.__doc__ = func_with_doc.__doc__
        return func_wo_doc

    return inner


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
    mask.Mask: mask.Mask,
    mask.StructuralMask: mask.StructuralMask,
    mask.ValueMask: mask.ValueMask,
    mask.ComplementedStructuralMask: mask.ComplementedStructuralMask,
    mask.ComplementedValueMask: mask.ComplementedValueMask,
}
_output_types.update((k, k) for k in np.cast)


def output_type(val):
    try:
        return _output_types[type(val)]
    except KeyError:
        return type(val)


def ints_to_numpy_buffer(array, dtype, *, name="array", copy=False, ownable=False, order="C"):
    if (
        isinstance(array, np.ndarray)
        and not np.issubdtype(array.dtype, np.integer)
        and not np.issubdtype(array.dtype, np.bool8)
    ):
        raise ValueError(f"{name} must be integers, not {array.dtype.name}")
    array = np.array(array, dtype, copy=copy, order=order)
    if ownable and (not array.flags.owndata or not array.flags.writeable):
        array = array.copy(order)
    return array


def values_to_numpy_buffer(array, dtype=None, *, copy=False, ownable=False, order="C"):
    if dtype is not None:
        dtype = lookup_dtype(dtype)
        array = np.array(array, dtype.np_type, copy=copy, order=order)
    else:
        is_input_np = isinstance(array, np.ndarray)
        array = np.array(array, copy=copy, order=order)
        if array.dtype == object:
            raise ValueError("object dtype for values is not allowed")
        if not is_input_np and array.dtype == np.int32:  # pragma: no cover
            # fix for win64 numpy handling of ints
            array = array.astype(np.int64)
        dtype = lookup_dtype(array.dtype)
    if ownable and (not array.flags.owndata or not array.flags.writeable):
        array = array.copy(order)
    return array, dtype


def get_shape(nrows, ncols, **arrays):
    if nrows is None or ncols is None:
        # Get nrows and ncols from the first 2d array
        arr = next((array for array in arrays.values() if array.ndim == 2), None)
        if arr is None:
            raise ValueError(
                "Either nrows and ncols must be provided, or one of the following arrays"
                f'must be 2d (from which to get nrows and ncols): {", ".join(arrays)}'
            )
        if nrows is None:
            nrows = arr.shape[0]
        if ncols is None:
            ncols = arr.shape[1]
    return nrows, ncols


class class_property:
    __slots__ = "classval", "member_property"

    def __init__(self, member_property, classval):
        self.classval = classval
        self.member_property = member_property

    def __get__(self, obj, type=None):
        if obj is None:
            return self.classval
        return self.member_property.__get__(obj, type)

    @property
    def __set__(self):
        return self.member_property.__set__


# A similar object may eventually make it to the GraphBLAS spec.
# Hide this from the user for now.
class _CArray:
    __slots__ = "array", "_carg", "dtype", "_name"

    def __init__(self, array=None, dtype=_INDEX, *, size=None, name=None):
        if size is not None:
            self.array = np.empty(size, dtype=dtype.np_type)
        else:
            self.array = np.array(array, dtype=dtype.np_type, copy=False, order="C")
        self._carg = ffi.cast(f"{dtype.c_type}*", ffi.from_buffer(self.array))
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
    __slots__ = "val"

    def __init__(self, val):
        self.val = val

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


def _autogenerate_code(
    filename,
    text,
    specializer=None,
    begin="# Begin auto-generated code",
    end="# End auto-generated code",
):
    """Super low-tech auto-code generation used by _automethods.py and _infixmethods.py"""
    with open(filename) as f:
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
    with open(filename, "w") as f:
        f.write(new_text)
    import subprocess

    subprocess.check_call(["black", filename])

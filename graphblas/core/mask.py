from .. import backend, monoid
from ..binary import land, lor, pair
from ..dtypes import BOOL
from ..select import valuene
from ..unary import one
from . import utils


class Mask:
    __slots__ = "parent", "__weakref__"
    complement = False
    structure = False
    value = False

    def __init__(self, mask):
        self.parent = mask

    def __eq__(self, other):
        raise TypeError(f"__eq__ not defined for objects of type {type(self)}.")

    def __bool__(self):
        raise TypeError(f"__bool__ not defined for objects of type {type(self)}.")

    def __repr__(self):
        return self.parent.__repr__(mask=self)

    def _repr_html_(self):
        return self.parent._repr_html_(mask=self)

    @property
    def _carg(self):
        return self.parent.gb_obj[0]

    def new(self, dtype=None, *, complement=False, mask=None, name=None, **opts):
        """Return a new object with True values determined by the mask(s).

        By default, the result is True wherever the mask(s) would have been applied,
        and empty otherwise.  If `complement` is True, then these are switched:
        the result is empty where the mask(s) would have been applied, and True otherwise.

        In other words, these are equivalent if complement is False (and mask keyword is None):

        >>> C(self) << expr
        >>> C(result.S) << expr  # equivalent when complement is False

        And these are equivalent if complement is True (and mask keyword is None):

        >>> C(self) << expr
        >>> C(~result.S) << expr  # equivalent when complement is True

        This can also efficiently merge two masks by using the `mask=` argument.
        This is equivalent to the following (but uses more efficient recipes):

        >>> val = Matrix(...)
        >>> val(self) << True
        >>> val(mask, replace=True) << val

        If `complement=` argument is True, then the *complement* will be returned.
        This is equivalent to the following (but uses more efficient recipes):

        >>> val = Matrix(...)
        >>> val(~self) << True
        >>> val(~mask) << True

        """
        if dtype is None:
            dtype = BOOL
        if mask is None:
            val = type(self.parent)(dtype, *self.parent.shape, name=name)
            if complement:
                val(~self, **opts) << True
            else:
                val(self, **opts) << True
            return val

        from .base import _check_mask

        mask = _check_mask(mask)
        d = _COMPLEMENT_MASKS if complement else _COMBINE_MASKS
        func = d[type(self), type(mask)]
        return func(self, mask, dtype, name, opts)

    def __and__(self, other, **opts):
        """Return the intersection of two masks as a new mask.

        `new_mask = mask1 & mask2` is equivalent to the following:

        >>> val = Matrix(bool, nrows, ncols)
        >>> val(mask1) << True
        >>> val(mask2, replace=True) << val
        >>> new_mask = val.S

        This uses faster recipes than the above for all combinations of input mask types,
        and aims to be memory efficient when operating on complemented masks.
        """
        from .base import _check_mask

        other = _check_mask(other)
        complement = self.complement or other.complement
        d = _COMPLEMENT_MASKS if complement else _COMBINE_MASKS
        func = d[type(self), type(other)]
        val = func(self, other, bool, None, opts)
        if complement:
            return ComplementedStructuralMask(val)
        return StructuralMask(val)

    __rand__ = __and__

    def __or__(self, other, **opts):
        """Return the union of two masks as a new mask.

        `new_mask = mask1 | mask2` is equivalent to the following:

        >>> val = Matrix(bool, nrows, ncols)
        >>> val(mask1) << True
        >>> val(mask2) << True
        >>> new_mask = val.S

        This uses faster recipes than the above for all combinations of input mask types,
        and aims to be memory efficient when operating on complemented masks.
        """
        from .base import _check_mask

        other = _check_mask(other)
        func = _MASK_OR[type(self), type(other)]
        return func(self, other, opts)

    __ror__ = __or__


class StructuralMask(Mask):
    __slots__ = ()
    complement = False
    structure = True
    value = False

    def __invert__(self):
        return ComplementedStructuralMask(self.parent)

    @property
    def name(self):
        return f"{self.parent.name}.S"

    @property
    def _name_html(self):
        return f"{self.parent._name_html}.S"


class ValueMask(Mask):
    __slots__ = ()
    complement = False
    structure = False
    value = True

    def __invert__(self):
        return ComplementedValueMask(self.parent)

    @property
    def name(self):
        return f"{self.parent.name}.V"

    @property
    def _name_html(self):
        return f"{self.parent._name_html}.V"


class ComplementedStructuralMask(Mask):
    __slots__ = ()
    complement = True
    structure = True
    value = False

    def __invert__(self):
        return StructuralMask(self.parent)

    @property
    def name(self):
        return f"~{self.parent.name}.S"

    @property
    def _name_html(self):
        return f"~{self.parent._name_html}.S"


class ComplementedValueMask(Mask):
    __slots__ = ()
    complement = True
    structure = False
    value = True

    def __invert__(self):
        return ValueMask(self.parent)

    @property
    def name(self):
        return f"~{self.parent.name}.V"

    @property
    def _name_html(self):
        return f"~{self.parent._name_html}.V"


# Recipes to combine two masks.
# Legend:
#    A: any
#    S: structural
#    V: value
#    CS: complemented structural
#    CV: complemented value
def _combine_S_S(m1, m2, dtype, name, opts):
    """S-S."""
    return pair(m1.parent & m2.parent).new(dtype, name=name, **opts)


def _combine_S_A(m1, m2, dtype, name, opts):
    """S-S, S-V, S-CS, S-CV."""
    return one(m1.parent).new(dtype, mask=m2, name=name, **opts)


def _combine_A_S(m1, m2, dtype, name, opts):
    """S-S, V-S, CS-S, CV-S."""
    return one(m2.parent).new(dtype, mask=m1, name=name, **opts)


def _combine_V_A(m1, m2, dtype, name, opts):
    """V-S, V-V, V-CS, V-CV."""
    if isinstance(m2, ValueMask) and m1.parent._nvals > m2.parent._nvals:
        m1, m2 = m2, m1
    val = valuene(m1.parent).new(dtype, mask=m2, name=name, **opts)
    if dtype != BOOL or backend != "suitesparse" or not val.ss.is_iso:
        val(val.S, **opts) << True
    return val


def _combine_A_V(m1, m2, dtype, name, opts):
    """S-V, V-V, CS-V, CV-V."""
    val = valuene(m2.parent).new(dtype, mask=m1, name=name, **opts)
    if dtype != BOOL or backend != "suitesparse" or not val.ss.is_iso:
        val(val.S, **opts) << True
    return val


def _combine_CS_CS(m1, m2, dtype, name, opts):
    """CS-CS."""
    val = pair(m1.parent | m2.parent).new(dtype, name=name, **opts)
    val(~val.S, replace=True, **opts) << True
    return val


def _combine_CS_CV(m1, m2, dtype, name, opts):
    """CS-CV."""
    val = pair(one(m1.parent).new(**opts) | m2.parent).new(dtype, name=name, **opts)
    val(~val.V, replace=True, **opts) << True
    return val


def _combine_CV_CS(m1, m2, dtype, name, opts):
    """CV-CS."""
    val = pair(m1.parent | one(m2.parent).new(**opts)).new(dtype, name=name, **opts)
    val(~val.V, replace=True, **opts) << True
    return val


def _combine_CV_CV(m1, m2, dtype, name, opts):
    """CV-CV."""
    val = lor(m1.parent | m2.parent).new(dtype, name=name, **opts)
    val(~val.V, replace=True, **opts) << True
    return val


_COMBINE_MASKS = {
    # S-S
    (StructuralMask, StructuralMask): _combine_S_S,
    # S-V, V-S
    (StructuralMask, ValueMask): _combine_S_A,
    (ValueMask, StructuralMask): _combine_A_S,
    # S-CS, CS-S
    (StructuralMask, ComplementedStructuralMask): _combine_S_A,
    (ComplementedStructuralMask, StructuralMask): _combine_A_S,
    # S-CV, CV-S
    (StructuralMask, ComplementedValueMask): _combine_S_A,
    (ComplementedValueMask, StructuralMask): _combine_A_S,
    # V-V
    (ValueMask, ValueMask): _combine_V_A,
    # V-CS, CS-V
    (ValueMask, ComplementedStructuralMask): _combine_V_A,
    (ComplementedStructuralMask, ValueMask): _combine_A_V,
    # V-CV, CV-V
    (ValueMask, ComplementedValueMask): _combine_V_A,
    (ComplementedValueMask, ValueMask): _combine_A_V,
    # CS-CS
    (ComplementedStructuralMask, ComplementedStructuralMask): _combine_CS_CS,
    # CS-CV, CV-CS
    (ComplementedStructuralMask, ComplementedValueMask): _combine_CS_CV,
    (ComplementedValueMask, ComplementedStructuralMask): _combine_CV_CS,
    # CV-CV
    (ComplementedValueMask, ComplementedValueMask): _combine_CV_CV,
}


# Recipes to return the *complement* of combining two masks
def _complement_S_S(m1, m2, dtype, name, opts):
    """S-S."""
    val = pair(m1.parent & m2.parent).new(dtype, name=name, **opts)
    val(~val.S, replace=True, **opts) << True
    return val


def _complement_S_A(m1, m2, dtype, name, opts):
    """S-S, S-V, S-CS, S-CV."""
    val = one(m1.parent).new(dtype, mask=m2, name=name, **opts)
    val(~val.S, replace=True, **opts) << True
    return val


def _complement_A_S(m1, m2, dtype, name, opts):
    """S-S, V-S, CS-S, CV-S."""
    val = one(m2.parent).new(dtype, mask=m1, name=name, **opts)
    val(~val.S, replace=True, **opts) << True
    return val


def _complement_V_V(m1, m2, dtype, name, opts):
    """V-V."""
    val = land(m1.parent & m2.parent).new(dtype, name=name, **opts)
    val(~val.V, replace=True, **opts) << True
    return val


def _complement_CS_CS(m1, m2, dtype, name, opts):
    """CS-CS."""
    return pair(one(m1.parent).new(**opts) | one(m2.parent).new(**opts)).new(
        dtype, name=name, **opts
    )


def _complement_CS_A(m1, m2, dtype, name, opts):
    """CS-S, CS-V, CS-CS, CS-CV."""
    val = one(m1.parent).new(dtype, name=name, **opts)
    val(~m2, **opts) << True
    return val


def _complement_A_CS(m1, m2, dtype, name, opts):
    """S-CS, V-CS, CS-CS, CV-CS."""
    val = one(m2.parent).new(dtype, name=name, **opts)
    val(~m1, **opts) << True
    return val


def _complement_CS_CV(m1, m2, dtype, name, opts):
    """CS-CV."""
    val = pair(one(m1.parent).new(**opts) | m2.parent).new(dtype, name=name, **opts)
    val(val.V, replace=True, **opts) << True
    return val


def _complement_CV_CS(m1, m2, dtype, name, opts):
    """CV-CS."""
    val = pair(m1.parent | one(m2.parent).new(**opts)).new(dtype, name=name, **opts)
    val(val.V, replace=True, **opts) << True
    return val


def _complement_CV_CV(m1, m2, dtype, name, opts):
    """CV-CV."""
    val = lor(m1.parent | m2.parent).new(dtype, name=name, **opts)
    val(val.V, replace=True, **opts) << True
    return val


def _complement_CV_A(m1, m2, dtype, name, opts):
    """CV-S, CV-V, CV-CS, CV-CV."""
    val = one(m1.parent).new(dtype, mask=~m1, name=name, **opts)
    val(~m2, **opts) << True
    return val


def _complement_A_CV(m1, m2, dtype, name, opts):
    """S-CV, V-CV, CS-CV, CV-CV."""
    val = one(m2.parent).new(dtype, mask=~m2, name=name, **opts)
    val(~m1, **opts) << True
    return val


_COMPLEMENT_MASKS = {
    # S-S
    (StructuralMask, StructuralMask): _complement_S_S,
    # S-V, V-S
    (StructuralMask, ValueMask): _complement_S_A,
    (ValueMask, StructuralMask): _complement_A_S,
    # S-CS, CS-S
    (StructuralMask, ComplementedStructuralMask): _complement_A_CS,  # or _complement_S_A
    (ComplementedStructuralMask, StructuralMask): _complement_CS_A,  # or _complement_A_S
    # S-CV, CV-S
    (StructuralMask, ComplementedValueMask): _complement_A_CV,  # or _complement_S_A
    (ComplementedValueMask, StructuralMask): _complement_CV_A,  # or _complement_A_S
    # V-V
    (ValueMask, ValueMask): _complement_V_V,
    # V-CS, CS-V
    (ValueMask, ComplementedStructuralMask): _complement_A_CS,
    (ComplementedStructuralMask, ValueMask): _complement_CS_A,
    # V-CV, CV-V
    (ValueMask, ComplementedValueMask): _complement_A_CV,
    (ComplementedValueMask, ValueMask): _complement_CV_A,
    # CS-CS
    (ComplementedStructuralMask, ComplementedStructuralMask): _complement_CS_CS,
    # CS-CV, CV-CS
    (ComplementedStructuralMask, ComplementedValueMask): _complement_CS_CV,
    (ComplementedValueMask, ComplementedStructuralMask): _complement_CV_CS,
    # CV-CV
    (ComplementedValueMask, ComplementedValueMask): _complement_CV_CV,
}


def _combine_S_S_mask_or(m1, m2, opts):
    """S-S."""
    val = monoid.any(one(m1.parent).new(bool, **opts) | one(m2.parent).new(bool, **opts)).new(
        **opts
    )
    return StructuralMask(val)


def _combine_S_SV_mask_or(m1, m2, opts):
    """S-V."""
    val = monoid.any(
        one(m1.parent).new(bool, **opts) | one(m2.parent).new(bool, mask=m2, **opts)
    ).new(**opts)
    return StructuralMask(val)


def _combine_SV_S_mask_or(m1, m2, opts):
    """V-S."""
    val = monoid.any(
        one(m1.parent).new(bool, mask=m1, **opts) | one(m2.parent).new(bool, **opts)
    ).new(**opts)
    return StructuralMask(val)


def _complement_A_CS_mask_or(m1, m2, opts):
    """~S-CS, ~V-CS, ~CV-CS."""
    val = one(m2.parent).new(bool, mask=~m1, **opts)
    return ComplementedStructuralMask(val)


def _complement_CS_A_mask_or(m1, m2, opts):
    """~CS-S, ~CS-V, ~CS-CV."""
    val = one(m1.parent).new(bool, mask=~m2, **opts)
    return ComplementedStructuralMask(val)


def _complement_A_CV_mask_or(m1, m2, opts):
    """~S-CV, ~V-CV."""
    val = valuene(m2.parent).new(bool, mask=~m1, **opts)
    return ComplementedStructuralMask(val)


def _complement_CV_A_mask_or(m1, m2, opts):
    """~CV-S, ~CV-V."""
    val = valuene(m1.parent).new(bool, mask=~m2, **opts)
    return ComplementedStructuralMask(val)


def _combine_V_V_mask_or(m1, m2, opts):
    """V-V."""
    val = monoid.any(valuene(m1.parent).new(**opts) | valuene(m2.parent).new(**opts)).new(
        bool, **opts
    )
    return StructuralMask(val)


def _complement_CS_CS_mask_or(m1, m2, opts):
    """~CS-CS."""
    val = pair(m1.parent & m2.parent).new(bool, **opts)
    return ComplementedStructuralMask(val)


def _complement_CV_CV_mask_or(m1, m2, opts):
    """~CV-CV."""
    val = valuene(land(m1.parent & m2.parent).new(bool, **opts)).new(**opts)
    return ComplementedStructuralMask(val)


_MASK_OR = {
    # S-S
    (StructuralMask, StructuralMask): _combine_S_S_mask_or,
    # S-V, V-S
    (StructuralMask, ValueMask): _combine_S_SV_mask_or,
    (ValueMask, StructuralMask): _combine_SV_S_mask_or,
    # S-CS, CS-S
    (StructuralMask, ComplementedStructuralMask): _complement_A_CS_mask_or,
    (ComplementedStructuralMask, StructuralMask): _complement_CS_A_mask_or,
    # S-CV, CV-S
    (StructuralMask, ComplementedValueMask): _complement_A_CV_mask_or,
    (ComplementedValueMask, StructuralMask): _complement_CV_A_mask_or,
    # V-V
    (ValueMask, ValueMask): _combine_V_V_mask_or,
    # V-CS, CS-V
    (ValueMask, ComplementedStructuralMask): _complement_A_CS_mask_or,
    (ComplementedStructuralMask, ValueMask): _complement_CS_A_mask_or,
    # V-CV, CV-V
    (ValueMask, ComplementedValueMask): _complement_A_CV_mask_or,
    (ComplementedValueMask, ValueMask): _complement_CV_A_mask_or,
    # CS-CS
    (ComplementedStructuralMask, ComplementedStructuralMask): _complement_CS_CS_mask_or,
    # CS-CV, CV-CS
    (ComplementedStructuralMask, ComplementedValueMask): _complement_CS_A_mask_or,
    (ComplementedValueMask, ComplementedStructuralMask): _complement_A_CS_mask_or,
    # CV-CV
    (ComplementedValueMask, ComplementedValueMask): _complement_CV_CV_mask_or,
}

utils._output_types[StructuralMask] = StructuralMask
utils._output_types[ValueMask] = ValueMask
utils._output_types[ComplementedStructuralMask] = ComplementedStructuralMask
utils._output_types[ComplementedValueMask] = ComplementedValueMask

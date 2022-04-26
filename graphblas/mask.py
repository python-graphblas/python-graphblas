import warnings


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

    @property
    def mask(self):
        warnings.warn(
            "`mask.mask` is deprecated; please use `mask.parent` instead.",
            DeprecationWarning,
        )
        return self.parent


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

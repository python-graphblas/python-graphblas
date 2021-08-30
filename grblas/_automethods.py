""" Define functions ot use as property methods on expressions.

These will automatically compute the value, so often avoids the need for `.new()`.

To automatically create the functions, run:

```python

common = {
    "_get_value",
    "_name_html",
    "_nvals",
    "gb_obj",
    "isclose",
    "isequal",
    "name",
    "nvals",
    # to_pygraphblas,
}
scalar = {
    "__bool__",
    "__complex__",
    "__eq__",
    "__float__",
    "__index__",
    "__int__",
    "__neg__",
    "is_empty",
    "value",
}
vector_matrix = {
    "S",
    "V",
    "__and__",
    "__contains__",
    "__getitem__",
    "__iter__",
    "__matmul__",
    "__or__",
    "_carg",
    "apply",
    "ewise_add",
    "ewise_mult",
    "to_values",
}
vector = {
    "inner",
    "outer",
    "reduce",
    "vxm",
}
matrix = {
    "T",
    "kronecker",
    "mxm",
    "mxv",
    "reduce_columns",
    "reduce_rows",
    "reduce_scalar",
}

for name in sorted(common | scalar | vector_matrix | vector | matrix):
    print(f'''def {name}(self):
    if self._value is None:
        self._value = self.new()
    return self._value.{name}\n\n''')

for name in sorted(common | scalar):
    print(f"    {name} = wrapdoc(Scalar.{name})(property(_automethods.{name}))")
print()
for name in sorted(common | vector_matrix | vector):
    print(f"    {name} = wrapdoc(Vector.{name})(property(_automethods.{name}))")
print()
for name in sorted(common | vector_matrix | matrix):
    print(f"    {name} = wrapdoc(Matrix.{name})(property(_automethods.{name}))")

```
"""


def S(self):
    if self._value is None:
        self._value = self.new()
    return self._value.S


def T(self):
    if self._value is None:
        self._value = self.new()
    return self._value.T


def V(self):
    if self._value is None:
        self._value = self.new()
    return self._value.V


def __and__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__and__


def __bool__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__bool__


def __complex__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__complex__


def __contains__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__contains__


def __eq__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__eq__


def __float__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__float__


def __getitem__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__getitem__


def __index__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__index__


def __int__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__int__


def __iter__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__iter__


def __matmul__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__matmul__


def __neg__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__neg__


def __or__(self):
    if self._value is None:
        self._value = self.new()
    return self._value.__or__


def _carg(self):
    if self._value is None:
        self._value = self.new()
    return self._value._carg


def _get_value(self):
    if self._value is None:
        self._value = self.new()
    return self._value


def _name_html(self):
    if self._value is None:
        self._value = self.new()
    return self._value._name_html


def _nvals(self):
    if self._value is None:
        self._value = self.new()
    return self._value._nvals


def apply(self):
    if self._value is None:
        self._value = self.new()
    return self._value.apply


def ewise_add(self):
    if self._value is None:
        self._value = self.new()
    return self._value.ewise_add


def ewise_mult(self):
    if self._value is None:
        self._value = self.new()
    return self._value.ewise_mult


def gb_obj(self):
    if self._value is None:
        self._value = self.new()
    return self._value.gb_obj


def inner(self):
    if self._value is None:
        self._value = self.new()
    return self._value.inner


def is_empty(self):
    if self._value is None:
        self._value = self.new()
    return self._value.is_empty


def isclose(self):
    if self._value is None:
        self._value = self.new()
    return self._value.isclose


def isequal(self):
    if self._value is None:
        self._value = self.new()
    return self._value.isequal


def kronecker(self):
    if self._value is None:
        self._value = self.new()
    return self._value.kronecker


def mxm(self):
    if self._value is None:
        self._value = self.new()
    return self._value.mxm


def mxv(self):
    if self._value is None:
        self._value = self.new()
    return self._value.mxv


def name(self):
    if self._value is None:
        self._value = self.new()
    return self._value.name


def nvals(self):
    if self._value is None:
        self._value = self.new()
    return self._value.nvals


def outer(self):
    if self._value is None:
        self._value = self.new()
    return self._value.outer


def reduce(self):
    if self._value is None:
        self._value = self.new()
    return self._value.reduce


def reduce_columns(self):
    if self._value is None:
        self._value = self.new()
    return self._value.reduce_columns


def reduce_rows(self):
    if self._value is None:
        self._value = self.new()
    return self._value.reduce_rows


def reduce_scalar(self):
    if self._value is None:
        self._value = self.new()
    return self._value.reduce_scalar


def to_values(self):
    if self._value is None:
        self._value = self.new()
    return self._value.to_values


def value(self):
    if self._value is None:
        self._value = self.new()
    return self._value.value


def vxm(self):
    if self._value is None:
        self._value = self.new()
    return self._value.vxm

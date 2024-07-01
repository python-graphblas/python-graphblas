from .. import backend, binary
from ..dtypes import BOOL
from ..exceptions import DimensionMismatch
from ..monoid import land, lor
from ..semiring import any_pair
from . import automethods, recorder, utils
from .base import _expect_op, _expect_type
from .expr import InfixExprBase
from .mask import Mask
from .matrix import Matrix, MatrixExpression, TransposedMatrix
from .scalar import Scalar, ScalarExpression
from .utils import output_type, wrapdoc
from .vector import Vector, VectorExpression

InfixExprBase._expect_op = _expect_op
InfixExprBase._expect_type = _expect_type


def _ewise_add_to_expr(self):
    if self._expr is not None:
        return self._expr
    if self.left.dtype == BOOL and self.right.dtype == BOOL:
        self._expr = self.left.ewise_add(self.right, lor)
        return self._expr
    raise TypeError(
        "Bad dtypes for `x | y`!  Automatic computation of `x | y` infix expressions is only valid "
        f"for BOOL dtypes.  The argument dtypes are {self.left.dtype} and {self.right.dtype}.\n\n"
        "When auto-computed for boolean dtypes, `x | y` performs ewise_add (union) using LOR.\n\n"
        "Typical usage is to create an ewise_add expression such as `monoid.plus(x | y)`."
    )


def _ewise_mult_to_expr(self):
    if self._expr is not None:
        return self._expr
    if self.left.dtype == BOOL and self.right.dtype == BOOL:
        self._expr = self.left.ewise_mult(self.right, land)
        return self._expr
    raise TypeError(
        "Bad dtypes for `x & y`!  Automatic computation of `x & y` infix expressions is only valid "
        f"for BOOL dtypes.  The argument dtypes are {self.left.dtype} and {self.right.dtype}.\n\n"
        "When auto-computed for boolean dtypes, `x & y` performs ewise_mult (intersection) using "
        "LAND.\n\n"
        "Typical usage is to create an ewise_mult expression such as `monoid.times(x & y)`."
    )


class ScalarInfixExpr(InfixExprBase):
    __slots__ = ()
    ndim = 0
    shape = ()
    output_type = ScalarExpression
    _is_scalar = True
    _is_cscalar = False
    is_cscalar = False

    def new(self, dtype=None, *, is_cscalar=False, name=None, **opts):
        expr = self._to_expr()
        return expr.new(dtype, is_cscalar=is_cscalar, name=name, **opts)

    @wrapdoc(Scalar.dup)
    def dup(self, dtype=None, *, clear=False, is_cscalar=False, name=None, **opts):
        if dtype is None:
            dtype = self.dtype
        if clear:
            return Scalar(dtype, is_cscalar=is_cscalar, name=name)
        return self.new(dtype, is_cscalar=is_cscalar, name=name, **opts)

    @property
    def is_grbscalar(self):
        return not self.is_cscalar

    # Begin auto-generated code: Scalar
    __and__ = wrapdoc(Scalar.__and__)(property(automethods.__and__))
    __array__ = wrapdoc(Scalar.__array__)(property(automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(automethods.__int__))
    __ne__ = wrapdoc(Scalar.__ne__)(property(automethods.__ne__))
    __or__ = wrapdoc(Scalar.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Scalar.__rand__)(property(automethods.__rand__))
    __ror__ = wrapdoc(Scalar.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Scalar._as_matrix)(property(automethods._as_matrix))
    _as_vector = wrapdoc(Scalar._as_vector)(property(automethods._as_vector))
    _is_empty = wrapdoc(Scalar._is_empty)(property(automethods._is_empty))
    _name_html = wrapdoc(Scalar._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Scalar._nvals)(property(automethods._nvals))
    apply = wrapdoc(Scalar.apply)(property(automethods.apply))
    ewise_add = wrapdoc(Scalar.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Scalar.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Scalar.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Scalar.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Scalar.get)(property(automethods.get))
    is_empty = wrapdoc(Scalar.is_empty)(property(automethods.is_empty))
    isclose = wrapdoc(Scalar.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Scalar.isequal)(property(automethods.isequal))
    name = wrapdoc(Scalar.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Scalar.nvals)(property(automethods.nvals))
    select = wrapdoc(Scalar.select)(property(automethods.select))
    value = wrapdoc(Scalar.value)(property(automethods.value))
    wait = wrapdoc(Scalar.wait)(property(automethods.wait))
    # These raise exceptions
    __matmul__ = Scalar.__matmul__
    __rmatmul__ = Scalar.__rmatmul__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Scalar


class ScalarEwiseAddExpr(ScalarInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _example_op = "plus"
    _infix = "|"

    _to_expr = _ewise_add_to_expr

    # Allow e.g. `plus(x | y | z)`
    __or__ = Scalar.__or__
    __ror__ = Scalar.__ror__
    _ewise_add = Scalar._ewise_add
    _ewise_union = Scalar._ewise_union

    # Don't allow e.g. `plus(x | y & z)`
    def __and__(self, other):
        raise TypeError("XXX")

    def __rand__(self, other):
        raise TypeError("XXX")


class ScalarEwiseMultExpr(ScalarInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _example_op = "times"
    _infix = "&"

    _to_expr = _ewise_mult_to_expr

    # Allow e.g. `plus(x & y & z)`
    __and__ = Scalar.__and__
    __rand__ = Scalar.__rand__
    _ewise_mult = Scalar._ewise_mult

    # Don't allow e.g. `plus(x | y & z)`
    def __or__(self, other):
        raise TypeError("XXX")

    def __ror__(self, other):
        raise TypeError("XXX")


class ScalarMatMulExpr(ScalarInfixExpr):
    __slots__ = ()
    method_name = "inner"
    _example_op = "plus_times"
    _infix = "@"


utils._output_types[ScalarEwiseAddExpr] = Scalar
utils._output_types[ScalarEwiseMultExpr] = Scalar
utils._output_types[ScalarMatMulExpr] = Scalar


class VectorInfixExpr(InfixExprBase):
    __slots__ = "_size"
    ndim = 1
    output_type = VectorExpression

    def __init__(self, left, right):
        super().__init__(left, right)
        self._size = left._size

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    @wrapdoc(Vector.dup)
    def dup(self, dtype=None, *, clear=False, mask=None, name=None, **opts):
        if clear:
            expr = self._to_expr()
            return expr.dup(dtype, clear=clear, name=name, **opts)
        return self.new(dtype, mask=mask, name=name, **opts)

    # Begin auto-generated code: Vector
    S = wrapdoc(Vector.S)(property(automethods.S))
    V = wrapdoc(Vector.V)(property(automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(automethods.apply))
    diag = wrapdoc(Vector.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Vector.get)(property(automethods.get))
    inner = wrapdoc(Vector.inner)(property(automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(automethods.isequal))
    name = wrapdoc(Vector.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(automethods.reduce))
    reposition = wrapdoc(Vector.reposition)(property(automethods.reposition))
    select = wrapdoc(Vector.select)(property(automethods.select))
    if backend == "suitesparse":
        ss = wrapdoc(Vector.ss)(property(automethods.ss))
    else:
        ss = Vector.__dict__["ss"]  # raise if used
    to_coo = wrapdoc(Vector.to_coo)(property(automethods.to_coo))
    to_dense = wrapdoc(Vector.to_dense)(property(automethods.to_dense))
    to_dict = wrapdoc(Vector.to_dict)(property(automethods.to_dict))
    vxm = wrapdoc(Vector.vxm)(property(automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imatmul__ = automethods.__imatmul__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Vector


class VectorEwiseAddExpr(VectorInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _example_op = "plus"
    _infix = "|"

    _to_expr = _ewise_add_to_expr

    # Allow e.g. `plus(x | y | z)`
    __or__ = Vector.__or__
    __ror__ = Vector.__ror__
    _ewise_add = Vector._ewise_add
    _ewise_union = Vector._ewise_union
    # Don't allow e.g. `plus(x | y & z)`
    __and__ = ScalarEwiseAddExpr.__and__  # raises
    __rand__ = ScalarEwiseAddExpr.__rand__  # raises


class VectorEwiseMultExpr(VectorInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _example_op = "times"
    _infix = "&"

    _to_expr = _ewise_mult_to_expr

    # Allow e.g. `plus(x & y & z)`
    __and__ = Vector.__and__
    __rand__ = Vector.__rand__
    _ewise_mult = Vector._ewise_mult
    # Don't allow e.g. `plus(x | y & z)`
    __or__ = ScalarEwiseMultExpr.__or__  # raises
    __ror__ = ScalarEwiseMultExpr.__ror__  # raises


class VectorMatMulExpr(VectorInfixExpr):
    __slots__ = "method_name"
    _example_op = "plus_times"
    _infix = "@"

    def __init__(self, left, right, *, method_name, size):
        InfixExprBase.__init__(self, left, right)
        self.method_name = method_name
        self._size = size

    __matmul__ = Vector.__matmul__
    __rmatmul__ = Vector.__rmatmul__
    _inner = Vector._inner
    _vxm = Vector._vxm


utils._output_types[VectorEwiseAddExpr] = Vector
utils._output_types[VectorEwiseMultExpr] = Vector
utils._output_types[VectorMatMulExpr] = Vector


class MatrixInfixExpr(InfixExprBase):
    __slots__ = "_nrows", "_ncols"
    ndim = 2
    output_type = MatrixExpression
    _is_transposed = False
    __networkx_backend__ = "graphblas"
    __networkx_plugin__ = "graphblas"

    def __init__(self, left, right):
        super().__init__(left, right)
        if left.ndim == 1:
            self._nrows = right._nrows
            self._ncols = right._ncols
        else:
            self._nrows = left._nrows
            self._ncols = left._ncols

    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols

    @property
    def shape(self):
        return (self._nrows, self._ncols)

    @wrapdoc(Matrix.dup)
    def dup(self, dtype=None, *, clear=False, mask=None, name=None, **opts):
        if clear:
            expr = self._to_expr()
            return expr.dup(dtype, clear=clear, name=name, **opts)
        return self.new(dtype, mask=mask, name=name, **opts)

    # Begin auto-generated code: Matrix
    S = wrapdoc(Matrix.S)(property(automethods.S))
    T = wrapdoc(Matrix.T)(property(automethods.T))
    V = wrapdoc(Matrix.V)(property(automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(automethods.__ror__))
    _as_vector = wrapdoc(Matrix._as_vector)(property(automethods._as_vector))
    _carg = wrapdoc(Matrix._carg)(property(automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(automethods.gb_obj))
    get = wrapdoc(Matrix.get)(property(automethods.get))
    isclose = wrapdoc(Matrix.isclose)(property(automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(automethods.mxv))
    name = wrapdoc(Matrix.name)(property(automethods.name)).setter(automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(automethods.nvals))
    power = wrapdoc(Matrix.power)(property(automethods.power))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(automethods.reduce_scalar))
    reposition = wrapdoc(Matrix.reposition)(property(automethods.reposition))
    select = wrapdoc(Matrix.select)(property(automethods.select))
    if backend == "suitesparse":
        ss = wrapdoc(Matrix.ss)(property(automethods.ss))
    else:
        ss = Matrix.__dict__["ss"]  # raise if used
    to_coo = wrapdoc(Matrix.to_coo)(property(automethods.to_coo))
    to_csc = wrapdoc(Matrix.to_csc)(property(automethods.to_csc))
    to_csr = wrapdoc(Matrix.to_csr)(property(automethods.to_csr))
    to_dcsc = wrapdoc(Matrix.to_dcsc)(property(automethods.to_dcsc))
    to_dcsr = wrapdoc(Matrix.to_dcsr)(property(automethods.to_dcsr))
    to_dense = wrapdoc(Matrix.to_dense)(property(automethods.to_dense))
    to_dicts = wrapdoc(Matrix.to_dicts)(property(automethods.to_dicts))
    to_edgelist = wrapdoc(Matrix.to_edgelist)(property(automethods.to_edgelist))
    wait = wrapdoc(Matrix.wait)(property(automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
    __iadd__ = automethods.__iadd__
    __iand__ = automethods.__iand__
    __ifloordiv__ = automethods.__ifloordiv__
    __imatmul__ = automethods.__imatmul__
    __imod__ = automethods.__imod__
    __imul__ = automethods.__imul__
    __ior__ = automethods.__ior__
    __ipow__ = automethods.__ipow__
    __isub__ = automethods.__isub__
    __itruediv__ = automethods.__itruediv__
    __ixor__ = automethods.__ixor__
    # End auto-generated code: Matrix


class MatrixEwiseAddExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _example_op = "plus"
    _infix = "|"

    _to_expr = _ewise_add_to_expr

    # Allow e.g. `plus(x | y | z)`
    __or__ = Matrix.__or__
    __ror__ = Matrix.__ror__
    _ewise_add = Matrix._ewise_add
    _ewise_union = Matrix._ewise_union
    # Don't allow e.g. `plus(x | y & z)`
    __and__ = VectorEwiseAddExpr.__and__  # raises
    __rand__ = VectorEwiseAddExpr.__rand__  # raises


class MatrixEwiseMultExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _example_op = "times"
    _infix = "&"

    _to_expr = _ewise_mult_to_expr

    # Allow e.g. `plus(x & y & z)`
    __and__ = Matrix.__and__
    __rand__ = Matrix.__rand__
    _ewise_mult = Matrix._ewise_mult
    # Don't allow e.g. `plus(x | y & z)`
    __or__ = VectorEwiseMultExpr.__or__  # raises
    __ror__ = VectorEwiseMultExpr.__ror__  # raises


class MatrixMatMulExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "mxm"
    _example_op = "plus_times"
    _infix = "@"

    def __init__(self, left, right, *, nrows, ncols):
        super().__init__(left, right)
        self._nrows = nrows
        self._ncols = ncols

    __matmul__ = Matrix.__matmul__
    __rmatmul__ = Matrix.__rmatmul__
    _mxm = Matrix._mxm
    _mxv = Matrix._mxv


utils._output_types[MatrixEwiseAddExpr] = Matrix
utils._output_types[MatrixEwiseMultExpr] = Matrix
utils._output_types[MatrixMatMulExpr] = Matrix


def _dummy(obj, obj_type):
    with recorder.skip_record:
        return output_type(obj)(BOOL, *obj.shape, name="")


def _mismatched(left, right, method, op):
    # Create dummy expression to raise on incompatible dimensions
    getattr(_dummy(left) if isinstance(left, InfixExprBase) else left, method)(
        _dummy(right) if isinstance(right, InfixExprBase) else right, op
    )
    raise DimensionMismatch  # pragma: no cover


def _ewise_infix_expr(left, right, *, method, within):
    left_type = output_type(left)
    right_type = output_type(right)

    types = {Vector, Matrix, TransposedMatrix}
    if left_type in types and right_type in types:
        if left_type is Vector:
            if right_type is Vector:
                if left._size != right._size:
                    _mismatched(left, right, method, binary.first)
                if method == "ewise_mult":
                    return VectorEwiseMultExpr(left, right)
                return VectorEwiseAddExpr(left, right)
            if left._size != right._nrows:
                _mismatched(left, right, method, binary.first)
        elif right_type is Vector:
            if left._ncols != right._size:
                _mismatched(left, right, method, binary.first)
        elif left.shape != right.shape:
            _mismatched(left, right, method, binary.first)
        if method == "ewise_mult":
            return MatrixEwiseMultExpr(left, right)
        return MatrixEwiseAddExpr(left, right)

    if within == "__or__" and isinstance(right, Mask):
        return right.__ror__(left)
    if within == "__and__" and isinstance(right, Mask):
        return right.__rand__(left)
    if left_type in types:
        left._expect_type(right, tuple(types), within=within, argname="right")
    if right_type in types:
        right._expect_type(left, tuple(types), within=within, argname="left")
    if left_type is Scalar:
        if method == "ewise_mult":
            return ScalarEwiseMultExpr(left, right)
        return ScalarEwiseAddExpr(left, right)
    if right_type is Scalar:
        if method == "ewise_mult":
            return ScalarEwiseMultExpr(right, left)
        return ScalarEwiseAddExpr(right, left)
    raise TypeError(  # pragma: no cover (sanity)
        f"Bad types for ewise infix: {type(left).__name__}, {type(right).__name__}"
    )


def _matmul_infix_expr(left, right, *, within):
    left_type = output_type(left)
    right_type = output_type(right)

    if left_type is Vector:
        if right_type is Matrix or right_type is TransposedMatrix:
            if left._size != right._nrows:
                _mismatched(left, right, "vxm", any_pair[BOOL])
            return VectorMatMulExpr(left, right, method_name="vxm", size=right._ncols)
        if right_type is Vector:
            if left._size != right._size:
                _mismatched(left, right, "inner", any_pair[BOOL])
            return ScalarMatMulExpr(left, right)
        left._expect_type(
            right,
            (Matrix, TransposedMatrix, Vector),
            within=within,
            argname="right",
        )
    if left_type is Matrix or left_type is TransposedMatrix:
        if right_type is Vector:
            if left._ncols != right._size:
                _mismatched(left, right, "mxv", any_pair[BOOL])
            return VectorMatMulExpr(left, right, method_name="mxv", size=left._nrows)
        if right_type is Matrix or right_type is TransposedMatrix:
            if left._ncols != right._nrows:
                _mismatched(left, right, "mxm", any_pair[BOOL])
            return MatrixMatMulExpr(left, right, nrows=left._nrows, ncols=right._ncols)
        left._expect_type(
            right,
            (Vector, Matrix, TransposedMatrix),
            within=within,
            argname="right",
        )
    if right_type is Vector:
        right._expect_type(
            left,
            (Matrix, TransposedMatrix),
            within=within,
            argname="left",
        )
    if right_type is Matrix or right_type is TransposedMatrix:
        right._expect_type(
            left,
            (Vector, Matrix, TransposedMatrix),
            within=within,
            argname="left",
        )
    raise TypeError(  # pragma: no cover (sanity)
        f"Bad types for matmul infix: {type(left).__name__}, {type(right).__name__}"
    )


_ewise_add_expr_types = (MatrixEwiseAddExpr, VectorEwiseAddExpr, ScalarEwiseAddExpr)
_ewise_mult_expr_types = (MatrixEwiseMultExpr, VectorEwiseMultExpr, ScalarEwiseMultExpr)

# Import infixmethods, which has side effects
from . import infixmethods  # noqa: E402, F401 isort:skip

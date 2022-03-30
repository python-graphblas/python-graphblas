from . import _automethods, binary, utils
from .base import _expect_op, _expect_type
from .dtypes import BOOL
from .expr import InfixExprBase
from .matrix import Matrix, MatrixExpression, TransposedMatrix
from .monoid import land, lor
from .scalar import Scalar, ScalarExpression
from .semiring import any_pair
from .utils import output_type, wrapdoc
from .vector import Vector, VectorExpression

InfixExprBase._expect_op = _expect_op
InfixExprBase._expect_type = _expect_type


def _ewise_add_to_expr(self):
    if self._value is not None:
        return self._value
    elif self.left.dtype == BOOL and self.right.dtype == BOOL:
        self._value = self.left.ewise_add(self.right, lor)
        return self._value
    raise TypeError(
        "Bad dtypes for `x | y`!  Automatic computation of `x | y` infix expressions is only valid "
        f"for BOOL dtypes.  The argument dtypes are {self.left.dtype} and {self.right.dtype}.\n\n"
        "When auto-computed for boolean dtypes, `x | y` performs ewise_add (union) using LOR.\n\n"
        "Typical usage is to create an ewise_add expression such as `monoid.plus(x | y)`."
    )


def _ewise_mult_to_expr(self):
    if self._value is not None:
        return self._value
    elif self.left.dtype == BOOL and self.right.dtype == BOOL:
        self._value = self.left.ewise_mult(self.right, land)
        return self._value
    raise TypeError(
        "Bad dtypes for `x & y`!  Automatic computation of `x & y` infix expressions is only valid "
        f"for BOOL dtypes.  The argument dtypes are {self.left.dtype} and {self.right.dtype}.\n\n"
        "When auto-computed for boolean dtypes, `x & y` performs ewise_mult (intersection) using "
        "LAND.\n\n"
        "Typical usage is to create an ewise_mult expression such as `monoid.times(x & y)`."
    )


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

    # Begin auto-generated code: Vector
    S = wrapdoc(Vector.S)(property(_automethods.S))
    V = wrapdoc(Vector.V)(property(_automethods.V))
    __and__ = wrapdoc(Vector.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Vector.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Vector.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Vector.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Vector.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Vector.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Vector.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Vector.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Vector.__ror__)(property(_automethods.__ror__))
    _as_matrix = wrapdoc(Vector._as_matrix)(property(_automethods._as_matrix))
    _carg = wrapdoc(Vector._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Vector._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Vector._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Vector.apply)(property(_automethods.apply))
    diag = wrapdoc(Vector.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Vector.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Vector.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Vector.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Vector.gb_obj)(property(_automethods.gb_obj))
    inner = wrapdoc(Vector.inner)(property(_automethods.inner))
    isclose = wrapdoc(Vector.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Vector.isequal)(property(_automethods.isequal))
    name = wrapdoc(Vector.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Vector.nvals)(property(_automethods.nvals))
    outer = wrapdoc(Vector.outer)(property(_automethods.outer))
    reduce = wrapdoc(Vector.reduce)(property(_automethods.reduce))
    ss = wrapdoc(Vector.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Vector.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Vector.to_values)(property(_automethods.to_values))
    vxm = wrapdoc(Vector.vxm)(property(_automethods.vxm))
    wait = wrapdoc(Vector.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Vector.__array__
    __bool__ = Vector.__bool__
    __iadd__ = _automethods.__iadd__
    __iand__ = _automethods.__iand__
    __ifloordiv__ = _automethods.__ifloordiv__
    __imatmul__ = _automethods.__imatmul__
    __imod__ = _automethods.__imod__
    __imul__ = _automethods.__imul__
    __ior__ = _automethods.__ior__
    __ipow__ = _automethods.__ipow__
    __isub__ = _automethods.__isub__
    __itruediv__ = _automethods.__itruediv__
    __ixor__ = _automethods.__ixor__
    # End auto-generated code: Vector


class VectorEwiseAddExpr(VectorInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _example_op = "plus"
    _infix = "|"

    _to_expr = _ewise_add_to_expr


class VectorEwiseMultExpr(VectorInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _example_op = "times"
    _infix = "&"

    _to_expr = _ewise_mult_to_expr


class VectorMatMulExpr(VectorInfixExpr):
    __slots__ = "method_name"
    _example_op = "plus_times"
    _infix = "@"

    def __init__(self, left, right, *, method_name, size):
        InfixExprBase.__init__(self, left, right)
        self.method_name = method_name
        self._size = size


utils._output_types[VectorEwiseAddExpr] = Vector
utils._output_types[VectorEwiseMultExpr] = Vector
utils._output_types[VectorMatMulExpr] = Vector


class MatrixInfixExpr(InfixExprBase):
    __slots__ = "_nrows", "_ncols"
    ndim = 2
    output_type = MatrixExpression
    _is_transposed = False

    def __init__(self, left, right):
        super().__init__(left, right)
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

    # Begin auto-generated code: Matrix
    S = wrapdoc(Matrix.S)(property(_automethods.S))
    T = wrapdoc(Matrix.T)(property(_automethods.T))
    V = wrapdoc(Matrix.V)(property(_automethods.V))
    __and__ = wrapdoc(Matrix.__and__)(property(_automethods.__and__))
    __contains__ = wrapdoc(Matrix.__contains__)(property(_automethods.__contains__))
    __getitem__ = wrapdoc(Matrix.__getitem__)(property(_automethods.__getitem__))
    __iter__ = wrapdoc(Matrix.__iter__)(property(_automethods.__iter__))
    __matmul__ = wrapdoc(Matrix.__matmul__)(property(_automethods.__matmul__))
    __or__ = wrapdoc(Matrix.__or__)(property(_automethods.__or__))
    __rand__ = wrapdoc(Matrix.__rand__)(property(_automethods.__rand__))
    __rmatmul__ = wrapdoc(Matrix.__rmatmul__)(property(_automethods.__rmatmul__))
    __ror__ = wrapdoc(Matrix.__ror__)(property(_automethods.__ror__))
    _carg = wrapdoc(Matrix._carg)(property(_automethods._carg))
    _name_html = wrapdoc(Matrix._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Matrix._nvals)(property(_automethods._nvals))
    apply = wrapdoc(Matrix.apply)(property(_automethods.apply))
    diag = wrapdoc(Matrix.diag)(property(_automethods.diag))
    ewise_add = wrapdoc(Matrix.ewise_add)(property(_automethods.ewise_add))
    ewise_mult = wrapdoc(Matrix.ewise_mult)(property(_automethods.ewise_mult))
    ewise_union = wrapdoc(Matrix.ewise_union)(property(_automethods.ewise_union))
    gb_obj = wrapdoc(Matrix.gb_obj)(property(_automethods.gb_obj))
    isclose = wrapdoc(Matrix.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Matrix.isequal)(property(_automethods.isequal))
    kronecker = wrapdoc(Matrix.kronecker)(property(_automethods.kronecker))
    mxm = wrapdoc(Matrix.mxm)(property(_automethods.mxm))
    mxv = wrapdoc(Matrix.mxv)(property(_automethods.mxv))
    name = wrapdoc(Matrix.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Matrix.nvals)(property(_automethods.nvals))
    reduce_columnwise = wrapdoc(Matrix.reduce_columnwise)(property(_automethods.reduce_columnwise))
    reduce_rowwise = wrapdoc(Matrix.reduce_rowwise)(property(_automethods.reduce_rowwise))
    reduce_scalar = wrapdoc(Matrix.reduce_scalar)(property(_automethods.reduce_scalar))
    ss = wrapdoc(Matrix.ss)(property(_automethods.ss))
    to_pygraphblas = wrapdoc(Matrix.to_pygraphblas)(property(_automethods.to_pygraphblas))
    to_values = wrapdoc(Matrix.to_values)(property(_automethods.to_values))
    wait = wrapdoc(Matrix.wait)(property(_automethods.wait))
    # These raise exceptions
    __array__ = Matrix.__array__
    __bool__ = Matrix.__bool__
    __iadd__ = _automethods.__iadd__
    __iand__ = _automethods.__iand__
    __ifloordiv__ = _automethods.__ifloordiv__
    __imatmul__ = _automethods.__imatmul__
    __imod__ = _automethods.__imod__
    __imul__ = _automethods.__imul__
    __ior__ = _automethods.__ior__
    __ipow__ = _automethods.__ipow__
    __isub__ = _automethods.__isub__
    __itruediv__ = _automethods.__itruediv__
    __ixor__ = _automethods.__ixor__
    # End auto-generated code: Matrix


class MatrixEwiseAddExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "ewise_add"
    _example_op = "plus"
    _infix = "|"

    _to_expr = _ewise_add_to_expr


class MatrixEwiseMultExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "ewise_mult"
    _example_op = "times"
    _infix = "&"

    _to_expr = _ewise_mult_to_expr


class MatrixMatMulExpr(MatrixInfixExpr):
    __slots__ = ()
    method_name = "mxm"
    _example_op = "plus_times"
    _infix = "@"

    def __init__(self, left, right, *, nrows, ncols):
        super().__init__(left, right)
        self._nrows = nrows
        self._ncols = ncols


utils._output_types[MatrixEwiseAddExpr] = Matrix
utils._output_types[MatrixEwiseMultExpr] = Matrix
utils._output_types[MatrixMatMulExpr] = Matrix


class ScalarMatMulExpr(InfixExprBase):
    __slots__ = ()
    method_name = "inner"
    ndim = 0
    output_type = ScalarExpression
    shape = ()
    _example_op = "plus_times"
    _infix = "@"
    _is_scalar = True

    def new(self, dtype=None, *, is_cscalar=None, name=None):
        # Rely on the default operator for the method
        expr = getattr(self.left, self.method_name)(self.right)
        return expr.new(dtype, is_cscalar=is_cscalar, name=name)

    dup = new

    @property
    def is_cscalar(self):
        if self._value is not None:
            return self._value._is_cscalar
        return self._to_expr()._is_cscalar

    _is_cscalar = is_cscalar

    @property
    def is_grbscalar(self):
        if self._value is not None:
            return not self._value._is_cscalar
        return not self._to_expr()._is_cscalar

    # Begin auto-generated code: Scalar
    __array__ = wrapdoc(Scalar.__array__)(property(_automethods.__array__))
    __bool__ = wrapdoc(Scalar.__bool__)(property(_automethods.__bool__))
    __complex__ = wrapdoc(Scalar.__complex__)(property(_automethods.__complex__))
    __eq__ = wrapdoc(Scalar.__eq__)(property(_automethods.__eq__))
    __float__ = wrapdoc(Scalar.__float__)(property(_automethods.__float__))
    __index__ = wrapdoc(Scalar.__index__)(property(_automethods.__index__))
    __int__ = wrapdoc(Scalar.__int__)(property(_automethods.__int__))
    __invert__ = wrapdoc(Scalar.__invert__)(property(_automethods.__invert__))
    __neg__ = wrapdoc(Scalar.__neg__)(property(_automethods.__neg__))
    _as_matrix = wrapdoc(Scalar._as_matrix)(property(_automethods._as_matrix))
    _as_vector = wrapdoc(Scalar._as_vector)(property(_automethods._as_vector))
    _is_empty = wrapdoc(Scalar._is_empty)(property(_automethods._is_empty))
    _name_html = wrapdoc(Scalar._name_html)(property(_automethods._name_html))
    _nvals = wrapdoc(Scalar._nvals)(property(_automethods._nvals))
    gb_obj = wrapdoc(Scalar.gb_obj)(property(_automethods.gb_obj))
    is_empty = wrapdoc(Scalar.is_empty)(property(_automethods.is_empty))
    isclose = wrapdoc(Scalar.isclose)(property(_automethods.isclose))
    isequal = wrapdoc(Scalar.isequal)(property(_automethods.isequal))
    name = wrapdoc(Scalar.name)(property(_automethods.name))
    name = name.setter(_automethods._set_name)
    nvals = wrapdoc(Scalar.nvals)(property(_automethods.nvals))
    to_pygraphblas = wrapdoc(Scalar.to_pygraphblas)(property(_automethods.to_pygraphblas))
    value = wrapdoc(Scalar.value)(property(_automethods.value))
    wait = wrapdoc(Scalar.wait)(property(_automethods.wait))
    # These raise exceptions
    __and__ = Scalar.__and__
    __matmul__ = Scalar.__matmul__
    __or__ = Scalar.__or__
    __rand__ = Scalar.__rand__
    __rmatmul__ = Scalar.__rmatmul__
    __ror__ = Scalar.__ror__
    # End auto-generated code: Scalar


utils._output_types[ScalarMatMulExpr] = Scalar


def _ewise_infix_expr(left, right, *, method, within):
    left_type = output_type(left)
    right_type = output_type(right)

    if left_type in {Vector, Matrix, TransposedMatrix}:
        if not (
            left_type is right_type
            or (left_type is Matrix and right_type is TransposedMatrix)
            or (left_type is TransposedMatrix and right_type is Matrix)
        ):
            if left_type is Vector:
                right = left._expect_type(right, Vector, within=within, argname="right")
            else:
                right = left._expect_type(
                    right, (Matrix, TransposedMatrix), within=within, argname="right"
                )
    elif right_type is Vector:
        left = right._expect_type(left, Vector, within=within, argname="left")
    elif right_type is Matrix or right_type is TransposedMatrix:
        left = right._expect_type(left, (Matrix, TransposedMatrix), within=within, argname="left")
    else:  # pragma: no cover
        raise TypeError(f"Bad types for ewise infix: {type(left).__name__}, {type(right).__name__}")

    # Create dummy expression to check compatibility of dimensions, etc.
    expr = getattr(left, method)(right, binary.any)
    if expr.output_type is Vector:
        if method == "ewise_mult":
            return VectorEwiseMultExpr(left, right)
        return VectorEwiseAddExpr(left, right)
    elif method == "ewise_mult":
        return MatrixEwiseMultExpr(left, right)
    return MatrixEwiseAddExpr(left, right)


def _matmul_infix_expr(left, right, *, within):
    left_type = output_type(left)
    right_type = output_type(right)

    if left_type is Vector:
        if right_type is Matrix or right_type is TransposedMatrix:
            method = "vxm"
        elif right_type is Vector:
            method = "inner"
        else:
            right = left._expect_type(
                right,
                (Matrix, TransposedMatrix),
                within=within,
                argname="right",
            )
    elif left_type is Matrix or left_type is TransposedMatrix:
        if right_type is Vector:
            method = "mxv"
        elif right_type is Matrix or right_type is TransposedMatrix:
            method = "mxm"
        else:
            right = left._expect_type(
                right,
                (Vector, Matrix, TransposedMatrix),
                within=within,
                argname="right",
            )
    elif right_type is Vector:
        left = right._expect_type(
            left,
            (Matrix, TransposedMatrix),
            within=within,
            argname="left",
        )
    elif right_type is Matrix or right_type is TransposedMatrix:
        left = right._expect_type(
            left,
            (Vector, Matrix, TransposedMatrix),
            within=within,
            argname="left",
        )
    else:  # pragma: no cover
        raise TypeError(
            f"Bad types for matmul infix: {type(left).__name__}, {type(right).__name__}"
        )

    # Create dummy expression to check compatibility of dimensions, etc.
    expr = getattr(left, method)(right, any_pair[bool])
    if expr.output_type is Vector:
        return VectorMatMulExpr(left, right, method_name=method, size=expr._size)
    elif expr.output_type is Matrix:
        return MatrixMatMulExpr(left, right, nrows=expr._nrows, ncols=expr._ncols)
    return ScalarMatMulExpr(left, right)


# Import _infixmethods, which has side effects
from . import _infixmethods  # noqa isort:skip

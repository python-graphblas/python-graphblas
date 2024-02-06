from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.binary import BinaryOp, TypedUserBinaryOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitBinaryOp(TypedOpBase):
    __slots__ = "_monoid", "_jit_c_definition"
    opclass = "BinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._monoid = None
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    monoid = TypedUserBinaryOp.monoid
    commutes_to = TypedUserBinaryOp.commutes_to
    _semiring_commutes_to = TypedUserBinaryOp._semiring_commutes_to
    is_commutative = TypedUserBinaryOp.is_commutative
    type2 = TypedUserBinaryOp.type2
    __call__ = TypedUserBinaryOp.__call__


def register_new(name, jit_c_definition, left_type, right_type, ret_type):
    """Register a new BinaryOp using the SuiteSparse:GraphBLAS JIT compiler.

    This creates a BinaryOp by compiling the C string definition of the function.
    It requires a shell call to a C compiler. The resulting operator will be as
    fast as if it were built-in to SuiteSparse:GraphBLAS and does not have the
    overhead of additional function calls as when using ``gb.binary.register_new``.

    This is an advanced feature that requires a C compiler and proper configuration.
    Configuration is handled by ``gb.ss.config``; see its docstring for details.
    By default, the JIT caches results in ``~/.SuiteSparse/``. For more information,
    see the SuiteSparse:GraphBLAS user guide.

    Only one type signature may be registered at a time, but repeated calls using
    the same name with different input types is allowed.

    Parameters
    ----------
    name : str
        The name of the operator. This will show up as ``gb.binary.ss.{name}``.
        The name may contain periods, ".", which will result in nested objects
        such as ``gb.binary.ss.x.y.z`` for name ``"x.y.z"``.
    jit_c_definition : str
        The C definition as a string of the user-defined function. For example:
        ``"void absdiff (double *z, double *x, double *y) { (*z) = fabs ((*x) - (*y)) ; }"``.
    left_type : dtype
        The dtype of the left operand of the binary operator.
    right_type : dtype
        The dtype of the right operand of the binary operator.
    ret_type : dtype
        The dtype of the result of the binary operator.

    Returns
    -------
    BinaryOp

    See Also
    --------
    gb.binary.register_new
    gb.binary.register_anonymous
    gb.unary.ss.register_new

    """
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.binary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    left_type = lookup_dtype(left_type)
    right_type = lookup_dtype(right_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = BinaryOp._remove_nesting(name, strict=False)
    if hasattr(module, funcname):
        rv = getattr(module, funcname)
        if not isinstance(rv, BinaryOp):
            BinaryOp._remove_nesting(name)
        if (
            (left_type, right_type) in rv.types
            or rv._udt_types is not None
            and (left_type, right_type) in rv._udt_types
        ):
            raise TypeError(
                f"BinaryOp gb.binary.{name} already defined for "
                f"({left_type}, {right_type}) input types"
            )
    else:
        # We use `is_udt=True` to make dtype handling flexible and explicit.
        rv = BinaryOp(name, is_udt=True)
    gb_obj = ffi_new("GrB_BinaryOp*")
    check_status_carg(
        lib.GxB_BinaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            left_type._carg,
            right_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "BinaryOp",
        gb_obj[0],
    )
    op = TypedJitBinaryOp(
        rv, funcname, left_type, ret_type, gb_obj[0], jit_c_definition, dtype2=right_type
    )
    rv._add(op, is_jit=True)
    setattr(module, funcname, rv)
    return rv

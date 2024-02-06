from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.unary import TypedUserUnaryOp, UnaryOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitUnaryOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "UnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition):
        super().__init__(parent, name, type_, return_type, gb_obj, name)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    __call__ = TypedUserUnaryOp.__call__


def register_new(name, jit_c_definition, input_type, ret_type):
    """Register a new UnaryOp using the SuiteSparse:GraphBLAS JIT compiler.

    This creates a UnaryOp by compiling the C string definition of the function.
    It requires a shell call to a C compiler. The resulting operator will be as
    fast as if it were built-in to SuiteSparse:GraphBLAS and does not have the
    overhead of additional function calls as when using ``gb.unary.register_new``.

    This is an advanced feature that requires a C compiler and proper configuration.
    Configuration is handled by ``gb.ss.config``; see its docstring for details.
    By default, the JIT caches results in ``~/.SuiteSparse/``. For more information,
    see the SuiteSparse:GraphBLAS user guide.

    Only one type signature may be registered at a time, but repeated calls using
    the same name with different input types is allowed.

    Parameters
    ----------
    name : str
        The name of the operator. This will show up as ``gb.unary.ss.{name}``.
        The name may contain periods, ".", which will result in nested objects
        such as ``gb.unary.ss.x.y.z`` for name ``"x.y.z"``.
    jit_c_definition : str
        The C definition as a string of the user-defined function. For example:
        ``"void square (float *z, float *x) { (*z) = (*x) * (*x) ; } ;"``
    input_type : dtype
        The dtype of the operand of the unary operator.
    ret_type : dtype
        The dtype of the result of the unary operator.

    Returns
    -------
    UnaryOp

    See Also
    --------
    gb.unary.register_new
    gb.unary.register_anonymous
    gb.binary.ss.register_new

    """
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.unary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    input_type = lookup_dtype(input_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = UnaryOp._remove_nesting(name, strict=False)
    if hasattr(module, funcname):
        rv = getattr(module, funcname)
        if not isinstance(rv, UnaryOp):
            UnaryOp._remove_nesting(name)
        if input_type in rv.types or rv._udt_types is not None and input_type in rv._udt_types:
            raise TypeError(f"UnaryOp gb.unary.{name} already defined for {input_type} input type")
    else:
        # We use `is_udt=True` to make dtype handling flexible and explicit.
        rv = UnaryOp(name, is_udt=True)
    gb_obj = ffi_new("GrB_UnaryOp*")
    check_status_carg(
        lib.GxB_UnaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            input_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "UnaryOp",
        gb_obj[0],
    )
    op = TypedJitUnaryOp(rv, funcname, input_type, ret_type, gb_obj[0], jit_c_definition)
    rv._add(op, is_jit=True)
    setattr(module, funcname, rv)
    return rv

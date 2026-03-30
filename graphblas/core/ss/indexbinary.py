from ... import backend
from ...dtypes import lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.indexbinary import (
    IndexBinaryOp,
    TypedBuiltinIndexBinaryOp,
    TypedUserIndexBinaryOp,
    _has_idxbinop,
)

ffi_new = ffi.new


class TypedJitIndexBinaryOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "IndexBinaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    thunk_type = TypedUserIndexBinaryOp.thunk_type

    def __call__(self, theta=None):
        return TypedBuiltinIndexBinaryOp.__call__(self, theta)

    __call__.__doc__ = TypedBuiltinIndexBinaryOp.__call__.__doc__


def register_new(name, jit_c_definition, x_type, y_type, theta_type, ret_type):
    """Register a new IndexBinaryOp using the SuiteSparse:GraphBLAS JIT compiler.

    This creates an IndexBinaryOp by compiling the C string definition of the function.
    It requires a shell call to a C compiler. The resulting operator will be as
    fast as if it were built-in to SuiteSparse:GraphBLAS and does not have the
    overhead of additional function calls as when using ``gb.indexbinary.register_new``.

    This is an advanced feature that requires a C compiler and proper configuration.
    Configuration is handled by ``gb.ss.config``; see its docstring for details.
    By default, the JIT caches results in ``~/.SuiteSparse/``. For more information,
    see the SuiteSparse:GraphBLAS user guide.

    Only one type signature may be registered at a time, but repeated calls using
    the same name with different input types is allowed.

    Parameters
    ----------
    name : str
        The name of the operator. This will show up as ``gb.indexbinary.ss.{name}``.
        The name may contain periods, ".", which will result in nested objects
        such as ``gb.indexbinary.ss.x.y.z`` for name ``"x.y.z"``.
    jit_c_definition : str
        The C definition as a string of the user-defined function. For example:
        ``"void my_idxbin (double *z, double *x, GrB_Index ix, GrB_Index jx, "``
        ``"double *y, GrB_Index iy, GrB_Index jy, double *theta) "``
        ``"{ (*z) = (*x) + (*y) + (*theta) ; }"``
    x_type : dtype
        The dtype of the first operand (x).
    y_type : dtype
        The dtype of the second operand (y).
    theta_type : dtype
        The dtype of the theta parameter.
    ret_type : dtype
        The dtype of the result.

    Returns
    -------
    IndexBinaryOp

    See Also
    --------
    gb.indexbinary.register_new
    gb.indexbinary.register_anonymous

    """
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.indexbinary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if not _has_idxbinop:
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "IndexBinaryOp requires SuiteSparse:GraphBLAS 9.4+; "
            f"current version is {ssgb.__version__}"
        )
    x_type = lookup_dtype(x_type)
    y_type = lookup_dtype(y_type)
    theta_type = lookup_dtype(theta_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = IndexBinaryOp._remove_nesting(name, strict=False)
    if hasattr(module, funcname):
        rv = getattr(module, funcname)
        if not isinstance(rv, IndexBinaryOp):
            IndexBinaryOp._remove_nesting(name)
        if (
            (x_type, theta_type) in rv.types
            or rv._udt_types is not None
            and (x_type, theta_type) in rv._udt_types
        ):
            raise TypeError(
                f"IndexBinaryOp gb.indexbinary.{name} already defined for "
                f"({x_type}, {theta_type}) input types"
            )
    else:
        rv = IndexBinaryOp(name, is_udt=True)
    gb_obj = ffi_new("GxB_IndexBinaryOp*")
    check_status_carg(
        lib.GxB_IndexBinaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            x_type._carg,
            y_type._carg,
            theta_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "IndexBinaryOp",
        gb_obj[0],
    )
    op = TypedJitIndexBinaryOp(
        rv, funcname, x_type, ret_type, gb_obj[0], jit_c_definition, dtype2=theta_type
    )
    rv._add(op, is_jit=True)
    setattr(module, funcname, rv)
    return rv

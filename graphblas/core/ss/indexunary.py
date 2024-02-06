from ... import backend
from ...dtypes import BOOL, lookup_dtype
from ...exceptions import check_status_carg
from .. import NULL, ffi, lib
from ..operator.base import TypedOpBase
from ..operator.indexunary import IndexUnaryOp, TypedUserIndexUnaryOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitIndexUnaryOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "IndexUnaryOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    thunk_type = TypedUserIndexUnaryOp.thunk_type
    __call__ = TypedUserIndexUnaryOp.__call__


def register_new(name, jit_c_definition, input_type, thunk_type, ret_type):
    """Register a new IndexUnaryOp using the SuiteSparse:GraphBLAS JIT compiler.

    This creates a IndexUnaryOp by compiling the C string definition of the function.
    It requires a shell call to a C compiler. The resulting operator will be as
    fast as if it were built-in to SuiteSparse:GraphBLAS and does not have the
    overhead of additional function calls as when using ``gb.indexunary.register_new``.

    This is an advanced feature that requires a C compiler and proper configuration.
    Configuration is handled by ``gb.ss.config``; see its docstring for details.
    By default, the JIT caches results in ``~/.SuiteSparse/``. For more information,
    see the SuiteSparse:GraphBLAS user guide.

    Only one type signature may be registered at a time, but repeated calls using
    the same name with different input types is allowed.

    This will also create a SelectOp operator under ``gb.select.ss`` if the return
    type is boolean.

    Parameters
    ----------
    name : str
        The name of the operator. This will show up as ``gb.indexunary.ss.{name}``.
        The name may contain periods, ".", which will result in nested objects
        such as ``gb.indexunary.ss.x.y.z`` for name ``"x.y.z"``.
    jit_c_definition : str
        The C definition as a string of the user-defined function. For example:
        ``"void diffy (double *z, double *x, GrB_Index i, GrB_Index j, double *y) "``
        ``"{ (*z) = (i + j) * fabs ((*x) - (*y)) ; }"``
    input_type : dtype
        The dtype of the operand of the indexunary operator.
    thunk_type : dtype
        The dtype of the thunk of the indexunary operator.
    ret_type : dtype
        The dtype of the result of the indexunary operator.

    Returns
    -------
    IndexUnaryOp

    See Also
    --------
    gb.indexunary.register_new
    gb.indexunary.register_anonymous
    gb.select.ss.register_new

    """
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.indexunary.ss.register_new` invalid when not using 'suitesparse' backend"
        )
    if _IS_SSGB7:
        # JIT was introduced in SuiteSparse:GraphBLAS 8.0
        import suitesparse_graphblas as ssgb

        raise RuntimeError(
            "JIT was added to SuiteSparse:GraphBLAS in version 8; "
            f"current version is {ssgb.__version__}"
        )
    input_type = lookup_dtype(input_type)
    thunk_type = lookup_dtype(thunk_type)
    ret_type = lookup_dtype(ret_type)
    name = name if name.startswith("ss.") else f"ss.{name}"
    module, funcname = IndexUnaryOp._remove_nesting(name, strict=False)
    if hasattr(module, funcname):
        rv = getattr(module, funcname)
        if not isinstance(rv, IndexUnaryOp):
            IndexUnaryOp._remove_nesting(name)
        if (
            (input_type, thunk_type) in rv.types
            or rv._udt_types is not None
            and (input_type, thunk_type) in rv._udt_types
        ):
            raise TypeError(
                f"IndexUnaryOp gb.indexunary.{name} already defined for "
                f"({input_type}, {thunk_type}) input types"
            )
    else:
        # We use `is_udt=True` to make dtype handling flexible and explicit.
        rv = IndexUnaryOp(name, is_udt=True)
    gb_obj = ffi_new("GrB_IndexUnaryOp*")
    check_status_carg(
        lib.GxB_IndexUnaryOp_new(
            gb_obj,
            NULL,
            ret_type._carg,
            input_type._carg,
            thunk_type._carg,
            ffi_new("char[]", funcname.encode()),
            ffi_new("char[]", jit_c_definition.encode()),
        ),
        "IndexUnaryOp",
        gb_obj[0],
    )
    op = TypedJitIndexUnaryOp(
        rv, funcname, input_type, ret_type, gb_obj[0], jit_c_definition, dtype2=thunk_type
    )
    rv._add(op, is_jit=True)
    if ret_type == BOOL:
        from ..operator.select import SelectOp
        from .select import TypedJitSelectOp

        select_module, funcname = SelectOp._remove_nesting(name, strict=False)
        if hasattr(select_module, funcname):
            selectop = getattr(select_module, funcname)
            if not isinstance(selectop, SelectOp):
                SelectOp._remove_nesting(name)
            if (
                (input_type, thunk_type) in selectop.types
                or selectop._udt_types is not None
                and (input_type, thunk_type) in selectop._udt_types
            ):
                raise TypeError(
                    f"SelectOp gb.select.{name} already defined for "
                    f"({input_type}, {thunk_type}) input types"
                )
        else:
            # We use `is_udt=True` to make dtype handling flexible and explicit.
            selectop = SelectOp(name, is_udt=True)
        op2 = TypedJitSelectOp(
            selectop, funcname, input_type, ret_type, gb_obj[0], jit_c_definition, dtype2=thunk_type
        )
        selectop._add(op2, is_jit=True)
        setattr(select_module, funcname, selectop)
    setattr(module, funcname, rv)
    return rv

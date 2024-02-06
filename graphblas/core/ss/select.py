from ... import backend, indexunary
from ...dtypes import BOOL, lookup_dtype
from .. import ffi
from ..operator.base import TypedOpBase
from ..operator.select import SelectOp, TypedUserSelectOp
from . import _IS_SSGB7

ffi_new = ffi.new


class TypedJitSelectOp(TypedOpBase):
    __slots__ = "_jit_c_definition"
    opclass = "SelectOp"

    def __init__(self, parent, name, type_, return_type, gb_obj, jit_c_definition, dtype2=None):
        super().__init__(parent, name, type_, return_type, gb_obj, name, dtype2=dtype2)
        self._jit_c_definition = jit_c_definition

    @property
    def jit_c_definition(self):
        return self._jit_c_definition

    thunk_type = TypedUserSelectOp.thunk_type
    __call__ = TypedUserSelectOp.__call__


def register_new(name, jit_c_definition, input_type, thunk_type):
    """Register a new SelectOp using the SuiteSparse:GraphBLAS JIT compiler.

    This creates a SelectOp by compiling the C string definition of the function.
    It requires a shell call to a C compiler. The resulting operator will be as
    fast as if it were built-in to SuiteSparse:GraphBLAS and does not have the
    overhead of additional function calls as when using ``gb.select.register_new``.

    This is an advanced feature that requires a C compiler and proper configuration.
    Configuration is handled by ``gb.ss.config``; see its docstring for details.
    By default, the JIT caches results in ``~/.SuiteSparse/``. For more information,
    see the SuiteSparse:GraphBLAS user guide.

    Only one type signature may be registered at a time, but repeated calls using
    the same name with different input types is allowed.

    This will also create an IndexUnary operator under ``gb.indexunary.ss``

    Parameters
    ----------
    name : str
        The name of the operator. This will show up as ``gb.select.ss.{name}``.
        The name may contain periods, ".", which will result in nested objects
        such as ``gb.select.ss.x.y.z`` for name ``"x.y.z"``.
    jit_c_definition : str
        The C definition as a string of the user-defined function. For example:
        ``"void woot (bool *z, const int32_t *x, GrB_Index i, GrB_Index j, int32_t *y) "``
        ``"{ (*z) = ((*x) + i + j == (*y)) ; }"``
    input_type : dtype
        The dtype of the operand of the select operator.
    thunk_type : dtype
        The dtype of the thunk of the select operator.

    Returns
    -------
    SelectOp

    See Also
    --------
    gb.select.register_new
    gb.select.register_anonymous
    gb.indexunary.ss.register_new

    """
    if backend != "suitesparse":  # pragma: no cover (safety)
        raise RuntimeError(
            "`gb.select.ss.register_new` invalid when not using 'suitesparse' backend"
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
    name = name if name.startswith("ss.") else f"ss.{name}"
    # Register to both `gb.indexunary.ss` and `gb.select.ss.`
    indexunary.ss.register_new(name, jit_c_definition, input_type, thunk_type, BOOL)
    module, funcname = SelectOp._remove_nesting(name, strict=False)
    return getattr(module, funcname)

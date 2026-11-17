from suitesparse_graphblas import burble

from .. import backend
from ._core import _IS_SSGB7, about, concat, config, diag

if not _IS_SSGB7:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0.
    from ..core.ss.context import Context, global_context

    # SuiteSparse:GraphBLAS 8.0 introduced the JIT.
    from ..core.ss.jit_config import _auto_fix_jit_at_import as _auto_fix
    from ..core.ss.jit_config import fix_jit_config, jit_compiler_is_usable

    # Repair the JIT compiler setting; see ``_auto_fix_jit_at_import``. Not on
    # vanilla: the repair reads ``gb.ss.config``, which needs GxB functions that
    # vanilla leaves out.
    if backend == "suitesparse":
        _auto_fix()
    del _auto_fix

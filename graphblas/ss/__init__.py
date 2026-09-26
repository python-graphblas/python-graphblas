from suitesparse_graphblas import burble

from .. import backend
from ._core import _IS_SSGB7, about, concat, config, diag

if not _IS_SSGB7:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0.
    from ..core.ss.context import Context, global_context

    # JIT compiler config helpers, introduced alongside JIT in SS 8.
    from ..core.ss.jit_config import _auto_fix_jit_at_import as _auto_fix
    from ..core.ss.jit_config import fix_jit_config, jit_compiler_is_usable

    # Repair the JIT compiler path on import. Conda-built psg bakes in a
    # compiler path that almost never exists at runtime, so JIT silently falls
    # back to the cfunc path (2-3x slowdown). This only rewrites the compiler
    # name and flags; ``jit_c_control`` is left where SuiteSparse set it, so
    # importing this submodule cannot change what a later operation computes.
    # Compilation is enabled on demand by ``_enable_jit_for_udt``. Skipped on
    # vanilla: the repair reads ``gb.ss.config``, which needs GxB callables
    # vanilla strips.
    if backend == "suitesparse":
        _auto_fix()
    del _auto_fix

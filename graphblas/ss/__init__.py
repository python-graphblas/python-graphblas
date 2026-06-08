from suitesparse_graphblas import burble

from .. import backend
from ._core import _IS_SSGB7, about, concat, config, diag

if not _IS_SSGB7:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0.
    from ..core.ss.context import Context, global_context

    # JIT compiler config helpers, introduced alongside JIT in SS 8.
    from ..core.ss.jit_config import _auto_fix_jit_at_import as _auto_fix
    from ..core.ss.jit_config import fix_jit_config, jit_compiler_is_usable

    # Auto-fix the JIT config on import. Conda-built psg bakes in a compiler
    # path that almost never exists at runtime, so JIT silently falls back to
    # the cfunc path (2-3x slowdown). When the compiler is already usable, the
    # call only bumps ``jit_c_control`` from SS's default ``run`` to ``on`` so
    # first-time UDT auto-lift actually JIT-compiles. Skipped on vanilla: the
    # auto-fix reads ``gb.ss.config``, which needs GxB callables vanilla strips.
    if backend == "suitesparse":
        _auto_fix()
    del _auto_fix

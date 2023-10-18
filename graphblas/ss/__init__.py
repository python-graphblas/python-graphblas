from suitesparse_graphblas import burble

from ._core import _IS_SSGB7, about, concat, config, diag

if not _IS_SSGB7:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0
    from ..core.ss.context import Context, global_context

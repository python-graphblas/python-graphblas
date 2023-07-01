from ..core.ss import _IS_SSGB7

if not _IS_SSGB7:
    from ..core.ss.dtypes import register_new  # noqa: F401

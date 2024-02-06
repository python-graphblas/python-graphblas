from .. import backend
from ..core.matrix import Matrix
from ._scipy import to_scipy_sparse


def mmread(source, engine="auto", *, dup_op=None, name=None, **kwargs):
    """Create a GraphBLAS Matrix from the contents of a Matrix Market file.

    This uses `scipy.io.mmread
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html>`_
    or `fast_matrix_market.mmread
    <https://github.com/alugowski/fast_matrix_market/tree/main/python>`_.

    By default, ``fast_matrix_market`` will be used if available, because it
    is faster. Additional keyword arguments in ``**kwargs`` will be passed
    to the engine's ``mmread``. For example, ``parallelism=8`` will set the
    number of threads to use to 8 when using ``fast_matrix_market``.

    Parameters
    ----------
    source : str or file
        Filename (.mtx or .mtz.gz) or file-like object
    engine : {"auto", "scipy", "fmm", "fast_matrix_market"}, default "auto"
        How to read the matrix market file. "scipy" uses ``scipy.io.mmread``,
        "fmm" and "fast_matrix_market" uses ``fast_matrix_market.mmread``,
        and "auto" will use "fast_matrix_market" if available.
    dup_op : BinaryOp, optional
        Aggregation function for duplicate coordinates (if found)
    name : str, optional
        Name of resulting Matrix

    Returns
    -------
    :class:`~graphblas.Matrix`

    """
    try:
        # scipy is currently needed for *all* engines
        from scipy.io import mmread
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to read Matrix Market files") from None
    engine = engine.lower()
    if engine in {"auto", "fmm", "fast_matrix_market"}:
        try:
            from fast_matrix_market import mmread  # noqa: F811
        except ImportError:  # pragma: no cover (import)
            if engine != "auto":
                raise ImportError(
                    "fast_matrix_market is required to read Matrix Market files "
                    f'using the "{engine}" engine'
                ) from None
    elif engine != "scipy":
        raise ValueError(
            f'Bad engine value: {engine!r}. Must be "auto", "scipy", "fmm", or "fast_matrix_market"'
        )
    array = mmread(source, **kwargs)
    if getattr(array, "format", None) == "coo":
        nrows, ncols = array.shape
        return Matrix.from_coo(
            array.row, array.col, array.data, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    return Matrix.from_dense(array, name=name)


def mmwrite(
    target,
    matrix,
    engine="auto",
    *,
    comment="",
    field=None,
    precision=None,
    symmetry=None,
    **kwargs,
):
    """Write a Matrix Market file from the contents of a GraphBLAS Matrix.

    This uses `scipy.io.mmwrite
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html>`_.

    Parameters
    ----------
    target : str or file target
        Filename (.mtx) or file-like object opened for writing
    matrix : Matrix
        Matrix to be written
    engine : {"auto", "scipy", "fmm", "fast_matrix_market"}, default "auto"
        How to read the matrix market file. "scipy" uses ``scipy.io.mmwrite``,
        "fmm" and "fast_matrix_market" uses ``fast_matrix_market.mmwrite``,
        and "auto" will use "fast_matrix_market" if available.
    comment : str, optional
        Comments to be prepended to the Matrix Market file
    field : str
        {"real", "complex", "pattern", "integer"}
    precision : int, optional
        Number of digits to write for real or complex values
    symmetry : str, optional
        {"general", "symmetric", "skew-symmetric", "hermetian"}

    """
    try:
        # scipy is currently needed for *all* engines
        from scipy.io import mmwrite
    except ImportError:  # pragma: no cover (import)
        raise ImportError("scipy is required to write Matrix Market files") from None
    engine = engine.lower()
    if engine in {"auto", "fmm", "fast_matrix_market"}:
        try:
            from fast_matrix_market import __version__, mmwrite  # noqa: F811
        except ImportError:  # pragma: no cover (import)
            if engine != "auto":
                raise ImportError(
                    "fast_matrix_market is required to write Matrix Market files "
                    f'using the "{engine}" engine'
                ) from None
        else:
            import scipy as sp

            engine = "fast_matrix_market"
    elif engine != "scipy":
        raise ValueError(
            f'Bad engine value: {engine!r}. Must be "auto", "scipy", "fmm", or "fast_matrix_market"'
        )
    if backend == "suitesparse" and matrix.ss.format in {"fullr", "fullc"}:
        array = matrix.ss.export()["values"]
    else:
        array = to_scipy_sparse(matrix, format="coo")
        if engine == "fast_matrix_market" and __version__ < "1.7." and sp.__version__ > "1.11.":
            # 2023-06-25: scipy 1.11.0 added `sparray` and changed e.g. `ss.isspmatrix_coo`.
            # fast_matrix_market updated to handle this in version 1.7.0
            # Also, it looks like fast_matrix_market has special writers for csr and csc;
            # should we see if using those are faster?
            array = sp.sparse.coo_matrix(array)  # FLAKY COVERAGE
    mmwrite(
        target,
        array,
        comment=comment,
        field=field,
        precision=precision,
        symmetry=symmetry,
        **kwargs,
    )

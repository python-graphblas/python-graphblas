"""Tensor conversions to and from pydata/sparse (COO, GCXS, DOK) and scipy.sparse n-d arrays.

The Tensor methods ``from_pydata_sparse`` / ``to_pydata_sparse`` and ``from_scipy_sparse`` /
``to_scipy_sparse`` call these; they follow ``graphblas.io``'s functions of the same names.
"""

import itertools

import numpy as np

from .. import binary
from ..dtypes import BOOL, lookup_dtype
from . import tensor_layout as tl
from .tensor_layout import Layout


def _gcxs_states(shape):
    """The layout states GCXS can hold: both groups nonempty and in increasing order (its
    compressed axes, sorted, are the rows; the others, sorted, the columns).
    """
    ndim = len(shape)
    states = set()
    for p in range(1, ndim):
        for rows in itertools.combinations(range(ndim), p):
            seq = rows + tuple(a for a in range(ndim) if a not in rows)
            if tl._valid_split(shape, seq, p):
                states.add((seq, p))
    return frozenset(states)


def _unravel_into(idx, shape, axes, out):
    """``tl.unravel`` writing each axis of ``axes`` into its row of the 2-d array ``out``,
    with no copy past the one into ``out``.
    """
    if not axes:
        return
    if len(axes) == 1:
        out[axes[0]] = idx
        return
    inner = axes[1:]
    for i, a in enumerate(reversed(inner)):
        q = out[axes[0]] if i == len(inner) - 1 else np.empty_like(idx)
        np.divmod(idx, shape[a], out=(q, out[a]))
        idx = q


def _with_layout(cls, M, shape, lay, layout, name):
    """The Tensor of ``M`` held in ``lay``, moved to ``layout`` (and pinned) if one is given."""
    if layout is None:
        return cls._from_M(M, shape, lay, name=name)
    target = tl.as_layout(layout, shape)
    M, lay = tl.relayout(M, shape, lay, target)
    return cls._from_M(M, shape, lay, pinned=True, name=name)


def from_pydata_sparse(cls, s, *, dup_op=None, layout=None, name=None):
    try:
        import sparse
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to import from pydata sparse") from None
    if not isinstance(s, sparse.SparseArray):
        raise TypeError(
            "from_pydata_sparse only accepts objects from the `sparse` library; "
            "see https://sparse.pydata.org"
        )
    if not s.fill_value == 0:
        raise ValueError(
            f"from_pydata_sparse needs a fill value of 0; got {s.fill_value!r}. A Tensor has "
            "no fill value (a missing entry has no value), so another fill value would be "
            "lost. Convert ``s - s.fill_value`` (with a fill value of 0) instead."
        )
    dtype = lookup_dtype(s.dtype)
    shape = tuple(int(n) for n in s.shape)
    if isinstance(s, sparse.GCXS) and s.compressed_axes is not None:
        # GCXS is a CSR Matrix of the compressed axes (sorted) by the others (sorted): exactly
        # a Tensor layout, so its arrays import as they are (one copy each).
        rows = tuple(int(a) for a in s.compressed_axes)
        lay = Layout(rows, tuple(a for a in range(len(shape)) if a not in rows))
        if tl._valid_split(shape, lay.seq, len(rows)):
            from .matrix import Matrix

            M = Matrix.ss.import_csr(
                nrows=tl.prod(shape, lay.rows),
                ncols=tl.prod(shape, lay.cols),
                indptr=s.indptr,
                col_indices=s.indices,
                values=s.data,
                dtype=dtype,
                sorted_cols=False,  # (GCXS does not promise sorted columns)
            )
            return _with_layout(cls, M, shape, lay, layout, name)
    if not isinstance(s, sparse.COO):
        s = s.asformat("coo")
    return cls.from_coo(
        s.coords, s.data, dtype, shape=shape, layout=layout, dup_op=dup_op, name=name
    )


def to_pydata_sparse(T, format="coo", *, compressed_axes=None):
    try:
        import sparse
    except ImportError:  # pragma: no cover (import)
        raise ImportError("sparse is required to export to pydata sparse") from None
    format = format.lower()
    if format not in {"coo", "dok", "gcxs"}:
        raise ValueError(f"Invalid format: {format}")
    if compressed_axes is not None and format != "gcxs":
        raise ValueError("compressed_axes applies to the 'gcxs' format only")
    shape = T._shape
    if format == "gcxs" and T.ndim >= 2:
        if compressed_axes is None:
            targets = _gcxs_states(shape)
        else:
            from .tensor import _normalize_axis

            rows = tuple(sorted({_normalize_axis(a, T.ndim) for a in compressed_axes}))
            cols = tuple(a for a in range(T.ndim) if a not in rows)
            if not rows or not cols:
                raise ValueError(
                    f"compressed_axes must be some but not all of the {T.ndim} axes; "
                    f"got {compressed_axes!r}"
                )
            targets = tl.as_layout((rows, cols), shape)
        if targets:
            # (no move when the layout is one GCXS holds; else the cheapest move to one)
            M, lay = tl.relayout(T._M, shape, T._layout, targets)
            indptr, cols, values = M.to_csr()
            return sparse.GCXS(
                (values, cols, indptr), shape=shape, compressed_axes=lay.rows, fill_value=0
            )
    # pydata COO is in C order without duplicates: the sorted coordinates (a relayout when
    # the layout is not the identity axis order), unraveled into the (ndim, nvals) array
    M, lay = T._M, T._layout
    identity = tuple(range(T.ndim))
    if lay.seq != identity:
        targets = frozenset(
            (identity, p) for p in range(T.ndim + 1) if tl._valid_split(shape, identity, p)
        )
        M, lay = tl.relayout(M, shape, lay, targets)
    r, c, values = M.to_coo()
    indices = np.empty((T.ndim, values.size), np.int64)
    _unravel_into(r.view(np.int64), shape, lay.rows, indices)
    _unravel_into(c.view(np.int64), shape, lay.cols, indices)
    s = sparse.COO(indices, values, shape=shape, sorted=True, has_duplicates=False, fill_value=0)
    return s if format == "coo" else s.asformat(format)


def _scipy_coo_needs(ndim):
    """The oldest scipy whose ``coo_array`` holds ``ndim``-d arrays, or None if this scipy
    does: scipy.sparse is 2-d only before 1.13, has 1-d COO from 1.13 and n-d from 1.15.
    """
    import scipy

    need = (1, 13) if ndim == 1 else (1, 15) if ndim > 2 else (0,)
    if tuple(int(v) for v in scipy.__version__.split(".")[:2]) < need:
        return ".".join(map(str, need))
    return None


def from_scipy_sparse(cls, A, *, dup_op=None, layout=None, name=None):
    import scipy.sparse as ss

    if not ss.issparse(A):
        raise TypeError(f"from_scipy_sparse needs a scipy.sparse array or matrix; got {type(A)}")
    dtype = lookup_dtype(A.dtype)
    shape = tuple(int(n) for n in A.shape)
    if A.format in {"csr", "csc"} and A.ndim == 2 and A.has_canonical_format:
        from ..io import from_scipy_sparse as matrix_from_scipy_sparse

        # (CSR or CSC imported, one copy; a CSC Matrix is held as its transpose by row)
        M = matrix_from_scipy_sparse(A)
        if tl.is_by_row(M):
            lay = Layout((0,), (1,))
        else:
            M = tl.flip_orientation(M)
            lay = Layout((1,), (0,))
        return _with_layout(cls, M, shape, lay, layout, name)
    if A.format != "coo":
        A = A.tocoo()
    if dup_op is None:
        # As scipy: repeated coordinates add up
        dup_op = binary.lor if dtype == BOOL else binary.plus
    # (scipy before 1.13 has no ``coords``; 2-d COO has ``row`` and ``col`` in every version)
    coords = (A.row, A.col) if A.ndim == 2 else A.coords
    return cls.from_coo(coords, A.data, dtype, shape=shape, layout=layout, dup_op=dup_op, name=name)


def to_scipy_sparse(T, format="coo"):
    import scipy
    import scipy.sparse as ss

    format = format.lower()
    if T.ndim == 2 and format != "coo":
        from ..io import to_scipy_sparse as matrix_to_scipy_sparse

        if T._layout == Layout((0,), (1,)):
            A = T._M
        elif T._layout == Layout((1,), (0,)):
            A = T._M.T  # (its CSC is the stored CSR)
        else:
            A = T.to_matrix((0,), copy=False)
        return matrix_to_scipy_sparse(A, format)
    if format != "coo":
        raise ValueError(
            f"scipy.sparse holds {T.ndim}-d arrays as 'coo' only; got format {format!r}"
        )
    if T.ndim == 0:
        raise ValueError("scipy.sparse has no 0-d arrays")
    if need := _scipy_coo_needs(T.ndim):
        raise NotImplementedError(
            f"a {T.ndim}-d scipy.sparse array requires scipy {need} or later; "
            f"this is scipy {scipy.__version__}"
        )
    # scipy's COO takes any order: the stored order, in C order (canonical) only when the
    # layout is the identity axis order, so no relayout
    canonical = T._layout.seq == tuple(range(T.ndim))
    coords, values = T.to_coo(sort=canonical)
    A = ss.coo_array((values, coords), shape=T._shape)
    A.has_canonical_format = canonical
    return A

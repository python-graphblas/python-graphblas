from math import ceil, log2

import numpy as np

from .. import binary
from ..operator import get_semiring, get_typed_op
from .matrix import compact_indices


# By default, scans on matrices are done along rows.
# To perform scans along columns, pass a transposed matrix.
def prefix_scan(A, monoid, *, name=None, within):
    from .. import Matrix, Vector
    from ..matrix import TransposedMatrix

    monoid = get_typed_op(monoid, A.dtype, kind="binary")
    A._expect_op(monoid, ("BinaryOp", "Monoid"), argname="op", within=within)
    if monoid.opclass == "BinaryOp":
        if monoid.monoid is not None:
            monoid = monoid.monoid
        else:
            A._expect_op(monoid, "Monoid", argname="op", within=within)
    semiring = get_semiring(monoid, binary.first)
    binaryop = semiring.monoid.binaryop

    is_transposed = type(A) is TransposedMatrix
    N_orig = A.shape[-1]
    if N_orig < 2:
        if is_transposed:
            return A.T.dup()
        return A.dup()

    # Compactify all the elements
    is_vector = type(A) is Vector
    if is_vector:
        info = A.ss.export("sparse", sort=True)
        N_cols = len(info["indices"])
        compact_info = dict(info, indices=np.arange(N_cols, dtype=np.uint64), size=N_cols)
    elif is_transposed:
        info = A.T.ss.export("hypercsc", sort=True)
        _, row_indices, N_cols = compact_indices(info["indptr"], None)
        compact_info = dict(
            info,
            col_indices=row_indices,
            ncols=N_cols,
            nrows=info["ncols"],
            rows=info["cols"],
            format="hypercsr",
            sorted_cols=True,
        )
        del compact_info["cols"]
        del compact_info["row_indices"]
        del compact_info["sorted_rows"]
    else:
        info = A.ss.export("hypercsr", sort=True)
        _, col_indices, N_cols = compact_indices(info["indptr"], None)
        compact_info = dict(info, col_indices=col_indices, ncols=N_cols)

    if N_cols < 2:
        if is_transposed:
            return A.T.dup()
        return A.dup()
    N_half = N_cols // 2
    val_t = np.int8
    index_t = np.uint64
    index = 1
    if is_vector:
        A = Vector.ss.import_sparse(**compact_info)
    else:
        A = Matrix.ss.import_hypercsr(**compact_info)

    # First iteration
    S = Matrix.ss.import_csc(
        nrows=N_cols,
        ncols=N_half,
        indptr=np.arange(0, 2 * N_half + 2, 2, dtype=index_t),
        row_indices=np.arange(2 * N_half, dtype=np.uint64),
        values=np.ones(1, dtype=val_t),  # 2 * N_half
        is_iso=True,
        sorted_rows=True,
        take_ownership=True,
        name="Up_0",
    )
    B = semiring(A @ S).new(name="B")
    if is_vector:
        mask = None
    else:
        mask = B.S

    # Upsweep
    stride = 1
    stride2 = 2
    while stride2 <= N_half:
        k = (N_half - stride2) // stride2 + 1
        cols = np.arange(stride2 - 1, N_half, stride2, dtype=index_t)
        # assert k == cols.size
        S = Matrix.ss.import_hypercsc(
            nrows=N_half,
            ncols=N_half,
            indptr=np.arange(k + 1, dtype=index_t),
            cols=cols,
            row_indices=cols - stride,
            values=np.ones(1, dtype=val_t),  # k
            is_iso=True,
            sorted_rows=True,
            take_ownership=True,
            name=f"Up_{index}",
        )
        B(binaryop, mask=mask) << semiring(B @ S)
        index += 1
        stride = stride2
        stride2 *= 2

    # Downsweep
    index = 0
    if N_half > 2:
        stride2 = max(2, 2 ** ceil(log2(N_half // 2)))
        stride = stride2 // 2
        while stride > 0:
            k = (N_half - stride2 - stride) // stride2 + 1
            if k == 0:
                stride2 = stride
                stride //= 2
                continue
            cols = np.arange(stride2 + stride - 1, N_half, stride2, dtype=index_t)
            # assert k == cols.size
            S = Matrix.ss.import_hypercsc(
                nrows=N_half,
                ncols=N_half,
                indptr=np.arange(k + 1, dtype=index_t),
                cols=cols,
                row_indices=cols - stride,
                values=np.ones(1, dtype=val_t),  # k
                is_iso=True,
                sorted_rows=True,
                take_ownership=True,
                name=f"Down_{index}",
            )
            B(binaryop, mask=mask) << semiring(B @ S)
            index += 1
            stride2 = stride
            stride //= 2

    # Last iteration
    indptr = np.arange(0, 2 * N_half + 2, 2)
    indptr[-1] = N_cols - 1
    col_indices = np.arange(1, N_cols, dtype=index_t)
    S = Matrix.ss.import_csr(
        nrows=N_half,
        ncols=N_cols,
        indptr=indptr,
        col_indices=col_indices,
        values=np.ones(1, dtype=val_t),  # N_cols - 1
        is_iso=True,
        sorted_cols=True,
        take_ownership=True,
        name=f"Down_{index}",
    )
    RV = semiring(B @ S).new(mask=A.S, name="RV")

    indices = np.arange(0, N_cols, 2, dtype=index_t)
    d = Vector.ss.import_sparse(
        size=N_cols,
        indices=indices,
        values=np.ones(1, dtype=val_t),  # (N_cols + 1) // 2
        is_iso=True,
        sorted_index=True,
        take_ownership=True,
        name="d",
    )
    D = d.diag(name="D")
    RV(binaryop) << semiring(A @ D)
    # De-compactify into final result
    if is_vector:
        rv_info = RV.ss.export("sparse", sort=True, give_ownership=True)
        RV = Vector.ss.import_sparse(name=name, **dict(info, values=rv_info["values"]))
    elif is_transposed:
        rv_info = RV.ss.export("hypercsr", sort=True, give_ownership=True)
        RV = Matrix.ss.import_hypercsc(name=name, **dict(info, values=rv_info["values"]))
    else:
        rv_info = RV.ss.export("hypercsr", sort=True, give_ownership=True)
        RV = Matrix.ss.import_hypercsr(name=name, **dict(info, values=rv_info["values"]))
    return RV

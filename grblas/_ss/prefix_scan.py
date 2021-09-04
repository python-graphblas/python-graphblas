from math import ceil, log2

import numpy as np

import grblas as gb

from .. import binary


# TODO: make this smarter and move to grblas.operator
def get_semiring(monoid, binaryop):
    return gb.operator.Semiring.register_anonymous(monoid, binaryop)


# TODO: use iso values for S matrices
# By default, scans on matrices are done along rows.
# To perform scans along columns, pass a transposed matrix.
def prefix_scan(A, monoid):
    from .. import Matrix, Vector
    from ..matrix import TransposedMatrix

    is_transposed = type(A) is TransposedMatrix
    semiring = get_semiring(monoid, binary.first)
    if is_transposed:
        semiring2 = get_semiring(monoid, binary.second)
    binaryop = semiring.monoid.binaryop
    N_orig = A.shape[-1]
    if N_orig < 2:
        if is_transposed:
            return A.T.dup()
        return A.dup()

    # Which columns have data?
    if type(A) is Vector:
        # Can we export the indices w/o the values?
        nonempty_cols = A.ss.export("sparse", sort=True)["indices"]
    else:
        nonempty_cols = (
            A.reduce_columns(gb.monoid.any)
            .new()
            .ss.export("sparse", sort=True, give_ownership=True)["indices"]
        )
    N_cols = len(nonempty_cols)
    if N_cols < 2:
        if is_transposed:
            return A.T.dup()
        return A.dup()
    N_half = N_cols // 2
    val_t = np.int8
    index_t = np.uint64
    is_condensed = N_cols != N_orig
    index = 1

    # First iteration
    S = Matrix.ss.import_csc(
        nrows=N_orig,
        ncols=N_half,
        indptr=np.arange(0, 2 * N_half + 2, 2, dtype=index_t),
        row_indices=nonempty_cols[: 2 * N_half].copy(),
        values=np.ones(2 * N_half, dtype=val_t),
        sorted_rows=True,
        take_ownership=True,
        name="Up_0",
    )
    B = semiring(A @ S).new(name="B")

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
            values=np.ones(k, dtype=val_t),
            sorted_rows=True,
            take_ownership=True,
            name=f"Up_{index}",
        )
        index += 1
        B(binaryop) << semiring(B @ S)
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
                values=np.ones(k, dtype=val_t),
                sorted_rows=True,
                take_ownership=True,
                name=f"Down_{index}",
            )
            index += 1
            B(binaryop) << semiring(B @ S)
            stride2 = stride
            stride //= 2

    # Last iteration
    indptr = np.arange(0, 2 * N_half + 2, 2)
    indptr[-1] = N_cols - 1
    if is_condensed:
        col_indices = nonempty_cols[1:].copy()
    else:
        col_indices = np.arange(1, N_orig, dtype=index_t)
    S = Matrix.ss.import_csr(
        nrows=N_half,
        ncols=N_orig,
        indptr=indptr,
        col_indices=col_indices,
        values=np.ones(N_cols - 1, dtype=val_t),
        sorted_cols=True,
        take_ownership=True,
        name=f"Down_{index}",
    )
    if is_transposed:
        RV = semiring2(S.T @ B.T).new(mask=A.T.S, name="RV")
    else:
        RV = semiring(B @ S).new(mask=A.S, name="RV")

    if is_condensed:
        indices = nonempty_cols[::2].copy()
    else:
        indices = np.arange(0, N_cols, 2, dtype=index_t)
    d = Vector.ss.import_sparse(
        size=N_orig,
        indices=indices,
        values=np.ones((N_cols + 1) // 2, dtype=val_t),
        sorted_index=True,
        take_ownership=True,
        name="d",
    )
    D = gb.ss.diag(d, name="D")
    if is_transposed:
        RV(binaryop) << semiring2(D @ A.T)
    else:
        RV(binaryop) << semiring(A @ D)
    return RV

# Tensor: design decisions and why

The decisions behind the experimental `gb.Tensor`, for reviewers and contributors. Each one
says what was decided and why, and some say what would change it. "Maintainers" means the
python-graphblas maintainers. Status: everything here is experimental and may change before
the API settles. `design.md` describes how the code implements these.

The dated decisions (26 to 36) were made while the Tensor was written, without waiting for
review. Like the rest, they are open to review.

## Semantics

1. **GraphBLAS semantics, numpy shapes.** Values are partial: a position either holds a value
   or is missing. There is no fill value. Shapes, axes and broadcasting follow numpy and the
   [Array API](https://data-apis.org/array-api/). Element-wise ops follow GraphBLAS:
   `ewise_mult` is the intersection, `ewise_add` the union, and `ewise_union` the union with
   defaults. Reductions use monoids and have no initial value, so an empty fiber stays
   missing. _Why:_ this is what a GraphBLAS user expects, and it generalizes Matrix and
   Vector exactly. A fill-value model, as in [pydata/sparse](https://github.com/pydata/sparse),
   clashes with semirings and masks.
2. **Results are always Tensors.** A 2-d result is a Tensor, not a Matrix; a full reduction
   (`reduce(axis=None)`) is a 0-d Tensor, while `reduce_scalar` gives a Scalar. Use
   `to_matrix()`, `to_vector()` or `to_scalar()` to convert. _Why:_ the type of a result
   should not depend on its rank, which would break generic code.
3. **`T[...]` is outer (orthogonal) indexing,** as in python-graphblas. Vectorized gathers are
   separate and explicit: `take`, `take_along_axis` and `T.vindex[...]`. _Why:_ one indexing
   rule in `T[...]`, with no behavior change from Matrix. The Array API (since its 2024.12
   revision) asks for vectorized semantics when every key is an integer or an integer array;
   that belongs to an Array API namespace over the Tensor (#667), not to `T[...]`.
4. **Negative indices wrap once,** as in numpy and python-graphblas. `take`,
   `take_along_axis` and `T.vindex(...)` accept `wraparound=False`, which raises IndexError
   on a negative index instead. Out-of-range indices raise; `take` also has numpy's
   `mode="wrap"` and `mode="clip"`.
5. **Index assignment uses GraphBLAS assign semantics,** with the value broadcast to the
   region's shape as numpy does. The region takes the value's structure: positions in the
   region that are missing from the value are deleted, as in Matrix `C[I, J] << A`. A mask
   applies to the whole Tensor in `T(mask)[idx] << v` (assign) and to the region in
   `T[idx](mask) << v` (subassign), as for Matrix. With repeated indices, the last one wins,
   as for Matrix.
6. **Index arrays are integers, slices, lists, ndarrays and Scalars,** not Vectors or Tensors.
   python-graphblas refuses a Vector as a Matrix index today, and sparse index arrays have no
   agreed meaning yet. `None` (a new axis) and boolean indices are refused too; use
   `expand_dims` for the first.
7. **Union broadcasting is numpy's, with a guard.** A broadcast `ewise_add` can create far more
   entries than its operands have. The result size is predicted in O(1). The operation
   raises above `graphblas.config["tensor_max_expansion"]` (default 1e9 entries) unless
   `allow_expansion=True` is passed. Broadcast `ewise_mult` never replicates the operand
   that covers the result.
8. **Shape stays logical.** Internal engines may hold blocks of dense values (array-valued
   entries), but a Tensor never exposes them as an array dtype.
9. **Masks are Tensor masks of the output's exact shape.** A mask of a Matrix or Vector of
   that shape is accepted and treated as a Tensor mask. Masks do not broadcast. Masks combine
   across kinds (`T.S & A.V`).
10. **Gathers compute at once; indexing is lazy.** `take`, `take_along_axis` and `vindex`
    return a Tensor, while `T[...]` returns an expression, as for Matrix. _Why:_ a gather's
    result shape and cost depend on the index arrays, and nothing would be gained by
    delaying it. A mask can still be applied with `C(mask) << R`.

## Representation

11. **One SuiteSparse Matrix, held by row, plus a layout.** The layout is an ordered
    partition of the logical axes into a row group and a column group, each raveled in C
    order. `permute_dims`, `.T`, `expand_dims`, `squeeze` and most `reshape`s are O(1)
    changes to the layout. Views share storage with copy-on-write, so a Tensor never sees
    another's writes. _Why:_ it reuses every SuiteSparse kernel (hypersparse storage, parallel
    transpose, reshape, build and mxm) and needs no new SuiteSparse object. See `design.md`.
12. **Relayouts happen only inside an operation that needs them,** chosen by a planner as the
    cheapest sequence of SuiteSparse steps (reshape, transpose, row or column permutation,
    rebuild), much like an LSD radix sort. A layout the user sets explicitly (constructor
    `layout=`, `relayout`, `dup(layout=)`) is pinned: `<<` into that Tensor keeps it.
13. **N-d from day one,** tuned for 3-d and then 4-d. Each axis group's raveled size must fit
    in 2^60.

## Engines and dependencies

14. **SuiteSparse:GraphBLAS 10 or later** is required, for the Container API, which gives
    O(1) zero-copy load and unload. The rest of python-graphblas keeps its floor (9.3.1). Every
    way to make a Tensor raises one clear error on an older SuiteSparse or another backend,
    and the Tensor test modules skip there.
15. **Numba where it pays, never required.** MTTKRP, TTM and point gathers have Numba kernels
    over the stored arrays. Every operation also has a pure-GraphBLAS path, and the Tensor
    tests pass with Numba unavailable (see `verification.md`).
16. **The SuiteSparse JIT is assumed to be on for speed,** not for correctness. Every path is
    correct with the JIT off. The array-UDT TTM engine needs the JIT. When it runs, it raises
    `jit_c_control` where python-graphblas's UDT auto-lift does, through the same function
    (`jit_config._enable_jit_for_udt`): SuiteSparse's default "run" becomes "on", and every
    other setting and a "run" the user set are left alone. So is the default where
    SuiteSparse compiles through cmake (Windows); that rule comes from #659, a pull request
    below this one in the stack. A case it declines leaves the control alone. If a compile
    fails, it leaves the control at "load", where SuiteSparse puts it after a failed
    compile, warns once, and turns itself off for the process. _Why "load":_ auto-lift never
    raises "load", so nothing in python-graphblas retries the failed compile; a control put
    back to the default "run" would be raised to "on" again by the next UDT operation, which
    would retry the compile and raise JitError from an unrelated Matrix operation.
17. **Alternative engines are kept only where measurements justify them,** and every engine
    stays selectable so that tests and benchmarks can force each one. A census of every
    decision point, over the test suite, benchmark rows and real workloads, decides what
    stays (`design.md`, "Engine selection and the census").
18. **Generated Numba kernels are cached on disk** under `$GRAPHBLAS_KERNEL_CACHE`, by default
    `python-graphblas/tensor-kernels` under `$XDG_CACHE_HOME` or `$HOME/.cache`; an empty
    value turns it off. A cached file runs only if it matches the generated source exactly.
    _Why:_ a kernel takes 0.2-0.6 s to compile, every process. The maintainers accepted the
    default location.
19. **opt_einsum is optional.** Without it, `einsum` uses a greedy pairwise order.

## API surface

20. **`gb.Tensor` and `gb.einsum` are exported, marked experimental.** They load lazily, so
    `import graphblas` does not load any Tensor code.
21. **Interop follows Tensor rules.** When a Matrix, Vector, Scalar or ndarray meets a Tensor
    in an infix expression, the result is a Tensor computed by Tensor rules. Matrix methods
    called with a Tensor operand raise with a hint, because only infix defers. The exceptions
    are `isequal` and `isclose`, which defer to the Tensor, so `A.isequal(T)` and
    `T.isequal(A)` agree. In-place operators on a Matrix with a Tensor operand raise rather
    than rebind the name.
22. **Tensor may go beyond Matrix and Vector.** It is a new type and can adopt better behavior.
    Features that Matrix and Vector could also have are listed in `backport-candidates.md`.
23. **Core changes come in their own commits,** when they are justified and easy to review:
    the interop hooks. Two that the Tensor needs went to main in their own pull request
    (#652): the JIT C definitions of lifted ops between a scalar and an array or record UDT,
    and the Windows compiler search in `jit_config`.
24. **`Tensor.wait()` materializes, as `Matrix.wait()` does,** including the hash SuiteSparse
    builds of a hypersparse Matrix's rows (when its hyperlist is longer than the
    `GxB_HYPER_HASH` setting, 1024 by default). _Why:_ the Tensor's own row lookups
    (extracts, gathers, `lookup`) use that hash; SuiteSparse documents `GrB_MATERIALIZE` as
    finishing all work on an object, after which several user threads can use it as an input
    at the same time (`GraphBLAS.h`, `GrB_WaitMode`); and a Tensor should not differ from
    Matrix silently. `wait("complete")` skips it, and with it everything else (SuiteSparse
    does no work for "complete"), and most operations on the Tensor finish their operand
    first, so the next one pays it. Measured on a
    TTM of NELL-2 (R = 16, 2026-10-28): the hash takes 0.68-0.97 s at 1 thread and 86-116 ms
    at 8 for modes 0 and 1, about 40% and 55% of the TTM at 1 thread and 24% and 37% at 8,
    and it is all the result's wait does (no pending work). After `wait("complete")`, a row
    extract that took 1-3 ms took about the hash (0.68-0.96 s at 1 thread, 89-117 ms at 8),
    and `to_coo` and a sum took about the hash longer. The benchmark kit times
    `wait()`, so its TTM numbers include the hash (`benchmarks.md`).
25. **No per-mode copies, for now.** The Tensor keeps one stored Matrix, and a contraction
    moves an operand stored in another order inside the call: TTM along a mode other than
    the stored order relayouts the tensor each time, where TACO and SPLATT keep a copy of
    the data per mode, built ahead. _Why not:_ memory, a full copy per mode (about 0.9 GB
    per mode for NELL-2); and keeping copies coherent with copy-on-write views and in-place
    updates would add real complexity to an experimental API. An explicit `relayout()`
    before a loop already gives a user that copy, and the user guide recommends it. With it,
    NELL-2's TTM along modes 0 and 1 takes 445 and 548 ms at 1 thread (TACO: 263 and 262
    from its own copy), plus the output's hash in `wait()` (`benchmarks.md`). _Checked_
    on such sweeps by the SAPS pilot (2026-10-28: HOOI and CP-ALS on FROSTT's Uber, Chicago
    and NIPS; `limitations-and-roadmap.md` says what it was), and kept. Copies built once
    saved 1-19% on HOOI: only Uber's first-axis TTM chain gained much (173 to 156 ms per
    sweep; the 45 ms build repaid in about 1.3 sweeps; a full SAPS run 8.97 to 7.26 s), and
    at 1 thread Chicago and NIPS gained 1.4% and 1.1%. They made
    CP-ALS 1.6-6x slower per sweep, because the Numba MTTKRP runs in the stored layout for
    every mode (Uber 7.2 to 11.6 ms, Chicago 7.7 to 48.9, NIPS 9.7 to 41.1). Where copies
    help, `relayout()` already gives them (six lines in the pilot's HOOI). _Revisit_ if a
    TTM-chain (TTMc) kernel turns out to need each mode's data stored mode first.
26. **Broadcast unions whose broadcast axes would fit only split across the row and column
    groups are not supported.** A union (`ewise_add`, `ewise_union`) replicates the
    lower-rank operand over the broadcast axes (`_expand`): before its row group, or, when
    that passes 2^60 positions, after its column group. When neither fits but a split of the
    broadcast axes across both groups would, it raises a clear ValueError ("... more than one
    Matrix dimension can hold (2**60) next to its own axes"). _Why not:_ splitting them needs
    a kron on both sides and a layout choice in `_expand`, for a rare case. It takes 2^60
    positions or more in total, a replication that `tensor_max_expansion` lets through (or
    `allow_expansion=True`), and an operand whose own groups crowd both sides. An
    intersection never replicates and has no such limit. Decided 2026-10-28. _Revisit_ when a
    workload meets the error.
27. **`X | Y` and `X & Y` on their own follow Matrix:** lor and land of BOOL Tensors, and a
    TypeError ("Bad dtypes for `x | y`", Matrix's message) for other dtypes, whichever way
    the expression is used: `.new()`, `C << (X | Y)`, a region's value, an operand,
    autocompute. With an operator (`binary.plus(X | Y)`, `monoid.max(X & Y)`) every dtype
    works, as for Matrix. They used to compute the default operator, plus or times, for
    other dtypes. _Why:_ the user guide says infix operators work "as for Matrix"; with
    numpy, `|` on integers is bitwise or and on floats raises, so quietly computing plus
    would surprise numpy users; `+` and `*` already give plus and times; and nothing
    recorded a reason for the old default. A behavior change to the experimental API,
    toward the parity it claims. Decided 2026-10-28. _Revisit_ if users ask for a default
    operator.
28. **`select` and `apply` take a 0-d Tensor as their scalar operand** (`thunk`, `left`,
    `right`), converted with `to_scalar()` when the expression is built. _Why:_ 0-d is the
    Tensor's scalar (2), a full `reduce` returns one, and `ewise_mult` already broadcasts
    one, so a threshold that came from a reduction needed a `to_scalar()` that no other
    operand needs. An empty 0-d Tensor raises `EmptyObject`, as an empty Scalar does: these
    operators need a value, while `ewise_mult` works by position and gives an empty result.
    The value is read when the expression is built, where a Scalar operand is read when it
    computes; only a 0-d Tensor changed between the two would show it. The narrower
    alternative was to leave the signature alone and make the refusal name `Tensor.select`
    and say `.to_scalar()`. Decided 2026-11-08. _Revisit_ if a Tensor
    operand of another rank is proposed here (it stays refused).
29. **A float fiber's NaN or -0.0 is not replaced by a monoid's identity,** except where a
    reduction hands the work to a Matrix method, which it then matches. SuiteSparse 10.5.1
    and 10.5.2, in their factory and JIT kernels, start a full column of a saxpy product from
    the monoid's identity and add each term (`GB_COMPUTE_DENSE_C_j`), and the identity is not
    exact for every float: under min a NaN came back +inf, under plus a -0.0 came back +0.0.
    Two kinds of Tensor path met it and were changed. Row permutations of float and complex
    values (relayout, `to_coo`, row picks, selector expansions) use `any_second` where they
    used `min_second` or `plus_second`, and a broadcast's diagonal product of floats and
    complex uses `any` where it used plus (integers keep plus, which is exact for them).
    `reduce` over part of an axis group, whose monoid is the caller's, pads its selector
    product with one column that nothing fills, so no output row is full (sparse and
    hypersparse data; float, complex and user monoids). _Cost,_ without the JIT where a build
    lacks the `any` kernels: the permutations up to 2x for FP64 on 10.5.1 (1e7 entries in 1e6
    sparse rows; 1.1-1.25x in hypersparse rows) and 1.5x for complex on conda-forge's 10.5.2,
    and the diagonal product 0-15%; and in every build, the padded reduce 0.92-1.23x, worst
    under plus over wide sparse rows. _Where Matrix's result is kept:_ a whole column group
    goes through `Matrix.reduce_rowwise`, and full data under plus through a dot product, and
    SuiteSparse's dot kernels fold plus from 0 (`GB_AxB_dot_cij.c`), so a fiber of two or
    more -0.0 sums to +0.0 there (on sparse, hypersparse and full data; bitmap keeps -0.0),
    as it does for Matrix. _Why not that too:_ a copy and a format conversion on every dense
    sum over row axes, for the sign of a zero, and a Tensor that disagreed with Matrix there
    would be a second surprise. Decided 2026-11-10. _Revisit_ if
    SuiteSparse's dot kernels or Matrix change.
30. **A reduction over every axis returns what `Matrix.reduce_scalar` returns,** including
    on bitmap data, where SuiteSparse 10.5.1 and 10.5.2 start from the monoid's identity
    (`GB_reduce_to_scalar.c`): entries all NaN reduce to +inf under min (-inf under max),
    and all -0.0 to +0.0 under plus. Sparse, hypersparse and full data keep the NaN. _Why
    not work around it in the Tensor:_ the behavior is Matrix's, and a fix belongs in Matrix
    or SuiteSparse, where it fixes both; a Tensor-only fix would make `T.reduce()` and
    `T.to_matrix().reduce_scalar()` disagree. Decided 2026-11-10.
    _Revisit_ when Matrix or SuiteSparse changes.
31. **An index out of range raises what the same Matrix call raises,** with the Tensor's own
    message naming the axis and its size. An int out of range, or any index below minus the
    axis length, raises `IndexError` when `T[key]` is written. Otherwise the error comes when
    the extraction is computed or assigned to, as GraphBLAS reports it for Matrix:
    `InvalidIndex` for assigning to one element or extracting with a Scalar out of range,
    else `IndexOutOfBound`. `take`, `take_along_axis` and `vindex` follow numpy and raise
    `IndexError` (3). Outer indexing used to raise `IndexOutOfBound` for every kind of index,
    when `T[key]` was written. _Why:_ `IndexOutOfBound` is not an `IndexError`, so
    `except IndexError` around `T[i]` missed it, and code written against Matrix's exceptions
    should work on a Tensor. Matrix's choice depends on the call (one element or a region,
    get or set); following it call by call leaves one rule to learn. Going beyond Matrix (22)
    is for better behavior, and a second set of names for the same mistakes is not. Decided
    2026-11-10. _Revisit_ if Matrix's exceptions change; a test compares
    28 keys under five operations (and on a Vector and a 3-d Tensor) against the same Matrix
    call.
32. **Composite aggregators sum integer and BOOL values in their float result type.**
    `T.reduce(agg.mean)` (and `varp`, `stds`, `geometric_mean` and the rest) copies integer and
    BOOL values into the type Matrix gives the result (FP64 for the mean of integers), then
    computes each part with its own aggregator (`agg.count`, `agg.sum`, ...), as
    `T.reduce(agg.count)` would, and combines them with Matrix's finalize function
    (`sum/count`), stored in that type. So the count is INT64 and the sums, squares and
    products are FP64, as numpy accumulates a mean or a variance. Matrix's composite reductions
    type each part by the values' type instead, so INT8 values are counted, summed and squared
    in INT8, which wraps: the mean of [100, 100] is -28 there and the sample variance of [3, 3,
    3, 3] is 1.42, and 100 and 0 here. BOOL means and variances work here (numpy's 2/3 for the
    mean of [True, False, True]) where Matrix's raise KeyError from the inner sum;
    `harmonic_mean` does not declare BOOL and raises, as for Matrix. `peak_to_peak` keeps the
    values' type, which its result has. _Why not follow Matrix:_ those are wrong values, and
    their fix belongs in Matrix (recorded for it). _Cost:_ a copy of integer values in FP64;
    and FP32 results can differ from Matrix's by FP32 rounding, since Matrix also counts,
    divides and squares in FP32 (the variance of one value is 0 here, about 1e-7 there). For
    complex values the variances are Matrix's E[x^2] - E[x]^2 with the complex square, not
    numpy's real variance (limitations-and-roadmap.md). Decided 2026-11-22. _Revisit_ when
    Matrix's composite aggregators change.
33. **Aggregators of positions follow Matrix's rows, along one axis.** `agg.argmin`,
    `agg.argmax`, `agg.first_index` and `agg.last_index` reduce one axis, as
    `reduce_rowwise` reduces a row and numpy's `argmin` an axis, and give an index along it
    (INT64); more axes, or `axis=None` on more than one axis, raise ValueError, as
    `Matrix.reduce_scalar` does for them. `argmin` is the first index whose value equals the
    fiber's minimum by `eq`, so a fiber whose minimum is NaN has no entry, as for Matrix.
    `agg.first` and `agg.last` take any axes, in C order over them, which is Matrix's
    `reduce_scalar` (row-major) for every axis of a 2-d Tensor. _Why not numpy's flat index
    for `argmin(axis=None)`:_ it is an index into a dense array that a sparse Tensor does
    not have, Matrix refuses it, and `T.ravel().reduce(agg.argmin)` gives it. Decided
    2026-11-22. _Revisit_ if a user needs an index over several axes.
34. **`unfold` takes `traversal_order=`, not `order=`.** Its "C" or "F" is the order in which
    the columns traverse the other axes, numpy's `reshape(order=)` sense. python-graphblas
    reads "C" and "F" as aliases of "rowwise" and "columnwise" wherever it takes `order=`
    (`utils.get_order`): the orientation of `Matrix.from_dicts` and `to_dicts`, and the
    traversal of `M.ss.reshape` and `M.ss.flatten`. The matrix-orientation pull request (#495,
    open) proposes `order=` for the storage orientation of a new Matrix on `Matrix()`, `dup`,
    `from_coo`, `from_dense` and other methods that make one. `unfold` returns a Matrix, so
    with that, `unfold(order="F")` would read as asking for a columnwise result. _Roads not
    taken:_ `order=`, for that clash; `index_order=`, numpy's own documentation term, but in a
    sparse library it suggests the order of stored indices; `ravel_order=`, the Tensor's word
    for how a group's axes combine, a close second; `convention="tensorly"` or `"kolda"`, which
    names a source rather than the effect. TensorLy's `unfold(tensor, mode)` has no such
    keyword (C order only), and pydata/sparse has no `unfold`. Decided 2026-11-22: the
    maintainer kept `order=` for #495 and suggested `traversal_order`.
35. **Positional operators raise wherever a Tensor takes an operator.** `apply`, `select`,
    `ewise_add`, `ewise_mult`, `ewise_union`, `tensordot`, `matmul`, `mttkrp`, `einsum` and an
    accumulator refuse an operator that reads an entry's indices (`indexunary.rowindex`,
    `colindex`, `diagindex`, `select.tril` and the other index selectors, `binary.ss.firsti`
    and the other positional binary ops, `semiring.ss.min_secondi` and the like) with
    NotImplementedError. SuiteSparse would hand them the stored Matrix's indices, which depend
    on the layout: in layout (1|0), `apply(indexunary.rowindex)` returned each entry's column.
    Only the built-in value ops (`valueeq`, ..., `valuele`) pass: a user's index unary or
    select op gets the indices whatever its name, so it is refused too, even one that reads
    only the value, since nothing tells the two apart. _Road not taken:_ logical coordinates,
    as numpy's `tril` applies to the last two axes; that needs a meaning for "row" and "column"
    of an n-d Tensor, for the maintainers to choose. Decided 2026-11-22. _Revisit_ with that
    design.
36. **`index in T`, `get` and iteration follow Matrix.** `(i, j, k) in T` says whether an entry
    is stored there (an int for a 1-d Tensor, as for a Vector, or a 1-tuple; an integer Scalar
    is an index and a bool is not, as for Matrix), and `iter(T)` gives the stored entries'
    coordinates in C order, as `iter(A)` gives a Matrix's (row, col); a 1-d Tensor gives
    1-tuples, where a Vector gives ints, so every Tensor gives tuples of `ndim` ints. Without
    them Python fell back to `__getitem__`, so `in` compared the sub-Tensors along axis 0 with
    the index and said False for a stored entry, and iteration made one sub-Tensor per position
    of axis 0, up to 2^60. _Road not taken:_ numpy's iteration over the first axis; it lists
    every position, stored or not, which a sparse Tensor should not, and Matrix and Vector
    iterate over their entries. `T.get(i, j, k, default=None)` returns a stored value as
    `Matrix.get(row, col)` does, with one index argument per axis (or one tuple) and `default`
    by keyword. `len(T)` stays undefined, as for Matrix. `in` ravels the index into the stored
    Matrix's (row, col) and asks it: 1.3 to 1.6 us on a 3 x 4 or 2 x 3 x 4 Tensor, against 0.4
    us for Matrix and 20 to 25 us through an extract (a loaded machine, interleaved runs).
    Decided 2026-11-22.

## Deferred, on purpose

- An Array API namespace: a separate namespace, `graphblas.array_api`, with fill values as
  in pydata/sparse, in its own pull request (#667).
- [binsparse](https://github.com/GraphBLAS/binsparse-specification) interchange: when the
  `__binsparse__` protocol settles.
- `graphblas.io` returning a Tensor for ndim > 2: depends on how else the Tensor integrates.
  For now the conversions are Tensor methods.
- Vector and Tensor index arrays, sparse ones included: undecided.
- Per-mode copies, for workloads that contract every mode, as
  [TACO](https://github.com/tensor-compiler/taco) and
  [SPLATT](https://github.com/ShadenSmith/splatt) keep: not now (25), only opt-in if ever.
- A storage form with dense trailing dimensions (one index per fiber, a dense block of
  values): only if a workload needs it beyond what the TTM engines already give.

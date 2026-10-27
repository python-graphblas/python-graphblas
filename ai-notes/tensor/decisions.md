# Tensor: design decisions and why

The decisions behind the experimental `gb.Tensor`, for reviewers and contributors. Each one
says what was decided and why, and some say what would change it. "Maintainers" means the
python-graphblas maintainers. Status: everything here is experimental and may change before
the API settles. `design.md` describes how the code implements these.

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
   rule in `T[...]`, with no behavior change from Matrix. The Array API 2024.12 asks for
   vectorized semantics when every key is an integer or an integer array; we will revisit
   that with an Array API namespace.
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
    O(1) zero-copy load and unload. The rest of python-graphblas keeps its older floor. Every
    way to make a Tensor raises one clear error on an older SuiteSparse or another backend,
    and the Tensor test modules skip there.
15. **Numba where it pays, never required.** MTTKRP, TTM and point gathers have Numba kernels
    over the stored arrays. Every operation also has a pure-GraphBLAS path, and the Tensor
    tests pass with Numba unavailable (see `verification.md`).
16. **The SuiteSparse JIT is assumed to be on for speed,** not for correctness. Every path is
    correct with the JIT off. The array-UDT TTM engine needs the JIT. When it runs, it raises
    `jit_c_control` from SuiteSparse's default "run" to "on", as python-graphblas's UDT
    auto-lift does; a case it declines leaves the control alone. If a compile fails, it
    restores the previous setting, warns once, and turns itself off for the process. _Why
    restore:_ the engine changed a user-visible setting itself, so it puts it back.
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
    the interop hooks, the mixed scalar x array-UDT JIT C definitions, and the Windows
    compiler search in `jit_config`.

## Deferred, on purpose

- An Array API namespace: later, as a separate namespace, perhaps with implicit-zero
  semantics.
- [binsparse](https://github.com/GraphBLAS/binsparse-specification) interchange: when the
  `__binsparse__` protocol settles.
- `graphblas.io` returning a Tensor for ndim > 2: depends on how else the Tensor integrates.
  For now the conversions are Tensor methods.
- Vector and Tensor index arrays, sparse ones included: undecided.
- Opt-in per-mode copies, for workloads that contract every mode, as
  [TACO](https://github.com/tensor-compiler/taco) and
  [SPLATT](https://github.com/ShadenSmith/splatt) do: only opt-in, if ever.
- A storage form with dense trailing dimensions (one index per fiber, a dense block of
  values): only if a workload needs it beyond what the TTM engines already give.

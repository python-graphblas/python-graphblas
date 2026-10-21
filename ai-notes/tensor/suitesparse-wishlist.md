# Tensor: SuiteSparse:GraphBLAS wishlist

The [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) features that
would simplify the experimental `gb.Tensor` or make it faster, in priority order, each with the
measurements behind it, a proposed API, how it could be built from SuiteSparse's existing
internals, and who else would use it. Some of what it once asked for turned out to be possible
already; that is in "Already possible today". It is written for the python-graphblas
maintainers and for SuiteSparse's author; nothing here has been filed upstream.

The Tensor stores an n-d sparse array as one SuiteSparse Matrix whose rows and columns each
ravel an ordered group of axes (`design.md`, "Data model"). Almost every item below is about
moving entries between such layouts, which the Tensor does today by composing reshapes,
transposes, products with permutation or selector matrices, and rebuilds (`design.md`, "The
relayout planner").

## How to read this page

- **Tags.** [source] was read in the SuiteSparse sources at the cited path and line;
  [measured] was timed; [inferred] is an estimate or a reading, not checked.
- **Machine and library.** Every measurement was made on an Apple M5 Pro (6 performance and
  12 efficiency cores; 8 threads span 6 P and 2 E cores) with conda-forge SuiteSparse:GraphBLAS
  10.5.1, best of several runs. Times are written "8 / 1" for 8 and 1 threads; other thread
  counts are named. The JIT was at SuiteSparse's default `"run"` unless noted.
- **Sources.** Paths are in the GraphBLAS repository (`Source/...`, `Include/GraphBLAS.h`, the
  User Guide in `Doc/UserGuide/`). Line numbers are from the v10.5.2 sources (v10.5.2-beta2);
  they are the same in 10.5.1 except in `Source/reshape/GB_reshape.c`, as noted where it is
  cited.
- **Spec issues** are those of the GraphBLAS C API repository, `GraphBLAS/graphblas-api-c`, as
  of October 2026. LAGraph paths and lines are from its `stable` branch.
- **Data sets.**

  | name   | what it is                                                        |
  | ------ | ----------------------------------------------------------------- |
  | u7     | uniform random 1000^3 tensor with 1e7 entries                     |
  | skew   | uniform random (100000, 1000, 10) with 1e7 entries                |
  | hyper  | uniform random (2^19)^3 with 1e6 entries                          |
  | Zipf   | 100000^3 with 1e7 entries, coordinates Zipf-distributed (a = 1.5) |
  | NELL-2 | the FROSTT tensor, 12092 x 9184 x 28818 with 76.9M entries        |
  | Uber   | the FROSTT tensor, 4-d, 183 x 24 x 1140 x 1717 with 3.3M entries  |
  | Enron  | the FROSTT tensor, 4-d, 54M entries                               |

  FROSTT tensors are at [frostt.io](http://frostt.io/). "torch" is PyTorch's sparse COO
  tensor, at 8 threads.

## Summary

| #   | item                                                                                                 | GxB or GrB          | exists today                                                             | expected gain                                                                      | effort |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------ |
| P1  | **Axis-map reindex**: relayout, reshape, reduce dropped axes, embed or select diagonals, in one call | GxB                 | pieces: reshape (below 2^60 total), transpose, `P @ M`, selectors, build | 1-3x on uniform 3-d relayouts, 2-4x where the Tensor rebuilds; no 2^60 total limit | L      |
| P2  | **Radix sort in the builder** (`GrB_Matrix_build`, `GrB_wait`, the builder-based transpose)          | GxB (internal)      | a parallel merge sort; sorted input skips it                             | unsorted builds are 9x (4 threads) to 21x (1 thread) slower than presorted ones    | M      |
| P3  | **Transpose and column reductions into many rows** at 4-8 threads (a performance bug; an MRE exists) | GxB (internal)      | an atomic method chosen by a table tuned on a 4-core machine             | 2-7x at 8 threads; a two-pass radix bucket 2-3x more [inferred]                    | M      |
| P4  | **Factory kernels for `ANY_FIRST` / `ANY_SECOND`**, off since v10.0 (a regression; an MRE exists)    | GxB (internal)      | `MIN_SECOND` / `MAX_SECOND` are factory kernels and keep iso values      | 2.8x on a permutation product without the JIT                                      | S      |
| P5  | **`GxB_Matrix_reshape` past 2^60 total size**                                                        | GxB                 | refused with `GrB_OUT_OF_MEMORY`                                         | a split move at reshape speed instead of a rebuild (about 0.5 vs 6.5 ns per entry) | S      |
| P6  | **Broadcast element-wise**: `A .* v` along rows or columns, and a gathered `v(g(i,j))`               | GrB (spec #72, #24) | `GrB_Matrix_diag` plus `GrB_mxm`, fast only for a full diagonal          | 7x for a sparse `v`; parity with torch where the Tensor is at 0.13-0.54x           | M      |
| P7  | **Relabel by vector maps** with a dup operator: permute, contract, filter and renumber               | GxB; spec later     | one-to-one maps (`GrB_extract`, `GrB_assign`); many-to-one as `S A S'`   | 2-4x on graph contraction [inferred]                                               | M      |
| P8  | **Zero-copy load and unload in the C API spec** (the Container)                                      | GrB (spec #31)      | the GxB Container, O(1); the spec's import and export copy               | O(1) instead of O(nnz) copies, in every binding                                    | M      |
| P9  | **Array-valued entries**: works today; small follow-ups                                              | GxB                 | a fixed-size array UDT with JIT C definitions                            | the Tensor's fastest TTM without Numba already uses it                             | S      |
| P10 | **No n-d object in the spec yet**                                                                    | GrB                 |                                                                          |                                                                                    |        |

P1, P2, P3 and P7 can share one new internal kernel, described next. P3 and P4 are performance
bugs or regressions in released versions, each with a minimal reproducer (an MRE) that the
maintainers can provide.

## A shared kernel: the stable digit pass

Three measurements point to one kernel.

- **A scatter into many buckets stops fitting in cache.** A materialized transpose of a matrix
  with 1e7 uniform entries costs, per entry, 0.52 / 1.82 ns into 1e3 output rows, 2.2 / 6.2 into
  1e5, and 8-16 / 10-15 into 3e5 rows or more [measured]. So the Tensor relayouts like an LSD
  radix sort, as a chain of reshapes and of transposes into one axis's worth of rows. On u7,
  `(k | i,j) -> (i,j | k)` takes 112 / 110 ms as one transpose and 34 / 136 ms as the chain
  [measured].
- **A rebuild's cost is the builder's sort.** On u7, `(i,j | k) -> (k,i | j)` by
  `to_coo`, numpy index arithmetic and `from_coo` takes 249 ms at 4 threads: `to_coo` 8.0,
  numpy 32.7, `from_coo` 211. The same tuples, presorted, build in 22.9 ms, and the Tensor's
  planned chain does the whole relayout in 12.8 ms. At 1 thread: 858 ms in all, `from_coo` 807,
  presorted 37.5, the chain 37.7 [measured]. The builder sorts with a parallel merge sort
  (`GB_msort_2` and `GB_msort_3`, `Source/builder/GB_builder.c:617-709`) [source]; nothing in
  `Source/` is a radix sort [source].
- **Sorted input is cheap.** The builder checks whether its tuples are already sorted and skips
  the sort if so (`GB_builder.c:299-310, 438-490`) [source]. A rebuild whose new keys arrive
  sorted takes 64.7 ms against 245.6 for one that must sort, for 1e7 uniform entries in a
  (2^21, 2^21, 2^20) tensor at 4 threads [measured].

**The kernel**: a parallel, stable bucket pass keyed by one digit of a computed key. Each
thread counts the digit's values over its slice, a prefix sum gives each (thread, bucket) its
output offset, and each thread scatters its slice. It is the non-atomic method of
`GB_transpose_bucket` (per-thread workspaces, `Source/transpose/GB_transpose_bucket.c:150-166`)
[source] keyed by a digit instead of the column index, with at most about 2^12 buckets per pass
so that the scatter stays in cache. Passes move a permutation index and the digits still
needed; the values move once, at the end. Its users:

- P1: a relayout is a stable sort by the axes that change position, at most `ndims - 1`
  passes and often 0 or 1;
- P2: an LSD radix sort on the builder's `(j, i)` keys, whose bounds are known;
- P3: a transpose into many rows as two passes of about 1e3 buckets;
- P7: sorting the new keys of the nonempty rows.

## P1. Axis-map reindex of a matricized n-d array (GxB)

### Why

The Tensor needs index maps that couple the row and column indices, and builds each one from
parts (`design.md`, "The relayout planner", "Reductions", "Contractions and batched products"):

- **Relayout** is a Dijkstra search over N!(N+1) layout states, with reshape, transpose,
  row-permutation, column-permutation and rebuild edges, priced by 17 groups of constants fitted
  on one machine (`design.md`, "Fitted constants"). On u7 the five 3-d permutations take
  7.9-30.0 / 20.8-135 ms, against 71-112 / 444-780 ms for a rebuild [measured]. Searching a plan
  costs about 35 us in 3-d, 374 us in 4-d and 4.4 ms in 5-d, and 0.13 us from the cache
  [measured]; above 5 axes there is no search, only a rebuild.
- **A row permutation** (`P @ M`) costs about 0.6 ns per entry plus 6-50 ns per nonempty row at
  8 threads, and 12-185 ns per row at 1 thread, the most for hypersparse rows (the Tensor's
  fitted constants) [measured]. P itself is built in numpy when it cannot be cached.
- **A column permutation** rewrites the column indices through the Container and marks the
  rows jumbled; SuiteSparse sorts each row later with a quicksort
  (`Source/wait/GB_unjumble.c:83-140`) [source]. For rows of 1e4 entries that sort costs about
  34 ns per entry at 1 thread (u7 in layout `(0 | 1,2)`: 39 / 39 ms for the rewrite, 81 / 378
  with the sort) [measured], although the new order within each row is a known permutation.
- **The rebuild fallback** is 85% builder sort (above). In the last full measurement on NELL-2,
  `permute_dims((0, 2, 1))` took 633 / 2868 ms, against torch's 2844 [measured].
- **Reductions over part of a group** use 0/1 selector matrices, whose cost follows the group's
  positions, not the data: about 2.2 / 12 ns per position plus 0.7 / 2.7 per entry (uniform
  (3000, 2000, 10000) with 7.7e6 entries) [measured]. On NELL-2 in layout `(0 | 1,2)`, a sum over
  axis 2 took 1.65 s, against 2.8 ms in `(0,1 | 2)` [measured]. The Tensor now chooses
  its default layout from the data partly to avoid that.
- **TTM along a non-stored axis** relayouts the tensor before the kernel runs; it is 1.9-2.7x
  behind TACO at 1 thread on u7 and 6.8-7.4x on NELL-2 modes 0 and 1 (`limitations-and-roadmap.md`,
  "Known performance gaps") [measured].
- **Batched contractions** embed one operand block-diagonally by editing its column indices in
  numpy through the Container (about 2 ms per 1e6 entries) [measured].
- **Tensors of 2^60 positions or more in total** cannot be reshaped (P5), so each split move
  becomes a rebuild.

### What exists today [source]

- `GxB_Matrix_reshape` and `GxB_Matrix_reshapeDup` (`Include/GraphBLAS.h:7448, 7467`) move the
  split of the row-major or column-major ravel: one pass over the entries plus the builder with
  `known_sorted` (`Source/reshape/GB_reshape.c:331-435` in 10.5.2); below 2^60 total size only.
- `GrB_transpose` (`GraphBLAS.h:6638`) swaps the two groups.
- `GrB_extract` and `GxB_Matrix_extract_Vector` (`GraphBLAS.h:5482`) gather by inverse index
  lists, which are dense over the rows, so a hypersparse group needs a detour.
- `GrB_assign` with I and J scatters forward, one-to-one; with duplicate indices the last one
  wins (User Guide, `GrB_operations_assign_duplicates.tex`).
- `GrB_Matrix_build` and `GxB_Matrix_build_Vector` (`GraphBLAS.h:4110`) take any map, and sort.
- `GrB_kronecker` (`GraphBLAS.h:6651`) builds the selectors; the Container
  (`GraphBLAS.h:6896-6993`) gives O(1) access to the row list `h` and column indices `i`.

Nothing computes a new `(i, j)` from both old indices without a full sort.

### Proposed API

A one-operand einsum on matricized arrays:

```c
GrB_Info GxB_Matrix_reindex_axes    // C<M> = accum (C, T), T = A re-indexed by an axis map
(
    GrB_Matrix C,               // input/output matrix
    const GrB_Matrix Mask,      // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,   // optional accum for Z=accum(C,T)
    const GrB_BinaryOp dup,     // combines entries of A that land on the same T(i,j), in the
                                // order A stores them; NULL: GrB_INVALID_VALUE if any do
    const GrB_Matrix A,         // an n-d array, matricized
    int in_ndims,               // number of axes of A (>= 1)
    const uint64_t *in_dims,    // size of each axis of A
    const int *in_labels,       // label of each axis of A; a label used twice selects that
                                // diagonal: entries whose two coordinates differ are dropped
    int in_nrow,                // A's row index ravels axes 0..in_nrow-1 in C order (last
                                // fastest); its column index ravels axes in_nrow..in_ndims-1
    int out_ndims,              // number of axes of T
    const int *out_labels,      // label of each axis of T; each must appear in in_labels.
                                // An input label missing here is reduced with dup; a label
                                // used twice here puts the entry on that diagonal
    int out_nrow,               // T's row index ravels axes 0..out_nrow-1 of T
    const GrB_Descriptor desc   // GrB_INP0 = GrB_TRAN: the axes describe A' instead of A;
                                // mask, replace and nthreads as usual
) ;
```

| use                                            | A's layout     | in_labels / in_nrow | out_labels / out_nrow | dup  |
| ---------------------------------------------- | -------------- | ------------------- | --------------------- | ---- |
| relayout `ijk -> kij`                          | `(i,j \| k)`   | 0 1 2 / 2           | 2 0 1 / 2             | NULL |
| reshape (move the split)                       | `(i \| j,k)`   | 0 1 2 / 1           | 0 1 2 / 2             | NULL |
| transpose                                      | `(i \| j)`     | 0 1 / 1             | 1 0 / 1               | NULL |
| sum over j                                     | `(i,j \| k)`   | 0 1 2 / 2           | 0 2 / 1               | PLUS |
| block-diagonal embedding for `bij,bjk->bik`    | `(b,i \| j)`   | 0 1 2 / 2           | 0 1 0 2 / 2           | NULL |
| undo it after the product                      | `(b,i \| b,k)` | 0 1 0 2 / 2         | 0 1 2 / 2             | NULL |
| the diagonal of a square matrix, as n x 1      | `(i \| i)`     | 0 0 / 1             | 0 / 1                 | NULL |
| `GrB_Matrix_diag` (n x 1 to n x n)             | `(i \| )`      | 0 / 1               | 0 0 / 1               | NULL |
| `kron(B,A)` from `kron(A,B)` (perfect shuffle) | `(a,b \| c,d)` | 0 1 2 3 / 2         | 1 0 3 2 / 2           | NULL |

### Semantics and edge cases

- A must be `prod(in_dims[0:in_nrow]) x prod(in_dims[in_nrow:])` and C must be T's size, else
  `GrB_DIMENSION_MISMATCH`. Each group of T must be at most 2^60, else `GrB_INVALID_VALUE`. No
  product of all the dims is ever formed, so there is no total-size limit and no 128-bit
  arithmetic: each coordinate is below its dim, and each output group index below 2^60. C's
  32/64-bit index widths come from `GB_determine_pji_is_32`, as in `GB_reshape`.
- Repeated input labels select a diagonal; repeated output labels embed on one; dropped labels
  are reduced with dup, combining entries in A's storage order, so the result is deterministic
  (as for `GrB_Matrix_build`). An output label that is not an input label is
  `GrB_INVALID_VALUE`: no replication (broadcasting is P6).
- Types: T has A's type; dup follows `GrB_Matrix_build`'s rules; `C<M> = accum (C,T)` with the
  usual typecasting.
- An empty A, or a zero dim, gives an empty T. Pending work in A is finished first, as for
  `GrB_transpose`. C may alias A.
- iso: T is iso when A is iso and no entries combine, or dup is idempotent (ANY, FIRST, SECOND,
  MIN, MAX, LAND, LOR; compare the "nice" monoids of `Source/mxm/GB_AxB_iso.c:158-165`).
- Any sparsity of A; bitmap and full A are enumerated by position, and a full A under a
  bijective map gives a full T (a strided copy).
- Orientation: the plan is made against storage order, so a by-column A or C costs no extra
  transpose (a by-column `(R | C)` is a by-row `(C | R)`).
- A jumbled A is allowed; its column labels then count as unsorted.
- Errors: `GrB_NULL_POINTER`, `GrB_UNINITIALIZED_OBJECT`, `GrB_INVALID_VALUE` (labels, counts,
  a group too large, duplicates with a NULL dup), `GrB_DIMENSION_MISMATCH`,
  `GrB_DOMAIN_MISMATCH`, `GrB_OUT_OF_MEMORY`.

### Implementation sketch [inferred]

1. **Plan**, in O(ndims^2) with no data. Let S be A's storage order as a sequence of labels (by
   row: the input labels in order; by column: the column labels, then the row labels); the
   stored entries are sorted lexicographically by S. Let K be C's storage order of the output
   labels, repeated ones collapsed. Find the smallest m such that S without
   {K_1, ..., K_m} equals (K_{m+1}, ..., K_end) followed only by dropped labels. A stable sort by
   (K_1, ..., K_m), done least significant first, then gives K order with duplicates adjacent.
   An axis of more than about 2^12 positions is split into digits, one pass each. For 3 axes
   stored in the order (i,j,k):

   | target order       | passes (keys, first to last)         | the Tensor today on u7, ms  |
   | ------------------ | ------------------------------------ | --------------------------- |
   | (i,j,k), any split | 0: one sweep                         | a reshape: 5.1 / 21.2       |
   | (j,i,k)            | 1 (j); with split 2, whole rows move | `P @ M`: 7.9 / 20.8         |
   | (k,i,j)            | 1 (k)                                | T, RS: 14.9 / 62.8          |
   | (i,k,j)            | 1, inside each i segment (k)         | T, RS, `P @ M`: 30.0 / 135  |
   | (j,k,i)            | 2 (k, then j)                        | T, RS, T, RS: 27.0 / 105    |
   | (k,j,i)            | 2 (j, then k)                        | `P @ M`, T, RS: 23.4 / 84.7 |

   (T is a transpose, RS a reshape; measured.) Fast paths: no passes is the reshape sweep; when
   every key is a leading row label, whole row segments move (sort the nonempty rows' keys and
   copy segments: what `P @ M` does, without its hash lookup per hypersparse row); a prefix that
   S and K share makes the passes segmented, inside each row or block of rows.

2. **Keys**: one parallel sweep over A computes only the digits the passes need, dividing by
   constant strides (precomputed reciprocals, or a JIT kernel specialized on the plan); entries
   that fail a diagonal selection are dropped here.
3. **Passes**: the stable digit pass above.
4. **Assemble**: compute the new `(i, j)` in K order and call `GB_builder` with
   `known_sorted = true` (`GB_builder.c:159`) [source], which combines duplicates with dup
   (factory `GB_bld` kernels or `Source/jit_kernels/template/GB_jit_kernel_build.c`) and builds
   the row pointers.
5. `C<M> = accum (C,T)` through `GB_accum_mask`, as `GrB_transpose` does.

Work is O(nnz x (passes + 1)), parallel in every step; memory is two index buffers of nnz words
plus T. Reused: `GB_builder`, `GB_transpose_bucket`'s slicing, `GB_determine_pji_is_32`,
`GB_accum_mask`, the build JIT kernel. New: the plan and the digit pass.

A first version could implement the API on existing internals, running the radix chain inside
SuiteSparse (`GB_reshape` with P5's fix, `GB_transpose` into one axis's rows, and the builder for
the rest). That would deliver the API, the end of the 2^60 limit and of planning in Python, at
today's speed. The digit pass, dup, diagonals and segmented passes would follow.

### Workaround today, and its cost

The planner and its edges, selectors built from the shape or from the nonempty rows,
`from_coo` with a dup operator for hypersparse reductions, and the Container embedding
(`design.md`, "The relayout planner", "Reductions", "Contractions and batched products"). Most
of `tensor_layout.py` (the planner, its cost model, the caches of permutation and selector
matrices, the embedding) and the strategy code of the reduction, TTV and broadcast families
exist to route around the missing call.

### Expected gain [inferred]

A pass into about 1e3 buckets should cost about what a transpose into 1e3 rows costs (0.5-0.7
ns per entry at 8 threads, 1.8 at 1), and the assemble step about what a reshape costs (0.5 /
2.1 ns per entry).

- u7: about 10 ms for one pass and 15-19 ms for two at 8 threads (today 7.9-30), 40-60 ms at 1
  thread (today 21-135). So 1-3x; the (j,i,k) swap is already as fast.
- NELL-2 `permute_dims((0, 2, 1))`: about four digit passes, or one segmented sort, so
  150-300 ms at 8 threads against 633 today.
- 4-d and 5-d tensors: at most ndims - 1 passes instead of chains of five steps or more, and no
  planning.
- Tensors past 2^60 total: split moves at reshape speed.
- Reductions over part of a group: one call whatever the group's size, so the default layout
  can be chosen for other reasons.
- TTM and MTTKRP along non-stored axes: a cheaper relayout before the kernel.

### Who needs it

Every multi-axis use case, because each needs a relayout before a contraction over a middle
axis:

- metapath counting on heterogeneous knowledge graphs stored as one (relation, source, target)
  tensor (Hetionet, for example), where the contraction runs over the node axis;
- multiplex and multilayer graphs (layer, source, target): coupling over layers, aggregating
  over them, taking the inter-layer diagonal;
- temporal and streaming graphs (window, source, target): statistics per window and across
  windows;
- sparse CP and Tucker decompositions: TTM and MTTKRP along each mode;
- regular path queries: LAGraph's RPQ takes one matrix per edge label, and the product-graph
  form is one contraction over the label axis;
- graph products: `kron(A,B)` and `kron(B,A)` differ by the perfect-shuffle permutation, the map
  `(a,b | c,d) -> (b,a | d,c)`;
- block-diagonal batching and unbatching of many small graphs.

### Effort

L overall (API, plan, digit pass, tests); M for a first version on existing internals.

## P2. A radix sort in the builder (GxB, internal)

### Why

The builder sorts unsorted tuples with a parallel merge sort (`GB_builder.c:617-709`) [source].
For 1e7 entries (u7's keys in another order), `from_coo` takes 211 ms at 4 threads and 807 at 1,
against 22.9 and 37.5 for the same tuples presorted: 9x and 21x [measured]. The keys have known
bounds (`vdim` and `vlen`, each at most 2^60), which is the case where an LSD radix sort wins.

### Proposal

Inside `GB_builder` (step 2, `GB_builder.c:617`), sort `(j, i)` keys with the stable digit pass:
`ceil(log2(vdim) / 12) + ceil(log2(vlen) / 12)` passes at most, fewer when the tuples are
already sorted by a prefix of the key (the existing check finds full sortedness only). Keep the
merge sort for small inputs. No API change.

### Who needs it

Everyone who builds a Matrix from an edge list: streaming ingestion (one Matrix per time window
of a traffic stream), knowledge-graph loads of millions of edges (one `from_coo` per relation
type), `GrB_wait` after many `setElement` calls (its pending tuples go through the same
builder), and the builder-based transpose, which `GB_transpose_method` picks when the bucket
method's work estimate is higher (`GB_transpose_method.c:131-165`). For the Tensor: every
rebuild, and `from_coo` of unsorted input.

### Expected gain and effort

Not measured beyond the gap above [inferred]: with about three passes of 0.6-1 ns per entry, an
unsorted build of 1e7 entries would take 40-60 ms at 4 threads instead of 211. Effort M.

## P3. Transpose and column reductions into many rows, at 4-8 threads (GxB, performance bug)

### Why

A materialized transpose, or a column sum, into 3e5 or more output rows is 2-7x slower at 4 or
8 threads than at 2, and often slower than at 1. Through python-graphblas, best of 7, at 1 / 2 /
4 / 8 threads [measured]:

| case                                                    | operation           | ms at 1 / 2 / 4 / 8 threads |
| ------------------------------------------------------- | ------------------- | --------------------------- |
| uniform 1e3 x 1e6 with 1e7 entries                      | `A.T`               | 104 / 60 / 153 / 116        |
|                                                         | `reduce_columnwise` | 18.4 / 19.1 / 12.7 / 56.7   |
| Uber as `(date,hour \| lat,lon)`, 4392 x 1957380, 3.3e6 | `A.T`               | 13.9 / 10.8 / 94.2 / 67.0   |
|                                                         | `reduce_columnwise` | 4.5 / 4.6 / 17.5 / 14.1     |
| control: uniform 1e4 x 1e5 with 1e7 entries             | `A.T`               | 78.3 / 48.9 / 31.3 / 23.5   |

A diagnostic build of SuiteSparse that forces the non-atomic method is 3-7x faster at 8 threads
on the slow cases, and the Tensor's radix chain (above) beats a direct transpose 3-5x at 8
threads [measured]. A C reproducer (MRE) exists. The Tensor runs wide transposes and column
reductions at 2 threads, which pick the non-atomic method (`design.md`, "Fitted constants").

### Mechanism [source]

- `Source/transpose/GB_transpose_method.c`: at 1 or 2 threads the bucket method is sequential or
  non-atomic (line 49); at more threads it is atomic when `nthreads * avlen > anz` (line 55),
  and otherwise when `log2(anz) - log2(avlen) <= atol` (line 95), from a table (lines 74-92)
  "based on performance on a 4-core system with 4 threads with gcc 7.5" (lines 62-64).
- The atomic method increments one shared counter array per entry
  (`Source/transpose/factory/GB_transpose_bucket_template.c:59-75`) and places each entry with an
  atomic fetch-and-increment (`Source/transpose/template/GB_transpose_sparse_template.c:90`).
- The column sum is a bitmap saxpy; `Source/mxm/GB_AxB_saxpy4_tasks.c:91` takes the fine
  non-atomic tasks only while `relwspace < 0.5`, which 8 threads into 1e6 outputs exceed.

On this machine an atomic update of a cache line that other cores also write costs far more
than the rule assumes; with skewed data many threads hit the same counters.

### Proposal (no API change)

1. Re-tune the atomic and non-atomic choice on ARM and current x86 (x86 numbers are needed
   first). Prefer the non-atomic method whenever its per-thread workspaces fit a budget (for
   example `max (anz, 2^24)` words), and cap the threads of the atomic method.
2. Above about 2^16 output rows, a two-pass radix transpose: the stable digit pass on the low,
   then the high digit of the new row index, each with at most about 2^12 buckets. That is the
   Tensor's radix chain without the reshapes.
3. For reductions to a vector into many outputs: per-thread partial results whenever they fit
   the same budget [inferred].

### Who needs it

`A'` of any large graph (pull-direction traversals, in-degrees), column degrees of power-law
graphs (Uber's skew is typical of them), per-window statistics over destinations (fan-in) in
traffic analysis, and reversing metapaths. Expected gain: 2-7x at 8 threads from the re-tune
[measured, on the forced build], and a further 2-3x from the radix pass [inferred]. Effort M (S
for the re-tune alone, once x86 numbers exist).

## P4. Factory kernels for `ANY_FIRST` and `ANY_SECOND` (GxB, regression since v10.0)

### Why

Without the JIT (or at `"run"` before a kernel is compiled), `ANY_FIRST` and `ANY_SECOND`
semirings run SuiteSparse's generic kernel. A C reproducer (MRE) exists. For `C = P*M`, P a
1e6 x 1e6 permutation and M 1e6 x 1000 FP64 with 10 entries per row, JIT off, 4 threads, on
10.5.1 (10.3.1 is within 1%) [measured]:

| semiring               | ms   | kernel  |
| ---------------------- | ---- | ------- |
| `GxB_ANY_SECOND_FP64`  | 41.8 | generic |
| `GxB_MIN_SECOND_FP64`  | 14.9 | factory |
| `GxB_PLUS_SECOND_FP64` | 15.0 | factory |
| `GxB_LOR_SECOND_BOOL`  | 33.2 | generic |
| `GxB_ANY_SECOND_BOOL`  | 32.9 | generic |

Through python-graphblas, on 1e6 entries at 4 threads: FP64 `any_second` 2.14 ms against
`min_second` 1.13; UINT32 `plus_second` 2.26 (generic) against `min_second` 1.02 [measured].

### Cause [source]

`Source/GB_control.h` lines 1831-1858 define `GxB_NO_ANY_FIRST_*` and `GxB_NO_ANY_SECOND_*`
for all 13 types, while the comment just above (lines 1826-1828) still says "33 semirings
enabled with the ANY monoid: ANY: FIRST, SECOND, PAIR". In v9.4.5 the same block left every real
type enabled (only the complex types were off), so this changed in v10.0.0. Also off:
`LOR_FIRST_BOOL` and `LOR_SECOND_BOOL` (lines 878-879), and every kernel of INT8, INT16, UINT8
and UINT16 (lines 103-110). The SECOND semirings still on are MIN_SECOND and MAX_SECOND for
INT32, INT64, UINT32, UINT64, FP32 and FP64, and PLUS_SECOND for FP32, FP64, INT32 and INT64.
The factory sources (`FactoryKernels/GB_AxB__any_second_fp64.c` and the rest) are still there.

### Proposal

- Zero code size: when an `ANY_FIRST_T` or `ANY_SECOND_T` kernel is compiled out and
  `MIN_FIRST_T` or `MIN_SECOND_T` is in, run the MIN kernel. A minimum is a valid ANY result,
  the factory already renames operators to share kernels (`GB_control.h:34-69`), and MIN keeps
  iso results iso (`GB_AxB_iso.c:158-165`). It loses only ANY's early exit, which matters in dot
  products.
- Or re-enable ANY_FIRST and ANY_SECOND for BOOL, INT32, INT64, UINT32, UINT64, FP32 and FP64,
  and LOR_FIRST_BOOL and LOR_SECOND_BOOL: 16 semirings, a change to `GB_control.h` only.
- Either way, correct the comment.

### Who needs it

Every permutation or selection product (relabeling vertices, extracting rows as `P*A`), where
ANY is the natural monoid and libraries reach for it first; the comment in `GB_control.h` notes
these semirings serve `GrB_reduce` to a vector too. Breadth-first search's `ANY_SECONDI` is
positional and still has its kernels. The Tensor uses `min_second` (`design.md`, "The
SuiteSparse JIT"). Gain 2.8x on the case above; effort S.

## P5. `GxB_Matrix_reshape` past 2^60 total size (GxB)

### Why [source, measured]

`GB_reshape` multiplies `nrows * ncols` with `GB_int64_multiply` and returns
`GrB_OUT_OF_MEMORY` when it overflows (`Source/reshape/GB_reshape.c:85-93`). The helper is
pessimistic: it fails whenever both factors are at least 2^30
(`Source/math/GB_uint64_multiply.h`). The new indices then come from a 64-bit linear index
`iold + jold * tvlen` (`GB_reshape.c:354, 383-386` in 10.5.2; lines 333 and 362 in 10.5.1). A
total of 2^59 works and 2^60 fails, even for a matrix with 2 entries; a (2^21, 2^21, 2^20)
tensor with 1e7 entries is refused with `GrB_OUT_OF_MEMORY` [measured]. The Tensor then replaces
each split move by a rebuild with sorted keys (`design.md`, "States and edges"): 64.7 ms per 1e7
entries at 4 threads, where a reshape costs about 5 ms at 8 threads [measured].

### Proposal

Same API. Accept any reshape whose new dimensions are each at most 2^60 and whose size products
are equal (compared in 128-bit arithmetic, or by cross-dividing), and compute the new indices
without the linear index:

- if `vlen_new` divides `tvlen` (f = tvlen / vlen_new): `inew = iold % vlen_new`,
  `jnew = jold * f + iold / vlen_new`;
- if `tvlen` divides `vlen_new` (f = vlen_new / tvlen): `inew = iold + (jold % f) * tvlen`,
  `jnew = jold / f`;
- otherwise use 128-bit arithmetic (`unsigned __int128` with gcc and clang; `_umul128` and
  `_udiv128` with MSVC; or a two-limb helper beside `GB_uint64_multiply`), or keep refusing that
  rare case.

Every reshape that moves the split along one axis sequence is one of the first two cases, since
`vlen` before and after is the product of a suffix of the dims. A size product that overflows
should return `GrB_INVALID_VALUE`, not `GrB_OUT_OF_MEMORY`. python-graphblas needs a matching
change: `Matrix.ss.reshape` checks the shape with numpy, which fails at 2^63.

### Who needs it

Traffic matrices over the 2^32 IPv4 address space (2^64 positions each, past the limit even
before a window axis is added), as in the Graph Challenge's anonymized network-traffic data;
FROSTT's NELL-1 (about 1.5e20 positions) and Amazon reviews (about 1.5e19); 3-d cubes past
2^20 per side and most large 4-d tensors; reshapes between `2^40 x 2^30` and `2^30 x 2^40`
hypersparse matrices. Effort S. It is the cheapest way to close the 2^60 gap before P1 exists.

## P6. Broadcast element-wise operations (GrB candidate; the gathered form GxB)

### Why [measured]

- On u7 in layout `(i,j | k)`, `X * w[k]` as `M @ diag(w)` runs SuiteSparse's column-scaling
  kernel, 2.7 / 7.5 ms. With w holding 10 entries the product becomes a saxpy: 19.3 / 128.9 ms,
  7x slower.
- A vector along an axis inside a group has no direct form. The Tensor gathers its values onto
  the stored entries with numpy through the Container, then runs one same-pattern `ewise_mult`
  (`design.md`, "Element-wise operations and broadcasting"). That is still behind torch's
  per-entry gather at 8 threads: hyper 7.9 and 13.4 ms against 1.3 and 1.7 (axes 1 and 2), Zipf
  47.0 against 15.2 (axis 1), and NELL-2 with a dense matrix over axes 1 and 2, 322 against 174.
- Union broadcasting replicates the smaller operand first: on u7, an outer product with a
  10-entry vector and then `ewise_add`, 21.5 / 126.5 ms for a 2e7-entry result.

### What exists today [source]

`GrB_Matrix_diag` and `GxB_Matrix_diag` (`GraphBLAS.h:4452, 4478`) plus `GrB_mxm`.
`GB_AxB_meta` switches to `GB_rowscale` and `GB_colscale` (and their JIT kernels,
`Source/jit_kernels/template/GB_jit_kernel_rowscale.c` and `_colscale.c`) only when
`GB_is_diagonal` holds, which requires exactly n entries in n vectors and refuses bitmap and
full matrices (`Source/mxm/GB_is_diagonal.c:46-65`; `Source/mxm/GB_AxB_meta.c`, lines 316,
386 and 626-634). LAGraph uses the idiom to normalize columns and rows and to compare against
a vector (`experimental/algorithm/LAGr_MarkovClustering.c:126-128, 188-189`;
`LAGr_PeerPressureClustering.c:157-160`); at least ten of its experimental algorithms call
`GrB_Matrix_diag`. Spec issues #72 ("Allow broadcasting in ewise operations") and #24 ("Support
rank promotion") are open; #67 (boolean index vectors in extract) is the select-flavored
relative.

### Proposed API

```c
// T(i,j) = op (A(i,j), v(k)) for each entry A(i,j) whose v(k) is present; k = i by default
// (v is a column vector, replicated across the columns of A); with GrB_INP1 = GrB_TRAN, k = j
// (v as a row vector). With gop non-NULL, k = gop (A(i,j), i, j, gthunk): the gathered form.
GrB_Info GxB_eWiseMult_MatrixVector         // C<M> = accum (C, T)
(
    GrB_Matrix C,
    const GrB_Matrix Mask,
    const GrB_BinaryOp accum,
    const GrB_BinaryOp op,                  // plus _Monoid and _Semiring variants
    const GrB_Matrix A,
    const GrB_Vector v,
    const GrB_IndexUnaryOp gop,             // NULL, or an integer-valued index op
    const GrB_Scalar gthunk,
    const GrB_Descriptor desc
) ;
// The operand-order twin, T(i,j) = op (v(k), A(i,j)), for non-commutative ops:
GrB_Info GxB_eWiseMult_VectorMatrix (C, Mask, accum, op, v, A, gop, gthunk, desc) ;
// Union: T(i,j) = op (a, b) on the union of A's pattern and {(i,j) : v(k) present}, a missing
// a or b taking alpha or beta (as GxB_eWiseUnion); a row with v(i) present becomes full.
GrB_Info GxB_eWiseUnion_MatrixVector (C, Mask, accum, op, A, alpha, v, beta, desc) ;
```

The names avoid the `_Vector` suffix, which SuiteSparse uses for index lists given as
GrB_Vectors. For the spec: a polymorphic `GrB_eWiseMult (C, M, accum, op, A, v, desc)` and
`GrB_eWiseAdd`, chosen by the operand types, with the descriptor's transpose of the vector
meaning "row vector"; no gathered form.

### Semantics

- `size(v)` is `nrows(A)` (or `ncols(A)` with `GrB_INP1` transposed); with gop, a `k` past the
  end is `GrB_INDEX_OUT_OF_BOUNDS` (or a dropped entry).
- Intersection: entries of A with no `v(k)` are dropped, so T's pattern is a subset of A's.
- iso when A and v are iso and op is not positional. Positional ops and IndexBinaryOp operators
  see A's `(i, j)` and v's `(k, 0)`.
- Any format of A and v; a full v and a full A give a full T.
- The gathered form covers the plain ones: `gop = GrB_ROWINDEX_INT64` or `GrB_COLINDEX_INT64`
  with thunk 0. For an axis inside a group, `gop` is `z = (j / stride) % dim`, a user op with a C
  definition that the JIT compiles.

### Implementation sketch [inferred]

A full v: run `GB_colscale` or `GB_rowscale` with the diagonal's values taken from `v->x`, which
is laid out as a full diagonal's values are, so the factory and JIT kernels are reused as they
are and no diagonal is built. A sparse or bitmap v: scatter it into a dense workspace with a
presence map, compute T on A's pattern, turn the entries without `v(k)` into zombies (as
`GB_select` does) and let `GB_wait` prune them. The gathered form: an apply-like kernel (the
index-op path of `GB_apply`), with gop and op fused by the JIT. O(nnz(A) + size(v)), parallel.

### Who needs it

Degree scaling everywhere: the random-walk matrix `D^-1 A`, the normalized adjacency
`D^-1/2 A D^-1/2`, PageRank, degree-weighted path counts on metapaths, and multiplex
normalization `A(l,u,v) / outdeg(l,u)`; edge weights from vertex attributes; filtering edges by a
vertex property; LAGraph's Markov and peer-pressure clustering. Expected gain [inferred]: 7x for
a sparse `v` (the full-diagonal speed), and parity with torch on the gathered rows above (it is
torch's algorithm, run in parallel). Effort M; S for a full `v` only.

## P7. Relabel by vector maps, with a dup operator (GxB; spec later)

### Proposal

`T (r(i), c(j)) = A (i,j)`, entries that land on the same position combined with dup. The maps
are GrB_Vectors of any 32- or 64-bit integer type; a row with no `r(i)` is dropped, so one call
can filter, renumber and contract.

```c
GrB_Info GxB_Matrix_relabel         // C<M> = accum (C, T), T (r(i), c(j)) = A (i,j)
(
    GrB_Matrix C,
    const GrB_Matrix Mask,
    const GrB_BinaryOp accum,
    const GrB_BinaryOp dup,         // combines entries landing on the same T(i,j); NULL: error
    const GrB_Matrix A,
    const GrB_Vector r,             // size nrows(A); NULL: r(i) = i; a missing r(i) drops row i
    const GrB_Vector c,             // size ncols(A); NULL: c(j) = j; a missing c(j) drops col j
    const GrB_Descriptor desc       // GrB_INP0: use A'; mask, replace, nthreads as usual
) ;
```

A value `r(i)` past `nrows(C)` is `GrB_INDEX_OUT_OF_BOUNDS`; iso and dup rules are P1's. An
extension for maps that couple i and j would take two integer-valued IndexUnaryOps
`z = f(aij, i, j, thunk)` instead of the vectors (JIT-able, as in `GrB_Matrix_apply_IndexOp`,
`GraphBLAS.h:6360`); it can be approximated today (see "Already possible today").

### What exists, and who needs it [source]

One-to-one maps: `GrB_extract` with an inverse permutation (LAGraph permutes by degree before
counting triangles this way, `src/algorithm/LAGr_TriangleCount.c:278-284`) and `GrB_assign`
with a forward one (no dup). Many-to-one maps: `S A S'` as two products (LAGraph's coarsening
computes `S = S*A`, then `A = S*S'`: `experimental/algorithm/LAGraph_Coarsen_Matching.c`,
lines 478 and 490). The users: graph coarsening and quotient graphs (community aggregation in
Louvain-type methods, condensing connected components), relabeling, and induced subgraphs with
renumbering in one call. For the Tensor: row permutations and the hypersparse reduction
branch, which go through `to_coo` and `from_coo` with a dup operator today (`design.md`,
"Reductions").

### Implementation sketch and effort [inferred]

Row map: read `r` at each nonempty row, sort those keys (nvec keys, not nnz), copy row segments
in that order; rows that merge are concatenated and assembled per row (sort within the row,
combine with dup, as `GB_wait` does for pending tuples). Column map: gather `c(j)` per entry
(the `GB_I_inverse` machinery of extract), mark the rows jumbled, sort within rows, combine
duplicates. Dropped rows and columns are skipped. O(nnz + nvec log nvec + the per-row sorts),
parallel. Gain on contraction: one pass plus per-row sorts instead of two products and their
workspaces, 2-4x. Effort M.

## P8. Zero-copy load and unload in the C API spec (GrB)

### Why

SuiteSparse 10's Container (`GraphBLAS.h:6896-6993`; User Guide, `GrB_objects_Container.tex`)
moves a matrix's arrays (`p`, `h`, `b`, `i`, `x`) in and out as GrB_Vectors in O(1), at their
native 32- or 64-bit widths, with the iso and jumbled flags [source]. The Tensor depends on it:
O(1) changes of orientation, nonempty-row counts, column permutations and the block-diagonal
embedding in place, the gather broadcast, its Numba kernels reading the stored arrays in place,
and the array-UDT engine's output handover (`design.md`, "Data model", "Numba kernels and the
kernel cache"). The legacy GxB export copies, and the legacy unpack widens 32-bit indices to
64-bit for good [measured]. `to_coo` copies (8.0 ms per 1e7 entries at 4 threads), where COO
libraries return their arrays [measured].

### What exists in the spec [source]

The C API spec 2.1's `GrB_Matrix_import` takes `const` arrays, so it copies (section 4.2.5.17).
Spec issue #31 ("Filling GraphBLAS matrices without calling GrB_Matrix_build()") and #83 (more
formats) are open, as is #3 (allocators). SuiteSparse's own plans (`Doc/future/FUTURE.txt`)
include COO through the Container, with a binary operator to combine duplicates.

### Proposal

For the spec: a move-semantics pair (`GrB_Matrix_unload` and `GrB_Matrix_load`, or the Container
itself) whose components are GrB_Vectors, an implementation-defined set of formats with a
query (`GrB_get` of the storage format), and allocator rules. For SuiteSparse: the COO Container
from `FUTURE.txt`, which would make `to_coo` and a rebuild's input nearly free. For
python-graphblas: a supported wrapper in `ss` (the Tensor calls the C functions directly today).
Who needs it: every language binding that shares memory with numpy or Arrow, and graph stores
that keep one CSR per relation and hand it to GraphBLAS. Effort M for the spec; S for the
python-graphblas wrapper.

## P9. Array-valued entries: works today, small follow-ups (GxB)

### What works [measured]

A fixed-size array type (R doubles) with a C definition, operators with a NULL function pointer
and C definitions (`GxB_BinaryOp_new`), including mixed types (FP64 times FP64[R]), and a
semiring built from them all run, and the JIT compiles `plus_times[FP64, FP64[R]]` into one dot
product kernel. With it, a TTM is one `GrB_mxv` whose result holds, for each output fiber, its
R values (`design.md`, "TTM engines"):

- NELL-2 mode 2, R = 16, 1 thread: 139 ms, against 174 for the Tensor's Numba kernel and 278 for
  TACO (single-threaded);
- without Numba it is the fast path: 1.5-5.6x over plain products for TTM, and the first step of
  MTTKRP takes NELL-2 mode 0 from 702 to 142 ms at 1 thread (R = 16);
- a cold JIT costs 1.0-1.3 s per R on the first call.

### Asks

- **A dense trailing axis without per-value indices.** The product's Vector of R-arrays becomes
  an (n x R) FP64 Matrix through the Container in O(1) when it is full. When it is bitmap, the
  Tensor repeats each row's bitmap byte R times; when it is sparse, it writes a column index
  for every value and a row pointer per row (`tensor_udt.py`). A format, or a load mode, in which
  a row is either absent or full (TACO's sparse-sparse-dense output for TTM) would remove that
  [inferred: not designed].
- **Documentation: how a JIT failure surfaces.** A failed compile returns `GxB_JIT_ERROR` and
  sets the JIT control to `"load"`, and it can be reported by a later call that runs a JIT
  kernel of its own. The Tensor saw it from a `GrB_Vector_set_INT32` of the sparsity control,
  which converted an empty array-UDT Vector at once (`design.md`, "The SuiteSparse JIT"). One
  paragraph in the User Guide's JIT chapter would help every caller that catches it.

### Who needs it

Multi-edges, where an entry holds a vector indexed by edge id (FalkorDB, which is built on
SuiteSparse, describes this in its blog post "Edges in FalkorDB"); TTM and CP with rank-R
blocks; recommender factors. Effort S for the documentation; the output format is L and needs a
design first.

## P10. Spec position on n-d (GrB)

Do not add an n-d object to the C API yet. The Tensor carries 3-d to 5-d data well on the 2-d
API (each axis group limited to 2^60 positions, so about 2^120 in total), and P1 to P8 would
close most of its gaps. Revisit if higher-order workloads hit that wall. Spec candidates from this
page, in order of readiness: broadcasting (P6; issues #72 and #24), load and unload (P8; #31),
and a 2-d `GrB_Matrix_reshape` with P5's semantics. The reindex family (P1, P7) should stay GxB
until SuiteSparse has experience with it.

## Already possible today

Each of these was an ask until a combination of existing calls turned out to do it.

1. **`min_second` for permutation and selection products** [measured]. A factory kernel for
   INT32, INT64, UINT32, UINT64, FP32 and FP64, and it keeps an iso operand iso; UINT32
   `plus_second` is generic (2.26 against 1.02 ms on 1e6 entries at 4 threads). BOOL has no
   factory SECOND semiring. The Tensor uses it (`design.md`, "States and edges").
2. **A column permutation through the Container** [measured]. Unload, gather the column
   indices through the map with `GxB_Vector_extract_Vector` (`GraphBLAS.h:5457`), set
   `jumbled`, load; SuiteSparse sorts each row only when something needs it. On a 1e6 x 1e3
   matrix with 1e7 entries at 4 threads: 10.3 ms without the sort, 33.7 with it, against 39.7
   for `M @ Q` and 50.1 for `M[:, perm]`. The map must be one-to-one within each row, and a
   jumbled Container must not hold read-only arrays
   (`Source/container/GB_load_from_container.c:294-298`) [source]. The Tensor uses it
   (`PERM_C`, `design.md`, "States and edges").
3. **Rebuilds with sorted keys**, priced as such [measured]: the builder skips its sort, 64.7
   against 245.6 ms per 1e7 entries at 4 threads. The Tensor prices them separately and uses
   them for split moves past 2^60 (`design.md`, "Cost model").
4. **A rebuild's cost is the sort, not numpy** [measured]: `to_coo` 8.0, numpy index arithmetic
   32.7, unsorted build 211 and presorted build 22.9 ms (u7, 4 threads). Moving the index
   arithmetic into GraphBLAS would gain little; avoiding unsorted rebuilds gains a lot.
5. **A general reindex without numpy** [inferred, not run]. `GrB_Matrix_apply_IndexOp`
   (`GraphBLAS.h:6360`) with an integer-valued IndexUnaryOp computes each entry's new row (or
   column) index as the values of a matrix with A's pattern; `GxB_unload_Matrix_into_Container`
   (`GraphBLAS.h:6963`) hands those values over as a GrB_Vector in O(1); and
   `GxB_Matrix_build_Vector` (`GraphBLAS.h:4110`) builds C from the three vectors with dup. It is
   parallel and JIT-able, and pays two pattern copies plus the full sort of item 4.
6. **A gather in GraphBLAS** [measured in item 2].
   `GxB_Vector_extract_Vector (w, NULL, NULL, v, I, desc)` computes `w = v(I)` for a per-entry
   index vector in parallel. Loaded as the values of a pattern copy, it could replace the numpy
   gather of the broadcast multiply (`tensor_layout.gather_operand`) [inferred].
7. **Per-call thread counts**, through the descriptor or `GxB_Context` (`GraphBLAS.h:4518`),
   are the stopgap for P3; the Tensor uses them.

## Smaller notes

- **The in-place reshape bug** (`GxB_Matrix_reshape` keeping 32-bit indices past 2^32,
  SuiteSparse issue 462) is fixed in 10.5.2: see `GB_reshape.c:126-127` and the width check at
  its line 258 [source]. The Tensor's workaround, `tensor_layout.reshape_owned`, is marked
  `TODO(SuiteSparse 10.5.2)` (`design.md`, "Executing a plan").
- **The reshape error code**: "too large" is reported as `GrB_OUT_OF_MEMORY`
  (`GB_reshape.c:88-93`) [source]; `GrB_INVALID_VALUE` would say what happened.
- **Editing `h` through the Container**: `GxB_load_Matrix_from_Container` checks only sizes, in
  O(1) (`GB_load_from_container.c:261-264`) [source], so a stale hyper hash `Y` or an unsorted
  `h` goes unnoticed. One sentence in the User Guide's Container section ("free Y when you
  change h, and keep h sorted") would make such edits safe.
- **A column slab** (`A(:, j)` of a matrix held by row) scans every row. That is inherent to CSR,
  so slicing the last stored axis stays slower than in COO libraries; P1's diagonal selection,
  or `GrB_select` with an index op, is the parallel route.

## Open questions

For SuiteSparse:

1. P1's shape: labels in the einsum style (repeated input labels select a diagonal), or a
   simpler form in which the output axes name input axes and no input label repeats? And its
   name (`GxB_Matrix_reindex_axes`, `GxB_Matrix_permute_axes`, or a generalized
   `GxB_Matrix_reshape`)?
2. Is the stable digit pass welcome as an internal kernel shared by the builder, the transpose
   and P1 and P7? What workspace budget is acceptable for its per-thread counts?
3. Was turning off `ANY_FIRST` and `ANY_SECOND` in v10.0 deliberate (code size for the 32/64-bit
   variants)? Is running the MIN kernel in their place acceptable?
4. P3 needs x86 numbers before `GB_transpose_method` is re-tuned. Which machines matter?
5. P6: is reusing `GB_rowscale` and `GB_colscale` with `v->x` as the diagonal sound for every
   format of `v`? Should a gather index out of range be an error or a dropped entry?

For the python-graphblas maintainers:

6. Wrap the Container as a supported `ss` API (P8)?
7. Which of these to raise first, and in what form: issues, MREs, or a branch SuiteSparse can
   run?

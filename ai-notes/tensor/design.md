# Tensor: design

How the experimental `gb.Tensor` is built, as the code stands. It is for contributors and
reviewers. It explains why the code is shaped this way and where each piece lives. The user
guide (`docs/user_guide/tensor.rst`) covers how to use it. The decisions behind the API are
in `decisions.md`.

Code comments may refer to the sections below by their headings, so keep the headings
stable. If you rename one, update the references (`grep -rn "ai-notes" graphblas/`).

Timings quoted here were measured on an Apple M5 Pro (6 performance and 12 efficiency cores)
with SuiteSparse:GraphBLAS 10.5.1, written "8 / 1" for 8 and 1 threads. The data sets they
name (u7, skew, hyper, Zipf, NELL-2, Uber, NIPS, Enron) are defined in the module docstring of
`graphblas/core/tensor.py`. The FROSTT tensors are at [frostt.io](http://frostt.io/).

## Data model

A Tensor is `(_M, shape, layout)`:

- `_M` is one SuiteSparse Matrix, **always held by row** (formats `csr`, `hypercsr`,
  `bitmapr`, `fullr`).
- `layout` is a `Layout(rows, cols)` (`tensor_layout.Layout`). It is an ordered partition of
  the logical axes into a row group and a column group.

Logical position `x` is stored at `_M[ravel(x[rows]), ravel(x[cols])]`, and each group is
raveled in **C order, in the order the layout lists it**. So `Layout((1, 0), (2,))` stores
`T[i, j, k]` at row `j * shape[0] + i` and column `k`. An empty group has size 1. A 0-d
Tensor is a 1 x 1 Matrix. A 1-d Tensor defaults to `Layout((), (0,))`, one row, which holds
the same bytes as a Vector. A 2-d one defaults to the plain Matrix, `Layout((0,), (1,))`.

Why one Matrix: every SuiteSparse kernel works on it as it is. That includes hypersparse
storage (essential when a group's positions far outnumber its entries), parallel transpose,
reshape and build, every semiring product, masks and accumulators. No n-d object is needed
from SuiteSparse.

Why always by row: the planner and every strategy then reason about one physical form.
`tensor_layout.new_matrix` and `tensor_layout.from_coo` make by-row Matrices whatever the
global format is. `flip_orientation` turns a by-column Matrix into the by-row Matrix of its
transpose in O(1) through the Container. `Tensor._check_invariants` asserts the invariants in
tests.

### Layouts and raveling

`tensor_layout.ravel` and `unravel` convert between per-axis coordinates and a group's linear
index, as int64 (every group is under 2^60). `coords_of` gives the logical coordinates of a
stored Matrix's entries.

The planner works on **states** `(seq, split)`, where `seq = rows + cols` and
`split = len(rows)`. `Layout.state` and `Layout.from_state` convert.

### Limits

- Each group holds at most 2^60 positions, one Matrix dimension (`MAX_GROUP`).
  `as_layout` refuses layouts that break it.
- SuiteSparse's reshape refuses matrices of 2^60 positions or more in total. Past that, the
  planner replaces every split move by an order-preserving REBUILD (see "States and edges").
- The planner searches states only up to 5 axes (`MAX_PLANNED_NDIM`). Above that, a relayout
  is one REBUILD.

## Copy-on-write views

`permute_dims`, `.T`, `.mT`, `expand_dims`, `squeeze` and `reshape` (where the layout holds
the result) return a new Tensor that **shares `_M`** with a different shape and layout, in
O(1) (`Tensor._view`). Sharing is never visible:

- A `_Store` holds the set of Tensors that share a Matrix (`_Owners`, a weak dictionary keyed
  by `id`, so no `__del__` bookkeeping is needed). It also holds an `exported` flag, set when
  the Matrix is handed out (`to_matrix(copy=False)`, `T.ss.matrix`,
  `from_matrix(copy=False)`).
- Every path that changes `_M` in place calls `Tensor._own()` first. It copies the Matrix
  when another owner exists or the Matrix was exported.
- A result that replaces `_M` wholesale goes through `_set_M`, which leaves the old Matrix to
  its other owners.
- `_share` lets a target adopt another Tensor's store, so `T << U` copies nothing when no
  relayout is needed.

This gives value semantics, as python-graphblas has elsewhere (`A.T` is not a writable view
either). It also survives an update that replaces `_M` with a Matrix in another layout, which
numpy-style mutable views could not.

## Expressions and storing results

Operations return a `TensorExpression` (`tensor.py`), a recipe: a function plus its
arguments. Nothing is computed until `.new()` or `T(mask, accum, replace) << expr`.
`BaseExpression._new` and `Tensor._update` then call the recipe with the target's `Updater`.
So **planning happens at assignment time**, when the target's layout and mask are known.
Recipes issue ordinary Matrix operations, so the Recorder, burble and the JIT see them.

Every recipe ends in `_store(updater, src, shape, layout)`, the only place where the mask and
the accumulator apply:

- The output's layout is `_fixed_layout`, which is the target's own layout when old values
  can survive (an accumulator, a mask without `replace`) or when the target is **pinned**.
  A layout is pinned when the user chose it, through the constructor's `layout=`,
  `relayout`, or `dup(layout=)`. Otherwise the result keeps the layout it was computed in.
  With a mask in another layout, whichever is smaller moves, the result or the mask.
- Masks are Tensor masks. A mask of a Matrix or Vector of the output's shape is converted
  (`_output_mask`). Masks do not broadcast. The mask is relaid out to the output's layout
  (`_mask_in_layout`).
- `fresh=True` lets the target adopt a temporary Matrix instead of copying it. `share=`
  adopts another Tensor's store.

Same-shape operations with operands in different layouts pick one layout to compute in
(`_choose_layout`). The candidates are each operand's layout, the fixed output layout and the
mask's layout, scored by the summed relayout costs (`tensor_layout.relayout_cost`). Ties go
to the largest operand.

## Choosing a default layout

New data gets the identity axis sequence (logical C order, so `to_coo` and the repr are
cheap) split after as many axes as the data fills:

- `tensor_layout.default_layout` is the shape rule: the largest split `p` with
  `prod(shape[:p]) * FULL_ROW_ENTRIES <= nnz` (`FULL_ROW_ENTRIES = 4`), else 1.
- `Tensor.from_coo` uses `_data_split_layout`, which also looks at the data. Rows are worth
  having when they hold enough entries each (`_rows_ok`):
  - 32 entries per row if the rows would be hypersparse (`MIN_HYPER_ROW_ENTRIES`), because
    each hypersparse row costs a hash lookup in element-wise products;
  - 4 entries per row (`FULL_ROW_ENTRIES`);
  - or 1.5 (`MIN_ROW_ENTRIES`), when one axis fewer in the rows would give a column group
    with more positions than entries.

  Sorted input is counted exactly from its row keys, which the build reuses. Large or
  unsorted input is judged from a sample of 2^14 entries (`_sampled_nonempty_rows`).
  Unsorted input is then settled after the build (`_settle_split`).

- A result with an empty column group (an n x 1 Matrix) is reshaped to the default split
  (`_healthy`). An empty row group is left alone: one row holding every entry costs the same
  as a Vector.

## The relayout planner

`tensor_layout.relayout(M, shape, src, targets)` moves a Matrix from one layout to the
cheapest of a set of target states. A relayout is a sort by the new major key, so the planner
builds it like an LSD radix sort: it chains reshapes and transposes, each cheap, rather than
doing one big scatter. A transpose costs 0.5 ns per entry into 1e3 output rows but 8-16 ns
into 3e5 or more, where more threads do not help.

### States and edges

`plan` runs Dijkstra over states. Its nodes also record whether the rows come straight from a
reshape: such rows are hypersparse with no hash of their row ids yet, so a following
`P @ M` costs more. The edges (`_neighbors`):

- **`RS`** moves the split and keeps the sequence: `ss.reshape` in storage order (rowwise
  on a by-row Matrix).
- **`T`** swaps the groups: a materialized transpose. At more than 1 thread into 1e5 or more
  rows it is also offered at 2 threads, which runs SuiteSparse's non-atomic method, faster
  than 4 or 8 threads on the measuring machine.
- **`PERM_R`** applies any permutation of the row axes: one `P @ M` with a permutation matrix
  `P` (`perm_rows`), with a factory-kernel semiring (`_perm_semiring`: `min_second`, or
  `lor_land` for BOOL, as `P` holds True).
- **`PERM_C`** applies any permutation of the column axes: the column indices are rewritten
  in the Container through a cached map (`perm_cols`), and the rows are marked jumbled, so
  SuiteSparse sorts each only when something needs it. Values are untouched, so it takes any
  dtype.
- **`REBUILD`** goes from any state to any state: `to_coo`, new indices, `from_coo`
  (`rebuild`). It is the fallback, and its cost bounds every search.

`allow_perm` restricts the permutation edges: True, `"rows"`, `"cols"` or False. A UDT
Tensor gets `"cols"` only, because `PERM_R` needs a semiring over its values. A search stops
expanding any node that already costs more than a REBUILD.

When the tensor has 2^60 positions or more in total, SuiteSparse cannot reshape it, so each
`RS` edge becomes a REBUILD into the same axis order. That REBUILD needs no sort (its keys
arrive sorted), and the fallback prefers a target in the source's axis order for the same
reason.

### Cost model

Each edge is priced in nanoseconds from per-entry and per-row constants. Most constants are
pairs, (more than 1 thread, 1 thread). Some terms need explaining:

- **Expected nonempty rows.** `expected_rows(nrows, nnz)` assumes uniform data. Reshapes and
  permutations pay per nonempty row, and pay differently for sparse rows, rows straight from
  a reshape, and hypersparse rows (`C_PERM_ROW`).
- **Transposes** are priced from a table of measured costs by output rows (`T_TABLE`,
  interpolated log-linearly by `t_cost`). Into hypersparse rows they use `C_T_HYPER`. Clustered
  data has far fewer nonempty columns than uniform data, so for sources of 2^22 entries or
  more the first transpose is priced on a sample of the source's column indices
  (`sampled_nonempty_cols`, 2^14 samples).
- **About one entry per row** (`density`): when rows hold fewer than about 2 entries,
  `P @ M` pays a hash lookup per row, so it is priced at least like a REBUILD.
- **REBUILD** is mostly SuiteSparse's sort in `from_coo`. When the new keys arrive sorted
  (the same axis order) the builder skips it, priced as `C_REBUILD_SORTED`.
- **A fixed cost per step**, `C_STEP` (10 us), plus `C_REBUILD_FIXED` (40 us), so tiny
  tensors take the plan with the fewest steps.

The constants **rank plans; they do not predict times**. They run 1.3-2x low on uniform data
and can be further off on clustered data (see "Fitted constants").

### Plan cache and matrix cache

- **Plans** are cached in an LRU of 4096 (`_plan_cache`). The key is the shape, the source,
  the target set, 1 thread or more, the one-entry-per-row flag, `allow_perm`, and the bit
  lengths of the entry count and of the sampled column count. A 3-d plan costs about 35 us
  to search and 0.13 us from the cache. A choice among many layouts by their cost to reach
  (`_mttkrp_order`) prices them all from one search (`plan_each`): searches from one source
  settle the same nodes in the same order up to each target, as long as the target is within
  its own REBUILD cost (past it, both give the REBUILD), so each gets the plan its own search
  would, and is cached as such.
- **Shape-only matrices** (permutation matrices `P`, reduction selectors, column maps) depend
  only on the shape, so they are cached too (`_matrix_cache`). It is bounded at 64 items and
  2^25 stored entries in all, about 400 MB. A `P` for a group far larger than the data is
  built from the nonempty rows instead, per call, and not cached.

### Fitted constants

The cost constants are measured, not derived. Each comment says what was timed. They are in:

- `tensor_layout.py`: `C_RS`, `C_RS_ROW`, `C_RS_HYPER_ROW`, `C_T_HYPER`, `T_TABLE`,
  `C_T_TWO_THREADS`, `C_REBUILD`, `C_REBUILD_SORTED`, `C_REBUILD_FIXED`, `C_PERM_ENTRY`,
  `C_PERM_ROW`, `C_PBUILD_FULL`, `C_PBUILD_ROW`, `C_PERM_C_MAP`, `C_PERM_C_ARITH`,
  `C_PERM_C_SORT`, `C_STEP`;
- `tensor.py`, the strategy costs shared by the families: `C_SEL_ENTRY`, `C_SEL_POS`,
  `C_SEL_BUILD`, `C_HROW`, `C_MXV`, `C_VXM`, `C_DIAG`, `C_GATHER`, `C_EWISE_SAME`,
  `C_EWISE_HROW`;
- `tensor_index.py`: `C_X_ROW`, `C_X_LIST`, `C_FILTER`, `C_FILTER_AXIS`, `C_CFILTER`,
  `C_CFILTER_OUT`;
- `tensor_ewise.py`: `C_NONEMPTY_COLS`;
- `tensor_engines.py`: `C_MTTKRP_INTER`.

**All of them were fitted on one machine**: an Apple M5 Pro (6 performance + 12 efficiency
cores; 8 threads span 6 P + 2 E) with SuiteSparse:GraphBLAS 10.5.1, on the uniform synthetic
tensors and the FROSTT tensors named above. Atomics are slow on that machine, which is why
wide transposes and column reductions prefer 2 threads (`C_T_TWO_THREADS`, and the 2-thread
cap in "Reductions"). x86 machines may well differ; see `limitations-and-roadmap.md`.

A census of every decision point (see "Engine selection and the census") checked the model
against forced alternatives. Its ranking of options is right where it matters most (relayout,
reductions, extraction), but its magnitudes are off by 2-5x. A few cases are mispriced; they
are listed in `limitations-and-roadmap.md`. Refitting the constants with a script driven by
the census is a planned step.

### Executing a plan

`execute` runs the edges in order and never changes the source Matrix. Intermediates are
owned, so later reshapes happen in place, which skips a copy of the values. Past 2^31 rows or
columns, `reshape_owned` copies instead, because SuiteSparse 10.5.1's in-place reshape keeps
32-bit indices (SuiteSparse issue 462, fixed in 10.5.2; marked `TODO(SuiteSparse 10.5.2)`).
Reshapes always run in storage order, which also avoids 10.5.1's other reshape bug (order
different from orientation). A final reshape's output gets SuiteSparse's global hyper switch
back when at least half its rows are nonempty (`_restore_hyper_switch`). Otherwise it would
stay hypersparse, and a hypersparse operand sends mxm to a slower Gustavson saxpy.

## Element-wise operations and broadcasting

`tensor_ewise.py`. Same-shape `ewise_add`, `ewise_mult` and `ewise_union` run as one Matrix
operation in a common layout (see "Expressions and storing results"). Infix operators map as
for Matrix: `+` is `ewise_add`, `-` is `ewise_union` with zero defaults, and `*`, `/` and the
comparisons are `ewise_mult`.

Broadcasting follows numpy's shape rules with GraphBLAS semantics:

- **Intersection (`ewise_mult`)** never replicates the operand that covers the result (`big`).
  `_bcast_mult` prices options and takes the cheapest. Most are diagonal scalings, which
  SuiteSparse runs as row or column scaling:
  - **diagonal, cols / rows**: the small operand's axes are exactly one group, possibly after
    a split move (a reshape). The split move pays the per-row lookups of hypersparse rows if
    it creates them.
  - **selector-expanded**: the small operand's axes lie inside a group, and its values are
    expanded over that group by a shape-only selector, cached (`_matrix_cache`).
  - **nonempty rows / nonempty cols**: the same inside a hypersparse row group, or a group
    wider than `max(4 * nnz, 4096)` positions and than what the selector cache holds (2^25
    entries), expanded over the nonempty rows or columns only. Below the cache's bound the
    cached selector is kept: a repeated broadcast pays only a product over the group, where
    the nonempty columns are found again each call (a sort of the column indices; priced
    per entry, `C_NONEMPTY_COLS`). The small operand's values at those rows or columns come
    from `tensor_layout.lookup_sorted`: indexed directly when it is full, through a dense
    table while its positions are within `max(4 * (entries + lookups), 4096)`, else by a
    binary search, so the memory follows the data however long the axis.
  - **gather**: the small operand's values are gathered onto the stored entries through the
    Container (`tensor_layout.gather_operand`), then one same-pattern `ewise_mult`.
  - **relayout, diagonal**: a planned relayout to a layout in another axis order whose group
    is exactly the small operand's axes.
  - **lookup at the entries**, only when none of those fits (past 2^60 positions in total,
    with an op that has no flipped form of its kind, as pow): the small operand's values at
    each stored entry's coordinates (the gather's `tensor_index.lookup`), as a Matrix of
    the big operand's pattern, then one `ewise_mult`. It costs by the entries.

  A one-value operand becomes an `apply`. An empty operand gives an empty result.

- **Union (`ewise_add`, `ewise_union`)** replicates the lower-rank operand over the broadcast
  axes (`_expand`), as numpy semantics require. `_check_expansion` predicts the replicated
  entries in O(1). Past `graphblas.config["tensor_max_expansion"]` (1e9 by default) it raises,
  unless `allow_expansion=True`. The broadcast axes lead the row group, or trail the column
  group when the rows would pass 2^60 positions; when neither fits, or the copies would pass
  2^60 entries, it raises.

`apply` and value `select` run on `_M` in its layout. Index-based selectors are refused
because they would need logical coordinates.

## Reductions

`tensor_reduce.py`, `_reduce_matrix`, works group by group until every reduced axis is gone:

- **All of them:** `reduce_scalar`, into a 0-d Tensor.
- **The whole column group:** `reduce_rowwise`.
- **Part of the column group:** a column selector (`M @ S^T`, shape-only), or a REBUILD with
  the monoid as `dup_op`, whichever costs less.
- **The whole row group:** `reduce_columnwise`, at most 2 threads into 1e5 outputs or more.
  When rows hold about one entry each, a REBUILD instead (4-5x faster). For wide column groups
  at several threads, a reshape that moves one column axis into the rows, then a selector
  (u7: 6.6 vs 55.6 ms at 8 threads).
- **Part of the row group:** a row selector (`S @ M`: shape-only, or built from the nonempty
  rows of a hypersparse group), or a REBUILD, whichever costs less.

Selectors are 0/1 matrices: kron-structured and shape-only (`tensor_layout.selector`,
cached), or built from the nonempty rows of a hypersparse group (`rows_selector`). One mxm
with `(monoid, second)` or `(monoid, first)` reduces any subset of a group's axes in place,
with no relayout. UDT values always take the REBUILD path. `reduce` also takes
aggregators made of one monoid or one semiring (`_reduce_aggregator`), typed as Matrix types
them. Composite aggregators (`mean`) and positional ones raise `NotImplementedError`.

## Contractions and batched products

`tensor_contract.py`, `_contract`. For `tensordot` and the pairwise steps of `einsum`:

- The contracted axes must be `a`'s whole column group and `b`'s whole row group, in the same
  order. `_contract` tries every order of the contracted axes, of the batch axes and of each
  operand's free axes (`_orders`), when there are at most 3 of each. It plans both operands'
  relayouts with `tensor_layout.plan` and takes the cheapest pair. Then one mxm with the
  semiring runs.
  The output layout is `(free a | free b)`, in whatever free order was cheapest; the output
  is never relaid out for its own sake.
- **Batch axes** (in both operands and the output) lead both row groups. `a`'s columns are
  embedded block-diagonally through the Container (`embed_block_diagonal`, which keeps 32-bit
  indices where they fit), and one mxm does every batch. Only `a` is embedded: the product
  keeps `b`'s columns, so it is narrow and needs no unembedding. If the new column indices
  need a wider integer type, the embedding is rebuilt instead. SuiteSparse then picks the
  embedded operand's format again: with few batches it is dense enough to be bitmap.
- `matmul` / `@` follows numpy: 1-d operands are promoted and squeezed back, size-1 batch
  axes broadcast (`_squeezed_batch`), and other batch axes are paired.

## TTV

A contraction with a 1-d operand (`tensor_contract._ttv`) avoids the general path where it
can. The options, by cost:

- `mxv` when the axis is the column group;
- `vxm` when it is the row group, if the column group is at most about 4x the data;
- a weighted **row selector** or **column selector** when the axis lies inside a group;
- each of these possibly after a reshape that moves the split;
- the **general** path (a relayout, then mxm) when it is cheaper;
- past 2^60 positions in total, where neither split moves nor the general relayout exist
  and a selector costs by its group's positions, **entries**: `w` at each entry's
  coordinate (the gather's lookup), one `ewise_mult`, then the axis reduced
  (`_ttv_by_entries`), which costs by the entries.

`mxv` or `vxm` on the stored layout is taken outright, without letting relayouts compete,
because their estimates run low (a relayout plus mxm measured 56 ms against vxm's 36).

## TTM engines

A TTM is `tensordot` of one axis with a 2-d factor. `tensor_engines._ttm_engines` tries the
engines in `_TTM_ENGINES = ("udt", "numba", "products")`. The first two run only for FP64
values, a full FP64 factor, `plus_times`, and at least 2^16 entries
(`_TTM_KERNEL_MIN_NNZ`), because they compile per R on first use.

- **`udt`** (`_ttm_udt`, `tensor_udt.ttm`): one `GrB_mxv` whose values are arrays of R
  doubles. The factor's rows become a full Vector of FP64[R]. The product gives, for each
  nonempty row (each output fiber), its R values, which become the output Matrix through the
  Containers with no copy. It needs the JIT (see "The SuiteSparse JIT"). Where the Numba
  engine also runs, it is taken only when it is faster there: in the stored order (no
  relayout), with R up to 32 (`_TTM_UDT_MAX_R`), and with a hypersparse output (nonempty rows
  under SuiteSparse's hyper switch, 1/16). NELL-2 mode 2 at 1 thread: 139 ms against
  Numba's 174, R = 16.
- **`numba`** (`_ttm_numba`, `tensor_numba.ttm`): a generated kernel over the stored arrays.
  The contracted axis must be the last column axis; the tensor moves to the cheapest layout
  that has it there. Its output arrays (full, bitmap or hypersparse) are handed to SuiteSparse
  without a copy (`tensor_layout.matrix_from_arrays`).
- **`products`**: one product.
  - In the stored order on hypersparse rows with at most half of them nonempty, the rows are
    compacted first (`_ttm_compacted`: `h` dropped through the Container, so the Matrix has
    no empty rows), and the result's rows are given `h` back. On compacted rows SuiteSparse
    runs its bitmap/full kernel instead of a Gustavson saxpy: NELL-2 mode 2 404-430 against
    689-769 ms at 1 thread. Any type and semiring.
  - Otherwise `_contract`. The products run every TTM, so any engine list falls back to them.

## MTTKRP engines

`Tensor.mttkrp(factors, mode)` (and the einsum pattern `'ijk,jr,kr->ir'`) never forms the
Khatri-Rao product. `tensor_engines._mttkrp_recipe` tries
`_MTTKRP_ENGINES = ("numba", "udt", "products")`:

- **`numba`** (`_mttkrp_numba`, `tensor_numba.mttkrp_nd`): kernels generated per R, per
  layout and per mode position, for any number of axes (3 or more). They read the stored
  arrays in place, with both groups nonempty, and need FP64 values, full FP64 factors,
  `plus_times`, and at least 2^16 entries.
  - Each row of the stored Matrix is a fiber of the row axes: its factors multiply once per
    row, and runs of entries that share leading column axes multiply once per run, as a CSF
    tree would.
  - Writes are race-free in one of three ways. When the mode is the first row axis, chunks
    start where the mode changes, so each owns its output rows. When the mode is the first
    column axis and per-thread buffers would be large, each thread owns a range of the
    mode's coordinates and finds its range in every row by binary search (column blocks: no
    buffers, no sum). Otherwise each thread gets an output buffer, summed at the end; past
    `max(R * nnz, 2^22)` entries in all, the kernel declines (next point).
  - When the stored layout does not suit, the tensor is relaid out to the cheapest
    `(mode, ... | last)` layout and the owning kernel runs.

  R is a compile-time constant, which is worth 1.7-1.9x over reading it at run time.

- **`udt`**: the products below, with their first step (a TTM over the last axis) done by
  the array-UDT engine. This is the fast no-Numba path: NELL-2 mode 0 702 to 142 ms at 1
  thread, R = 16.
- **`products`**: one TTM over the last axis, then for each remaining axis a reshape and one
  mxm with the factor's embedding `W[j * R + r, r] = F[j, r]` (`_krp_embedding`). The axis
  order is chosen by relayout cost **plus the intermediates' sizes**
  (`C_MTTKRP_INTER * R * fibers`, the fibers estimated from a sample: `_mttkrp_order`,
  `_sampled_fibers`). Relayout cost alone once picked an intermediate 55x too large.

## einsum

`tensor_einsum.einsum(subscripts, *operands, op=semiring.plus_times)`. The semiring's
multiply combines the operands and its monoid sums over the contracted indices. The plan
depends only on the subscripts, the shapes and `optimize`, so it is cached
(`_einsum_plan`, an LRU of 1024). It gives one of three routes:

- **The MTTKRP pattern** (one operand of 3 or more axes and 2-d factors sharing a rank index):
  `Tensor.mttkrp`.
- **One contraction already in the output's order** (`_direct_axes`): the contraction's own
  recipe runs on einsum's target, so a mask or accumulator reaches the product.
- **Pairwise**: the order comes from `opt_einsum.contract_path` when opt_einsum is installed
  and there are 3 or more operands, else from a greedy choice by result size (`_einsum_path`).
  In each pair (`_einsum_pair`), indices found in only one operand and not needed later are
  summed out first. Shared indices that are kept become batch axes, the others are
  contracted, and a pair with no contracted index is an element-wise product with
  broadcasting. The last contraction runs on the target.

Not supported: `...`, and an index repeated within one operand (diagonals).

## Reshape

`tensor_shape.py`. `reshape(shape, copy=None)` is numpy's C-order reshape, with -1.

- It is an **O(1) view** when every run of old axes that maps to a run of new axes
  (`_reshape_chunks`) lies within one layout group, in order (`_reshape_layout`).
- Otherwise the data moves once: a planned relayout to a layout that holds the result
  (`_reshape_targets`), then a reshape.
- `copy=False` raises where data would move; `copy=True` always copies.
- `ravel` is `reshape(-1)` and `flatten` is `reshape(-1, copy=True)`.

## Extraction

`T[...]` is **outer** (orthogonal) indexing, as for Matrix: ints drop their axis; slices,
lists and integer ndarrays keep it; negative indices wrap once; GraphBLAS Scalars index as the
ints they hold. `None` and boolean indices are refused. (`tensor_index._extract_expr`,
`_extract_recipe`.)

An outer region of any layout is an I x J submatrix of the stored Matrix: each group's
positions are the raveled outer product of its axes' indices (`_group_index`).
`_extract_recipe` chooses by cost among:

- **extract**: one SuiteSparse extract with those index lists. With slices only, this is
  taken directly. It is priced by the rows it visits (a listed row is found directly, or by a
  hash lookup when the Matrix is hypersparse), plus, for a column list shorter than 1/32 of
  the columns, a hash lookup per entry of those rows (`_extract_cost`).
- **column range from the stored arrays**: a range of the columns of every row of a sparse
  (not hypersparse) Matrix, by one numpy pass over its column indices, read where they are
  through the Container (the Matrix is not changed, and only the kept entries are copied),
  the kept entries' rows found by a binary search of the row pointers (`_column_range`). The extract visits
  every row there, empty or not: Uber's 5e6 rows hold 3.3e6 entries, and its slab along the
  last axis took 31 ms at 1 thread against 1.2.
- **filter**: `to_coo`, a numpy mask on the coordinates, a rebuild. This is cheaper than
  listing many indices (a slab of a wide column group), and it is the only path when a group
  would list more than 2^26 positions (`_MAX_LISTED`).
- **hypersparse row pick**: for ints and ranges inside a hypersparse row group, pick among the
  nonempty rows (`h`) and run `P @ M`. NELL-2 range slices: 17 against 36 ms at 8 threads,
  32 against 134 at 1.

Repeated indices repeat the entries, as for Matrix.

## Assignment

`tensor_index.assign`. `T[idx] << v`, `T[idx] = v` and `del T[idx]` follow Matrix
assignment:

- `T(mask, accum, replace)[idx] << v` is GrB_assign, with the mask of `T`'s shape.
- `T[idx](mask, accum, replace) << v` is GxB_subassign, with the mask of the region's shape.
- The value (a Tensor, Matrix, Vector, Scalar or Python scalar) is broadcast to the region's
  shape by numpy rules.
- The region takes the value's structure: a position missing from the value is deleted, and
  a scalar fills the region.
- With repeated indices the last occurrence wins, as SuiteSparse does for Matrix.

Because the region is an I x J submatrix (see "Extraction"), each assignment is **one**
SuiteSparse assign or subassign on the stored Matrix. It runs in place after copy-on-write,
with the value and the mask relaid out to the region's matching layout. A region listing more
than 2^26 rows or columns raises, with a hint to relayout.

## Gather

`take`, `take_along_axis` and `T.vindex[...]` (`tensor_index.py`) gather numpy-style. Unlike
`T[...]`, they **compute at once** and return a Tensor; a mask can still be applied with
`C(mask) << R`.

- Negative indices wrap once, unless `wraparound=False`, which raises IndexError.
- `take`'s `mode` is numpy's: `"raise"`, `"wrap"` or `"clip"` for indices out of range.
- A position that lands on a missing entry is missing.
- Index arrays are ints, Scalars and integer ndarrays. Boolean, Vector and Tensor indices are
  refused.

Engines, in `_GATHER_ENGINES = ("extract", "numba", "graphblas")`:

- **`extract`** (`take` only): `take` along one axis is an outer extraction followed by a
  reshape, so it never enumerates the output positions.
- **`numba`** (`_lookup_numba`, `tensor_numba.lookup`): a binary search for each point in its
  row's column indices, over the Container's arrays, parallel over points. Bitmap and full
  storage are indexed directly. About 15-28 ms per 1e6 points.
- **`graphblas`** (`_lookup_graphblas`): the distinct points as an iso pattern Matrix,
  `ewise_mult(first)` with the stored Matrix, then each point's hit by binary search. This is
  the no-Numba path, about 5-6x slower than the Numba lookup at 1e6 points.

## Input and output

`tensor_io.py`. `Tensor.from_pydata_sparse` / `to_pydata_sparse` (COO, GCXS, DOK) and
`Tensor.from_scipy_sparse` / `to_scipy_sparse` (n-d `coo_array`), named and with arguments
as `graphblas.io`'s functions. `graphblas.io` itself is unchanged and still refuses more than
2 dimensions.

- **GCXS** maps directly onto the Tensor's storage: its compressed axes, sorted, are the row
  group, and the other axes, sorted, the column group (`_gcxs_states`). So it converts with
  one copy of each array. NELL-2 at 8 threads: from GCXS 134 ms against `from_coo`'s 335.
- Explicit zeros are kept. Duplicates are summed, or combined with `dup_op`.
- A pydata `fill_value` other than 0 raises, because a Tensor has no fill value.
  (`graphblas.io.from_pydata_sparse` silently drops it today; see `backport-candidates.md`.)

## Interop with Matrix, Vector and Scalar

Tensor rules apply whenever a Tensor is involved. The core hooks live in their own small
changes:

- `graphblas.core.utils._tensor_types` lists the Tensor classes, so core modules need not
  import the Tensor.
- Infix operators on Matrix, Vector and Scalar (`infix._ewise_infix_expr`,
  `_matmul_infix_expr`, `infixmethods.call_op`) return `NotImplemented` when either operand
  is a Tensor, and Python then calls the Tensor's reflected operator. A Matrix is then a 2-d
  Tensor and a Vector a 1-d one, aligned on trailing axes as numpy aligns them.
- In-place operators on a Matrix with a Tensor operand raise. Returning `NotImplemented`
  would let Python rebind `A` to a Tensor expression.
- Matrix and Vector methods keep their rules and raise with a hint when given a Tensor
  (`base._expect_type_message`). `isequal` and `isclose` are the exceptions: they return what
  the Tensor's methods return.
- A Tensor mask on a Matrix or Vector output raises (`base.py`, `mask.py`). Masks combine
  across kinds (`T.S & A.V` is a Tensor mask).
- Named operators dispatch to the Tensor (`operator/base.py`, `operator/unary.py`,
  `operator/agg.py`): `binary.plus(T, 1)`, `unary.abs(T)`, `agg.sum(T)`, and the infix forms
  `binary.plus(A | T)` and `semiring.min_plus(A @ T)`. As for Matrix, two collections go
  through the infix form: `binary.plus(A, T)` raises, as `binary.plus(A, B)` does.

## Engine selection and the census

The code keeps several implementations where measurements justify them. The maintainers'
principle: build the alternatives, measure them, keep every engine selectable, and decide by
data. Engines are lists that tests and benchmarks can set (`_TTM_ENGINES`, `_MTTKRP_ENGINES`,
`tensor_index._GATHER_ENGINES`). Strategies chosen by cost can be forced through the census.

`tensor_census.py` records every decision point. It is private and off by default: each
point tests `tensor_census.on` first, and costs one attribute read when off. On (`start()`,
or the environment variable `GRAPHBLAS_TENSOR_CENSUS` naming a file for JSON lines), each
decision records:

- its choice, and every option's estimated cost where it chooses by cost;
- forcing: `force[point] = label` takes another option wherever it is offered, and relayouts
  can be forced to plan without permutations or to REBUILD.

The census ran over the test suite (103k decisions), 576 benchmark rows (u7, skew, hyper,
Zipf, Uber, NIPS, part of Enron, each alternative forced and timed), and real workloads. What
it kept, one line each ("margin" is the forced alternative's time over the choice's):

- **TTM `numba`**: the most chosen TTM engine; keep.
- **TTM `udt`**: the only fast TTM without Numba (1.5-5.6x over the products); keep, taken
  with Numba only in its niche (see "TTM engines").
- **TTM `products`**: runs every case; its compacted-rows branch keeps a 1.6-1.7x niche
  (NELL-2 mode 2 without Numba or the JIT, any semiring).
- **MTTKRP `numba`**: median margin 5.6x; keep.
- **MTTKRP `udt` step**: the no-Numba path, 1.1-2.4x over the products on u7 and skew and 5x
  on NELL-2 mode 0; keep.
- **MTTKRP `products`**: the fallback; keep.
- **Gather `numba`**: 5-6x over the GraphBLAS lookup at 1e6 points; keep.
- **Gather `graphblas`**: the no-Numba path; keep.
- **Gather `extract`**: `take` on large tensors would otherwise enumerate every output
  position; keep.
- **Broadcast diagonal options**: median margins 2-6x; keep. The relayout option is offered
  only into another axis order (same-order targets are the split moves).
- **Element-wise layout choice**: keep.
- **Reduction selectors**: 4-9x; keep.
- **Reduction REBUILD**: for about one entry per row, and for UDTs; keep.
- **Column reductions capped at 2 threads**, and the wide case by reshape then selector:
  keep.
- **TTV options**: keep (pricing issues only).
- **Extraction**: extract, filter and the hypersparse row pick (2.1x and 4.2x on NELL-2 range
  slices); keep all three.
- **Planner edges**: RS and T, keep (a forced REBUILD is 2.5x slower median, up to 36x).
  PERM_R, keep (up to 19x). PERM_C, keep (1.71x median). REBUILD is the fallback.
- **Plan cache and kernel disk cache**: keep (the Tensor tests take 10-12 s warm against
  45-51 s cold).
- **einsum**: the direct route, the MTTKRP pattern and the pairwise route; keep.
- **Batched products**: block-diagonal embedding; keep.
- **Default-layout heuristics**: keep, with the three "4 entries per row" rules merged into
  one (`FULL_ROW_ENTRIES`).

## Numba kernels and the kernel cache

Numba is optional: `graphblas.core._has_numba` gates every kernel, and every caller has a
GraphBLAS fallback. `tensor_numba.py` generates kernel source per case (R, the group sizes,
the kind of write) and compiles it with `njit(parallel=True)`. The kernels read the stored
Matrix's arrays in place through the Container (`tensor_layout._Unloaded` and `_VectorArray`)
and hand their output arrays back the same way, unchecked. That is why the tests compile them
with Numba bounds checks (the `numba_boundscheck` fixture in `test_tensor.py`), and why the
TTM checks validate every result with `GxB_Matrix_fprint`.

Compiling costs 0.2-0.6 s per kernel, so kernels are cached:

- In the process (`_compiled`).
- On disk (`_compile`, `_load_cached`): the source is written to a file named by its SHA-256
  under `$GRAPHBLAS_KERNEL_CACHE`. The default is `python-graphblas/tensor-kernels` under
  `$XDG_CACHE_HOME`, or under `$HOME/.cache`. An empty value turns the cache off.
- A cached file runs only if its bytes equal the generated source; otherwise the kernel
  compiles in memory and the file is left alone. Writes go through a temporary file and a
  rename. A failed rename onto a file another process just wrote with the same source uses
  that file (the Windows race).
- Numba's own `cache=True` then caches the compiled code beside it.

## The SuiteSparse JIT

The maintainers treat the SuiteSparse JIT as on by default: performance is judged with it on.
Every path must stay correct with it off.

- Internal products choose semirings with factory kernels where the JIT may be off. For
  example, `min_second` for permutations: SuiteSparse 10's factory has no `any_second`, which
  runs SuiteSparse's generic kernel when the JIT cannot compile it (1.3-2.3x slower, measured
  with the JIT at `"run"`).
- **The array-UDT engine** (`tensor_udt.py`) needs the JIT:
  - It makes its own type per R, with a fixed C name (`_gbtensor_fp64x16`), so every process
    finds compiled kernels again in the JIT cache. An anonymous UDT gets a per-process
    `_gbudt_N` name instead.
  - Its operators have a NULL function pointer and the C definitions that
    `operator/udt_codegen.py` generates (`GxB_BinaryOp_new`), so no Numba is involved.
  - Like UDT auto-lift, it raises `jit_c_control` from SuiteSparse's default `"run"` to
    `"on"` when a compiler works (`jit_ready`), and honors an explicit `"off"`, `"pause"` or
    `"load"`. It raises it only once it has decided to run: before that, `jit_possible`
    answers with no side effect, so a TTM it leaves to Numba leaves the control at `"run"`.
- **A failed compile** returns `GxB_JIT_ERROR`, and SuiteSparse drops the control to
  `"load"`. `tensor_udt.run` catches that: it leaves the control at `"load"`, as UDT
  auto-lift does after a failed compile (putting back `"run"` made auto-lift raise it to
  `"on"` at the next UDT operation, which retried the compile and raised), warns once
  (`NoJITWarning`), and turns the engine off for the process. The next engine then runs.
- Lifted operators between a scalar and an array or record UDT get JIT C definitions in core
  (`operator/udt_codegen.py`, `operator/binary.py`). This is a separate change on this
  branch, also usable by Matrix and Vector.

## Module map

All under `graphblas/core/`. `tensor.py` imports the family modules at its end, and each
family imports what it needs from the hub at its own end (`# isort: off` keeps them there).
Knobs that tests set (`_has_numba`, the engine lists, the kernel thresholds, `_MAX_LISTED`,
the engine functions) are read through the hub as `_t.NAME`, so setting them on
`graphblas.core.tensor` works.

- **`tensor.py`**, the hub: `Tensor` and its methods; `TensorExpression`, `TensorIndexExpr`
  and the infix classes; `_store`, `_fill`, `_output_mask`, `_mask_in_layout`,
  `_fixed_layout`, `_choose_layout`, `_in_layout`; copy-on-write (`_Owners`, `_Store`); the
  default-split heuristics (`_rows_ok`, `_data_split_layout`, `_sampled_nonempty_rows`,
  `_settle_split`, `_healthy`); the shared strategy costs (`C_*`); the engine knobs
  (`_TTM_ENGINES`, `_MTTKRP_ENGINES`, `_TTM_UDT_MAX_R`, `*_KERNEL_MIN_NNZ`, `_MAX_LISTED`);
  the repr; `_check_backend`.
- **`tensor_layout.py`**: `Layout`; raveling; by-row Matrix helpers and the Container calls
  (`_Unloaded`, `_VectorArray`, `flip_orientation`, `matrix_from_arrays`, `give_rows`,
  `nonempty_row_ids`); the planner (`plan`, `_neighbors`, the cost constants, `relayout`,
  `execute`, `relayout_cost`); `perm_rows`, `perm_cols`, `rebuild`, `transpose`,
  `reshape_owned`; `selector`, `rows_selector`; `embed_block_diagonal`; `gather_operand`;
  the caches.
- **`tensor_shape.py`**: reshape (`_reshape`, `_reshape_chunks`, `_reshape_layout`,
  `_reshape_targets`).
- **`tensor_ewise.py`**: `_ewise_expr`; broadcasting (`_bcast_mult`, `_expand`,
  `_check_expansion`, `_squeezed`, `_lift_rank`); the ewise, select and apply recipes.
- **`tensor_reduce.py`**: `_reduce_expr`, `_reduce_aggregator`, `_reduce_matrix`,
  `_reduce_rebuild`, `_reduce_scalar_recipe`.
- **`tensor_contract.py`**: `tensordot`, `_contract`, `matmul`, batched products, `_ttv`.
- **`tensor_engines.py`**: the TTM engines (`_ttm_engines`, `_ttm_udt`, `_ttm_numba`,
  `_ttm_compacted`) and MTTKRP (`_mttkrp_recipe`, `_mttkrp_numba`, `_mttkrp_order`,
  `_sampled_fibers`, `_krp_embedding`).
- **`tensor_einsum.py`**: `einsum`, `_einsum_plan`, `_direct_axes`, `_mttkrp_pattern`,
  `_einsum_path`, `_einsum_pair`.
- **`tensor_index.py`**: extraction (`_extract_expr`, `_extract_recipe`, `_group_index`,
  `_hyper_rows_pick`), assignment (`assign`, `TensorUpdater`, `TensorAssigner`) and gather
  (`take`, `take_along_axis`, `vindex`, `gather`, the lookups).
- **`tensor_numba.py`**: the generated MTTKRP and TTM kernels, the point lookup, the kernel
  caches.
- **`tensor_udt.py`**: the array-UDT TTM engine and its JIT policy.
- **`tensor_io.py`**: the pydata/sparse and scipy conversions.
- **`tensor_census.py`**: the decision census (private, off by default).

Outside these modules, the Tensor touches:

- `graphblas/__init__.py`: the lazy `gb.Tensor` and `gb.einsum` exports;
- `graphblas.yaml`: the `tensor_max_expansion` setting;
- the interop hooks listed in "Interop with Matrix, Vector and Scalar";
- `core/ss/jit_config.py`: finding conda compilers on Windows;
- the docs: `docs/user_guide/tensor.rst` and `docs/api_reference/collections.rst`;
- the tests: `graphblas/tests/test_tensor.py`, `test_tensor_io.py`, and the Tensor checks in
  `test_core.py` and `test_ssjit.py`.

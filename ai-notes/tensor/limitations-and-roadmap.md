# Tensor: limitations and roadmap

What the experimental `gb.Tensor` does not do yet, what has not been tested, and the order in
which the maintainers plan to work on it. Timings are written "8 / 1" for 8 and 1 threads,
and the data sets are those named in `design.md`.

## Not tested

- **Windows.** Nothing has run on Windows. Specifically:
  - **The kernel cache's file locking.** A rename onto a file another process holds open
    raises there; the cache then uses that process's file if it holds the same source, or
    compiles in memory. That was tested by simulating the failure, not on Windows.
  - **The JIT compiler flags.** `fix_jit_config` now finds conda's compilers under
    `Library\bin` and with `.exe` names. But the flags it repairs are POSIX-minded
    (`-isysroot`, `-arch`), and whether conda-forge's Windows SuiteSparse JIT compiles that
    way is unknown. If it does not, the array-UDT engine warns once and turns itself off,
    and the Numba or GraphBLAS engines run: correct, slower.
  - **Numba's `parallel=True`.** It should work with its workqueue or TBB layer (the kernels
    use only `prange` and plain arrays), but it has not run.
  - **numpy 1.x's 32-bit default integer.** Coordinates are converted to int64 explicitly,
    and no arithmetic on a default-int array that could pass 2^31 was found by reading. Not
    run.
- **x86 planner constants.** The cost model was fitted on one machine, an Apple M5 Pro
  (`design.md`, "Fitted constants"). Atomics are slow there, which is why wide transposes
  and column reductions prefer 2 threads. That rule, and the rest of the ranking, may differ
  on x86.
- **Linux, and SuiteSparse other than 10.5.1.** The branch has not run in CI.
- **NELL-2 and Enron were not fully re-censused.** The census of decision points covered the
  test suite, u7, skew, hyper, Zipf, Uber, NIPS and part of Enron. Its later re-measurement
  finished only u7, because machine load spoiled the rest. For NELL-2, the TTM and MTTKRP
  engine choices rest on earlier forced-engine measurements, and only its slices were rerun
  after the last changes. Uber's and NIPS's contraction and slice rows were not run by the census
  (Uber's slices were measured later, with the benchmark kit: see "Known performance gaps").
- **The independent oracles** are not in the repository (see `verification.md`), so the
  checks they made cannot be rerun from it yet. The benchmark harness and the workload scripts
  are, in `scripts/tensor_bench/` (`benchmarks.md`).

## Missing features

- `None` (new axis) and boolean indices; Vector or Tensor index arrays (undecided: see
  `decisions.md`).
- `einsum` with `...` or a repeated index within one operand (diagonals).
- Composite aggregators (`agg.mean`, `agg.varp`) and positional ones (`agg.ss.argmin`) in
  `reduce`.
- Index-based `select` (`rowindex`, `tril` and the like), which needs logical coordinates.
- `pad`.
- binsparse interchange; `graphblas.io` dispatching ndim > 2 to the Tensor methods.
- An Array API namespace.
- `Tensor.dtype` is a python-graphblas type, not a numpy dtype, so libraries that expect
  numpy dtypes (TensorLy, for example) need a small adapter.

## Known performance gaps

- **TTM along an axis other than the stored order** relayouts the tensor and then runs the
  kernel. In the last full measurement, made before the latest planner refits, it was
  1.9-2.7x behind TACO at 1 thread on u7, and 6.8-7.4x on NELL-2 modes 0 and 1. TACO's
  numbers are for a copy of the tensor already in that mode's order, built outside its timing
  (7.6 s on NELL-2). An in-place kernel for those modes, or opt-in per-mode copies, would
  close it.
- **Slices of Uber (4-d, layout (0, 1, 2 | 3))** are 2.2-16x behind pydata/sparse and scipy at 1
  thread, and 2.9-11x at 8 (`benchmarks.md`, the fresh-clone run). Each slab touches a few
  entries in many rows of the stored Matrix.
- **einsum's per-call overhead** shows at millisecond scale: about 45 us for one contraction,
  97 us for an MTTKRP through einsum, and 128 us for three operands.
- **Moved operands are not kept.** A relayout made for one operation is dropped afterwards, so
  repeated contractions of the same operand repay it. Keeping it on the store needs a memory
  policy.
- **4-d MTTKRP** is within 1.0-1.4x of SPLATT at 8 threads (geometric mean on Uber, NIPS and
  Enron). The worst row is 1.66x (Uber mode 3, R = 16).
- **Batched products with few large batches** can trail a plain loop: multiplex PageRank on
  the Higgs data set (4 layers) is 1.4-1.8x slower as one batched `'lu,luv->lv'` than as four
  `vxm`.
- **Known mispricings** found by the census and not yet refitted:
  - a broadcast multiply on Zipf at 8 threads (29 to 36 ms after the last refit: clustered
    rows, priced as uniform);
  - materializing Enron's `permute_dims((0, 2, 1, 3))` at 1 thread: the planned chain is
    1.23x behind a REBUILD;
  - the TTM engine rule: on u7 mode 2 at 1 thread and skew mode 0 at 8, the array-UDT engine
    is 1.27-1.38x faster than the one chosen, and on hyper mode 0 at 8 threads the products
    are 1.2x faster than Numba;
  - Uber's two-axis sums (a REBUILD where the column selector is 4x faster), NIPS's TTV
    column selector (1.6x behind mxv after a reshape), and Zipf's extract against the
    coordinate filter (2.2x).

## Roadmap, in order

1. **Maintainer review** of the API (`decisions.md`) and of the riskier code: copy-on-write,
   the Container calls that hand arrays to SuiteSparse unchecked, the planner's pricing, and
   the engine rules.
2. **CI on every platform** python-graphblas supports, with a SuiteSparse 10 job on each.
   This closes the Windows questions above, and runs the suite without Numba and with the JIT
   off there too.
3. **Bring the rest of the verification into the repository:** the independent oracles, as
   slow tests, so that anyone can rerun them (the benchmark kit is in `scripts/tensor_bench/`,
   and asv benchmarks for the Tensor in `benchmarks/tensor_ops.py`).
4. **Refit the cost constants** on x86, and with a script driven by the census instead of by
   hand. Then fix the mispricings listed above, and finish the census on NELL-2 and Enron.
5. **Performance:** TTM along non-stored modes (an in-place kernel, or opt-in per-mode copies);
   einsum's per-call overhead; keeping moved operands on the store.
6. **API:** `pad`, more aggregators, index-based `select`, einsum diagonals, and the open
   questions (Vector and Tensor index arrays, `graphblas.io` dispatch).
7. **When SuiteSparse 10.5.2 is the oldest supported version:** drop the 10.5.1 reshape
   workaround (`tensor_layout.reshape_owned`, marked `TODO(SuiteSparse 10.5.2)`).
8. **Backports** to Matrix and Vector (`backport-candidates.md`).
9. **Later:** binsparse, when its protocol settles; an Array API namespace.

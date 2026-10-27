# Tensor: benchmarks

How the experimental `gb.Tensor` compares with other sparse-tensor libraries, how sure we are
of the numbers, and how to rerun them. Everything here can be reproduced with the kit in
`scripts/tensor_bench/` (its README has the details). The asv suite in `benchmarks/`
(`tensor_ops.py`) tracks the Tensor's own regressions; it is not a comparison.

Times are best-of-N wall times in milliseconds, written "8 / 1" for 8 and 1 threads.
"Faster" ratios are the other library's time over the Tensor's (above 1: the Tensor is
faster); "GB / X" ratios are the other way round. The data sets are those of `design.md`:
uniform u6 and u7 (1000^3 with 1e6 and 1e7 entries), skew (1e5 x 1e3 x 10, 1e7), hyper
((2^19)^3, 1e6), Zipf ((1e5)^3, 1e7, each mode Zipf-distributed), and the
[FROSTT](http://frostt.io/) tensors NELL-2 (3-d, 77M entries), Enron (4-d, 54M), Uber (4-d,
3.3M) and NIPS (4-d, 3.1M).

## The machine, the versions, and which run each number comes from

Everything ran on one machine: an **Apple M5 Pro** with 6 performance and 12 efficiency cores
(macOS 26.6 names the two clusters "Super" and "Performance"), 48 GB, macOS 26.6.2. Eight
threads span the 6 fast cores and 2 slow ones. Another workload shared the machine, so every
result records the 1-minute load average, and each table below states the load it ran at.

| component             | version                                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| SuiteSparse:GraphBLAS | 10.5.1 (python-suitesparse-graphblas 10.5.1.1), JIT on, compiled with conda-forge clang 21.1.8                                                  |
| Python stack          | Python 3.14.8, numpy 2.5.4, Numba 0.68.0                                                                                                        |
| pydata/sparse         | 0.19.2 (COO, and GCXS for reductions and permutations)                                                                                          |
| scipy                 | 1.18.1 (n-d `coo_array`)                                                                                                                        |
| torch                 | 2.14.1, CPU build from conda-forge (OpenMP), `torch.sparse` COO                                                                                 |
| SPLATT                | [ShadenSmith/splatt](https://github.com/ShadenSmith/splatt) 6cb8628 (2022-03-03), 64-bit indices, double, OpenMP, clang 21.1.8                  |
| TACO                  | [tensor-compiler/taco](https://github.com/tensor-compiler/taco) 0e79acb5 (2025-04-14), Release, OpenMP; kernels compiled with `-O3 -ffast-math` |
| Finch                 | [Finch.jl](https://github.com/finch-tensor/Finch.jl) 1.4.0 on Julia 1.12.6, one thread                                                          |
| hand-written Numba    | parallel TTV, TTM and MTTKRP kernels over per-mode sorted copies (the copies are built outside the timing)                                      |

The Tensor changed while it was measured, so each number below names its run, and each run
the commit it measured (by letter; the commits are named below by subject, which a rebase
keeps, unlike a commit id):

| run                | Tensor at | what it measured                                                                                                                                            | 1-minute load  |
| ------------------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 2026-10-20 morning | A         | pydata/sparse, scipy, torch, hand-written Numba and the Tensor, every kernel group on every data set. The other libraries' numbers below come from this run | 3-100 (bursts) |
| 2026-10-20 16:50   | B         | the Tensor again, on every data set and group                                                                                                               | 4.5 median     |
| 2026-10-20 evening | B         | SPLATT, TACO and Finch, and the Tensor's rows on u7 and NELL-2 again                                                                                        | 1.3-3.3        |
| 2026-10-20 19:50   | C         | the n-d MTTKRP kernel on u7, NELL-2, Uber, NIPS and Enron                                                                                                   | 3.0-5.2        |
| 2026-10-21 04:30   | D         | TTM and MTTKRP with each engine forced, on u7 and NELL-2                                                                                                    | 3.8-6.4        |
| 2026-10-21         | E to F    | the census of decision points (every alternative forced and timed), and the refits it led to                                                                | 2-18           |
| 2026-10-21 evening | G         | the kit from a fresh clone: u7 and Uber, the Tensor against pydata/sparse and scipy (last section)                                                          | 1.4-2.7        |

- A: "Tensor: lift broadcast operands to the result's rank before choosing a strategy"
- B: "Tensor: price a sorted REBUILD fallback only where reshape is unavailable"
- C: "Tensor: test MTTKRP on layouts with an empty group"
- D: "Tensor: UDT engine, no bitmap request on the empty output; test a new kernel failing"
- E: "Tensor: a private census of the decision points (off by default, for measurement)"
- F: "Tensor broadcast: a split move into hypersparse rows pays the diagonal's row lookups"
- G: "scripts/tensor_bench: a portable benchmark kit for the experimental Tensor"

The commits after F split `tensor.py` into modules and edited text only; an AST comparison
with the docstrings removed found no code change.

## Correctness: fingerprints

Every timed result carries a fingerprint of its output: the shape, the entry count, an
order-independent hash of the entries' linear indices, the sum of the values and a weighted
sum (`scripts/tensor_bench/harness/common.py`). A result counts only if its fingerprint agrees
with the Tensor's for the same case: counts and hashes exactly, sums to 1e-8 relative. Inputs
are positive, so libraries with and without missing-value semantics compare directly. The C,
C++ and Julia drivers compute the same fingerprint.

| run                      | cross-library pairs (large data)                                | against dense numpy (small data)  | mismatches |
| ------------------------ | --------------------------------------------------------------- | --------------------------------- | ---------- |
| 2026-10-20 morning       | 2,040                                                           | 756                               | 0          |
| 2026-10-20 evening       | 606                                                             | 272 (every driver, R = 16 and 64) | 0          |
| 2026-10-20 19:50         | 828 (against the evening's SPLATT and Tensor results)           |                                   | 0          |
| 2026-10-21 04:30         | 1,688 (TTM 392, MTTKRP 1,296; every engine)                     |                                   | 0          |
| 2026-10-21 evening (kit) | 292 (and 820 against the 2026-10-20 results for the same cases) |                                   | 0          |

Each check has a teeth mode that plants wrong fingerprints and must flag them
(`report.py --teeth`). Each run's raw results were also re-checked by a separate script that
shares no code with the harness. These comparisons found no correctness bug in the Tensor;
the bugs they found were in other libraries (see "What wins and what loses").

## Against pydata/sparse, scipy and torch

From the 2026-10-20 16:50 run (the Tensor at B), against the morning's other libraries.
For each row, the fastest of torch at 8 threads, scipy, pydata COO and GCXS, over the Tensor
at 8 threads; geometric means:

| data set | rows | Tensor faster | other faster | geomean |
| -------- | ---- | ------------- | ------------ | ------- |
| u6       | 42   | 26            | 16           | 2.4x    |
| u7       | 41   | 37            | 4            | 5.9x    |
| skew     | 40   | 36            | 4            | 6.9x    |
| hyper    | 33   | 22            | 11           | 1.5x    |
| Zipf     | 35   | 28            | 7            | 5.5x    |
| NELL-2   | 31   | 28            | 3            | 8.4x    |

| kernel group           | geomean | range     |
| ---------------------- | ------- | --------- |
| element-wise add / mul | 24x     | 8.5-68x   |
| contractions           | 15x     | 1.8-2500x |
| permute + materialize  | 13x     | 2.5-118x  |
| MTTKRP                 | 7.9x    | 0.59-30x  |
| TTV                    | 4.2x    | 0.43-79x  |
| reductions             | 3.1x    | 0.12-210x |
| construction from COO  | 3.0x    | 0.96-6.3x |
| TTM                    | 1.9x    | 0.71-5.3x |
| slices                 | 1.6x    | 0.04-21x  |
| broadcast multiply     | 1.6x    | 0.13-9.3x |

Export to COO is left out: it is a copy for the Tensor (22 ms on u7) and free for the COO
libraries, which return their arrays. TTM and MTTKRP have improved since (below). The u7 and
NELL-2 rows were run again at a lower load (1.3-2.7) that evening, 8-10% faster at both
thread counts.

Some kernels the other libraries could not run at NELL-2's size. TTM needs dense outputs of
14-34 GB in scipy, pydata and torch. For MTTKRP, pydata's `einsum` materializes 1.2e9
entries, and torch's gather needs about 30 GB. The Tensor runs both in a fraction of a
second, with sparse outputs.

## MTTKRP against SPLATT and TACO

MTTKRP is the kernel of CP decompositions, and [SPLATT](https://github.com/ShadenSmith/splatt)
is its reference implementation. The Tensor's Numba kernel reads the stored arrays in place,
in whatever layout the tensor is stored (`design.md`, "MTTKRP engines"). SPLATT is given its
best tree allocation per row: usually one CSF tree per mode, rooted at that mode, which takes
3-4x the Tensor's memory (NELL-2: 3.8 GB of trees against the Tensor's 0.9 GB). TACO gets a
copy rooted at the output mode, built outside its timing. GB / SPLATT, geometric mean per data
set:

| data set     | 8 threads | 1 thread | run                                 |
| ------------ | --------- | -------- | ----------------------------------- |
| u7 (3-d)     | 1.13      | 0.98     | 2026-10-21 04:30 (D), R = 16 and 64 |
| NELL-2 (3-d) | 0.75      | 0.88     | same                                |
| Uber (4-d)   | 1.31      | 1.09     | 2026-10-20 19:50 (C)                |
| NIPS (4-d)   | 1.38      | 0.99     | same                                |
| Enron (4-d)  | 1.00      | 0.90     | same                                |

Selected rows (ms, 8 / 1 threads):

| case                  | Tensor      | SPLATT, best allocation | SPLATT, one tree | TACO        | Finch (1 thread) |
| --------------------- | ----------- | ----------------------- | ---------------- | ----------- | ---------------- |
| u7 mode 0, R = 16     | 4.55 / 23.3 | 4.74 / 29.9             | 4.84 / 31.1      | 5.59 / 34.1 | 39.9             |
| u7 mode 0, R = 64     | 16.8 / 104  | 13.9 / 86.4             | 14.8 / 88.6      | 15.6 / 101  | 106              |
| NELL-2 mode 0, R = 16 | 28.4 / 161  | 45.1 / 279              | 47.4 / 290       | 43.0 / 302  | 313              |
| NELL-2 mode 2, R = 16 | 47.2 / 204  | 44.0 / 301              | 47.0 / 1138      | 44.2 / 308  | 425              |
| NELL-2 mode 2, R = 64 | 159 / 1106  | 155 / 840               | 383 / 1899       | 130 / 885   | 982              |
| Enron mode 0, R = 16  | 28.1 / 165  | 29.9 / 201              | 32.6 / 215       | 51.6 / 269  |                  |
| Enron mode 2, R = 16  | 46.1 / 219  | 61.1 / 391              | 624 / 764        | 39.2 / 270  |                  |
| Uber mode 3, R = 16   | 2.91 / 14.3 | 1.75 / 13.3             | 1.75 / 45.4      | 4.44 / 13.5 |                  |

- At R = 16 the Tensor matches or beats SPLATT. A likely reason, not isolated: its kernels are
  compiled with R as a constant, while SPLATT and TACO read it at run time. At R = 64 it is
  1.2-1.3x behind on u7.
- A mode that is the stored layout's column group (NELL-2 mode 2) was the Tensor's worst row
  until threads got blocks of the output column instead of private buffers (R = 64: 391 to
  146-159 ms).
- The remaining gaps are small 4-d calls at 8 threads (2-8 ms), 1.3-1.7x behind. Fixed costs
  show at that size, and the schedule is static while threads 7 and 8 land on slow cores.
- Without Numba, the GraphBLAS products are 3-17x slower than the kernel. The array-UDT engine,
  which needs the SuiteSparse JIT, falls in between.
- Through TensorLy (`scripts/tensor_bench/demos/cp_als_tensorly.py`), a CP-ALS iteration at
  rank 16 takes 0.073 / 0.017 s on u7, against TensorLy's pydata path at 17 s and SPLATT's own
  `cpd` at 0.107 / 0.018 s. On NELL-2 it takes 0.54 / 0.14 s, against SPLATT's 0.90 / 0.17 s;
  TensorLy's pydata path would form a 34 GB Khatri-Rao product there. Load 5.0-5.8,
  2026-10-21.

## TTV, TTM, sums and element-wise against TACO and Finch

TTV, sums and element-wise ops are from the 2026-10-20 evening run (load 1.4-2.3); TTM is from
the 2026-10-21 04:30 run, with the engines the Tensor picks (load 3.8-5.4). TACO's default
schedule runs any kernel with a sparse output on one thread, and Finch's `@finch` loops are
single-threaded, so the Tensor's 1-thread column is the fair comparison. TACO and Finch were
also given a copy of the tensor with the contracted or summed axis at the leaf, built outside
their timing (2.8-3.3 s per mode on NELL-2). The Tensor starts from its one layout, (0, 1 | 2).

| case (ms)              | Tensor 8t | Tensor 1t | TACO 1t | Finch 1t |
| ---------------------- | --------- | --------- | ------- | -------- |
| u7 TTV mode 0          | 7.97      | 29.9      | 7.77    | 10.3     |
| u7 TTV mode 2          | 1.34      | 8.01      | 8.12    | 10.3     |
| u7 TTM mode 0          | 22.1      | 75.0      | 29.5    | 39.0     |
| u7 TTM mode 2          | 8.42      | 26.9      | 29.4    | 38.8     |
| u7 add                 | 16.8      | 110       | 121     | 100      |
| NELL-2 TTV mode 0      | 82.9      | 472       | 91.4    | 140      |
| NELL-2 TTV mode 2      | 5.44      | 28.1      | 28.5    | 47.1     |
| NELL-2 TTM mode 0      | 337       | 1736      | 263     | 390      |
| NELL-2 TTM mode 2      | 24.3      | 139       | 278     | 290      |
| NELL-2 sum over axis 0 | 81.3      | 469       | 84.5    | 129      |
| NELL-2 add             | 64.2      | 429       | 405     | 388      |
| NELL-2 multiply        | 57.5      | 394       | 542     | 331      |

- Where the work is the same (the stored order, and element-wise ops), the Tensor at 1 thread
  matches or beats TACO. At 8 threads it is fastest everywhere except u7 TTV mode 0 (by 3%)
  and NELL-2 TTM modes 0 and 1.
- Along the other axes TACO's prepared copy wins at 1 thread: 2.8-5.6x on TTV and sums, and
  1.8-6.6x on TTM. The Tensor relayouts there, inside its timing.

## The census: what each engine and strategy is worth

The Tensor keeps alternative engines and strategies where measurements justify them
(`design.md`, "Engine selection and the census"). The census forced each alternative on 576
benchmark rows (u7, skew, hyper, Zipf, Uber, NIPS and part of Enron) at 8 and 1 threads.
"Margin" is the best forced alternative's time over the choice's, on rows of at least 1 ms:

| decision point                              | margin, median / worst                                   | what runs without it                                          |
| ------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| MTTKRP: Numba kernel                        | 5.6x / 1.58x                                             | the array-UDT step plus products, or products: 1.6-17x slower |
| TTM: Numba kernel                           | 1.23x / 0.71x                                            | the array-UDT engine (needs the JIT), else the products       |
| TTM: array-UDT engine, in its niche         | 1.24x over Numba (NELL-2 mode 2); 1.5-5.6x over products | the only fast TTM without Numba                               |
| gather: Numba binary search                 | 5-6x over GraphBLAS at 1e6 points                        | the GraphBLAS lookup                                          |
| reductions: row and column selectors        | 4-9x                                                     | a REBUILD or a relayout                                       |
| extraction: extract; coordinate filter      | 4.2x / 0.46x; 1.25x / 1.04x                              | the other one                                                 |
| broadcast: diagonal with expanded selectors | 2.1-5.9x / 1.45x                                         | a gather or a relayout                                        |
| element-wise: layout choice                 | 1.6x / 0.70x                                             | the largest operand's layout                                  |
| relayout: reshape and transpose edges       | 2.5x median, up to 36x                                   | a REBUILD                                                     |
| relayout: row permutation edge (PERM_R)     | up to 19x                                                | other plans                                                   |
| relayout: column-group reorder (PERM_C)     | 1.71x / 1.24x                                            | other plans                                                   |
| plan cache and kernel disk cache            | the Tensor tests in 10-12 s warm, 45-51 s cold           | recomputing every plan and kernel                             |

The census also found the mispricings listed in `limitations-and-roadmap.md`. The worst ones
were refitted: Enron's `permute_dims((1, 2, 3, 0))` at 1 thread went from 4.0 to 0.82 s, and
its broadcast along the last axis from 1787 to 369 ms.

## Real workloads

Each workload in `scripts/tensor_bench/demos/` asserts that the Tensor's result equals that of
the per-Matrix code python-graphblas users write today (ms, 1 / 8 threads, 2026-10-21, load
2.9-5.8):

| workload                                                          | Tensor          | per-Matrix loop | other                    |
| ----------------------------------------------------------------- | --------------- | --------------- | ------------------------ |
| Hetionet: 9 two-hop metapaths, counts and overlaps                | 3.8 / 3.6       | 4.9 / 6.1       | scipy 3.0                |
| LBNL traffic: 9 statistics for every minute window, 15 timescales | 300 / 260       | 6,130 / 6,130   |                          |
| LBNL traffic: every hour window, 9 timescales                     | 103 / 68        | 160 / 162       |                          |
| multiplex PageRank, EU airlines (37 layers)                       | 55 / 64         | 234 / 236       | scipy supra-matrix 25    |
| multiplex PageRank, Higgs Twitter (4 layers, 15.4M edges)         | 1.4-1.8x slower | the reference   | scipy supra-matrix 2,704 |
| regular path query on Hetionet, 100 sources, union                | 2.5 / 2.0       | 3.2 / 3.5       |                          |
| the same, per source                                              | 130 / 34        | 320 / 265       |                          |

The Tensor wins these mostly by making one call where the loop pays Python overhead per
Matrix (the minute windows), or by one batched product in place of many small ones. Where
the layers are few and large (Higgs), the loop of `vxm` is faster.

## What wins and what loses

The Tensor is fastest, often by an order of magnitude, at element-wise ops (one sorted Matrix;
SuiteSparse relayouts a second operand stored in another layout), construction from COO
(SuiteSparse's parallel build), materialized permutations (the relayout planner), reductions
and TTV along the stored axes, sparse contractions and batched products, and slabs on the
leading axes. It scales to NELL-2 and Enron where the COO libraries run out of memory. Its
MTTKRP is at SPLATT's level with one copy of the data, where SPLATT's fastest setting uses
three or four.

It loses, with the cause:

- **TTM along an axis other than the stored order**, against a library given a copy in that
  order (TACO: 1.8-6.6x at 1 thread). The Tensor relayouts inside its timing; an in-place
  kernel for those modes is on the roadmap.
- **Hypersparse broadcasts and slices** against torch (hyper `X * v[j]`: 7.1 ms against torch's
  1.33 at 8 threads; slabs along the last stored axis, 2-23x), and Uber's slices against
  pydata/sparse and scipy (2.2-16x at 1 thread). Each touches a few entries of many rows, which
  costs per row in a Matrix and per entry in COO.
- **Small tensors against scipy** (u6's sums and TTVs: scipy 2.3-3.2 ms against the Tensor's
  3.1-6.3 at 8 threads): fixed costs, and scipy returns dense arrays there.
- **Export to COO** (a copy; free for COO libraries), and **sums over all axes** against torch
  (0.45 against 0.43 ms on u7).
- **Small 4-d MTTKRP calls at 8 threads** (1.3-1.7x behind SPLATT, above).
- **einsum's per-call overhead**: about 45 us for one contraction and 97 us for an MTTKRP.

The comparisons also found problems in other libraries:

- torch's `permute` keeping dimension 0 marks its result coalesced although it is not sorted,
  so `coalesce()` skips the sort;
- scipy's n-d `sum` allocates a dense array of ones over the reduced axes (the OS killed it on
  (2^19)^3);
- pydata/sparse's `einsum` materializes the full broadcast product for MTTKRP;
- TACO's command-line tool cannot run kernels at the measured commit (its generated module
  defines `compute` twice).

## Reproducing the numbers

With the kit (`scripts/tensor_bench/README.md`):

```bash
conda env create -f scripts/tensor_bench/environment.yml
conda activate graphblas-tensor-bench
python -m pip install --no-deps --no-build-isolation -e .
cd scripts/tensor_bench
python run.py --datasets u6,u7,skew,hyper,zipf                   # synthetic data
python fetch_data.py frostt-nell2 frostt-uber frostt-nips frostt-enron
python run.py --datasets nell2,uber,nips,enron
# SPLATT and TACO, in the environment of compiled/environment-build.yml
bash compiled/build_splatt.sh && bash compiled/build_taco.sh
python run.py --datasets u7,nell2,uber,nips,enron --groups mttkrp --libs graphblas,splatt,taco
python report.py results/*.jsonl --teeth
```

The demos are listed in `scripts/tensor_bench/demos/README.md`. The hand-written Numba
kernels are `--libs numba`; the Tensor with one engine forced is `graphblas-numba`,
`graphblas-udt` or `graphblas-products`.

### A check from a fresh clone

The kit was run from a fresh clone of this branch (commit G, 2026-10-21 evening, load
1.4-2.7), on u7 and on Uber downloaded by `fetch_data.py`, against pydata/sparse and scipy.
`report.py --teeth` passed. Geometric mean speedup of the Tensor over the faster of the two:

| data set | threads | cases | geomean | where the Tensor is slower                                                                                                                                                   |
| -------- | ------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| u7       | 8       | 42    | 11.7x   | one case: a slab on axis 0 (0.041 against pydata's 0.031 ms)                                                                                                                 |
| u7       | 1       | 42    | 3.0x    | 10 cases: the slab along the last stored axis (2.9x), sums (up to 1.7x), `'ijk,kl->ijl'` (1.4x), TTV and TTM mode 1 (within 9%), all against scipy's dense outputs or pydata |
| Uber     | 8       | 43    | 3.7x    | 7 cases, all slices: 2.9-11x behind single-threaded pydata/sparse and scipy                                                                                                  |
| Uber     | 1       | 43    | 1.06x   | slices (2.2-16x), sums (up to 4.3x), TTV (up to 2.5x) and TTM modes 2-3 (up to 1.6x) against scipy's dense outputs                                                           |

The fingerprints of these 386 results agree with each other (292 pairs) and with the
2026-10-20 results for the same cases (820 pairs, every library). Uber's slices had not been
measured before; they join the known gaps in `limitations-and-roadmap.md`. The demos
`rpq_lagraph.py` (both cases), `hetionet_metapaths.py` and `multiplex_pagerank.py eu` also ran
from the clone, on data downloaded by `fetch_data.py`, with every equality assert passing.

## Contributing results from other machines

The planner's cost constants were fitted on the machine above (`design.md`, "Fitted
constants"), so results from other machines, x86 most of all, are the most useful
contribution. The kit's README says how: run on a quiet machine, check that
`report.py --teeth` passes, and open an issue on python-graphblas with the results file
attached. That file holds the machine record, the times and the fingerprints, and no data.
Fingerprints from different machines must agree, since the synthetic data are seeded and the
downloaded files are checked by SHA-256, so every contributed file is also a correctness
check.

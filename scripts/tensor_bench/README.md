# Tensor benchmark kit

Benchmarks of the experimental n-d `graphblas.Tensor` against other sparse-tensor libraries,
on synthetic tensors and on [FROSTT](http://frostt.io/) tensors, plus five application demos
that check the Tensor against the code python-graphblas users write today. The results and
what they mean are in `ai-notes/tensor/benchmarks.md`.

This directory is separate from `benchmarks/`, the asv suite that tracks the Tensor's own
regressions: asv imports every module under `benchmarks/`, and this kit loads torch,
pydata/sparse and the compiled comparators, which asv must not.

## Quick start

From a clone of python-graphblas:

```bash
conda env create -f scripts/tensor_bench/environment.yml
conda activate graphblas-tensor-bench
python -m pip install --no-deps --no-build-isolation -e .   # python-graphblas from this clone
cd scripts/tensor_bench
python run.py --datasets u7                  # the Tensor, pydata/sparse, scipy, torch on u7
python report.py results/*.jsonl             # tables and the fingerprint check
```

In an environment that already has python-graphblas's dependencies, `PYTHONPATH=/path/to/clone`
instead of the `pip install` works too; the results record which `graphblas` was imported.

`python run.py --datasets u7 --libs graphblas,pydata,scipy` took 13 minutes on an Apple M5 Pro,
most of it in pydata/sparse (its MTTKRP alone takes 3 minutes); `--libs graphblas` alone takes
under a minute.

## What runs

- **Libraries** (`--libs`; the default is `graphblas` plus whichever of `pydata`, `scipy` and
  `torch` import):

  | name                                                     | what                                                                                                                                          |
  | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
  | `graphblas`                                              | the Tensor, with the engines it picks                                                                                                         |
  | `graphblas-numba`, `graphblas-udt`, `graphblas-products` | the Tensor with one TTM / MTTKRP engine forced (Numba kernels; the array-UDT engine, which needs the SuiteSparse JIT; the GraphBLAS products) |
  | `pydata`, `gcxs`                                         | pydata/sparse COO, and GCXS for reductions and permutations                                                                                   |
  | `scipy`                                                  | scipy's n-d `coo_array`                                                                                                                       |
  | `torch`                                                  | torch.sparse COO                                                                                                                              |
  | `numba`                                                  | hand-written Numba kernels for TTV, TTM and MTTKRP over per-mode sorted copies (their build is not timed)                                     |
  | `numpy`                                                  | dense numpy, the reference on the small datasets                                                                                              |
  | `splatt`, `taco`, `finch`                                | compiled comparators, once built (`compiled/README.md`)                                                                                       |

- **Datasets** (`--datasets`, default `u6,u7`):
  - synthetic, generated on first use and cached in the data directory: `small` and
    `small_skew` (checked against dense numpy), `u6` and `u7` (uniform 1000^3 with 1e6 and 1e7
    entries), `skew` (1e5 x 1e3 x 10, 1e7), `hyper` ((2^19)^3, 1e6), `zipf` ((1e5)^3, 1e7, each
    mode Zipf-distributed), `bmm10`, `bmm100`, `bmm1000` (batched matmul operands);
  - FROSTT, after `python fetch_data.py frostt-<name>`: `nell2` (3-d, 77M entries), `enron`
    (4-d, 54M), `uber` (4-d, 3.3M), `nips` (4-d, 3.1M), `chicago` (4-d, 5.3M).
- **Kernel groups** (`--groups`, default all): `construct` (from COO, sorted and shuffled;
  export), `ewise` (add and multiply; the second operand also as a permuted view), `bcast`
  (multiply by a dense vector along each axis, and by a dense matrix), `reduce` (sum over
  every subset of axes), `permute` (every permutation, materialized), `ttv`, `ttm` (R = 16 and
  64), `mttkrp` (R = 16 and 64 for the Tensor, SPLATT, TACO and Finch; 16 for the Python
  libraries), `contract` (`'ijk,kl->ijl'` with a sparse matrix, `'ijk,jkl->il'` with a sparse
  tensor), `bmm` (`'bij,bjk->bik'`), `slice` (the middle index and a 10% range of each axis).
- **Threads** (`--threads`, default `min(8, cores),1`): multithreaded libraries (the Tensor,
  torch, Numba, SPLATT, TACO) run at each count; pydata/sparse and scipy are single-threaded.

## How it measures

- Every (library, dataset, group) runs in its own process (`harness/runner.py`), so caches, JIT
  state and memory high-water marks do not leak between libraries. Only one benchmark process
  runs at a time.
- Each case: one warm-up call (it absorbs JIT and Numba compiles), then the best of N calls,
  with N = 9 below 0.1 s per call, 5 below 1 s, 3 below 30 s. GC is off while timing. Lazy work
  is included: the Tensor's results are materialized (`.new()`, then `.wait()`), torch's are
  coalesced where it leaves them uncoalesced, scipy's are made canonical where they can hold
  duplicates (the non-canonical times are extra `*_raw` cases).
- Inputs are built outside the timed region, in each library's native canonical form. Dense
  operands are numpy arrays (or a Vector or Matrix for the Tensor).
- A case is skipped (`n/a`, with the reason) when a library would allocate a dense output or
  intermediate above 4 GiB, or materialize more than about 12 GiB of COO entries.
- Every row records the 1-minute load average. Benchmark on a quiet machine, and look at the
  load column before trusting a row.
- The first line of each results file records the machine: CPU model, cores (with performance
  and efficiency cores where the OS reports them), memory, OS, Python and library versions,
  the SuiteSparse:GraphBLAS version, whether the SuiteSparse JIT is on and with which
  compiler, whether Numba is available, and the git commit.

## Correctness: fingerprints

Every result is fingerprinted: its shape, the number of entries, an order-independent hash of
the entries' linear indices, the sum of the values, and a sum weighted by a function of each
index (`harness/common.py`). Inputs are positive, so "present" and "nonzero" agree, and
libraries with and without missing-value semantics compare directly. `report.py` compares
every result with the Tensor's at the highest thread count for the same case (counts and hashes
exactly, sums to 1e-8 relative), across libraries, thread counts, engines and machines. The
synthetic data are seeded and FROSTT files are checked by SHA-256, so results from different
machines must agree too.

```bash
python report.py results/*.jsonl            # exit status 1 on any mismatch
python report.py results/*.jsonl --teeth    # plants three wrong fingerprints; fails unless all are caught
python report.py results/*.jsonl --csv all.csv
```

On `small` and `small_skew` the `numpy` library gives dense references:
`python run.py --datasets small,small_skew --libs graphblas,numpy,pydata,scipy`.

## Data

`python fetch_data.py --list` shows every dataset, its size and license. Files are downloaded
from their publishers and checked against the SHA-256 recorded in `fetch_data.py`; nothing is
redistributed with python-graphblas. Each dataset's license and citation are printed and
written next to the files (`SOURCES-<name>.md`). Cite the sources you use.

The data directory is `data/` here by default (git ignores it); set `TENSOR_BENCH_DATA` or
pass `--data-dir` to `run.py`, `fetch_data.py` and the demos to put it elsewhere. Results go
to `results/` (`TENSOR_BENCH_RESULTS`, or `--out FILE`).

## Comparators built from source, and demos

- `compiled/README.md`: SPLATT and TACO (build scripts and drivers), and Finch.jl (optional).
- `demos/README.md`: Hetionet metapaths, network-traffic windows, CP-ALS through TensorLy,
  multiplex PageRank, and regular path queries against LAGraph.

## Adding a machine's results

The cost model inside the Tensor was fitted on one machine, so results from other machines,
x86 in particular, are the most useful contribution.

1. Use a quiet machine (no builds, test suites or video calls); keep the threads at or below
   the number of performance cores if the machine has two kinds.
2. Run the synthetic set, and one FROSTT tensor if you can:

   ```bash
   python run.py --datasets u6,u7,skew,hyper,zipf
   python fetch_data.py frostt-nell2 && python run.py --datasets nell2
   python report.py results/*.jsonl --teeth
   ```

3. Check that the fingerprint check passes. A mismatch is a bug in one of the libraries (or in
   the kit): please report it.
4. Share the results file: open an issue on python-graphblas titled "Tensor benchmark results:
   <CPU model>", attach the `.jsonl` file (it holds the machine record, times and fingerprints,
   no data) and paste `report.py`'s output. The maintainers merge results by machine.

# Tensor: how it is verified

The Tensor was built quickly, with much of the code machine-written, so its correctness
argument does not rest on the unit tests alone. This page describes the layers of checking
and how to run them.

## Unit tests

`graphblas/tests/test_tensor.py` and `graphblas/tests/test_tensor_io.py` (904 tests) check
every operation against a dense numpy oracle. That means the values plus _presence_, a
boolean array of which positions hold a value. Three things recur throughout:

- **Every layout.** The same logical tensor is built in many layouts (axis order and
  row/column split, empty groups included) and as permuted, expanded and squeezed views.
  Layout bugs are the main risk in this design, and a test that uses only the default layout
  misses them. `Tensor._check_invariants` checks the stored Matrix against the shape and
  layout.
- **Every engine.** Where several engines can compute an operation (Numba kernels, the
  array-UDT JIT engine, pure GraphBLAS products), tests force each in turn through the engine
  lists (`_TTM_ENGINES`, `_MTTKRP_ENGINES`, `_GATHER_ENGINES`). A spy checks which path ran.
- **The user guide.** `test_user_guide_examples` runs every example in
  `docs/user_guide/tensor.rst` (81 of them).

Numba kernels write arrays that are handed to SuiteSparse unchecked, so the kernel tests
compile them with Numba bounds checks (the `numba_boundscheck` fixture): an out-of-bounds
write there corrupts memory without an error.

Run the tests under the configurations they are meant for:

```bash
# blocking and non-blocking mode (904 passed each)
pytest graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse --blocking
pytest graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse --no-blocking
# another backend: the modules skip cleanly
pytest graphblas/tests/test_tensor.py --backend suitesparse-vanilla
```

With warm caches each run takes about 35 s on an idle machine. With an empty SuiteSparse
JIT cache, as in CI, the first took 7 to 10 minutes, most of it compiling.

**Without Numba.** Block it in the interpreter, so that every `import numba` fails. pydata/sparse
needs Numba too, so its io tests skip as well (770 passed, 134 skipped):

```python
# run_without_numba.py
import sys

sys.modules["numba"] = None
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
```

```bash
python run_without_numba.py graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse
```

**With the JIT off.** A plugin turns it off before every test. The array-UDT engine's tests
skip (887 passed, 17 skipped):

```python
# jit_off_plugin.py (put it on PYTHONPATH)
import pytest


@pytest.fixture(autouse=True)
def _jit_off():
    import graphblas as gb

    gb.ss.config["jit_c_control"] = "off"
    yield
```

```bash
PYTHONPATH=. pytest graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse -p jit_off_plugin
```

The counts above were measured on 2026-11-23 and the times on 2026-11-22, on macOS (Apple
M5 Pro) with SuiteSparse:GraphBLAS 10.5.1, Python 3.14, numpy 2.5.4, Numba 0.68, scipy 1.18
and pydata/sparse 0.19.

**With a by-column global format.** The Tensor keeps its Matrix by row whatever SuiteSparse's
global format is (`design.md`, "Data model"), so the modules also run with the global set to
by column before every test, by a plugin that sets `GrB_STORAGE_ORIENTATION_HINT` to
`GrB_COLMAJOR` in `pytest_runtest_setup`. All three modules passed that way on 2026-11-23
(904 tests, and the 27 oracles with `--runslow`; SuiteSparse 10.5.2).

`graphblas/tests/test_core.py` checks the SuiteSparse floor in subprocesses. With the Tensor
module's `lib` replaced by one without the Container API, every constructor raises the same
`NotImplementedError` and the Tensor test modules skip.

## Independent randomized oracles

`graphblas/tests/test_tensor_oracles.py` holds randomized oracles that share no code with the
unit tests or the implementation. Each draws random cases and compares every result with a
numpy model of its values and presence:

- shapes of 1-5 dims, including size-0 and size-1 axes;
- random layouts, and permuted, expanded and squeezed views;
- iso values and explicit zeros;
- masks: structural, value and complemented;
- accumulators and `replace`.

They use the Tensor's public API, except for the switches that force an engine
(`_TTM_ENGINES`, `_MTTKRP_ENGINES`, `tensor_index._GATHER_ENGINES`, `_has_numba`, the kernel
thresholds `_TTM_KERNEL_MIN_NNZ` and `_MTTKRP_KERNEL_MIN_NNZ`, and `tensor_numba._FORCE_RUNS` /
`_FORCE_COLBLOCK` for the MTTKRP kernel's variants). They read the census (`tensor_census`,
"Engine selection and the census" in `design.md`) to count which engine ran, so a pass that
never reached an engine cannot hide: a test that forces Numba fails if no kernel ran, and the
one that forces the array-UDT engine skips, with the counts, where that engine cannot run (no
JIT). The TTM and MTTKRP oracles also have SuiteSparse validate every result's stored Matrix
(`GxB_Matrix_fprint`), since the kernels hand it their arrays unchecked.

| test                                 | covers                                                                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_ewise_reduce`                  | `ewise_add`, `ewise_mult` and `ewise_union` with a non-commutative operator, reductions over every axis subset, tensordot, updates with masks and accum        |
| `test_broadcast_contract`            | broadcasting against lower-rank and size-1 operands, Matrix and Vector operands, matmul (1-d and batch axes, `min_plus`), einsum, mttkrp, outer indexing       |
| `test_mttkrp`                        | every mode (negative too), layout and view, R in {1, 2, 5, 16}, numpy and Matrix factors, the kernel's variants forced; with and without Numba                 |
| `test_ttm`, `test_ttm_without_numba` | every axis, factors in either orientation (numpy, full or sparse Matrix), FP64 and INT64, `plus_times` and `min_plus`, hypersparse storage; each engine forced |
| `test_reshape`                       | random splits, fuses and -1, every `copy` mode, copy-on-write in both directions, `ravel` and `flatten`                                                        |
| `test_assign_model_matches_matrix`   | the assignment model, validated against Matrix (GraphBLAS itself is the ground truth in 2-d)                                                                   |
| `test_assign`                        | the same model on n-d Tensors: assign and subassign, Tensor, Scalar and Python values with broadcasting, masks, accum, `replace`                               |
| `test_gather`                        | `take`, `take_along_axis` and `vindex` against numpy, with each gather engine list                                                                             |
| `test_einsum`                        | random subscripts reused with new layouts (the plan cache), masked products, batched `@`                                                                       |
| `test_io`                            | round trips through pydata/sparse COO and GCXS, scipy n-d COO with duplicates, a refused `fill_value`                                                          |

Each oracle has **teeth**: `test_oracle_teeth[<name>]` plants a known error in its model and
requires failures, to show that the comparison would catch a real one. The planted errors are
the operands of a non-commutative operator swapped, the wrong mode or factor axis, F order for
C order, `replace` ignored (in the Matrix validation too), the mask dropped, a gather one
position off, and an entry dropped.

They are slow tests, so they run with `--runslow`, as in CI's slow jobs:

```bash
pytest graphblas/tests/test_tensor_oracles.py --runslow --backend suitesparse
```

The default case counts are sized for CI: about 2,200 checks, plus the runs that force each
engine (2026-10-21). On 2026-11-22 the 27 tests took 72-80 s with the caches warm from the
unit tests. Much of a run is compiling the JIT kernels its seed's cases need, so the time varies
by seed. Three runs at 10 times the counts (`-k "not teeth"`) took about 4 minutes each
(2026-10-21). For a thorough run, multiply the counts:

```bash
GRAPHBLAS_TENSOR_ORACLE_SCALE=20 pytest graphblas/tests/test_tensor_oracles.py --runslow --backend suitesparse -k "not teeth"
```

Each run draws new cases: the seed is the session's `GRAPHBLAS_TEST_SEED`, which
`conftest.py` prints. A failure names its seed; `GRAPHBLAS_TENSOR_ORACLE_SEED=<seed>` reruns
it. Set `NUMBA_BOUNDSCHECK=1` to compile the kernels with bounds checks.

## Notebooks

Two notebooks run the Tensor end to end, committed with their outputs:

- `notebooks/Tensor Quickstart.ipynb`: the model, construction, views and copy-on-write,
  broadcasting, reductions, contractions, masks and assignment, indexing and gathering, and
  io, on one small example.
- `notebooks/Tensor Multiplex PageRank.ipynb`: a multilayer PageRank over the 37
  airline layers of the EU airline multiplex, as one Tensor and one einsum per step, checked
  against scipy's sparse direct solve. It downloads the data (24 KB) from Netzschleuder and
  checks the SHA-256 of the CSV files it reads (not of the zip, which the server may rebuild).

CI's notebook job executes every notebook, the Tensor ones only where SuiteSparse:GraphBLAS 10
or later is installed. To rerun them in place, with Jupyter installed:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Tensor*.ipynb
```

## Cross-library agreement in the benchmarks

Every benchmark result carries a fingerprint: the shape, the entry count, a hash of the
coordinates, and two weighted sums of the values. Fingerprints are compared across
libraries, and a benchmark row counts only if they agree. The libraries are pydata/sparse,
scipy, torch, [SPLATT](https://github.com/ShadenSmith/splatt),
[TACO](https://github.com/tensor-compiler/taco), [Finch](https://github.com/finch-tensor/Finch.jl),
and the Tensor itself at different thread counts and with each engine forced. The comparison
has its own teeth check (tampered fingerprints must be flagged). Small data sets are also
checked against dense numpy. The harness is in `scripts/tensor_bench/`; `benchmarks.md` gives
the counts and the results, and the kit's README says how to rerun it.

## Real workloads

Each workload computes a result two ways and asserts they are equal: with the Tensor, and with
the code that python-graphblas users write today (usually a loop over Matrices):

- two-hop metapath counts on [Hetionet](https://het.io);
- per-window traffic statistics over network traces, with the time axis reshaped into
  windows;
- CP-ALS through [TensorLy](https://github.com/tensorly/tensorly), with `Tensor.mttkrp`
  registered as its MTTKRP backend;
- multiplex PageRank, compared also with a sparse direct solve;
- regular path queries, against [LAGraph](https://github.com/GraphBLAS/LAGraph)'s expected
  outputs.

These scripts are in `scripts/tensor_bench/demos/`, reading data that `fetch_data.py`
downloads from the original sources.

## What this does not cover

- **Platforms.** Everything above ran on macOS on Apple silicon. CI runs the Tensor test
  modules wherever it draws SuiteSparse 10 or later, at random per job: on the branch's run
  of 2026-11-22 they ran on Linux x86-64 (10.4 and 10.5), macOS arm64 and x86-64, and
  Windows (10.5.2), and skipped on Linux aarch64, which drew 9.4. On Windows the array-UDT
  engine's tests skip: SuiteSparse compiles through cmake there, and python-graphblas then
  leaves the JIT control at "run" (a rule from #659, below this pull request; `design.md`,
  "The SuiteSparse JIT"). The slow oracles
  ran in that run's Linux x86-64 and Windows jobs.
- **SuiteSparse versions.** Locally, 10.5.1 and 10.3.1; in CI, whichever 10.x versions the
  jobs draw. Versions before 10 are refused by design.
- **The cost model.** The planner's constants were fitted on one machine, and its choices are
  checked by measurement only there (`design.md`, "Fitted constants").
- **Size.** The largest inputs were FROSTT tensors: NELL-2 has 77M entries and Enron 54M.

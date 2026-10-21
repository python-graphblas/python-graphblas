# Tensor: how it is verified

The Tensor was built quickly, with much of the code machine-written, so its correctness
argument does not rest on the unit tests alone. This page describes the layers of checking,
what is in the repository and what is not yet, and how to run what is.

## Unit tests

`graphblas/tests/test_tensor.py` and `graphblas/tests/test_tensor_io.py` (506 tests) check
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
  `docs/user_guide/tensor.rst` (68 of them).

Numba kernels write arrays that are handed to SuiteSparse unchecked, so the kernel tests
compile them with Numba bounds checks (the `numba_boundscheck` fixture). This was added after
an early kernel wrote past an array's end.

Run the tests under the configurations they are meant for:

```bash
# blocking and non-blocking mode (506 passed each, about 30 s)
pytest graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse --blocking
pytest graphblas/tests/test_tensor.py graphblas/tests/test_tensor_io.py --backend suitesparse --no-blocking
# another backend: the modules skip cleanly
pytest graphblas/tests/test_tensor.py --backend suitesparse-vanilla
```

**Without Numba.** Block it in the interpreter, so that every `import numba` fails. pydata/sparse
needs Numba too, so its io tests skip as well (394 passed, 112 skipped):

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
skip (494 passed, 12 skipped):

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

The counts above were checked on macOS with SuiteSparse:GraphBLAS 10.5.1 and Python 3.14.
The same tests also passed on Python 3.11 with numpy 1.24 (490 passed, 16 skipped: that
environment had no C compiler, so the JIT tests skipped).

`graphblas/tests/test_core.py` checks the SuiteSparse floor in subprocesses. With the Tensor
module's `lib` replaced by one without the Container API, every constructor raises the same
`NotImplementedError` and the Tensor test modules skip.

## Independent randomized oracles

During development, each change was also checked by separate oracle scripts. They share no
code with the tests or the implementation, and **they are not in the repository yet**:
moving them in, as slow tests, is on the roadmap (`limitations-and-roadmap.md`). Each one
draws random cases and compares against numpy:

- shapes of 1-5 dims, including size-0 and size-1 axes;
- random layouts and views;
- iso values and explicit zeros;
- masks: structural, value and complemented;
- accumulators and `replace`.

Each has a **teeth** mode that plants a known error in the oracle, to show the check would
catch a real one.

| oracle                        | covers                                                                                                                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| element-wise and reductions   | ewise ops with non-commutative operators, every axis subset, masks and accum on permuted targets                                                                                             |
| broadcasting and contractions | broadcasting with operands of lower rank, tensordot, matmul                                                                                                                                  |
| MTTKRP                        | every mode, every layout (empty groups included), R in {1, 2, 5, 16}, Numba variants forced, Numba blocked                                                                                   |
| TTM                           | every mode, factors in either orientation (numpy, full or sparse Matrix), FP64 and INT64, two semirings; every result validated by `GxB_Matrix_fprint`; with and without Numba bounds checks |
| reshape                       | random splits, fuses and -1, every `copy` mode, copy-on-write in both directions                                                                                                             |
| assignment                    | a numpy model of GraphBLAS assign and subassign, **validated first against Matrix** (2-d ground truth), then run on n-d                                                                      |
| gather                        | `take`, `take_along_axis` and `vindex` against numpy                                                                                                                                         |
| einsum                        | repeated subscripts and shapes against fresh layouts (to exercise the plan cache), masked products, batched `@`                                                                              |
| io                            | round trips through pydata/sparse COO and GCXS, and scipy n-d COO with duplicates                                                                                                            |

Typical runs are hundreds to thousands of cases per oracle per seed, with 0 failures
required. Every oracle was rerun after each round of changes to the Tensor's code.

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

- **Platforms.** Everything above ran on macOS on Apple silicon. python-graphblas's CI matrix
  includes Linux (x86-64 and aarch64), macOS (x86-64 and arm64) and Windows, but this branch
  has not run in CI. CI picks SuiteSparse versions at random, and the Tensor tests skip on
  any below 10. Windows in particular is untested; see `limitations-and-roadmap.md`.
- **SuiteSparse versions.** Only 10.5.1 was run. Versions before 10 are refused by design.
- **The cost model.** The planner's constants were fitted on one machine, and its choices are
  checked by measurement only there (`design.md`, "Fitted constants").
- **Size.** The largest inputs were FROSTT tensors: NELL-2 has 77M entries and Enron 54M.

# ai-notes

Curated notes for contributors, reviewers and developers of python-graphblas. They were written
with AI assistance and edited by the maintainers. They explain why the code is the way it is:
design, decisions, how it is verified and measured, and what is known to be missing. They are
not user documentation; that is in `docs/`.

Each topic has its own directory. Each page is meant to be readable on its own, without access
to anything outside this repository.

## tensor/: the experimental n-d `gb.Tensor`

| page                                | read it for                                                                                                                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensor/design.md`                  | the data model (one Matrix plus a layout), copy-on-write views, the relayout planner and its cost model, how each operation family maps to SuiteSparse, the engines, and a map of the modules |
| `tensor/decisions.md`               | the API and semantic decisions, with their reasons                                                                                                                                            |
| `tensor/verification.md`            | how correctness is checked: unit tests and how to run them without Numba or the JIT, independent randomized oracles, cross-library fingerprints, real workloads                               |
| `tensor/limitations-and-roadmap.md` | what is untested, missing or slow, and the planned order of work                                                                                                                              |
| `tensor/backport-candidates.md`     | Tensor behavior that Matrix and Vector could adopt                                                                                                                                            |
| `tensor/benchmarks.md`              | results against pydata/sparse, scipy, torch, SPLATT, TACO and Finch, the real workloads, and how to rerun them (`scripts/tensor_bench/`)                                                      |
| `tensor/suitesparse-wishlist.md`    | SuiteSparse:GraphBLAS features that would simplify or speed up the Tensor, in priority order, with evidence and proposed APIs                                                                 |

The user guide's "Tensors" page (`docs/user_guide/tensor.rst`) covers how to use it.

Code comments may point at sections of these pages by their headings. If you change the
Tensor's design or a decision, update the matching page in the same pull request, and keep the
headings stable (or update the comments that cite them: `grep -rn "ai-notes" graphblas/`).

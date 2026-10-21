# Compiled comparators: SPLATT, TACO, Finch

Optional. `run.py --libs splatt,taco,finch` runs them on the same data as the Python
libraries (exported once as raw binary by `export_bin.py` into `<data dir>/bin/`), and their
results carry the same fingerprints, so `report.py` checks them against the Tensor.

## Build

```bash
conda env create -f scripts/tensor_bench/compiled/environment-build.yml
conda activate graphblas-tensor-build
bash scripts/tensor_bench/compiled/build_splatt.sh   # a minute or so
bash scripts/tensor_bench/compiled/build_taco.sh     # a few minutes
conda activate graphblas-tensor-bench                # then run.py as usual
```

Sources, builds and the drivers go to `TENSOR_BENCH_BUILD` (default `../build`, ignored by
git). The scripts check out the commits the maintainers measured (SPLATT 6cb8628 of
2022-03-03, TACO 0e79acb5 of 2025-04-14; override with `SPLATT_COMMIT` and `TACO_COMMIT`), use
the active environment's compilers and libraries, and clear compiler variables that another
environment may have exported. They were run on macOS (Apple silicon) with conda-forge
clang 21; on Linux they should work with conda-forge's gcc, but that is untested.

## SPLATT

[SPLATT](https://github.com/ShadenSmith/splatt) is the reference for MTTKRP (the kernel of CP
decompositions). `splatt_drv.c` builds SPLATT's own CSF with `csf_alloc` and times
`mttkrp_csf`, the routine SPLATT's CPD calls, per mode and thread count, for three ways of
storing the tensor:

- `splatt-onemode`: one CSF tree for all modes (the memory of one copy, as the Tensor);
- `splatt-twomode`: two trees, SPLATT's default for CPD;
- `splatt-allmode`: one tree per mode, rooted at that mode (fastest, 3-4x the memory).

The driver keeps empty slices (SPLATT's file reader would renumber them), so output row `i`
is index `i`. `splatt bench` is not used: it reports times in 1 ms steps and only tiled
ONEMODE. The CSF build time is reported as `prep_ms`, outside the kernel time.

## TACO

[TACO](https://github.com/tensor-compiler/taco) generates C for each kernel and compiles it at
run time. `taco_drv.cpp` drives it through its C++ API, because its command-line tool cannot
run kernels at this commit (its benchmark path compiles a module whose C source defines
`compute` twice: "redefinition of 'compute'"). Kernels: MTTKRP (R = 16, 64), TTM (R = 16),
TTV, sums over each axis, element-wise add and multiply. Things to know when reading its
numbers:

- The input is given to TACO as a CSF in the mode order each kernel wants (the output mode at
  the root for MTTKRP, so that TACO's default schedule parallelizes the outer loop without
  races; the contracted or summed mode at the leaf for TTV, TTM and sums). The driver builds
  that copy per mode outside the timing (`prep_ms`); the Tensor starts from its one layout.
- TACO's default schedule emits no parallel loop when the output is sparse, so TTV, sums and
  element-wise ops run single-threaded (run.py runs them at 1 thread only), and TTM's only
  parallel loop zeroes the output.
- The time is assemble + compute (the output's structure is part of the result, as for the
  Tensor). `TensorBase::assemble()` and `compute()` do nothing unless the tensor is flagged as
  needing them, so the driver sets the flags before every timed call.
- Kernels compile with TACO's default `-O3 -ffast-math`; the compile time is `compile_ms`.

## Finch (optional)

[Finch.jl](https://github.com/finch-tensor/Finch.jl) runs through `finch_drv.jl`: `@finch` loop
nests over Dense(SparseList(SparseList)) tensors in the same mode orders as TACO, 3-d only,
single-threaded (`@finch` loops need explicit parallel annotations). It was measured with
Finch 1.4.0 on Julia 1.12; the plain loops do not use Finch's array API, so they should run on
later versions too. Install Julia and Finch in a project of your own, then:

```bash
export JULIA_PROJECT=/path/to/a/julia/project/with/Finch   # julia -e 'using Pkg; Pkg.add("Finch")'
export TENSOR_BENCH_JULIA=julia                            # or the path to the julia binary
python run.py --datasets u7 --libs finch
```

Each new kernel compiles on its first call in a process (4-13 s measured); the reported time
is the best of the later calls, and `first_ms` has the first. `mttkrpfib16` / `mttkrpfib64`
(run `finch_drv.jl` directly) are a hand-factored MTTKRP that sums each fiber into a
workspace first; it was slower than the plain loop.

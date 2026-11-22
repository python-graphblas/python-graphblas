# asv benchmarks for python-graphblas

An airspeed velocity (asv) suite for the library's hot paths. Its configuration is
`asv.conf.json` at the repository root. `_verify.py` runs every benchmark once
without asv.

## Layout

```
asv.conf.json            asv configuration (at the repository root). It has `//`
                         comments, which asv strips, so the strict `check-json`
                         pre-commit hook skips it.
benchmarks/
  __init__.py            package marker, and how the modules import common
  common.py              seeded data builders, called from setup (not timed)
  scalar_access.py       get, extract and assign of one element, present or missing
  index_parse.py         parse_index for a Python int and for a numpy int
  small_ops.py           ewise, apply and reduce on about 100 elements (overhead)
  large_kernels.py       mxm, mxv, ewise and reduce on about 1e6 nonzeros (C library)
  conversions.py         from_coo, to_coo, from_dense, to_dense and scipy interop
  imports.py             cold import and first use (timeraw, in a fresh process)
  repr_bench.py          repr and _repr_html_ of small and large objects
  _verify.py             runs every benchmark once (not an asv file)
```

## Running

```bash
pip install asv
asv machine --yes             # once: record the machine's details
asv run                       # benchmark the current commit
asv continuous main HEAD      # compare a branch with main and flag regressions
asv publish && asv preview    # build and serve the HTML report
asv run --bench scalar_access # run the benchmarks that match a regex
```

To check that every benchmark runs, without asv:

```bash
python benchmarks/_verify.py
```

## Design

- `setup()` builds all data once, outside the timed region, and every builder is
  seeded, so runs are comparable across commits.
- "Small" is 10 to 1000 elements, where the time is mostly Python's. "Large" is
  about 1e6 nonzeros, where it is mostly the C library's. The large matrix has
  about one entry per row, so `mxm` gives about as many entries as its inputs
  instead of quadratic fill-in. On one machine, the large kernels took 1 to 5 ms
  a call, the conversions under 10 ms, `import graphblas` about 32 ms, and
  initialization with the first operator about 0.2 s. Raise `common.LARGE_NNZ`,
  or add a denser matrix, for a heavier `mxm`.
- The large kernels set `number = 1`, `warmup_time = 0`, a small `repeat` and a
  `timeout`, so asv does not batch them into runs of several seconds.
- `imports.py` uses `timeraw_*` benchmarks, which run the returned code in a
  fresh process. A normal benchmark would read about 0 for an import, since the
  module is already imported.
- `asv.conf.json` pins `python-suitesparse-graphblas`, so a run measures changes
  on the python-graphblas side alone. Drop the pin to measure both. Keep the
  pinned version among the psg versions in `scripts/ci_pick_versions.py`.
- `OMP_NUM_THREADS=1` runs GraphBLAS on one thread, for repeatable timings. Raise
  it in a separate run to measure parallel scaling.

## Notes

- asv is not in `dev-requirements.txt` or `environment.yml`: nothing in the tests
  or CI runs it, so only people who benchmark need it. An optional `benchmark`
  dependency group (`pip install python-graphblas[benchmark]`) would suit a suite
  that CI runs.
- CI does not run asv. Shared GitHub runners are too noisy to compare the runs of
  a pull request; a weekly `asv continuous` on one fixed machine would not be.
- The first use of some operators compiles with Numba or the SuiteSparse JIT.
  `setup()` uses the operators each benchmark times, but the first `asv run` in a
  new environment may still show some slow one-off times.

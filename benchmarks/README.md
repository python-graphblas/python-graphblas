# asv benchmark scaffold for python-graphblas

A starting airspeed velocity (asv) benchmark suite covering the library's hot
paths. Built as a scaffold: review, then move `asv.conf.json` and `benchmarks/`
to the repo root (next to `pyproject.toml`). `_verify.py` is a local sanity
checker and does not need to ship.

## Layout

```
asv.conf.json            asv config, tuned for this repo (see caveats below).
                         JSONC: asv documents `//` comments and strips them in
                         its loader, so `.pre-commit-config.yaml` excludes this
                         one path from the strict `check-json` hook. asv also
                         accepts `asv.conf.jsonc`, which would need no exclude,
                         but asv is not a pinned dependency here and older
                         versions resolve only the `.json` name.
benchmarks/
  __init__.py            package marker + note on the dual-import shim
  common.py              shared, seeded data builders (called from setup, not timed)
  scalar_access.py       single-element get / extract / assign, hit vs miss
  index_parse.py         parse_index int fast lane vs numpy-int lane (index build)
  small_ops.py           ewise / apply / reduce on ~100-element objects (overhead)
  large_kernels.py       mxm / mxv / ewise / reduce on ~1e6 nnz (C-library bound)
  conversions.py         from_coo/to_coo, from_dense/to_dense, scipy interop
  imports.py             cold import + first-use timing (timeraw, fresh subprocess)
  repr_bench.py          repr / _repr_html_ for small and large objects
_verify.py               standalone check: runs every benchmark once (not an asv file)
```

## Running

```bash
pip install asv
asv machine --yes            # one-time: record machine info
asv run                      # benchmark the current commit
asv continuous main HEAD     # compare a branch against main, flag regressions
asv publish && asv preview   # build and serve the HTML report
asv run --bench scalar_access # run a subset by regex
```

Quick correctness check without asv (runs each benchmark once):

```bash
python _verify.py
```

## Design choices

- `setup()` builds all data once, outside the timed region; every builder is
  seeded so runs are comparable across commits.
- Sizes: "small" is 10 to 1000 elements (overhead-dominated, so we track
  Python-side cost); "large" is ~1e6 nonzeros (C-library dominated). Average
  matrix degree is ~1 so `mxm` output stays near the input size instead of
  exploding into quadratic fill-in. Verified per-call times on one dev machine:
  large kernels ran in 1 to 5 ms, conversions under 10 ms, cold `import
graphblas` ~32 ms, init/first-operator ~0.2 s. Bump `common.LARGE_NNZ` or add a
  higher-degree matrix if you want heavier `mxm`.
- Large kernels use `number = 1`, `warmup_time = 0`, small `repeat`, and a
  `timeout`, so asv does not batch them into multi-second runs.
- `imports.py` uses `timeraw_*`, which runs the returned code string in a fresh
  subprocess. That is the only way to measure true cold import cost; a normal
  benchmark would read ~0 because the module is already imported.
- `asv.conf.json` pins `python-suitesparse-graphblas` so a run isolates
  python-graphblas-side regressions from C-library changes. Drop the pin to track
  end-to-end (library + C) performance. Keep the pinned version in sync with the
  PSG pools in `scripts/ci_pick_versions.py`.
- `OMP_NUM_THREADS=1` pins GraphBLAS to one thread for reproducible single-core
  timings. Raise it in a dedicated run to benchmark parallel scaling.

## Caveats to resolve before merging

- **dev deps**: `asv` is deliberately NOT added to `dev-requirements.txt` or
  `environment.yml`. Nothing in the test suite or CI invokes it, so adding it
  would make every contributor's dev install fetch a package only used by
  people who opt into benchmarking; and this repo ties dependency changes to
  `scripts/check_versions.sh` and the CI version pools, which a benchmarks-only
  change has no business editing. `pip install asv` is in the usage section
  above. Adding a `benchmark` optional-dependencies group (so that
  `pip install python-graphblas[benchmark]` works) is the natural move if the
  suite graduates from scaffold to something CI runs.
- **CI wiring**: asv is not wired into CI here. A weekly cron running `asv
continuous` against a fixed machine (or asv's own regression detection) is the
  natural follow-up; per-PR benchmarking on shared GitHub runners is too noisy to
  gate on.
- **First-run JIT/compile noise**: the first use of some operators triggers
  numba/JIT work. `setup()` touches the operators the benchmark uses, but the
  very first `asv run` on a fresh env may still show inflated one-off numbers;
  asv's repeats and its own warmup mitigate this.
- **pandas dependency**: `repr_bench.py` needs pandas (the default/test extras
  include it). The env matrix installs pandas, so this is satisfied.

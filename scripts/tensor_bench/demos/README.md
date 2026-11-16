# Demos: the Tensor against the code users write today

Each demo computes the same result twice, with the Tensor and with the per-Matrix code that
python-graphblas users write today (usually a loop over Matrices, sometimes scipy or another
library too), asserts that the results are equal, and prints timings at 1 and 8 threads.
Measured results are in `ai-notes/tensor/benchmarks.md`. Run them from `scripts/tensor_bench/`
after fetching their data; each takes `--data-dir` (or `TENSOR_BENCH_DATA`).

| demo                                                                                                     | data                                             | what it checks                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hetionet_metapaths.py`: all two-hop Compound-Gene-Disease metapaths of Hetionet in two `einsum`s        | `python fetch_data.py hetionet`                  | the 9 metapaths' path counts, how many known treats / palliates pairs each reaches, and the degree-weighted counts (DWPC), against a per-predicate loop over Matrices (one `mxm` per metapath, as `cbizon/metapath-counts` does) and scipy; plus a larger Compound-x-Gene-y-Gene family                                                         |
| `lbnl_windows.py`: Graph Challenge traffic statistics for every time window, by reshaping the time axis  | `python fetch_data.py frostt-lbnl`               | the nine per-window statistics of the MIT LL / CAIDA `ILANDS-sensor` code at every binary timescale, against its loop of one Matrix per window (`--window 3600` and `--window 60`)                                                                                                                                                              |
| `cp_als_tensorly.py`: CP-ALS through TensorLy's `parafac`, with `Tensor.mttkrp` registered as its MTTKRP | none (u7) or `python fetch_data.py frostt-nell2` | the reconstruction errors at every iteration against TensorLy's own sparse path (pydata/sparse) from the same initialization, and across thread counts; SPLATT's `cpd` for reference with `--tns-dir` (needs `compiled/build_splatt.sh`). Needs `tensorly`. u7 is random, so its fit is meaningless: there the demo checks agreement and timing |
| `multiplex_pagerank.py`: multilayer PageRank (De Domenico et al. 2015) on a layer x src x dst Tensor     | `python fetch_data.py eu-airlines higgs`         | the PageRank of every (layer, node) against a per-layer loop of `vxm` (to 1e-14), and against a sparse direct solve on the supra-graph (EU airlines) or scipy's power iteration (Higgs), to 1e-10                                                                                                                                               |
| `rpq_lagraph.py`: regular path queries with the edge labels as an axis                                   | `python fetch_data.py lagraph-rpq hetionet`      | `lagraph`: the four cases of LAGraph's `test_RegularPathQuery.c` against its expected outputs and a per-label Matrix loop written as LAGraph's C is; `hetionet`: Compound-binds-Gene (-interacts-Gene)\* -associates-Disease from 100 compounds, union and per source, against the loop                                                         |

```bash
python fetch_data.py hetionet && python demos/hetionet_metapaths.py
python fetch_data.py frostt-lbnl && python demos/lbnl_windows.py --window 3600
python demos/cp_als_tensorly.py --datasets u7
python fetch_data.py eu-airlines && python demos/multiplex_pagerank.py eu
python fetch_data.py lagraph-rpq hetionet && python demos/rpq_lagraph.py lagraph
```

An `AssertionError` means the Tensor and the baseline disagree: please report it.

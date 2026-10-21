"""Export benchmark datasets and operands as raw binary files for the compiled drivers.

    python export_bin.py u7 nell2 ...

Writes to ``<data dir>/bin/`` (``TENSOR_BENCH_DATA``, see ``paths.py``):
  <ds>.tb      X: magic b"TNSB0001", int64 nmodes, int64 nnz, int64 dims[nmodes],
               int64 inds[nmodes][nnz] (0-based, C-order sorted), float64 vals[nnz]
  <ds>_y.tb    the element-wise second operand Y (datasets.second_operand), same format
  <ds>_f<R>.f64  the MTTKRP/TTM factors (datasets.factors), mode after mode, each I_m x R
               row-major float64
  <ds>_v.f64   the TTV vectors (datasets.vectors), mode after mode, float64
Same seeds as the Python harness (``harness/datasets.py``), so every library sees the same
numbers. Files already exported are kept.
"""

import os
import sys
import time

import numpy as np

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import datasets as D
import paths

# isort: on


OUT = str(paths.data_dir() / "bin")


def write_tb(path, coords, vals, shape):
    with open(path, "wb") as f:
        f.write(b"TNSB0001")
        np.array([len(shape), vals.size, *shape], np.int64).tofile(f)
        for c in coords:
            np.ascontiguousarray(c, np.int64).tofile(f)
        np.ascontiguousarray(vals, np.float64).tofile(f)


def export(ds):
    if os.path.exists(os.path.join(OUT, f"{ds}_v.f64")):
        return
    t0 = time.perf_counter()
    coords, vals, shape = D.load(ds)
    write_tb(os.path.join(OUT, f"{ds}.tb"), coords, vals, shape)
    yc, yv = D.second_operand(ds, coords, vals, shape)
    write_tb(os.path.join(OUT, f"{ds}_y.tb"), yc, yv, shape)
    for R in (16, 64):
        Fs = D.factors(ds, shape, R)
        np.concatenate([np.ascontiguousarray(F).ravel() for F in Fs]).tofile(
            os.path.join(OUT, f"{ds}_f{R}.f64")
        )
    np.concatenate(D.vectors(ds, shape)).tofile(os.path.join(OUT, f"{ds}_v.f64"))
    print(f"{ds}: shape {shape}, nnz {vals.size}, {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for ds in sys.argv[1:]:
        export(ds)

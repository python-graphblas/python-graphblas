"""Sparse CP-ALS with TensorLy's own ``parafac`` loop and the Tensor's ``mttkrp`` underneath.

TensorLy routes every MTTKRP of ``parafac`` through ``tenalg.unfolding_dot_khatri_rao``, which it
documents as swappable (``tl.tenalg.register_backend_method``). Registering one that calls
``gb.Tensor.mttkrp`` lets TensorLy's numpy backend run CP-ALS on a sparse tensor that it could
not hold densely: TensorLy keeps the factors (dense numpy) and the ALS algebra; the Tensor does
the sparse work. The only other things ``parafac`` asks of the tensor are its shape, ndim,
dtype and 2-norm, which a 20-line adapter provides.

The check: TensorLy's own sparse path (``tensorly.contrib.sparse`` on pydata/sparse) from the
same random initialization gives the same reconstruction errors, iteration by iteration, where
it runs (u7). SPLATT's ``cpd`` is timed for reference (its own initialization).

    python demos/cp_als_tensorly.py [--datasets u7,nell2] [--rank 16] [--iters 5] [--threads 1,8]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

import graphblas as gb
from graphblas import Tensor
from graphblas.core.ss import jit_config

# isort: off
# the kit's own modules (paths.py, harness/), wherever the kit is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
import paths

# isort: on


jit_config.fix_jit_config()


class GBTensor:
    """What TensorLy's numpy backend asks of the data tensor in ``parafac``: shape, ndim, a
    numpy dtype, and the 2-norm (registered below); the MTTKRPs go to ``T.mttkrp``.
    """

    def __init__(self, T):
        self.T = T
        self.shape = T.shape
        self.ndim = T.ndim
        self.dtype = np.dtype(np.float64)
        self._norm = float(np.sqrt((T * T).new().sum().new().to_scalar().value or 0.0))


tl.set_backend("numpy")
# (the backends' own methods: tl.norm and tl.tenalg.* dispatch, so they would call ours)
_numpy_norm = tl.backend.BackendManager.current_backend().norm
_tenalg_default = tl.tenalg.TenalgBackendManager.current_backend().unfolding_dot_khatri_rao


def norm(tensor, order=2, axis=None):
    if isinstance(tensor, GBTensor) and order == 2 and axis is None:
        return tensor._norm
    return _numpy_norm(tensor, order, axis)


def unfolding_dot_khatri_rao(tensor, cp_tensor, mode):
    """TensorLy's MTTKRP hook: ``unfold(X, mode) @ khatri_rao(factors, skip=mode)``, scaled by
    the weights, as ``X.mttkrp(factors, mode)`` with the weights folded into one factor.
    """
    if not isinstance(tensor, GBTensor):
        return _tenalg_default(tensor, cp_tensor, mode)
    weights, factors = cp_tensor
    factors = [np.ascontiguousarray(f, np.float64) for f in factors]
    other = next(i for i in range(tensor.ndim) if i != mode)
    factors[other] = factors[other] * np.asarray(weights)[None, :]
    out = tensor.T.mttkrp(factors, mode).new()
    return out.to_dense(fill_value=0.0)


tl.register_backend_method("norm", norm)
tl.tenalg.register_backend_method("unfolding_dot_khatri_rao", unfolding_dot_khatri_rao)


def timed(f, *args, **kwargs):
    t0 = time.perf_counter()
    rv = f(*args, **kwargs)
    return time.perf_counter() - t0, rv


def run_parafac(X, rank, iters, fn=parafac):
    """``fn`` (a parafac) from random_state 0 for ``iters`` iterations: the seconds per iteration
    (from runs of 1 and 1 + iters iterations, so initialization and the norm drop out; a
    callback would make parafac build the dense reconstruction) and the relative errors.
    """
    kw = dict(init="random", random_state=0, tol=0, return_errors=True)
    t1, _ = timed(fn, X, rank, n_iter_max=1, **kw)
    tk, (cp, errors) = timed(fn, X, rank, n_iter_max=1 + iters, **kw)
    return cp, [float(e) for e in errors], (tk - t1) / iters


def tensorly_sparse(coords, vals, shape, rank, iters):
    """TensorLy's own sparse path (pydata/sparse COO through contrib.sparse)."""
    import sparse
    from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac

    S = sparse.COO(np.array(coords), vals, shape=shape)
    _cp, errors, per_iter = run_parafac(S, rank, iters, fn=sparse_parafac)
    return errors, per_iter


def tns_file(ds, coords, vals, tns_dir):
    """The tensor as a .tns text file for SPLATT (1-based indices), written once."""
    import pandas as pd

    path = os.path.join(tns_dir, f"{ds}.tns")
    if not os.path.exists(path):
        os.makedirs(tns_dir, exist_ok=True)
        cols = {f"i{a}": c + 1 for a, c in enumerate(coords)}
        pd.DataFrame({**cols, "v": vals}).to_csv(
            path + ".tmp", sep=" ", header=False, index=False, float_format="%.17g"
        )
        os.replace(path + ".tmp", path)
    return path


def splatt_exe():
    """The splatt binary that compiled/build_splatt.sh builds."""
    import platform

    return (
        paths.build_dir()
        / "splatt"
        / "build"
        / f"{platform.system()}-{platform.machine()}"
        / "bin"
        / "splatt"
    )


def splatt_cpd(exe, path, rank, iters, nthreads):
    """SPLATT's own CPD (its initialization; CSF built first): seconds per iteration, fit."""
    out = subprocess.run(
        [
            str(exe),
            "cpd",
            path,
            "-r",
            str(rank),
            "-i",
            str(iters),
            "-t",
            str(nthreads),
            "--tol",
            "0",
            "--nowrite",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    its = [float(line.split("(")[1].split("s)")[0]) for line in out.splitlines() if "its =" in line]
    fit = [line for line in out.splitlines() if line.startswith("Final fit")]
    total = [line for line in out.splitlines() if line.strip().startswith("TOTAL")]
    return float(np.median(its)), fit[0].split(":")[1].strip(), total[0].split()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="u7,nell2")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--sparse-ref", default="u7", help="datasets to run TensorLy's sparse path on")
    ap.add_argument("--sparse-iters", type=int, default=2)
    ap.add_argument(
        "--tns-dir", default="", help="where to write .tns files for SPLATT (none: skip)"
    )
    ap.add_argument(
        "--splatt", default=None, help="the splatt executable (default: the kit's build)"
    )
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: TENSOR_BENCH_DATA or ../data)"
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    import datasets as D

    for ds in args.datasets.split(","):
        coords, vals, shape = D.load(ds)
        T = Tensor.from_coo(coords, vals, shape=shape)
        T.wait()
        X = GBTensor(T)
        print(
            f"{ds}: shape {shape}, {T.nvals} entries, layout {T.layout._short()}, "
            f"load {os.getloadavg()[0]:.1f}",
            flush=True,
        )
        ref = None
        for nt in [int(x) for x in args.threads.split(",")]:
            gb.ss.config["nthreads"] = nt
            run_parafac(X, args.rank, 1)  # (kernels compiled or loaded)
            _cp, errors, per_iter = run_parafac(X, args.rank, args.iters)
            ref = errors if ref is None else ref
            assert np.allclose(errors, ref, rtol=1e-9), (errors, ref)
            print(
                json.dumps(
                    dict(
                        dataset=ds,
                        engine="tensorly parafac + gb Tensor.mttkrp",
                        nthreads=nt,
                        rank=args.rank,
                        iters=args.iters,
                        s_per_iter=round(per_iter, 4),
                        rel_errors=[round(e, 8) for e in errors],
                        load1=round(os.getloadavg()[0], 2),
                    )
                ),
                flush=True,
            )
        if ds in args.sparse_ref.split(","):
            k = args.sparse_iters
            errs, per_iter = tensorly_sparse(coords, vals, shape, args.rank, k)
            same = bool(np.allclose(errs, ref[: len(errs)], rtol=1e-8))
            print(
                json.dumps(
                    dict(
                        dataset=ds,
                        engine="tensorly contrib.sparse (pydata)",
                        rank=args.rank,
                        iters=k,
                        s_per_iter=round(per_iter, 4),
                        rel_errors=[round(e, 8) for e in errs],
                        same_errors_as_gb=same,
                        load1=round(os.getloadavg()[0], 2),
                    )
                ),
                flush=True,
            )
            assert same, (errs, ref)
        exe = Path(args.splatt) if args.splatt else splatt_exe()
        if args.tns_dir and not exe.exists():
            print(f"(no SPLATT at {exe}: skipped)", flush=True)
        elif args.tns_dir:
            path = tns_file(ds, coords, vals, os.path.expanduser(args.tns_dir))
            for nt in [int(x) for x in args.threads.split(",")]:
                per_iter, fit, total = splatt_cpd(exe, path, args.rank, args.iters + 1, nt)
                print(
                    json.dumps(
                        dict(
                            dataset=ds,
                            engine="SPLATT cpd (own init, CSF two-mode)",
                            nthreads=nt,
                            rank=args.rank,
                            iters=args.iters + 1,
                            s_per_iter=per_iter,
                            final_fit=fit,
                            total_s_incl_read_and_csf=total,
                            load1=round(os.getloadavg()[0], 2),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()

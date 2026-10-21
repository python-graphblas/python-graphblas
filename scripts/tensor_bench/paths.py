"""Where the kit keeps data, results and compiled tools.

Every path can be set by an environment variable, and the scripts' ``--data-dir`` flags set
``TENSOR_BENCH_DATA`` for the processes they start:

- ``TENSOR_BENCH_DATA``: downloaded data and the generated synthetic tensors
  (default: ``data/`` next to this file);
- ``TENSOR_BENCH_RESULTS``: result files (default: ``results/`` next to this file);
- ``TENSOR_BENCH_BUILD``: SPLATT and TACO sources, builds and drivers (default: ``build/``
  next to this file).

The defaults are listed in this directory's ``.gitignore``.
"""

import os
from pathlib import Path

KIT = Path(__file__).resolve().parent


def _dir(env, default):
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else KIT / default


def data_dir():
    return _dir("TENSOR_BENCH_DATA", "data")


def results_dir():
    return _dir("TENSOR_BENCH_RESULTS", "results")


def build_dir():
    return _dir("TENSOR_BENCH_BUILD", "build")


def set_data_dir(path):
    """Use ``path`` in this process and in the processes it starts."""
    if path:
        os.environ["TENSOR_BENCH_DATA"] = str(Path(path).expanduser().resolve())

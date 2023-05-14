#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
conda search 'numpy[channel=conda-forge]>=1.24.3'
conda search 'pandas[channel=conda-forge]>=2.0.1'
conda search 'scipy[channel=conda-forge]>=1.10.1'
conda search 'networkx[channel=conda-forge]>=3.1'
conda search 'awkward[channel=conda-forge]>=2.2.0'
conda search 'sparse[channel=conda-forge]>=0.14.0'
conda search 'fast_matrix_market[channel=conda-forge]>=1.5.1'
conda search 'numba[channel=conda-forge]>=0.56.4'
conda search 'pyyaml[channel=conda-forge]>=6.0'
conda search 'flake8-bugbear[channel=conda-forge]>=23.5.9'
conda search 'flake8-simplify[channel=conda-forge]>=0.20.0'
# conda search 'python[channel=conda-forge]>=3.8 *pypy*'

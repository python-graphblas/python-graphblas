#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
conda search 'flake8-bugbear[channel=conda-forge]>=24.1.17'
conda search 'flake8-simplify[channel=conda-forge]>=0.21.0'
conda search 'numpy[channel=conda-forge]>=1.26.3'
conda search 'pandas[channel=conda-forge]>=2.2.0'
conda search 'scipy[channel=conda-forge]>=1.12.0'
conda search 'networkx[channel=conda-forge]>=3.2.1'
conda search 'awkward[channel=conda-forge]>=2.5.2'
conda search 'sparse[channel=conda-forge]>=0.15.1'
conda search 'fast_matrix_market[channel=conda-forge]>=1.7.6'
conda search 'numba[channel=conda-forge]>=0.59.0'
conda search 'pyyaml[channel=conda-forge]>=6.0.1'
# conda search 'python[channel=conda-forge]>=3.10 *pypy*'

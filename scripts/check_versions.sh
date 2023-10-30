#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
conda search 'flake8-bugbear[channel=conda-forge]>=23.9.16'
conda search 'flake8-simplify[channel=conda-forge]>=0.21.0'
conda search 'numpy[channel=conda-forge]>=1.26.0'
conda search 'pandas[channel=conda-forge]>=2.1.2'
conda search 'scipy[channel=conda-forge]>=1.11.3'
conda search 'networkx[channel=conda-forge]>=3.2.1'
conda search 'awkward[channel=conda-forge]>=2.4.6'
conda search 'sparse[channel=conda-forge]>=0.14.0'
conda search 'fast_matrix_market[channel=conda-forge]>=1.7.4'
conda search 'numba[channel=conda-forge]>=0.57.1'
conda search 'pyyaml[channel=conda-forge]>=6.0.1'
# conda search 'python[channel=conda-forge]>=3.9 *pypy*'

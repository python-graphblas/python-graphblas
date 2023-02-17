#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
conda search 'numpy[channel=conda-forge]>=1.24.2'
conda search 'pandas[channel=conda-forge]>=1.5.3'
conda search 'scipy[channel=conda-forge]>=1.10.0'
conda search 'networkx[channel=conda-forge]>=3.0'
conda search 'awkward[channel=conda-forge]>=2.0.7'
conda search 'numba[channel=conda-forge]>=0.56.4'
conda search 'pyyaml[channel=conda-forge]>=6.0'
conda search 'flake8-comprehensions[channel=conda-forge]>=3.10.1'
conda search 'flake8-bugbear[channel=conda-forge]>=23.2.13'
conda search 'flake8-simplify[channel=conda-forge]>=0.19.3'

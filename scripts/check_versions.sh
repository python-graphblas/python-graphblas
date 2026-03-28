#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
#
# When updating versions throughout the repo (CI, pyproject.toml, pre-commit, etc.),
# also update these version numbers to match the latest versions we currently test.
conda search 'flake8-bugbear[channel=conda-forge]>=25.11.29'
conda search 'flake8-simplify[channel=conda-forge]>=0.30.0'
conda search 'numpy[channel=conda-forge]>=2.4'
conda search 'pandas[channel=conda-forge]>=3.0'
conda search 'scipy[channel=conda-forge]>=1.17'
conda search 'networkx[channel=conda-forge]>=3.6'
conda search 'awkward[channel=conda-forge]>=2.9'
conda search 'sparse[channel=conda-forge]>=0.15'
# fast_matrix_market is deprecated (no longer maintained; last supported Python is 3.12)
conda search 'numba[channel=conda-forge]>=0.64'
conda search 'pyyaml[channel=conda-forge]>=6.0'
conda search 'python-suitesparse-graphblas[channel=conda-forge]>=10.3.1'
# conda search 'python[channel=conda-forge]>=3.11 *pypy*'

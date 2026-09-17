#!/usr/bin/env bash
# Simple helper to check versions of dependencies.
# Use, adjust, copy/paste, etc. as necessary to answer your questions.
# This may be helpful when updating dependency versions in CI.
# Tip: add `--json` for more information.
#
# When updating versions throughout the repo (CI, pyproject.toml, pre-commit, etc.),
# also update these version numbers to match the latest versions we currently test.
#
# scripts/audit_pypi_versions.py and scripts/audit_conda_versions.py ask the same
# questions automatically, and also check the pools and tables in ci_pick_versions.py
# against what is published. The "Audit dependency versions" workflow runs both weekly.
conda search 'flake8-bugbear[channel=conda-forge]>=26.9.9'
conda search 'flake8-simplify[channel=conda-forge]>=0.30.0'
conda search 'numpy[channel=conda-forge]>=2.5'
conda search 'pandas[channel=conda-forge]>=3.0'
conda search 'scipy[channel=conda-forge]>=1.18'
conda search 'networkx[channel=conda-forge]>=3.6'
conda search 'awkward[channel=conda-forge]>=2.13'
conda search 'sparse[channel=conda-forge]>=0.19'
# fast_matrix_market is deprecated (no longer maintained; last supported Python is 3.12)
conda search 'numba[channel=conda-forge]>=0.67'
conda search 'pyyaml[channel=conda-forge]>=6.0'
# matplotlib is installed unpinned for the notebooks job; ci_pick_versions.py caps it
# below 3.11 when the numpy pick predates matplotlib's numpy >=1.25 runtime floor.
conda search 'matplotlib-base[channel=conda-forge]>=3.11'
conda search 'python-suitesparse-graphblas[channel=conda-forge]>=10.5.0'
# conda search 'python[channel=conda-forge]>=3.11 *pypy*'

#!/usr/bin/env bash
#
# Warning!  This will search-and-replace code in a way that is difficult to undo.
# This script may be helpful to debug and test locally if there are errors when
# tests are run the second time in CI (i.e., after the following is applied).
#
# Because this is a dangerous operation, we automatically create `pre_bizarro.patch`
#
# Since Python-graphblas supports both GrB_Scalar and C-scalars, we perform
# search-and-replace to switch defaults in the code to ensure better coverage.
git diff > pre_bizarro.patch
find graphblas -type f -name "*.py" -print0 | xargs -0 sed -i -s \
  -e '/# pragma: is_grbscalar/! s/is_cscalar=False/is_cscalar=True/g' \
  -e '/# pragma: is_grbscalar/! s/is_cscalar = False/is_cscalar = True/g' \
  -e '/# pragma: to_grb/ s/is_cscalar=True/is_cscalar=False/g' \
  -e '/# pragma: to_grb/ s/is_cscalar = True/is_cscalar = False/g'

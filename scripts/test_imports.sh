#!/usr/bin/env bash
#
# Make sure imports work.  Also, this is a good way to measure import performance.
(for attr in Matrix Scalar Vector Recorder _agg agg base binary descriptor \
  dtypes exceptions expr ffi formatting infix init io lib mask matrix monoid \
  op operator scalar select semiring tests unary vector recorder _ss ss
  do echo python -c \"from graphblas import $attr\"
    if ! python -c "from graphblas import $attr"
      then exit 1
    fi
  done
)
(for attr in _agg agg base binary descriptor dtypes exceptions \
  expr formatting infix io mask matrix monoid op operator scalar \
  select semiring tests unary vector recorder _ss ss
  do echo python -c \"import graphblas.$attr\"
    if ! python -c "import graphblas.$attr"
      then exit 1
    fi
  done
)

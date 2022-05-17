#!/usr/bin/env bash
#
# Make sure imports work.  Also, this is a good way to measure import performance.
if ! python -c "from graphblas import * ; Matrix" ; then exit 1 ; fi
if ! python -c "from graphblas import agg, _agg" ; then exit 1 ; fi
if ! python -c "from graphblas.agg import count" ; then exit 1 ; fi
if ! python -c "from graphblas.binary import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.indexunary import tril" ; then exit 1 ; fi
if ! python -c "from graphblas.monoid import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.op import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.select import tril" ; then exit 1 ; fi
if ! python -c "from graphblas.semiring import plus_times" ; then exit 1 ; fi
if ! python -c "from graphblas.unary import exp" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.agg.count" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.binary.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.indexunary.tril" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.monoid.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.op.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.select.tril" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.semiring.plus_times" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.unary.exp" ; then exit 1 ; fi
if ! python -c "from graphblas import agg ; agg.count" ; then exit 1 ; fi
if ! python -c "from graphblas import binary ; binary.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import indexunary ; indexunary.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import monoid ; monoid.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import op ; op.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import select ; select.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import semiring ; semiring.plus_times" ; then exit 1 ; fi
if ! python -c "from graphblas import unary ; unary.exp" ; then exit 1 ; fi

if ! (for attr in Matrix Scalar Vector Recorder _agg agg base binary descriptor \
  dtypes exceptions expr ffi formatting infix init io lib mask matrix monoid \
  op operator scalar select semiring tests unary vector recorder _ss ss \
  _automethods _infixmethods _slice
  do echo python -c \"from graphblas import $attr\"
    if ! python -c "from graphblas import $attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi
if ! (for attr in _agg agg base binary binary.numpy descriptor dtypes exceptions \
  expr formatting infix io mask matrix monoid monoid.numpy op op.numpy operator scalar \
  select semiring semiring.numpy tests unary unary.numpy vector recorder _ss ss \
  _automethods _infixmethods _slice
  do echo python -c \"import graphblas.$attr\"
    if ! python -c "import graphblas.$attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi

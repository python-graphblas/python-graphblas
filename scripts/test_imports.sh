#!/usr/bin/env bash
#
# Make sure imports work.  Also, this is a good way to measure import performance.
if ! python -c "from graphblas import * ; Matrix" ; then exit 1 ; fi
if ! python -c "from graphblas import agg" ; then exit 1 ; fi
if ! python -c "from graphblas.core.operator import agg" ; then exit 1 ; fi
if ! python -c "from graphblas.agg import count" ; then exit 1 ; fi
if ! python -c "from graphblas.binary import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.indexunary import tril" ; then exit 1 ; fi
if ! python -c "from graphblas.monoid import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.op import plus" ; then exit 1 ; fi
if ! python -c "from graphblas.select import tril" ; then exit 1 ; fi
if ! python -c "from graphblas.semiring import plus_times" ; then exit 1 ; fi
if ! python -c "from graphblas.unary import exp" ; then exit 1 ; fi
if ! (for attr in Matrix Scalar Vector Recorder agg binary dtypes exceptions \
  init io monoid op select semiring tests unary ss viz MAX_SIZE
  do echo python -c \"from graphblas import $attr\"
    if ! python -c "from graphblas import $attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi
if ! (for attr in base descriptor expr formatting ffi infix lib mask \
  matrix operator scalar vector recorder automethods infixmethods slice ss
  do echo python -c \"from graphblas.core import $attr\"
    if ! python -c "from graphblas.core import $attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.agg.count" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.binary.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.indexunary.tril" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.monoid.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.op.plus" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.select.tril" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.semiring.plus_times" ; then exit 1 ; fi
if ! python -c "import graphblas as gb ; gb.unary.exp" ; then exit 1 ; fi
if ! (for attr in agg binary binary.numpy dtypes exceptions io monoid monoid.numpy \
  op op.numpy select semiring semiring.numpy tests unary unary.numpy ss viz
  do echo python -c \"import graphblas.$attr\"
    if ! python -c "import graphblas.$attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi
if ! (for attr in base descriptor expr formatting infix mask matrix \
  operator scalar vector recorder automethods infixmethods slice ss
  do echo python -c \"import graphblas.core.$attr\"
    if ! python -c "import graphblas.core.$attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi
if ! python -c "from graphblas import agg ; agg.count" ; then exit 1 ; fi
if ! python -c "from graphblas import binary ; binary.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import indexunary ; indexunary.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import monoid ; monoid.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import op ; op.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import select ; select.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import semiring ; semiring.plus_times" ; then exit 1 ; fi
if ! python -c "from graphblas import unary ; unary.exp" ; then exit 1 ; fi
if ! (for attr in agg unary binary monoid semiring select indexunary base utils
  do echo python -c \"import graphblas.core.operator.$attr\"
    if ! python -c "import graphblas.core.operator.$attr"
      then exit 1
    fi
  done
) ; then exit 1 ; fi

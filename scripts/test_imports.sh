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
if ! (for attr in base descriptor dtypes expr formatting ffi infix lib mask \
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
if ! python -c "from graphblas import agg ; agg.count" ; then exit 1 ; fi
if ! python -c "from graphblas import binary ; binary.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import indexunary ; indexunary.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import monoid ; monoid.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import op ; op.plus" ; then exit 1 ; fi
if ! python -c "from graphblas import select ; select.tril" ; then exit 1 ; fi
if ! python -c "from graphblas import semiring ; semiring.plus_times" ; then exit 1 ; fi
if ! python -c "from graphblas import unary ; unary.exp" ; then exit 1 ; fi

# Every module must import as the first import of a fresh process. These cannot yet:
# a module each one imports imports it back while it is half-initialized
# (core.ss.binary and core.ss.unary through graphblas.binary.ss and graphblas.unary.ss
# as the operators initialize, the others through core.vector and core.matrix).
# The loop fails when one of them starts working, so it can be taken off the list.
not_first=(graphblas.core.ss.binary graphblas.core.ss.matrix graphblas.core.ss.prefix_scan
  graphblas.core.ss.unary graphblas.core.ss.vector)
# udf.py needs Numba, and only code that checks for Numba imports it
needs_numba=()
if ! python -c "import numba" 2> /dev/null ; then needs_numba=(graphblas.core.operator.udf) ; fi
listed() {  # whether $1 is one of the other arguments
  local name=$1 ; shift
  for x in "$@" ; do if [ "$x" = "$name" ] ; then return 0 ; fi ; done
  return 1
}
modules=$(find graphblas -name '*.py' \( ! -path 'graphblas/tests/*' -o -path graphblas/tests/__init__.py \) |
  sed -e 's#/__init__\.py$##' -e 's#\.py$##' -e 's#/#.#g' | sort)
if ! (for module in $modules
  do
    if listed "$module" "${needs_numba[@]}" ; then continue ; fi
    if listed "$module" "${not_first[@]}" ; then
      if python -c "import $module" 2> /dev/null
        then echo "$module now imports first; take it off not_first" ; exit 1
      fi
      continue
    fi
    echo "python -c \"import $module\""
    if ! python -c "import $module"
      then exit 1
    fi
  done
) ; then exit 1 ; fi

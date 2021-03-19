import os
import sys
from cffi import FFI

is_win = sys.platform.startswith("win")

ffibuilder = FFI()

ffibuilder.set_source(
    "grblas.backends.suitesparse._suitesparse_grblas",
    r"""#include "GraphBLAS.h" """,
    libraries=["graphblas"],
    include_dirs=[os.path.join(sys.prefix, "include")],
)

thisdir = os.path.dirname(__file__)

header = "suitesparse_graphblas_4.0.3.h"
if is_win:
    header = "suitesparse_graphblas_no_complex_4.0.3.h"
gb_cdef = open(os.path.join(thisdir, header))

ffibuilder.cdef(gb_cdef.read())

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

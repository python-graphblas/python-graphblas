import os
from cffi import FFI

ffibuilder = FFI()

ffibuilder.set_source(
    "grblas.backends.suitesparse._suitesparse_grblas",
    r"""#include "GraphBLAS.h" """,
    libraries=["graphblas"],
)

thisdir = os.path.dirname(__file__)

gb_cdef = open(os.path.join(thisdir, "suitesparse_graphblas_3.3.1.h"))

ffibuilder.cdef(gb_cdef.read())

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source(
    "_grblas",
    r"""#include "GraphBLAS.h" """,
    libraries=['graphblas'])

gb_cdef = open('cdef/suitesparse_graphblas_3.1.1.h')

ffibuilder.cdef(gb_cdef.read())

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)

from numpy cimport ndarray
from libc.stdint cimport uint64_t

cdef extern from "numpy/arrayobject.h" nogil:
    # These aren't public (i.e., "extern"), but other projects use them too
    void *PyDataMem_NEW(size_t)
    void *PyDataMem_NEW_ZEROED(size_t, size_t)
    void *PyDataMem_RENEW(void *, size_t)
    void PyDataMem_FREE(void *)
    # These are available in newer Cython versions
    void PyArray_ENABLEFLAGS(ndarray, int flags)
    void PyArray_CLEARFLAGS(ndarray, int flags)

ctypedef enum GrB_Mode:
    GrB_NONBLOCKING
    GrB_BLOCKING

ctypedef uint64_t (*GxB_init)(
    GrB_Mode,
    void *(*user_malloc_function)(size_t),
    void *(*user_calloc_function)(size_t, size_t),
    void *(*user_realloc_function)(void *, size_t),
    void (*user_free_function)(void *),
    bint,  # user_malloc_is_thread_safe
)

cpdef int call_gxb_init(ffi, lib, int mode)

cpdef ndarray claim_buffer(ffi, cdata, size_t size, dtype)

cpdef ndarray claim_buffer_2d(ffi, cdata, size_t cdata_size, size_t nrows, size_t ncols, dtype, bint is_c_order)

cpdef unclaim_buffer(ndarray array)


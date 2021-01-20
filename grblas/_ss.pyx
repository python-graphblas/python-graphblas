import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t, uint64_t

np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void *PyDataMem_NEW(size_t)
    void *PyDataMem_NEW_ZEROED(size_t, size_t)
    void *PyDataMem_RENEW(void *, size_t)
    void PyDataMem_FREE(void *)

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

cpdef int call_gxb_init(ffi, lib, int mode):
    cdef GxB_init func = <GxB_init><uintptr_t>int(ffi.cast("uintptr_t", ffi.addressof(lib, "GxB_init")))
    return func(<GrB_Mode>mode, PyDataMem_NEW, PyDataMem_NEW_ZEROED, PyDataMem_RENEW, PyDataMem_FREE, True)


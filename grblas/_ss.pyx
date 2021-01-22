import numpy as np
from numpy cimport import_array, ndarray, npy_intp, PyArray_SimpleNewFromData, NPY_ARRAY_OWNDATA, NPY_ARRAY_WRITEABLE
from libc.stdint cimport uintptr_t

import_array()

cpdef int call_gxb_init(ffi, lib, int mode):
    cdef GxB_init func = <GxB_init><uintptr_t>int(ffi.cast("uintptr_t", ffi.addressof(lib, "GxB_init")))
    return func(<GrB_Mode>mode, PyDataMem_NEW, PyDataMem_NEW_ZEROED, PyDataMem_RENEW, PyDataMem_FREE, True)


cpdef ndarray claim_buffer(ffi, cdata, size_t size, dtype):
    cdef npy_intp dims = size
    cdef uintptr_t ptr = int(ffi.cast("uintptr_t", cdata))
    cdef ndarray array = PyArray_SimpleNewFromData(1, &dims, dtype.num, <void*>ptr)
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA)
    return array


cpdef unclaim_buffer(ndarray array):
    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE)

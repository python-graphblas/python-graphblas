import numpy as np
from numpy cimport import_array, ndarray, npy_intp, PyArray_SimpleNewFromData, NPY_ARRAY_OWNDATA, NPY_ARRAY_WRITEABLE
from libc.stdint cimport uintptr_t

import_array()

cpdef int call_gxb_init(ffi, lib, int mode):
    # We need to call `GxB_init`, but we didn't compile Cython against GraphBLAS.  So, we get it from cffi.
    # Step 1: ffi.addressof(lib, "GxB_init")
    #    Return type: cffi.cdata object of a function pointer.  Can't cast to int.
    # Step 2: ffi.cast("uintptr_t", ...)
    #    Return type: cffi.cdata object of a uintptr_t type, an unsigned pointer.  Can cast to int.
    # Step 3: int(...)
    #    Return type: int.  The physical address of the function.
    # Step 4: <uintptr_t>(...)
    #    Return type: uintptr_t in Cython.  Cast Python int to Cython integer for pointers.
    # Step 5: <GsB_init>(...)
    #    Return: function pointer in Cython!

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

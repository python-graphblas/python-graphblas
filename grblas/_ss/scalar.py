from .. import ffi, lib
from ..exceptions import check_status_carg


def free(scalar):
    status = lib.GxB_Scalar_free(scalar)
    check_status_carg(status, "Scalar", scalar[0])


def gxb_scalar(dtype, value=None):
    scalar = ffi.new("GxB_Scalar*")
    status = lib.GxB_Scalar_new(scalar, dtype._carg)
    check_status_carg(status, "Scalar", scalar[0])
    scalar = ffi.gc(scalar, free)
    if value is not None:
        func = getattr(lib, f"GxB_Scalar_setElement_{dtype.name}")
        status = func(scalar[0], value)
        check_status_carg(status, "Scalar", scalar[0])
    return scalar

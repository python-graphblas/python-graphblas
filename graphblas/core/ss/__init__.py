import suitesparse_graphblas as _ssgb

# Version of the SuiteSparse:GraphBLAS C library, which is what every version
# gate in this package is really asking about. Read from the library's own
# constants rather than parsed out of ``_ssgb.__version__``: the wrapper's
# version normally tracks the library, but a development build between
# releases reports the previous one (10.4.1.0 against a 10.5.0 library), and a
# gate keyed on that takes the wrong branch without saying so. These are cffi
# ``#define`` constants, so they read at import time, before ``GrB_init``.
try:
    version_major = _ssgb.lib.GxB_IMPLEMENTATION_MAJOR
    version_minor = _ssgb.lib.GxB_IMPLEMENTATION_MINOR
    version_bug = _ssgb.lib.GxB_IMPLEMENTATION_SUB
except AttributeError:  # pragma: no cover (build without the GxB defines)
    version_major, version_minor, version_bug = map(int, _ssgb.__version__.split(".")[:3])

_IS_SSGB7 = version_major == 7

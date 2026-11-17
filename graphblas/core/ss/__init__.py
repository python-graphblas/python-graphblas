import suitesparse_graphblas as _ssgb

# The version of the SuiteSparse:GraphBLAS C library, which every version gate in
# this package is about. It comes from the library's constants, not from
# ``_ssgb.__version__``: a development build of the wrapper can report the previous
# release (10.4.1.0 with a 10.5.0 library), and a gate keyed on that would silently
# take the wrong branch. The constants are cffi ``#define`` values, so they can be
# read before ``GrB_init``.
try:
    version_major = _ssgb.lib.GxB_IMPLEMENTATION_MAJOR
    version_minor = _ssgb.lib.GxB_IMPLEMENTATION_MINOR
    version_bug = _ssgb.lib.GxB_IMPLEMENTATION_SUB
except AttributeError:  # pragma: no cover (build without the GxB defines)
    version_major, version_minor, version_bug = map(int, _ssgb.__version__.split(".")[:3])

_IS_SSGB7 = version_major == 7

import suitesparse_graphblas as _ssgb

(version_major, version_minor, version_bug) = map(int, _ssgb.__version__.split(".")[:3])

_IS_SSGB7 = version_major == 7

import sysconfig
from pprint import pprint

import suitesparse_graphblas as _ssgb

(version_major, version_minor, version_bug) = map(int, _ssgb.__version__.split(".")[:3])

_IS_SSGB7 = version_major == 7

# Why are ssjit tests being run for SSGB 7.3.2?
print(  # noqa: T201
    "python-suitesparse-graphblas version:",
    _ssgb.__version__,
    version_major,
    version_minor,
    version_bug,
    _IS_SSGB7,
)
pprint(sysconfig.get_config_vars())  # noqa: T203

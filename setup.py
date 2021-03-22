from setuptools import setup, find_packages, Extension
from glob import glob

try:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import get_directive_defaults

    use_cython = True
except ImportError:
    use_cython = False
import numpy as np
import os
import sys
import versioneer

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if use_cython:
    suffix = ".pyx"
    directive_defaults = get_directive_defaults()
    directive_defaults["binding"] = True
    directive_defaults["language_level"] = 3
    if os.environ.get("CYTHON_COVERAGE"):
        directive_defaults["linetrace"] = True
        define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
else:
    suffix = ".c"

include_dirs = [np.get_include(), os.path.join(sys.prefix, "include")]
ext_modules = [
    Extension(
        name[: -len(suffix)].replace("/", ".").replace("\\", "."),
        [name],
        include_dirs=include_dirs,
        define_macros=define_macros,
    )
    for name in glob(f"grblas/**/*{suffix}", recursive=True)
]
if use_cython:
    ext_modules = cythonize(ext_modules, include_path=include_dirs)

with open("README.md") as f:
    long_description = f.read()

setup(
    name="grblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python interface to GraphBLAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jim Kitchen and Erik Welch",
    author_email="erik.n.welch@gmail.com,jim22k@gmail.com",
    url="https://github.com/metagraph-dev/grblas",
    ext_modules=ext_modules,
    packages=find_packages(exclude=["grblas.backends.python"]),
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    cffi_modules=["grblas/backends/suitesparse/build.py:ffibuilder"],
    python_requires=">=3.7",
    install_requires=["cffi>=1.0.0", "numpy>=1.15", "numba"],
    tests_require=["pytest", "pandas"],
    license="Apache License 2.0",
    keywords=["graphblas", "graph", "sparse", "matrix", "lagraph", "suitesparse"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_data={"grblas": ["*/*.pyx", "*/*.pxd", "backends/suitesparse/*.h"]},
    include_package_data=True,
    zip_safe=False,
)

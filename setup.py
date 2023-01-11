from setuptools import find_packages, setup

extras_require = {
    "repr": ["pandas >=1.2"],
    "io": ["networkx >=2.8", "scipy >=1.8", "awkward >=1.9"],
    "viz": ["matplotlib"],
    "test": ["pytest", "pandas >=1.2", "scipy >=1.8"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

with open("README.md") as f:
    long_description = f.read()

setup(
    name="python-graphblas",
    description=(
        "Python library for GraphBLAS: high-performance sparse linear algebra "
        "for scalable graph analytics"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Erik Welch and Jim Kitchen",
    author_email="erik.n.welch@gmail.com,jim22k@gmail.com",
    url="https://github.com/python-graphblas/python-graphblas",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "suitesparse-graphblas >=7.4.0.0, <7.5",
        "numpy >=1.21",
        "numba >=0.55",
        "donfig >=0.6",
        "pyyaml >=5.4",
    ],
    extras_require=extras_require,
    include_package_data=True,
    license="Apache License 2.0",
    keywords=["graphblas", "graph", "sparse", "matrix", "lagraph", "suitesparse"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
)

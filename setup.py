from setuptools import setup, find_packages
import versioneer

extras_require = {
    "repr": ["pandas"],
    "io": ["networkx", "scipy"],
    "viz": ["matplotlib"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

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
    packages=find_packages(exclude=["grblas.backends.python"]),
    setup_requires=["pytest-runner"],
    python_requires=">=3.7",
    install_requires=["suitesparse-graphblas", "numba"],
    tests_require=["pytest", "pandas"],
    extras_require=extras_require,
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
    zip_safe=False,
)

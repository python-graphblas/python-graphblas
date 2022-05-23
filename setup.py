from setuptools import find_packages, setup

import versioneer

extras_require = {
    "repr": ["pandas"],
    "io": ["networkx", "scipy"],
    "viz": ["matplotlib"],
    "test": ["pytest", "pandas"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

with open("README.md") as f:
    long_description = f.read()

setup(
    name="python-graphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python interface to GraphBLAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jim Kitchen and Erik Welch",
    author_email="erik.n.welch@gmail.com,jim22k@gmail.com",
    url="https://github.com/metagraph-dev/python-graphblas",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["suitesparse-graphblas >=7.1.0.0, <7.2", "numba", "donfig", "pyyaml"],
    extras_require=extras_require,
    include_package_data=True,
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

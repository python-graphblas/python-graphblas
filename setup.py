from setuptools import setup, find_packages

setup(
    name='grblas',
    version='1.3.0',
    description='Python interface to GraphBLAS',
    author='Jim Kitchen and Erik Welch',
    url='https://github.com/metagraph-dev/grblas',
    packages=find_packages(exclude=['grblas.backends.python']),
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    cffi_modules=["grblas/backends/suitesparse/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    tests_require=["pytest", "pandas"],
)

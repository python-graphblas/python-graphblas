from setuptools import setup

setup(
    name='grblas',
    version='1.2.6',
    description='Python interface to GraphBLAS',
    author='Jim Kitchen',
    packages=['grblas', 'grblas/backends', 'grblas/backends/suitesparse',
              'grblas/binary', 'grblas/monoid', 'grblas/semiring', 'grblas/unary'],
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    cffi_modules=["grblas/backends/suitesparse/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    tests_require=["pytest"],
)

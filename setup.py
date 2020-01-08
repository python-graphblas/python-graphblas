from setuptools import setup

setup(
    name='grblas',
    version='1.2.2',
    description='Python interface to GraphBLAS',
    author='Jim Kitchen',
    packages=['grblas'],
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    cffi_modules=["grblas/backends/suitesparse/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    tests_require=["pytest"],
)

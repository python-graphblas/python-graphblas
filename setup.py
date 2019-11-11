from setuptools import setup

setup(
    name='grblas',
    version='1.2.0',
    description='Python interface to GraphBLAS',
    author='Jim Kitchen',
    packages=['grblas'],
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
)

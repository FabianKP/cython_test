from setuptools import setup, Extension


# Build with "python setup.py build_ext --inplace"


module = Extension ('apgd_cython', sources=['apgd_cython.pyx'])

setup(
    name='cythonTest',
    author='fabian',
    ext_modules=[module]
)
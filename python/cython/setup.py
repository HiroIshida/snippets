from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include 

ext = Extension("sample", sources=["sample.pyx"], include_dirs=['.', get_include()])
setup(name="sample", ext_modules=cythonize([ext]))

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include 

ext = Extension("low_variance_sampler", sources=["low_variance_sampler.pyx"], include_dirs=['.', get_include()])
setup(name="sampler", ext_modules=cythonize([ext]))

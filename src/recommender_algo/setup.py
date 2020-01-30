from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Editable SVD',
      ext_modules=cythonize("editable_svd.pyx"),
      include_dirs=[numpy.get_include()])
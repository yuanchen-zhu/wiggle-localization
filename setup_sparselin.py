#!/usr/bin/python

from distutils.core import setup, Extension
import numpy

module1 = Extension('sparselin',
                    include_dirs = [numpy.get_include()],
                    libraries = ['superlu', 'arpack++'],
                    library_dirs = [],
                    sources = ['sparselin.cc'],
                    extra_compile_args = ['-fpermissive'])

setup (name = 'sparselin',
       version = '0.1',
       description = 'Sparse linear algebra routines',
       ext_modules = [module1])

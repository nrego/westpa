from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    from Cython.Distutils import build_ext
    ext_modules = [Extension("utils", ["utils.pyx"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])] 
    setup(name = 'state assignment util',
          cmdclass = {'build_ext': build_ext},
          ext_modules = ext_modules)
                
except:
    ext_modules = [Extension("utils", ["utils.c"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])]
               
    setup(
      name = 'state assignment util',
      ext_modules = ext_modules
    )

from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    from Cython.Distutils import build_ext
    ext_modules = [Extension("ElasticNetwork", ["elasticnetwork.pyx", "randomkit.c"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])]
                
    setup(name = 'Two-state Elastic Network Model',
          cmdclass = {'build_ext': build_ext},
          ext_modules = ext_modules)
                
except:
    ext_modules = [Extension("ElasticNetwork", ["elasticnetwork.c", "randomkit.c"],
                include_dirs=[numpy_include],
                extra_compile_args=["-O3","-ffast-math"])]
               
    setup(
      name = 'Two-state Elastic Network Model',
      ext_modules = ext_modules
    )


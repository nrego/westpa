from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# Handle cython modules
try:
    from Cython.Distutils import build_ext
    use_cython = True
    cmdclass = {'build_ext': build_ext}
except ImportError:
    use_cython = False
    cmdclass = {}
finally:
    print '---------------------------------'
    print 'Using cython: {}'.format(use_cython)
    print '---------------------------------'

fext = 'pyx' if use_cython is True else 'c'

ff_files = ["ForceFields.{}".format(fext),"ForceFields.pxd"] if use_cython else ["ForceFields.{}".format(fext),]

ext_modules = [Extension("ForceFields", ff_files,
            include_dirs=[numpy_include],
            extra_compile_args=["-O3","-ffast-math"]),
            Extension("cIntegrator", ["cIntegrator.{}".format(fext),"randomkit.c"],
            include_dirs=[numpy_include],
            extra_compile_args=["-O3","-ffast-math"]),
            Extension("cIntegratorSimple", ["cIntegratorSimple.{}".format(fext),"randomkit.c"],
            include_dirs=[numpy_include],
            extra_compile_args=["-O3","-ffast-math"])]
            
setup(name = 'Python ForceFields and Integrator modules',
      cmdclass = cmdclass, 
      ext_modules = ext_modules)
                


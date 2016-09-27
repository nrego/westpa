import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double exp(double)
    double cos(double)
    double sin(double)
    double atan2(double,double)
    double sqrt(double)

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef class Force:
    cpdef int evaluate(self,np.ndarray[DTYPE_t, ndim=1] coords, np.ndarray[DTYPE_t, ndim=1] force)

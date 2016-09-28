import numpy as np
cimport numpy as np

import cython

cdef extern from "math.h":
    float rintf(float)

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.float32_t coord_t

@cython.boundscheck(False)
@cython.wraparound(False)
def dfunc(coord_t[:] p, coord_t[:,::1] centers):
    cdef unsigned int k
    cdef int ncenters = centers.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty((ncenters,), dtype=DTYPE)
    cdef float pp_x, pp_y

    for k in xrange(ncenters):
        pp_x = p[0]
        pp_y = p[1] - centers[k,1]
        pp_y -= rintf(pp_y)
        pp_y += centers[k,1]

        d[k] = (pp_x - centers[k,0])**2 + (pp_y - centers[k,1])**2

    return d


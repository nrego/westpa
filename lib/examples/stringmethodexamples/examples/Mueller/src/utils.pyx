import numpy as np
cimport numpy as np

import cython
from libc.math cimport fabs

DEF FLOAT_INFP = float('+inf')
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.float32_t coord_t

@cython.boundscheck(False)
@cython.wraparound(False)
def update_rate_stats(np.ndarray[DTYPE_t, ndim=2] coords, unsigned int last_state):
    cdef unsigned int k, trans_to_0, trans_to_1, time_in_0, time_in_1, state
    cdef double R2, sigma, x0, x1

    trans_to_0 = trans_to_1 = time_in_0 = time_in_1 = 0
    R2 = 1.0
    x0 = -3.0
    x1 = 3.0

    for k in xrange(coords.shape[0]):
        if (coords[k,0] - x0)**2 + coords[k,1]**2 < R2:
            time_in_0 += 1
            if last_state == 1:
                trans_to_0 += 1
                last_state = 0
        elif (coords[k,0] - x1)**2 + coords[k,1]**2 < R2:
            time_in_1 += 1
            if last_state == 0:
                trans_to_1 += 1
                last_state = 1
        else:
            if last_state == 0:
                time_in_0 += 1
            else:
                time_in_1 += 1

    return last_state, trans_to_0, trans_to_1, time_in_0, time_in_1

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_wemd_rate(np.ndarray[DTYPE_t, ndim=3] pcoords, np.ndarray[np.float64_t, ndim=1] weight):
    cdef unsigned int k
    cdef double flux_to_A, flux_to_B, wA, wB
    cdef double start_state, end_state, w, tol

    flux_to_A = flux_to_B = wA = wB = 0.0
    tol = 1.0E-3

    for k in xrange(pcoords.shape[0]):
        start_state = pcoords[k,0,2]
        end_state = pcoords[k,1,2]
        w = weight[k]

        if fabs(start_state - 0.0) < tol:
            wA += w
            if fabs(end_state - 1.0) < tol:
                flux_to_B += w
        else:
            wB += w
            if fabs(end_state - 0.0) < tol:
                flux_to_A += w

    return flux_to_A, flux_to_B, wA, wB
            
@cython.boundscheck(False)
@cython.wraparound(False)
def dfunc(coord_t[:] p, coord_t[:,::1] centers):
    cdef unsigned int k
    cdef int slen = centers.shape[0] // 2
    cdef double state = p[2]
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty((centers.shape[0],),dtype=DTYPE)

    for k in xrange(d.shape[0]):
        d[k] = FLOAT_INFP

    if fabs(state - 0.0) < 1.0E-3:
        for k in xrange(slen):
            d[k] = (p[0] - centers[k,0])**2 + (p[1] - centers[k,1])**2
    else:
        for k in xrange(slen):
            d[k+slen] = (p[0] - centers[k+slen,0])**2 + (p[1] - centers[k+slen,1])**2

    return d

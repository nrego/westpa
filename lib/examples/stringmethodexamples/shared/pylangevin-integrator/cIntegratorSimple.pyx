import numpy as np
cimport numpy as np
cimport cython
cimport ForceFields

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

include "Python.pxi"

cdef extern from "math.h":
    double sqrt(double)
    double floor(double)
 
cdef extern from "randomkit.h": 
    ctypedef struct rk_state: 
        unsigned long key[624] 
        int pos 
        int has_gauss 
        double gauss 
    void rk_seed(unsigned long seed, rk_state *state) 
    double rk_gauss(rk_state *state)

cdef class SimpleIntegrator:
    cdef double _beta, _mass, _xi, _dt, _invmass, _sigma, _fp
    cdef unsigned int _dims
    cdef ForceFields.Force _ff
    cdef rk_state *rng_state
    cdef DTYPE_t* boxsize
    cdef DTYPE_int_t* isperiodic 
    
    def __cinit__(self,ForceFields.Force forcefield, double mass, double xi, double beta, double dt, unsigned int dims,
                    np.ndarray[DTYPE_int_t,ndim=1] isperiodic, np.ndarray[DTYPE_t, ndim=1] boxsize, unsigned long seed):
        
        self.rng_state = <rk_state*>PyMem_Malloc(sizeof(rk_state)) 
        self.boxsize = <DTYPE_t*>PyMem_Malloc(dims*sizeof(DTYPE_t))
        self.isperiodic = <DTYPE_int_t*>PyMem_Malloc(dims*sizeof(DTYPE_int_t))
        
    def __dealloc__(self): 
        if self.rng_state != NULL: 
            PyMem_Free(self.rng_state) 
            self.rng_state = NULL
        if self.boxsize != NULL: 
            PyMem_Free(self.boxsize) 
            self.boxsize = NULL
        if self.isperiodic != NULL: 
            PyMem_Free(self.isperiodic) 
            self.isperiodic = NULL
    
    def __init__(self,ForceFields.Force forcefield, double mass, double xi, double beta, double dt, unsigned int dims,
                    np.ndarray[DTYPE_int_t,ndim=1] isperiodic, np.ndarray[DTYPE_t, ndim=1] boxsize, unsigned long seed):

        cdef unsigned int k

        self._beta = beta
        self._mass = mass
        self._xi = xi
        self._dt = dt
        self._dims = dims

        self._ff = forcefield
        
        self._sigma = sqrt(2.0*dt/(mass*beta*xi))
        self._fp = dt/(mass*xi)
    
        for k in xrange(self._dims):
            self.boxsize[k] = boxsize[k]
            self.isperiodic[k] = isperiodic[k] 
        
        rk_seed(seed, self.rng_state)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def step(self, np.ndarray[DTYPE_t, ndim=1] coords, int steps):
        cdef unsigned int k,d
        cdef double eta
        cdef np.ndarray[DTYPE_t, ndim=1] F = np.zeros((self._dims,),dtype=DTYPE)
        
        for k in xrange(steps):
            self._ff.evaluate(coords,F)
            
            for d in xrange(self._dims):
                eta = rk_gauss(self.rng_state)
                #print(eta)
                coords[d] = coords[d] + self._fp*F[d] + self._sigma*eta

                # Wrap the coordinates if periodic in dimension
                if self.isperiodic[d] == 1:
                    coords[d] = coords[d] - floor(coords[d]/self.boxsize[d])*self.boxsize[d]

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def step_save(self, np.ndarray[DTYPE_t, ndim=1] coords, int steps, int save_freq):
        cdef unsigned int k,d
        cdef double eta
        cdef np.ndarray[DTYPE_t, ndim=1] F = np.zeros((self._dims,),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] traj = np.zeros((steps/save_freq, self._dims), dtype=DTYPE)
        
        cdef int c = 0

        for k in xrange(steps):
            self._ff.evaluate(coords,F)
            
            for d in xrange(self._dims):
                eta = rk_gauss(self.rng_state)
                coords[d] = coords[d] + self._fp*F[d] + self._sigma*eta

                # Wrap the coordinates if periodic in dimension
                if self.isperiodic[d] == 1:
                    coords[d] = coords[d] - floor(coords[d]/self.boxsize[d])*self.boxsize[d]

            if k % save_freq == 0:
                for d in xrange(self._dims):
                    traj[c,d] = coords[d]
                c += 1

        return traj


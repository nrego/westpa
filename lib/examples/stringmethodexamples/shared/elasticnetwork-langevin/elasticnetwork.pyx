import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t

include "Python.pxi"

from libc.math cimport sqrt, exp, log


cdef extern from "randomkit.h": 
    ctypedef struct rk_state: 
        unsigned long key[624] 
        int pos 
        int has_gauss 
        double gauss 
    void rk_seed(unsigned long seed, rk_state *state)
    double rk_gauss(rk_state *state)


cdef class ElasticNetwork:
    cdef double timefac,h,h2,halfh,h32
    cdef double boltz,mass,rmass,T,kBT,beta,gamma,noise
    cdef double sigma,sigma2,eps,eps12,betam,rbetam

    cdef public unsigned int natoms # number of atoms 

    cdef rk_state *rng_state
    cdef np.ndarray kAB,DA,DB,distA,distB

    def __cinit__(self):
        self.rng_state = <rk_state*>PyMem_Malloc(sizeof(rk_state))


    def __dealloc__(self):
        if self.rng_state != NULL: 
            PyMem_Free(self.rng_state) 
            self.rng_state = NULL

    def __init__(self,distA=None,distB=None,DA=None,DB=None,kAB=None,
                dt=0.0025,gamma=30.0,mass=100.0,temp=300.0,sigma=2.5,
                eps=1.0,betamix=0.005,unsigned long seed=332894241):
        #------------------------------------------
        # CHARMM UNITS
        #Variable   AKMA            SI
        #Length     1 A             1 x 10-10 m  
        #Energy     1 kcal/mol      4186 J/mol  
        #Mass       1 AMU           1.661 x 10-27 kg  
        #Charge     1 electron      1.602 x 10-19 C  
        #Time       1 time unit     0.04888 x 10-12 sec  
        #Force      1 kcal/mol-A    6.95 x 10-1 N  


        # physical constants in SI units
        # ------------------------------
        #     Kb = 1.380662 E-23 J/K
        #     Na = 6.022045 E23  1/mol
        #     e = 1.6021892 E-19 C
        #     eps = 8.85418782 E-12 F/m
        #     1 Kcal = 4184.0 J
        #     1 amu = 1.6605655 E-27 Kg
        #     1 A = 1.0 E-10 m
        # ----------------------------------------- 
        #conversion factor from AKMA time to picoseconds.
        #(TIMFAC = SQRT ( ( 1A )**2 * 1amu * Na  / 1Kcal )  (ps)
        self.timefac=0.04888821

        #timestep ps
        self.h=dt/self.timefac #(2.5 fs)
        self.h2=self.h*self.h
        self.halfh=0.5*self.h 
        self.h32=sqrt(self.h**3)

        # Boltzmann constant in AKMA units
        self.boltz=0.001987191

        # mass in amu
        self.mass=mass
        self.rmass=1.0/self.mass

        #temperature K 
        self.T=temp
        self.kBT=self.boltz*self.T
        self.beta=1.0/self.kBT

        # friction ps^{-1} converted into akma
        self.gamma=gamma*self.timefac
        self.noise=sqrt(2.0*self.kBT*self.gamma/self.mass)

        #---------------------------------------
        # potential

        # for UR
        self.sigma=sigma
        self.sigma2=self.sigma*self.sigma
        self.eps=eps
        self.eps12 = 12.0*self.eps
        # for UA and UB
        self.betam=betamix
        self.rbetam=1.0/self.betam
        # -----------------------------------------
        
        self.kAB = kAB
        self.DA = DA
        self.DB = DB
        self.distA = distA
        self.distB = distB

        self.natoms = self.distA.shape[0]

        rk_seed(seed, self.rng_state)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double calculate_force(self,np.ndarray[DTYPE_t,ndim=2] pos, np.ndarray[DTYPE_t,ndim=2] force):
        cdef unsigned int i,j,k
        cdef double rij2,rij,rrij,dfac
        cdef double UR,UA,UB,expUA,expUB
        cdef np.ndarray[DTYPE_t, ndim=1] dij = np.empty((3,),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] fR = np.zeros((self.natoms,3),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] fA = np.zeros((self.natoms,3),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] fB = np.zeros((self.natoms,3),dtype=DTYPE)

        # Create a method local fast buffer
        cdef np.ndarray[DTYPE_t, ndim=2] kAB, distA, distB 
        cdef np.ndarray[DTYPE_int_t, ndim=2] DA, DB 

        kAB = self.kAB
        DA = self.DA.astype(np.int32)
        DB = self.DB.astype(np.int32)
        distA = self.distA
        distB = self.distB

        cdef double alpha=0.73
        cdef double alpha13=alpha**13
        cdef double rcap=self.sigma*alpha
        cdef double fcap=self.eps12/(self.sigma*alpha13)

        UA = 0.0
        UB = 0.0
        UR = 0.0

        for i in xrange(0,self.natoms-1):
            for j in xrange(i+1,self.natoms):
                
                rij2 = 0.0
                
                for k in xrange(3):
                    dij[k] = pos[i,k] - pos[j,k]
                    rij2 += dij[k]**2

                rij  = sqrt(rij2)
                rrij = 1.0/rij

                for k in xrange(3):
                    dij[k] *= rrij

                # UR
                dfac = (self.sigma2/rij2)**6
                UR += dfac

                if rij < rcap:
                    dfac = fcap
                else:
                    dfac = self.eps12*dfac*rrij

                for k in xrange(3):
                    fR[i,k] += dfac*dij[k]
                    fR[j,k] -= dfac*dij[k]

                # UA and UB
                if DA[i,j] == 1:
                    dfac = rij-distA[i,j]
                    UA += kAB[i,j]*dfac*dfac
                    dfac *= -kAB[i,j]  #minus because it's the force

                    for k in xrange(3):
                        fA[i,k] += dfac*dij[k]
                        fA[j,k] -= dfac*dij[k]

                if DB[i,j] == 1:
                    dfac = rij - distB[i,j]
                    UB += kAB[i,j]*dfac*dfac
                    dfac *= -kAB[i,j] 

                    for k in xrange(3):
                        fB[i,k] += dfac*dij[k]
                        fB[j,k] -= dfac*dij[k]

        UR = self.eps*UR
        UA = 0.5*UA
        UB = 0.5*UB

        expUA = exp(-self.betam*UA)
        expUB = exp(-self.betam*UB)
        dfac = expUA + expUB

        epot = -self.rbetam*log(dfac) + UR

        expUA=expUA/dfac
        expUB=expUB/dfac

        for i in xrange(0,self.natoms):
            for k in xrange(3):
                fA[i,k] *= expUA
                fB[i,k] *= expUB
                force[i,k] = fA[i,k] + fB[i,k] + fR[i,k]
                # acceleration
                force[i,k] *= self.rmass

        return epot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def step(self,np.ndarray[DTYPE_t,ndim=2] pos, np.ndarray[DTYPE_t,ndim=2] vel, unsigned int nsteps):
        cdef unsigned int i,k,m
        cdef double eta,xsi

        cdef np.ndarray[DTYPE_t, ndim=2] force = np.empty((self.natoms,3),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] randq = np.empty((self.natoms,3),dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] randv = np.empty((self.natoms,3),dtype=DTYPE)

        cdef double b1 = 1.0-self.halfh*self.gamma + 0.125*self.h2*self.gamma**2
        cdef double b2 = self.halfh - 0.125*self.h2*self.gamma

        cdef double r1 = self.noise*(sqrt(self.h) - 0.25*self.h32*self.gamma)
        cdef double r2 = self.h32*self.noise*0.5/sqrt(3.0)

        # Calculate initial forces based on position
        self.calculate_force(pos,force)

        for m in xrange(nsteps):

            # First update
            # vel(n) --> vel(n+1/2)
            # pos(n) --> pos(n+1)
            for i in xrange(self.natoms):
                for k in xrange(3):
                    xsi = rk_gauss(self.rng_state)
                    eta = rk_gauss(self.rng_state)
                    randq[i,k] = eta*r2
                    randv[i,k] = 0.5*(xsi*r1-randq[i,k]*self.gamma)

                for k in xrange(3):
                    # (1) velocities half-step
                    vel[i,k] = vel[i,k]*b1 + force[i,k]*b2 + randv[i,k]
                    # (2) positions full step 
                    pos[i,k] += self.h*vel[i,k] + randq[i,k]

            self.calculate_force(pos,force)

            # update velocities to velocities full step
            for i in xrange(self.natoms):
                for k in xrange(3):
                    vel[i,k] = vel[i,k]*b1 + force[i,k]*b2 + randv[i,k]

        return 


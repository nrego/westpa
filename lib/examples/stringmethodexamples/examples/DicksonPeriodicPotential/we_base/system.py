from __future__ import division, print_function; __metaclass__ = type

import time
import os
import numpy as np
import scipy

import west
from west.propagators import WESTPropagator
from west import Segment, WESTSystem
from westpa.binning import VoronoiBinMapper
from westext.stringmethod import DefaultStringMethod
from westext.stringmethod.fourier_fitting import FourierFit

import cIntegratorSimple
import ForceFields
from utils import dfunc as dfunc

import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

pcoord_dtype = np.float32


def genrandint():
    'Generates a random integer between 0 and (2^32)-1'
    x = 0
    for i in range(4):
        x = (x << 8)+ord(os.urandom(1))
    return x


class SimpleLangevinPropagator(WESTPropagator):

    def __init__(self, rc=None):
        super(SimpleLangevinPropagator, self).__init__(rc)

        rc = self.rc.config['west', 'simplelangevin']
        self.ndim = rc.get('ndim', 2)
        self.nsteps = rc.get('blocks_per_iteration', 2)
        self.nsubsteps = rc.get('steps_per_block')
        self.alpha = rc.get('alpha')

        ff = ForceFields.Dickson2dPeriodicForce_revised(self.alpha)

        MASS = 1.0
        XI = 1.5
        BETA = 4.0 
        NDIMS = 2
        DT = 0.002
        ISPERIODIC = np.array([0,1],dtype=np.int)
        BOXSIZE = np.array([1.0E8,1.0], dtype=pcoord_dtype)

        self.integrator = cIntegratorSimple.SimpleIntegrator(ff,MASS,XI,BETA,DT,NDIMS,ISPERIODIC,BOXSIZE,genrandint())

    def get_pcoord(self, state):
        pcoord = None
        if state.label == 'initA':
            pcoord = [0.05, 0.5]

        state.pcoord = pcoord

    def propagate(self,segments):

        for segment in segments:
            starttime = time.time()

            new_pcoords = np.empty((self.nsteps,self.ndim), dtype=pcoord_dtype)
            new_pcoords[0,:] = segment.pcoord[0,:]

            x = new_pcoords[0,:].copy()
            
            for istep in xrange(1,self.nsteps):
                self.integrator.step(x,self.nsubsteps)
                new_pcoords[istep,:] = x

            segment.pcoord = new_pcoords[...]
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime

        return segments


def dfunc_orig(p, centers):
    isperiodic = np.array([0,1],dtype=np.int)
    boxsize = np.array([1.0e8,1.0])

    pp = p - centers

    for k in xrange(len(isperiodic)):
        if isperiodic[k] == 1:
            pp[:,k] -= np.rint(pp[:,k]/boxsize[k])*boxsize[k]

    pp += centers
    return np.sqrt(np.sum((pp-centers)**2,axis=1))


def average_position(self, n_iter):

    isperiodic = np.array([0,1],dtype=np.int)
    boxsize = np.array([1.0e8,1.0])

    nbins = self.system.bin_mapper.nbins
    ndim = self.system.pcoord_ndim

    avg_pos = np.zeros((nbins, ndim), dtype=self.system.pcoord_dtype)
    sum_bin_weight = np.zeros((nbins,), dtype=self.system.pcoord_dtype)

    start_iter = max(n_iter - min(self.windowsize, n_iter), 1)
    stop_iter = n_iter + 1

    for n in xrange(start_iter, stop_iter):
        with self.data_manager.lock:
            iter_group = self.data_manager.get_iter_group(n)
            seg_index = iter_group['seg_index'][...]

            pcoords = iter_group['pcoord'][:,-1,:]  # Only read final point
            bin_indices = self.system.bin_mapper.assign(pcoords)
            weights = seg_index['weight']

            uniq_indices = np.unique(bin_indices)

            for indx in uniq_indices:

                ii = np.where(bin_indices == indx)[0]
                bpc = pcoords[ii,:].copy()

                # Wrap coordinates into the same image as the current center
                xref = self.strings.centers[indx,:]

                for k in xrange(ndim):
                    if isperiodic[k] == 1:
                        xoffset = bpc[:,k] - xref[k]
                        xoffset -= np.rint(xoffset/boxsize[k])*boxsize[k]
                        bpc[:,k] = xoffset + xref[k]

                pcoord_w = bpc * weights[ii][:,np.newaxis]

                avg_pos[indx,:] += pcoord_w.sum(axis=0)

            sum_bin_weight += np.bincount(bin_indices.astype(np.int),weights=weights,minlength=nbins)

    # Some bins might have zero samples so exclude to avoid divide by zero
    occ_ind = np.nonzero(sum_bin_weight)
    avg_pos[occ_ind] /= sum_bin_weight[occ_ind][:,np.newaxis]

    return avg_pos, sum_bin_weight


class System(WESTSystem):

    def initialize(self):
        rc = self.rc.config['west', 'system']

        # pcoord parameters
        self.pcoord_ndim = 2
        self.pcoord_len = 2
        self.pcoord_dtype = pcoord_dtype
        self.target_count = rc.get('target_count')
        self.nbins = rc.get('nbins')

        y = np.linspace(0.05,0.95,self.nbins)
        centers = np.zeros((self.nbins,self.pcoord_ndim),dtype=self.pcoord_dtype)
        centers[:,1] = y

        self.bin_mapper = VoronoiBinMapper(dfunc, centers)
        self.bin_target_counts = np.zeros((self.bin_mapper.nbins,), dtype=np.int_)
        self.bin_target_counts[...] = self.target_count

        # string method parameters
        self.sm_params = {'slen':[self.nbins],
                          'kappa':0.01,
                          'fixed_ends':False,
                          'isperiodic':np.array([0,1],dtype=np.int),
                          'boxsize':np.array([1.0e8,1.0]),
                          'sciflag':True
                         }


class PeriodicLinkedStringMethod(DefaultStringMethod):
    def __init__(self, centers,isperiodic=None,boxsize=None,**kwargs):
        super(PeriodicLinkedStringMethod, self).__init__(centers,**kwargs)

        # Create dict to hold kappan and A objects for all unique lengths of strings
        self._kappan = {}
        self._A = {}

        self._isperiodic = isperiodic
        self._boxsize = boxsize

        self.finalize_init()

    def finalize_init(self):
        # Set up A and kappan for each string
        uslen = np.unique(self._slen)

        for ulen in uslen:
            self._kappan[ulen] = self._kappa * self._dtau * ulen
            self._A[ulen] = None

            if False: #self._SCIPY_FLAG:
                ud = np.zeros((ulen+2,))
                ld = np.zeros((ulen+2,))
                d = np.ones((ulen+2,))

                d[1:-1] = 2.0*self._kappan[ulen] + 1.0
                ud[2:] = -self._kappan[ulen]
                ud[1] = -1.0
                ld[:-2] = -self._kappan[ulen]
                ld[-2] = -1.0
                self._A[ulen] = np.mat([ud,d,ld])

            else:
                self._A[ulen] = np.eye(ulen+2)
                di = np.diag_indices(ulen+2, ndim=2)
                ii = (di[0][1:-1],di[1][1:-1])

                self._A[ulen][ii] = 2.0*self._kappan[ulen] + 1.0

                dd = np.zeros((ulen+2-1,))
                dd[1:] = -self._kappan[ulen]
                self._A[ulen] += np.diag(dd,k=1)

                dd = np.zeros((ulen+2-1,))
                dd[:-1] = -self._kappan[ulen]
                self._A[ulen] += np.diag(dd,k=-1)

                self._A[ulen][-1,1] = -1.0
                self._A[ulen][0,-2] = -1.0

            print('A')
            print(self._A[ulen])

    def update_string_centers(self, avgcoords, binprob):
        """ Update the position of all string centers
        **Parameters**
        avgcoords:      Average position of replicas in each voronoi cell
        binprob:        The total weight in each voronoi cell

        """
        assert self.centers.shape == avgcoords.shape

        # If centers are paired, calculate their weighted average position
        if self._mpairs is not None:
            for pi in self._mpairs:
                totprob = np.sum(binprob[pi])
                if totprob == 0.0:
                    continue
                wt = binprob[pi]/totprob
                idx = np.ix_(pi,self._indx_take)
                wavg = np.average(avgcoords[idx],weights=wt,axis=0)
                avgcoords[idx] = wavg

        for sid,si in enumerate(self._strindx):

            x = avgcoords.copy()[si]
            centers = self.centers[si]
            occupied = np.nonzero(binprob[si[0]])
            N = self._slen[sid]

            # if avgcoords has missing values fill them by linearly interpolating
            # present data
            if occupied[0].shape != N:
                notocc = np.ones((N,),dtype=np.bool)  # unoccupied
                notocc[occupied] = False
                cfunc = lambda z: z.nonzero()[0]

                # Handle ends first
                if notocc[0]:
                    x[0,:] = centers[0,:]
                    notocc[0] = False
                if notocc[-1]:
                    x[-1,:] = centers[-1,:] 
                    notocc[-1] = False

                # interpolate values for unoccupied bins
                if self._SCIPY_FLAG:
                    for k in xrange(self._ndim_take):
                        f = scipy.interpolate.interp1d(cfunc(~notocc),x[~notocc,k],kind='linear')
                        x[notocc,k] = f(cfunc(notocc))
                else:
                    for k in xrange(self._ndim_take):
                        x[notocc,k] = np.interp(cfunc(notocc),cfunc(~notocc),x[~notocc,k])

            if self._fixed_ends:
                x[0,:] = centers[0,:]
                x[-1,:] = centers[-1,:]

            psi = centers
            b = psi - self._dtau*(psi - x)

            # Update and smooth the string
            psi_aug = np.zeros((psi.shape[0]+2,psi.shape[1]))
            b_aug = np.zeros_like(psi_aug)
            b_aug[1:-1,:] = b

            # determine sign conventions to fill in b_aug
            for k in xrange(self._ndim_take):
                if self._isperiodic[k]:
                    dx = b[0,k] - b[-1,k]
                    s = np.rint(dx/self._boxsize[k])
                    b_aug[0,k] = s*self._boxsize[k]
                    b_aug[-1,k] = -s*self._boxsize[k]

            if False: #self._SCIPY_FLAG:
                for k in xrange(self._ndim_take):
                    psi_aug[:,k] = scipy.linalg.solve_banded((1,1),self._A[N],b_aug[:,k])
            else:
                for k in xrange(self._ndim_take):
                    psi_aug[:,k] = np.linalg.solve(self._A[N],b_aug[:,k])

            # smooth using fourier method
            P = 2
            w0 = np.zeros((2,P),np.float)
            t0 = np.linspace(0,1,psi_aug.shape[0])

            ff = FourierFit(P=P,maxiters=100)
            ff.optimize(psi_aug,None,w0,t0)
            psi_aug = ff.pp[-1][:]

            # Enforce equal spacing between centers along the string
            L = self.calculate_length(psi_aug)
            L /= L[-1]
            g2 = np.linspace(0,1,N+2)

            if self._SCIPY_FLAG:
                for k in xrange(self._ndim_take):
                    f = scipy.interpolate.interp1d(L,psi_aug[:,k],kind='linear')
                    psi_aug[:,k] = f(g2)
            else:
                for k in xrange(self._ndim_take):
                    psi_aug[:,k] = np.interp(g2,L,psi_aug[:,k])

            psi_new = psi_aug[1:-1,:]
            for k in xrange(self._ndim_take):
                if self._isperiodic[k]:
                    psi_new[:,k] -= np.floor(psi_new[:,k]/self._boxsize[k])*self._boxsize[k]
                    sort_indx = np.argsort(psi_new[:,k])
                    psi_new = psi_new[sort_indx,:]


            self.centers[si] = psi_new.copy()

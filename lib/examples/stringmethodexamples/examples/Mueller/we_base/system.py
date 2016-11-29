from __future__ import division, print_function
__metaclass__ = type
import time
import os
import numpy as np

import west
from west.propagators import WESTPropagator
from west import Segment, WESTSystem
from westpa.binning import VoronoiBinMapper

import cIntegratorSimple
import ForceFields
from utils import dfunc

import logging
log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

pcoord_dtype = np.float32


def genrandint():
    """Generates a random integer between 0 and (2^32)-1"""
    x = 0
    for i in range(4):
        x = (x << 8) + ord(os.urandom(1))

    return x


class SimpleLangevinPropagator(WESTPropagator):

    def __init__(self, rc=None):
        super(SimpleLangevinPropagator, self).__init__(rc)

        rc = self.rc.config['west','simplelangevin']
        self.ndim = rc.get('ndim', 2)
        self.nsteps = rc.get('blocks_per_iteration', 2)
        self.nsubsteps = rc.get('steps_per_block')
        self.beta = rc.get('beta')

        ff = ForceFields.MuellerForce()
        MASS = 1.0
        XI = 1.5
        BETA = self.beta
        NDIMS = 2
        DT = 1e-4
        ISPERIODIC = np.array([0, 0], dtype=np.int)
        BOXSIZE = np.array([1.0E8, 1.0E8], dtype=pcoord_dtype)

        self.integrator = cIntegratorSimple.SimpleIntegrator(ff, MASS, XI, BETA, DT, NDIMS, ISPERIODIC, BOXSIZE, genrandint())

    def get_pcoord(self, state):
        pcoord = None
        if state.label == 'initA':
            pcoord = [-1.0, 1.0]
        if state.label == 'initB':
            pcoord = [0.5, 0.5]

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


class System(WESTSystem):

    def initialize(self):

        rc = self.rc.config['west', 'system']

        self.pcoord_ndim = 2
        self.pcoord_len = 2
        self.pcoord_dtype = pcoord_dtype
        self.target_count = rc.get('target_count')
        self.nbins = rc.get('nbins')

        slen = self.nbins
        #y[:] = 1.0
        centers = np.zeros((self.nbins, self.pcoord_ndim), dtype=self.pcoord_dtype)

        #endpoint1 = np.array([-0.55918841,  1.44078036])
        #endpoint2 = np.array([ 0.61810024,  0.03152928])
        endpoint1 = np.array([-1.5, 1.0])
        endpoint2 = np.array([1.0, 1.0])
        centers[:,0] = np.linspace(endpoint1[0], endpoint2[0], self.nbins)
        centers[:,1] = np.linspace(endpoint1[1], endpoint2[1], self.nbins)

        self.bin_mapper = VoronoiBinMapper(dfunc, centers)
        self.bin_target_counts = np.zeros((self.bin_mapper.nbins,), dtype=np.int_)
        self.bin_target_counts[...] = self.target_count

        slen = self.nbins 
        self.sm_params = {'slen': [slen],
                          'kappa': 0.1,
                          'dtau': 0.1,
                          'fixed_ends': False,
                          'sciflag': True,
                          'fourierflag': False}

from __future__ import division, print_function; __metaclass__ = type

import time
import os
import numpy as np

import west
from west.propagators import WESTPropagator
from west import Segment, WESTSystem
from westpa.binning import VoronoiBinMapper

from ElasticNetwork import ElasticNetwork
import str_utils
import pyqcprot as qcp

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


def average_position(self, n_iter):
    """Get average position of replicas in each bin as of n_iter for the
    the user selected update interval
    """

    nbins = self.system.bin_mapper.nbins
    ndim = self.system.pcoord_ndim

    natoms = ndim // 6
    cpos = natoms * 3

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
                bpc = pcoords[ii,:cpos]
                nmemb = bpc.shape[0]

                # (1) Align all structures to confA
                xref = self.strings.centers[0,:cpos].reshape((natoms, 3))
                xref -= np.mean(xref, axis=0)
                xref = xref.T.astype(np.float64)

                for k in xrange(nmemb):
                    xi = bpc[k,:].reshape((natoms, 3))

                    R = np.zeros((9,), dtype=np.float64)

                    qcp.CalcRMSDRotationalMatrix(xref,xi.T.astype(np.float64), natoms, R, None, center_ref=0, center_conf=1, copy=0)
                    R = np.matrix(R.reshape(3,3))

                    xi[:] = xi * R
                    avg_pos[indx,:cpos] += xi.ravel() * weights[ii[k]]

            sum_bin_weight += np.bincount(bin_indices.astype(np.int), weights=weights, minlength=nbins)

        # Some bins might have zero samples so exclude to avoid divide by zero
        occ_ind = np.nonzero(sum_bin_weight)
        avg_pos[occ_ind] /= sum_bin_weight[occ_ind][:,np.newaxis]

        return avg_pos, sum_bin_weight


class ElasticNetworkPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(ElasticNetworkPropagator, self).__init__(rc)

        rc = self.rc.config['west', 'elasticnetwork']

        self.model = {}
        self.model['mass'] = rc.get('mass')
        self.model['gamma'] = rc.get('gamma')
        self.model['temp'] = rc.get('temp')
        self.model['dt'] = rc.get('dt')
        self.model['sigma'] = rc.get('sigma')
        self.model['eps'] = rc.get('eps')
        self.model['betamix'] = rc.get('betamix')

        self.model.update(np.load(rc.get('ff_data')))
        self.model['seed'] = genrandint()

        self.init_pos = {}
        self.init_pos['coordsA'] = self.model['coordsA']
        self.init_pos['coordsB'] = self.model['coordsB']

        del self.model['coordsA']
        del self.model['coordsB']

        assert self.init_pos['coordsA'].shape == self.init_pos['coordsB'].shape

        self.ndim = 2 * self.init_pos['coordsA'].size
        self.nsteps = rc.get('blocks_per_iteration', 2)
        self.nsubsteps = rc.get('steps_per_block', 100)

    def get_pcoord(self, state):

        # Assign velocities drawn from Maxwell-Boltzmann distribution
        temp = self.model['temp']
        mass = self.model['mass']
        sigma = np.sqrt(temp*0.001987191/mass)

        if state.label == 'initA':
            init_coords = self.init_pos['coordsA']
        elif state.label == 'initB':
            init_coords = self.init_pos['coordsB']

        init_vel = sigma*np.random.normal(size=init_coords.shape)

        state.pcoord = np.vstack((init_coords,init_vel)).ravel()

    def propagate(self, segments):
        # Instantiate the integrator here since cython extension classes do not
        # automatically implement the pickling protocol
        self.model['seed'] = genrandint()
        integrator = ElasticNetwork(**self.model)

        for segment in segments:
            starttime = time.time()

            new_pcoords = np.empty((self.nsteps,self.ndim), dtype=pcoord_dtype)
            new_pcoords[0,:] = segment.pcoord[0,:]

            cpos = self.ndim // 2
            x = new_pcoords[0,:cpos].copy().reshape((-1,3)).astype(np.float64)
            v = new_pcoords[0,cpos:].copy().reshape((-1,3)).astype(np.float64)

            for istep in xrange(1, self.nsteps):
                integrator.step(x, v, self.nsubsteps)
                new_pcoords[istep,:] = np.hstack((x.ravel(), v.ravel()))

            segment.pcoord = new_pcoords[...].astype(pcoord_dtype)
            segment.status = Segment.SEG_STATUS_COMPLETE

            segment.walltime = time.time() - starttime

        del integrator

        return segments


def dfunc(p, centers):

    #cpos = 372; natoms = 124
    #indr = np.arange(9, 122 * 3)
    natoms_r = 119

    xi = np.asarray(p[9:122*3]).reshape((natoms_r, 3))
    xi -= np.mean(xi, axis=0)
    xi = xi.T.astype(np.float64)

    d = np.empty((centers.shape[0],))

    for ci, ck in enumerate(centers):
        cen = np.asarray(ck[9:122*3]).reshape((natoms_r, 3))

        d[ci] = qcp.CalcRMSDRotationalMatrix(xi, cen.T.astype(np.float64), natoms_r, None, None, center_ref=0, center_conf=1, copy=0)

    return d.astype(pcoord_dtype)


class System(WESTSystem):

    def initialize(self):

        rc = self.rc.config

        # pcoord parameters
        ff_data = np.load(rc.require(['west', 'elasticnetwork', 'ff_data']))
        self.pcoord_ndim = 2 * ff_data['coordsA'].size
        self.pcoord_len = rc.require(['west', 'elasticnetwork', 'blocks_per_iteration'])
        self.pcoord_dtype = pcoord_dtype

        self.target_count = rc.require(['west', 'system', 'target_count'])
        self.nbins = rc.require(['west', 'system', 'nbins'])

        # string method parameters
        self.cpos = self.pcoord_ndim // 2
        self.natoms = ff_data['coordsA'].shape[0]
        self.sm_params = {'slen': [self.nbins],
                          'dtau': 0.1,
                          'kappa': 0.2,
                          'fixed_ends': rc.require(['west', 'plugins', 1, 'fixed_ends'], type_=bool),
                          'slabels': range(self.cpos, self.pcoord_ndim)}

        self.init_pos = {}
        self.init_pos['coordsA'] = ff_data['coordsA']
        self.init_pos['coordsB'] = ff_data['coordsB']

        try:
            self.init_str_ends = np.load(rc.get(['west', 'plugins', 1, 'init_string_ends'], None))
        except:
            self.init_str_ends = np.vstack((self.init_pos['coordsA'].ravel(), self.init_pos['coordsB'].ravel()))

        confA = self.init_str_ends[0]
        confB = self.init_str_ends[1]

        # Check and reshape coordinates
        assert confA.shape == confB.shape
        if len(confA.shape) == 1:
            confA = confA.reshape((self.cpos//3,3))

        if len(confB.shape) == 1:
            confB = confB.reshape((self.cpos//3,3))

        assert confA.shape == confB.shape == (self.cpos//3,3)

        # Build initial string
        centers = str_utils.interpolate_conformations(confA, confB, self.nbins)

        # Add dummy progress coordinates for velocities
        centers = np.hstack((centers,np.zeros_like(centers))).astype(pcoord_dtype)

        self.bin_mapper = VoronoiBinMapper(dfunc, centers)
        self.bin_target_counts = np.zeros((self.bin_mapper.nbins,), dtype=np.int_)
        self.bin_target_counts[...] = self.target_count

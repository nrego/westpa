# Copyright (C) 2013 Joshua L. Adelman
#
# This file is part of WESTPA.
#
# WESTPA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WESTPA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WESTPA.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division; __metaclass__ = type
import logging
log = logging.getLogger(__name__)

import numpy as np
import types

import westpa, west
from westpa import extloader
from westpa.yamlcfg import check_bool, ConfigItemMissing
from westext.stringmethod import WESTStringMethod, DefaultStringMethod
from westpa.binning import VoronoiBinMapper

from pickle import PickleError

tensor_dtype = np.float64


# Get the average tensor for a single iteration, n_iter
#   This calls the user-supplied tensor_func on each segment of iteration n_iter,
#   sums them up (weighted by segment weight), and returns the total estimated avg
#   tensor, along with the iter number for future processing
#
# n_iter - iter for which we will calculate avg tensor
# pcoords - (n_seg, pcoord_ndim) Pcoords for each segment for a *single* time point during n_iter (usually the last)
# coords - (n_seg, coord_ndim) coords for each segment for a *single* time point during n_iter
# weights - (n_seg,) weights for each segment; sum to unity
# tensor_func - function that returns metric tensor from a single segment's coords and pcoords
#    called as:  tensor_func(n_seg, pcoord, coord)
#
def _avg_tensor_func(n_iter, pcoords, coords, weights, tensor_func):
    ndim = pcoords.shape[1]
    my_tensor = np.zeros((ndim, ndim), dtype=tensor_dtype)
    for n_seg in xrange(pcoords.shape[0]):
        pcoord = pcoords[n_seg]
        coord = coords[n_seg]
        curr_tensor = tensor_func(n_seg, pcoord, coord)
        my_tensor += curr_tensor * weights[n_seg]

    return my_tensor, n_iter


class StringDriver(object):
    def __init__(self, sim_manager, plugin_config):
        super(StringDriver, self).__init__()

        if not sim_manager.work_manager.is_master:
                return

        self.work_manager = sim_manager.work_manager
        self.sim_manager = sim_manager
        self.data_manager = sim_manager.data_manager
        self.system = sim_manager.system

        # Parameters from config file
        self.windowsize = plugin_config.get('windowsize', 10)
        self.tensor_windowsize = plugin_config.get('tensor_windowsize', self.windowsize)
        self.update_interval = plugin_config.get('update_interval', 10)
        self.initial_update = plugin_config.get('initial_update', 20)
        self.priority = plugin_config.get('priority', 0)

        self.write_avg_pos = check_bool(plugin_config.get('write_avgpos', True))
        self.do_update = check_bool(plugin_config.get('do_update', True))
        self.init_from_data = check_bool(plugin_config.get('init_from_data', True))

        self.update_metric_tensor = check_bool(plugin_config.get('do_tensor_update', False))

        # Try to load a supplied function to calculate metric tensor, if provided
        #    Otherwise, set 'tensor_func' to None, and take care of it later
        #    NOTE: if no tensor_func provided, will automatically set the metric tensor to None (eg. use default dfunc)
        try:
            methodname = plugin_config['tensor_function']
            self.tensor_func = extloader.get_object(methodname)
        except:
            self.tensor_func = None

        self.dfunc = self.get_dfunc_method(plugin_config)

        # Load method to calculate average position in a bin
        # If the method is defined in an external module, correctly bind it
        ap = self.get_avgpos_method(plugin_config)
        if hasattr(ap, 'im_class'):
            self.get_avgpos = ap
        else:
            self.get_avgpos = types.MethodType(ap, self)

        # Get initial set of string centers
        centers = self.get_initial_centers()
        ndim = centers.shape[1]
        # Grab inverse metric tensor from h5 file or system, if provided - otherwise set to None
        self.inv_tensor = self.get_initial_tensor()

        try:
            sm_params = self.system.sm_params
        except AttributeError as e:
            log.error('String Driver Error: system does not define sm_params.' \
                        'This is required and should be added to the system definition; {}'.format(e))
            raise

        # Initialize the string
        str_method = self.get_string_method(plugin_config)

        try:
            self.strings = str_method(centers, **sm_params)
        except (TypeError, AssertionError) as e:
            log.error('String Driver Error: Failed during initialization of string method: {}'.format(e))
            raise

        # Update the BinMapper
        self.update_bin_mapper()

        # Register callback
        sim_manager.register_callback(sim_manager.prepare_new_iteration, self.prepare_new_iteration, self.priority)

        westpa.rc.pstatus('-westext.stringmethod -----------------\n')
        westpa.rc.pstatus('windowsize: {}\n'.format(self.windowsize))
        westpa.rc.pstatus('update interval: {}\n'.format(self.update_interval))
        westpa.rc.pstatus('initial update: {}\n'.format(self.initial_update))
        westpa.rc.pstatus('priority: {}\n'.format(self.priority))
        westpa.rc.pstatus('write average positions: {}\n'.format(self.write_avg_pos))
        westpa.rc.pstatus('do update: {}\n'.format(self.do_update))
        westpa.rc.pstatus('initialize from WE data: {}\n'.format(self.init_from_data))
        westpa.rc.pstatus('----------------------------------------\n')
        westpa.rc.pflush()

    def dfunc(self):
        raise NotImplementedError

    def get_avgpos(self, n_iter):
        raise NotImplementedError

    def get_dfunc_method(self, plugin_config):
        try:
            methodname = plugin_config['dfunc_method']
        except KeyError:
            raise ConfigItemMissing('dfunc_method')

        dfunc_method = extloader.get_object(methodname)

        log.info('loaded stringmethod dfunc method {!r}'.format(dfunc_method))

        return dfunc_method

    def get_avgpos_method(self, plugin_config):
        try:
            methodname = plugin_config['avgpos_method']
        except KeyError:
            raise ConfigItemMissing('avgpos_method')

        if methodname.lower() == 'cartesian':
            avgpos_method = self.avgpos_cartesian
        else:
            avgpos_method = extloader.get_object(methodname)

        log.info('loaded stringmethod avgpos method {!r}'.format(avgpos_method))

        return avgpos_method

    def get_string_method(self, plugin_config):
        try:
            methodname = plugin_config['string_method']
        except KeyError:
            raise ConfigItemMissing('string_method')

        if methodname.lower() == 'default':
            str_method = DefaultStringMethod
        else:
            str_method = extloader.get_object(methodname)

        assert issubclass(str_method, WESTStringMethod)
        log.debug('loaded stringmethod string method {!r}'.format(str_method))

        return str_method

    def get_initial_centers(self):
        self.data_manager.open_backing()

        with self.data_manager.lock:


            # First attempt to initialize string from data rather than system
            centers = None
            if self.init_from_data:
                log.info('Attempting to initialize stringmethod from data at iteration {}'.format(self.data_manager.current_iteration))

                # First try to grab from current iteration
                try:
                    n_iter = self.data_manager.current_iteration
                    iter_group = self.data_manager.get_iter_group(n_iter)
                    binhash = iter_group.attrs['binhash']
                    bin_mapper = self.data_manager.get_bin_mapper(binhash)

                    centers = bin_mapper.centers

                except:
                    log.info('...Attempting to initialize stringmethod from data at iteration {}'.format(max(self.data_manager.current_iteration-1, 1)))

                    try:
                        n_iter = max(self.data_manager.current_iteration-1, 1)
                        iter_group = self.data_manager.get_iter_group(n_iter)
                        binhash = iter_group.attrs['binhash']
                        bin_mapper = self.data_manager.get_bin_mapper(binhash)

                        centers = bin_mapper.centers
                    except:
                        log.warning('Initializing string centers from data failed; Using definition in system instead.')
                        centers = self.system.bin_mapper.centers
            else:
                log.info('Initializing string centers from system definition')
                centers = self.system.bin_mapper.centers

        self.data_manager.close_backing()

        return centers

    def get_initial_tensor(self):
        '''Get the initial (inverse) metric tensor, if provided; otherwise send back None'''
        self.data_manager.open_backing()

        with self.data_manager.lock:
            n_iter = max(self.data_manager.current_iteration - 1, 1)
            iter_group = self.data_manager.get_iter_group(n_iter)

            # Try to get stored tensor, if present
            inv_tensor = None
            if self.init_from_data:
                log.info('Attempting to initialize get metric tensor from data')

                try:
                    binhash = iter_group.attrs['binhash']
                    bin_mapper = self.data_manager.get_bin_mapper(binhash)

                    inv_tensor = bin_mapper.dfkwargs['inv_tensor']

                except:
                    log.warning('Initializing inverse tensor from data failed; Using definition in system instead.')
                    inv_tensor = getattr(self.system, 'inv_tensor', None)
            else:
                log.info('Initializing inverse tensor from system definition')

                inv_tensor = getattr(self.system, 'inv_tensor', None)

        self.data_manager.close_backing()

        return inv_tensor

    def update_bin_mapper(self):
        '''Update the bin_mapper using the current string'''

        # This does no harm if already open
        self.data_manager.open_backing()

        westpa.rc.pstatus('westext.stringmethod: Updating bin mapper\n')
        westpa.rc.pflush()

        try:
            dfargs = getattr(self.system, 'dfargs', None)
            dfkwargs = getattr(self.system, 'dfkwargs', None)
            if dfkwargs:
                dfkwargs['inv_tensor'] = self.inv_tensor
            log.debug('dfkwargs: {}'.format(dfkwargs))
            
            self.system.bin_mapper = VoronoiBinMapper(self.dfunc, self.strings.centers, 
                                                      dfargs=dfargs, 
                                                      dfkwargs=dfkwargs)
        except (ValueError, TypeError) as e:
            log.error('StringDriver Error: Failed updating the bin mapper: {}'.format(e))
            raise

        # Try to save the bin mapper immediately
        try:
            n_iter = self.data_manager.current_iteration
            iter_group = self.data_manager.get_iter_group(n_iter)

            try:
                pickled, hashed = self.system.bin_mapper.pickle_and_hash()
            except PickleError:
                pickled = hashed = ''

            if hashed and pickled:
                # save_bin_mapper checks that mapper has not already been added to table
                self.data_manager.save_bin_mapper(hashed, pickled)
                iter_group.attrs['binhash'] = hashed
            else:
                iter_group.attrs['binhash'] = '' 

        except:
            log.error('String Driver Error: Failed storing the updated bin mapper')
            raise

    def avgpos_cartesian(self, n_iter):
        '''Get average position of replicas in each bin as of n_iter for the
        the user selected update interval'''

        nbins = self.system.bin_mapper.nbins
        ndim = self.system.pcoord_ndim

        avg_pos = np.zeros((nbins, ndim), dtype=self.system.pcoord_dtype)
        sum_bin_weight = np.zeros((nbins,), dtype=self.system.pcoord_dtype)

        start_iter = max(n_iter - min(self.windowsize, n_iter), 2)
        stop_iter = n_iter + 1

        for n in xrange(start_iter, stop_iter):
            with self.data_manager.lock:
                iter_group = self.data_manager.get_iter_group(n)
                seg_index = iter_group['seg_index'][...]

                pcoords = iter_group['pcoord'][:,-1,:]  # Only read final point
                bin_indices = self.system.bin_mapper.assign(pcoords)
                weights = seg_index['weight']

                pcoord_w = pcoords * weights[:,np.newaxis]
                uniq_indices = np.unique(bin_indices)

                for indx in uniq_indices:
                    avg_pos[indx,:] += pcoord_w[bin_indices == indx].sum(axis=0)

                sum_bin_weight += np.bincount(bin_indices.astype(np.int), weights=weights, minlength=nbins)

        # Some bins might have zero samples so exclude to avoid divide by zero
        occ_ind = np.nonzero(sum_bin_weight)
        avg_pos[occ_ind] /= sum_bin_weight[occ_ind][:,np.newaxis]

        return avg_pos, sum_bin_weight

    def avg_metric_tensor(self, n_iter):
        '''Get average metric tensor up to iteration n_iter'''
        nbins = self.system.bin_mapper.nbins
        ndim = self.system.pcoord_ndim

        start_iter = max(n_iter - min(self.tensor_windowsize, n_iter), 2)
        stop_iter = n_iter + 1
        iter_count = stop_iter - start_iter

        # Keep a running total of metric tensor
        metric_tensor = np.zeros((iter_count, ndim, ndim), dtype=tensor_dtype)

        # These two functions are shannanagins necessary for interfacing with work manager...

        # Generator for work_manager
        #  Go through each iteration, send out _avg_tensor_func (which calls user-supplied tensor func - yikes)
        def task_gen():
            for n in xrange(start_iter, stop_iter):
                log.info("for iter: {}".format(n))
                with self.data_manager.lock:
                    iter_group = self.data_manager.get_iter_group(n)
                    seg_index = iter_group['seg_index'][...]

                    pcoords = iter_group['pcoord'][:,-1,:]  # Only read final point
                    weights = seg_index['weight']
                    try:
                        coords = iter_group['auxdata/coord'][...]
                    except KeyError:
                        continue

                args = ()
                kwargs = dict(n_iter=n, pcoords=pcoords, coords=coords, weights=weights,
                              tensor_func=self.tensor_func)

                yield (_avg_tensor_func, args, kwargs)

        # Send out avg tensor calculation to each worker (one for each iteration)
        #    Collect results - a tensor for iteration n_iter - and put it into proper spot in total array
        for future in self.work_manager.submit_as_completed(task_gen()):
            n_iter_tensor, n_iter = future.get_result(discard=True)
            metric_tensor[n_iter-start_iter, ...] = n_iter_tensor


        # Some bins might have zero samples so exclude to avoid divide by zero
        return np.sum(metric_tensor, axis=0) / iter_count

    def prepare_new_iteration(self):

        n_iter = self.sim_manager.n_iter

        with self.data_manager.lock:
            iter_group = self.data_manager.get_iter_group(n_iter)

            try:
                del iter_group['stringmethod']
            except KeyError:
                pass

            sm_global_group = self.data_manager.we_h5file.require_group('stringmethod')
            last_update = long(sm_global_group.attrs.get('last_update', 0))

        # Update metric tensor (if desired) everytime we update the string
        if n_iter - last_update < self.update_interval or n_iter < self.initial_update or not self.do_update:
            log.debug('Not updating string this iteration')
            return
        else:
            log.debug('Updating string - n_iter: {}'.format(n_iter))

        westpa.rc.pstatus('-westext.stringmethod -----------------\n')
        if self.update_metric_tensor:
            westpa.rc.pstatus('westext.stringmethod: Updating metric tensor\n')
            westpa.rc.pflush()
            new_tensor = self.avg_metric_tensor(n_iter)

            self.inv_tensor = np.linalg.inv(new_tensor)

        westpa.rc.pstatus('westext.stringmethod: Calculating average position in string images\n')
        westpa.rc.pflush()

        avg_pos, sum_bin_weight = self.get_avgpos(n_iter)

        westpa.rc.pstatus('westext.stringmethod: Updating string\n')
        westpa.rc.pflush()

        prev_centers = self.strings.centers.copy()
        self.strings.update_string_centers(avg_pos, sum_bin_weight)
        rmsds = []
        L = self.strings.length
        for sid, si in enumerate(self.strings._strindx):
            rmsds.append(np.sqrt(np.sum((self.strings.centers[si] - prev_centers[si])**2) / (L[sid]*self.strings._slen[sid])))

        westpa.rc.pstatus('westext.stringmethod: RMSD/len w.r.t previous string: {}\n'.format(rmsds))

        westpa.rc.pstatus('westext.stringmethod: String lengths: {}\n'.format(self.strings.length))
        westpa.rc.pflush()

        # Update the bin definitions
        self.update_bin_mapper()

        sm_global_group.attrs['last_update'] = n_iter

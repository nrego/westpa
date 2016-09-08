# Copyright (C) 2013 Matthew C. Zwier and Lillian T. Chong
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

from __future__ import division, print_function

import logging, argparse, traceback
log = logging.getLogger('w_run_debug')

import westpa
import work_managers
from work_managers import make_work_manager

from IPython import embed
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('w_run_debug', 'start/continue a WEST simulation')
westpa.rc.add_args(parser)
parser.add_argument('--oneseg', dest='only_one_segment', action='store_true',
                    help='only propagate one segment (useful for debugging propagators)')

args = parser.parse_args()
westpa.rc.process_args(args)
# Force WM to use 'serial' work manager, regardless of environment or cmd line options
#setattr(args, 'work_manager', 'serial')
work_managers.environment.process_wm_args(args)
work_manager = westpa.rc.work_manager = make_work_manager()

# Load the sim manager and other drivers
sim_manager = westpa.rc.get_sim_manager()
system = westpa.rc.get_system_driver()
data_manager = westpa.rc.get_data_manager()
we_driver = westpa.rc.get_we_driver()
propagator = westpa.rc.get_propagator()

propagator.system = system
data_manager.system = system
we_driver.system = system

sim_manager.data_manager = data_manager
sim_manager.system = system
sim_manager.propagator = propagator
sim_manager.we_driver = we_driver

with work_manager:

    work_manager.install_sigint_handler()
    sim_manager.load_plugins()

    # a list of plugin instances (which callback 'prepare_new_iteration', 
    #  since WESTPA doesn't provide access to instances for some weird reason
    plugins = [fn.im_self for (_,_,fn) in sim_manager._callback_table[sim_manager.prepare_new_iteration]]
    
    log.debug('preparing run')
    sim_manager.prepare_run()
    sim_manager.n_iter = sim_manager.data_manager.current_iteration

    self = sim_manager
    embed()




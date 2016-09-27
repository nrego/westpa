import numpy as np
import h5py
import argparse
import sys
import os

import west

print '-----------------------'
print os.path.basename(__file__)
print '-----------------------'
env = os.environ
for k in env:
    if 'WEST' in k:
        print k, env[k]


parser = argparse.ArgumentParser('get_strings', description='''\
        Retrieve number of replicas from west.h5 file and write them to new file
        ''')

west.rc.add_args(parser)
parser.add_argument('-o', dest='h5out', help='name of output file')

args = parser.parse_args()
west.rc.process_args(args)

data_manager = west.rc.get_data_manager()
data_manager.open_backing(mode='r')

h5out = h5py.File(args.h5out, 'w')

# Get number of replicas per iteration
nrep = data_manager.we_h5file['summary']['n_particles'][:]
n_iters = nrep.size
iter_ds = h5out.require_dataset('n_particles', shape=(n_iters,), dtype=np.int64)
iter_ds[:] = nrep

h5out.close()
data_manager.close_backing()

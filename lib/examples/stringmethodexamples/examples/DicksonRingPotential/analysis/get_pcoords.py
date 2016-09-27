import numpy as np
import h5py
import argparse
import sys
import os

import west

step = 100
pcoord_dtype = np.float32

print '-----------------------'
print os.path.basename(__file__)
print '-----------------------'
env = os.environ
for k in env:
    if 'WEST' in k:
        print k, env[k]


parser = argparse.ArgumentParser('get_pcoords', description='''\
        Retrieve pcoords from west.h5 file and write them to new file
        ''')

west.rc.add_args(parser)
parser.add_argument('-o', dest='h5out', help='name of output file')

args = parser.parse_args()
west.rc.process_args(args)

data_manager = west.rc.get_data_manager()
data_manager.open_backing(mode='r')

h5out = h5py.File(args.h5out, 'w')

n_iters = data_manager.current_iteration - 1
iter_prec = data_manager.iter_prec

data_group = h5out.require_group('/iterations')

# Get first iteration
iiter = 2
iter_group = data_manager.get_iter_group(iiter)
crd = iter_group['pcoord'][:,-1,:]
iter_name ='iter_{:0{prec}d}'.format(long(iiter), prec=iter_prec)
pcoord_ds = data_group.require_dataset(iter_name, shape=crd.shape, dtype=pcoord_dtype)
pcoord_ds[...] = crd

for iiter in xrange(step, n_iters, step):
    h5out.flush()

    try:
        iter_group = data_manager.get_iter_group(iiter)
        crd = iter_group['pcoord'][:,-1,:]
    except:
        print 'Error processing iteration: {}'.format(iiter)
        print sys.exc_info()
        break

    iter_name = 'iter_{:0{prec}d}'.format(long(iiter), prec=iter_prec)
    pcoord_ds = data_group.require_dataset(iter_name, shape=crd.shape, dtype=pcoord_dtype)
    pcoord_ds[...] = crd

h5out.close()
data_manager.close_backing()

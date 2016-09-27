import numpy as np
import h5py
import argparse
import sys
import os

import west
from utils import calc_wemd_rate

print '-----------------------'
print os.path.basename(__file__)
print '-----------------------'
env = os.environ
for k in env:
    if 'WEST' in k:
        print k, env[k]


parser = argparse.ArgumentParser('calculate_rate', description='''\
        Calculate distribution statistics
        ''')

west.rc.add_args(parser)
parser.add_argument('-o', dest='h5out', help='name of output file')

args = parser.parse_args()
west.rc.process_args(args)

data_manager = west.rc.get_data_manager()
data_manager.open_backing(mode='r')

h5out = h5py.File(args.h5out, 'a')

n_iters = data_manager.current_iteration - 1
iter_prec = data_manager.iter_prec

if 'data' in h5out:
    data_ds = h5out['data']
    dshape = data_ds.shape
    if dshape[0] < n_iters:
        data_ds.resize((n_iters - 2, 4))

    start_iter = h5out.attrs['last_completed_iter']
else:
    print n_iters
    data_ds = h5out.require_dataset('data', shape=(n_iters - 2, 4), dtype=np.float64, maxshape=(None, 4))
    start_iter = 2
    h5out.attrs['last_completed_iter'] = 2

for iiter in xrange(start_iter, n_iters):
    if iiter % 1000 == 0:
        print 'Processing {} of {}'.format(iiter, n_iters - 1)
        h5out.flush()

    try:
        iter_group = data_manager.get_iter_group(iiter)
        weight = iter_group['seg_index']['weight']
        crd = iter_group['pcoord'][:]

        assert weight.shape[0] == crd.shape[0]
        flux_to_A, flux_to_B, wA, wB = calc_wemd_rate(crd, weight)
        data_ds[iiter-2,:] = np.array([flux_to_A, flux_to_B, wA, wB])
        h5out.attrs['last_completed_iter'] = iiter

    except:
        print 'Error in processing iteration: {}'.format(iiter)
        print sys.exc_info()
        break

h5out.close()
data_manager.close_backing()

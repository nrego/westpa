import numpy as np
import h5py
import argparse
import sys
import os

import west
import pyqcprot as qcp 

def dfunc(p,centers):

    indr = np.arange(9,122*3)
    natoms_r = 119

    xi = p[indr].reshape((natoms_r, 3))
    xi -= np.mean(xi,axis=0)
    xi = xi.T.astype(np.float64)

    d = np.empty((centers.shape[0],))

    for ci,ck in enumerate(centers):
        cen = ck[indr].reshape((natoms_r,3))

        d[ci] = qcp.CalcRMSDRotationalMatrix(xi,cen.T.astype(np.float64),natoms_r,None,None,center_ref=0,center_conf=1,copy=0)

    return d


# h5py storage types
vstr_dtype = h5py.new_vlen(str)
idtype = np.dtype([('iter_name', vstr_dtype), ('string_index', np.int32)])

print '-----------------------'
print os.path.basename(__file__)
print '-----------------------'
env = os.environ
for k in env:
    if 'WEST' in k:
        print k, env[k]


parser = argparse.ArgumentParser('get_bin_populations_all', description='''\
        Retrieve strings from west.h5 file and write them to new file
        ''')

west.rc.add_args(parser)
parser.add_argument('-o', dest='h5out', help='name of output file')

args = parser.parse_args()
west.rc.process_args(args)

data_manager = west.rc.get_data_manager()
data_manager.open_backing(mode='r')

last_update = data_manager.we_h5file['/stringmethod'].attrs['last_update']

print('last_update: {}'.format(last_update))

# Get each string stored in the unique bin_mapper objects
string_hashes = data_manager.we_h5file['bin_topologies']['index'][:]['hash']
n_strings = string_hashes.size

binhash = string_hashes[0]

bin_mapper = data_manager.get_bin_mapper(binhash)
n_centers = bin_mapper.nbins
nbins = n_centers
n_dims = bin_mapper.ndim
print('n_centers: {}, n_strings: {}, n_dims: {}'.format(n_centers, n_strings, n_dims))

shash = string_hashes[-1]
bin_mapper = data_manager.get_bin_mapper(shash)
centers = bin_mapper.centers

# Set up output file
h5out = h5py.File(args.h5out, 'a')
n_iters = data_manager.current_iteration - 1

if 'data' in h5out:
    data_ds = h5out['data']
    dshape = data_ds.shape
    if dshape[0] < n_iters:
        data_ds.resize((n_iters - 2, nbins))

    start_iter = h5out.attrs['last_completed_iter']
    if shash != h5out.attrs['shash']:
        sys.exit('Bin hash does not match the bash used to extract previous data')
else:
    data_ds = h5out.require_dataset('data', (n_iters - 2, nbins), np.float64, exact=False, maxshape=(None, nbins))
    start_iter = 2
    h5out.attrs['last_completed_iter'] = 2
    h5out.attrs['shash'] = shash

print('Beginning from iteration: {}'.format(start_iter))

for iiter in xrange(start_iter, n_iters):
    if iiter % 100 == 0:
        print('Processing {} of {}'.format(iiter, n_iters -1))
        h5out.flush()

    try:
        iter_group = data_manager.get_iter_group(iiter)
        weight = iter_group['seg_index']['weight']
        crd = iter_group['pcoord'][:,-1,:]

        assert weight.shape[0] == crd.shape[0]

        bin_indices = bin_mapper.assign(crd)
        data_ds[iiter-2,:] = np.bincount(bin_indices, weights=weight, minlength=nbins)
        h5out.attrs['last_completed_iter'] = iiter

    except:
        print 'Error in processing iteration: {}'.format(iiter)
        print sys.exc_info()
        break


h5out.close()
data_manager.close_backing()

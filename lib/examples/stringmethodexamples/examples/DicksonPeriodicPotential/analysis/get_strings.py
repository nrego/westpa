import numpy as np
import h5py
import argparse
import sys
import os

import west

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


parser = argparse.ArgumentParser('get_strings', description='''\
        Retrieve strings from west.h5 file and write them to new file
        ''')

west.rc.add_args(parser)
parser.add_argument('-o', dest='h5out', help='name of output file')

args = parser.parse_args()
west.rc.process_args(args)

data_manager = west.rc.get_data_manager()
data_manager.open_backing(mode='r')

h5out = h5py.File(args.h5out, 'w')

last_update = data_manager.we_h5file['/stringmethod'].attrs['last_update']

print('last_update: {}'.format(last_update))

# Get each string stored in the unique bin_mapper objects
string_hashes = data_manager.we_h5file['bin_topologies']['index'][:]['hash']
n_strings = string_hashes.size

binhash = string_hashes[0]

bin_mapper = data_manager.get_bin_mapper(binhash)
n_centers = bin_mapper.nbins
n_dims = bin_mapper.ndim
print('n_centers: {}, n_strings: {}, n_dims: {}'.format(n_centers, n_strings, n_dims))

# Get all strings
string_ds = h5out.require_dataset('strings', shape=(n_strings, n_centers, n_dims), dtype=np.float32)

for si, shash in enumerate(string_hashes):
    bin_mapper = data_manager.get_bin_mapper(shash)
    centers = bin_mapper.centers
    string_ds[si,:,:] = centers

# Get per iteration string
iterations = data_manager.we_h5file['iterations'].keys()
n_iters = len(iterations)

iter_ds = h5out.require_dataset('iterations', shape=(n_iters,), dtype=idtype)

for iiter, iter_name in enumerate(iterations):
    if iiter % 100 == 0:
        print 'Processing {} of {}'.format(iiter, n_iters)
        h5out.flush()

    try:
        iter_group = data_manager.we_h5file['iterations'][iter_name]
        if 'binhash' in iter_group.attrs:
            binhash = iter_group.attrs['binhash']
            string_index = data_manager.find_bin_mapper(binhash)
        else:
            string_index = -1
    except:
        print 'Error in processing iteration {}'.format(iter_name)
        print sys.exc_info()
        break

    iter_ds[iiter] = (iter_name, string_index)

h5out.close()
data_manager.close_backing()

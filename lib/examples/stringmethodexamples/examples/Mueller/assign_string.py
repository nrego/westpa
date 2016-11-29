import numpy as np
import westpa

dm = westpa.rc.get_data_manager()

def construct(n_iter, iter_group):
    binhash = iter_group.attrs['binhash']

    mapper = dm.get_bin_mapper(binhash)

    pcoords = iter_group['pcoord'][...]
    ret_arr = np.zeros((pcoords.shape[0], pcoords.shape[1], 1), dtype=np.int)
    ret_arr[:,0,0] = mapper.assign(pcoords[:,0,:])
    ret_arr[:,1,0] = mapper.assign(pcoords[:,1,:])

    return ret_arr

import h5py
import numpy as np
import westpa
import cPickle as pickle

def dist(pt1, pt2):
    return np.sum((pt1-pt2)**2)

def mueller(x, y):
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]

    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    V1 = 0
    for j in range(4):
        V1 += AA[j] * np.exp(aa[j] * (x - XX[j])**2 + \
              bb[j] * (x - XX[j]) * (y - YY[j]) + cc[j] * (y - YY[j])**2)

    return V1

def calculate_length(x):
    dd = x - np.roll(x, 1, axis=0)
    dd[0,:] = 0.0
    return np.cumsum(np.sqrt((dd*dd).sum(axis=1)))

# Get the (free) energy as a function of string images
xx, yy = np.mgrid[-1.5:1.2:0.01, -0.2:2.0:0.01]

assert xx.shape == yy.shape
nx = xx.shape[0]
ny = xx.shape[1]

energy = mueller(xx, yy)
energy -= energy.min()

dm = westpa.rc.get_data_manager()
dm.open_backing()
hashes = dm.we_h5file['bin_topologies']['index']['hash']
mapper = dm.get_bin_mapper(hashes[0])
strings = np.zeros((len(hashes), mapper.centers.shape[0], mapper.centers.shape[1]))

for i, hashval in enumerate(hashes):
    mapper = dm.get_bin_mapper(hashval)

    strings[i] = mapper.centers

pickle.dump(strings, open('strings.pkl', 'w'))
import numpy as np
from scipy.spatial.distance import pdist,squareform
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.split(file_dir)[0]

pdbfiles = ['1DC7.pdb', '1DC8.pdb']
pdbfiles = [os.path.join(basedir, 'data', x) for x in pdbfiles]
kmax = 0.2
dcut = 11.5
eps = 0.5

coords = [[], []]

# Extract coordinates
for k, pdb in enumerate(pdbfiles):
    ca_pdb = pdb.split('.')[0] + '_ca.pdb'
    with open(pdb,'r') as f, open(ca_pdb,'w') as fout:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if ' CA ' in line:
                    fout.write(line)
                    x = [float(y) for y in line[28:56].split()]
                    coords[k].append(x)

# Calculate
print len(coords[0])
print len(coords[1])
assert len(coords[0]) == len(coords[1])

natoms = len(coords[0])
coords = [np.array(coords[0]), np.array(coords[1])]

dist = [[], []]
D = [[], []]
kAB = np.zeros((natoms, natoms))

# Calculate distance and contact matrix
for k in xrange(2):
    dist[k] = squareform(pdist(coords[k]))
    D[k] = np.zeros((natoms,natoms), dtype=np.int)
    ii = np.where(dist[k] < dcut)
    D[k][ii] = 1

dd = (dist[0] - dist[1])**2
di = np.diag_indices_from(dd)
dd[di] = 10000.0
kAB = eps*np.ones((natoms, natoms))/dd

kAB[kAB > kmax] = kmax

# Save arrays
np.savez(os.path.join(basedir, 'data', 'data.npz'),
         kAB=kAB, distA=dist[0], distB=dist[1],
         coordsA=coords[0], coordsB=coords[1], DA=D[0], DB=D[1])

# Write xyz files
for pdb,name,x in zip(pdbfiles,['coordsA','coordsB'],coords):
    with open(os.path.join(basedir, 'data', name + '.xyz'),'w') as f:
        f.write(str(natoms) + '\n')
        f.write('CA model of {}\n'.format(pdb))

        for xi in x:
            f.write('CA {:5.3f} {:5.3f} {:5.3f}\n'.format(xi[0], xi[1], xi[2]))

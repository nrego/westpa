import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
from glob import glob

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.
#mpl.rcParams['text.usetex'] = True

fsize = (3.375, 3.0)

beta = [1.0, 1.5, 2.0, 2.5]
nbins = 100
theta = np.linspace(-1.0,1.0,nbins)
lastN = 500

# Setup figure
fig = plt.figure(1, figsize=fsize, dpi=200)
ax = fig.add_subplot(111)


# Accumulate distributions from brute force
for b in beta:
    fnames = glob('../bruteforce_{beta}/rawcounts_{beta}_*.npy'.format(beta=b))
    H = np.zeros((nbins,))

    for f in fnames[:10]:
        H += np.load(f).sum(0)

    H /= np.sum(H)

    ax.semilogy(theta, H, ls='-', lw=0.5, c='k')

# Accumulate distributions for WE simulations
for b in beta:
    fnames = glob('../we_{beta}/analysis/*/distribution.h5'.format(beta=b))
    nsims = len(fnames)
    H = np.zeros((nsims,nbins))

    for fi,f in enumerate(fnames):
        h5file = h5py.File(f, 'r')
        h = h5file['data'][-lastN:,:].sum(0)
        h /= h.sum()
        H[fi,:] = h
        h5file.close()

    ax.semilogy(theta, H.mean(0), lw=0.5, ls='None', marker='o', ms=2, c='r', mfc='None')

ax.set_xlabel(r'${\theta}/{\pi}$')
ax.set_ylabel(r'$P({\theta})$')
plt.tight_layout()
plt.savefig('distribution.eps', dpi=600, format='eps', bbox_inches='tight')

plt.show()

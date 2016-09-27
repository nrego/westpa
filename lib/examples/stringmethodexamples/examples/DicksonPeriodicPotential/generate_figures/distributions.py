import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from glob import glob

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

legfont = fm.FontProperties(size=8)

sims = ['common', 'rare']
markers = ['o', 'v']
bins = np.linspace(0.0, 1.0, 100)
nsims = 10

fsize = (3.375, 3.0)


# Figure setup
fig = plt.figure(1,figsize=fsize)
ax = fig.add_subplot(111)

# Brute force
for sni,sname in enumerate(sims):
    H = np.zeros((nsims, bins.size))

    data_files = glob('../bruteforce_{}/rawcounts_*.npy'.format(sname))

    for dfi,df in enumerate(data_files[:10]):
        h = np.load(df)
        H[dfi,:] += h.sum(0)
        H[dfi,:] /= H[dfi,:].sum()

    H = H.mean(0)

    ax.semilogy(bins,H,'-',lw=0.5)

ax.set_xlabel('y')
ax.set_ylabel('Probability')

# Weighted Ensemble
lastN = 20000

for sni, sname in enumerate(sims):

    data_files = glob('../we_{}/analysis/*/distribution.h5'.format(sname))
    H = np.zeros((len(data_files),bins.size))
    
    for dfi,df in enumerate(data_files):
        f = h5py.File(df,'r')
        h = f['data'][:]
        H[dfi,:] = np.sum(h[-lastN:,:],axis=0)
        H[dfi,:] /= H[dfi,:].sum()
        f.close()

    H = H.mean(axis=0)
    ax.semilogy(bins,H,ls='None',marker=markers[sni],ms=2,mfc='None',mew=0.5)

ax.axis([0,1,1E-9,1])
#ax.legend(('CONV common','CONV rare', 'WE common', 'WE rare'),loc=1,frameon=False,prop=legfont)
fig.set_size_inches(fsize)
plt.tight_layout()
plt.savefig('distribution.eps',dpi=600,format='eps',bbox_inches='tight')

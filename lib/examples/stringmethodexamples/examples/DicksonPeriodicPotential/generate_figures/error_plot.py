import numpy as np
import h5py
import yaml

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from glob import glob
import os

basedir = '..'


# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

legfont = fm.FontProperties(size=10)

sims = ['common', 'rare']

# Get parameters from yaml files
with open(os.path.join(basedir,'configs/bruteforce_run_config.yaml'),'r') as f:
    bf_cfg_data = [grp for grp in yaml.load_all(f)]

we_cfg_data = {}
for sname in sims:
    with open(os.path.join(basedir,'configs/we_{}_run_config.yaml'.format(sname)),'r') as f:
        we_cfg_data[sname] = yaml.load(f)

bf_dt = {}
for cfg_data in bf_cfg_data:
    for sname in sims:
        if sname in cfg_data['name']:
            bf_dt[sname] = cfg_data['steps_per_block'] * cfg_data['blocks_per_dump']

we_dt = {}
for sname in sims:
    we_dt[sname] = we_cfg_data[sname]['tau'] * we_cfg_data[sname]['nbins'] * we_cfg_data[sname]['target_count']


we_offset = [50, 50]
#T = [20.0*1E8,20.0*1E9] # Numbers used for this study
T = [1.6E8, 1.0E10] # Values used in the Dickson paper
ntarget = [1, 10]
nbins = 100

fsize = (3.375, 6.0)

logfunc = np.log10

# Figure setup
fig = plt.figure(1,figsize=fsize)
ax = {}
ax['common'] = fig.add_subplot(211)
ax['rare'] = fig.add_subplot(212)

for sni,sname in enumerate(sims):
    # Get all brute force data
    fnames = glob(os.path.join(basedir,'bruteforce_{sname}/rawcounts_*.npy'.format(sname=sname)))

    # Target data
    target = np.zeros((nbins,))

    for k in xrange(ntarget[sni]):
        tk = np.load(fnames[k])
        target += np.sum(tk,axis=0)

    target /= np.sum(target)
    log_Pt = logfunc(target)
    nframes = tk.shape[0]
    print '{} -- nframes: {} bf_dt: {}'.format(sname, nframes,bf_dt[sname])
    t = bf_dt[sname] * np.arange(nframes,dtype=np.int64)
    print np.min(t), np.max(t)
    
    # Brute Force
    err = np.zeros((len(fnames)-ntarget[sni],len(t)))
    maxi = []
    for ki,k in enumerate(xrange(ntarget[sni],len(fnames))):
        P = np.cumsum(np.load(fnames[k]),axis=0)
        Psum = np.sum(P,axis=1)
        P /= Psum[:,np.newaxis]

        log_Pi = logfunc(P)
        Ei = log_Pi - log_Pt[np.newaxis,:]
        ii = np.where(np.abs(log_Pi) == np.inf)
        maxi.append(np.max(ii[0]))
    
        Ei[ii] = logfunc(1.0/T[sni]) - np.tile(log_Pt[np.newaxis,:],(Ei.shape[0],1))[ii]
        err[ki,:] = np.sqrt(np.mean(Ei**2,1))
   
    ax[sname].loglog(t,np.mean(err,axis=0),color='black',ls=':',label='CONV')
    maxi = np.array(maxi)
    maxii = np.ceil(np.mean(maxi))
    ax[sname].loglog([t[maxii]],[np.mean(err,axis=0)[maxii]],marker='o',ms=4,color='black',label='_nolegend_')

    # WE
    fnames = glob(os.path.join(basedir,'we_{}/analysis/*/distribution.h5'.format(sname)))
    data = []
    data_shape = []
    for fn in fnames:
        h5file = h5py.File(fn,'r')
        d = h5file['data'][:]
        
        data.append(d)
        data_shape.append(list(d.shape))

    data_shape = np.array(data_shape)
    assert np.alltrue(data_shape == data_shape[0])

    offset = we_offset[sni]

    we_nframes = data_shape[0,0]
    werr = np.zeros((10,we_nframes-offset))
    tt = we_dt[sname]*np.arange(we_nframes-offset) + we_dt[sname]*offset
    maxi = []

    for k in xrange(len(data)):
        print 'WE {}'.format(k)
        hk = data[k][offset:,:]
        hk = np.cumsum(hk,0)
        hk_sum = np.sum(hk,1)
        hk /= hk_sum[:,np.newaxis]

        log_hk = logfunc(hk)

        Ei = log_hk - log_Pt[np.newaxis,:]
        ii = np.where(hk == 0.0)

        if ii[0].size == 0:
            maxi.append(0)
        else:
            maxi.append(np.max(ii[1]))

        Ei[ii] = logfunc(1.0/T[sni]) - np.tile(log_Pt[np.newaxis,:],(Ei.shape[0],1))[ii]
        werr[k,:] = np.sqrt(np.mean(Ei**2,1))

    maxi = np.array(maxi)
    maxii = np.ceil(np.mean(maxi))
    ax[sname].loglog([tt[maxii]],[np.mean(werr,axis=0)[maxii]],marker='o',ms=4,color='g',label='_nolegend_')
    print 'we max inf time: %.3e %.3e %.3e' % (np.max(tt[maxi]),np.min(tt[maxi]), np.mean(tt[maxi]))

    ax[sname].loglog(tt,np.mean(werr,0),'g',label='WE')
    ax[sname].set_xlabel('Time $({\delta}t)$')
    ax[sname].set_ylabel('Error')
    ax[sname].legend(loc=1,frameon=False,prop=legfont)


fig.set_size_inches(fsize)
plt.tight_layout()
plt.savefig('error.eps',dpi=600,format='eps',bbox_inches='tight')
plt.show()

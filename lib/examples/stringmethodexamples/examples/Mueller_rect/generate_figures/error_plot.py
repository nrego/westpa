import numpy as np
import h5py
import yaml
import bottleneck as bn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from glob import glob
import os

from smooth import smooth

basedir = '..'

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

legfont = fm.FontProperties(size=10)
fsize = (3.375, 6.0)

sims = ['we_1.5', 'we_2.5']
logfunc = np.log10


def estimate_rate(cdata, basedir, beta):
    # Estimate the rate at a given temperature beta supposing and Arrhenius dependence
    # using all other brute force data

    bf_data_dirs = glob(os.path.join(basedir,'bruteforce_*'))

    rates = np.zeros((len(bf_data_dirs),2))
    temps = np.zeros((len(bf_data_dirs),))

    for bi,bfd in enumerate(bf_data_dirs):
        temps[bi] = float(bfd[-3:])
        rates[bi,:] = calc_target_rate(cdata,basedir,temps[bi])

    rates = np.mean(rates,axis=1)

    # Linear fit data to desired temp
    z = np.polyfit(temps,rates,deg=1)
    p = np.poly1d(z)

    target_rate = 10.0**p(beta)

    return np.array([target_rate, target_rate]), np.zeros((2,))

def calc_target_rate(cdata,basedir,beta=None):

    if beta is None:
        beta = cdata['beta']

    if beta not in [1.0, 1.5, 2.0, 2.5]:
        target_rate, target_rate_std = estimate_rate(cdata,basedir,beta)
    else:
        bf_dir = os.path.join(basedir,'bruteforce_{}'.format(beta))
        bf_dt = cdata['analysis']['bf_dt']

        bf_data_files = glob(os.path.join(bf_dir,'rate_stats_{}_*.npy'.format(beta)))

        # Calculate target rate
        target_rates = np.zeros((10,2),np.float)

        for k,f in enumerate(bf_data_files[:10]):
            s = np.load(f)
            ss = np.cumsum(s,0)[-1,:]

            # A -> B
            target_rates[k,0] = (1.0*ss[1]) / (bf_dt*ss[2])

            # B -> A
            target_rates[k,1] = (1.0*ss[0]) / (bf_dt*ss[3])

        # Take the target as the average of all forward and backward rates
        target_rate = np.mean(target_rates,axis=0)
        target_rate_std = np.std(target_rates,axis=0)

    log_target_rate = logfunc(target_rate)

    print '{} target -- kAB: {} +/- {}, kBA: {} +/-'.format(beta,target_rate[0],target_rate_std[0],target_rate[1],target_rate_std[1])

    return log_target_rate

def calc_bruteforce(cdata,basedir,log_target_rate):
    # Brute force parameters
    beta = cdata['beta']

    bf_dir = os.path.join(basedir,'bruteforce_{}'.format(beta))
    bf_dt = cdata['analysis']['bf_dt']
    bf_blocksize = cdata['analysis']['bf_blocksize']

    bf_data_files = glob(os.path.join(bf_dir,'rate_stats_{}_*.npy'.format(beta)))

    # Determine number of frames
    s = np.load(bf_data_files[0])
    bf_nframes = s.shape[0]

    # Calculate time-dependent error estimate for brute force
    if len(bf_data_files) > 10:
        start = 10
    else:
        start = 0

    bf_err = np.zeros((10,2,bf_nframes))

    for k,f in enumerate(bf_data_files[start:]):
        s = np.load(f)
        ss = np.cumsum(s,0)

        rAB = (1.0*ss[:,1]) / (bf_dt*ss[:,2])
        rBA = (1.0*ss[:,0]) / (bf_dt*ss[:,3])

        bf_err[k,0,:] = logfunc(rAB) - log_target_rate[0]
        bf_err[k,1,:] = logfunc(rBA) - log_target_rate[1]

    bf_err = np.abs(bf_err)
    bf_err_avg = np.sqrt(np.mean(bf_err**2,0))
    bf_t = bf_dt * bf_blocksize * np.arange(bf_nframes)

    return bf_err_avg, bf_t

def calc_we(cdata,basedir,log_target_rate):
     # WE parameters
    we_dir = os.path.join(basedir,cdata['name'],'analysis')
    we_dt = cdata['tau']
    we_nbins = cdata['nbins']
    we_target_count = cdata['target_count']

    winsize_flux = cdata['analysis']['winsize_flux']
    winsize_err = cdata['analysis']['winsize_err']
    last_n = cdata['analysis']['last_n']

    we_nframes = 0

    we_data_files = glob(os.path.join(we_dir,'*/rate.h5'))

    for fname in we_data_files:
        f = h5py.File(fname,'r')
        we_nframes = max(we_nframes,f.attrs['last_completed_iter']-2)
        f.close() 

    we_err = np.empty((len(we_data_files),2,we_nframes))
    we_err.fill(np.nan)

    for k,fname in enumerate(we_data_files):
        print 'we: {}'.format(fname)
        f = h5py.File(fname,'r')
        s = f['data'][:]
        dget = min(we_nframes,s.shape[0])
        s = s[:dget,:]

        ss = np.empty_like(s)
        for i in xrange(4):
            ss[:,i] = smooth(s[:,i],winsize_flux,'flat')

        sm = s[-last_n:,:].sum(0)

        rAB = (1.0*ss[:,1]) / (we_dt*ss[:,2])
        rBA = (1.0*ss[:,0]) / (we_dt*ss[:,3])

        rABm = (1.0*sm[1]) / (we_dt*sm[2])
        rBAm = (1.0*sm[0]) / (we_dt*sm[3])

        print 'we_{} -- kAB: {}, kBA: {}'.format(k,rABm,rBAm)

        we_err[k,0,:rAB.shape[0]] = logfunc(rAB) - log_target_rate[0]
        we_err[k,1,:rBA.shape[0]] = logfunc(rBA) - log_target_rate[1]

        f.close()

    we_err = np.abs(we_err)

    #we_err_avg = np.sqrt(np.mean(we_err**2,0))
    we_err_avg = np.sqrt(bn.nanmean(we_err**2,0))

    for i in xrange(2):
        we_err_avg[i,:] = smooth(we_err_avg[i,:],winsize_err,'flat')

    we_t = we_dt * we_nbins * we_target_count * np.arange(we_nframes)

    return we_err_avg, we_t

# --------------------------------
# --- Main script ----------------
# --------------------------------
if __name__ == '__main__':
    fig = plt.figure(1,figsize=fsize)

    nsubplots = len(sims)

    ax = {}
    for si,sname in enumerate(sims):
        ax[sname] = fig.add_subplot(nsubplots,1,si+1)
        
        config_file = os.path.join(basedir,'configs/{}_run_config.yaml'.format(sname))
        with open(config_file,'r') as f:
            config_data = [grp for grp in yaml.load_all(f)]
            config_data[:] = [grp for grp in config_data if grp['name'] in sims]

        config_data = config_data[0]

        beta = config_data['beta']
        log_target_rate = calc_target_rate(config_data,basedir)

        if beta in [1.0, 1.5, 2.0, 2.5]:
            bf_err_avg, bf_t = calc_bruteforce(config_data, basedir, log_target_rate)

            ax[sname].loglog(bf_t,bf_err_avg[0,:],ls=':',color='black',label='CONV')

        we_err_avg, we_t = calc_we(config_data,basedir,log_target_rate)
        ax[sname].loglog(we_t[:-200],we_err_avg[0,:-200],color='black',label='WE')

        ax[sname].set_xlabel('Time $({\delta}t)$')
        ax[sname].set_ylabel('Error')
        ax[sname].legend(loc=1,frameon=False,prop=legfont)
        


    fig.set_size_inches(fsize)
    plt.tight_layout()
    plt.savefig('error.eps',dpi=600,format='eps',bbox_inches='tight')
    plt.show()



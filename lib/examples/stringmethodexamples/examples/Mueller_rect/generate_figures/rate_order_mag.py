import numpy as np
import yaml

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

from error_plot import calc_target_rate, calc_we, calc_bruteforce
basedir = '..'

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 10.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

legfont = fm.FontProperties(size=10)
fsize = (3.375,3.0)

sims = ['we_1.0','we_1.5','we_2.0','we_2.5','we_3.0']

fig = plt.figure(1,figsize=fsize)
ax = fig.add_subplot(111)

betas = np.zeros((len(sims),))
oom = np.zeros_like(betas)
factx = np.zeros_like(betas)

oom_bf = np.zeros_like(betas)
factx_bf = np.zeros_like(betas)


for si,sname in enumerate(sims):
    config_file = os.path.join(basedir,'configs/{}_run_config.yaml'.format(sname))
    with open(config_file,'r') as f:
        config_data = [grp for grp in yaml.load_all(f)]
        config_data[:] = [grp for grp in config_data if grp['name'] in sims]

    config_data = config_data[0]

    betas[si] = config_data['beta']
    log_target_rate = calc_target_rate(config_data,basedir)

    we_err_avg, we_t = calc_we(config_data,basedir,log_target_rate)

    # Find time at WE last crosses order of magnitude estimate of the rate
    ii = np.argwhere(we_err_avg[0,:] > 1.0)
    jj = 1 if np.max(ii) < 1 else np.max(ii)
    we_oom = we_t[jj]
    mfpt = 1.0/(10.0**log_target_rate[0])
    oom[si] = we_oom/mfpt

    # Find time at which WE last crosses factor of 3 estimate of rate
    ii = np.argwhere(we_err_avg[0,:] > 0.5)
    jj = 1 if np.max(ii) < 1 else np.max(ii)
    we_factx = we_t[jj]
    mfpt = 1.0/(10.0**log_target_rate[0])
    factx[si] = we_factx/mfpt


data, = ax.semilogy(betas,oom,marker='o',ls='-',zorder=1000,color='black',label='$T_1$')
data.set_clip_on(False)
data, = ax.semilogy(betas,factx,marker='v',ls='-',zorder=1000,color='black',label='$T_{0.3}$')
data.set_clip_on(False)

ax.semilogy(betas,np.ones_like(oom),ls=':',lw=2.0,color='black')
ax.set_xlabel('${\\beta}$')
ax.set_ylabel('$T_X$/MFPT')
ax.legend(loc=1,frameon=False,prop=legfont)

ax.axis([1.0,3.0,1.0E-5,100])


fig.set_size_inches(fsize)
plt.tight_layout()
plt.savefig('rate_order_mag.eps',dpi=600,format='eps',bbox_inches='tight')
plt.show()




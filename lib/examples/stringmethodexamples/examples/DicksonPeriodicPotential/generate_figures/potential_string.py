import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid

import voronoi
from utils import dfunc as dfunc


simname = 'we_common'
pcoord_h5 = '../{}/analysis/0/pcoords.h5'.format(simname)
string_h5 = '../{}/analysis/0/strings.h5'.format(simname)

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 8.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 8.
mpl.rcParams['xtick.labelsize'] = 8.
mpl.rcParams['ytick.labelsize'] = 8.

fsize = (3.375,6.0)
clines = np.arange(20)
my_cmap = cm.get_cmap('RdYlBu_r')


# Potential Data
x = np.linspace(-1,1,100)
y = np.linspace(0,1,100)

X,Y = np.meshgrid(x,y)

gamma = 2.25
alpha = 1.125
beta = 4.0
F = 1.8

V = gamma*(X - 0.5*np.sin(2.0*np.pi*Y))**2 + alpha*np.cos(2.0*np.pi*Y)
V *= beta
V -= np.min(V)

# Get string data
f = h5py.File(string_h5, 'r')

centers = f['strings'][-1,...]
f.close()

offset = np.zeros_like(centers)
offset[:,1] = 1.0
cen_rep = np.r_[centers + offset, centers, centers - offset]

# Get coordinates
f = h5py.File(pcoord_h5, 'r')
iters = [x for x in f['iterations'].keys() if 'iter_' in x]
crd = []
for iiter in iters[-10:]:
    crd.append(f['iterations'][iiter][:])

crd = np.vstack(crd)
assignments = np.empty((crd.shape[0],))

for pi,p in enumerate(crd):
    d = dfunc(p,cen_rep)
    assignments[pi] = np.argmin(d)

#fig = plt.figure(1,figsize=fsize,dpi=600)
fig = plt.figure()

grid = ImageGrid(fig, 111,
                      aspect='equal',
                      nrows_ncols = (1, 1),
                      direction="row",
                      axes_pad = 0.05,
                      add_all=True,
                      label_mode = "L",
                      share_all = True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="5%",
                      cbar_pad=0.08,
                      )


# Plot potential
im = grid[0].contourf(X,Y,V,clines,cmap=my_cmap,zorder=0)

# Plot coordinates
#grid[0].plot(crd[:,0],crd[:,1],c='0.2',mec='0.2',mfc='0.2',marker='.',ms=3,ls='None')
uassign = np.unique(assignments)
for k in uassign:
    ii = np.where(assignments == k)[0]
    np.random.shuffle(ii)
    num_points = min(ii.size,500)
    jj = ii[:num_points]

    if k % 2 == 0:
        cc = '0.8'
    else:
        cc = '0.4'

    grid[0].plot(crd[jj,0],crd[jj,1],c=cc,mec=cc,mfc=cc,marker='.',ms=3,ls='None',zorder=1)


# Plot string
grid[0].plot(centers[:,0],centers[:,1],ms=3,marker='o',color='k',ls='None',zorder=2)

# Plot Voronoi Cells
segments = voronoi.voronoi(cen_rep[:,0],cen_rep[:,1])
lines = mpl.collections.LineCollection(segments, color='k',lw=1.0,zorder=3)
grid[0].add_collection(lines)

grid[0].set_xlabel('X')
grid[0].set_ylabel('Y')


grid[0].axis([-1,1,0,1])
grid[0].cax.colorbar(im)
grid[0].cax.toggle_label(True)

fig.set_size_inches(fsize)
plt.tight_layout()
plt.savefig('potential_string.eps',dpi=600,format='eps',bbox_inches='tight')
plt.show()



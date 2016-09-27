import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
import h5py

import voronoi


beta = 1.0
simname = 'we_1.0'
pcoord_h5 = '../{}/analysis/0/pcoords.h5'.format(simname)
string_h5 = '../{}/analysis/0/strings.h5'.format(simname)

# Font settings
### rcParams are the default parameters for matplotlib
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 12.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

## Potential Data
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)

X,Y = np.meshgrid(x,y)

alpha = 3.0
gamma  = 3.0
chi1 = 2.25
chi2 = 4.5
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y,X)
V = alpha*(r - gamma)**2 - chi1*np.cos(2*theta) - chi2*np.cos(4*theta)
V -= np.min(V)

clines = np.arange(18)
my_cmap = cm.get_cmap('RdYlBu_r')


## String data
f = h5py.File(string_h5,'r')

centers = f['strings'][-1,...]
ncenters = centers.shape[0] // 2
f.close()

## Coordinate data
f = h5py.File(pcoord_h5,'r')
iters = [x for x in f['iterations'].keys() if 'iter_' in x]
iiter = max(iters)
crd = f['iterations'][iiter][:]

f.close()

# Plot figure
fig = plt.figure(1,figsize=(6.69,3.5),dpi=600)

grid = ImageGrid(fig, 111,
                      aspect='equal',
                      nrows_ncols = (1, 3),
                      direction="row",
                      axes_pad = 0.05,
                      add_all=True,
                      label_mode = "L",
                      share_all = True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="10%",
                      cbar_pad=0.05,
                      )


for k in xrange(3):
    grid[k].set_xlabel('X')
    grid[k].set_ylabel('Y')
#grid[0].set_xlabel("X")
#grid[1].set_xlabel("X")
#grid[2].set_xlabel("X")
#grid[0].set_ylabel("Y")

# Plot pcoords
iiA = np.where(crd[:,2] == 0)[0]
iiB = np.where(crd[:,2] == 1)[0]

grid[0].plot(crd[iiA,0],crd[iiA,1],c='r',mec='r',mfc='r',marker='.',ms=3,ls='None')
grid[0].plot(crd[iiB,0],crd[iiB,1],c='b',mec='b',mfc='b',marker='.',ms=3,ls='None')
cirA = plt.Circle((-3,0), radius=1.0,  ec='k', fill=False, lw=1.5,zorder=1000)
cirB = plt.Circle((3,0), radius=1.0,  ec='k', fill=False, lw=1.5,zorder=1000)
grid[0].add_patch(cirA)
grid[0].add_patch(cirB)

# Plot string
for k,gk in enumerate(xrange(1,3)):

    # Draw contour map of potential
    im = grid[gk].contourf(X,Y,V,clines,cmap=my_cmap)

    # Draw circles defining state boundaries
    cirA = plt.Circle((-3,0), radius=1.0,  ec='k', fill=False, lw=0.5)
    cirB = plt.Circle((3,0), radius=1.0,  ec='k', fill=False, lw=0.5)
    grid[gk].add_patch(cirA)
    grid[gk].add_patch(cirB)

    if k == 0:
        cen = centers[:ncenters,:]
    else:
        cen = centers[ncenters:,:]

    grid[gk].plot(cen[:,0],cen[:,1],'k.-',lw=1,ms=2,marker='o')

    segments = voronoi.voronoi(cen[:,0],cen[:,1])
    lines = mpl.collections.LineCollection(segments, color='k',lw=0.5)
    grid[gk].add_collection(lines)

    grid[gk].axis([-5,5,-5,5])

grid[2].cax.colorbar(im)
grid[2].cax.toggle_label(True)

    
#plt.show()
plt.savefig('potential_string.eps',dpi=600,format='eps',bbox_inches='tight')


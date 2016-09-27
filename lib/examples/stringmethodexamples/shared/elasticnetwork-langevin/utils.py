import numpy as np
import pyqcprot as qcp

def interpolate_conformations(confA,confB,N):

    assert confA.shape == confB.shape

    natoms = confA.shape[0]

    x0 = confA.copy()
    xN = confB.copy()
    
    # Align confA and confB
    x0 -= np.mean(x0,axis=0)
    xN -= np.mean(xN,axis=0)

    R = np.zeros((9,),dtype=np.float64)
    
    rmsd = qcp.CalcRMSDRotationalMatrix(x0.T.astype(np.float64),xN.T.astype(np.float64),natoms,R,None)
    print('RMSD: {}'.format(rmsd))

    R = np.matrix(R.reshape(3,3))
    xN = xN*R

    # Interpolate coordinates form x0 to xN
    confs = np.zeros((N,natoms*3))

    x0 = np.ravel(x0)
    xN = np.ravel(xN)

    for k in xrange(natoms*3):
        confs[:,k] = np.linspace(x0[k],xN[k],N)
        
    return confs

def test():
    from netcdf4storage import NetCDF4Storage

    x = np.load('data.npz')
    natoms = x['coordsA'].shape[0]
    N = 15
    
    nc = NetCDF4Storage(natoms=natoms)
    nc.initialize_netcdf('confs.nc')


    confs = interpolate_conformations(x['coordsA'],x['coordsB'],N)
    confs = confs.reshape((N,natoms,3))

    for k in xrange(N):
        nc.write_frame(confs[k,:,:])

    nc.ncfile.close()


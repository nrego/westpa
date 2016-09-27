import netCDF4 as netcdf

class NetCDF4Storage(object):
    """docstring for NetCDF4Storage"""
    def __init__(self, storedir='./',natoms=None,dt=0.001,zlib=False, cmplevel=1):
        super(NetCDF4Storage, self).__init__()
        self._storedir = storedir
        self._natoms = natoms
        self._zlib = zlib
        self._clvl = cmplevel
        self.ncfile = None
        self.curr_frame = 0
        self.dt = dt

    def initialize_netcdf(self,fname,store_vel=False):
        """docstring for initialize_netcdf"""
        ncfile = netcdf.Dataset(self._storedir + '/' + fname, clobber=True,mode='w',format='NETCDF3_64BIT')

        # Set global attributes.
        setattr(ncfile, 'program', 'custom simulation')
        setattr(ncfile, 'programVersion', '0.1')
        setattr(ncfile, 'Conventions', 'AMBER')
        setattr(ncfile, 'ConventionVersion', '1.0')

        # Create dimensions
        ncfile.createDimension('frame', None) # unlimited number of replicas
        ncfile.createDimension('atom', self._natoms) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions

        # Create variables.
        ncvar_coords = ncfile.createVariable('coordinates','f8', ('frame','atom','spatial'),zlib=self._zlib,complevel=self._clvl)
        setattr(ncvar_coords, 'units', 'angstrom')

        if store_vel:
            ncvar_velocs = ncfile.createVariable('velocities','f8', ('frame','atom','spatial'),zlib=self._zlib,complevel=self._clvl)
            setattr(ncvar_velocs, 'units', 'angstrom/picosecond')
        
        ncfile.sync()
        self.ncfile = ncfile

        return
        
    def write_frame(self,coords,velocs=None):
        """docstring for write_iteration"""
        
        self.ncfile.variables['coordinates'][self.curr_frame,:,:] = coords[:,:]
        if velocs is not None and 'velocities' in self.ncfile.variables:
            self.ncfile.variables['velocities'][self.curr_frame,:,:] = velocs[:,:]

        self.curr_frame += 1

        self.ncfile.sync()

        return

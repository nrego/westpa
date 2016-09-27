import numpy as np
from ElasticNetwork import ElasticNetwork
from netcdf4storage import NetCDF4Storage as trajstore
import sys,os,time
import logging
from ConfigParser import SafeConfigParser

def genrandint():
    'Generates a random integer between 0 and (2^32)-1'
    x = 0
    for i in range(4):
        x = (x << 8)+ord(os.urandom(1))
    return x

def run(config):

    # Set up logging
    logname = config.get('outputs','log')
    print('Setting up logging: {}'.format(logname))
    logging.basicConfig(filename=logname,level=logging.DEBUG,filemode='w')

    NUM_BLOCKS = config.getint('parameters','num_blocks')
    STEPS_PER_BLOCK = config.getint('parameters','steps_per_block')
    
    model = {}
    model['mass'] = config.getfloat('model','mass')
    model['gamma'] = config.getfloat('model','gamma')
    model['temp'] = config.getfloat('model','temp') 
    model['dt'] = config.getfloat('model','dt')
    model['sigma'] = config.getfloat('model','sigma')
    model['eps'] = config.getfloat('model','eps')
    model['betamix'] = config.getfloat('model','betamix')

    model.update(np.load(config.get('model','ff_data')))
    model['seed'] = genrandint()    
    
    init_pos = {}
    init_pos['coordsA'] = model['coordsA']
    init_pos['coordsB'] = model['coordsB']
    del model['coordsA']
    del model['coordsB']

    system = ElasticNetwork(**model)

    # Setup storage 
    print('Setting up netcdf4 trajectory storage')
    nc = trajstore(natoms=system.natoms)
    nc.initialize_netcdf(config.get('outputs','trajname'))

    # Initial coords and velocities
    pos = init_pos[config.get('model','init_pos')]
    
    # Assign velocities drawn from Maxwell-Boltzmann distribution
    sigma = np.sqrt(model['temp']*0.001987191/model['mass'])
    vel = sigma*np.random.normal(size=pos.shape)
    
    print('Starting Simulation')
    for dk in xrange(NUM_BLOCKS):
        t1 = time.time()
        system.step(pos,vel,STEPS_PER_BLOCK)    
        nc.write_frame(pos,vel)
        EKin = 0.5*model['mass']*np.sum(vel**2)
        T = EKin*(2./3)/(0.001987191*system.natoms)
        logging.info('Completed {} of {} steps: {} s Ekin: {} Temp: {}'.format(dk,NUM_BLOCKS-1,time.time() - t1,EKin,T))

    nc.ncfile.close()


if __name__ == '__main__':
    config = SafeConfigParser()
    config.read(sys.argv[1])

    run(config)


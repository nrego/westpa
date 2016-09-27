import numpy as np
import cIntegratorSimple
import ForceFields
import os
import time
import logging
import argparse


def genrandint():
    'Generates a random integer between 0 and (2^32)-1'
    x = 0
    for i in range(4):
        x = (x << 8) + ord(os.urandom(1))
    return x


def run(sid,alpha,NUM_BLOCKS,STEPS_PER_BLOCK,BLOCKS_PER_DUMP):

    print('Setting up logging')
    logging.basicConfig(filename='sim_{}.log'.format(sid),level=logging.DEBUG,filemode='w')
    logging.info('Sim ID: {}'.format(sid))
    logging.info('alpha: {}'.format(alpha))
    logging.info('NUM_BLOCKS: {}'.format(NUM_BLOCKS))
    logging.info('STEPS_PER_BLOCK: {}'.format(STEPS_PER_BLOCK))
    logging.info('BLOCKS_PER_DUMP: {}'.format(BLOCKS_PER_DUMP))

    ff = ForceFields.Dickson2dPeriodicForce_revised(alpha)

    MASS = 1.0
    XI = 1.5
    BETA = 4.0
    NDIMS = 2
    DT = 0.002
    ISPERIODIC = np.array([0,1], dtype=np.int)
    BOXSIZE = np.array([1.0E8,1.0], dtype=np.float32)

    print('Instantiating Integrator')
    integrator = cIntegratorSimple.SimpleIntegrator(ff,MASS,XI,BETA,DT,NDIMS,ISPERIODIC,BOXSIZE,genrandint())

    # Initial coords and velocities
    x = np.array([0.0,0.5], dtype=np.float32)

    totblocks = NUM_BLOCKS//BLOCKS_PER_DUMP

    bins = np.linspace(0.0,1.0,100)
    hist = np.empty((totblocks,len(bins)))

    print('Starting Simulation')
    for dk in xrange(totblocks):
        t1 = time.time()
        ctemp = integrator.step_save(x,BLOCKS_PER_DUMP*STEPS_PER_BLOCK,STEPS_PER_BLOCK)
        x = ctemp[-1,:]

        # bin data
        h,edges = np.histogram(ctemp[:,1],range=(0.0,1.0),bins=100)
        hist[dk,:] = h
        logging.info('Completed {} of {} steps: {} s'.format(dk,totblocks-1,time.time() - t1))

    np.save('rawcounts_{}'.format(sid),hist)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate Dickson Ring Potential')
    parser.add_argument('-a',dest='alpha', action='store',type=float)
    parser.add_argument('-i',dest='sid', action='store',type=int)
    parser.add_argument('--nblocks',dest='nblocks',action='store',type=int)
    parser.add_argument('--steps',dest='steps',action='store',type=int)
    parser.add_argument('--dsize',dest='dsize',action='store',type=int)

    args = parser.parse_args()

    run(args.sid, args.alpha, args.nblocks, args.steps, args.dsize)

import sys
if not any(p in sys.path for p in ['', '.']):
    sys.path.insert(0, '')

import numpy as np
import cIntegratorSimple
import ForceFields

from utils import update_rate_stats

import logging
import os
import time
import argparse

coord_dtype = np.float32


def genrandint():
    'Generates a random integer between 0 and (2^32)-1'
    x = 0
    for i in range(4):
        x = (x << 8) + ord(os.urandom(1))
    return x


def run(sid,beta,NUM_BLOCKS,STEPS_PER_BLOCK,BLOCKS_PER_DUMP):

    print('Setting up logging')
    logging.basicConfig(filename='sim_{}_{}.log'.format(beta,sid),level=logging.DEBUG)
    logging.info('Sim ID: {}'.format(sid))
    logging.info('beta: {}'.format(beta))
    logging.info('NUM_BLOCKS: {}'.format(NUM_BLOCKS))
    logging.info('STEPS_PER_BLOCK: {}'.format(STEPS_PER_BLOCK))
    logging.info('BLOCKS_PER_DUMP: {}'.format(BLOCKS_PER_DUMP))

    ff = ForceFields.Dickson2dRingForce()

    MASS = 1.0
    XI = 1.5
    BETA = beta
    NDIMS = 2
    DT = 0.005
    ISPERIODIC = np.array([0,0],dtype=np.int)
    BOXSIZE = np.array([1.0E8,1.0E8], dtype=coord_dtype)

    print('Instantiating Integrator')
    integrator = cIntegratorSimple.SimpleIntegrator(ff,MASS,XI,BETA,DT,NDIMS,ISPERIODIC,BOXSIZE,genrandint())

    # Initial coords
    x = np.array([-3.0,0.0], dtype=coord_dtype)
    last_state = 0

    totblocks = NUM_BLOCKS//BLOCKS_PER_DUMP
    hist = np.empty((totblocks,100))
    rate_stats = np.empty((totblocks,4))

    print('Starting Simulation')
    for dk in xrange(totblocks):
        t1 = time.time()
        ctemp = integrator.step_save(x,BLOCKS_PER_DUMP*STEPS_PER_BLOCK,STEPS_PER_BLOCK)
        x = ctemp[-1,:]

        # Collect transition data
        last_state, trans_to_0, trans_to_1, time_in_0, time_in_1 = update_rate_stats(ctemp,last_state)
        rate_stats[dk,:] = [trans_to_0, trans_to_1, time_in_0, time_in_1]

        # bin data
        theta = np.arctan2(ctemp[:,1],ctemp[:,0])/np.pi
        h,edges = np.histogram(theta,range=(-1.0,1.0),bins=100)
        hist[dk,:] = h
        logging.info('Completed {} of {} steps: {} s'.format(dk,totblocks-1,time.time() - t1))

    np.save('rawcounts_{}_{}'.format(beta,sid),hist)
    np.save('rate_stats_{}_{}'.format(beta,sid),rate_stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate Dickson Ring Potential')
    parser.add_argument('-b',dest='beta', action='store',type=float)
    parser.add_argument('-i',dest='sid', action='store',type=int)
    parser.add_argument('--nblocks',dest='nblocks',action='store',type=int)
    parser.add_argument('--steps',dest='steps',action='store',type=int)
    parser.add_argument('--dsize',dest='dsize',action='store',type=int)

    args = parser.parse_args()

    run(args.sid, args.beta, args.nblocks, args.steps, args.dsize)

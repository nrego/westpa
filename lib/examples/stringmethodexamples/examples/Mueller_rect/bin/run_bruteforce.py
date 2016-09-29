import multiprocessing
import os
import argparse

file_dir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.split(file_dir)[0]

script_template = """
mkdir -p {rundir}
cd {rundir}
ln -s {basedir}/we_base/ForceFields.so
ln -s {basedir}/we_base/utils.so
ln -s {basedir}/we_base/cIntegratorSimple.so
ln -s {basedir}/bin/simulate.py

python simulate.py -b {beta} -i {sid} --nblocks {nblocks} --steps {steps} --dsize {dsize} >/dev/null &
wait
"""


def run_job(kwargs):
    script = script_template.format(**kwargs)

    # Create directory
    if not os.path.exists(kwargs['rundir']):
        try:
            os.makedirs(kwargs['rundir'])
        except:
            pass

    # Setup external run script
    sname = os.path.join(kwargs['rundir'], kwargs['script_name'])
    with open(sname, 'w') as f:
        for line in script:
            f.write(line)

    os.system('chmod u+x {}'.format(sname))
    print('Running {}'.format(sname))
    os.system('{}'.format(sname))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bruteforce run script')
    parser.add_argument('-n', dest='nsims', type=int, default=10, help='number of simulations to run')
    parser.add_argument('-w', dest='nworkers', type=int, default=multiprocessing.cpu_count(),
                        help='number of cores to use')
    parser.add_argument('--start-sim', dest='start_sim', type=int, default=0, help='Starting sim number to run')

    args = parser.parse_args()

    pool_size = args.nworkers
    pool = multiprocessing.Pool(processes=pool_size)

    # Build inputs
    simp = {}
    simp[1.0] = {'steps': 20, 'nblocks': int(2e8), 'dsize': 10000}
    simp[1.5] = {'steps': 20, 'nblocks': int(1e9), 'dsize': 50000}
    simp[2.0] = {'steps': 20, 'nblocks': int(3e9), 'dsize': 150000}
    simp[2.5] = {'steps': 20, 'nblocks': int(1e10), 'dsize': 500000}

    inputs = []

    for beta in [1.0, 1.5, 2.0, 2.5]:
        for k in xrange(args.start_sim, args.start_sim + args.nsims):
            dict_in = {}
            dict_in['basedir'] = basedir
            dict_in['rundir'] = os.path.join(basedir, 'bruteforce_{}'.format(beta))
            dict_in['script_name'] = 'run_{}_{}.sh'.format(beta, k)
            dict_in['sid'] = k
            dict_in['beta'] = beta
            dict_in['nblocks'] = simp[beta]['nblocks']
            dict_in['steps'] = simp[beta]['steps']
            dict_in['dsize'] = simp[beta]['dsize']
            inputs.append(dict_in)

    for job in inputs:
        print('Beta: {} Job ID: {}'.format(job['beta'], job['sid']))

    pool.map(run_job, inputs, chunksize=1)

import multiprocessing
import os, sys
import yaml
import argparse
import glob


file_dir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.split(file_dir)[0]

scripts = ['calculate_rate', 'calculate_distribution', 'get_pcoords',
           'get_strings', 'get_nreplicas', 'all']


def run_job(kwargs):
    grp_root = os.path.join(basedir, kwargs['config_data']['name']) 
    sim_root = os.path.join(grp_root, kwargs['sim_index'])
    cfg_file = os.path.join(sim_root, 'we_phase1.cfg')

    print kwargs['config_data']['name']

    if kwargs['script'] != 'all':
        script = os.path.join(basedir, 'analysis', kwargs['script'])
        h5out_file = os.path.join(grp_root, 'analysis', kwargs['sim_index'], script.split('_')[-1] + '.h5')
        os.system('cd {} && $WEST_ROOT/bin/west {}.py -r {} -o {}'.format(sim_root, script, cfg_file, h5out_file))
    else:
        print 'Running all analysis scripts'
        for s in scripts:
            if s != 'all':
                sp = os.path.join(basedir, 'analysis', s)
                h5out_file = os.path.join(grp_root, 'analysis', kwargs['sim_index'], sp.split('_')[-1] + '.h5')
                os.system('cd {} && $WEST_ROOT/bin/west {}.py -r {} -o {}'.format(sim_root, sp, cfg_file, h5out_file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config_file', required=True, nargs='+', help='yaml config file name')
    parser.add_argument('-n', dest='name', nargs='*', help='simulation name to run; by default run all')
    parser.add_argument('-w', dest='nworkers', type=int, default=multiprocessing.cpu_count(), help='number of cores to use')
    parser.add_argument('-s', dest='script', required=True, choices=scripts, help='analysis script to run')
    args = parser.parse_args()

    # Setup worker pool
    pool_size = args.nworkers
    pool = multiprocessing.Pool(processes=pool_size)

    # Build inputs
    inputs = []

    config_data = []
    for cf in args.config_file:
        with open(cf, 'r') as f:
            config_data.extend([grp for grp in yaml.load_all(f)])

    # Remove simulations not in args.name
    if not isinstance(args.name, list):
        args.name = [args.name]

    if args.name is not None:
        config_data[:] = [grp for grp in config_data if grp['name'] in args.name]

    if len(config_data) == 0:
        print('ERROR: No simulations to run')
        sys.exit(1)

    for grp in config_data:
        h5files = glob.glob(os.path.join(basedir, grp['name'], '*/west.h5'))
        for h5f in h5files:
            si = os.path.basename(os.path.dirname(h5f))
            dict_in = {}
            dict_in['config_data'] = grp
            dict_in['script'] = args.script
            dict_in['sim_index'] = si
            inputs.append(dict_in)

            # Create run directory
            if not os.path.exists(os.path.join(basedir, grp['name'], 'analysis', si)):
                os.makedirs(os.path.join(basedir, grp['name'], 'analysis', si))

    pool.map(run_job, inputs, chunksize=1)

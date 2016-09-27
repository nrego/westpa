import multiprocessing
import os, sys
import yaml
import argparse
import subprocess

file_dir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.split(file_dir)[0]

script_template = """
#!/bin/bash
cd {rundir}

export WEST_SIM_ROOT={rundir}

{env_variables}

echo west_pythonpath: $WEST_PYTHONPATH
echo west_sim_root: $WEST_SIM_ROOT
echo west_root: $WEST_ROOT

BSTATE_ARGS_0="--bstate initA,0.5"
BSTATE_ARGS_1="--bstate initB,0.5"

if [ ! -f west.h5 ];
then
    $WEST_ROOT/bin/w_init -r we_{sim_name}.cfg --work-manager=serial $BSTATE_ARGS_0 $BSTATE_ARGS_1 > sim_{sim_name}_init.log &
    wait
fi

if [ {nworkers} -eq 1 ]
then
    echo 'Running in serial mode'
    $WEST_ROOT/bin/w_run {profile} -r we_{sim_name}.cfg --verbose --work-manager=serial > sim_{sim_name}.log &
else
    echo 'Using zmq work manager'
    $WEST_ROOT/bin/w_run {profile} -r we_{sim_name}.cfg --verbose --work-manager=processes --n-workers={nworkers} &> sim_{sim_name}.log &
fi

wait
"""


def build_west_cfg(config_data, protocol):
    opts = ['init_pos', 'nbins', 'target_count', 'tau',
            'propagator_block_size', 'adjust_counts']
    wd = {k:config_data[k] for k in opts}
    wd['max_iterations'] = protocol['max_iterations']

    # weed parameters
    for p in protocol['weed']:
        wd['weed_' + p] = protocol['weed'][p]

    # string method parameters
    for p in protocol['stringmethod']:
        wd['sm_' + p] = protocol['stringmethod'][p]

    return wd


def run_job(kwargs):

    rundir = kwargs['rundir']
    config_data = kwargs['config_data']
    args = kwargs['args']

    # Setup run directory if it does not already exist
    if not os.path.exists(rundir):
        os.system('cp -RP {} {}'.format('we_base', rundir))

    # Get protocols to run
    protocols = config_data['protocols']

    if args.protocols is not None:
        protocols[:] = [p for p in protocols if p['name'] in args.protocols]

    for p in protocols:
        # Setup external run script
        env_file = os.path.join(rundir, 'env.sh')
        with open(env_file, 'r') as f:
            ev = f.read()

        sname = os.path.join(rundir,'run_{}.sh'.format(p['name']))
        prof_flag = '--profile' if args.profile else ''
        
        script = script_template.format(rundir=rundir, sim_name=p['name'],nworkers=args.nworkers, 
                                        env_variables=ev, profile=prof_flag,timeout=args.timeout)
        
        with open(sname,'w') as f:
            for line in script:
                f.write(line)

        # Setup west config script
        wcfg_dict = build_west_cfg(config_data,p)
        print wcfg_dict
        with open('we_base/we_base.cfg','r') as fin, open(os.path.join(rundir,'we_{}.cfg'.format(p['name'])),'w') as fout:
            we_cfg_template = fin.read()
            wcfg = we_cfg_template.format(**wcfg_dict)

            for line in wcfg:
                fout.write(line)

        os.system('chmod u+x {}'.format(sname))
        if not args.norun:
            print('Running {}'.format(sname))
            subprocess.check_call('{}'.format(sname), shell=True, stderr=subprocess.STDOUT,)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WEST run script')
    parser.add_argument('-c', dest='config_file', required=True, help='yaml config file name')
    parser.add_argument('-n', dest='name', required=True, help='simulation name to run')
    parser.add_argument('-p', dest='protocols', nargs='*', help='protocols to run; by default run all')
    parser.add_argument('-w', dest='nworkers', type=int, default=multiprocessing.cpu_count(), 
                                help='number of cores to use')
    parser.add_argument('--timeout', dest='timeout', type=int, default=60, help='timeout when running with zmq work manager')
    parser.add_argument('--profile', dest='profile', action='store_true', default=False, help='Profile code')
    parser.add_argument('--no-run', dest='norun', default=False, action='store_true', 
                                help='Only setup simulations but do not run them')

    args = parser.parse_args()

    config_data_all = []
    with open(args.config_file,'r') as f:
        config_data_all.extend([grp for grp in yaml.load_all(f)])

    sim_names = [grp['name'] for grp in config_data_all]

    try:
        ii = sim_names.index(args.name)
    except ValueError:
        print('ERROR: simulation name {} not found in config file [{}]'.format(args.name,sim_names))
        sys.exit(1)

    config_data = config_data_all[ii]

    dict_in = {}
    dict_in['rundir'] = os.path.join(basedir,config_data['name'])
    dict_in['config_data'] = config_data
    dict_in['args'] = args

    run_job(dict_in)

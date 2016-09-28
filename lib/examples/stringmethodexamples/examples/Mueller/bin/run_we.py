import multiprocessing
import os, sys
import yaml
import argparse
from IPython import embed

basedir = os.getcwd()

script_template = """
#!/bin/bash

cd {rundir}

export WEST_SIM_ROOT={rundir}

{env_variables}

echo west_pythonpath: $WEST_PYTHONPATH
echo west_sim_root: $WEST_SIM_ROOT
echo west_root: $WEST_ROOT

BSTATE_ARGS="--bstate initA,1.0"

if [ ! -f west.h5 ];
then
    $WEST_ROOT/bin/w_init -r we_{sim_name}.cfg $BSTATE_ARGS > sim_{sim_name}_init.log &
    wait
fi

$WEST_ROOT/bin/w_run -r we_{sim_name}.cfg --verbose > sim_{sim_name}.log &
wait
"""


def build_west_cfg(config_data, protocol):
    wd = {k:config_data[k] for k in  ['beta','nbins','target_count','tau','propagator_block_size','adjust_counts']}
    wd['max_iterations'] = protocol['max_iterations']

    # wess parameters
    for p in protocol['wess']:
        wd['wess_' + p] = protocol['wess'][p]

    # string method parameters
    for p in protocol['stringmethod']:
        wd['sm_' + p] = protocol['stringmethod'][p]

    return wd


def run_job(kwargs):

    rundir = kwargs['rundir']
    config_data = kwargs['config_data']
    args = kwargs['args']
    sim_id = kwargs['sim_id']

    # Setup run directory if it does not already exist
    if not os.path.exists(rundir):
        os.system('cp -RP {} {}'.format('we_base',rundir))

    # Get protocols to run
    protocols = config_data['protocols']

    if args.protocols is not None:
        protocols[:] = [p for p in protocols if p['name'] in args.protocols]

    for p in protocols:
        # Setup external run script
        env_file = os.path.join(rundir, 'env.sh')
        with open(env_file, 'r') as f:
            ev = f.read()

        sname = os.path.join(rundir,'run_{}_{}.sh'.format(sim_id,p['name']))
        script = script_template.format(rundir=rundir, sim_name=p['name'], env_variables=ev)

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
            os.system('{}'.format(sname))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WEST run script')
    parser.add_argument('-s', dest='nsims', type=int, default=10, help='number of simulations to run')
    parser.add_argument('-c', dest='config_file', required=True, nargs='+',help='yaml config file name')
    parser.add_argument('-n', dest='name', nargs='*', required=True,help='simulation name to run')
    parser.add_argument('-p', dest='protocols', nargs='*', help='protocols to run; by default run all')
    parser.add_argument('-w', dest='nworkers', type=int, default=multiprocessing.cpu_count(), 
                        help='number of cores to use')
    parser.add_argument('--sid_offset', dest='sid_offset', type=int, default=0,
                        help='offset for numbering simulations')
    parser.add_argument('--no-run', dest='norun', default=False, action='store_true',
                        help='Only setup simulations but do not run them')
    
    args = parser.parse_args()

    # Setup worker pool
    pool_size = args.nworkers
    pool = multiprocessing.Pool(processes=pool_size)

    # Build inputs
    inputs = []

    config_data = []
    for cf in args.config_file:
        with open(cf,'r') as f:
            config_data.extend([grp for grp in yaml.load_all(f)])

    # Remove simulations not in args.name
    if not isinstance(args.name,list):
        args.name = [args.name]

    if args.name is not None:
        config_data[:] = [grp for grp in config_data if grp['name'] in args.name]

    if len(config_data) == 0:
        print('ERROR: No simulations to run')
        sys.exit(1)

    for grp in config_data:
        for si in xrange(args.sid_offset, args.nsims + args.sid_offset):
            dict_in = {}
            dict_in['rundir'] = os.path.join(basedir,grp['name'],'{}'.format(si))
            dict_in['config_data'] = grp
            dict_in['args'] = args
            dict_in['sim_id'] = si
            inputs.append(dict_in)

        # Create run directory
        if not os.path.exists(os.path.join(basedir,grp['name'])):
            os.makedirs(os.path.join(basedir,grp['name']))
            os.makedirs(os.path.join(basedir,grp['name'],'analysis'))
    #embed()
    pool.map(run_job, inputs,chunksize=1)

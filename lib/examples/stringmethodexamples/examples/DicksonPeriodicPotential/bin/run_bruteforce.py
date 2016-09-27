import multiprocessing
import os, sys
import yaml
import argparse

basedir = os.getcwd()

script_template = """
#!/bin/bash


cd {rundir}
python simulate.py -a {alpha} -i {sid} --nblocks {num_blocks} --steps {steps_per_block} --dsize {blocks_per_dump} >/dev/null &
wait

"""

def build_cfg_dict(config_data):
    wd = {}
    wd['alpha'] = float(config_data['alpha'])
    wd['num_blocks'] = int(float(config_data['num_blocks']))
    wd['steps_per_block'] = int(config_data['steps_per_block'])
    wd['blocks_per_dump'] = int(config_data['blocks_per_dump'])
    
    return wd


def run_job(kwargs):

    rundir = kwargs['rundir']
    config_data = kwargs['config_data']
    args = kwargs['args']
    sim_id = kwargs['sim_id']

    cfg_dict = build_cfg_dict(config_data)

    # Setup external run script
    sname = os.path.join(rundir,'run_{}_{}.sh'.format(sim_id,config_data['name']))
    script = script_template.format(rundir=rundir,sid=sim_id,**cfg_dict)
    
    with open(sname,'w') as f:
        for line in script:
            f.write(line)

    os.system('chmod u+x {}'.format(sname))
    print('Running {}'.format(sname))
    os.system('{}'.format(sname))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bruteforce run script')
    parser.add_argument('-s', dest='nsims', type=int, default=10, help='number of simulations to run')
    parser.add_argument('-c', dest='config_file', required=True, nargs='+',help='yaml config file name')
    parser.add_argument('-n', dest='name', nargs='*', required=True,help='simulation name to run')
    parser.add_argument('-w', dest='nworkers', type=int, default=multiprocessing.cpu_count(), 
                                help='number of cores to use')
    parser.add_argument('--sid_offset', dest='sid_offset', type=int, default=0,
                                help='offset for numbering simulations')


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
            dict_in['rundir'] = os.path.join(basedir,grp['name'])
            dict_in['config_data'] = grp
            dict_in['args'] = args
            dict_in['sim_id'] = si
            inputs.append(dict_in)

        # Create run directory
        if not os.path.exists(os.path.join(basedir,grp['name'])):
            rundir = os.path.join(basedir,grp['name'])
            templatedir = os.path.join(basedir,'bruteforce_base')
            os.system('cp -RP {} {}'.format(templatedir,rundir))
            os.makedirs(os.path.join(rundir,'analysis'))
    
    print inputs
    pool.map(run_job, inputs,chunksize=1)

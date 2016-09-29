==========================
Dickson Ring Potential
==========================

Driven Brownian particle on a 2D surface with two pathways based on the potential and parameterization
found in:

    A. Dickson, A. Warmflash, and A. R. Dinner, “Separating forward and backward pathways 
    in nonequilibrium umbrella sampling,” J Chem Phys 131, 154104 (2009). http://dx.doi.org/10.1063/1.3244561

and

    A. Dickson, A. Warmflash, and A. R. Dinner, “Erratum: ”Separating forward and backward pathways in 
    nonequilibrium umbrella sampling” [J. Chem. Phys. 131, 154104 (2009)],” J Chem Phys 136, 239901 (2012).
    http://dx.doi.org/10.1063/1.4730937

Running conventional simulations
--------------------------------

To run the brute force simulations, use the script, ``bin/run_bruteforce.py``

::

    $ python bin/run_bruteforce.py -h
    usage: run_bruteforce.py [-h] [-n NSIMS] [-w NWORKERS] [--start-sim START_SIM]

    Bruteforce run script

    optional arguments:
      -h, --help            show this help message and exit
      -n NSIMS              number of simulations to run
      -w NWORKERS           number of cores to use
      --start-sim START_SIM
                            Starting sim number to run

This and other scripts uses python's `multiprocessing`_ module to run multiple independent
simulations on multiple cores simultaneously. The number of workers defaults to the number
of cores available, although this can be specified via the ``-w`` flag. The ``--start-sim``
command line argument offsets the simulation identifier.

The command used for the paper was::

    $ python bin/run_bruteforce.py -n 10 -w 6

Since these and other simulations can take a long time to complete, it is advisable 
to use ``nohup``. This creates a directory ``bruteforce_{BETA}/`` containing all of the outputs,
where *BETA* is the inverse temperature.

Running Weighted Ensemble simulations
-------------------------------------

To run the Weighted Ensemble simulations, use the script, ``bin/run_we.py``

::

    $ python bin/run_we.py -h
    usage: run_we.py [-h] [-s NSIMS] -c CONFIG_FILE [CONFIG_FILE ...] -n
                     [NAME [NAME ...]] [-p [PROTOCOLS [PROTOCOLS ...]]]
                     [-w NWORKERS] [--sid_offset SID_OFFSET] [--no-run]

    WEST run script

    optional arguments:
      -h, --help            show this help message and exit
      -s NSIMS              number of simulations to run
      -c CONFIG_FILE [CONFIG_FILE ...]
                            yaml config file name
      -n [NAME [NAME ...]]  simulation name to run
      -p [PROTOCOLS [PROTOCOLS ...]]
                            protocols to run; by default run all
      -w NWORKERS           number of cores to use
      --start-sim START_SIM
                            starting sim number to run
      --no-run              Only setup simulations but do not run them

This tool is similar to ``run_bruteforce.py``, but adds two additional command line arguments;
The ``-p`` flag specifies which *protocols* to run (DEFAULT: run all), and ``--no-run`` sets up all of the simulation
directories and scripts, but does not launch the actual simulations. The protocols are specified
in the yaml configuration file in ``configs`` and the are selected by the ``name`` variable.

The command used for the paper was::

    $ python bin/run_we.py -s 10 -c configs/we_{BETA}_run_config.yaml -n we_{BETA} -w 6

to run the Weighted Ensemble simulations at temperature *BETA* equals ``1.0``, ``1.5``, ``2.0``, ``2.5`` and ``3.0``. 

Analyzing data
--------------

To analyze the Weighted Ensemble simulations, use the script, ``bin/run_we_analysis.py``

::

    $ python bin/run_we_analysis.py -h
    usage: run_we_analysis.py [-h] -c CONFIG_FILE [CONFIG_FILE ...]
                              [-n [NAME [NAME ...]]] [-w NWORKERS] -s
                              {calculate_rate,calculate_distribution,get_pcoords,get_strings,get_nreplicas,all}

    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG_FILE [CONFIG_FILE ...]
                            yaml config file name
      -n [NAME [NAME ...]]  simulation name to run; by default run all
      -w NWORKERS           number of cores to use
      -s {calculate_rate,calculate_distribution,get_pcoords,get_strings,get_nreplicas,all}
                            analysis script to run

This script runs the analysis tools found in the ``analysis`` directory. The command used for the paper was::

    $ python bin/run_we_analysis.py -c configs/we_{BETA}_run_config.yaml -n we_{BETA} -w 3 -s all

These scripts generate a number of ``.h5`` files in ``we_{BETA}/analysis/{sim_id}`` containing the calculated quantities 
used to create the figures.
    
Generating figures
------------------

To generate the figures that appear in the manuscript after running all of the simulations and the scripts
that analyze them, run::

    $ cd generate_figures
    $ python distributions.py
    $ python error_plot.py
    $ python potential_string.py
    $ python rate_order_mag.py

This will create one *.eps* file per figure.


.. LINKS

.. _`multiprocessing`: http://docs.python.org/2/library/multiprocessing.html





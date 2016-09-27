=====================
Elastic Network Model
=====================

Double-well Elastic Network model of NtrC based on the parameterization in

    E. Vanden-Eijnden and M. Venturoli, “Revisiting the finite temperature string method for the calculation 
    of reaction tubes and free energies,” J Chem Phys 130, 194103 (2009).


Running Weighted Ensemble simulations
-------------------------------------

To run the Weighted Ensemble simulation, use the script, ``bin/run_we.py``

::

    $ python bin/run_we.py -h
    usage: run_we.py [-h] -c CONFIG_FILE -n NAME [-p [PROTOCOLS [PROTOCOLS ...]]]
                     [-w NWORKERS] [--profile] [--timeout TIMEOUT]

    WEST run script

    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG_FILE        yaml config file name
      -n NAME               simulation name to run
      -p [PROTOCOLS [PROTOCOLS ...]]
                            protocols to run; by default run all
      -w NWORKERS           number of cores to use
      --timeout TIMEOUT     timeout when running with zmq work manager
      --profile             Profile code
      --no-run              Only setup simulations but do not run them

The ``-p`` flag specifies which *protocols* to run (DEFAULT: run all), and ``--no-run`` sets up the simulation
directorie and scripts, but does not launch the actual simulation. The protocols are specified
in the yaml configuration file in ``configs`` and the are selected by the ``name`` variable.
When running with more than one worker, the simulation is run with the zmq work manager, and the ``--timeout``
flag should be set. Since the re-weighting step can be expensive the *timeout* parameter should be set between 1800
and 3600 seconds, although this will depend on the number of cores available for the calculation and the processor
speed.

The command used for the paper was::

    $ python bin/run_we.py -c configs/we_run_config_freestr.yaml -n we_freestr -w 6 --timeout 3600

Since this calculation can take some time, ``nohup`` should probably be used.

.. NOTE::

    The ``propagator_block_size`` in ``we_run_config_freestr.yaml``, which sets the ``block_size`` parameter in the configuration file,
    should be adjusted so that ``(nbins * target_count) / propagator_block_size >= nworkers`` to ensure proper scaling of the simulation
    when varying the number of workers or the total number of replicas when the bin space is fully populated.

Analyzing data
--------------

To analyze the Weighted Ensemble simulations and calculate the probability distribution based on the final string
configuration, use the script, ``analysis/get_bin_populations_all.py``

::

    $WEST_ROOT/bin/west analysis/get_bin_populations_all.py -h
    Entering Python shell using python2.7
    -----------------------
    get_bin_populations_all.py
    -----------------------
    usage: get_bin_populations_all [-h] [-r RCFILE]
                                   [--quiet | --verbose | --debug] [--version]
                                   [-o H5OUT]

    Retrieve strings from west.h5 file and write them to new file

    optional arguments:
      -h, --help            show this help message and exit
      -o H5OUT              name of output file

    general options:
      -r RCFILE, --rcfile RCFILE
                            use RCFILE as the WEST run-time configuration file
                            (default: west.cfg)

Run this script as::

    $ cd we_freestr
    $ $WEST_ROOT/bin/west ../analysis/get_bin_populations_all.py -o bin_populations_all.h5 -r we_phase1.cfg 


Generating figures
------------------

To generate the figures that appear in the manuscript after running all of the simulations and the scripts
that analyze them, run::

    $ cd generate_figures
    $ python plot_pmf.py

This will create one *.eps* file per figure.


.. LINKS

.. _`multiprocessing`: http://docs.python.org/2/library/multiprocessing.html





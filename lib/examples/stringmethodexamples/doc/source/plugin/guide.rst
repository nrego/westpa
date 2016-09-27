Activating the plugin
---------------------

To use the `WESTPA`_ string method plugin, first activate it in the ``west.cfg`` file, as in the 
following example::

    ---
    west: 
        ...
        plugins:
            - plugin: westext.stringmethod.StringDriver
              string_method: system.PeriodicLinkedStringMethod
              avgpos_method: system.average_position
              dfunc_method: system.dfunc
              do_update: True
              windowsize: 100
              update_interval: 10
              initial_update: 10
              priority: 1

The ``plugin`` block specific to the string method, must contain each of the following elements:

=====================================================================   =====================================================================
Setting name (default value)                                            What does it do?
=====================================================================   =====================================================================
``plugin``                                                              Defines that the string method should be used. This should be set to 
                                                                        ``westext.stringmethod.StringDriver``. 
``string_method``                                                       Should be set to ``default`` or a custom subclass of the ``WESTStringMethod``
                                                                        class.
``avgpos_method``                                                       Method defining how the average position in a bin is to be calculated. 
                                                                        This should be set to ``cartesian`` or a custom method.
``dfunc_method``                                                        Custom method defining how the ``VoronoiBinMapper`` should calculate the
                                                                        distance between a test point in progress coordinates space and all of 
                                                                        the bin generators.
``do_update`` (``True``)                                                Update the string during an iteration corresponding to the ``update_interval``.
``update_interval`` (``10``)                                            The number of iterations of the WE method between string updates.
``initial_update`` (``20``)                                             First iteration to attempt string update.
``windowsize`` (``10``)                                                 The number of iterations previous to the current iteration that should be 
                                                                        considered when calculating the average position of replicas within each bin.
``init_from_data`` (``True``)                                           Use the last stored string definition to initialize the string. Otherwise
                                                                        use the definition in the ``WESTSystem.initialize`` method.
``priority`` (``0``)                                                    If multiple plugins are specified for the same callback hook, execute them
                                                                        in ascending order based on the ``priority``.
=====================================================================   =====================================================================

For the parameters of the plugin listed above, it is often convenient to define any custom methods within the ``system.py`` file as is shown in 
the example.

String Method parameters
------------------------

While the string method plugin defines a generic interface for performing a string calculation in `WESTPA`_, most 
users will likely use the ``DefaultStringMethod`` (specified by ``string_method: default``, or a custom subclasss of 
``DefaultStringMethod``. Regardless of the implementation, the plugin driver expects a that a `python dict`_ can be found
that is a instance variable of your ``system`` class, called ``system.sm_params``, as in the following example::

    class System(WESTSystem):

        def initialize(self):

            ...
            self.sm_params = {'slen': [slen, slen],
                              'kappa': 0.001,
                              'dtau': 0.15,
                              'fixed_ends': False,
                              'sciflag': True,
                              'mpairs': [[0, self.nbins - 1], [slen - 1, slen]],
                              'slabels': [2],
                              'fourierflag': True,
                              'fourier_P': 2}

The parameters available via the ``DefaultStringMethod`` are

=====================================================================   =====================================================================
Parameter name (default value)                                            What does it do?
=====================================================================   =====================================================================
``slen``                                                                An iterable containing the number of centers in each string.
``slabels`` (``None``)                                                  A list containing the relative positions in each string of any state label
                                                                        progress coordinates if present. These progress coordinates will be ignored in the
                                                                        calculation. ``None`` if no labels.
``mpairs`` (``None``)                                                   A list of lists containing the indices of pairs of centers that should move together.
                                                                        ``None`` if strings move independently. 
``dtau`` (``0.1``)                                                      Parameter controlling the rate at which centers move toward the average value in the bin.
``kappa`` (``0.1``)                                                     Parameter controlling the smoothing of the string.
``fixed_ends`` (``True``)                                               Boolean flag specifying whether to fix the ends of the strings.
``sciflag`` (``None``)                                                  Boolean flag specifying whether to attempt to use SciPy methods which are 
                                                                        generally more efficient. If ``None`` import SciPy modules if available. 
``fourierflag`` (``False``)                                             Boolean flag specifying whether to user fourier fitting method
``fourier_P`` (``2``)                                                   Integer value specifying how many fourier modes to use in fitting
``fourier_maxiters`` (``100``)                                          Maximum number of iterations of fourier fitting procedure
``fourier_tol`` (``1.0E-6``)                                            Tolerance for ending fourier fitting
=====================================================================   =====================================================================

If the ``dfunc`` method requires additional positional or keyword arguments, the plugin will take them from ``system.dfargs`` and 
``system.dfkwargs`` respectively when updating the ``BinMapper``. For example, if the ``dfunc`` method required a set of weights 
to calculate a weighted RMSD and a boolean flag parameter, with the call signature ``dfunc(p, centers, weights, flag=False)``, 
the system initialization might look like::

    class System(WESTSystem):

        def initialize(self):

            ...
            weights = ... # Define weight array
            df_flag = True

            self.dfargs = (weights, )
            self.dfkwargs = {'flag': df_flag}
            self.bin_mapper = VoronoiBinMapper(dfunc, centers, dfargs=self.dfargs, dfkwargs=self.dfkwargs)


.. LINKS

.. _`WESTPA`: http://chong.chem.pitt.edu/WESTPA
.. _`python dict`: http://docs.python.org/2/library/stdtypes.html#mapping-types-dict


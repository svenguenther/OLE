Sampler
===============================

Following settings are relevant for the sampler:


General Sampler Settings:

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``output_directory``
     - ``output``
     - Directory where the results are written to.
   * - ``force``
     - ``False``
     - Overwrites results (TODO). Currently, results are appended every time.
   * - ``covmat``
     - ``None``
     - File of parameter covmat. It is used for initial guess for the samplers. TODO: currently not for Minimizer
   * - ``nwalkers``
     - ``10``
     - Number of walkers in enselbe sampler
   * - ``compute_data_covmats``    
     - ``False``
     - If your likelihood is differentiable, you can compute the covmats of your data. This can help you with normalization. Can be instabel for high dimensional likelihoods.
   * - ``status_print_frequency``
     - ``100``
     - Frequency with which the status updates are printed.
   * - ``debug``
     - ``False``
     - If set to ``True`` the sampler will print out a lot of debugging information. This is very helpful when investigating a new problem.

Evaluate sampler (computes likelihood for a given parameter set):

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``use_emulator``
     - ``True``
     - Flag whether the emulator is to be used or not.
   * - ``return_uncertainty``
     - ``False``
     - Gives uncertainty estimate from emulator.
   * - ``logposterior``
     - ``False``
     - Flag whether we compute the logposterior or loglikelihood
   * - ``nsamples``
     - ``20``
     - Number of samples computed at this point to estimate the uncertainty
    


Minimize sampler (computes likelihood for a given parameter set):


.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``use_emulator``
     - ``True``
     - Flag whether the emulator is to be used or not.
   * - ``use_gradients``
     - ``True``
     - Flag to indicate if we want to use gradients for the minimization (only for differentiable likelihoods).
   * - ``logposterior``
     - ``False``
     - Flag whether we compute the logposterior or loglikelihood
   * - ``method``
     - ``TNC``
     - Minimization method. You can select any of scipy.optimize.minimize


NUTS sampler (computes likelihood for a given parameter set):

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``nwalkers``
     - ``10``
     - Number of walkers in Mestropolis hastings sampler in the early stage of the emulator before the emulator is trained. More walkers increase the variety in the training data set
   * - ``target_acceptance``
     - ``0.5``
     - Target acceptance of NUTS
   * - ``M_adapt``
     - ``200``
     - Number of steps until stepsize is fixed
   * - ``delta_max``
     - ``1000``
     - NUTS parameter
   * - ``minimize_nuisance_parameters``
     - ``True``
     - This flag indicates that during the training stage, the nuisance parameters are fitted. This allows faster acceptance of data points in particular for high dimensional nuisance parameter space.


Cobaya:

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``cobaya_state_file``
     - ``None``
     - Path to pickled file which stores a cobaya state. If set, it will be either created if the file does not exist or loaded when it does exist. If this file exists the emulator can be build without running the theory code.

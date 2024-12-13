Emulator
=================================================

Note that some parameters are SHARED in different parts of the code.


Data collection and cache. The cache is shared between all chains and processes.
These settings are independent of the sampling method. 


.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``cache_size``
     - ``1000``
     - Maximum size of stored training data points. If more data points are to be added, the one with the smallest loglikelihood is removed.
   * - ``cache_file``
     - ``cache.pkl``
     - File in which the cache is going to be stored in.
   * - ``shared_cache``
     - ``True``
     - If this flag is set to ``True`` the cache is shared between all chains and processes. This is useful if you want to run the sampler in parallel. If set to ``False`` each chain has its own cache. Shared cache allows for faster training of the emulator. However, biases of the emulator can be shared between chains. If each emulator uses its own cache, this can lead to a minimal R-1 (due to the emulation bias). Actually, thts a nice estimate of the emulation bias :)
   * - ``load_cache``
     - ``False``
     - If set ``True``, the cache of a previous run is loaded. Note that if the likelihood is changed, this can corrupt your cache leading to bugs! Thus, if you change the theory or likelihood code, always create a new cache or set this flag to ``False``. In this case the old cache file will be overwritten.
   * - ``delta_loglike``
     - ``50.0``
     - This parameter discriminates between relevant data points for the cache or outliers. Therefore, all states in the cache with a loglikelihood smaller that the maximum loglikelihood in the cache minus ``delta_loglike`` are to be removed since they are classified as outlier. If ``N_sigma`` and ``dimensionality`` is set, this parameter is omitted. In general it is better to give ``N_sigma`` and ``dimensionality``!
   * - ``dimensionality``
     - ``None``
     - As an alternative to the ``delta_loglike`` we can compute an educated guess for this parameter by computing the delta loglike of a gaussian distribution of dimension ``dimenstionality`` from its best fit point to ``N_sigma``. Thus, if the posterior would be gaussian, points in the cache would lay inside a ``N_sigma`` contour but all points outside would be classifies as outlier. If no ``dimenstionality`` is given ``delta_loglike`` is used. Imporant for eficiency!
   * - ``N_sigma``
     - ``4.0``
     - See ``dimensionality``. Important parameter for efficiency.


These parameters are used to specify the PCA compression of the data.

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``min_variance_per_bin``
     - ``1e-4``
     - The level of compression of each observable is determined by the number of PCA components. Therefore, we increase the number of PCA components until the explained variance per bin times the bin size exceeds the parameters value. The value of ``1e-4`` can be interpreted in a way that for each observable the systematic uncertainty due to insufficient projection of the PCA will lead to a relative error (of the normalized observables) of ``1e-2``. Thus, it is a maximal achievable precision of the emulator. If it is selected too large an error message appears that indicates possible biases. Here we can directly trade between speed and accuracy. For highly correlated quantities it is adviseable to reduce this number by 1-2 magnitudes! This is an important parameter.
   * - ``max_output_dimensions``
     - ``30``
     - The maximal number of PCA components. Unlikely to exceed that
   * - ``data_covmat_directory``
     - ``None``
     - We can provide the emulator with a dictionary of data covmats (keys are the names of the observables). They can be either the full (2-dimensional) covariance matrix or the (1-dimensional) diagonal of the covariance matrix. These covariance matrices are used to normalize the data. This is particular helpful to indicate the emulator which parts of the observable have to be computed precisesly and which parts have only a low significance for the total likelihood. If no covariance matrices are provided, the normalization is performed bin wise and the code assumes the entire range of the output to be of same relevance for the total likelihood.
   * - ``normalize_by_full_covmat``
     - ``False``
     - If the flag is set to true, we normalize the observables by the full covariance, thus, go into the data eigenspace. This is already partly that what the PCA is supposed to do. It can be computationally expensive for high dimensional observables.



Following parameters are used to specify the training of the GP and when it is supposed to happen.
It also deals with the possible compression of data by sparse GPs.


.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``kernel``
     - ``RBF``
     - GP Kernel. Currently implemented: [RBF] In fact it is a RBF + linear + WhiteNoise kernel.
   * - ``learning_rate``
     - ``0.1``
     - Learning rate for ADAM optimizer when fitting the GP parameters. Note that sparse GP typically require a smaller learning rate than ordinary ones
   * - ``num_iters``
     - ``100``
     - Proposed number of training epochs. If we see that the loss is still falling (more than ``early_stopping`` within two batches of ``early_stopping_window`` iterations)
   * - ``max_num_iters``
     - ``400``
     - Maximal training epochs if early stopping is not triggered. Should not be reached. Produces a warning when exceeded!
   * - ``early_stopping``
     - ``0.05``
     - Early stopping criterium. See ``num_iters``.
   * - ``early_stopping_window``
     - ``10``
     - Window for early stopping. See ``num_iters``.
   * - ``min_data_points``
     - ``80``
     - Number of minimal states in the cache before the emulator can be trained. This is an important parameter. If it is selected too small, the emulator will require too many retrainings. If too large, the initial data collection phase of OLE is unnecessary long.
   * - ``kernel_fitting_frequency``
     - ``20``
     - Frequency of how many new data points are added to the cache until a new compression is computed and the parameters of the GP are fitted again. Since this step is rather computational expensive we do not want to refit every step. Note however, that every new point in the cache will be utilized in the prediction even if the kernels are not refitted!
   * - ``sparse_GP_points``
     - ``0``
     - If not set to ``0`` we try to use condensate the information of all training points into a reduced training set (sparse GPs). The initial guess of the number of estimated sparse data points is ``sparse_GP_points``. However, in the iterative search for the best number of data points there is a certain error tolerance that we deem acceptable for the acceleration. It should be choosen rather small as the subleading PCA components can be fit with very few data points.
   * - ``white_noise_ratio``
     - ``1.``
     - If not set to ``0`` a noise term is added to the Kernel that is determined by the ``explained_variance_cutoff`` for each PCA component. This prevents the GP from fitting random noise introduced in the PCA analysis. It is also a central component of the sparse GP method since it is used to determine the optimal number of sparse points. A value of one sets the white noise error such that is comparable to the dropped PCA components
   * - ``error_boost`` 
     - ``2.``
     - This parameter allocates a noise budget to the sparse GP relative to the existing white noise term. A value of 2. means that the total allowed error is twice the white noise and thus the average error of the sparse GP may be as large as the white noise term. A value of 1. means that the sparse GP error is zero, so it can never be used. Reasonable values are between 1.5 and 5. 
    

Uncertainty qualification related to the precision criterium of the emulator and when to test it.


.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``testing_strategy``
     - ``'test_stochastic'``
     - Specify testing strategy. Possible stragies: ``'test_all','test_early','test_none','test_stochastic'``. When ``'test_all'`` is selected each emulator call will be tested. When ``'test_none'`` is selected none emulator call will be tested. If ``'test_early'`` is selected we test all points until we tested ``test_early_points`` consecutive points positive. 
     Afterwards we turn off the testing. ``test_stochastik`` starts with a 100% testing probability. However, the chance of testing will exponentially decrease with the number of consecutive successful emulator calls. The scale of the ``test_stochastic_scale`` times ``dimensionality`` is the scale of the exponential decrease. 
     If ``test_stochastic_rate`` is set, even after the exponential decay we will test at least with a ``test_stochastic_rate`` the points. 
     If it is not set, it will be determined by ``test_stochastic_testing_time_fraction``. In this case, the time for testing and the actual emulator call is balanced, such that the testing time is a fraction of the total time.
   * - ``test_early_points``
     - ``1000``
     - Number of consective positive test calls until testing is switched off. See ``testing_strategy``
   * - ``test_stochastic_scale``
     - ``20``
     - Scale of each dimension for the stochastik testing. See ``testing_strategy``.
   * - ``test_stochastic_rate``
     - ``None``
     - See ``testing_strategy``.
   * - ``test_stochastic_testing_time_fraction``
     - ``0.1``
     - See ``testing_strategy``.
   * - ``N_quality_samples``   
     - ``5``
     - Number of samples which are drawn from the emulator to estimate the performance of the emulator. The runtime is about linear in that parameter! From this number of samples we compute the mean loglikelihood $m$  and its standard deviation $\sigma_m$. In general we want the emulator to be very precise at the best fit point with its loglikelihood $b$ and less accurate for points more away. We accept the prediction of the emulator if $\sigma_m < \mathrm{quality.threshold.constant} +  \mathrm{quality.threshold.linear}*(b-m) +  \mathrm{quality.threshold.quadratic} * (b-m)^2 $
   * - ``DEPRECATED: tail_cut_fraction``   
     - ``0.2``
     - The distribution of the sampled loglikes is found to be non-gaussian. However, by cutting the low end tail, we find sufficient estimates for the mean and standard deviation. This parameter specifies the fraction of the tail that is cut. 
   * - ``quality_threshold_constant``
     - ``0.1``
     - See ``N_quality_samples``
   * - ``quality_threshold_linear``
     - ``0.05``
     - See ``N_quality_samples``. Note that this factor can be reformulated in a precision criterium of your confidence bounds (for a gaussian distribution). If we set this factor to ``0.01`` the emulator can estimate the position of the N sigma contour to a precision of ``N*0.01``.
   * - ``quality_threshold_quadratic``
     - ``0.0001``
     - See ``N_quality_samples``. In general we want the quadratic term to be state the absolute ignorance outside the relevant parameter space. To provide you with a better handle this parameter is overwritten if one provides values for ``dimensionality`` and ``N_sigma``. In this case, the contribution of ``quality_threshold_quadratic`` starts to dominate over the constant and linear term exactly at ``N_sigma``.
   * - ``quality_points_radius``
     - ``0.0``
     - One way to reduce the number of performance tests is to create a sphere around each tested emulator call and whenever the emulator predicts the performance within a radius of ``quality_points_radius`` (in normalized units), no testing is required and the emulator can be used. If set to 0.0 ever call will be tested.



Other:

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``load_initial_state``
     - ``False``
     - If flag is set to ``True`` the state from which the emulator is initialized is loaded from an already existing cache file. Otherwise the emulator is initialized once the theory code was run for the first time. By setting this to ``True`` and setting ``test_emulator`` to ``False``, one can use the emulator without calling the theory code at all.
   * - ``skip_emulation_quantities``
     - ``None``
     - List of quantities that are provided by the theory code but which should not be emulated. As a consequence the output of the veto quantities will be constant with the value the emulator was initialized with.
   * - ``jit``
     - ``True``
     - Flag if we want to use 'jax.jit' to accelerate the emulator by just-in-time compilation.
   * - ``jit_threshold``
     - ``10``
     - Using 'jit' gives a small overhead due to compiling the code. In the early phase when there are a lot of new data points it can be ineffcient to do that every time. Thus, we can wait for a certain number of successful emulator calls until we jit the emulator.


Debugging. Very recommended when investigating a new problem:

.. list-table:: Title
   :widths: 10 10 50
   :header-rows: 1

   * - parameter
     - default
     - description
   * - ``plotting_directory``
     - ``None``
     - Path to a directory in which (if set) debugging plots are saved to.
   * - ``testset_fraction``
     - ``None``
     - If set (for example ``0.1``) a certain fraction of the training samples will not be used for training but for testing the performance of the emulator. Additional plots will be created in the ``plotting_directory``
   * - ``logfile``
     - ``None``
     - If set to a text file, the emulator writes a log.   
   * - ``status_print_frequency``
     - ``200``
     - Every ``status_print_frequency`` runs the status of the emulator will be printed.   
   * - ``debug``
     - ``False``
     - If set to ``True`` the emulator will print out a lot of debugging information. This is very helpful when investigating a new problem.
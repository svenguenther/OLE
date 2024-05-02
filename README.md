# Online Learning Émulator
The Online Learning Émulator - OLÉ is a framework to efficiently perform statistical analyeses in cases where one can distinguish between a Theory (simulation) code in which observables are computed and likelihood codes that compute a likelihood for a given computed observable. The efficiency comes from emulating the computationally expensive theory codes with 1-O(20) parameters. 

The required training sample for the emulator is gathered during the inference process (online learning) to ensure a coverage of the relevant parameter space but not of unintesting domains. Additionally this emulator provides the user with an uncertainty estimate of the given emulation call. As a consequence we can use an active sampling algorithm that only adds new training points when the accuracy is insufficient.

This implementation involves both the emulator that can be used independently (also with cobaya interface) and a collection of sampler which includes Ensemble/Minimizer/NUTS - Sampler. The minimizer and NUTS benefit from a differentiable likelihood.
## Installing 
```
git clone git@github.com:svenguenther/OLE.git
pip install ./OLE
```

## Documentation
There are some test examples in $\texttt{OLE/examples}$. 



### Emulator settings
Note that some parameters are SHARED in different parts of the code.

Data collection:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```cache_size``` | ```1000``` | Maximum size of stored training data points. If more data points are to be added, the one with the smallest loglikelihood is removed. |
| ```cache_file``` | ```cache.pkl``` | File in which the cache is going to be stored in. |
| ```load_cache``` | ```True``` | If set `True`, the cache of a previous run is loaded. Note that if the likelihood is changed, this can corrupt your cache leading to bugs! Thus, if you change the theory or likelihood code, always create a new cache or set this flag to ```False```. In this case the old cache file will be overwritten. |
| ```delta_loglike``` | ```100.0``` | This parameter discriminates between relevant data points for the cache or outliers. Therefore, all states in the cache with a loglikelihood smaller that the maximum loglikelihood in the cache minus ```delta_loglike``` are to be removed since they are classified as outlier. |
| ```dimensionality``` | ```None``` | As an alternative to the ```delta_loglike``` we can compute an educated guess for this parameter by computing the delta loglike of a gaussian distribution of dimension ```dimenstionality``` from its best fit point to ```N_sigma```. Thus, if the posterior would be gaussian, points in the cache would lay inside a ```N_sigma``` contour but all points outside would be classifies as outlier. If no ```dimenstionality``` is given ```delta_loglike``` is used. |
| ```N_sigma``` | ```6.0``` | See ```dimensionality``` |


Normalization:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```explained_variance_cutoff``` | ```0.9999``` | The level of compression of each observable is determined by the number of PCA components. Therefore, we increase the number of PCA components until the explained variance exceeds that value |
| ```max_output_dimensions``` | ```30``` | The maximal number of PCA components. |
| ```data_covmat_directory``` | ```None``` | We can provide the emulator with a dictionary of data covmats (keys are the names of the observables). They can be either the full (2-dimensional) covariance matrix or the (1-dimensional) diagonal of the covariance matrix. These covariance matrices are used to normalize the data. This is particular helpful to indicate the emulator which parts of the observable have to be computed precisesely and which parts have only a low significance for the total likelihood. If no covariance matrices are provided, the normalization is performed bin wise and the code assumes the entire range of the output to be of same relevance for the total likelihood. |
| ```normalize_by_full_covmat``` | ```False``` | If the flag is set to true, we normalize the observables by the full covariance, thus, go into the data eigenspace. This is already partly that what the PCA is supposed to do. It can be computationally expensive for high dimensional observables. |


Training:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```kernel``` | ```RBF``` | GP Kernel. Currently implemented: [RBF] |
| ```learning_rate``` | ```0.02``` | Learning rate for ADAM optimizer when fitting the GP parameters |
| ```num_iters``` | ```100``` | Number of training epochs. |
| ```min_data_points``` | ```80``` | Number of minimal states in the cache before the emulator can be trained. |
| ```kernel_fitting_frequency``` | ```20``` | Frequency of how many new data points are added to the cache until a new compression is computed and the parameters of the GP are fitted again. Since this step is rather computational expensive we do not want to refit every step. Note however, that every new point in the cache will be utilized in the prediction even if the kernels are not refitted! |
| ```sparse_GP_points``` | ```0``` | If not set to ```0``` we try to use condensate the information of all training points into a reduced training set (sparse GPs). The initial guess of the number of estimated sparse data points is ```sparse_GP_points```. However, in the iterative search for the best number of data points there is a certain error tolerance that we deem acceptable for the acceleration. |
| ```noise_percentage``` | ```0.0``` | Sets the noise percentage for the error tolerance for computing sparse GPs. (see ```sparse_GP_points```). |
| ```error_tolerance``` | ```0.1``` | Default value for error tolerance for sparse GPs (see ```sparse_GP_points```). |


Uncertainty qualification:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```test_emulator``` | ```True``` | Flag on whether the emulator should test its performance. |
| ```N_quality_samples``` | ```5``` | Number of samples which are drawn from the emulator to estimate the performance of the emulator. The runtime is about linear in that parameter! From this number of samples we compute the mean loglikelihood $m$  and its standard deviation $\sigma_m$. In general we want the emulator to be very precise at the best fit point with its loglikelihood $b$ and less accurate for points more away. We accept the prediction of the emulator if $\sigma_m < \mathrm{quality.threshold.constant} +  \mathrm{quality.threshold.linear}*(b-m) +  \mathrm{quality.threshold.quadratic} * (b-m)^2 $ |
| ```quality_threshold_constant``` | ```0.1``` | See ```N_quality_samples``` |
| ```quality_threshold_linear``` | ```0.01``` | See ```N_quality_samples```. Note that this factor can be reformulated in a precision criterium of your confidence bounds (for a gaussian distribution). If we set this factor to ```0.01``` the emulator can estimate the position of the N sigma contour to a precision of ```N*0.01```. |
| ```quality_threshold_quadratic``` | ```0.0001``` | See ```N_quality_samples```. In general we want the quadratic term to be state the absolute ignorance outside the relevant parameter space. To provide you with a better handle this parameter is overwritten if one provides values for ```dimensionality``` and ```N_sigma```. In this case, the contribution of ```quality_threshold_quadratic``` starts to dominate over the constant and linear term exactly at ```N_sigma```.  |
| ```quality_points_radius``` | ```0.0``` | One way to reduce the number of performance tests is to create a sphere around each tested emulator call and whenever the emulator predicts the performance within a radius of ```quality_points_radius``` (in normalized units), no testing is required and the emulator can be used. If set to 0.0 ever call will be tested. |


Other:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```load_initial_state``` | ```False``` | If flag is set to ```True``` the state from which the emulator is initialized is loaded from an already existing cache file. Otherwise the emulator is initialized once the theory code was run for the first time. By setting this to ```True``` and setting ```test_emulator``` to ```False```, one can use the emulator without calling the theory code at all. |
| ```veto_list``` | ```None``` | List of quantities that are provided by the theory code but which should not be emulated. As a consequence the output of the veto quantities will be constant with the value the emulator was initialized with. |


Debugging:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```plotting_directory``` | ```None``` | Path to a directory in which (if set) debugging plots are saved to. |
| ```testset_fraction``` | ```None``` | If set (for example ```0.1```) a certain fraction of the training samples will not be used for training but for testing the performance of the emulator. Additional plots will be created in the ```plotting_directory``` |
| ```logfile``` | ```None``` | If set to a text file, the emulator writes a log. |


### Sampler settings
Folloring settings are relevant for the sampler:


Sampler:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```output_directory``` | ```output``` | Directory where the results are written to. |
| ```force``` | ```False``` | Overwrites results (TODO) |
| ```covmat``` | ```None``` | File of parameter covmat. It is used for initial guess for the samplers. TODO: currently not for Minimizer |
| ```nwalkers``` | ```10``` | Number of walkers in enselbe sampler |
| ```compute_data_covmats``` | ```False``` | If your likelihood is differentiable, you can compute the covmats of your data. This can help you with normalization. TODO: use this for your PCA. |



Evaluate sampler (computes likelihood for a given parameter set):
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```use_emulator``` | ```True``` | Flag whether the emulator is to be used or not. |
| ```return_uncertainty``` | ```False``` | Gives uncertainty estimate from emulator. |
| ```logposterior``` | ```False``` | Flag whether we compute the logposterior or loglikelihood |
| ```nsamples``` | ```20``` | Number of samples computed at this point to estimate the uncertainty |


Minimize sampler (computes likelihood for a given parameter set):
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```use_emulator``` | ```True``` | Flag whether the emulator is to be used or not. |
| ```use_gradients``` | ```True``` | Flag to indicate if we want to use gradients for the minimization (only for differentiable likelihoods). |
| ```logposterior``` | ```False``` | Flag whether we compute the logposterior or loglikelihood |
| ```method``` | ```L-BFGS-B``` | Minimization method. You can select any of scipy.optimize.minimize |


NUTS sampler (computes likelihood for a given parameter set):
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```nwalkers``` | ```10``` | Number of walkers in enselbe sampler in the early stage of the emulator before the emulator is trained|
| ```target_acceptance``` | ```0.5``` | Target acceptance of NUTS |
| ```M_adapt``` | ```1000``` | Number of steps until stepsize is fixed |
| ```delta_max``` | ```1000``` | NUTS parameter |


Cobaya:
| parameter   | default    | description       |
| :---    | :---   | :---     |
| ```cobaya_state_file``` | ```None``` | Path to pickled file which stores a cobaya state. If set, it will be either created if the file does not exist or loaded when it does exist. If this file exists the emulator can be build without running the theory code. |
| ```jit_threshold``` | ```10``` | The emulator will be used this number of times in cobaya before it will be jitted. If the accuracy of the emulator was not sufficient and the emulator is to be updated the counter is set back to 0. This should help to reduce wasting time in jitting the emulator in the early stage of the inference task when it is not still |




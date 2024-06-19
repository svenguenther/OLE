# This example script is a NUTS sampler using the OLE emulator for doing parameter estiamtion using the candl likelihood.

# The script has the following structure:

# 1. Define a theory which computes Cls. It has 2 functions: initialize and compute.
#    The initialize function sets the requirements of the theory.
#    The compute function computes the observables for the given parameters. The input is a dictionary with the parameters and the output is a dictionary with the observables.
#    It has the attributes 'requirements' which states the parameters it requires to compute the observables.
# 2. Define a likelihood. It has 2 functions: initialize and loglike.
#    The initialize function sets the requirements of the likelihood.
#    The loglike function computes the loglikelihood for the given parameters.
#    We use the candl likelihood. Be aware that it can give nan values for some parameters. In this case, we set the loglike to -1e100. This is not diffbar. TODO: Fix that
# 3. Define the settings for the emulator.
# 4. Define the sampler. It has 2 functions: initialize and run_mcmc.
#    The initialize function sets the theory, likelihood, parameters and emulator settings.
#    The run_mcmc function runs the mcmc. It has the number of steps as input.
# 5. Define the parameters. It is a dictionary with the parameters and their priors.
# 6. Run the mcmc.











from jax import config
# config.update("jax_debug_nans", True)


from OLE.theory import Theory
from OLE.likelihood import Likelihood
import jax.numpy as jnp
import time
import jax

import classy

import numpy as np

from functools import partial


# append parent directory to path
import candl
import candl.data

from OLE.theories.CLASS import CLASS
from OLE.likelihoods.cosmo.candl import candl_likelihood

# Init theory and likelihood
my_theory = CLASS()
my_likelihood = candl_likelihood()



emulator_settings = {
    # the number of data points in cache before the emulator is to be trained
    'min_data_points': 80,

    # name of the cache file
    'cache_file': './output_clang_spt_nuts/cache.pkl',

    # accuracy parameters for loglike:
    'quality_threshold_constant': 1.0,
    'quality_threshold_linear': 0.1,

    # related so sampler
    'explained_variance_cutoff': 0.9999,

    # cache criteria
    'dimensionality': 39,
    'N_sigma': 5.0,

    # M adapt # burn-in of NUTS
    'M_adapt': 200,

    # 'plotting_directory': './plots_spt_clang_nuts',
    # 'testset_fraction': 0.1,
    'logfile': './output_clang_spt_nuts/log.txt',

    'learning_rate': 0.1,
    'num_iters': 300,

}


likelihood_settings = {
    'candl_dataset': 'candl.data.SPT3G_2018_TTTEEE',
    'clear_priors': False,
}

theory_settings = {
    # here we could add some class settings
}

sampling_settings = {
    # output directory
    'output_directory': './output_clang_spt_nuts',

    'nwalkers':1,

    # 'compute_data_covmat': True,
    'data_covmat_directory': './spt_data_covmats',

    # M adapt # burn-in of NUTS
    'M_adapt': 200,
}



# load sampler 
from OLE.sampler import EnsembleSampler, Sampler, NUTSSampler
my_sampler = NUTSSampler(debug=False)
# my_sampler = EnsembleSampler(debug=False)


my_parameters = {'h': {'prior': {'min': 0.60, 'max': 0.80}, 
                       'ref': {'mean': 0.68, 'std': 0.01},
                       'proposal': 0.01,},
                    'n_s': {'prior': {'min': 0.9, 'max': 1.1}, 'ref': {'mean': 0.965, 'std': 0.005},
                            'proposal': 0.01,}, 
                    'omega_b': {'prior': {'min': 0.02, 'max': 0.024}, 'ref': {'mean': 0.0223, 'std': 0.0003},
                                'proposal': 0.0002,}, 
                    'omega_cdm': {'prior': {'min': 0.10, 'max': 0.14}, 'ref': {'mean': 0.120, 'std': 0.002},
                                  'proposal': 0.002,}, 
                    'tau_reio': {'prior': {'min': 0.01, 'max': 0.1}, 'ref': {'mean': 0.055, 'std': 0.01},
                                 'proposal': 0.01,},
                    'logA': {'prior': {'min': 2.8, 'max': 3.3}, 'ref': {'mean': 3.1, 'std': 0.05},
                             'proposal': 0.05},

                    # default SPT3G_2018_TTTEEE nuisance parameters are set automaticially. If you want to overwrite them you can state them here nonetheless.

                    # 'Kappa': {'prior': {'min': -0.01, 'max': 0.01, 'mean': 0.0, 'std': 0.00045}, 
                    #           'ref': {'mean': 0.0, 'std': 0.00045}, 
                    #           'proposal': 0.00045},

}

covmat_path = None#'./covmat_candl.txt'


start = time.time()

my_sampler.initialize(theory=my_theory, 
                      likelihood=my_likelihood, 
                      parameters=my_parameters, 
                      covmat = covmat_path, 
                      emulator_settings = emulator_settings,
                      likelihood_settings = likelihood_settings,
                      theory_settings = theory_settings,
                      sampling_settings = sampling_settings,)

# Note the total run steps are   (nsteps * nwalkers * MPI_size)
n_steps = 1000
my_sampler.run_mcmc(n_steps)

end = time.time()
print("Time elapsed: ", end - start)

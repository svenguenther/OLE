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

import time

from OLE.theories.CLASS import CLASS
from OLE.likelihoods.cosmo.candl import candl_likelihood
from OLE.likelihoods.cosmo.bao.sdss_dr12_consensus_bao import sdss_dr12_consensus_bao

# Init theory and likelihood
my_theory = CLASS()

my_candl = candl_likelihood()
my_bao = sdss_dr12_consensus_bao()
my_likelihood_collection = {'candl': my_candl, 'bao': my_bao}


emulator_settings = {
    # the number of data points in cache before the emulator is to be trained
    'min_data_points': 30,

    # name of the cache file
    'cache_file': './output_clang_act_bao_nuts/cache.pkl',
    # load the cache from previous runs if possible. If set to false, the cache is overwritten.
    'load_cache': True,

    # accuracy parameters for loglike:
    'quality_threshold_constant': 1.0,
    'quality_threshold_linear': 0.1,

    # related so sampler
    'min_variance_per_bin': 1e-4,
    'num_iters': 300,

    # cache criteria
    'dimensionality': 7,
    'N_sigma': 4.0,

    # 'plotting_directory': './output_clang_act_bao_nuts/plots_sampler_clang_nuts',
    # 'testset_fraction': 0.1,
    'logfile': './output_clang_act_bao_nuts/log',

    # 'compute_data_covmat': True,
    'data_covmat_directory': './act_data_covmats',

    # 'jit': False,

}

likelihood_settings = {
    'candl_dataset': 'candl.data.ACT_DR4_TTTEEE',
    'clear_priors': False,
}

my_likelihood_collection_settings = {'candl': likelihood_settings, 'bao': {}}

theory_settings = {
    # here we could add some class settings
    'parameters': ['h', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'logA'],

    # input parameters of the theory
    'class_settings': {'output':'tCl,pCl,lCl', 'lensing':'yes', 'N_ur':3.048, 'output_verbose':1},
}

sampling_settings = {
    # output directory
    'output_directory': './output_clang_act_bao_nuts',

    # M adapt # burn-in of NUTS
    'M_adapt': 200,
    'minimize_nuisance_parameters': True,
    # 'covmat': './fisher.covmat',
    # 'logfile': './output_clang_act_bao_nuts/log_sampler',

}


# load sampler 
from OLE.sampler import EnsembleSampler, Sampler, NUTSSampler
my_sampler = NUTSSampler()
# my_sampler = EnsembleSampler()


my_parameters = {'h': {'prior': {'min': 0.6, 'max': 0.8, 'type': 'uniform'},
                       'ref': {'mean': 0.68, 'std': 0.01},
                       'proposal': 0.01,},
                    'n_s': {'prior': {'min': 0.9, 'max': 1.1, 'type': 'uniform'}, 
                            'ref': {'mean': 0.965, 'std': 0.005},
                            'proposal': 0.01,}, 
                    'omega_b': {'prior': {'min': 0.02, 'max': 0.024, 'type': 'uniform'}, 
                                'ref': {'mean': 0.0223, 'std': 0.0003},
                                'proposal': 0.0002,}, 
                    'omega_cdm': {'prior': {'min': 0.10, 'max': 0.14, 'type': 'uniform'}, 
                                    'ref': {'mean': 0.120, 'std': 0.002},
                                    'proposal': 0.002,}, 
                    'tau_reio': {'prior': {'min': 0.01, 'max': 0.1, 'type': 'uniform'}, 
                                'ref': {'mean': 0.055, 'std': 0.01},
                                 'proposal': 0.01,},
                    'logA': {'prior': {'min': 2.8, 'max': 3.3, 'type': 'uniform'}, 
                                'ref': {'mean': 3.1, 'std': 0.05},
                             'proposal': 0.05},

                    # ACT_DR4_TTTEEE
                    # Nuisance parameters are laoded automaticially
                    'yp': {'prior': {'min': 0.9, 'max': 1.1, 'type': 'uniform'}, 'ref': {'mean': 1.0, 'std': 0.0001}, 'proposal': 0.0001},
}
    

start = time.time()

my_sampler.initialize(theory=my_theory, 
                      likelihood_collection=my_likelihood_collection, 
                      parameters=my_parameters, 
                      emulator_settings = emulator_settings,
                      likelihood_collection_settings = my_likelihood_collection_settings,
                      theory_settings = theory_settings,
                      sampling_settings = sampling_settings,
                    #   debug = True
                      )

# Note the total run steps are   (nsteps * nwalkers * MPI_size)
n_steps = 1000
my_sampler.run_mcmc(n_steps)

end = time.time()
print("Time elapsed: ", end - start)
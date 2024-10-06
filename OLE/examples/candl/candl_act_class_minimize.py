# example Minimize script

import time

from OLE.theories import CLASS, CAMB
from OLE.likelihoods.cosmo.candl import candl_likelihood


my_theory = CLASS()
# my_theory = CAMB()
my_likelihood = candl_likelihood()
my_likelihood_collection = {'candl': my_likelihood}

emulator_settings = {
    # the number of data points in cache before the emulator is to be trained
    'min_data_points': 80,

    # name of the cache file
    'cache_file': './candl_act_minimize/cache.pkl',
    # load the cache from previous runs if possible. If set to false, the cache is overwritten.
    'load_cache': True,

    # accuracy parameters for loglike:
    'quality_threshold_constant': 1.0,
    'quality_threshold_linear': 0.1,

    # related so sampler
    'min_variance_per_bin': 1e-6,
    'num_iters': 300,

    # cache criteria
    'dimensionality': 7,
    'N_sigma': 4.0,

    # 'plotting_directory': './output_clang_act_nuts/plots_sampler_clang_nuts',
    # 'testset_fraction': 0.1,
    'logfile': './candl_act_minimize/log',

    # 'compute_data_covmat': True,
    'data_covmat_directory': './act_data_covmats',
}

likelihood_settings = {
    'candl_dataset': 'candl.data.ACT_DR4_TTTEEE',
    'clear_priors': False,
}
my_likelihood_collection_settings = {'candl': likelihood_settings}

theory_settings = {
    # here we could add some class settings
    'parameters': ['h', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'logA'],

    # input parameters of the theory
    # 'cosmo_settings': {}, # some parameters for the theory
}

sampling_settings_NUTS = {
    # output directory
    'output_directory': './candl_act_minimize',
}

sampling_settings_minimize = {
    # output directory
    'output_directory': './candl_act_minimize',

    'use_emulator': True,    
    'logposterior': True,
    'use_gradients': True,
    'method': 'TNC',# scipy optimization method: 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'trust-ncg', 'trust-exact', 'trust-krylov'
    'n_restarts': 5,
}


# load sampler 
from OLE.sampler import EnsembleSampler, Sampler, NUTSSampler, MinimizeSampler
my_sampler_NUTS = NUTSSampler()
my_sampler_minimizer = MinimizeSampler()


my_parameters = {'h': {'prior': {'min': 0.6, 'max': 0.8, 'type': 'uniform'},
                       'ref': {'mean': 0.68, 'std': 0.01},
                       'proposal': 0.01,
                    #    'value': 0.68
                       },
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
}


start = time.time()

# run NUTS jsut to get some samples and train the emulator
my_sampler_NUTS.initialize(theory=my_theory,
                        likelihood_collection=my_likelihood_collection, 
                        parameters=my_parameters, 
                        emulator_settings = emulator_settings,
                        likelihood_collection_settings = my_likelihood_collection_settings,
                        theory_settings = theory_settings,
                        sampling_settings = sampling_settings_NUTS,
                        #   debug = True
                        )

my_sampler_NUTS.run_mcmc(nsteps = 10)



# now minimize
my_sampler_minimizer.initialize(theory=my_theory, 
                      likelihood_collection=my_likelihood_collection, 
                      parameters=my_parameters, 
                      emulator_settings = emulator_settings,
                      likelihood_collection_settings = my_likelihood_collection_settings,
                      theory_settings = theory_settings,
                      sampling_settings = sampling_settings_minimize,
                      emulator = my_sampler_NUTS.emulator,
                    #   debug = True
                      )
my_sampler_minimizer.minimize()

end = time.time()
print("Time elapsed: ", end - start)

# chains = np.array(my_sampler.chain)

# # do corner
# import corner
# import matplotlib.pyplot as plt

# fig = corner.corner(chains.reshape(-1, chains.shape[-1]), labels=[r"$h$", r"$n_s$", r"$\omega_b$", r"$\omega_{cdm}$", r"$\tau_{reio}$", r"$A_{planck}$", r"$\log(10^{10} A_s)$"])
# plt.savefig('corner.png')
# plt.show()
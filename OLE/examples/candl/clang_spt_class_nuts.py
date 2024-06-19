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




class my_theory(Theory):
    def initialize(self, **kwargs):
        super().initialize(**kwargs)   

        # input parameters of the theory
        self.requirements = ['h', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'logA']

        self.cosmo = classy.Class()

        self.class_settings = {'output':'tCl,pCl,lCl,mPk',
                        'lensing':'yes',
                        'l_max_scalars':3200, #  for SPT3G_2018_TTTEEE
                        # 'l_max_scalars':7925, #  for ACT

                        'N_ur':3.048,
                        'output_verbose':1,
                        }
        return 
    

    def compute(self, state):
        # Compute the observable for the given parameters.
        class_input = self.class_settings.copy()
        class_input['A_s'] = 1e-10*np.exp(state['parameters']['logA'][0])
        class_input['h'] = state['parameters']['h'][0]
        class_input['n_s'] = state['parameters']['n_s'][0]
        class_input['omega_b'] = state['parameters']['omega_b'][0]
        class_input['omega_cdm'] = state['parameters']['omega_cdm'][0]
        class_input['tau_reio'] = state['parameters']['tau_reio'][0]

        if state['parameters']['tau_reio'][0] < 0.01:
            class_input['tau_reio'] = 0.01

        self.cosmo.set(class_input)
        self.cosmo.compute()

        cls = self.cosmo.lensed_cl(3200)
        # cls = self.cosmo.lensed_cl(7925)
        # 3200 for SPT3G_2018_TTTEEE

        state['quantities']['tt'] = cls['tt']
        state['quantities']['ee'] = cls['ee']
        state['quantities']['te'] = cls['te']
        state['quantities']['bb'] = cls['bb']


        return state







class my_likelihood(Likelihood):
    def initialize(self, **kwargs):

        super().initialize(**kwargs)   

        self.use_likelihood = 'SPT3G_2018_TTTEEE'
        
        # define your candl likelihood
        if self.use_likelihood == 'SPT3G_2018_TTTEEE':
            self.candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)
        elif self.use_likelihood == 'ACT_DR4_TTTEEE':
            self.candl_like = candl.Like(candl.data.ACT_DR4_TTTEEE)



        
        return

    # @partial(jax.jit, static_argnums=(0,))
    def loglike(self, state):
        # Compute the loglikelihood for the given parameters.

        if self.use_likelihood == 'SPT3G_2018_TTTEEE':
            input_keys = ['EE_Poisson_150x150', 'EE_Poisson_150x220', 'EE_Poisson_220x220', 'EE_Poisson_90x150', 'EE_Poisson_90x220', 'EE_Poisson_90x90', 'EE_PolGalDust_Alpha', 'EE_PolGalDust_Amp', 'EE_PolGalDust_Beta', 'Ecal150', 'Ecal220', 'Ecal90', 'Kappa', 'TE_PolGalDust_Alpha', 'TE_PolGalDust_Amp', 'TE_PolGalDust_Beta', 'TT_CIBClustering_Amp', 'TT_CIBClustering_Beta', 'TT_GalCirrus_Alpha', 'TT_GalCirrus_Amp', 'TT_GalCirrus_Beta', 'TT_Poisson_150x150', 'TT_Poisson_150x220', 'TT_Poisson_220x220', 'TT_Poisson_90x150', 'TT_Poisson_90x220', 'TT_Poisson_90x90', 'TT_kSZ_Amp', 'TT_tSZ_Amp', 'TT_tSZ_CIB_Corr_Amp', 'Tcal150', 'Tcal220', 'Tcal90']
            ell = jnp.float32(jnp.arange(3199)+2)
        elif self.use_likelihood == 'ACT_DR4_TTTEEE':
            input_keys = ['yp']
            ell = jnp.float32(jnp.arange(7924)+2)\
        
        Dl = {'TT': state['quantities']['tt'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
              'TE': state['quantities']['te'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
              'EE': state['quantities']['ee'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
            #   'BB': state['quantities']['bb'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
              'ell': ell}

        candl_input = {key: state['parameters'][key][0] for key in input_keys}
        candl_input['Dl'] = Dl
        candl_input['tau'] = state['parameters']['tau_reio'][0]


        loglike = self.candl_like.log_like(candl_input)

        return jnp.array([loglike])



my_theory = my_theory()
my_likelihood = my_likelihood()


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

    'compute_data_covmat': True,
    # 'data_covmat_directory': './act_data_covmats',

    # output directory
    'output_directory': './output_clang_spt_nuts',

    # M adapt # burn-in of NUTS
    'M_adapt': 200,

    # 'plotting_directory': './plots_spt_clang_nuts',
    # 'testset_fraction': 0.1,
    'logfile': './output_clang_spt_nuts/log.txt',

    'learning_rate': 0.1,
    'num_iters': 300,

    'nwalkers':1,

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

                    # SPT3G_2018_TTTEEE

                    'EE_Poisson_150x150': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.011495, 'std': 0.01}, 
                                           'ref': {'mean': 0.011495, 'std': 0.01}, 
                                           'proposal': 0.01},
                    'EE_Poisson_150x220': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.018962, 'std': 0.01}, 
                                           'ref': {'mean': 0.018962, 'std': 0.01}, 
                                           'proposal': 0.01},
                    'EE_Poisson_220x220': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.047557, 'std': 0.01}, 
                                           'ref': {'mean': 0.047557, 'std': 0.01}, 
                                           'proposal': 0.01},
                    'EE_Poisson_90x150': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.018048, 'std': 0.01}, 
                                          'ref': {'mean': 0.018048, 'std': 0.01}, 
                                          'proposal': 0.01},
                    'EE_Poisson_90x220': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.015719, 'std': 0.01}, 
                                          'ref': {'mean': 0.015719, 'std': 0.01},
                                            'proposal': 0.01},
                    'EE_Poisson_90x90': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.040469, 'std': 0.01}, 
                                         'ref': {'mean': 0.040469, 'std': 0.01},
                                           'proposal': 0.01},
                    'EE_PolGalDust_Alpha': {'prior': {'min': -5.0, 'max': -1.0, 'mean': -2.42, 'std': 0.04}, 
                                            'ref': {'mean': -2.42, 'std': 0.04},
                                              'proposal': 0.04},
                    'EE_PolGalDust_Amp': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.05, 'std': 0.02}, 
                                          'ref': {'mean': 0.05, 'std': 0.02}, 
                                          'proposal': 0.02},
                    'EE_PolGalDust_Beta': {'prior': {'min': 0.5, 'max': 2.5, 'mean': 1.51, 'std': 0.04}, 
                                           'ref': {'mean': 1.51, 'std': 0.04},
                                             'proposal': 0.04},
                    'Kappa': {'prior': {'min': -0.01, 'max': 0.01, 'mean': 0.0, 'std': 0.00045}, 
                              'ref': {'mean': 0.0, 'std': 0.00045}, 
                              'proposal': 0.00045},
                    'TE_PolGalDust_Alpha': {'prior': {'min': -5.0, 'max': -1.0, 'mean': -2.42, 'std': 0.04},
                                             'ref': {'mean': -2.42, 'std': 0.04}, 
                                             'proposal': 0.04},
                    'TE_PolGalDust_Amp': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 0.12, 'std': 0.051},
                                           'ref': {'mean': 0.12, 'std': 0.051},
                                             'proposal': 0.051},
                    'TE_PolGalDust_Beta': {'prior': {'min': 0.5, 'max': 2.5, 'mean': 1.51, 'std': 0.04}, 
                                           'ref': {'mean': 1.51, 'std': 0.04}, 
                                           'proposal': 0.04},
                    'TT_CIBClustering_Amp': {'prior': {'min': 0.0, 'max': 20.0, 'mean': 3.2263, 'std': 1.0}, 
                                             'ref': {'mean': 3.2263, 'std': 1.0}, 
                                             'proposal': 1.0},
                    'TT_CIBClustering_Beta': {'prior': {'min': 0.0, 'max': 5.0, 'mean': 2.2952, 'std': 0.5}, 
                                              'ref': {'mean': 2.2952, 'std': 0.5}, 
                                              'proposal': 0.5},
                    'TT_GalCirrus_Alpha': {'prior': {'min': -3.0, 'max': -2.0,'mean': -2.53, 'std': 0.05}, 
                                           'ref': {'mean': -2.53, 'std': 0.05}, 
                                           'proposal': 0.05},
                    'TT_GalCirrus_Amp': {'prior': {'min': 0.0, 'max': 10.0, 'mean': 1.88, 'std': 0.2}, 
                                         'ref': {'mean': 1.88, 'std': 0.2}, 
                                         'proposal': 0.2},
                    'TT_GalCirrus_Beta': {'prior': {'min': 0.0, 'max': 2.0, 'mean': 1.48, 'std': 0.1}, 
                                          'ref': {'mean': 1.48, 'std': 0.1}, 
                                          'proposal': 0.1},
                    'TT_Poisson_150x150': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 15.8306, 'std': 5}, 
                                           'ref': {'mean': 15.8306, 'std': 5}, 
                                           'proposal': 5},
                    'TT_Poisson_150x220': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 15.8306, 'std': 5}, 
                                           'ref': {'mean': 15.8306, 'std': 5}, 
                                           'proposal': 5},
                    'TT_Poisson_220x220': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 68.2457, 'std': 5}, 
                                           'ref': {'mean': 68.2457, 'std': 5}, 
                                           'proposal': 5},
                    'TT_Poisson_90x150': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 23.9268, 'std': 5}, 
                                          'ref': {'mean': 23.9268, 'std': 5}, 
                                          'proposal': 5},
                    'TT_Poisson_90x220': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 21.5341, 'std': 5}, 
                                          'ref': {'mean': 21.5341, 'std': 5}, 
                                          'proposal': 5},
                    'TT_Poisson_90x90': {'prior': {'min': 0.0, 'max': 200.0, 'mean': 52.8146, 'std': 5.0}, 
                                         'ref': {'mean': 52.8146, 'std': 5.0}, 
                                         'proposal': 5.0},
                    'TT_kSZ_Amp': {'prior': {'min': 0.0, 'max': 20.0, 'mean': 3.0, 'std': 1.0}, 
                                   'ref': {'mean': 3.0, 'std': 1.0}, 
                                   'proposal': 1.0},
                    'TT_tSZ_Amp': {'prior': {'min': 0.0, 'max': 20.0, 'mean': 3.42, 'std': 0.5}, 
                                   'ref': {'mean': 3.42, 'std': 0.5}, 
                                   'proposal': 0.5},
                    'TT_tSZ_CIB_Corr_Amp': {'prior': {'min': -1.0, 'max': 1.0, 'mean': 0.1144, 'std': 0.1}, 
                                            'ref': {'mean': 0.1144, 'std': 0.1}, 
                                            'proposal': 0.1},                    
                    'Tcal150': {'prior': {'min': 0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                                'ref': {'mean': 1.0, 'std': 0.01}, 
                                'proposal': 0.01},
                    'Tcal220': {'prior': {'min': 0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                                'ref': {'mean': 1.0, 'std': 0.01}, 
                                'proposal': 0.01},
                    'Tcal90': {'prior': {'min': 0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                               'ref': {'mean': 1.0, 'std': 0.01}, 
                               'proposal': 0.01},
                    'Ecal150': {'prior': {'min': 0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                                'ref': {'mean': 1.0, 'std': 0.01}, 
                                'proposal': 0.01},
                    'Ecal220': {'prior': {'min':0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                                'ref': {'mean': 1.0, 'std': 0.01}, 
                                'proposal': 0.01},
                    'Ecal90': {'prior': {'min':0.9, 'max': 1.1, 'mean': 1.0, 'std': 0.01}, 
                               'ref': {'mean': 1.0, 'std': 0.01}, 
                               'proposal': 0.01},
}
    

covmat_path = None#'./covmat_candl.txt'


start = time.time()

my_sampler.initialize(theory=my_theory, likelihood=my_likelihood, parameters=my_parameters, covmat = covmat_path, **emulator_settings)

# Note the total run steps are   (nsteps * nwalkers * MPI_size)
n_steps = 1000
my_sampler.run_mcmc(n_steps)

end = time.time()
print("Time elapsed: ", end - start)

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














from OLE.theory import Theory
from OLE.likelihood import Likelihood
import jax.numpy as jnp
import time
import jax

import classy

import numpy as np


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
                        'l_max_scalars':3200,
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

        state['quantities']['tt'] = cls['tt']
        state['quantities']['ee'] = cls['ee']
        state['quantities']['te'] = cls['te']
        state['quantities']['bb'] = cls['bb']


        return state







class my_likelihood(Likelihood):
    def initialize(self, **kwargs):

        super().initialize(**kwargs)   
        
        # define your candl likelihood
        self.candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)




        
        return


    def loglike(self, state):
        # Compute the loglikelihood for the given parameters.
        ell = jnp.arange(3199)+2

        candl_input = {
            'Dl': {
                'TT': state['quantities']['tt'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
                'TE': state['quantities']['te'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
                'EE': state['quantities']['ee'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
                'BB': state['quantities']['bb'][2:]*ell*(ell+1)* 2.7255e6**2 / (2*jnp.pi),
                'ell': ell,
            },
            'tau': state['parameters']['tau_reio'][0],
            'EE_Poisson_150x150': state['parameters']['EE_Poisson_150x150'][0],
            'EE_Poisson_150x220': state['parameters']['EE_Poisson_150x220'][0],
            'EE_Poisson_220x220': state['parameters']['EE_Poisson_220x220'][0],
            'EE_Poisson_90x150': state['parameters']['EE_Poisson_90x150'][0],
            'EE_Poisson_90x220': state['parameters']['EE_Poisson_90x220'][0],
            'EE_Poisson_90x90': state['parameters']['EE_Poisson_90x90'][0],
            'EE_PolGalDust_Alpha': state['parameters']['EE_PolGalDust_Alpha'][0],
            'EE_PolGalDust_Amp': state['parameters']['EE_PolGalDust_Amp'][0],
            'EE_PolGalDust_Beta': state['parameters']['EE_PolGalDust_Beta'][0],
            'Ecal150': state['parameters']['Ecal150'][0],
            'Ecal220': state['parameters']['Ecal220'][0],
            'Ecal90': state['parameters']['Ecal90'][0],
            'Kappa': state['parameters']['Kappa'][0],
            'TE_PolGalDust_Alpha': state['parameters']['TE_PolGalDust_Alpha'][0],
            'TE_PolGalDust_Amp': state['parameters']['TE_PolGalDust_Amp'][0],
            'TE_PolGalDust_Beta': state['parameters']['TE_PolGalDust_Beta'][0],
            'TT_CIBClustering_Amp': state['parameters']['TT_CIBClustering_Amp'][0],
            'TT_CIBClustering_Beta': state['parameters']['TT_CIBClustering_Beta'][0],
            'TT_GalCirrus_Alpha': state['parameters']['TT_GalCirrus_Alpha'][0],
            'TT_GalCirrus_Amp': state['parameters']['TT_GalCirrus_Amp'][0],
            'TT_GalCirrus_Beta': state['parameters']['TT_GalCirrus_Beta'][0],
            'TT_Poisson_150x150': state['parameters']['TT_Poisson_150x150'][0],
            'TT_Poisson_150x220': state['parameters']['TT_Poisson_150x220'][0],
            'TT_Poisson_220x220': state['parameters']['TT_Poisson_220x220'][0],
            'TT_Poisson_90x150': state['parameters']['TT_Poisson_90x150'][0],
            'TT_Poisson_90x220': state['parameters']['TT_Poisson_90x220'][0],
            'TT_Poisson_90x90': state['parameters']['TT_Poisson_90x90'][0],
            'TT_kSZ_Amp': state['parameters']['TT_kSZ_Amp'][0],
            'TT_tSZ_Amp': state['parameters']['TT_tSZ_Amp'][0],
            'TT_tSZ_CIB_Corr_Amp': state['parameters']['TT_tSZ_CIB_Corr_Amp'][0],
            'Tcal150': state['parameters']['Tcal150'][0],
            'Tcal220': state['parameters']['Tcal220'][0],
            'Tcal90': state['parameters']['Tcal90'][0],
        }

        loglike = self.candl_like.log_like(candl_input)

        if jnp.isnan(loglike):
            loglike = -1e100

        return jnp.array([loglike])



my_theory = my_theory()
my_likelihood = my_likelihood()


emulator_settings = {
    # the number of data points in cache before the emulator is to be trained
    'min_data_points': 80,

    # maximal cache size
    'cache_size': 1000,

    # name of the cache file
    'cache_file': './output_sampler_clang_nuts/cache.pkl',

    # load the cache from the cache file
    'load_cache': True,

    # load inifile from the cache file
    # 'load_initial_state': True,
    # 'test_emulator': False, # if True the emulator is tested and potentially retrained with new data

    # delta loglike for what states to be accepted to the cache
    'delta_loglike': 300.0,

    # accuracy parameters for loglike:
    'quality_threshold_constant': 0.1,
    'quality_threshold_linear': 0.0,
    'quality_threshold_quadratic': 0.001,

    # related so sampler

    # number of walker
    'nwalkers': 10,

    # output directory
    'output_directory': './output_sampler_clang_nuts',

    # force by overwriting previous results
    'force': True,

    # M adapt
    'M_adapt': 1000,
}


# load sampler 
from OLE.sampler import EnsembleSampler, Sampler, NUTSSampler
my_sampler = NUTSSampler(debug=False)


my_parameters = {'h': {'prior': {'min': 0.64, 'max': 0.72}, 
                       'ref': {'mean': 0.68, 'std': 0.01},
                       'proposal': 0.01,},
                    'n_s': {'prior': {'min': 0.92, 'max': 1.0}, 'ref': {'mean': 0.965, 'std': 0.005},
                            'proposal': 0.01,}, 
                    'omega_b': {'prior': {'min': 0.0216, 'max': 0.024}, 'ref': {'mean': 0.0223, 'std': 0.0003},
                                'proposal': 0.0002,}, 
                    'omega_cdm': {'prior': {'min': 0.110, 'max': 0.13}, 'ref': {'mean': 0.120, 'std': 0.002},
                                  'proposal': 0.002,}, 
                    'tau_reio': {'prior': {'min': 0.01, 'max': 0.1}, 'ref': {'mean': 0.055, 'std': 0.01},
                                 'proposal': 0.01,},
                    'A_planck': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'mean': 1.00, 'std': 0.05},
                                 'proposal': 0.005,},
                    'logA': {'prior': {'min': 2.9, 'max': 3.3}, 'ref': {'mean': 3.1, 'std': 0.05},
                             'proposal': 0.05},

                    'EE_Poisson_150x150': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.040469, 'std': 0.003448}, 'proposal': 0.003448},
                    'EE_Poisson_150x220': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.018962, 'std': 0.005689}, 'proposal': 0.005689},
                    'EE_Poisson_220x220': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.047557, 'std': 0.014267}, 'proposal': 0.014267},
                    'EE_Poisson_90x150': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.018048, 'std': 0.005414}, 'proposal': 0.005414},
                    'EE_Poisson_90x220': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.015719, 'std': 0.004716}, 'proposal': 0.004716},
                    'EE_Poisson_90x90': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.040469, 'std': 0.012141}, 'proposal': 0.012141},
                    'EE_PolGalDust_Alpha': {'prior': {'min': -10.0, 'max': 1.0}, 'ref': {'mean': -2.42, 'std': 0.04}, 'proposal': 0.04},
                    'EE_PolGalDust_Amp': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.05, 'std': 0.022}, 'proposal': 0.022},
                    'EE_PolGalDust_Beta': {'prior': {'min': -1.0, 'max': 10.0}, 'ref': {'mean': 1.51, 'std': 0.04}, 'proposal': 0.04},
                    'Ecal150': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01},
                    'Ecal220': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01},
                    'Ecal90': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01},
                    'Kappa': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.0, 'std': 0.00045}, 'proposal': 0.00045},
                    'TE_PolGalDust_Alpha': {'prior': {'min': -10.0, 'max': 1.0}, 'ref': {'mean': -2.42, 'std': 0.04}, 'proposal': 0.04},
                    'TE_PolGalDust_Amp': {'prior': {'min': -1.0, 'max': 1.0}, 'ref': {'mean': 0.12, 'std': 0.051}, 'proposal': 0.051},
                    'TE_PolGalDust_Beta': {'prior': {'min': -1.0, 'max': 10.0}, 'ref': {'mean': 1.51, 'std': 0.04}, 'proposal': 0.04},
                    'TT_CIBClustering_Amp': {'prior': {'min': -10.0, 'max': 100.0}, 'ref': {'mean': 3.2263, 'std': 1.8354}, 'proposal': 1.8354},
                    'TT_CIBClustering_Beta': {'prior': {'min': -10.0, 'max': 100.0}, 'ref': {'mean': 2.2642, 'std': 0.3814}, 'proposal': 0.3814},
                    'TT_GalCirrus_Alpha': {'prior': {'min': -10.0, 'max': 1.0}, 'ref': {'mean': -2.53, 'std': 0.05}, 'proposal': 0.05},
                    'TT_GalCirrus_Amp': {'prior': {'min': -1.0, 'max': 10.0}, 'ref': {'mean': 1.88, 'std': 0.48}, 'proposal': 0.48},
                    'TT_GalCirrus_Beta': {'prior': {'min': -1.0, 'max': 10.0}, 'ref': {'mean': 1.48, 'std': 0.02}, 'proposal': 0.02},
                    'TT_Poisson_150x150': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 15.3455, 'std': 4.132}, 'proposal': 4.132},
                    'TT_Poisson_150x220': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 28.3573, 'std': 4.1925}, 'proposal': 4.1925},
                    'TT_Poisson_220x220': {'prior': {'min': -100.0, 'max': 1000.0}, 'ref': {'mean': 75.9719, 'std': 14.8624}, 'proposal': 14.8624},
                    'TT_Poisson_90x150': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 22.4417, 'std': 7.0881}, 'proposal': 7.0881},
                    'TT_Poisson_90x220': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 20.7004, 'std': 5.9235}, 'proposal': 5.9235},
                    'TT_Poisson_90x90': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 51.3204, 'std': 9.442}, 'proposal': 9.442},
                    'TT_kSZ_Amp': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 3.7287, 'std': 4.644}, 'proposal': 4.644},
                    'TT_tSZ_Amp': {'prior': {'min': -100.0, 'max': 100.0}, 'ref': {'mean': 3.2279, 'std': 2.3764}, 'proposal': 2.3764},
                    'TT_tSZ_CIB_Corr_Amp': {'prior': {'min': -10.0, 'max': 10.0}, 'ref': {'mean': 0.1801, 'std': 0.3342}, 'proposal': 0.3342},                    
                    'Tcal150': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01},
                    'Tcal220': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01},
                    'Tcal90': {'prior': {'min': 0.0, 'max': 2.0}, 'ref': {'mean': 1.0, 'std': 0.1}, 'proposal': 0.01}
}

start = time.time()

my_sampler.initialize(theory=my_theory, likelihood=my_likelihood, parameters=my_parameters, **emulator_settings)

# Note the total run steps are   (nsteps * nwalkers * MPI_size)
n_steps = 1000
my_sampler.run_mcmc(n_steps)

end = time.time()
print("Time elapsed: ", end - start)

# chains = np.array(my_sampler.chain)

# # do corner
# import corner
# import matplotlib.pyplot as plt

# fig = corner.corner(chains.reshape(-1, chains.shape[-1]), labels=[r"$h$", r"$n_s$", r"$\omega_b$", r"$\omega_{cdm}$", r"$\tau_{reio}$", r"$A_{planck}$", r"$\log(10^{10} A_s)$"])
# plt.savefig('corner.png')
# plt.show()